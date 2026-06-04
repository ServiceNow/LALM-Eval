"""LLM-based judge metrics for evaluation."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import httpx
import regex as _regex
import yaml
from openai import AsyncAzureOpenAI, AsyncOpenAI, APIConnectionError
from tqdm import tqdm

from metrics.metrics import Metrics
from postprocessors.base import Postprocessor
from utils import util
from utils.custom_logging import write_record_log, append_final_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper to pull a JSON object/array out of messy LLM responses (e.g. when the
# judge interleaves reasoning around the structured output we asked for).
# ---------------------------------------------------------------------------

_JSON_START = re.compile(r"[{[]")
_STRAY_BACKSLASH_CHARS = "_+-[]*'$"
_UNESCAPED_BACKSLASH = re.compile(
    r"""(?<!\\)(?=\\(?!["/bfnrt]|u[0-9A-Fa-f]{4}|\\(?:\\\\)*(?!\\)))""",
    re.VERBOSE,
)
_UNESCAPED_DOUBLE_QUOTE = _regex.compile(
    r"""(?<![,:[{]\s*|(?<!\\)\\)(?="(?!\s*[,:\]}]))""",
    _regex.VERBOSE,
)


def _extract_json(text: str) -> dict | list | None:
    """Extract the first JSON object/array from messy text, or None."""
    if not text:
        return None

    text = re.sub(rf"\\\\([{re.escape(_STRAY_BACKSLASH_CHARS)}])", r"\1", text)
    text = re.sub(rf"\\([{re.escape(_STRAY_BACKSLASH_CHARS)}])", r"\1", text)

    repaired = _UNESCAPED_BACKSLASH.sub(r"\\", _UNESCAPED_DOUBLE_QUOTE.sub(r"\\", text))
    for candidate in (text, repaired):
        decoder = json.JSONDecoder()
        pos = 0
        while m := _JSON_START.search(candidate, pos):
            try:
                obj, _end = decoder.raw_decode(candidate, m.start())
            except json.JSONDecodeError:
                pos = m.start() + 1
                continue
            if isinstance(obj, (dict, list)):
                return obj
            pos = m.start() + 1

    return None


# ---------------------------------------------------------------------------
# Helper to load prompt templates shipped with the package
# ---------------------------------------------------------------------------

_template_cache: dict[str, str] | None = None
PROMPT_FILE_PATH = Path(__file__).resolve().parents[1] / "prompts/judge_prompts.yaml"

NestedPromptType = Union[
    str,
    dict[str, str],
    dict[str, dict[str, str]],
]


def _get_prompt(kind: str) -> NestedPromptType:
    """Load and return the prompt string for *kind* every call (no caching)."""
    data = yaml.safe_load(PROMPT_FILE_PATH.read_text()) or {}
    if kind not in data:
        raise KeyError(f"Prompt '{kind}' not found in {PROMPT_FILE_PATH}")
    return data[kind]


# ---------------------------------------------------------------------------
# Base LLM judge – uses gpt-4o via async OpenAI SDK
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_CONCURRENCY = 5


class _BaseLLMJudge(Metrics):
    """Common LLM-as-judge class."""

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__()
        # Store the properties dictionary
        self._judge_properties = judge_properties or {}
        # Extract parameters from judge_properties or use defaults
        self._max_concurrency = self._judge_properties.get("judge_concurrency") or _DEFAULT_MAX_CONCURRENCY
        self._model = self._judge_properties.get("judge_model") or _DEFAULT_OPENAI_MODEL
        self._judge_type = self._judge_properties.get("judge_type") or "openai"
        self._request_manager = None  # Set in Engine
        # Initialize the appropriate client based on judge_type
        if self._judge_type == "openai":
            self._client = AsyncAzureOpenAI(
                api_key=self._judge_properties.get("judge_api_key"),
                api_version=self._judge_properties.get("judge_api_version"),
                azure_endpoint=self._judge_properties.get("judge_api_endpoint"),
            )
        elif self._judge_type == "vllm":
            self._client = AsyncOpenAI(
                base_url=self._judge_properties.get("judge_api_endpoint"),
                api_key=self._judge_properties.get("judge_api_key"),
            )

    def set_request_manager(self, manager):
        """Set the request manager and register the model."""
        self._request_manager = manager
        if self._request_manager is not None:
            # Register the model type directly with the central controller
            self._request_manager.central_controller.register_model(
                self._model, self._max_concurrency
            )

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict | None:

        max_retries = 8
        for attempt in range(max_retries):
            try:
                # Check if we should override the system prompt
                model_name = self._judge_properties.get("prompt_override", None)
                if model_name is not None:
                    # Construct the key using model_name + metric_name
                    # Extract the metric name from self._prompt_key (which is set in subclasses)
                    metric_name = self._prompt_key
                    prompt_key = f"{model_name}_{metric_name}"

                    try:
                        # Load the prompt from judge_prompts.yaml using the constructed key
                        system_prompt = _get_prompt(prompt_key)
                    except KeyError as e:
                        logger.warning("Prompt key '%s' not found in judge_prompts.yaml: %s", prompt_key, e)
                        # Keep using the original system prompt if the constructed key is not found
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Get temperature from judge_properties or use default 0.1
                temperature = self._judge_properties.get("judge_temperature", 0.1)
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content.strip()
                # Clean response to remove thinking content
                cleaned_content = Postprocessor.remove_thinking_content(content)
                content = cleaned_content
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    return content
            except (APIConnectionError, httpx.ConnectError, httpx.HTTPError) as connection_error:
                logger.warning("API connection failed (attempt %d/%d): %s", attempt + 1, max_retries, connection_error)
                await asyncio.sleep(2)  # Wait before retrying
            except Exception as e:
                error_message = str(e)
                # Handle content policy violations separately
                if "content management policy" in error_message and attempt < max_retries - 1:
                    logger.warning(
                        "Content filter triggered (attempt %d/%d). Modifying prompt...", attempt + 1, max_retries)

                    # Apply progressively stronger modifications to avoid content filter
                    if attempt == 0:
                        # First retry: Add a safety prefix to the system prompt
                        system_prompt = f"Please provide an academic evaluation only. Avoid any harmful, unethical, or inappropriate content. {system_prompt}"
                    else:
                        # Second retry: Add a safety prefix to the user prompt
                        user_prompt = f"For academic evaluation purposes only: {user_prompt}"

                    # Continue to retry with modified prompts
                    await asyncio.sleep(1)
                    continue

                # For other unexpected errors, log and potentially break
                logger.error("Unexpected error in _score_once: %s", e)
                break
        logger.error("All %d attempts failed for this sample. Skipping.", max_retries)
        return None

    async def _judge_all(
        self,
        candidates: list[str],
        references: list[str],
        task_name: str | None = None,
        model_name: str | None = None,
    ) -> list:
        """Run the LLM judge over *candidates* vs *references*.

        *prompt_add_on* is extra text appended to the base system prompt. Each
        concrete metric can inject task-specific instructions without changing
        the core helper.
        """
        if self._request_manager is None:
            raise ValueError("Request manager must be set before calling _judge_all")

        # Setup for token management
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        pending_samples = list(range(len(candidates)))  # Samples waiting for tokens
        processing_samples = set(range(len(candidates)))
        completed_samples = set()  # Samples that are completed
        results = [None] * len(candidates)
        sys_prompt_template = _get_prompt(self._prompt_key)

        # Generate unique evaluator instance ID
        evaluator_id = f"llm_judge_{self._prompt_key}_{id(self)}"
        # Continuously ask for tokens based on pending samples
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(candidates)

            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))

            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0

            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0
            # Continue running until all samples have been given tokens
            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as needed for pending samples, up to max_concurrency
                request_amount = min(self._max_concurrency, len(pending_samples))

                granted = await self._request_manager.request_tokens(
                    self._model, evaluator_id, request_amount)
                if granted > 0:
                    # Process the granted tokens
                    for _ in range(granted):
                        if pending_samples:
                            pending_samples.pop(0)
                            # Release semaphore permits for each granted token
                            token_sem.release()
                    # Wait based on dataset size when tokens were granted
                    await asyncio.sleep(token_wait)
                else:
                    # Backoff when no tokens were granted, based on dataset size
                    # Apply a small multiplier for repeated failures, but cap it
                    backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                    await asyncio.sleep(no_token_wait * backoff_multiplier)

        # Start the token management task
        token_task = asyncio.create_task(token_manager())
        # Process each candidate-reference pair with token management
        async def _evaluate_with_token_mgmt(idx, cand, ref):
            # Acquire a token
            await token_sem.acquire()

            try:
                # Support prompts split into system/user sub-keys
                if isinstance(sys_prompt_template, dict) and "system" in sys_prompt_template and "user" in sys_prompt_template:
                    sys_prompt = sys_prompt_template["system"]
                    user_prompt = sys_prompt_template["user"].replace("{{reference}}", str(ref)).replace("{{hypothesis}}", str(cand))
                else:
                    sys_prompt = sys_prompt_template
                    user_prompt = f"candidate: {cand}\nreference: {ref}"

                # Call the scoring function
                result = await self._score_once(sys_prompt, user_prompt)

                # Move from processing to completed
                processing_samples.remove(idx)
                completed_samples.add(idx)

                # Return token to model's pool
                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                return idx, result
            except Exception as eval_error:
                # Make sure to return token on error
                logger.error("[_BaseLLMJudge._evaluate_with_token_mgmt] Error processing sample %d: %s", idx, eval_error)

                # Move sample from processing to completed
                processing_samples.remove(idx)
                completed_samples.add(idx)

                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                return idx, None
        # Create tasks for all samples
        tasks = [_evaluate_with_token_mgmt(i, c, r) for i, (c, r) in enumerate(zip(candidates, references))]

        # Use tqdm to show progress
        desc = f"{self._prompt_key}"
        if task_name and model_name:
            desc = f"Evaluating {task_name} | {model_name} | {desc}"
        with tqdm(total=len(tasks), desc=desc) as progress_bar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
                progress_bar.update(1)

        # Cancel the token_manager task when all samples are processed
        token_task.cancel()
        try:
            await token_task
        except asyncio.CancelledError:
            pass

        return results



# All of these judges always return on a scale of 100

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):
    """Binary LLM judge metric for yes/no evaluations."""
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None
        self.instructions = None
        self.model_responses = None

    async def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        overall = await super().get_score(candidates, references, task_name, model_name)
        if self.name in overall:
            overall[self.name] = util.smart_round(overall[self.name] * 100.0 , 2)
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, task_name, model_name, explanations,
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, task_name: str | None = None, model_name: str | None = None):
        """Compute record-level scores for binary evaluation."""
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, task_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


class RedTeamingJudgeMetric(_BaseLLMJudge):
    """Binary LLM judge metric for yes/no evaluations."""
    name: str = "llm_judge_redteaming"
    _prompt_key: str = "redteaming_judge_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None
        self.instructions = None
        self.model_responses = None

    async def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        overall = await super().get_score(candidates, references, task_name, model_name)
        if self.name in overall:
            overall[self.name] = util.smart_round(overall[self.name] * 100.0 , 2)
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, task_name, model_name, explanations,
                           instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, task_name: str | None = None, model_name: str | None = None):
        """Compute record-level scores for binary evaluation."""
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, task_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


class DetailedLLMJudgeMetric(_BaseLLMJudge):
    """Detailed LLM judge metric.
    On a scale of 0 to 5, how well does the candidate text match the reference text?
    Returns:
        float: Overall score
    """
    name: str = "llm_judge_detailed"
    _prompt_key: str = "detailed_judge_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None

    async def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        overall = await super().get_score(candidates, references, task_name, model_name)
        if self.name in overall:
            # From 0-5 scale to 0-100 scale
            overall[self.name] = util.smart_round(overall[self.name] * 20.0, 2)
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, task_name, model_name, explanations,
                          instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, task_name: str | None = None, model_name: str | None = None):
        """Compute record-level scores for detailed evaluation."""
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, task_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


class CallHomeLLMJudgeMetric(_BaseLLMJudge):
    """CallHome-specific LLM judge metric."""
    name: str = "llm_judge_callhome"
    _prompt_key: str = "callhome_judge_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None

    async def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        overall = await super().get_score(candidates, references, task_name, model_name)
        if self.name in overall:
            overall[self.name] += 1
            overall[self.name] *= 10
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, task_name, model_name, explanations,
                      instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, task_name: str | None = None, model_name: str | None = None):
        """Compute record-level scores for CallHome evaluation."""
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, task_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


class LLMJudgeMetric(Metrics):
    """Simple LLM judge metric."""
    name: str = "llm_judge"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self.instructions = None

    def __call__(self, ref: str, hyp: str, instructions=None):
        # Store instructions for potential later use
        self.instructions = instructions
        return float(ref.strip() == hyp.strip())


class BigBenchAudioLLMJudgeMetric(_BaseLLMJudge):
    """
    A judge metric for evaluating BigBenchAudio predictions using an LLM to determine correctness.

    This class compares model predictions (candidates) against references (transcript, official_answer pairs)
    using a prompt-based LLM, which returns either "CORRECT" or "INCORRECT" for each comparison.
    """
    name: str = "llm_judge_big_bench_audio"
    _prompt_key: str = "big_bench_audio_judge_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None

    async def __call__(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]],
        instructions=None,
        *,
        task_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_responses=None
    ) -> dict:
        """
        Evaluate the predictions using LLM-based judgment and return overall accuracy.

        Args:
            candidates (List[str]): List of model predictions.
            references (List[Tuple[str, str]]): List of (transcript, official_answer) pairs.
            task_name (Optional[str]): Optional dataset name for logging purposes.
            model_name (Optional[str]): Optional model name for logging purposes.

        Returns:
            dict: Evaluation result with accuracy and counts for correct, incorrect, and failed responses.
        """
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        scores = await self.compute_record_level_scores(candidates, references, task_name, model_name)
        all_scores = scores[self.name]

        num_correct = sum(1 for s in all_scores if s == 1.0)
        num_incorrect = sum(1 for s in all_scores if s == 0.0)
        total = num_correct + num_incorrect

        overall = {
            self.name: util.smart_round((num_correct * 100.0) / total, 2) if total > 0 else 0.0,
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "num_failed": len(all_scores) - total,
        }

        if task_name and model_name:
            write_record_log(self, references, candidates, all_scores, task_name, model_name,
                       instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)

        return overall

    async def compute_record_level_scores(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]],
        task_name: str | None = None,
        model_name: str | None = None
    ) -> dict:
        """
        Calls the LLM to judge each (transcript, prediction, official_answer) triple and returns scores.

        Args:
            candidates (List[str]): List of model predictions.
            references (List[Tuple[str, str]]): List of (transcript, official_answer) pairs.

        Returns:
            dict: A mapping from metric name to list of 1.0 (correct), 0.0 (incorrect), or None (failed).
        """
        # LLM input: (transcript, prediction, official_answer)
        raw_responses = await self._judge_all(candidates, references, task_name, model_name)
        scores = []

        for response in raw_responses:
            if isinstance(response, str):
                normalized = response.strip().upper()
                if normalized == "CORRECT":
                    scores.append(1.0)
                elif normalized == "INCORRECT":
                    scores.append(0.0)
                else:
                    scores.append(None)
            else:
                scores.append(None)

        self.explanations = raw_responses  # Save raw LLM responses for inspection/logging
        return {self.name: scores}

    def _append_final_score(self, overall: dict, task_name: str, model_name: str) -> None:
        """
        Appends the final score summary to a structured log file.

        Args:
            overall (dict): Final evaluation scores and metadata.
            task_name (str): Dataset identifier.
            model_name (str): Model identifier.
        """

        def _slug(text: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]+", "_", text)

        log_path = Path(".") / f"{_slug(task_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")


class MtbenchLLMJudgeMetric(_BaseLLMJudge):
    """MT-bench LLM judge metric.
    On a scale of 0 to 10, how well does the candidate text match the reference text?
    Returns:
        float: Overall score
    """
    name: str = "mt_bench_llm_judge"
    _prompt_key: str = "mt_bench"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)

        self.instructions = None
        self.model_responses = None
        self.explanations = None
        self.NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict | None:

        max_retries = 8
        for attempt in range(max_retries):
            try:
                # Check if we should override the system prompt
                model_name = self._judge_properties.get("prompt_override", None)
                if model_name is not None:
                    # Construct the key using model_name + metric_name
                    # Extract the metric name from self._prompt_key (which is set in subclasses)
                    metric_name = self._prompt_key
                    prompt_key = f"{model_name}_{metric_name}"

                    try:
                        # Load the prompt from judge_prompts.yaml using the constructed key
                        system_prompt = _get_prompt(prompt_key)
                    except KeyError as e:
                        logger.warning("Prompt key '%s' not found in judge_prompts.yaml: %s", prompt_key, e)
                        # Keep using the original system prompt if the constructed key is not found
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Get temperature from judge_properties or use default 0.1
                temperature = self._judge_properties.get("judge_temperature", 0.1)
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content.strip()
                # Clean response to remove thinking content
                cleaned_content = Postprocessor.remove_thinking_content(content)
                content = cleaned_content
                score_pattern = re.compile(r"\[\[?\s*(\d+(?:\.\d+)?)\s*\]?\]")
                # Backup pattern: just finds any number (int or float) in the text
                score_pattern_backup = re.compile(r"\d+(?:\.\d+)?")
                match = score_pattern.search(content)
                if not match:
                    match = score_pattern_backup.search(content)
                if match:
                    rating = float(match.group(1) if match.lastindex else match.group(0))
                else:
                    logger.warning("Score decoding failed, returned 0 score")
                    rating = 0.0
                return rating
            except (APIConnectionError, httpx.ConnectError, httpx.HTTPError) as connection_error:
                logger.warning("API connection failed (attempt %d/%d): %s", attempt + 1, max_retries, connection_error)
                await asyncio.sleep(2)  # Wait before retrying
            except Exception as e:
                error_message = str(e)
                # Handle content policy violations separately
                if "content management policy" in error_message and attempt < max_retries - 1:
                    logger.warning(
                        "Content filter triggered (attempt %d/%d). Modifying prompt...", attempt + 1, max_retries)

                    # Apply progressively stronger modifications to avoid content filter
                    if attempt == 0:
                        # First retry: Add a safety prefix to the system prompt
                        system_prompt = f"Please provide an academic evaluation only. Avoid any harmful, unethical, or inappropriate content. {system_prompt}"
                    else:
                        # Second retry: Add a safety prefix to the user prompt
                        user_prompt = f"For academic evaluation purposes only: {user_prompt}"

                    # Continue to retry with modified prompts
                    await asyncio.sleep(1)
                    continue

                # For other unexpected errors, log and potentially break
                logger.error("Unexpected error in _score_once: %s", e)
                break
        logger.error("All %d attempts failed for this sample. Skipping.", max_retries)
        return 0.0

    def _build_conversation_history(self, instructions: list, responses: list, up_to_turn: int, use_references: bool = False) -> str:
        """Build formatted conversation history for judge prompt.
        
        Args:
            instructions: List of user instructions/questions for all turns
            responses: List of assistant responses or reference answers for all turns
            up_to_turn: Build history up to (but not including) this turn index
            use_references: If True, format as reference answers; if False, format as assistant responses
            
        Returns:
            Formatted conversation history string
        """
        if up_to_turn == 0:
            return "(No previous conversation)"
        
        history = []
        response_label = "Reference answer" if use_references else "Assistant A"
        for i in range(up_to_turn):
            history.append(f"### User (Turn {i + 1}):\n{instructions[i]}")
            resp = responses[i] if responses and i < len(responses) else ""
            history.append(f"### {response_label} (Turn {i + 1}):\n{resp}")
        return "\n\n".join(history)

    async def _judge_all(
            self,
            candidates: list[str],
            references: list[str],
            task_name: str | None = None,
            model_name: str | None = None,
    ) -> list:
        """Run the LLM judge over *candidates* vs *references*.

        *prompt_add_on* is extra text appended to the base system prompt. Each
        concrete metric can inject task-specific instructions without changing
        the core helper.
        """
        if self._request_manager is None:
            raise ValueError("Request manager must be set before calling _judge_all")

        # Setup for token management
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        pending_samples = list(range(len(candidates)))  # Samples waiting for tokens
        processing_samples = set(range(len(candidates)))
        completed_samples = set()  # Samples that are completed
        results = [None] * len(candidates)
        sys_prompt_template = _get_prompt(self._prompt_key)

        # Extract N-turn prompt templates (generalized for any number of turns)
        reference_turn_n_sys_prompt = sys_prompt_template.get(
            'judge', {}).get('reference_turn_n', {}).get('system', {}).get('en')
        reference_turn_n_user_prompt = sys_prompt_template.get(
            'judge', {}).get('reference_turn_n', {}).get('user', {}).get('en')
        no_reference_turn_n_sys_prompt = sys_prompt_template.get(
            'judge', {}).get('no_reference_turn_n', {}).get('system', {}).get('en')
        no_reference_turn_n_user_prompt = sys_prompt_template.get(
            'judge', {}).get('no_reference_turn_n', {}).get('user', {}).get('en')

        # Generate unique evaluator instance ID
        evaluator_id = f"llm_judge_{self._prompt_key}_{id(self)}"

        # Continuously ask for tokens based on pending samples
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(candidates)

            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))

            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0

            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0
            # Continue running until all samples have been given tokens
            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as needed for pending samples, up to max_concurrency
                request_amount = min(self._max_concurrency, len(pending_samples))

                granted = await self._request_manager.request_tokens(
                    self._model, evaluator_id, request_amount)
                if granted > 0:
                    # Process the granted tokens
                    for _ in range(granted):
                        if pending_samples:
                            pending_samples.pop(0)
                            # Release semaphore permits for each granted token
                            token_sem.release()
                    # Wait based on dataset size when tokens were granted
                    await asyncio.sleep(token_wait)
                else:
                    # Backoff when no tokens were granted, based on dataset size
                    # Apply a small multiplier for repeated failures, but cap it
                    backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                    await asyncio.sleep(no_token_wait * backoff_multiplier)

        # Start the token management task
        token_task = asyncio.create_task(token_manager())

        # Process each candidate-reference pair with token management
        async def _evaluate_with_token_mgmt(idx, cand, ref):
            # Acquire a token
            await token_sem.acquire()
            assert isinstance(cand, dict)
            targets = cand['targets']
            responses = cand['responses']
            instructions = cand['instructions']
            category = cand['category']
            assert isinstance(instructions, list)

            # Select appropriate prompt templates based on category
            if category in self.NEED_REF_CATS:
                sys_prompt_template = reference_turn_n_sys_prompt
                user_prompt_template = reference_turn_n_user_prompt
            else:
                sys_prompt_template = no_reference_turn_n_sys_prompt
                user_prompt_template = no_reference_turn_n_user_prompt

            try:
                # Dynamic result dict for N turns
                # Use min of instructions and responses to handle mismatched lengths
                num_turns = min(len(instructions), len(responses)) if responses and not isinstance(responses, str) else len(instructions)
                result = {f"turn{i + 1}": 0.0 for i in range(num_turns)}
                result["overall"] = 0.0
                
                if isinstance(responses, str) or responses is None or len(responses) == 0:
                    return idx, result

                # Process each turn dynamically
                for turn in range(num_turns):
                    # Build conversation history up to current turn
                    conversation_history = self._build_conversation_history(
                        instructions, responses, up_to_turn=turn
                    )
                    
                    # Format user prompt with N-turn template
                    if category in self.NEED_REF_CATS:
                        # Build reference history for reference-based categories
                        conversation_history_with_references = self._build_conversation_history(
                            instructions, targets, up_to_turn=turn, use_references=True
                        )
                        current_reference = targets[turn] if targets and turn < len(targets) else ""
                        user_prompt = user_prompt_template.format(
                            conversation_history=conversation_history,
                            conversation_history_with_references=conversation_history_with_references,
                            current_question=instructions[turn],
                            current_reference=current_reference,
                            current_response=responses[turn],
                            turn_number=turn + 1
                        )
                    else:
                        user_prompt = user_prompt_template.format(
                            conversation_history=conversation_history,
                            current_question=instructions[turn],
                            current_response=responses[turn],
                            turn_number=turn + 1
                        )

                    # Call the scoring function
                    rating = await self._score_once(sys_prompt_template, user_prompt)
                    result[f'turn{turn + 1}'] = rating

                # Compute overall score as average of all turns, scaled to [0, 100]
                turn_scores = [result[f'turn{i + 1}'] for i in range(num_turns)]
                result['overall'] = util.smart_round(sum(turn_scores) * 10 / len(turn_scores), 2) if turn_scores else 0.0 
                # Move from processing to completed
                processing_samples.remove(idx)
                completed_samples.add(idx)

                # Return token to model's pool
                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                return idx, result
            except Exception as eval_error:
                # Make sure to return token on error
                logger.error("[_BaseLLMJudge._evaluate_with_token_mgmt] Error processing sample %d: %s", idx,
                             eval_error)

                # Move sample from processing to completed
                processing_samples.remove(idx)
                completed_samples.add(idx)

                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                return idx, None

        # Create tasks for all samples
        tasks = [_evaluate_with_token_mgmt(i, c, r) for i, (c, r) in enumerate(zip(candidates, references))]

        # Use tqdm to show progress
        desc = f"{self._prompt_key}"
        if task_name and model_name:
            desc = f"Evaluating {task_name} | {model_name} | {desc}"
        with tqdm(total=len(tasks), desc=desc) as progress_bar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
                progress_bar.update(1)

        # Cancel the token_manager task when all samples are processed
        token_task.cancel()
        try:
            await token_task
        except asyncio.CancelledError:
            pass

        return results

    async def get_score(self, candidates, references, task_name=None, model_name=None) -> dict:
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list
            task_name: optional dataset name for progress bar
            model_name: optional model name for progress bar

        Returns:
            vary based on implementation, should be a dict
        """
        assert self.name is not None
        assert len(candidates) == len(references)

        if not self.record_level_scores:
            self.record_level_scores = await self.compute_record_level_scores(candidates, references, task_name,
                                                                              model_name)

        res = {}
        for name, score_list in self.record_level_scores.items():
            assert isinstance(score_list, list)
            score_list = [score['overall'] for score in score_list if score is not None]
            score = util.smart_round(sum(score_list) / len(score_list), 2) if len(score_list) else 0.0
            res[name] = score
        return res

    async def __call__(self, candidates, references, instructions=None, *, task_name: str | None = None,
                       model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        overall = await self.get_score(candidates, references, task_name, model_name)
        if task_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, task_name, model_name, explanations,
                             instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, task_name: str | None = None,
                                          model_name: str | None = None):
        """Compute record-level scores for detailed evaluation."""
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, task_name, model_name)
        # Expect {"score": number, "explanation": str}
        self.explanations = [""] * len(candidates)
        return {self.name: raw_scores}


# ---------------------------------------------------------------------------
# Answer Error Rate (AER) — judges two parallel answer arrays. The transcript
# -> answers extraction step is upstream (CodeSwitchAerPostprocessor).
# ---------------------------------------------------------------------------

class AERJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_aer"
    display_name: str = "Answer Error Rate"
    description: str = "Fraction of questions where the ASR transcript leads to an incorrect answer."
    higher_is_better: bool = False
    range: tuple = (0, 1)
    _prompt_key: str = "aer_comparison_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)
        self.model_responses = []
        self.instructions = None
        self._mismatches: list[int | None] = []
        self._questions: list[int | None] = []

    async def __call__(self, candidates, references, instructions=None, *,
                       task_name=None, model_name=None, model_responses=None):
        self.instructions = instructions
        self.model_responses = model_responses or []
        overall = await self.get_score(candidates, references, task_name, model_name)
        if task_name and model_name:
            scores = self.record_level_scores.get("aer_per_row", [])
            extras = self._build_log_extras(len(scores))
            write_record_log(self, references, candidates, scores, task_name, model_name,
                             instructions=self.instructions, model_responses=self.model_responses,
                             extras=extras)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def get_score(self, candidates, references, task_name=None, model_name=None) -> dict:
        if not self.record_level_scores:
            self.record_level_scores = await self.compute_record_level_scores(
                candidates, references, task_name, model_name
            )

        per_row = self.record_level_scores.get("aer_per_row", [])
        total_miss = sum(m for m in self._mismatches if m is not None)
        total_q = sum(q for q in self._questions if q)
        overall_aer = total_miss / total_q if total_q > 0 else 0.0
        scored = [s for s in per_row if s is not None]
        average_sample_aer = sum(scored) / len(scored) if scored else 0.0

        return {
            "overall_aer": util.smart_round(overall_aer, 4),
            "aer_per_row": util.smart_round(average_sample_aer, 4),
        }

    def _build_log_extras(self, n: int) -> dict:
        """Pull transcript / extracted answers off dataset records for side-by-side logging."""
        contexts = getattr(self, "contexts", None) or []
        transcripts = [contexts[i].get("_aer_transcript", "") if i < len(contexts) else "" for i in range(n)]
        extracted = [contexts[i].get("_aer_extracted_answers", "") if i < len(contexts) else "" for i in range(n)]
        return {"transcript": transcripts, "extracted_answers": extracted}

    async def compute_record_level_scores(self, candidates, references,
                                          task_name=None, model_name=None):
        raw_results = await self._judge_all(candidates, references, task_name, model_name)
        aer_scores: list[float | None] = []
        misses: list[int | None] = []
        qs: list[int | None] = []
        for cand_json, raw in zip(candidates, raw_results):
            n_expected = _aer_safe_len(cand_json)
            matches = _aer_parse_bool_array(raw, n_expected)
            if matches is None:
                aer_scores.append(None)
                misses.append(None)
                qs.append(None)
            else:
                m = sum(1 for x in matches if not x)
                aer_scores.append(m / len(matches))
                misses.append(m)
                qs.append(len(matches))
        self._mismatches = misses
        self._questions = qs
        return {"aer_per_row": aer_scores}


def _aer_safe_len(json_array_str: str) -> int:
    try:
        parsed = json.loads(json_array_str)
        return len(parsed) if isinstance(parsed, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


def _aer_parse_bool_array(raw, n_expected: int) -> list[bool] | None:
    if isinstance(raw, list):
        parsed = raw
    elif isinstance(raw, str):
        parsed = _extract_json(raw)
        if parsed is None:
            # Judges sometimes capitalize True/False; lowercase and retry.
            parsed = _extract_json(raw.lower())
    else:
        return None
    if not isinstance(parsed, list):
        return None
    if n_expected and len(parsed) != n_expected:
        return None
    return [bool(x) for x in parsed]


# ---------------------------------------------------------------------------
# Semantic Word Error Rate (SER) — judge counts insertions/deletions/
# substitutions and reports the reference word count. Per-row SER = (S+D+I)/N.
# Aggregates mirror WER: overall_ser (micro) and average_sample_ser (macro).
# ---------------------------------------------------------------------------

class SERJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_ser"
    display_name: str = "Semantic Word Error Rate"
    description: str = "Semantic WER computed from judge-reported insertion/deletion/substitution counts."
    higher_is_better: bool = False
    range: tuple = (0, 1)
    _prompt_key: str = "semantic_wer_prompt"

    def __init__(self, *_, judge_properties: Dict | None = None, **__):
        super().__init__(judge_properties=judge_properties)
        self.instructions = None
        self.model_responses = []
        self._numerators: list[int | None] = []
        self._denominators: list[int | None] = []
        self._judge_raw: list[str] = []
        self._parse_failures: int = 0

    async def __call__(self, candidates, references, instructions=None, *,
                       task_name=None, model_name=None, model_responses=None):
        self.instructions = instructions
        self.model_responses = model_responses or []
        overall = await self.get_score(candidates, references, task_name, model_name)
        if task_name and model_name:
            scores = self.record_level_scores.get("ser_per_row", [])
            extras = {"judge_raw": self._judge_raw}
            write_record_log(self, references, candidates, scores, task_name, model_name,
                             instructions=self.instructions, model_responses=self.model_responses,
                             extras=extras)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    async def get_score(self, candidates, references, task_name=None, model_name=None) -> dict:
        if not self.record_level_scores:
            self.record_level_scores = await self.compute_record_level_scores(
                candidates, references, task_name, model_name
            )

        per_row = self.record_level_scores.get("ser_per_row", [])
        total_num = sum(n for n in self._numerators if n is not None)
        total_den = sum(d for d in self._denominators if d)
        overall_ser = total_num / total_den if total_den > 0 else 0.0
        scored = [s for s in per_row if s is not None]
        average_sample_ser = sum(scored) / len(scored) if scored else 0.0

        return {
            "overall_ser": util.smart_round(overall_ser, 4),
            "average_sample_ser": util.smart_round(average_sample_ser, 4),
            "ser_parse_failures": self._parse_failures,
            "ser_rows_scored": len(scored),
        }

    async def compute_record_level_scores(self, candidates, references,
                                          task_name=None, model_name=None):
        raw_results = await self._judge_all(candidates, references, task_name, model_name)
        per_row: list[float | None] = []
        nums: list[int | None] = []
        dens: list[int | None] = []
        raw_str: list[str] = []
        failures = 0

        for raw in raw_results:
            raw_str.append(raw if isinstance(raw, str) else ("" if raw is None else json.dumps(raw, ensure_ascii=False)))
            counts = _ser_parse_counts(raw)
            if counts is None:
                per_row.append(None)
                nums.append(None)
                dens.append(None)
                failures += 1
                continue
            s, d, i, n = counts
            num = s + d + i
            per_row.append(num / n)
            nums.append(num)
            dens.append(n)

        self._numerators = nums
        self._denominators = dens
        self._judge_raw = raw_str
        self._parse_failures = failures
        return {"ser_per_row": per_row}


def _ser_parse_counts(raw) -> tuple[int, int, int, int] | None:
    """Extract (substitutions, deletions, insertions, reference_words) or None."""
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        obj = _extract_json(raw)
    else:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        s = int(obj["substitutions"])
        d = int(obj["deletions"])
        i = int(obj["insertions"])
        n = int(obj["reference_words"])
    except (KeyError, TypeError, ValueError):
        return None
    if n <= 0 or s < 0 or d < 0 or i < 0:
        return None
    return s, d, i, n
