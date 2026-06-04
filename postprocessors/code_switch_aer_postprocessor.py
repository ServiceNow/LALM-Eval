"""Postprocessor for the Code-Switching AER metric.

For the ``llm_judge_aer`` metric, runs an extraction model over each ASR
transcript to derive one answer per dataset question, then hands the metric two
JSON answer arrays (predicted vs reference) for comparison. All other metrics
pass through unchanged from ``GeneralPostprocessor``.
"""
import asyncio
import json
import logging
import re
from pathlib import Path

import yaml
from openai import AsyncAzureOpenAI, AsyncOpenAI

from postprocessors.general_postprocessor import GeneralPostprocessor

logger = logging.getLogger(__name__)

_EXTRACTOR_HELPER_NAME = "aer_extractor"
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts/judge_prompts.yaml"


def _load_extraction_prompt() -> str:
    data = yaml.safe_load(_PROMPT_PATH.read_text()) or {}
    return data["aer_extraction_prompt"]


def _parse_answers(raw: str, n_expected: int) -> list[str] | None:
    """Parse a JSON array of strings, falling back to a numbered list."""
    if not raw:
        return None
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) == n_expected:
                return [str(x) if x is not None else "N/A" for x in parsed]
        except json.JSONDecodeError:
            pass
    items = re.findall(r"^\s*\d+\.\s*(.+)$", raw, re.MULTILINE)
    if len(items) == n_expected:
        return [item.strip() or "N/A" for item in items]
    return None


class CodeSwitchAerPostprocessor(GeneralPostprocessor):
    """Replaces transcripts with extracted-answer JSON for the AER metric only."""

    def __init__(self):
        super().__init__()
        self.helpers: dict = {}
        self._extraction_prompt = _load_extraction_prompt()

    def process(self, dataset: list[dict], predictions, metric) -> dict:
        out = super().process(dataset=dataset, predictions=predictions, metric=metric)
        if metric != "llm_judge_aer":
            return out

        helper = self.helpers.get(_EXTRACTOR_HELPER_NAME)
        if helper is None:
            raise ValueError(
                f"CodeSwitchAerPostprocessor requires a helper named '{_EXTRACTOR_HELPER_NAME}' "
                "in the run config (models: with role: helper)"
            )
        client, model_name = self._build_client(helper)
        concurrency = helper.batch_size or 10

        new_preds = {}
        for run_model_name, transcripts in out["processed_predictions"].items():
            extracted = asyncio.run(
                self._extract_all(client, model_name, concurrency, dataset, transcripts)
            )
            new_preds[run_model_name] = extracted
            # Stash on dataset records so the AER metric can log transcript +
            # extracted side-by-side with the comparison score.
            for idx, (transcript, extracted_json) in enumerate(zip(transcripts, extracted)):
                if idx < len(dataset):
                    dataset[idx]["_aer_transcript"] = transcript
                    dataset[idx]["_aer_extracted_answers"] = extracted_json
        out["processed_predictions"] = new_preds
        out["model_targets"] = [
            json.dumps([str(a) for a in (rec.get("expected_answers") or [])])
            for rec in dataset
        ]
        return out

    @staticmethod
    def _build_client(helper):
        info = helper.model_info
        if info.get("inference_type") == "openai_chat_completion" and info.get("api_version"):
            client = AsyncAzureOpenAI(
                api_key=info.get("auth_token"),
                api_version=info.get("api_version"),
                azure_endpoint=info.get("url"),
            )
        else:
            client = AsyncOpenAI(
                base_url=info.get("url"),
                api_key=info.get("auth_token") or "EMPTY",
            )
        return client, info.get("model")

    async def _extract_all(self, client, model_name, concurrency, dataset, transcripts):
        sem = asyncio.Semaphore(concurrency)

        async def one(idx, transcript):
            questions = dataset[idx].get("questions") or []
            if not questions or not transcript:
                return json.dumps([])
            numbered = "\n".join(f"{j+1}. {q}" for j, q in enumerate(questions))
            user_prompt = f"Text: {transcript}\nQuestions:\n{numbered}\n"
            async with sem:
                raw = await self._call(client, model_name, user_prompt)
            answers = _parse_answers(raw, len(questions))
            if answers is None:
                logger.warning("AER extraction failed for record %d, raw=%r", idx, raw)
                return json.dumps([])
            return json.dumps(answers)

        return await asyncio.gather(*[one(i, t) for i, t in enumerate(transcripts)])

    async def _call(self, client, model_name, user_prompt: str) -> str:
        max_retries = 4
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": self._extraction_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("AER extractor call failed (attempt %d/%d): %s",
                               attempt + 1, max_retries, e)
                await asyncio.sleep(2 ** attempt)
        return ""
