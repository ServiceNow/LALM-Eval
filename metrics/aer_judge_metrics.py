"""Answer Error Rate (AER) judge metric.

The transcript-to-answers extraction is done upstream by
``CodeSwitchAerPostprocessor``. This metric only compares the two answer
arrays (predicted vs reference) and returns the fraction of mismatches.
"""
import json
import logging
import re

from metrics.llm_judge import _BaseLLMJudge
from utils.custom_logging import write_record_log, append_final_score

logger = logging.getLogger(__name__)


class AERJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_aer"
    display_name: str = "Answer Error Rate"
    description: str = "Fraction of questions where the ASR transcript leads to an incorrect answer."
    higher_is_better: bool = False
    range: tuple = (0, 1)
    _prompt_key: str = "aer_comparison_prompt"

    def __init__(self, *_, judge_properties=None, **__):
        super().__init__(judge_properties=judge_properties)
        self.model_responses = []
        self.instructions = None

    async def __call__(self, candidates, references, instructions=None, *,
                       task_name=None, model_name=None, model_responses=None):
        self.instructions = instructions
        self.model_responses = model_responses or []
        overall = await super().get_score(candidates, references, task_name, model_name)
        if task_name and model_name:
            scores = self.record_level_scores.get("aer_per_row", [])
            extras = self._build_log_extras(len(scores))
            write_record_log(self, references, candidates, scores, task_name, model_name,
                             instructions=self.instructions, model_responses=self.model_responses,
                             extras=extras)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    def _build_log_extras(self, n: int) -> dict:
        """Pull transcript / extracted answers off dataset records for side-by-side logging."""
        contexts = getattr(self, "contexts", None) or []
        transcripts = [contexts[i].get("_aer_transcript", "") if i < len(contexts) else "" for i in range(n)]
        extracted = [contexts[i].get("_aer_extracted_answers", "") if i < len(contexts) else "" for i in range(n)]
        return {"transcript": transcripts, "extracted_answers": extracted}

    async def compute_record_level_scores(self, candidates, references,
                                          task_name=None, model_name=None):
        raw_results = await self._judge_all(candidates, references, task_name, model_name)
        aer_scores = []
        for cand_json, raw in zip(candidates, raw_results):
            n_expected = _safe_len(cand_json)
            matches = _parse_bool_array(raw, n_expected)
            if matches is None:
                aer_scores.append(None)
            else:
                aer_scores.append(sum(1 for m in matches if not m) / len(matches))
        return {"aer_per_row": aer_scores}


def _safe_len(json_array_str: str) -> int:
    try:
        parsed = json.loads(json_array_str)
        return len(parsed) if isinstance(parsed, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


def _parse_bool_array(raw, n_expected: int) -> list[bool] | None:
    if isinstance(raw, list):
        return [bool(x) for x in raw] if (not n_expected or len(raw) == n_expected) else None
    if not isinstance(raw, str):
        return None
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group().lower())
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    if n_expected and len(parsed) != n_expected:
        return None
    return [bool(x) for x in parsed]
