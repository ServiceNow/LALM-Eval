"""Word Error Rate (WER) metrics implementation.

This module provides WER calculation capabilities with text normalization,
language-specific handling, and detailed scoring breakdowns.
"""
import logging
import re
import unicodedata
from collections import defaultdict

from jiwer import process_words
from num2words import num2words
from tqdm import tqdm

from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score
from utils import constants, util

logger = logging.getLogger(__name__)


def convert_unicode_to_characters(text: str) -> str:
    """Convert unicode to composed form."""
    try:
        return unicodedata.normalize("NFC", text)
    except Exception as e:
        # Optionally log the error
        logger.warning("Unicode normalization failed: %s. Returning original text.", e)
        return text


def convert_digits_to_words(text: str, language: str):
    """Convert numbers to words (e.g., "3" to "three")."""
    if not language:
        return text
    try:
        return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=language), text)
    except Exception as e:
        logger.info("Failed to convert digits to words for language %s - continuing...", language)
        logger.warning("Non-fatal error: %s - continuing...", e)
        return text


def normalize_text(text: str, language: str = 'en') -> str:
    """Normalize text based on language.

    Args:
        text: input text
        language: language code (e.g. 'en', 'es')
    """
    # Use language code directly without conversion
    # Get the appropriate normalizer
    normalizer = constants.NORMALIZERS.get(language, constants.DEFAULT_NORMALIZER)

    # Process the text
    text = convert_unicode_to_characters(text)
    text = convert_digits_to_words(text, language)
    return constants.BASIC_TRANSFORMATIONS([normalizer(text)])[0]


class WERMetrics(Metrics):
    """Word Error Rate metrics implementation.
    
    Computes WER scores with text normalization and language-specific handling.
    Provides overall, per-conversation, and length-bucketed WER calculations.
    """
    def __call__(self, candidates, references, ids=None, lengths=None, instructions=None, *, task_name: str | None = None, model_name: str | None = None, model_responses=None):
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []

        overall = self.get_score(candidates, references, ids, lengths)
        if task_name and model_name:
            # WER record scores are stored under 'wer_per_row'
            scores = self.record_level_scores.get("wer_per_row", [])
            write_record_log(self, references, candidates, scores, task_name, model_name, 
                          instructions=self.instructions, model_responses=self.model_responses)
            append_final_score(self, overall, task_name, model_name, self.model_responses)
        return overall

    def __init__(self, language="en"):
        super().__init__()
        self.name = "word_error_rate"
        self.lower_better = True
        # Use language code directly without conversion
        self.language = language
        self.description = "The proportion of words that are incorrectly predicted, when compared to the reference text. The dataset is considered as one big conversation."
        self.instructions = None
        self.model_responses = []

    def compute_attributes(self, incorrect: list[int | float], total: list[int | float], attributes: list[str]) -> dict:
        """Compute the attributes (e.g., accent, gender) that should be saved in the record level file for analysis."""
        results = {}
        for attribute in attributes:
            current_attr = self.record_level_scores.get(attribute, [])
            incorrect_per_attr = defaultdict(int)
            total_per_attr = defaultdict(int)
            for _incorrect, _total, attr_value in zip(incorrect, total, current_attr):
                if attr_value:
                    incorrect_per_attr[attr_value] += _incorrect
                    total_per_attr[attr_value] += _total

            for attr in incorrect_per_attr:
                total_attr = total_per_attr.get(attr, 0)
                if total_attr:
                    results[f"wer_{attribute}_{attr}"] = incorrect_per_attr[attr] / total_attr
        return results

    def get_score(self, candidates, references, ids=None, lengths=None):
        """Get overall score.

        Args:
            candidates: generated text list
            references: reference text list
            ids: optional list of conversation IDs (first 4 letters)
            lengths: optional list of audio sample lengths in seconds

        Returns:
            Dict with WER metrics by overall, conversation, and length buckets
        """
        scores = self.compute_record_level_scores(candidates, references)

        # Compute the overall WER
        incorrect_chars = sum(scores["incorrect"])
        total_chars = sum(scores["total"])
        # Overall WER is the sum of incorrect divided by sum of total
        overall_wer = incorrect_chars / total_chars if total_chars > 0 else 0

        # We also track per-sample average for a more balanced view
        avg_sample_wer = sum(scores["wer_per_row"]) / len(scores["wer_per_row"]) if scores["wer_per_row"] else 0

        # Initialize the result with both WER metrics
        result = {
            "average_sample_wer": util.smart_round(avg_sample_wer * 100.0, 2),
            "overall_wer": util.smart_round(overall_wer * 100.0, 2)
        }

        if ids and len(ids) == len(scores["wer_per_row"]):
            conversation_wer = {}
            # Group WERs by conversation ID
            id_to_wers = defaultdict(list)
            id_to_incorrect = defaultdict(int)
            id_to_total = defaultdict(int)

            for i, conv_id in enumerate(ids):
                if i < len(scores["wer_per_row"]) and i < len(scores["incorrect"]) and i < len(scores["total"]):
                    id_to_wers[conv_id].append(scores["wer_per_row"][i])
                    id_to_incorrect[conv_id] += scores["incorrect"][i]
                    id_to_total[conv_id] += scores["total"][i]

            # Calculate average WER for each conversation ID
            for conv_id in id_to_wers:
                # Using ratio of sums for conversation WER
                conv_wer = id_to_incorrect[conv_id] / id_to_total[conv_id] if id_to_total[conv_id] > 0 else 0
                conversation_wer[conv_id] = conv_wer

            result["conversation_wer"] = conversation_wer

        # If lengths are provided, calculate WER by length buckets
        if lengths and len(lengths) == len(scores["wer_per_row"]):
            # Define length buckets
            buckets = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, float('inf'))]
            bucket_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
            length_wer = {}

            # Group WERs by length bucket
            bucket_to_incorrect = {label: 0 for label in bucket_labels}
            bucket_to_total = {label: 0 for label in bucket_labels}

            for i, length in enumerate(lengths):
                if i < len(scores["wer_per_row"]) and i < len(scores["incorrect"]) and i < len(scores["total"]):
                    # Find which bucket this length belongs to
                    bucket_idx = next((j for j, (min_len, max_len) in enumerate(buckets)
                                       if min_len <= length < max_len), len(buckets) - 1)
                    bucket_label = bucket_labels[bucket_idx]

                    bucket_to_incorrect[bucket_label] += scores["incorrect"][i]
                    bucket_to_total[bucket_label] += scores["total"][i]

            # Calculate WER for each length bucket
            for bucket_label in bucket_labels:
                if bucket_to_total[bucket_label] > 0:
                    bucket_wer = bucket_to_incorrect[bucket_label] / bucket_to_total[bucket_label]
                    length_wer[bucket_label] = bucket_wer
                else:
                    length_wer[bucket_label] = 0.0

            result["length_wer"] = length_wer

        # Store the scores for later record level reporting
        # Important to use setdefault which is a no-op if the value already exists
        # As users can evaluate multiple models and call compute_record_level_scores multiple times
        self.record_level_scores.setdefault("wer_per_row", scores["wer_per_row"])
        self.record_level_scores.setdefault("incorrect", scores["incorrect"])
        self.record_level_scores.setdefault("total", scores["total"])
        return result

    def compute_record_level_scores(self, candidates: list, references: list):
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """
        incorrect_scores = []
        total_scores = []
        scores = []
        references_clean = []
        candidates_clean = []

        for reference, candidate in tqdm(zip(references, candidates), desc="word_error_rate", total=len(references)):
            # Use the normalized language code from instance variable
            references_clean.append(normalize_text(reference, self.language))
            candidates_clean.append(normalize_text(candidate, self.language))
            if references_clean[-1].strip() == "":
                logger.warning(
                    "After normalization, '%s' is empty. Considering all words in '%s' as incorrect.",
                    reference, candidate
                )
                incorrect_scores.append(len(candidates_clean[-1].split()))
                total_scores.append(1)
            else:
                kwargs = (
                    {kwarg: constants.CER_DEFAULTS for kwarg in ("truth_transform", "hypothesis_transform")}
                    if self.language in constants.CER_LANGUAGES
                    else {}
                )
                measures = process_words(references_clean[-1], candidates_clean[-1], **kwargs)

                # Newer jiwer returns a dataclass-like object with attributes
                substitutions = measures.substitutions
                deletions = measures.deletions
                insertions = measures.insertions
                hits = measures.hits

                incorrect_scores.append(substitutions + deletions + insertions)
                total_scores.append(substitutions + deletions + hits)
            wer = incorrect_scores[-1] / total_scores[-1]
            scores.append(wer)

        results = {
            "wer_per_row": scores,
            "candidates_clean": candidates_clean,
            "references_clean": references_clean,
            "incorrect": incorrect_scores,
            "total": total_scores,
        }
        accents = [record.get("accent") for record in self.contexts]
        gender = [record.get("gender") for record in self.contexts]
        if any(accents):
            results["accent"] = accents
        if any(gender):
            results["gender"] = gender
        return results


class CodeSwitchedWERMetrics(Metrics):
    """Per-language and CMI-bucketed WER for code-switching ASR.

    Expects each record in self.contexts to have:
      - words: list[str]          — reference words (from HF dataset column)
      - word_languages: list[str] — language tag per word (e.g. "EN", "ES")
    """

    name: str = "codeswitched_wer"
    display_name: str = "Code-Switched WER"
    description: str = "Per-language and CMI-bucketed WER for code-switching ASR."
    higher_is_better: bool = False
    range: tuple = (0, 1)

    _CMI_BUCKETS = [
        ("low", 0.0, 25.0),
        ("medium", 25.0, 40.0),
        ("high", 40.0, 101.0),
    ]

    def __call__(self, candidates, references, ids=None, lengths=None, instructions=None, *, task_name=None, model_name=None, model_responses=None):
        return self.get_score(candidates, references, ids, lengths)

    def get_score(self, candidates, references, ids=None, lengths=None):
        """Compute aggregate per-language and CMI-bucketed WER."""
        lang_stats: dict[str, dict] = defaultdict(lambda: {"errors": 0, "total": 0})
        bucket_stats = {label: {"errors": 0, "total": 0, "count": 0} for label, *_ in self._CMI_BUCKETS}
        cross_stats: dict[tuple, dict] = defaultdict(lambda: {"errors": 0, "total": 0})
        records_counted = 0

        for i, (candidate, reference) in enumerate(zip(candidates, references)):
            ctx = self.contexts[i] if i < len(self.contexts) else {}
            words = ctx.get("words")
            word_languages = ctx.get("word_languages")

            if not words or not word_languages or not reference or not candidate:
                continue

            cmi = self._compute_cmi(word_languages)
            if cmi is None:
                continue

            bucket_label = next(
                (label for label, lo, hi in self._CMI_BUCKETS if lo <= cmi < hi),
                None,
            )
            if bucket_label is None:
                continue

            norm_ref = normalize_text(reference, "en")
            norm_hyp = normalize_text(candidate, "en")

            position_langs = []
            for word, lang in zip(words, word_languages):
                n_tokens = max(len(normalize_text(word, "en").split()), 1)
                position_langs.extend([lang] * n_tokens)

            try:
                alignment = process_words(norm_ref, norm_hyp).alignments[0]
            except Exception:
                continue

            for chunk in alignment:
                is_error = chunk.type != "equal"
                span = chunk.ref_end_idx - chunk.ref_start_idx
                bucket_stats[bucket_label]["total"] += span
                if is_error:
                    bucket_stats[bucket_label]["errors"] += span

                for ref_idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if ref_idx >= len(position_langs):
                        continue
                    lang = position_langs[ref_idx]
                    lang_stats[lang]["total"] += 1
                    cross_stats[(bucket_label, lang)]["total"] += 1
                    if is_error:
                        lang_stats[lang]["errors"] += 1
                        cross_stats[(bucket_label, lang)]["errors"] += 1

            bucket_stats[bucket_label]["count"] += 1
            records_counted += 1

        if not lang_stats:
            return {}

        scores: dict = {"cs_wer_records_counted": records_counted}

        for lang, s in lang_stats.items():
            t = s["total"]
            e = s["errors"]
            scores[f"{lang}_wer"] = round(e / t, 4) if t else 0.0
            scores[f"{lang}_error_words"] = e
            scores[f"{lang}_total_words"] = t

        for label, s in bucket_stats.items():
            t = s["total"]
            e = s["errors"]
            scores[f"cmi_{label}_wer"] = round(e / t, 4) if t else 0.0
            scores[f"cmi_{label}_error_words"] = e
            scores[f"cmi_{label}_total_words"] = t
            scores[f"cmi_{label}_num_records"] = s["count"]

        for (label, lang), s in cross_stats.items():
            t = s["total"]
            e = s["errors"]
            scores[f"cmi_{label}_{lang}_wer"] = round(e / t, 4) if t else 0.0

        return scores

    def compute_record_level_scores(self, candidates: list, references: list) -> dict[str, list]:
        """Required by the abstract base class. Aggregate metrics live in get_score."""
        return {"codeswitched_wer_placeholder": [None] * len(candidates)}

    @staticmethod
    def _compute_cmi(word_languages: list[str]) -> float | None:
        """Code Mixing Index: fraction of non-dominant-language words × 100."""
        if not word_languages:
            return None
        counts: dict[str, int] = defaultdict(int)
        for lang in word_languages:
            counts[lang] += 1
        if len(counts) == 1:
            return 0.0
        total = sum(counts.values())
        dominant = max(counts.values())
        return (total - dominant) / total * 100
