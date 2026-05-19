from collections import defaultdict

from jiwer import process_words

from metrics.metrics import Metrics
from metrics.word_error_rate_metrics import normalize_text


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
