# 📏 Metrics Overview

This document provides a summary and detailed explanation of all evaluation metrics used in the framework. <br />
For more detailed documentation regarding which metrics can be used for which tasks and task categories, refer to [Task Config Overview](../tasks/README.md).

**NOTE** For consistency across metrics, the final reported score of each supported metric is standardized within the range of **[0.0, 100.0]** with **2-decimal precision**.

---

## 📊 Metric Overview Table

| Metric Name                | Description                                      | Reported metric values                                |
|---------------------------|--------------------------------------------------|-----------------------------------------------|
| `word_error_rate_metrics` (&darr;) | Measures ASR errors via insertions, deletions | average_sample_wer <br /> overall_wer |
| `diarization_metrics` (&darr;) | LLM-Adaptive diarization-relevent metrics | avg_sample_wder <br /> overall_wder <br /> avg_sample_cpwer <br /> overall_cpwer <br /> avg_speaker_count_absolute_error |
| `llm_judge_binary` (&uarr;)        | Binary LLM-based correctness judgment            | llm_judge_binary                 |
| `llm_judge_detailed` (&uarr;)      | Detailed scoring across multiple dimensions      | llm_judge_detailed            |
| `llm_judge_big_bench_audio` (&uarr;) | LLM-based evaluations for BigBench-like tasks           | llm_judge_big_bench_audio                                      |
| `llm_judge_redteaming` (&uarr;)      | LLM-based evaluations for red-teaming/ safety      | llm_judge_redteaming            |
| `mt_bench_llm_judge` (&uarr;)    | LLM-based evaluation for Multi-turn systems (i.e. MT-Bench)                           | mt_bench_llm_judge                            |
| `bleu` (&uarr;)                    | N-gram overlap score                             | bleu                    |
| `bertscore` (&uarr;)               | Semantic similarity using BERT embeddings        | bertscore                       |
| `comet` (&uarr;)               | Semantic similarity measure for translation tasks        | comet                       |
| `meteor` (&uarr;)                  | Alignment-based score with synonym handling      | meteor                    |
| `bfcl_match_score` (&uarr;)        | Structured logic form comparison                 | bfcl_match_score                          |
| `sql_score` (&uarr;)               | SQL correctness and execution match              | text2sql_score                                 |
| `instruction_following` (&uarr;)   | LLM-judged instruction following capability                 | final              |
| `gsm8k_exact_match` (&uarr;)   | Exact-match accuracy of the final numerical answer.             | gsm8k_exact_match              |
| `joint_goal_accuracy` (&uarr;)   | Dialogue state tracking - all slots match             | joint_goal_accuracy              |
| `slot_accuracy` (&uarr;)   | Dialogue state tracking - per-slot accuracy             | slot_accuracy              |
| `slot_f1` (&uarr;)   | Dialogue state tracking - slot extraction F1             | slot_f1              |
| `multiple_choice_accuracy` (&uarr;)   | Accuracy of prediction the correct option letter in multiple choice tasks     | multiple_choice_accuracy              |

--- 

## 📋 Metric Details

### `word_error_rate_metrics`
- **Type**: Speech recognition metric
- **Description**: Measure the correctness of the generated hypothesis vs reference transcript
- **Reported Value**: 
    - `average_sample_wer`: Averaging WER of each sample across the evaluated dataset samples
    - `overall_wer`: (Total deletions, insertions and substitutions) / Total words
- **Scoring (record-level)**: Scoring between 0.0 and 1.0 (lower is better)
- **Used In**: `asr`, `long_form_asr`, `code_switching_asr`

---

### `diarization_error_rate_metrics`
- **Type**: LLM-Adaptive Diariziation-relevant metric
- **Description**: Measure the diarization-relevant metrics for the generated LLM-generated hypothesis. Metrics are mostly computed for who-spoke-what to avoid the requirements of exact timestamp predictions.
- **Reported Value**:
    - `average_sample_wder`: Averaging WDER of each sample across the evaluated dataset samples
    - `overall_wder`: Overall Errors / Overall Total Counts
    - `avg_sample_cpwer`: Averaging cpWER of each sample across the evaluated dataset samples
    - `overall_cpwer`: Overall Errors / Overall Total Counts
    - `avg_speaker_count_absolute_error`:  Mean absolute errors (MAE) of the predicted number of speakers
- **Scoring (record-level)** Scoring between 0.0 and 1.0 (lower is better)
- **Used In**: `speaker_diarization`

---

### `llm_judge_binary`
- **Type**: Binary classification metric
- **Description**: Judges whether a model output is correct or not using an LLM.
- **Scoring (record-level)** `1` for correct, `0` for incorrect. Higher is better.
- **Used In**: `emotion_recognition`, `accent_recognition`, `gender_recognition`, `intent_classification`,`spoofing`

---

### `llm_judge_detailed`
- **Type**: Multi-dimensional judgment metric
- **Description**: Uses an LLM to assess output quality based on attributes like fluency, relevance, and completenes (with or without ground truth reference)
- **Scoring (record-level)** Scoring between `0` and `5` for each sample. Higher is better.
- **Used In**: `spoken_dialogue_summrization`, `scene_understanding`

---

### `llm_judge_big_bench_audio`
- **Type**: LLM-based QA judgment metric
- **Description**: Evaluates performance on BigBench-like audio QA tasks.
- **Scoring (record-level)** Scoring `correct` or `incorrect` based on different aspects of QA tasks. Higher is better
- **Used In**: `sqqa`

---

### `llm_judge_redteaming`
- **Type**: LLM-based judgement metric for red-teaming/ safety.
- **Description**: Evaluates performance on the safety-related aspects for LALMs.
- **Scoring (record-level)** Scoring `1` for refusing to answer the given audio (correct), `0` for anwering the given audio (incorrect). Higher is better.
- **Used In**: `safety`

---

### `mt_bench_llm_judge`
- **Type**: LLM-based judgement metric for Multi-turn systems. 
- **Description**: Evaluates performance at multiple-turn conversation systems
- **Scoring (record-level)** Scoring between `0` and `10` for each sample. Higher is better.
- **Used In**: `mtbench`

---

### `bleu`
- **Type**: N-gram precision metric
- **Description**: Measures how many n-grams in the prediction match the reference.
- **Scoring (record-level)** Score between `0` and `100`, higher is better.
- **Used In**: `translation` 

---

### `bertscore`
- **Type**: Semantic similarity
- **Description**: Uses contextual BERT embeddings to match tokens semantically.
- **Scoring (record-level)** Outputs F1 (between `0` and `1`), higher is better.
- **Used In**: `translation`

---

### `comet`
- **Type**: Semantic similarity for translation tasks
- **Description**: Uses contextual embeddings to compute semantic similarity between source and target language pair
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: `translation`

---

### `meteor`
- **Type**: Alignment metric
- **Description**: Improves on BLEU by considering synonyms, stemming, and paraphrase.
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: `translation`

---

### `bfcl_match_score`
- **Type**: Function calling metric
- **Description**: Evaluate the function calling capabilities. 
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: Speech Function calling (`bfcl`)

---

### `sql_score`
- **Type**: Coding correctness metric
- **Description**: Evaluate the correctness of the generated SQL. 
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: Speech-to-SQL-coding (`speech_to_sql`)

---

### `instruction_following`
- **Type**: Instruction following evaluation metric
- **Description**: Measure the instruction following capabilities of LALMs by averaging accuracy across (1) strict-prompt, (2) strict-instruction, (3)loose-prompt and (4) loose-instruction. 
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: Audio Instruction Following (`ifeval`)

---

### `gsm8k_exact_match`
- **Type**: Math correctness metric
- **Description**: Measure the exact-match accuracy of the final numerical answer (expected within `\boxed{}`) with the reference numerical answer.
- **Scoring (record-level)** Score between `0` and `100`, higher is better.
- **Used In**: Math (`gsm8k`)

---

### `joint_goal_accuracy`
- **Type**: Dialogue state tracking metric
- **Description**: Evaluates whether all predicted slots exactly match the ground truth dialogue state. A sample scores 1 only if every slot-value pair is correct.
- **Scoring (record-level)** Score `0` or `1`, higher is better.
- **Used In**: Task-Oriented Dialogue (`spoken_dialogue`)

---

### `slot_accuracy`
- **Type**: Dialogue state tracking metric
- **Description**: Computes the proportion of individual slots correctly predicted across all samples.
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: Task-Oriented Dialogue (`spoken_dialogue`)

---

### `slot_f1`
- **Type**: Dialogue state tracking metric
- **Description**: Computes F1 score for slot value extraction, balancing precision and recall of predicted slot-value pairs.
- **Scoring (record-level)** Score between `0` and `1`, higher is better.
- **Used In**: Task-Oriented Dialogue (`spoken_dialogue`)
---

### `multiple_choice_accuracy`
- **Type**: Multiple choice accuracy metric
- **Description**: Measure the accuracy of prediction the correct option letter in multiple choice tasks. The correct option is expected in the format `Answer: A`
- **Scoring (record-level)** Score between `0` and `100`, higher is better.
- **Used In**: Audio Instruction Following (`ifeval`)