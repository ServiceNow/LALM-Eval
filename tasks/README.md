# 📊 Task Overview

## 📝 Summary

This framework supports structured **evaluation runs** across a range of **Task Categories** and **Task Names**. Each valid combination corresponds to an accepted configuration in your `config.yaml` file. For clarity, each task lists its associated **Supported Metrics**, which are automatically handled by the evaluation pipeline.

Use this documentation to configure and understand supported evaluation targets and their corresponding metrics.

For more detailed documentation regarding individual metrics, refer to [Metrics Overview](../metrics/README.md).


---

## 🧾 Task Category Descriptions

| Task Category                     | Description                                                                                 |
|----------------------------------|---------------------------------------------------------------------------------------------|
| **🗣️ speech_recognition**         | Tasks involving automatic speech recognition (ASR), including standard ASR, long-form ASR, and code-switching ASR. |
| **🎭 paralinguistics**            | Tasks that analyze non-verbal aspects of speech such as emotion, gender, accent, and speaker characteristics. |
| **🔊 audio_understanding** | Tasks that require understanding of the general audio signals including but not limited to music, noise, sound. |
| **🧠 spoken_language_understanding** | Tasks that require understanding of spoken language and/or audio information including QA, translation, summarization, and intent classification. |
| **🧩 spoken_language_reasoning**  | Tasks that require reasoning over spoken input, such as instruction following or logical/mathematical reasoning. |
| **🔐 safety_and_security**        | Tasks related to assessing model behavior around safety, robustness, and vulnerability to spoofing or adversarial content. |

---

## ✅ Supported Tasks and Metrics

| Task Category                     | Task Name                          | Supported Metrics                                 |
|----------------------------------|------------------------------------|--------------------------------------------------|
| `speech_recognition`             | `asr`                              | `word_error_rate`                                |
| `speech_recognition`             | `code_switching_asr`               | `word_error_rate`                                |
| `speech_recognition`             | `long_form_asr`                    | `word_error_rate`                                |
| `paralinguistics`                | `emotion_recognition`              | `llm_judge_binary`                               |
| `paralinguistics`                | `gender_recognition`               | `llm_judge_binary`                               |
| `paralinguistics`                | `accent_recognition`               | `llm_judge_binary`                               |
| `paralinguistics`                | `speaker_recognition`              | `llm_judge_binary`                               |
| `paralinguistics`                | `speaker_diarization`              | `diarization_metrics`                            |
| `audio_understanding`           | `scene_understanding`              | `llm_judge_detailed`, `llm_judge_binary`         |
| `audio_understanding`           | `music_understanding`              | `llm_judge_binary`                               |
| `spoken_language_understanding` | `speech_qa`                        | `llm_judge_binary`, `llm_judge_detailed`         |
| `spoken_language_understanding` | `sqqa`                             | `llm_judge_big_bench_audio`, `llm_judge_binary`  |
| `spoken_language_understanding` | `translation`                      | `bleu`, `bertscore`, `meteor`, `comet`           |
| `spoken_language_understanding` | `spoken_dialogue_summarization`    | `llm_judge_detailed`                             |
| `spoken_language_understanding` | `intent_classification`            | `llm_judge_binary`                               |
| `spoken_language_understanding` | `spoken_dialogue`                  | `joint_goal_accuracy`, `slot_accuracy`, `slot_f1`, `bleu` |
| `spoken_language_reasoning`     | `bfcl`                             | `bfcl_match_score`                               |
| `spoken_language_reasoning`     | `speech_to_sql`                    | `sql_score`                                      |
| `spoken_language_reasoning`     | `ifeval`                           | `instruction_following`                          |
| `spoken_language_reasoning`     | `mtbench`                           | `mt_bench_llm_judge`                            |
| `spoken_language_reasoning`     | `gsm8k`                             | `gsm8k_exact_match`                               |
| `spoken_language_reasoning`     | `gpqa_diamond`                     | `multiple_choice_accuracy`                               |
| `safety_and_security`           | `safety`                           | `detailed_judge_prompt`                          |
| `safety_and_security`           | `spoofing`                         | `detailed_judge_prompt`, `llm_judge_binary`      |

---

🔍 **Notes**
- Only the listed combinations of categories, tasks, and metrics are currently supported.

- Make sure to use exact variable names in your `config.yaml`.

- This framework can be extended with new task types and metric integrations.
