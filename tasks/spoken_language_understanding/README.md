# 🧠 Spoken Language Understanding
Spoken Language Understanding refers to the task category involving the process of understanding the knowledge extracted from spoken language. Unlike Audio Understanding,  the objective of this task category is to extract semantic information from the input speech input.

## Sample Config
```yaml
dataset_metric:  
  # All datasets within **translation** sub-task category
  - ['translation', 'bleu']

  # Individual dataset
  - ["covost2_en_zh-CN", "bleu"]
```

## 📊 Supported Datasets for Spoken Language Understanding

| Dataset Name                   | Task type       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **COVOST2**               | Translation          | [translation/covost2](./translation/base.yaml)|  Large-scale multilingual speech translation corpus       |    CC-BY-NC-4.0     |
| **SLURP**               | Intent Classification          | [intent_classification/SLURP](./intent_classification/SLURP-intent.yaml)| Multi-domain spoken dialogue understanding benchmark    |    CC BY-NC 4.0     |
| **BIG_BENCH_AUDIO**               | SQQA          | [sqqa/big_bench_audio](./sqqa/big_bench_audio/base.yaml)| Benchmark evaluating the reasoning capabilities of models that support audio and text input.       |    MIT    |
| **MMSU**               | SQQA          | [sqqa/mmsu](./sqqa/mmsu/base.yaml)|   Multi-choice Question Answering dataset    |    Apache-2.0     |
| **OPENBOOKQA**               | SQQA          | [sqqa/openbookqa](./sqqa/openbookqa/openbookqa.yaml)| Multi-choice Question Answering dataset     |     Apache-2.0     |
| **SD-QA**               | SQQA          | [sqqa/sd-qa](./sqqa/sd-qa/base.yaml)| Multi-choice Question Answering dataset      |     Apache-2.0    |
| **MNSC_SQA**               | Speech QA          | [speech_qa/mnsc_sqa](./speech_qa/mnsc_sqa/base.yaml)| Comprehensive benchmark designed specifically for understanding and reasoning in spoken language      |    NSC License    |
| **CN_COLLEGE_LISTEN_MCQ**               | Speech QA          | [speech_qa/cn_college_listen_mcq](./speech_qa/cn_college_listen_mcq_test.yaml)| Multimodal and multi-speaker dataset with humans' emotion expression elicitation      |    MERaLiON Public License     |
| **DREAM_TTS_MCQ**               | Speech QA          | [speech_qa/dream_tts_mcq](./speech_qa/dream_tts_mcq_test.yaml)| Dialogue-based multiple-choice reading comprehension dataset with audio support      |    MIT     |
| **SLUE_SQA**               | Speech QA          | [speech_qa/slue_sqa](./speech_qa/slue_p2_sqa5_test.yaml)| Spoken Language Understanding Evaluation (SLUE) benchmark      |    CC-BY-4.0     |
| **OPENHERMES**               | Speech QA          | [speech_qa/openhermes](./speech_qa/openhermes_instruction_test.yaml)| Speech-based Question Answering benchmark with audio instruction following capability testing     |    CC-BY-NC     |
| **ALPACA_AUDIO**               | Speech QA          | [speech_qa/alpaca_audio](./speech_qa/alpaca_audio_test.yaml)| Speech-based Question Answering benchmark with audio instruction following capability testing      |    Apache-2.0     |
| **PUBLIC_SG**               | Speech QA          | [speech_qa/public_sg](./speech_qa/public_sg_speech_qa_test.yaml)|  Speech Question Answering becnhmark     |    NSC License    |
| **SPOKEN_SQUAD**               | Speech QA          | [speech_qa/spoken_squad](./speech_qa/spoken_squad_test.yaml)|  Extraction-based Speech QA task    |    CC-BY-SA-4.0    |
| **SPOKENWOZ**               | Spoken Dialogue          | [spoken_dialogue/spokenwoz](./spoken_dialogue/spokenwoz/base.yaml)| Large-scale speech-text benchmark for task-oriented dialogue agents. Metrics: `joint_goal_accuracy`, `slot_accuracy`, `slot_f1`, `bleu`      |    CC-BY-NC-4.0    |
  