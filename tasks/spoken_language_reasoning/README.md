# 🧩 Spoken Language Reasoning
Spoken Language Reasoning refers to the task category involving performing complex, logical operations on information conveyed through speech. These tasks go beyond simple transcription or understanding a single utterance, as they require the model to comprehend context, make inferences, user external tools, folow complex multi-step instructions and answer questions that demand a deeper understanding of the spoken content.

## Sample Config
```yaml
dataset_metric:
  
  # All datasets within **scene_understanding** sub-task category
  - ['mtbench', 'mt_bench_llm_judge']

  # Individual dataset
  - ["mtbench_audio", "mt_bench_llm_judge"]
```
## SPEECH_TO_SQL
Before running evaluations for **SPEECH_TO_SQL** task, it is required to follow the installation steps using the script in [`data/scripts/download_spider.sh`](../../data/scripts/download_spider.sh). The `data/spider/` directory and its subsidiary files would be created if the installation completes successfully. Failure to install these will result in the inability to run evaluations of 
  - `[spider_audio, sql_score]`
  - `[spider_text, sql_score]`

 More specifically, execute the below before running the evaluation for `speech_to_sql`.

```bash
cd AU-Harness/
bash data/scripts/downnload_spider.sh
```

## 📊 Supported Datasets for Spoken Language Reasoning

| Dataset Name                   | Task type       | config | Description                                                                                       | License              |
|-------------------------------|------------------|----- | ---------------------------------------------------------------------------------------------------|----------------------|
| **MTBench**               | Speech Instruction Following          | [spoken_language_reasoning/mtbench](./mtbench/base.yaml)| Speech-based multi-turn complex instruction following dataset      |    Apache-2.0     |
| **IFEVAL**               | Speech Instruction Following          | [spoken_language_reasoning/ifeval](./ifeval/base.yaml)| Speech-based complex instruction following dataset    |    Apache-2.0     |
| **BFCL**               | Speech Function Calling          | [spoken_language_reasoning/bfcl](./bfcl/base.yaml)| Speech-based complex function calling dataset with audio input       |    Apache-2.0    |
| **SPEECH_TO_SQL**               | Speech-to-Coding         | [spoken_language_reasoning/speech_to_sql](./speech_to_sql/base.yaml)| Speech-based dataset involving following instructions to produce executable code        |    Apache-2.0     |
| **GSM8k**               | Grade School Math         | [spoken_language_reasoning/gsm8k](./gsm8k/base.yaml)| Speech-based math dataset with grade school math word problems       |    MIT (text dataset)     |
| **GPQA Diamond**               | Grade School Math         | [spoken_language_reasoning/gpqa_diamond](./gpqa_diamond/base.yaml)| Speech based questions considered difficult, written and validated by experts in biology, physics, and chemistry.       |   cc-by-4.0     |