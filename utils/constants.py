"""Constants and mappings for AU-Harness framework.

This module contains configuration constants, mappings between different
components (metrics, models, tasks), and normalization settings used
throughout the evaluation framework.
"""

from jiwer import (
    Compose,
    ReduceToListOfListOfChars,
    RemovePunctuation,
    RemoveWhiteSpace,
    Strip,
    ToLowerCase,
)

from metrics.wer.normalizers import JapaneseTextNormalizer
from metrics.wer.whisper_normalizer.basic import BasicTextNormalizer
from metrics.wer.whisper_normalizer.english import EnglishTextNormalizer

# Inference server types
INFERENCE_SERVER_VLLM_CHAT_COMPLETION = 'vllm'
OPENAI_CHAT_COMPLETION = 'openai'
GEMINI_CHAT_COMPLETION = 'gemini'
TRANSCRIPTION = 'transcription'

# WER/CER metrics constants
# Define WER/CER related constants
NORMALIZERS = {'en': EnglishTextNormalizer(), 'ja': JapaneseTextNormalizer()}
DEFAULT_NORMALIZER = BasicTextNormalizer()
BASIC_TRANSFORMATIONS = Compose(
    [
        ToLowerCase(),
        RemovePunctuation(),
        Strip(),
    ]
)
# CER stands for Character Error Rate
CER_LANGUAGES = {'ja'}
CER_DEFAULTS = Compose(
    [
        RemoveWhiteSpace(),
        ReduceToListOfListOfChars(),
    ]
)

# Other constants
ROUND_DIGITS = 3
VLLM_MAX_TOKEN_RETRY_BUFFER = 50
INVERTED_METRIC_INDICATOR = '↓'

# Default generation params
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_COMPLETION_TOKENS = 4096 

# Dictionary mapping metric names to their implementation details (module, class)
metric_map = {
    'bertscore': ("metrics.bertscore", "BertScore"),
    'bfcl_match_score': ("metrics.bfcl_metric", "BFCLMatchScore"),
    "bleu": ("metrics.bleu_metrics", "BleuMetrics"),
    'diarization_metrics': ("metrics.diarization_metrics", "DiarizationMetrics"),
    'gsm8k_exact_match': ("metrics.gsm8k_metrics", "Gsm8kExactMatch"),
    'instruction_following': ("metrics.voice_bench_ifeval_score", "InstructionFollowingScore"),
    "llm_judge_big_bench_audio": ("metrics.llm_judge", "BigBenchAudioLLMJudgeMetric"),
    "llm_judge_binary": ("metrics.llm_judge", "BinaryLLMJudgeMetric"),
    "llm_judge_callhome": ("metrics.llm_judge", "CallHomeLLMJudgeMetric"),
    "llm_judge_detailed": ("metrics.llm_judge", "DetailedLLMJudgeMetric"),
    "llm_judge_redteaming": ("metrics.llm_judge", "RedTeamingJudgeMetric"),
    "meteor": ("metrics.meteor_score", "MeteorScore"),
    'sql_score': ("metrics.sql_score", "SqlScore"),
    "word_error_rate": ("metrics.word_error_rate_metrics", "WERMetrics"),
    "comet": ("metrics.comet_score", "CometScore"),
    "mt_bench_llm_judge": ("metrics.llm_judge", "MtbenchLLMJudgeMetric"),

}

task_temp_map = {
    # ASR
    "asr": 0.1,
    "code_switching_asr": 0.1,
    "long_form_asr": 0.1,

    # Paralinguistics
    "emotion_recognition": 0.2,
    "gender_recognition": 0.2,
    "accent_recognition": 0.2,
    "speaker_recognition": 0.2,

    # Spoken Language Understanding
    "speech_qa": 0.5,
    "sqqa": 0.5,
    "translation": 0.2,
    "scene_understanding": 0.5,
    "spoken_dialogue_summarization": 0.8,
    "music_understanding": 0.7,

}

mtbench_temp_map = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

allowed_task_metrics = {
    # ASR
    'asr': ['word_error_rate', 'meteor', 'bleu', 'bertscore'],
    'code_switching_asr': ['word_error_rate', 'meteor', 'bleu', 'bertscore'],
    'long_form_asr': ['word_error_rate', 'meteor', 'bleu', 'bertscore'],

    # Paralinguistics
    'emotion_recognition': ['llm_judge_binary'],
    'gender_recognition': ['llm_judge_binary'],
    'accent_recognition': ['llm_judge_binary'],
    'speaker_recognition': ['llm_judge_binary'],
    'speaker_diarization': ['diarization_metrics'],


    # Spoken Language Understanding
    'speech_qa': ['llm_judge_detailed', 'llm_judge_binary'],
    'sqqa': ['llm_judge_big_bench_audio', 'llm_judge_binary'],
    'translation': ['word_error_rate', 'meteor', 'bleu', 'bertscore', 'comet'],
    'scene_understanding': ['llm_judge_binary', 'llm_judge_detailed'],
    'spoken_dialogue_summarization': ['llm_judge_detailed'],
    'intent_classification': ['llm_judge_binary'],
    'music_understanding': ['llm_judge_binary'],

    # Spoken Language Reasoning
    'bfcl': ['bfcl_match_score'],
    'ifeval': ['instruction_following'],
    'speech_to_sql': ['sql_score'],
    'gsm8k': ['gsm8k_exact_match'],
    
    # Safety and Security
    'safety': ['llm_judge_redteaming'],
    'spoofing': ['llm_judge_detailed', 'llm_judge_binary'],
    'mtbench': ['mt_bench_llm_judge'],
}

metric_output = {
    "mt_bench_llm_judge": ["mt_bench_llm_judge"],
    "llm_judge_detailed": ["llm_judge_detailed"],
    "word_error_rate": ["average_sample_wer", "overall_wer"],
    "bertscore": ["bertscore"],
    "bleu": ["BLEU"],
    "llm_judge_callhome": ["llm_judge_callhome"],
    "llm_judge_binary": ["llm_judge_binary"],
    "llm_judge_redteaming": ["llm_judge_redteaming"],
    "llm_judge_big_bench_audio": ["llm_judge_big_bench_audio"],
    "meteor": ["meteor"],
    "bfcl_match_score": ["final"],
    "sql_score": ["sql_score"],  # need to find real metric
    "instruction_following": ["strict_instruction", "loose_instruction", "final"],
    "diarization_metrics": ["average_sample_wder", "overall_wder", "average_sample_cpwer", "overall_cpwer", "speaker_count_absolute_error"],
    "comet": ["comet"]
}

# Dictionary mapping language names to their standard codes
language_map = {
    'ab': 'abkhaz',
    'af': 'afrikaans',
    'am': 'amharic',
    'ar': 'arabic',
    'as': 'assamese',
    'ast': 'asturian',
    'az': 'azerbaijani',
    'ba': 'bashkir',
    'bas': 'basaa',
    'be': 'belarusian',
    'bg': 'bulgarian',
    'bn': 'bengali',
    'br': 'breton',
    'bs': 'bosnian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ckb': 'sorani-kurdish',
    'cmn': 'mandarin chinese',
    'cnh': 'hakha chin',
    'cs': 'czech',
    'cv': 'chuvash',
    'cy': 'welsh',
    'da': 'danish',
    'de': 'german',
    'dv': 'dhivehi',
    'dyu': 'dioula',
    'el': 'greek',
    'en': 'english',
    'eo': 'esperanto',
    'es': 'spanish',
    'et': 'estonian',
    'eu': 'basque',
    'fa': 'persian',
    'ff': 'fula',
    'fi': 'finnish',
    'fil': 'filipino',
    'fr': 'french',
    'fy': 'frisian',
    'ga': 'irish',
    'gl': 'galician',
    'gn': 'guarani',
    'gu': 'gujarati',
    'ha': 'hausa',
    'he': 'hebrew',
    'hi': 'hindi',
    'hr': 'croatian',
    'hsb': 'sorbian, upper',
    'hu': 'hungarian',
    'hy': 'armenian',
    'ia': 'interlingua',
    'id': 'indonesian',
    'ig': 'igbo',
    'is': 'icelandic',
    'it': 'italian',
    'ja': 'japanese',
    'jv': 'javanese',
    'ka': 'georgian',
    'kab': 'kabyle',
    'kam': 'kamba',
    'kea': 'kabuverdianu',
    'kk': 'kazakh',
    'km': 'khmer',
    'kmr': 'kurmanji kurdish',
    'kn': 'kannada',
    'ko': 'korean',
    'ky': 'kyrgyz',
    'lb': 'luxembourgish',
    'lg': 'ganda',
    'ln': 'lingala',
    'lo': 'lao',
    'lt': 'lithuanian',
    'luo': 'luo',
    'lv': 'latvian',
    'mdf': 'moksha',
    'mhr': 'meadow mari',
    'mi': 'maori',
    'mk': 'macedonian',
    'ml': 'malayalam',
    'mn': 'mongolian',
    'mr': 'marathi',
    'mrj': 'hill mari',
    'ms': 'malay',
    'mt': 'maltese',
    'my': 'burmese',
    'myv': 'erzya',
    'nan-tw': 'taiwanese (minnan)',
    'nb': 'norwegian',
    'ne': 'nepali',
    'nl': 'dutch',
    'nn': 'norwegian nynorsk',
    'nso': 'northern-sotho',
    'ny': 'nyanja',
    'oc': 'occitan',
    'om': 'oromo',
    'or': 'oriya',
    'pa': 'punjabi',
    'pl': 'polish',
    'ps': 'pashto',
    'pt': 'portuguese',
    'quy': 'ayacucho quechua',
    'rm-sursilv': 'romansh-sursilv',
    'rm-vallader':'romansh-vallader',
    'ro': 'romanian',
    'ru': 'russian',
    'rw': 'kinyarwanda',
    'sah': 'sakha',
    'sat': 'santali (ol chiki)',
    'sc': 'sardinian',
    'sd': 'sindhi',
    'sk': 'slovak',
    'skr': 'saraiki',
    'sl': 'slovenian',
    'sn': 'shona',
    'so': 'somali',
    'sq': 'albanian',
    'sr': 'serbian',
    'sv': 'swedish',
    'sv-SE': 'swedish',
    'sw': 'swahili',
    'ta': 'tamil',
    'te': 'telugu',
    'tg': 'tajik',
    'th': 'thai',
    'ti': 'tigrinya',
    'tig': 'tigre',
    'tk': 'turkmen',
    'tok': 'toki pona',
    'tr': 'turkish',
    'tt': 'tatar',
    'tw': 'twi',
    'ug': 'uyghur',
    'uk': 'ukrainian',
    'umb': 'umbundu',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'vot': 'votic',
    'wo': 'wolof',
    'xh': 'xhosa',
    'yo': 'yoruba',
    'yue': 'cantonese chinese',
    'zgh': 'standard moroccan tamazight',
    'zh': 'chinese',
    'zh-CN': 'chinese',
    'zu': 'zulu',
}
