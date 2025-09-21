"""Reasoning-based preprocessor module for AU-Harness framework.

This module provides a preprocessor for audio benchmarks
from AudioLLMs and other HuggingFace datasets, with focus on support of MMAR/MMAU-PRO
where local audio files need to be downloaded, unzipped and loaded from LOCAL_DATA_DIR
when preprocessing. LOCAL_DATA_DIR needs to be set from environment (.env).
"""

import logging
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from preprocessors.base import Preprocessor
from scipy.signal import resample
import soundfile as sf
from urllib.request import urlopen
import io
import os
from dotenv import load_dotenv
from pathlib import Path


logger = logging.getLogger(__name__)

class MmarPreprocessor(Preprocessor):
    """Preprocessor for standard Audio benchmarks where output references are ALWAYS expected."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on standard/ general Audio datasets.
        
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """

        # Load the local_data_dir saved in predefined .env file        
        load_dotenv()
        local_data_dir = os.getenv("LOCAL_DATA_DIR")
        dataset_name = task_config['dataset_path'].split('/')[-1].lower()

        # Extract common properties using base class method
        category_name = task_config.get('category_name', 'speech')
        audio_column_name = task_config.get('audio_column', None)
        target_column_name = task_config.get('target_column', None)
        choices_column_name = task_config.get('choices_column', None)
        category_column_name = task_config.get('category_column', '')
        sample_instruction_column_name = task_config.get('instruction_column', None)
        user_query_column_name = task_config.get('textual_input_column', None)

        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')
        
        # Get dataset info
        dataset_keys = list(dataset.features.keys())
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)

        processed_data = []
        total_duration = 0
        sample_count = 0

        for i, row in enumerate(tqdm(dataset, desc="Processing samples")):
            instruction = user_prompt
            if (row[category_column_name] != category_name):
                continue
            # Create record by accessing each feature by index
            record = {k: row[k] for k in dataset_keys}
            audio_path = record[audio_column_name]
            if (isinstance(audio_path, list)):
                audio_path = audio_path[0]

            # Mapping audio path to local audio path (sample: $HOME/mmau-pro/data/xyz.wav)
            local_audio_path = os.path.join(local_data_dir, dataset_name, audio_path)
            audio_array, samplerate = sf.read(local_audio_path)

            # Resample samples if not in 16kHz sampling rate
            target_sr = 16000
            if samplerate != target_sr:
                num_samples = int(round(audio_array.shape[0] * target_sr / samplerate))
                audio_array = resample(audio_array, num_samples)
                samplerate = target_sr
            record['array'] = audio_array
            record['sampling_rate'] = samplerate
            
            # Calculate audio duration in seconds
            audio_duration = len(record["array"]) / record["sampling_rate"]
            total_duration += audio_duration

            # Apply dataset filtering
            if (length_filter):
                if not self.check_audio_length(record["array"], record["sampling_rate"], length_filter):
                    continue
            if (num_samples_filter):
                if sample_count >= num_samples_filter:
                    break

            # General processor requires reference. Otherwise, implement your own preprocessor.
            if target_column_name and target_column_name in record:
                record["model_target"] = record.get(target_column_name, None)
            else:
                raise ValueError("No valid target key found in record")

            # Add sample-specific instructions if they exist in the dataset
            if sample_instruction_column_name and sample_instruction_column_name in record:
                instruction += record.get(sample_instruction_column_name, "")
            
            # Append any user-specified prompt add-ons and choices
            if choices_column_name and choices_column_name in record:
                choices = record.get(choices_column_name, [])
                if isinstance(choices, list):
                    choices_text = " ".join(choices)
                else:
                    choices_text = str(choices)
                instruction += "\n Choices: " + choices_text
            
            # Warning users if no instruction is provided. This can cause evaluated models to hallucinate.
            if not instruction:
                logger.warning("Instruction is empty for sample %d, add user_prompt for instruction insertion", i)
            record["instruction"] = instruction.strip()

            metric_name = task_config.get('metrics')
            if ('judge' in metric_name):
                judge_type = metric_name.split('_')[-1]
                record['judge_type'] = judge_type
            else:
                record['judge_type'] = 'detailed'
            processed_data.append(record)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)
        return processed_data
