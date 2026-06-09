"""BFCL preprocessor for function calling evaluation."""

import ast
import json
import logging
import re
from typing import Dict, List, Any
from datasets import Dataset 

import numpy as np
from tqdm import tqdm

from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class BfclPreprocessor(Preprocessor):
    """
    A preprocessor for the BFCL dataset, designed for
    instruction following evaluation of audio LLMs.
    """

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on standard/ general Audio datasets.
        
        Args:
            dataset: The task dataset to pre-process (Dataset object)
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """

        # Extract common properties using base class method
        modality = task_config.get('modality', 'audio')
        audio_column_name = task_config.get('audio_column', None)
        target_column_name = task_config.get('target_column', None)
        sample_instruction_column_name = task_config.get('instruction_column', None)

        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')

        # Get dataset info using base class method
        dataset_keys = list(dataset.features.keys())
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(run_config.get('filter', None), dataset_size)

        indices = range(dataset_size)
        processed_data = []
        sample_count = 0
        total_duration = 0
        
        for i in tqdm(indices, desc="Processing samples"):
            ######## RULE TO PARSE BFCL INSTRUCTIONS #################################################
            ### Without prompt: 
            #   Audio Modality: instruction = ''
            #   Text Modality: insttruction = prompt (from dataset)
            ### With prompt: 
            #   + Audio Modality: instruction = user_prompt + function + prompt (from dataset)
            #   + Text Modality: instruction = user_prompt + user_prompt + function + prompt (from dataset) 
            #################### END RULE ############################################################

            id = dataset["id"][i]
            prompt = dataset[sample_instruction_column_name][i]
            # Default without prompt
            instruction = '' if modality =='audio' else prompt
            
            if isinstance(prompt, str):
                prompt = ast.literal_eval(prompt)

            if len(prompt) == 1:  # For now handle single turn only
                prompt = prompt[0]
            else:
                logger.warning(f"[{id}] Support only single turn")
                continue

            # Ensure prompt exists. Otherwise, skip this example.
            if not prompt:
                logger.warning(f"[{id}] Missing prompt. Skipping sample.")
                continue
            
            if modality == "text":
                audio_data = {
                    "array": np.array([]),  # Placeholder, not used in text-only evals
                    "sampling_rate": 16000
                }
            else:
                audio_data = dataset[audio_column_name][i]
                if isinstance(audio_data, list) and len(audio_data) == 1:
                    audio_data = audio_data[0]
                elif isinstance(audio_data, list) and len(audio_data) > 1:
                    logger.warning(f"[{id}] Support single audio only right now!")
                    continue
                elif isinstance(audio_data, dict):
                    audio_data = audio_data

            function = dataset["tools"][i]
            reference = dataset[target_column_name][i]

            if isinstance(reference, str):
                reference = json.loads(reference)
            if isinstance(function, str):
                function = json.loads(function)

            new_references = []
            for item in reference:
                func_name = list(item.keys())[0]
                func_args = item[func_name]
                for func in function:
                    name = func["name"]
                    if func_name == name.split(".")[-1]:
                        func_name = name
                        break
                new_func_name = re.sub(r"\.", "_", func_name)
                new_references.append({new_func_name: func_args})
            reference = new_references

            # Lot of models don't support dot(.) in function name so to handle that we replace it with _ for easy evals
            new_functions = []
            for item in function:
                item["name"] = re.sub(r"\.", "_", item["name"])
                new_functions.append(item)
            function = new_functions

            required_fields = {
                tool['name']: [
                    (
                        param,
                        props[param]['type'],
                        props[param].get('items', {}).get('type'),   # nested element type (for array/tuple)
                        param in tool['parameters'].get('required', []),  # True = required, False = optional
                    )
                    for param, props in [
                        (p, tool['parameters']['properties'])
                        for p in tool['parameters']['properties']
                    ]
                ]
                for tool in function
            }

            # Validate audio data structure
            if not isinstance(audio_data, dict):
                logger.warning(f"[{id}] Invalid audio format. Skipping sample.")
                continue

            # Convert to NumPy array
            audio_array = np.array(audio_data.get("array"))
            sr = audio_data.get("sampling_rate")

            if modality == "audio":
                if sr is None:
                    logger.warning("[%s] Sampling rate missing. Assuming 16kHz.", id)
                    sr = 16000

                # Use base class method to resample audio
                audio_array, sr = self.resample_audio(audio_array, sr)
                if (length_filter and not self.check_audio_length(audio_array, sr, length_filter)):
                    logger.warning(
                        f"Audio in item {id} filtered because of length "
                        f"Supported: {length_filter[0]}–{length_filter[1]} sec."
                    )
                    continue

                # Calculate audio duration in seconds
                audio_duration = len(audio_array) / sr
                total_duration += audio_duration
                
            # If prompt is provided, either modality will require function appended to instruction and setting function = None
            if (user_prompt != ''):
                instruction = user_prompt + 'Here is a list of functions in JSON format that you can invoke.\n{functions}\n'.format(
                    functions=json.dumps(function, indent=4))
                function = None

                # If modality is text, adding the sample-specific prompt 
                if (modality == 'text'):
                    instruction += '\n\n' + prompt

            # In prompt mode we get tool calls from llm response,
            # This doesn't evals the model on tool calling via capability sense,
            # but still allow to check ability to choose function


            # Stop processing if num_samples filtering is set and more than num_samples_filter samples are processed
            if (num_samples_filter and sample_count >= num_samples_filter):
                break
            # Create structured sample
            sample = {
                "id": id,
                "array": audio_array,
                "sampling_rate": sr,
                "audio_content_in_text": prompt,
                "instruction": instruction,
                "tools": function,
                "reference": reference,
                "model_target": [reference, required_fields],
            }

            processed_data.append(sample)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)

        return processed_data