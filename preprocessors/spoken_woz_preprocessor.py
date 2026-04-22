"""SpokenWoz preprocessor module for AU-Harness framework.

This module provides a preprocessor for the SpokenWoz task-oriented dialogue
benchmark, handling conversation context formatting and audio processing.
"""

import logging
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from preprocessors.base import Preprocessor

logger = logging.getLogger(__name__)


class SpokenWozPreprocessor(Preprocessor):
    """Preprocessor for SpokenWoz task-oriented dialogue benchmark."""

    def process(self, dataset: Dataset, task_config: Dict[str, Any], 
                run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run pre-processing on SpokenWoz dataset.
        
        Args:
            dataset: The task dataset to pre-process
            task_config: Dictionary containing task configuration parameters
            run_config: Dictionary containing run configuration parameters
            
        Returns:
            List of dictionaries where each dictionary represents a pre-processed sample
        """
        
        # Extract common properties from task_config
        modality = task_config.get('modality', 'audio')
        audio_column_name = task_config.get('audio_column', 'audio')
        target_column_name = task_config.get('target_column', 'agent_text')
        context_column_name = task_config.get('context_column', 'context')
        user_text_column_name = task_config.get('user_text_column', 'text')
        
        # Obtain task-specific prompt (if provided)
        user_prompt = task_config.get('user_prompt', '')
                
        # Get dataset info
        dataset_keys = list(dataset.features.keys())
        dataset_size = len(dataset)
        self.log_dataset_info(dataset_keys, dataset_size)

        # Get dataset filters
        length_filter, num_samples_filter = self.get_dataset_filters(
            run_config.get('filter', None), dataset_size
        )

        processed_data = []
        total_duration = 0
        sample_count = 0

        for i, row in enumerate(tqdm(dataset, desc="Processing SpokenWoz samples")):
            # Only copy essential keys to avoid memory bloat from large audio/context data
            record = {
                target_column_name: row.get(target_column_name),
                context_column_name: row.get(context_column_name),
                user_text_column_name: row.get(user_text_column_name),
                'slots': row.get('slots', {}),
            }

            if modality == "text":
                # Text-only mode: use placeholder audio
                record["array"] = np.array([], dtype=np.float32)
                record["sampling_rate"] = 16000
            else:
                # Audio mode: extract audio information
                record[audio_column_name] = row.get(audio_column_name)
                self.extract_audio_info(record, audio_column_name=audio_column_name)

                # Convert to float32 to save memory (float64 -> float32 = 50% reduction)
                if record["array"].dtype == np.float64:
                    record["array"] = record["array"].astype(np.float32)

                # Calculate audio duration in seconds
                audio_duration = len(record["array"]) / record["sampling_rate"]
                total_duration += audio_duration

                # Apply length filtering (only for audio mode)
                if length_filter:
                    if not self.check_audio_length(record["array"], record["sampling_rate"], length_filter):
                        continue
            if num_samples_filter:
                if sample_count >= num_samples_filter:
                    break

            # Extract target (agent response)
            if target_column_name and target_column_name in record:
                record["model_target"] = record.get(target_column_name, None)
            else:
                raise ValueError(f"No valid target key '{target_column_name}' found in record")

            # Build instruction with dialogue context
            instruction = self._build_instruction_with_context(
                record, 
                context_column_name, 
                user_text_column_name,
                user_prompt
            )
            record["instruction"] = instruction.strip()

            # Preserve slots for SpokenWoz metrics (JGA, slot accuracy)
            record['ground_truth_slots'] = record.get('slots', {})

            # Remove unnecessary data to free memory
            record.pop(context_column_name, None)
            record.pop(user_text_column_name, None)
            record.pop(target_column_name, None)
            record.pop('slots', None)

            # Set judge type for evaluation
            metric_name = task_config.get('metrics', '')
            if isinstance(metric_name, list) and len(metric_name) > 0:
                metric_name = metric_name[0].get('metric', '')
            if 'judge' in str(metric_name):
                judge_type = str(metric_name).split('_')[-1]
                record['judge_type'] = judge_type
            else:
                record['judge_type'] = 'detailed'

            processed_data.append(record)
            sample_count += 1

        self.log_dataset_info(dataset_keys, dataset_size, sample_count, total_duration)

        return processed_data

    def _build_instruction_with_context(
        self, 
        record: Dict[str, Any], 
        context_column: str,
        user_text_column: str,
        user_prompt: str
    ) -> str:
        """Build instruction string including dialogue context.
        
        Args:
            record: The current sample record
            context_column: Name of the context column
            user_text_column: Name of the user text column
            user_prompt: Task-specific prompt prefix
            
        Returns:
            Formatted instruction string with context
        """
        parts = []
        
        # Add task prompt if provided
        if user_prompt:
            parts.append(user_prompt)
        
        # Add dialogue context if available
        context = record.get(context_column, {})
        if context and isinstance(context, dict):
            turn_indices = context.get('turn_index', [])
            user_texts = context.get('text', [])
            agent_texts = context.get('agent_text', [])
            
            if turn_indices and len(turn_indices) > 0:
                parts.append("\n[Dialogue History]")
                for j in range(len(turn_indices)):
                    user_text = user_texts[j] if j < len(user_texts) else ""
                    agent_text = agent_texts[j] if j < len(agent_texts) else ""
                    parts.append(f"User: {user_text}")
                    parts.append(f"Agent: {agent_text}")
        
        # Add current user input indicator
        current_user_text = record.get(user_text_column, "")
        if current_user_text:
            parts.append(f"\n[Current User Input (Audio Transcription)]: {current_user_text}")
        
        parts.append("\nPlease provide an appropriate agent response based on the user's spoken input and dialogue history.")
        
        return "\n".join(parts)
