"""Engine module for evaluating language models across various tasks and datasets.

This module provides the core Engine class responsible for running model evaluations,
managing inference, handling concurrent model execution, and computing metrics.
It orchestrates the entire evaluation pipeline from dataset loading to final scoring.
"""

import asyncio
import logging
import math
from tqdm import tqdm

from models.model import Model
from metrics.llm_judge import _BaseLLMJudge
from postprocessors.base import Postprocessor
from utils.data_utils import load_dataset_with_args
from utils.metric_utils import load_metric
from utils.model_utils import get_generation_params_override
from utils.util import get_class_from_module, get_system_prompt_override, get_instruction_prompt_override

logger = logging.getLogger(__name__)

class Engine:
    """Evaluate one or many models over the same dataset concurrently."""
    def __init__(self, models: list[Model], task_info: dict, run_config: dict,
                 engine_id: str = None, request_manager = None):
        
        self.engine_id = engine_id

        self.request_manager = request_manager

        # Unpack task_info
        self.task_name, self.metric_name, self.task_config, self.task_ancestry = task_info

        self.run_config = run_config

        # Load dataset
        self.dataset = self.get_dataset(self.task_config, self.task_name)

        # Pre-process dataset
        self.dataset = self.preprocess_dataset(self.dataset, self.task_config, run_config)

        # Temperature overrides
        self.generation_params_override = self.run_config.get("generation_params_override", None)

        # Prompt overrides
        self.prompt_overrides = self.run_config.get("prompt_overrides", None)

        # Load all metrics from task_config
        self.metrics = self.load_metrics()

        # Load the postprcessor
        self.postprocessor = self.get_postprocessor(self.task_config)

        # Group models by their model attribute for sharding
        self.model_groups = self.get_model_groups(models)
        
    def get_dataset(self, task_config, task_name):
        """
        Load the dataset based on the task configuration parameters.
        
        Args:
            task_config: Dictionary containing task configuration parameters,
                         including dataset_path, split, and subset
            task_name: Name of the task being evaluated
            
        Returns:
            The loaded dataset
        """
        dataset_path = task_config.get("dataset_path", None)
        split = task_config.get("split", None)
        subset = task_config.get("subset", None)
        return load_dataset_with_args(dataset_path, split, subset, task_name)
    
    def preprocess_dataset(self, dataset, task_config, run_config):
        """
        Preprocess the dataset using the specified preprocessor from the task configuration.
        
        Args:
            dataset: The dataset to preprocess
            task_config: Dictionary containing task configuration parameters,
                         including the preprocessor to use
                         
        Returns:
            The preprocessed dataset
            
        Raises:
            ValueError: If the specified preprocessor cannot be loaded
        """
        preprocessor_name = task_config.get("preprocessor", "GeneralPreprocessor")
        preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
        if preprocessor_class is None:
            error_msg = f"Could not load preprocessor {preprocessor_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        preprocessed = preprocessor_class().process(dataset, task_config, run_config)
        return preprocessed
    
    def load_metrics(self):
        """
        Load all metrics specified in the task configuration.
        Returns a dictionary mapping metric names to metric instances.
        """
        language = self.task_config.get("language", "en")
        judge_settings = self.run_config.get("judge_settings", None)
        
        # Initialize metrics dict
        metrics = {}

        # if a metric is specified in run_config, compute only that
        if self.metric_name != "all":
            try:
                metrics[self.metric_name] = load_metric(self.metric_name, language=language, judge_settings=judge_settings)
            except Exception as e:  
                logger.warning("[Engine.load_metrics] Failed to load metric %s: %s", self.metric_name, e)
            return metrics

        # Load all metrics for the task from task_config
        for metric_config in self.task_config.get('metrics', []):
            metric_name = metric_config['metric']
            try:
                metrics[metric_name] = load_metric(metric_name, language=language, judge_settings=judge_settings)
                logger.info("[Engine.load_metrics] Loaded metric: %s", metric_name)
            except Exception as e:
                logger.warning("[Engine.load_metrics] Failed to load metric %s: %s", metric_name, e)

        return metrics

    def get_postprocessor(self, task_config: dict) -> Postprocessor:
        postprocessor_name = task_config.get("postprocessor", "GeneralPostprocessor")
        postprocessor_class = get_class_from_module('postprocessors', postprocessor_name)
        if postprocessor_class is None:
            logger.warning("Could not load postprocessor %s, using default GeneralPostprocessor", postprocessor_name)
            postprocessor_class = get_class_from_module('postprocessors', 'GeneralPostprocessor')
        return postprocessor_class()
    
    def get_model_groups(self, models: list[Model]) -> dict[str, list[Model]]:
        """
        Group models by their model attribute for sharding
        """
        model_groups = {}
        for model in models:
            model_type = model.model  # The model attribute we're sharding on
            if model_type not in model_groups:
                model_groups[model_type] = []
            model_groups[model_type].append(model)
        return model_groups

    # ---------------- internal helpers ----------------x
    # infer by batch size, calling generate text with retry for each sample
    async def _infer_single_model(self, model: Model, samples=None) -> list[str]:
        samples = samples if samples is not None else self.dataset  # Use provided samples or full dataset
        
        # Check for generation params override for this specific model and task combination
        override_generation_params = get_generation_params_override(model.model, self.task_ancestry, self.generation_params_override)

        if override_generation_params is not None:
            # Use the override generation params directly
            model.set_generation_params(override_generation_params)
        else:
            # Use the standard task-based generation params setting
            model.set_generation_params(self.task_config.get("generation_kwargs", {"temperature": 0.7, "max_completions_tokens": 4096}))

        # Check for system prompt for this specific model and task combination from run_config
        system_prompt_override = get_system_prompt_override(model.model, 
                                                            self.task_ancestry, 
                                                            self.run_config.get("prompt_overrides", None))
        if system_prompt_override is not None:
            model.set_system_prompt(system_prompt_override)

        # Check for instruction override for this specific model and task combination from run_config
        instruction_override = get_instruction_prompt_override(model.model, 
                                                        self.task_ancestry, 
                                                        self.run_config.get("prompt_overrides", None))
        if instruction_override is not None:
            model.set_or_override_instruction(instruction_override)
        
        # Add long audio processing
        long_audio_processing_logic = self.task_config.get("long_audio_processing_logic", None)
        if long_audio_processing_logic not in ["chunk", "truncate"]:
            raise ValueError("Invalid long audio processing logic for task %s. Must be 'chunk' or 'truncate'." % self.task_name)
        model.set_long_audio_processing_logic(long_audio_processing_logic)

        # Get model type for request management
        model_name = model.name()  # The actual model name (e.g., "gpt-4o-mini-audio-preview-1")
        # Generate a unique model instance ID
        model_instance_id = f"{model.name()}_{id(model)}"
        
        # Create an adjustable semaphore based on granted tokens
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        
        # Keep track of pending and completed samples
        pending_samples = list(range(len(samples)))  # Indices of samples waiting for tokens
        completed_samples = set()  # Indices of samples that have completed processing
        
        # Continuously ask for tokens(with backoff if none available) based on pending samples
        # Have dynamic wait times for by dataset size - this "layering" of Engine priority gives models in the same Engine similar priority, so they don't wait on each other as often
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(samples)
            
            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))
            
            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0
            
            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0
            
            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as we need for pending samples
                request_amount = min(model.batch_size, len(pending_samples))
                
                if request_amount > 0:
                    granted = await self.request_manager.request_tokens(
                        model_name, model_instance_id, request_amount)
                    
                    if granted > 0:
                        # Remove samples from pending list based on granted tokens
                        pending_samples[:] = pending_samples[granted:]
                        # Release semaphore permits for each granted token
                        for _ in range(granted):
                            token_sem.release()
                        # Wait based on dataset size when tokens were granted
                        await asyncio.sleep(token_wait)
                    else:
                        # Backoff when no tokens were granted, based on dataset size
                        # Apply a small multiplier for repeated failures, but cap it
                        backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                        await asyncio.sleep(no_token_wait * backoff_multiplier)
                else:
                    break
        
        asyncio.create_task(token_manager())
            
        # Wait for token to be available, run task, and add to completed
        async def _call_with_token_mgmt(idx: int, sample: dict):
            # Acquire a token
            await token_sem.acquire()
            try:
                # Process the sample
                resp = await model.generate_text_with_retry(sample, {"chunk_size": model.chunk_size})
                result = resp
                
                # Add to completed set
                completed_samples.add(idx)
                
                # Return token to model's pool
                await self.request_manager.return_tokens(model_name, model_instance_id, 1)
                
                return idx, result
            except Exception as e:
                # Make sure to return token on error
                logger.error("[Engine._infer_single_model] Error processing sample %d in %s: %s", idx, self.task_name, e)
                completed_samples.add(idx)
                await self.request_manager.return_tokens(model_name, model_instance_id, 1)
                return idx, ""
        
        # Create tasks paired with their original index
        tasks = [_call_with_token_mgmt(i, ex) for i, ex in enumerate(samples)]
        
        # Process results in order of completion
        results: list[str | None] = [None] * len(tasks)
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                          desc="Inference (%s | %s)" % (self.task_name, model.name())):
            idx, resp = await coro
            results[idx] = resp
            
        return results

    # Infer all models concurrently
    async def _infer_all(self):
        results = {}
        all_tasks = []
        task_info = {}

        # Prepare all tasks for concurrent execution
        for model_type, models in self.model_groups.items():
            if len(models) > 1:  # Multiple instances of the same model type - need sharding
                # Calculate proportional sharding based on batch sizes
                total_batch_capacity = sum(model.batch_size for model in models)
                dataset_size = len(self.dataset)
                
                # Track the mapping of original indices to shard indices for recombination
                index_mappings = {}
                sharded_tasks = {}
                
                # Distribute samples and create tasks
                current_idx = 0
                for i, model in enumerate(models):
                    # Calculate proportional shard size based on batch size
                    if i < len(models) - 1:
                        shard_size = int(dataset_size * model.batch_size / total_batch_capacity)
                        end_idx = current_idx + shard_size
                    else:
                        # Last model gets any remaining samples
                        end_idx = dataset_size
                    
                    shard = self.dataset[current_idx:end_idx]

                    # Keep track of original indices
                    index_mappings[model.name()] = list(range(current_idx, end_idx))
                    current_idx = end_idx
                    
                    task = asyncio.create_task(self._infer_single_model(model, shard))
                    sharded_tasks[model.name()] = task
                    all_tasks.append(task)
                
                # Store info for later reconstruction
                task_info[model_type] = {
                    "is_sharded": True,
                    "tasks": sharded_tasks,
                    "index_mappings": index_mappings
                }
            else:
                # Single instance, normal processing
                model = models[0]
                model_name = model.name()
                task = asyncio.create_task(self._infer_single_model(model))
                all_tasks.append(task)
                task_info[model_name] = {
                    "is_sharded": False,
                    "task": task
                }
        # Wait for all tasks to complete concurrently
        await asyncio.gather(*all_tasks)
        # Process results according to task type
        for key, info in task_info.items():
            if info["is_sharded"]:
                # Reconstruct sharded results
                shard_results = {name: task.result() for name, task in info["tasks"].items()}
                
                # Combine results under model_type as the key
                combined_results = [None] * len(self.dataset)

                # Use index mappings to put results back in correct order
                for model_name, model_results in shard_results.items():
                    original_indices = info["index_mappings"][model_name]
                    for shard_idx, orig_idx in enumerate(original_indices):
                        if shard_idx < len(model_results):
                            combined_results[orig_idx] = model_results[shard_idx]

                # Use the model_type as the key for combined results
                results[key] = combined_results
            else:
                # Single instance result
                results[key] = info["task"].result()
        
        return results

    async def run(self):
        metrics_str = ', '.join([m for m in self.metrics.keys()])
        logger.info("[Engine.run] Starting evaluation run for %s with metrics: %s", self.task_name, metrics_str)
        raw_predictions = await self._infer_all()
        logger.info("[Engine.run] Predictions complete for %s. Calculating scores...", self.task_name)
        scores = {model_name: {} for model_name in raw_predictions.keys()}
        
        # We'll iterate through each metric and process the results
        for metric_name, metric_instance in self.metrics.items():
            # Pass the metric name to the postprocessor
            process_result = self.postprocessor.process(dataset=self.dataset, predictions=raw_predictions, metric=metric_name)
            # Extract values from the dictionary returned by the postprocessor
            model_targets = process_result["model_targets"]
            predictions = process_result["processed_predictions"]
            instructions = process_result.get("instructions", None)
            ids = process_result.get("ids", [])
            lengths = process_result.get("lengths", [])

            # Determine if this is an LLM-judge metric
            is_llm_judge = isinstance(metric_instance, _BaseLLMJudge)
        
            # Get judge_properties from the metric if it's a LLM judge
            judge_settings = getattr(metric_instance, '_judge_properties', None) if is_llm_judge else None
        
            # Get language attribute from the metric if available or default to 'en'
            language = getattr(metric_instance, 'language', 'en')
        
            # Create metric instances for each model using load_metric
            # load_metric will handle deciding which parameters to use based on the metric type
            metric_instances = {
                model_name: load_metric(metric_name, language=language, judge_settings=judge_settings)
                for model_name in predictions.keys()
            }
        
            async def score_model_with_tokens(model_name, outs):
                metric = metric_instances[model_name]
                model_responses = raw_predictions.get(model_name, [])
                # Check if this is an LLM judge
                if is_llm_judge:
                    # For LLM judges, set the request manager
                    # The metric will handle token management internally with the Engine Manager
                    metric.set_request_manager(self.request_manager)
                
                    if ids and lengths:
                        result = await metric(outs, model_targets, ids, lengths,
                                            instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses)
                    else:
                        result = await metric(outs, model_targets,
                                            instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses)
                    return model_name, result
                else:
                    # For regular metrics, just run them directly (no token management needed)
                    if ids and lengths:
                        if metric_name == 'comet':
                            source_sentences = process_result["source_sentences"]
                            result = await asyncio.to_thread(
                                metric, outs, model_targets, ids, lengths,
                                instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses
                            )
                        else:
                            result = await asyncio.to_thread(
                                metric, outs, model_targets, ids, lengths,
                                instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses
                            )
                    else:
                        if metric_name == 'comet':
                            source_sentences = process_result["source_sentences"]
                            result = await asyncio.to_thread(
                                metric, outs, model_targets, source_sentences,
                                instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses
                            )
                        elif metric_name in ('joint_goal_accuracy', 'slot_accuracy', 'slot_f1'):
                            ground_truth_slots = process_result.get("ground_truth_slots", [])
                            result = await asyncio.to_thread(
                                metric, outs, model_targets,
                                instructions=instructions, task_name=self.task_name, model_name=model_name, 
                                model_responses=model_responses, ground_truth_slots=ground_truth_slots
                            )
                        else:
                            result = await asyncio.to_thread(
                                metric, outs, model_targets,
                                instructions=instructions, task_name=self.task_name, model_name=model_name, model_responses=model_responses
                            )
                    return model_name, result

            # Run all model scoring concurrently for this metric
            tasks = [score_model_with_tokens(model_name, outs) for model_name, outs in predictions.items()]
            results = await asyncio.gather(*tasks)
        
            # Store results for this metric
            for model_name, model_score in results:
                scores[model_name][metric_name] = model_score
            
            logger.info("[Engine.run] Completed evaluation with metric %s", metric_name)
    
        # All metrics have been processed
        logger.info("[Engine.run] All evaluations complete. Returning scores.")
        return scores