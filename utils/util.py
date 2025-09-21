
import importlib
import json
import logging
import os
import statistics
import yaml
from pathlib import Path
from typing import Any, Dict
import tarfile
import zipfile
from . import constants
from utils.custom_logging import configure
from utils.task_utils import _validate_task_metric_pairs, get_groups, get_tasks 

logger = logging.getLogger(__name__)

def get_class_from_module(module_prefix, module_name):
    try:
        # Convert class name (CamelCase) to filename (snake)
        # Get pre or post processor
        module_filename = ''.join(['_' + c.lower() if c.isupper() else c for c in module_name]).lstrip('_')
        module = importlib.import_module(f"{module_prefix}.{module_filename}")
        return getattr(module, module_name)
    except Exception as e:
        logger.warning(f"Could not import {module_name} from {module_prefix}: {e}")
        return None

def extract_tar_gz(file_path, extract_path="."):
    """
        Extracts a .tar.gz file to a specified path.
        Args:
        ----
        file_path: str: Path to the archive `.tar.gz` file.
        extract_path: str: Directory to extract the contents to.
    """
    try:
        print ("Tar gz extraction")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        logger.warning(f"Successfully extracted {file_path} to {extract_path}")
    except tarfile.ReadError as e:
        logger.warning(f"Error reading tar.gz file: {e}")
    except Exception as e:
        logger.warning(f"An unexpected error occurred: {e}")

def extract_zip(file_path, extract_path="."):
    """
        Extracts a .zip file to a specified path.
        Args:
        ----
        file_path: str: Path to the archive `.zip` file.
        extract_path: str: Directory to extract the contents to.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.warning(f"Successfully extracted {file_path} to {extract_path}")
    except zipfile.BadZipFile as e:
        logger.warning(f"Error reading zip file: {e}")
    except Exception as e:
        logger.warning(f"An unexpected error occurred: {e}")

def extract_archive(file_path, extract_path="."):
    """
        Extracts either a .tar.gz or .zip file based on its extension.

        Args:
        ----
        file_path: str: Path to the archive file.
        extract_path: str: Directory to extract the contents to.
    """
    if file_path.endswith(".tar.gz"):
        extract_tar_gz(file_path, extract_path)
    elif file_path.endswith(".zip"):
        extract_zip(file_path, extract_path)
    else:
        logger.warnning(f"Unsupported archive format for file: {file_path}")

def smart_round(val: float, precision: int = constants.ROUND_DIGITS) -> float:
    """Round off metrics to global precision value.

    References:
        1. https://bugs.python.org/msg358467
        2. https://en.wikipedia.org/wiki/IEEE_754

    Args:
    ----
        precision: int: Precision up to which value should be rounded off.
        val: float: Value

    Returns:
    -------
        float: Rounded off value
    """
    if not isinstance(precision, int) or precision <= 0:
        logger.warning(
            "Invalid precision provided: %s. Using the default precision: %s",
            precision, constants.ROUND_DIGITS
        )
        precision = constants.ROUND_DIGITS
    rounded_off_val = round(val * 10 ** precision) / 10 ** precision
    return rounded_off_val


def get_context_indices_for_filter(key: str, value: Any, contexts: list[dict]) -> list[int]:
    """Get indices for rows satisfying the given filter.

    Given key-value pair, it returns the list of indices of contexts satisfying key = value.

    Args:
        key: The key to match against in each row of context/data.
        value: The value to compare against
        contexts: list of dictionaries containing additional key-value pairs in data.

    Returns:
        List of integer indices.

    """
    indices = [_ for _, c in enumerate(contexts) if c[key] == value]
    return indices

def validate_config(config: dict, task_configs: dict[Path, list[dict]]) -> Dict:
    """
    Validate configuration file against expected structure and types.

    Args:
        config: Configuration dictionary
        task_configs: List of task configs
    """
    try:
        logger.info("---------Start validation---------")

        # Validate filters as a dictionary
        logger.info("---------Validating filters---------")
        if 'filters' in config:
            if not isinstance(config['filters'], dict):
                raise ValueError("'filters' must be a dictionary")
            _validate_filter_values(config['filters'])
        
        # Validate judge_properties as a dictionary
        logger.info("---------Validating judge properties---------")
        if 'judge_properties' in config:
            if not isinstance(config['judge_properties'], dict):
                raise ValueError("'judge_properties' must be a dictionary")
            _validate_judge_properties(config['judge_properties'])

        # Delegate validation for complex sections
        logger.info("---------Validating models---------")
        _validate_models(config)

        logger.info("---------Validating aggregate---------")
        if 'aggregate' in config:
            _validate_aggregate(config['aggregate'])

        logger.info("---------Validating temperature overrides---------")
        if 'temperature_overrides' in config:   
            _validate_temperature_overrides(config['temperature_overrides'])

        logger.info("---------Validating prompt overrides---------")
        if 'prompt_overrides' in config:
            infer_models = [x['name'] for x in config['models']]
            _validate_prompt_overrides(config['prompt_overrides'], task_configs, infer_models)

        logger.info("---------Validating task-metric---------")
        if 'task_metric' not in config:
            raise ValueError("'task_metric' is required")

        if not isinstance(config['task_metric'], list):
            raise ValueError("'task_metric' must be a list")

        task_metric = config['task_metric']
        if len(task_metric) == 0:
            raise ValueError("'task_metric' must have at least one element")

        for i, item in enumerate(task_metric):
            if not isinstance(item, list):
                raise ValueError(
                    f"'task_metric' item {i+1} must be a list, not {type(item).__name__}"
                )

            if len(item) == 0:
                raise ValueError(f"'task_metric' item {i+1} must not be an empty list")

            for j, element in enumerate(item):
                if not isinstance(element, str) or not element.strip():
                    raise ValueError(
                        f"'task_metric' item {i+1}, element {j+1} must be a non-empty string"
                    )
        _validate_task_metric_pairs(config.get('task_metric', []), task_configs)

        logger.info("---------Configuration validated successfully---------")
        return

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {str(e)}") from e


def _validate_filter_values(filters: Dict) -> None:
    """Validate the values in the filters dictionary.
    
    Args:
        filters: Dictionary of filter values to validate
    
    Raises:
        ValueError: If any filter value is invalid
    """
    # Validate num_samples if present
    if 'num_samples' in filters and not isinstance(filters['num_samples'], int):
        raise ValueError("'num_samples' must be an integer")
    
    # Validate length_filter if present
    if 'length_filter' in filters:
        if not isinstance(filters['length_filter'], list):
            raise ValueError("'length_filter' must be a list")
        filter_list = filters['length_filter']
        if len(filter_list) != 2:
            raise ValueError("'length_filter' must have exactly 2 elements")
        if not all(isinstance(value, (int, float)) for value in filter_list):
            raise ValueError("'length_filter' elements must be numbers")
    
    # Validate accented if present
    if 'accented' in filters and not isinstance(filters['accented'], bool):
        raise ValueError("'accented' must be a boolean")
    
    # Validate language if present
    if 'language' in filters and not isinstance(filters['language'], str):
        raise ValueError("'language' must be a string")


def _validate_judge_properties(judge_props: Dict) -> None:
    """Validate the values in the judge_properties dictionary.
    
    Args:
        judge_props: Dictionary of judge properties to validate
    
    Raises:
        ValueError: If any judge property is invalid
    """
    # Validate judge_concurrency if present
    if 'judge_concurrency' in judge_props and not isinstance(judge_props['judge_concurrency'], int):
        raise ValueError("'judge_concurrency' must be an integer")
    
    # Validate judge_model if present
    if 'judge_model' in judge_props and not isinstance(judge_props['judge_model'], str):
        raise ValueError("'judge_model' must be a string")
    
    # Validate judge_type if present
    if 'judge_type' in judge_props:
        if not isinstance(judge_props['judge_type'], str):
            raise ValueError("'judge_type' must be a string")
        if judge_props['judge_type'] not in ['vllm', 'openai']:
            raise ValueError("'judge_type' must be either 'vllm' or 'openai'")
    
    # Validate string properties
    string_props = ['judge_api_version', 'judge_api_endpoint', 'judge_api_key', 'judge_prompt_model_override']
    for prop in string_props:
        if prop in judge_props and not isinstance(judge_props[prop], str):
            raise ValueError(f"'{prop}' must be a string")
    
    # Validate judge_temperature if present
    if 'judge_temperature' in judge_props and not isinstance(judge_props['judge_temperature'], (int, float)):
        raise ValueError("'judge_temperature' must be a number")

def _validate_models(config: Dict) -> None:
    """Validate the models section of the configuration.

    Args:
        config: The configuration dictionary

    Raises:
        ValueError: If the models section is invalid
    """
    def validate_required_fields(info: Dict, index: int) -> None:
        required_fields = ['name', 'model', 'inference_type', 'url']
        for field in required_fields:
            if not info.get(field) or not isinstance(info[field], str) or not info[field].strip():
                raise ValueError(f"Model {index}: '{field}' must be a non-empty string")
    def validate_optional_fields(info: Dict, index: int) -> None:
        optional_fields = {
            'delay': int, 'retry_attempts': int, 'timeout': int,
            'auth_token': str, 'api_version': str, 'batch_size': int, 'chunk_size': int
        }
        for field, field_type in optional_fields.items():
            if field in info and not isinstance(info[field], field_type):
                raise ValueError(f"Model {index}: '{field}' must be of type {field_type.__name__}")
    if 'models' not in config or not isinstance(config['models'], list):
        raise ValueError("'models' section is required and must be a list")
    for i, model_entry in enumerate(config['models'], start=1):
        if not isinstance(model_entry, dict):
            raise ValueError(f"Model entry {i} must be a dictionary")
        validate_required_fields(model_entry, i)
        validate_optional_fields(model_entry, i)


def _validate_aggregate(aggregate_section) -> None:
    """
    Validate the aggregate section of the configuration.

    Structure should be:
    aggregate:
      - ["metric_name", ["dataset1", "dataset2", ...]]
    """
    # Validate top-level structure is a list
    if not isinstance(aggregate_section, list):
        raise ValueError("'aggregate' must be a list")

    # Validate each aggregate entry
    for i, entry in enumerate(aggregate_section, start=1):
        # Check entry is a list with exactly 2 elements
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"Aggregate entry {i} must be a list with exactly 2 elements")

        # First element must be a non-empty string (metric name)
        if not isinstance(entry[0], str) or not entry[0].strip():
            raise ValueError(
                f"Aggregate entry {i}: first element must be a non-empty string representing the metric name"
            )

        # Second element must be a list of dataset names
        if not isinstance(entry[1], list):
            raise ValueError(f"Aggregate entry {i}: second element must be a list of dataset names")

        # Validate each dataset name in the list
        for j, dataset in enumerate(entry[1], start=1):
            if not isinstance(dataset, str) or not dataset.strip():
                raise ValueError(f"Aggregate entry {i}, dataset {j} must be a non-empty string")


def _validate_temperature_overrides(temperature_overrides) -> None:
    """
    Validate the temperature_overrides section of the configuration.

    Structure should be:
    temperature_overrides:
      - model: "model_name" (optional)
        task: "task_name" (optional)
        temperature: 0.5 (required)

    Either model or task (or both) must be present.

    Args:
        temperature_overrides: The temperature_overrides section to validate

    Raises:
        ValueError: If the temperature_overrides section is invalid
    """
    if not isinstance(temperature_overrides, list):
        raise ValueError("'temperature_overrides' must be a list")

    for i, override in enumerate(temperature_overrides):
        if not isinstance(override, dict):
            raise ValueError(f"Temperature override {i+1} must be a dictionary")

        # Check for required temperature field
        if 'temperature' not in override:
            raise ValueError(f"Temperature override {i+1} is missing required field: 'temperature'")

        if not isinstance(override['temperature'], (int, float)):
            raise ValueError(f"Temperature override {i+1}: 'temperature' must be a number")

        # Check that at least one of model or task is present
        if 'model' not in override and 'task' not in override:
            raise ValueError(
                f"Temperature override {i+1} must have at least one of 'model' or 'task'"
            )

        # Validate types if present and ensure non-empty values
        if 'model' in override:
            if not isinstance(override['model'], str):
                raise ValueError(f"Temperature override {i+1}: 'model' must be a string")
            if len(override['model'].strip()) == 0:
                raise ValueError(f"Temperature override {i+1}: 'model' cannot be empty")

        if 'task' in override:
            if not isinstance(override['task'], str):
                raise ValueError(f"Temperature override {i+1}: 'task' must be a string")
            if len(override['task'].strip()) == 0:
                raise ValueError(f"Temperature override {i+1}: 'task' cannot be empty")

def _validate_prompt_overrides(prompt_overrides, task_configs, inference_models) -> None:
    """
    Validate the prompt_overrides section of the configuration.

    Structure of the prompt overrides should looking like this -

    prompt_overrides:
      user_prompt:
        - task: "task_name"
          model: "model_name" (optional)
          prompt: "prompt_text"
      system_prompt:
        - model: "model_name"
          task: "task_name" (optional)
          prompt: "prompt_text"
    
    Args:
        prompt_overrides: The prompt_overrides section to validate
        task_configs: All the task configs
        inference_models: All the inference models in the run config
    """
    if not isinstance(prompt_overrides, dict):
        raise ValueError("'prompt_overrides' must be a dictionary")
    
    # "user_prompt" and "system_prompt" are the only allowed keys
    if not all(key in ["user_prompt", "system_prompt"] for key in prompt_overrides.keys()):
        raise ValueError("'prompt_overrides' keys must be either 'user_prompt' or 'system_prompt'")

    # Get all groups of tasks. These are generally folders
    groups = get_groups(task_configs)

    # Get all tasks. These are generally YAML files
    tasks = get_tasks(task_configs)

    # Validate each override
    for key, value in prompt_overrides.items():
        if not isinstance(value, list):
            raise ValueError(f"'{key}' in 'prompt_overrides' must be a list")
        
        for i, override in enumerate(value):
            # Validate that each override is a dictionary
            if not isinstance(override, dict):
                raise ValueError(f"Override {i} in '{key}' must be a dictionary")
            
            # Prompt is a mandatory field
            if "prompt" not in override:
                raise ValueError(f"Override {i} in '{key}' must have a 'prompt' key")
            
            # Task is a mandatory field for user_prompt. This can be a task name or a task group name
            if ("task" not in override) and (key == "user_prompt"):
                raise ValueError(f"Override {i} in '{key}' must have a 'task' key")
            
            # Model is a mandatory field for system_prompt.
            if ("model" not in override) and (key == "system_prompt"):
                raise ValueError(f"Override {i} in '{key}' must have a 'model' key")
            
            # Make sure only expected keys are present
            for override_key in override.keys():
                if override_key not in ["prompt", "task", "model"]:
                    raise ValueError(f"Invalid key '{override_key}' in override {i} of '{key}'. Only 'prompt', 'task', and 'model' are allowed")
            
            # Make sure that the task is valid
            if "task" in override.keys():
                if (override['task'] not in groups.keys()) and (override['task'] not in tasks.keys()):
                    raise ValueError(f"Invalid task name: {override['task']}")
            
            # Make sure the model name if present is valid
            if "model" in override.keys():
                if override['model'] not in inference_models:
                    raise ValueError(f"Model override of {override['model']} is not presented in the list of models specified in the config")
    return

def setup_logging(log_file: str):
    """
    Set up logging with default.log
    """
    
    # Configure logging using the custom_logging module
    configure(log_file)
    
    # Set root logger level to INFO
    logging.getLogger().setLevel(logging.INFO)
    
    # Set httpx logger to WARNING level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

def read_config(cfg_path: str):
    """
    Read configuration file and set up logging.
    
    Args:
        cfg_path: Path to configuration file
        
    Returns:
        Tuple of (cfg, judge_properties, filters, temperature_overrides)
    """
    # Set up logging
    with open(cfg_path, encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)
    log_file = raw_cfg.get("logging", {}).get("log_file", "default.log")
    setup_logging(log_file)
    
    return raw_cfg
    
def calculate_aggregates(aggregates, all_scores, model_configs, task_configs):
    """
    Process aggregate metrics by calculating means across multiple tasks for a specific metric.
    
    Args:
        aggregates: List of aggregate configurations from the config file in format [metric_name, [task1, task2, ...]]
        all_scores: Dictionary of scores keyed by task_metric pairs
        model_configs: List of model configurations used for evaluation
        task_configs: List of task configs
    """
    logger.info("[calculate_aggregates] Processing aggregate metrics...")

    aggregate_scores = {}
    
    # Get unique model types
    model_types = set()
    for model_config in model_configs:
        model_type = model_config.get("model")  # The model type (e.g., "gpt-4o-mini-audio-preview")
        if model_type:
            model_types.add(model_type)
        
    # Get all groups of tasks. These are generally folders
    groups = get_groups(task_configs)

    # Get all tasks. These are generally YAML files
    all_tasks = get_tasks(task_configs)

    for agg_item in aggregates:
        # Skip invalid aggregates
        if not isinstance(agg_item, (list, tuple)) or len(agg_item) != 2:
            logger.warning(f"[calculate_aggregates] Invalid aggregate format: {agg_item}")
            continue
            
        metric_name, tasks = agg_item
        if not isinstance(tasks, list) or not tasks:
            logger.warning(f"[calculate_aggregates] Invalid tasks list in aggregate for metric '{metric_name}'")
            continue
        
        # Step 1: Look up metric keys from constants.py
        if metric_name not in constants.metric_output:
            logger.warning(f"[calculate_aggregates] Metric '{metric_name}' not found in metric_output dict")
            continue
        
        metric_keys = constants.metric_output[metric_name]
        
        # Step 2: Process each task
        processed_tasks = []  # For actual calculations

        for task in tasks:
            if task in groups.keys():
                # Add all the tasks belonging to the group
                processed_tasks.extend([x['task_name'] for x in task_configs[groups[task]]])
            elif task in all_tasks.keys():
                processed_tasks.append(task)
            else:
                raise ValueError(f"Invalid task name: {task}")

        if not processed_tasks:
            logger.warning(f"[calculate_aggregates] No valid tasks found for metric '{metric_name}'")
            continue
        
        # Step 3: Calculate aggregates for each model using the metric keys
        model_agg_scores = {}
        
        for model_type in model_types:
            model_scores = {}
            
            # For each metric key, collect values across all datasets
            for metric_key in metric_keys:
                values = []
                dataset_sizes = []
                
                # Process each dataset
                for task_name in processed_tasks:
                    try:
                        # Check if this dataset and model combination exists
                        if task_name in all_scores and model_type in all_scores[task_name]:
                            # Direct access to metrics from all_scores
                            metrics_dict = all_scores[task_name][model_type]
                            dataset_size = 1  # Default size if not specified
                            
                            # Check if this specific metric exists
                            if metric_key in metrics_dict[metric_name]:
                                value = metrics_dict[metric_name][metric_key]
                                if isinstance(value, (int, float)):
                                    values.append(value)
                                    dataset_sizes.append(dataset_size)
                        else:
                            logger.warning(f"[calculate_aggregates] Task '{task_name}' not found in all_scores")
                    except KeyError as e:
                        logger.warning(f"[calculate_aggregates] Error accessing data for {model_type} in {task_name}: {str(e)}")
                
                # Calculate weighted average for this metric key
                if values:
                    if sum(dataset_sizes) > 0:
                        weighted_avg = sum(v * w for v, w in zip(values, dataset_sizes)) / sum(dataset_sizes)
                        model_scores[metric_key] = weighted_avg
                    else:
                        # Fallback to simple mean if weights are all zero
                        model_scores[metric_key] = statistics.mean(values)
            
            # Add scores for this model
            if model_scores:
                model_agg_scores[model_type] = model_scores
            else:
                logger.warning(f"[calculate_aggregates] No scores to aggregate for {model_type} in '{metric_name}'")
        
        # Add aggregate scores to the results
        if model_agg_scores:
            # Create a key with metric name and original task names
            display_names_str = ", ".join(processed_tasks)
            aggregate_key = f"{metric_name} - {display_names_str}"
            aggregate_scores[aggregate_key] = model_agg_scores
    
    # Add aggregate scores to all_scores
    if aggregate_scores:
        all_scores["aggregates"] = aggregate_scores

def get_prompt_override(model: str, task_ancestry: list, prompt_overrides: dict, prompt_type_key: str) -> str | None:
    """Common helper function to get prompt overrides for a model and task combination.
    
    Args:
        model: The model
        task_ancestry: The ancestry path of the task (base_dir, intermediate dirs, task_name)
        prompt_overrides: Prompt override config from run config
        prompt_type_key: The key to look for in prompt_overrides ("system_prompts" or "user_prompts")
        
    Returns:
        Prompt if found, None otherwise
    """
    if not prompt_overrides or prompt_type_key not in prompt_overrides.keys():
        return None

    # Get the task name (last element in ancestry)
    task_name = task_ancestry[-1] if task_ancestry else None
    
    # Store best match score and prompt for hierarchical matching
    best_match_score = -1
    best_match_prompt = None
    
    for override in prompt_overrides[prompt_type_key]:
        prompt_override = override.get("prompt", None)
        if not prompt_override:
            continue
        
        # Check if this override applies to our model/task
        override_model = override.get("model", None)
        override_task = override.get("task", None)

        # Skip if model doesn't match and override has model constraint
        if override_model and override_model != model:
            continue

        # Calculate match score for hierarchical task matching
        match_score = -1
        
        # Case 1: Exact task name match (highest priority)
        if override_task == task_name:
            match_score = 100  # Highest priority
        
        # Case 2: Match with any folder in the ancestry path (medium priority)
        elif override_task and override_task in task_ancestry:
            # Find where in the hierarchy this folder/task appears
            # Items deeper in the hierarchy (closer to task) get higher scores
            position = task_ancestry.index(override_task)
            depth_score = position + 1  # Add 1 to avoid zero scores
            match_score = 10 + depth_score * 5  # Base of 10 plus position bonus
        
        # Case 3: No task constraint but model matches
        elif override_model == model and not override_task:
            match_score = 5  # Lower priority than task-specific overrides
        
        # Update best match if we found a better one
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_prompt = prompt_override
    
    return best_match_prompt if best_match_prompt else None


def get_system_prompt_override(model: str, task_ancestry: list, prompt_overrides: dict) -> str | None:
    """Get system prompt override for this model and task combination.
    
    Args:
        model: The model
        task_ancestry: The ancestry path of the task (base_dir, intermediate dirs, task_name)
        prompt_overrides: Prompt override config from run config
        
    Returns:
        System prompt if found, None otherwise
    """
    return get_prompt_override(model, task_ancestry, prompt_overrides, "system_prompts")


def get_instruction_prompt_override(model: str, task_ancestry: list, prompt_overrides: dict) -> str | None:
    """Get instruction prompt override for this model and task combination.
    
    Args:
        model: The model
        task_ancestry: The ancestry path of the task (base_dir, intermediate dirs, task_name)
        prompt_overrides: Prompt override config from run config
        
    Returns:
        Instruction prompt if found, None otherwise
    """
    return get_prompt_override(model, task_ancestry, prompt_overrides, "user_prompts")