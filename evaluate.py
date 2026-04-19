import argparse
import asyncio
import logging
import json
# Apply nest_asyncio to allow nested event loops in Azure OpenAI client calls
import nest_asyncio

from utils.engine_utils import create_engine, run_all_engines
from utils.model_utils import register_models_with_controller
from utils.util import read_config, validate_config, calculate_aggregates
from utils.task_utils import find_all_task_configs, expand_task_metric_pairs

nest_asyncio.apply()

# Module-level logger - will be configured when setup_logging is called
logger = logging.getLogger(__name__)


def main(cfg_path='config.yaml'):
    """Main function to run the evaluation benchmark.

    Args:
        cfg_path: Path to configuration file

    Returns:
        Dictionary containing evaluation scores
    """
    # 1. Read run config
    run_config = read_config(cfg_path)

    # 2. Get all task configs and their ancestry
    task_configs, task_ancestry = find_all_task_configs()

    # Log task ancestry information for debugging
    if logger.isEnabledFor(logging.DEBUG):
        for task_name, ancestry in task_ancestry.items():
            logger.debug(f"Task '{task_name}' ancestry: {' > '.join(ancestry) if ancestry else 'root'}") 

    # 3. Validate config
    try:
        validate_config(run_config, task_configs)
    except ValueError as e:
        logger.error("[validate_config] Run config validation error: %s", e)
        raise

    # 4. Load models and initialize central request controller
    central_request_controller, model_configs = register_models_with_controller(run_config.get("models", []), run_config.get("judge_settings", {}))

    # 5. Expand task-metric pairs
    task_payload = expand_task_metric_pairs(run_config, task_configs, task_ancestry)

    # 6. Create engines for each expanded task-metric pair
    all_engines = []
    for task_info in task_payload:
        engine, task_name = create_engine(
            task_info=task_info,
            run_config=run_config,
            central_request_controller=central_request_controller
        )
        all_engines.append((engine, task_name))

    # 7. Run all engines concurrently
    scores = asyncio.run(run_all_engines(all_engines))

    # 8. Log final results and process aggregates
    aggregates = run_config.get("aggregate", [])
    if aggregates:
        calculate_aggregates(aggregates, scores, model_configs, task_configs)

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio evaluation benchmark')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    # Pass the config path to main
    all_scores = main(cfg_path=args.config)
    logger.info("[main] Evaluation complete. Final results:")
    logger.info(json.dumps(all_scores, indent=2))
