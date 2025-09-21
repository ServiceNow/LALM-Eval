import os
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from utils.util import get_class_from_module
from huggingface_hub import hf_hub_download, HfApi
from . import util
import logging
import os

logger = logging.getLogger(__name__)

def _load_callhome_dataset(repo, preprocessor_name, num_samples, properties):
    """Load and process a CallHome dataset using the specified preprocessor."""
    repo = Path(repo).resolve()
    # Dynamically load the preprocessor
    preprocessor_class = get_class_from_module('preprocessors', preprocessor_name)
    if preprocessor_class is None:
        error_msg = f"Could not load preprocessor {preprocessor_name}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    dataset = preprocessor_class().process(repo, num_samples=num_samples, properties=properties)
    dataset_size = len(dataset) if dataset else 0
    return dataset, dataset_size

def load_dataset_with_args(dataset_path: str, split: str, subset: str, task_name: str):
    """ Load the dataset
    
    Args:
        dataset_path: Path to dataset
        split: Split to load
        subset: Subset to load
        task_name: Name of the task
    
    Returns:
        dataset: Dataset loaded and transformed
    """
    if dataset_path is None:
        raise ValueError(f'Dataset path is missing for task {task_name}')
    
    if split is None:
        raise ValueError(f'Dataset split is missing for task {task_name}')
    
    # Load local environment file
    load_dotenv()

    token=os.getenv("HF_TOKEN")
    local_data_dir = os.getenv("LOCAL_DATA_DIR")
    api = HfApi()

    # Load dataset
    try: 
        dataset_load_args = {"path": dataset_path, "split": split, "trust_remote_code": True}
        if subset:
            dataset_load_args["name"] = subset
        if token:
            dataset_load_args["token"] = token

        # Handle processing separately for MMAU-Pro and MMAR
        if ('MMAU-Pro' in dataset_path or 'MMAR' in dataset_path):
            data_name = dataset_path.split('/')[-1].lower()
            private_local_path = os.path.join(local_data_dir, data_name)
            if not os.path.exists(private_local_path):
                os.mkdir(private_local_path)

            # Find all archive files
            files_info = api.list_repo_files(repo_id=dataset_path, repo_type="dataset")
            archive_files = []
            for file_info in files_info:
                if (file_info.endswith('.zip') or file_info.endswith('.tar.gz')):
                    archive_files.append(file_info)

            # Download, unzip and store all zip files into local_data_dir
            for archive_file in archive_files:
                archive_filename = archive_file.split('.')[0] # filename without .zip
                desired_audio_storge_path = os.path.join(private_local_path, archive_file)
                if (not os.path.exists(desired_audio_storge_path)):
                    audio_data_dir = hf_hub_download(
                        repo_id=dataset_path,
                        filename=archive_file,
                        repo_type="dataset",
                        local_dir = private_local_path
                    )
                    util.extract_archive(audio_data_dir, private_local_path)
        dataset = load_dataset(**dataset_load_args)
    except Exception as e:
        raise ValueError(e)

    if dataset is None:
        raise ValueError(f"Dataset with path {dataset_path}, split {split} and subset {subset} not found")
    
    return dataset