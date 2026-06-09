"""Central logging setup

IMPORTANT: Call ``configure()`` ONLY ONCE from evaluate.py.
DO NOT call this from any other module! All logging will be sent to default.log in the project root.

Other modules should simply use:
  import logging
  logger = logging.getLogger(__name__)
"""
from __future__ import annotations
import logging
import csv
import re
import json
import datetime
from pathlib import Path
from itertools import zip_longest

# Default log file name for central logging
_DEFAULT_NAME = "default.log"

# Class to manage configuration state
class LoggingState:
    """Class to manage logging configuration state."""
    CONFIGURED = False
    LOGGING_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _install_handlers(log_path: Path):
    """(Re)configure the root logger to write to *log_path* and stderr."""
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        try:
            h.close()
        except (OSError, IOError):
            pass

    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)


def configure(log_file: str):
    """Configure root logger. Always uses default.log in the project root."""
    if LoggingState.CONFIGURED:
        return
    log_path = log_file or _DEFAULT_NAME
    path = Path(f"{LoggingState.LOGGING_TIMESTAMP}_{log_path}")

    _install_handlers(path)

    # Disable httpx INFO logs by setting its logger to WARNING level
    logging.getLogger("httpx").setLevel(logging.WARNING)

    LoggingState.CONFIGURED = True


def write_record_log(
    self, refs, cands, scores, task_name, model_name,
    explanations=None, instructions=None, model_responses=None, extras=None
):
    """
    Write record-level logs to a file specific to the dataset, metric, and model.

    Args:
        self: The metric object instance with a 'name' attribute
        refs: List of reference texts
        cands: List of candidate texts
        scores: List of scores
        task_name: Name of the dataset
        model_name: Name of the model
        explanations: Optional list of explanations for each score
        instructions: Optional list of instructions
        model_responses: Optional list of ModelResponse objects with detailed info
        extras: Optional dict mapping column name -> per-record list of values,
            appended after the standard columns
    """
    if not refs or not scores:
        return

    def _slug(s):
        return re.sub(r"[^A-Za-z0-9_]+", "_", s)

    log_dir = Path("run_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp_log_dir = log_dir / f"{LoggingState.LOGGING_TIMESTAMP}"
    timestamp_log_dir.mkdir(exist_ok=True)

    log_subdir = timestamp_log_dir / f"{_slug(task_name)}"
    log_subdir.mkdir(exist_ok=True)
    log_path = log_subdir / f"{_slug(task_name)}_{_slug(self.name)}_{_slug(model_name)}.csv"

    # Use provided explanations or an empty list
    if explanations is None:
        explanations = [""] * len(scores)

    # Use provided instructions or an empty list
    if instructions is None:
        instructions = [""] * len(scores)

    # Define headers for the CSV file
    headers = [
        "instruction", "reference", "candidate", "score", "explanation",
        "response_code", "raw_response_type", "wait_time",
        "error_rate_limit", "error_connection", "error_api",
        "error_timeout", "error_server", "error_other",
        "is_final_score"
    ]
    extra_columns = list(extras.keys()) if extras else []
    headers = headers + extra_columns

    with open(log_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        # Write the header row
        writer.writerow(headers)

        for i, (ref, cand, sc, expl, inst) in enumerate(
            zip_longest(refs, cands, scores, explanations, instructions, fillvalue=None)
        ):
            # Initialize with empty values
            row_values = [""] * len(headers)

            # Map values to their respective header positions
            header_to_index = {header: index for index, header in enumerate(headers)}

            # Set basic fields
            row_values[header_to_index["instruction"]] = inst if inst is not None else ""
            row_values[header_to_index["reference"]] = ref if ref is not None else ""
            row_values[header_to_index["candidate"]] = cand if cand is not None else ""
            row_values[header_to_index["explanation"]] = expl if expl else ""
            row_values[header_to_index["is_final_score"]] = "False"

            # Set score if available
            if sc is not None:
                row_values[header_to_index["score"]] = sc

            # Add ModelResponse data if available
            if model_responses and i < len(model_responses) and model_responses[i]:
                resp = model_responses[i]

                # Add core ModelResponse fields
                row_values[header_to_index["response_code"]] = resp.response_code
                row_values[header_to_index["raw_response_type"]] = type(resp.raw_response).__name__

                if hasattr(resp, "wait_time") and resp.wait_time is not None:
                    row_values[header_to_index["wait_time"]] = resp.wait_time

                # Add error tracker data if available
                if resp.error_tracker:
                    row_values[header_to_index["error_rate_limit"]] = resp.error_tracker.rate_limit
                    row_values[header_to_index["error_connection"]] = (
                        resp.error_tracker.connection_error
                    )
                    row_values[header_to_index["error_api"]] = resp.error_tracker.api_error
                    row_values[header_to_index["error_timeout"]] = (
                        resp.error_tracker.request_timeout
                    )
                    row_values[header_to_index["error_server"]] = resp.error_tracker.internal_server
                    row_values[header_to_index["error_other"]] = resp.error_tracker.other

            for col in extra_columns:
                values = extras.get(col, [])
                row_values[header_to_index[col]] = values[i] if i < len(values) else ""

            # Write the row
            writer.writerow(row_values)

    # Write to shared run.log
    write_to_run_json(
        self, refs, cands, scores, task_name, model_name, explanations, instructions
    )

    return log_path


# Flag to track if run.log has been reset for this session
class RunLogState:
    """Class to manage run log state."""
    reset = False

# Flag to track if final_scores.json has been reset for this session
class FinalScoresState:
    """Class to manage final scores state."""
    reset = False


def write_to_run_json(
    self, refs, cands, scores, task_name, model_name,
    explanations=None, instructions=None
):
    """
    Write each sample's prediction to a shared run.log file that resets with every run.
    The file is truncated on the first call to this function in each program execution.

    Args:
        self: The metric object instance with a 'name' attribute
        refs: List of reference texts
        cands: List of candidate texts
        scores: List of scores
        task_name: Name of the dataset
        model_name: Name of the model
        explanations: Optional list of explanations for each score
    """
    run_directory = Path("run_logs")
    run_directory.mkdir(exist_ok=True)
    run_timestamp_directory = run_directory / f"{LoggingState.LOGGING_TIMESTAMP}"
    run_timestamp_directory.mkdir(exist_ok=True)
    run_path = run_timestamp_directory / "run.log"

    # Use provided explanations or an empty list
    if explanations is None:
        explanations = [""] * len(scores)

    # Use provided instructions or an empty list
    if instructions is None:
        instructions = [""] * len(scores)

    # Determine file mode: 'w' to reset file on first call, 'a' to append on subsequent calls
    file_mode = "w" if not RunLogState.reset else "a"
    RunLogState.reset = True

    # Open run.log in appropriate mode
    with open(run_path, file_mode, encoding="utf-8") as f:
        # Add entries for this metric/dataset/model
        for ref, cand, sc, expl, inst in zip_longest(
            refs, cands, scores, explanations, instructions, fillvalue=None
        ):
            entry = {
                "dataset": task_name,
                "metric": self.name,
                "model": model_name,
                "instruction": inst,
                "reference": ref,
                "candidate": cand,
            }
            if sc is not None:
                entry["score"] = sc
            if expl:
                entry["explanation"] = expl
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_final_score(self, overall, task_name, model_name, model_responses=None):
    """
    Append the final aggregated score to final_scores.json in run_logs directory.

    Args:
        self: The metric object instance with a 'name' attribute
        overall: Dict containing overall metrics and scores
        task_name: Name of the dataset
        model_name: Name of the model
        model_responses: Optional list of ModelResponse objects for additional stats
    
    Returns:
        Path to the JSON file where the final score was appended
    """
    log_dir = Path("run_logs")
    log_dir.mkdir(exist_ok=True)
    timestamp_log_dir = log_dir / f"{LoggingState.LOGGING_TIMESTAMP}"
    timestamp_log_dir.mkdir(exist_ok = True)
    json_path = timestamp_log_dir / "final_scores.json"
    
    # Calculate additional statistics from model_responses
    total_samples = 0
    total_failures = 0
    avg_wait_time = 0.0
    
    if model_responses:
        total_samples = len(model_responses)
        total_failures = sum(1 for resp in model_responses if hasattr(resp, 'llm_response') and not resp.llm_response)
        
        # Calculate average wait time
        wait_times = [resp.wait_time for resp in model_responses if hasattr(resp, 'wait_time') and resp.wait_time is not None]
        if wait_times:
            avg_wait_time = sum(wait_times) / len(wait_times)

    # Create the score entry
    score_entry = {
        "dataset": task_name,
        "metric": self.name,
        "model": model_name,
        "score": overall,
        "total_samples": total_samples,
        "failures": total_failures,
        "avg_wait_time": avg_wait_time
    }

    # Load existing data or create new array
    existing_data = []
    
    # Only reset on the first call of a new run
    if not FinalScoresState.reset:
        FinalScoresState.reset = True
        existing_data = []  # Start fresh for new run
    else:
        # Load existing data for subsequent calls
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    existing_data = []

    # Add new entry to the data
    existing_data.append(score_entry)

    # Write the complete JSON array
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    return json_path
