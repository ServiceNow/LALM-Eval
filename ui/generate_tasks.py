#!/usr/bin/env python3
"""
Script to dynamically generate tasks.json from the tasks folder structure.
Loads metrics directly from the YAML files under the tasks directory instead of parsing README tables.
"""

import json
from pathlib import Path
from typing import Dict, List

import yaml


def get_category_description(category):
    """Return a human-friendly description for a task category."""
    descriptions = {
        "speech_recognition": "Tasks involving automatic speech recognition (ASR), including standard ASR, long-form ASR, and code-switching ASR.",
        "paralinguistics": "Tasks that analyze non-verbal aspects of speech such as emotion, gender, accent, and speaker characteristics.",
        "audio_understanding": "Tasks that require understanding of the general audio signals including but not limited to music, noise, sound.",
        "spoken_language_understanding": "Tasks that require understanding of spoken language and/or audio information including QA, translation, summarization, and intent classification.",
        "spoken_language_reasoning": "Tasks that require reasoning over spoken input, such as instruction following or logical/mathematical reasoning.",
        "safety_and_security": "Tasks related to assessing model behavior around safety, robustness, and vulnerability to spoofing or adversarial content.",
        "speech_enhancement": "Tasks related to speech quality improvement, noise detection, and audio enhancement.",
        "speech_disorder": "Tasks related to detecting and analyzing speech disorders and voice pathologies.",
        "phonetics": "Tasks related to phonetic analysis, phoneme recognition, and speech sound processing."
    }

    return descriptions.get(category, f"Tasks related to {category.replace('_', ' ')}.")


def get_category_display_name(category):
    """Get display name with emoji for category."""
    display_names = {
        "speech_recognition": "🗣️ Speech Recognition",
        "paralinguistics": "🎭 Paralinguistics",
        "audio_understanding": "🔊 Audio Understanding",
        "spoken_language_understanding": "🧠 Spoken Language Understanding",
        "spoken_language_reasoning": "🧩 Spoken Language Reasoning",
        "safety_and_security": "🔐 Safety and Security",
        "speech_enhancement": "✨ Speech Enhancement",
        "speech_disorder": "🩺 Speech Disorder",
        "phonetics": "📢 Phonetics"
    }

    return display_names.get(category, category.replace('_', ' ').title())


def format_task_name(task_name):
    """Format task name for display."""
    return task_name.replace('_', ' ').title()


def safe_load_yaml(yaml_path: Path) -> Dict:
    """Safely load YAML content, returning an empty dict on failure."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"Warning: Failed to parse YAML file {yaml_path}: {exc}")
    except OSError as exc:
        print(f"Warning: Failed to read YAML file {yaml_path}: {exc}")
    return {}


def extract_metrics_from_yaml(data: Dict) -> List[str]:
    """Extract a list of metric names from a YAML object."""
    metrics = []
    yaml_metrics = data.get("metrics") if isinstance(data, dict) else None

    if isinstance(yaml_metrics, list):
        for item in yaml_metrics:
            if isinstance(item, dict) and "metric" in item:
                metrics.append(item["metric"])
            elif isinstance(item, str):
                metrics.append(item)

    # Deduplicate while preserving order
    seen = set()
    unique_metrics = []
    for metric in metrics:
        if metric and metric not in seen:
            seen.add(metric)
            unique_metrics.append(metric)

    return unique_metrics


def collect_metrics_from_task_dir(task_dir: Path) -> List[str]:
    """
    Collect metrics for a task directory by inspecting its YAML files.
    Preference is given to base.yaml if present; otherwise all YAML files under the directory
    are scanned until metrics are found.
    """
    yaml_files = []
    base_yaml = task_dir / "base.yaml"

    if base_yaml.exists():
        yaml_files.append(base_yaml)

    for yaml_path in sorted(task_dir.rglob("*.yaml")):
        if yaml_path == base_yaml:
            continue
        yaml_files.append(yaml_path)

    for yaml_file in yaml_files:
        data = safe_load_yaml(yaml_file)
        metrics = extract_metrics_from_yaml(data)
        if metrics:
            return metrics

    return []


def collect_configs_from_task_dir(task_dir: Path) -> List[str]:
    """Collect config identifiers from YAML files excluding base definitions."""
    configs = []

    for yaml_path in sorted(task_dir.rglob("*.yaml")):
        # Skip base.yaml files at any level
        if yaml_path.name.lower() == "base.yaml":
            continue

        data = safe_load_yaml(yaml_path)
        config_name = data.get("task_name") or yaml_path.stem
        if config_name and config_name not in configs:
            configs.append(config_name)

    return configs


def load_task_categories_from_yaml(tasks_dir: Path):
    """
    Build the task categories dictionary by traversing the tasks directory.
    Each top-level directory is treated as a category and each immediate
    sub-directory is treated as a task whose metrics/configs are discovered from YAML.
    Returns both the categories dictionary and a flat task metadata dictionary.
    """
    task_categories = {}
    task_details = {}

    for category_path in sorted(tasks_dir.iterdir(), key=lambda p: p.name):
        if not category_path.is_dir():
            continue

        category_key = category_path.name
        tasks = {}
        for task_path in sorted(category_path.iterdir(), key=lambda p: p.name):
            if not task_path.is_dir():
                continue

            metrics = collect_metrics_from_task_dir(task_path)
            if not metrics:
                continue

            task_key = task_path.name

            # Collect configs for this task
            configs = collect_configs_from_task_dir(task_path)

            task_info = {
                "name": format_task_name(task_key),
                "metrics": metrics
            }

            # Add configs to task_info if available
            if configs:
                task_info["configs"] = configs

            tasks[task_key] = task_info

            task_metadata = dict(task_info)
            task_metadata["category"] = category_key

            task_details[task_key] = task_metadata

        if not tasks:
            continue

        task_categories[category_key] = {
            "name": get_category_display_name(category_key),
            "description": get_category_description(category_key),
            "tasks": tasks
        }

    return task_categories, task_details


def main():
    """Main function to generate tasks.json and tasks.js from tasks folder."""

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    tasks_dir = project_root / "tasks"

    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found at {tasks_dir}")
        return

    task_categories, task_details = load_task_categories_from_yaml(tasks_dir)

    if not task_categories:
        print("No task categories were discovered from YAML files.")
        return

    combined_output = {**task_categories, **task_details}

    # Generate tasks.json
    json_output_file = script_dir / "tasks.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_output, f, indent=2, ensure_ascii=False)

    # Generate tasks.js for local file access (no server needed)
    js_output_file = script_dir / "tasks.js"
    with open(js_output_file, 'w', encoding='utf-8') as f:
        f.write("// Auto-generated by generate_tasks.py\n")
        f.write("// This file allows the UI to work when opening index.html directly in a browser\n")
        f.write("window.TASKS_DATA = ")
        json.dump(combined_output, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    print(f"Successfully generated {json_output_file} and {js_output_file} with {len(task_categories)} task categories")


if __name__ == "__main__":
    main()
