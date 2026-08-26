# coding=utf-8
#
# Copyright 2026
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Renato Vukovic (renato.vukovic@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate hallucination-detection results on the RAGTruth benchmark.

The evaluator follows the response-level procedure used in
``evaluation_notebook.ipynb``. Ground-truth hallucinations are taken from the
RAGTruth labels, while a prediction is positive when the saved model response
contains either of the JSON decisions used by the inference prompts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_processing.ragtruth_processing import (
    combine_sources_and_responses,
    group_by_task_type,
    load_jsonl,
)
from inference.hallucination_detection_configs import get_config


@dataclass(frozen=True)
class EvaluationMetrics:
    """Classification metrics for one RAGTruth task and inference config."""

    config_name: str
    task_type: str
    sample_count: int
    error_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def load_ragtruth_data(dataset_directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and group RAGTruth responses by task type.

    Args:
        dataset_directory: Directory containing ``response.jsonl`` and
            ``source_info.jsonl``.

    Returns:
        A mapping from RAGTruth task type to combined source-response entries.
    """
    response_entries = load_jsonl(dataset_directory / "response.jsonl")
    source_entries = load_jsonl(dataset_directory / "source_info.jsonl")
    combined_entries = combine_sources_and_responses(
        response_entries,
        source_entries,
    )
    return group_by_task_type(combined_entries)


def extract_response(prediction: Any) -> str:
    """Extract a model response from either supported saved-result shape."""
    if isinstance(prediction, Mapping):
        response = prediction.get("response", "")
    else:
        response = prediction
    return response if isinstance(response, str) else ""


def predicts_hallucination(response: str) -> bool:
    """Return whether an inference response predicts a hallucination."""
    normalized_response = response.lower()
    return (
        '"hallucinations in response": true' in normalized_response
        or '"response is subset of reference": false' in normalized_response
    )


def calculate_metrics(
    config_name: str,
    task_type: str,
    labels: Sequence[bool],
    predictions: Sequence[bool],
    error_count: int,
) -> EvaluationMetrics:
    """Calculate response-level classification metrics."""
    matrix = confusion_matrix(labels, predictions, labels=[False, True])
    roc_auc: float | None
    if len(set(labels)) == 2:
        roc_auc = float(roc_auc_score(labels, predictions))
    else:
        roc_auc = None

    return EvaluationMetrics(
        config_name=config_name,
        task_type=task_type,
        sample_count=len(labels),
        error_count=error_count,
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=roc_auc,
        true_negatives=int(matrix[0, 0]),
        false_positives=int(matrix[0, 1]),
        false_negatives=int(matrix[1, 0]),
        true_positives=int(matrix[1, 1]),
    )


def evaluate_config(
    config_name: str,
    grouped_data: Mapping[str, Sequence[Mapping[str, Any]]],
    results_directory: Path,
) -> EvaluationMetrics:
    """Evaluate one saved inference configuration on its RAGTruth subset."""
    config = get_config(config_name)
    task_data = [
        entry
        for entry in grouped_data.get(config.tasktype, [])
        if entry.get("split") in config.splits
    ]
    data_by_id = {
        f"{entry.get('split')}-{entry.get('response_id')}": entry
        for entry in task_data
    }

    result_path = results_directory / f"{config_name}_responses.pt"
    predictions = torch.load(result_path, weights_only=False)
    labels: list[bool] = []
    predicted_labels: list[bool] = []
    error_count = 0

    for response_id, prediction in predictions.items():
        entry = data_by_id.get(response_id)
        if entry is None:
            continue

        response = extract_response(prediction)
        if not response:
            error_count += 1
        labels.append(bool(entry.get("hallucinations")))
        predicted_labels.append(predicts_hallucination(response))

    if not labels:
        raise ValueError(f"No matching RAGTruth datapoints found for {config_name}")

    return calculate_metrics(
        config_name,
        config.tasktype,
        labels,
        predicted_labels,
        error_count,
    )


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print a human-readable metric report for one configuration."""
    print(f"\n--- {metrics.config_name} ({metrics.task_type}) ---")
    print(f"Total datapoints: {metrics.sample_count}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1 Score:  {metrics.f1:.4f}")
    print(f"ROC-AUC:   {metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "ROC-AUC:   unavailable")
    print(
        "Confusion matrix: "
        f"TN={metrics.true_negatives}, FP={metrics.false_positives}, "
        f"FN={metrics.false_negatives}, TP={metrics.true_positives}"
    )
    if metrics.error_count:
        print(f"Warning: {metrics.error_count} empty or invalid responses.")


def print_latex_row(metrics: Sequence[EvaluationMetrics]) -> None:
    """Print the notebook-compatible task metrics and macro averages."""
    if not metrics:
        return
    macro_precision = sum(item.precision for item in metrics) / len(metrics)
    macro_recall = sum(item.recall for item in metrics) / len(metrics)
    macro_f1 = sum(item.f1 for item in metrics) / len(metrics)
    values = [value for item in metrics for value in (item.precision, item.recall, item.f1)]
    values.extend((macro_precision, macro_recall, macro_f1))
    print("\nLaTeX row:")
    print(" & ".join(f"{value * 100:.1f}" for value in values))


def main() -> None:
    """Parse command-line arguments and evaluate the requested configs."""
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_name",
        nargs="+",
        required=True,
        help="One or more RAGTruth inference config names.",
    )
    parser.add_argument(
        "--dataset_directory",
        type=Path,
        default=repository_root / "Data" / "ragtruth_data",
        help="Directory containing the RAGTruth JSONL files.",
    )
    parser.add_argument(
        "--results_directory",
        type=Path,
        default=repository_root / "src" / "inference" / "results",
        help="Directory containing saved inference responses.",
    )
    args = parser.parse_args()

    grouped_data = load_ragtruth_data(args.dataset_directory)
    metrics = [
        evaluate_config(config_name, grouped_data, args.results_directory)
        for config_name in args.config_name
    ]
    for item in metrics:
        print_metrics(item)
    if len(metrics) > 1:
        print_latex_row(metrics)


if __name__ == "__main__":
    main()