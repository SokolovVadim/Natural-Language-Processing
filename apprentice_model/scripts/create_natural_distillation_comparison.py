#!/usr/bin/env python
"""Tune distilled-student threshold and compare natural-split models."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.threshold_tuning import (  # noqa: E402
    compute_threshold_metrics,
    predictions_from_threshold,
    select_best_threshold,
    sweep_thresholds,
)


RESULTS_DIR = PROJECT_ROOT / "results"
DISTILLED_PREFIX = "bert_tiny_distilled_t2_a07_natural"
DEFAULT_THRESHOLD = 0.5


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file with a clear missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a pretty JSON file."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def require_columns(dataframe: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Validate required dataframe columns."""
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {missing_columns}")


def normalize_metric_values(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metric values to JSON-safe Python scalars."""
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            normalized[key] = [
                [int(item) for item in row]
                for row in value
            ]
        elif key in {"tn", "fp", "fn", "tp"}:
            normalized[key] = int(value)
        elif key == "support":
            normalized[key] = {
                str(label): int(count)
                for label, count in value.items()
            }
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value

    if {"tn", "fp", "fn", "tp"}.issubset(normalized) and "support" not in normalized:
        normalized["support"] = {
            "0": int(normalized["tn"]) + int(normalized["fp"]),
            "1": int(normalized["fn"]) + int(normalized["tp"]),
        }
    return normalized


def tune_distilled_threshold() -> dict[str, Any]:
    """Tune distilled model threshold on validation and update artifacts."""
    metrics_path = RESULTS_DIR / f"{DISTILLED_PREFIX}_metrics.json"
    validation_predictions_path = RESULTS_DIR / f"{DISTILLED_PREFIX}_validation_predictions.csv"
    test_predictions_path = RESULTS_DIR / f"{DISTILLED_PREFIX}_predictions.csv"
    threshold_sweep_path = RESULTS_DIR / f"{DISTILLED_PREFIX}_threshold_sweep.csv"

    metrics = read_json(metrics_path)
    validation_df = pd.read_csv(validation_predictions_path)
    test_df = pd.read_csv(test_predictions_path)
    required_columns = ["original_label", "student_prob_1"]
    require_columns(validation_df, validation_predictions_path, required_columns)
    require_columns(test_df, test_predictions_path, required_columns)

    threshold_sweep = sweep_thresholds(
        y_true=validation_df["original_label"],
        prob_1=validation_df["student_prob_1"],
        start=0.05,
        stop=0.95,
        step=0.01,
    )
    best_threshold, best_validation_metrics = select_best_threshold(threshold_sweep)
    default_validation_metrics = compute_threshold_metrics(
        y_true=validation_df["original_label"],
        prob_1=validation_df["student_prob_1"],
        threshold=DEFAULT_THRESHOLD,
    )
    default_test_metrics = compute_threshold_metrics(
        y_true=test_df["original_label"],
        prob_1=test_df["student_prob_1"],
        threshold=DEFAULT_THRESHOLD,
    )
    tuned_test_metrics = compute_threshold_metrics(
        y_true=test_df["original_label"],
        prob_1=test_df["student_prob_1"],
        threshold=best_threshold,
    )

    validation_df["student_pred_tuned_threshold"] = predictions_from_threshold(
        validation_df["student_prob_1"],
        best_threshold,
    )
    test_df["student_pred_tuned_threshold"] = predictions_from_threshold(
        test_df["student_prob_1"],
        best_threshold,
    )
    validation_df.to_csv(validation_predictions_path, index=False)
    test_df.to_csv(test_predictions_path, index=False)
    threshold_sweep.drop(columns=["confusion_matrix"]).to_csv(
        threshold_sweep_path,
        index=False,
    )

    metrics["default_threshold"] = DEFAULT_THRESHOLD
    metrics["default_threshold_original_validation_metrics"] = normalize_metric_values(
        default_validation_metrics
    )
    metrics["default_threshold_original_test_metrics"] = normalize_metric_values(
        default_test_metrics
    )
    metrics["best_validation_threshold"] = best_threshold
    metrics["best_validation_threshold_metrics"] = normalize_metric_values(
        best_validation_metrics
    )
    metrics["tuned_threshold_original_test_metrics"] = normalize_metric_values(
        tuned_test_metrics
    )
    write_json(metrics_path, metrics)

    print(f"Saved distilled threshold sweep to {threshold_sweep_path}")
    print(f"Updated distilled metrics at {metrics_path}")
    print(f"Updated validation predictions at {validation_predictions_path}")
    print(f"Updated test predictions at {test_predictions_path}")
    return metrics


def metric_row(
    model: str,
    threshold: float,
    metrics: dict[str, Any],
    result_type: str,
) -> dict[str, Any]:
    """Build one flat comparison row."""
    return {
        "model": model,
        "result_type": result_type,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    }


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a compact GitHub-flavored Markdown table."""
    headers = list(dataframe.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in dataframe.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_markdown(
    default_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    recovery: dict[str, float],
    output_path: Path,
) -> None:
    """Write final Markdown comparison summary."""
    markdown = [
        "# Natural Split Distillation Comparison",
        "",
        "## Default Threshold Comparison",
        "",
        dataframe_to_markdown(default_df),
        "",
        "## Validation-Tuned Threshold Comparison",
        "",
        dataframe_to_markdown(tuned_df),
        "",
        "## F1 Gap Recovery",
        "",
        (
            f"Using default-threshold F1, the supervised BERT-tiny to BERT-base gap "
            f"is {recovery['gap']:.4f}. The distilled BERT-tiny recovers "
            f"{recovery['recovered']:.4f}, or {recovery['recovery_percent']:.2f}% "
            "of that gap."
        ),
        "",
        "## Interpretation",
        "",
        "- TF-IDF is the weakest baseline.",
        "- BERT-tiny supervised improves recall but has many false positives.",
        "- BERT-base is the strongest teacher.",
        "- Soft-label distillation improves BERT-tiny over the supervised BERT-tiny baseline.",
        "- Distilled BERT-tiny moves closer to BERT-base while keeping compact model size.",
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def create_comparison(distilled_metrics: dict[str, Any]) -> None:
    """Create final JSON, CSV, and Markdown comparison artifacts."""
    tfidf_metrics = read_json(RESULTS_DIR / "tfidf_natural_metrics.json")
    tiny_metrics = read_json(RESULTS_DIR / "bert_tiny_supervised_natural_metrics.json")
    base_metrics = read_json(RESULTS_DIR / "bert_base_supervised_metrics.json")

    default_results = [
        metric_row(
            "TF-IDF + Logistic Regression",
            DEFAULT_THRESHOLD,
            tfidf_metrics["default_threshold_test_metrics"],
            "default_threshold",
        ),
        metric_row(
            "BERT-tiny supervised",
            DEFAULT_THRESHOLD,
            tiny_metrics["default_threshold_test_metrics"],
            "default_threshold",
        ),
        metric_row(
            "BERT-base supervised teacher",
            DEFAULT_THRESHOLD,
            base_metrics["default_threshold_test_metrics"],
            "default_threshold",
        ),
        metric_row(
            "BERT-tiny distilled T=2 alpha=0.7",
            DEFAULT_THRESHOLD,
            distilled_metrics["default_threshold_original_test_metrics"],
            "default_threshold",
        ),
    ]
    tuned_results = [
        metric_row(
            "TF-IDF + Logistic Regression",
            tfidf_metrics["best_validation_threshold"],
            tfidf_metrics["tuned_threshold_test_metrics"],
            "validation_tuned_threshold",
        ),
        metric_row(
            "BERT-tiny supervised",
            tiny_metrics["best_validation_threshold"],
            tiny_metrics["tuned_threshold_test_metrics"],
            "validation_tuned_threshold",
        ),
        metric_row(
            "BERT-base supervised teacher",
            base_metrics["best_validation_threshold"],
            base_metrics["tuned_threshold_test_metrics"],
            "validation_tuned_threshold",
        ),
        metric_row(
            "BERT-tiny distilled T=2 alpha=0.7",
            distilled_metrics["best_validation_threshold"],
            distilled_metrics["tuned_threshold_original_test_metrics"],
            "validation_tuned_threshold",
        ),
    ]

    supervised_tiny_f1 = tiny_metrics["default_threshold_test_metrics"]["f1"]
    bert_base_f1 = base_metrics["default_threshold_test_metrics"]["f1"]
    distilled_f1 = distilled_metrics["default_threshold_original_test_metrics"]["f1"]
    gap = bert_base_f1 - supervised_tiny_f1
    recovered = distilled_f1 - supervised_tiny_f1
    recovery_percent = recovered / gap * 100 if gap else 0.0
    recovery = {
        "bert_base_default_f1": bert_base_f1,
        "bert_tiny_supervised_default_f1": supervised_tiny_f1,
        "bert_tiny_distilled_default_f1": distilled_f1,
        "gap": gap,
        "recovered": recovered,
        "recovery_percent": recovery_percent,
    }

    comparison = {
        "split": "data/processed_natural",
        "default_threshold_results": default_results,
        "validation_tuned_threshold_results": tuned_results,
        "f1_gap_recovery_default_threshold": recovery,
        "interpretation": [
            "TF-IDF is the weakest baseline.",
            "BERT-tiny supervised improves recall but has many false positives.",
            "BERT-base is the strongest teacher.",
            "Soft-label distillation improves BERT-tiny over the supervised BERT-tiny baseline.",
            "Distilled BERT-tiny moves closer to BERT-base while keeping compact model size.",
        ],
    }

    json_path = RESULTS_DIR / "natural_distillation_comparison.json"
    csv_path = RESULTS_DIR / "natural_distillation_comparison.csv"
    markdown_path = RESULTS_DIR / "natural_distillation_comparison.md"
    write_json(json_path, comparison)

    combined_csv = pd.concat(
        [pd.DataFrame(default_results), pd.DataFrame(tuned_results)],
        ignore_index=True,
        sort=False,
    )
    for count_column in ["tn", "fp", "fn", "tp"]:
        combined_csv[count_column] = combined_csv[count_column].astype("Int64")
    combined_csv.to_csv(csv_path, index=False)
    write_markdown(
        pd.DataFrame(default_results),
        pd.DataFrame(tuned_results),
        recovery,
        markdown_path,
    )

    print(f"Saved final comparison JSON to {json_path}")
    print(f"Saved final comparison CSV to {csv_path}")
    print(f"Saved final comparison Markdown to {markdown_path}")
    print(
        "Recovered "
        f"{recovery_percent:.2f}% of the default-threshold F1 gap "
        "between supervised BERT-tiny and BERT-base."
    )


def main() -> None:
    """Run Stage 3 threshold tuning and final comparison export."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    distilled_metrics = tune_distilled_threshold()
    create_comparison(distilled_metrics)


if __name__ == "__main__":
    main()
