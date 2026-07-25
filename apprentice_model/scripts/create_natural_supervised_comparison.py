#!/usr/bin/env python
"""Create supervised baseline comparison artifacts for natural splits."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


DEFAULT_RESULTS = [
    {
        "model": "TF-IDF + Logistic Regression",
        "threshold": 0.50,
        "accuracy": 0.8967,
        "precision": 0.3719,
        "recall": 0.4473,
        "f1": 0.4061,
        "tn": 2584,
        "fp": 179,
        "fn": 131,
        "tp": 106,
    },
    {
        "model": "BERT-tiny supervised",
        "threshold": 0.50,
        "accuracy": 0.8863,
        "precision": 0.3844,
        "recall": 0.7300,
        "f1": 0.5036,
        "tn": 2486,
        "fp": 277,
        "fn": 64,
        "tp": 173,
    },
    {
        "model": "BERT-base supervised teacher",
        "threshold": 0.50,
        "accuracy": 0.9313,
        "precision": 0.5508,
        "recall": 0.7089,
        "f1": 0.6199,
        "tn": 2626,
        "fp": 137,
        "fn": 69,
        "tp": 168,
    },
]


TUNED_RESULTS = [
    {
        "model": "TF-IDF + Logistic Regression",
        "best_validation_threshold": 0.52,
        "test_f1_at_default_threshold": 0.4061,
        "test_f1_at_tuned_threshold": 0.4000,
    },
    {
        "model": "BERT-tiny supervised",
        "best_validation_threshold": 0.75,
        "test_f1_at_default_threshold": 0.5036,
        "test_f1_at_tuned_threshold": 0.5434,
    },
    {
        "model": "BERT-base supervised teacher",
        "best_validation_threshold": 0.49,
        "test_f1_at_default_threshold": 0.6199,
        "test_f1_at_tuned_threshold": 0.6179,
    },
]


def write_markdown(
    default_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a compact Markdown comparison report."""
    default_table = dataframe_to_markdown(default_df)
    tuned_table = dataframe_to_markdown(tuned_df)
    markdown = [
        "# Natural Split Supervised Baseline Comparison",
        "",
        "## Default Threshold Comparison",
        "",
        default_table,
        "",
        "## Validation-Tuned Threshold Comparison",
        "",
        tuned_table,
        "",
        "## Interpretation",
        "",
        "- TF-IDF + Logistic Regression is the weakest baseline by F1.",
        "- BERT-tiny improves recall and F1, but over-predicts the toxic class.",
        "- BERT-base gives the best F1 and the strongest precision-recall balance.",
        "- This justifies using BERT-base as the teacher for soft-label distillation.",
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table."""
    headers = list(dataframe.columns)
    rows = []
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join("---" for _ in headers) + " |")

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


def main() -> None:
    """Create JSON, CSV, and Markdown comparison artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    default_df = pd.DataFrame(DEFAULT_RESULTS)
    tuned_df = pd.DataFrame(TUNED_RESULTS)

    comparison = {
        "split": "data/processed_natural",
        "default_threshold_results": DEFAULT_RESULTS,
        "validation_tuned_threshold_results": TUNED_RESULTS,
        "interpretation": [
            "TF-IDF + Logistic Regression is the weakest baseline.",
            "BERT-tiny improves recall and F1 but over-predicts the toxic class.",
            "BERT-base gives the best F1 and precision-recall balance.",
            "This justifies using BERT-base as the teacher for soft-label distillation.",
        ],
    }

    json_path = RESULTS_DIR / "natural_supervised_comparison.json"
    csv_path = RESULTS_DIR / "natural_supervised_comparison.csv"
    markdown_path = RESULTS_DIR / "natural_supervised_comparison.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    default_csv = default_df.assign(result_type="default_threshold")
    tuned_csv = tuned_df.assign(result_type="validation_tuned_threshold")
    combined_csv = pd.concat([default_csv, tuned_csv], ignore_index=True, sort=False)
    for count_column in ["tn", "fp", "fn", "tp"]:
        combined_csv[count_column] = combined_csv[count_column].astype("Int64")
    combined_csv.to_csv(csv_path, index=False)
    write_markdown(default_df, tuned_df, markdown_path)

    print(f"Saved JSON comparison to {json_path}")
    print(f"Saved CSV comparison to {csv_path}")
    print(f"Saved Markdown comparison to {markdown_path}")


if __name__ == "__main__":
    main()
