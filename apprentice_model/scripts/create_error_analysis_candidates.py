#!/usr/bin/env python
"""Create error-analysis candidate tables for natural-split models."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPERVISED_PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "bert_tiny_supervised_natural_predictions.csv"
)
DISTILLED_PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "bert_tiny_distilled_t2_a07_natural_predictions.csv"
)
TEACHER_LOGITS_PATH = (
    PROJECT_ROOT / "data" / "teacher_logits" / "bert_base_test_logits.csv"
)
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "error_analysis_candidates.csv"
OUTPUT_MD_PATH = PROJECT_ROOT / "results" / "error_analysis_candidates.md"
OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "error_analysis_summary.json"

CATEGORY_ORDER = [
    "distillation_fixed_supervised_fp",
    "distillation_fixed_supervised_fn",
    "distillation_worsened_to_fp",
    "distillation_worsened_to_fn",
    "distilled_false_positive",
    "distilled_false_negative",
    "teacher_student_disagreement",
    "all_correct",
    "teacher_correct_distilled_wrong",
    "distilled_correct_teacher_wrong",
]


def fail(message: str) -> None:
    """Exit with a clear validation error."""
    raise SystemExit(f"Error-analysis candidate generation failed: {message}")


def read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a CSV file and validate required columns."""
    if not path.exists():
        fail(f"missing required file: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        fail(f"{path} is missing columns: {missing_columns}")
    return dataframe


def normalize_text_series(series: pd.Series) -> pd.Series:
    """Normalize text values for exact row-order alignment checks."""
    return series.fillna("").astype(str).reset_index(drop=True)


def label_series(series: pd.Series) -> pd.Series:
    """Normalize binary label/prediction values."""
    return series.astype(int).reset_index(drop=True)


def validate_alignment(
    teacher_df: pd.DataFrame,
    supervised_df: pd.DataFrame,
    distilled_df: pd.DataFrame,
) -> None:
    """Validate that all prediction files describe the same ordered examples."""
    lengths = {
        "teacher_logits": len(teacher_df),
        "bert_tiny_supervised": len(supervised_df),
        "bert_tiny_distilled": len(distilled_df),
    }
    if len(set(lengths.values())) != 1:
        fail(f"row counts differ: {lengths}")

    teacher_text = normalize_text_series(teacher_df["text"])
    supervised_text = normalize_text_series(supervised_df["text"])
    distilled_text = normalize_text_series(distilled_df["text"])
    if not teacher_text.equals(supervised_text):
        mismatch_index = int((teacher_text != supervised_text).idxmax())
        fail(
            "teacher logits and supervised predictions text order mismatch "
            f"at row {mismatch_index}"
        )
    if not teacher_text.equals(distilled_text):
        mismatch_index = int((teacher_text != distilled_text).idxmax())
        fail(
            "teacher logits and distilled predictions text order mismatch "
            f"at row {mismatch_index}"
        )

    teacher_labels = label_series(teacher_df["original_label"])
    supervised_labels = label_series(supervised_df["true_label"])
    distilled_labels = label_series(distilled_df["original_label"])
    if not teacher_labels.equals(supervised_labels):
        mismatch_index = int((teacher_labels != supervised_labels).idxmax())
        fail(
            "teacher logits and supervised predictions label mismatch "
            f"at row {mismatch_index}"
        )
    if not teacher_labels.equals(distilled_labels):
        mismatch_index = int((teacher_labels != distilled_labels).idxmax())
        fail(
            "teacher logits and distilled predictions label mismatch "
            f"at row {mismatch_index}"
        )


def truncate_excerpt(text: str, max_chars: int = 180) -> str:
    """Create a short, Markdown-safe text excerpt."""
    excerpt = re.sub(r"\s+", " ", str(text)).strip()
    excerpt = excerpt.replace("|", "\\|")
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3].rstrip() + "..."


def categories_for_row(row: pd.Series) -> list[str]:
    """Return all matching error-analysis categories for one example."""
    original = int(row["original_label"])
    teacher = int(row["teacher_pred"])
    supervised = int(row["bert_tiny_supervised_pred"])
    distilled = int(row["bert_tiny_distilled_pred"])

    categories: list[str] = []
    if original == 0 and supervised == 1 and distilled == 0:
        categories.append("distillation_fixed_supervised_fp")
    if original == 1 and supervised == 0 and distilled == 1:
        categories.append("distillation_fixed_supervised_fn")
    if original == 0 and supervised == 0 and distilled == 1:
        categories.append("distillation_worsened_to_fp")
    if original == 1 and supervised == 1 and distilled == 0:
        categories.append("distillation_worsened_to_fn")
    if original == 0 and distilled == 1:
        categories.append("distilled_false_positive")
    if original == 1 and distilled == 0:
        categories.append("distilled_false_negative")
    if teacher != distilled:
        categories.append("teacher_student_disagreement")
    if teacher == original and supervised == original and distilled == original:
        categories.append("all_correct")
    if teacher == original and distilled != original:
        categories.append("teacher_correct_distilled_wrong")
    if distilled == original and teacher != original:
        categories.append("distilled_correct_teacher_wrong")

    return categories or ["uncategorized"]


def build_candidates_dataframe(
    teacher_df: pd.DataFrame,
    supervised_df: pd.DataFrame,
    distilled_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full row-level error-analysis candidate dataframe."""
    candidates = pd.DataFrame(
        {
            "row_id": range(len(teacher_df)),
            "text_excerpt": [
                truncate_excerpt(text)
                for text in teacher_df["text"].fillna("").astype(str)
            ],
            "original_label": label_series(teacher_df["original_label"]),
            "teacher_pred": label_series(teacher_df["teacher_pred_default_threshold"]),
            "bert_tiny_supervised_pred": label_series(
                supervised_df["pred_label_default_threshold"]
            ),
            "bert_tiny_distilled_pred": label_series(
                distilled_df["student_pred_default_threshold"]
            ),
            "teacher_prob_1": teacher_df["teacher_prob_1"].astype(float),
            "supervised_prob_1": supervised_df["prob_1"].astype(float),
            "distilled_prob_1": distilled_df["student_prob_1"].astype(float),
        }
    )
    candidates["category"] = [
        ";".join(categories_for_row(row))
        for _, row in candidates.iterrows()
    ]
    return candidates


def count_categories(candidates_df: pd.DataFrame) -> dict[str, int]:
    """Count occurrences of each possibly overlapping category."""
    counts = {category: 0 for category in CATEGORY_ORDER}
    counts["uncategorized"] = 0
    for category_cell in candidates_df["category"]:
        for category in str(category_cell).split(";"):
            counts[category] = counts.get(category, 0) + 1
    return counts


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table."""
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


def examples_for_category(
    candidates_df: pd.DataFrame,
    category: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Return deterministic example rows for one category."""
    mask = candidates_df["category"].str.split(";").apply(lambda items: category in items)
    columns = [
        "row_id",
        "text_excerpt",
        "original_label",
        "teacher_pred",
        "bert_tiny_supervised_pred",
        "bert_tiny_distilled_pred",
        "teacher_prob_1",
        "supervised_prob_1",
        "distilled_prob_1",
    ]
    return candidates_df.loc[mask, columns].head(limit)


def write_markdown(candidates_df: pd.DataFrame, category_counts: dict[str, int]) -> None:
    """Write Markdown summary with counts and representative examples."""
    count_rows = [
        {"category": category, "count": count}
        for category, count in category_counts.items()
        if category != "uncategorized" or count > 0
    ]
    sections = [
        "# Error Analysis Candidates",
        "",
        "This file lists candidate examples for manual inspection.",
        "Excerpts are truncated to keep the table report-safe.",
        "",
        "## Category Counts",
        "",
        dataframe_to_markdown(pd.DataFrame(count_rows)),
        "",
    ]

    for category in CATEGORY_ORDER:
        count = category_counts.get(category, 0)
        if count == 0:
            continue
        examples = examples_for_category(candidates_df, category, limit=5)
        sections.extend(
            [
                f"## {category}",
                "",
                dataframe_to_markdown(examples),
                "",
            ]
        )

    OUTPUT_MD_PATH.write_text("\n".join(sections), encoding="utf-8")


def write_summary_json(
    candidates_df: pd.DataFrame,
    category_counts: dict[str, int],
) -> None:
    """Write machine-readable summary counts."""
    summary: dict[str, Any] = {
        "num_examples": int(len(candidates_df)),
        "alignment_verified": True,
        "source_files": {
            "teacher_logits": str(TEACHER_LOGITS_PATH),
            "bert_tiny_supervised_predictions": str(SUPERVISED_PREDICTIONS_PATH),
            "bert_tiny_distilled_predictions": str(DISTILLED_PREDICTIONS_PATH),
        },
        "category_counts": {
            category: int(count)
            for category, count in category_counts.items()
        },
        "output_files": {
            "csv": str(OUTPUT_CSV_PATH),
            "markdown": str(OUTPUT_MD_PATH),
        },
    }
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    """Create CSV, Markdown, and JSON error-analysis candidate artifacts."""
    teacher_df = read_csv(
        TEACHER_LOGITS_PATH,
        [
            "text",
            "original_label",
            "teacher_pred_default_threshold",
            "teacher_prob_1",
        ],
    )
    supervised_df = read_csv(
        SUPERVISED_PREDICTIONS_PATH,
        ["text", "true_label", "pred_label_default_threshold", "prob_1"],
    )
    distilled_df = read_csv(
        DISTILLED_PREDICTIONS_PATH,
        ["text", "original_label", "student_pred_default_threshold", "student_prob_1"],
    )

    validate_alignment(teacher_df, supervised_df, distilled_df)
    candidates_df = build_candidates_dataframe(
        teacher_df=teacher_df,
        supervised_df=supervised_df,
        distilled_df=distilled_df,
    )
    category_counts = count_categories(candidates_df)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(OUTPUT_CSV_PATH, index=False)
    write_markdown(candidates_df, category_counts)
    write_summary_json(candidates_df, category_counts)

    print("Error-analysis candidate generation complete.")
    print(f"Rows: {len(candidates_df)}")
    print("Category counts:")
    for category in CATEGORY_ORDER:
        print(f"  {category}: {category_counts.get(category, 0)}")
    if category_counts.get("uncategorized", 0):
        print(f"  uncategorized: {category_counts['uncategorized']}")
    print(f"Saved CSV to {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to {OUTPUT_MD_PATH}")
    print(f"Saved summary JSON to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
