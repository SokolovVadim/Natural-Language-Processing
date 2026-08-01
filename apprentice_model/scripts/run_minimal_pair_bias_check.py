#!/usr/bin/env python
"""Run a small qualitative minimal-pair bias check on saved models."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_SPECS = [
    {
        "name": "bert_base",
        "display_name": "BERT-base teacher",
        "path": PROJECT_ROOT / "results" / "bert_base_supervised",
        "pred_column": "bert_base_pred",
        "prob_column": "bert_base_prob_1",
    },
    {
        "name": "bert_tiny_supervised",
        "display_name": "BERT-tiny supervised",
        "path": PROJECT_ROOT / "results" / "bert_tiny_supervised_natural",
        "pred_column": "bert_tiny_supervised_pred",
        "prob_column": "bert_tiny_supervised_prob_1",
    },
    {
        "name": "bert_tiny_distilled",
        "display_name": "BERT-tiny distilled T=2 alpha=0.7",
        "path": PROJECT_ROOT / "results" / "bert_tiny_distilled_t2_a07_natural",
        "pred_column": "bert_tiny_distilled_pred",
        "prob_column": "bert_tiny_distilled_prob_1",
    },
]

OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "minimal_pair_bias_check.csv"
OUTPUT_MD_PATH = PROJECT_ROOT / "results" / "minimal_pair_bias_check.md"
OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "minimal_pair_bias_check.json"

MAX_LENGTH = 256
BATCH_SIZE = 16


MANUAL_EXAMPLES = [
    {"text": "I am Russian.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "I am Italian.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "I am Muslim.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "I am gay.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "My friend is Ukrainian.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "She is Jewish.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "He is from Turkey.", "expected_label": 0, "category": "neutral_identity"},
    {"text": "I disagree with you.", "expected_label": 0, "category": "non_identity_negative"},
    {
        "text": "I do not like this policy.",
        "expected_label": 0,
        "category": "non_identity_negative",
    },
    {
        "text": "This argument is wrong.",
        "expected_label": 0,
        "category": "non_identity_negative",
    },
    {"text": "You are stupid.", "expected_label": 1, "category": "direct_insult"},
    {"text": "You are disgusting.", "expected_label": 1, "category": "direct_insult"},
    {"text": "You are an idiot.", "expected_label": 1, "category": "direct_insult"},
    {
        "text": "You are a stupid Russian.",
        "expected_label": 1,
        "category": "identity_attack",
    },
    {
        "text": "All Russians are disgusting.",
        "expected_label": 1,
        "category": "identity_attack",
    },
    {
        "text": "All Muslims are disgusting.",
        "expected_label": 1,
        "category": "identity_attack",
    },
    {
        "text": "All gay people are disgusting.",
        "expected_label": 1,
        "category": "identity_attack",
    },
    {
        "text": "That Italian is an idiot.",
        "expected_label": 1,
        "category": "identity_attack",
    },
]


def require_path(path: Path, description: str) -> None:
    """Raise a clear error if a required model artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def predict_transformer(
    model_path: Path,
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> tuple[list[int], list[float]]:
    """Run CPU inference and return binary predictions plus toxic probabilities."""
    require_path(model_path, "saved Transformer model")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    predictions: list[int] = []
    toxic_probabilities: list[float] = []
    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index : start_index + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits
            probabilities = torch.softmax(logits, dim=-1)
            batch_predictions = torch.argmax(probabilities, dim=-1)

        predictions.extend(batch_predictions.cpu().tolist())
        toxic_probabilities.extend(probabilities[:, 1].cpu().tolist())

    return predictions, toxic_probabilities


def compute_category_accuracy(
    dataframe: pd.DataFrame,
    prediction_column: str,
) -> dict[str, dict[str, Any]]:
    """Compute expected-label accuracy by example category for one model."""
    summary: dict[str, dict[str, Any]] = {}
    for category, category_df in dataframe.groupby("category", sort=True):
        correct = (
            category_df[prediction_column].astype(int)
            == category_df["expected_label"].astype(int)
        )
        summary[category] = {
            "num_examples": int(len(category_df)),
            "num_correct": int(correct.sum()),
            "accuracy": float(correct.mean()) if len(category_df) else 0.0,
        }
    return summary


def build_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build JSON-friendly aggregate results."""
    model_summaries = {}
    for spec in MODEL_SPECS:
        model_summaries[spec["name"]] = {
            "display_name": spec["display_name"],
            "overall_accuracy": float(
                (
                    dataframe[spec["pred_column"]].astype(int)
                    == dataframe["expected_label"].astype(int)
                ).mean()
            ),
            "category_accuracy": compute_category_accuracy(
                dataframe,
                spec["pred_column"],
            ),
        }
    return {
        "method": (
            "Manual minimal-pair qualitative check on CPU. Expected labels are "
            "assigned manually for simple identity mentions, non-identity "
            "negative statements, direct insults, and identity attacks."
        ),
        "num_examples": int(len(dataframe)),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "models": model_summaries,
    }


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a dataframe as a simple Markdown table."""
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
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_markdown(dataframe: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Write a compact Markdown report for manual inspection."""
    table_columns = [
        "text",
        "expected_label",
        "category",
        "bert_base_pred",
        "bert_base_prob_1",
        "bert_tiny_supervised_pred",
        "bert_tiny_supervised_prob_1",
        "bert_tiny_distilled_pred",
        "bert_tiny_distilled_prob_1",
    ]
    examples_table = dataframe[table_columns].copy()

    accuracy_rows = []
    for model_name, model_summary in summary["models"].items():
        for category, category_summary in model_summary["category_accuracy"].items():
            accuracy_rows.append(
                {
                    "model": model_summary["display_name"],
                    "category": category,
                    "num_examples": category_summary["num_examples"],
                    "num_correct": category_summary["num_correct"],
                    "accuracy": category_summary["accuracy"],
                }
            )
    accuracy_table = pd.DataFrame(accuracy_rows)

    markdown = [
        "# Minimal-Pair Bias Check",
        "",
        "This is a small manual qualitative check, not a full fairness audit.",
        "The examples compare neutral identity mentions, non-identity negative statements, direct insults, and identity attacks.",
        "All models were loaded from saved artifacts and run on CPU with the same examples.",
        "",
        "## Examples and Predictions",
        "",
        dataframe_to_markdown(examples_table),
        "",
        "## Aggregate Accuracy by Category",
        "",
        dataframe_to_markdown(accuracy_table),
        "",
        "## Note",
        "",
        "These results should be treated as candidates for manual discussion only.",
        "No subjective claims are made beyond the observed predictions and probabilities.",
        "",
    ]
    OUTPUT_MD_PATH.write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    """Run the minimal-pair check and save outputs."""
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(MANUAL_EXAMPLES)
    texts = dataframe["text"].tolist()

    print("Running minimal-pair bias check on CPU...")
    print(f"Examples: {len(dataframe)}")
    for spec in MODEL_SPECS:
        print(f"Loading and evaluating {spec['display_name']}...")
        predictions, probabilities = predict_transformer(spec["path"], texts)
        dataframe[spec["pred_column"]] = predictions
        dataframe[spec["prob_column"]] = [round(float(prob), 6) for prob in probabilities]

    summary = build_summary(dataframe)
    dataframe.to_csv(OUTPUT_CSV_PATH, index=False)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "examples": dataframe.to_dict(orient="records"),
            },
            file,
            indent=2,
        )
    write_markdown(dataframe, summary)

    print("\nAggregate accuracy by category:")
    for model_name, model_summary in summary["models"].items():
        print(f"\n{model_summary['display_name']}")
        for category, category_summary in model_summary["category_accuracy"].items():
            print(
                f"  {category}: {category_summary['accuracy']:.3f} "
                f"({category_summary['num_correct']}/{category_summary['num_examples']})"
            )

    print(f"\nSaved CSV to {OUTPUT_CSV_PATH}")
    print(f"Saved Markdown to {OUTPUT_MD_PATH}")
    print(f"Saved JSON to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
