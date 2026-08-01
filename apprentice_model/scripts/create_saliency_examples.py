#!/usr/bin/env python
"""Create qualitative token-saliency examples for BERT-tiny models."""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
import warnings
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_EXAMPLES_PATH = PROJECT_ROOT / "results" / "error_analysis_report_examples.csv"
FULL_TEXT_SOURCE_PATH = (
    PROJECT_ROOT / "results" / "bert_tiny_distilled_t2_a07_natural_predictions.csv"
)
SUPERVISED_MODEL_PATH = PROJECT_ROOT / "results" / "bert_tiny_supervised_natural"
DISTILLED_MODEL_PATH = PROJECT_ROOT / "results" / "bert_tiny_distilled_t2_a07_natural"
OUTPUT_CSV_PATH = PROJECT_ROOT / "results" / "saliency_examples.csv"
OUTPUT_MD_PATH = PROJECT_ROOT / "results" / "saliency_examples.md"
OUTPUT_JSON_PATH = PROJECT_ROOT / "results" / "saliency_examples.json"

MAX_LENGTH = 256
TOP_K = 10
TOXIC_CLASS_INDEX = 1

warnings.filterwarnings(
    "ignore",
    message="CUDA initialization:.*",
    category=UserWarning,
)


MODEL_SPECS = [
    {
        "model_name": "bert_tiny_supervised",
        "display_name": "BERT-tiny supervised",
        "path": SUPERVISED_MODEL_PATH,
        "prediction_column": "supervised_pred",
        "probability_column": "supervised_prob_1",
    },
    {
        "model_name": "bert_tiny_distilled_t2_a07",
        "display_name": "BERT-tiny distilled T=2 alpha=0.7",
        "path": DISTILLED_MODEL_PATH,
        "prediction_column": "distilled_pred",
        "probability_column": "distilled_prob_1",
    },
]


MASK_PATTERNS = [
    (re.compile(r"\ba\s*hole\b", flags=re.IGNORECASE), "a***"),
    (re.compile(r"\basshole\b", flags=re.IGNORECASE), "a***"),
    (re.compile(r"\bfuck\w*\b", flags=re.IGNORECASE), "f***"),
    (re.compile(r"\bshit\w*\b", flags=re.IGNORECASE), "s***"),
    (re.compile(r"\bbitch\w*\b", flags=re.IGNORECASE), "b***"),
]


def fail(message: str) -> None:
    """Exit with a clear generation error."""
    raise SystemExit(f"Saliency example generation failed: {message}")


def require_path(path: Path, description: str) -> None:
    """Validate that a required file or directory exists."""
    if not path.exists():
        fail(f"missing {description}: {path}")


def mask_report_text(text: str) -> str:
    """Mask a small set of offensive words in report-facing text."""
    masked = str(text)
    for pattern, replacement in MASK_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def clean_excerpt(text: str, max_chars: int = 120) -> str:
    """Create a short Markdown-safe excerpt."""
    excerpt = re.sub(r"\s+", " ", mask_report_text(text)).strip()
    excerpt = excerpt.replace("|", "\\|")
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 3].rstrip() + "..."


def load_curated_examples() -> pd.DataFrame:
    """Load curated examples and validate columns."""
    require_path(CURATED_EXAMPLES_PATH, "curated error-analysis examples")
    examples = pd.read_csv(CURATED_EXAMPLES_PATH)
    required_columns = [
        "row_id",
        "case_type",
        "text_excerpt",
        "original_label",
        "supervised_pred",
        "distilled_pred",
        "supervised_prob_1",
        "distilled_prob_1",
        "interpretation",
    ]
    missing_columns = [
        column for column in required_columns if column not in examples.columns
    ]
    if missing_columns:
        fail(f"{CURATED_EXAMPLES_PATH} is missing columns: {missing_columns}")
    return examples


def load_full_texts() -> pd.DataFrame:
    """Load full test texts for row-id based lookup."""
    require_path(FULL_TEXT_SOURCE_PATH, "full-text prediction source")
    dataframe = pd.read_csv(FULL_TEXT_SOURCE_PATH)
    if "text" not in dataframe.columns:
        fail(f"{FULL_TEXT_SOURCE_PATH} is missing text column")
    return dataframe


def merge_wordpieces(tokens: list[str], scores: list[float]) -> list[tuple[str, float]]:
    """Merge WordPiece continuation tokens and aggregate saliency scores."""
    merged: list[tuple[str, float]] = []
    for token, score in zip(tokens, scores):
        if token.startswith("##") and merged:
            previous_token, previous_score = merged[-1]
            merged[-1] = (previous_token + token[2:], previous_score + score)
        else:
            merged.append((token, score))
    return merged


def compute_token_saliency(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, Any]:
    """Compute gradient-times-embedding token saliency for toxic-class logit."""
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids).detach()
    embeddings.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    toxic_logit = logits[0, TOXIC_CLASS_INDEX]
    toxic_logit.backward()

    if embeddings.grad is None:
        fail("embedding gradients were not populated")

    token_scores = torch.norm(embeddings.grad * embeddings, dim=-1)[0].detach().cpu()
    token_ids = input_ids[0].detach().cpu().tolist()
    attention_values = attention_mask[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    special_tokens = set(tokenizer.all_special_tokens)

    filtered_tokens: list[str] = []
    filtered_scores: list[float] = []
    for token, score, is_active in zip(tokens, token_scores.tolist(), attention_values):
        if not is_active:
            continue
        if token in special_tokens:
            continue
        filtered_tokens.append(mask_report_text(token))
        filtered_scores.append(float(score))

    merged = merge_wordpieces(filtered_tokens, filtered_scores)
    if not merged:
        top_tokens: list[str] = []
        top_scores: list[float] = []
    else:
        max_score = max(score for _, score in merged)
        normalized = [
            (token, score / max_score if max_score > 0 else 0.0)
            for token, score in merged
        ]
        normalized.sort(key=lambda item: item[1], reverse=True)
        top_items = normalized[:TOP_K]
        top_tokens = [token for token, _ in top_items]
        top_scores = [round(float(score), 4) for _, score in top_items]

    prediction = int(torch.argmax(probabilities, dim=-1).item())
    toxic_probability = float(probabilities[0, TOXIC_CLASS_INDEX].detach().cpu().item())
    return {
        "prediction": prediction,
        "toxic_probability": toxic_probability,
        "top_saliency_tokens": top_tokens,
        "top_saliency_scores": top_scores,
    }


def load_model_bundle(model_path: Path) -> tuple[Any, Any]:
    """Load tokenizer and model from a saved local directory."""
    require_path(model_path, "model directory")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model


def build_saliency_rows() -> list[dict[str, Any]]:
    """Compute saliency rows for each curated example and model."""
    curated_examples = load_curated_examples()
    full_text_df = load_full_texts()
    device = torch.device("cpu")
    rows: list[dict[str, Any]] = []

    for model_spec in MODEL_SPECS:
        print(f"Loading {model_spec['display_name']}...")
        tokenizer, model = load_model_bundle(model_spec["path"])
        model.to(device)

        for _, example in curated_examples.iterrows():
            row_id = int(example["row_id"])
            if row_id < 0 or row_id >= len(full_text_df):
                fail(f"row_id {row_id} is outside full-text source range")

            full_text = str(full_text_df.loc[row_id, "text"])
            saliency = compute_token_saliency(
                text=full_text,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            rows.append(
                {
                    "row_id": row_id,
                    "case_type": example["case_type"],
                    "text_excerpt": clean_excerpt(example["text_excerpt"]),
                    "original_label": int(example["original_label"]),
                    "model_name": model_spec["model_name"],
                    "model_prediction": saliency["prediction"],
                    "toxic_probability": round(saliency["toxic_probability"], 4),
                    "top_saliency_tokens": ", ".join(
                        saliency["top_saliency_tokens"]
                    ),
                    "top_saliency_scores": ", ".join(
                        f"{score:.4f}"
                        for score in saliency["top_saliency_scores"]
                    ),
                    "interpretation": example.get("interpretation", ""),
                    "reference_prediction": int(
                        example[model_spec["prediction_column"]]
                    ),
                    "reference_prob_1": round(
                        float(example[model_spec["probability_column"]]),
                        4,
                    ),
                }
            )

        del model

    return rows


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown."""
    headers = list(dataframe.columns)
    lines = [
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
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(saliency_df: pd.DataFrame) -> None:
    """Write grouped Markdown saliency examples."""
    sections = [
        "# Token Saliency Examples",
        "",
        "Gradient-based token saliency is qualitative and illustrative.",
        "It is not a formal proof of fairness, causality, or model reasoning.",
        "Scores are normalized per example and model.",
        "",
    ]

    for row_id, group in saliency_df.groupby("row_id", sort=False):
        first_row = group.iloc[0]
        sections.extend(
            [
                f"## row_id {row_id}: {first_row['case_type']}",
                "",
                f"Original label: {first_row['original_label']}",
                "",
                f"Excerpt: {first_row['text_excerpt']}",
                "",
                f"Interpretation: {first_row['interpretation']}",
                "",
            ]
        )
        table = group[
            [
                "model_name",
                "model_prediction",
                "toxic_probability",
                "top_saliency_tokens",
                "top_saliency_scores",
            ]
        ]
        sections.extend([dataframe_to_markdown(table), ""])

    OUTPUT_MD_PATH.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    """Compute saliency examples and write CSV, Markdown, and JSON outputs."""
    rows = build_saliency_rows()
    saliency_df = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    saliency_df.to_csv(OUTPUT_CSV_PATH, index=False)
    write_markdown(saliency_df)

    json_rows = saliency_df.to_dict(orient="records")
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "gradient_times_embedding_norm_for_toxic_class_logit",
                "qualitative_note": (
                    "Token saliency is illustrative and not causal proof."
                ),
                "max_length": MAX_LENGTH,
                "top_k": TOP_K,
                "rows": json_rows,
            },
            file,
            indent=2,
        )

    print(f"Saved saliency CSV to {OUTPUT_CSV_PATH}")
    print(f"Saved saliency Markdown to {OUTPUT_MD_PATH}")
    print(f"Saved saliency JSON to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
