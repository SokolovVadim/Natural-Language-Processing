#!/usr/bin/env python
"""Tune BERT-base classification threshold from saved validation logits."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.threshold_tuning import (
    compute_threshold_metrics,
    predictions_from_threshold,
    select_best_threshold,
    sweep_thresholds,
)


DEFAULT_THRESHOLD = 0.5


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_required_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV and check required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {missing_columns}")
    return dataframe


def load_best_epoch(history_path: Path) -> tuple[int, float]:
    """Read training history and return the best validation-F1 epoch."""
    history = load_required_csv(
        history_path,
        [
            "epoch",
            "step",
            "train_loss",
            "eval_loss",
            "eval_accuracy",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "learning_rate",
        ],
    )
    best_row = history.loc[history["eval_f1"].idxmax()]
    return int(best_row["epoch"]), float(best_row["eval_f1"])


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy/pandas scalars to JSON-safe Python values."""
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            normalized[key] = [
                [int(item) for item in row]
                for row in value
            ]
        elif key in {"tn", "fp", "fn", "tp"}:
            normalized[key] = int(value)
        elif key == "threshold":
            normalized[key] = float(value)
        elif isinstance(value, float):
            normalized[key] = float(value)
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def build_prediction_export(
    logits_df: pd.DataFrame,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Create a predictions CSV from teacher logits."""
    return pd.DataFrame(
        {
            "text": logits_df["text"],
            "true_label": logits_df["original_label"].astype(int),
            "pred_label_default_threshold": predictions_from_threshold(
                logits_df["teacher_prob_1"],
                DEFAULT_THRESHOLD,
            ),
            "prob_0": logits_df["teacher_prob_0"],
            "prob_1": logits_df["teacher_prob_1"],
            "logit_0": logits_df["teacher_logit_0"],
            "logit_1": logits_df["teacher_logit_1"],
            "pred_label_tuned_threshold": predictions_from_threshold(
                logits_df["teacher_prob_1"],
                tuned_threshold,
            ),
        }
    )


def build_teacher_logits_export(
    logits_df: pd.DataFrame,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Create the canonical teacher-logits CSV schema."""
    return pd.DataFrame(
        {
            "text": logits_df["text"],
            "original_label": logits_df["original_label"].astype(int),
            "teacher_logit_0": logits_df["teacher_logit_0"],
            "teacher_logit_1": logits_df["teacher_logit_1"],
            "teacher_prob_0": logits_df["teacher_prob_0"],
            "teacher_prob_1": logits_df["teacher_prob_1"],
            "teacher_pred_default_threshold": predictions_from_threshold(
                logits_df["teacher_prob_1"],
                DEFAULT_THRESHOLD,
            ),
            "teacher_pred_tuned_threshold": predictions_from_threshold(
                logits_df["teacher_prob_1"],
                tuned_threshold,
            ),
        }
    )


def main() -> None:
    """Tune threshold and update BERT-base supervised artifacts."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    bert_config = config["bert_base_supervised"]

    output_dir = PROJECT_ROOT / bert_config["output_dir"]
    teacher_logits_dir = PROJECT_ROOT / bert_config["teacher_logits_dir"]
    results_dir = PROJECT_ROOT / "results"

    history_path = output_dir / "training_history.csv"
    best_epoch, best_validation_f1 = load_best_epoch(history_path)
    best_checkpoint_path = output_dir / "checkpoints" / f"epoch_{best_epoch}"
    saved_model_path = output_dir / "model.safetensors"
    checkpoint_model_path = best_checkpoint_path / "model.safetensors"

    if not checkpoint_model_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint file: {checkpoint_model_path}")
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Missing saved model file: {saved_model_path}")

    saved_model_hash = file_sha256(saved_model_path)
    best_checkpoint_hash = file_sha256(checkpoint_model_path)
    model_matches_best_checkpoint = saved_model_hash == best_checkpoint_hash

    required_logits_columns = [
        "text",
        "original_label",
        "teacher_logit_0",
        "teacher_logit_1",
        "teacher_prob_0",
        "teacher_prob_1",
    ]
    validation_logits = load_required_csv(
        teacher_logits_dir / "bert_base_validation_logits.csv",
        required_logits_columns,
    )
    test_logits = load_required_csv(
        teacher_logits_dir / "bert_base_test_logits.csv",
        required_logits_columns,
    )

    threshold_sweep = sweep_thresholds(
        y_true=validation_logits["original_label"],
        prob_1=validation_logits["teacher_prob_1"],
        start=0.05,
        stop=0.95,
        step=0.01,
    )
    sweep_path = results_dir / "bert_base_supervised_threshold_sweep.csv"
    threshold_sweep.to_csv(sweep_path, index=False)

    best_threshold, best_threshold_metrics = select_best_threshold(threshold_sweep)
    default_test_metrics = compute_threshold_metrics(
        y_true=test_logits["original_label"],
        prob_1=test_logits["teacher_prob_1"],
        threshold=DEFAULT_THRESHOLD,
    )
    tuned_test_metrics = compute_threshold_metrics(
        y_true=test_logits["original_label"],
        prob_1=test_logits["teacher_prob_1"],
        threshold=best_threshold,
    )

    validation_predictions_path = (
        results_dir / "bert_base_supervised_validation_predictions.csv"
    )
    test_predictions_path = results_dir / "bert_base_supervised_predictions.csv"
    build_prediction_export(
        validation_logits,
        tuned_threshold=best_threshold,
    ).to_csv(validation_predictions_path, index=False)
    build_prediction_export(
        test_logits,
        tuned_threshold=best_threshold,
    ).to_csv(test_predictions_path, index=False)

    for split_name in ("train", "validation", "test"):
        logits_path = teacher_logits_dir / f"bert_base_{split_name}_logits.csv"
        logits_df = load_required_csv(logits_path, required_logits_columns)
        build_teacher_logits_export(
            logits_df,
            tuned_threshold=best_threshold,
        ).to_csv(logits_path, index=False)

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "best_checkpoint_path": str(best_checkpoint_path.relative_to(PROJECT_ROOT)),
        "model_matches_best_checkpoint": model_matches_best_checkpoint,
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_test_metrics": normalize_metrics(default_test_metrics),
        "best_validation_threshold": best_threshold,
        "best_validation_threshold_metrics": normalize_metrics(best_threshold_metrics),
        "tuned_threshold_test_metrics": normalize_metrics(tuned_test_metrics),
    }
    metrics_path = results_dir / "bert_base_supervised_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("BERT-base threshold tuning complete.")
    print(f"  best epoch: {best_epoch}")
    print(f"  final model matches best checkpoint: {model_matches_best_checkpoint}")
    print(f"  best validation threshold: {best_threshold:.2f}")
    print(f"  validation F1 at tuned threshold: {best_threshold_metrics['f1']:.4f}")
    print(f"  test F1 at default threshold: {default_test_metrics['f1']:.4f}")
    print(f"  test F1 at tuned threshold: {tuned_test_metrics['f1']:.4f}")
    print(f"  saved threshold sweep to {sweep_path}")
    print(f"  updated metrics at {metrics_path}")


if __name__ == "__main__":
    main()
