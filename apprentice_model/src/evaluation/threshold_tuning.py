"""Helpers for threshold-based binary classification evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score


def compute_threshold_metrics(
    y_true: list[int] | pd.Series,
    prob_1: list[float] | pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Compute binary metrics after thresholding positive-class probabilities."""
    predictions = (pd.Series(prob_1).astype(float) >= threshold).astype(int)
    labels = pd.Series(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_thresholds(
    y_true: list[int] | pd.Series,
    prob_1: list[float] | pd.Series,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate thresholds from start to stop, inclusive."""
    thresholds = np.round(np.arange(start, stop + (step / 2), step), 2)
    rows = [
        compute_threshold_metrics(y_true=y_true, prob_1=prob_1, threshold=threshold)
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def select_best_threshold(sweep_df: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    """Select the threshold with highest validation F1."""
    if sweep_df.empty:
        raise ValueError("Cannot select a threshold from an empty sweep.")

    best_index = sweep_df["f1"].idxmax()
    best_row = sweep_df.loc[best_index]
    best_metrics = best_row.to_dict()

    if "confusion_matrix" not in best_metrics:
        best_metrics["confusion_matrix"] = [
            [int(best_metrics["tn"]), int(best_metrics["fp"])],
            [int(best_metrics["fn"]), int(best_metrics["tp"])],
        ]

    return float(best_row["threshold"]), best_metrics


def predictions_from_threshold(
    prob_1: list[float] | pd.Series,
    threshold: float,
) -> pd.Series:
    """Convert positive-class probabilities into hard labels."""
    return (pd.Series(prob_1).astype(float) >= threshold).astype(int)
