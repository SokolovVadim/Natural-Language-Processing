"""Evaluation metric helpers."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Compute standard binary classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing accuracy, precision, recall, F1 score, and
        confusion matrix.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
