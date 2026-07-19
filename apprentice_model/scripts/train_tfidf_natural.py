#!/usr/bin/env python
"""Train TF-IDF + Logistic Regression on natural-distribution splits."""

from __future__ import annotations

from pathlib import Path
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
from src.models.tfidf_baseline import TfidfLogisticRegressionBaseline


DEFAULT_THRESHOLD = 0.5


def load_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    """Load and validate one natural processed split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split CSV: {path}. "
            "Run `python scripts/prepare_natural_splits.py` first."
        )

    dataframe = pd.read_csv(path)
    missing_columns = [
        column
        for column in (text_column, label_column)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {missing_columns}")

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe[label_column] = dataframe[label_column].astype(int)
    return dataframe


def class_distribution(dataframe: pd.DataFrame, label_column: str) -> dict[str, Any]:
    """Return class counts and percentages for one split."""
    counts = dataframe[label_column].value_counts().sort_index()
    total = int(len(dataframe))
    label_0_count = int(counts.get(0, 0))
    label_1_count = int(counts.get(1, 0))
    return {
        "rows": total,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "label_0_percentage": label_0_count / total if total else 0.0,
        "label_1_percentage": label_1_count / total if total else 0.0,
    }


def get_probabilities(model: TfidfLogisticRegressionBaseline, texts: pd.Series) -> pd.DataFrame:
    """Return class-ordered probabilities for labels 0 and 1."""
    classifier = model.pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Expected classifier classes [0, 1], got {classes}")

    class_0_index = classes.index(0)
    class_1_index = classes.index(1)
    probabilities = model.predict_proba(texts)
    return pd.DataFrame(
        {
            "prob_0": [row[class_0_index] for row in probabilities],
            "prob_1": [row[class_1_index] for row in probabilities],
        }
    )


def build_predictions_dataframe(
    dataframe: pd.DataFrame,
    probabilities: pd.DataFrame,
    text_column: str,
    label_column: str,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Build validation/test prediction exports."""
    return pd.DataFrame(
        {
            "text": dataframe[text_column],
            "true_label": dataframe[label_column].astype(int),
            "pred_label_default_threshold": predictions_from_threshold(
                probabilities["prob_1"],
                DEFAULT_THRESHOLD,
            ),
            "pred_label_tuned_threshold": predictions_from_threshold(
                probabilities["prob_1"],
                tuned_threshold,
            ),
            "prob_0": probabilities["prob_0"],
            "prob_1": probabilities["prob_1"],
        }
    )


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metric values to JSON-safe Python values."""
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
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value

    tn = int(normalized["tn"])
    fp = int(normalized["fp"])
    fn = int(normalized["fn"])
    tp = int(normalized["tp"])
    normalized["support"] = {
        "0": tn + fp,
        "1": fn + tp,
    }
    return normalized


def save_top_features(
    model: TfidfLogisticRegressionBaseline,
    output_path: Path,
    top_n: int = 50,
) -> None:
    """Save top positive and negative Logistic Regression coefficients."""
    vectorizer = model.pipeline.named_steps["tfidf"]
    classifier = model.pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    positive = (
        feature_df.sort_values("coefficient", ascending=False)
        .head(top_n)
        .assign(direction="toxic")
    )
    negative = (
        feature_df.sort_values("coefficient", ascending=True)
        .head(top_n)
        .assign(direction="non-toxic")
    )
    top_features = pd.concat([positive, negative], ignore_index=True)
    top_features.to_csv(output_path, index=False)


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Print a compact metric summary."""
    print(f"\n{title}:")
    print(f"  threshold: {metrics['threshold']:.2f}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(
        "  confusion: "
        f"TN={metrics['tn']}, FP={metrics['fp']}, "
        f"FN={metrics['fn']}, TP={metrics['tp']}"
    )


def main() -> None:
    """Train and evaluate the natural-split TF-IDF baseline."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    seed = int(config.get("seed", 12345))
    natural_config = config["natural_splits"]
    text_column = config["dataset"]["text_column"]
    label_column = config["dataset"]["label_column"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    processed_natural_dir = PROJECT_ROOT / natural_config["output_dir"]
    train_csv = processed_natural_dir / "train.csv"
    validation_csv = processed_natural_dir / "validation.csv"
    test_csv = processed_natural_dir / "test.csv"

    train_df = load_split(train_csv, text_column, label_column)
    validation_df = load_split(validation_csv, text_column, label_column)
    test_df = load_split(test_csv, text_column, label_column)

    vectorizer_settings = {
        "max_features": 20_000,
        "ngram_range": [1, 2],
        "lowercase": True,
        "stop_words": None,
    }
    logistic_regression_settings = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": seed,
    }

    print("Training TF-IDF + Logistic Regression on natural splits...")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")

    model = TfidfLogisticRegressionBaseline(
        max_features=vectorizer_settings["max_features"],
        ngram_range=tuple(vectorizer_settings["ngram_range"]),
        lowercase=vectorizer_settings["lowercase"],
        stop_words=vectorizer_settings["stop_words"],
        class_weight=logistic_regression_settings["class_weight"],
        max_iter=logistic_regression_settings["max_iter"],
        solver=logistic_regression_settings["solver"],
        random_state=logistic_regression_settings["random_state"],
    )
    model.fit(train_df[text_column], train_df[label_column])

    validation_probabilities = get_probabilities(model, validation_df[text_column])
    test_probabilities = get_probabilities(model, test_df[text_column])

    threshold_sweep = sweep_thresholds(
        y_true=validation_df[label_column],
        prob_1=validation_probabilities["prob_1"],
        start=0.05,
        stop=0.95,
        step=0.01,
    )
    best_threshold, best_validation_metrics = select_best_threshold(threshold_sweep)

    default_validation_metrics = compute_threshold_metrics(
        y_true=validation_df[label_column],
        prob_1=validation_probabilities["prob_1"],
        threshold=DEFAULT_THRESHOLD,
    )
    default_test_metrics = compute_threshold_metrics(
        y_true=test_df[label_column],
        prob_1=test_probabilities["prob_1"],
        threshold=DEFAULT_THRESHOLD,
    )
    tuned_test_metrics = compute_threshold_metrics(
        y_true=test_df[label_column],
        prob_1=test_probabilities["prob_1"],
        threshold=best_threshold,
    )

    model_path = results_dir / "tfidf_natural_model.joblib"
    metrics_path = results_dir / "tfidf_natural_metrics.json"
    validation_predictions_path = results_dir / "tfidf_natural_validation_predictions.csv"
    test_predictions_path = results_dir / "tfidf_natural_predictions.csv"
    threshold_sweep_path = results_dir / "tfidf_natural_threshold_sweep.csv"
    top_features_path = results_dir / "tfidf_natural_top_features.csv"

    model.save(str(model_path))
    threshold_sweep.drop(columns=["confusion_matrix"]).to_csv(
        threshold_sweep_path,
        index=False,
    )
    build_predictions_dataframe(
        validation_df,
        validation_probabilities,
        text_column,
        label_column,
        best_threshold,
    ).to_csv(validation_predictions_path, index=False)
    build_predictions_dataframe(
        test_df,
        test_probabilities,
        text_column,
        label_column,
        best_threshold,
    ).to_csv(test_predictions_path, index=False)
    save_top_features(model, top_features_path)

    metrics = {
        "model_name": "tfidf_logistic_regression",
        "split": "processed_natural",
        "train_csv": str(train_csv.relative_to(PROJECT_ROOT)),
        "validation_csv": str(validation_csv.relative_to(PROJECT_ROOT)),
        "test_csv": str(test_csv.relative_to(PROJECT_ROOT)),
        "vectorizer_settings": vectorizer_settings,
        "logistic_regression_settings": logistic_regression_settings,
        "class_distribution": {
            "train": class_distribution(train_df, label_column),
            "validation": class_distribution(validation_df, label_column),
            "test": class_distribution(test_df, label_column),
        },
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_validation_metrics": normalize_metrics(
            default_validation_metrics
        ),
        "default_threshold_test_metrics": normalize_metrics(default_test_metrics),
        "best_validation_threshold": best_threshold,
        "best_validation_threshold_metrics": normalize_metrics(best_validation_metrics),
        "tuned_threshold_test_metrics": normalize_metrics(tuned_test_metrics),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print_metrics("Validation at default threshold", default_validation_metrics)
    print_metrics("Validation at tuned threshold", best_validation_metrics)
    print_metrics("Test at default threshold", default_test_metrics)
    print_metrics("Test at tuned threshold", tuned_test_metrics)
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved validation predictions to {validation_predictions_path}")
    print(f"Saved test predictions to {test_predictions_path}")
    print(f"Saved threshold sweep to {threshold_sweep_path}")
    print(f"Saved top features to {top_features_path}")


if __name__ == "__main__":
    main()
