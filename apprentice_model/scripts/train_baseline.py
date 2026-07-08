#!/usr/bin/env python
"""Train and evaluate the TF-IDF + Logistic Regression baseline."""

from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.metrics import compute_classification_metrics
from src.models.tfidf_baseline import TfidfLogisticRegressionBaseline


def load_split(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    """Load one processed CSV split and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed split: {path}. "
            "Run `python scripts/prepare_data.py` first."
        )

    dataframe = pd.read_csv(path)
    missing_columns = {
        column
        for column in (text_column, label_column)
        if column not in dataframe.columns
    }
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {sorted(missing_columns)}")

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe[label_column] = dataframe[label_column].astype(int)
    return dataframe


def print_metric_summary(split_name: str, metrics: dict) -> None:
    """Print a readable metric summary for one split."""
    print(f"\n{split_name} metrics:")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")


def main() -> None:
    """Train the baseline, evaluate it, and save outputs."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    text_column = config["dataset"]["text_column"]
    label_column = config["dataset"]["label_column"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(processed_dir / "train.csv", text_column, label_column)
    validation_df = load_split(
        processed_dir / "validation.csv",
        text_column,
        label_column,
    )
    test_df = load_split(processed_dir / "test.csv", text_column, label_column)

    print("Training TF-IDF + Logistic Regression baseline...")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")

    model = TfidfLogisticRegressionBaseline(
        random_state=int(config.get("seed", 12345)),
    )
    model.fit(train_df[text_column], train_df[label_column])

    validation_predictions = model.predict(validation_df[text_column])
    test_predictions = model.predict(test_df[text_column])

    validation_metrics = compute_classification_metrics(
        validation_df[label_column],
        validation_predictions,
    )
    test_metrics = compute_classification_metrics(
        test_df[label_column],
        test_predictions,
    )

    all_metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_path = results_dir / "tfidf_baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = test_predictions
    probabilities = model.predict_proba(test_df[text_column])
    if probabilities and len(probabilities[0]) > 1:
        predictions_df["predicted_positive_probability"] = [
            row[1] for row in probabilities
        ]

    predictions_path = results_dir / "tfidf_baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    model_path = results_dir / "tfidf_baseline.joblib"
    model.save(str(model_path))

    print_metric_summary("Validation", validation_metrics)
    print_metric_summary("Test", test_metrics)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
