#!/usr/bin/env python
"""Train and evaluate the BERT-tiny neural baseline."""

from pathlib import Path
import json
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.torch_dataset import TextClassificationDataset
from src.models.student_model import load_student_model, load_student_tokenizer
from src.training.train_student import (
    evaluate_model,
    get_device,
    set_seed,
    train_student_model,
)


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
    print(f"  loss:      {metrics['loss']:.4f}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")


def make_data_loader(
    dataset: TextClassificationDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def main() -> None:
    """Train the neural student baseline and save metrics, predictions, model."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_config = config["dataset"]
    student_config = config["student_baseline"]
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]

    processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    output_model_dir = results_dir / "student_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(processed_dir / "train.csv", text_column, label_column)
    validation_df = load_split(
        processed_dir / "validation.csv",
        text_column,
        label_column,
    )
    test_df = load_split(processed_dir / "test.csv", text_column, label_column)

    model_name = student_config.get("model_name", "prajjwal1/bert-tiny")
    max_length = int(student_config.get("max_length", 128))
    batch_size = int(student_config.get("batch_size", 16))
    learning_rate = float(student_config.get("learning_rate", 2e-5))
    epochs = int(student_config.get("epochs", 3))
    num_labels = int(student_config.get("num_labels", 2))

    print("Training Transformer student baseline...")
    print(f"  model: {model_name}")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  epochs: {epochs}")
    print(f"  batch size: {batch_size}")

    tokenizer = load_student_tokenizer(model_name)
    model = load_student_model(model_name, num_labels=num_labels)

    train_dataset = TextClassificationDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    validation_dataset = TextClassificationDataset(
        dataframe=validation_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    test_dataset = TextClassificationDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )

    train_loader = make_data_loader(train_dataset, batch_size, shuffle=True, seed=seed)
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = make_data_loader(test_dataset, batch_size, shuffle=False, seed=seed)

    device = get_device()
    print(f"  device: {device}")

    history = train_student_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )

    validation_result = evaluate_model(model, validation_loader, device)
    test_result = evaluate_model(model, test_loader, device)

    metrics = {
        "model_name": model_name,
        "history": history,
        "validation": validation_result["metrics"],
        "test": test_result["metrics"],
    }

    metrics_path = results_dir / "student_baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions_df = test_df.copy()
    predictions_df["predicted_label"] = test_result["predictions"]
    predictions_df["predicted_positive_probability"] = test_result[
        "positive_probabilities"
    ]

    predictions_path = results_dir / "student_baseline_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    print_metric_summary("Validation", validation_result["metrics"])
    print_metric_summary("Test", test_result["metrics"])
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(f"Saved student baseline model to {output_model_dir}")


if __name__ == "__main__":
    main()
