#!/usr/bin/env python
"""Train a hard-label distilled BERT-tiny student on OpenAI teacher labels."""

from __future__ import annotations

from pathlib import Path
import copy
import json
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.torch_dataset import TextClassificationDataset
from src.evaluation.metrics import compute_classification_metrics
from src.models.student_model import load_student_model, load_student_tokenizer
from src.training.train_student import evaluate_model, get_device, set_seed, train_one_epoch


REQUIRED_TEACHER_COLUMNS = {
    "example_id",
    "text",
    "original_label",
    "teacher_label",
    "teacher_confidence",
    "teacher_reason",
}


def resolve_device(device_config: str) -> torch.device:
    """Resolve configured device, supporting automatic CPU/CUDA detection."""
    if device_config == "auto":
        return get_device()

    device = torch.device(device_config)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Configured CUDA device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def resolve_teacher_split_paths(config: dict, distilled_config: dict) -> dict[str, Path]:
    """Resolve teacher-label split paths, defaulting to finalized labels when configured."""
    labels_config_name = distilled_config.get("teacher_labels_config", "teacher_labels_final")
    labels_config = config.get(labels_config_name)

    if labels_config:
        return {
            "train": PROJECT_ROOT / labels_config["train_path"],
            "validation": PROJECT_ROOT / labels_config["validation_path"],
            "test": PROJECT_ROOT / labels_config["test_path"],
        }

    teacher_dir = PROJECT_ROOT / config["paths"]["teacher_labels_dir"]
    return {
        "train": teacher_dir / "train_teacher.csv",
        "validation": teacher_dir / "validation_teacher.csv",
        "test": teacher_dir / "test_teacher.csv",
    }


def load_distilled_components(
    distilled_config: dict,
    model_name: str,
    num_labels: int = 2,
):
    """Load tokenizer/model either from supervised checkpoint or the base model."""
    if distilled_config.get("initialize_from_supervised_student", False):
        checkpoint_path = PROJECT_ROOT / distilled_config["supervised_student_path"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing supervised student checkpoint: {checkpoint_path}")
        print(f"  initialization checkpoint: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
        )
        return tokenizer, model

    tokenizer = load_student_tokenizer(model_name)
    model = load_student_model(model_name, num_labels=num_labels)
    return tokenizer, model


def load_teacher_split(
    path: Path,
    required_columns: set[str],
    text_column: str,
    teacher_label_column: str,
    original_label_column: str,
) -> pd.DataFrame:
    """Load one teacher-labelled split and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing teacher-label split: {path}. "
            "Run `python scripts/generate_teacher_labels.py` first."
        )

    dataframe = pd.read_csv(path)
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {sorted(missing_columns)}")

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    dataframe[teacher_label_column] = dataframe[teacher_label_column].astype(int)
    dataframe[original_label_column] = dataframe[original_label_column].astype(int)
    return dataframe


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


def strip_loss(metrics: dict) -> dict:
    """Return metrics without loss for final JSON sections."""
    return {
        key: value
        for key, value in metrics.items()
        if key != "loss"
    }


def metrics_against_labels(labels: pd.Series, predictions: list[int]) -> dict:
    """Compute classification metrics against a dataframe label column."""
    return compute_classification_metrics(labels.astype(int).tolist(), predictions)


def build_predictions_dataframe(
    test_df: pd.DataFrame,
    predictions: list[int],
    positive_probabilities: list[float],
) -> pd.DataFrame:
    """Build the required test predictions output dataframe."""
    prob_toxic = positive_probabilities
    prob_not_toxic = [1.0 - probability for probability in positive_probabilities]

    return pd.DataFrame(
        {
            "example_id": test_df["example_id"],
            "text": test_df["text"],
            "original_label": test_df["original_label"],
            "teacher_label": test_df["teacher_label"],
            "predicted_label": predictions,
            "prob_not_toxic": prob_not_toxic,
            "prob_toxic": prob_toxic,
            "teacher_confidence": test_df["teacher_confidence"],
            "teacher_reason": test_df["teacher_reason"],
        }
    )


def print_metric_summary(title: str, metrics: dict) -> None:
    """Print a compact metric summary."""
    print(f"\n{title}:")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")


def print_label_distribution(name: str, labels: pd.Series | list[int]) -> None:
    """Print a small label distribution table."""
    counts = pd.Series(labels).astype(int).value_counts().sort_index()
    print(f"{name}:")
    for label in (0, 1):
        print(f"  label={label}: {int(counts.get(label, 0))}")


def main() -> None:
    """Train and evaluate a hard-label distilled BERT-tiny student."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    distilled_config = config["distilled_student"]

    seed = int(distilled_config.get("seed", 12345))
    set_seed(seed)

    teacher_split_paths = resolve_teacher_split_paths(config, distilled_config)
    output_dir = PROJECT_ROOT / distilled_config["output_dir"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    text_column = distilled_config["text_column"]
    teacher_label_column = distilled_config["teacher_label_column"]
    original_label_column = distilled_config["original_label_column"]

    required_columns = set(REQUIRED_TEACHER_COLUMNS)
    required_columns.update({text_column, teacher_label_column, original_label_column})

    train_df = load_teacher_split(
        teacher_split_paths["train"],
        required_columns,
        text_column,
        teacher_label_column,
        original_label_column,
    )
    validation_df = load_teacher_split(
        teacher_split_paths["validation"],
        required_columns,
        text_column,
        teacher_label_column,
        original_label_column,
    )
    test_df = load_teacher_split(
        teacher_split_paths["test"],
        required_columns,
        text_column,
        teacher_label_column,
        original_label_column,
    )

    model_name = distilled_config["model_name"]
    max_length = int(distilled_config.get("max_length", 128))
    batch_size = int(distilled_config.get("batch_size", 16))
    learning_rate = float(distilled_config.get("learning_rate", 2e-5))
    num_epochs = int(distilled_config.get("num_epochs", 5))
    weight_decay = float(distilled_config.get("weight_decay", 0.01))
    device = resolve_device(str(distilled_config.get("device", "auto")))

    print("Training hard-label distilled BERT-tiny student...")
    print(f"  model: {model_name}")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  target column: {teacher_label_column}")
    print(f"  train labels: {teacher_split_paths['train']}")
    print(f"  validation labels: {teacher_split_paths['validation']}")
    print(f"  test labels: {teacher_split_paths['test']}")
    print(f"  epochs: {num_epochs}")
    print(f"  device: {device}")

    tokenizer, model = load_distilled_components(
        distilled_config=distilled_config,
        model_name=model_name,
        num_labels=2,
    )

    train_dataset = TextClassificationDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=teacher_label_column,
        max_length=max_length,
    )
    validation_dataset = TextClassificationDataset(
        dataframe=validation_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=teacher_label_column,
        max_length=max_length,
    )
    test_dataset = TextClassificationDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=teacher_label_column,
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

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history_rows: list[dict] = []
    best_validation_f1 = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        validation_result = evaluate_model(model, validation_loader, device)
        validation_metrics = validation_result["metrics"]
        validation_f1 = validation_metrics["f1"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy_against_teacher": validation_metrics["accuracy"],
                "validation_precision_against_teacher": validation_metrics["precision"],
                "validation_recall_against_teacher": validation_metrics["recall"],
                "validation_f1_against_teacher": validation_metrics["f1"],
            }
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)

    validation_teacher_result = evaluate_model(model, validation_loader, device)
    test_teacher_result = evaluate_model(model, test_loader, device)

    validation_predictions = validation_teacher_result["predictions"]
    test_predictions = test_teacher_result["predictions"]

    validation_original_metrics = metrics_against_labels(
        validation_df[original_label_column],
        validation_predictions,
    )
    test_original_metrics = metrics_against_labels(
        test_df[original_label_column],
        test_predictions,
    )

    metrics = {
        "validation_against_teacher": strip_loss(
            validation_teacher_result["metrics"]
        ),
        "test_against_teacher": strip_loss(test_teacher_result["metrics"]),
        "validation_against_original": validation_original_metrics,
        "test_against_original": test_original_metrics,
    }

    metrics_path = results_dir / "student_distilled_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    predictions_df = build_predictions_dataframe(
        test_df=test_df,
        predictions=test_predictions,
        positive_probabilities=test_teacher_result["positive_probabilities"],
    )
    predictions_path = results_dir / "student_distilled_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    history_path = results_dir / "student_distilled_training_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nDistilled student results:")
    print_metric_summary(
        "Validation against teacher",
        metrics["validation_against_teacher"],
    )
    print_metric_summary("Test against teacher", metrics["test_against_teacher"])
    print_metric_summary(
        "Test against original labels",
        metrics["test_against_original"],
    )

    print("\nTest label distribution comparison:")
    print_label_distribution("Original labels", test_df[original_label_column])
    print_label_distribution("Teacher labels", test_df[teacher_label_column])
    print_label_distribution("Predicted labels", test_predictions)

    print(f"\nSaved best distilled model to {output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
