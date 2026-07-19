#!/usr/bin/env python
"""Fine-tune BERT-tiny on natural-distribution original labels."""

from __future__ import annotations

from pathlib import Path
import copy
import json
import sys
import warnings
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.torch_dataset import TextClassificationDataset
from src.evaluation.threshold_tuning import (
    compute_threshold_metrics,
    predictions_from_threshold,
    select_best_threshold,
    sweep_thresholds,
)
from src.models.student_model import load_student_model, load_student_tokenizer
from src.training.train_student import set_seed


DEFAULT_THRESHOLD = 0.5


def cuda_is_available() -> bool:
    """Check CUDA availability while suppressing noisy driver warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.cuda.is_available()


def mps_is_available() -> bool:
    """Return whether Apple MPS is available."""
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_device(device_config: str) -> torch.device:
    """Resolve configured device with accelerator fallback."""
    if device_config == "auto":
        if cuda_is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_config)
    if device.type == "cuda" and not cuda_is_available():
        print("Configured CUDA device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not mps_is_available():
        print("Configured MPS device is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


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


def make_data_loader(
    dataframe: pd.DataFrame,
    tokenizer,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic tokenized DataLoader."""
    dataset = TextClassificationDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


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


def compute_class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    """Compute class weights as N / (num_classes * count_c)."""
    counts = labels.astype(int).value_counts().sort_index()
    total_count = int(counts.sum())
    weights = []

    for label in range(2):
        label_count = int(counts.get(label, 0))
        if label_count == 0:
            raise ValueError(f"Cannot compute class weight for missing label={label}.")
        weights.append(total_count / (2 * label_count))

    return torch.tensor(weights, dtype=torch.float, device=device)


def train_one_epoch_weighted(
    model,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch with weighted cross-entropy."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress_bar:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        labels = batch.pop("labels")

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(data_loader), 1)


def evaluate_model_weighted(
    model,
    data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model and return metrics, logits, and probabilities."""
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    logits_rows: list[list[float]] = []
    probability_rows: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            probabilities = torch.softmax(logits, dim=-1)

            total_loss += loss.item()
            labels_all.extend(labels.cpu().tolist())
            logits_rows.extend(logits.cpu().tolist())
            probability_rows.extend(probabilities.cpu().tolist())

    prob_1 = [row[1] for row in probability_rows]
    metrics = compute_threshold_metrics(
        y_true=labels_all,
        prob_1=prob_1,
        threshold=DEFAULT_THRESHOLD,
    )
    metrics["loss"] = total_loss / max(len(data_loader), 1)

    return {
        "metrics": metrics,
        "labels": labels_all,
        "logits": logits_rows,
        "probabilities": probability_rows,
    }


def save_epoch_checkpoint(model, tokenizer, output_dir: Path, epoch: int) -> None:
    """Save one epoch checkpoint."""
    checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def save_history(history_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Save training history as CSV and JSON."""
    history_csv = output_dir / "training_history.csv"
    history_json = output_dir / "training_history.json"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    with history_json.open("w", encoding="utf-8") as file:
        json.dump(history_rows, file, indent=2)


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metric values to JSON-safe Python values and add support."""
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

    if {"tn", "fp", "fn", "tp"}.issubset(normalized):
        normalized["support"] = {
            "0": int(normalized["tn"]) + int(normalized["fp"]),
            "1": int(normalized["fn"]) + int(normalized["tp"]),
        }
    return normalized


def build_predictions_dataframe(
    dataframe: pd.DataFrame,
    eval_result: dict[str, Any],
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Build validation/test prediction exports."""
    logits = eval_result["logits"]
    probabilities = eval_result["probabilities"]
    prob_1 = [row[1] for row in probabilities]

    return pd.DataFrame(
        {
            "text": dataframe[text_column],
            "true_label": dataframe[label_column].astype(int),
            "pred_label_default_threshold": predictions_from_threshold(
                prob_1,
                DEFAULT_THRESHOLD,
            ),
            "prob_0": [row[0] for row in probabilities],
            "prob_1": prob_1,
            "logit_0": [row[0] for row in logits],
            "logit_1": [row[1] for row in logits],
        }
    )


def is_cuda_oom(error: RuntimeError) -> bool:
    """Return whether an error looks like accelerator out-of-memory."""
    message = str(error).lower()
    return ("cuda" in message or "mps" in message) and "out of memory" in message


def run_training(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Run BERT-tiny supervised fine-tuning with the given batch size."""
    tiny_config = config["bert_tiny_supervised_natural"]
    text_column = tiny_config["text_column"]
    label_column = tiny_config["label_column"]
    seed = int(tiny_config.get("seed", config.get("seed", 12345)))
    set_seed(seed)

    output_dir = PROJECT_ROOT / tiny_config["output_dir"]
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(PROJECT_ROOT / tiny_config["train_csv"], text_column, label_column)
    validation_df = load_split(
        PROJECT_ROOT / tiny_config["validation_csv"],
        text_column,
        label_column,
    )
    test_df = load_split(PROJECT_ROOT / tiny_config["test_csv"], text_column, label_column)

    model_name = tiny_config["model_name"]
    max_length = int(tiny_config.get("max_length", 256))
    learning_rate = float(tiny_config.get("learning_rate", 2e-5))
    weight_decay = float(tiny_config.get("weight_decay", 0.01))
    max_epochs = int(tiny_config.get("max_epochs", 5))
    patience = int(tiny_config.get("early_stopping_patience", 2))
    device = resolve_device(str(tiny_config.get("device", "auto")))

    print("Fine-tuning supervised BERT-tiny on natural splits...")
    print(f"  model: {model_name}")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  batch size: {batch_size}")
    print(f"  max epochs: {max_epochs}")
    print(f"  device: {device}")

    tokenizer = load_student_tokenizer(model_name)
    model = load_student_model(model_name, num_labels=2)
    model.to(device)

    train_loader = make_data_loader(
        train_df,
        tokenizer,
        text_column,
        label_column,
        max_length,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_data_loader(
        validation_df,
        tokenizer,
        text_column,
        label_column,
        max_length,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = make_data_loader(
        test_df,
        tokenizer,
        text_column,
        label_column,
        max_length,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    class_weights = compute_class_weights(train_df[label_column], device)
    print(f"  class weights: {class_weights.detach().cpu().tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history_rows: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_f1 = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_weighted(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )
        validation_result = evaluate_model_weighted(
            model=model,
            data_loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        validation_metrics = validation_result["metrics"]
        history_rows.append(
            {
                "epoch": epoch,
                "step": epoch * len(train_loader),
                "train_loss": train_loss,
                "eval_loss": validation_metrics["loss"],
                "eval_accuracy": validation_metrics["accuracy"],
                "eval_precision": validation_metrics["precision"],
                "eval_recall": validation_metrics["recall"],
                "eval_f1": validation_metrics["f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        save_history(history_rows, output_dir)
        save_epoch_checkpoint(model, tokenizer, output_dir, epoch)

        if validation_metrics["f1"] > best_validation_f1:
            best_epoch = epoch
            best_validation_f1 = validation_metrics["f1"]
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"eval_loss={validation_metrics['loss']:.4f}, "
            f"eval_f1={validation_metrics['f1']:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epoch(s).")
            break

    model.load_state_dict(best_state_dict)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_validation_result = evaluate_model_weighted(
        model=model,
        data_loader=validation_loader,
        loss_fn=loss_fn,
        device=device,
    )
    final_test_result = evaluate_model_weighted(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    validation_prob_1 = [row[1] for row in final_validation_result["probabilities"]]
    test_prob_1 = [row[1] for row in final_test_result["probabilities"]]
    threshold_sweep = sweep_thresholds(
        y_true=final_validation_result["labels"],
        prob_1=validation_prob_1,
        start=0.05,
        stop=0.95,
        step=0.01,
    )
    best_threshold, best_threshold_metrics = select_best_threshold(threshold_sweep)
    default_validation_metrics = compute_threshold_metrics(
        y_true=final_validation_result["labels"],
        prob_1=validation_prob_1,
        threshold=DEFAULT_THRESHOLD,
    )
    default_test_metrics = compute_threshold_metrics(
        y_true=final_test_result["labels"],
        prob_1=test_prob_1,
        threshold=DEFAULT_THRESHOLD,
    )
    tuned_test_metrics = compute_threshold_metrics(
        y_true=final_test_result["labels"],
        prob_1=test_prob_1,
        threshold=best_threshold,
    )

    metrics = {
        "model_name": model_name,
        "split": "processed_natural",
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_validation_metrics": normalize_metrics(
            default_validation_metrics
        ),
        "default_threshold_test_metrics": normalize_metrics(default_test_metrics),
        "best_validation_threshold": best_threshold,
        "best_validation_threshold_metrics": normalize_metrics(best_threshold_metrics),
        "tuned_threshold_test_metrics": normalize_metrics(tuned_test_metrics),
        "class_distribution": {
            "train": class_distribution(train_df, label_column),
            "validation": class_distribution(validation_df, label_column),
            "test": class_distribution(test_df, label_column),
        },
    }

    metrics_path = results_dir / "bert_tiny_supervised_natural_metrics.json"
    threshold_sweep_path = (
        results_dir / "bert_tiny_supervised_natural_threshold_sweep.csv"
    )
    validation_predictions_path = (
        results_dir / "bert_tiny_supervised_natural_validation_predictions.csv"
    )
    test_predictions_path = results_dir / "bert_tiny_supervised_natural_predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    threshold_sweep.drop(columns=["confusion_matrix"]).to_csv(
        threshold_sweep_path,
        index=False,
    )
    build_predictions_dataframe(
        validation_df,
        final_validation_result,
        text_column,
        label_column,
    ).to_csv(validation_predictions_path, index=False)
    build_predictions_dataframe(
        test_df,
        final_test_result,
        text_column,
        label_column,
    ).to_csv(test_predictions_path, index=False)

    print("\nBERT-tiny supervised natural results:")
    print(f"  best epoch: {best_epoch}")
    print(f"  best validation F1: {best_validation_f1:.4f}")
    print(f"  best validation threshold: {best_threshold:.2f}")
    print(f"  test F1 at default threshold: {default_test_metrics['f1']:.4f}")
    print(f"  test F1 at tuned threshold: {tuned_test_metrics['f1']:.4f}")
    print(f"\nSaved best model to {output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved threshold sweep to {threshold_sweep_path}")
    print(f"Saved validation predictions to {validation_predictions_path}")
    print(f"Saved test predictions to {test_predictions_path}")

    return metrics


def main() -> None:
    """Run supervised BERT-tiny natural training with small-batch OOM fallback."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    tiny_config = config["bert_tiny_supervised_natural"]
    batch_size = int(tiny_config.get("batch_size", 16))
    fallback_batch_size = int(tiny_config.get("fallback_batch_size", 8))

    try:
        run_training(config, batch_size=batch_size)
    except RuntimeError as error:
        if batch_size > fallback_batch_size and is_cuda_oom(error):
            print(
                f"Out of memory at batch_size={batch_size}; "
                f"retrying with batch_size={fallback_batch_size}."
            )
            if cuda_is_available():
                torch.cuda.empty_cache()
            run_training(config, batch_size=fallback_batch_size)
        else:
            raise


if __name__ == "__main__":
    main()
