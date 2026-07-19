#!/usr/bin/env python
"""Fine-tune BERT-base as a supervised teacher model."""

from __future__ import annotations

from pathlib import Path
import copy
import json
import sys
import warnings

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from src.training.train_student import set_seed


DEFAULT_THRESHOLD = 0.5


def cuda_is_available() -> bool:
    """Check CUDA availability while suppressing noisy driver warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.cuda.is_available()


def resolve_device(device_config: str) -> torch.device:
    """Resolve configured device with CPU fallback."""
    if device_config == "auto":
        return torch.device("cuda" if cuda_is_available() else "cpu")

    device = torch.device(device_config)
    if device.type == "cuda" and not cuda_is_available():
        print("Configured CUDA device is unavailable; falling back to CPU.")
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


def compute_class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    """Compute class weights as N / (num_classes * count_c)."""
    counts = labels.astype(int).value_counts().sort_index()
    total_count = int(counts.sum())
    num_classes = 2
    weights = []

    for label in range(num_classes):
        label_count = int(counts.get(label, 0))
        if label_count == 0:
            raise ValueError(f"Cannot compute class weight for missing label={label}.")
        weights.append(total_count / (num_classes * label_count))

    return torch.tensor(weights, dtype=torch.float, device=device)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute classification metrics including support."""
    labels = [0, 1]
    support_counts = pd.Series(y_true).value_counts().sort_index()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "support": {
            str(label): int(support_counts.get(label, 0))
            for label in labels
        },
    }


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
) -> dict:
    """Evaluate model with weighted loss and return logits/probabilities."""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
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
            predictions = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            logits_rows.extend(logits.cpu().tolist())
            probability_rows.extend(probabilities.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(data_loader), 1)

    return {
        "metrics": metrics,
        "labels": y_true,
        "predictions": y_pred,
        "logits": logits_rows,
        "probabilities": probability_rows,
    }


def save_epoch_checkpoint(model, tokenizer, output_dir: Path, epoch: int) -> None:
    """Save an epoch checkpoint."""
    checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def save_history(history_rows: list[dict], output_dir: Path) -> None:
    """Save training history as CSV and JSON."""
    history_csv = output_dir / "training_history.csv"
    history_json = output_dir / "training_history.json"
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(history_csv, index=False)
    with history_json.open("w", encoding="utf-8") as file:
        json.dump(history_rows, file, indent=2)


def build_predictions_dataframe(
    dataframe: pd.DataFrame,
    eval_result: dict,
    text_column: str,
    label_column: str,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Build final test predictions dataframe."""
    logits = eval_result["logits"]
    probabilities = eval_result["probabilities"]
    prob_1 = [row[1] for row in probabilities]

    return pd.DataFrame(
        {
            "text": dataframe[text_column],
            "true_label": dataframe[label_column],
            "pred_label_default_threshold": predictions_from_threshold(
                prob_1,
                DEFAULT_THRESHOLD,
            ),
            "prob_0": [row[0] for row in probabilities],
            "prob_1": prob_1,
            "logit_0": [row[0] for row in logits],
            "logit_1": [row[1] for row in logits],
            "pred_label_tuned_threshold": predictions_from_threshold(
                prob_1,
                tuned_threshold,
            ),
        }
    )


def build_teacher_logits_dataframe(
    dataframe: pd.DataFrame,
    eval_result: dict,
    text_column: str,
    label_column: str,
    tuned_threshold: float,
) -> pd.DataFrame:
    """Build teacher logits/probabilities dataframe for one split."""
    logits = eval_result["logits"]
    probabilities = eval_result["probabilities"]
    prob_1 = [row[1] for row in probabilities]

    return pd.DataFrame(
        {
            "text": dataframe[text_column],
            "original_label": dataframe[label_column],
            "teacher_logit_0": [row[0] for row in logits],
            "teacher_logit_1": [row[1] for row in logits],
            "teacher_prob_0": [row[0] for row in probabilities],
            "teacher_prob_1": prob_1,
            "teacher_pred_default_threshold": predictions_from_threshold(
                prob_1,
                DEFAULT_THRESHOLD,
            ),
            "teacher_pred_tuned_threshold": predictions_from_threshold(
                prob_1,
                tuned_threshold,
            ),
        }
    )


def normalize_metrics(metrics: dict) -> dict:
    """Convert metric values into JSON-safe Python values."""
    normalized = {}
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
    return normalized


def is_cuda_oom(error: RuntimeError) -> bool:
    """Return whether an error looks like CUDA out-of-memory."""
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def run_training(config: dict, batch_size: int) -> dict:
    """Run BERT-base supervised fine-tuning with a given batch size."""
    teacher_config = config["bert_base_supervised"]
    text_column = teacher_config["text_column"]
    label_column = teacher_config["label_column"]
    seed = int(teacher_config.get("seed", config.get("seed", 42)))
    set_seed(seed)

    output_dir = PROJECT_ROOT / teacher_config["output_dir"]
    teacher_logits_dir = PROJECT_ROOT / teacher_config["teacher_logits_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_logits_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(PROJECT_ROOT / teacher_config["train_csv"], text_column, label_column)
    validation_df = load_split(
        PROJECT_ROOT / teacher_config["validation_csv"],
        text_column,
        label_column,
    )
    test_df = load_split(PROJECT_ROOT / teacher_config["test_csv"], text_column, label_column)

    model_name = teacher_config["model_name"]
    max_length = int(teacher_config.get("max_length", 256))
    learning_rate = float(teacher_config.get("learning_rate", 2e-5))
    weight_decay = float(teacher_config.get("weight_decay", 0.01))
    max_epochs = int(teacher_config.get("max_epochs", 5))
    patience = int(teacher_config.get("early_stopping_patience", 2))
    device = resolve_device(str(teacher_config.get("device", "auto")))

    print("Fine-tuning supervised BERT-base teacher...")
    print(f"  model: {model_name}")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  batch size: {batch_size}")
    print(f"  max epochs: {max_epochs}")
    print(f"  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    model.to(device)

    train_loader = make_data_loader(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    train_eval_loader = make_data_loader(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    validation_loader = make_data_loader(
        dataframe=validation_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = make_data_loader(
        dataframe=test_df,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
        batch_size=batch_size,
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

    history_rows: list[dict] = []
    best_validation_f1 = -1.0
    best_epoch = 0
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
        current_learning_rate = optimizer.param_groups[0]["lr"]

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
                "learning_rate": current_learning_rate,
            }
        )
        save_history(history_rows, output_dir)
        save_epoch_checkpoint(model, tokenizer, output_dir, epoch)

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_epoch = epoch
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
    validation_prob_1 = [
        row[1] for row in final_validation_result["probabilities"]
    ]
    test_prob_1 = [
        row[1] for row in final_test_result["probabilities"]
    ]
    threshold_sweep = sweep_thresholds(
        y_true=final_validation_result["labels"],
        prob_1=validation_prob_1,
        start=0.05,
        stop=0.95,
        step=0.01,
    )
    threshold_sweep_path = PROJECT_ROOT / "results" / "bert_base_supervised_threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    tuned_threshold, tuned_validation_metrics = select_best_threshold(threshold_sweep)
    default_test_metrics = compute_threshold_metrics(
        y_true=final_test_result["labels"],
        prob_1=test_prob_1,
        threshold=DEFAULT_THRESHOLD,
    )
    tuned_test_metrics = compute_threshold_metrics(
        y_true=final_test_result["labels"],
        prob_1=test_prob_1,
        threshold=tuned_threshold,
    )

    metrics_path = PROJECT_ROOT / "results" / "bert_base_supervised_metrics.json"
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "best_checkpoint_path": str(
            (output_dir / "checkpoints" / f"epoch_{best_epoch}").relative_to(PROJECT_ROOT)
        ),
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_test_metrics": normalize_metrics(default_test_metrics),
        "best_validation_threshold": tuned_threshold,
        "best_validation_threshold_metrics": normalize_metrics(tuned_validation_metrics),
        "tuned_threshold_test_metrics": normalize_metrics(tuned_test_metrics),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    validation_predictions_path = (
        PROJECT_ROOT / "results" / "bert_base_supervised_validation_predictions.csv"
    )
    build_predictions_dataframe(
        dataframe=validation_df,
        eval_result=final_validation_result,
        text_column=text_column,
        label_column=label_column,
        tuned_threshold=tuned_threshold,
    ).to_csv(validation_predictions_path, index=False)
    predictions_path = PROJECT_ROOT / "results" / "bert_base_supervised_predictions.csv"
    build_predictions_dataframe(
        dataframe=test_df,
        eval_result=final_test_result,
        text_column=text_column,
        label_column=label_column,
        tuned_threshold=tuned_threshold,
    ).to_csv(predictions_path, index=False)

    split_eval_items = {
        "train": (train_df, train_eval_loader),
        "validation": (validation_df, validation_loader),
        "test": (test_df, test_loader),
    }
    for split_name, (split_df, split_loader) in split_eval_items.items():
        split_result = evaluate_model_weighted(
            model=model,
            data_loader=split_loader,
            loss_fn=loss_fn,
            device=device,
        )
        logits_path = teacher_logits_dir / f"bert_base_{split_name}_logits.csv"
        build_teacher_logits_dataframe(
            dataframe=split_df,
            eval_result=split_result,
            text_column=text_column,
            label_column=label_column,
            tuned_threshold=tuned_threshold,
        ).to_csv(logits_path, index=False)

    print("\nBERT-base supervised teacher test metrics:")
    print(f"  default threshold: {DEFAULT_THRESHOLD:.2f}")
    print(f"  tuned threshold:   {tuned_threshold:.2f}")
    print(f"  accuracy:          {tuned_test_metrics['accuracy']:.4f}")
    print(f"  precision:         {tuned_test_metrics['precision']:.4f}")
    print(f"  recall:            {tuned_test_metrics['recall']:.4f}")
    print(f"  f1:                {tuned_test_metrics['f1']:.4f}")
    print(f"\nSaved best model to {output_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved threshold sweep to {threshold_sweep_path}")
    print(f"Saved validation predictions to {validation_predictions_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved teacher logits to {teacher_logits_dir}")

    return metrics_payload


def main() -> None:
    """Run supervised BERT-base fine-tuning, with small-batch OOM fallback."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    teacher_config = config["bert_base_supervised"]
    batch_size = int(teacher_config.get("batch_size", 8))
    fallback_batch_size = int(teacher_config.get("fallback_batch_size", 4))

    try:
        run_training(config, batch_size=batch_size)
    except RuntimeError as error:
        if batch_size > fallback_batch_size and is_cuda_oom(error):
            print(
                f"CUDA out of memory at batch_size={batch_size}; "
                f"retrying with batch_size={fallback_batch_size}."
            )
            torch.cuda.empty_cache()
            run_training(config, batch_size=fallback_batch_size)
        else:
            raise


if __name__ == "__main__":
    main()
