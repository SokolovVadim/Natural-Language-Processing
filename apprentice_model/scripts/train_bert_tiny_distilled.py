#!/usr/bin/env python
"""Train BERT-tiny with soft-label distillation from BERT-base logits."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import copy
import json
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.distillation_dataset import (  # noqa: E402
    REQUIRED_TEACHER_LOGIT_COLUMNS,
    DistillationTextDataset,
)
from src.models.student_model import load_student_model, load_student_tokenizer  # noqa: E402
from src.training.distillation_loss import compute_distillation_loss  # noqa: E402
from src.training.train_student import set_seed  # noqa: E402


DEFAULT_THRESHOLD = 0.5
REQUIRED_PREDICTION_COLUMNS = [
    "text",
    "original_label",
    "teacher_pred_default_threshold",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_logit_0",
    "teacher_logit_1",
]


def parse_args() -> Namespace:
    """Parse distillation training arguments."""
    parser = ArgumentParser(
        description="Train BERT-tiny using BERT-base teacher logits."
    )
    parser.add_argument(
        "--train_logits_csv",
        default="data/teacher_logits/bert_base_train_logits.csv",
    )
    parser.add_argument(
        "--val_logits_csv",
        default="data/teacher_logits/bert_base_validation_logits.csv",
    )
    parser.add_argument(
        "--test_logits_csv",
        default="data/teacher_logits/bert_base_test_logits.csv",
    )
    parser.add_argument("--student_model_name", default="prajjwal1/bert-tiny")
    parser.add_argument("--teacher_model", default="bert-base-uncased")
    parser.add_argument(
        "--output_dir",
        default="results/bert_tiny_distilled_t2_a07_natural",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fallback_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    """Resolve absolute or project-relative paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


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


def validate_input_paths(args: Namespace) -> None:
    """Validate required teacher-logit inputs."""
    for argument_name in ["train_logits_csv", "val_logits_csv", "test_logits_csv"]:
        path = resolve_project_path(getattr(args, argument_name))
        if not path.exists():
            raise FileNotFoundError(f"Missing {argument_name}: {path}")


def load_logits_split(path: Path) -> pd.DataFrame:
    """Load one teacher-logit split and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher logits CSV: {path}")

    dataframe = pd.read_csv(path)
    required_columns = sorted(
        set(REQUIRED_TEACHER_LOGIT_COLUMNS + REQUIRED_PREDICTION_COLUMNS)
    )
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise KeyError(f"{path} is missing columns: {missing_columns}")

    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    dataframe["original_label"] = dataframe["original_label"].astype(int)
    dataframe["teacher_pred_default_threshold"] = dataframe[
        "teacher_pred_default_threshold"
    ].astype(int)

    numeric_columns = [
        "teacher_logit_0",
        "teacher_logit_1",
        "teacher_prob_0",
        "teacher_prob_1",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        if dataframe[column].isna().any():
            raise ValueError(f"{path} contains missing values in {column}")
        if not np.isfinite(dataframe[column]).all():
            raise ValueError(f"{path} contains non-finite values in {column}")

    labels = set(dataframe["original_label"].dropna().unique())
    teacher_predictions = set(
        dataframe["teacher_pred_default_threshold"].dropna().unique()
    )
    if not labels.issubset({0, 1}):
        raise ValueError(f"{path} original_label must contain only 0/1")
    if not teacher_predictions.issubset({0, 1}):
        raise ValueError(f"{path} teacher predictions must contain only 0/1")

    return dataframe


def class_distribution(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return class counts and percentages for original labels."""
    counts = dataframe["original_label"].value_counts().sort_index()
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
    weights: list[float] = []
    for label in range(2):
        label_count = int(counts.get(label, 0))
        if label_count == 0:
            raise ValueError(f"Cannot compute class weight for missing label={label}.")
        weights.append(total_count / (2 * label_count))
    return torch.tensor(weights, dtype=torch.float, device=device)


def make_data_loader(
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic distillation DataLoader."""
    dataset = DistillationTextDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
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


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Compute binary metrics with fixed 0/1 confusion-matrix ordering."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": {
            "0": int(tn + fp),
            "1": int(fn + tp),
        },
    }


def compute_teacher_imitation_metrics(
    teacher_predictions: list[int],
    student_predictions: list[int],
) -> dict[str, Any]:
    """Compute agreement and F1 against teacher hard predictions."""
    metrics = compute_binary_metrics(teacher_predictions, student_predictions)
    agreement = float(
        np.mean(np.array(teacher_predictions) == np.array(student_predictions))
    )
    metrics["agreement"] = agreement
    return metrics


def train_one_epoch_distilled(
    model,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
    class_weights: torch.Tensor | None,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch with hard-label CE plus soft teacher-logit KL."""
    model.train()
    total_loss_sum = 0.0
    hard_loss_sum = 0.0
    soft_raw_sum = 0.0
    soft_scaled_sum = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress_bar:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        labels = batch.pop("labels")
        teacher_logits = batch.pop("teacher_logits")

        optimizer.zero_grad()
        outputs = model(**batch)
        total_loss, hard_ce_loss, soft_kl_loss_raw = compute_distillation_loss(
            student_logits=outputs.logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=temperature,
            alpha=alpha,
            class_weights=class_weights,
        )
        total_loss.backward()
        optimizer.step()

        soft_kl_loss_scaled = (temperature ** 2) * soft_kl_loss_raw
        total_loss_sum += total_loss.item()
        hard_loss_sum += hard_ce_loss.item()
        soft_raw_sum += soft_kl_loss_raw.item()
        soft_scaled_sum += soft_kl_loss_scaled.item()
        progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

    denominator = max(len(data_loader), 1)
    return {
        "train_loss": total_loss_sum / denominator,
        "hard_ce_loss": hard_loss_sum / denominator,
        "soft_kl_loss_raw": soft_raw_sum / denominator,
        "soft_kl_loss_scaled": soft_scaled_sum / denominator,
    }


def evaluate_distilled_model(
    model,
    data_loader: DataLoader,
    device: torch.device,
    temperature: float,
    alpha: float,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    """Evaluate distilled student and return losses, logits, and predictions."""
    model.eval()
    total_loss_sum = 0.0
    hard_loss_sum = 0.0
    soft_raw_sum = 0.0
    soft_scaled_sum = 0.0
    labels_all: list[int] = []
    logits_rows: list[list[float]] = []
    probability_rows: list[list[float]] = []
    predictions_all: list[int] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            labels = batch.pop("labels")
            teacher_logits = batch.pop("teacher_logits")
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            total_loss, hard_ce_loss, soft_kl_loss_raw = compute_distillation_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature,
                alpha=alpha,
                class_weights=class_weights,
            )
            soft_kl_loss_scaled = (temperature ** 2) * soft_kl_loss_raw

            total_loss_sum += total_loss.item()
            hard_loss_sum += hard_ce_loss.item()
            soft_raw_sum += soft_kl_loss_raw.item()
            soft_scaled_sum += soft_kl_loss_scaled.item()
            labels_all.extend(labels.cpu().tolist())
            logits_rows.extend(logits.cpu().tolist())
            probability_rows.extend(probabilities.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    denominator = max(len(data_loader), 1)
    return {
        "losses": {
            "eval_loss": total_loss_sum / denominator,
            "hard_ce_loss": hard_loss_sum / denominator,
            "soft_kl_loss_raw": soft_raw_sum / denominator,
            "soft_kl_loss_scaled": soft_scaled_sum / denominator,
        },
        "labels": labels_all,
        "predictions": predictions_all,
        "logits": logits_rows,
        "probabilities": probability_rows,
    }


def save_epoch_checkpoint(model, tokenizer, output_dir: Path, epoch: int) -> Path:
    """Save one epoch checkpoint and return its path."""
    checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def save_history(history_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Save training history as CSV and JSON."""
    history_csv = output_dir / "training_history.csv"
    history_json = output_dir / "training_history.json"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    with history_json.open("w", encoding="utf-8") as file:
        json.dump(history_rows, file, indent=2)


def make_predictions_dataframe(
    dataframe: pd.DataFrame,
    eval_result: dict[str, Any],
) -> pd.DataFrame:
    """Build prediction CSV rows for validation or test."""
    logits = eval_result["logits"]
    probabilities = eval_result["probabilities"]
    return pd.DataFrame(
        {
            "text": dataframe["text"].tolist(),
            "original_label": dataframe["original_label"].astype(int).tolist(),
            "teacher_pred_default_threshold": dataframe[
                "teacher_pred_default_threshold"
            ].astype(int).tolist(),
            "student_pred_default_threshold": eval_result["predictions"],
            "student_prob_0": [row[0] for row in probabilities],
            "student_prob_1": [row[1] for row in probabilities],
            "student_logit_0": [row[0] for row in logits],
            "student_logit_1": [row[1] for row in logits],
            "teacher_prob_0": dataframe["teacher_prob_0"].astype(float).tolist(),
            "teacher_prob_1": dataframe["teacher_prob_1"].astype(float).tolist(),
            "teacher_logit_0": dataframe["teacher_logit_0"].astype(float).tolist(),
            "teacher_logit_1": dataframe["teacher_logit_1"].astype(float).tolist(),
        }
    )


def is_accelerator_oom(error: RuntimeError) -> bool:
    """Return whether an error looks like accelerator out-of-memory."""
    message = str(error).lower()
    return ("cuda" in message or "mps" in message) and "out of memory" in message


def run_training(args: Namespace, batch_size: int) -> dict[str, Any]:
    """Run soft-label distillation for BERT-tiny."""
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    set_seed(args.seed)
    validate_input_paths(args)

    train_path = resolve_project_path(args.train_logits_csv)
    validation_path = resolve_project_path(args.val_logits_csv)
    test_path = resolve_project_path(args.test_logits_csv)
    output_dir = resolve_project_path(args.output_dir)
    results_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_logits_split(train_path)
    validation_df = load_logits_split(validation_path)
    test_df = load_logits_split(test_path)

    device = resolve_device(args.device)
    print("Training BERT-tiny with BERT-base soft-label distillation...")
    print(f"  student model: {args.student_model_name}")
    print(f"  teacher model: {args.teacher_model}")
    print(f"  train rows: {len(train_df)}")
    print(f"  validation rows: {len(validation_df)}")
    print(f"  test rows: {len(test_df)}")
    print(f"  temperature: {args.temperature}")
    print(f"  alpha: {args.alpha}")
    print(f"  batch size: {batch_size}")
    print(f"  max epochs: {args.epochs}")
    print(f"  patience: {args.patience}")
    print(f"  device: {device}")

    tokenizer = load_student_tokenizer(args.student_model_name)
    model = load_student_model(args.student_model_name, num_labels=2)
    model.to(device)

    train_loader = make_data_loader(
        train_df,
        tokenizer,
        max_length=args.max_length,
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = make_data_loader(
        validation_df,
        tokenizer,
        max_length=args.max_length,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_data_loader(
        test_df,
        tokenizer,
        max_length=args.max_length,
        batch_size=batch_size,
        shuffle=False,
        seed=args.seed,
    )

    class_weights = compute_class_weights(train_df["original_label"], device)
    print(f"  hard-label CE class weights: {class_weights.detach().cpu().tolist()}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history_rows: list[dict[str, Any]] = []
    best_epoch = 0
    best_validation_f1_original = -1.0
    best_checkpoint_path: Path | None = None
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch_distilled(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
            class_weights=class_weights,
            epoch=epoch,
        )
        validation_result = evaluate_distilled_model(
            model=model,
            data_loader=validation_loader,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
            class_weights=class_weights,
        )
        validation_original_metrics = compute_binary_metrics(
            validation_result["labels"],
            validation_result["predictions"],
        )
        validation_teacher_metrics = compute_teacher_imitation_metrics(
            validation_df["teacher_pred_default_threshold"].astype(int).tolist(),
            validation_result["predictions"],
        )
        checkpoint_path = save_epoch_checkpoint(model, tokenizer, output_dir, epoch)

        history_row = {
            "epoch": epoch,
            "step": epoch * len(train_loader),
            "train_loss": train_losses["train_loss"],
            "hard_ce_loss": train_losses["hard_ce_loss"],
            "soft_kl_loss_raw": train_losses["soft_kl_loss_raw"],
            "soft_kl_loss_scaled": train_losses["soft_kl_loss_scaled"],
            "eval_loss": validation_result["losses"]["eval_loss"],
            "eval_accuracy_original": validation_original_metrics["accuracy"],
            "eval_precision_original": validation_original_metrics["precision"],
            "eval_recall_original": validation_original_metrics["recall"],
            "eval_f1_original": validation_original_metrics["f1"],
            "eval_teacher_agreement": validation_teacher_metrics["agreement"],
            "eval_f1_teacher": validation_teacher_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(history_row)
        save_history(history_rows, output_dir)

        if validation_original_metrics["f1"] > best_validation_f1_original:
            best_epoch = epoch
            best_validation_f1_original = validation_original_metrics["f1"]
            best_checkpoint_path = checkpoint_path
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_losses['train_loss']:.4f}, "
            f"hard_ce={train_losses['hard_ce_loss']:.4f}, "
            f"soft_kl_raw={train_losses['soft_kl_loss_raw']:.4f}, "
            f"soft_kl_scaled={train_losses['soft_kl_loss_scaled']:.4f}, "
            f"eval_loss={validation_result['losses']['eval_loss']:.4f}, "
            f"eval_f1_original={validation_original_metrics['f1']:.4f}, "
            f"eval_teacher_agreement={validation_teacher_metrics['agreement']:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epoch(s).")
            break

    model.load_state_dict(best_state_dict)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_validation_result = evaluate_distilled_model(
        model=model,
        data_loader=validation_loader,
        device=device,
        temperature=args.temperature,
        alpha=args.alpha,
        class_weights=class_weights,
    )
    final_test_result = evaluate_distilled_model(
        model=model,
        data_loader=test_loader,
        device=device,
        temperature=args.temperature,
        alpha=args.alpha,
        class_weights=class_weights,
    )

    validation_original_metrics = compute_binary_metrics(
        final_validation_result["labels"],
        final_validation_result["predictions"],
    )
    test_original_metrics = compute_binary_metrics(
        final_test_result["labels"],
        final_test_result["predictions"],
    )
    validation_teacher_metrics = compute_teacher_imitation_metrics(
        validation_df["teacher_pred_default_threshold"].astype(int).tolist(),
        final_validation_result["predictions"],
    )
    test_teacher_metrics = compute_teacher_imitation_metrics(
        test_df["teacher_pred_default_threshold"].astype(int).tolist(),
        final_test_result["predictions"],
    )

    metrics = {
        "model_name": args.student_model_name,
        "teacher_model": args.teacher_model,
        "split": "processed_natural",
        "temperature": args.temperature,
        "alpha": args.alpha,
        "best_epoch": best_epoch,
        "best_validation_f1_original": best_validation_f1_original,
        "best_checkpoint_path": (
            str(best_checkpoint_path) if best_checkpoint_path is not None else None
        ),
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_original_validation_metrics": validation_original_metrics,
        "default_threshold_original_test_metrics": test_original_metrics,
        "teacher_imitation_validation_metrics": validation_teacher_metrics,
        "teacher_imitation_test_metrics": test_teacher_metrics,
        "class_distribution": {
            "train": class_distribution(train_df),
            "validation": class_distribution(validation_df),
            "test": class_distribution(test_df),
        },
        "training_config": {
            "train_logits_csv": str(train_path),
            "validation_logits_csv": str(validation_path),
            "test_logits_csv": str(test_path),
            "max_length": args.max_length,
            "batch_size": batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(device),
        },
    }

    metrics_path = results_dir / "bert_tiny_distilled_t2_a07_natural_metrics.json"
    validation_predictions_path = (
        results_dir / "bert_tiny_distilled_t2_a07_natural_validation_predictions.csv"
    )
    test_predictions_path = (
        results_dir / "bert_tiny_distilled_t2_a07_natural_predictions.csv"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    make_predictions_dataframe(validation_df, final_validation_result).to_csv(
        validation_predictions_path,
        index=False,
    )
    make_predictions_dataframe(test_df, final_test_result).to_csv(
        test_predictions_path,
        index=False,
    )

    print("\nBERT-tiny distilled results:")
    print(f"  best epoch: {best_epoch}")
    print(f"  best validation F1 against original labels: {best_validation_f1_original:.4f}")
    print(
        "  validation F1 against teacher predictions: "
        f"{validation_teacher_metrics['f1']:.4f}"
    )
    print(f"  test F1 against original labels: {test_original_metrics['f1']:.4f}")
    print(f"  test F1 against teacher predictions: {test_teacher_metrics['f1']:.4f}")
    print(f"\nSaved best model to {output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved validation predictions to {validation_predictions_path}")
    print(f"Saved test predictions to {test_predictions_path}")
    print(f"Saved training history to {output_dir / 'training_history.csv'}")

    return metrics


def main() -> None:
    """Run distilled BERT-tiny training with small-batch OOM fallback."""
    args = parse_args()
    try:
        run_training(args, batch_size=args.batch_size)
    except RuntimeError as error:
        if args.batch_size > args.fallback_batch_size and is_accelerator_oom(error):
            print(
                f"Out of memory at batch_size={args.batch_size}; "
                f"retrying with batch_size={args.fallback_batch_size}."
            )
            if cuda_is_available():
                torch.cuda.empty_cache()
            run_training(args, batch_size=args.fallback_batch_size)
        else:
            raise


if __name__ == "__main__":
    main()
