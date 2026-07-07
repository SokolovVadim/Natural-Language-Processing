"""Training utilities for the Transformer student baseline."""

from typing import Any
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from src.evaluation.metrics import compute_classification_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if _cuda_is_available() else "cpu")


def _cuda_is_available() -> bool:
    """Check CUDA availability while avoiding noisy driver warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.cuda.is_available()


def train_one_epoch(
    model: PreTrainedModel,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train the model for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress_bar:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(data_loader), 1)


def evaluate_model(
    model: PreTrainedModel,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a model and return metrics plus raw predictions."""
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    positive_probabilities: list[float] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            labels = batch["labels"]
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            total_loss += outputs.loss.item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            if probabilities.shape[-1] > 1:
                positive_probabilities.extend(probabilities[:, 1].cpu().tolist())
            else:
                positive_probabilities.extend(probabilities[:, 0].cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(len(data_loader), 1)

    return {
        "metrics": metrics,
        "labels": y_true,
        "predictions": y_pred,
        "positive_probabilities": positive_probabilities,
    }


def train_student_model(
    model: PreTrainedModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Train the student baseline and return per-epoch history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        validation_result = evaluate_model(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": validation_result["metrics"],
            }
        )

    return history
