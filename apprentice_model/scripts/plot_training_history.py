#!/usr/bin/env python
"""Plot BERT-base supervised teacher training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "results" / "bert_base_supervised" / "training_history.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main() -> None:
    """Read training history and save learning-curve figures."""
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing training history: {HISTORY_PATH}")

    history = pd.read_csv(HISTORY_PATH)
    required_columns = {"epoch", "train_loss", "eval_loss", "eval_f1"}
    missing_columns = sorted(required_columns - set(history.columns))
    if missing_columns:
        raise KeyError(f"Training history is missing columns: {missing_columns}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="train_loss")
    plt.plot(history["epoch"], history["eval_loss"], marker="o", label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BERT-base Supervised Teacher Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = FIGURES_DIR / "bert_base_learning_curve_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["eval_f1"], marker="o", label="eval_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("BERT-base Supervised Teacher Validation F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    f1_path = FIGURES_DIR / "bert_base_learning_curve_f1.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()

    print(f"Saved loss curve to {loss_path}")
    print(f"Saved F1 curve to {f1_path}")


if __name__ == "__main__":
    main()
