#!/usr/bin/env python
"""Prepare natural-distribution stratified dataset splits."""

from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset_loader import load_toxic_conversations
from src.data.preprocessing import preprocess_dataframe


def collect_label_indices(
    dataset: Dataset,
    label_column: str,
) -> dict[int, list[int]]:
    """Collect row indices for binary labels."""
    if label_column not in dataset.column_names:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")

    indices_by_label = {0: [], 1: []}
    for index, label in enumerate(dataset[label_column]):
        label = int(label)
        if label not in indices_by_label:
            raise ValueError(f"Expected binary labels 0/1, got {label}.")
        indices_by_label[label].append(index)

    return indices_by_label


def natural_label_counts(size: int, source_positive_ratio: float) -> dict[int, int]:
    """Compute class counts that preserve the source distribution."""
    positive_count = round(size * source_positive_ratio)
    positive_count = min(max(positive_count, 0), size)
    return {
        0: size - positive_count,
        1: positive_count,
    }


def stratified_natural_indices(
    dataset: Dataset,
    size: int,
    label_column: str,
    seed: int,
    start_offsets: dict[int, int] | None = None,
) -> tuple[list[int], dict[int, int], float]:
    """Select deterministic stratified indices matching source distribution."""
    start_offsets = start_offsets or {0: 0, 1: 0}
    indices_by_label = collect_label_indices(dataset, label_column)
    source_rows = len(dataset)
    source_positive_ratio = len(indices_by_label[1]) / source_rows
    target_counts = natural_label_counts(size, source_positive_ratio)

    selected_indices: list[int] = []
    for label in (0, 1):
        shuffled_indices = indices_by_label[label].copy()
        random.Random(seed + label).shuffle(shuffled_indices)

        start = start_offsets.get(label, 0)
        end = start + target_counts[label]
        if len(shuffled_indices) < end:
            raise ValueError(
                f"Not enough label={label} examples: requested through {end}, "
                f"available {len(shuffled_indices)}."
            )
        selected_indices.extend(shuffled_indices[start:end])

    random.Random(seed + 1000).shuffle(selected_indices)
    return selected_indices, target_counts, source_positive_ratio


def combine_available_splits(dataset: DatasetDict) -> Dataset:
    """Combine all available dataset splits as a fallback source."""
    return concatenate_datasets(list(dataset.values()))


def create_natural_splits(
    dataset: DatasetDict,
    train_size: int,
    validation_size: int,
    test_size: int,
    label_column: str,
    seed: int,
) -> dict[str, Dataset]:
    """Create train/validation/test splits preserving natural class ratios."""
    train_source = dataset["train"] if "train" in dataset else combine_available_splits(dataset)
    test_source = dataset["test"] if "test" in dataset else train_source

    train_indices, train_counts, _ = stratified_natural_indices(
        dataset=train_source,
        size=train_size,
        label_column=label_column,
        seed=seed,
        start_offsets={0: 0, 1: 0},
    )
    validation_indices, validation_counts, _ = stratified_natural_indices(
        dataset=train_source,
        size=validation_size,
        label_column=label_column,
        seed=seed,
        start_offsets=train_counts,
    )
    test_indices, _, _ = stratified_natural_indices(
        dataset=test_source,
        size=test_size,
        label_column=label_column,
        seed=seed + 10,
        start_offsets={0: 0, 1: 0},
    )

    print("Source selection:")
    print("  train: Hugging Face train split")
    print("  validation: Hugging Face train split")
    print(
        "  test: Hugging Face test split"
        if "test" in dataset
        else "  test: fallback source because no Hugging Face test split was found"
    )

    return {
        "train": train_source.select(train_indices),
        "validation": train_source.select(validation_indices),
        "test": test_source.select(test_indices),
    }


def split_summary(dataframe: pd.DataFrame, label_column: str) -> dict[str, float | int]:
    """Build a count and percentage summary for one split."""
    counts = dataframe[label_column].astype(int).value_counts().sort_index()
    rows = len(dataframe)
    label_0_count = int(counts.get(0, 0))
    label_1_count = int(counts.get(1, 0))
    toxic_percentage = (label_1_count / rows * 100) if rows else 0.0

    return {
        "rows": rows,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "toxic_percentage": toxic_percentage,
    }


def print_split_summary(split_name: str, summary: dict[str, float | int]) -> None:
    """Print one split distribution summary."""
    print(f"\n{split_name} distribution:")
    print(f"  rows: {summary['rows']}")
    print(f"  label=0: {summary['label_0_count']}")
    print(f"  label=1: {summary['label_1_count']}")
    print(f"  toxic percentage: {summary['toxic_percentage']:.2f}%")


def main() -> None:
    """Create and save natural-distribution stratified splits."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    dataset_config = config["dataset"]
    natural_config = config["natural_splits"]

    dataset_name = dataset_config["name"]
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]
    seed = int(natural_config.get("seed", config.get("seed", 42)))

    dataset = load_toxic_conversations(dataset_name)
    splits = create_natural_splits(
        dataset=dataset,
        train_size=int(natural_config["train_size"]),
        validation_size=int(natural_config["validation_size"]),
        test_size=int(natural_config["test_size"]),
        label_column=label_column,
        seed=seed,
    )

    output_dir = PROJECT_ROOT / natural_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int]] = {}
    for split_name, split_dataset in splits.items():
        dataframe = split_dataset.to_pandas()
        dataframe = preprocess_dataframe(dataframe, text_column)
        csv_path = output_dir / f"{split_name}.csv"
        dataframe.to_csv(csv_path, index=False)

        summary[split_name] = split_summary(dataframe, label_column)
        print_split_summary(split_name, summary[split_name])
        print(f"  saved to: {csv_path}")

    summary_path = PROJECT_ROOT / natural_config["summary_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nSaved natural split summary to {summary_path}")


if __name__ == "__main__":
    main()
