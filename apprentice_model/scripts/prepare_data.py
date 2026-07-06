#!/usr/bin/env python
"""Prepare small processed CSV splits for the toxic conversations dataset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset_loader import (
    create_small_splits,
    datasets_to_dataframes,
    load_toxic_conversations,
    save_splits_to_csv,
)
from src.data.preprocessing import preprocess_dataframe


def print_label_distribution(
    split_name: str,
    dataframe,
    label_column: str,
) -> None:
    """Print label counts and proportions for one split."""
    if label_column not in dataframe.columns:
        print(f"{split_name}: label column '{label_column}' not found.")
        return

    counts = dataframe[label_column].value_counts(dropna=False).sort_index()
    proportions = dataframe[label_column].value_counts(
        normalize=True,
        dropna=False,
    ).sort_index()

    print(f"\n{split_name} label distribution:")
    for label, count in counts.items():
        percentage = proportions[label] * 100
        print(f"  label={label}: {count} ({percentage:.2f}%)")


def main() -> None:
    """Load, preprocess, and save deterministic small dataset splits."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    dataset_config = config["dataset"]
    subset_config = config["subsets"]
    path_config = config["paths"]
    seed = int(config["seed"])

    dataset = load_toxic_conversations(dataset_config["name"])
    splits = create_small_splits(
        dataset=dataset,
        train_size=int(subset_config["train_size"]),
        validation_size=int(subset_config["validation_size"]),
        test_size=int(subset_config["test_size"]),
        seed=seed,
        label_column=dataset_config["label_column"],
        sampling_strategy=subset_config.get("sampling_strategy", "random"),
        positive_label_ratio=float(subset_config["positive_label_ratio"]),
    )

    dataframes = datasets_to_dataframes(splits)
    processed_dataframes = {
        split_name: preprocess_dataframe(
            dataframe,
            text_column=dataset_config["text_column"],
        )
        for split_name, dataframe in dataframes.items()
    }

    output_dir = PROJECT_ROOT / path_config["processed_data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPrepared split sizes:")
    for split_name, dataframe in processed_dataframes.items():
        print(f"  {split_name}: {len(dataframe)} rows")
        print_label_distribution(
            split_name=split_name,
            dataframe=dataframe,
            label_column=dataset_config["label_column"],
        )

    print()
    save_splits_to_csv(processed_dataframes, str(output_dir))


if __name__ == "__main__":
    main()
