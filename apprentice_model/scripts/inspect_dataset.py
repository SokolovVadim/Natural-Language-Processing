#!/usr/bin/env python
"""Check the toxic conversations dataset structure and sample rows"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset_loader import load_toxic_conversations


def print_label_distribution(split_name: str, split_data, label_column: str) -> None:
    """Print label distribution for a dataset split when possible."""
    if label_column not in split_data.column_names:
        print(f"{split_name}: label column '{label_column}' not found.")
        return

    dataframe = split_data.to_pandas()
    counts = dataframe[label_column].value_counts(dropna=False).sort_index()
    print(f"\n{split_name} label distribution:")
    for label, count in counts.items():
        print(f"  label={label}: {count}")


def print_examples(split_name: str, split_data, num_examples: int = 3) -> None:
    """Print a few examples from a dataset split."""
    print(f"\nExamples from {split_name}:")
    for index, example in enumerate(split_data.select(range(min(num_examples, len(split_data))))):
        print(f"\nExample {index + 1}:")
        for column_name, value in example.items():
            print(f"  {column_name}: {value}")


def main() -> None:
    """Load and inspect the configured Hugging Face dataset."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    dataset_name = config["dataset"]["name"]
    label_column = config["dataset"]["label_column"]

    dataset = load_toxic_conversations(dataset_name)

    print("\nDataset structure:")
    print(dataset)

    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Rows: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")
        print_examples(split_name, split_data)
        print_label_distribution(split_name, split_data, label_column)


if __name__ == "__main__":
    main()
