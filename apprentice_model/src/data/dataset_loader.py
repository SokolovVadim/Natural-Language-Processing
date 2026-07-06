"""Reusable dataset loading and splitting utilities."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def load_toxic_conversations(dataset_name: str) -> DatasetDict:
    """Load the toxic conversations dataset from Hugging Face.

    Args:
        dataset_name: Hugging Face dataset identifier.

    Returns:
        A DatasetDict containing the available dataset splits.
    """
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    print(f"Loaded dataset: {dataset_name}")
    print(f"Available splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(
            f"- {split_name}: {len(split_data)} rows, "
            f"columns: {split_data.column_names}"
        )

    return dataset


def create_small_splits(
    dataset: DatasetDict,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
    label_column: str | None = None,
    sampling_strategy: str = "random",
    positive_label_ratio: float | None = None,
    positive_label: int = 1,
) -> dict[str, Dataset]:
    """Create deterministic small train, validation, and test splits.

    If the source dataset has a test split, test examples are sampled from it.
    Training and validation examples are sampled from the train split when
    available. If expected splits are missing, all available splits are combined
    before sampling.

    Args:
        dataset: Loaded Hugging Face dataset dictionary.
        train_size: Number of examples in the train subset.
        validation_size: Number of examples in the validation subset.
        test_size: Number of examples in the test subset.
        seed: Random seed used for deterministic shuffling.
        label_column: Name of the label column, required for stratified sampling.
        sampling_strategy: Either "random" or "stratified".
        positive_label_ratio: Target ratio of positive examples in each split.
        positive_label: Label value representing the positive class.

    Returns:
        Dictionary with train, validation, and test Dataset objects.
    """
    if not dataset:
        raise ValueError("Dataset is empty; cannot create splits.")

    if "train" in dataset:
        train_source = dataset["train"].shuffle(seed=seed)
    else:
        train_source = _combine_splits(dataset).shuffle(seed=seed)

    sampling_strategy = sampling_strategy.lower()
    if sampling_strategy not in {"random", "stratified"}:
        raise ValueError(
            "sampling_strategy must be either 'random' or 'stratified', "
            f"got '{sampling_strategy}'."
        )

    if sampling_strategy == "stratified":
        _validate_stratified_sampling_config(label_column, positive_label_ratio)

    required_train_rows = train_size + validation_size
    if len(train_source) < required_train_rows:
        raise ValueError(
            "Not enough training rows to create requested train and validation "
            f"splits: requested {required_train_rows}, available {len(train_source)}."
        )

    if sampling_strategy == "stratified":
        train_indices = _stratified_indices(
            dataset=train_source,
            size=train_size,
            label_column=label_column,
            positive_label_ratio=positive_label_ratio,
            positive_label=positive_label,
            start_offsets={"positive": 0, "negative": 0},
        )
        validation_indices = _stratified_indices(
            dataset=train_source,
            size=validation_size,
            label_column=label_column,
            positive_label_ratio=positive_label_ratio,
            positive_label=positive_label,
            start_offsets=_class_counts(train_size, positive_label_ratio),
        )
        train_split = train_source.select(train_indices)
        validation_split = train_source.select(validation_indices)
    else:
        train_split = train_source.select(range(train_size))
        validation_split = train_source.select(range(train_size, required_train_rows))

    if "test" in dataset:
        test_source = dataset["test"].shuffle(seed=seed)
    else:
        used_train_indices = set(range(required_train_rows))
        remaining_indices = [
            index for index in range(len(train_source)) if index not in used_train_indices
        ]
        test_source = train_source.select(remaining_indices)

    if len(test_source) < test_size:
        raise ValueError(
            "Not enough test rows to create requested test split: "
            f"requested {test_size}, available {len(test_source)}."
        )

    if sampling_strategy == "stratified":
        test_indices = _stratified_indices(
            dataset=test_source,
            size=test_size,
            label_column=label_column,
            positive_label_ratio=positive_label_ratio,
            positive_label=positive_label,
            start_offsets={"positive": 0, "negative": 0},
        )
        test_split = test_source.select(test_indices)
    else:
        test_split = test_source.select(range(test_size))

    return {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }


def save_splits_to_csv(
    splits: dict[str, Dataset] | dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Save dataset or dataframe splits as CSV files.

    Args:
        splits: Mapping from split name to Hugging Face Dataset or DataFrame.
        output_dir: Directory where CSV files should be saved.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        if isinstance(split_data, pd.DataFrame):
            dataframe = split_data
        else:
            dataframe = split_data.to_pandas()

        csv_path = output_path / f"{split_name}.csv"
        dataframe.to_csv(csv_path, index=False)
        print(f"Saved {split_name} split to {csv_path} ({len(dataframe)} rows)")


def datasets_to_dataframes(splits: dict[str, Dataset]) -> dict[str, pd.DataFrame]:
    """Convert Hugging Face dataset splits to pandas DataFrames."""
    return {
        split_name: split_data.to_pandas()
        for split_name, split_data in splits.items()
    }


def _combine_splits(dataset: DatasetDict) -> Dataset:
    """Combine all available splits into a single Dataset."""
    split_datasets: list[Dataset] = []

    for split_data in dataset.values():
        if not isinstance(split_data, Dataset):
            raise TypeError(f"Expected Dataset split, got {type(split_data)!r}")
        split_datasets.append(split_data)

    return concatenate_datasets(split_datasets)


def _validate_stratified_sampling_config(
    label_column: str | None,
    positive_label_ratio: float | None,
) -> None:
    """Validate required stratified sampling options."""
    if not label_column:
        raise ValueError("label_column is required for stratified sampling.")

    if positive_label_ratio is None:
        raise ValueError("positive_label_ratio is required for stratified sampling.")

    if not 0 < positive_label_ratio < 1:
        raise ValueError(
            "positive_label_ratio must be between 0 and 1, "
            f"got {positive_label_ratio}."
        )


def _class_counts(size: int, positive_label_ratio: float) -> dict[str, int]:
    """Compute target positive and negative counts for a split size."""
    positive_count = round(size * positive_label_ratio)
    positive_count = max(1, min(size - 1, positive_count)) if size > 1 else size
    negative_count = size - positive_count

    return {
        "positive": positive_count,
        "negative": negative_count,
    }


def _stratified_indices(
    dataset: Dataset,
    size: int,
    label_column: str,
    positive_label_ratio: float,
    positive_label: int,
    start_offsets: dict[str, int],
) -> list[int]:
    """Select deterministic indices with a controlled positive class ratio.

    The dataset is expected to already be shuffled deterministically. This helper
    then takes the next requested number of positive and negative examples.
    """
    if label_column not in dataset.column_names:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")

    class_counts = _class_counts(size, positive_label_ratio)
    positive_count = class_counts["positive"]
    negative_count = class_counts["negative"]

    positive_start = start_offsets.get("positive", 0)
    negative_start = start_offsets.get("negative", 0)

    positive_indices: list[int] = []
    negative_indices: list[int] = []

    labels = dataset[label_column]
    for index, label in enumerate(labels):
        if label == positive_label:
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    positive_end = positive_start + positive_count
    negative_end = negative_start + negative_count

    if len(positive_indices) < positive_end:
        raise ValueError(
            "Not enough positive examples for stratified sampling: "
            f"requested through index {positive_end}, available {len(positive_indices)}."
        )

    if len(negative_indices) < negative_end:
        raise ValueError(
            "Not enough negative examples for stratified sampling: "
            f"requested through index {negative_end}, available {len(negative_indices)}."
        )

    selected_indices = (
        positive_indices[positive_start:positive_end]
        + negative_indices[negative_start:negative_end]
    )

    return selected_indices
