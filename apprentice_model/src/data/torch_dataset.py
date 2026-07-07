"""PyTorch dataset wrappers for text classification."""

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextClassificationDataset(Dataset):
    """Tokenized text classification dataset for Transformer models."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        label_column: str,
        max_length: int = 128,
    ) -> None:
        """Tokenize text and store labels as tensors.

        Args:
            dataframe: Source dataframe containing text and labels.
            tokenizer: Hugging Face tokenizer.
            text_column: Name of the text column.
            label_column: Name of the label column.
            max_length: Maximum token sequence length.
        """
        if text_column not in dataframe.columns:
            raise KeyError(f"Text column '{text_column}' not found.")
        if label_column not in dataframe.columns:
            raise KeyError(f"Label column '{label_column}' not found.")

        texts = dataframe[text_column].fillna("").astype(str).tolist()
        labels = dataframe[label_column].astype(int).tolist()

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one tokenized example."""
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item
