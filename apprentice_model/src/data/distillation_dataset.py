"""Dataset helpers for BERT teacher-logit distillation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


REQUIRED_TEACHER_LOGIT_COLUMNS = [
    "text",
    "original_label",
    "teacher_logit_0",
    "teacher_logit_1",
    "teacher_prob_0",
    "teacher_prob_1",
    "teacher_pred_default_threshold",
]


class DistillationTextDataset(Dataset):
    """Tokenized text dataset with hard labels and teacher logits."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "text",
        label_column: str = "original_label",
        teacher_logit_columns: tuple[str, str] = ("teacher_logit_0", "teacher_logit_1"),
        max_length: int = 256,
    ) -> None:
        """Tokenize texts and store original labels plus teacher logits."""
        required_columns = [
            text_column,
            label_column,
            teacher_logit_columns[0],
            teacher_logit_columns[1],
        ]
        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise KeyError(f"Missing distillation columns: {missing_columns}")

        texts = dataframe[text_column].fillna("").astype(str).tolist()
        labels = dataframe[label_column].astype(int).tolist()
        teacher_logits = dataframe[
            [teacher_logit_columns[0], teacher_logit_columns[1]]
        ].astype(float)

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_logits = torch.tensor(teacher_logits.values, dtype=torch.float)

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one tokenized distillation example."""
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        item["teacher_logits"] = self.teacher_logits[index]
        return item


def create_distillation_dataloader(
    dataframe: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int = 256,
    shuffle: bool = False,
    seed: int = 12345,
) -> DataLoader:
    """Create a deterministic DataLoader for distillation batches."""
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
