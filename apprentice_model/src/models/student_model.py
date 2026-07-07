"""Student model helpers for Transformer-based baselines."""

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_student_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer for the student baseline model."""
    return BertTokenizer.from_pretrained(model_name)


def load_student_model(
    model_name: str,
    num_labels: int = 2,
) -> PreTrainedModel:
    """Load a Transformer sequence classification model."""
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return BertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
