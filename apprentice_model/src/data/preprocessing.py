"""Basic text preprocessing utilities."""

import re

import pandas as pd


def clean_text(text: str) -> str:
    """Clean a single text value without destroying tokenizer-friendly signal.

    The current preprocessing is intentionally conservative: it handles missing
    values, strips leading/trailing whitespace, and normalizes repeated spaces.

    Args:
        text: Raw text value.

    Returns:
        Cleaned text string.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Apply basic text preprocessing to a dataframe.

    Args:
        df: Input dataframe.
        text_column: Name of the column containing text.

    Returns:
        Copy of the dataframe with cleaned text.

    Raises:
        KeyError: If the text column is missing.
    """
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' not found in dataframe.")

    processed_df = df.copy()
    processed_df[text_column] = processed_df[text_column].apply(clean_text)
    return processed_df
