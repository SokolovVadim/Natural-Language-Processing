#!/usr/bin/env python
"""Create final deterministic teacher-label splits from already labelled examples."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_DIR = PROJECT_ROOT / "data" / "teacher_labels"
OUTPUT_DIR = PROJECT_ROOT / "data" / "teacher_labels_final"
SEED = 42

SPLIT_TARGETS = {
    "train": {0: 1050, 1: 450},
    "validation": {0: 210, 1: 90},
    "test": {0: 210, 1: 90},
}


def load_split(split: str) -> pd.DataFrame:
    path = INPUT_DIR / f"{split}_teacher.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher-label file: {path}")

    df = pd.read_csv(path)
    required_columns = {"example_id", "original_label", "teacher_label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    return df.drop_duplicates(subset="example_id", keep="first").copy()


def stratified_sample(df: pd.DataFrame, targets: dict[int, int], split: str) -> pd.DataFrame:
    sampled_parts = []
    for label, target_count in targets.items():
        label_df = df[df["original_label"].astype(int) == label]
        if len(label_df) < target_count:
            raise ValueError(
                f"{split}: need {target_count} rows with original_label={label}, "
                f"but only {len(label_df)} are available"
            )
        sampled_parts.append(label_df.sample(n=target_count, random_state=SEED))

    sampled = pd.concat(sampled_parts, ignore_index=False)
    sampled = sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return sampled


def agreement(df: pd.DataFrame) -> float:
    return (df["original_label"].astype(int) == df["teacher_label"].astype(int)).mean()


def print_summary(split: str, df: pd.DataFrame) -> None:
    print(f"\n{split}")
    print(f"rows: {len(df)}")
    print("original:")
    print(df["original_label"].astype(int).value_counts().sort_index())
    print("teacher:")
    print(df["teacher_label"].astype(int).value_counts().sort_index())
    print(f"agreement: {agreement(df):.4f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, targets in SPLIT_TARGETS.items():
        df = load_split(split)
        final_df = stratified_sample(df, targets, split)
        output_path = OUTPUT_DIR / f"{split}_teacher.csv"
        final_df.to_csv(output_path, index=False)
        print_summary(split, final_df)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
