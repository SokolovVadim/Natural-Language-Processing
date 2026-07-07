#!/usr/bin/env python
"""Generate OpenAI teacher labels for processed dataset splits."""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import os
import sys
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.teacher.openai_teacher import OpenAIToxicityTeacher
from src.teacher.prompting import build_toxicity_teacher_prompt


SPLITS = ("train", "validation", "test")
SAVE_EVERY_N_EXAMPLES = 10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate OpenAI teacher labels for processed CSV splits."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a few prompts/examples without calling the OpenAI API.",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        help="Process only one split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum examples to consider for the selected split(s).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore configured split limits and process all rows.",
    )
    return parser.parse_args()


def maybe_load_dotenv() -> None:
    """Load local .env values when python-dotenv is installed."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


def require_openai_api_key() -> None:
    """Ensure the OpenAI API key is available without printing it."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it before running teacher labeling: "
            'export OPENAI_API_KEY="your_key_here"'
        )


def load_processed_split(
    split_name: str,
    processed_dir: Path,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Load one processed split with required columns."""
    split_path = processed_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing processed split: {split_path}. "
            "Run `python scripts/prepare_data.py` first."
        )

    dataframe = pd.read_csv(split_path)
    required_columns = {text_column, label_column, "label_text"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(f"{split_path} is missing columns: {sorted(missing_columns)}")

    dataframe[text_column] = dataframe[text_column].fillna("").astype(str)
    return dataframe


def create_example_id(split_name: str, row_index: int, text: str) -> str:
    """Create a stable example identifier from split, row index, and text hash."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{split_name}:{row_index}:{text_hash}"


def get_split_limit(
    split_name: str,
    dataframe: pd.DataFrame,
    teacher_config: dict[str, Any],
    cli_limit: int | None,
    use_all: bool,
) -> int:
    """Resolve how many examples to consider for one split."""
    if use_all:
        return len(dataframe)
    if cli_limit is not None:
        return min(cli_limit, len(dataframe))

    config_key = f"max_examples_{split_name}"
    return min(int(teacher_config[config_key]), len(dataframe))


def select_teacher_examples(
    dataframe: pd.DataFrame,
    limit: int,
    label_column: str,
    teacher_config: dict[str, Any],
) -> pd.DataFrame:
    """Select teacher-labeling examples with deterministic sampling."""
    sampling_strategy = teacher_config.get("sampling_strategy", "stratified").lower()
    seed = int(teacher_config.get("seed", 42))

    if sampling_strategy == "stratified":
        selected = select_stratified_examples(
            dataframe=dataframe,
            limit=limit,
            label_column=label_column,
            positive_label_ratio=float(
                teacher_config.get("positive_label_ratio", 0.3)
            ),
            seed=seed,
        )
    elif sampling_strategy == "random":
        selected = dataframe.sample(
            n=limit,
            random_state=seed,
            replace=False,
        )
    else:
        raise ValueError(
            "teacher.sampling_strategy must be 'stratified' or 'random', "
            f"got {sampling_strategy!r}."
        )

    return selected


def select_stratified_examples(
    dataframe: pd.DataFrame,
    limit: int,
    label_column: str,
    positive_label_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic stratified subset by the original label column."""
    if label_column not in dataframe.columns:
        raise KeyError(f"Label column '{label_column}' not found.")
    if not 0 < positive_label_ratio < 1:
        raise ValueError(
            "teacher.positive_label_ratio must be between 0 and 1, "
            f"got {positive_label_ratio}."
        )

    positive_count = round(limit * positive_label_ratio)
    positive_count = max(1, min(limit - 1, positive_count)) if limit > 1 else limit
    negative_count = limit - positive_count

    positive_examples = dataframe[dataframe[label_column].astype(int) == 1]
    negative_examples = dataframe[dataframe[label_column].astype(int) == 0]

    if len(positive_examples) < positive_count:
        raise ValueError(
            "Not enough positive examples for teacher stratified sampling: "
            f"requested {positive_count}, available {len(positive_examples)}."
        )
    if len(negative_examples) < negative_count:
        raise ValueError(
            "Not enough negative examples for teacher stratified sampling: "
            f"requested {negative_count}, available {len(negative_examples)}."
        )

    sampled_positive = positive_examples.sample(
        n=positive_count,
        random_state=seed,
        replace=False,
    )
    sampled_negative = negative_examples.sample(
        n=negative_count,
        random_state=seed + 1,
        replace=False,
    )

    selected = pd.concat([sampled_positive, sampled_negative])
    return selected.sample(
        n=len(selected),
        random_state=seed + 2,
        replace=False,
    )


def print_selected_label_distribution(
    selected_dataframe: pd.DataFrame,
    label_column: str,
    requested_positive_label_ratio: float,
) -> None:
    """Print selected original label distribution and warn on large drift."""
    counts = selected_dataframe[label_column].astype(int).value_counts().sort_index()

    print("Selected original label distribution:")
    for label in (0, 1):
        print(f"  label={label}: {int(counts.get(label, 0))}")

    if selected_dataframe.empty:
        return

    actual_positive_ratio = counts.get(1, 0) / len(selected_dataframe)
    tolerance = max(0.05, 1 / len(selected_dataframe))
    if abs(actual_positive_ratio - requested_positive_label_ratio) > tolerance:
        print(
            "Warning: selected positive label ratio "
            f"({actual_positive_ratio:.3f}) differs from requested ratio "
            f"({requested_positive_label_ratio:.3f})."
        )


def load_existing_labels(output_path: Path) -> pd.DataFrame:
    """Load existing teacher labels if a previous run exists."""
    if output_path.exists():
        dataframe = pd.read_csv(output_path)
        if "example_id" not in dataframe.columns:
            raise KeyError(
                f"Existing label file is missing required example_id column: "
                f"{output_path}"
            )
        return dataframe.drop_duplicates(subset="example_id", keep="last")

    return pd.DataFrame(
        columns=[
            "example_id",
            "text",
            "original_label",
            "original_label_text",
            "teacher_label",
            "teacher_label_text",
            "teacher_prob_not_toxic",
            "teacher_prob_toxic",
            "teacher_confidence",
            "teacher_reason",
        ]
    )


def save_labels_incrementally(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Safely save labels through a temporary file then replace final CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)


def append_new_label_rows(
    existing_labels: pd.DataFrame,
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Append newly labeled rows without concat warnings on empty frames."""
    if not new_rows:
        return existing_labels

    new_labels = pd.DataFrame(new_rows)
    if existing_labels.empty:
        return new_labels

    return pd.concat([existing_labels, new_labels], ignore_index=True)


def classify_with_retries(
    teacher: OpenAIToxicityTeacher,
    text: str,
    max_retries: int,
    sleep_between_requests_sec: float,
) -> dict[str, Any]:
    """Call the teacher with retry handling."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return teacher.classify(text)
        except Exception as exc:
            last_error = exc
            if is_non_retryable_openai_error(exc):
                raise RuntimeError(
                    "OpenAI teacher labeling stopped because the API returned a "
                    f"non-retryable error: {exc}\n"
                    "If this is `insufficient_quota`, check your OpenAI billing, "
                    "quota, or project limits before retrying."
                ) from exc

            if attempt == max_retries:
                break

            sleep_time = sleep_between_requests_sec * attempt
            print(
                f"OpenAI request failed on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {sleep_time:.1f}s."
            )
            time.sleep(sleep_time)

    raise RuntimeError(
        f"OpenAI teacher labeling failed after {max_retries} attempts: {last_error}"
    )


def is_non_retryable_openai_error(exc: Exception) -> bool:
    """Return whether an OpenAI error should fail immediately."""
    message = str(exc).lower()
    non_retryable_markers = (
        "insufficient_quota",
        "exceeded your current quota",
        "invalid_api_key",
        "incorrect api key",
        "billing",
    )
    return any(marker in message for marker in non_retryable_markers)


def build_output_row(
    example_id: str,
    source_row: pd.Series,
    teacher_output: dict[str, Any],
    text_column: str,
    label_column: str,
) -> dict[str, Any]:
    """Build one teacher-label output row."""
    return {
        "example_id": example_id,
        "text": source_row[text_column],
        "original_label": int(source_row[label_column]),
        "original_label_text": source_row["label_text"],
        **teacher_output,
    }


def print_split_summary(split_name: str, labeled_rows: pd.DataFrame) -> None:
    """Print label distribution, agreement, and disagreement examples."""
    print(f"\nSummary for {split_name}:")
    print(f"  labeled examples: {len(labeled_rows)}")

    if labeled_rows.empty:
        return

    distribution = labeled_rows["teacher_label"].value_counts().sort_index()
    print("  teacher label distribution:")
    for label, count in distribution.items():
        print(f"    label={label}: {count}")

    agreement = (
        labeled_rows["original_label"].astype(int)
        == labeled_rows["teacher_label"].astype(int)
    )
    print(f"  agreement with original labels: {agreement.mean() * 100:.2f}%")

    disagreements = labeled_rows.loc[~agreement]
    print(f"  disagreements: {len(disagreements)}")
    for _, row in disagreements.head(5).iterrows():
        text_preview = str(row["text"]).replace("\n", " ")[:120]
        print(
            f"    {row['example_id']}: original={row['original_label']} "
            f"teacher={row['teacher_label']} text={text_preview!r}"
        )


def print_dry_run_examples(
    split_name: str,
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> None:
    """Print prompts/examples without calling the API."""
    print(f"\nDry run for {split_name}: showing {min(len(dataframe), 3)} example(s)")
    for row_index, row in dataframe.head(3).iterrows():
        example_id = create_example_id(split_name, row_index, row[text_column])
        print(f"\nexample_id: {example_id}")
        print(f"original_label: {int(row[label_column])}")
        print(f"original_label_text: {row['label_text']}")
        print("prompt:")
        print(build_toxicity_teacher_prompt(row[text_column]))


def label_split(
    split_name: str,
    config: dict[str, Any],
    teacher: OpenAIToxicityTeacher,
    cli_limit: int | None,
    use_all: bool,
) -> None:
    """Generate or resume teacher labels for one split."""
    dataset_config = config["dataset"]
    teacher_config = config["teacher"]
    text_column = dataset_config["text_column"]
    label_column = dataset_config["label_column"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
    output_dir = PROJECT_ROOT / teacher_config["output_dir"]
    output_path = output_dir / f"{split_name}_teacher.csv"

    dataframe = load_processed_split(
        split_name=split_name,
        processed_dir=processed_dir,
        text_column=text_column,
        label_column=label_column,
    )
    limit = get_split_limit(split_name, dataframe, teacher_config, cli_limit, use_all)
    target_dataframe = select_teacher_examples(
        dataframe=dataframe,
        limit=limit,
        label_column=label_column,
        teacher_config=teacher_config,
    )
    target_ids = {
        create_example_id(split_name, row_index, row[text_column])
        for row_index, row in target_dataframe.iterrows()
    }
    existing_labels = load_existing_labels(output_path)
    existing_ids = set(existing_labels["example_id"].dropna().astype(str))
    new_rows: list[dict[str, Any]] = []

    print(f"\nLabeling {split_name}: target={len(target_dataframe)}")
    print_selected_label_distribution(
        selected_dataframe=target_dataframe,
        label_column=label_column,
        requested_positive_label_ratio=float(
            teacher_config.get("positive_label_ratio", 0.3)
        ),
    )
    progress = tqdm(target_dataframe.iterrows(), total=len(target_dataframe))
    for row_index, row in progress:
        example_id = create_example_id(split_name, row_index, row[text_column])
        if example_id in existing_ids:
            continue

        try:
            teacher_output = classify_with_retries(
                teacher=teacher,
                text=row[text_column],
                max_retries=int(teacher_config["max_retries"]),
                sleep_between_requests_sec=float(
                    teacher_config["sleep_between_requests_sec"]
                ),
            )
        except Exception:
            if new_rows:
                existing_labels = append_new_label_rows(existing_labels, new_rows)
                save_labels_incrementally(existing_labels, output_path)
                print(f"\nSaved {len(new_rows)} new row(s) before stopping.")
            raise

        new_rows.append(
            build_output_row(
                example_id=example_id,
                source_row=row,
                teacher_output=teacher_output,
                text_column=text_column,
                label_column=label_column,
            )
        )
        existing_ids.add(example_id)

        if len(new_rows) % SAVE_EVERY_N_EXAMPLES == 0:
            existing_labels = append_new_label_rows(existing_labels, new_rows)
            save_labels_incrementally(existing_labels, output_path)
            new_rows = []

        time.sleep(float(teacher_config["sleep_between_requests_sec"]))

    if new_rows:
        existing_labels = append_new_label_rows(existing_labels, new_rows)
        save_labels_incrementally(existing_labels, output_path)

    labeled_target = existing_labels[
        existing_labels["example_id"].astype(str).isin(target_ids)
    ]
    print_split_summary(split_name, labeled_target)


def main() -> None:
    """Generate teacher labels according to config and CLI options."""
    args = parse_args()
    maybe_load_dotenv()

    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    teacher_config = config["teacher"]
    dataset_config = config["dataset"]
    selected_splits = [args.split] if args.split else list(SPLITS)

    if args.dry_run:
        processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
        for split_name in selected_splits:
            dataframe = load_processed_split(
                split_name=split_name,
                processed_dir=processed_dir,
                text_column=dataset_config["text_column"],
                label_column=dataset_config["label_column"],
            )
            limit = get_split_limit(
                split_name,
                dataframe,
                teacher_config,
                args.limit,
                args.all,
            )
            selected_dataframe = select_teacher_examples(
                dataframe=dataframe,
                limit=limit,
                label_column=dataset_config["label_column"],
                teacher_config=teacher_config,
            )
            print_selected_label_distribution(
                selected_dataframe=selected_dataframe,
                label_column=dataset_config["label_column"],
                requested_positive_label_ratio=float(
                    teacher_config.get("positive_label_ratio", 0.3)
                ),
            )
            print_dry_run_examples(
                split_name=split_name,
                dataframe=selected_dataframe,
                text_column=dataset_config["text_column"],
                label_column=dataset_config["label_column"],
            )
        print("\nDry run complete. No API calls were made and no files were written.")
        return

    require_openai_api_key()
    teacher = OpenAIToxicityTeacher(
        model=teacher_config["model"],
        temperature=float(teacher_config["temperature"]),
        timeout_sec=int(teacher_config["request_timeout_sec"]),
    )

    for split_name in selected_splits:
        label_split(
            split_name=split_name,
            config=config,
            teacher=teacher,
            cli_limit=args.limit,
            use_all=args.all,
        )


if __name__ == "__main__":
    main()
