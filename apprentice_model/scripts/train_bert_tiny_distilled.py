#!/usr/bin/env python
"""Skeleton for future BERT-tiny soft-label distillation training."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> Namespace:
    """Parse distillation training arguments."""
    parser = ArgumentParser(
        description=(
            "Prepare BERT-tiny soft-label distillation from BERT-base logits. "
            "Stage 1 does not run full training."
        )
    )
    parser.add_argument(
        "--train_logits_csv",
        default="data/teacher_logits/bert_base_train_logits.csv",
    )
    parser.add_argument(
        "--val_logits_csv",
        default="data/teacher_logits/bert_base_validation_logits.csv",
    )
    parser.add_argument(
        "--test_logits_csv",
        default="data/teacher_logits/bert_base_test_logits.csv",
    )
    parser.add_argument("--student_model_name", default="prajjwal1/bert-tiny")
    parser.add_argument(
        "--output_dir",
        default="results/bert_tiny_distilled_t2_a07_natural",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    """Resolve project-relative paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def validate_input_paths(args: Namespace) -> None:
    """Validate required teacher-logit inputs."""
    for argument_name in ["train_logits_csv", "val_logits_csv", "test_logits_csv"]:
        path = resolve_project_path(getattr(args, argument_name))
        if not path.exists():
            raise FileNotFoundError(f"Missing {argument_name}: {path}")


def main() -> None:
    """Print the future training configuration without starting training."""
    args = parse_args()
    validate_input_paths(args)

    print("BERT-tiny soft-label distillation skeleton is ready.")
    print("Stage 1 intentionally does not start full training.")
    print(f"  train logits: {resolve_project_path(args.train_logits_csv)}")
    print(f"  validation logits: {resolve_project_path(args.val_logits_csv)}")
    print(f"  test logits: {resolve_project_path(args.test_logits_csv)}")
    print(f"  student model: {args.student_model_name}")
    print(f"  output dir: {resolve_project_path(args.output_dir)}")
    print(f"  temperature: {args.temperature}")
    print(f"  alpha: {args.alpha}")
    print(f"  max length: {args.max_length}")
    print(f"  batch size: {args.batch_size}")
    print(f"  learning rate: {args.learning_rate}")
    print(f"  weight decay: {args.weight_decay}")
    print(f"  epochs: {args.epochs}")


if __name__ == "__main__":
    main()
