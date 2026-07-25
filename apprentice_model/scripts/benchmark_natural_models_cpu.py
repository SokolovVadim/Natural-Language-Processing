#!/usr/bin/env python
"""CPU inference benchmark for natural-split model artifacts."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
import json
import statistics
import sys
from typing import Any, Callable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tfidf_baseline import TfidfLogisticRegressionBaseline  # noqa: E402


BYTES_PER_MB = 1024 * 1024
DEPLOYABLE_TRANSFORMER_FILES = {
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
}


ACCEPTED_DEFAULT_F1 = {
    "tfidf_logistic_regression": 0.4061,
    "bert_tiny_supervised": 0.5036,
    "bert_tiny_distilled_t2_a07": 0.5649,
    "bert_base_teacher": 0.6199,
}


def parse_args() -> Namespace:
    """Parse CPU benchmark arguments."""
    parser = ArgumentParser(
        description="Benchmark natural-split TF-IDF, BERT-tiny, and BERT-base models on CPU."
    )
    parser.add_argument(
        "--test_csv",
        default="data/processed_natural/test.csv",
        help="Project-relative or absolute test CSV path.",
    )
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 16],
        help="Transformer batch sizes to benchmark.",
    )
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--warmup_examples", type=int, default=16)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit. Omit for the full natural test split.",
    )
    parser.add_argument(
        "--output_json",
        default="results/cpu_benchmark_comparison.json",
    )
    parser.add_argument(
        "--output_csv",
        default="results/cpu_benchmark_comparison.csv",
    )
    parser.add_argument(
        "--output_md",
        default="results/cpu_benchmark_comparison.md",
    )
    return parser.parse_args()


def resolve_project_path(path: str) -> Path:
    """Resolve an absolute or project-relative path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def require_path(path: Path, description: str) -> None:
    """Raise a clear error if a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_texts(test_csv: Path, text_column: str, limit: int | None = None) -> list[str]:
    """Load benchmark texts from the natural test split."""
    require_path(test_csv, "natural test split")
    dataframe = pd.read_csv(test_csv)
    if text_column not in dataframe.columns:
        raise KeyError(f"Column '{text_column}' not found in {test_csv}")
    texts = dataframe[text_column].fillna("").astype(str).tolist()
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        texts = texts[:limit]
    if not texts:
        raise ValueError("Cannot benchmark an empty test set")
    return texts


def file_size_mb(path: Path) -> float:
    """Return file size in MB."""
    return path.stat().st_size / BYTES_PER_MB


def deployable_model_size_mb(path: Path) -> float:
    """Return size of files needed to load the model for inference.

    Checkpoints and training-history files are intentionally excluded so that
    saved training artifacts do not distort deployable model-size comparisons.
    """
    require_path(path, "model artifact")
    if path.is_file():
        return file_size_mb(path)

    total_mb = 0.0
    for child in path.iterdir():
        if child.is_file() and child.name in DEPLOYABLE_TRANSFORMER_FILES:
            total_mb += file_size_mb(child)
    if total_mb == 0.0:
        raise FileNotFoundError(
            f"No deployable Transformer files found at top level of {path}"
        )
    return total_mb


def artifact_size_mb(path: Path) -> float:
    """Return recursive artifact directory/file size in MB."""
    require_path(path, "model artifact")
    if path.is_file():
        return file_size_mb(path)
    total_bytes = sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )
    return total_bytes / BYTES_PER_MB


def measure_predict_function(
    predict_fn: Callable[[list[str]], Any],
    texts: list[str],
    num_repeats: int,
    warmup_examples: int,
) -> dict[str, Any]:
    """Warm up and time one full prediction function."""
    if num_repeats <= 0:
        raise ValueError("--num_repeats must be positive")

    warmup_texts = texts[: min(warmup_examples, len(texts))]
    if warmup_texts:
        predict_fn(warmup_texts)

    run_times: list[float] = []
    for _ in range(num_repeats):
        start_time = perf_counter()
        predict_fn(texts)
        run_times.append(perf_counter() - start_time)

    average_time = statistics.mean(run_times)
    return {
        "avg_total_inference_time_sec": average_time,
        "std_total_inference_time_sec": (
            statistics.stdev(run_times) if len(run_times) > 1 else 0.0
        ),
        "avg_inference_time_per_example_ms": average_time / len(texts) * 1000,
        "examples_per_second": len(texts) / average_time,
        "run_times_sec": run_times,
    }


def transformer_predict(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[int]:
    """Run CPU Transformer inference including tokenization."""
    predictions: list[int] = []
    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index : start_index + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {
            key: value.to(device)
            for key, value in encodings.items()
        }
        with torch.no_grad():
            outputs = model(**encodings)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(batch_predictions.cpu().tolist())
    return predictions


def benchmark_tfidf(
    model_path: Path,
    texts: list[str],
    num_repeats: int,
    warmup_examples: int,
) -> dict[str, Any]:
    """Benchmark TF-IDF vectorizer plus Logistic Regression together."""
    print("Benchmarking TF-IDF + Logistic Regression on CPU...")
    require_path(model_path, "TF-IDF natural model")
    model = TfidfLogisticRegressionBaseline.load(str(model_path))
    timing = measure_predict_function(
        predict_fn=model.predict,
        texts=texts,
        num_repeats=num_repeats,
        warmup_examples=warmup_examples,
    )
    return {
        "model_name": "tfidf_logistic_regression",
        "display_name": "TF-IDF + Logistic Regression",
        "model_path": str(model_path),
        "batch_size": "all",
        "device": "cpu",
        "max_length": None,
        "num_examples": len(texts),
        "num_repeats": num_repeats,
        "model_size_mb": deployable_model_size_mb(model_path),
        "artifact_size_mb": artifact_size_mb(model_path),
        "accepted_default_f1": ACCEPTED_DEFAULT_F1["tfidf_logistic_regression"],
        **timing,
    }


def benchmark_transformer(
    model_name: str,
    display_name: str,
    model_path: Path,
    texts: list[str],
    batch_size: int,
    max_length: int,
    num_repeats: int,
    warmup_examples: int,
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark one saved Transformer model on CPU."""
    print(f"Benchmarking {display_name} on CPU with batch_size={batch_size}...")
    require_path(model_path, f"{display_name} model directory")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    def predict_fn(batch_texts: list[str]) -> list[int]:
        return transformer_predict(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    timing = measure_predict_function(
        predict_fn=predict_fn,
        texts=texts,
        num_repeats=num_repeats,
        warmup_examples=warmup_examples,
    )
    del model
    return {
        "model_name": model_name,
        "display_name": display_name,
        "model_path": str(model_path),
        "batch_size": batch_size,
        "device": "cpu",
        "max_length": max_length,
        "num_examples": len(texts),
        "num_repeats": num_repeats,
        "model_size_mb": deployable_model_size_mb(model_path),
        "artifact_size_mb": artifact_size_mb(model_path),
        "accepted_default_f1": ACCEPTED_DEFAULT_F1[model_name],
        **timing,
    }


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown."""
    headers = list(dataframe.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in dataframe.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_markdown(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write benchmark table and short interpretation."""
    table_columns = [
        "display_name",
        "batch_size",
        "accepted_default_f1",
        "model_size_mb",
        "avg_total_inference_time_sec",
        "avg_inference_time_per_example_ms",
        "examples_per_second",
    ]
    table_df = pd.DataFrame(results)[table_columns].rename(
        columns={
            "display_name": "Model",
            "batch_size": "Batch size",
            "accepted_default_f1": "Default F1",
            "model_size_mb": "Model size MB",
            "avg_total_inference_time_sec": "Total sec",
            "avg_inference_time_per_example_ms": "ms/example",
            "examples_per_second": "examples/sec",
        }
    )
    markdown = [
        "# CPU Benchmark Comparison",
        "",
        "Benchmark uses the same natural-distribution test examples for every model.",
        "Transformer timings include tokenization and model forward pass on CPU.",
        "Model size reports deployable top-level model/tokenizer files and excludes saved training checkpoints.",
        "",
        "## Benchmark Table",
        "",
        dataframe_to_markdown(table_df),
        "",
        "## Interpretation",
        "",
        "- BERT-base gives the best F1 but is expected to be slower and larger than BERT-tiny.",
        "- BERT-tiny distilled improves over supervised BERT-tiny while keeping the same BERT-tiny inference cost.",
        "- This supports the distillation objective: a better compact model with lower cost than BERT-base.",
        "",
    ]
    output_path.write_text("\n".join(markdown), encoding="utf-8")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a readable benchmark summary."""
    print("\nCPU benchmark results:")
    for result in results:
        print(f"\n{result['display_name']} (batch_size={result['batch_size']})")
        print(f"  F1 used for comparison: {result['accepted_default_f1']:.4f}")
        print(f"  deployable model size: {result['model_size_mb']:.2f} MB")
        print(f"  examples: {result['num_examples']}")
        print(
            "  avg total inference time: "
            f"{result['avg_total_inference_time_sec']:.4f} sec"
        )
        print(
            "  avg time/example: "
            f"{result['avg_inference_time_per_example_ms']:.4f} ms"
        )
        print(f"  examples/sec: {result['examples_per_second']:.2f}")


def main() -> None:
    """Run the CPU-only natural model benchmark."""
    args = parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    device = torch.device("cpu")

    test_csv = resolve_project_path(args.test_csv)
    output_json = resolve_project_path(args.output_json)
    output_csv = resolve_project_path(args.output_csv)
    output_md = resolve_project_path(args.output_md)
    for output_path in [output_json, output_csv, output_md]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = load_texts(test_csv, args.text_column, limit=args.limit)
    print(f"Loaded {len(texts)} benchmark examples from {test_csv}")
    print("Running on CPU only.")

    model_specs = [
        (
            "bert_tiny_supervised",
            "BERT-tiny supervised",
            PROJECT_ROOT / "results" / "bert_tiny_supervised_natural",
        ),
        (
            "bert_tiny_distilled_t2_a07",
            "BERT-tiny distilled T=2 alpha=0.7",
            PROJECT_ROOT / "results" / "bert_tiny_distilled_t2_a07_natural",
        ),
        (
            "bert_base_teacher",
            "BERT-base teacher",
            PROJECT_ROOT / "results" / "bert_base_supervised",
        ),
    ]

    results: list[dict[str, Any]] = [
        benchmark_tfidf(
            model_path=PROJECT_ROOT / "results" / "tfidf_natural_model.joblib",
            texts=texts,
            num_repeats=args.num_repeats,
            warmup_examples=args.warmup_examples,
        )
    ]
    for model_name, display_name, model_path in model_specs:
        for batch_size in args.batch_sizes:
            results.append(
                benchmark_transformer(
                    model_name=model_name,
                    display_name=display_name,
                    model_path=model_path,
                    texts=texts,
                    batch_size=batch_size,
                    max_length=args.max_length,
                    num_repeats=args.num_repeats,
                    warmup_examples=args.warmup_examples,
                    device=device,
                )
            )

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    write_markdown(results, output_md)
    print_summary(results)
    print(f"\nSaved JSON to {output_json}")
    print(f"Saved CSV to {output_csv}")
    print(f"Saved Markdown to {output_md}")


if __name__ == "__main__":
    main()
