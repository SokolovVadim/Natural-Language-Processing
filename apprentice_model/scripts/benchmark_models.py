#!/usr/bin/env python
"""Benchmark saved baseline models for size and inference speed."""

from pathlib import Path
import json
import sys
import warnings

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.benchmarking import get_path_size_mb, measure_inference_time
from src.models.tfidf_baseline import TfidfLogisticRegressionBaseline


def require_existing_path(path: Path, description: str) -> None:
    """Raise a clear error if a required path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")


def load_test_texts(path: Path, text_column: str) -> list[str]:
    """Load benchmark texts from the processed test split."""
    require_existing_path(path, "processed test split")

    dataframe = pd.read_csv(path)
    if text_column not in dataframe.columns:
        raise KeyError(f"Text column '{text_column}' not found in {path}.")

    return dataframe[text_column].fillna("").astype(str).tolist()


def predict_with_transformer(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[int]:
    """Run batched Transformer inference including tokenization."""
    predictions: list[int] = []

    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index : start_index + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
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


def benchmark_tfidf_model(
    model_path: Path,
    texts: list[str],
    num_repeats: int,
) -> dict:
    """Benchmark the saved TF-IDF + Logistic Regression baseline."""
    require_existing_path(model_path, "TF-IDF baseline model")

    model = TfidfLogisticRegressionBaseline.load(str(model_path))
    timing = measure_inference_time(
        predict_fn=model.predict,
        texts=texts,
        num_repeats=num_repeats,
        warmup=True,
    )

    return {
        "model_name": "tfidf_logistic_regression",
        "model_path": str(model_path),
        "model_size_mb": get_path_size_mb(str(model_path)),
        "num_examples": len(texts),
        "avg_total_inference_time_sec": timing["avg_total_inference_time_sec"],
        "avg_inference_time_per_example_ms": timing[
            "avg_inference_time_per_example_ms"
        ],
        "examples_per_second": timing["examples_per_second"],
        "device": "cpu",
        "batch_size": None,
        "num_repeats": num_repeats,
    }


def benchmark_student_model(
    model_path: Path,
    texts: list[str],
    num_repeats: int,
    batch_size: int,
    max_length: int,
    model_name: str = "bert_tiny_student",
) -> dict:
    """Benchmark a saved Transformer sequence-classification model."""
    require_existing_path(model_path, f"{model_name} model directory")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    def predict_fn(batch_texts: list[str]) -> list[int]:
        return predict_with_transformer(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )

    timing = measure_inference_time(
        predict_fn=predict_fn,
        texts=texts,
        num_repeats=num_repeats,
        warmup=True,
    )

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "model_size_mb": get_path_size_mb(str(model_path)),
        "num_examples": len(texts),
        "avg_total_inference_time_sec": timing["avg_total_inference_time_sec"],
        "avg_inference_time_per_example_ms": timing[
            "avg_inference_time_per_example_ms"
        ],
        "examples_per_second": timing["examples_per_second"],
        "device": str(device),
        "batch_size": batch_size,
        "num_repeats": num_repeats,
    }


def print_summary(results: list[dict]) -> None:
    """Print a readable benchmark summary."""
    display_names = {
        "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
        "bert_tiny_student": "BERT-tiny Student",
        "bert_tiny_student_distilled": "BERT-tiny Distilled Student",
    }

    print("\nBenchmark results:")
    for result in results:
        print(f"\n{display_names.get(result['model_name'], result['model_name'])}")
        print(f"  size: {result['model_size_mb']:.2f} MB")
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
        print(f"  device: {result['device']}")
        if result["batch_size"] is not None:
            print(f"  batch size: {result['batch_size']}")


def main() -> None:
    """Benchmark saved baseline model artifacts."""
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    text_column = config["dataset"]["text_column"]
    benchmark_config = config.get("benchmark", {})
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    num_repeats = int(benchmark_config.get("num_repeats", 5))
    batch_size = int(benchmark_config.get("batch_size", 32))
    max_length = int(
        benchmark_config.get(
            "max_length",
            config.get("student_baseline", {}).get("max_length", 128),
        )
    )

    test_path = PROJECT_ROOT / benchmark_config.get(
        "test_path",
        "data/processed/test.csv",
    )
    output_json = PROJECT_ROOT / benchmark_config.get(
        "output_json",
        "results/benchmark_results.json",
    )
    output_csv = PROJECT_ROOT / benchmark_config.get(
        "output_csv",
        "results/benchmark_results.csv",
    )

    tfidf_model_path = results_dir / "tfidf_baseline.joblib"
    student_model_path = results_dir / "student_baseline"
    distilled_student_model_path = results_dir / "student_distilled"

    texts = load_test_texts(test_path, text_column)

    results = [
        benchmark_tfidf_model(
            model_path=tfidf_model_path,
            texts=texts,
            num_repeats=num_repeats,
        ),
        benchmark_student_model(
            model_path=student_model_path,
            texts=texts,
            num_repeats=num_repeats,
            batch_size=batch_size,
            max_length=max_length,
            model_name="bert_tiny_student",
        ),
        benchmark_student_model(
            model_path=distilled_student_model_path,
            texts=texts,
            num_repeats=num_repeats,
            batch_size=batch_size,
            max_length=max_length,
            model_name="bert_tiny_student_distilled",
        ),
    ]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(results).to_csv(output_csv, index=False)

    print_summary(results)
    print(f"\nSaved benchmark JSON to {output_json}")
    print(f"Saved benchmark CSV to {output_csv}")


if __name__ == "__main__":
    main()
