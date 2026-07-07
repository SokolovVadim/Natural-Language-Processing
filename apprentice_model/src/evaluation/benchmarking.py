"""Utilities for model size and inference speed benchmarking."""

from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable
import statistics


BYTES_PER_MB = 1024 * 1024


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Expected a model file, but not found: {file_path}")

    return file_path.stat().st_size / BYTES_PER_MB


def get_directory_size_mb(path: str) -> float:
    """Return recursive directory size in megabytes."""
    directory_path = Path(path)
    if not directory_path.is_dir():
        raise FileNotFoundError(
            f"Expected a model directory, but not found: {directory_path}"
        )

    total_bytes = sum(
        file_path.stat().st_size
        for file_path in directory_path.rglob("*")
        if file_path.is_file()
    )
    return total_bytes / BYTES_PER_MB


def get_path_size_mb(path: str) -> float:
    """Return model artifact size in megabytes for a file or directory."""
    artifact_path = Path(path)
    if artifact_path.is_file():
        return get_file_size_mb(path)
    if artifact_path.is_dir():
        return get_directory_size_mb(path)

    raise FileNotFoundError(f"Model artifact path does not exist: {artifact_path}")


def measure_inference_time(
    predict_fn: Callable[[list[str]], object],
    texts: Iterable[str],
    num_repeats: int = 5,
    warmup: bool = True,
) -> dict[str, float | list[float]]:
    """Measure average inference time for a prediction function.

    Args:
        predict_fn: Callable that performs full inference for a list of texts.
        texts: Text examples to pass to the prediction function.
        num_repeats: Number of timed runs.
        warmup: Whether to run one untimed warm-up pass.

    Returns:
        Timing summary with total time, per-example time, throughput, and runs.
    """
    if num_repeats <= 0:
        raise ValueError(f"num_repeats must be positive, got {num_repeats}.")

    text_list = list(texts)
    if not text_list:
        raise ValueError("Cannot benchmark inference on an empty text collection.")

    if warmup:
        predict_fn(text_list)

    run_times: list[float] = []
    for _ in range(num_repeats):
        start_time = perf_counter()
        predict_fn(text_list)
        run_times.append(perf_counter() - start_time)

    avg_total_time = statistics.mean(run_times)
    num_examples = len(text_list)

    return {
        "avg_total_inference_time_sec": avg_total_time,
        "avg_inference_time_per_example_ms": (
            avg_total_time / num_examples
        )
        * 1000,
        "examples_per_second": num_examples / avg_total_time,
        "run_times_sec": run_times,
    }
