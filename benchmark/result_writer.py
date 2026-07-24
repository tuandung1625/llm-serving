from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.metrics import aggregate_to_dict
from benchmark.schemas import AggregateMetrics, BenchmarkConfig, RequestResult, WorkloadTrace


VLLM_REQUIRED_ARGS = [
    "--model=/model",
    "--served-model-name=LFM2.5-1.2B-Instruct",
    "--host=0.0.0.0",
    "--port=8000",
    "--max-model-len=32768",
    "--gpu-memory-utilization=0.95",
    "--tensor-parallel-size=1",
    "--enable-prefix-caching",
]


def create_experiment_dir(output_dir: str | Path, experiment_id: str | None = None) -> tuple[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if experiment_id is None:
        experiment_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ_baseline")
    path = root / experiment_id
    suffix = 1
    original = path
    while path.exists():
        path = Path(f"{original}_{suffix}")
        suffix += 1
    path.mkdir(parents=True)
    return path.name, path


def write_results(
    experiment_dir: str | Path,
    request_results: list[RequestResult],
    aggregate: AggregateMetrics,
    metadata: dict[str, Any],
    *,
    include_output_text: bool = False,
) -> None:
    destination = Path(experiment_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request_rows = [row.to_public_dict(include_text=include_output_text) for row in request_results]
    _write_json(destination / "requests.json", request_rows)
    _write_csv(destination / "requests.csv", request_rows)
    aggregate_dict = aggregate_to_dict(aggregate)
    _write_json(destination / "aggregate.json", aggregate_dict)
    _write_csv(destination / "aggregate.csv", [aggregate_dict])
    _write_json(destination / "metadata.json", metadata)


def collect_metadata(
    *,
    config: BenchmarkConfig,
    trace: WorkloadTrace,
    trace_path: str | Path,
    experiment_id: str,
    vllm_image: str,
    vllm_version: str,
    start_timestamp: str,
    end_timestamp: str,
) -> dict[str, Any]:
    model_metadata = _read_model_metadata(config.model_path)
    return {
        "experiment_id": experiment_id,
        "git_commit_hash": _command_output(["git", "rev-parse", "HEAD"]),
        "docker_image": vllm_image,
        "gpu": _gpu_info(),
        "nvidia_driver_version": _nvidia_driver_version(),
        "docker_version": _command_output(["docker", "--version"]),
        "docker_compose_version": _command_output(["docker", "compose", "version"]),
        "vllm_version": vllm_version,
        "python_version": platform.python_version(),
        "model_path": config.model_path,
        "model_revision": model_metadata.get("revision"),
        "model_repo_id": model_metadata.get("repo_id"),
        "vllm_entrypoint": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "vllm_arguments": VLLM_REQUIRED_ARGS,
        "benchmark_config": asdict(config),
        "workload_trace": asdict(trace),
        "workload_trace_hash_sha256": sha256_file(trace_path),
        "tokenizer_required": config.tokenizer_required,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _command_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _gpu_info() -> dict[str, Any] | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {"raw": first}
    return {
        "name": parts[0],
        "memory_total_mib": _int_or_none(parts[1]),
        "memory_free_mib": _int_or_none(parts[2]),
        "driver_version": parts[3],
    }


def _nvidia_driver_version() -> str | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    return output.splitlines()[0].strip()


def _read_model_metadata(model_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(model_path) / ".baseline_model_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None

