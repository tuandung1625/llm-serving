#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import aggregate_results
from benchmark.result_writer import collect_metadata, create_experiment_dir, write_results
from benchmark.scheduler import build_request_plans
from benchmark.schemas import BenchmarkConfig
from benchmark.streaming_client import OpenAIStreamingClient
from benchmark.tokenizer_utils import load_token_counter
from benchmark.trace_loader import load_trace
from benchmark.workload_runner import WorkloadRunner


LOGGER = logging.getLogger("benchmark")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[handler], force=True)


async def run_benchmark(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args)
    config.validate()
    trace = load_trace(config.trace_path)
    plans = build_request_plans(trace)
    experiment_id, experiment_dir = create_experiment_dir(config.output_dir, args.experiment_id)
    start_timestamp = datetime.now(timezone.utc).isoformat()
    LOGGER.info(
        "benchmark_starting",
        extra={
            "experiment_id": experiment_id,
            "experiment_dir": str(experiment_dir),
            "request_count": len(plans),
            "trace": trace.name,
        },
    )
    token_counter = load_token_counter(config.model_path, required=config.tokenizer_required and not args.allow_tokenizer_fallback)
    client = OpenAIStreamingClient(
        config.base_url,
        config.model,
        token_counter,
        request_timeout_s=config.request_timeout_s,
        connect_timeout_s=config.connect_timeout_s,
        max_connections=config.max_connections,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    started_s = time.perf_counter()
    try:
        runner = WorkloadRunner(
            experiment_id=experiment_id,
            trace=trace,
            plans=plans,
            client=client,
            token_counter=token_counter,
            server_model_name=config.model,
        )
        request_results = await runner.run()
    finally:
        await client.close()
    total_duration_s = time.perf_counter() - started_s
    aggregate = aggregate_results(experiment_id, request_results, total_duration_s)
    end_timestamp = datetime.now(timezone.utc).isoformat()
    metadata = collect_metadata(
        config=config,
        trace=trace,
        trace_path=config.trace_path,
        experiment_id=experiment_id,
        vllm_image=args.vllm_image,
        vllm_version=args.vllm_version,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    metadata["token_counter"] = getattr(token_counter, "name", token_counter.__class__.__name__)
    write_results(
        experiment_dir,
        request_results,
        aggregate,
        metadata,
        include_output_text=config.include_output_text,
    )
    print(json.dumps({"experiment_dir": str(experiment_dir), "aggregate": asdict(aggregate)}, indent=2, sort_keys=True))
    return 0


def load_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Benchmark config root must be a mapping")
    trace_override = args.trace or os.environ.get("TRACE_PATH")
    trace_path = trace_override or raw.get("trace_path", "configs/sample_trace.json")
    model_path = args.model_path or os.environ.get("MODEL_PATH") or raw.get("model_path", "./model")
    output_dir = args.output_dir or os.environ.get("RESULTS_DIR") or raw.get("output_dir", "./results")
    trace_path = str(_resolve_relative(PROJECT_ROOT if trace_override else config_path.parent, trace_path))
    model_path = str(_resolve_relative(PROJECT_ROOT, model_path))
    output_dir = str(_resolve_relative(PROJECT_ROOT, output_dir))
    return BenchmarkConfig(
        base_url=args.base_url or os.environ.get("VLLM_BASE_URL") or raw.get("base_url", "http://127.0.0.1:8000"),
        model=args.model or os.environ.get("SERVED_MODEL_NAME") or raw.get("model", "LFM2.5-1.2B-Instruct"),
        model_path=model_path,
        trace_path=trace_path,
        output_dir=output_dir,
        request_timeout_s=float(args.timeout_s or os.environ.get("REQUEST_TIMEOUT_S") or raw.get("request_timeout_s", 180.0)),
        connect_timeout_s=float(raw.get("connect_timeout_s", 10.0)),
        max_connections=int(raw.get("max_connections", 128)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        include_output_text=bool(raw.get("include_output_text", False)),
        tokenizer_required=bool(raw.get("tokenizer_required", True)),
    )


def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-turn streaming benchmark against vLLM.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--trace", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--allow-tokenizer-fallback", action="store_true")
    parser.add_argument("--vllm-image", default=os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.22.1"))
    parser.add_argument("--vllm-version", default=os.environ.get("VLLM_VERSION", "0.22.1"))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
