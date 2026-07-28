#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    suite_path = Path(args.suite)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir or results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_json(suite_path)
    traces = suite.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("golden suite must contain a non-empty traces list")

    rows: list[dict[str, Any]] = []
    weighted_score = 0.0
    total_weight = 0.0
    hard_failures = 0

    for trace in traces:
        trace_id = str(trace["id"])
        weight = float(trace["weight"])
        total_weight += weight
        aggregate_path = results_root / f"{args.run_id}_{trace_id}" / "aggregate.json"
        row = {
            "run_id": args.run_id,
            "compose_file": args.compose_file,
            "trace_id": trace_id,
            "trace_path": trace.get("path"),
            "weight": weight,
            "aggregate_path": str(aggregate_path),
            "aggregate_found": aggregate_path.exists(),
            "ers": 0.0,
            "weighted_ers": 0.0,
            "request_count": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "timeout_requests": 0,
            "zero_token_responses": 0,
            "ttft_mean_ms": None,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p50_ms": None,
            "tpot_p95_ms": None,
            "tpot_p99_ms": None,
            "latency_mean_ms": None,
            "output_tokens_per_second": 0.0,
            "requests_per_second": 0.0,
            "total_benchmark_duration_s": 0.0,
        }
        if aggregate_path.exists():
            aggregate = load_json(aggregate_path)
            ers = float(aggregate.get("ers", 0.0))
            row.update(
                {
                    "ers": ers,
                    "weighted_ers": ers * weight,
                    "request_count": int(aggregate.get("request_count", 0)),
                    "successful_requests": int(aggregate.get("successful_requests", 0)),
                    "failed_requests": int(aggregate.get("failed_requests", 0)),
                    "timeout_requests": int(aggregate.get("timeout_requests", 0)),
                    "zero_token_responses": int(aggregate.get("zero_token_responses", 0)),
                    "ttft_mean_ms": aggregate.get("ttft_mean_ms"),
                    "ttft_p50_ms": aggregate.get("ttft_p50_ms"),
                    "ttft_p95_ms": aggregate.get("ttft_p95_ms"),
                    "ttft_p99_ms": aggregate.get("ttft_p99_ms"),
                    "tpot_mean_ms": aggregate.get("tpot_mean_ms"),
                    "tpot_p50_ms": aggregate.get("tpot_p50_ms"),
                    "tpot_p95_ms": aggregate.get("tpot_p95_ms"),
                    "tpot_p99_ms": aggregate.get("tpot_p99_ms"),
                    "latency_mean_ms": aggregate.get("latency_mean_ms"),
                    "output_tokens_per_second": aggregate.get("output_tokens_per_second", 0.0),
                    "requests_per_second": aggregate.get("requests_per_second", 0.0),
                    "total_benchmark_duration_s": aggregate.get("total_benchmark_duration_s", 0.0),
                }
            )
            weighted_score += ers * weight
            if row["failed_requests"] or row["timeout_requests"] or row["zero_token_responses"]:
                hard_failures += 1
        else:
            hard_failures += 1
        rows.append(row)

    if total_weight <= 0:
        raise ValueError("golden suite total weight must be > 0")

    normalized_score = weighted_score / total_weight
    summary = {
        "run_id": args.run_id,
        "compose_file": args.compose_file,
        "suite": suite.get("name"),
        "suite_path": str(suite_path),
        "results_root": str(results_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_score": normalized_score,
        "weighted_score_sum": weighted_score,
        "total_weight": total_weight,
        "trace_count": len(rows),
        "hard_failure_trace_count": hard_failures,
        "total_requests": sum(int(row["request_count"]) for row in rows),
        "successful_requests": sum(int(row["successful_requests"]) for row in rows),
        "failed_requests": sum(int(row["failed_requests"]) for row in rows),
        "timeout_requests": sum(int(row["timeout_requests"]) for row in rows),
        "zero_token_responses": sum(int(row["zero_token_responses"]) for row in rows),
        "traces": rows,
    }

    write_json(output_dir / "golden_summary.json", summary)
    write_csv(output_dir / "golden_summary.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize weighted golden workload scores.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[golden-summary][error] {exc}", file=sys.stderr)
        raise
