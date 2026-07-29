#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = load_rows(results_root)
    rows.sort(key=lambda row: float(row["golden_score"]), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, args.top)

    print(f"[summary] rows={len(rows)}")
    print(f"[summary] csv={output_csv}")
    print(f"[summary] md={output_md}")
    if rows:
        best = rows[0]
        print(
            "[summary] best="
            f"{best['golden_score']:.6f} "
            f"{Path(str(best['compose_file'])).name}"
        )
    return 0


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(results_root.glob("*/golden_summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row: dict[str, Any] = {
            "rank": 0,
            "compose_file": summary.get("compose_file", ""),
            "run_dir": summary_path.parent.name,
            "golden_score": float(summary.get("golden_score", 0.0)),
            "hard_failure_trace_count": int(summary.get("hard_failure_trace_count", 0)),
            "failed_requests": int(summary.get("failed_requests", 0)),
            "timeout_requests": int(summary.get("timeout_requests", 0)),
            "zero_token_responses": int(summary.get("zero_token_responses", 0)),
            "successful_requests": int(summary.get("successful_requests", 0)),
            "total_requests": int(summary.get("total_requests", 0)),
        }
        for trace in summary.get("traces", []):
            trace_id = str(trace.get("trace_id", "trace")).replace("golden_", "")
            row[f"{trace_id}_ers"] = trace.get("ers")
            row[f"{trace_id}_ttft_p95_ms"] = trace.get("ttft_p95_ms")
            row[f"{trace_id}_tpot_p95_ms"] = trace.get("tpot_p95_ms")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], top: int) -> None:
    lines = [
        "# Golden Results Ranking",
        "",
        "Generated from local golden workload results. This is not the official BTC trace.",
        "",
        f"Total runs: {len(rows)}",
        "",
        f"## Top {min(top, len(rows))}",
        "",
        "| Rank | GoldenScore | Failed | Timeout | Zero | Compose |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:top]:
        lines.append(
            "| {rank} | {score:.6f} | {failed} | {timeout} | {zero} | `{compose}` |".format(
                rank=row["rank"],
                score=float(row["golden_score"]),
                failed=row["failed_requests"],
                timeout=row["timeout_requests"],
                zero=row["zero_token_responses"],
                compose=Path(str(row["compose_file"])).name,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all golden benchmark runs into one ranking.")
    parser.add_argument("--results-root", default="results/golden_runs")
    parser.add_argument("--output-csv", default="results/golden_runs_ranking.csv")
    parser.add_argument("--output-md", default="GOLDEN_RESULTS_ANALYSIS.md")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
