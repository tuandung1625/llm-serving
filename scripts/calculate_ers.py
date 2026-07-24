#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import request_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalculate ERS from saved per-request benchmark results.")
    parser.add_argument("requests_file", help="Path to requests.json or requests.csv")
    parser.add_argument("--write-json", default=None, help="Optional path for a recalculated aggregate JSON.")
    args = parser.parse_args()
    rows = _read_rows(Path(args.requests_file))
    scores = []
    for row in rows:
        ttft = _float_or_none(row.get("ttft_ms"))
        tpot = _float_or_none(row.get("tpot_ms"))
        output_tokens = int(float(row.get("output_token_count") or 0))
        timeout = _bool(row.get("timeout"))
        error = bool(row.get("error_type"))
        scores.append(
            request_score(
                ttft,
                tpot,
                output_tokens,
                error=error,
                timeout=timeout,
            )
        )
    ers = sum(scores) / len(scores) if scores else 0.0
    payload = {
        "request_count": len(rows),
        "ers": ers,
        "mean_request_score": ers,
        "successful_score_count": sum(1 for score in scores if score > 0),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.write_json:
        with Path(args.write_json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file does not exist: {path}")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("requests.json must contain a list")
        return data
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Input must be .json or .csv")


def _float_or_none(value: object) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())

