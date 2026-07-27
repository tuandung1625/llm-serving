#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import final_submission_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate final post-accuracy-gate score from ERS and GPQA accuracy.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ers", type=float, help="ERS in [0, 1].")
    source.add_argument("--aggregate-json", help="Path to aggregate.json containing an ers field.")
    parser.add_argument("--baseline-accuracy", type=float, required=True)
    parser.add_argument("--submission-accuracy", type=float, required=True)
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    ers = args.ers if args.ers is not None else _read_ers(Path(args.aggregate_json))
    payload = final_submission_score(
        ers=ers,
        baseline_accuracy=args.baseline_accuracy,
        submission_accuracy=args.submission_accuracy,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.write_json:
        with Path(args.write_json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


def _read_ers(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Aggregate JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "ers" not in payload:
        raise ValueError(f"Aggregate JSON does not contain an ers field: {path}")
    return float(payload["ers"])


if __name__ == "__main__":
    raise SystemExit(main())

