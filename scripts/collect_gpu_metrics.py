#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.gpu_metrics import run_gpu_monitor


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect lightweight GPU metrics to CSV using nvidia-smi.")
    parser.add_argument("--output", default="results/gpu_metrics.csv")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()
    return run_gpu_monitor(args.output, args.interval_s, args.duration_s)


if __name__ == "__main__":
    raise SystemExit(main())

