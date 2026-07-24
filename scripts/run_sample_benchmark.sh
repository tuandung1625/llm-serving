#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="${TRACE:-configs/sample_trace.json}"

cd "$PROJECT_ROOT"
. .venv/bin/activate

python scripts/benchmark.py --config configs/benchmark.yaml --trace "$TRACE"

EXP_DIR="$(ls -td results/*_baseline* | head -1)"
python scripts/calculate_ers.py "$EXP_DIR/requests.json"

printf '\nResults: %s\n' "$EXP_DIR"

