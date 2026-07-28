#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/run_golden_score.sh <docker-compose-file>"
  echo
  echo "Example:"
  echo "  bash scripts/run_golden_score.sh docker-compose-260725-101045-rtx4090-mimic-h200mig.yaml"
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="$ROOT_DIR/llm-serving-baseline"
COMPOSE_FILE="$1"
COMPOSE_PATH="$ROOT_DIR/$COMPOSE_FILE"
LIMIT_FILE="$ROOT_DIR/docker-compose-260725-h200mig-limits.yaml"
LOCAL_MODEL_FILE="$ROOT_DIR/docker-compose-260725-local-model.yaml"
SUITE_PATH="$BASELINE_DIR/configs/golden_suite.json"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(basename "$COMPOSE_FILE" .yaml)"
RESULTS_ROOT="$BASELINE_DIR/results/golden_runs/$RUN_ID"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-1800}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark.yaml}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-240}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-golden_${RUN_ID//[^a-zA-Z0-9]/_}}"

if [ ! -f "$COMPOSE_PATH" ]; then
  echo "[golden][error] compose file not found: $COMPOSE_PATH" >&2
  exit 1
fi
if [ ! -f "$LIMIT_FILE" ]; then
  echo "[golden][error] limit override file not found: $LIMIT_FILE" >&2
  exit 1
fi
if [ ! -f "$LOCAL_MODEL_FILE" ]; then
  echo "[golden][error] local model override file not found: $LOCAL_MODEL_FILE" >&2
  exit 1
fi
if [ ! -d "$BASELINE_DIR/model" ]; then
  echo "[golden][error] missing model directory: $BASELINE_DIR/model" >&2
  echo "[golden][hint] download model first, for example:" >&2
  echo "  cd llm-serving-baseline && python scripts/download_model.py --repo-id LiquidAI/LFM2.5-1.2B-Instruct --local-dir model" >&2
  exit 1
fi
if [ ! -d "$BASELINE_DIR/.venv" ]; then
  echo "[golden][error] missing Python venv: $BASELINE_DIR/.venv" >&2
  echo "[golden][hint] create it first:" >&2
  echo "  cd llm-serving-baseline && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-benchmark.txt" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT"

compose() {
  docker compose \
    -p "$COMPOSE_PROJECT_NAME" \
    -f "$COMPOSE_PATH" \
    -f "$LIMIT_FILE" \
    -f "$LOCAL_MODEL_FILE" \
    "$@"
}

cleanup() {
  status=$?
  echo "[golden] stopping compose project: $COMPOSE_PROJECT_NAME"
  compose down --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

echo "[golden] run_id=$RUN_ID"
echo "[golden] compose=$COMPOSE_FILE"
echo "[golden] results=$RESULTS_ROOT"
echo "[golden] starting vLLM server"
compose up -d

echo "[golden] waiting for health: $HEALTH_URL"
deadline=$((SECONDS + SERVER_START_TIMEOUT_S))
until python3 "$BASELINE_DIR/scripts/healthcheck.py" --url "$HEALTH_URL" --timeout 5 >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "[golden][error] server did not become healthy within ${SERVER_START_TIMEOUT_S}s" >&2
    echo "[golden][hint] recent logs:" >&2
    compose logs --tail=160 model >&2 || true
    exit 1
  fi
  sleep 10
done

echo "[golden] server healthy"

mapfile -t TRACE_ROWS < <(
  python3 - "$SUITE_PATH" <<'PY'
import json
import sys

suite = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for trace in suite["traces"]:
    print(f"{trace['id']}|{trace['path']}")
PY
)

cd "$BASELINE_DIR"
. .venv/bin/activate

for row in "${TRACE_ROWS[@]}"; do
  trace_id="${row%%|*}"
  trace_path="${row#*|}"
  echo "[golden] running trace=$trace_id"
  if ! python scripts/benchmark.py \
    --config "$BENCHMARK_CONFIG" \
    --trace "$trace_path" \
    --output-dir "$RESULTS_ROOT" \
    --experiment-id "${RUN_ID}_${trace_id}" \
    --timeout-s "$REQUEST_TIMEOUT_S"; then
    echo "[golden][warn] trace failed, it will contribute 0 score: $trace_id" >&2
  fi
done

echo "[golden] summarizing"
python scripts/summarize_golden_score.py \
  --suite "$SUITE_PATH" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --compose-file "$COMPOSE_FILE" \
  --output-dir "$RESULTS_ROOT"

echo
echo "[golden] done"
echo "[golden] summary: $RESULTS_ROOT/golden_summary.json"
echo "[golden] csv:     $RESULTS_ROOT/golden_summary.csv"
