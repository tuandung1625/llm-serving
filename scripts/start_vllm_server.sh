#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-LFM2.5-1.2B-Instruct}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
SERVER_COMPOSE_FILE="${SERVER_COMPOSE_FILE:-docker-compose/docker-compose.server.yml}"

log() {
  printf '[server] %s\n' "$*"
}

cd "$PROJECT_ROOT"

if ! docker info >/dev/null 2>&1; then
  log "docker daemon chua chay, thu start dockerd"
  mkdir -p /var/run /var/lib/docker /var/log
  pkill dockerd 2>/dev/null || true
  pkill containerd 2>/dev/null || true
  nohup dockerd \
    --host=unix:///var/run/docker.sock \
    --iptables=false \
    --bridge=none \
    --ip-forward=false \
    --ip-masq=false \
    > /var/log/dockerd.log 2>&1 &
  sleep 8
fi

docker info >/dev/null 2>&1 || {
  tail -100 /var/log/dockerd.log >&2 || true
  printf '[server][error] docker daemon chua san sang\n' >&2
  exit 1
}

if [[ ! -d model ]]; then
  printf '[server][error] thieu ./model, hay chay scripts/setup_gpu_server.sh truoc\n' >&2
  exit 1
fi

log "start vLLM docker compose"
docker compose --env-file configs/baseline.env -f "$SERVER_COMPOSE_FILE" up -d

log "wait health http://127.0.0.1:$PORT/health"
. .venv/bin/activate
deadline=$((SECONDS + HEALTH_TIMEOUT_S))
until python scripts/healthcheck.py --url "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    docker compose -f "$SERVER_COMPOSE_FILE" logs --tail=200 lfm-vllm >&2 || true
    printf '[server][error] health check timeout sau %ss\n' "$HEALTH_TIMEOUT_S" >&2
    exit 1
  fi
  sleep 10
done

log "smoke test"
python scripts/smoke_test.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL_NAME"

cat <<EOF

Server da san sang.

Logs:
docker compose -f $SERVER_COMPOSE_FILE logs -f lfm-vllm

Benchmark:
. .venv/bin/activate
python scripts/benchmark.py --config configs/benchmark.yaml --trace configs/sample_trace.json
EOF
