#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/run_h200mig_compose.sh <compose-file> [compose args...]"
  echo
  echo "Examples:"
  echo "  bash scripts/run_h200mig_compose.sh docker-compose-260725-101040-e00-bf16-bf16-prefixoff-balanced.yaml up -d"
  echo "  bash scripts/run_h200mig_compose.sh docker-compose-260725-101040-e00-bf16-bf16-prefixoff-balanced.yaml logs -f model"
  echo "  bash scripts/run_h200mig_compose.sh docker-compose-260725-101040-e00-bf16-bf16-prefixoff-balanced.yaml down"
  exit 2
fi

compose_file="$1"
shift

if [ ! -f "$compose_file" ]; then
  echo "[error] compose file not found: $compose_file" >&2
  exit 1
fi

limit_file="docker-compose-260725-h200mig-limits.yaml"
if [ ! -f "$limit_file" ]; then
  echo "[error] limit override file not found: $limit_file" >&2
  exit 1
fi

local_model_file="docker-compose-260725-local-model.yaml"
if [ ! -f "$local_model_file" ]; then
  echo "[error] local model override file not found: $local_model_file" >&2
  exit 1
fi

exec docker compose -f "$compose_file" -f "$limit_file" -f "$local_model_file" "$@"
