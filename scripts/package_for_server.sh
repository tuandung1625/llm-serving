#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
ARCHIVE="${1:-${PROJECT_NAME}_$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"

cd "$(dirname "$PROJECT_ROOT")"

tar \
  --exclude="${PROJECT_NAME}/model" \
  --exclude="${PROJECT_NAME}/results/*" \
  --exclude="${PROJECT_NAME}/.venv" \
  --exclude="${PROJECT_NAME}/venv" \
  --exclude="${PROJECT_NAME}/.pytest_cache" \
  --exclude="*/__pycache__" \
  -czf "$ARCHIVE" \
  "$PROJECT_NAME"

printf '%s\n' "$ARCHIVE"

