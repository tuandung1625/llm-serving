#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN="${LEVEL5_PATTERN:-docker-compose/docker-compose-level5-*.yaml}"

cd "$ROOT_DIR"

mapfile -t FILES < <(find docker-compose -maxdepth 1 -type f -name "$(basename "$PATTERN")" | sort)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "[level5][error] no files matched: $PATTERN" >&2
  exit 1
fi

echo "[level5] files=${#FILES[@]}"
for file in "${FILES[@]}"; do
  echo
  echo "[level5] running $file"
  if ! bash run_golden_score.sh "$file"; then
    echo "[level5][warn] failed: $file" >&2
  fi
done

echo
echo "[level5] summarizing all golden runs"
python3 scripts/summarize_all_golden_results.py
python3 scripts/summarize_golden_by_compose.py
