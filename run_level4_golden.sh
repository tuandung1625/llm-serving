#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN="${LEVEL4_PATTERN:-docker-compose/docker-compose-level4-*.yaml}"

cd "$ROOT_DIR"

mapfile -t FILES < <(find docker-compose -maxdepth 1 -type f -name "$(basename "$PATTERN")" | sort)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "[level4][error] no files matched: $PATTERN" >&2
  exit 1
fi

echo "[level4] files=${#FILES[@]}"
for file in "${FILES[@]}"; do
  echo
  echo "[level4] running $file"
  if ! bash run_golden_score.sh "$file"; then
    echo "[level4][warn] failed: $file" >&2
  fi
done

echo
echo "[level4] summarizing all golden runs"
python3 scripts/summarize_all_golden_results.py
