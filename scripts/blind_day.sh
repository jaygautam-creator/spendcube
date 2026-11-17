#!/bin/sh
# Blind-day driver (docs/BLIND_DAY.md): C -> B2 -> A2 strictly in sequence, each guarded so one failure never stops
# the next. Usage: scripts/blind_day.sh <blind_dir>   (expects <blind_dir>/C, /B2, /A2 and submission/C.yaml edited)
# Only configs are used; no code is changed. Outputs go to submission/.
set -u
B="${1:?usage: blind_day.sh <blind_dir>}"
PY="${PY:-.venv/bin/python}"
run() { echo "=== $*"; "$PY" -m spendcube run "$@" || echo "!!! run failed (continuing): $*"; }
run --config submission/C.yaml --llm-mode blind --out submission
run --config configs/B.yaml --input-dir "$B/B2" --suffix B2 --llm-mode blind --out submission
run --config configs/A.yaml --input-dir "$B/A2" --suffix A2 --llm-mode blind --out submission
ls -l submission/jaygautam__*.csv
