#!/bin/sh
# Human-operated blind day driver. See docs/BLIND_DAY.md.
set -u
PY="${PY:-.venv/bin/python}"
MODE="${MODE:-blind}"
cmd="${1:-}"
run() { echo "=== $*"; "$PY" -m spendcube run "$@" || { echo "!!! run failed: $*"; return 1; }; }
case "$cmd" in
  ab)
    B="${2:?usage: blind_day.sh ab <blind_dir>}"
    run --config configs/B.yaml --input-dir "$B/B2" --suffix B2 --llm-mode "$MODE" --out submission || exit 1
    run --config configs/A.yaml --input-dir "$B/A2" --suffix A2 --llm-mode "$MODE" --out submission || exit 1
    "$PY" scripts/blind_summary.py
    ;;
  c)
    B="${2:?usage: blind_day.sh c <blind_dir> [YYYY-MM-DDTHH:MM]}"
    DEADLINE="${3:-2026-11-19T11:20}"
    CCFG="${C_CONFIG:-submission/C.yaml}"
    if ! "$PY" -c 'import sys; from spendcube.config.loader import load_config; from spendcube.taxonomy.tree import Taxonomy; c=load_config(sys.argv[1]); Taxonomy.load(c.taxonomy)' "$CCFG"; then
      echo "NO-GO: invalid C config ($CCFG)" >&2; exit 2
    fi
    CDIR="$B/C"; [ -d "$CDIR" ] || CDIR="$B"
    run --config "$CCFG" --input-dir "$CDIR" --deadline-at "$DEADLINE" --llm-mode "$MODE" --out submission || exit 1
    "$PY" scripts/blind_summary.py
    ;;
  summary) "$PY" scripts/blind_summary.py ;;
  *) echo "usage: scripts/blind_day.sh ab <blind_dir> | c <blind_dir> [deadline] | summary" >&2; exit 2 ;;
esac
