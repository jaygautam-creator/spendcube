#!/bin/sh
# Usage: scripts/fresh_clone_test.sh <repo path or url> <ABSOLUTE kit dir containing company_A/> [python]
# Clones the repo to a temp dir, builds a venv, installs requirements, runs the tests, then runs A (--limit 500, no LLM).
set -eu
REPO="${1:?repo path}"; KIT="${2:?absolute kit dir}"; PY="${3:-python3}"
T0=$(date +%s); TMP=$(mktemp -d); echo "temp dir: $TMP"
git clone -q "$REPO" "$TMP/repo"; cd "$TMP/repo"
"$PY" -m venv .venv; . .venv/bin/activate
pip install -q --upgrade pip; pip install -q -r requirements.txt
T1=$(date +%s); echo "install: $((T1 - T0))s"
python -m pytest -q 2>&1 | tail -3
T2=$(date +%s); echo "pytest: $((T2 - T1))s"
python -m spendcube run --config configs/A.yaml --input-dir "$KIT/company_A" --kit-dir "$KIT/company_A" \
  --out "$TMP/out" --limit 500 --no-llm 2>&1 | tail -2
python -m spendcube validate --config configs/A.yaml --input-dir "$KIT/company_A" --kit-dir "$KIT/company_A" \
  --limit 500 --output "$TMP/out/jaygautam__A.csv"
T3=$(date +%s); echo "run+validate: $((T3 - T2))s  total: $((T3 - T0))s"
