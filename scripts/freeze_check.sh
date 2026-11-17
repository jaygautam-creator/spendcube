#!/bin/sh
# Run immediately before creating/pushing the freeze tag. Prints a PASS/FAIL for each gate.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
PY="${PY:-.venv/bin/python}"
fail=0
check() { name="$1"; shift; if "$@"; then echo "PASS: $name"; else echo "FAIL: $name"; fail=1; fi; }
clean() { [ -z "$(git status --porcelain)" ]; }
head_remote() { [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ]; }
root_ok() { [ "$(git rev-list --max-parents=0 HEAD)" = 5a917959d248da44418fee36e9b86246e3392952 ]; }
tracked_data_ok() {
  bad=$(git ls-files | grep -Ei '\.(csv|xlsx|xls|parquet|jsonl|zip|pkl|joblib|sqlite|db|pdf)$' | grep -Ev '^(outputs|submission|tests/fixtures)/' || true)
  [ -z "$bad" ] && ! git ls-files | grep -Eiq '(^|/)(Cantill_DataML_Task_Kit|\.env$)'
}
keys_ok() {
  ! git ls-files -z | xargs -0 grep -IlE '(AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,}|(API_KEY|SECRET|TOKEN)=[^[:space:]]{8,})' 2>/dev/null | grep -v '^\.env\.example$' | grep -q .
}
leaks() { "$PY" -m pytest -q tests/test_no_data_in_docs.py tests/test_guards.py; }
tests() { "$PY" -m pytest -q; }
outputs() {
  "$PY" - <<'PY'
import csv
from pathlib import Path
from spendcube.config.loader import load_config
from spendcube.ingest.canonical import ingest_all
from spendcube.output.validator import validate_file
from spendcube.taxonomy.tree import Taxonomy
for comp, n in (("A",65549),("B",41028)):
    path=Path(f"outputs/jaygautam__{comp}.csv")
    with path.open(newline="", encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    if len(rows)!=n: raise SystemExit(f"{comp}: expected {n} rows, got {len(rows)}")
    cfg=load_config(f"configs/{comp}.yaml")
    ids=ingest_all(cfg)["record_id_original"].tolist()
    errors=validate_file(path,Taxonomy.load(cfg.taxonomy),ids,expected_name=None)
    if errors: raise SystemExit(f"{comp}: validation failed: {errors[:3]}")
PY
}
configs() { "$PY" - <<'PY'
from spendcube.config.loader import load_config
for p in ("configs/A.yaml","configs/B.yaml","configs/C.example.yaml"):
    load_config(p)
PY
}
docs() { [ -f README.md ] && [ -f APPROACH.md ] && [ -f DECISIONS.md ] && [ "$(wc -w < APPROACH.md)" -le 700 ]; }
prompt_pool() { "$PY" - <<'PY'
from spendcube.llm.prompts import PROMPT_VERSION
from spendcube.config.loader import load_config
c=load_config("configs/A.yaml")
print("PROMPT_VERSION="+PROMPT_VERSION)
print("llm.parallel_pool.enabled="+str(c.classification.get("llm",{}).get("parallel_pool",{}).get("enabled",False)))
PY
}
doctor_safe() {
  if [ -n "${GEMINI_API_KEY:-}" ]; then echo "WARN: blind key present; skipped doctor --blind --list-only"; return 0; fi
  "$PY" -m spendcube doctor --blind --list-only
}
check "clean worktree" clean
check "HEAD equals origin/main" head_remote
check "public snapshot root" root_ok
check "tracked data files and kit paths" tracked_data_ok
check "tracked key patterns" keys_ok
check "leak tests" leaks
check "full pytest suite" tests
check "A/B output row counts and validation" outputs
check "A/B/C configs load" configs
check "required docs and APPROACH word limit" docs
check "prompt version and pool switch" prompt_pool
check "blind doctor list-only safe check" doctor_safe
[ "$fail" -eq 0 ]
