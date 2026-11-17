#!/usr/bin/env python3
"""Pre-commit data guard: refuse data files outside the allowed folders, and kit file names anywhere.

Allowed data-like files: outputs/, submission/, tests/fixtures/.
"""
import subprocess
import sys
from pathlib import PurePosixPath

DATA_EXT = {".csv", ".xlsx", ".xls", ".parquet", ".jsonl", ".sqlite", ".db", ".pkl", ".joblib", ".zip"}
ALLOWED_PREFIX = ("outputs/", "submission/", "tests/fixtures/")
KIT_NAMES = {
    "erp_goods_receipts.csv", "invoice_tool_export.csv", "labels_a.csv", "taxonomy_a.csv",
    "invoice_lines.csv", "fx_rates_usd.csv", "seed_labels.csv", "taxonomy_b.csv",
    "output_template.csv", "cantill_dataml_task_kit_vf.zip", "cantill_dataml_task_brief.pdf",
}


def staged() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def violations(paths: list[str]) -> list[str]:
    bad = []
    for p in paths:
        pp = PurePosixPath(p)
        name = pp.name.lower()
        if name in KIT_NAMES or pp.parts[0] == "Cantill_DataML_Task_Kit" or ".env" == name or (name.startswith(".env.") and name != ".env.example"):
            bad.append(f"{p}  (kit/secret file)")
        elif pp.suffix.lower() in DATA_EXT and not p.startswith(ALLOWED_PREFIX):
            bad.append(f"{p}  (data file outside outputs/, submission/, tests/fixtures/)")
    return bad


def content_violations(paths: list[str]) -> list[str]:
    """Light scan (supplier + taxonomy names) of staged .md/.yaml/.py files; only when the kit and pandas exist."""
    try:
        sys.path.insert(0, str(PurePosixPath(__file__).parent))
        import leak_scan
        if not leak_scan.KIT.exists():
            return []
        strings = leak_scan.light_strings()
    except Exception:
        return []
    bad = []
    for p in paths:
        if PurePosixPath(p).suffix.lower() not in {".md", ".yaml", ".yml", ".py"} or p.startswith(leak_scan.SKIP_PREFIX):
            continue
        text = subprocess.run(["git", "show", f":{p}"], capture_output=True, text=True).stdout
        bad += [f"{p}: {h}  (task data string)" for h in leak_scan.scan_text(text, strings)]
    return bad


if __name__ == "__main__":
    files = staged()
    v = violations(files) + content_violations(files)
    if v:
        print("DATA GUARD: commit refused. Offending files:", file=sys.stderr)
        for x in v:
            print("  -", x, file=sys.stderr)
        sys.exit(1)
