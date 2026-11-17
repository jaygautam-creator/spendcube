#!/usr/bin/env python3
"""Summarize blind outputs and run reports without printing input data."""
import csv
import re
import os
from pathlib import Path

wanted = ("A2", "B2", "C")
out_dir = Path(os.environ.get("OUT", "submission"))
problems = []
for suffix in wanted:
    path = out_dir / f"jaygautam__{suffix}.csv"
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    total = len(rows)
    excluded = [r for r in rows if r.get("include") != "1"]
    output_spend = sum(float(r.get("spend") or 0) for r in rows)
    reports = sorted(Path("runs").glob("*/report.md"))
    matching = [p for p in reports if f"({path.name})" in p.read_text(encoding="utf-8")[:160]]
    report = matching[-1] if matching else None
    text = report.read_text(encoding="utf-8") if report else ""
    validation = "OK" if "- validation: OK" in text else "FAILED" if "- validation: FAILED" in text else "MISSING"
    def note(name):
        for line in text.splitlines():
            if line.startswith(f"- {name}:"):
                return line.split(":", 1)[1].strip()
        return "n/a"
    excluded_spend = 0.0
    for line in text.splitlines():
        match = re.search(r"\(spend ([\d,]+(?:\.\d+)?)\)", line)
        if match:
            excluded_spend += float(match.group(1).replace(",", ""))
    total_spend = output_spend + excluded_spend
    calls = note("calls")
    if calls == "n/a":
        calls = "0"
    print(f"{path.name}: rows={total}, validator={validation}, excluded_count={len(excluded)}/{total}, "
          f"excluded_spend={excluded_spend:.2f}/{total_spend:.2f} ({excluded_spend/max(abs(total_spend),1e-9):.2%}), "
          f"tier_spend_share={note('method_spend_share')}, requests={calls}, deadline_hit={note('deadline_hit')}")
    if validation != "OK":
        problems.append(f"{suffix} validation {validation}")
if not any((out_dir / f"jaygautam__{s}.csv").exists() for s in wanted):
    problems.append("no output files")
print("GO" if not problems else "NO-GO: " + "; ".join(problems))
