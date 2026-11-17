"""Run report (report.md): counts, totals, timings, notes, warnings."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(ctx, out: pd.DataFrame, path: Path, errors: list[str]) -> None:
    inc = out["include"] == "1"
    L = [f"# Run report: {ctx.cfg.company} ({ctx.cfg.out_filename})", ""]
    L.append(f"- rows: {len(out)}  included: {int(inc.sum())}  excluded: {int((~inc).sum())}")
    L.append(f"- total spend ({ctx.cfg.reporting_currency}): {out['spend'].sum():,.2f}")
    L.append(f"- validation: {'OK' if not errors else 'FAILED'}")
    for e in errors:
        L.append(f"  - {e}")
    L.append("")
    L.append("## Exclusions")
    for k, v in out.loc[~inc, "exclusion_reason"].value_counts().items():
        sp = ctx.state.get("excluded_spend", {}).get(k)
        L.append(f"- {k}: {v}" + (f" (spend {sp:,.2f})" if sp is not None else ""))
    L.append("")
    L.append("## Timings (s)")
    for k, v in ctx.timings.items():
        L.append(f"- {k}: {v}")
    for sec, d in ctx.notes.items():
        L.append("")
        L.append(f"## {sec}")
        for k, v in d.items():
            L.append(f"- {k}: {v}")
    if ctx.warnings:
        L.append("")
        L.append("## Warnings")
        L.extend(f"- {w}" for w in ctx.warnings[:100])
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
