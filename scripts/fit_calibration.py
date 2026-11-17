"""Fit the per-tier confidence calibration from held-out evaluation runs (D-48).

usage: python scripts/fit_calibration.py RUN_DIR:OUTPUT_CSV [...] [--extra records.csv] [--out configs/_calibration.json]
`records.csv` (optional, e.g. zero-label LLM units) has columns method,conf,correct[,run].
Prints reliability per band, before and after, using leave-one-run-out so the "after" numbers are out-of-sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.classify.calibration import Calibrator, fit, gather, group_of, reliability  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--out", default="configs/_calibration.json")
    ap.add_argument("--min-n", type=int, default=60)
    a = ap.parse_args()
    parts = [gather([tuple(p.split(":", 1)) for p in a.pairs])] if a.pairs else []
    if a.extra:
        e = pd.read_csv(a.extra)
        e["run"] = e.get("run", "extra")
        e["spend"] = e.get("spend", 1.0)
        parts.append(e[["run", "method", "conf", "correct", "spend"]])
    d = pd.concat(parts, ignore_index=True)
    d["group"] = d["method"].map(group_of)
    report: dict = {"samples": d.groupby(["group", "run"]).size().rename("n").reset_index().to_dict("records")}
    # out-of-sample: fit without run r, apply to run r
    cv = []
    for r in sorted(set(d["run"])):
        tr, te = d[d.run != r], d[d.run == r].copy()
        if len(tr) == 0:
            continue
        cal = fit(tr, a.min_n)
        te["cal"] = cal.apply(te["method"], te["conf"]).to_numpy()
        cv.append(te)
    if cv:
        cv = pd.concat(cv)
        for g in sorted(set(cv["group"])):
            x = cv[cv.group == g]
            report[f"{g}_before"] = reliability(x, "conf")
            report[f"{g}_after_loro"] = reliability(x, "cal")
            report[f"{g}_ece_before_after"] = [round(float(np.abs(x["conf"] - x["correct"]).mean()), 4),
                                               round(float(np.abs(x["cal"] - x["correct"]).mean()), 4)]
    final = fit(d, a.min_n)
    final.save(a.out)
    report["fitted_groups"] = {g: m["n"] for g, m in final.groups.items()}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
