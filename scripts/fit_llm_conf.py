"""Conservative LLM-tier confidence mapping chosen by MINIMAX across companies (item 4).

Data: per-unit (conf, correct) for the LLM tier from held-out evaluations, one group per company:
  A = zero-label A units; B = zero-label B units + the B seed-holdout LLM-answered rows.
Families (grid searched): `cap` (conf' = min(conf, c)) and `bands` (three-piece constant: below t1 -> v0, [t1,t2) -> v1, >= t2 -> v2).
Objective: minimise the WORST company ECE; ship only if it beats raw confidence on BOTH companies.
Writes numbers only to stdout (and optionally --out json). usage:
  python scripts/fit_llm_conf.py runs/eval_zero_label_cal/zero_label_A_nodefs_units.csv runs/eval_zero_label_cal/zero_label_B_nodefs_units.csv \
      runs/20261117_210708_B:runs/demand_Bbase/jaygautam__B.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.classify.calibration import gather  # noqa: E402


def ece_groups(conf: np.ndarray, correct: np.ndarray, groups: np.ndarray) -> float:
    n, tot = len(conf), 0.0
    for g in np.unique(groups):
        m = groups == g
        tot += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def raw_ece(conf, correct) -> float:
    bands = np.digitize(conf, [0.3, 0.5, 0.7, 0.85, 0.95])
    return ece_groups(conf, correct, bands)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zl_a")
    ap.add_argument("zl_b")
    ap.add_argument("b_holdout", help="RUN_DIR:OUTPUT_CSV of the B holdout run with LLM answers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = pd.read_csv(a.zl_a)[["conf", "correct"]]
    zb = pd.read_csv(a.zl_b)[["conf", "correct"]]
    g = gather([tuple(a.b_holdout.split(":", 1))])
    g = g[g["method"].isin(["llm", "llm_sc"])][["conf", "correct"]]
    B = pd.concat([zb, g], ignore_index=True)
    data = {"A": (A["conf"].to_numpy(), A["correct"].to_numpy().astype(float)),
            "B": (B["conf"].to_numpy(), B["correct"].to_numpy().astype(float))}
    out = {"n": {k: len(v[0]) for k, v in data.items()},
           "acc": {k: round(float(v[1].mean()), 3) for k, v in data.items()},
           "mean_raw_conf": {k: round(float(v[0].mean()), 3) for k, v in data.items()},
           "raw_ece": {k: round(raw_ece(*v), 4) for k, v in data.items()}}

    def worst(fn):
        e = {}
        for k, (c, y) in data.items():
            m = fn(c)
            e[k] = ece_groups(m, y, np.round(m, 4))
        return max(e.values()), e

    best = {"cap": None, "bands": None}
    for c in np.arange(0.3, 1.0, 0.025):
        w, e = worst(lambda x, c=c: np.minimum(x, c))
        if best["cap"] is None or w < best["cap"][0]:
            best["cap"] = (w, e, {"c": round(float(c), 3)})
    vals = np.arange(0.2, 0.96, 0.05)
    for t1, t2 in itertools.combinations([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95], 2):
        for v0, v1, v2 in itertools.product(vals, vals, vals):
            if not v0 <= v1 <= v2:
                continue
            w, e = worst(lambda x, t1=t1, t2=t2, v0=v0, v1=v1, v2=v2: np.where(x < t1, v0, np.where(x < t2, v1, v2)))
            if best["bands"] is None or w < best["bands"][0]:
                best["bands"] = (w, e, {"t1": t1, "t2": t2, "v0": round(float(v0), 2), "v1": round(float(v1), 2), "v2": round(float(v2), 2)})
    # four pieces: keeps the top band distinguishable (ranking information), still minimax
    best["bands4"] = None
    vals4 = np.arange(0.2, 0.96, 0.05)
    for t1, t2, t3 in itertools.combinations([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95], 3):
        for v0, v1, v2, v3 in itertools.product(vals4, vals4, vals4, vals4):
            if not (v0 <= v1 <= v2 <= v3) or v3 - v0 < 0.3:       # require real resolution between low and high
                continue
            w, e = worst(lambda x, t1=t1, t2=t2, t3=t3, v0=v0, v1=v1, v2=v2, v3=v3:
                         np.where(x < t1, v0, np.where(x < t2, v1, np.where(x < t3, v2, v3))))
            if best["bands4"] is None or w < best["bands4"][0]:
                best["bands4"] = (w, e, {"t1": t1, "t2": t2, "t3": t3, "v0": round(float(v0), 2), "v1": round(float(v1), 2),
                                         "v2": round(float(v2), 2), "v3": round(float(v3), 2)})
    for fam, (w, e, p) in best.items():
        out[fam] = {"worst_ece": round(w, 4), "ece": {k: round(v, 4) for k, v in e.items()}, "params": p,
                    "beats_raw_on_both": all(e[k] < out["raw_ece"][k] for k in e)}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
