"""Confidence calibration (D-48): per-tier monotone (isotonic) map  raw confidence -> P(L3 path correct).

Fitted offline on held-out evaluations (A label holdout, B seeds, zero-label LLM samples); stored as plain numbers in
`configs/_calibration.json` (knots only, no task data). Applied once at the end of the cascade; the raw value is kept in
`cat_conf_raw`. Tiers with too few held-out samples keep the identity map. Disabled with classification.calibration.enabled=false.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("spendcube.calibration")

GROUPS = {"supervised": "supervised", "llm": "llm", "llm_sc": "llm", "llm_coherence": "llm", "memory": "llm",
          "llm_context": "blank", "ctx_vote": "blank", "ctx_model": "blank",
          "distill": "distill", "distill_supervised": "distill", "propagate_supplier": "propagate",
          "propagate_gl": "propagate", "fallback": "fallback", "offline": "fallback"}
BANDS = [0, .3, .5, .7, .85, .95, 1.01]


def group_of(method: str) -> str:
    return GROUPS.get(method, "other")


class Calibrator:
    def __init__(self, groups: dict | None = None):
        self.groups = groups or {}

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls(json.loads(p.read_text())["groups"])
        except Exception as e:      # a damaged artefact must never break a run
            log.warning("calibration file unreadable (%s); using raw confidences", e)
            return None

    def apply(self, methods: pd.Series, conf: pd.Series) -> pd.Series:
        out = conf.astype(float).copy()
        grp = methods.map(group_of)
        for g, m in self.groups.items():
            sel = (grp == g).to_numpy()
            if sel.any():
                raw = conf.to_numpy()[sel]
                v = np.interp(raw, m["x"], m["y"])
                if m.get("tiebreak"):       # keep ranking inside a flat band: + tb*(raw - band_mid), monotone in raw per band
                    mid = np.zeros_like(raw)
                    for lo, hi, mdl in m["mids"]:
                        mid[(raw >= lo) & (raw <= hi)] = mdl
                    v = np.clip(v + float(m["tiebreak"]) * (raw - mid), m.get("lo", 0.05), m.get("hi", 0.95))
                out[sel] = v
        return out.clip(0.01, 0.99)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"groups": self.groups}, indent=1))


def fit(records: pd.DataFrame, min_n: int = 60, max_knots: int = 12) -> Calibrator:
    """records: columns method, conf, correct (0/1). Isotonic per tier group with >= min_n samples."""
    from sklearn.isotonic import IsotonicRegression
    groups = {}
    g = records["method"].map(group_of)
    for name in sorted(set(g)):
        r = records[g == name]
        if len(r) < min_n or r["correct"].nunique() < 2:
            continue
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(r["conf"].astype(float), r["correct"].astype(float))
        x, y = np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_)
        if len(x) > max_knots:
            k = np.unique(np.linspace(0, len(x) - 1, max_knots).round().astype(int))
            x, y = x[k], y[k]
        groups[name] = {"x": [round(float(v), 4) for v in x], "y": [round(float(v), 4) for v in y], "n": int(len(r))}
    return Calibrator(groups)


def reliability(records: pd.DataFrame, conf_col: str = "conf") -> list[dict]:
    out = []
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        m = (records[conf_col] >= a) & (records[conf_col] < b)
        out.append({"band": f"{a}-{b}", "n": int(m.sum()), "mean_conf": round(float(records.loc[m, conf_col].mean()), 3) if m.any() else None,
                    "acc": round(float(records.loc[m, "correct"].mean()), 3) if m.any() else None})
    return out


def gather(pairs: list[tuple[str | Path, str | Path]]) -> pd.DataFrame:
    """Rows of (run, method, conf, correct, spend) from e2e holdout runs: (run folder, output csv) pairs."""
    parts = []
    for d, out_csv in ((Path(a), Path(b)) for a, b in pairs):
        fr = pd.read_parquet(d / "final_frame.parquet")
        raw = "cat_conf_raw" if "cat_conf_raw" in fr.columns else "cat_conf"
        ho = pd.read_csv(d / "heldout_labels.csv", dtype=str, keep_default_na=False)
        out = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        m = ho.merge(out, on="record_id", suffixes=("_t", "")).merge(fr[["record_id", "cat_method", raw]], on="record_id")
        m = m[m["include"] == "1"]
        cols = [c for c in ("l1", "l2", "l3", "l4") if f"{c}_t" in m.columns]
        correct = np.all(m[[f"{c}_t" for c in cols]].to_numpy() == m[cols].to_numpy(), axis=1)
        parts.append(pd.DataFrame({"run": d.name, "method": m["cat_method"].to_numpy(), "conf": pd.to_numeric(m[raw]).to_numpy(),
                                   "correct": correct.astype(int), "spend": pd.to_numeric(m["spend"]).abs().to_numpy()}))
    return pd.concat(parts, ignore_index=True)
