"""End-to-end holdout evaluation: hide labels of an id-gap-run holdout, run the FULL pipeline, compare the output
categories to the hidden labels (count- and spend-weighted), by classification method. LLM on/off comparable."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.loader import load_config
from .. import pipeline as _pl
from ..pipeline import run_pipeline

log = logging.getLogger("spendcube.eval")


def run_e2e(config_path: str, frac: float = 0.18, no_llm: bool = False, llm_mode: str = "dev",
            out_dir: str = "runs/e2e_out", seed: int = 7, overrides: dict | None = None) -> dict:
    cfg = load_config(config_path, overrides=overrides) if overrides else load_config(config_path)
    cfg.labels.holdout_frac = frac
    cfg.labels.holdout_seed = seed
    path, errors = run_pipeline(cfg, out_dir=Path(out_dir), no_llm=no_llm, llm_mode=llm_mode, cfg_path=config_path)
    rd = _pl.LAST_RUN_DIR
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    ho = pd.read_csv(rd / "heldout_labels.csv", dtype=str, keep_default_na=False)
    fr = pd.read_parquet(rd / "final_frame.parquet")[["record_id", "cat_method", "cat_conf"]]
    m = ho.merge(out, on="record_id", suffixes=("_t", "")).merge(fr, on="record_id")
    m["spend"] = pd.to_numeric(m["spend"])
    m = m[m["include"] == "1"]
    res = {"company": cfg.company, "no_llm": no_llm, "heldout_rows_included": len(m), "validation_errors": errors}
    w = m["spend"].abs().to_numpy()

    def acc(sub: pd.DataFrame) -> dict:
        d = {}
        ww = sub["spend"].abs().to_numpy()
        for k in (1, 2, 3):
            cols_t = [f"l{i}_t" for i in range(1, k + 1)]
            cols_p = [f"l{i}" for i in range(1, k + 1)]
            hit = np.all(sub[cols_t].to_numpy() == sub[cols_p].to_numpy(), axis=1)
            d[f"L{k}"] = round(float(hit.mean()), 4) if len(sub) else None
            d[f"L{k}_sw"] = round(float((hit * ww).sum() / ww.sum()), 4) if len(sub) and ww.sum() > 0 else None
        d["n"] = len(sub)
        d["spend_share"] = round(float(ww.sum() / max(w.sum(), 1e-9)), 4)
        return d
    res["all"] = acc(m)
    res["by_method"] = {meth: acc(g) for meth, g in m.groupby("cat_method")}
    p = pd.to_numeric(m["cat_conf"], errors="coerce").fillna(0)
    hit3 = (m[["l1_t", "l2_t", "l3_t"]].to_numpy() == m[["l1", "l2", "l3"]].to_numpy()).all(axis=1)
    bins = [0, .3, .5, .7, .85, .95, 1.01]
    res["reliability_L3"] = [{"conf": f"{a}-{b}", "n": int(((p >= a) & (p < b)).sum()),
                              "acc": round(float(hit3[(p >= a) & (p < b)].mean()), 3) if ((p >= a) & (p < b)).any() else None}
                             for a, b in zip(bins[:-1], bins[1:])]
    return res
