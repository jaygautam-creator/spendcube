"""Supervised-classifier evaluations (real data, not CI).

run_a2_sim(config): hold out 18% of ERP line ids in short runs (mimicking the A2 gaps), train on the rest.
run_b_seeds(config): 5-fold CV on described seed lines.
Both report L1/L2/L3 accuracy count- and spend-weighted, feature ablations, lookup and calibration bins.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..classify.lookup import build_lookup
from ..classify.supervised import SupervisedModel
from ..classify.units import build_units
from ..config.loader import load_config
from ..context import RunContext
from ..ingest.canonical import ingest_all
from ..labels import load_labels
from ..normalize.stage import normalize_stage
from ..taxonomy.tree import Taxonomy

log = logging.getLogger("spendcube.eval")


def _prepare(config_path: str, source: Optional[str] = None):
    cfg = load_config(config_path)
    tax = Taxonomy.load(cfg.taxonomy)
    ctx = RunContext(cfg=cfg, run_dir=Path(tempfile.mkdtemp()), taxonomy=tax)
    df = ingest_all(cfg)
    df = df[df["source_role"] == "primary"] if source is None else df[df["source"] == source]
    df = normalize_stage(ctx, df.copy()).reset_index(drop=True)
    lab = load_labels(ctx)
    keep = ["record_id", "leaf_id"] + [c for c in ("include",) if lab is not None and c in lab.columns]
    df = df.merge(lab[keep].rename(columns={"leaf_id": "y"}), on="record_id", how="left")
    df["y"] = df["y"].fillna("")
    df["spend_abs"] = pd.to_numeric(df["spend_rc"], errors="coerce").abs().fillna(0.0)
    return cfg, tax, ctx, df


def _level_ok(tax: Taxonomy, truth: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    cache: dict = {}

    def p(l):
        if l not in cache:
            cache[l] = tuple(tax.leaf(l).path[:k]) if (l and tax.has(l)) else ("?",)
        return cache[l]
    return np.array([p(t) == p(q) for t, q in zip(truth, pred)])


def _metrics(tax, truth, pred, w) -> dict:
    out = {"n": int(len(truth))}
    for k in (1, 2, 3):
        ok = _level_ok(tax, truth, pred, k)
        out[f"L{k}"] = round(float(ok.mean()), 4) if len(ok) else None
        out[f"L{k}_sw"] = round(float((ok * w).sum() / w.sum()), 4) if len(ok) and w.sum() > 0 else None
    return out


def _bins(p: np.ndarray, correct: np.ndarray) -> list[dict]:
    edges = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append({"bin": f"[{lo:.2f},{min(hi, 1):.2f})", "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 3),
                         "acc": round(float(correct[m].mean()), 3)})
    return rows


def _unit_level(df: pd.DataFrame, pred: np.ndarray, p: np.ndarray, tax) -> dict:
    d = df.copy()
    d["pred"] = pred
    d["p"] = p
    units, uid = build_units(d, "spend_abs")
    d["uid"] = uid.to_numpy()
    first = d.drop_duplicates("uid")
    truth = d.groupby("uid")["y"].agg(lambda s: s.value_counts().index[0])
    spend = d.groupby("uid")["spend_abs"].sum()
    f = first.set_index("uid")
    ids = f.index
    return _metrics(tax, truth.loc[ids].to_numpy(), f["pred"].to_numpy(), spend.loc[ids].to_numpy())


def _gap_run_holdout(order_ids: pd.Series, frac: float = 0.18, seed: int = 7) -> np.ndarray:
    """Hold out short runs of consecutive ids (1: 65%, 2: 17%, 3: 12%, 4: 6%) until ~frac of rows are held out."""
    rng = np.random.RandomState(seed)
    srt = np.argsort(order_ids.to_numpy(), kind="stable")
    n = len(srt)
    held = np.zeros(n, bool)
    i = 0
    while i < n:
        L = rng.choice([1, 2, 3, 4], p=[0.65, 0.17, 0.12, 0.06])
        if rng.rand() < frac:   # run lengths are independent of selection, so held fraction ~= frac
            held[srt[i:i + L]] = True
        i += L
    return held


def run_a2_sim(config_path: str, out_file: Optional[str] = None, seed: int = 7) -> dict:
    t0 = time.time()
    cfg, tax, ctx, df = _prepare(config_path, source="erp")
    df = df[df["y"] != ""].reset_index(drop=True)
    held = _gap_run_holdout(df["record_id"].str.replace(r"\D", "", regex=True).astype(int), 0.18, seed)
    tr, te = df[~held], df[held]
    res: dict = {"company": cfg.company, "train_rows": len(tr), "holdout_rows": len(te),
                 "holdout_frac": round(len(te) / len(df), 3)}
    variants = {"text_only": ("text",), "text+supplier": ("text", "supplier"),
                "text+supplier+gl": ("text", "supplier", "gl"),
                "all": ("text", "supplier", "gl", "ctx", "material")}
    for name, groups in variants.items():
        t1 = time.time()
        m = SupervisedModel(groups=groups).fit(tr, tr["y"].to_numpy())
        pr = m.predict(te)
        pred, p = pr["leaf_id"].to_numpy(), pr["p"].to_numpy()
        r = {"row": _metrics(tax, te["y"].to_numpy(), pred, te["spend_abs"].to_numpy()),
             "unit": _unit_level(te, pred, p, tax), "T": round(m.T, 2), "fit_s": round(time.time() - t1, 1)}
        if name == "all":
            ok3 = _level_ok(tax, te["y"].to_numpy(), pred, 3)
            r["reliability"] = _bins(p, ok3)
            r["acc_at_p>=0.8"] = {"coverage": round(float((p >= 0.8).mean()), 3),
                                  "acc": round(float(ok3[p >= 0.8].mean()), 4) if (p >= 0.8).any() else None}
        res[name] = r
    lk = build_lookup(tr, tr["y"].to_numpy())
    q = lk.query(te)
    hit = q["leaf_id"].notna().to_numpy()
    res["lookup"] = {"coverage_rows": round(float(hit.mean()), 3),
                     "acc_L3": round(float((q["leaf_id"][hit].to_numpy() == te["y"].to_numpy()[hit]).mean()), 4) if hit.any() else None,
                     "coverage_spend": round(float(te["spend_abs"].to_numpy()[hit].sum() / te["spend_abs"].sum()), 3),
                     "generic_materials": len(lk.generic)}
    res["seconds"] = round(time.time() - t0, 1)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(res, indent=2))
    return res


def run_b_seeds(config_path: str, out_file: Optional[str] = None, folds: int = 5, seed: int = 11) -> dict:
    t0 = time.time()
    cfg, tax, ctx, df = _prepare(config_path)
    d = df[(df["y"] != "") & (~df["desc_blank"].astype(bool))]
    if "include" in d.columns:
        d = d[d["include"].fillna("1") != "0"]
    d = d.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    fold = rng.permutation(len(d)) % folds
    res: dict = {"company": cfg.company, "seed_rows_described": len(d), "folds": folds}
    variants = {"text_only": ("text",), "text+vendor": ("text", "supplier"), "text+GL": ("text", "gl"),
                "text+vendor+GL": ("text", "supplier", "gl"), "all": ("text", "supplier", "gl", "ctx", "material")}
    for name, groups in variants.items():
        pred = np.empty(len(d), dtype=object)
        p = np.zeros(len(d))
        for f in range(folds):
            tr, te = d[fold != f], d[fold == f]
            m = SupervisedModel(groups=groups, min_cal=10**9 if len(tr) < 600 else 300).fit(tr, tr["y"].to_numpy())
            pr = m.predict(te)
            pred[fold == f] = pr["leaf_id"].to_numpy()
            p[fold == f] = pr["p"].to_numpy()
        r = {"row": _metrics(tax, d["y"].to_numpy(), pred, d["spend_abs"].to_numpy())}
        if name == "all":
            r["reliability"] = _bins(p, _level_ok(tax, d["y"].to_numpy(), pred, 3))
        res[name] = r
    res["seconds"] = round(time.time() - t0, 1)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(res, indent=2))
    return res
