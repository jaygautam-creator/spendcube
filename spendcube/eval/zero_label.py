"""Zero-label rehearsal (C proxy): classify a sample of units with labels hidden, compare to labels.

Variants: with/without compiled definitions; single model vs two-model agreement (+ arbiter on disagreement).
Reports L1/L2/L3 accuracy count- and spend-weighted. Sample is persisted so reruns hit the LLM cache.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..classify.llm import LLMClassifier
from ..classify.units import build_units
from ..config.loader import load_config
from ..ingest.canonical import ingest_all
from ..llm.router import Router
from ..normalize.stage import normalize_stage
from ..taxonomy.compile import compile_taxonomy
from ..taxonomy.tree import Taxonomy

log = logging.getLogger("spendcube.eval")


def _truth_frame(cfg, df) -> pd.DataFrame:
    """Join labels (column names from cfg.labels.columns) onto df rows; returns df with l1_t,l2_t,l3_t."""
    lc = cfg.labels
    lab = pd.read_csv(next(iter(__import__("glob").glob(lc.file))), dtype=str, keep_default_na=False)
    key = lc.join_on
    if key == "record_id" and "record_id" in lab.columns:
        df = df.merge(lab, on="record_id", how="left", suffixes=("", "_lab"))
    else:  # B: seeds keyed by line_id == record_id
        lab = lab.rename(columns={key: "record_id"})
        df = df.merge(lab, on="record_id", how="left", suffixes=("", "_lab"))
    cols = lc.columns
    for i, k in enumerate(("l1", "l2", "l3")):
        df[f"{k}_t"] = df[cols[k]].fillna("") if cols[k] in df.columns else ""
    if "include" in cols and cols["include"] in df.columns:
        df["incl_t"] = df[cols["include"]].fillna("")
    return df


def select_sample(cfg, tax, top: int, strat: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = ingest_all(cfg)
    df = df[df["source_role"] == "primary"].copy()

    from ..context import RunContext
    c = RunContext(cfg=cfg, run_dir=Path("runs"), taxonomy=tax)
    df = normalize_stage(c, df)
    try:   # exclusions (IC/tax rules) so that excluded lines do not enter the sample
        from ..exclusions.stage import exclusions_stage
        df = exclusions_stage(c, df)
        df = df[df["excl_code"].fillna("") == ""].copy()
    except Exception as e:
        log.warning("exclusions skipped in zero-label sample: %s", e)
    df = _truth_frame(cfg, df)
    labelled = df["l1_t"] != ""
    if "incl_t" in df.columns:
        labelled &= df["incl_t"] == "1"
    df["spend_rc"] = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0.0)
    units, row_uid = build_units(df)
    df["uid"] = row_uid
    lab_rows = df[labelled & ~df["desc_blank"]]
    truth = lab_rows.groupby("uid").apply(
        lambda g: Counter(zip(g.l1_t, g.l2_t, g.l3_t)).most_common(1)[0][0], include_groups=False)
    cand = units[units["uid"].isin(truth.index) & units["described"]].copy()
    cand["truth"] = cand["uid"].map(truth)
    cand = cand[cand["truth"].map(lambda t: tax.leaf_by_names(list(t)) is not None)]
    top_u = cand.head(top)
    rest = cand.iloc[top:]
    rng = np.random.default_rng(seed)
    strat_u = rest.iloc[rng.permutation(len(rest))[:strat]] if len(rest) else rest
    sample = pd.concat([top_u, strat_u]).reset_index(drop=True)
    sample["is_top"] = [True] * len(top_u) + [False] * len(strat_u)
    return sample, units


def _acc(sample: pd.DataFrame, pred: dict[int, tuple], tax, subset=None) -> dict:
    rows = sample if subset is None else sample[sample["uid"].isin(subset)]
    out = {}
    for k in (1, 2, 3):
        hit = np.array([uid in pred and tuple(pred[uid][:k]) == tuple(t[:k]) for uid, t in zip(rows.uid, rows.truth)])
        w = rows.spend.to_numpy()
        out[f"L{k}"] = round(float(hit.mean()), 4) if len(rows) else None
        out[f"L{k}_sw"] = round(float((hit * w).sum() / w.sum()), 4) if len(rows) and w.sum() > 0 else None
    out["n"] = len(rows)
    return out


def run_zero_label(config_path: str, top: int = 150, strat: int = 150, out_dir: Optional[Path] = None,
                   variants: tuple = ("nodefs", "defs"), sample_file: Optional[str] = None,
                   mode: str = "dev", use_llm: bool = True) -> dict:
    cfg = load_config(config_path)
    tax = Taxonomy.load(cfg.taxonomy)
    llm_cfg = cfg.classification.get("llm", {})
    out_dir = Path(out_dir or "runs/eval_zero_label"); out_dir.mkdir(parents=True, exist_ok=True)
    if sample_file and Path(sample_file).exists():
        sample = pd.read_pickle(sample_file)
        units = None
    else:
        sample, units = select_sample(cfg, tax, top, strat)
        if sample_file:
            Path(sample_file).parent.mkdir(parents=True, exist_ok=True)
            sample.to_pickle(sample_file)
    router = Router(llm_cfg, mode=mode, enabled=use_llm)
    items = [{"uid": int(r.uid), "supplier": r.supplier, "gl": r.gl, "ctx": r.ctx, "text": r.text}
             for r in sample.itertuples()]
    results: dict = {"company": cfg.company, "n_units": len(sample), "top": top, "strat": strat,
                     "weights_note": "spend-weighted uses unit spend from the (stub) normalise stage"}
    definitions = {}
    for var in variants:
        if var == "defs" and not definitions:
            texts = list(dict.fromkeys(sample["text"].tolist()))
            definitions = compile_taxonomy(router, tax, cfg.industry_hint, texts)
            results["defs_compiled"] = len(definitions)
        clf = LLMClassifier(router, tax, cfg.industry_hint, definitions if var == "defs" else None, llm_cfg)
        r1 = clf.classify(items, "top_spend", workers=3)
        e1 = {v.entry for v in r1.values()}
        r2 = clf.classify(items, "second_opinion", exclude_entries=tuple(e1), workers=3)
        leaf_path = lambda lid: tax.leaf(lid).path
        p1 = {u: leaf_path(v.leaf_id) for u, v in r1.items()}
        p2 = {u: leaf_path(v.leaf_id) for u, v in r2.items()}
        agree = {u for u in p1 if u in p2 and p1[u] == p2[u]}
        dis = [u for u in p1 if u in p2 and p1[u] != p2[u]]
        idx1 = {l.id: i for i, l in enumerate(tax.leaves, 1)}
        arb_items = [dict(it, cand=(idx1[r1[it["uid"]].leaf_id], idx1[r2[it["uid"]].leaf_id])) for it in items if it["uid"] in dis]
        r3 = clf.classify(arb_items, "arbiter", workers=2) if arb_items else {}
        pf = dict(p1)
        for u, v in r3.items():
            pf[u] = leaf_path(v.leaf_id)
        from collections import Counter as _C
        pure1 = {u for u, v in r1.items() if v.entry == "gemini_flash_lite_35"}
        pure2 = {u for u, v in r2.items() if v.entry == "gemini_flash_lite_31"}
        results[var] = {
            "answered_by_primary": dict(_C(v.entry for v in r1.values())),
            "answered_by_second": dict(_C(v.entry for v in r2.values())),
            "answered_by_arbiter": dict(_C(v.entry for v in r3.values())),
            "primary_pure_3.5lite": _acc(sample, p1, tax, pure1),
            "second_pure_3.1lite": _acc(sample, p2, tax, pure2),
            "primary": _acc(sample, p1, tax), "second": _acc(sample, p2, tax),
            "agreed_subset": {**_acc(sample, p1, tax, agree), "coverage": round(len(agree) / max(1, len(sample)), 3)},
            "disagreed_subset_primary": _acc(sample, p1, tax, dis),
            "arbitrated_final": _acc(sample, pf, tax),
            "top_only_primary": _acc(sample[sample.is_top], p1, tax),
            "disagree_n": len(dis), "arbiter_resolved": len(r3),
        }
        # Preserve the spend-rank bands used for T4 model decisions. The standard
        # zero-label run usually samples a 150-unit head plus a random tail; when
        # top=900 is used, these slices make the head/tail quality visible.
        results[var]["by_rank"] = {
            label: {
                "primary_3.5lite": _acc(sample.iloc[start:stop], p1, tax),
                "second_3.1lite": _acc(sample.iloc[start:stop], p2, tax),
            }
            for label, start, stop in (("top150", 0, 150), ("150-300", 150, 300), ("300-900", 300, 900))
            if start < len(sample)
        }
        # per-unit records for confidence calibration (same confidence formula as the cascade)
        recs = []
        tmap = dict(zip(sample.uid, sample.truth))
        smap = dict(zip(sample.uid, sample.spend))
        tx = {r.uid: (r.supplier, r.gl, r.ctx, r.text) for r in sample.itertuples()}
        for u, v in r1.items():
            fin, agree = v, 0.7
            if u in r2:
                if p2[u] == p1[u]:
                    agree = 1.0
                elif u in r3:
                    fin, agree = r3[u], 0.4
                else:
                    agree = 0.3
            c = float(np.clip(0.5 * fin.conf + 0.5 * agree, 0.05, 0.99))
            recs.append({"method": "llm_sc" if u in r2 else "llm", "conf": round(c, 4),
                         "correct": int(tuple(pf[u][:3]) == tuple(tmap[u][:3])), "spend": float(smap[u]), "run": f"zl_{cfg.company}_{var}",
                         "uid": u, "entry": fin.entry, "pred": " > ".join(pf[u]), "truth": " > ".join(tmap[u]),
                         "supplier": tx[u][0], "gl": tx[u][1], "ctx": tx[u][2], "text": tx[u][3]})   # local analysis only (runs/ is git-ignored)
        pd.DataFrame(recs).to_csv(out_dir / f"zero_label_{cfg.company}_{var}_units.csv", index=False)
    results["router"] = router.report()
    (out_dir / f"zero_label_{cfg.company}.json").write_text(json.dumps(results, indent=2, default=str))
    return results
