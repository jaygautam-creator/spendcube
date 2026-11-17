"""Supplier-rich zero-label experiment (architect task 5): the top-N suppliers by spend, ALL their units, labels hidden.

Variants (spend-weighted L1/L2/L3 vs the label truth):
  single      first-opinion LLM answers only
  cascade     + second opinion on every unit + arbiter on disagreements (what the pipeline does for the top units)
  coherence   cascade + supplier-coherence re-ask (classify/coherence.py)
Reports which entries answered, request counts and the router report. Dev key only (the router mode is `dev`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..classify.coherence import run_coherence
from .blank_eval import _unit_truth
from .supervised_eval import _prepare


def _sw(tax, truth, pred, w) -> dict:
    out = {}
    for k in (1, 2, 3):
        hit = np.array([bool(p) and tax.has(p) and bool(t) and tax.has(t) and
                        tax.leaf(p).path[:k] == tax.leaf(t).path[:k] for p, t in zip(pred, truth)])
        out[f"L{k}_sw"] = round(float((hit * w).sum() / w.sum()), 4)
        out[f"L{k}_count"] = round(float(hit.mean()), 4)
    return out


def run_coherence_eval(config_path: str, top_suppliers: int = 50, max_units: int = 1200, out_file: Optional[str] = None,
                       coherence_cfg: Optional[dict] = None, source: Optional[str] = None) -> dict:
    t0 = time.time()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass
    from ..classify.cascade import _make_router
    from ..classify.llm import LLMClassifier
    from ..classify.units import build_units
    cfg, tax, ctx, df = _prepare(config_path, source=source)
    df = df[(df["y"] != "") & (~df["desc_blank"])].reset_index(drop=True)
    units, uid = build_units(df, "spend_rc")
    truth = units["uid"].map(_unit_truth(df, uid)).to_numpy()
    sk = units["supplier_core"].astype(str)
    sup_spend = units.assign(k=sk).query("k != ''").groupby("k")["spend"].sum().sort_values(ascending=False)
    pick = set(sup_spend.head(top_suppliers).index)
    sel = units.index[sk.isin(pick)].to_numpy()
    sel = sel[np.argsort(-units["spend"].to_numpy()[sel])][:max_units]
    sub = units.iloc[sel].reset_index(drop=True)
    tr = truth[sel]
    w = sub["spend"].to_numpy()
    router = _make_router(ctx, budget=False)
    if router is None or not router.has_llm("bulk"):
        return {"error": "no LLM available"}
    clf = LLMClassifier(router, tax, cfg.industry_hint, {}, (cfg.classification or {}).get("llm", {}) or {})
    idx1 = {l.id: i for i, l in enumerate(tax.leaves, 1)}
    items = [{"uid": i, "supplier": sub["supplier"].iat[i], "gl": sub["gl"].iat[i], "ctx": sub["ctx"].iat[i],
              "text": sub["text"].iat[i]} for i in range(len(sub))]
    res: dict = {"company": cfg.company, "suppliers": len(pick), "units": len(sub), "spend_covered_of_all_units":
                 round(float(w.sum() / units["spend"].sum()), 3)}
    r1 = clf.classify(items, "top_spend", workers=3)
    used = tuple({v.entry for v in r1.values()})
    r2 = clf.classify([it for it in items if it["uid"] in r1], "second_opinion", exclude_entries=used, workers=3)
    dis = [dict(it, cand=(idx1[r1[it["uid"]].leaf_id], idx1[r2[it["uid"]].leaf_id])) for it in items
           if it["uid"] in r1 and it["uid"] in r2 and r1[it["uid"]].leaf_id != r2[it["uid"]].leaf_id]
    r3 = clf.classify(dis, "arbiter", workers=2) if dis else {}
    res["answered_by"] = {"first": _count(r1), "second": _count(r2), "arbiter": _count(r3)}
    res["disagreements"] = len(dis)

    U = len(sub)
    leaf = np.array([r1[i].leaf_id if i in r1 else "" for i in range(U)], dtype=object)
    res["single"] = _sw(tax, tr, leaf, w)
    leaf_c = leaf.copy()
    conf = np.zeros(U); meth = np.full(U, "", dtype=object); selfc = np.zeros(U)
    for i in range(U):
        if i not in r1:
            continue
        v, agree = r1[i], 0.7
        if i in r2:
            if r2[i].leaf_id == v.leaf_id:
                agree = 1.0
            elif i in r3:
                v, agree = r3[i], 0.4
            else:
                agree = 0.3
        leaf_c[i], selfc[i] = v.leaf_id, v.conf
        conf[i] = float(np.clip(0.5 * v.conf + 0.5 * agree, 0.05, 0.99)); meth[i] = "llm_sc" if i in r2 else "llm"
    res["cascade"] = _sw(tax, tr, leaf_c, w)
    res["arbiter_effect_on_disagreed"] = {
        "first_opinion": _sw(tax, tr[[u["uid"] for u in dis]], leaf[[u["uid"] for u in dis]], w[[u["uid"] for u in dis]]) if dis else None,
        "arbitrated": _sw(tax, tr[[u["uid"] for u in dis]], leaf_c[[u["uid"] for u in dis]], w[[u["uid"] for u in dis]]) if dis else None}
    leaf_h, conf_h, meth_h, selfc_h = leaf_c.copy(), conf.copy(), meth.copy(), selfc.copy()
    stats: dict = {}
    cc = {"enabled": True, **(coherence_cfg or {})}
    lab = np.array([m != "" for m in meth_h])
    changed = run_coherence(sub, leaf_h, conf_h, meth_h, selfc_h, lab, clf, tax, cc, stats)
    res["coherence"] = _sw(tax, tr, leaf_h, w)
    res["coherence_stats"] = stats
    res["coherence_cfg"] = cc
    res["router"] = router.report()
    res["seconds"] = round(time.time() - t0, 1)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(res, indent=2, default=str))
    return res


def _count(r) -> dict:
    out: dict = {}
    for v in r.values():
        out[v.entry] = out.get(v.entry, 0) + 1
    return out
