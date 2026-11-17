"""Blank-description accuracy: blank the text of labelled described lines, keep all other context, classify.

Mode "label": the model/votes are trained on the other (described, labelled) rows of an id-gap holdout.
Mode "llm":   C-like, no labels: the known units are LLM-labelled described units (cached answers reused;
              new requests go through the DEV key, capped by `max_llm_requests`).
Reports L1/L2/L3 count- and spend-weighted for: the old tier (text-trained model fed blank text), key voting,
context-only model, and the combination.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..classify.blank import predict_blank
from ..classify.supervised import SupervisedModel
from ..classify.units import build_units
from .supervised_eval import _gap_run_holdout, _metrics, _prepare


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("desc_raw", "desc_norm", "desc_compact"):
        if c in d.columns:
            d[c] = ""
    d["desc_blank"] = True
    return d


def _unit_truth(df: pd.DataFrame, uid: pd.Series) -> pd.Series:
    g = df.assign(uid=uid.to_numpy()).groupby(["uid", "y"])["spend_abs"].sum().reset_index()
    return g.sort_values("spend_abs", ascending=False).drop_duplicates("uid").set_index("uid")["y"]


def _score(tax, truth, pred, w, method=None, extra=None) -> dict:
    r = {"all": _metrics(tax, truth, pred, w)}
    if method is not None:
        r["by_method"] = {m: _metrics(tax, truth[method == m], pred[method == m], w[method == m])
                          for m in sorted(set(method)) if m}
    return r


def run_blank_eval(config_path: str, out_file: Optional[str] = None, mode: str = "both", seed: int = 7,
                   max_llm_units: int = 900, max_llm_requests: int = 25, variants: Optional[dict] = None,
                   source: Optional[str] = None) -> dict:
    t0 = time.time()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass
    cfg, tax, ctx, df = _prepare(config_path, source=source)
    df = df[(df["y"] != "") & (~df["desc_blank"])].reset_index(drop=True)
    ids = df["record_id"].str.replace(r"\D", "", regex=True)
    ids = pd.to_numeric(ids, errors="coerce").fillna(pd.Series(np.arange(len(df)), index=df.index)).astype(int)
    held = _gap_run_holdout(ids, 0.18, seed)
    tr, te = df[~held].reset_index(drop=True), _blank(df[held].reset_index(drop=True))
    tu, tuid = build_units(tr, "spend_rc")
    eu, euid = build_units(te, "spend_rc")
    ttruth = _unit_truth(tr, tuid)
    etruth = _unit_truth(te, euid)
    tu["leaf"] = tu["uid"].map(ttruth)
    truth = eu["uid"].map(etruth).to_numpy()
    spend_e = eu["spend"].to_numpy()
    res: dict = {"company": cfg.company, "train_units": len(tu), "blank_units": len(eu),
                 "blank_rows": len(te), "blank_spend_share_of_df": None}
    variants = variants or {
        "vote": {"mode": "vote"}, "model": {"mode": "model"}, "combo": {"mode": "combo"},
        "vote_first": {"mode": "vote_first"}, "vote_first_s50": {"mode": "vote_first", "vote_min_share": 0.5},
        "vote_first_s80": {"mode": "vote_first", "vote_min_share": 0.8}}

    def run_variants(name: str, known_units: pd.DataFrame, known_leaf: np.ndarray, known_spend: np.ndarray):
        allu = pd.concat([known_units, eu], ignore_index=True)
        known = np.r_[np.ones(len(known_units), bool), np.zeros(len(eu), bool)]
        kl = np.r_[known_leaf, np.full(len(eu), "", dtype=object)]
        sp = np.r_[known_spend, spend_e]
        out = {}
        for vn, vc in variants.items():
            l, c, m = predict_blank(allu, known, kl, sp, vc)
            l, c, m = l[len(known_units):], c[len(known_units):], m[len(known_units):]
            miss = l == ""
            out[vn] = {**_score(tax, truth, l, spend_e, np.where(miss, "none", m)), "coverage": round(float((~miss).mean()), 3)}
        return out

    if mode in ("both", "label"):
        # old behaviour: text-trained supervised model fed blank text (the prior blank tier)
        m = SupervisedModel().fit(tu, tu["leaf"], weights=np.sqrt(tu["rows"].astype(float)))
        pr = m.predict(eu)
        res["label_mode"] = {"old_supervised_top1": _score(tax, truth, pr["leaf_id"].to_numpy(), spend_e)}
        res["label_mode"].update(run_variants("label", tu, tu["leaf"].to_numpy(), tu["spend"].to_numpy()))
    if mode in ("both", "llm"):
        from ..classify.cascade import _make_router
        from ..classify.llm import LLMClassifier
        router = _make_router(ctx, budget=False)
        if router is None or not router.has_llm("bulk"):
            res["llm_mode"] = {"error": "no LLM available"}
        else:
            top = tu.head(max_llm_units)
            clf = LLMClassifier(router, tax, cfg.industry_hint, {}, (cfg.classification or {}).get("llm", {}) or {})
            items = [{"uid": int(i), "supplier": top["supplier"].iat[i], "gl": top["gl"].iat[i],
                      "ctx": top["ctx"].iat[i], "text": top["text"].iat[i]} for i in range(len(top))]
            items = items[: max_llm_requests * clf.batch_max]
            r = clf.classify(items, "bulk", workers=3)
            idx = np.array(sorted(r))
            kl = np.array([r[i].leaf_id for i in idx], dtype=object)
            ku = top.iloc[idx].reset_index(drop=True)
            truth_known = top["leaf"].to_numpy()[idx]
            ok3 = [tax.leaf(a).path[:3] == tax.leaf(b).path[:3] if (a and tax.has(a) and b and tax.has(b)) else False
                   for a, b in zip(kl, truth_known)]
            ok3 = np.array(ok3, dtype=bool)
            ok1 = np.array([tax.leaf(a).path[:1] == tax.leaf(b).path[:1] if (a and tax.has(a) and b and tax.has(b)) else False
                            for a, b in zip(kl, truth_known)])
            w = top["spend"].to_numpy()[idx]
            ents = np.array([r[i].entry for i in idx])
            by_entry = {e: {"units": int((ents == e).sum()),
                            "L3_count": round(float(ok3[ents == e].mean()), 4),
                            "L3_sw": round(float((ok3 * w)[ents == e].sum() / w[ents == e].sum()), 4)}
                        for e in sorted(set(ents))}
            res["llm_mode"] = {"llm_labelled_units": int(len(idx)), "llm_label_L3_vs_truth": round(float(np.mean(ok3)), 4),
                               "llm_label_L3_vs_truth_sw": round(float((ok3 * w).sum() / w.sum()), 4),
                               "llm_label_L1_vs_truth_sw": round(float((ok1 * w).sum() / w.sum()), 4),
                               "llm_label_by_entry": by_entry,
                               "llm_label_by_rank": {
                                   lab: {"units": int(m.sum()), "L3_sw": round(float((ok3 * w)[m].sum() / w[m].sum()), 4),
                                         "L3_count": round(float(ok3[m].mean()), 4)}
                                   for lab, m in (("top150", idx < 150), ("150-300", (idx >= 150) & (idx < 300)),
                                                  ("300-900", idx >= 300)) if m.any()},
                               "llm_spend_covered_of_train": round(float(top["spend"].to_numpy()[idx].sum() / tu["spend"].sum()), 3)}
            res["llm_mode"].update(run_variants("llm", ku, kl, ku["spend"].to_numpy()))
            res["llm_mode"]["router"] = router.report()
    res["seconds"] = round(time.time() - t0, 1)
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(res, indent=2, default=str))
    return res
