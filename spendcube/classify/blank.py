"""Blank-description units: resolve them from context alone, learned from the run's own labelled described units.

Two generic, text-free mechanisms (no company-specific strings):
  * key voting: for each context key (vendor/supplier, GL account, supplier x GL, ...) the spend-weighted leaf
    distribution of known units; a blank unit takes the most specific key that has a confident majority.
  * context model: a supervised model on categorical tokens only (supplier, vendor, GL, context columns, material),
    trained on the known units, so it never sees (absent) description text.
The two are combined by confidence. `known` units are those whose leaf came from labels or the LLM.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

CTX_GROUPS = ("supplier", "gl", "ctx", "material")
DEFAULTS = {"enabled": True, "mode": "vote_first", "vote_min_share": 0.6, "vote_min_spend": 0.0, "cap": 0.6,
            "min_known": 30}


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper()


def vote_keys(units: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Most specific first. Blank key values never vote."""
    def c(n):
        return _norm(units[n]) if n in units.columns else pd.Series("", index=units.index)
    sup = c("supplier_core").where(c("supplier_core") != "", c("supplier_raw"))
    ven, gl = c("vendor_id"), c("gl_account")
    ctxs = c("llm_context")
    both = (sup + "|" + gl).where((sup != "") & (gl != ""), "")
    sup_ctx = (sup + "|" + ctxs).where((sup != "") & (ctxs != ""), "")
    ven_gl = (ven + "|" + gl).where((ven != "") & (gl != ""), "")
    return [("sup_gl", both), ("ven_gl", ven_gl), ("sup_ctx", sup_ctx), ("supplier", sup), ("vendor", ven), ("gl", gl)]


def vote_predict(units: pd.DataFrame, known_units: pd.DataFrame, known_leaf: np.ndarray, weight: np.ndarray,
                 min_share: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Per unit: (leaf, share) from the most specific key whose known units have a >=min_share majority."""
    n = len(units)
    leaf = np.full(n, "", dtype=object)
    share = np.zeros(n)
    if len(known_units) == 0:
        return leaf, share
    for (_, k), (_, kk) in zip(vote_keys(units), vote_keys(known_units)):
        kv, kkv = k.to_numpy(), kk.to_numpy()
        df = pd.DataFrame({"k": kkv, "l": known_leaf, "w": weight})
        df = df[df["k"] != ""]
        if df.empty:
            continue
        g = df.groupby(["k", "l"])["w"].sum().reset_index()
        g["s"] = g["w"] / g.groupby("k")["w"].transform("sum")
        top = g.sort_values("s", ascending=False).drop_duplicates("k").set_index("k")
        cand = pd.Series(kv).map(top["l"]).to_numpy()
        sh = pd.Series(kv).map(top["s"]).fillna(0.0).to_numpy()
        take = (leaf == "") & (sh >= min_share) & (kv != "")
        leaf[take] = cand[take]
        share[take] = sh[take]
    return leaf, share


def _vote_weight(spend: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(spend, 0.0)) + 1.0


def context_model_predict(train_units: pd.DataFrame, train_leaf, train_w, units: pd.DataFrame) -> pd.DataFrame:
    from .supervised import SupervisedModel
    m = SupervisedModel(groups=CTX_GROUPS).fit(train_units, list(train_leaf), weights=train_w)
    return m.predict(units)


def predict_blank(units: pd.DataFrame, known: np.ndarray, known_leaf: np.ndarray, spend: np.ndarray,
                  bcfg: Optional[dict] = None, extra_train: Optional[pd.DataFrame] = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (leaf, conf, method) for every unit (callers pick the ones they need). `extra_train` is an optional
    labelled unit frame with columns `leaf` and `w` (e.g. label-derived training units)."""
    cfg = {**DEFAULTS, **(bcfg or {})}
    n = len(units)
    w = _vote_weight(spend)
    mode = cfg["mode"]
    cap = float(cfg["cap"])
    leaf = np.full(n, "", dtype=object)
    conf = np.zeros(n)
    meth = np.full(n, "", dtype=object)

    parts = [units[known].assign(leaf=known_leaf[known], w=w[known])]
    if extra_train is not None and len(extra_train):
        parts.append(extra_train[[c for c in parts[0].columns if c in extra_train.columns]])
    tr = pd.concat(parts, ignore_index=True)
    vl, vs = vote_predict(units, tr, tr["leaf"].to_numpy(), tr["w"].to_numpy(), float(cfg["vote_min_share"]))
    cm = None
    if mode in ("combo", "model", "vote_first") and len(tr) >= int(cfg["min_known"]):
        try:
            cm = context_model_predict(tr, tr["leaf"], tr["w"].to_numpy(), units)
        except Exception:
            cm = None
    if mode == "vote" or cm is None:
        ok = vl != ""
        leaf[ok], conf[ok], meth[ok] = vl[ok], np.minimum(vs[ok], cap), "ctx_vote"
        return leaf, conf, meth
    ml, mp = cm["leaf_id"].to_numpy(), cm["p"].to_numpy()
    if mode == "model":
        return ml.astype(object), np.minimum(mp, cap), np.full(n, "ctx_model", dtype=object)
    if mode == "combo":   # vote wins only when its share beats the model probability
        use_vote = (vl != "") & (vs >= mp)
    else:                 # vote_first: a confident key majority wins, the context model fills the rest
        use_vote = vl != ""
    leaf[:] = np.where(use_vote, vl, ml)
    conf[:] = np.minimum(np.where(use_vote, vs, mp), cap)
    meth[:] = np.where(use_vote, "ctx_vote", "ctx_model")
    return leaf, conf, meth


def resolve_blank(units: pd.DataFrame, leaf: np.ndarray, conf: np.ndarray, meth: np.ndarray, spend: np.ndarray,
                  described: np.ndarray, bcfg: Optional[dict], stats: dict,
                  extra_train: Optional[pd.DataFrame] = None) -> int:
    """Fill unresolved units in place. Known = resolved described units. Returns the number of units filled."""
    cfg = {**DEFAULTS, **(bcfg or {})}
    rest = leaf == ""
    if not cfg["enabled"] or not rest.any():
        return 0
    known = (~rest) & described
    if known.sum() == 0 and (extra_train is None or len(extra_train) == 0):
        return 0
    kl, kc, km = predict_blank(units, known, leaf.copy(), spend, cfg, extra_train)
    take = rest & (kl != "")
    leaf[take], conf[take], meth[take] = kl[take], kc[take], km[take]
    stats["blank_context_units"] = int(take.sum())
    stats["blank_context_spend_share"] = round(float(spend[take].sum() / max(spend.sum(), 1e-9)), 4)
    return int(take.sum())
