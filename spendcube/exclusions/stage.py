"""Stage 3: status map, ordered rules, learned rules, vendor propagation, precedence (D-03).

Adds: status_class, excl_code, excl_detail, excl_conf, rule_include (bool: row protected by an `include`
rule; never excluded by the LLM verdict, D-17). DUPLICATE (reconcile, later) is never assigned here and an
existing DUPLICATE is only replaced by a higher-precedence code.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..schema_cols import EXCLUSION_PRECEDENCE
from .learned import DEFAULT_FIELDS, apply_learned, learn_rules
from .propagation import propagate_vendor
from .rules import apply_rules
from .status import classify_status

log = logging.getLogger("spendcube.exclusions")


def _cfg_extra(cfg, key, default):
    v = getattr(cfg, key, None)
    if v is None and getattr(cfg, "model_extra", None):
        v = cfg.model_extra.get(key)
    return default if v is None else v


def _rank(code: str) -> int:
    return EXCLUSION_PRECEDENCE.index(code) if code in EXCLUSION_PRECEDENCE else 99


def exclusions_stage(ctx, df: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.cfg
    n = len(df)
    # ---- 1. status ----
    status = pd.Series("INCLUDE", index=df.index, dtype=object)
    unknown_all: dict[str, dict] = {}
    for src in cfg.sources:
        if src.status_map is None or not src.columns.get("status"):
            continue
        m = df["source"] == src.name
        cls, unknown = classify_status(df.loc[m, "status_raw"], src.status_map)
        status[m] = cls
        if unknown:
            unknown_all[src.name] = unknown
            log.warning("source %s: unknown status values -> default %s: %s", src.name,
                        src.status_map.default, unknown)
    df["status_class"] = status
    if unknown_all:
        ctx.note("exclusions", "unknown_status_values", unknown_all)

    # ---- 2. config rules ----
    r_code, r_detail, protect, hits = apply_rules(df, cfg.exclusion_rules)
    ctx.note("exclusions", "config_rule_hits", hits)
    r_conf = np.where(r_code != "", 0.95, 0.0)

    # ---- 3. learned rules (below config rules) ----
    lcfg = _cfg_extra(cfg, "learned_rules", {})
    learned_on = bool(lcfg.get("enabled", True)) and not ctx.state.get("disable_learned", False)
    learned: list[dict] = []
    if learned_on and cfg.labels is not None:
        from ..labels import load_labels
        labels = load_labels(ctx)
        if labels is not None and "include" in labels.columns:
            # Context columns (plant, company, system...) are NOT learned by default: seed sampling is not
            # uniform, so such rules look pure but over-exclude. List them explicitly in config to opt in.
            fields = lcfg.get("fields") or DEFAULT_FIELDS
            learned = learn_rules(df, labels, fields, int(lcfg.get("min_support", 5)),
                                  float(lcfg.get("min_purity", 0.95)))
    if learned:
        l_code, l_detail, l_conf = apply_learned(df, learned)
        take = (r_code == "") & (l_code != "") & ~protect
        r_code = np.where(take, l_code, r_code)
        r_detail = np.where(take, l_detail, r_detail)
        r_conf = np.where(take, l_conf, r_conf)
        ctx.note("exclusions", "learned_rules", f"{len(learned)} rules: " +
                 "; ".join(f"{r['field']}={r['value']}->{r['code']} (n={r['support']},p={r['purity']})"
                           for r in learned[:40]))
    ctx.state["learned_rules"] = learned

    # ---- 4. vendor propagation ----
    vp = cfg.vendor_propagation
    if vp.enabled:
        for code in vp.codes:
            cand = propagate_vendor(df, np.asarray(r_code, dtype=object), protect, code, vp.min_share, vp.min_rows)
            cand &= (r_code == "")
            if cand.any():
                r_code = np.where(cand, code, r_code)
                r_detail = np.where(cand, "vendor_propagation", r_detail)
                r_conf = np.where(cand, 0.85, r_conf)
                ctx.note("exclusions", f"vendor_propagation_{code}", int(cand.sum()))

    # ---- 5. precedence: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE ----
    prev = df["excl_code"].fillna("").astype(str).to_numpy() if "excl_code" in df.columns else np.array([""] * n, dtype=object)
    prev_detail = df["excl_detail"].fillna("").astype(str).to_numpy() if "excl_detail" in df.columns else np.array([""] * n, dtype=object)
    prev_conf = pd.to_numeric(df["excl_conf"], errors="coerce").fillna(1.0).to_numpy() if "excl_conf" in df.columns else np.ones(n)
    st = status.to_numpy(dtype=object)
    sraw = df["status_raw"].fillna("").astype(str).to_numpy()
    code_out = np.array([""] * n, dtype=object)
    detail_out = np.array([""] * n, dtype=object)
    conf_out = np.ones(n)
    for i in range(n):
        cands = []
        if st[i] in ("CANCELLED", "NOT_APPROVED"):
            cands.append((st[i], f"status:{sraw[i]}", 0.99))
        if r_code[i]:
            cands.append((r_code[i], r_detail[i], float(r_conf[i])))
        if prev[i]:
            cands.append((prev[i], prev_detail[i], float(prev_conf[i])))
        if cands:
            best = min(cands, key=lambda c: _rank(c[0]))
            code_out[i], detail_out[i], conf_out[i] = best
    df["excl_code"] = code_out
    df["excl_detail"] = detail_out
    df["excl_conf"] = conf_out
    df["rule_include"] = protect
    summ = pd.Series(code_out[code_out != ""]).value_counts().to_dict()
    ctx.note("exclusions", "counts_after_stage3", summ)
    return df
