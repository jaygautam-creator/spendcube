"""Learned exclusion rules from labels that carry include/exclusion_reason (e.g. B seeds).

For categorical fields, a value with support >= min_support whose labelled rows are >= min_purity one exclusion
code becomes a rule (applies to all rows with that value). Ranked below config rules; written to ctx.notes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("spendcube.exclusions")

DEFAULT_FIELDS = ["gl_desc", "gl_account", "vendor_id"]


def learn_rules(df: pd.DataFrame, labels: pd.DataFrame, fields: list[str], min_support: int = 5,
                min_purity: float = 0.95) -> list[dict]:
    """labels: record_id, include ('1'/'0'), exclusion_reason. Returns rule dicts."""
    if labels is None or labels.empty or "include" not in labels.columns:
        return []
    lab = labels[["record_id", "include"] + (["exclusion_reason"] if "exclusion_reason" in labels.columns else [])].copy()
    if "exclusion_reason" not in lab.columns:
        lab["exclusion_reason"] = ""
    lab["cls"] = np.where(lab["include"].astype(str) == "1", "", lab["exclusion_reason"].fillna("").astype(str))
    base = df[df["source_role"].isin(("primary", "secondary"))]
    j = base.merge(lab[["record_id", "cls"]], on="record_id", how="inner")
    rules: list[dict] = []
    for f in fields:
        if f not in j.columns:
            continue
        g = j[j[f].fillna("").astype(str).str.strip() != ""]
        for val, sub in g.groupby(f):
            sup = len(sub)
            if sup < min_support:
                continue
            top = sub["cls"].value_counts()
            code, cnt = top.index[0], int(top.iloc[0])
            if code == "":
                continue
            purity = cnt / sup
            if purity >= min_purity:
                rules.append({"field": f, "value": str(val), "code": code, "support": sup,
                              "purity": round(purity, 3)})
    return rules


def apply_learned(df: pd.DataFrame, rules: list[dict]):
    n = len(df)
    code = np.array([""] * n, dtype=object)
    detail = np.array([""] * n, dtype=object)
    conf = np.zeros(n)
    by_field: dict[str, list[dict]] = {}
    for r in rules:
        by_field.setdefault(r["field"], []).append(r)
    for f, rs in by_field.items():
        s = df[f].fillna("").astype(str)
        for r in sorted(rs, key=lambda r: r["purity"]):          # best purity written last wins
            m = (s == r["value"]).to_numpy()
            code[m] = r["code"]
            detail[m] = f"learned:{f}={r['value']}"
            conf[m] = min(0.9, r["purity"] * 0.9)
    return code, detail, conf
