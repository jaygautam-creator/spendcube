"""Ordered config exclusion rules (regex / equals / prefix / in on any canonical column)."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger("spendcube.exclusions")


def rule_mask(df: pd.DataFrame, rule) -> np.ndarray:
    field = rule.field
    if field not in df.columns:
        log.warning("exclusion rule field '%s' not in frame; rule skipped", field)
        return np.zeros(len(df), bool)
    s = df[field].fillna("").astype(str)
    m = np.zeros(len(df), bool)
    if rule.regex:
        rx = re.compile(rule.regex)
        hit = {u for u in s.unique() if rx.search(u)}      # unique-ised: descriptions repeat massively
        if hit:
            m |= s.isin(hit).to_numpy()
    if rule.equals:
        eq = {str(x).strip().casefold() for x in rule.equals}
        m |= s.str.strip().str.casefold().isin(eq).to_numpy()
    if rule.in_:
        iv = {str(x).strip().casefold() for x in rule.in_}
        m |= s.str.strip().str.casefold().isin(iv).to_numpy()
    if rule.prefix:
        m |= s.str.strip().str.casefold().str.startswith(rule.prefix.strip().casefold()).to_numpy()
    return m


def apply_rules(df: pd.DataFrame, rules) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """First hit wins. Returns (code[str obj], detail[str obj], protect[bool], hits_per_rule).

    action='include' rules protect a row: later exclude rules (and the LLM spend verdict, D-17) never act on it.
    """
    n = len(df)
    code = np.array([""] * n, dtype=object)
    detail = np.array([""] * n, dtype=object)
    protect = np.zeros(n, bool)
    done = np.zeros(n, bool)
    hits: dict[int, int] = {}
    for i, r in enumerate(rules):
        try:
            m = rule_mask(df, r) & ~done
        except re.error as e:
            log.warning("bad regex in exclusion rule %d: %s", i, e)
            continue
        if not m.any():
            hits[i] = 0
            continue
        hits[i] = int(m.sum())
        done |= m
        if r.action == "include":
            protect |= m
        else:
            code[m] = r.code
            detail[m] = f"rule#{i}:{r.field}"
    return code, detail, protect, hits
