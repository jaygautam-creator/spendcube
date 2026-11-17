"""Feature text builders for the supervised classifier.

Every frame column is optional (missing columns are treated as blank). Categorical identifiers become
tokens `<NAME>_<VALUE>`; every `ctx__<name>` column becomes a generic token so a new company's context
columns need no code. Feature groups can be switched off for ablations.
"""
from __future__ import annotations

import re

import pandas as pd

GROUPS = ("text", "supplier", "gl", "ctx", "material")
_NON = re.compile(r"[^0-9A-Z]+")


def col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def _tok(prefix: str, s: pd.Series) -> pd.Series:
    """Series of `PREFIX_VALUE` tokens ('' where the value is blank). Computed over uniques for speed."""
    u = s.unique()
    m = {}
    for v in u:
        c = _NON.sub("", str(v).upper())
        m[v] = f"{prefix}_{c}" if c else ""
    return s.map(m)


def text_series(df: pd.DataFrame) -> pd.Series:
    t = col(df, "desc_norm")
    if (t == "").all():
        t = col(df, "desc_raw").str.upper()
    return t


def _ctx_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ctx__")]


def _kv_tokens(items: list[str]) -> str:
    out = []
    for i in items:
        k, _, v = i.partition("=")
        v = _NON.sub("", v.upper())
        if k.strip() and v:
            out.append(f"{k.strip().upper()}_{v}")
    return " ".join(out)


def cat_series(df: pd.DataFrame, groups: tuple[str, ...] = GROUPS) -> pd.Series:
    """Space-joined categorical tokens per row."""
    parts: list[pd.Series] = []
    if "supplier" in groups:
        sup = col(df, "supplier_core")
        raw = col(df, "supplier_raw").str.upper().str.replace(_NON, "", regex=True)
        parts += [_tok("SUP", sup.where(sup != "", raw)), _tok("VEN", col(df, "vendor_id"))]
    if "gl" in groups:
        parts += [_tok("GL", col(df, "gl_account")), _tok("GLD", col(df, "gl_desc"))]
    if "material" in groups:
        parts += [_tok("MAT", col(df, "material_id"))]
    if "ctx" in groups:
        parts += [_tok("CC", col(df, "cost_centre"))]
        cc = _ctx_columns(df)
        if cc:
            for c in cc:
                parts.append(_tok(c[len("ctx__"):].upper(), df[c].fillna("").astype(str)))
        else:  # distillation frames may only carry the compact llm_context "k=v;k=v"
            lc = col(df, "llm_context")
            if (lc != "").any():
                parts.append(lc.str.split(";").map(_kv_tokens))
    if not parts:
        return pd.Series("", index=df.index, dtype=object)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    return out.str.replace(r"\s+", " ", regex=True).str.strip()
