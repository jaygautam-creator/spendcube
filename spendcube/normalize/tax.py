"""VAT / tax handling (D-04): 15, 0.15, 15%, blank -> default, 0 -> 0% (NOT blank)."""
from __future__ import annotations

import math

import pandas as pd

from .numbers import parse_number


def parse_rate(s, default: float | None) -> float:
    """Fraction (0.15). Blank -> default (NaN when no default). '%' or >=1 -> percentage (1 -> 1%)."""
    t = "" if s is None else str(s).replace(" ", "").strip()
    if t == "":
        return default if default is not None else math.nan
    pct = "%" in t
    t = t.replace("%", "").strip()
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    v = parse_number(t, ".")
    if math.isnan(v):
        return default if default is not None else math.nan
    if pct or v >= 1:
        v = v / 100.0
    return v


def parse_rate_series(raw: pd.Series, default: float | None) -> pd.Series:
    raw = raw.fillna("").astype(str)
    uniq = pd.Series(raw.unique())
    mp = {u: parse_rate(u, default) for u in uniq}
    return raw.map(mp).astype(float)


def net_amount(amount: pd.Series, basis: str, rate: pd.Series,
               tax_amount: pd.Series | None = None) -> pd.Series:
    """net = gross - tax_amount (when given) else gross/(1+rate); net basis passes through."""
    if basis != "gross":
        return amount
    net = amount / (1.0 + rate.fillna(0.0))
    if tax_amount is not None:
        use = tax_amount.notna() & amount.notna()
        net = net.where(~use, amount - tax_amount)
    return net
