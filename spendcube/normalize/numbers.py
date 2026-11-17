"""Robust number parsing: 1,234.56 / 1.234,56 / (123.45) / 123.45- / currency symbols / NBSP / spaces."""
from __future__ import annotations

import math
import re

import pandas as pd

_STRIP = re.compile(r"[^\d.,\-+()]")
_US_THOU = re.compile(r"\d,\d{3}(?!\d)")
_EU_THOU = re.compile(r"\d\.\d{3}(?!\d)")
_EU_DEC = re.compile(r",\d{1,2}$")
_US_DEC = re.compile(r"\.\d{1,2}$")


def _clean(s) -> str:
    # NBSP, narrow NBSP, thin space and apostrophe thousands separators (Swiss 1'234.56) are removed
    t = str(s)
    for ch in ("\u00a0", "\u202f", "\u2009", "'", "\u2019", "\u02bc", "`"):
        t = t.replace(ch, "")
    return t.replace(" ", "").strip()


def detect_decimal(values) -> str:
    """Decimal separator for a column by majority evidence ('.' default)."""
    eu = us = 0
    for v in values:
        t = _STRIP.sub("", _clean(v))
        if not t:
            continue
        has_c, has_d = "," in t, "." in t
        if has_c and has_d:
            if t.rfind(",") > t.rfind("."):
                eu += 1
            else:
                us += 1
        elif has_c:
            if _US_THOU.search(t):
                us += 1
            elif _EU_DEC.search(t):
                eu += 1
        elif has_d:
            if t.count(".") > 1:
                eu += 1          # 1.234.567
            elif _US_DEC.search(t):
                us += 1
    return "," if eu > us else "."


def parse_number(s, decimal: str = ".") -> float:
    """Parse one value; NaN when blank/unparseable."""
    if s is None:
        return math.nan
    t = _clean(s)
    if t == "":
        return math.nan
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    # strip only LEADING/TRAILING currency symbols or letters ("R 5,266.63", "12 USD"); letters inside
    # the number ("1x2", "x1y") make it unparseable rather than silently becoming 12.
    t = re.sub(r"^[^\d\-+(.,]+|[^\d)\-.,]+$", "", t).replace(" ", "")
    if re.search(r"[^\d.,\-+()]", t):
        return math.nan
    t = t.replace("(", "").replace(")", "")
    if t.endswith("-"):
        neg, t = True, t[:-1]
    if t.startswith("-"):
        neg, t = True, t[1:]
    elif t.startswith("+"):
        t = t[1:]
    if t == "" or not re.search(r"\d", t):
        return math.nan
    if decimal == ",":
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", "")
    if t.count(".") > 1 or "-" in t:
        return math.nan
    try:
        v = float(t)
    except ValueError:
        return math.nan
    return -v if neg else v


def parse_series(raw: pd.Series, decimal: str | None = None) -> tuple[pd.Series, str]:
    """Parse a str Series (unique-ised for speed). decimal None/'auto' -> detect. Returns (floats, decimal)."""
    raw = raw.fillna("").astype(str)
    uniq = pd.Series(raw.unique())
    if decimal in (None, "auto"):
        decimal = detect_decimal(uniq.tolist())
    mp = {u: parse_number(u, decimal) for u in uniq}
    return raw.map(mp).astype(float), decimal
