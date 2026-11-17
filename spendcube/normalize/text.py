"""Text normalisation: desc_norm / desc_compact / desc_blank (architecture 5.2)."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

_WS = re.compile(r"\s+")
_TRAIL = re.compile(r"(?:\s*[-–—]\s*)+$")
_NONALNUM = re.compile(r"[^0-9A-Z]")

DEFAULT_NULL_TOKENS = {"", "-", "--", "N/A", "NA", "NULL", "NONE", "NO OBJECT", "NO DESCRIPTION", "."}


def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = _WS.sub(" ", s).strip()
    s = _TRAIL.sub("", s).strip()
    return s.upper()


def compact(s: str) -> str:
    return _NONALNUM.sub("", s)


def add_text_columns(df: pd.DataFrame, null_tokens: list[str] | None = None) -> pd.DataFrame:
    tokens = DEFAULT_NULL_TOKENS | {t.upper() for t in (null_tokens or [])}
    raw = df["desc_raw"].fillna("").astype(str)
    # unique-ise: descriptions repeat massively
    uniq = pd.Series(raw.unique())
    n = uniq.map(norm_text)
    mp = dict(zip(uniq, n))
    df["desc_norm"] = raw.map(mp)
    cp = {k: compact(v) for k, v in mp.items()}
    df["desc_compact"] = raw.map(cp)
    df["desc_blank"] = df["desc_norm"].isin(tokens) | (df["desc_compact"] == "")
    return df
