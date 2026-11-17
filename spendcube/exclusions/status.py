"""Status mapping: raw status value -> INCLUDE / CANCELLED / NOT_APPROVED via per-source status_map."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("spendcube.exclusions")


def build_lookup(status_map) -> tuple[dict[str, str], str]:
    lk: dict[str, str] = {}
    for cls in ("INCLUDE", "CANCELLED", "NOT_APPROVED"):
        for v in getattr(status_map, cls, []) or []:
            lk[str(v).strip().casefold()] = cls
    return lk, getattr(status_map, "default", "INCLUDE")


def classify_status(raw: pd.Series, status_map) -> tuple[pd.Series, dict[str, int]]:
    """Returns (status_class, unknown_value_counts). Unknown / blank values use status_map.default."""
    lk, default = build_lookup(status_map)
    key = raw.fillna("").astype(str).str.strip().str.casefold()
    cls = key.map(lk)
    unknown = cls.isna()
    counts = raw[unknown].fillna("").astype(str).str.strip().value_counts().to_dict() if unknown.any() else {}
    return cls.fillna(default).astype(object), {str(k): int(v) for k, v in counts.items()}
