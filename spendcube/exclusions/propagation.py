"""Vendor propagation: a vendor whose rows are mostly excluded by rules has all its rows excluded."""
from __future__ import annotations

import numpy as np
import pandas as pd


def vendor_key(df: pd.DataFrame) -> pd.Series:
    vid = df["vendor_id"].fillna("").astype(str).str.strip()
    core = df["supplier_core"].fillna("").astype(str) if "supplier_core" in df.columns else pd.Series("", index=df.index)
    return pd.Series(np.where(vid != "", "V:" + vid, np.where(core != "", "S:" + core, "")), index=df.index)


def propagate_vendor(df: pd.DataFrame, rule_code: np.ndarray, protect: np.ndarray, code: str,
                     min_share: float, min_rows: int) -> np.ndarray:
    """Boolean mask of rows to newly exclude with `code` (rows not already carrying a rule code)."""
    vk = vendor_key(df)
    hit = (rule_code == code)
    t = pd.DataFrame({"vk": vk.to_numpy(), "hit": hit})
    t = t[t["vk"] != ""]
    agg = t.groupby("vk")["hit"].agg(["sum", "size"])
    ok = agg[(agg["size"] >= min_rows) & (agg["sum"] / agg["size"] >= min_share)].index
    cand = vk.isin(ok).to_numpy() & ~hit & ~protect
    return cand
