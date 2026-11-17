"""Classification units: unique (supplier, description, GL, context) tuples ordered by |spend| desc.

The unit frame carries the canonical feature columns of the first row of each unit so the supervised model can
be fitted/predicted on it directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIT_KEY_FIELDS = ["supplier_core", "vendor_id", "desc_compact", "gl_account", "llm_context"]
FEATURE_FIELDS = ["desc_norm", "desc_compact", "supplier_core", "supplier_raw", "vendor_id", "gl_account",
                  "gl_desc", "material_id", "llm_context", "desc_blank"]


def build_units(df: pd.DataFrame, spend_col: str = "spend_rc") -> tuple[pd.DataFrame, pd.Series]:
    """Returns (units, row_uid). units: uid(1..n), supplier, gl, ctx, text, described, spend, rows + feature cols."""
    key_cols = [c for c in UNIT_KEY_FIELDS if c in df.columns]
    keys = df[key_cols].astype(str).agg("\x1f".join, axis=1)
    codes, _ = pd.factorize(keys)
    d = pd.DataFrame({"code": codes, "spend": pd.to_numeric(df[spend_col], errors="coerce").abs().fillna(0.0)})
    agg = d.groupby("code").agg(spend=("spend", "sum"), rows=("spend", "size"))
    first = df.assign(code=codes).drop_duplicates("code").set_index("code")
    keep = [c for c in FEATURE_FIELDS if c in first.columns] + [c for c in first.columns if c.startswith("ctx__")]
    units = first[keep].copy()
    units["supplier"] = first["supplier_raw"].where(first["supplier_raw"].astype(str).str.strip() != "", first["vendor_id"])
    units["gl"] = first["gl_desc"].where(first["gl_desc"].astype(str).str.strip() != "", first["gl_account"])
    units["ctx"] = first["llm_context"]
    units["text"] = first["desc_raw"].astype(str).str.split().str.join(" ")
    units["described"] = ~first["desc_blank"]
    units = units.join(agg).sort_values(["spend", "rows"], ascending=False)
    units["uid"] = np.arange(1, len(units) + 1)
    remap = pd.Series(units["uid"].to_numpy(), index=units.index)
    row_uid = pd.Series(codes, index=df.index).map(remap)
    return units.reset_index(drop=True), row_uid
