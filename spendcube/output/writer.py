"""Stage 7: assemble the output frame from the canonical frame and write the CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema_cols import EMIT_ROLES, EXCLUSION_CODES, OUTPUT_COLUMNS


def ensure_columns(ctx, df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every column the assembler needs exists (a failed stage must not break output)."""
    n = len(df)
    tax = ctx.taxonomy
    first_leaf = tax.leaves[0].id if tax is not None else ""
    defaults = {
        "excl_code": "", "excl_detail": "", "excl_conf": 1.0, "spend_rc": 0.0,
        "supplier_key": None, "item_key": None, "leaf_id": first_leaf, "cat_conf": 0.01,
        "cat_method": "placeholder", "flags": "",
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    if df["supplier_key"].isna().any() or (df["supplier_key"].astype(str).str.strip() == "").any():
        sk = df["supplier_key"].fillna("").astype(str)
        fb = np.where(df["vendor_id"].astype(str).str.strip() != "", "V:" + df["vendor_id"].astype(str).str.strip(),
                      "S:UNKNOWN")
        df["supplier_key"] = np.where(sk.str.strip() == "", fb, sk)
    if df["item_key"].isna().any() or (df["item_key"].astype(str).str.strip() == "").any():
        ik = df["item_key"].fillna("").astype(str)
        fb = "U:" + df["supplier_key"].astype(str)
        df["item_key"] = np.where(ik.str.strip() == "", fb, ik)
    bad_leaf = ~df["leaf_id"].astype(str).map(lambda x: tax.has(x) if tax is not None else True)
    if bad_leaf.any():
        df.loc[bad_leaf, "leaf_id"] = first_leaf
        df.loc[bad_leaf, "cat_conf"] = 0.01
    df["excl_code"] = df["excl_code"].fillna("").astype(str)
    df.loc[~df["excl_code"].isin(EXCLUSION_CODES), "excl_code"] = ""
    df["spend_rc"] = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0.0)
    return df


def assemble(ctx, df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(ctx, df)
    emit = df[df["source_role"].isin(EMIT_ROLES)].sort_values("row_order", kind="stable")
    tax = ctx.taxonomy
    inc = emit["excl_code"] == ""
    paths = {lid: tax.path_padded(lid) for lid in emit["leaf_id"].unique()}
    lv = np.array([paths[x] for x in emit["leaf_id"]], dtype=object).reshape(len(emit), 4)
    lv[~inc.to_numpy()] = ""
    dec = ctx.cfg.output.float_decimals
    spend = np.where(inc, emit["spend_rc"].round(dec), 0.0)
    conf = np.where(inc, emit["cat_conf"], emit["excl_conf"])
    out = pd.DataFrame({
        "record_id": emit.get("record_id_original", emit["record_id"]).to_numpy(),
        "include": np.where(inc, "1", "0"),
        "exclusion_reason": "",
        "supplier_key": emit["supplier_key"].to_numpy(),
        "item_key": emit["item_key"].to_numpy(),
        "spend": spend,
        "l1": lv[:, 0], "l2": lv[:, 1], "l3": lv[:, 2], "l4": lv[:, 3],
        "confidence": np.clip(pd.to_numeric(conf, errors="coerce"), 0, 1).round(3),
    })
    out["exclusion_reason"] = np.where(inc.to_numpy(), "", emit["excl_code"].to_numpy())
    out["confidence"] = out["confidence"].fillna(0.0)
    return out[OUTPUT_COLUMNS]


def write_output(out: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    return path
