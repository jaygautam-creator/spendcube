"""Output contract validator (architecture section 9). Used by the pipeline and `validate` CLI."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ..schema_cols import EXCLUSION_CODES, OUTPUT_COLUMNS
from ..taxonomy.tree import Taxonomy


def validate_frame(out: pd.DataFrame, tax: Taxonomy, expected_ids: Optional[list[str]] = None,
                   max_errors: int = 20) -> list[str]:
    errs: list[str] = []

    def add(msg: str):
        if len(errs) < max_errors:
            errs.append(msg)

    if list(out.columns) != OUTPUT_COLUMNS:
        add(f"columns {list(out.columns)} != {OUTPUT_COLUMNS}")
        return errs
    if expected_ids is not None:
        if len(out) != len(expected_ids):
            add(f"row count {len(out)} != expected {len(expected_ids)}")
        elif list(out["record_id"]) != list(expected_ids):
            bad = int((out["record_id"].reset_index(drop=True) != pd.Series(expected_ids)).sum())
            add(f"record_id order/content differs from input in {bad} rows")
    if out["record_id"].duplicated().any():
        add(f"{int(out['record_id'].duplicated().sum())} duplicated record_id")
    if (out["record_id"].astype(str).str.strip() == "").any():
        add("blank record_id")

    inc = out["include"].astype(str)
    if not inc.isin(["0", "1"]).all():
        add("include not in {0,1}")
    included = inc == "1"
    reason = out["exclusion_reason"].fillna("").astype(str)
    if (reason[included] != "").any():
        add("exclusion_reason set on included rows")
    if (~reason[~included].isin(EXCLUSION_CODES)).any():
        add("excluded rows with missing/invalid exclusion_reason")

    spend = pd.to_numeric(out["spend"], errors="coerce")
    if spend.isna().any():
        add(f"{int(spend.isna().sum())} non-numeric spend")
    if (spend[~included].fillna(0) != 0).any():
        add("non-zero spend on excluded rows")
    conf = pd.to_numeric(out["confidence"], errors="coerce")
    if conf.isna().any() or ((conf < 0) | (conf > 1)).any():
        add("confidence outside [0,1] or non-numeric")
    for k in ("supplier_key", "item_key"):
        if (out[k].fillna("").astype(str).str.strip() == "").any():
            add(f"blank {k}")

    lv = out[["l1", "l2", "l3", "l4"]].fillna("").astype(str)
    exc = lv[~included]
    if (exc != "").any().any():
        add("category set on excluded rows")
    inc_lv = lv[included]
    if len(inc_lv):
        valid = tax.valid_paths()
        keys = list(zip(*(inc_lv[c] for c in ("l1", "l2", "l3", "l4"))))
        bad = 0
        for k in set(keys):
            path = tuple(x for x in k if x != "")
            # blanks only allowed below the leaf depth
            if path not in valid or tuple(k[len(path):]) != ("",) * (4 - len(path)):
                bad += keys.count(k)
        if bad:
            add(f"{bad} included rows have a category path not in the tree")
    return errs


def validate_file(path: Path, tax: Taxonomy, expected_ids: Optional[list[str]] = None,
                  expected_name: Optional[str] = None) -> list[str]:
    errs: list[str] = []
    if expected_name and Path(path).name != expected_name:
        errs.append(f"file name {Path(path).name} != expected {expected_name}")
    if not re.search(r"__[A-Za-z0-9]+\.csv$", Path(path).name):
        errs.append("file name must look like <name>__<A|B|A2|B2|C>.csv")
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    return errs + validate_frame(out, tax, expected_ids)
