"""R1: per-row reconciliation features, computed from RAW ingest columns (with small local helpers).

If the normalize stage already produced net_local / po_number / invoice_no_norm / quantity /
desc_compact / supplier_core / invoice_date, those are preferred where non-null.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..entities.legal_forms import supplier_core

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "primary_sources": ["erp"],          # source names that play the "ERP / receipts" role
    "secondary_sources": ["tool"],       # source names that play the "invoice tool" role
    "po_regex": r"45\d{8}",
    "po_item_regex": r"/\s*(\d+)(?:\.\d+)?",
    "default_tax_rate": 0.15,
    "tolerance": {"abs": 0.02, "rel": 0.001},
    "near_pct": 0.01,                    # amount within this share still scores (PO route only)
    "min_score": 5.5,
    "weights": {"amount_exact": 4.0, "amount_near": 1.5, "qty": 1.0, "po_item": 1.0,
                "desc": 2.0, "supplier": 1.0, "po_route": 2.0},
    # supplier+amount route (no PO evidence). OFF by default: measured on real A it is mostly coincidence
    # (hold-out sim FP 34%; matched partners sit ~half the PO range away from the id-order prediction).
    # when: no_po | unknown (also PO absent from pool) | any (also PO present but amount mismatch)
    # guards (D-19 re-test): one-to-one capacity vs the PO route, |net|<=min_abs ignored, dates (when both exist)
    # must be within max_days.
    "core_route": {"enabled": False, "when": "no_po", "max_days": 120, "min_abs": 0.01},
    # D-22 capacity match for partial-quantity invoices (same PO + supplier core, unit price within price_tol,
    # cumulative tool qty <= receipt qty). Off unless measured safe.
    "partial_capacity": {"enabled": False, "price_tol": 0.005, "max_candidates": 60},
    "max_po_candidates": 300,
    "max_amount_candidates": 60,
    "reexport": {"desc_sim": 0.85, "date_days": 45, "block_cap": 400,
                 "link_without_invoice": False},     # D-21 variant, off: bulk lines repeat price
    # R6 orphan rescue. require_same_invoice: on real A all 49 'rescued' pairs had DIFFERENT invoice numbers
    # (distinct invoices on one PO line), so rescuing them would drop real spend; default requires equal digits.
    "rescue": {"enabled": True, "desc_sim": 0.80, "require_same_invoice": True},
    "aggregate": {"enabled": False, "max_items": 3, "max_candidates": 30, "tol_abs": 0.02},
    # History (D-02): during A2 the original A files are loaded as reference rows (never emitted).
    # enabled: false | true | "auto". files: {<source name>: <glob>}; reference sources are "<name>_hist".
    # With "auto"/true, history is skipped when its files equal the run's own files (build runs).
    "history": {"enabled": False, "files": {}, "suffix": "_hist"},
    "status_rank": {"paid": 3, "approved": 2},
    "ineligible_status_words": ["cancel", "reject", "hold", "dispute", "draft", "void"],
}


def get_cfg(ctx_cfg) -> dict[str, Any]:
    """Merge config.reconciliation over DEFAULTS (one level deep)."""
    raw = getattr(ctx_cfg, "reconciliation", None) or {}
    out: dict[str, Any] = {}
    for k, v in DEFAULTS.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            out[k] = {**v, **raw[k]}
        else:
            out[k] = raw.get(k, v)
    return out


# ---------------------------------------------------------------------------- parsers
_NUM_JUNK = re.compile(r"[^0-9.\-()]")


def parse_amount_series(s: pd.Series) -> pd.Series:
    """Amounts with thousands commas, parentheses negatives, trailing minus; NaN if unparseable."""
    x = s.fillna("").astype(str).str.strip()
    neg = x.str.startswith("(") & x.str.endswith(")") | x.str.endswith("-")
    y = x.str.replace(",", "", regex=False).map(lambda v: _NUM_JUNK.sub("", v))
    y = y.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.rstrip("-")
    v = pd.to_numeric(y, errors="coerce")
    return v.where(~neg, -v.abs())


def parse_vat_series(s: pd.Series, default: float) -> pd.Series:
    """15 / 0.15 / 15% / blank(->default) / 0 (->0%, NOT blank). Values >1 are percentages."""
    x = s.fillna("").astype(str).str.strip().str.replace("%", "", regex=False).str.strip()
    blank = x == ""
    v = pd.to_numeric(x.str.replace(",", ".", regex=False), errors="coerce")
    v = v.where(v.isna() | (v <= 1), v / 100.0)
    v = v.where(~blank, default)
    return v.fillna(default)


def extract_po(s: pd.Series, regex: str) -> pd.Series:
    return s.fillna("").astype(str).str.extract("(" + regex + ")", expand=False).fillna("")


def extract_po_item_tool(s: pd.Series, regex: str) -> pd.Series:
    x = s.fillna("").astype(str).str.extract(regex, expand=False)
    return pd.to_numeric(x, errors="coerce")


def parse_po_item_erp(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.fillna("").astype(str).str.strip(), errors="coerce")


_ALNUM = re.compile(r"[^0-9A-Z]")


def norm_alnum(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.upper().map(lambda v: _ALNUM.sub("", v))


def compact_text(s: pd.Series) -> pd.Series:
    return norm_alnum(s)


_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%d.%m.%Y", "%Y/%m/%d", "%Y%m%d"]


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse the observed formats (ISO, dd/mm/yyyy day-first, dd-Mon-yy); NaT otherwise."""
    x = s.fillna("").astype(str).str.strip()
    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")
    todo = x != ""
    for fmt in _DATE_FORMATS:
        if not todo.any():
            break
        p = pd.to_datetime(x[todo], format=fmt, errors="coerce")
        ok = p.notna()
        out.loc[p.index[ok]] = p[ok].astype("datetime64[ns]")
        todo.loc[p.index[ok]] = False
    return out


_GLUED = ("PTYLTD", "LIMITED", "LTD", "PTY", "CC")


def strip_glued_legal(core: str) -> str:
    """Whitespace-free cores can end in a glued legal suffix ('ACMEPTYLTD'); strip it when >=4 chars remain."""
    for _ in range(2):
        for suf in _GLUED:
            if core.endswith(suf) and len(core) - len(suf) >= 4:
                core = core[: -len(suf)]
                break
        else:
            break
    return core


# ---------------------------------------------------------------------------- feature frame
def _prefer(df: pd.DataFrame, col: str, computed: pd.Series) -> pd.Series:
    if col in df.columns:
        existing = df[col]
        if existing.notna().any():
            if existing.dtype == object or str(existing.dtype).startswith("str"):
                e = existing.fillna("").astype(str)
                return e.where(e != "", computed)
            return existing.where(existing.notna(), computed)
    return computed


def build_features(df: pd.DataFrame, cfg: dict[str, Any], is_tool: pd.Series) -> pd.DataFrame:
    """Return a frame aligned to df.index with prefixed feature columns f_*.

    `is_tool` marks invoice-tool-like rows (gross amounts + VAT); others are receipt-like (net as given).
    """
    f = pd.DataFrame(index=df.index)
    raw_amt = parse_amount_series(df["amount_raw"])
    vat = parse_vat_series(df["tax_rate_raw"], cfg["default_tax_rate"])
    net = np.where(is_tool, raw_amt / (1.0 + vat), raw_amt)
    f["f_gross"] = raw_amt
    f["f_net"] = _prefer(df, "net_local", pd.Series(net, index=df.index))
    f["f_net"] = pd.to_numeric(f["f_net"], errors="coerce")
    qty = pd.to_numeric(df["quantity_raw"].fillna("").astype(str).str.replace(",", "", regex=False), errors="coerce")
    f["f_qty"] = _prefer(df, "quantity", qty)

    po_tool = extract_po(df["po_raw"], cfg["po_regex"])
    po = po_tool
    f["f_po"] = _prefer(df, "po_number", po).fillna("").astype(str)
    poi_tool = extract_po_item_tool(df["po_raw"], cfg["po_item_regex"])
    poi_erp = parse_po_item_erp(df["po_item_raw"]) if "po_item_raw" in df.columns else poi_tool * np.nan
    poi = pd.Series(np.where(is_tool, poi_tool, poi_erp.fillna(poi_tool)), index=df.index)
    # a tool row without an explicit /item falls back to the ERP-side column (blank) -> NaN
    f["f_poi"] = pd.to_numeric(poi, errors="coerce")

    sup = df["supplier_raw"].fillna("").astype(str)
    u = pd.Series(sup.unique())
    core_map = dict(zip(u, u.map(supplier_core)))
    core = _prefer(df, "supplier_core", sup.map(core_map)).fillna("").astype(str)
    cu = pd.Series(core.unique())
    f["f_core"] = core.map(dict(zip(cu, cu.map(strip_glued_legal))))
    f["f_desc"] = _prefer(df, "desc_compact", compact_text(df["desc_raw"])).fillna("").astype(str)
    inv = norm_alnum(df["invoice_no_raw"])
    f["f_inv"] = _prefer(df, "invoice_no_norm", inv).fillna("").astype(str)
    f["f_invd"] = f["f_inv"].str.replace(r"\D", "", regex=True)
    dates = parse_dates(df["invoice_date_raw"])
    f["f_date"] = dates
    pd_raw = df["posting_date_raw"] if "posting_date_raw" in df.columns else pd.Series("", index=df.index)
    f["f_pdate"] = parse_dates(pd_raw)
    st = df["status_raw"].fillna("").astype(str).str.strip().str.lower()
    ranks = cfg["status_rank"]
    f["f_rank"] = st.map(lambda v: next((r for k, r in ranks.items() if k in v), 1)).astype(int)
    bad = cfg["ineligible_status_words"]
    status_bad = st.map(lambda v: any(w in v for w in bad)) & (st != "")
    already = df["excl_code"].fillna("").astype(str) != "" if "excl_code" in df.columns else False
    f["f_eligible"] = ~(status_bad | already) if not isinstance(already, bool) else ~status_bad
    f["f_gross_r"] = f["f_gross"].round(2)
    return f
