"""Stage 2: numbers, VAT, dates, FX, PO/invoice extraction, text, supplier norm (architecture 5.2)."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..entities.legal_forms import supplier_norm
from .dates import parse_date_series
from .fx import FxTable, _month_ord, convert
from .numbers import parse_series
from .tax import net_amount, parse_rate_series
from .text import add_text_columns

log = logging.getLogger("spendcube.normalize")

_PO_PREFIX = re.compile(r"^\s*(?:PO|P\.O\.|PURCHASE ORDER)?\s*[#:.\-]*\s*", re.I)
_PO_ITEM_DEFAULT = re.compile(r"/\s*(\d+)(?:\.\d+)?\s*$")


def add_flag(df: pd.DataFrame, mask, name: str) -> None:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    cur = df.loc[m, "flags"].fillna("").astype(str)
    df.loc[m, "flags"] = np.where(cur == "", name, cur + "|" + name)


def _opt(columns: dict, key: str) -> dict:
    v = columns.get(key)
    return v if isinstance(v, dict) else {}


def _int_str(s: str) -> str:
    t = str(s).strip()
    if not t:
        return ""
    try:
        return str(int(float(t)))
    except ValueError:
        return t.upper()


def extract_po(raw: pd.Series, po_regex: str | None, item_regex: str | None, item_raw: pd.Series):
    """Returns (po_number, po_item). po_regex: first group (or whole match) of the first match."""
    raw = raw.fillna("").astype(str)
    uniq = pd.Series(raw.unique())
    rx = re.compile(po_regex) if po_regex else None
    irx = re.compile(item_regex) if item_regex else _PO_ITEM_DEFAULT

    def one(s: str) -> tuple[str, str]:
        t = s.strip()
        if not t:
            return "", ""
        item = ""
        mi = irx.search(t)
        if mi:
            item = _int_str(mi.group(1))
        if rx:
            m = rx.search(t)
            return (m.group(1) if m and m.groups() else (m.group(0) if m else "")), item
        t = _PO_PREFIX.sub("", t.split("/")[0])
        return re.sub(r"[^0-9A-Za-z]", "", t).upper(), item

    mp = {u: one(u) for u in uniq}
    po = raw.map(lambda u: mp[u][0])
    item = raw.map(lambda u: mp[u][1])
    ir = item_raw.fillna("").astype(str).map(_int_str)
    item = item.where(item != "", ir)
    return po, item


def normalize_stage(ctx, df: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.cfg
    df = add_text_columns(df, cfg.null_description_tokens)
    sup = df["supplier_raw"].fillna("")
    u = pd.Series(sup.unique())
    df["supplier_norm"] = sup.map(dict(zip(u, u.map(supplier_norm))))
    df["supplier_core"] = df["supplier_norm"].str.replace(" ", "", regex=False)
    if "flags" not in df.columns:
        df["flags"] = ""
    n = len(df)
    df["quantity"] = np.nan
    df["amount"] = np.nan
    df["tax_rate"] = np.nan
    df["net_local"] = np.nan
    df["fx_rate_used"] = np.nan
    df["fx_method"] = ""
    df["spend_rc"] = 0.0
    df["invoice_date"] = pd.NaT
    df["posting_date"] = pd.NaT
    df["po_number"] = ""
    df["po_item"] = ""
    df["invoice_no_norm"] = df["invoice_no_raw"].fillna("").astype(str).str.upper().str.replace(r"[^0-9A-Z]", "", regex=True)

    table = FxTable.load(cfg.fx)
    reporting = cfg.reporting_currency.upper()
    fx_counts: dict[str, int] = {}
    for src in cfg.sources:
        idx = df.index[df["source"] == src.name]
        if len(idx) == 0:
            continue
        sub = df.loc[idx]
        cols = src.columns
        nf = src.number_format
        dec = nf.get("decimal") if isinstance(nf, dict) else None
        # --- amounts ---
        amt, used_dec = parse_series(sub["amount_raw"], dec)
        blank_amt = sub["amount_raw"].fillna("").str.strip() == ""
        add_flag_sub = lambda mask, name: add_flag(df, pd.Series(mask, index=sub.index).reindex(df.index, fill_value=False), name)  # noqa: E731
        add_flag_sub(blank_amt, "AMOUNT_BLANK")
        add_flag_sub(amt.isna() & ~blank_amt, "AMOUNT_UNPARSEABLE")
        qty, _ = parse_series(sub["quantity_raw"], dec)
        basis = str(cols.get("amount_basis", "net")).lower()
        dtr = cols.get("default_tax_rate")
        dtr = float(dtr) if dtr not in (None, "") else None
        has_rate_col = bool(cols.get("tax_rate"))
        rate = (parse_rate_series(sub["tax_rate_raw"], dtr) if has_rate_col
                else pd.Series(dtr if dtr is not None else 0.0, index=sub.index))
        tax_amt = None
        if cols.get("tax_amount"):
            tax_amt, _ = parse_series(sub["tax_amount_raw"], dec)
            tax_amt = tax_amt.where(sub["tax_amount_raw"].fillna("").str.strip() != "")
        net = net_amount(amt, basis, rate, tax_amt)
        # --- dates ---
        d_out = {}
        for col, raw_col in (("invoice_date", "invoice_date_raw"), ("posting_date", "posting_date_raw")):
            o = _opt(cols, col)
            if (sub[raw_col].fillna("").str.strip() == "").all():
                d_out[col] = pd.Series(pd.NaT, index=sub.index)
                continue
            d, unp, imp, info = parse_date_series(sub[raw_col], o.get("dayfirst"),
                                                  tuple(cfg.valid_years) if cfg.valid_years else None)
            d_out[col] = d
            add_flag_sub(unp, f"{col.upper()}_UNPARSEABLE".replace("INVOICE_DATE", "DATE").replace("POSTING_DATE", "POSTING_DATE"))
            add_flag_sub(imp, "DATE_IMPOSSIBLE" if col == "invoice_date" else "POSTING_DATE_IMPOSSIBLE")
            ctx.note("dates", f"{src.name}.{col}", f"dayfirst={info['dayfirst']} window={info['window']} "
                     f"unparseable={int(unp.sum())} impossible={int(imp.sum())}")
        # --- FX ---
        lr, _ = parse_series(sub["fx_rate_raw"], dec)
        inv_m = d_out["invoice_date"].map(_month_ord).astype(float) if d_out["invoice_date"].notna().any() \
            else pd.Series(np.nan, index=sub.index)
        post_m = d_out["posting_date"].map(_month_ord).astype(float) if d_out["posting_date"].notna().any() \
            else pd.Series(np.nan, index=sub.index)
        spend, rate_used, method, flags = convert(
            net, sub["currency"], lr, inv_m, post_m, reporting, table, tuple(cfg.fx.plausibility_band),
            str(cols.get("fx_rate_convention", "units_per_reporting")))
        for fname, mask in flags.items():
            add_flag_sub(mask & net.notna() if fname == "FX_MISSING" else mask, fname)
        for k, v in method.value_counts().items():
            fx_counts[f"{src.name}:{k}"] = int(v)
        # --- PO / invoice ---
        po, item = extract_po(sub["po_raw"], cols.get("po_regex"), cols.get("po_item_regex"), sub["po_item_raw"])
        # --- write back ---
        df.loc[idx, "quantity"] = qty
        df.loc[idx, "amount"] = amt
        df.loc[idx, "tax_rate"] = rate if (basis == "gross" or has_rate_col) else np.nan
        df.loc[idx, "net_local"] = net
        df.loc[idx, "fx_rate_used"] = rate_used
        df.loc[idx, "fx_method"] = method
        df.loc[idx, "spend_rc"] = spend.where(net.notna(), 0.0)
        df.loc[idx, "invoice_date"] = d_out["invoice_date"]
        df.loc[idx, "posting_date"] = d_out["posting_date"]
        df.loc[idx, "po_number"] = po
        df.loc[idx, "po_item"] = item
        ctx.note("amounts", src.name, f"decimal={used_dec!r} basis={basis} total_net_local={float(net.sum()):,.2f} "
                 f"total_spend_rc={float(spend.sum()):,.2f}")
    ctx.note("fx", "method_counts", fx_counts)
    ctx.state["fx_method_counts"] = fx_counts
    return df
