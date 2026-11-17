"""Stage 5: supplier_key / item_key (+ confidences). Never drops rows (architecture section 7).

Order: cluster suppliers and items (unsupervised) -> label alias maps (when labels carry golden keys) ->
record-level golden keys for rows that are themselves labelled -> inheritance along reconcile's match_ref
(tool -> ERP, re-export -> canonical, history -> golden keys of the original ERP rows).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..labels import load_labels
from ..normalize.text import add_text_columns
from .items import build_item_alias, resolve_items, text_key
from .legal_forms import supplier_norm
from .suppliers import build_supplier_alias, resolve_suppliers

log = logging.getLogger("spendcube.entities")

INHERIT_TYPES = ("erp", "history", "reexport")


def _prep_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Make sure raw-ingest rows (history) carry supplier_core / desc_* columns."""
    rows = rows.copy()
    if "desc_norm" not in rows.columns:
        rows = add_text_columns(rows, None)
    if "supplier_core" not in rows.columns:
        s = rows["supplier_raw"].fillna("").astype(str)
        u = pd.Series(s.unique())
        mp = dict(zip(u, u.map(lambda x: supplier_norm(x).replace(" ", ""))))
        rows["supplier_core"] = s.map(mp)
    return rows


def label_rows(ctx, df: pd.DataFrame) -> pd.DataFrame | None:
    """Rows (current frame + history reference rows) joined to labels, with gold_supplier / gold_item."""
    if "label_rows" in ctx.state:
        return ctx.state["label_rows"]
    out = None
    try:
        lab = load_labels(ctx)
        if lab is not None and ("supplier_key" in lab.columns or "item_key" in lab.columns):
            lab = lab.drop_duplicates("record_id")
            parts = []
            have = df[df["record_id"].isin(lab["record_id"])]
            if len(have):
                parts.append(have)
            missing = len(lab) - len(have)
            if missing > 0 and getattr(ctx.cfg, "reconciliation", None) is not None:
                try:
                    from ..reconcile.features import get_cfg
                    from ..reconcile.stage import _load_history
                    ref = _load_history(ctx, get_cfg(ctx.cfg))
                    if ref is not None:
                        ref = ref[ref["record_id"].isin(lab["record_id"])]
                        if len(ref):
                            parts.append(_prep_rows(ref))
                            log.info("entities: %d history rows used for label alias maps", len(ref))
                except Exception as e:
                    ctx.warn(f"entities: history rows for aliases unavailable: {type(e).__name__}: {e}")
            if parts:
                cols = ["record_id", "supplier_core", "vendor_id", "material_id", "desc_norm"]
                rows = pd.concat([p[cols] for p in parts], ignore_index=True).drop_duplicates("record_id")
                rows = rows.merge(lab[["record_id"] + [c for c in ("supplier_key", "item_key") if c in lab.columns]],
                                  on="record_id", how="left")
                rows["gold_supplier"] = rows["supplier_key"].fillna("") if "supplier_key" in rows else ""
                rows["gold_item"] = rows["item_key"].fillna("") if "item_key" in rows else ""
                rows["text_key"] = [text_key(x) for x in rows["desc_norm"].fillna("")]
                out = rows
    except Exception as e:
        ctx.warn(f"entities: label rows unavailable: {type(e).__name__}: {e}")
    ctx.state["label_rows"] = out
    return out


def _inherit(df: pd.DataFrame, sk: pd.Series, sc: pd.Series, ik: pd.Series, ic: pd.Series,
             gold: pd.DataFrame | None) -> int:
    if "match_type" not in df.columns or "match_ref" not in df.columns:
        return 0
    mt = df["match_type"].fillna("none").astype(str)
    ref = df["match_ref"].fillna("").astype(str)
    sel = mt.isin(INHERIT_TYPES) & (ref != "")
    if not sel.any():
        return 0
    gs = gi = None
    if gold is not None:
        g = gold.drop_duplicates("record_id").set_index("record_id")
        gs = g["supplier_key"].where(g["supplier_key"] != "") if "supplier_key" in g else None
        gi = g["item_key"].where(g["item_key"] != "") if "item_key" in g else None
    rid = df["record_id"]
    n_inh = 0
    for _ in range(3):                      # re-export -> canonical -> ERP chains
        s_by = pd.Series(sk.to_numpy(), index=rid.to_numpy())
        i_by = pd.Series(ik.to_numpy(), index=rid.to_numpy())
        s_by = s_by[~s_by.index.duplicated()]
        i_by = i_by[~i_by.index.duplicated()]
        is_hist = ref.str.startswith("hist:")
        r_clean = ref.str.replace("^hist:", "", regex=True)
        tsk = r_clean.map(s_by)
        tik = r_clean.map(i_by)
        if gs is not None:                  # history rows are not in the frame: use golden keys
            hs = r_clean.map(gs)
            tsk = tsk.where(~is_hist, hs)
        if gi is not None:
            hi = r_clean.map(gi)
            tik = tik.where(~is_hist, hi)
        ms = sel & tsk.notna()
        mi = sel & tik.notna() & ~tik.astype(str).str.startswith("U:")
        changed = int((sk[ms] != tsk[ms]).sum())
        sk.loc[ms], sc.loc[ms] = tsk[ms], 0.97
        ik.loc[mi], ic.loc[mi] = tik[mi], 0.97
        n_inh = max(n_inh, int(ms.sum()))
        if changed == 0:
            break
    return n_inh


def entities_stage(ctx, df: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.cfg.entities or {}
    sup_cfg, item_cfg = cfg.get("suppliers", {}) or {}, cfg.get("items", {}) or {}
    purity = float(cfg.get("alias_purity", 0.9))
    use_labels = bool(cfg.get("use_labels", True))
    emitted = df["source_role"].isin(("primary", "secondary")) if "source_role" in df.columns else pd.Series(True, index=df.index)

    a_core = a_vend = a_mat = a_txt = None
    lab = label_rows(ctx, df) if use_labels else None
    if lab is not None and len(lab):
        if lab["gold_supplier"].astype(str).ne("").any():
            a_core, a_vend = build_supplier_alias(lab, purity)
        if lab["gold_item"].astype(str).ne("").any():
            a_mat, a_txt = build_item_alias(lab, purity)
        ctx.note("entities", "alias_maps", {"supplier_cores": len(a_core or {}), "vendors": len(a_vend or {}),
                                            "materials": len(a_mat or {}), "texts": len(a_txt or {})})

    sk, sc, sstats = resolve_suppliers(df, sup_cfg, a_core, a_vend)
    prim = (df["source_role"] == "primary") if "source_role" in df.columns else None
    ik, ic, istats = resolve_items(df, item_cfg, sk, a_mat, a_txt, prim)

    # record-level golden keys (rows that are themselves labelled: build runs / A on its own labels)
    if use_labels:
        try:
            labels = load_labels(ctx)
            if labels is not None:
                g = labels.drop_duplicates("record_id").set_index("record_id")
                if "supplier_key" in g.columns:
                    gs = df["record_id"].map(g["supplier_key"].where(g["supplier_key"] != ""))
                    m = gs.notna()
                    sk.loc[m], sc.loc[m] = gs[m], 1.0
                    ctx.note("entities", "record_level_supplier_keys", int(m.sum()))
                if "item_key" in g.columns:
                    gi = df["record_id"].map(g["item_key"].where(g["item_key"] != ""))
                    m = gi.notna()
                    ik.loc[m], ic.loc[m] = gi[m], 1.0
                    ctx.note("entities", "record_level_item_keys", int(m.sum()))
        except Exception as e:
            ctx.warn(f"entities: record-level labels skipped: {type(e).__name__}: {e}")

    gold = None
    try:
        gold = load_labels(ctx) if use_labels else None
    except Exception:
        gold = None
    n_inh = _inherit(df, sk, sc, ik, ic, gold)
    df["supplier_key"], df["supplier_conf"] = sk.astype(str), sc.astype(float)
    df["item_key"], df["item_conf"] = ik.astype(str), ic.astype(float)
    ctx.note("entities", "supplier_keys", int(df.loc[emitted, "supplier_key"].nunique()))
    ctx.note("entities", "item_keys", int(df.loc[emitted, "item_key"].nunique()))
    ctx.note("entities", "inherited_rows", n_inh)
    ctx.note("entities", "supplier_stats", {k: v for k, v in sstats.items() if k != "distinct_keys"})
    ctx.note("entities", "item_stats", istats)
    return df
