"""Real-data reconciliation hold-outs (kit required; numbers only are reported, nothing is stored).

 * core_route_holdout: blank the PO on a random share of PO-bearing tool rows, run the no-PO route ON vs OFF and
   compare DUPLICATE flags (the scored quantity) against the full-information run.
 * partial_holdout: hide a share of matched ERP receipts; rows whose partner vanished are 'true tool-only' for
   the visible data, so any DUPLICATE the partial-capacity route gives them is a false positive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config.loader import load_config
from ..context import RunContext
from ..ingest.canonical import ingest_all
from ..reconcile.features import get_cfg
from ..reconcile.stage import reconcile_core

log = logging.getLogger("spendcube.eval")


def load_prepared(config_path: str = "configs/A.yaml") -> tuple[pd.DataFrame, Any]:
    from ..exclusions.stage import exclusions_stage
    from ..normalize.stage import normalize_stage
    from ..taxonomy.tree import Taxonomy
    cfg = load_config(config_path)
    ctx = RunContext(cfg=cfg, run_dir=Path("runs/_tmp"))
    ctx.taxonomy = Taxonomy.load(cfg.taxonomy)
    df = ingest_all(cfg)
    df = normalize_stage(ctx, df)
    df = exclusions_stage(ctx, df)
    return df, cfg


def make_rc(cfg, **over) -> dict:
    rc = get_cfg(cfg)
    rc["primary_sources"] = rc["primary_sources"] or ["erp"]
    rc["secondary_sources"] = rc["secondary_sources"] or ["tool"]
    rc["history"] = {**rc["history"], "enabled": False}
    for k, v in over.items():
        rc[k] = {**rc[k], **v} if isinstance(rc.get(k), dict) else v
    for s in cfg.sources:
        if s.name in rc["secondary_sources"] and s.columns.get("default_tax_rate") is not None:
            rc["default_tax_rate"] = float(s.columns["default_tax_rate"])
    return rc


def dup_flags(df: pd.DataFrame) -> np.ndarray:
    return ((df["source"] == "tool") & (df["excl_code"] == "DUPLICATE")).to_numpy()


def confusion(truth: np.ndarray, pred: np.ndarray, w: np.ndarray, mask: np.ndarray) -> dict:
    t, p, w = truth[mask], pred[mask], np.abs(w[mask])
    out = {}
    for name, ww in (("n", np.ones(len(t))), ("zar", w)):
        tp = float(ww[t & p].sum()); fp = float(ww[~t & p].sum())
        fn = float(ww[t & ~p].sum()); tn = float(ww[~t & ~p].sum())
        out[name] = {"TP": round(tp), "FP": round(fp), "FN": round(fn), "TN": round(tn),
                     "correct": round(tp + tn), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return out


def core_route_holdout(df: pd.DataFrame, cfg, frac: float = 0.18, seeds=(1, 2, 3), core: Optional[dict] = None) -> dict:
    """Truth = flags of the unblanked run (core off). Blank PO on `frac` of PO-bearing tool rows (any partner status)."""
    truth_df, _ = reconcile_core(df, make_rc(cfg, core_route={"enabled": False}))
    truth = dup_flags(truth_df)
    elig_tool = ((df["source"] == "tool") & (df["excl_code"].fillna("") == "")).to_numpy()
    has_po = (df["po_raw"].astype(str).str.contains(r"45\d{8}")).to_numpy()
    w = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0).to_numpy()
    res: dict = {"seeds": {}}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cand = np.flatnonzero(elig_tool & has_po)
        pick = rng.choice(cand, size=int(frac * len(cand)), replace=False)
        d2 = df.copy()
        for c in ("po_raw", "po_number", "po_item"):     # normalised columns win over raw in the features
            if c in d2.columns:
                d2.loc[d2.index[pick], c] = ""
        mask = np.zeros(len(df), dtype=bool)
        mask[pick] = True
        row = {"n_blanked": int(mask.sum())}
        for label, en in (("off", False), ("on", True)):
            cr = {"enabled": en, **(core or {})}
            out, st = reconcile_core(d2, make_rc(cfg, core_route={**cr, "when": "no_po"}))
            pred = dup_flags(out)
            row[label] = {"blanked_rows": confusion(truth, pred, w, mask),
                          "po_bearing_rows": confusion(truth, pred, w, elig_tool & has_po),
                          "core_assigned": st.get("core_route_assigned", 0)}
        res["seeds"][seed] = row
    return res


def real_nopo_effect(df: pd.DataFrame, cfg, core: Optional[dict] = None) -> dict:
    """What the core route does on the real no-PO rows."""
    elig = ((df["source"] == "tool") & (df["excl_code"].fillna("") == "")).to_numpy()
    nopo = (df["source"] == "tool").to_numpy() & ~df["po_raw"].astype(str).str.contains(r"45\d{8}").to_numpy()
    w = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0).to_numpy()
    out = {}
    for label, en in (("off", False), ("on", True)):
        o, st = reconcile_core(df, make_rc(cfg, core_route={"enabled": en, "when": "no_po", **(core or {})}))
        d = dup_flags(o)
        out[label] = {"no_po_eligible_rows": int((nopo & elig).sum()), "flagged_dup": int((d & nopo & elig).sum()),
                      "flagged_zar": round(float(np.abs(w[d & nopo & elig]).sum())),
                      "total_tool_dup": int(d.sum()), "core_assigned": st.get("core_route_assigned", 0)}
    return out


def pseudo_a2(df: pd.DataFrame, cfg, seed: int = 1, po_frac: float = 0.18, other_tool_frac: float = 0.25,
              **rc_over) -> dict:
    """A2 simulation: emit ERP rows of 18% of the POs + their tool rows + a random quarter of the other tool rows;
    everything else becomes `*_hist` reference rows. Agreement of the emitted tool rows' DUPLICATE flags with the
    full-data run (history-aware matching, D-02)."""
    full, _ = reconcile_core(df, make_rc(cfg, **rc_over))
    full_dup = dup_flags(full)
    rng = np.random.default_rng(seed)
    po = df["po_number"].astype(str).to_numpy() if "po_number" in df.columns else df["po_raw"].astype(str).to_numpy()
    pos = np.unique(po[po != ""])
    a2_pos = set(rng.choice(pos, size=int(po_frac * len(pos)), replace=False))
    in_a2po = np.array([p in a2_pos for p in po])
    is_tool = (df["source"] == "tool").to_numpy()
    emit = np.where(is_tool, in_a2po | (rng.random(len(df)) < other_tool_frac), in_a2po)
    d2 = df.copy()
    ref = ~emit
    d2.loc[ref, "source"] = d2.loc[ref, "source"] + "_hist"
    d2.loc[ref, "source_role"] = "reference"
    rc = make_rc(cfg, **rc_over)
    rc["history"] = {**rc["history"], "enabled": True}
    out, st = reconcile_core(d2, rc)
    sel = emit & is_tool
    pred = ((out["source"] == "tool") & (out["excl_code"] == "DUPLICATE")).to_numpy()
    agree = float((pred[sel] == full_dup[sel]).mean())
    return {"emitted_tool_rows": int(sel.sum()), "agreement_with_full_run": round(agree, 4),
            "history_matches": int((out["match_type"] == "history").sum()),
            "ref_rows_flagged": int(((out["source_role"] == "reference") & (out["excl_code"].fillna("") != "")).sum()
                                    - ((df["excl_code"].fillna("") != "") & ref).sum())}


def partial_effect(df: pd.DataFrame, cfg) -> dict:
    """Real A: what partial-capacity adds (count/ZAR) and proof that no existing flag changes."""
    w = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0).to_numpy()
    off, st0 = reconcile_core(df, make_rc(cfg, partial_capacity={"enabled": False}))
    on, st1 = reconcile_core(df, make_rc(cfg, partial_capacity={"enabled": True}))
    d0, d1 = dup_flags(off), dup_flags(on)
    new = d1 & ~d0
    same_ref = bool((off.loc[d0, "match_ref"].to_numpy() == on.loc[d0, "match_ref"].to_numpy()).all())
    return {"flagged_off": int(d0.sum()), "flagged_on": int(d1.sum()), "newly_flagged": int(new.sum()),
            "newly_flagged_zar": round(float(np.abs(w[new]).sum())), "lost_flags": int((d0 & ~d1).sum()),
            "existing_matches_unchanged": same_ref,
            "po_in_pool_unmatched_off": st0.get("po_in_pool_unmatched"),
            "po_in_pool_unmatched_on": st1.get("po_in_pool_unmatched"),
            "partial_assigned": st1.get("partial_assigned")}


def partial_hidden_holdout(df: pd.DataFrame, cfg, frac: float = 0.18, seeds=(1, 2, 3)) -> dict:
    """Delete the ERP receipt of a random share of exactly-matched tool rows: those rows become true tool-only for
    the visible data. DUPLICATE flags they still receive are false positives (baseline = partial off)."""
    base, _ = reconcile_core(df, make_rc(cfg, partial_capacity={"enabled": False}))
    w_all = pd.to_numeric(df["spend_rc"], errors="coerce").fillna(0).to_numpy()
    matched = np.flatnonzero((base["match_type"] == "erp").to_numpy() & (base["source"] == "tool").to_numpy())
    rid_to_pos = {r: i for i, r in enumerate(df["record_id"].to_numpy())}
    res = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pick = rng.choice(matched, size=int(frac * len(matched)), replace=False)
        hide_erp = {rid_to_pos[r] for r in base["match_ref"].to_numpy()[pick] if r in rid_to_pos}
        keep = np.ones(len(df), dtype=bool)
        keep[list(hide_erp)] = False
        d2 = df[keep]
        row = {"n_hidden_tool_rows": len(pick)}
        pos_in_d2 = {r: i for i, r in enumerate(d2["record_id"].to_numpy())}
        idx = np.array([pos_in_d2[r] for r in df["record_id"].to_numpy()[pick]])
        w = w_all[pick]
        for label, en in (("off", False), ("on", True)):
            o, st = reconcile_core(d2, make_rc(cfg, partial_capacity={"enabled": en}))
            fl = dup_flags(o)[idx]
            row[label] = {"false_dup_n": int(fl.sum()), "false_dup_zar": round(float(np.abs(w[fl]).sum())),
                          "pct_n": round(float(fl.mean()), 4), "partial_assigned": st.get("partial_assigned", 0)}
        row["incremental_FP_n"] = row["on"]["false_dup_n"] - row["off"]["false_dup_n"]
        row["incremental_FP_zar"] = row["on"]["false_dup_zar"] - row["off"]["false_dup_zar"]
        res[seed] = row
    return res
