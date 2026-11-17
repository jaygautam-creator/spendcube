"""`eval entities` (MASTER_ARCHITECTURE 7.3): pairwise P/R/F1 + B-cubed for supplier and item grouping.

(a) A ERP rows, unsupervised (alias maps OFF) vs golden keys; (a2) alias maps ON using a run-structured 18%
holdout (maps built from the other 82%, scored only on the holdout); (b) A tool rows against pseudo-truth =
golden keys of the matched ERP row (the noisy-names / C-like test). Suppliers are also reported with the
catch-all golden bucket removed.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config.loader import load_config
from ..context import RunContext
from ..entities.items import build_item_alias, resolve_items, text_key
from ..entities.stage import _prep_rows
from ..entities.suppliers import build_supplier_alias, resolve_suppliers
from ..ingest.canonical import ingest_all
from ..labels import load_labels

log = logging.getLogger("spendcube.eval")


def cluster_metrics(pred: pd.Series, gold: pd.Series) -> dict:
    """Pairwise precision/recall/F1 and B-cubed precision/recall/F1 of a predicted grouping vs a gold one."""
    d = pd.DataFrame({"p": pred.to_numpy(), "g": gold.to_numpy()})
    N = len(d)
    if N == 0:
        return {"n": 0}
    nij = d.groupby(["p", "g"]).size()
    ni = d.groupby("p").size()
    mj = d.groupby("g").size()
    pairs = lambda s: float((s * (s - 1) / 2).sum())
    tp, pp, gp = pairs(nij), pairs(ni), pairs(mj)
    P = tp / pp if pp else 1.0
    R = tp / gp if gp else 1.0
    f1 = 2 * P * R / (P + R) if P + R else 0.0
    ij = nij.reset_index(name="n")
    bp = float((ij["n"] ** 2 / ij["p"].map(ni)).sum() / N)
    br = float((ij["n"] ** 2 / ij["g"].map(mj)).sum() / N)
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    return {"n": N, "pred_groups": int(len(ni)), "gold_groups": int(len(mj)),
            "pair_P": round(P, 4), "pair_R": round(R, 4), "pair_F1": round(f1, 4),
            "b3_P": round(bp, 4), "b3_R": round(br, 4), "b3_F1": round(bf, 4)}


def prepare(config_path: str, input_dir: Optional[str] = None):
    """Ingest + normalize + exclusions + reconcile (build mode) -> (ctx, df). No LLM, no entities."""
    from ..exclusions.stage import exclusions_stage
    from ..normalize.stage import normalize_stage
    from ..reconcile.stage import reconcile_stage
    from ..taxonomy.tree import Taxonomy
    cfg = load_config(config_path, input_dir=input_dir)
    ctx = RunContext(cfg=cfg, run_dir=Path(tempfile.mkdtemp(prefix="ent_eval_")))
    ctx.taxonomy = Taxonomy.load(cfg.taxonomy)
    df = ingest_all(cfg)
    for fn in (normalize_stage, exclusions_stage, reconcile_stage):
        df = fn(ctx, df)
    return ctx, df


def _catch_all_key(df: pd.DataFrame, gold: pd.Series, col: str = "supplier_core") -> Optional[str]:
    """Golden key that bundles the most distinct names/materials (the client master's 'unmapped' bucket)."""
    d = pd.DataFrame({"g": gold, "c": df[col]})
    d = d[(d.g != "") & (d.c != "")]
    if d.empty:
        return None
    nn = d.groupby("g")["c"].nunique()
    top = nn.idxmax()
    return top if nn.max() >= 10 else None


def _holdout_ids(ids: list[str], frac: float = 0.18, seed: int = 7) -> set[str]:
    """Run-structured holdout: contiguous runs of 1-3 consecutive ids (mimics the A2 id gaps)."""
    rng = np.random.default_rng(seed)
    order = sorted(ids)
    n, i, hold = len(order), 0, set()
    target = int(frac * n)
    # place runs at random start points until the target share is reached
    while len(hold) < target:
        start = int(rng.integers(0, n))
        ln = int(rng.choice([1, 2, 3], p=[0.73, 0.18, 0.09]))
        hold.update(order[start:start + ln])
    return hold


def _pseudo_truth(df: pd.DataFrame, gold_sup: pd.Series, gold_item: pd.Series):
    """Tool rows -> golden keys of the ERP row they matched (re-exports follow their canonical)."""
    ref = df["match_ref"].fillna("").astype(str)
    mt = df["match_type"].fillna("none").astype(str)
    rid = df["record_id"]
    by_sup = pd.Series(gold_sup.to_numpy(), index=rid.to_numpy())
    by_item = pd.Series(gold_item.to_numpy(), index=rid.to_numpy())
    mt_by = pd.Series(mt.to_numpy(), index=rid.to_numpy())
    ref_by = pd.Series(ref.to_numpy(), index=rid.to_numpy())
    tgt = ref.where(mt.isin(["erp", "reexport"]), "")
    for _ in range(2):         # re-export -> canonical -> ERP
        hop = tgt.map(mt_by).eq("erp") | ~tgt.map(mt_by).isin(["erp", "reexport"])
        nxt = tgt.map(ref_by).fillna("")
        tgt = tgt.where(hop | (tgt == ""), nxt)
    return tgt.map(by_sup), tgt.map(by_item)


def run_entities_eval(config_path: str = "configs/A.yaml", input_dir: Optional[str] = None,
                      holdout_frac: float = 0.18) -> dict:
    ctx, df = prepare(config_path, input_dir)
    cfg = ctx.cfg.entities or {}
    sup_cfg, item_cfg = cfg.get("suppliers", {}) or {}, cfg.get("items", {}) or {}
    lab = load_labels(ctx)
    if lab is None or "supplier_key" not in lab.columns:
        return {"error": "no labels with supplier_key/item_key configured"}
    lab = lab.drop_duplicates("record_id").set_index("record_id")
    gold_sup = df["record_id"].map(lab["supplier_key"]).fillna("")
    gold_item = df["record_id"].map(lab["item_key"]).fillna("") if "item_key" in lab.columns else pd.Series("", index=df.index)
    erp = (df["source_role"] == "primary").to_numpy()
    tool = (df["source_role"] == "secondary").to_numpy()
    res: dict = {"rows_erp": int(erp.sum()), "rows_tool": int(tool.sum())}
    cat = _catch_all_key(df[erp], gold_sup[erp])
    cat_i = _catch_all_key(df[erp], gold_item[erp], "material_id")
    res["catch_all_present"] = {"supplier": cat is not None, "item": cat_i is not None}

    def item_metrics(ik_: pd.Series, mask, gold: pd.Series) -> dict:
        m = np.asarray(mask) & (gold != "").to_numpy()
        out = {"with_catch_all": cluster_metrics(ik_[m], gold[m])}
        if cat_i:
            m2 = m & (gold != cat_i).to_numpy()
            out["without_catch_all"] = cluster_metrics(ik_[m2], gold[m2])
        return out

    def sup_metrics(sk: pd.Series, mask, gold: pd.Series) -> dict:
        m = np.asarray(mask) & (gold != "").to_numpy()
        out = {"with_catch_all": cluster_metrics(sk[m], gold[m])}
        if cat:
            m2 = m & (gold != cat).to_numpy()
            out["without_catch_all"] = cluster_metrics(sk[m2], gold[m2])
        return out

    # ---- (a)+(b) unsupervised: alias maps off, no inheritance; ERP and tool clustered jointly like the pipeline
    sk, _, sst = resolve_suppliers(df, sup_cfg)
    ik, _, ist = resolve_items(df, item_cfg, sk, None, None, df["source_role"] == "primary")
    p_sup, p_item = _pseudo_truth(df, gold_sup, gold_item)
    p_sup, p_item = p_sup.fillna(""), p_item.fillna("")
    res["unsupervised"] = {
        "supplier_stats": {k: v for k, v in sst.items()},
        "item_stats": {k: v for k, v in ist.items() if k != "generic_materials"},
        "generic_materials": ist.get("generic_materials"),
        "erp_supplier": sup_metrics(sk, erp, gold_sup),
        "erp_item": item_metrics(ik, erp, gold_item),
        "tool_supplier_vs_matched_erp_golden": sup_metrics(sk, tool, p_sup),
        "tool_item_vs_matched_erp_golden": item_metrics(ik, tool, p_item),
    }

    # ---- (a2) alias maps ON, run-structured holdout (maps from the other 82%, score the 18%)
    erp_ids = df.loc[erp, "record_id"].tolist()
    hold = _holdout_ids(erp_ids, holdout_frac)
    is_hold = df["record_id"].isin(hold).to_numpy() & erp
    is_train = ~df["record_id"].isin(hold).to_numpy() & erp
    train = df[is_train].copy()
    train["gold_supplier"] = gold_sup[is_train]
    train["gold_item"] = gold_item[is_train]
    train["text_key"] = [text_key(x) for x in train["desc_norm"].fillna("")]
    a_core, a_vend = build_supplier_alias(train, 0.9)
    a_mat, a_txt = build_item_alias(train, 0.9)
    H = df[is_hold]
    hs, _, _ = resolve_suppliers(H, sup_cfg, a_core, a_vend)
    hi, _, hst = resolve_items(H, item_cfg, hs, a_mat, a_txt, H["source_role"] == "primary")
    hg_s, hg_i = gold_sup[is_hold], gold_item[is_hold]
    hs0, _, _ = resolve_suppliers(H, sup_cfg)                       # same rows, aliases off (before/after)
    hi0, _, _ = resolve_items(H, item_cfg, hs0, None, None, H["source_role"] == "primary")
    mi = (hg_i != "").to_numpy()
    res["alias_on_holdout"] = {
        "holdout_rows": int(is_hold.sum()), "alias_sizes": {"supplier_cores": len(a_core), "vendors": len(a_vend),
                                                            "materials": len(a_mat), "texts": len(a_txt)},
        "supplier_alias_on": sup_metrics(hs, np.ones(len(H), bool), hg_s) if False else _sup_h(hs, hg_s, cat, H),
        "supplier_alias_off": _sup_h(hs0, hg_s, cat, H),
        "item_alias_on": item_metrics(hi, np.ones(len(H), bool), hg_i),
        "item_alias_off": item_metrics(hi0, np.ones(len(H), bool), hg_i),
        "item_methods": hst.get("methods"),
    }
    # share of holdout rows whose golden supplier key / item key is covered by the alias maps
    res["alias_on_holdout"]["supplier_golden_covered"] = round(float(hg_s.isin(set(a_core.values())).mean()), 4)
    res["alias_on_holdout"]["item_golden_covered"] = round(float(hg_i[mi].isin(set(a_mat.values())).mean()), 4)
    return res


def _sup_h(sk, gold, cat, H) -> dict:
    m = (gold != "").to_numpy()
    out = {"with_catch_all": cluster_metrics(sk[m], gold[m])}
    if cat:
        m2 = m & (gold != cat).to_numpy()
        out["without_catch_all"] = cluster_metrics(sk[m2], gold[m2])
    return out
