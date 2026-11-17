"""Label loader (config `labels:`): DataFrame keyed by record_id; cached in ctx.state['labels']. Never raises."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("spendcube.labels")

CANON = ["l1", "l2", "l3", "l4", "supplier_key", "item_key", "include", "exclusion_reason"]


def load_labels(ctx) -> pd.DataFrame | None:
    """Columns: record_id, any of l1..l4/supplier_key/item_key/include/exclusion_reason, plus leaf_id when a
    taxonomy is available and the label path resolves to a leaf. None when no labels are configured/readable."""
    if "labels" in ctx.state:
        return ctx.state["labels"]
    out = None
    try:
        lc = ctx.cfg.labels
        if lc is not None:
            from .config.loader import resolve_files
            from .ingest.readers import read_many
            files = resolve_files(lc.file, required=False)
            if files:
                raw = read_many(files, lc.read.delimiter, lc.read.encoding, lc.read.sheet)
                if lc.join_on not in raw.columns:
                    raise ValueError(f"join column {lc.join_on} not in labels file")
                out = pd.DataFrame({"record_id": lc.id_prefix + raw[lc.join_on].astype(str).str.strip()})
                for canon, col in lc.columns.items():
                    if col in raw.columns:
                        out[canon] = raw[col].astype(str).str.strip()
                    else:
                        log.warning("labels: column %s missing", col)
                if "include" in out.columns:
                    out["include"] = out["include"].str.casefold().map(
                        {"1": "1", "true": "1", "yes": "1", "0": "0", "false": "0", "no": "0"}).fillna(out["include"])
                hf = (lc.model_extra or {}).get("holdout_frac")
                if hf:   # evaluation only: hide an id-gap-run holdout of labels (kept in ctx.state / run dir)
                    from .eval.supervised_eval import _gap_run_holdout
                    held = _gap_run_holdout(out["record_id"], float(hf), int((lc.model_extra or {}).get("holdout_seed", 7)))
                    ctx.state["labels_heldout"] = out[held].copy()
                    out = out[~held].copy()
                    log.info("labels: %d rows held out for evaluation", int(held.sum()))
                tax = getattr(ctx, "taxonomy", None)
                lv = [c for c in ("l1", "l2", "l3", "l4") if c in out.columns]
                if tax is not None and lv:
                    cache: dict[tuple, str] = {}

                    def leaf_of(vals: tuple) -> str:
                        if vals not in cache:
                            lf = tax.leaf_by_names(list(vals))
                            cache[vals] = lf.id if lf else ""
                        return cache[vals]
                    out["leaf_id"] = [leaf_of(v) for v in zip(*(out[c] for c in lv))]
                    unm = int(((out["leaf_id"] == "") & (out[lv[0]] != "")).sum())
                    if unm:
                        log.warning("labels: %d rows with no matching tree leaf", unm)
                log.info("labels loaded: %d rows, columns %s", len(out), list(out.columns))
    except Exception as e:  # never raise
        log.warning("labels not loaded: %s", e)
        out = None
    ctx.state["labels"] = out
    return out
