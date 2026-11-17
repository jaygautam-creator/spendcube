"""Stage 4: reconcile (invoice tool -> receipts/ERP), re-exports, history (D-02, D-03, D-19, D-21).

Adds: dup_group, match_ref, match_type (erp|reexport|history|none), match_score.
Sets excl_code=DUPLICATE (+excl_conf) on emitted secondary rows that match, unless an earlier stage already
excluded the row (precedence CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE).
Never drops rows. Runs only when config.reconciliation is present and enabled.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ..schema_cols import ROLE_REFERENCE
from .features import build_features, get_cfg
from .match import PoolIndex, assign_greedy, generate_and_score, orphan_rescue, partial_capacity_match
from .reexport import cluster_reexports, propagate_po

log = logging.getLogger("spendcube.reconcile")


def _score_conf(score: float) -> float:
    return float(min(0.99, 0.55 + 0.045 * score))


def _init_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c, v in (("dup_group", ""), ("match_ref", ""), ("match_type", "none"), ("match_score", 0.0)):
        if c not in df.columns:
            df[c] = v
    return df


def reconcile_core(df: pd.DataFrame, rc: dict) -> tuple[pd.DataFrame, dict]:
    """Pure function used by the stage and by the evaluation harness. df needs the canonical raw columns,
    `source`, `source_role`, `record_id`. Returns (df with result columns, stats)."""
    t0 = time.time()
    df = _init_cols(df.copy())
    if "excl_code" not in df.columns:
        df["excl_code"] = ""
        df["excl_detail"] = ""
        df["excl_conf"] = 1.0
    stats: dict = {}
    src = df["source"].astype(str).to_numpy()
    role = df["source_role"].astype(str).to_numpy()
    h = rc["history"]
    prim_names = set(rc["primary_sources"] or [])
    sec_names = set(rc["secondary_sources"] or [])
    hist_on = h.get("enabled") is True
    if hist_on:
        sfx = h.get("suffix", "_hist")
        prim_names |= {n + sfx for n in list(prim_names)}
        sec_names |= {n + sfx for n in list(sec_names)}
    is_ref = role == ROLE_REFERENCE
    is_pool = np.isin(src, list(prim_names)) & (~is_ref | hist_on)
    is_tool = np.isin(src, list(sec_names)) & (~is_ref | hist_on)
    stats.update(pool_rows=int(is_pool.sum()), tool_rows=int(is_tool.sum()))
    if not is_tool.any():
        stats["skipped"] = "no tool rows"
        return df, stats

    rid = df["record_id"].astype(str).to_numpy()
    f = build_features(df, rc, pd.Series(is_tool, index=df.index))
    elig = f["f_eligible"].to_numpy(dtype=bool)
    tool_rows = np.flatnonzero(is_tool & elig)
    pool_rows = np.flatnonzero(is_pool)

    # R2: re-export clusters inside the tool (including history rows)
    cluster, canon, rstats = cluster_reexports(f, rid, tool_rows, rc)
    stats["reexport"] = rstats
    stats["reexport"]["po_propagated"] = propagate_po(f, canon, tool_rows)
    claim = np.array(sorted({int(canon[r]) for r in tool_rows}), dtype=np.int64)

    # R3-R5: tool canonical -> ERP pool
    px = PoolIndex(f, pool_rows)
    cands, cstats = generate_and_score(f, claim, px, rc, routes=("po",))
    assigned, taken = assign_greedy(cands)
    core_assigned: set[int] = set()
    if rc["core_route"].get("enabled"):
        # phase 2 (guarded): only claimants the PO route could not place, only pool rows it left unclaimed
        rest = np.array([int(t) for t in claim if int(t) not in assigned], dtype=np.int64)
        c2, cs2 = generate_and_score(f, rest, px, rc, routes=("core",), exclude_pool=set(taken))
        before = set(assigned)
        assign_greedy(c2, assigned, taken)
        core_assigned = set(assigned) - before
        cands = cands + c2
        for k, v in cs2.items():
            cstats[k] = cstats.get(k, 0) + v
        stats["core_route_assigned"] = len(core_assigned)
    stats["candidates"] = {"pairs": len(cands), **cstats}
    # R6: orphan rescue
    rescued: dict[int, int] = {}
    if rc["rescue"].get("enabled", True):
        rescued = orphan_rescue(f, cands, assigned, taken, claim, rc)
    stats["assigned"] = len(assigned)
    stats["rescued"] = len(rescued)

    # D-22 partial-quantity capacity match (flag)
    partial_set: set[int] = set()
    partial_pool: set[int] = set()
    if rc["partial_capacity"].get("enabled"):
        un = np.array([int(t) for t in claim if int(t) not in assigned and int(t) not in rescued], dtype=np.int64)
        part = partial_capacity_match(f, un, px, taken, rc)
        for t, (e, sc) in part.items():
            assigned[t] = (e, sc)
            partial_set.add(t)
            partial_pool.add(e)
        stats["partial_assigned"] = len(part)

    # R7 aggregate (flag)
    if rc["aggregate"].get("enabled"):
        from .aggregate import aggregate_matches
        unmatched = np.array([int(t) for t in claim if t not in assigned and t not in rescued], dtype=np.int64)
        extra = aggregate_matches(f, unmatched, px, rc, set(taken) | partial_pool)
        for t, (e, s) in extra.items():
            assigned[t] = (e, s)
        stats["aggregate_assigned"] = len(extra)

    mt = df["match_type"].to_numpy(dtype=object).copy()
    mref = df["match_ref"].to_numpy(dtype=object).copy()
    mscore = df["match_score"].to_numpy(dtype=float).copy()
    dgrp = df["dup_group"].to_numpy(dtype=object).copy()
    ref_flag = is_ref

    def tag(i: int) -> str:
        return ("hist:" if ref_flag[i] else "") + str(rid[i])

    group_of_canon: dict[int, str] = {}
    for t in claim:
        t = int(t)
        if t in assigned:
            group_of_canon[t] = "G:" + tag(int(pool_rows[assigned[t][0]]))
    for t, u in rescued.items():
        group_of_canon[t] = group_of_canon.get(u) or ("G:" + tag(u))
    for t in claim:
        group_of_canon.setdefault(int(t), "G:" + tag(int(t)))

    for r in tool_rows:
        r = int(r)
        c = int(canon[r])
        if c in assigned:
            e_local, sc = assigned[c]
            e = int(pool_rows[e_local])
            if r == c:
                mt[r], mref[r], mscore[r] = ("history" if ref_flag[e] else "erp"), tag(e), sc
            else:
                mt[r] = "history" if (ref_flag[e] or ref_flag[c]) else "reexport"
                mref[r], mscore[r] = tag(c), sc
        elif c in rescued:
            u = rescued[c]
            mt[r] = "history" if ref_flag[u] else "reexport"
            mref[r], mscore[r] = tag(u), 6.0
        elif c != r:
            mt[r] = "history" if ref_flag[c] else "reexport"
            mref[r], mscore[r] = tag(c), 8.0
        dgrp[r] = group_of_canon.get(c, "G:" + tag(c))
    for t, (e_local, _) in assigned.items():       # matched receipt rows share the group id
        e = int(pool_rows[e_local])
        if not dgrp[e]:
            dgrp[e] = group_of_canon[t]
    df["match_type"], df["match_ref"], df["match_score"], df["dup_group"] = mt, mref, mscore, dgrp

    # exclusion: emitted tool rows that duplicate something (D-03: only if not already excluded)
    if "excl_code" not in df.columns:
        df["excl_code"] = ""
        df["excl_detail"] = ""
        df["excl_conf"] = 1.0
    canon_keys = {int(r) for r in tool_rows}
    is_dupmatch = pd.Series(mt, index=df.index).isin(["erp", "reexport", "history"]).to_numpy()
    free = df["excl_code"].fillna("").astype(str).to_numpy() == ""
    sel = (~is_ref) & is_tool & is_dupmatch & free & elig
    df.loc[sel, "excl_code"] = "DUPLICATE"
    df.loc[sel, "excl_detail"] = [f"{t}:{r}" for t, r in zip(df.loc[sel, "match_type"], df.loc[sel, "match_ref"])]
    df.loc[sel, "excl_conf"] = [_score_conf(s) for s in df.loc[sel, "match_score"]]
    if partial_set:
        pmask = sel & np.array([int(canon[i]) in partial_set if is_tool[i] and elig[i] and i in canon_keys else False
                                for i in range(len(df))])
        df.loc[pmask, "excl_detail"] = "partial_" + df.loc[pmask, "excl_detail"].astype(str)
        stats["partial_rows_flagged"] = int(pmask.sum())

    emitted_tool = is_tool & ~is_ref
    stats["match_type_counts"] = {k: int(v) for k, v in pd.Series(mt[emitted_tool]).value_counts().items()}
    stats["duplicates_flagged"] = int(sel.sum())
    po = f["f_po"].to_numpy(dtype=object)
    stats["po_in_pool_unmatched"] = sum(
        1 for t in claim if int(t) not in assigned and int(t) not in rescued and not ref_flag[t]
        and po[t] and po[t] in px.po_index)
    stats["seconds"] = round(time.time() - t0, 1)
    return df, stats


def reconcile_stage(ctx, df: pd.DataFrame) -> pd.DataFrame:
    if getattr(ctx.cfg, "reconciliation", None) is None:
        return _init_cols(df)
    rc = get_cfg(ctx.cfg)
    if not rc.get("enabled", True):
        return _init_cols(df)
    if not rc["primary_sources"]:
        rc["primary_sources"] = [s.name for s in ctx.cfg.sources if s.role == "primary"]
    if not rc["secondary_sources"]:
        rc["secondary_sources"] = [s.name for s in ctx.cfg.sources if s.role == "secondary"]
    for s in ctx.cfg.sources:      # tool default VAT comes from the tool source config
        if s.name in rc["secondary_sources"] and s.columns.get("default_tax_rate") is not None:
            rc["default_tax_rate"] = float(s.columns["default_tax_rate"])
            break
    n0 = len(df)
    ref = None
    if rc["history"].get("enabled") in (True, "auto"):
        ref = _load_history(ctx, rc)
        if ref is not None:
            ids = set(df.loc[df["source"].isin(rc["primary_sources"]), "record_id"])
            sfx = rc["history"].get("suffix", "_hist")
            rp = ref.loc[ref["source"].isin([n + sfx for n in rc["primary_sources"]]), "record_id"]
            overlap = len(set(rp) & ids) / max(1, len(rp))
            if overlap > 0.5:
                ctx.warn(f"reconcile: history rows overlap this run's rows ({overlap:.0%}); history off")
                ref = None
    rc["history"]["enabled"] = ref is not None
    big = pd.concat([df, ref], ignore_index=True) if ref is not None else df
    big, stats = reconcile_core(big, rc)
    out = big.iloc[:n0].copy() if ref is not None else big
    out.index = df.index
    if ref is not None:
        stats["history_rows"] = len(ref)
    for k, v in stats.items():
        ctx.note("reconcile", k, v)
    log.info("reconcile: %s", {k: v for k, v in stats.items()
                               if k in ("match_type_counts", "po_in_pool_unmatched", "history_rows", "seconds")})
    return out


def _load_history(ctx, rc: dict) -> pd.DataFrame | None:
    """Ingest the configured history files as reference rows. None when history does not apply."""
    from ..config.loader import resolve_files
    from ..ingest.canonical import ingest_source
    h = rc["history"]
    if not h.get("files"):
        return None
    sfx = h.get("suffix", "_hist")
    frames = []
    by_name = {s.name: s for s in ctx.cfg.sources}
    for name, pattern in h["files"].items():
        src = by_name.get(name)
        if src is None:
            ctx.warn(f"reconcile.history: unknown source '{name}'")
            continue
        try:
            own = {str(p) for p in resolve_files(src.file, required=False)}
            hist = {str(p) for p in resolve_files(pattern, required=False)}
        except Exception as e:
            ctx.warn(f"reconcile.history: cannot resolve files for {name}: {e}")
            continue
        if not hist:
            ctx.warn(f"reconcile.history: no files for {name} ({pattern}); skipped")
            continue
        if hist == own:
            ctx.log.info("reconcile.history: %s history files equal the run's own files -> skipped (build run)", name)
            continue
        hs = src.model_copy(update={"file": pattern, "name": name + sfx, "role": "reference", "required": False})
        try:
            frames.append(ingest_source(ctx.cfg, hs, limit=None, row_offset=-1))
        except Exception as e:
            ctx.warn(f"reconcile.history: ingest of {name} history failed: {type(e).__name__}: {e}")
    if not frames:
        return None
    ref = pd.concat(frames, ignore_index=True)
    ref["source_role"] = ROLE_REFERENCE
    return ref
