"""R2: re-export clustering inside the invoice tool (D-21).

Copies of one invoice share supplier core + gross + quantity + invoice-number digits. Different
invoice numbers with the same price are distinct purchases (bulk lines repeat prices), so the
architecture's invoice-free "desc+date" link is OFF by default (config reexport.link_without_invoice).
Blocking: groupby on the exact key (no pairwise comparison across blocks); pairwise checks only inside
a key group, capped by reexport.block_cap.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

log = logging.getLogger("spendcube.reconcile")


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 1.0          # no text evidence against the link
    return fuzz.ratio(a, b) / 100.0


def _date_ok(d1, d2, days: int) -> bool:
    if pd.isna(d1) or pd.isna(d2):
        return True
    return abs((d1 - d2) / np.timedelta64(1, "D")) <= days


def propagate_po(f: pd.DataFrame, canon: np.ndarray, rows: np.ndarray) -> int:
    """Copies of one invoice may differ in PO format or drop the PO entirely. Give the canonical row the PO
    (and PO item) of the first member that has one, so the cluster matches as a whole. Returns #filled."""
    po = f["f_po"].to_numpy(dtype=object).copy()
    poi = f["f_poi"].to_numpy(dtype=float).copy()
    filled = 0
    for r in rows:
        c = int(canon[r])
        if c == r:
            continue
        if po[c] == "" and po[r] != "":
            po[c] = po[r]
            filled += 1
            if np.isnan(poi[c]) and not np.isnan(poi[r]):
                poi[c] = poi[r]
        elif po[c] == po[r] and po[c] != "" and np.isnan(poi[c]) and not np.isnan(poi[r]):
            poi[c] = poi[r]
    f["f_po"] = po
    f["f_poi"] = poi
    return filled


def cluster_reexports(f: pd.DataFrame, ids: np.ndarray, rows: np.ndarray, rcfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cluster tool-like rows.

    f      feature frame (aligned to df, positional)
    ids    record ids aligned to df (for deterministic tie-breaking)
    rows   positional indices of eligible tool-like rows (emitted + reference)
    Returns (cluster_id[len(df)], canon[len(df)], stats) with -1 for rows not in any cluster; canon[i]==i
    for singleton/canonical rows.
    """
    n = len(f)
    cluster = np.full(n, -1, dtype=np.int64)
    canon = np.arange(n, dtype=np.int64)
    rc = rcfg["reexport"]
    cap, days, thr = int(rc["block_cap"]), int(rc["date_days"]), float(rc["desc_sim"])
    stats = {"groups": 0, "rows_in_groups": 0, "overflow_blocks": 0, "reexport_rows": 0}
    if len(rows) == 0:
        return cluster, canon, stats

    sub = f.iloc[rows]
    key = ["f_core", "f_gross_r", "f_qty", "f_invd"]
    link_blank = bool(rc.get("link_without_invoice", False))
    has_inv = sub["f_invd"].astype(str) != ""
    work = sub[has_inv] if not link_blank else sub
    groups = work.groupby(key, dropna=False, sort=False).indices
    next_cid = 0
    descs = f["f_desc"].to_numpy()
    dates = f["f_date"].to_numpy()
    rank = f["f_rank"].to_numpy()
    wrows = rows[np.flatnonzero(has_inv.to_numpy())] if not link_blank else rows
    for _, loc in groups.items():
        if len(loc) < 2:
            continue
        members = wrows[loc]
        if len(members) > cap:
            stats["overflow_blocks"] += 1
            members = members[:cap]
        # union-find inside the exact-key group
        parent = {int(m): int(m) for m in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ml = [int(m) for m in members]
        for i in range(len(ml)):
            for j in range(i + 1, len(ml)):
                a, b = ml[i], ml[j]
                if find(a) == find(b):
                    continue
                if _sim(descs[a], descs[b]) >= thr and _date_ok(dates[a], dates[b], days):
                    parent[find(a)] = find(b)
        comp: dict[int, list[int]] = {}
        for m in ml:
            comp.setdefault(find(m), []).append(m)
        for mem in comp.values():
            if len(mem) < 2:
                continue
            mem.sort(key=lambda r: (-rank[r], str(ids[r])))
            c = mem[0]
            for r in mem:
                cluster[r] = next_cid
                canon[r] = c
            next_cid += 1
            stats["groups"] += 1
            stats["rows_in_groups"] += len(mem)
            stats["reexport_rows"] += len(mem) - 1
    return cluster, canon, stats
