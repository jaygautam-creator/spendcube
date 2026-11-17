"""R3-R6: candidate generation, additive scoring, one-to-one greedy assignment, orphan rescue.

Claimants are canonical invoice-tool rows (emitted + reference); the pool holds receipt-like rows
(emitted primary + reference primary). Everything is dict/searchsorted driven; no O(n^2) work.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

log = logging.getLogger("spendcube.reconcile")


class PoolIndex:
    def __init__(self, f: pd.DataFrame, pool: np.ndarray):
        self.pool = pool
        sub = f.iloc[pool]
        self.net = sub["f_net"].to_numpy(dtype=float)
        self.qty = sub["f_qty"].to_numpy(dtype=float)
        self.poi = sub["f_poi"].to_numpy(dtype=float)
        self.core = sub["f_core"].to_numpy(dtype=object)
        self.desc = sub["f_desc"].to_numpy(dtype=object)
        self.pday = _days(sub["f_pdate"]) if "f_pdate" in sub.columns else np.full(len(sub), np.nan)
        self.po_index: dict[str, np.ndarray] = {}
        po = sub["f_po"].astype(str)
        for k, loc in po[po != ""].groupby(po[po != ""], sort=False).indices.items():
            self.po_index[k] = loc
        self.core_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        valid = ~np.isnan(self.net) & (self.core != "")
        pos = np.flatnonzero(valid)
        if len(pos):
            cs = pd.Series(self.core[pos])
            for k, loc in cs.groupby(cs, sort=False).indices.items():
                real = pos[loc]
                order = np.argsort(self.net[real], kind="stable")
                self.core_index[k] = (self.net[real][order], real[order])


def _days(dt: pd.Series) -> np.ndarray:
    """datetime series -> float day numbers (NaN for NaT)."""
    d = pd.to_datetime(dt, errors="coerce")
    out = (d - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=float, na_value=np.nan)
    return out


def _tol(val: np.ndarray | float, rc: dict) -> np.ndarray | float:
    t = rc["tolerance"]
    return t["abs"] + t["rel"] * np.abs(val)


def _qty_eq(a: float, b: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def generate_and_score(f: pd.DataFrame, claim: np.ndarray, px: PoolIndex, rc: dict,
                       routes: tuple = ("po", "core"), exclude_pool: set | None = None
                       ) -> tuple[list[tuple[float, int, int]], dict]:
    """Return candidate triples (score, claimant_row, pool_local_idx) above min_score.

    routes: "po" = same-PO candidates; "core" = supplier+amount without PO evidence (guarded: pool rows in
    `exclude_pool` (already claimed by the PO route) are never offered, |net|<=min_abs ignored, dates must be
    within max_days when both are known)."""
    exclude_pool = exclude_pool or set()
    c_date = _days(f["f_date"])
    w, min_score = rc["weights"], float(rc["min_score"])
    near = float(rc["near_pct"])
    cap_po, cap_amt = int(rc["max_po_candidates"]), int(rc["max_amount_candidates"])
    c_net = f["f_net"].to_numpy(dtype=float)
    c_qty = f["f_qty"].to_numpy(dtype=float)
    c_poi = f["f_poi"].to_numpy(dtype=float)
    c_po = f["f_po"].to_numpy(dtype=object)
    c_core = f["f_core"].to_numpy(dtype=object)
    c_desc = f["f_desc"].to_numpy(dtype=object)
    out: list[tuple[float, int, int]] = []
    stats = {"po_in_pool": 0, "cand_po_route": 0, "cand_core_route": 0, "capped": 0}
    for t in claim:
        net = c_net[t]
        if np.isnan(net):
            continue
        po = c_po[t]
        cand_loc: np.ndarray | None = None
        via_po = False
        if "po" in routes and po and po in px.po_index:
            stats["po_in_pool"] += 1
            loc = px.po_index[po]
            if len(loc) > cap_po:
                stats["capped"] += 1
                loc = loc[:cap_po]
            vals = px.net[loc]
            d = np.abs(vals - net)
            keep = d <= np.maximum(_tol(vals, rc), near * np.abs(vals))
            if keep.any():
                cand_loc, via_po = loc[keep], True
        if cand_loc is None and "core" in routes:
            cr = rc["core_route"]
            if not cr.get("enabled"):
                continue
            if abs(net) <= float(cr.get("min_abs", 0.01)):
                continue
            when = cr.get("when", "no_po")
            allowed = (not po) or (when in ("unknown", "any") and po not in px.po_index) or \
                      (when == "any" and po in px.po_index)
            if not allowed:
                continue
            core = c_core[t]
            ci = px.core_index.get(core) if core else None
            if ci is None:
                continue
            vals_sorted, locs = ci
            tol = float(_tol(net, rc))
            lo, hi = np.searchsorted(vals_sorted, [net - tol, net + tol + 1e-9])
            if hi <= lo:
                continue
            cand_loc = locs[lo:hi]
            if exclude_pool:
                cand_loc = np.array([e for e in cand_loc if int(e) not in exclude_pool], dtype=np.int64)
            maxd = cr.get("max_days")
            if maxd is not None and len(cand_loc) and not np.isnan(c_date[t]):
                pd_ = px.pday[cand_loc]
                cand_loc = cand_loc[np.isnan(pd_) | (np.abs(pd_ - c_date[t]) <= float(maxd))]
            if len(cand_loc) == 0:
                continue
            if len(cand_loc) > cap_amt:
                stats["capped"] += 1
                cand_loc = cand_loc[:cap_amt]
        if cand_loc is None:
            continue
        stats["cand_po_route" if via_po else "cand_core_route"] += len(cand_loc)
        for e in cand_loc:
            ev = px.net[e]
            d = abs(ev - net)
            s = w["amount_exact"] if d <= float(_tol(ev, rc)) else w["amount_near"]
            if _qty_eq(c_qty[t], px.qty[e]):
                s += w["qty"]
            if not np.isnan(c_poi[t]) and not np.isnan(px.poi[e]) and c_poi[t] == px.poi[e]:
                s += w["po_item"]
            a, b = c_desc[t], px.desc[e]
            if a and b:
                s += w["desc"] * fuzz.ratio(a, b) / 100.0
            if c_core[t] and c_core[t] == px.core[e]:
                s += w["supplier"]
            if via_po:
                s += w["po_route"]
            if s >= min_score:
                out.append((s, int(t), int(e)))
    return out, stats


def assign_greedy(cands: list[tuple[float, int, int]], assigned: dict | None = None, taken: dict | None = None
                  ) -> tuple[dict[int, tuple[int, float]], dict[int, int]]:
    """Greedy one-to-one by descending score; ties broken by claimant row then pool index (deterministic).

    Returns (assigned: claimant -> (pool_local, score), taken: pool_local -> claimant)."""
    cands = sorted(cands, key=lambda x: (-round(x[0], 6), x[1], x[2]))
    assigned = {} if assigned is None else assigned
    taken = {} if taken is None else taken
    for s, t, e in cands:
        if t in assigned or e in taken:
            continue
        assigned[t] = (e, s)
        taken[e] = t
    return assigned, taken


def orphan_rescue(f: pd.DataFrame, cands: list[tuple[float, int, int]], assigned: dict[int, tuple[int, float]],
                  taken: dict[int, int], claim: np.ndarray, rc: dict) -> dict[int, int]:
    """R6: an unassigned claimant whose best candidate pool row was taken by a twin (same supplier, amount,
    quantity, similar text) is a re-export that clustering missed -> rescued_to[claimant] = twin claimant."""
    thr = float(rc["rescue"]["desc_sim"])
    need_inv = bool(rc["rescue"].get("require_same_invoice", True))
    invd = f["f_invd"].to_numpy(dtype=object)
    best: dict[int, list[tuple[float, int]]] = {}
    for s, t, e in cands:
        if t not in assigned:
            best.setdefault(t, []).append((s, e))
    net = f["f_net"].to_numpy(dtype=float)
    qty = f["f_qty"].to_numpy(dtype=float)
    core = f["f_core"].to_numpy(dtype=object)
    desc = f["f_desc"].to_numpy(dtype=object)
    rescued: dict[int, int] = {}
    for t, lst in best.items():
        for s, e in sorted(lst, reverse=True):
            u = taken.get(e)
            if u is None or u == t:
                continue
            if core[t] != core[u]:
                continue
            if need_inv and invd[t] != invd[u]:      # different invoice numbers = distinct invoices (measured on A)
                continue
            if abs(net[t] - net[u]) > float(_tol(net[u], rc)):
                continue
            if not _qty_eq(qty[t], qty[u]):
                continue
            a, b = desc[t], desc[u]
            if a and b and fuzz.ratio(a, b) / 100.0 < thr:
                continue
            rescued[t] = u
            break
    return rescued


def partial_capacity_match(f: pd.DataFrame, claim_unmatched: np.ndarray, px: PoolIndex, taken: dict[int, int],
                           rc: dict) -> dict[int, tuple[int, float]]:
    """D-22: partial-quantity invoices. A tool invoice with the same PO + supplier core as a receipt, unit net price
    within `price_tol`, whose quantity fits into the receipt's remaining capacity counts as (part of) that receipt.

    Capacity of a receipt = its quantity unless an exact match already consumed it (then 0); each accepted partial
    decrements it. Largest quantities first, best-fit receipt (smallest leftover), ties by price then index."""
    pc = rc.get("partial_capacity", {})
    tol = float(pc.get("price_tol", 0.005))
    cap_n = int(pc.get("max_candidates", 60))
    c_net = f["f_net"].to_numpy(dtype=float)
    c_qty = f["f_qty"].to_numpy(dtype=float)
    c_po = f["f_po"].to_numpy(dtype=object)
    c_core = f["f_core"].to_numpy(dtype=object)
    c_poi = f["f_poi"].to_numpy(dtype=float)
    cap = {}
    out: dict[int, tuple[int, float]] = {}
    order = sorted((int(t) for t in claim_unmatched), key=lambda t: (-(c_qty[t] if not np.isnan(c_qty[t]) else 0), t))
    for t in order:
        net, qty, po, core = c_net[t], c_qty[t], c_po[t], c_core[t]
        if not po or po not in px.po_index or not core:
            continue
        if np.isnan(net) or np.isnan(qty) or net <= 0.01 or qty <= 0:
            continue
        unit = net / qty
        best, best_key = None, None
        for e in px.po_index[po][:cap_n]:
            e = int(e)
            if px.core[e] != core:
                continue
            ev, eq = px.net[e], px.qty[e]
            if np.isnan(ev) or np.isnan(eq) or ev <= 0.01 or eq <= 0:
                continue
            if not np.isnan(c_poi[t]) and not np.isnan(px.poi[e]) and c_poi[t] != px.poi[e]:
                continue
            eu = ev / eq
            if abs(unit - eu) > tol * eu:
                continue
            left = cap.get(e, 0.0 if e in taken else eq)
            if qty > left + 1e-6 * max(1.0, eq):
                continue
            key = (round(left - qty, 6), abs(unit - eu), e)
            if best_key is None or key < best_key:
                best, best_key = e, key
        if best is not None:
            cap[best] = cap.get(best, 0.0 if best in taken else px.qty[best]) - qty
            out[t] = (best, 5.0)
    return out
