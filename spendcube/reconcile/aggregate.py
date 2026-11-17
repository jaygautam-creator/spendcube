"""R7: aggregate matches (D-22), behind reconciliation.aggregate.enabled (default off).

Same PO only, bounded subset search (<= max_items items, <= max_candidates candidates per PO):
  * combined invoicing: one tool line == sum of 2..max_items free receipt lines
  * split invoicing:    2..max_items unmatched tool lines == one free receipt line
Matched tool rows are assigned to (the first) receipt line with a fixed score so they become
DUPLICATE through the normal path; all involved receipt lines are marked taken.
"""
from __future__ import annotations

import itertools

import numpy as np

AGG_SCORE = 6.0


def aggregate_matches(f, unmatched: np.ndarray, px, rc: dict, taken: set[int]) -> dict[int, tuple[int, float]]:
    ag = rc["aggregate"]
    kmax, cmax, tol = int(ag["max_items"]), int(ag["max_candidates"]), float(ag["tol_abs"])
    net = f["f_net"].to_numpy(dtype=float)
    po = f["f_po"].to_numpy(dtype=object)
    core = f["f_core"].to_numpy(dtype=object)
    by_po: dict[str, list[int]] = {}
    for t in unmatched:
        t = int(t)
        if po[t] and po[t] in px.po_index and not np.isnan(net[t]):
            by_po.setdefault(po[t], []).append(t)
    out: dict[int, tuple[int, float]] = {}
    used_tool: set[int] = set()
    for p, tools in by_po.items():
        free = [int(e) for e in px.po_index[p] if int(e) not in taken][:cmax]
        if not free:
            continue
        # combined: tool line = sum of several receipt lines
        for t in tools:
            if t in used_tool:
                continue
            done = False
            for k in range(2, kmax + 1):
                for comb in itertools.combinations(free, k):
                    if any(c in taken for c in comb):
                        continue
                    if abs(sum(px.net[c] for c in comb) - net[t]) <= tol and all(core[t] == px.core[c] for c in comb):
                        out[t] = (comb[0], AGG_SCORE)
                        taken.update(comb)
                        used_tool.add(t)
                        done = True
                        break
                if done:
                    break
        # split: several tool lines = one receipt line
        rest = [t for t in tools if t not in used_tool][:cmax]
        for e in free:
            if e in taken or len(rest) < 2:
                continue
            found = None
            for k in range(2, kmax + 1):
                for comb in itertools.combinations(rest, k):
                    if abs(sum(net[c] for c in comb) - px.net[e]) <= tol and all(core[c] == px.core[e] for c in comb):
                        found = comb
                        break
                if found:
                    break
            if found:
                for c in found:
                    out[c] = (e, AGG_SCORE)
                    used_tool.add(c)
                taken.add(e)
                rest = [t for t in rest if t not in used_tool]
    return out
