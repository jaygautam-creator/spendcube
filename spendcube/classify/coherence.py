"""Supplier coherence pass (D-46, experimental, off by default until measured).

After the LLM tier: for every supplier whose LLM-labelled units mostly agree on one leaf (spend-weighted share >=
min_share over >= min_units units), re-ask the LLM about that supplier's low-confidence units that DISAGREE with the
majority, giving the majority path as a hint. The model may still keep its answer. One extra batched call set per run.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

log = logging.getLogger("spendcube.coherence")


def supplier_majorities(units, leaf, labelled, min_units: int, min_share: float) -> dict[str, tuple[str, float, int]]:
    skey = units["supplier_core"].astype(str).to_numpy()
    w = np.log1p(units["spend"].to_numpy()) + 1.0
    dist: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt: dict[str, int] = defaultdict(int)
    for i in np.where(labelled)[0]:
        k = skey[i]
        if k:
            dist[k][leaf[i]] += w[i]
            cnt[k] += 1
    out = {}
    for k, d in dist.items():
        if cnt[k] < min_units:
            continue
        tot = sum(d.values())
        l, v = max(d.items(), key=lambda kv: kv[1])
        if v / tot >= min_share:
            out[k] = (l, v / tot, cnt[k])
    return out


def run_coherence(units, leaf, conf, meth, selfc, labelled, clf, tax, cfg: dict, stats: dict) -> int:
    """Mutates leaf/conf/selfc/meth in place; returns the number of units whose label changed."""
    if not cfg.get("enabled", False):
        return 0
    maj = supplier_majorities(units, leaf, labelled, int(cfg.get("min_units", 3)), float(cfg.get("min_share", 0.5)))
    below = float(cfg.get("conf_below", 0.8))
    skey = units["supplier_core"].astype(str).to_numpy()
    cand = [i for i in np.where(labelled)[0]
            if skey[i] in maj and conf[i] < below and leaf[i] != maj[skey[i]][0]]
    cand.sort(key=lambda i: -units["spend"].iat[i])
    cand = cand[: int(cfg.get("max_units", 600))]
    stats["coherence_candidates"] = len(cand)
    if not cand:
        return 0
    items = []
    for i in cand:
        m = maj[skey[i]]
        hint = f"supplier_majority={' > '.join(tax.leaf(m[0]).path)} (share {m[1]:.0%} of {m[2]} lines)"
        items.append({"uid": int(i), "supplier": units["supplier"].iat[i], "gl": units["gl"].iat[i],
                      "ctx": (str(units["ctx"].iat[i]) + " | " + hint).strip(" |"), "text": units["text"].iat[i]})
    res = clf.classify(items, str(cfg.get("role", "bulk")), workers=3)
    changed = 0
    for uid, v in res.items():
        if v.leaf_id != leaf[uid]:
            changed += 1
        leaf[uid] = v.leaf_id
        selfc[uid] = v.conf
        conf[uid] = float(np.clip(0.5 * v.conf + 0.5 * (1.0 if v.leaf_id == maj[skey[uid]][0] else 0.5), 0.05, 0.99))
        meth[uid] = "llm_coherence"
    stats["coherence_reasked"], stats["coherence_changed"] = len(res), changed
    log.info("supplier coherence: %d candidates, %d answered, %d changed", len(cand), len(res), changed)
    return changed
