"""Supplier resolution (MASTER_ARCHITECTURE 7.1): exact cores, vendor-id union, conservative blocked fuzzy
merge with OCR equivalence, optional label alias maps. Nothing here knows any company name."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .legal_forms import supplier_norm

log = logging.getLogger("spendcube.entities")

_OCR = str.maketrans("01L58", "OIISB")
DEFAULTS = {
    "mode": "auto",            # auto | vendor_only (supplier_key = V:<vendor_id> regardless of names)
    "fuzzy": True, "jw_min": 0.95, "ratio_min": 92.0, "max_len_diff": 3, "min_len": 5, "block_cap": 300,
    "vendor_union": True, "vendor_min_sim": 50.0,
}


def ocr_fold(s: str) -> str:
    return s.translate(_OCR)


def sorted_tokens_key(norm: str) -> str:
    return " ".join(sorted(norm.split()))


class UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _similar(a: str, b: str, c: dict) -> float:
    """Return a similarity score >0 when a and b should merge, else 0."""
    if min(len(a), len(b)) < c["min_len"]:
        return 0.0
    fa, fb = ocr_fold(a), ocr_fold(b)
    if fa == fb:
        return 1.0
    jw = JaroWinkler.normalized_similarity(fa, fb)
    ra = fuzz.ratio(fa, fb)
    # numbers inside names (e.g. 'DEPOT 12' vs 'DEPOT 21') are identity, never typos: after OCR folding the only
    # digits left are 2,3,4,6,7,9; no edit between the two names may touch one of them.
    for op in Levenshtein.editops(fa, fb):
        ch_a = fa[op.src_pos] if op.src_pos < len(fa) else ""
        ch_b = fb[op.dest_pos] if op.dest_pos < len(fb) else ""
        if (op.tag != "insert" and ch_a.isdigit()) or (op.tag != "delete" and ch_b.isdigit()):
            return 0.0
    if jw >= c["jw_min"] or (ra >= c["ratio_min"] and abs(len(a) - len(b)) <= c["max_len_diff"]):
        return max(jw, ra / 100.0)
    return 0.0


def cluster_cores(cores: dict[str, float], norms: dict[str, str], vendor_groups: list[list[str]],
                  cfg: dict) -> tuple[dict[str, str], dict[str, float], dict]:
    """cores: core -> weight. Returns (core -> cluster center core, core -> confidence, stats)."""
    c = {**DEFAULTS, **cfg}
    stats = {"cores": len(cores), "vendor_unions": 0, "vendor_conflicts": 0, "fuzzy_merges": 0, "block_overflow": 0}
    uf = UF()
    for k in cores:
        uf.find(k)
    # vendor-id union with dissimilarity guard
    if c["vendor_union"]:
        for grp in vendor_groups:
            grp = sorted(set(g for g in grp if g in cores), key=lambda g: (-cores[g], g))   # deterministic
            if len(grp) < 2:
                continue
            subs: list[list[str]] = []          # greedy sub-clusters of mutually similar names
            for g in grp:
                for sc in subs:
                    if any(fuzz.ratio(m, g) >= c["vendor_min_sim"] for m in sc):
                        sc.append(g)
                        break
                else:
                    subs.append([g])
            for sc in subs:
                for g in sc[1:]:
                    uf.union(sc[0], g)
                    stats["vendor_unions"] += 1
            stats["vendor_conflicts"] += len(subs) - 1
    conf: dict[str, float] = {k: 0.9 for k in cores}
    # fuzzy: star around heaviest cluster, one hop, blocked
    if c["fuzzy"]:
        weights: dict = defaultdict(float)
        members: dict = defaultdict(list)
        for k, w in cores.items():
            r = uf.find(k)
            weights[r] += w
            members[r].append(k)
        center = {r: max(ms, key=lambda m: cores[m]) for r, ms in members.items()}
        roots = sorted(weights, key=lambda r: (-weights[r], r))
        blocks: dict[str, list] = defaultdict(list)
        for r in roots:
            cc = center[r]
            blocks["p:" + ocr_fold(cc)[:3]].append(r)
            blocks["t:" + sorted_tokens_key(norms.get(cc, cc))].append(r)
        assigned: dict = {}
        order = {r: i for i, r in enumerate(roots)}
        for bk, rs in blocks.items():
            if len(rs) < 2:
                continue
            if len(rs) > c["block_cap"]:
                stats["block_overflow"] += 1
                rs = rs[:c["block_cap"]]
            rs = sorted(rs, key=lambda r: order[r])
            for i, r in enumerate(rs):
                if r in assigned:
                    continue
                for s in rs[i + 1:]:
                    if s in assigned or uf.find(s) == uf.find(r):
                        continue
                    if bk.startswith("t:"):
                        sim = 1.0 if min(len(center[r]), len(center[s])) >= c["min_len"] else 0.0
                    else:
                        sim = _similar(center[r], center[s], c)
                    if sim > 0:
                        assigned[s] = r
                        for m in members[s]:
                            conf[m] = min(conf[m], 0.5 + 0.4 * sim if sim < 1 else 0.85)
                        stats["fuzzy_merges"] += 1
        for s, r in assigned.items():
            uf.union(r, s)
    final: dict[str, list] = defaultdict(list)
    for k in cores:
        final[uf.find(k)].append(k)
    out = {}
    for r, ms in final.items():
        ctr = max(ms, key=lambda m: cores[m])
        for m in ms:
            out[m] = ctr
    return out, conf, stats


def build_supplier_alias(rows: pd.DataFrame, purity: float = 0.9) -> tuple[dict, dict]:
    """rows: supplier_core, vendor_id, gold_supplier (golden key). -> (core->key, vendor->key), purity-filtered."""
    r = rows[(rows["gold_supplier"].astype(str) != "")]
    out = []
    for col in ("supplier_core", "vendor_id"):
        d = {}
        sub = r[r[col].astype(str).str.strip() != ""]
        if len(sub):
            ct = sub.groupby([col, "gold_supplier"]).size().rename("n").reset_index()
            tot = ct.groupby(col)["n"].transform("sum")
            ct["share"] = ct["n"] / tot
            top = ct.sort_values(["share", "n"], ascending=False).drop_duplicates(col)
            d = dict(zip(top.loc[top.share >= purity, col], top.loc[top.share >= purity, "gold_supplier"]))
        out.append(d)
    return out[0], out[1]


def resolve_suppliers(df: pd.DataFrame, cfg: Optional[dict] = None, alias_core: Optional[dict] = None,
                      alias_vendor: Optional[dict] = None) -> tuple[pd.Series, pd.Series, dict]:
    """Returns (supplier_key, supplier_conf, stats) indexed like df."""
    c = {**DEFAULTS, **(cfg or {})}
    alias_core, alias_vendor = alias_core or {}, alias_vendor or {}
    core = df["supplier_core"].fillna("").astype(str) if "supplier_core" in df.columns else pd.Series("", index=df.index)
    vid = df["vendor_id"].fillna("").astype(str).str.strip()
    src = df["source"].astype(str) if "source" in df.columns else pd.Series("", index=df.index)
    w = pd.to_numeric(df["spend_rc"], errors="coerce").abs().fillna(0.0) if "spend_rc" in df.columns else pd.Series(0.0, index=df.index)
    w = w + 1e-6                                       # tie-break by row count
    stats: dict = {}
    if c["mode"] == "vendor_only":
        core = pd.Series("", index=df.index)
    named = core != ""
    cw = w[named].groupby(core[named]).sum().to_dict()
    norms = {}
    if "supplier_norm" in df.columns:
        norms = dict(zip(core[named], df.loc[named, "supplier_norm"]))
    else:
        norms = {k: k for k in cw}
    vg: dict = defaultdict(set)
    for k, v, s in zip(core[named], vid[named], src[named]):
        if v:
            vg[(s, v)].add(k)
    center, conf, st = cluster_cores(cw, norms, [sorted(v) for v in vg.values()], c) if cw else ({}, {}, {})
    stats.update(st)
    # vendor -> core (for rows with a vendor id but no name)
    vend_core: dict = {}
    if named.any():
        vv = pd.DataFrame({"v": vid[named], "c": core[named].map(center), "w": w[named]})
        vv = vv[vv.v != ""]
        if len(vv):
            top = vv.groupby(["v", "c"])["w"].sum().reset_index().sort_values("w", ascending=False)
            tot = top.groupby("v")["w"].transform("sum")
            top = top[top["w"] / tot >= 0.8].drop_duplicates("v")
            vend_core = dict(zip(top.v, top.c))
    # cluster-level alias (heaviest aliased member)
    clus_alias: dict = {}
    for k, ctr in center.items():
        if k in alias_core:
            cur = clus_alias.get(ctr)
            if cur is None or cw[k] > cur[1]:
                clus_alias[ctr] = (alias_core[k], cw[k])
    n = len(df)
    key = np.empty(n, dtype=object)
    cf = np.zeros(n)
    core_a, vid_a = core.to_numpy(), vid.to_numpy()
    for i in range(n):
        k, v = core_a[i], vid_a[i]
        if k:
            if k in alias_core:
                key[i], cf[i] = alias_core[k], 0.95
            elif v and v in alias_vendor:
                key[i], cf[i] = alias_vendor[v], 0.9
            else:
                ctr = center.get(k, k)
                if ctr in clus_alias:
                    key[i], cf[i] = clus_alias[ctr][0], 0.8
                else:
                    key[i], cf[i] = "S:" + ctr, conf.get(k, 0.9)
        elif v:
            if v in alias_vendor:
                key[i], cf[i] = alias_vendor[v], 0.9
            elif v in vend_core:
                ctr = vend_core[v]
                key[i], cf[i] = (clus_alias[ctr][0] if ctr in clus_alias else "S:" + ctr), 0.75
            else:
                key[i], cf[i] = "V:" + v, 0.7
        else:
            key[i], cf[i] = "S:UNKNOWN", 0.1
    stats["distinct_keys"] = int(len(set(key)))
    return pd.Series(key, index=df.index), pd.Series(cf, index=df.index), stats
