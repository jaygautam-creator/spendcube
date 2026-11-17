"""Item resolution (MASTER_ARCHITECTURE 7.2): generic-material detection, material keys, description
clusters (decimal-separator + OCR unification, blocked fuzzy with a digit-signature guard), linking of
free-text/secondary rows to the material master, label alias maps."""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .suppliers import UF, ocr_fold

log = logging.getLogger("spendcube.entities")

DEFAULTS = {
    "generic_min_texts": 8, "generic_min_ratio": 0.2, "generic_ids": [],
    "fuzzy": True, "fuzzy_mode": "strict", "fuzzy_ratio": 93.0, "fuzzy_min_len": 10, "block_cap": 300, "block_prefix": 6,
    "link_share": 0.6, "merge_materials_by_text": False,
}
_DECIMAL = re.compile(r"(?<=\d),(?=\d)")
_ALNUM = re.compile(r"[^0-9A-Z]")


def text_key(desc_norm: str) -> str:
    """Decimal-comma unification -> compact -> OCR fold. Empty for blank."""
    if not desc_norm:
        return ""
    return ocr_fold(_ALNUM.sub("", _DECIMAL.sub(".", desc_norm.upper())))


def detect_generic(material: pd.Series, desc_compact: pd.Series, cfg: dict) -> set[str]:
    c = {**DEFAULTS, **cfg}
    m = material.fillna("").astype(str).str.strip()
    ok = (m != "") & (desc_compact.fillna("") != "")
    g = pd.DataFrame({"m": m[ok], "d": desc_compact[ok]}).groupby("m")["d"].agg(["nunique", "size"])
    gen = g[(g["nunique"] >= c["generic_min_texts"]) & (g["nunique"] / g["size"] >= c["generic_min_ratio"])].index
    return set(gen) | set(c["generic_ids"])


def build_item_alias(rows: pd.DataFrame, purity: float = 0.9) -> tuple[dict, dict]:
    """rows: material_id, text_key, gold_item. -> (material->key, text_key->key for texts spanning >=2 materials)."""
    r = rows[rows["gold_item"].astype(str) != ""]
    mat_alias, txt_alias = {}, {}
    sub = r[r["material_id"].astype(str).str.strip() != ""]
    if len(sub):
        ct = sub.groupby(["material_id", "gold_item"]).size().rename("n").reset_index()
        tot = ct.groupby("material_id")["n"].transform("sum")
        ct["share"] = ct["n"] / tot
        top = ct.sort_values(["share", "n"], ascending=False).drop_duplicates("material_id")
        top = top[top.share >= purity]
        mat_alias = dict(zip(top.material_id, top.gold_item))
    sub = r[(r["text_key"] != "") & (r["material_id"].astype(str).str.strip() != "")]
    if len(sub):
        g = sub.groupby("text_key").agg(nm=("material_id", "nunique"), nk=("gold_item", "nunique"), n=("gold_item", "size"))
        cand = g[(g.nm >= 2) & (g.nk == 1)].index
        one = sub[sub.text_key.isin(cand)].drop_duplicates("text_key")
        txt_alias = dict(zip(one.text_key, one.gold_item))
    return mat_alias, txt_alias


_FOLDED_DIGITS = set("0123456789OISBL")     # chars that may stand for a digit after OCR folding


def _strict_ok(a: str, b: str) -> bool:
    """Folded keys differ only by inserted/deleted characters that cannot be digits. Any substitution
    (e.g. 1800 vs 1500 -> IBOO vs ISOO) or digit-like indel is a genuine difference, not OCR noise."""
    if len(a) < len(b):
        a, b = b, a
    ops = Levenshtein.editops(a, b)
    if len(ops) > max(2, int(0.07 * len(a))):
        return False
    for op in ops:
        if op.tag == "replace":
            return False
        ch = a[op.src_pos] if op.tag == "delete" else b[op.dest_pos]
        if ch in _FOLDED_DIGITS:
            return False
    return True


def _cluster_texts(keys: dict[str, float], c: dict) -> tuple[dict[str, str], dict]:
    """keys: text_key -> weight. Star-merge near-identical keys (blocked). Returns key -> center key."""
    stats = {"text_keys": len(keys), "fuzzy_merges": 0, "block_overflow": 0}
    center = {k: k for k in keys}
    if not c["fuzzy"]:
        return center, stats
    blocks: dict[str, list] = defaultdict(list)
    for k in sorted(keys, key=lambda k: -keys[k]):
        if len(k) >= c["fuzzy_min_len"]:
            blocks[k[:c["block_prefix"]]].append(k)
    for bk, ks in blocks.items():
        if len(ks) < 2:
            continue
        if len(ks) > c["block_cap"]:
            stats["block_overflow"] += 1
            ks = ks[:c["block_cap"]]
        done: set = set()
        for i, a in enumerate(ks):
            if a in done:
                continue
            da = re.sub(r"\D", "", a)
            for b in ks[i + 1:]:
                if b in done or center[b] != b:
                    continue
                if abs(len(a) - len(b)) > 3:
                    continue
                if c["fuzzy_mode"] == "loose" and re.sub(r"\D", "", b) != da:
                    continue
                if fuzz.ratio(a, b) >= c["fuzzy_ratio"] and (c["fuzzy_mode"] == "loose" or _strict_ok(a, b)):
                    center[b] = a
                    done.add(b)
                    stats["fuzzy_merges"] += 1
    return center, stats


def resolve_items(df: pd.DataFrame, cfg: Optional[dict] = None, supplier_key: Optional[pd.Series] = None,
                  mat_alias: Optional[dict] = None, txt_alias: Optional[dict] = None,
                  primary_mask: Optional[pd.Series] = None) -> tuple[pd.Series, pd.Series, dict]:
    """Returns (item_key, item_conf, stats). primary_mask marks master-like rows used to build the
    material index that free-text/secondary rows link to (default: rows with a non-generic material)."""
    c = {**DEFAULTS, **(cfg or {})}
    mat_alias, txt_alias = mat_alias or {}, txt_alias or {}
    n = len(df)
    mat = df["material_id"].fillna("").astype(str).str.strip() if "material_id" in df.columns else pd.Series("", index=df.index)
    dn = df["desc_norm"].fillna("").astype(str) if "desc_norm" in df.columns else pd.Series("", index=df.index)
    dc = df["desc_compact"].fillna("").astype(str) if "desc_compact" in df.columns else pd.Series("", index=df.index)
    blank = df["desc_blank"].fillna(True).astype(bool) if "desc_blank" in df.columns else (dc == "")
    gl = df["gl_account"].fillna("").astype(str) if "gl_account" in df.columns else pd.Series("", index=df.index)
    sk = supplier_key.astype(str) if supplier_key is not None else pd.Series("", index=df.index)
    stats: dict = {}
    generic = detect_generic(mat, dc.where(~blank, ""), c)
    stats["generic_materials"] = sorted(generic)[:20]
    tk = pd.Series([text_key(x) if not b else "" for x, b in zip(dn, blank)], index=df.index)
    has_text = tk != ""
    use_mat = (mat != "") & ~mat.isin(generic)
    # text clusters
    w = pd.Series(1.0, index=df.index)
    kw = w[has_text].groupby(tk[has_text]).sum().to_dict()
    center, st = _cluster_texts(kw, c)
    stats.update(st)
    tkc = tk.map(lambda k: center.get(k, k) if k else "")
    # optional: merge materials that share an identical normalised text
    mat_root: dict = {}
    if c["merge_materials_by_text"]:
        uf = UF()
        first: dict = {}
        for m, k in zip(mat[use_mat & has_text], tkc[use_mat & has_text]):
            if k in first:
                uf.union(first[k], m)
            else:
                first[k] = m
        mat_root = {m: uf.find(m) for m in set(mat[use_mat])}
    # linking index: cluster -> majority material key among master rows
    master = use_mat if primary_mask is None else (use_mat & primary_mask)
    link: dict = {}
    sub = pd.DataFrame({"k": tkc[master & has_text], "m": mat[master & has_text]})
    if len(sub):
        ct = sub.groupby(["k", "m"]).size().rename("n").reset_index()
        tot = ct.groupby("k")["n"].transform("sum")
        ct["share"] = ct["n"] / tot
        top = ct.sort_values(["share", "n"], ascending=False).drop_duplicates("k")
        top = top[top.share >= c["link_share"]]
        link = dict(zip(top.k, top.m))
    key = np.empty(n, dtype=object)
    cf = np.zeros(n)
    mat_a, tkc_a, tk_a, gl_a, sk_a, use_a = mat.to_numpy(), tkc.to_numpy(), tk.to_numpy(), gl.to_numpy(), sk.to_numpy(), use_mat.to_numpy()
    cnt = Counter()
    for i in range(n):
        m, k = mat_a[i], tkc_a[i]
        if m and m in mat_alias:
            key[i], cf[i] = mat_alias[m], 0.95; cnt["alias_material"] += 1
        elif k and tk_a[i] in txt_alias:
            key[i], cf[i] = txt_alias[tk_a[i]], 0.85; cnt["alias_text"] += 1
        elif use_a[i]:
            key[i], cf[i] = "M:" + mat_root.get(m, m), 0.95; cnt["material"] += 1
        elif k:
            if k in link:
                lm = link[k]
                key[i], cf[i] = "M:" + mat_root.get(lm, lm), 0.8; cnt["linked_to_material"] += 1
            else:
                key[i], cf[i] = "D:" + hashlib.md5(k.encode()).hexdigest()[:10], 0.7; cnt["description"] += 1
        else:
            key[i], cf[i] = "U:" + sk_a[i] + "|" + gl_a[i], 0.3; cnt["unknown"] += 1
    stats["methods"] = dict(cnt)
    stats["distinct_keys"] = int(len(set(key)))
    return pd.Series(key, index=df.index), pd.Series(cf, index=df.index), stats
