"""Label lookup: exact (non-generic) material id, else exact compact description, with purity/support floors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import col


def generic_materials(df: pd.DataFrame, min_texts: int = 8, min_ratio: float = 0.2) -> set[str]:
    """A material id is generic (a free-text placeholder) when it carries many distinct descriptions."""
    d = pd.DataFrame({"m": col(df, "material_id").str.strip().to_numpy(), "t": col(df, "desc_compact").to_numpy()})
    d = d[d["m"] != ""]
    if d.empty:
        return set()
    g = d.groupby("m")["t"].agg(["nunique", "size"])
    return set(g.index[(g["nunique"] >= min_texts) & (g["nunique"] / g["size"] >= min_ratio)])


def _table(keys: pd.Series, y: pd.Series, min_support: int, min_purity: float) -> dict[str, tuple[str, float]]:
    d = pd.DataFrame({"k": keys.to_numpy(), "y": y.to_numpy()})
    d = d[(d["k"] != "") & (d["y"] != "")]
    if d.empty:
        return {}
    cnt = d.groupby(["k", "y"]).size().reset_index(name="n")
    tot = cnt.groupby("k")["n"].sum().rename("tot")
    best = cnt.sort_values(["k", "n", "y"], ascending=[True, False, True]).drop_duplicates("k").set_index("k").join(tot)
    best["pur"] = best["n"] / best["tot"]
    best = best[(best["tot"] >= min_support) & (best["pur"] >= min_purity)]
    return {k: (r.y, float(r.pur)) for k, r in best.iterrows()}


@dataclass
class Lookup:
    material: dict = field(default_factory=dict)
    desc: dict = field(default_factory=dict)
    generic: set = field(default_factory=set)

    def query(self, df: pd.DataFrame) -> pd.DataFrame:
        mat = col(df, "material_id").str.strip().to_numpy()
        dc = col(df, "desc_compact").to_numpy()
        blank = df["desc_blank"].astype(bool).to_numpy() if "desc_blank" in df.columns else np.zeros(len(df), bool)
        leaf = np.full(len(df), None, dtype=object)
        conf = np.full(len(df), np.nan)
        for i in range(len(df)):
            hit = None if mat[i] in self.generic else self.material.get(mat[i])
            if hit is None and not blank[i]:
                hit = self.desc.get(dc[i])
            if hit is not None:
                leaf[i], conf[i] = hit
        return pd.DataFrame({"leaf_id": leaf, "conf": conf}, index=df.index)


def build_lookup(df: pd.DataFrame, leaf_ids, min_support: int = 2, min_purity: float = 0.9,
                 generic_min_texts: int = 8) -> Lookup:
    y = pd.Series(np.asarray(leaf_ids, dtype=object), index=df.index).astype(str)
    gen = generic_materials(df, generic_min_texts)
    mat = col(df, "material_id").str.strip()
    dc = col(df, "desc_compact")
    if "desc_blank" in df.columns:
        dc = dc.where(~df["desc_blank"].astype(bool), "")
    return Lookup(_table(mat.where(~mat.isin(gen), ""), y, min_support, min_purity),
                  _table(dc, y, min_support, min_purity), gen)
