"""Unit memory (D-45): LLM answers of earlier runs for the SAME taxonomy, keyed by unit identity.

The LLM response cache is keyed on whole-batch prompts, so a new extract (B2/A2) with a different unit mix misses it
even when most descriptions recur. This store answers individual units: key = sha1(text|supplier|gl|ctx) (exact) or
sha1(text|supplier) (looser). Entries carry the taxonomy hash, so a different company's tree never sees them.
Written only from LLM-answered units with confidence >= min_conf; read before any LLM request.
Lives in .cache/ (git-ignored); never holds more than normalised text, so it is treated like the LLM cache.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .units import FEATURE_FIELDS


def _h(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8", "ignore")).hexdigest()


def unit_keys(units) -> tuple[list[str], list[str]]:
    t, s, g, c = (units[k].astype(str).to_numpy() for k in ("text", "supplier", "gl", "ctx"))
    exact = [_h("x", a, b, d, e) for a, b, d, e in zip(t, s, g, c)]
    loose = [_h("l", a, b) for a, b in zip(t, s)]
    return exact, loose


class UnitMemory:
    def __init__(self, path: str | Path, tax_hash: str):
        self.path, self.tax = Path(path), tax_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.execute("CREATE TABLE IF NOT EXISTS m (tax TEXT, k TEXT, leaf TEXT, conf REAL, n INT, "
                         "PRIMARY KEY(tax, k))")
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(m)")}
        if "feat" not in cols:      # older store: add the feature column (exact-key rows only carry it)
            self._db.execute("ALTER TABLE m ADD COLUMN feat TEXT")
        self._db.commit()

    def lookup(self, units, mask: np.ndarray, loose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """-> (leaf ids ('' = miss), conf) for the masked units."""
        ex, lo = unit_keys(units)
        leaf = np.full(len(units), "", dtype=object)
        conf = np.zeros(len(units))
        idx = np.where(mask)[0]
        if not len(idx):
            return leaf, conf
        keys = {ex[i] for i in idx} | ({lo[i] for i in idx} if loose else set())
        found: dict[str, tuple[str, float]] = {}
        ks = list(keys)
        with self._lock:
            for j in range(0, len(ks), 500):
                part = ks[j:j + 500]
                q = "SELECT k, leaf, conf FROM m WHERE tax=? AND k IN (%s)" % ",".join("?" * len(part))
                for k, l, c in self._db.execute(q, [self.tax, *part]):
                    found[k] = (l, c)
        for i in idx:
            for k in ((ex[i], lo[i]) if loose else (ex[i],)):
                if k in found:
                    leaf[i], conf[i] = found[k]
                    break
        return leaf, conf

    def store(self, units, mask: np.ndarray, leaf: np.ndarray, conf: np.ndarray) -> int:
        ex, lo = unit_keys(units)
        fcols = [c for c in list(FEATURE_FIELDS) + ["text", "supplier", "gl", "ctx"] if c in units.columns] + \
                [c for c in units.columns if str(c).startswith("ctx__")]
        rows = []
        for i in np.where(mask)[0]:
            feat = json.dumps({c: (bool(units[c].iat[i]) if c == "desc_blank" else str(units[c].iat[i])) for c in fcols})
            rows.append((self.tax, ex[i], leaf[i], float(conf[i]), feat))
            rows.append((self.tax, lo[i], leaf[i], float(conf[i]), None))
        with self._lock:
            self._db.executemany(
                "INSERT INTO m(tax,k,leaf,conf,n,feat) VALUES(?,?,?,?,1,?) ON CONFLICT(tax,k) DO UPDATE SET "
                "leaf=CASE WHEN excluded.conf>=m.conf THEN excluded.leaf ELSE m.leaf END, "
                "conf=MAX(m.conf,excluded.conf), n=m.n+1, feat=COALESCE(excluded.feat, m.feat)",
                [(t, k, l, c, f) for t, k, l, c, f in rows])
            self._db.commit()
        return len(rows) // 2

    def training_units(self, min_conf: float = 0.7, weight: float = 0.3) -> pd.DataFrame:
        """Earlier confident LLM answers as extra supervised training rows: unit feature columns + leaf + w."""
        with self._lock:
            got = list(self._db.execute("SELECT feat, leaf, conf FROM m WHERE tax=? AND feat IS NOT NULL AND conf>=?",
                                        (self.tax, min_conf)))
        if not got:
            return pd.DataFrame()
        recs = []
        for f, l, c in got:
            d = json.loads(f)
            d["leaf"], d["w"] = l, weight * float(c)
            recs.append(d)
        df = pd.DataFrame(recs)
        if "desc_blank" in df.columns:
            df["desc_blank"] = df["desc_blank"].astype(bool)
        df["rows"], df["spend"] = 1, 0.0
        return df
