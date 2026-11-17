"""Taxonomy model: wide / path / adjacency formats, leaf ids, exact-string paths.

Names are kept as the exact raw strings found in the file (output must match).
Comparisons elsewhere use casefolded copies. Leaves are identified by id, never by
name, so duplicate names under different parents are fine.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config.loader import ConfigError, resolve_files
from ..config.schema import TaxonomyCfg
from ..ingest.readers import read_many

log = logging.getLogger("spendcube.taxonomy")

MAX_DEPTH = 4


@dataclass(frozen=True)
class Leaf:
    id: str
    path: tuple[str, ...]
    depth: int


class Taxonomy:
    def __init__(self, paths: list[tuple[str, ...]], max_depth: int = MAX_DEPTH):
        # keep first occurrence order, drop exact duplicate paths
        seen, uniq = set(), []
        for p in paths:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        if not uniq:
            raise ConfigError("taxonomy has no categories")
        self.depth = min(max(len(p) for p in uniq), max_depth)
        if max(len(p) for p in uniq) > max_depth:
            log.warning("taxonomy deeper than %d levels; truncating", max_depth)
            uniq = list(dict.fromkeys(p[:max_depth] for p in uniq))
        self.leaves: list[Leaf] = [
            Leaf(id=f"T{i + 1:03d}", path=p, depth=len(p)) for i, p in enumerate(uniq)
        ]
        self._by_id = {l.id: l for l in self.leaves}
        self._by_path = {l.path: l for l in self.leaves}
        self.hash = hashlib.sha256("\n".join(" > ".join(l.path) for l in self.leaves).encode()).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.leaves)

    def leaf(self, leaf_id: str) -> Leaf:
        return self._by_id[leaf_id]

    def has(self, leaf_id: str) -> bool:
        return leaf_id in self._by_id

    def path_padded(self, leaf_id: str, n: int = MAX_DEPTH) -> list[str]:
        p = list(self._by_id[leaf_id].path)
        return (p + [""] * n)[:n]

    def leaf_by_path(self, path: tuple[str, ...]) -> Optional[Leaf]:
        return self._by_path.get(tuple(path))

    def leaf_by_names(self, names: list[str]) -> Optional[Leaf]:
        """Resolve label names (stripped, casefolded) to a leaf; deepest non-blank wins."""
        key = tuple(n.strip().casefold() for n in names if n and n.strip())
        for l in self.leaves:
            if tuple(x.strip().casefold() for x in l.path) == key:
                return l
        return None

    def path_str(self, leaf_id: str) -> str:
        return " > ".join(self._by_id[leaf_id].path)

    def summary(self) -> str:
        by_depth: dict[int, int] = {}
        for l in self.leaves:
            by_depth[l.depth] = by_depth.get(l.depth, 0) + 1
        l1 = {l.path[0] for l in self.leaves}
        return (f"{len(self.leaves)} leaves, max depth {self.depth}, "
                f"leaves/depth={dict(sorted(by_depth.items()))}, {len(l1)} L1 nodes")

    def valid_paths(self) -> set[tuple[str, ...]]:
        return {l.path for l in self.leaves}

    @classmethod
    def load(cls, cfg: TaxonomyCfg) -> "Taxonomy":
        files = resolve_files(cfg.file)
        df = read_many(files, cfg.read.delimiter, cfg.read.encoding, cfg.sheet or cfg.read.sheet)
        if cfg.columns:
            missing = [c for c in cfg.columns if c not in df.columns]
            if missing:
                raise ConfigError(f"taxonomy.columns {missing} not in file columns {list(df.columns)}")
            df = df[cfg.columns]
        fmt = cfg.format if cfg.format != "auto" else detect_format(df, cfg.path_separator)
        log.info("taxonomy %s: format=%s columns=%s", files[0].name, fmt, list(df.columns))
        if fmt == "wide":
            paths = _from_wide(df)
        elif fmt == "path":
            paths = _from_path(df, cfg.path_separator)
        elif fmt == "adjacency":
            paths = _from_adjacency(df)
        else:
            raise ConfigError(f"unknown taxonomy format {fmt}")
        t = cls(paths)
        log.info("taxonomy: %s", t.summary())
        return t


_LEVEL_RE = re.compile(r"(?:^|[^a-z0-9])(?:l|level|lvl|cat(?:egory)?[ _-]?(?:l|level)?)[ _-]?(\d)(?:$|[^a-z0-9])", re.I)


def _level_columns(df: pd.DataFrame) -> list[str]:
    scored = []
    for c in df.columns:
        m = _LEVEL_RE.search(str(c)) or re.search(r"(\d)$", str(c))
        if m:
            scored.append((int(m.group(1)), c))
    scored.sort()
    return [c for _, c in scored]


def detect_format(df: pd.DataFrame, sep: str) -> str:
    cols = [str(c).lower() for c in df.columns]
    if len(_level_columns(df)) >= 2:
        return "wide"
    if any("parent" in c for c in cols):
        return "adjacency"
    # single-column or a path-like column
    for c in df.columns:
        sample = df[c].astype(str).head(50)
        for s in (sep, ">", "/", "|", "\\"):
            if s and (sample.str.contains(re.escape(s))).mean() > 0.5:
                return "path"
    if df.shape[1] >= 2:
        return "wide"   # unnamed level columns: take columns in order
    return "path"


def _from_wide(df: pd.DataFrame) -> list[tuple[str, ...]]:
    cols = _level_columns(df)
    if not cols:   # no level-named columns: drop a leading code/id-like column (mostly numeric/short codes)
        cols = list(df.columns)
        first = df[cols[0]].astype(str)
        if len(cols) > 2 and (first.str.fullmatch(r"[\d.\-_/ ]+").mean() > 0.8 or first.nunique() > 0.9 * len(df)):
            cols = cols[1:]
    cols = cols[:MAX_DEPTH + 1]
    paths = []
    for row in df[cols].itertuples(index=False):
        cells = [str(v).strip() for v in row]
        # strip trailing blanks; blanks in the middle are a data problem -> cut there
        p: list[str] = []
        for v in cells:
            if v == "":
                break
            p.append(v)
        if p:
            paths.append(tuple(p))
    return paths


def _from_path(df: pd.DataFrame, sep: str) -> list[tuple[str, ...]]:
    col = df.columns[0]
    named = [c for c in df.columns if re.search(r"path|categor|taxonom|hierarch", str(c).lower())]
    if named:
        col = named[0]
    elif df.shape[1] > 1:   # pick the column whose values contain a separator most often
        col = max(df.columns, key=lambda c: df[c].astype(str).str.contains(r">|/|\||\\").sum())
    vals = df[col].astype(str)
    cands = [sep, ">", "/", "|", "\\", ";"]
    best = max(cands, key=lambda s: vals.str.contains(re.escape(s)).sum() if s else -1)
    if not vals.str.contains(re.escape(best)).any():
        best = sep
    paths = []
    for v in vals:
        parts = tuple(x.strip() for x in v.split(best) if x.strip())
        if parts:
            paths.append(parts)
    return paths


def _from_adjacency(df: pd.DataFrame) -> list[tuple[str, ...]]:
    cl = {str(c).lower(): c for c in df.columns}
    par_c = next((c for k, c in cl.items() if "parent" in k), None)
    id_c = next((cl[k] for k in ("id", "node_id", "code", "category_id", "category_code") if k in cl), df.columns[0])
    rest = [c for c in df.columns if c not in (par_c, id_c)]
    name_c = next((cl[k] for k in ("name", "label", "category", "title", "description", "desc") if k in cl), rest[0] if rest else df.columns[1])
    if par_c is None:
        raise ConfigError("adjacency taxonomy needs a parent column")
    names, parent = {}, {}
    for i, p, n in zip(df[id_c].astype(str), df[par_c].astype(str), df[name_c].astype(str)):
        names[i.strip()] = n.strip()
        parent[i.strip()] = p.strip()
    children = {p for p in parent.values() if p}
    paths = []
    for i in names:
        if i in children:
            continue  # internal node, not a leaf
        chain, cur, guard = [], i, 0
        while cur and cur in names and guard < 10:
            chain.append(names[cur])
            cur = parent.get(cur, "")
            guard += 1
        paths.append(tuple(reversed(chain)))
    return paths
