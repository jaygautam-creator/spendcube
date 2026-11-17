"""Stage 1: ingest sources into the canonical frame (columns in schema_cols)."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..config.loader import ConfigError, resolve_files
from ..config.schema import Config, SourceCfg
from ..schema_cols import CONCAT_FIELDS, INGEST_COLUMNS, MAPPED_FIELDS
from .readers import read_many

log = logging.getLogger("spendcube.ingest")

CTX_PREFIX = "ctx__"


def _col_name(spec: Any) -> Any:
    """Column spec may be a str, a list[str], or a dict {column: ..., ...}."""
    if isinstance(spec, dict):
        return spec.get("column")
    return spec


def _series(df: pd.DataFrame, cols: Any, concat: bool, where: str) -> pd.Series:
    if cols is None or cols == "" or cols == []:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    names = [cols] if isinstance(cols, str) else list(cols)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ConfigError(f"{where}: mapped column(s) {missing} not in file (have {list(df.columns)[:40]})")
    if not concat or len(names) == 1:
        return df[names[0]].astype(str)
    parts = [df[c].astype(str).str.strip() for c in names]
    out = parts[0]
    for p in parts[1:]:
        out = (out + " " + p).str.strip()
    return out


def ingest_source(cfg: Config, src: SourceCfg, limit: int | None = None,
                  row_offset: int = 0) -> pd.DataFrame:
    files = resolve_files(src.file, required=src.required)
    if not files:
        log.warning("source %s: no files, skipped (not required)", src.name)
        return pd.DataFrame(columns=INGEST_COLUMNS)
    df = read_many(files, src.read.delimiter, src.read.encoding, src.read.sheet)
    n_limit = limit if limit is not None else src.limit
    if n_limit:
        df = df.head(n_limit).copy()
    where = f"source[{src.name}]"
    out = pd.DataFrame(index=df.index)

    # record id
    rid_col = src.record_id.column
    if rid_col:
        if rid_col not in df.columns:
            raise ConfigError(f"{where}: record_id column '{rid_col}' not in file")
        ids = df[rid_col].astype(str).str.strip()
    else:
        ids = pd.Series(range(1, len(df) + 1), index=df.index).astype(str)
    blank = ids == ""
    if blank.any():
        ids = ids.where(~blank, pd.Series((df.index + 1).astype(str), index=df.index))
    out["record_id_original"] = src.record_id.prefix + ids
    out["record_id"] = out["record_id_original"]
    out["source"] = src.name
    out["source_role"] = src.role
    out["row_order"] = range(row_offset, row_offset + len(df))
    flags = pd.Series([""] * len(df), index=df.index, dtype=object)
    dup = out["record_id"].duplicated(keep=False)
    if dup.any():
        # one row per input record must survive: disambiguate repeated ids
        n = out.groupby("record_id").cumcount()
        out.loc[dup, "record_id"] = out.loc[dup, "record_id"] + "#" + n[dup].astype(str)
        flags[dup] = "DUP_RECORD_ID"
        log.warning("%s: %d rows had duplicated ids; internal ids suffixed with #n", where, int(dup.sum()))

    cols = src.columns
    for key, canon in MAPPED_FIELDS.items():
        spec = _col_name(cols.get(key))
        out[canon] = _series(df, spec, key in CONCAT_FIELDS, where)
    if cols.get("currency") in (None, "") and cols.get("currency_constant"):
        out["currency"] = str(cols["currency_constant"])
    elif cols.get("currency_constant"):
        c = out["currency"].str.strip()
        out["currency"] = c.where(c != "", str(cols["currency_constant"]))

    # date options (dayfirst etc.) are read in normalize; keep the raw strings here
    ctx_cols = cols.get("context") or []
    ctx_cols = [ctx_cols] if isinstance(ctx_cols, str) else list(ctx_cols)
    parts = []
    for c in ctx_cols:
        if c not in df.columns:
            raise ConfigError(f"{where}: context column '{c}' not in file")
        out[CTX_PREFIX + c] = df[c].astype(str).str.strip()
        parts.append(out[CTX_PREFIX + c].map(lambda v, c=c: f"{c}={v}" if v else ""))
    if parts:
        ctx = parts[0]
        for p in parts[1:]:
            ctx = ctx + "|" + p
        out["context"] = ctx.str.replace(r"\|+", "|", regex=True).str.strip("|")
    else:
        out["context"] = ""
    # compact key=value context shown to the LLM (subset of columns; default = all context columns)
    llm_cols = cols.get("llm_context", ctx_cols) or []
    llm_cols = [llm_cols] if isinstance(llm_cols, str) else list(llm_cols)
    lparts = []
    for c in llm_cols:
        if c not in df.columns:
            raise ConfigError(f"{where}: llm_context column '{c}' not in file")
        v = df[c].astype(str).str.strip()
        lparts.append(v.map(lambda x, c=c: f"{c}={x}" if x else ""))
    if lparts:
        lc = lparts[0]
        for p_ in lparts[1:]:
            lc = lc + ";" + p_
        out["llm_context"] = lc.str.replace(r";+", ";", regex=True).str.strip(";")
    else:
        out["llm_context"] = ""
    # warn on entirely blank mapped columns
    for key, canon in MAPPED_FIELDS.items():
        if cols.get(key) and (out[canon].astype(str).str.strip() == "").all() and len(out):
            log.warning("%s: mapped column for '%s' is entirely blank", where, key)
    out["flags"] = flags
    return out.reset_index(drop=True)


def ingest_all(cfg: Config, limit: int | None = None) -> pd.DataFrame:
    frames, offset = [], 0
    for src in cfg.sources:
        f = ingest_source(cfg, src, limit=limit, row_offset=offset)
        offset += len(f)
        frames.append(f)
        log.info("ingested %s [%s]: %d rows", src.name, src.role, len(f))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    ctx_cols = [c for c in df.columns if c.startswith(CTX_PREFIX)]
    for c in ctx_cols:
        df[c] = df[c].fillna("")
    df["context"] = df["context"].fillna("")
    df["llm_context"] = df["llm_context"].fillna("")
    return df
