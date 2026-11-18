"""Robust table readers. Everything is read as str; nothing is type-inferred."""
from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("spendcube.ingest")

_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
_DELIMS = [",", ";", "\t", "|"]


def _decode(raw: bytes, encoding: str) -> tuple[str, str]:
    encs = _ENCODINGS if encoding == "auto" else [encoding]
    last: Optional[Exception] = None
    for enc in encs:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError as e:
            last = e
    return raw.decode("latin-1", errors="replace"), "latin-1(replace)"


def sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    try:
        d = csv.Sniffer().sniff(sample, delimiters="".join(_DELIMS)).delimiter
        if d in _DELIMS:
            return d
    except csv.Error:
        pass
    first = sample.splitlines()[0] if sample else ""
    counts = {d: first.count(d) for d in _DELIMS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_text(path: Path, delimiter: str = "auto", encoding: str = "auto") -> pd.DataFrame:
    raw = path.read_bytes()
    text, enc = _decode(raw, encoding)
    delim = sniff_delimiter(text) if delimiter == "auto" else ("\t" if delimiter == "\\t" else delimiter)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False,
                         na_filter=False, skipinitialspace=False)
    except Exception as e:  # malformed quoting etc. -> python engine, tolerate ragged rows
        log.warning("csv C-engine read failed for %s (%s); retrying with python engine", path.name, e)
        df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str, keep_default_na=False,
                         na_filter=False, engine="python", on_bad_lines="warn")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    log.info("read %s: %d rows x %d cols (enc=%s delim=%r)", path.name, len(df), df.shape[1], enc, delim)
    return df


def _excel_header_fix(path: Path, sheet, df: pd.DataFrame) -> pd.DataFrame:
    """Title/banner rows above the real header (e.g. a report title in A1) make pandas name most columns
    'Unnamed: n'. Then use the first of the top 20 rows that fills >= 60% of the columns as the header."""
    cols = [str(c) for c in df.columns]
    if not cols or sum(c.startswith("Unnamed:") for c in cols) <= len(cols) / 2:
        return df
    raw = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str, header=None,
                        keep_default_na=False, na_filter=False)
    need = max(2, int(0.6 * raw.shape[1] + 0.999))
    for i in range(min(20, len(raw))):
        if (raw.iloc[i].astype(str).str.strip() != "").sum() >= need:
            out = raw.iloc[i + 1:].reset_index(drop=True)
            out.columns = [str(c).strip() or f"col_{j}" for j, c in enumerate(raw.iloc[i])]
            log.warning("%s: header found on sheet row %d (title rows above it skipped)", path.name, i + 1)
            return out
    return df


def read_table(path: Path, delimiter: str = "auto", encoding: str = "auto",
               sheet: Optional[str | int] = None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str,
                           keep_default_na=False, na_filter=False)
        df = _excel_header_fix(path, sheet, df)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.astype(str)
        log.info("read %s (excel): %d rows", path.name, len(df))
        return df
    if suf == ".parquet":
        return pd.read_parquet(path).astype(str)
    if suf in (".jsonl", ".ndjson"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows).astype(str)
    if suf == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))).astype(str)
    return read_csv_text(path, delimiter, encoding)


def read_many(paths: list[Path], delimiter: str = "auto", encoding: str = "auto",
              sheet: Optional[str | int] = None) -> pd.DataFrame:
    frames = [read_table(p, delimiter, encoding, sheet) for p in paths]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
