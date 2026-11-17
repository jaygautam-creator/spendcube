"""Builds a set of sensitive strings from the (gitignored) task kit and scans text for them.

Used by tests/test_no_data_in_docs.py (full scan of tracked files) and scripts/data_guard.py (light scan of staged
files). Never prints kit rows beyond the offending string itself.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "Cantill_DataML_Task_Kit"
TEXT_EXT = {".md", ".yaml", ".yml", ".py", ".txt", ".toml", ".sh", ".json", ".cfg", ".ini"}
SKIP_PREFIX = ("outputs/", "submission/", "tests/fixtures/")


# Generic words/phrases that happen to be taxonomy names; they are ordinary prose, not task data.
GENERIC = {"training", "safety", "facilities", "logistics", "aluminium", "software", "cleaning", "transformers",
           "professional services", "maintenance services", "office supplies", "office", "consulting", "insurance",
           "security", "utilities", "marketing", "travel", "catering", "legal", "audit", "freight", "packaging"}


def _read(p: Path, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, usecols=cols)


def light_strings(kit: Path = KIT) -> set[str]:
    """Supplier names (>=5 chars) and taxonomy names (>=6 chars)."""
    out: set[str] = set()
    for f, col in (("company_A/erp_goods_receipts.csv", "supplier_name"),
                   ("company_A/invoice_tool_export.csv", "supplier_name_as_captured")):
        p = kit / f
        if p.exists():
            out.update(v.strip() for v in _read(p, [col])[col].unique() if len(v.strip()) >= 5)
    for p in list(kit.glob("company_*/taxonomy_*.csv")):
        df = _read(p)
        for c in df.columns:
            out.update(v.strip() for v in df[c].unique() if len(v.strip()) >= 6)
    return {s for s in out if s.lower() not in GENERIC}


def heavy_strings(kit: Path = KIT) -> set[str]:
    """Material short texts (>=12), B line descriptions (>=15), B vendor ids (V + 6 digits)."""
    out: set[str] = set()
    p = kit / "company_A/erp_goods_receipts.csv"
    if p.exists():
        df = _read(p, ["material_short_text", "service_short_text"])
        for c in df.columns:
            out.update(v.strip() for v in df[c].unique() if len(v.strip()) >= 12 and not v.strip().isalpha())
    p = kit / "company_B/invoice_lines.csv"
    if p.exists():
        df = _read(p, ["line_description", "vendor_id"])
        out.update(v.strip() for v in df["line_description"].unique() if len(v.strip()) >= 15)
        out.update(v.strip() for v in df["vendor_id"].unique() if re.fullmatch(r"V\d{6}", v.strip()))
    return out


def _compile(strings: set[str]):
    if not strings:
        return None
    # longest first; word-boundary-ish match, case-insensitive
    parts = sorted((re.escape(s) for s in strings), key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])", re.I)


def scan_text(text: str, strings: set[str], rx=None) -> list[str]:
    rx = rx or _compile(strings)
    if rx is None:
        return []
    return sorted({m.group(0) for m in rx.finditer(text)})


def scan_files(paths: list[Path], strings: set[str]) -> list[tuple[str, str]]:
    rx = _compile(strings)
    hits = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for s in scan_text(text, strings, rx):
            hits.append((str(p.relative_to(ROOT)) if p.is_absolute() else str(p), s))
    return hits
