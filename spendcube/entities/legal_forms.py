"""Supplier-name normalisation: strip accents/punctuation/legal forms, remove all whitespace.

Legal-form tokens are generic multilingual business suffixes (not company specific).
"""
from __future__ import annotations

import re
import unicodedata

LEGAL_TOKENS = {
    "PTY", "LTD", "LIMITED", "CC", "SA", "GMBH", "BV", "NV", "AG", "LLC", "INC", "INCORPORATED",
    "CORP", "CORPORATION", "CO", "COMPANY", "LTDA", "SRL", "SPA", "SAS", "SL", "OY", "AB", "AS", "PLC",
    "KK", "LP", "LLP", "PTE", "BHD", "SDN", "PVT", "PRIVATE", "SARL", "SASU", "KG", "OHG", "UG", "EV",
    "THE", "DE", "CV", "SAB", "SNC", "SCA", "DOO", "ZOO", "SP", "SPZOO", "OOO", "ZAO", "EIRL", "SRO",
}
# multi-token legal forms (checked on the space-joined normalised string)
LEGAL_PHRASES = [r"\bS\s*A\s+DE\s+C\s*V\b", r"\bS\s*DE\s+R\s*L\b", r"\bPROPRIETARY\b", r"\bPTY\b"]

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)
_WS = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def supplier_norm(name: str) -> str:
    """Uppercase, accent-free, punctuation-free, legal forms removed, single spaces."""
    if not name:
        return ""
    s = strip_accents(unicodedata.normalize("NFKC", str(name))).upper()
    s = s.replace("&", " AND ")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    for pat in LEGAL_PHRASES:
        s = re.sub(pat, " ", s)
    toks = [t for t in s.split(" ") if t and t not in LEGAL_TOKENS]
    return " ".join(toks)


def supplier_core(name: str) -> str:
    """supplier_norm with all whitespace removed (reproduces golden spacing merges)."""
    return supplier_norm(name).replace(" ", "")
