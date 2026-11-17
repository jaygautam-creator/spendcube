"""Profiler: inspect an unseen company file (+ tree) and report what a human needs to write the config.

`profile_file(path, taxonomy_path)` -> dict (machine readable); `format_report(profile)` -> text.
Role guessing uses BOTH column names (EN/DE/FR/ES/PT/IT keywords) and content (numeric / date / currency
code / low-cardinality text). Nothing here is company specific.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..ingest.readers import _decode, read_table, sniff_delimiter
from ..normalize.dates import detect_dayfirst, parse_date, parse_date_series
from ..normalize.numbers import detect_decimal, parse_number

log = logging.getLogger("spendcube.profile")

SAMPLE = 5000

# --------------------------------------------------------------------------- name vocabularies
_TAXISH = r"(vat|mwst|ust|tax|iva|tva|steuer|impuesto|imposto|gst|btw|moms)"
_IDISH = r"(id|no|nr|num|number|nummer|code|key|kto|numero|numéro)"
_AMT = r"(amount|betrag|value|valor|montant|importe|importo|total|summe|wert|net|gross|brutto|netto|ttc|bruto|neto|lordo|price|preis|cost|kosten|spend)"

ROLE_NAME = {
    "line_id": r"(^|\b)(line ?(id|no|nr|number)|row ?(id|no)|record ?(id|no)?|beleg ?(nr|no|id)?|belegnr|document ?(id|no|nr)|doc ?(id|no)|id)(\b|$)",
    "invoice_number": r"(invoice|rechnung|factura|facture|fattura|fatura|bill|inv)[ _]?(no|nr|num|number|nummer|id|numero|ref)",
    "po_reference": r"(^|\b)(po|p o|purchase ?order|bestell|commande|pedido|ordem|ordine|order ?(no|nr|num|number)?)(\b|$)",
    "material_id": r"(material|artikel ?(nr|no)|item ?(code|no|nr|number)|sku|article|product ?(code|id)|part ?(no|nr|number)|matnr|produit|producto)",
    "quantity": r"(^|\b)(qty|quantity|menge|quantit[eé]|cantidad|quantidade|quantit[aà]|units?)(\b|$)",
    "uom": r"(^|\b)(uom|unit|einheit|unit[eé]|unidad|unidade|unit[aà]|me)(\b|$)",
    "cost_centre": r"(cost ?cent(er|re)|kostenstelle|centre de co[uû]t|kst|costcentre|centro de coste|centro de custo|cc)",
    "currency": r"(curr|ccy|w[aä]hrung|devise|moneda|moeda|valuta|iso)",
    "fx_rate": r"(exchange|wechselkurs|kurs|taux|tipo de cambio|fx|rate)",
    "status": r"(status|state|zustand|statut|estado|stato|freigabe|approval)",
    "tax_rate": _TAXISH + r".*(rate|satz|taux|tasa|aliquota|pct|percent|%)|(rate|satz|taux|tasa).*" + _TAXISH + "|^" + _TAXISH + "$",
    "tax_amount": _TAXISH + r".*(amount|betrag|montant|importe|value|wert)|(amount|betrag|montant|importe).*" + _TAXISH,
    "supplier_name": r"(supplier|vendor|lieferant|fournisseur|proveedor|fornecedor|creditor|kreditor|payee|beneficiary|creditore|name)",
    "vendor_id": r"(supplier|vendor|lieferant|fournisseur|proveedor|fornecedor|kreditor|creditor)[ _]?" + _IDISH + r"|" + _IDISH + r"[ _]?(supplier|vendor)",
    "description": r"(desc|text|beschreibung|bezeichnung|libell|descripci|descri|narrative|memo|particular|item name|line ?text|artikel ?text)",
    "gl_account": r"(^|\b)(gl|account|konto|sachkonto|compte|cuenta|conta|ledger|nominal)",
    "gl_description": r"(gl|account|konto|sachkonto|compte|cuenta|conta|ledger)[ _]?(desc|text|name|bezeichnung|descr|libell)|(desc|text|name)[ _]?(of )?(gl|account|konto)",
    "invoice_date": r"(invoice|rechnung|factura|facture|fattura|bill|belegdatum|document)[ _]?(date|datum|fecha|data)|^(date|datum|fecha|data)$",
    "posting_date": r"(post|buchung|booking|gl|accounting)[ _]?(ing)?[ _]?(date|datum|fecha|data)",
}
_NOT_DATE = re.compile(r"(due|f[aä]llig|paid|zahl|payment|created|erstellt|modified|changed|vencim|echeance|update)", re.I)

# multilingual status vocabulary (casefolded substring/exact match) -> status class
STATUS_VOCAB = {
    "INCLUDE": ["paid", "approved", "posted", "released", "cleared", "settled", "complete", "bezahlt", "freigegeben",
                "gebucht", "verbucht", "beglichen", "pay[eé]", "approuv", "valid", "r[eé]gl", "pagad", "aprobad",
                "contabilizado", "pago", "aprovado", "pagato", "approvato", "ok", "open", "offen"],
    "CANCELLED": ["cancel", "annul", "storn", "void", "revers", "reject[eé]e? et", "anulad", "cancelad", "stornato",
                  "annullato", "deleted", "gel[oö]scht"],
    "NOT_APPROVED": ["reject", "abgelehnt", "refus", "rechaz", "rejeit", "rifiut", "on hold", "hold", "pending",
                     "draft", "submitted", "dispute", "kl[aä]rung", "gesperrt", "blocked", "en attente", "attente",
                     "pendiente", "pendente", "in sospeso", "entwurf", "brouillon", "borrador", "eingereicht",
                     "unapproved", "nicht freigegeben", "review", "kl(ae|[aä])rung", "gepr[uü]ft werden", "in pr[uü]fung"],
}
# NON_PROCUREMENT keyword scan (multilingual); word-ish matching
NP_KEYWORDS = [
    "tax", "taxes", "vat", "gst", "iva", "tva", "mwst", "ust", "umsatzsteuer", "steuer", "impuesto", "imp[oô]t", "taxe",
    "intercompany", "inter-company", "inter company", "i/c", "ic", "konzern", "verrechnung", "intragroup",
    "payroll", "salary", "salaries", "wage", "wages", "gehalt", "lohn", "n[oó]mina", "salaire", "sueldo",
    "interest", "zins", "int[eé]r[eê]t", "inter[eé]s", "loan", "darlehen", "pr[eê]t", "dividend", "dividende",
    "royalt", "lizenzgeb", "redevance", "revenue", "umsatzerl", "sales", "rebate", "refund", "r[uü]ckerstattung",
    "bank charge", "bankgeb", "fee", "donation", "spende", "penalty", "fine", "bu[sß]geld", "depreciation",
    "abschreibung", "amortization", "dues", "customs duty", "zoll",
]
_NP_RX = re.compile(r"(?<![a-z0-9])(" + "|".join(NP_KEYWORDS) + r")(?![a-z0-9])", re.I)


def _norm_name(c: str) -> str:
    return re.sub(r"[^a-z0-9%äöüéèàáíóúñçß]+", " ", str(c).lower()).strip()


# --------------------------------------------------------------------------- loading
def load_any(path: str | Path) -> tuple[pd.DataFrame, dict]:
    p = Path(path)
    meta: dict[str, Any] = {"file": p.name, "suffix": p.suffix.lower()}
    if p.suffix.lower() in (".csv", ".txt", ".tsv", ""):
        raw = p.read_bytes()
        text, enc = _decode(raw, "auto")
        delim = sniff_delimiter(text)
        meta.update(encoding=enc, delimiter=delim)
    else:
        meta.update(encoding="n/a", delimiter="n/a")
    df = read_table(p)
    meta["rows"], meta["cols"] = len(df), df.shape[1]
    return df, meta


# --------------------------------------------------------------------------- column features
def _vals(s: pd.Series) -> list[str]:
    v = s.astype(str).str.strip()
    v = v[v != ""]
    if len(v) > SAMPLE:
        v = v.sample(SAMPLE, random_state=0)
    return v.tolist()


_DATE_LIKE = re.compile(r"^\d{1,4}[./\-]\d{1,2}[./\-]\d{2,4}|^\d{1,2}[\s\-./]*[A-Za-z]{3,9}[\s\-./,]*\d{2,4}$|^\d{8}$|^\d{4}-\d{2}-\d{2}")


# strict numeric shape (parse_number is lenient: it strips letters, so "gloves 22" would parse as 22)
_STRICT_NUM = re.compile(r"^\(?[-+]?\s*(?:[A-Za-z]{1,3}\s+|[$€£¥]\s*)?\d[\d.,\s]*\)?[-+]?\s*(?:%|[A-Za-z]{3})?$")


def column_features(name: str, s: pd.Series) -> dict:
    n = len(s)
    v = _vals(s)
    blank = float((s.astype(str).str.strip() == "").mean()) if n else 1.0
    f: dict[str, Any] = {"name": name, "blank_share": round(blank, 4), "n_distinct": int(s.nunique()),
                         "unique_ratio": round(s.nunique() / max(n, 1), 4), "sample": v[:3]}
    if not v:
        f.update(num_frac=0.0, date_frac=0.0, cur_frac=0.0, mean_len=0.0, max_num=None, kind="blank")
        return f
    dec = detect_decimal(v)
    nums = [parse_number(x, dec) if _STRICT_NUM.match(x) else math.nan for x in v]
    ok = [x for x in nums if not math.isnan(x)]
    f["num_frac"] = round(len(ok) / len(v), 4)
    f["decimal"] = dec
    f["max_num"] = float(np.max(np.abs(ok))) if ok else None
    f["neg_share"] = round(sum(1 for x in ok if x < 0) / max(len(ok), 1), 4) if ok else 0.0
    f["has_decimals"] = bool(ok) and any(abs(x - round(x)) > 1e-9 for x in ok)
    f["date_frac"] = round(sum(1 for x in v if _DATE_LIKE.match(x) and parse_date(x, True) is not None) / len(v), 4)
    f["cur_frac"] = round(sum(1 for x in v if re.fullmatch(r"[A-Z]{3}", x) is not None) / len(v), 4)
    f["mean_len"] = round(float(np.mean([len(x) for x in v])), 2)
    f["has_space_frac"] = round(sum(1 for x in v if " " in x) / len(v), 3)
    f["kind"] = ("date" if f["date_frac"] >= 0.9 else "numeric" if f["num_frac"] >= 0.9
                 else "currency" if f["cur_frac"] >= 0.9 else "text")
    return f


def _name_hit(role: str, nname: str) -> bool:
    return re.search(ROLE_NAME[role], nname, re.I) is not None


def guess_roles(df: pd.DataFrame, feats: dict[str, dict]) -> tuple[dict[str, str], dict[str, list[str]], dict]:
    """Greedy best (role, column) assignment. Returns (roles, description_columns, scores)."""
    cols = list(df.columns)
    nn = {c: _norm_name(c) for c in cols}
    cand: list[tuple[float, str, str]] = []
    for c in cols:
        f, n = feats[c], nn[c]
        taxish = re.search(_TAXISH, n) is not None
        idish = re.search(r"\b" + _IDISH + r"\b|" + _IDISH + "$", n) is not None
        kind = f["kind"]

        def add(role, score):
            if score > 0:
                cand.append((score, role, c))
        # numeric roles
        if kind == "numeric":
            if re.search(_AMT, n) and not taxish and not re.search(r"(rate|kurs|qty|menge|quantity)", n):
                add("amount", 3 + (1 if f["n_distinct"] > 20 else 0) + (0.5 if f.get("has_decimals") else 0))
            elif f.get("has_decimals") and f["n_distinct"] > 50 and not taxish and not idish:
                add("amount", 1.5 + (0.5 if (f["max_num"] or 0) > 100 else 0))
            if taxish and (f["max_num"] or 0) <= 100 and f["n_distinct"] <= 25:
                add("tax_rate", 4)
            elif _name_hit("tax_rate", n) and (f["max_num"] or 0) <= 100 and f["n_distinct"] <= 40:
                add("tax_rate", 3.5)
            if taxish and f.get("has_decimals") and (f["max_num"] or 0) > 30:
                add("tax_amount", 4)
            if _name_hit("quantity", n):
                add("quantity", 4)
            if re.search(r"(rate|kurs|taux|exchange|fx)", n) and not taxish:
                add("fx_rate", 3.5 if f.get("has_decimals") else 2)
        if kind in ("numeric", "text") and _name_hit("tax_rate", n) and "%" in " ".join(f["sample"]):
            add("tax_rate", 4)
        # dates
        if kind == "date" and not _NOT_DATE.search(n):
            if _name_hit("posting_date", n):
                add("posting_date", 4)
            elif _name_hit("invoice_date", n) or re.search(r"(date|datum|fecha|data)", n):
                add("invoice_date", 3.5 if re.search(r"(invoice|rechnung|factura|facture|bill|beleg)", n) else 3)
            else:
                add("invoice_date", 1.5)
        # currency (content alone is strong)
        if kind == "currency" and f["n_distinct"] <= 40:
            add("currency", 3.5 + (1 if _name_hit("currency", n) else 0))
        # ids and text
        if _name_hit("line_id", n) and not re.search(r"(invoice|rechnung|po|vendor|supplier|material|gl|account|konto|lieferant)", n):
            add("line_id", 3 + (1 if f["unique_ratio"] >= 0.99 else 0))
        if _name_hit("invoice_number", n) and kind != "date":
            add("invoice_number", 4)
        if _name_hit("po_reference", n) and kind != "date" and not taxish:
            add("po_reference", 3.5)
        if _name_hit("material_id", n) and kind != "date":
            add("material_id", 3.5)
        if _name_hit("uom", n) and kind in ("text", "currency") and f["n_distinct"] <= 60:
            add("uom", 3)
        if _name_hit("cost_centre", n) and kind != "date" and not re.search(r"desc|text", n):
            add("cost_centre", 3)
        if _name_hit("status", n) and kind in ("text", "currency") and f["n_distinct"] <= 25:
            add("status", 4.5)
        elif kind == "text" and f["n_distinct"] <= 12 and f["mean_len"] <= 20 and f["blank_share"] < 0.5 \
                and f["unique_ratio"] < 0.05 and re.search(r"(stat|approv|release|freig)", n):
            add("status", 3)
        if _name_hit("vendor_id", n) and not re.search(r"name", n):
            add("vendor_id", 4)
        elif re.search(r"(vendor|supplier|lieferant|fournisseur|kreditor|creditor)", n) and not re.search(r"name|desc", n) \
                and f["mean_len"] <= 12 and f["has_space_frac"] < 0.1 and kind != "numeric_big":
            add("vendor_id", 2.5)
        if kind == "text" and _name_hit("supplier_name", n) and not (idish and not re.search(r"name", n)):
            sc = 3.5 if re.search(r"(supplier|vendor|lieferant|fournisseur|proveedor|fornecedor|creditor|kreditor)", n) else 1.5
            if re.search(r"(supplier|vendor|lieferant|fournisseur|kreditor|creditor|proveedor)", n) and (
                    re.search(r"name", n) or f["has_space_frac"] > 0.2 or f["mean_len"] > 12):
                sc += 1.5
            if re.search(r"(type|class|group|typ)\b", n):
                sc -= 3
            add("supplier_name", sc)
        if kind in ("text",) and _name_hit("gl_description", n):
            add("gl_description", 4.5)
        elif _name_hit("gl_account", n) and kind != "date":
            if re.search(r"(desc|text|name|bezeichnung|libell)", n):
                add("gl_description", 4.5)
            else:
                add("gl_account", 3.5)
        if kind == "text" and _name_hit("description", n) and not re.search(r"(gl|account|konto|status|supplier|vendor|lieferant)", n):
            add("description", 3 + (1 if f["mean_len"] >= 8 else 0))
        elif kind == "text" and f["mean_len"] >= 18 and f["has_space_frac"] > 0.5 and f["unique_ratio"] > 0.05:
            add("description", 1.5)
    cand.sort(reverse=True)
    roles: dict[str, str] = {}
    used: set[str] = set()
    scores: dict[str, float] = {}
    for sc, role, c in cand:
        if role in roles or (c in used):
            continue
        if sc < 2.0:
            continue
        roles[role] = c
        used.add(c)
        scores[role] = sc
    # description may span several columns (e.g. material text + service text)
    desc_cols = [roles["description"]] if "description" in roles else []
    for sc, role, c in cand:
        if role == "description" and c not in desc_cols and (c not in used or c == roles.get("description")) and sc >= 3:
            desc_cols.append(c)
    return roles, {"description": desc_cols}, scores


# --------------------------------------------------------------------------- analyses
def _status_suggest(values: dict[str, int]) -> dict:
    out: dict[str, list[str]] = {"INCLUDE": [], "CANCELLED": [], "NOT_APPROVED": []}
    unknown: list[str] = []
    for v in values:
        k = v.strip().casefold()
        hit = None
        for cls in ("CANCELLED", "NOT_APPROVED", "INCLUDE"):       # most specific first
            if any(re.search(r"(^|\b|(?<=[a-z]))" + pat, k) for pat in STATUS_VOCAB[cls]):
                hit = cls
                break
        if k == "" or hit is None:
            unknown.append(v)
        else:
            out[hit].append(v)
    return {"map": out, "unknown": unknown}


def _date_info(s: pd.Series) -> dict:
    vals = _vals(s)
    a_big = b_big = 0
    for v in vals:
        m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})", v)
        if m:
            a_big += int(m[1]) > 12
            b_big += int(m[2]) > 12
    fmts = {"iso": r"^\d{4}-\d{2}-\d{2}", "dd/mm/yyyy": r"^\d{1,2}/\d{1,2}/\d{4}$", "dd.mm.yyyy": r"^\d{1,2}\.\d{1,2}\.\d{4}$",
            "dd-mm-yyyy": r"^\d{1,2}-\d{1,2}-\d{4}$", "dd-Mon-yy": r"^\d{1,2}[\s\-][A-Za-z]{3}[\s\-]\d{2,4}$",
            "yyyymmdd": r"^\d{8}$", "excel serial": r"^\d{5}(\.\d+)?$"}
    counts = {k: sum(1 for v in vals if re.match(p, v)) for k, p in fmts.items()}
    counts = {k: v for k, v in counts.items() if v}
    dayfirst = detect_dayfirst(vals, True)
    d, unp, imp, info = parse_date_series(s, dayfirst)
    ok = d.dropna()
    return {"formats": counts, "dayfirst": bool(dayfirst), "first_field_gt12": a_big, "second_field_gt12": b_big,
            "ambiguous": bool(a_big == 0 and b_big == 0 and any(counts.get(k) for k in ("dd/mm/yyyy", "dd.mm.yyyy", "dd-mm-yyyy"))),
            "unparseable": int(unp.sum()), "impossible": int(imp.sum()), "window": info["window"],
            "min": str(ok.min().date()) if len(ok) else None, "max": str(ok.max().date()) if len(ok) else None}


def _amount_series(df, roles, feats) -> Optional[pd.Series]:
    c = roles.get("amount")
    if not c:
        return None
    dec = feats[c].get("decimal", ".")
    return df[c].astype(str).map(lambda x: parse_number(x, dec)).astype(float)


def _np_candidates(df, roles, amount: Optional[pd.Series]) -> list[dict]:
    out = []
    fields = [("gl_description", "gl_desc"), ("description", "desc"), ("supplier_name", "supplier"),
              ("gl_account", "gl_account")]
    for role, short in fields:
        cols = roles.get(role)
        if role == "description":
            cols = roles.get("description")
        if not cols or cols not in df.columns:
            continue
        s = df[cols].astype(str).str.strip()
        if role == "gl_account":
            continue          # accounts are numeric codes; they are surfaced through their GL description
        hit = s.map(lambda x: (_NP_RX.search(x).group(1).lower() if _NP_RX.search(x) else ""))
        m = hit != ""
        if not m.any():
            continue
        key = s[m].str.replace(r"\d+", "#", regex=True)         # group "Charge entity 7" / "Charge entity 12"
        g = pd.DataFrame({"value": s[m], "key": key, "kw": hit[m]})
        if amount is not None:
            g["spend"] = amount[m].abs().fillna(0).values
        else:
            g["spend"] = 0.0
        agg = g.groupby("key").agg(rows=("value", "size"), example=("value", "first"), keyword=("kw", "first"),
                                   spend=("spend", "sum")).sort_values(["spend", "rows"], ascending=False).head(25)
        for k, r in agg.iterrows():
            out.append({"field": short, "pattern": k, "example": r["example"], "keyword": r["keyword"],
                        "rows": int(r["rows"]), "spend": round(float(r["spend"]), 2),
                        "distinct_values": int(s[m][key == k].nunique())})
    return out


# ---- suggested NON_PROCUREMENT exclusion rules (emitted COMMENTED by draft_config; D-43) ---------------------------
# Heuristic: non-procurement postings are TEMPLATE lines ("<Tax Type> Expense - <period>"), i.e. a short leading phrase
# repeated many times, whereas a professional-services line that merely mentions tax is unique and carries service words.
SUGGEST_CATEGORIES = {
    "intercompany": r"inter-?\s?company|intra-?\s?group|i\s?/\s?c|ic|konzern|verrechnung",
    "tax": r"(sales|use|state|local|property|income|withholding|payroll|excise|accrued)?(\s*&\s*(use|local))?\s*tax(es)?|vat|gst|iva|tva|mwst|umsatzsteuer|steuer|impuesto|imp[oô]t",
    "payroll": r"payroll|salary|salaries|wages?|gehalt|lohn|n[oó]mina|salaire|sueldo",
    "bank_interest": r"bank\s*(charges?|fees?)|bankgeb[uü]hr\w*|interest|zins\w*|int[eé]r[eê]ts?|loan|darlehen",
    "other_non_spend": r"dividends?|depreciation|abschreibung|amortization|penalt(y|ies)|donations?|customs\s+dut(y|ies)|zoll",
}
# words that mark a genuine purchased service/good: veto (keeps "Acme Tax Consultant", "Tax Services ...")
SUGGEST_VETO = re.compile(r"(consult|advis|audit|service|software|licen[sc]e|prepar|training|assist|legal|law\b|work\b|"
                          r"compliance|system|tool|equipment|support|project|program|accounting|solutions?)", re.I)


def _leading_phrase(v: str) -> str:
    v = re.split(r"\s[-\u2013\u2014:|]\s|\d|\(|#|/20", v.strip().lower(), maxsplit=1)[0]
    return re.sub(r"\s+", " ", v).strip(" -,.")


def suggest_exclusions(df, roles, amount: Optional[pd.Series], min_rows_free_text: int = 3, max_per_cat: int = 8) -> list[dict]:
    fields = [("gl_description", "gl_desc", 1, True), ("description", "desc_raw", min_rows_free_text, False),
              ("supplier_name", "supplier_raw", min_rows_free_text, True)]
    out: list[dict] = []
    for role, rfield, min_rows, allow_ic_token in fields:
        col = roles.get(role)
        if not col or col not in df.columns:
            continue
        s = df[col].astype(str)
        ph = s.map(_leading_phrase)
        cnt = ph.value_counts()
        accepted: dict[str, list[str]] = {}
        for cat, rx in SUGGEST_CATEGORIES.items():
            if cat == "intercompany" and not allow_ic_token:
                rx = r"inter-?\s?company|intra-?\s?group|konzern|verrechnung"
            crx = re.compile(r"^(" + rx + r")(?![a-z])", re.I) if cat != "tax" else re.compile(r"^(" + rx + r")(?![a-z0-9])", re.I)
            for phrase, n in cnt.items():
                if not phrase or len(phrase.split()) > 5 or SUGGEST_VETO.search(phrase):
                    continue
                if not crx.match(phrase) and not (cat == "tax" and re.search(r"(^|\s)(tax(es)?|vat|gst|iva)$", phrase)):
                    continue
                if any(phrase.startswith(a + " ") or phrase == a for lst in accepted.values() for a in lst):
                    continue
                m = ph == phrase
                # also swallow longer phrases that start with this one when they carry no veto word
                rows_n = int(m.sum())
                if rows_n < min_rows:
                    continue
                accepted.setdefault(cat, []).append(phrase)
        for cat, phrases in accepted.items():
            for phrase in phrases[:max_per_cat * 3]:
                toks = [re.escape(t).replace("\\&", "&") for t in phrase.split()]
                rx_str = r"(?i)^\s*" + r"\s+".join(toks) + r"(?![a-z])"
                m = s.str.match(rx_str)
                m &= ~s.map(lambda x: bool(SUGGEST_VETO.search(_leading_phrase(x))))
                out.append({"category": cat, "field": rfield, "regex": rx_str, "phrase": phrase, "rows": int(m.sum()),
                            "spend": round(float(amount.abs()[m].sum()), 2) if amount is not None else 0.0})
    out = [o for o in out if o["rows"] > 0]
    out = [o for o in out if not any(p2["field"] == o["field"] and o["phrase"].startswith(p2["phrase"] + " ") for p2 in out)]
    out.sort(key=lambda o: (o["category"], -o["spend"]))
    return out


def _dup_stats(df, roles) -> dict:
    idc = roles.get("line_id")
    cols = [c for c in df.columns if c != idc]
    d = df.duplicated(subset=cols, keep=False)
    out: dict[str, Any] = {"exact_dup_rows_excl_id": int(d.sum()), "exact_dup_extra_copies": int(df.duplicated(subset=cols).sum())}
    if idc and d.any():
        nums = pd.to_numeric(df.loc[d, idc].astype(str).str.extract(r"(\d+)\s*$")[0], errors="coerce")
        grp = df.loc[d, cols].astype(str).agg("\x1f".join, axis=1)
        cons = 0
        tot = 0
        for _, idx in grp.groupby(grp).groups.items():
            v = sorted(nums.loc[idx].dropna().tolist())
            if len(v) > 1:
                tot += 1
                cons += int(all(b - a <= 3 for a, b in zip(v, v[1:])))
        out["dup_groups"] = tot
        out["dup_groups_consecutive_ids"] = cons
        out["hint"] = ("consecutive ids: looks like re-exports, consider intra_source_dedup"
                       if tot and cons / tot > 0.7 else
                       "non-consecutive ids: probably genuine repeat purchases, leave dedup OFF (D-14)")
    inv, amt, sup = roles.get("invoice_number"), roles.get("amount"), roles.get("supplier_name")
    if inv and amt:
        keys = [c for c in (inv, amt, sup) if c]
        out["same_invoice_amount_supplier_rows"] = int(df.duplicated(subset=keys, keep=False).sum())
    return out


def _concentration(df, roles, amount) -> dict:
    if amount is None:
        return {}
    key = df[roles["supplier_name"]].astype(str) if "supplier_name" in roles else (
        df[roles["vendor_id"]].astype(str) if "vendor_id" in roles else pd.Series("", index=df.index))
    dcol = roles.get("description")
    if dcol:
        key = key + "|" + df[dcol].astype(str).str.upper().str.replace(r"\s+", " ", regex=True)
    sp = amount.abs().fillna(0).groupby(key).sum().sort_values(ascending=False)
    tot = float(sp.sum())
    out: dict[str, Any] = {"units": int(len(sp)), "total_abs_amount": round(tot, 2)}
    if tot > 0:
        cum = sp.cumsum() / tot
        for q in (0.5, 0.8, 0.9, 0.95):
            out[f"units_for_{int(q * 100)}pct"] = int((cum < q).sum() + 1)
    return out


def taxonomy_summary(path: str | Path) -> dict:
    from ..config.schema import TaxonomyCfg
    from ..ingest.readers import read_many
    from ..taxonomy.tree import Taxonomy, detect_format, _level_columns
    out: dict[str, Any] = {"file": Path(path).name}
    try:
        df = read_many([Path(path)])
        out["raw_columns"] = list(df.columns)
        out["raw_rows"] = len(df)
        out["detected_format"] = detect_format(df, " > ")
        out["level_columns"] = _level_columns(df)
        t = Taxonomy.load(TaxonomyCfg(file=str(path), format="auto"))
        by_depth: dict[int, int] = {}
        for l in t.leaves:
            by_depth[l.depth] = by_depth.get(l.depth, 0) + 1
        names: dict[str, list] = {}
        for l in t.leaves:
            names.setdefault(l.path[-1].casefold(), []).append(" > ".join(l.path))
        out.update(ok=True, leaves=len(t), max_depth=t.depth, leaves_per_depth=dict(sorted(by_depth.items())),
                   l1_nodes=sorted({l.path[0] for l in t.leaves}),
                   duplicate_leaf_names={k: v for k, v in names.items() if len(v) > 1},
                   summary=t.summary(), sample_paths=[" > ".join(l.path) for l in t.leaves[:6]])
        warn = []
        if len(t) < 5:
            warn.append("fewer than 5 leaves: check delimiter/format")
        if len(df) != len(t.leaves):
            warn.append(f"{len(df)} file rows -> {len(t.leaves)} leaves (duplicates/internal rows collapsed)")
        if t.depth > 4:
            warn.append("deeper than 4 levels: truncated")
        l1 = out["l1_nodes"]
        if l1 and sum(bool(re.fullmatch(r"[\d.\-]+", x)) for x in l1) >= max(1, len(l1) // 2):
            warn.append("L1 names look like numeric codes: an ID/code column was probably parsed as a tree level. "
                        "Check the sample paths; the tree may need a cleaned copy of the file (drop the code column)")
        if len(l1) == len(t) and len(t) >= 5 and t.depth > 1:
            warn.append("every leaf has its own L1 node: suspicious (wrong columns or separator?)")
        low = [str(c).lower() for c in df.columns]
        if any(c.startswith("parent") for c in low) and out["detected_format"] != "adjacency":
            warn.append("a parent-like column exists but the tree was NOT parsed as adjacency: set taxonomy.format explicitly")
        if out["detected_format"] == "path" and df.shape[1] > 1:
            warn.append("path format with several columns: confirm the path column was the one used (see sample paths)")
        out["warnings"] = warn
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


# --------------------------------------------------------------------------- main API
def profile_file(path: str | Path, taxonomy_path: Optional[str | Path] = None,
                 fx_search_dir: Optional[str | Path] = None) -> dict:
    df, meta = load_any(path)
    feats = {c: column_features(c, df[c]) for c in df.columns}
    roles, multi, scores = guess_roles(df, feats)
    p: dict[str, Any] = {"meta": meta, "columns": feats, "roles": roles, "role_scores": scores,
                         "description_columns": multi["description"]}
    amt_col = roles.get("amount")
    # ---- amounts
    amount = _amount_series(df, roles, feats)
    if amt_col:
        f = feats[amt_col]
        nm = _norm_name(amt_col)
        basis = ("gross" if re.search(r"(gross|brutto|incl|ttc|bruto|lordo|inkl)", nm) else
                 "net" if re.search(r"(net|netto|neto|excl|\bht\b|netto)", nm) else "unknown")
        has_tax = "tax_rate" in roles or "tax_amount" in roles
        if basis == "unknown" and has_tax:
            basis = "gross?"      # a tax column next to an unlabeled amount usually means gross
        p["amounts"] = {"column": amt_col, "decimal": f["decimal"], "negative_share": f["neg_share"],
                        "zero_share": round(float((amount.fillna(0) == 0).mean()), 4) if amount is not None else None,
                        "blank_share": f["blank_share"], "basis_hint": basis, "has_tax_column": has_tax,
                        "total_abs": round(float(amount.abs().sum()), 2) if amount is not None else None}
    # ---- dates
    p["dates"] = {r: {"column": roles[r], **_date_info(df[roles[r]])} for r in ("invoice_date", "posting_date") if r in roles}
    # ---- currency / fx
    cur = {"column": roles.get("currency")}
    if "currency" in roles:
        vc = df[roles["currency"]].astype(str).str.strip().value_counts()
        cur["values"] = {k: int(v) for k, v in vc.items()}
    cur["rate_column"] = roles.get("fx_rate")
    if "fx_rate" in roles:
        r = pd.to_numeric(df[roles["fx_rate"]].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        cur["rate_blank_share"] = round(float(r.isna().mean()), 4)
        if "currency" in roles:
            cur["rate_blank_share_by_currency"] = {k: round(float(v), 3) for k, v in r.isna().groupby(df[roles["currency"]]).mean().items()}
    d = Path(fx_search_dir) if fx_search_dir else Path(path).parent
    cands = sorted(set(glob.glob(str(d / "*fx*")) + glob.glob(str(d / "*rate*")) + glob.glob(str(d / "*exchange*"))))
    cur["fx_files_nearby"] = [os.path.basename(x) for x in cands if os.path.abspath(x) != os.path.abspath(str(path))]
    p["currency"] = cur
    # ---- status
    if "status" in roles:
        vc = df[roles["status"]].astype(str).str.strip().value_counts()
        vals = {k: int(v) for k, v in vc.items()}
        p["status"] = {"column": roles["status"], "values": vals, **_status_suggest(vals)}
    # ---- description / blanks
    dcols = multi["description"]
    if dcols:
        txt = df[dcols].astype(str).apply(lambda s: s.str.strip()).agg(" ".join, axis=1).str.strip()
        p["description"] = {"columns": dcols, "blank_share": round(float((txt == "").mean()), 4),
                            "mean_len": round(float(txt[txt != ""].str.len().mean()), 1) if (txt != "").any() else 0}
        if amount is not None:
            p["description"]["blank_spend_share"] = round(float(amount.abs()[txt == ""].sum() / max(amount.abs().sum(), 1e-9)), 4)
    # ---- supplier
    if "supplier_name" in roles:
        s = df[roles["supplier_name"]].astype(str)
        from ..entities.legal_forms import supplier_core
        p["supplier"] = {"column": roles["supplier_name"], "distinct_raw": int(s.nunique()),
                         "distinct_cores": int(s.map(supplier_core).nunique())}
    p["duplicates"] = _dup_stats(df, roles)
    p["np_candidates"] = _np_candidates(df, roles, amount)
    p["suggested_exclusions"] = suggest_exclusions(df, roles, amount)
    p["concentration"] = _concentration(df, roles, amount)
    if taxonomy_path:
        p["taxonomy"] = taxonomy_summary(taxonomy_path)
    p["warnings"] = _warnings(p, roles)
    return p


def _warnings(p: dict, roles: dict) -> list[str]:
    w = []
    for need in ("amount", "description"):
        if need not in roles:
            w.append(f"NO column guessed for '{need}': map it manually")
    if "supplier_name" not in roles and "vendor_id" not in roles:
        w.append("no supplier/vendor column found: supplier_key will be 'S:UNKNOWN' for all rows")
    if "status" not in roles:
        w.append("no status column found: every row will count as approved (check for cancelled/rejected markers)")
    a = p.get("amounts", {})
    if a.get("basis_hint") in ("unknown", "gross?"):
        w.append(f"amount basis is {a.get('basis_hint')}: decide net or gross (set amount_basis; if gross also tax_rate/default_tax_rate)")
    cur = p.get("currency", {})
    if cur.get("values") and len(cur["values"]) > 1 and not cur.get("rate_column") and not cur.get("fx_files_nearby"):
        w.append("several currencies but no rate column and no fx file nearby: set fx.static_rates (units per 1 reporting currency)")
    st = p.get("status", {})
    if st.get("unknown"):
        w.append(f"status values not recognised: {st['unknown']}: classify them in status_map")
    for d, di in p.get("dates", {}).items():
        if di.get("ambiguous"):
            w.append(f"{d}: dd/mm vs mm/dd is ambiguous (no field >12): confirm dayfirst")
        if di.get("impossible"):
            w.append(f"{d}: {di['impossible']} dates outside the plausible window {di['window']}")
    if p.get("description", {}).get("blank_share", 0) > 0.3:
        w.append(f"{p['description']['blank_share']:.0%} of descriptions are blank: classification there relies on supplier/GL propagation")
    t = p.get("taxonomy", {})
    if t and not t.get("ok", True):
        w.append(f"TAXONOMY FAILED to parse: {t.get('error')}")
    for x in t.get("warnings", []) if t else []:
        w.append("taxonomy: " + x)
    return w


# --------------------------------------------------------------------------- report text
def format_report(p: dict) -> str:
    L: list[str] = []
    m = p["meta"]
    L.append(f"=== PROFILE of {m['file']} ===")
    L.append(f"rows={m['rows']} cols={m['cols']} delimiter={m['delimiter']!r} encoding={m['encoding']}")
    L.append("")
    L.append("-- column role guesses (score) --")
    for r, c in sorted(p["roles"].items(), key=lambda kv: -p["role_scores"].get(kv[0], 0)):
        f = p["columns"][c]
        L.append(f"  {r:15s} <- {c!r:28s} ({p['role_scores'][r]:.1f}) kind={f['kind']} blank={f['blank_share']:.0%} e.g. {f['sample'][:2]}")
    if len(p["description_columns"]) > 1:
        L.append(f"  description spans columns: {p['description_columns']}")
    un = [c for c in p["columns"] if c not in p["roles"].values()]
    L.append(f"  unassigned columns: {un}")
    a = p.get("amounts")
    if a:
        L.append("")
        L.append("-- amounts --")
        L.append(f"  column={a['column']!r} decimal={a['decimal']!r} basis_hint={a['basis_hint']} tax_column={a['has_tax_column']} "
                 f"negatives={a['negative_share']:.1%} zero={a['zero_share']:.1%} blank={a['blank_share']:.1%} sum|x|={a['total_abs']:,.0f}")
    if p.get("dates"):
        L.append("")
        L.append("-- dates --")
        for r, d in p["dates"].items():
            L.append(f"  {r} ({d['column']}): formats={d['formats']} dayfirst={d['dayfirst']} (first>12: {d['first_field_gt12']}, "
                     f"second>12: {d['second_field_gt12']}{', AMBIGUOUS' if d['ambiguous'] else ''}) range={d['min']}..{d['max']} "
                     f"unparseable={d['unparseable']} impossible={d['impossible']} window={d['window']}")
    c = p["currency"]
    L.append("")
    L.append("-- currency / fx --")
    L.append(f"  currency column={c.get('column')!r} values={c.get('values')} rate column={c.get('rate_column')!r} "
             f"rate_blank={c.get('rate_blank_share')} fx files nearby={c.get('fx_files_nearby')}")
    if "status" in p:
        s = p["status"]
        L.append("")
        L.append(f"-- status ({s['column']!r}) --")
        for v, n in s["values"].items():
            cls = next((k for k, vs in s["map"].items() if v in vs), "??? UNKNOWN")
            L.append(f"  {v!r}: {n}  -> {cls}")
    if "description" in p:
        d = p["description"]
        L.append("")
        L.append(f"-- description -- blank={d['blank_share']:.1%} mean_len={d['mean_len']} blank_spend_share={d.get('blank_spend_share')}")
    if "supplier" in p:
        s = p["supplier"]
        L.append(f"-- supplier -- {s['distinct_raw']} spellings -> {s['distinct_cores']} cores after legal-form/whitespace normalisation")
    L.append("")
    L.append(f"-- duplicates (D-14 evidence) -- {p['duplicates']}")
    L.append("")
    L.append("-- candidate NON_PROCUREMENT values (review each, then add as exclusion_rules) --")
    for x in p["np_candidates"][:40]:
        L.append(f"  [{x['field']}] kw={x['keyword']!r} rows={x['rows']} spend={x['spend']:,.0f} pattern={x['pattern']!r} e.g. {x['example']!r}")
    if not p["np_candidates"]:
        L.append("  (none found)")
    if p.get("concentration"):
        L.append("")
        L.append(f"-- spend concentration -- {p['concentration']}")
    if "taxonomy" in p:
        t = p["taxonomy"]
        L.append("")
        L.append("-- taxonomy --")
        if t.get("ok"):
            L.append(f"  format={t['detected_format']} {t['summary']}")
            L.append(f"  L1: {t['l1_nodes']}")
            L.append(f"  sample: {t['sample_paths']}")
            if t["duplicate_leaf_names"]:
                L.append(f"  duplicate leaf names (ok, leaves are id-based): {list(t['duplicate_leaf_names'])[:5]}")
        else:
            L.append(f"  FAILED: {t.get('error')}")
        for w in t.get("warnings", []):
            L.append("  ! " + w)
    L.append("")
    L.append("-- WARNINGS / TODO --")
    for w in p["warnings"] or ["(none)"]:
        L.append("  * " + w)
    return "\n".join(L)
