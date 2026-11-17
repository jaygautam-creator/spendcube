"""Profiler tests: SYNTHETIC data only. Checks role guessing by name+content across languages and formats."""
from pathlib import Path

import pandas as pd
import pytest

from spendcube.config.loader import load_config
from spendcube.profile.draft_config import draft_config
from spendcube.profile.profiler import _STRICT_NUM, _status_suggest, format_report, profile_file

FIX = Path(__file__).parent / "fixtures" / "synthC"


def test_synthC_roles_and_formats():
    p = profile_file(FIX / "lines_C.csv", FIX / "taxonomy_C.csv")
    r = p["roles"]
    assert p["meta"]["delimiter"] == ";"
    assert r["amount"] == "betrag_brutto" and r["tax_rate"] == "mwst_satz" and r["currency"] == "waehrung"
    assert r["status"] == "status" and r["supplier_name"] == "lieferant" and r["vendor_id"] == "lieferantennr"
    assert r["description"] == "beschreibung" and r["gl_description"] == "sachkontotext" and r["gl_account"] == "sachkonto"
    assert r["invoice_date"] == "rechnungsdatum" and r["line_id"] == "belegnr" and r["quantity"] == "menge"
    assert p["amounts"]["decimal"] == "," and p["amounts"]["basis_hint"] == "gross"
    d = p["dates"]["invoice_date"]
    assert d["formats"] == {"dd.mm.yyyy": 3000} and d["dayfirst"] is True and not d["ambiguous"]
    assert set(p["currency"]["values"]) == {"EUR", "CHF", "USD"}
    assert any("static_rates" in w for w in p["warnings"])           # no rate column and no fx file
    assert p["status"]["unknown"] == []
    assert p["status"]["map"]["CANCELLED"] == ["Storniert"]
    assert sorted(p["status"]["map"]["NOT_APPROVED"]) == ["Abgelehnt", "In Klaerung"]
    assert 0.1 < p["description"]["blank_share"] < 0.2
    assert p["supplier"]["distinct_cores"] < p["supplier"]["distinct_raw"]
    # non-procurement candidates come from the GL description
    kws = {(x["field"], x["keyword"]) for x in p["np_candidates"]}
    assert ("gl_desc", "intercompany") in kws and ("gl_desc", "umsatzsteuer") in kws
    t = p["taxonomy"]
    assert t["ok"] and t["detected_format"] == "path" and t["max_depth"] == 4
    assert t["leaves_per_depth"] == {1: 1, 2: 4, 3: 6, 4: 2}
    assert "PROFILE of lines_C.csv" in format_report(p)


def test_draft_config_loads(tmp_path):
    p = profile_file(FIX / "lines_C.csv", FIX / "taxonomy_C.csv")
    y = draft_config(p, FIX / "lines_C.csv", FIX / "taxonomy_C.csv", llm_include=None, input_dir=str(FIX))
    f = tmp_path / "C.yaml"
    f.write_text(y, encoding="utf-8")
    cfg = load_config(f)
    s = cfg.sources[0]
    assert s.columns["amount"] == "betrag_brutto" and s.columns["amount_basis"] == "gross"
    assert s.columns["invoice_date"]["dayfirst"] is True and s.number_format == {"decimal": ","}
    assert s.status_map.CANCELLED == ["Storniert"]
    assert cfg.exclusion_rules == [] and "# SUGGESTED" in y          # D-43: suggestions are commented out by default
    y2 = draft_config(p, FIX / "lines_C.csv", FIX / "taxonomy_C.csv", llm_include=None, input_dir=str(FIX),
                      accept_suggestions=True)
    f2 = tmp_path / "C2.yaml"
    f2.write_text(y2, encoding="utf-8")
    cfg2 = load_config(f2)
    assert len(cfg2.exclusion_rules) == 3 and {r.field for r in cfg2.exclusion_rules} == {"desc_raw", "gl_desc"}


def test_c_example_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "C.example.yaml", input_dir="blind/C")
    s = cfg.sources[0]
    assert cfg.company == "C" and cfg.out_filename == "jaygautam__C.csv" and cfg.reporting_currency == "USD"
    assert s.columns["amount_basis"] == "net" and s.status_map.CANCELLED[0] == "cancelled"
    assert cfg.taxonomy.file == "blind/C/taxonomy*.csv" and cfg.exclusion_rules == []
    assert cfg.classification["llm"]["chain"]                 # merged from _llm.yaml via `include`


def _write(tmp_path, df, name="x.csv", **kw):
    f = tmp_path / name
    df.to_csv(f, index=False, **kw)
    return f


def test_english_comma_file(tmp_path):
    n = 300
    df = pd.DataFrame({
        "invoice_id": [f"INV{i:05d}" for i in range(n)],
        "supplier_name": [f"Acme {i % 7} Ltd" for i in range(n)],
        "supplier_id": [f"S{i % 7}" for i in range(n)],
        "item_description": [f"Widget model {i % 40} steel bracket" for i in range(n)],
        "po_number": [f"4500{i:06d}" for i in range(n)],
        "amount": [f"{1000 + i * 3.17:,.2f}" for i in range(n)],
        "currency": ["USD" if i % 3 else "EUR" for i in range(n)],
        "exchange_rate": ["" if i % 3 else "0.91" for i in range(n)],
        "posting_date": [f"2024-{1 + i % 12:02d}-{1 + i % 27:02d}" for i in range(n)],
        "approval_status": ["Approved" if i % 9 else "Rejected" for i in range(n)],
        "gl_account": [str(6000 + i % 5) for i in range(n)],
        "gl_account_description": ["Stationery items"] * n,
        "cost_centre": [f"CC{i % 4}" for i in range(n)],
    })
    p = profile_file(_write(tmp_path, df))
    r = p["roles"]
    assert r["amount"] == "amount" and r["currency"] == "currency" and r["fx_rate"] == "exchange_rate"
    assert r["posting_date"] == "posting_date" and r["status"] == "approval_status"
    assert r["supplier_name"] == "supplier_name" and r["vendor_id"] == "supplier_id"
    assert r["po_reference"] == "po_number" and r["description"] == "item_description"
    assert r["gl_description"] == "gl_account_description" and r["gl_account"] == "gl_account"
    assert r["cost_centre"] == "cost_centre" and r["invoice_number"] == "invoice_id"
    assert p["amounts"]["decimal"] == "."
    assert p["status"]["map"]["NOT_APPROVED"] == ["Rejected"]
    assert p["currency"]["rate_column"] == "exchange_rate"


def test_french_and_spanish_headers(tmp_path):
    n = 120
    fr = pd.DataFrame({
        "numero_facture": [f"F{i:05d}" for i in range(n)],
        "fournisseur": [f"Dupont Fils {i % 5} SARL" for i in range(n)],
        "libelle": [f"Achat de fournitures de bureau lot {i % 9}" for i in range(n)],
        "montant_ttc": [f"{(100 + i) * 1.19:.2f}".replace(".", ",") for i in range(n)],
        "taux_tva": ["20"] * n,
        "devise": ["EUR"] * n,
        "statut": ["Payé" if i % 8 else "Annulé" for i in range(n)],
        "date_facture": [f"{1 + i % 28:02d}/{1 + i % 12:02d}/2024" for i in range(n)],
    })
    p = profile_file(_write(tmp_path, fr, "fr.csv", sep=";"))
    r = p["roles"]
    assert r["amount"] == "montant_ttc" and r["tax_rate"] == "taux_tva" and r["currency"] == "devise"
    assert r["status"] == "statut" and r["supplier_name"] == "fournisseur" and r["description"] == "libelle"
    assert p["amounts"]["decimal"] == "," and p["amounts"]["basis_hint"] == "gross"
    assert p["status"]["map"]["CANCELLED"] == ["Annulé"] and p["status"]["map"]["INCLUDE"] == ["Payé"]
    es = pd.DataFrame({"proveedor": [f"Suministros {i % 4} SA" for i in range(n)],
                       "descripcion": [f"Compra material {i % 11}" for i in range(n)],
                       "importe": [f"{200 + i:.2f}" for i in range(n)], "moneda": ["EUR"] * n,
                       "fecha": [f"{1 + i % 28}/{1 + i % 12}/2024" for i in range(n)]})
    p2 = profile_file(_write(tmp_path, es, "es.csv"))
    assert p2["roles"]["amount"] == "importe" and p2["roles"]["supplier_name"] == "proveedor"
    assert p2["roles"]["description"] == "descripcion" and p2["roles"]["currency"] == "moneda"
    assert any("no status column" in w for w in p2["warnings"])


def test_xlsx_and_ambiguous_dates(tmp_path):
    n = 80
    df = pd.DataFrame({"Vendor": [f"Corp {i % 6}" for i in range(n)],
                       "Description": [f"Service {i % 13} support contract" for i in range(n)],
                       "Net Amount": [str(100 + i) for i in range(n)],
                       "Invoice Date": [f"{1 + i % 12:02d}/{1 + i % 12:02d}/2024" for i in range(n)]})
    f = tmp_path / "x.xlsx"
    df.to_excel(f, index=False)
    p = profile_file(f)
    assert p["roles"]["amount"] == "Net Amount" and p["amounts"]["basis_hint"] == "net"
    assert p["roles"]["invoice_date"] == "Invoice Date" and p["dates"]["invoice_date"]["ambiguous"]
    assert any("ambiguous" in w for w in p["warnings"])


def test_strict_numeric_and_status_vocab():
    for ok in ("1,234.56", "1.234,56", "(123.45)", "123-", "R 5,266.63", "12 USD", "19%", "€ 4,50", "-7"):
        assert _STRICT_NUM.match(ok), ok
    for bad in ("V1281", "gloves 22", "ABC", "INV/2023/1", "2024-01-02x"):
        assert not _STRICT_NUM.match(bad), bad
    s = _status_suggest({"Paid": 1, "Cancelled": 1, "On Hold": 1, "Bezahlt": 1, "Storniert": 1, "Mystery": 1, "": 2})
    assert s["map"]["INCLUDE"] == ["Paid", "Bezahlt"] and s["map"]["CANCELLED"] == ["Cancelled", "Storniert"]
    assert s["map"]["NOT_APPROVED"] == ["On Hold"] and set(s["unknown"]) == {"Mystery", ""}


def test_duplicate_stats_hint(tmp_path):
    base = pd.DataFrame({"line_id": [f"L{i}" for i in range(40)], "supplier": ["Foo Inc"] * 40,
                         "description": [f"item number {i // 2}" for i in range(40)],
                         "amount": [str(10 + i // 2) for i in range(40)]})
    p = profile_file(_write(tmp_path, base))
    assert p["duplicates"]["exact_dup_extra_copies"] == 20
    assert "consecutive" in p["duplicates"]["hint"]


def test_silent_taxonomy_misparses_are_flagged(tmp_path):
    # id/code column first in a path file; code + level-name columns; parent_code adjacency: the loader parses
    # all three without error but wrongly, so the profiler must raise a visible warning
    cases = {
        "code_path.csv": "code,path_name\n10,Food > Dairy > Milk\n11,Food > Meat\n12,Tools > Hand > Saw\n13,Tools > Power\n14,IT > Soft\n15,IT > Hard\n",
        "code_levels.csv": "Code;Segment;Family;Class\n1;Food;Dairy;Milk\n2;Food;Meat;Beef\n3;Tools;Hand;Saw\n4;Tools;Power;Drill\n5;IT;Soft;Lic\n6;IT;Hard;Lap\n",
        "parent_code.csv": "code,name,parent_code\n1,Root,\n2,Left,1\n3,Right,1\n4,LL,2\n5,LR,2\n6,RR,3\n",
    }
    for name, text in cases.items():
        f = tmp_path / name
        f.write_text(text, encoding="utf-8")
        p = profile_file(FIX / "lines_C.csv", f)
        if name in ("code_levels.csv", "parent_code.csv"):
            # the loader now parses these correctly (code column skipped; any *parent* column = adjacency)
            want = "adjacency" if name == "parent_code.csv" else "wide"
            assert p["taxonomy"]["detected_format"] == want, (name, p["taxonomy"])
            assert not any("numeric L1" in w or "own root" in w for w in p["warnings"]), (name, p["warnings"])
        else:
            assert any(w.startswith("taxonomy:") for w in p["warnings"]) or p["taxonomy"].get("l1_sample"), (name, p["taxonomy"])


def test_bad_taxonomy_reports_not_raises(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("", encoding="utf-8")
    p = profile_file(FIX / "lines_C.csv", f)
    assert p["taxonomy"]["ok"] is False
    assert any("TAXONOMY FAILED" in w for w in p["warnings"])
