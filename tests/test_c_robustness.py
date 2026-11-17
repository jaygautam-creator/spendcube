"""Small synthetic C-like input contract and parser robustness sweep."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from spendcube.config.loader import load_config
from spendcube.output.validator import validate_file
from spendcube.pipeline import run_pipeline
from spendcube.taxonomy.tree import Taxonomy


def _case(tmp_path, rows, *, tree_format="path", depth4=False, delimiter=",", encoding="utf-8",
          bom=False, amount_format="auto", date_values=None, currency_col=True, duplicate_ids=False):
    root = tmp_path
    tax_path = root / "tree.csv"
    if tree_format == "path":
        paths = ["Root > Branch", "Root > Branch > Leaf > Deep"] if depth4 else ["Root > Leaf"]
        tax_path.write_text("path\n" + "\n".join(paths) + "\n", encoding="utf-8")
        taxonomy = {"file": str(tax_path), "format": "path"}
    else:
        if depth4:
            tax_rows = [["Root", "Branch", "Leaf", "Deep"], ["Root", "Branch", "", ""]]
            cols = ["level1", "level2", "level3", "level4"]
        else:
            tax_rows = [["Root", "Leaf", "", ""]]
            cols = ["level1", "level2", "level3", "level4"]
        with tax_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(tax_rows)
        taxonomy = {"file": str(tax_path), "format": "wide"}

    data_path = root / "lines.csv"
    fields = ["line_id", "description", "amount", "date"] + (["currency"] if currency_col else [])
    with data_path.open("w", encoding=encoding, newline="") as f:
        if bom:
            f.write("\ufeff")
        w = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        w.writeheader()
        for i, item in enumerate(rows):
            rid, desc, amount = item
            if duplicate_ids and i == len(rows) - 1:
                rid = rows[0][0]
            date = (date_values[i] if date_values else "2024-01-15")
            row = {"line_id": rid, "description": desc, "amount": amount, "date": date}
            if currency_col:
                row["currency"] = "USD"
            w.writerow(row)

    cfg = {
        "company": "SYN", "candidate_name": "synthetic", "input_dir": str(root), "reporting_currency": "USD",
        "taxonomy": taxonomy,
        "sources": [{"name": "lines", "role": "primary", "file": str(data_path),
                     "record_id": {"column": "line_id"},
                     "read": {"delimiter": delimiter, "encoding": "auto"},
                     "number_format": amount_format,
                     "columns": {"description": "description", "amount": "amount", "amount_basis": "net",
                                 **({"currency": "currency"} if currency_col else {"currency_constant": "USD"}),
                                 "invoice_date": {"column": "date"}},
                     "status_map": {"default": "INCLUDE"}}],
        "classification": {"llm": {"enabled": False}},
    }
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return load_config(cfg_path), data_path


def _run(tmp_path, rows, **options):
    cfg, data = _case(tmp_path, rows, **options)
    out, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=True, cfg_path=str(tmp_path / "config.yaml"))
    assert errors == []
    got = pd.read_csv(out, dtype=str, keep_default_na=False)
    ids = got["record_id"].tolist()
    assert len(got) == len(rows)
    assert validate_file(out, Taxonomy.load(cfg.taxonomy), ids, expected_name=None) == []
    return got


def test_two_level_path_tree_and_one_row(tmp_path):
    got = _run(tmp_path, [("only", "short", "12.50")])
    assert len(got) == 1 and got.loc[0, "include"] == "1"


def test_four_level_ragged_wide_tree(tmp_path):
    got = _run(tmp_path, [("a", "part", "10"), ("b", "service", "20")], tree_format="wide", depth4=True)
    assert len(got) == 2


def test_bom_semicolon_and_iso_latin1(tmp_path):
    rows = [("a", "text", "1,234.56"), ("b", "more text", "2,000.00")]
    got = _run(tmp_path, rows, delimiter=";", encoding="utf-8-sig", bom=True,
                amount_format={"decimal": ".", "thousands": ","})
    assert len(got) == 2


def test_latin1_input_encoding(tmp_path):
    got = _run(tmp_path, [("a", "café", "1.00"), ("b", "pièce", "2.00")], encoding="latin-1")
    assert len(got) == 2


def test_no_currency_column_and_all_one_currency(tmp_path):
    got = _run(tmp_path, [("a", "item", "5.00"), ("b", "item", "8.00")], currency_col=False)
    assert pd.to_numeric(got["spend"]).sum() == pytest.approx(13.0)


def test_currency_column_with_one_currency_value(tmp_path):
    got = _run(tmp_path, [("a", "item", "3.00"), ("b", "item", "7.00")])
    assert pd.to_numeric(got["spend"]).sum() == pytest.approx(10.0)


def test_excel_serial_dates_and_negative_credit_notes(tmp_path):
    got = _run(tmp_path, [("a", "purchase", "100.00"), ("b", "credit", "-20.00")],
                date_values=[45292, 45300])
    assert pd.to_numeric(got["spend"]).sum() == pytest.approx(80.0)


def test_empty_descriptions_and_duplicate_line_ids(tmp_path):
    rows = [(f"id{i}", "" if i % 10 < 3 else "generic item", "1.00") for i in range(10)]
    got = _run(tmp_path, rows, duplicate_ids=True)
    assert got["record_id"].tolist() == [*(f"id{i}" for i in range(9)), "id0"]


@pytest.mark.parametrize("value,fmt,expected", [
    ("1.234,56 €", {"decimal": ",", "thousands": "."}, 1234.56),
    ("$1,234.56", {"decimal": ".", "thousands": ","}, 1234.56),
])
def test_trailing_or_leading_currency_symbols(tmp_path, value, fmt, expected):
    got = _run(tmp_path, [("a", "item", value)], amount_format=fmt)
    assert float(got.loc[0, "spend"]) == pytest.approx(expected)
