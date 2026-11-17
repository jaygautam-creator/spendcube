"""R1 feature parsers (synthetic inputs only)."""
import numpy as np
import pandas as pd

from spendcube.reconcile.features import (
    build_features, extract_po, extract_po_item_tool, get_cfg, parse_amount_series, parse_dates,
    parse_vat_series, strip_glued_legal,
)
from spendcube.eval.synth import _blank_frame


def S(*v):
    return pd.Series(list(v), dtype=object)


def test_vat_encodings():
    v = parse_vat_series(S("15", "0.15", "15%", "", " 15 % ", "0", "0.0", "abc"), 0.15)
    assert list(v.round(4)) == [0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.15]   # '0' is zero-rated, blank is default


def test_vat_default_from_config():
    assert parse_vat_series(S(""), 0.10).iloc[0] == 0.10


def test_amounts():
    a = parse_amount_series(S("5,266.63", "-1,000.50", "(123.45)", "99.10-", "", "x", "12"))
    assert a.iloc[0] == 5266.63 and a.iloc[1] == -1000.5 and a.iloc[2] == -123.45 and a.iloc[3] == -99.10
    assert np.isnan(a.iloc[4]) and np.isnan(a.iloc[5]) and a.iloc[6] == 12


def test_po_extraction_and_item():
    raw = S("4510062123", "PO 4510062123", "PO#4510062123", "4510062123/10.0", "", "no po", "PO 12")
    po = extract_po(raw, r"45\d{8}")
    assert list(po) == ["4510062123"] * 4 + ["", "", ""]
    item = extract_po_item_tool(raw, r"/\s*(\d+)(?:\.\d+)?")
    assert item.iloc[3] == 10 and item.iloc[:3].isna().all()


def test_dates_three_formats_dayfirst():
    d = parse_dates(S("2024-03-29", "29/03/2024", "29-Mar-24", "03/04/2024", "", "garbage"))
    assert d.iloc[0] == d.iloc[1] == d.iloc[2] == pd.Timestamp("2024-03-29")
    assert d.iloc[3] == pd.Timestamp("2024-04-03")          # day-first
    assert pd.isna(d.iloc[4]) and pd.isna(d.iloc[5])


def test_glued_legal_suffix():
    assert strip_glued_legal("ACMEPTYLTD") == "ACME"
    assert strip_glued_legal("ACMECC") == "ACME"
    assert strip_glued_legal("ABCC") == "ABCC"               # too short to strip
    assert strip_glued_legal("PLAIN") == "PLAIN"


def test_build_features_net_and_eligibility():
    df = _blank_frame(3)
    df["source"] = ["erp", "tool", "tool"]
    df["amount_raw"] = ["1000.00", "1,150.00", "500.00"]
    df["tax_rate_raw"] = ["", "15%", "0"]
    df["status_raw"] = ["", "Paid", "Cancelled"]
    df["po_raw"] = ["4510000001", "PO#4510000001/20.0", ""]
    df["supplier_raw"] = ["Acme (Pty) Ltd", "ACME PTY LTD", "ACME"]
    f = build_features(df, get_cfg(type("C", (), {"reconciliation": {}})), pd.Series([False, True, True]))
    assert f["f_net"].round(2).tolist() == [1000.0, 1000.0, 500.0]
    assert f["f_core"].tolist() == ["ACME", "ACME", "ACME"]
    assert f["f_poi"].iloc[1] == 20
    assert f["f_eligible"].tolist() == [True, True, False]     # cancelled never claims a receipt
    assert f["f_rank"].tolist()[1] == 3
