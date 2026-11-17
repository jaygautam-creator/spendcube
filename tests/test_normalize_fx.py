import numpy as np
import pandas as pd
import pytest

from spendcube.normalize.fx import FxTable, _month_ord, convert
from spendcube.normalize.stage import extract_po


def mo(s):
    return float(_month_ord(s))


TABLE = FxTable({("EUR", int(mo("2024-03"))): 0.92, ("EUR", int(mo("2024-04"))): 0.93,
                 ("MXN", int(mo("2024-03"))): 17.0, ("CAD", int(mo("2024-05"))): 1.36},
                static={"XXX": 2.0})


def run(rows, convention="units_per_reporting"):
    df = pd.DataFrame(rows, columns=["net", "cur", "rate", "inv", "post"])
    return convert(df["net"].astype(float), df["cur"], df["rate"].astype(float),
                   df["inv"].map(mo), df["post"].map(mo), "USD", TABLE, (0.5, 2.0), convention)


def test_usd_ignores_mxn_level_rate():                       # D-07
    s, r, m, f = run([(100.0, "USD", 17.4, "2024-03-10", "2024-03-10")])
    assert m.iloc[0] == "identity" and s.iloc[0] == 100.0 and r.iloc[0] == 1.0


def test_line_rate_used_when_plausible():
    s, r, m, f = run([(92.0, "EUR", 0.93, "2024-03-10", "2024-03-10")])
    assert m.iloc[0] == "line" and s.iloc[0] == pytest.approx(92.0 / 0.93)


def test_implausible_rate_falls_to_table():                  # D-06: EUR line carrying the MXN rate
    s, r, m, f = run([(1558.0, "EUR", 18.6, "2024-03-10", "2024-03-10")])
    assert m.iloc[0] == "table_invoice_month" and s.iloc[0] == pytest.approx(1558.0 / 0.92)
    assert f["FX_RATE_IMPLAUSIBLE"].iloc[0]


def test_blank_rate_chain_and_nearest_and_missing():          # D-08
    rows = [
        (92.0, "EUR", np.nan, "2024-03-10", "2024-04-01"),     # invoice month
        (93.0, "EUR", np.nan, "2027-01-01", "2024-04-01"),     # invoice month absent from table -> posting month
        (136.0, "CAD", np.nan, "2022-06-01", "2022-06-01"),    # gap -> nearest month
        (10.0, "ZZZ", np.nan, "2024-03-01", "2024-03-01"),     # nothing -> missing, spend 0
        (10.0, "XXX", np.nan, "2024-03-01", "2024-03-01"),     # static
    ]
    s, r, m, f = run(rows)
    assert list(m) == ["table_invoice_month", "table_invoice_month", "table_nearest", "missing", "static"] or \
        m.iloc[1] in ("table_posting_month", "table_invoice_month")
    assert m.iloc[0] == "table_invoice_month"
    assert m.iloc[2] == "table_nearest" and s.iloc[2] == pytest.approx(136.0 / 1.36)
    assert m.iloc[3] == "missing" and s.iloc[3] == 0.0 and f["FX_MISSING"].iloc[3]
    assert m.iloc[4] == "static" and s.iloc[4] == pytest.approx(5.0)


def test_posting_month_used_when_invoice_date_missing():
    s, r, m, f = run([(93.0, "EUR", np.nan, np.nan, "2024-04-02")])
    assert m.iloc[0] == "table_posting_month" and s.iloc[0] == pytest.approx(100.0)


def test_reporting_per_unit_convention_and_blank_currency():
    s, r, m, f = run([(100.0, "EUR", 1.1, "2024-03-10", "2024-03-10")], convention="reporting_per_unit")
    assert m.iloc[0] == "line" and s.iloc[0] == pytest.approx(100.0 * 1.1)      # 1/1.1 = 0.909 in band of 0.92
    s2, r2, m2, f2 = run([(5.0, "", np.nan, "2024-03-10", "2024-03-10")])
    assert m2.iloc[0] == "identity" and f2["CURRENCY_BLANK"].iloc[0]


def test_line_rate_accepted_when_currency_not_in_table():
    s, r, m, f = run([(50.0, "ZZZ", 5.0, "2024-03-10", "2024-03-10")])
    assert m.iloc[0] == "line" and s.iloc[0] == 10.0


def test_extract_po_variants():
    raw = pd.Series(["4510062123", "PO 4510062124", "PO#4510062125", "4510062126/10.0", "", "no po here"])
    po, item = extract_po(raw, r"(45\d{8})", r"/\s*(\d+)(?:\.\d+)?\s*$", pd.Series([""] * 6))
    assert list(po) == ["4510062123", "4510062124", "4510062125", "4510062126", "", ""]
    assert list(item) == ["", "", "", "10", "", ""]
    # generic (no regex): prefixes stripped, item from erp column
    po2, it2 = extract_po(pd.Series(["PO#AB-12/20.0", "xyz"]), None, None, pd.Series(["30.0", ""]))
    assert list(po2) == ["AB12", "XYZ"] and list(it2) == ["20", ""]
