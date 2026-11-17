import math
from datetime import datetime

import pandas as pd
import pytest

from spendcube.normalize.dates import detect_dayfirst, parse_date, parse_date_series
from spendcube.normalize.numbers import detect_decimal, parse_number, parse_series
from spendcube.normalize.tax import net_amount, parse_rate, parse_rate_series


@pytest.mark.parametrize("s,dec,exp", [
    ("1,234.56", ".", 1234.56), ("1.234,56", ",", 1234.56), ("(123.45)", ".", -123.45),
    ("123.45-", ".", -123.45), ("-5", ".", -5.0), ("R 5,266.63", ".", 5266.63), (" 1 234,5", ",", 1234.5),
    ("USD 12", ".", 12.0), ("12 USD", ".", 12.0), ("1x2", ".", math.nan),
    ("", ".", math.nan), ("abc", ".", math.nan), ("1.2.3", ".", math.nan),
    ("0.01", ".", 0.01), ("+7", ".", 7.0),
])
def test_parse_number(s, dec, exp):
    v = parse_number(s, dec)
    assert (math.isnan(v) and math.isnan(exp)) or v == pytest.approx(exp)


def test_detect_decimal_us_eu_default():
    assert detect_decimal(["1,234.56", "99.5", "1,000"]) == "."
    assert detect_decimal(["1.234,56", "99,5", "7,25"]) == ","
    assert detect_decimal(["1,234", "5,678"]) == "."          # thousands, not decimals
    assert detect_decimal(["12", "7"]) == "."
    s, d = parse_series(pd.Series(["1.234,56", "2,5", ""]))
    assert d == "," and s.iloc[0] == pytest.approx(1234.56) and math.isnan(s.iloc[2])


@pytest.mark.parametrize("raw,exp", [
    ("15", 0.15), ("0.15", 0.15), ("15%", 0.15), (" 15 % ", 0.15), ("", 0.15),   # blank -> default
    ("0", 0.0),                                                                  # D-04: 0 is 0%, not blank
    ("0,15", 0.15), ("7.5", 0.075), ("garbage", 0.15),
])
def test_vat_encodings(raw, exp):
    assert parse_rate(raw, 0.15) == pytest.approx(exp)


def test_vat_no_default_is_nan_and_net():
    assert math.isnan(parse_rate("", None))
    rate = parse_rate_series(pd.Series(["15", "0", ""]), 0.15)
    gross = pd.Series([115.0, 100.0, 230.0])
    net = net_amount(gross, "gross", rate)
    assert list(net.round(2)) == [100.0, 100.0, 200.0]
    assert list(net_amount(gross, "net", rate)) == list(gross)
    # tax_amount takes precedence over the rate
    n2 = net_amount(pd.Series([115.0]), "gross", pd.Series([0.15]), pd.Series([10.0]))
    assert n2.iloc[0] == 105.0


def test_dates_formats():
    assert parse_date("2024-03-29") == datetime(2024, 3, 29)
    assert parse_date("29/03/2024") == datetime(2024, 3, 29)
    assert parse_date("03/29/2024", dayfirst=False) == datetime(2024, 3, 29)
    assert parse_date("29-Mar-24") == datetime(2024, 3, 29)
    assert parse_date("29.03.2024") == datetime(2024, 3, 29)
    assert parse_date("20240329") == datetime(2024, 3, 29)
    assert parse_date("45380") == datetime(2024, 3, 29)             # Excel serial
    assert parse_date("31/02/2024") is None or parse_date("31/02/2024").year == 2024
    assert parse_date("garbage") is None and parse_date("") is None


def test_dayfirst_evidence():
    assert detect_dayfirst(["13/01/2024", "01/02/2024"]) is True
    assert detect_dayfirst(["01/13/2024", "02/01/2024"]) is False
    assert detect_dayfirst(["01/02/2024"], default=False) is False   # ambiguous -> default


def test_impossible_dates_flagged():
    raw = pd.Series(["2024-%02d-15" % m for m in range(1, 13)] * 5 + ["2018-07-23", "2027-01-01", "notadate", ""])
    d, unp, imp, info = parse_date_series(raw)
    assert imp.sum() == 2 and unp.sum() == 1
    assert d.isna().sum() == 4          # 2 impossible + 1 unparseable + 1 blank
    d2, _, imp2, _ = parse_date_series(raw, valid_years=(2018, 2030))
    assert imp2.sum() == 0
