"""Date parsing: ISO, dd/mm/yyyy (dayfirst by evidence), dd-Mon-yy, dd.mm.yyyy, yyyymmdd, Excel serials."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T ].*)?$")
_YMD = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_NUM3 = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})(?:[T ].*)?$")
_DMON = re.compile(r"^(\d{1,2})[\s\-./]*([A-Za-z]{3,9})[\s\-./,]*(\d{2}|\d{4})$")
_SERIAL = re.compile(r"^\d{4,5}(?:\.\d+)?$")


def _year(y: str) -> int:
    v = int(y)
    return v if len(y) == 4 else (2000 + v if v < 70 else 1900 + v)


def _mk(y: int, m: int, d: int) -> Optional[datetime]:
    try:
        return datetime(y, m, d)
    except (ValueError, TypeError):
        return None


def parse_date(s: str, dayfirst: bool = True) -> Optional[datetime]:
    t = str(s).strip()
    if not t:
        return None
    m = _ISO.match(t)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]))
    m = _YMD.match(t)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]))
    m = _NUM3.match(t)
    if m:
        a, b, y = int(m[1]), int(m[2]), _year(m[3])
        d, mo = (a, b) if dayfirst else (b, a)
        r = _mk(y, mo, d)
        return r if r else _mk(y, d, mo)      # last resort: swapped
    m = _DMON.match(t)
    if m:
        mon = _MONTHS.get(m[2][:3].upper())
        return _mk(_year(m[3]), mon, int(m[1])) if mon else None
    if _SERIAL.match(t):
        v = float(t)
        if 20000 <= v <= 80000:
            return datetime(1899, 12, 30) + timedelta(days=v)
    return None


def detect_dayfirst(values, default: bool = True) -> bool:
    """Evidence test on a/b/yyyy values: first field >12 => day first; second field >12 => month first."""
    a_big = b_big = 0
    for v in values:
        m = _NUM3.match(str(v).strip())
        if m:
            a_big += int(m[1]) > 12
            b_big += int(m[2]) > 12
    if a_big and not b_big:
        return True
    if b_big and not a_big:
        return False
    return default


def parse_date_series(raw: pd.Series, dayfirst: Optional[bool] = None,
                      valid_years: Optional[tuple[int, int]] = None,
                      default_dayfirst: bool = True):
    """Returns (dates, unparseable_mask, impossible_mask, info). Out-of-window dates become NaT.

    Window (when not configured): [year(q02)-1, year(q98)+1] of the column, so a few outliers
    (2018, 2027 against 2024 data) are flagged while genuine multi-year spans are kept.
    """
    raw = raw.fillna("").astype(str)
    uniq = pd.Series(raw.unique())
    df_used = dayfirst if dayfirst is not None else detect_dayfirst(uniq.tolist(), default_dayfirst)
    parsed = {u: parse_date(u, df_used) for u in uniq}
    dates = pd.to_datetime(raw.map(parsed), errors="coerce")
    nonblank = raw.str.strip() != ""
    unparse = nonblank & dates.isna()
    ok = dates.dropna()
    if valid_years:
        lo, hi = valid_years
    elif len(ok) >= 20:
        lo = int(ok.quantile(0.02).year) - 1
        hi = int(ok.quantile(0.98).year) + 1
    else:
        lo, hi = 1900, 2100
    imposs = dates.notna() & ((dates.dt.year < lo) | (dates.dt.year > hi))
    dates = dates.where(~imposs)
    return dates, unparse, imposs, {"dayfirst": df_used, "window": (lo, hi)}
