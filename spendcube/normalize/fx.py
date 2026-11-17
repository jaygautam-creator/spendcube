"""FX chain (D-06/D-07/D-08). Convention default: units_per_reporting (rate = units of currency per 1 RC).

fx_method values: identity | line | table_invoice_month | table_posting_month | table_nearest | static | missing
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config.loader import resolve_files
from ..ingest.readers import read_many
from .numbers import parse_number

log = logging.getLogger("spendcube.fx")

_MON_RE = re.compile(r"^(\d{4})[-/](\d{1,2})")


def _month_ord(s) -> float:
    """'2024-03' / '2024-03-15' / Timestamp -> ordinal month (year*12+month-1); NaN if not parseable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    if isinstance(s, pd.Timestamp):
        return s.year * 12 + s.month - 1 if not pd.isna(s) else np.nan
    m = _MON_RE.match(str(s).strip())
    return int(m[1]) * 12 + int(m[2]) - 1 if m else np.nan


class FxTable:
    def __init__(self, rates: dict[tuple[str, int], float], static: dict[str, float] | None = None):
        self.rates = rates
        self.static = {k.upper(): float(v) for k, v in (static or {}).items()}
        self._months: dict[str, np.ndarray] = {}
        by: dict[str, list[int]] = {}
        for (c, m) in rates:
            by.setdefault(c, []).append(m)
        for c, ms in by.items():
            self._months[c] = np.array(sorted(ms))

    def get(self, cur: str, mon: float):
        if np.isnan(mon):
            return None
        return self.rates.get((cur, int(mon)))

    def nearest(self, cur: str, mon: float):
        ms = self._months.get(cur)
        if ms is None or len(ms) == 0:
            return None
        if np.isnan(mon):
            mon = float(np.median(ms))
        i = int(np.argmin(np.abs(ms - mon)))
        return self.rates[(cur, int(ms[i]))]

    def has_currency(self, cur: str) -> bool:
        return cur in self._months

    @classmethod
    def load(cls, fx_cfg) -> "FxTable":
        rates: dict[tuple[str, int], float] = {}
        try:
            files = resolve_files(fx_cfg.files, required=False) if fx_cfg.files else []
        except Exception:
            files = []
        for f in files:
            try:
                df = read_many([f])
                cc, mc, rc = fx_cfg.currency_col, fx_cfg.month_col, fx_cfg.rate_col
                if not {cc, mc, rc} <= set(df.columns):
                    log.warning("fx file %s lacks columns %s", f.name, (cc, mc, rc))
                    continue
                for c, m, r in zip(df[cc], df[mc], df[rc]):
                    mo, rv = _month_ord(m), parse_number(r)
                    if not np.isnan(mo) and not np.isnan(rv) and rv > 0:
                        rates.setdefault((str(c).strip().upper(), int(mo)), rv)   # first file wins
            except Exception as e:
                log.warning("fx file %s unreadable: %s", f, e)
        log.info("fx table: %d rates, %d currencies", len(rates), len({c for c, _ in rates}))
        return cls(rates, fx_cfg.static_rates)


def convert(net: pd.Series, cur: pd.Series, line_rate: pd.Series, inv_month: pd.Series, post_month: pd.Series,
            reporting: str, table: FxTable, band: tuple[float, float] = (0.5, 2.0),
            convention: str = "units_per_reporting"):
    """Vectorised-by-unique-combination conversion. Returns (spend_rc, rate_used, method, flags_dict(name->mask))."""
    n = len(net)
    reporting = reporting.upper()
    cur = cur.fillna("").astype(str).str.strip().str.upper()
    lr = line_rate.to_numpy(dtype=float).copy()
    if convention == "reporting_per_unit":
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(lr > 0, 1.0 / lr, np.nan)     # normalise to units_per_reporting
    lr = np.where(lr > 0, lr, np.nan)
    im = inv_month.to_numpy(dtype=float)
    pm = post_month.to_numpy(dtype=float)
    rate = np.full(n, np.nan)
    method = np.array(["missing"] * n, dtype=object)
    implaus = np.zeros(n, bool)
    cur_blank = (cur == "").to_numpy()
    cur_a = np.where(cur_blank, reporting, cur.to_numpy())

    ident = cur_a == reporting
    rate[ident] = 1.0
    method[ident] = "identity"

    todo = np.flatnonzero(~ident)
    if len(todo):
        keys = pd.DataFrame({"c": cur_a[todo], "im": im[todo], "pm": pm[todo], "lr": lr[todo]})
        keys["lr"] = keys["lr"].round(8)
        grp = keys.fillna({"im": -1, "pm": -1, "lr": -1}).groupby(["c", "im", "pm", "lr"]).indices
        for (c, imv, pmv, lrv), idx in grp.items():
            imv = np.nan if imv == -1 else imv
            pmv = np.nan if pmv == -1 else pmv
            lrv = np.nan if lrv == -1 else lrv
            t_inv, t_post = table.get(c, imv), table.get(c, pmv)
            ref = t_inv if t_inv is not None else (t_post if t_post is not None else
                                                   table.nearest(c, imv if not np.isnan(imv) else pmv))
            r, m, bad = None, None, False
            if not np.isnan(lrv):
                if ref is None:
                    r, m = lrv, "line"
                elif band[0] <= lrv / ref <= band[1]:
                    r, m = lrv, "line"
                else:
                    bad = True
            if r is None:
                if t_inv is not None:
                    r, m = t_inv, "table_invoice_month"
                elif t_post is not None:
                    r, m = t_post, "table_posting_month"
                else:
                    nr = table.nearest(c, imv if not np.isnan(imv) else pmv)
                    if nr is not None:
                        r, m = nr, "table_nearest"
                    elif c in table.static:
                        r, m = table.static[c], "static"
            sel = todo[idx]
            if r is not None:
                rate[sel] = r
                method[sel] = m
            implaus[sel] = bad
    spend = np.where(np.isnan(rate) | (rate <= 0), 0.0, net.to_numpy(dtype=float) / np.where(rate > 0, rate, 1.0))
    spend = np.where(np.isnan(spend), 0.0, spend)
    return (pd.Series(spend, index=net.index), pd.Series(rate, index=net.index),
            pd.Series(method, index=net.index),
            {"FX_RATE_IMPLAUSIBLE": pd.Series(implaus, index=net.index),
             "FX_MISSING": pd.Series(method == "missing", index=net.index),
             "CURRENCY_BLANK": pd.Series(cur_blank, index=net.index)})
