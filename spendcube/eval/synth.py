"""Synthetic matching test (`matching-synth`): invented ERP-shaped receipts + tool-like copies.

Nothing here is derived from kit rows. Generators mimic the *mechanisms* in DATA_FINDINGS section 4:
VAT encodings (15 / 0.15 / 15% / blank / 0), PO formats (plain / 'PO ' / 'PO#' / '/10.0' / dropped),
OCR noise, spacing/case/legal-suffix variation, re-exports with jittered dates, bulk repeated-amount
receipts, fabricated tool-only lines and coincidence negatives (same supplier+price on another PO).
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..schema_cols import MAPPED_FIELDS
from ..reconcile.features import get_cfg
from ..reconcile.stage import reconcile_core

_SYL = ["kel", "mor", "ta", "vex", "lu", "dan", "ri", "po", "ska", "bren", "tho", "nal", "quo", "zim", "fer",
        "han", "wil", "ost", "pra", "dun"]
_NOUN = ["VALVE", "BEARING", "PUMP", "HOSE", "FILTER", "BOLT", "CABLE", "GEAR", "SEAL", "ROD", "PANEL", "BRACKET",
         "MOTOR", "LINER", "SENSOR", "FAN", "COUPLING", "CHAIN", "BELT", "GASKET"]
_MOD = ["HYD", "ELEC", "STD", "HD", "SS", "ASSY", "KIT", "SPARE", "REPL", "MNT"]
_LEGAL = ["", " (PTY) LTD", " PTY LTD", " LTD", " CC", " (Pty) Ltd", " Pty Ltd"]
_OCR = {"O": "0", "I": "1", "S": "5", "B": "8", "L": "1", "0": "O", "1": "I", "5": "S"}


def _names(rng: np.random.Generator, n: int) -> list[str]:
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(_SYL, size=rng.integers(2, 4))).upper())
    return sorted(out)


def _noise(text: str, rng: np.random.Generator, p: float = 0.04) -> str:
    chars = []
    for ch in text:
        r = rng.random()
        if r < p and ch in _OCR:
            chars.append(_OCR[ch])
        elif r < p * 1.5 and ch == " ":
            continue                                   # dropped space
        elif r < p * 1.8 and ch != " ":
            chars.append(ch + " ")                     # inserted space
        else:
            chars.append(ch)
    s = "".join(chars)
    if rng.random() < 0.10 and len(s) > 12:
        s = s[: -int(rng.integers(1, 3))]              # truncation
    if rng.random() < 0.15:
        s = s.lower() if rng.random() < 0.5 else s.title()
    return s


def _fmt_po(po: str, item: int, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.35:
        return po
    if r < 0.55:
        return f"PO {po}"
    if r < 0.75:
        return f"PO#{po}"
    return f"{po}/{item}.0"


def _fmt_vat(vat: float, rng: np.random.Generator) -> str:
    if vat == 0.0:
        return "0"
    r = rng.random()
    return "15" if r < 0.3 else "0.15" if r < 0.6 else "15%" if r < 0.9 else ""


def _fmt_invoice(rng: np.random.Generator) -> str:
    n = int(rng.integers(100000, 999999))
    k = int(rng.integers(0, 5))
    return [f"INV/2023/{n}", f"IN0{n}", f"{n}", f"TAX INV {n}", f"SI-2024-{n}"][k]


def _fmt_date(d: pd.Timestamp, rng: np.random.Generator) -> str:
    k = int(rng.integers(0, 3))
    return [d.strftime("%Y-%m-%d"), d.strftime("%d/%m/%Y"), d.strftime("%d-%b-%y")][k]


_COLS = list(MAPPED_FIELDS.values()) + ["context", "llm_context"]


def _blank_frame(n: int) -> pd.DataFrame:
    """Canonical-shaped frame of blanks (+ the identity columns the stage needs)."""
    d = {c: [""] * n for c in _COLS}
    d.update(record_id=[""] * n, source=[""] * n, source_role=[""] * n, row_order=list(range(n)))
    return pd.DataFrame(d)


def make_synth(n_erp: int = 3000, seed: int = 0, drop_po: float = 0.18, reexport_rate: float = 0.08,
               tool_coverage: float = 0.75, n_tool_only: int = 500, n_coincidence: int = 150,
               history_split: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df, truth). truth is indexed like df for the tool rows (NaN for ERP rows):
    truth_type in {erp, reexport, only}; truth_erp = id of the true ERP receipt for erp/reexport-of-erp;
    dropped_po flags copies whose PO was removed.

    history_split>0: that share of ERP receipts + their tool copies go to *reference* sources (A2 simulation)."""
    rng = np.random.default_rng(seed)
    sup_names = _names(rng, 40)
    # ---- ERP side: POs with 1-6 lines; bulk POs with many identical receipts -------------
    rows = []
    po_no = 4510000000
    while len(rows) < n_erp:
        po_no += int(rng.integers(1, 6))
        sup = sup_names[int(rng.integers(0, len(sup_names)))]
        bulk = rng.random() < 0.08
        for li in range(int(rng.integers(1, 7))):
            item = 10 * (li + 1)
            desc = f"{rng.choice(_NOUN)},{rng.choice(_MOD)};{rng.choice(_NOUN)} {int(rng.integers(10, 999))}MM,{rng.choice(_MOD)}"
            unit = round(float(rng.lognormal(7.5, 1.0)), 2)
            if bulk:
                nrec, qty0 = int(rng.integers(5, 25)), float(rng.integers(20, 40))
                recs = [(qty0 + float(rng.integers(0, 2)) * 0.1, ) for _ in range(nrec)]
            else:
                nrec = int(rng.integers(1, 3))
                recs = [(float(rng.integers(1, 30)),) for _ in range(nrec)]
            for (q,) in recs:
                rows.append((po_no, item, sup, desc, q, round(unit * q, 2)))
    rows = rows[:n_erp]
    erp = _blank_frame(len(rows))
    erp["record_id"] = [f"ERP:S{i:06d}" for i in range(len(rows))]
    erp["source"], erp["source_role"] = "erp", "primary"
    erp["row_order"] = range(len(rows))
    erp["po_raw"] = [str(r[0]) for r in rows]
    erp["po_item_raw"] = [f"{r[1]}.0" for r in rows]
    erp["supplier_raw"] = [r[2] for r in rows]
    erp["desc_raw"] = [r[3] for r in rows]
    erp["quantity_raw"] = [str(r[4]) for r in rows]
    erp["amount_raw"] = [str(r[5]) for r in rows]
    erp["status_raw"] = ""

    # ---- tool side ------------------------------------------------------------------------
    t_rows: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    sel = np.flatnonzero(rng.random(len(erp)) < tool_coverage)
    date0 = pd.Timestamp("2022-01-03")

    def add_invoice(base: dict, n_copies: int, ttype: str, terp: Optional[str], dropped: bool):
        inv = _fmt_invoice(rng)
        d0 = date0 + pd.Timedelta(days=int(rng.integers(0, 900)))
        has_date = rng.random() < 0.35
        for c in range(n_copies):
            r = dict(base)
            r["invoice_no_raw"] = inv
            r["supplier_raw"] = _noise(base["_sup"], rng, 0.0) + _LEGAL[int(rng.integers(0, len(_LEGAL)))]
            if rng.random() < 0.25:
                r["supplier_raw"] = r["supplier_raw"].replace(" ", "")
            r["supplier_raw"] = r["supplier_raw"].upper() if rng.random() < 0.6 else r["supplier_raw"].title()
            r["desc_raw"] = _noise(base["_desc"], rng) if c or rng.random() < 0.5 else base["_desc"]
            po = "" if dropped else _fmt_po(base["_po"], base["_item"], rng)
            r["po_raw"] = po
            r["invoice_date_raw"] = _fmt_date(d0 + pd.Timedelta(days=int(rng.integers(-20, 20)) if c else 0), rng) if has_date else ""
            r["status_raw"] = "Paid"
            t_rows.append(r)
            truth.append({"truth_type": ttype if c == 0 else "reexport", "truth_erp": terp or "",
                          "dropped_po": dropped, "canonical": c == 0})

    for i in sel:
        r = erp.iloc[i]
        net = float(r["amount_raw"])
        vat = 0.0 if rng.random() < 0.05 else 0.15
        gross = round(net * (1 + vat), 2)
        base = {c: "" for c in _COLS}
        base.update(amount_raw=f"{gross:,.2f}", tax_rate_raw=_fmt_vat(vat, rng), quantity_raw=r["quantity_raw"],
                    _sup=r["supplier_raw"], _desc=r["desc_raw"], _po=r["po_raw"], _item=int(float(r["po_item_raw"])))
        copies = 1 + (int(rng.integers(1, 3)) if rng.random() < reexport_rate else 0)
        add_invoice(base, copies, "erp", r["record_id"], bool(rng.random() < drop_po))

    # fabricated tool-only invoices (PO not in ERP, or no PO)
    for k in range(n_tool_only):
        sup = sup_names[int(rng.integers(0, len(sup_names)))]
        net = round(float(rng.lognormal(7.5, 1.0)) * float(rng.integers(1, 30)), 2)
        po = str(po_no + 1000 + k * 3) if rng.random() < 0.6 else ""
        base = {c: "" for c in _COLS}
        base.update(amount_raw=f"{round(net * 1.15, 2):,.2f}", tax_rate_raw=_fmt_vat(0.15, rng),
                    quantity_raw=str(float(rng.integers(1, 30))), _sup=sup,
                    _desc=f"{rng.choice(_NOUN)},{rng.choice(_MOD)};{rng.choice(_NOUN)} {int(rng.integers(10, 999))}MM",
                    _po=po or "4510000000", _item=10)
        copies = 1 + (int(rng.integers(1, 3)) if rng.random() < reexport_rate else 0)
        add_invoice(base, copies, "only", None, dropped=(po == ""))

    # coincidence negatives: same supplier + price + qty + text as an ERP row, different (unknown) PO / no PO
    for k in range(n_coincidence):
        r = erp.iloc[int(rng.integers(0, len(erp)))]
        net = float(r["amount_raw"])
        base = {c: "" for c in _COLS}
        base.update(amount_raw=f"{round(net * 1.15, 2):,.2f}", tax_rate_raw="15", quantity_raw=r["quantity_raw"],
                    _sup=r["supplier_raw"], _desc=r["desc_raw"], _po=str(po_no + 5000 + k * 7), _item=10)
        add_invoice(base, 1, "only", None, dropped=bool(rng.random() < 0.5))

    tool = _blank_frame(len(t_rows))
    tdf = pd.DataFrame(t_rows)
    for c in _COLS:
        if c in tdf.columns:
            tool[c] = tdf[c].fillna("").astype(str).to_numpy()
    # shuffle tool rows so ids are not ordered by ERP position; keep re-export copies adjacent (as observed)
    order = np.arange(len(tool))
    blocks = np.cumsum([1 if tr["canonical"] else 0 for tr in truth]) - 1
    perm = rng.permutation(blocks.max() + 1)
    rank = perm[blocks]
    order = np.lexsort((np.arange(len(tool)), rank))
    tool = tool.iloc[order].reset_index(drop=True)
    tr = pd.DataFrame(truth).iloc[order].reset_index(drop=True)
    tool["record_id"] = [f"TOOL:T{i:06d}" for i in range(len(tool))]
    tool["source"], tool["source_role"] = "tool", "secondary"
    tool["row_order"] = range(len(erp), len(erp) + len(tool))

    df = pd.concat([erp, tool], ignore_index=True)
    truth_df = pd.concat([pd.DataFrame({"truth_type": [""] * len(erp), "truth_erp": "", "dropped_po": False,
                                         "canonical": False}), tr], ignore_index=True)
    return df, truth_df


def make_synth_history(n_erp: int = 1200, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A2 simulation: half of the receipts (and half of their invoices) are *reference* rows (history).

    Emitted tool rows then include invoices whose receipt lives only in history (-> 'history' matches) and
    cross-split re-exports of history tool rows. Reference rows are never emitted and never flagged."""
    df, truth = make_synth(n_erp=n_erp, seed=seed, drop_po=0.0)
    rng = np.random.default_rng(seed + 100)
    is_erp = (df["source"] == "erp").to_numpy()
    erp_ids = df.loc[is_erp, "record_id"].to_numpy()
    hist_erp = set(erp_ids[::2])
    df = df.copy()
    truth = truth.copy()
    # invoice groups (canonical row starts a group)
    canon = truth["canonical"].to_numpy()
    group = np.cumsum(canon) - 1
    group = np.where(is_erp, -1, group)
    tool_idx = np.flatnonzero(~is_erp)
    gids = sorted(set(group[tool_idx]))
    to_ref = {g for g in gids if (g % 2 == 0) and truth.loc[np.flatnonzero(group == g)[0], "truth_erp"] in hist_erp}
    for i in np.flatnonzero(is_erp):
        if df.at[i, "record_id"] in hist_erp:
            df.at[i, "source"], df.at[i, "source_role"] = "erp_hist", "reference"
    for i in tool_idx:
        if group[i] in to_ref:
            df.at[i, "source"], df.at[i, "source_role"] = "tool_hist", "reference"
    # cross-split re-exports: emitted copies of some reference invoices
    extra_rows, extra_truth = [], []
    ref_canon = [i for i in tool_idx if group[i] in to_ref and canon[i] and truth.at[i, "truth_erp"]]
    for i in rng.choice(ref_canon, size=min(40, len(ref_canon)), replace=False):
        r = df.loc[i].copy()
        r["record_id"] = f"TOOL:X{len(extra_rows):05d}"
        r["source"], r["source_role"] = "tool", "secondary"
        r["supplier_raw"] = r["supplier_raw"].replace(" ", "")
        extra_rows.append(r)
        extra_truth.append({"truth_type": "reexport", "truth_erp": truth.at[i, "truth_erp"], "dropped_po": False,
                            "canonical": False})
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        truth = pd.concat([truth, pd.DataFrame(extra_truth)], ignore_index=True)
    return df, truth


def evaluate(df_out: pd.DataFrame, truth: pd.DataFrame, subset: Optional[pd.Series] = None) -> dict[str, Any]:
    tool = (df_out["source"] == "tool").to_numpy().copy()
    if subset is not None:
        tool = tool & subset.to_numpy()
    pred_dup = df_out["excl_code"].eq("DUPLICATE").to_numpy()[tool]
    true_dup = truth["truth_type"].isin(["erp", "reexport"]).to_numpy()[tool]
    tp = int((pred_dup & true_dup).sum()); fp = int((pred_dup & ~true_dup).sum()); fn = int((~pred_dup & true_dup).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    # partner accuracy for rows typed 'erp': true ERP receipt, or an equivalent receipt (same PO + net + qty)
    mt = df_out["match_type"].to_numpy()[tool]
    ref = df_out["match_ref"].to_numpy()[tool]
    terp = truth["truth_erp"].to_numpy()[tool]
    erp = df_out[df_out["source"].astype(str).str.startswith("erp")].set_index("record_id")
    key = (erp["po_raw"] + "|" + erp["amount_raw"] + "|" + erp["quantity_raw"]).to_dict()
    pa_n = pa_ok = 0
    for m, r, t in zip(mt, ref, terp):
        if m == "erp" and t:
            pa_n += 1
            pa_ok += int(r == t or (r in key and t in key and key[r] == key[t]))
    return {"n_tool": int(tool.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "partner_accuracy": round(pa_ok / max(1, pa_n), 4), "partner_n": pa_n}


def matching_synth(n_erp: int = 3000, seed: int = 0, rc_overrides: Optional[dict] = None, verbose: bool = True) -> dict:
    """Entry for `spendcube eval matching-synth`. Returns metrics for the default config and a core-route variant."""
    class _Cfg:                      # minimal stand-in for Config (only .reconciliation is read)
        reconciliation = {"primary_sources": ["erp"], "secondary_sources": ["tool"], **(rc_overrides or {})}
    results: dict[str, Any] = {}
    df, truth = make_synth(n_erp=n_erp, seed=seed)
    for label, over in (("default", {}), ("core_route_no_po", {"core_route": {"enabled": True, "when": "no_po"}})):
        rc = get_cfg(_Cfg)
        for k, v in over.items():
            rc[k] = v
        out, st = reconcile_core(df, rc)
        dropped = truth["dropped_po"].astype(bool)
        res = {
            "all_tool_rows": evaluate(out, truth),
            "rows_with_po": evaluate(out, truth, ~dropped),
            "rows_without_po": evaluate(out, truth, dropped),
            "match_types": st.get("match_type_counts"),
        }
        results[label] = res
        if verbose:
            print(label, json.dumps(res, default=str))
    # A2 simulation with history reference rows (D-02)
    hdf, htruth = make_synth_history(n_erp=max(600, n_erp // 2), seed=seed + 1)
    rc = get_cfg(_Cfg)
    rc["history"]["enabled"] = True
    hout, hst = reconcile_core(hdf, rc)
    ref_flagged = int(hout.loc[hout["source_role"] == "reference", "excl_code"].fillna("").ne("").sum())
    results["history"] = {"emitted_tool_rows": evaluate(hout, htruth), "match_types": hst.get("match_type_counts"),
                          "reference_rows_flagged": ref_flagged}
    if verbose:
        print("history", json.dumps(results["history"], default=str))
    return results


def make_synth_partial(n_erp: int = 3000, seed: int = 0, split_share: float = 0.30, hide_share: float = 0.12
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-22 injection test. From a synthetic base: (a) `split_share` of single-invoice receipts become 2-3
    partial invoices (same PO, same unit price, quantities summing to the receipt, or short of it: the missing
    part was never invoiced); (b) `hide_share` of receipts are deleted so their invoices become true tool-only
    rows (a partial-capacity matcher must NOT attach them to other receipts of the same PO/price).
    truth adds: partial (row is one of the split parts), hidden (partner receipt deleted)."""
    df, truth = make_synth(n_erp=n_erp, seed=seed, drop_po=0.0, reexport_rate=0.0, n_coincidence=0)
    rng = np.random.default_rng(seed + 7)
    is_tool = (df["source"] == "tool").to_numpy()
    cand = np.flatnonzero(is_tool & (truth["truth_type"] == "erp").to_numpy() & truth["canonical"].to_numpy())
    rng.shuffle(cand)
    n_hide = int(hide_share * len(cand)); n_split = int(split_share * len(cand))
    hide_rows, split_rows = cand[:n_hide], cand[n_hide:n_hide + n_split]
    erp_pos = {rid: i for i, rid in enumerate(df["record_id"]) if str(rid).startswith("ERP:")}
    drop_idx = set(); new_rows = []; new_truth = []
    hidden_rows = set(int(i) for i in hide_rows)
    for i in hide_rows:
        drop_idx.add(erp_pos[truth.at[i, "truth_erp"]])
    n_new = 0
    for i in split_rows:
        e = df.loc[erp_pos[truth.at[i, "truth_erp"]]]
        q_tot = float(e["quantity_raw"]); net = float(e["amount_raw"])
        if q_tot < 4 or net <= 0:
            continue
        unit = net / q_tot
        k = int(rng.integers(2, 4))
        cuts = np.sort(rng.choice(np.arange(1, int(q_tot)), size=k - 1, replace=False)) if int(q_tot) > k else None
        if cuts is None:
            continue
        parts = np.diff(np.concatenate([[0], cuts, [q_tot]])).tolist()
        if rng.random() < 0.4:
            parts = parts[:-1]                       # incomplete: the rest was never invoiced
        base = df.loc[i]
        for q in parts:
            r = base.copy()
            pn = round(unit * q, 2)
            r["quantity_raw"] = str(round(q, 3))
            r["amount_raw"] = f"{round(pn * 1.15, 2):,.2f}"
            r["tax_rate_raw"] = "15"
            r["invoice_no_raw"] = _fmt_invoice(rng)
            r["record_id"] = f"TOOL:P{n_new:06d}"
            n_new += 1
            new_rows.append(r)
            new_truth.append({"truth_type": "erp", "truth_erp": truth.at[i, "truth_erp"], "dropped_po": False,
                              "canonical": True, "partial": True, "hidden": False})
        drop_idx.add(i)
    keep = [j for j in range(len(df)) if j not in drop_idx]
    truth = truth.copy()
    truth["partial"] = False
    truth["hidden"] = [j in hidden_rows for j in range(len(df))]
    truth.loc[list(hidden_rows), "truth_type"] = "only"
    df2 = pd.concat([df.iloc[keep], pd.DataFrame(new_rows)], ignore_index=True)
    tr2 = pd.concat([truth.iloc[keep], pd.DataFrame(new_truth)], ignore_index=True)
    df2["row_order"] = range(len(df2))
    return df2, tr2


def partial_synth(n_erp: int = 3000, seed: int = 0, verbose: bool = True) -> dict:
    """Partial-capacity ON vs OFF on the injection test (overall, split parts, hidden-receipt negatives)."""
    class _Cfg:
        reconciliation = {"primary_sources": ["erp"], "secondary_sources": ["tool"]}
    df, truth = make_synth_partial(n_erp=n_erp, seed=seed)
    res: dict[str, Any] = {}
    for label, en in (("off", False), ("on", True)):
        rc = get_cfg(_Cfg)
        rc["partial_capacity"] = {**rc["partial_capacity"], "enabled": en}
        out, st = reconcile_core(df, rc)
        res[label] = {"all": evaluate(out, truth), "split_parts": evaluate(out, truth, truth["partial"]),
                      "hidden_receipt_rows": {**evaluate(out, truth, truth["hidden"]),
                                              "flagged_dup": int(out.loc[truth["hidden"].to_numpy(), "excl_code"].eq("DUPLICATE").sum())},
                      "partial_assigned": st.get("partial_assigned", 0)}
        if verbose:
            print("partial", label, json.dumps(res[label], default=str))
    return res


if __name__ == "__main__":
    matching_synth()
