"""D-19 core-route guards and D-22 partial-capacity matching (synthetic only)."""
from spendcube.eval.synth import make_synth_partial, evaluate
from spendcube.reconcile.stage import reconcile_core

from test_reconcile_match import erp, flags, frame, rc, tool

CORE = dict(core_route={"enabled": True, "when": "no_po", "max_days": 120, "min_abs": 0.01})


def test_core_route_matches_no_po_row_to_unclaimed_receipt():
    out, st = reconcile_core(frame([erp(1), tool(1, po="")]), rc(**CORE))
    assert flags(out)["TOOL:1"] == "DUPLICATE" and st["core_route_assigned"] == 1


def test_core_route_cannot_reuse_receipt_claimed_by_po_route():
    # TOOL:1 matches ERP:1 through the PO; TOOL:2 (no PO, identical amount) must not steal the same receipt
    out, _ = reconcile_core(frame([erp(1), tool(1), tool(2, po="", inv="INV/2023/222222")]), rc(**CORE))
    assert flags(out)["TOOL:1"] == "DUPLICATE" and flags(out)["TOOL:2"] == ""


def test_core_route_ignores_zero_and_penny_amounts():
    rows = [erp(1, amt="0.01", qty="1"), erp(2, amt="0.00", qty="1", po="4510000002"),
            tool(1, po="", gross="0.01", qty="1"), tool(2, po="", gross="0.00", qty="1", inv="INV/9")]
    out, _ = reconcile_core(frame(rows), rc(**CORE))
    assert flags(out)["TOOL:1"] == "" and flags(out)["TOOL:2"] == ""


def test_core_route_date_window():
    far = frame([erp(1, **{}), tool(1, po="", invoice_date_raw="2024-12-01")])
    far.at[0, "posting_date_raw"] = "2022-01-10"
    near = frame([erp(1), tool(1, po="", invoice_date_raw="2022-02-01")])
    near.at[0, "posting_date_raw"] = "2022-01-10"
    assert flags(reconcile_core(far, rc(**CORE))[0])["TOOL:1"] == ""            # ~3 years apart
    assert flags(reconcile_core(near, rc(**CORE))[0])["TOOL:1"] == "DUPLICATE"  # 22 days
    missing = frame([erp(1), tool(1, po="")])                                   # no dates: allowed
    assert flags(reconcile_core(missing, rc(**CORE))[0])["TOOL:1"] == "DUPLICATE"


def test_core_route_off_by_default():
    out, _ = reconcile_core(frame([erp(1), tool(1, po="")]), rc())
    assert flags(out)["TOOL:1"] == ""


PARTIAL = dict(partial_capacity={"enabled": True, "price_tol": 0.005})


def test_partial_capacity_two_partials_fit_receipt():
    # receipt 10 units @ 100; invoices of 4 and 6 units (same PO, same unit price)
    rows = [erp(1, amt="1000.00", qty="10"),
            tool(1, gross="460.00", qty="4", inv="INV/1"), tool(2, gross="690.00", qty="6", inv="INV/2")]
    off, _ = reconcile_core(frame(rows), rc())
    on, st = reconcile_core(frame(rows), rc(**PARTIAL))
    assert flags(off)["TOOL:1"] == "" and flags(off)["TOOL:2"] == ""
    assert flags(on)["TOOL:1"] == "DUPLICATE" and flags(on)["TOOL:2"] == "DUPLICATE"
    assert st["partial_assigned"] == 2
    assert on.set_index("record_id").loc["TOOL:1", "excl_detail"].startswith("partial_")


def test_partial_capacity_respects_capacity_and_price():
    rows = [erp(1, amt="1000.00", qty="10"),
            tool(1, gross="690.00", qty="6", inv="INV/1"),
            tool(2, gross="690.00", qty="6", inv="INV/2"),          # 6+6 > 10: second does not fit
            tool(3, gross="575.00", qty="2", inv="INV/3")]          # unit price 250/unit != 100: no
    on, _ = reconcile_core(frame(rows), rc(**PARTIAL))
    f = flags(on)
    assert f["TOOL:1"] == "DUPLICATE" and f["TOOL:2"] == "" and f["TOOL:3"] == ""


def test_partial_does_not_touch_exact_matches_or_other_pos():
    rows = [erp(1), tool(1),                                         # exact match, receipt consumed
            tool(2, gross="575.00", qty="1", inv="INV/2"),           # half qty, same unit price, receipt already used
            erp(3, po="4510000009", amt="1000.00", qty="2"),
            tool(3, po="PO 4510000042", gross="575.00", qty="1", inv="INV/3")]  # other PO
    on, _ = reconcile_core(frame(rows), rc(**PARTIAL))
    f = flags(on)
    assert f["TOOL:1"] == "DUPLICATE" and f["TOOL:2"] == "" and f["TOOL:3"] == ""


def test_partial_other_supplier_not_matched():
    rows = [erp(1, amt="1000.00", qty="10"), tool(1, gross="460.00", qty="4", sup="OTHERCO (PTY) LTD")]
    on, _ = reconcile_core(frame(rows), rc(**PARTIAL))
    assert flags(on)["TOOL:1"] == ""


def test_partial_synthetic_injection_thresholds():
    df, truth = make_synth_partial(n_erp=2000, seed=3)
    out = {}
    for label, over in (("off", {}), ("on", PARTIAL)):
        o, st = reconcile_core(df, rc(**over))
        out[label] = (o, st)
    off, on = out["off"][0], out["on"][0]
    d0, d1 = off["excl_code"].eq("DUPLICATE"), on["excl_code"].eq("DUPLICATE")
    assert not (d0 & ~d1).any()                                      # never removes an existing flag
    parts = evaluate(on, truth, truth["partial"])
    assert parts["recall"] >= 0.85 and parts["precision"] >= 0.99
    new = d1 & ~d0
    tool_new = new & (on["source"] == "tool")
    true_dup = truth["truth_type"].isin(["erp", "reexport"])
    assert (true_dup & tool_new).sum() / max(1, tool_new.sum()) >= 0.95   # incremental precision
