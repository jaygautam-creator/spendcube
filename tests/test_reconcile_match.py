"""R2-R8 behaviours on tiny hand-made frames (synthetic only)."""
import pandas as pd

from spendcube.eval.synth import _blank_frame
from spendcube.reconcile.features import get_cfg
from spendcube.reconcile.stage import reconcile_core


def rc(**over):
    cfg = get_cfg(type("C", (), {"reconciliation": {"primary_sources": ["erp"], "secondary_sources": ["tool"]}}))
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    return cfg


def frame(rows):
    df = _blank_frame(len(rows))
    for i, r in enumerate(rows):
        for k, v in r.items():
            df.at[i, k] = v
        df.at[i, "row_order"] = i
        if not df.at[i, "source_role"]:
            df.at[i, "source_role"] = {"erp": "primary", "tool": "secondary"}.get(r["source"], "reference")
    return df


def erp(i, po="4510000001", amt="1000.00", qty="2", sup="Acme", desc="VALVE BALL 2IN", item="10.0", src="erp"):
    return dict(record_id=f"ERP:{i}", source=src, supplier_raw=sup, po_raw=po, po_item_raw=item,
                amount_raw=amt, quantity_raw=qty, desc_raw=desc)


def tool(i, po="PO 4510000001", gross="1,150.00", vat="15", qty="2", sup="ACME (PTY) LTD", desc="VALVE BALL 2IN",
         inv="INV/2023/111111", status="Paid", src="tool", **kw):
    return dict(record_id=f"TOOL:{i}", source=src, supplier_raw=sup, po_raw=po, amount_raw=gross, tax_rate_raw=vat,
                quantity_raw=qty, desc_raw=desc, invoice_no_raw=inv, status_raw=status, **kw)


def flags(out):
    return dict(zip(out.record_id, out.excl_code))


def test_po_exact_amount_is_duplicate_and_erp_counts():
    out, st = reconcile_core(frame([erp(1), tool(1)]), rc())
    assert flags(out) == {"ERP:1": "", "TOOL:1": "DUPLICATE"}
    r = out.set_index("record_id").loc["TOOL:1"]
    assert r.match_type == "erp" and r.match_ref == "ERP:1" and r.dup_group == "G:ERP:1"


def test_vat_zero_and_blank():
    out, _ = reconcile_core(frame([erp(1), erp(2, po="4510000002"),
                                   tool(1, gross="1,000.00", vat="0"),                       # zero-rated: net == gross
                                   tool(2, po="4510000002", vat="")]), rc())               # blank -> 15%
    assert flags(out)["TOOL:1"] == "DUPLICATE" and flags(out)["TOOL:2"] == "DUPLICATE"


def test_zero_vat_not_treated_as_blank():
    out, _ = reconcile_core(frame([erp(1), tool(1, gross="1,000.00", vat="")]), rc())      # blank -> 15% => 869.57
    assert flags(out)["TOOL:1"] == ""                                                       # not a match (and not near)


def test_one_to_one_assignment():
    out, _ = reconcile_core(frame([erp(1), tool(1, inv="A1"), tool(2, inv="B2")]), rc())
    f = flags(out)
    assert sorted([f["TOOL:1"], f["TOOL:2"]]) == ["", "DUPLICATE"]     # two different invoices, one receipt


def test_reexport_cluster_canonical_is_best_status_then_lowest_id():
    rows = [erp(1), tool(1, status="Approved"), tool(2, status="Paid", po="", sup="ACME(PTY)LTD"),
            tool(3, status="Paid", desc="VALVE BALL 2IN ")]
    out, st = reconcile_core(frame(rows), rc())
    o = out.set_index("record_id")
    assert all(o.loc[f"TOOL:{i}", "excl_code"] == "DUPLICATE" for i in (1, 2, 3))   # one copy matched ERP -> all dup
    assert o.loc["TOOL:2", "match_type"] == "erp"                                  # canonical = Paid, lowest id
    assert o.loc["TOOL:1", "match_type"] == "reexport" and o.loc["TOOL:1", "match_ref"] == "TOOL:2"


def test_reexport_of_tool_only_invoice_counts_once():
    rows = [tool(1, po="PO 4510009999", inv="X9"), tool(2, po="", inv="X9", sup="ACME PTY LTD"),
            tool(3, po="4510009999/10.0", inv="X9")]
    out, _ = reconcile_core(frame(rows), rc())
    f = flags(out)
    assert sorted(f.values()) == ["", "DUPLICATE", "DUPLICATE"]                       # exactly one copy counts
    assert f["TOOL:1"] == ""


def test_bulk_lines_with_different_invoices_are_not_reexports():
    rows = [tool(i, po="", inv=f"INV{i}00{i}", sup="STONE CO") for i in (1, 2, 3)]
    out, _ = reconcile_core(frame(rows), rc())
    assert set(flags(out).values()) == {""}


def test_precedence_existing_exclusion_is_kept():
    df = frame([erp(1), tool(1)])
    df["excl_code"] = ["", "NON_PROCUREMENT"]
    out, _ = reconcile_core(df, rc())
    assert flags(out)["TOOL:1"] == "NON_PROCUREMENT"                   # D-03: DUPLICATE ranks lowest


def test_cancelled_invoice_does_not_consume_receipt():
    rows = [erp(1), tool(1, status="Cancelled", inv="C1"), tool(2, status="Paid", inv="P2")]
    out, _ = reconcile_core(frame(rows), rc())
    f = flags(out)
    assert f["TOOL:2"] == "DUPLICATE"                                  # the paid invoice still finds its receipt
    assert f["TOOL:1"] == ""                                           # status exclusion belongs to the exclusions stage


def test_core_route_off_by_default_and_when_enabled():
    rows = [erp(1, po="4510000001"), tool(1, po="")]
    out, _ = reconcile_core(frame(rows), rc())
    assert flags(out)["TOOL:1"] == ""
    out, _ = reconcile_core(frame(rows), rc(core_route={"enabled": True, "when": "no_po"}))
    assert flags(out)["TOOL:1"] == "DUPLICATE"


def test_po_not_in_pool_is_tool_only():
    out, _ = reconcile_core(frame([erp(1), tool(1, po="PO 4510088888")]), rc())
    assert flags(out)["TOOL:1"] == ""


def test_orphan_rescue_needs_same_invoice_number():
    # different invoice numbers = distinct invoices (one receipt in the pool, the other receipt elsewhere)
    out, st = reconcile_core(frame([erp(1), tool(1, inv="AAA111"), tool(2, inv="BBB222")]), rc())
    assert st["rescued"] == 0
    # same invoice number, but R2 refuses to cluster (dates 150 days apart): rescue links the twin
    rows = [erp(1), tool(1, inv="AAA111", invoice_date_raw="2023-01-01"),
            tool(2, inv="AAA111", invoice_date_raw="2023-06-01")]
    out, st = reconcile_core(frame(rows), rc())
    o = out.set_index("record_id")
    assert st["rescued"] == 1
    assert {o.loc["TOOL:1", "match_type"], o.loc["TOOL:2", "match_type"]} == {"erp", "reexport"}
    # the architecture's looser rule stays available behind a flag
    _, st = reconcile_core(frame([erp(1), tool(1, inv="AAA111"), tool(2, inv="BBB222")]),
                           rc(rescue={"require_same_invoice": False}))
    assert st["rescued"] == 1


def test_history_reference_rows_match_and_are_never_flagged():
    rows = [erp(1, src="erp_hist"), erp(2, po="4510000002"), tool(1, src="tool_hist", inv="H1"),
            tool(2, inv="H1", po=""),                                              # re-export of the history invoice
            tool(3, po="PO 4510000002", inv="N3", gross="1,150.00"),               # matches a current receipt
            tool(4, po="PO 4510000001", inv="N4")]                                 # receipt in history is taken by H1
    cfg = rc(history={"enabled": True})
    out, st = reconcile_core(frame(rows), cfg)
    o = out.set_index("record_id")
    assert o.loc["TOOL:2", "match_type"] == "history" and o.loc["TOOL:2", "excl_code"] == "DUPLICATE"
    assert o.loc["TOOL:3", "match_type"] == "erp"
    assert o.loc["ERP:1", "excl_code"] == "" and (out.loc[out.source_role == "reference", "excl_code"] == "").all()


def test_history_receipt_for_new_invoice():
    rows = [erp(1, src="erp_hist"), tool(1)]
    out, _ = reconcile_core(frame(rows), rc(history={"enabled": True}))
    o = out.set_index("record_id")
    assert o.loc["TOOL:1", "match_type"] == "history" and o.loc["TOOL:1", "excl_code"] == "DUPLICATE"
    out, _ = reconcile_core(frame(rows), rc(history={"enabled": False}))
    assert out.set_index("record_id").loc["TOOL:1", "excl_code"] == ""   # without history the reference row is ignored


def test_aggregate_combined_and_split():
    agg = {"aggregate": {"enabled": True}}
    comb = [erp(1, amt="400.00", qty="1"), erp(2, amt="600.00", qty="1"), tool(1, gross="1,150.00", qty="2")]
    out, _ = reconcile_core(frame(comb), rc(**agg))
    assert flags(out)["TOOL:1"] == "DUPLICATE"
    split = [erp(1, amt="1000.00", qty="4"), tool(1, gross="575.00", qty="2", inv="S1"),
             tool(2, gross="575.00", qty="2", inv="S2", desc="VALVE BALL 2IN X")]
    out, _ = reconcile_core(frame(split), rc(**agg))
    assert flags(out)["TOOL:1"] == "DUPLICATE" and flags(out)["TOOL:2"] == "DUPLICATE"
    out, _ = reconcile_core(frame(comb), rc())                          # flag off by default
    assert flags(out)["TOOL:1"] == ""


def test_row_count_and_columns_preserved():
    df = frame([erp(1), tool(1), tool(2, inv="Q")])
    out, _ = reconcile_core(df, rc())
    assert len(out) == len(df) and {"dup_group", "match_ref", "match_type", "match_score"} <= set(out.columns)
