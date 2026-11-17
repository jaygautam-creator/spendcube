"""Synthetic-injection precision/recall (exit criteria: recall >= 0.97, precision >= 0.99 where a PO exists)."""
from spendcube.eval.synth import matching_synth


def test_matching_synth_exit_criteria():
    r = matching_synth(n_erp=1500, seed=3, verbose=False)
    po = r["default"]["rows_with_po"]
    assert po["recall"] >= 0.97 and po["precision"] >= 0.99, po
    assert po["partner_accuracy"] >= 0.99
    assert r["default"]["all_tool_rows"]["precision"] >= 0.99          # never flags tool-only/coincidence rows
    # the no-PO supplier+amount route recovers dropped-PO copies but costs precision (why it is off by default)
    cr = r["core_route_no_po"]["all_tool_rows"]
    assert cr["recall"] >= r["default"]["all_tool_rows"]["recall"]
    h = r["history"]
    assert h["emitted_tool_rows"]["precision"] >= 0.99 and h["emitted_tool_rows"]["recall"] >= 0.97
    assert h["reference_rows_flagged"] == 0
