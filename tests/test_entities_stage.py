"""Entities stage: alias maps, record-level golden keys, inheritance along match_ref. Synthetic only."""
from types import SimpleNamespace

import pandas as pd

from spendcube.entities.legal_forms import supplier_core, supplier_norm
from spendcube.entities.stage import entities_stage
from spendcube.normalize.text import add_text_columns


def mk_ctx(labels=None, entities=None):
    cfg = SimpleNamespace(entities=entities or {}, reconciliation=None)
    ctx = SimpleNamespace(cfg=cfg, state={}, notes={}, warnings=[])
    ctx.note = lambda sec, k, v: ctx.notes.setdefault(sec, {}).__setitem__(k, v)
    ctx.warn = lambda m: ctx.warnings.append(m)
    ctx.state["labels"] = labels
    return ctx


def mk_df(rows):
    df = pd.DataFrame(rows, columns=["record_id", "source", "source_role", "supplier_raw", "vendor_id",
                                     "material_id", "desc_raw", "match_type", "match_ref"])
    df["gl_account"] = ""
    df["spend_rc"] = 1.0
    df["supplier_norm"] = df.supplier_raw.map(supplier_norm)
    df["supplier_core"] = df.supplier_raw.map(supplier_core)
    return add_text_columns(df, None)


ROWS = [
    ("E1", "erp", "primary", "Acme Ltd", "V1", "M1", "BALL VALVE 2IN", "none", ""),
    ("E2", "erp", "primary", "Acme Ltd", "V1", "M2", "GASKET SET", "none", ""),
    ("T1", "tool", "secondary", "ACME LIMITED", "", "", "ball valve 2in", "erp", "E1"),
    ("T2", "tool", "secondary", "acme", "", "", "BALL VALVE 2IN", "reexport", "T1"),
    ("T3", "tool", "secondary", "Other Co", "", "", "MISC SERVICE", "none", ""),
    ("T4", "tool", "secondary", "Hist Co", "", "", "SOMETHING", "history", "hist:E999"),
]


def test_never_drops_rows_and_sets_all_columns():
    out = entities_stage(mk_ctx(), mk_df(ROWS))
    assert len(out) == len(ROWS)
    for c in ("supplier_key", "supplier_conf", "item_key", "item_conf"):
        assert out[c].notna().all() and (out[c].astype(str) != "").all()


def test_inheritance_chain_tool_erp_reexport_canonical():
    out = entities_stage(mk_ctx(), mk_df(ROWS)).set_index("record_id")
    assert out.loc["T1", "supplier_key"] == out.loc["E1", "supplier_key"]
    assert out.loc["T1", "item_key"] == out.loc["E1", "item_key"] == "M:M1"
    assert out.loc["T2", "item_key"] == out.loc["E1", "item_key"]        # re-export -> canonical -> ERP
    assert out.loc["T3", "supplier_key"] != out.loc["E1", "supplier_key"]
    assert out.loc["T1", "item_conf"] > 0.9


def test_labels_record_level_and_history_golden_keys():
    labels = pd.DataFrame({"record_id": ["E1", "E2", "E999"], "supplier_key": ["GS1", "GS1", "GSH"],
                           "item_key": ["GI1", "", "GIH"]})
    out = entities_stage(mk_ctx(labels), mk_df(ROWS)).set_index("record_id")
    assert out.loc["E1", "supplier_key"] == "GS1" and out.loc["E1", "item_key"] == "GI1"
    assert out.loc["E2", "item_key"] == "M:M2"            # blank golden key does not override ours
    assert out.loc["T1", "supplier_key"] == "GS1" and out.loc["T1", "item_key"] == "GI1"
    assert out.loc["T2", "item_key"] == "GI1"
    # history match: keys come from the golden keys of the (absent) original ERP row
    assert out.loc["T4", "supplier_key"] == "GSH" and out.loc["T4", "item_key"] == "GIH"


def test_alias_maps_generalise_to_unlabelled_rows():
    labels = pd.DataFrame({"record_id": ["E1", "E2"], "supplier_key": ["GS1", "GS1"], "item_key": ["GI1", "GI2"]})
    rows = ROWS + [("N1", "erp2", "primary", "ACME", "", "M1", "BALL VALVE 2IN", "none", "")]   # new record, known name+material
    out = entities_stage(mk_ctx(labels), mk_df(rows)).set_index("record_id")
    assert out.loc["N1", "supplier_key"] == "GS1" and out.loc["N1", "item_key"] == "GI1"


def test_vendor_only_company_without_names():
    rows = [("B1", "b", "primary", "", "V10", "", "STEEL COIL", "none", ""),
            ("B2", "b", "primary", "", "V10", "", "", "none", ""),
            ("B3", "b", "primary", "", "V11", "MAT1", "BOLTS", "none", "")]
    out = entities_stage(mk_ctx(entities={"use_labels": False, "suppliers": {"mode": "vendor_only"}}), mk_df(rows))
    assert list(out.supplier_key) == ["V:V10", "V:V10", "V:V11"]
    assert out.item_key[1].startswith("U:") and out.item_key[2] == "M:MAT1"
