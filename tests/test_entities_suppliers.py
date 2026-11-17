"""Supplier resolution: synthetic names only."""
import pandas as pd

from spendcube.entities.legal_forms import supplier_core, supplier_norm
from spendcube.entities.suppliers import build_supplier_alias, cluster_cores, resolve_suppliers


def frame(names, vendors=None, source="s", spend=None):
    n = len(names)
    return pd.DataFrame({
        "source": [source] * n,
        "supplier_raw": names,
        "supplier_norm": [supplier_norm(x) for x in names],
        "supplier_core": [supplier_core(x) for x in names],
        "vendor_id": vendors or [""] * n,
        "spend_rc": spend or [1.0] * n,
    })


def keys(df, **kw):
    k, c, st = resolve_suppliers(df, kw.pop("cfg", None), **kw)
    return list(k), list(c), st


def test_spacing_case_and_legal_suffix_collapse():
    k, _, _ = keys(frame(["Alpha Mining (Pty) Ltd", "ALPHA MINING", "Alpha  Min ing Pty Ltd", "Betaco Supplies CC"]))
    assert k[0] == k[1] == k[2] and k[3] != k[0]


def test_ocr_digit_letter_variants_merge():
    k, c, st = keys(frame(["ZORBIX TRADING", "ZORB1X TRADING", "ZORBIX TRADING"]))
    assert len(set(k)) == 1 and st["fuzzy_merges"] >= 1
    assert min(c) < 0.9            # fuzzy merge has lower confidence than an exact one


def test_similar_but_distinct_names_not_merged():
    k, _, _ = keys(frame(["DELTAM", "DELTAMWELDING", "DELTAM", "DEPOT 12 SUPPLIES", "DEPOT 21 SUPPLIES"]))
    assert k[0] != k[1]
    assert k[3] != k[4]            # digits inside names are identity, not typos


def test_short_names_never_fuzzy_merged():
    k, _, _ = keys(frame(["ABCD", "ABCE"]))
    assert k[0] != k[1]


def test_word_order_variants_merge():
    k, _, _ = keys(frame(["Smith Brothers Hardware", "Hardware Smith Brothers", "Other Corp Ltd"]))
    assert k[0] == k[1] != k[2]


def test_vendor_id_union_and_dissimilarity_guard():
    df = frame(["ACME FOUNDRY", "ACME FOUNDRIES GROUP", "TOTALLY DIFFERENT NAME"],
               vendors=["V1", "V1", "V1"])
    k, _, st = keys(df)
    assert k[0] == k[1]                       # same vendor id, similar names -> one supplier
    assert k[2] != k[0] and st["vendor_conflicts"] == 1   # dissimilar: flagged, not merged


def test_vendor_only_rows_inherit_name_cluster_else_v_key():
    df = frame(["ACME FOUNDRY", "", "", ""], vendors=["V1", "V1", "V9", ""])
    k, c, _ = keys(df)
    assert k[1] == k[0]                       # vendor known from a named row
    assert k[2] == "V:V9"
    assert k[3] == "S:UNKNOWN" and c[3] < 0.2


def test_vendor_only_mode_ignores_names():
    df = frame(["Name A", "Name B"], vendors=["V7", "V8"])
    k, _, _ = keys(df, cfg={"mode": "vendor_only"})
    assert k == ["V:V7", "V:V8"]


def test_alias_maps_apply_golden_keys_and_catch_all():
    train = frame(["AAA TRADING", "BBB TRADING", "CCC TRADING", "AAA TRADING"], vendors=["V1", "V2", "V3", "V1"])
    train["gold_supplier"] = ["G1", "CATCH", "CATCH", "G1"]
    a_core, a_vend = build_supplier_alias(train, 0.9)
    assert a_core["AAATRADING"] == "G1" and a_core["BBBTRADING"] == "CATCH" and a_vend["V3"] == "CATCH"
    test = frame(["aaa trading ltd", "BBB Trading", "ZZZ NEW CO", "CCC TRADING"], vendors=["", "", "", ""])
    k, c, _ = keys(test, alias_core=a_core, alias_vendor=a_vend)
    assert k[0] == "G1" and k[1] == "CATCH" and k[3] == "CATCH"
    assert k[2].startswith("S:")              # unseen name keeps its own key: never guess the catch-all (D-01)


def test_alias_purity_filter():
    train = frame(["AAA TRADING"] * 4)
    train["gold_supplier"] = ["G1", "G1", "G2", "G2"]
    a_core, _ = build_supplier_alias(train, 0.9)
    assert "AAATRADING" not in a_core


def test_fuzzy_member_inherits_cluster_alias():
    train = frame(["ZORBIX TRADING"])
    train["gold_supplier"] = ["GQ"]
    a_core, a_vend = build_supplier_alias(train, 0.9)
    test = frame(["ZORBIX TRADING", "ZORB1X TRADING"])
    k, _, _ = keys(test, alias_core=a_core, alias_vendor=a_vend)
    assert k == ["GQ", "GQ"]


def test_cluster_cores_block_cap_logged_not_crashing():
    cores = {f"NAME{i:04d}ZZ": 1.0 for i in range(50)}
    out, conf, st = cluster_cores(cores, {k: k for k in cores}, [], {"block_cap": 10})
    assert len(out) == 50 and st["block_overflow"] >= 1
