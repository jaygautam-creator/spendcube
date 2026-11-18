"""Synthetic end-to-end check of normalize + exclusions on a tiny in-memory company (no kit data)."""
import pandas as pd
import pytest

from spendcube.config.schema import Config
from spendcube.context import RunContext
from spendcube.exclusions.learned import apply_learned, learn_rules
from spendcube.exclusions.status import classify_status
from spendcube.exclusions.stage import exclusions_stage
from spendcube.ingest.canonical import ingest_source
from spendcube.normalize.stage import normalize_stage
from spendcube.taxonomy.tree import Taxonomy


def make(tmp_path, rows, extra=None):
    p = tmp_path / "lines.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    t = tmp_path / "tax.csv"
    pd.DataFrame({"category_l1": ["Goods"], "category_l2": ["Stuff"]}).to_csv(t, index=False)
    cfg = {
        "company": "T", "reporting_currency": "USD",
        "taxonomy": {"file": str(t)},
        "sources": [{
            "name": "lines", "role": "primary", "file": str(p),
            "record_id": {"column": "id"},
            "columns": {"supplier_name": "sup", "vendor_id": "vid", "description": "txt", "amount": "amt",
                        "amount_basis": "gross", "tax_rate": "vat", "default_tax_rate": 0.15, "currency": "cur",
                        "status": "status", "gl_description": "gld", "invoice_date": {"column": "d"}},
            "status_map": {"INCLUDE": ["paid", "approved"], "CANCELLED": ["cancelled"],
                           "NOT_APPROVED": ["on hold", "rejected"], "default": "NOT_APPROVED"},
        }],
    }
    cfg.update(extra or {})
    cfg = Config.model_validate(cfg)
    ctx = RunContext(cfg=cfg, run_dir=tmp_path)
    ctx.taxonomy = Taxonomy.load(cfg.taxonomy)
    df = ingest_source(cfg, cfg.sources[0])
    return ctx, df


ROWS = [
    dict(id="1", sup="A", vid="V1", txt="widget", amt="1,150.00", vat="15", cur="USD", status="Paid", gld="AP", d="2024-03-01"),
    dict(id="2", sup="A", vid="V1", txt="widget", amt="1150", vat="", cur="USD", status="Cancelled", gld="AP", d="2024-03-01"),
    dict(id="3", sup="B", vid="V2", txt="gadget", amt="100", vat="0", cur="USD", status="Rejected", gld="AP", d="2024-03-02"),
    dict(id="4", sup="B", vid="V2", txt="Sales Tax -", amt="50", vat="0", cur="USD", status="Paid", gld="AP", d="2024-03-02"),
    dict(id="5", sup="C", vid="V3", txt="x", amt="10", vat="0", cur="USD", status="mystery", gld="IC05", d="2024-03-02"),
    dict(id="6", sup="C", vid="V3", txt="y", amt="-20", vat="0", cur="USD", status="Approved", gld="IC05", d="2024-03-02"),
    dict(id="7", sup="C", vid="V3", txt="z", amt="30", vat="0", cur="USD", status="Approved", gld="Other", d="2024-03-02"),
    dict(id="8", sup="C", vid="V3", txt="w", amt="40", vat="0", cur="USD", status="Approved", gld="IC05", d="2024-03-02"),
    dict(id="9", sup="D", vid="V4", txt="Acme Tax Consultant", amt="70", vat="0", cur="USD", status="Paid", gld="AP", d="2024-03-02"),
]
RULES = [
    {"code": "NON_PROCUREMENT", "field": "gl_desc", "regex": r"(?i)^IC\d+"},
    {"code": "NON_PROCUREMENT", "field": "desc_raw", "regex": r"(?i)^\s*sales\s+tax\b"},
]


def test_status_vat_rules_precedence_and_vendor_propagation(tmp_path):
    ctx, df = make(tmp_path, ROWS, {"exclusion_rules": RULES,
                                    "vendor_propagation": {"enabled": True, "min_share": 0.7, "min_rows": 3}})
    n = len(df)
    df = normalize_stage(ctx, df)
    df = exclusions_stage(ctx, df)
    assert len(df) == n                                            # never drops rows
    r = df.set_index("record_id")
    assert r.loc["1", "spend_rc"] == pytest.approx(1000.0) and r.loc["1", "excl_code"] == ""   # 1150/1.15
    assert r.loc["2", "excl_code"] == "CANCELLED"
    assert r.loc["2", "spend_rc"] == pytest.approx(1000.0)         # spend kept in frame, zeroed at output
    assert r.loc["3", "excl_code"] == "NOT_APPROVED" and r.loc["3", "spend_rc"] == 100.0   # VAT 0 = 0%
    assert r.loc["4", "excl_code"] == "NON_PROCUREMENT"            # tax-posting description
    assert r.loc["5", "status_class"] == "NOT_APPROVED"            # unknown status -> default
    assert r.loc["5", "excl_code"] == "NOT_APPROVED"               # beats NON_PROCUREMENT (precedence)
    assert r.loc["6", "excl_code"] == "NON_PROCUREMENT" and r.loc["6", "spend_rc"] == -20.0   # negatives kept
    assert r.loc["7", "excl_code"] == "NON_PROCUREMENT"            # vendor propagation (3 of 4 rows IC)
    assert r.loc["7", "excl_detail"] == "vendor_propagation"
    assert r.loc["9", "excl_code"] == ""                           # tax consultant stays included
    assert "mystery" in ctx.notes["exclusions"]["unknown_status_values"]["lines"]


def test_include_rule_protects_and_existing_duplicate_preserved(tmp_path):
    rules = [{"code": "NON_PROCUREMENT", "field": "desc_raw", "regex": "(?i)^widget$", "action": "include"}] + RULES
    ctx, df = make(tmp_path, ROWS, {"exclusion_rules": rules,
                                    "vendor_propagation": {"enabled": False}})
    df = normalize_stage(ctx, df)
    df["excl_code"] = ""
    df.loc[df.record_id == "9", "excl_code"] = "DUPLICATE"        # as a later reconcile stage would set it
    df.loc[df.record_id == "9", "excl_detail"] = "reexport"
    df = exclusions_stage(ctx, df)
    r = df.set_index("record_id")
    assert bool(r.loc["1", "rule_include"]) and r.loc["1", "excl_code"] == ""
    assert r.loc["9", "excl_code"] == "DUPLICATE"                  # not clobbered
    assert r.loc["2", "excl_code"] == "CANCELLED"                  # status still beats protection? yes: intrinsic


def test_blank_amount_and_flags(tmp_path):
    rows = [dict(ROWS[0], id="a", amt="", d="31/02/2099"), dict(ROWS[0], id="b", amt="1x2")]
    ctx, df = make(tmp_path, rows)
    df = normalize_stage(ctx, df)
    r = df.set_index("record_id")
    assert "AMOUNT_BLANK" in r.loc["a", "flags"] and r.loc["a", "spend_rc"] == 0.0
    assert "AMOUNT_UNPARSEABLE" in r.loc["b", "flags"]


def test_intra_source_dedup_marks_only_exact_later_copy(tmp_path):
    rows = [dict(ROWS[0], id="a", txt="exact", amt="10", d="2024-01-01"),
            dict(ROWS[0], id="b", txt="exact", amt="10", d="2024-01-01"),
            dict(ROWS[0], id="c", txt="exact", amt="10", d="2024-01-02")]
    p, t = tmp_path / "lines.csv", tmp_path / "tax.csv"
    pd.DataFrame([dict(r, inv="I-1") for r in rows]).to_csv(p, index=False)
    pd.DataFrame({"category_l1": ["Goods"], "category_l2": ["Stuff"]}).to_csv(t, index=False)
    cfg = Config.model_validate({"company": "T", "taxonomy": {"file": str(t)}, "sources": [{"name": "lines",
        "role": "primary", "file": str(p), "record_id": {"column": "id"}, "columns": {"supplier_name": "sup",
        "vendor_id": "vid", "description": "txt", "amount": "amt", "amount_basis": "gross", "currency": "cur",
        "status": "status", "invoice_number": "inv", "invoice_date": {"column": "d"}},
        "status_map": {"default": "INCLUDE"}, "intra_source_dedup": {"enabled": True}}]})
    ctx = RunContext(cfg=cfg, run_dir=tmp_path)
    ctx.taxonomy = Taxonomy.load(cfg.taxonomy)
    from spendcube.ingest.canonical import ingest_source
    df = ingest_source(cfg, cfg.sources[0])
    df = normalize_stage(ctx, df)
    df = exclusions_stage(ctx, df)
    assert list(df["excl_code"]) == ["", "DUPLICATE", ""]


def test_intra_source_dedup_vendor_id_fallback_when_supplier_unmapped(tmp_path):
    rows = [dict(ROWS[0], id="a", inv="I1", vid="V1", txt="same", amt="10", d="2024-01-01"),
            dict(ROWS[0], id="b", inv="I1", vid="V1", txt="same", amt="10", d="2024-01-01"),
            dict(ROWS[0], id="c", inv="I1", vid="V2", txt="same", amt="10", d="2024-01-01")]
    p, t = tmp_path / "lines.csv", tmp_path / "tax.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    pd.DataFrame({"category_l1": ["Goods"], "category_l2": ["Stuff"]}).to_csv(t, index=False)
    cfg = Config.model_validate({"company": "T", "taxonomy": {"file": str(t)}, "sources": [{"name": "lines",
        "role": "primary", "file": str(p), "record_id": {"column": "id"}, "columns": {"vendor_id": "vid",
        "description": "txt", "amount": "amt", "amount_basis": "gross", "currency": "cur", "status": "status",
        "invoice_number": "inv", "invoice_date": {"column": "d"}}, "status_map": {"default": "INCLUDE"},
        "intra_source_dedup": {"enabled": True}}]})
    ctx = RunContext(cfg=cfg, run_dir=tmp_path)
    ctx.taxonomy = Taxonomy.load(cfg.taxonomy)
    df = ingest_source(cfg, cfg.sources[0])
    df = exclusions_stage(ctx, normalize_stage(ctx, df))
    assert list(df["excl_code"]) == ["", "DUPLICATE", ""]


def test_classify_status_unknowns_counted():
    from spendcube.config.schema import StatusMapCfg
    sm = StatusMapCfg(INCLUDE=["paid"], CANCELLED=["void"], default="INCLUDE")
    cls, unk = classify_status(pd.Series(["PAID", " Void ", "weird", "weird", ""]), sm)
    assert list(cls) == ["INCLUDE", "CANCELLED", "INCLUDE", "INCLUDE", "INCLUDE"]
    assert unk == {"weird": 2, "": 1}


def test_learned_rules_thresholds():
    df = pd.DataFrame({"record_id": [str(i) for i in range(40)], "source_role": "primary",
                       "gl_desc": ["IC1"] * 10 + ["AP"] * 20 + ["MIX"] * 10,
                       "gl_account": "", "vendor_id": ""})
    lab = pd.DataFrame({"record_id": [str(i) for i in range(40)],
                        "include": ["0"] * 10 + ["1"] * 20 + ["0"] * 5 + ["1"] * 5,
                        "exclusion_reason": ["NON_PROCUREMENT"] * 10 + [""] * 20 + ["NON_PROCUREMENT"] * 5 + [""] * 5})
    rules = learn_rules(df, lab, ["gl_desc", "gl_account"], min_support=5, min_purity=0.95)
    assert [(r["value"], r["code"]) for r in rules] == [("IC1", "NON_PROCUREMENT")]   # AP all included, MIX impure
    code, detail, conf = apply_learned(df, rules)
    assert list(code[:10]) == ["NON_PROCUREMENT"] * 10 and code[10] == ""
