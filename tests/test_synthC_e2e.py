"""End-to-end on the SYNTHETIC company C: profiler -> draft config (+ minimal manual fixes) -> pipeline -> validator.

Generalisation drill: European decimals, ';' delimiter, dd.mm.yyyy, German statuses, gross + tax rate, EUR/CHF
without line rates (static fx), 4-level ragged path tree, supplier variants, intercompany/tax lines.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from spendcube.config.loader import load_config
from spendcube.output.validator import validate_file
from spendcube.pipeline import run_pipeline
from spendcube.profile.draft_config import draft_config
from spendcube.profile.profiler import profile_file
from spendcube.taxonomy.tree import Taxonomy

FIX = Path(__file__).parent / "fixtures" / "synthC"
STATIC = "{EUR: 0.92, CHF: 0.88}"        # the "manual fix": units per 1 USD (what a human writes at blind time)


def _draft(tmp_path, with_static=True, extra=None):
    p = profile_file(FIX / "lines_C.csv", FIX / "taxonomy_C.csv")
    y = draft_config(p, FIX / "lines_C.csv", FIX / "taxonomy_C.csv", llm_include=None, input_dir=str(FIX),
                     accept_suggestions=True)       # the human step: uncomment the suggested exclusion rules
    if with_static:
        assert "static_rates: {}" in y
        y = y.replace("static_rates: {}", f"static_rates: {STATIC}", 1)
    f = tmp_path / "C.yaml"
    f.write_text(y, encoding="utf-8")
    return f


@pytest.fixture(scope="module")
def truth():
    return pd.read_csv(FIX / "truth.csv", dtype=str, keep_default_na=False).set_index("line_id")


def _run(cfg_file, tmp_path, **kw):
    cfg = load_config(cfg_file, **kw.pop("cfg_kw", {}))
    path, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=True)
    return cfg, path, errors


def test_e2e_no_llm(tmp_path, truth):
    cfg, path, errors = _run(_draft(tmp_path), tmp_path)
    assert errors == []
    assert path.name == "jaygautam__C.csv"
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(out) == 3000 and list(out["record_id"]) == list(truth.index)        # exact count, input order
    assert validate_file(path, Taxonomy.load(cfg.taxonomy), list(truth.index)) == []

    # exclusions match ground truth exactly (precedence CANCELLED > NOT_APPROVED > NON_PROCUREMENT)
    got = out.set_index("record_id")["exclusion_reason"]
    exp = truth["true_exclusion"]
    assert (got == exp).mean() == 1.0, pd.crosstab(exp, got)
    assert set(exp) == {"", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT"}

    # European decimals + gross->net (VAT) + static FX recovered per line (to the cent) and in total (<0.1%)
    spend = pd.to_numeric(out.set_index("record_id")["spend"])
    tru = pd.to_numeric(truth["true_spend_usd"])
    assert abs(spend.sum() - tru.sum()) / tru.sum() < 0.001
    inc = out.set_index("record_id")["include"] == "1"
    assert (spend[inc] - tru[inc]).abs().max() < 0.02
    assert (spend[~inc] == 0).all()
    # categories are exact tree strings (placeholder classifier here); excluded rows blank
    assert (out.loc[out.include == "0", ["l1", "l2", "l3", "l4"]] == "").all().all()
    assert (out.loc[out.include == "1", "l1"] != "").all()


def test_static_fx_applied_and_missing_is_not_silently_one(tmp_path, truth):
    # without static rates EUR/CHF lines must NOT be converted at 1 (spend 0 + FX_MISSING); USD lines still count
    cfg, path, errors = _run(_draft(tmp_path, with_static=False), tmp_path)
    assert errors == []
    out = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("record_id")
    raw = pd.read_csv(FIX / "lines_C.csv", sep=";", dtype=str, keep_default_na=False).set_index("belegnr")
    inc = out["include"] == "1"
    usd = raw["waehrung"] == "USD"
    spend = pd.to_numeric(out["spend"])
    assert (spend[inc & ~usd] == 0).all()
    tru = pd.to_numeric(truth["true_spend_usd"])
    assert (spend[inc & usd] - tru[inc & usd]).abs().max() < 0.02
    # with the static table the same lines are converted
    cfg2, path2, _ = _run(_draft(tmp_path / "..", with_static=True) if False else _draft(tmp_path), tmp_path)
    out2 = pd.read_csv(path2, dtype=str, keep_default_na=False).set_index("record_id")
    assert (pd.to_numeric(out2["spend"])[inc & ~usd] > 0).any()


def test_supplier_and_status_sanity(tmp_path):
    cfg, path, _ = _run(_draft(tmp_path), tmp_path)
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = pd.read_csv(FIX / "lines_C.csv", sep=";", dtype=str, keep_default_na=False)
    # spacing/case variants of one supplier collapse to one key; legal suffix variants too
    ok = raw["lieferant"].str.upper().str.replace(r"\s+", "", regex=True).str.contains("MEDICOSUPPLY")
    assert out.loc[ok.values, "supplier_key"].nunique() == 1
    assert out["supplier_key"].nunique() < raw["lieferant"].nunique()


needs_router = False


@pytest.mark.skipif(needs_router, reason="router->pipeline wiring (classify.cascade) not merged yet")
def test_e2e_with_mock_llm(tmp_path, truth):
    f = _draft(tmp_path)
    mock_llm = {"enabled": True, "allow_training_use": True,
                "chain": [{"id": "m", "provider": "mock", "model": "mock"}],
                "roles": {r: ["m"] for r in ("compile", "top_spend", "bulk", "second_opinion", "arbiter")},
                "cache_file": str(tmp_path / "c.sqlite"), "ledger_file": str(tmp_path / "l.sqlite")}
    cfg = load_config(f, overrides={"classification": {"llm": mock_llm}})
    path, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=False)
    assert errors == []
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(out) == 3000
    assert (out.loc[out.include == "1", "l1"] != "").all()
