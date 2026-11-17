"""Failure injection (plan P6): every case must still yield a valid output file."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from test_synthC_e2e import FIX, _draft  # noqa: E402

from spendcube.config.loader import load_config
from spendcube.llm.providers.base import ProviderResult
from spendcube.llm.router import Router
from spendcube.llm.prompts import RESPONSE_SCHEMA
from spendcube.output.validator import validate_file
from spendcube.pipeline import run_pipeline
from spendcube.taxonomy.tree import Taxonomy


class Scripted:
    """Provider returning scripted results."""
    def __init__(self, results):
        self.results, self.calls = list(results), 0

    def complete_json(self, system, user, schema, model, max_tokens=4096, thinking=None, timeout=120.0):
        self.calls += 1
        r = self.results[min(self.calls - 1, len(self.results) - 1)]
        return r() if callable(r) else r


def ok(rows=None):
    return ProviderResult(parsed={"r": rows or [[1, 1, 90, 0]]}, raw_text="x", usage={"in": 10, "out": 5, "thinking": 0})


def err(kind, retry=None):
    return ProviderResult(error=f"{kind} error", error_kind=kind, retry_after=retry)


def make_router(tmp, entries):
    cfg = {"enabled": True, "chain": [{"id": e, "provider": "mock", "model": e} for e in entries],
           "roles": {"bulk": list(entries)}, "cache_file": str(Path(tmp) / "c.sqlite"),
           "ledger_file": str(Path(tmp) / "l.sqlite")}
    return Router(cfg, "dev")


def inject(r, mapping):
    r._provider = lambda e: mapping[e.id]


def validate_one(p):
    return (True, "") if isinstance(p, dict) and p.get("r") else (False, "bad")


def call(r):
    return r.call("bulk", "sys", "user", RESPONSE_SCHEMA, validate_one)


def test_429_storm_fails_over_without_raising(tmp_path):
    r = make_router(tmp_path, ["a", "b"])
    inject(r, {"a": Scripted([err("rate_limit", 300)]), "b": Scripted([ok()])})
    out = call(r)
    assert out is not None and out.entry_id == "b" and "a" in r.dead


def test_daily_quota_kind_disables_entry(tmp_path):
    r = make_router(tmp_path, ["a", "b"])
    inject(r, dict(zip("ab", (Scripted([err("quota")]), Scripted([ok()])))))
    assert call(r).entry_id == "b" and r.dead["a"] == "quota exhausted"


def test_outage_all_entries_returns_none(tmp_path):
    r = make_router(tmp_path, ["a", "b"])
    import spendcube.llm.router as rt
    rt.time.sleep = lambda s: None       # no real backoff in the test
    inject(r, {"a": Scripted([err("server")]), "b": Scripted([err("timeout")])})
    assert call(r) is None and r.stats["failures"] == 1


def test_bad_json_gets_one_repair_then_succeeds(tmp_path):
    r = make_router(tmp_path, ["a"])
    bad = ProviderResult(parsed={"r": []}, raw_text="x", usage={"in": 1, "out": 1, "thinking": 0})
    inject(r, {"a": Scripted([bad, ok()])})
    out = call(r)
    assert out is not None and out.repaired and r.stats["repairs"] == 1


def test_unparseable_then_valid(tmp_path):
    r = make_router(tmp_path, ["a"])
    inject(r, {"a": Scripted([err("parse"), ok()])})
    assert call(r) is not None


def test_missing_keys_means_no_llm_chain(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY_DEV", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = {"enabled": True, "chain": [{"id": "g", "provider": "gemini", "model": "m", "key_env": "GEMINI_API_KEY"}],
           "roles": {"bulk": ["g"]}, "cache_file": str(tmp_path / "c"), "ledger_file": str(tmp_path / "l")}
    r = Router(cfg, "dev")
    assert not r.has_llm("bulk") and call(r) is None


def _run_with_llm(tmp_path, llm_cfg):
    f = _draft(tmp_path)
    cfg = load_config(f, overrides={"classification": {"llm": llm_cfg}})
    path, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=False)
    tax = Taxonomy.load(cfg.taxonomy)
    ids = pd.read_csv(FIX / "truth.csv", dtype=str, keep_default_na=False)["line_id"].tolist()
    return path, errors, validate_file(path, tax, ids)


def test_pipeline_valid_when_every_llm_call_fails(tmp_path):
    llm = {"enabled": True, "allow_training_use": True, "chain": [{"id": "m", "provider": "mock", "model": "mock"}],
           "roles": {r: ["m"] for r in ("compile", "top_spend", "bulk", "second_opinion", "arbiter")},
           "cache_file": str(tmp_path / "c.sqlite"), "ledger_file": str(tmp_path / "l.sqlite")}
    import spendcube.llm.providers.mock as mk
    orig = mk.MockProvider.complete_json
    mk.MockProvider.complete_json = lambda self, *a, **k: ProviderResult(error="boom", error_kind="server")
    import spendcube.llm.router as rt
    old = rt.time.sleep
    rt.time.sleep = lambda s: None
    try:
        path, errors, verrs = _run_with_llm(tmp_path, llm)
    finally:
        mk.MockProvider.complete_json = orig
        rt.time.sleep = old
    assert errors == [] and verrs == []


def test_pipeline_valid_when_deadline_already_passed(tmp_path):
    llm = {"enabled": True, "allow_training_use": True, "deadline_minutes": -1,
           "chain": [{"id": "m", "provider": "mock", "model": "mock"}],
           "roles": {r: ["m"] for r in ("compile", "top_spend", "bulk", "second_opinion", "arbiter")},
           "cache_file": str(tmp_path / "c.sqlite"), "ledger_file": str(tmp_path / "l.sqlite")}
    path, errors, verrs = _run_with_llm(tmp_path, llm)
    assert errors == [] and verrs == []
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(out) == 3000


def test_pipeline_survives_stage_exception(tmp_path, monkeypatch):
    import spendcube.pipeline as pl
    from spendcube.classify import stage as cs

    def boom(ctx, df):
        raise RuntimeError("injected")
    monkeypatch.setattr("spendcube.classify.stage.classify_stage", boom)
    f = _draft(tmp_path)
    cfg = load_config(f)
    path, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=True)
    assert errors == [] and len(pd.read_csv(path)) == 3000
