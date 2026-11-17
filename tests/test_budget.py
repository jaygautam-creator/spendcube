"""D-42: the per-job request budget is enforced in the router; the pipeline still ends with valid output."""
import pandas as pd

from spendcube.llm.router import Router
from spendcube.config.loader import load_config
from spendcube.pipeline import run_pipeline
from test_synthC_e2e import _draft


def _cfg(tmp_path, share=1.0, rpd=10, util=1.0, max_requests=None):
    b = {"enabled": True, "share": share, "util_cap": util, "pool": ["m"], "exempt_roles": ["compile", "arbiter"]}
    if max_requests is not None:
        b["max_requests"] = max_requests
    return {"enabled": True, "allow_training_use": True,
            "chain": [{"id": "m", "provider": "mock", "model": "mock", "rpd": rpd}],
            "roles": {r: ["m"] for r in ("compile", "top_spend", "bulk", "second_opinion", "arbiter")},
            "budget": b, "cache_file": str(tmp_path / "c.sqlite"), "ledger_file": str(tmp_path / "l.sqlite")}


def _call(r, role, i):
    return r.call(role, "sys", f"user {i}", {}, lambda p: (True, ""))


def test_budget_math_and_enforcement(tmp_path):
    r = Router(_cfg(tmp_path, share=0.5, rpd=10), job_budget=True)
    assert r.budget == 5
    outs = [_call(r, "bulk", i) for i in range(8)]
    assert sum(o is not None for o in outs) == 5            # stops after the budget, never raises
    assert r.budget_exhausted() and not r.has_llm("bulk")
    assert _call(r, "arbiter", 99) is not None              # exempt role still works
    assert r.report()["budget"]["spent"] == 5


def test_no_budget_by_default(tmp_path):
    r = Router(_cfg(tmp_path), job_budget=False)
    assert r.budget is None and all(_call(r, "bulk", i) is not None for i in range(8))


def test_pipeline_valid_when_budget_is_zero(tmp_path):
    f = _draft(tmp_path)
    cfg = load_config(f, overrides={"classification": {"llm": _cfg(tmp_path, share=0.0)}})
    path, errors = run_pipeline(cfg, out_dir=tmp_path / "out", no_llm=False)
    assert errors == []
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(out) == 3000 and (out.loc[out.include == "1", "l1"] != "").all()
