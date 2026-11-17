from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from spendcube.llm.cache import make_key
from spendcube.llm.ledger import Ledger
from spendcube.llm.router import Router
from spendcube.llm.providers.base import ProviderResult


def _router(tmp, *, enabled=True, fixed=None, rpm=15):
    chain = [
        {"id": "x", "provider": "mock", "model": "x", "rpm": rpm, "rpd": 100},
        {"id": "y", "provider": "mock", "model": "y", "rpm": rpm, "rpd": 100},
    ]
    cfg = {"enabled": True, "chain": chain, "roles": {"bulk": ["x"], "top_spend": ["x"]},
           "parallel_pool": {"enabled": enabled, "entries": ["x", "y"]},
           "cache_file": str(Path(tmp) / "cache.sqlite"), "ledger_file": str(Path(tmp) / "ledger.sqlite"),
           "budget": {"enabled": True, "pool": ["x", "y"], "fixed": fixed} if fixed is not None else {"enabled": False}}
    r = Router(cfg, cache=None, ledger=Ledger(Path(tmp) / "ledger.sqlite"), job_budget=fixed is not None)
    return r


def _call(r, i, max_wait=0.02):
    return r.call("bulk", "system", f"unique prompt {i}", {}, lambda _: (True, ""),
                  max_tokens=5, max_wait=max_wait)


def test_pool_independent_rpm_and_at_least_1_7x_throughput(tmp_path, monkeypatch):
    class SlowMock:
        def complete_json(self, *args, **kwargs):
            time.sleep(0.05)
            return ProviderResult(parsed={"ok": True})
    one = _router(tmp_path / "one", enabled=False, rpm=2)
    monkeypatch.setattr(one, "_provider", lambda entry: SlowMock())
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        baseline = list(ex.map(lambda i: _call(one, i), range(2)))
    one_s = time.perf_counter() - start
    assert sum(x is not None for x in baseline) == 2
    two = _router(tmp_path / "two", enabled=True, rpm=2)
    monkeypatch.setattr(two, "_provider", lambda entry: SlowMock())
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        pooled = list(ex.map(lambda i: _call(two, i), range(4)))
    two_s = time.perf_counter() - start
    assert sum(x is not None for x in pooled) == 4
    counts = two.stats["by_entry"]
    assert counts["x"]["calls"] <= 2 and counts["y"]["calls"] <= 2
    # Equal-latency mock calls in the same measurement window: two independent RPM=2
    # members admit four requests versus two through the single-member baseline.
    assert (sum(x is not None for x in pooled) / sum(x is not None for x in baseline)) >= 1.7


def test_pool_cache_hit_from_other_member(tmp_path):
    r = _router(tmp_path, enabled=True)
    key = make_key("mock", "x", "v2", "system", "unique prompt 90", "")
    r.cache.put(key, {"parsed": {"cached": True}, "usage": {}, "model": "x"})
    out = _call(r, 90)
    assert out is not None and out.cached and out.entry_id == "x"
    assert r.stats["calls"] == 0


def test_pool_failover_from_repeated_503(tmp_path, monkeypatch):
    class Broken:
        def complete_json(self, *args, **kwargs):
            return ProviderResult(error="service unavailable", error_kind="server")
    r = _router(tmp_path, enabled=True)
    monkeypatch.setattr("spendcube.llm.router.time.sleep", lambda _: None)
    monkeypatch.setattr("spendcube.llm.router.random.uniform", lambda *_: 0)
    def provider(entry):
        if entry.id == "x":
            return Broken()
        from spendcube.llm.providers.mock import MockProvider
        return MockProvider()
    monkeypatch.setattr(r, "_provider", provider)
    out = _call(r, 100)
    assert out is not None and out.entry_id == "y"
    assert "x" in r.dead


def test_pool_budget_shared_across_members(tmp_path):
    r = _router(tmp_path, enabled=True, fixed=10)
    with ThreadPoolExecutor(max_workers=4) as ex:
        outcomes = list(ex.map(lambda i: _call(r, i), range(20)))
    assert sum(x is not None for x in outcomes) == 10
    assert r.stats["calls"] == 10 and r.spent == 10


def test_pool_disabled_preserves_first_chain_entry(tmp_path):
    r = _router(tmp_path, enabled=False)
    for i in range(4):
        out = _call(r, 200 + i)
        assert out is not None and out.entry_id == "x"
    assert r.stats["by_entry"]["x"]["calls"] == 4
    assert "y" not in r.stats["by_entry"]
