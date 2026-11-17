import tempfile
from pathlib import Path

from spendcube.classify.llm import LLMClassifier
from spendcube.llm.cache import Cache
from spendcube.llm.ledger import Ledger
from spendcube.llm.limiter import Limiter
from spendcube.llm.router import Router
from spendcube.taxonomy.compile import compile_taxonomy
from spendcube.taxonomy.tree import Taxonomy

TAX = Taxonomy([("Food", "Dairy", "Milk"), ("Food", "Meat", "Beef"), ("Tools", "Hand", "Hammer")])


def mkrouter(tmp, chain=None, **kw):
    cfg = {"enabled": True, "chain": chain or [{"id": "m", "provider": "mock", "model": "mock"}],
           "roles": {"bulk": ["m"], "compile": ["m"]}, "cache_file": str(Path(tmp) / "c.sqlite"),
           "ledger_file": str(Path(tmp) / "l.sqlite"), **kw}
    return Router(cfg, mode="dev")


def test_router_classifies_and_caches():
    with tempfile.TemporaryDirectory() as tmp:
        r = mkrouter(tmp)
        clf = LLMClassifier(r, TAX, "test", None, {})
        units = [{"uid": 10, "supplier": "A", "gl": "", "ctx": "", "text": "BEEF STEAK"},
                 {"uid": 11, "supplier": "A", "gl": "", "ctx": "", "text": "HAMMER CLAW"}]
        a = clf.classify(units, "bulk")
        assert len(a) == 2
        assert r.stats["call_metrics"]
        assert r.stats["call_metrics"][0]["n_units"] == 2
        assert r.stats["call_metrics"][0]["model"] == "mock"
        assert r.stats["call_metrics"][0]["wall_ms"] >= 0
        assert "tokens_in" in r.stats["call_metrics"][0] and "tokens_out" in r.stats["call_metrics"][0]
        calls = r.stats["calls"]
        b = clf.classify(units, "bulk")
        assert r.stats["calls"] == calls and r.stats["cache_hits"] >= 1   # served from cache
        assert {k: v.leaf_id for k, v in a.items()} == {k: v.leaf_id for k, v in b.items()}


def test_bisect_on_failures():
    with tempfile.TemporaryDirectory() as tmp:
        r = mkrouter(tmp)
        clf = LLMClassifier(r, TAX, "", None, {})
        orig = clf._validator
        # reject any batch bigger than 2 units -> must bisect and still classify all
        def strict(uids):
            ok = orig(uids)
            return lambda p: (False, "too big") if len(uids) > 2 else ok(p)
        clf._validator = strict
        units = [{"uid": i, "supplier": "", "gl": "", "ctx": "", "text": f"MILK {i}"} for i in range(1, 8)]
        assert len(clf.classify(units, "bulk")) == 7


def test_no_llm_returns_empty_and_compile_noop():
    with tempfile.TemporaryDirectory() as tmp:
        r = Router({"enabled": True, "chain": [], "cache_file": str(Path(tmp) / "c"), "ledger_file": str(Path(tmp) / "l")}, "dev")
        assert compile_taxonomy(r, TAX, "", []) == {}
        assert LLMClassifier(r, TAX).classify([{"uid": 1, "text": "x"}], "bulk") == {}


def test_rpd_cap_marks_entry_dead_and_falls_through():
    with tempfile.TemporaryDirectory() as tmp:
        chain = [{"id": "a", "provider": "mock", "model": "m", "rpd": 1}, {"id": "b", "provider": "mock", "model": "m"}]
        r = mkrouter(tmp, chain, roles={"bulk": ["a", "b"]})
        clf = LLMClassifier(r, TAX, "", None, {})
        clf.classify([{"uid": 1, "text": "BEEF"}], "bulk")
        out = clf.classify([{"uid": 2, "text": "HAMMER"}], "bulk")
        assert out[2].entry == "b" and "a" in r.dead


def test_ledger_and_limiter():
    with tempfile.TemporaryDirectory() as tmp:
        L = Ledger(Path(tmp) / "l.sqlite")
        L.record("p", "e", 10, 5)
        assert L.requests_today("p", "e") == 1 and L.profile_requests("other") == 0
        lim = Limiter(rpm=2)
        assert lim.acquire(1, 1) and lim.acquire(1, 1) and not lim.acquire(1, 0.5)


def test_compile_with_mock():
    with tempfile.TemporaryDirectory() as tmp:
        r = mkrouter(tmp)
        d = compile_taxonomy(r, TAX, "x", ["milk"])
        assert len(d) == 3
