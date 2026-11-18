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


def test_blind_mode_fails_over_to_dev_gemini_key_on_quota_or_503(monkeypatch):
    from spendcube.llm.providers.base import ProviderResult
    monkeypatch.setenv("GEMINI_API_KEY", "fake_blind_key")
    monkeypatch.setenv("GEMINI_API_KEY_DEV", "fake_dev_key")
    with tempfile.TemporaryDirectory() as tmp:
        chain = [{"id": "g1", "provider": "gemini", "model": "gemini-3.5-flash-lite", "key_env": "GEMINI_API_KEY", "rpd": 10}]
        r = Router({"enabled": True, "chain": chain, "roles": {"bulk": ["g1"]},
                    "cache_file": str(Path(tmp) / "c.sqlite"), "ledger_file": str(Path(tmp) / "l.sqlite")}, mode="blind")
        assert r.entries["g1"].profiles == ["GEMINI_API_KEY", "GEMINI_API_KEY_DEV"]
        assert r.entries["g1"].profile == "GEMINI_API_KEY"

        class StubProv:
            def __init__(self, prof):
                self.prof = prof

            def complete_json(self, *args, **kwargs):
                if self.prof == "GEMINI_API_KEY":
                    return ProviderResult(error="daily quota exhausted", error_kind="quota")
                return ProviderResult(parsed={"r": [[1, 1, 90, 0]]}, usage={"in": 10, "out": 5})

        monkeypatch.setattr(r, "_provider", lambda e: StubProv(e.profile))
        out = r.call("bulk", "sys", "usr", {}, lambda p: (True, ""))
        assert out is not None and out.parsed == {"r": [[1, 1, 90, 0]]}
        assert r.entries["g1"].profile == "GEMINI_API_KEY_DEV"
        assert "g1" not in r.dead
        assert r.stats["key_failovers"] == [{"entry": "g1", "from": "GEMINI_API_KEY", "to": "GEMINI_API_KEY_DEV", "why": "quota exhausted"}]


def test_dev_mode_never_touches_blind_gemini_key(monkeypatch):
    from spendcube.llm.providers.base import ProviderResult
    monkeypatch.setenv("GEMINI_API_KEY", "fake_blind_key")
    monkeypatch.setenv("GEMINI_API_KEY_DEV", "fake_dev_key")
    with tempfile.TemporaryDirectory() as tmp:
        chain = [{"id": "g1", "provider": "gemini", "model": "gemini-3.5-flash-lite", "key_env": "GEMINI_API_KEY", "rpd": 10}]
        r = Router({"enabled": True, "chain": chain, "roles": {"bulk": ["g1"]},
                    "cache_file": str(Path(tmp) / "c.sqlite"), "ledger_file": str(Path(tmp) / "l.sqlite")}, mode="dev")
        assert r.entries["g1"].profiles == ["GEMINI_API_KEY_DEV"]
        called_profiles = []

        class StubProv:
            def __init__(self, prof):
                self.prof = prof

            def complete_json(self, *args, **kwargs):
                called_profiles.append(self.prof)
                return ProviderResult(error="daily quota exhausted", error_kind="quota")

        monkeypatch.setattr(r, "_provider", lambda e: StubProv(e.profile))
        out = r.call("bulk", "sys", "usr", {}, lambda p: (True, ""))
        assert out is None
        assert "g1" in r.dead
        assert called_profiles == ["GEMINI_API_KEY_DEV"]
        assert "GEMINI_API_KEY" not in called_profiles


def test_outage_chain_falls_back_to_groq_when_gemini_and_ollama_down(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake_groq_key")
    with tempfile.TemporaryDirectory() as tmp:
        chain = [
            {"id": "g1", "provider": "gemini", "model": "gemini-3.5-flash-lite", "key_env": "GEMINI_API_KEY"},
            {"id": "ollama_local", "provider": "ollama", "model": "qwen3:4b", "optional": True},
            {"id": "groq_gptoss120b", "provider": "groq", "model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY", "optional": True},
        ]
        r = Router({"enabled": True, "chain": chain, "roles": {"bulk": ["g1"]},
                    "outage": {"enabled": True, "chain": ["ollama_local", "groq_gptoss120b"]},
                    "cache_file": str(Path(tmp) / "c.sqlite"), "ledger_file": str(Path(tmp) / "l.sqlite")}, mode="blind")
        monkeypatch.setattr(r, "_ollama_up", lambda e: False)
        r.dead["g1"] = "all gemini keys exhausted"
        assert not r.has_llm("bulk")
        assert r.try_outage() is True
        assert [e.id for e in r.chain("bulk")] == ["groq_gptoss120b"]
