from spendcube.llm.router import Router


def _cfg(tmp_path):
    return {"enabled": True, "cache_file": str(tmp_path / "c.sqlite"), "ledger_file": str(tmp_path / "l.sqlite"),
            "roles": {"bulk": ["primary"]}, "outage": {"enabled": True, "chain": ["local"]},
            "chain": [{"id": "primary", "provider": "mock", "model": "m"}, {"id": "local", "provider": "mock", "model": "m"}]}


def test_outage_switch_when_primary_dead(tmp_path):
    r = Router(_cfg(tmp_path), mode="dev")
    assert [e.id for e in r.chain("bulk")] == ["primary"]
    r.dead["primary"] = "5xx"
    assert not r.has_llm("bulk")
    assert r.try_outage() and r.outage
    assert [e.id for e in r.chain("bulk")] == ["local"]
    assert r.report()["outage_switch"] is True


def test_no_outage_when_budget_spent(tmp_path):
    r = Router(_cfg(tmp_path), mode="dev")
    r.budget, r.spent = 5, 5
    assert not r.try_outage()
