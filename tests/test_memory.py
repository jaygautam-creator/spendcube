"""D-45: unit memory saves LLM requests on recurring units and never crosses taxonomies."""
import numpy as np
import pandas as pd

from spendcube.classify.memory import UnitMemory
from spendcube.config.loader import load_config
from spendcube.pipeline import run_pipeline
from test_synthC_e2e import _draft


def _units():
    return pd.DataFrame({"text": ["hex bolt", "pump seal", "audit fee"], "supplier": ["acme", "acme", "kpx"],
                         "gl": ["g1", "g1", "g2"], "ctx": ["", "", ""]})


def test_store_lookup_and_taxonomy_isolation(tmp_path):
    u = _units()
    m = UnitMemory(tmp_path / "m.sqlite", "taxA")
    assert m.store(u, np.array([True, False, True]), np.array(["T001", "", "T003"], dtype=object), np.array([.9, 0, .8])) == 2
    leaf, conf = m.lookup(u, np.ones(3, bool))
    assert list(leaf) == ["T001", "", "T003"] and conf[0] == 0.9
    # same text+supplier, different GL -> loose key still hits; strict mode does not
    u2 = u.assign(gl=["gX", "gX", "gX"])
    assert list(m.lookup(u2, np.ones(3, bool))[0]) == ["T001", "", "T003"]
    assert list(m.lookup(u2, np.ones(3, bool), loose=False)[0]) == ["", "", ""]
    # another taxonomy never sees them
    assert list(UnitMemory(tmp_path / "m.sqlite", "taxB").lookup(u, np.ones(3, bool))[0]) == ["", "", ""]


def test_second_run_needs_fewer_requests(tmp_path):
    f = _draft(tmp_path)

    def run(tag):
        llm = {"enabled": True, "allow_training_use": True,
               "chain": [{"id": "m", "provider": "mock", "model": "mock"}],
               "roles": {r: ["m"] for r in ("compile", "top_spend", "bulk", "second_opinion", "arbiter")},
               "cache_file": str(tmp_path / f"c_{tag}.sqlite"), "ledger_file": str(tmp_path / "l.sqlite")}
        cfg = load_config(f, overrides={"classification": {"llm": llm, "memory": {"file": str(tmp_path / "mem.sqlite"), "min_conf": 0.0}}})
        path, errors = run_pipeline(cfg, out_dir=tmp_path / f"out_{tag}", no_llm=False)
        assert errors == []
        import spendcube.pipeline as pl
        notes = (pl.LAST_RUN_DIR / "report.md").read_text()
        return pd.read_csv(path, dtype=str, keep_default_na=False), notes
    a, _ = run("1")
    b, notes = run("2")                     # fresh LLM cache, shared memory
    assert len(a) == len(b) == 3000 and list(a.record_id) == list(b.record_id)
    meth = pd.read_parquet(__import__("spendcube.pipeline", fromlist=["x"]).LAST_RUN_DIR / "final_frame.parquet")["cat_method"]
    assert (meth == "memory").sum() > 0


def test_training_units_roundtrip(tmp_path):
    u = _units().assign(desc_norm=["hex bolt", "pump seal", "audit fee"], desc_compact=["hexbolt", "pumpseal", "auditfee"],
                        supplier_core=["acme", "acme", "kpx"], vendor_id=["", "", ""], gl_account=["g1", "g1", "g2"],
                        desc_blank=[False, False, False])
    m = UnitMemory(tmp_path / "m.sqlite", "taxA")
    m.store(u, np.array([True, True, False]), np.array(["T001", "T002", ""], dtype=object), np.array([.9, .6, 0]))
    t = m.training_units(min_conf=0.7, weight=0.5)
    assert len(t) == 1 and t["leaf"].iat[0] == "T001" and abs(t["w"].iat[0] - 0.45) < 1e-9
    assert t["desc_blank"].iat[0] is np.False_ or t["desc_blank"].iat[0] == False   # noqa: E712
    assert {"desc_norm", "supplier_core", "text"} <= set(t.columns)
