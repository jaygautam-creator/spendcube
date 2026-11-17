"""D-46: supplier coherence re-asks only low-confidence units that disagree with a clear supplier majority."""
import numpy as np
import pandas as pd

from spendcube.classify.coherence import run_coherence, supplier_majorities
from spendcube.taxonomy.tree import Taxonomy


class FakeClf:
    def __init__(self):
        self.seen = []

    def classify(self, items, role, **kw):
        from spendcube.classify.llm import UnitResult
        self.seen = items
        return {it["uid"]: UnitResult(it["uid"], "T001", 0.9, 0, "fake") for it in items}


def test_majority_and_reask():
    tax = Taxonomy([("A", "x"), ("A", "y"), ("B", "z")])
    n = 8
    units = pd.DataFrame({"supplier_core": ["acme"] * 6 + ["solo", ""], "supplier": ["Acme"] * 6 + ["Solo", ""],
                          "gl": [""] * n, "ctx": [""] * n, "text": [f"t{i}" for i in range(n)],
                          "spend": [100.0, 90, 80, 70, 60, 50, 10, 10]})
    leaf = np.array(["T001", "T001", "T001", "T001", "T001", "T002", "T003", "T003"], dtype=object)
    conf = np.array([.95, .95, .95, .95, .95, .5, .5, .5])
    selfc, meth = conf.copy(), np.array(["llm"] * n, dtype=object)
    lab = np.ones(n, bool)
    maj = supplier_majorities(units, leaf, lab, 3, 0.5)
    assert maj["acme"][0] == "T001" and "solo" not in maj          # < min_units
    clf = FakeClf()
    st: dict = {}
    ch = run_coherence(units, leaf, conf, meth, selfc, lab, clf, tax, {"enabled": True}, st)
    assert [it["uid"] for it in clf.seen] == [5] and "supplier_majority=A > x" in clf.seen[0]["ctx"]
    assert ch == 1 and leaf[5] == "T001" and meth[5] == "llm_coherence" and st["coherence_candidates"] == 1
    assert (meth[:5] == "llm").all() and leaf[6] == "T003"          # confident / no-majority units untouched


def test_disabled_by_default():
    assert run_coherence(None, None, None, None, None, None, None, None, {}, {}) == 0
