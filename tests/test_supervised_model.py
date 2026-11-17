"""Synthetic tests for the supervised classifier, features and lookup (no kit data)."""
import numpy as np
import pandas as pd

from spendcube.classify.features import cat_series
from spendcube.classify.lookup import build_lookup, generic_materials
from spendcube.classify.supervised import SupervisedModel

ITEMS = {
    "T1": ["BEEF STEAK 5KG", "BEEF MINCE FROZEN", "BEEF RIBS", "STEAK SIRLOIN BEEF"],
    "T2": ["HAMMER CLAW 16OZ", "HAMMER SLEDGE 4KG", "CLAW HAMMER STEEL", "MALLET RUBBER HAMMER"],
    "T3": ["COPY PAPER A4 80GSM", "PAPER A3 BOX", "A4 PAPER REAM", "PRINTER PAPER 500 SHEETS"],
}


def make(n_per=40, seed=0, with_cat=True):
    rng = np.random.RandomState(seed)
    rows = []
    for leaf, texts in ITEMS.items():
        for _ in range(n_per):
            t = texts[rng.randint(len(texts))]
            rows.append({"desc_norm": t, "supplier_core": "SUP" + leaf if with_cat else "",
                         "vendor_id": "V" + leaf, "gl_account": "", "ctx__spend_type": "Material", "y": leaf})
    return pd.DataFrame(rows)


def test_fit_predict_and_interface():
    df = make()
    m = SupervisedModel().fit(df, df["y"])
    pr = m.predict(df)
    assert list(pr.columns) == ["leaf_id", "p", "top3"] and pr.index.equals(df.index)
    assert (pr["leaf_id"] == df["y"]).mean() > 0.99
    assert pr["p"].between(0, 1).all() and len(pr["top3"].iloc[0]) == 3


def test_missing_columns_tolerated_and_unseen_text():
    df = make()
    m = SupervisedModel().fit(df, df["y"])
    bare = pd.DataFrame({"desc_norm": ["BEEF SHORT RIB", "PAPER CARTON A4"]})
    pr = m.predict(bare)
    assert pr["leaf_id"].tolist() == ["T1", "T3"]


def test_single_class_and_single_sample_classes():
    df = make(n_per=2)
    one = df.iloc[:1]
    assert SupervisedModel().fit(one, ["T1"]).predict(df)["leaf_id"].eq("T1").all()
    d2 = pd.concat([make(n_per=30), pd.DataFrame([{"desc_norm": "ODDITY WIDGET", "y": "T9"}])], ignore_index=True)
    m = SupervisedModel().fit(d2, d2["y"])
    assert "T9" in set(m.classes_)


def test_weights_and_extra_docs_cover_unseen_leaf():
    df = make()
    m = SupervisedModel().fit(df, df["y"], weights=np.linspace(0.1, 5, len(df)),
                              extra_docs=[("WELDING ROD ELECTRODE", "T7"), ("WELDING WIRE", "T7")])
    assert "T7" in set(m.classes_)
    assert m.predict(pd.DataFrame({"desc_norm": ["WELDING ROD"]}))["leaf_id"].iloc[0] == "T7"


def test_deterministic_and_calibration_sets_temperature():
    df = make(n_per=120)
    a = SupervisedModel().fit(df, df["y"])
    b = SupervisedModel().fit(df, df["y"])
    assert a.T == b.T and np.allclose(a.predict(df)["p"], b.predict(df)["p"])
    assert 0.5 <= a.T <= 6.0
    assert SupervisedModel(calibrate=False).fit(df, df["y"]).T == 1.0


def test_feature_groups_and_ctx_tokens():
    df = make().head(3)
    s = cat_series(df, ("supplier", "ctx")).iloc[0]
    assert "SUP_SUPT1" in s and "SPEND_TYPE_MATERIAL" in s and "GL_" not in s
    only_llm = pd.DataFrame({"llm_context": ["spend_type=Material;opex_capex=Opex"], "desc_norm": ["X"]})
    assert "OPEX_CAPEX_OPEX" in cat_series(only_llm, ("ctx",)).iloc[0]
    assert cat_series(pd.DataFrame({"desc_norm": ["X"]}), ("supplier", "gl", "ctx", "material")).iloc[0] == ""


def test_text_only_group_ignores_categories():
    df = make()
    m = SupervisedModel(groups=("text",)).fit(df, df["y"])
    assert m.predict(df)["leaf_id"].eq(df["y"]).mean() > 0.99


def test_lookup_purity_support_and_generic():
    rows = []
    for i in range(10):       # material M1: pure -> hit
        rows.append({"material_id": "M1", "desc_compact": f"ITEMA{i % 2}", "y": "T1"})
    for i in range(12):       # generic placeholder: many distinct texts -> ignored as identity
        rows.append({"material_id": "MGEN", "desc_compact": f"FREETEXT{i}", "y": "T2" if i % 2 else "T3"})
    rows += [{"material_id": "", "desc_compact": "UNIQUETXT", "y": "T3"}] * 3
    rows += [{"material_id": "", "desc_compact": "MIXED", "y": "T1"}] * 2 + [{"material_id": "", "desc_compact": "MIXED", "y": "T2"}] * 2
    rows += [{"material_id": "", "desc_compact": "SINGLE", "y": "T1"}]
    df = pd.DataFrame(rows)
    assert generic_materials(df) == {"MGEN"}
    lk = build_lookup(df, df["y"])
    q = lk.query(pd.DataFrame({"material_id": ["M1", "MGEN", "", "", "", ""],
                               "desc_compact": ["X", "FREETEXT3", "UNIQUETXT", "MIXED", "SINGLE", "NEW"]}))
    assert q["leaf_id"].tolist()[0] == "T1" and q["conf"].iloc[0] == 1.0
    assert pd.isna(q["conf"].iloc[1])           # generic material id -> no identity; text not seen twice
    assert q["leaf_id"].iloc[2] == "T3"
    assert q["conf"].iloc[3:].isna().all()      # impure, unsupported, unseen
