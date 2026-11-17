"""Blank-description tier: synthetic units only."""
import numpy as np
import pandas as pd

from spendcube.classify.blank import predict_blank, resolve_blank, vote_predict


def _units(rows):
    df = pd.DataFrame(rows, columns=["supplier_core", "vendor_id", "gl_account", "llm_context", "spend"])
    df["supplier_raw"] = df["supplier_core"]
    df["gl_desc"] = ""
    df["material_id"] = ""
    df["desc_norm"] = ""
    return df


def _world():
    rows, leaves = [], []
    for i in range(40):  # supplier S1 -> L1 (two units known), S2 -> L2
        rows.append(["S1", "V1", "GL1", "t=a", 100 + i]); leaves.append("leaf1")
        rows.append(["S2", "V2", "GL2", "t=b", 100 + i]); leaves.append("leaf2")
    return _units(rows), np.array(leaves, dtype=object)


def test_vote_takes_supplier_majority_weighted_by_spend():
    ku, kl = _world()
    kl = kl.copy()
    kl[0] = "leaf2"   # one dissenting small unit does not flip S1
    blank = _units([["S1", "V1", "GL9", "t=a", 5], ["NEW", "VX", "GL9", "t=z", 5]])
    leaf, share = vote_predict(blank, ku, kl, np.full(len(ku), 5.0), 0.6)
    assert leaf[0] == "leaf1" and share[0] >= 0.9
    assert leaf[1] == ""                         # unseen keys never vote


def test_predict_blank_vote_first_then_context_model():
    ku, kl = _world()
    blank = _units([["S1", "V1", "GL1", "t=a", 50], ["NEW", "VX", "GL7", "t=b", 50]])
    allu = pd.concat([ku, blank], ignore_index=True)
    known = np.r_[np.ones(len(ku), bool), np.zeros(2, bool)]
    kleaf = np.r_[kl, np.array(["", ""], dtype=object)]
    spend = allu["spend"].to_numpy(float)
    leaf, conf, meth = predict_blank(allu, known, kleaf, spend, {"mode": "vote_first"})
    assert leaf[-2] == "leaf1" and meth[-2] == "ctx_vote"
    assert leaf[-1] == "leaf2" and meth[-1] == "ctx_model"      # unseen supplier+GL: context tokens decide
    assert (conf[-2:] <= 0.6).all() and (conf[-2:] > 0).all()


def test_resolve_blank_fills_in_place_and_never_touches_resolved():
    ku, kl = _world()
    blank = _units([["S2", "V2", "GL2", "t=b", 50]])
    u = pd.concat([ku, blank], ignore_index=True)
    leaf = np.r_[kl, np.array([""], dtype=object)]
    conf = np.r_[np.full(len(ku), 0.9), [0.0]]
    meth = np.r_[np.full(len(ku), "llm", dtype=object), [""]]
    described = np.r_[np.ones(len(ku), bool), [False]]
    before = leaf[:-1].copy()
    stats = {}
    n = resolve_blank(u, leaf, conf, meth, u["spend"].to_numpy(float), described, {}, stats)
    assert n == 1 and leaf[-1] == "leaf2" and meth[-1].startswith("ctx_")
    assert (leaf[:-1] == before).all() and (conf[:-1] == 0.9).all()
    assert stats["blank_context_units"] == 1


def test_resolve_blank_disabled_and_empty_known_are_noops():
    ku, _ = _world()
    leaf = np.full(len(ku), "", dtype=object)
    conf = np.zeros(len(ku)); meth = np.full(len(ku), "", dtype=object)
    sp = ku["spend"].to_numpy(float)
    assert resolve_blank(ku, leaf, conf, meth, sp, np.zeros(len(ku), bool), {}, {}) == 0
    assert resolve_blank(ku, leaf, conf, meth, sp, np.ones(len(ku), bool), {"enabled": False}, {}) == 0
