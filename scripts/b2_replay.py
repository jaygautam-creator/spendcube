"""B2 demand by REPLAY (no quota): date split of the finished full B run (first 75% of posting dates = build, last 25% = sim).

Unit memory = the build part's LLM-answered units (conf >= 0.7) from the full run's final frame (those answers came from the
cached LLM responses). For the sim part: how many of the units that needed the LLM in the full run are already in memory (exact or
loose key), how many requests the novel ones would still need (60 units per request + second opinion on the top 300), and how
accurate the memory answers are on held-out seed rows in the sim part.
usage: python scripts/b2_replay.py RUN_DIR OUTPUT_CSV [--build-frac 0.75] [--config configs/B.yaml]
Numbers only are printed/written (runs/b2_replay.json).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.classify.memory import unit_keys  # noqa: E402
from spendcube.classify.units import build_units  # noqa: E402
from spendcube.config.loader import load_config  # noqa: E402
from spendcube.schema_cols import EMIT_ROLES  # noqa: E402
from spendcube.taxonomy.tree import Taxonomy  # noqa: E402


def unit_table(df: pd.DataFrame):
    units, uid = build_units(df, "spend_rc")
    pos = (uid.to_numpy() - 1).astype(int)
    meth = df["cat_method"].to_numpy()
    leaf = df["leaf_id"].to_numpy()
    conf = pd.to_numeric(df["cat_conf"]).to_numpy()
    # unit-level answer = first row's (all rows of a unit share the answer)
    first = pd.DataFrame({"pos": pos, "meth": meth, "leaf": leaf, "conf": conf}).drop_duplicates("pos").set_index("pos").sort_index()
    units = units.reset_index(drop=True)
    units["meth"], units["leaf"], units["conf"] = first["meth"].to_numpy(), first["leaf"].to_numpy(), first["conf"].to_numpy()
    ex, lo = unit_keys(units)
    units["kx"], units["kl"] = ex, lo
    return units, pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--build-frac", type=float, default=0.75)
    ap.add_argument("--config", default="configs/B.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tax = Taxonomy.load(cfg.taxonomy)
    f = pd.read_parquet(Path(a.run_dir) / "final_frame.parquet")
    f = f[f["source_role"].isin(EMIT_ROLES) & (f["excl_code"].fillna("") == "")].copy()
    f["_d"] = pd.to_datetime(f["posting_date"], errors="coerce")
    f = f.sort_values("_d", kind="stable")
    cut = int(len(f) * a.build_frac)
    build, sim = f.iloc[:cut], f.iloc[cut:]
    llm_m = ("llm", "llm_sc")
    ub, _ = unit_table(build)
    us, spos = unit_table(sim)
    mem = ub[ub["meth"].isin(llm_m) & (ub["conf"] >= 0.7)]
    kx, kl = dict(zip(mem["kx"], mem["leaf"])), dict(zip(mem["kl"], mem["leaf"]))
    need = us[us["meth"].isin(llm_m)].copy()                       # units the full run sent to the LLM
    hit_x = need["kx"].isin(kx)
    hit_l = need["kl"].isin(kl) & ~hit_x
    novel = need[~(hit_x | hit_l)]
    n_need, n_novel = len(need), len(novel)
    reqs = lambda n: math.ceil(n / 60) + math.ceil(min(n, 300) / 60) if n else 0
    out = {"build_rows": len(build), "sim_rows": len(sim), "build_units": len(ub), "sim_units": len(us),
           "memory_units_from_build": len(mem),
           "sim_units_needing_llm_without_memory": n_need, "answered_by_memory_exact": int(hit_x.sum()),
           "answered_by_memory_loose": int(hit_l.sum()), "novel_units": n_novel,
           "memory_hit_share_units": round(1 - n_novel / max(n_need, 1), 3),
           "memory_hit_share_spend": round(float(1 - novel["spend"].sum() / max(need["spend"].sum(), 1e-9)), 3),
           "requests_sim_without_memory_est": reqs(n_need), "requests_sim_with_memory_est": reqs(n_novel),
           "requests_full_run_actual_approx": 255}
    # accuracy of memory answers on held-out seed rows of the sim part
    ho = pd.read_csv(Path(a.run_dir) / "heldout_labels.csv", dtype=str, keep_default_na=False)
    outp = pd.read_csv(a.output_csv, dtype=str, keep_default_na=False)[["record_id", "spend"]]
    sim_ids = sim.assign(pos=spos).reset_index(drop=True)
    sim_ids["kx"], sim_ids["kl"] = us.loc[spos, "kx"].to_numpy(), us.loc[spos, "kl"].to_numpy()
    sim_ids["mem_leaf"] = sim_ids["kx"].map(kx).fillna(sim_ids["kl"].map(kl))
    m = ho.merge(sim_ids[["record_id", "mem_leaf", "leaf_id", "cat_method"]], on="record_id").merge(outp, on="record_id")
    m["spend"] = pd.to_numeric(m["spend"]).abs()
    mm = m[m["mem_leaf"].notna() & m["cat_method"].isin(llm_m)]
    if len(mm):
        tp = ["l1", "l2", "l3"]
        truth = mm[tp].to_numpy()
        pred = np.array([list(tax.leaf(l).path[:3]) + [""] * (3 - len(tax.leaf(l).path[:3])) for l in mm["mem_leaf"]])
        full = np.array([list(tax.leaf(l).path[:3]) + [""] * (3 - len(tax.leaf(l).path[:3])) for l in mm["leaf_id"]])
        w = mm["spend"].to_numpy()
        for name, p in (("memory", pred), ("full_run_llm", full)):
            hit = (truth == p).all(axis=1)
            out[f"heldout_sim_{name}_L3"] = {"n": int(len(mm)), "count": round(float(hit.mean()), 3),
                                              "spend_weighted": round(float((hit * w).sum() / max(w.sum(), 1e-9)), 3)}
    else:
        out["heldout_sim_memory_L3"] = {"n": 0}
    # self-training lever: supervised model fed with the build part's own confident answers (weight 0.3 x conf for LLM answers),
    # how many of the sim part's LLM-bound units it would accept (p >= 0.8) without a request, and how often it agrees with the LLM
    try:
        from spendcube.classify.supervised import SupervisedModel
        lab = ub[ub["meth"].isin(("label_lookup", "supervised"))]
        tr = pd.concat([lab, mem], ignore_index=True)
        w = np.r_[np.sqrt(lab["rows"].astype(float).to_numpy()), 0.3 * mem["conf"].to_numpy()]
        mdl = SupervisedModel().fit(tr, tr["leaf"], weights=w)
        pr = mdl.predict(need)
        ok = pr["p"].to_numpy() >= 0.8
        out["selftrain_accept_p80_units"] = int(ok.sum())
        out["selftrain_accept_share_of_llm_bound"] = round(float(ok.mean()), 3)
        out["selftrain_agree_with_full_run_llm"] = round(float((pr["leaf_id"].to_numpy()[ok] == need["leaf"].to_numpy()[ok]).mean()), 3) if ok.any() else None
        rest = need[~ok]
        out["requests_sim_with_memory_and_selftrain_est"] = reqs(len(rest) - int(((rest["kx"].isin(kx)) | rest["kl"].isin(kl)).sum()))
    except Exception as e:
        out["selftrain_error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out, indent=1))
    Path("runs/b2_replay.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
