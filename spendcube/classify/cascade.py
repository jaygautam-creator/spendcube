"""Classification cascade (architecture 8.3): direct labels -> supervised -> LLM (spend-ordered, cross-model
second opinion, arbiter) -> distillation -> supplier/GL propagation -> offline fallback.

Works on unique *units* (supplier, description, GL, context) ordered by |spend|; results are broadcast to rows.
Nothing here raises past `run_cascade`: each tier degrades to the next.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from ..labels import load_labels
from ..schema_cols import EMIT_ROLES
from .calibration import Calibrator
from .coherence import run_coherence
from .memory import UnitMemory
from .units import build_units

log = logging.getLogger("spendcube.cascade")


# ---------------------------------------------------------------------------------------------
def _training_units(ctx, df: pd.DataFrame):
    """Label-trained units: (units_frame, leaf_per_unit, weights) or None. Uses the run frame when it contains the
    labelled rows (build runs), else ingests the kit files (A2/B2 runs: labels belong to the original data)."""
    ccfg = ctx.cfg.classification or {}
    if not ccfg.get("use_labels", True) or ctx.cfg.labels is None:
        return None
    labels = load_labels(ctx)
    if labels is None or "leaf_id" not in labels.columns:
        return None
    lab = labels[labels["leaf_id"] != ""]
    if lab.empty:
        return None
    have = df[df["record_id"].isin(set(lab["record_id"]))]
    if len(have) >= 0.2 * len(lab):
        train = have
    else:
        try:
            from ..config.loader import load_config
            from ..context import RunContext
            from ..ingest.canonical import ingest_all
            from ..normalize.stage import normalize_stage
            cfg2 = load_config(ctx.cfg_path, input_dir=ctx.cfg.kit_dir, kit_dir=ctx.cfg.kit_dir)
            kit = ingest_all(cfg2)
            kit = kit[kit["record_id"].isin(set(lab["record_id"]))].copy()
            c2 = RunContext(cfg=cfg2, run_dir=ctx.run_dir, taxonomy=ctx.taxonomy)
            train = normalize_stage(c2, kit)
            log.info("training frame from kit files: %d labelled rows", len(train))
        except Exception as e:
            ctx.warn(f"could not load kit training rows ({e}); no supervised tier")
            return None
    train = train.drop(columns=["leaf_id"], errors="ignore").merge(
        lab[["record_id", "leaf_id"]], on="record_id", how="inner")
    if "spend_rc" not in train.columns:
        train["spend_rc"] = 0.0
    tu, truid = build_units(train, "spend_rc")
    maj = train.assign(uid=truid.to_numpy()).groupby("uid")["leaf_id"].agg(lambda s: s.value_counts().index[0])
    tu["leaf"] = tu["uid"].map(maj)
    tu["w"] = np.sqrt(tu["rows"].astype(float))
    ctx.note("classification", "label_training_rows", int(len(train)))
    ctx.note("classification", "label_training_units", int(len(tu)))
    return tu, lab.set_index("record_id")["leaf_id"]


def _leaf_docs(taxonomy, definitions: dict) -> list[tuple[str, str]]:
    docs = []
    for l in taxonomy.leaves:
        docs.append((" ".join(l.path).upper(), l.id))
        d = definitions.get(l.id)
        if d:
            docs.append((d.upper(), l.id))
    return docs


def run_cascade(ctx, df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    cfg, tax = ctx.cfg, ctx.taxonomy
    ccfg = cfg.classification or {}
    lcfg = ccfg.get("llm", {}) or {}
    first_leaf = tax.leaves[0].id
    df["leaf_id"] = first_leaf
    df["cat_conf"] = 0.01
    df["cat_method"] = "placeholder"
    df["unit_id"] = ""

    excl = df["excl_code"].fillna("") if "excl_code" in df.columns else pd.Series("", index=df.index)
    elig = df["source_role"].isin(EMIT_ROLES) & (excl == "")
    sub = df[elig]
    if sub.empty:
        return df
    units, row_uid = build_units(sub, "spend_rc")
    U = len(units)
    leaf = np.full(U, "", dtype=object)
    conf = np.zeros(U)
    meth = np.full(U, "", dtype=object)
    spcls = np.zeros(U, dtype=int)
    selfc = np.zeros(U)
    described = units["described"].to_numpy()
    spend = units["spend"].to_numpy()
    stats: dict = {"units": U, "described_units": int(described.sum())}

    def resolved() -> np.ndarray:
        return leaf != ""

    # ---- tier: supervised on labels ------------------------------------------------------
    from .supervised import SupervisedModel
    tr = None
    sup_pred = None
    try:
        tr = _training_units(ctx, df)
    except Exception as e:
        ctx.warn(f"label training data failed: {e}")
    sup_cfg = ccfg.get("supervised", {}) or {}
    accept_p = float(sup_cfg.get("accept_p", 0.80))
    if tr is not None and sup_cfg.get("enabled", "auto") is not False and len(tr[0]) >= int(sup_cfg.get("min_labels", 50)):
        try:
            tu, _ = tr
            mcfg0 = ccfg.get("memory", {}) or {}
            if mcfg0.get("enabled", True) and mcfg0.get("train", False) and not ctx.no_llm:
                try:      # own earlier confident LLM answers (same taxonomy) as extra, down-weighted training rows (D-45)
                    mu = UnitMemory(mcfg0.get("file", ".cache/unit_memory.sqlite"), tax.hash).training_units(
                        float(mcfg0.get("min_conf", 0.7)), float(mcfg0.get("train_weight", 0.3)))
                    if len(mu):
                        common = [c for c in tu.columns if c in mu.columns]
                        tu = pd.concat([tu, mu[common]], ignore_index=True)
                        stats["memory_training_units"] = int(len(mu))
                except Exception as e:
                    ctx.warn(f"unit memory training rows skipped: {type(e).__name__}: {e}")
            model = SupervisedModel().fit(tu, tu["leaf"], weights=tu["w"])
            sup_pred = model.predict(units)
            ok = (sup_pred["p"].to_numpy() >= accept_p)
            leaf[ok] = sup_pred["leaf_id"].to_numpy()[ok]
            conf[ok] = sup_pred["p"].to_numpy()[ok]
            meth[ok] = "supervised"
            stats["supervised_accepted_units"] = int(ok.sum())
            stats["supervised_accepted_spend_share"] = round(float(spend[ok].sum() / max(spend.sum(), 1e-9)), 4)
        except Exception as e:
            ctx.warn(f"supervised tier failed: {type(e).__name__}: {e}")

    llm_labelled = np.zeros(U, dtype=bool)
    # ---- tier: unit memory (answers of earlier runs on the same taxonomy; saves LLM requests) -------------
    mem = None
    mcfg = ccfg.get("memory", {}) or {}
    if mcfg.get("enabled", True) and not ctx.no_llm:
        try:
            mem = UnitMemory(mcfg.get("file", ".cache/unit_memory.sqlite"), tax.hash)
            ml, mc = mem.lookup(units, (~resolved()) & described, loose=bool(mcfg.get("loose", True)))
            hit = ml != ""
            cap = float(mcfg.get("conf_cap", 0.9))
            leaf[hit], conf[hit], meth[hit] = ml[hit], np.minimum(mc[hit], cap), "memory"
            llm_labelled[hit], selfc[hit] = True, conf[hit]      # they train distillation like fresh LLM answers
            stats["memory_units"] = int(hit.sum())
            stats["memory_spend_share"] = round(float(spend[hit].sum() / max(spend.sum(), 1e-9)), 4)
        except Exception as e:
            ctx.warn(f"unit memory lookup failed: {type(e).__name__}: {e}")
            mem = None

    # ---- tier: LLM -------------------------------------------------------------------------
    definitions: dict = {}
    router = None
    clf_obj = None
    try:
        router = _make_router(ctx)
        if router is not None and (router.has_llm("bulk") or router.try_outage()):
            definitions, clf_obj = _run_llm(ctx, units, leaf, conf, meth, spcls, selfc, llm_labelled, described, router, lcfg, stats)
        else:
            ctx.warn("no LLM provider available: using supervised/propagation/offline fallback only")
    except Exception as e:
        ctx.warn(f"LLM tier failed: {type(e).__name__}: {e}")
    if clf_obj is not None and (ccfg.get("coherence") or {}).get("enabled", False):
        try:
            run_coherence(units, leaf, conf, meth, selfc, llm_labelled, clf_obj, tax, ccfg["coherence"], stats)
        except Exception as e:
            ctx.warn(f"supplier coherence failed: {type(e).__name__}: {e}")
    if router is not None:
        ctx.notes["llm"] = router.report()
    if mem is not None:
        try:
            keep = np.isin(meth, ("llm", "llm_sc", "llm_coherence")) & (conf >= float(mcfg.get("min_conf", 0.7)))
            if keep.any():
                stats["memory_stored_units"] = mem.store(units, keep, leaf, conf)
        except Exception as e:
            ctx.warn(f"unit memory store failed: {type(e).__name__}: {e}")

    # ---- tier: distillation / supervised leftovers --------------------------------------------
    todo = (~resolved()) & described
    if todo.any():
        try:
            n_llm = int(llm_labelled.sum())
            if n_llm >= int(lcfg.get("distill_min_units", 30)):
                parts, ys, ws = [units[llm_labelled]], list(leaf[llm_labelled]), list(
                    np.clip(selfc[llm_labelled], 0.3, 1.0) * np.log1p(spend[llm_labelled]) + 0.5)
                if tr is not None:
                    parts.append(tr[0]); ys += list(tr[0]["leaf"]); ws += list(tr[0]["w"])
                trainset = pd.concat(parts, ignore_index=True)
                dm = SupervisedModel().fit(trainset, ys, weights=ws, extra_docs=_leaf_docs(tax, definitions))
                pr = dm.predict(units[todo])
                idx = np.where(todo)[0]
                leaf[idx] = pr["leaf_id"].to_numpy()
                conf[idx] = pr["p"].to_numpy() * 0.85
                meth[idx] = "llm_distill"
                stats["distilled_units"] = int(todo.sum())
            elif sup_pred is not None:
                idx = np.where(todo)[0]
                leaf[idx] = sup_pred["leaf_id"].to_numpy()[idx]
                conf[idx] = sup_pred["p"].to_numpy()[idx]
                meth[idx] = "supervised"
                stats["supervised_leftover_units"] = int(todo.sum())
        except Exception as e:
            ctx.warn(f"distillation failed: {type(e).__name__}: {e}")

    # ---- tier: propagation ---------------------------------------------------------------------
    if (ccfg.get("propagation", {}) or {}).get("enabled", True) and (~resolved()).any():
        try:
            _propagate(units, leaf, conf, meth, spend, float((ccfg.get("propagation", {}) or {}).get("min_share", 0.6)), stats)
        except Exception as e:
            ctx.warn(f"propagation failed: {e}")

    # ---- tier: blank / unresolved units: label-trained supervised top-1, else context-only LLM ------------
    if (~resolved()).any():
        try:
            from .blank import resolve_blank
            resolve_blank(units, leaf, conf, meth, spend, described, ccfg.get("blank", {}) or {}, stats,
                          extra_train=tr[0] if tr is not None else None)
        except Exception as e:
            ctx.warn(f"blank context tier failed: {type(e).__name__}: {e}")
    rest = ~resolved()
    if rest.any() and sup_pred is not None:
        idx = np.where(rest)[0]
        leaf[idx] = sup_pred["leaf_id"].to_numpy()[idx]
        conf[idx] = np.minimum(sup_pred["p"].to_numpy()[idx], 0.6)
        meth[idx] = "supervised"
        stats["supervised_blank_units"] = int(len(idx))
    elif rest.any() and router is not None and clf_obj is None and router.has_llm("bulk"):
        try:
            from .llm import LLMClassifier
            clf_obj = LLMClassifier(router, tax, cfg.industry_hint, definitions, lcfg)
        except Exception:
            clf_obj = None
    if (leaf == "").any() and clf_obj is not None and router is not None and router.has_llm("bulk") and sup_pred is None:
        idx = np.where(leaf == "")[0][: int(lcfg.get("blank_llm_units", 1500))]
        items = [{"uid": int(i), "supplier": units["supplier"].iat[i], "gl": units["gl"].iat[i],
                  "ctx": units["ctx"].iat[i], "text": "(blank description)"} for i in idx]
        try:
            rb = clf_obj.classify(items, "bulk", workers=4,
                                  user_prefix="NOTE: descriptions below are blank; infer the category from supplier, GL and context only; use low confidence when unsure.\n")
            for uid, v in rb.items():
                leaf[uid], conf[uid], meth[uid] = v.leaf_id, min(0.5, v.conf * 0.6), "llm_context"
            stats["context_only_llm_units"] = len(rb)
        except Exception as e:
            ctx.warn(f"context-only LLM tier failed: {e}")

    # ---- tier: offline fallback ------------------------------------------------------------------
    rest = ~resolved()
    if rest.any():
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import linear_kernel
            docs = _leaf_docs(tax, definitions)
            utext = (units.loc[rest, "desc_norm"].astype(str) + " " + units.loc[rest, "supplier"].astype(str)).str.upper()
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
            vec.fit([d for d, _ in docs] + list(utext))
            sims = linear_kernel(vec.transform(utext), vec.transform([d for d, _ in docs]))
            doc_leaf = np.array([l for _, l in docs], dtype=object)
            best = sims.argmax(1)
            idx = np.where(rest)[0]
            leaf[idx] = doc_leaf[best]
            conf[idx] = np.minimum(sims[np.arange(len(best)), best], 0.4)
            meth[idx] = "fallback"
        except Exception as e:
            ctx.warn(f"offline fallback failed ({e}); using most frequent leaf")
            top = Counter(leaf[resolved()]).most_common(1)
            leaf[rest] = top[0][0] if top else first_leaf
            conf[rest] = 0.05
            meth[rest] = "fallback"
    stats["fallback_units"] = int(rest.sum())

    # ---- LLM spend verdict (D-17) --------------------------------------------------------------------
    vconf = float(lcfg.get("spend_verdict_conf", 0.85))
    verdict_units = np.where((spcls != 0) & llm_labelled & (selfc >= vconf))[0] if lcfg.get("llm_spend_verdict", True) else []

    # ---- broadcast to rows -------------------------------------------------------------------------------
    pos = (row_uid.to_numpy() - 1).astype(int)
    sub_idx = sub.index
    df.loc[sub_idx, "leaf_id"] = leaf[pos]
    df.loc[sub_idx, "cat_conf"] = np.clip(conf[pos], 0.01, 0.99)
    df.loc[sub_idx, "cat_method"] = meth[pos]
    df.loc[sub_idx, "unit_id"] = ["U%d" % u for u in row_uid.to_numpy()]
    if len(verdict_units):
        vmask = np.isin(pos, verdict_units)
        protect = df.loc[sub_idx, "rule_include"].to_numpy() if "rule_include" in df.columns else np.zeros(len(sub_idx), bool)
        hit = sub_idx[vmask & ~protect.astype(bool)]
        df.loc[hit, "excl_code"] = "NON_PROCUREMENT"
        df.loc[hit, "excl_detail"] = "llm_spend_verdict"
        df.loc[hit, "excl_conf"] = selfc[pos][vmask & ~protect.astype(bool)]
        stats["llm_non_procurement_rows"] = int(len(hit))
    # direct row labels (label file rows keep their label)
    labels = load_labels(ctx) if (ccfg.get("use_labels", True) and cfg.labels is not None) else None
    if labels is not None and "leaf_id" in labels.columns:
        lm = labels[labels["leaf_id"] != ""].set_index("record_id")["leaf_id"]
        direct = df.loc[sub_idx, "record_id"].map(lm)
        has = direct.notna()
        df.loc[direct.index[has], "leaf_id"] = direct[has]
        df.loc[direct.index[has], "cat_conf"] = 0.97
        df.loc[direct.index[has], "cat_method"] = "label_lookup"
        stats["direct_label_rows"] = int(has.sum())
    df["cat_conf_raw"] = df["cat_conf"]
    cal_cfg = ccfg.get("calibration", {}) or {}
    if cal_cfg.get("enabled", True):
        cal = Calibrator.load(cal_cfg.get("file", "configs/_calibration.json"))
        if cal is not None:
            m = df["cat_method"] != "label_lookup"
            df.loc[m, "cat_conf"] = cal.apply(df.loc[m, "cat_method"], df.loc[m, "cat_conf"])
            stats["calibrated_groups"] = sorted(cal.groups)
    stats["method_units"] = dict(Counter(meth))
    stats["method_spend_share"] = {m: round(float(spend[meth == m].sum() / max(spend.sum(), 1e-9)), 4) for m in set(meth)}
    for k, v in stats.items():
        ctx.note("classification", k, v)
    ctx.note("classification", "seconds", round(time.time() - t0, 1))
    try:
        top = units.head(60)[["uid", "supplier", "text", "gl", "spend", "rows"]].copy()
        top["leaf"] = [tax.path_str(l) if l else "" for l in leaf[:60]]
        top["conf"] = conf[:60].round(3)
        top["method"] = meth[:60]
        top.to_csv(ctx.run_dir / "top_units.csv", index=False)
    except Exception:
        pass
    return df


# ---------------------------------------------------------------------------------------------
def _make_router(ctx, budget: bool = True):
    lcfg = (ctx.cfg.classification or {}).get("llm", {}) or {}
    if ctx.no_llm or not lcfg.get("enabled", True):
        return None
    if not lcfg.get("allow_training_use", False) and any(e.get("provider") == "gemini" for e in lcfg.get("chain", [])):
        # D-35: Gemini free tier needs explicit opt-in; drop gemini entries otherwise
        lcfg = {**lcfg, "chain": [e for e in lcfg["chain"] if e.get("provider") != "gemini"]}
    from ..llm.router import Router
    return Router(lcfg, mode=ctx.llm_mode, job_budget=budget)


def _run_llm(ctx, units, leaf, conf, meth, spcls, selfc, llm_labelled, described, router, lcfg, stats):
    from ..taxonomy.compile import compile_taxonomy
    from .llm import LLMClassifier
    tax = ctx.taxonomy
    deadline = float(lcfg.get("deadline_minutes", 60))
    # unresolved described units in spend order
    cand = np.where((leaf == "") & described)[0]
    if len(cand) == 0:
        return {}, None
    cand = cand[: int(lcfg.get("max_units", 20000))]
    definitions: dict = {}
    if lcfg.get("compile", True):
        spend_order = units.iloc[cand]
        texts = list(dict.fromkeys(spend_order["text"].head(150).tolist() +
                                   units.iloc[cand].sort_values("rows", ascending=False)["text"].head(100).tolist()))
        try:
            definitions = compile_taxonomy(router, tax, ctx.cfg.industry_hint, texts)
        except Exception as e:
            ctx.warn(f"taxonomy compile failed: {e}")
    stats["definitions_compiled"] = len(definitions)
    clf = LLMClassifier(router, tax, ctx.cfg.industry_hint, definitions, lcfg)
    idx1 = {l.id: i for i, l in enumerate(tax.leaves, 1)}
    top_n = int(lcfg.get("top_n", 1000))
    sc_n = int(lcfg.get("self_consistency_top_n", 300))
    tranche = int(lcfg.get("tranche_units", 240))
    stats["llm_candidate_units"] = int(len(cand))
    entry_cap = lcfg.get("entry_conf_cap", {}) or {}
    done = 0
    outage_t0 = None
    for s in range(0, len(cand), tranche):
        if ctx.elapsed_min > deadline:
            ctx.warn(f"LLM deadline ({deadline} min) reached after {done} units; remaining units use lower tiers")
            break
        if not router.has_llm("bulk") and router.try_outage():
            stats["outage_switch_at_unit"] = stats.get("outage_switch_at_unit", done)
            outage_t0 = outage_t0 if outage_t0 is not None else ctx.elapsed_min
            ctx.warn("LLM OUTAGE: Gemini unavailable; switched to the local model for the remaining top-spend units "
                     "(offline fallback for the rest)")
        if router.outage:
            outage_t0 = outage_t0 if outage_t0 is not None else ctx.elapsed_min
            if ctx.elapsed_min - outage_t0 > float((lcfg.get("outage") or {}).get("max_minutes", 20)):
                ctx.warn(f"local-model outage time budget spent after {done} units; remaining units use offline fallback")
                break
        if not router.has_llm("bulk"):
            ctx.warn("LLM request budget spent (%s) or all entries exhausted; remaining units use lower tiers"
                     % (router.budget_info or "n/a"))
            break
        chunk = cand[s: s + tranche]
        role = "top_spend" if s < top_n else "bulk"
        items = [{"uid": int(i), "supplier": units["supplier"].iat[i], "gl": units["gl"].iat[i],
                  "ctx": units["ctx"].iat[i], "text": units["text"].iat[i]} for i in chunk]
        r1 = clf.classify(items, role, workers=4)
        sc_items = [] if router.outage else [it for it in items if it["uid"] in r1 and (s + items.index(it)) < sc_n]
        r2, r3 = {}, {}
        if sc_items:
            used = tuple({r1[it["uid"]].entry for it in sc_items})
            r2 = clf.classify(sc_items, "second_opinion", exclude_entries=used, workers=4)
            dis = [dict(it, cand=(idx1[r1[it["uid"]].leaf_id], idx1[r2[it["uid"]].leaf_id]))
                   for it in sc_items if it["uid"] in r2 and r2[it["uid"]].leaf_id != r1[it["uid"]].leaf_id]
            if dis:
                r3 = clf.classify(dis, "arbiter", workers=2)
        for uid, v in r1.items():
            agree = 0.7
            final = v
            if uid in r2:
                if r2[uid].leaf_id == v.leaf_id:
                    agree = 1.0
                elif uid in r3:
                    final, agree = r3[uid], 0.4
                else:
                    agree = 0.3
            leaf[uid] = final.leaf_id
            selfc[uid] = final.conf
            conf[uid] = float(np.clip(0.5 * final.conf + 0.5 * agree, 0.05, 0.99))
            conf[uid] = min(conf[uid], float(entry_cap.get(final.entry, 1.0)))     # weak failover models (D-47)
            meth[uid] = "llm_sc" if uid in r2 else "llm"
            spcls[uid] = final.spend_class
            llm_labelled[uid] = True
        done += len(chunk)
    stats["llm_units_labelled"] = int(llm_labelled.sum())
    stats["llm_spend_share_labelled"] = round(float(units["spend"].to_numpy()[llm_labelled].sum() / max(units["spend"].sum(), 1e-9)), 4)
    return definitions, clf


def _propagate(units, leaf, conf, meth, spend, min_share: float, stats: dict) -> None:
    res = leaf != ""
    skey = np.where(units["supplier_core"].astype(str) != "", units["supplier_core"].astype(str),
                    np.where(units["vendor_id"].astype(str) != "", "V:" + units["vendor_id"].astype(str), ""))
    gkey = units["gl_account"].astype(str).to_numpy()
    w = np.log1p(spend) + 1.0
    n_sup = n_gl = 0
    for name, keys, tag in (("supplier", skey, "propagate_supplier"), ("gl", gkey, "propagate_gl")):
        dist: dict[str, Counter] = {}
        for k, l, ww, r in zip(keys, leaf, w, res):
            if r and k:
                dist.setdefault(k, Counter())[l] += ww
        for i in np.where(leaf == "")[0]:
            k = keys[i]
            if not k or k not in dist:
                continue
            c = dist[k]
            tot = sum(c.values())
            l, v = c.most_common(1)[0]
            share = v / tot
            if share >= min_share:
                leaf[i], conf[i], meth[i] = l, min(0.6, share * 0.9), tag
                n_sup += name == "supplier"
                n_gl += name == "gl"
    stats["propagated_supplier_units"], stats["propagated_gl_units"] = int(n_sup), int(n_gl)
