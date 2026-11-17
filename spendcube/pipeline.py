"""Pipeline orchestration. Stage-level failures degrade (log + continue); the run always ends
with a validated file (architecture sections 1, 9)."""
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config.loader import load_config
from .config.schema import Config
from .context import RunContext, make_run_dir, setup_logging
from .ingest.canonical import ingest_all
from .output.report import write_report
from .output.validator import validate_frame
from .output.writer import assemble, ensure_columns, write_output
from .taxonomy.tree import Taxonomy

log = logging.getLogger("spendcube")
LAST_RUN_DIR: Optional[Path] = None


def _stages() -> list[tuple[str, Callable]]:
    from .classify.stage import classify_stage
    from .entities.stage import entities_stage
    from .exclusions.stage import exclusions_stage
    from .normalize.stage import normalize_stage
    stages = [("normalize", normalize_stage), ("exclusions", exclusions_stage)]
    try:
        from .reconcile.stage import reconcile_stage
        stages.append(("reconcile", reconcile_stage))
    except ImportError:
        pass
    stages += [("entities", entities_stage), ("classify", classify_stage)]
    return stages


def run_pipeline(cfg: Config, out_dir: Path = Path("outputs"), no_llm: bool = False,
                 limit: Optional[int] = None, llm_mode: str = "dev",
                 verbose: bool = False, cfg_path: Optional[str] = None,
                 mock_active: Optional[bool] = None) -> tuple[Path, list[str]]:
    global LAST_RUN_DIR
    run_dir = make_run_dir(cfg.company)
    LAST_RUN_DIR = run_dir
    setup_logging(run_dir, verbose)
    if mock_active is None:
        from .llm.safety import mock_active as is_mock_active
        mock_active = is_mock_active(cfg.classification.get("llm", {}))
    ctx = RunContext(cfg=cfg, run_dir=run_dir, no_llm=no_llm, limit=limit, out_dir=Path(out_dir), llm_mode=llm_mode, cfg_path=cfg_path)
    ctx.note("run", "mock_active", bool(mock_active))
    log.info("run %s -> %s", cfg.company, ctx.run_dir)

    with ctx.timer("taxonomy"):
        ctx.taxonomy = Taxonomy.load(cfg.taxonomy)          # hard failure: nothing can be output without a tree
    ctx.note("taxonomy", "summary", ctx.taxonomy.summary())
    with ctx.timer("ingest"):
        df = ingest_all(cfg, limit=limit)                    # hard failure
    expected_ids = df.loc[df["source_role"].isin(("primary", "secondary")), "record_id"].tolist()
    log.info("ingested %d rows (%d emitted)", len(df), len(expected_ids))

    for name, fn in _stages():
        with ctx.timer(name):
            try:
                before = len(df)
                df = fn(ctx, df)
                if len(df) != before:
                    raise RuntimeError(f"stage {name} changed row count {before} -> {len(df)}")
            except Exception as e:  # degrade: keep previous frame
                ctx.warn(f"STAGE FAILED {name}: {type(e).__name__}: {e}")
                log.debug(traceback.format_exc())
                (run_dir / f"stage_error_{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                df = _recover(df, before)
        log.info("stage %-11s done (%.1fs)", name, ctx.timings.get(name, 0))

    with ctx.timer("assemble"):
        df = ensure_columns(ctx, df)
        out = assemble(ctx, df)
        errors = validate_frame(out, ctx.taxonomy, expected_ids)
        if errors:
            ctx.warn(f"validation errors before repair: {errors}")
        path = Path(out_dir) / cfg.out_filename
        write_output(out, path)
    inc = df["excl_code"] == ""
    ctx.state["excluded_spend"] = df.loc[(~inc) & df["source_role"].isin(("primary", "secondary"))] \
        .groupby("excl_code")["spend_rc"].sum().to_dict()
    llm_note = ctx.notes.get("llm", {})
    call_metrics = llm_note.pop("call_metrics", [])
    if call_metrics:
        pd.DataFrame(call_metrics, columns=["model", "entry", "wall_ms", "n_units", "tokens_in", "tokens_out"]) \
            .to_csv(run_dir / "llm_call_metrics.csv", index=False)
        llm_note["call_metrics_file"] = "llm_call_metrics.csv"
        llm_note["call_metrics_count"] = len(call_metrics)
    write_report(ctx, out, run_dir / "report.md", errors)
    if "labels_heldout" in ctx.state:
        ho = ctx.state["labels_heldout"]
        if "leaf_id" not in ho.columns and "labels" in ctx.state:
            pass
        ho.to_csv(run_dir / "heldout_labels.csv", index=False)
    try:
        df.drop(columns=[c for c in df.columns if c.startswith("ctx__")]).to_parquet(run_dir / "final_frame.parquet")
    except Exception as e:  # checkpoint is best-effort
        log.debug("checkpoint skipped: %s", e)
    log.info("wrote %s (%d rows) validation=%s total_time=%.1fs", path, len(out),
             "OK" if not errors else f"FAILED {errors}", sum(ctx.timings.values()))
    return path, errors


def _recover(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df
