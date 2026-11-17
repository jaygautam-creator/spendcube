"""CLI: python -m spendcube <run|validate|doctor|profile|eval> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass


def cmd_run(a) -> int:
    import os
    from .config.loader import load_config
    from .llm.safety import mock_active, reject_mock_in_blind
    from .pipeline import run_pipeline
    _load_env()
    mode = a.llm_mode or os.environ.get("SPENDCUBE_LLM_MODE", "dev")
    if reject_mock_in_blind(mode):
        return 2
    cfg = load_config(a.config, input_dir=a.input_dir, kit_dir=a.kit_dir, suffix=a.suffix)
    active = mock_active(cfg.classification.get("llm", {}))
    if active:
        print("SPENDCUBE MOCK ACTIVE: mock_active: true", file=sys.stderr)
    path, errors = run_pipeline(cfg, out_dir=Path(a.out), no_llm=a.no_llm, limit=a.limit,
                                llm_mode=mode, verbose=a.verbose, cfg_path=a.config, mock_active=active)
    print(f"{path}  validation={'OK' if not errors else 'FAILED'}")
    for e in errors:
        print("  -", e)
    return 0 if not errors else 2


def cmd_validate(a) -> int:
    from .config.loader import load_config
    from .ingest.canonical import ingest_all
    from .output.validator import validate_file
    from .taxonomy.tree import Taxonomy
    cfg = load_config(a.config, input_dir=a.input_dir, kit_dir=a.kit_dir, suffix=a.suffix)
    tax = Taxonomy.load(cfg.taxonomy)
    df = ingest_all(cfg, limit=a.limit)
    ids = df.loc[df["source_role"].isin(("primary", "secondary")), "record_id"].tolist()
    errs = validate_file(Path(a.output), tax, ids, expected_name=None)
    print("OK" if not errs else "FAILED")
    for e in errs:
        print(" -", e)
    return 0 if not errs else 2


def cmd_doctor(a) -> int:
    from .llm.doctor import run_doctor
    _load_env()
    from .llm.safety import reject_mock_in_blind
    if reject_mock_in_blind("blind" if a.blind else "dev", doctor=True):
        return 2
    return run_doctor(a.config, "blind" if a.blind else "dev", generate=not getattr(a, "list_only", False))


def cmd_eval(a) -> int:
    import json
    _load_env()
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if a.what == "zero-label":
        from .eval.zero_label import run_zero_label
        res = run_zero_label(a.config, a.top, a.strat, Path(a.out_dir), tuple(a.variants.split(",")),
                             a.sample_file, a.llm_mode or "dev")
        print(json.dumps(res, indent=2, default=str))
        return 0
    if a.what in ("a2-sim-e2e", "e2e"):
        from .eval.e2e import run_e2e
        print(json.dumps(run_e2e(a.config, no_llm=a.no_llm, llm_mode=a.llm_mode or "dev"), indent=2, default=str)); return 0
    if a.what == "a2-sim":
        from .eval.supervised_eval import run_a2_sim
        print(json.dumps(run_a2_sim(a.config), indent=2, default=str)); return 0
    if a.what == "b-seeds":
        from .eval.supervised_eval import run_b_seeds
        print(json.dumps(run_b_seeds(a.config), indent=2, default=str)); return 0
    if a.what == "blank":
        from .eval.blank_eval import run_blank_eval
        print(json.dumps(run_blank_eval(a.config, a.sample_file or None, mode="llm" if not a.no_llm else "label"),
                         indent=2, default=str)); return 0
    if a.what == "coherence":
        from .eval.coherence_eval import run_coherence_eval
        print(json.dumps(run_coherence_eval(a.config, out_file=a.sample_file or None), indent=2, default=str)); return 0
    if a.what == "matching-synth":
        from .eval.synth import matching_synth
        print(json.dumps(matching_synth(), indent=2, default=str)); return 0
    if a.what == "entities":
        from .eval.entities import run_entities_eval
        print(json.dumps(run_entities_eval(a.config), indent=2, default=str)); return 0
    print("eval type not implemented yet:", a.what)
    return 1


def cmd_profile(a) -> int:
    from .profile.draft_config import main_profile
    return main_profile(a) or 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spendcube")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("--input-dir", default=None)
        sp.add_argument("--kit-dir", default=None)
        sp.add_argument("--suffix", default=None, help="output suffix, e.g. A2 -> <name>__A2.csv")
        sp.add_argument("--limit", type=int, default=None)

    r = sub.add_parser("run", help="run the pipeline")
    common(r)
    r.add_argument("--out", default="outputs")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--llm-mode", choices=["dev", "blind"], default=None)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="validate an output file against the inputs")
    common(v)
    v.add_argument("--output", required=True)
    v.set_defaults(fn=cmd_validate)

    d = sub.add_parser("doctor", help="check keys, models and quotas (dev key unless --blind)")
    d.add_argument("--config", default="configs/A.yaml")
    d.add_argument("--blind", action="store_true")
    d.add_argument("--list-only", action="store_true",
                   help="metadata only: key presence, models.list, ledger reads; ZERO generate requests")
    d.set_defaults(fn=cmd_doctor)

    e = sub.add_parser("eval", help="evaluation harness")
    e.add_argument("what", choices=["zero-label", "a2-sim", "a2-sim-e2e", "e2e", "b-seeds", "entities", "matching-synth", "blank", "coherence", "all"])
    e.add_argument("--config", required=True)
    e.add_argument("--top", type=int, default=150)
    e.add_argument("--strat", type=int, default=150)
    e.add_argument("--variants", default="nodefs,defs")
    e.add_argument("--sample-file", default=None)
    e.add_argument("--out-dir", default="runs/eval_zero_label")
    e.add_argument("--llm-mode", choices=["dev", "blind"], default=None)
    e.add_argument("--no-llm", action="store_true")
    e.set_defaults(fn=cmd_eval)

    pr = sub.add_parser("profile", help="profile a raw file and draft a config")
    pr.add_argument("--file", required=True)
    pr.add_argument("--taxonomy", default=None)
    pr.add_argument("--draft-config", default=None)
    pr.add_argument("--accept-suggestions", action="store_true", help="write the suggested exclusion rules uncommented")
    pr.add_argument("--company", default="C")
    pr.add_argument("--reporting-currency", default="USD")
    pr.add_argument("--input-dir", default=None)
    pr.add_argument("--fx-dir", default=None)
    pr.set_defaults(fn=cmd_profile)

    a = p.parse_args(argv)
    if getattr(a, "llm_mode", None) is None and hasattr(a, "llm_mode"):
        import os
        a.llm_mode = os.environ.get("SPENDCUBE_LLM_MODE", "dev")
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
