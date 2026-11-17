"""B2 demand simulation (time-ordered): BUILD run on the first `--build-frac` of the primary file populates the unit
memory; the SIM run on the full file (fresh LLM cache, memory from BUILD, labels = seeds) is what a new extract of the
same company would cost. Reports LLM requests (ledger delta) for both runs and holdout accuracy of the SIM run.
usage: python scripts/b2_sim.py configs/B.yaml [--build-frac 0.7] [--no-memory-train]
Dev key only (llm_mode=dev); the job budget is disabled so the raw demand is visible.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from spendcube.config.loader import load_config  # noqa: E402
from spendcube.eval.e2e import run_e2e  # noqa: E402
from spendcube.llm.ledger import Ledger  # noqa: E402


def _delta(led, before):
    after = {(r[0], r[1]): r[2:6] for r in led.summary()}
    d = {k[1]: [a - b for a, b in zip(v, before.get(k, (0, 0, 0, 0)))] for k, v in after.items() if v != before.get(k)}
    return d, sum(v[0] for v in d.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--build-frac", type=float, default=0.7)
    ap.add_argument("--no-memory-train", action="store_true")
    ap.add_argument("--tag", default="b2sim")
    a = ap.parse_args()
    base = load_config(a.config)
    tag = a.tag
    mem = f"runs/{tag}_memory.sqlite"
    Path(mem).unlink(missing_ok=True)
    led = Ledger(".cache/llm_ledger.sqlite")
    out: dict = {"config": a.config, "build_frac": a.build_frac}

    def overrides(cache: str, limit):
        cl = copy.deepcopy(base.classification)
        cl["llm"]["budget"]["enabled"] = False
        cl["llm"]["cache_file"] = cache
        cl["memory"] = {"enabled": True, "file": mem, "train": not a.no_memory_train}
        ov = {"classification": cl}
        if limit:
            src = [s.model_dump(by_alias=True) for s in base.sources]
            src[0]["limit"] = limit
            ov["sources"] = src
        return ov

    # row count of the primary file (to place the cut)
    from spendcube.ingest.canonical import ingest_all
    n = int((ingest_all(base)["source_role"] == "primary").sum())
    for name, limit in (("build", int(n * a.build_frac)), ("sim", None)):
        before = {(r[0], r[1]): r[2:6] for r in led.summary()}
        t0 = time.time()
        res = run_e2e(a.config, llm_mode="dev", out_dir=f"runs/{tag}_{name}",
                      overrides=overrides(f"runs/{tag}_{name}_cache.sqlite", limit))
        d, total = _delta(led, before)
        out[name] = {"rows_limit": limit, "requests": total, "by_entry": d, "seconds": round(time.time() - t0),
                     "holdout_all": res["all"], "by_method": {k: {"n": v["n"], "L3_sw": v["L3_sw"]} for k, v in res["by_method"].items()}}
        print(json.dumps({name: out[name]}, default=str), flush=True)
    Path(f"runs/{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
