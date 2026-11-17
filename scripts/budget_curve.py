"""Budget -> accuracy curve by CACHE REPLAY (no quota): holdout e2e with the LLM cache of an earlier full run, the router
counting every pool-entry answer (cached or not) against a fixed request budget, in its own priority order (top-spend
tranches first, then bulk). Past the budget the run falls to supervised / propagation / blank / offline tiers.

usage: python scripts/budget_curve.py configs/B.yaml runs/demand_Bbase_cache.sqlite [--budgets 0,25,50,100,140,unlimited]
Prints and writes runs/budget_curve_<company>.json (numbers only).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from spendcube.config.loader import load_config  # noqa: E402
from spendcube.eval.e2e import run_e2e  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("cache")
    ap.add_argument("--budgets", default="0,25,50,100,140,unlimited")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    base = load_config(a.config)
    out = {"company": base.company, "cache": Path(a.cache).name, "curve": []}
    for tok in a.budgets.split(","):
        cl = copy.deepcopy(base.classification)
        cl["llm"]["cache_file"] = a.cache
        cl["memory"] = {"enabled": False}            # isolate the budget effect
        if tok == "unlimited":
            cl["llm"]["budget"] = {"enabled": False}
        else:
            pool = cl["llm"]["budget"].get("pool", ["gemini_flash_lite_35", "gemini_flash_lite_31"])
            cl["llm"]["budget"] = {"enabled": True, "fixed": int(tok), "count_cached": True, "pool": pool,
                                   "exempt_roles": ["compile", "arbiter"]}
        res = run_e2e(a.config, llm_mode="dev", out_dir=f"runs/budget_curve_{base.company}_{tok}",
                      overrides={"classification": cl})
        row = {"budget": tok, "L1_sw": res["all"]["L1_sw"], "L3_sw": res["all"]["L3_sw"], "L3_count": res["all"]["L3"],
               "n": res["all"]["n"],
               "by_method_L3_sw": {k: [v["n"], v["L3_sw"], v["spend_share"]] for k, v in res["by_method"].items()}}
        out["curve"].append(row)
        print(json.dumps(row), flush=True)
    Path(f"runs/budget_curve_{base.company}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
