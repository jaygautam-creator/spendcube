"""P2 exit checks re-run after the D-19/D-22 follow-ups: synthetic P/R and pseudo-A2 agreement."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.eval.recon_holdout import load_prepared, pseudo_a2
from spendcube.eval.synth import matching_synth

print("== synthetic (default = core off/partial off; variant = core on)")
r = matching_synth(3000, 0, verbose=False)
for k, v in r.items():
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk in ("all_tool_rows", "rows_with_po", "emitted_tool_rows")}, default=str))

df, cfg = load_prepared()
for label, over in (("core on, partial off", dict(core_route={"enabled": True, "when": "no_po"}, partial_capacity={"enabled": False})),
                    ("core on, partial on", dict(core_route={"enabled": True, "when": "no_po"}, partial_capacity={"enabled": True})),
                    ("core off, partial off", dict(core_route={"enabled": False}, partial_capacity={"enabled": False}))):
    for seed in (1, 2):
        print("pseudo-A2", label, "seed", seed, json.dumps(pseudo_a2(df, cfg, seed, **over)), flush=True)
