import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.eval.recon_holdout import core_route_holdout, load_prepared, real_nopo_effect

core = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
df, cfg = load_prepared()
print("prepared", len(df), "core overrides", core, flush=True)
print("REAL no-PO rows:", json.dumps(real_nopo_effect(df, cfg, core)), flush=True)
r = core_route_holdout(df, cfg, core=core)
tot = {}
for seed, row in r["seeds"].items():
    for lab in ("off", "on"):
        for pop in ("blanked_rows", "po_bearing_rows"):
            for m in ("n", "zar"):
                c = row[lab][pop][m]
                print(f"seed{seed} {lab:3s} {pop:15s} {m:3s} TP={c['TP']:>12} FP={c['FP']:>12} FN={c['FN']:>12} "
                      f"TN={c['TN']:>12} correct={c['correct']:>12} P={c['precision']} R={c['recall']}")
        print(f"seed{seed} {lab} core_assigned={row[lab]['core_assigned']} n_blanked={row['n_blanked']}")
