"""Run the entity eval and print a compact table. Usage: python scripts/run_entities_eval.py [config] [out.json]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.eval.entities import run_entities_eval  # noqa: E402

cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/A.yaml"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/p3/eval.json"
Path(out).parent.mkdir(parents=True, exist_ok=True)
r = run_entities_eval(cfg)
json.dump(r, open(out, "w"), indent=1, default=str)
KEYS = ("n", "pred_groups", "gold_groups", "pair_P", "pair_R", "pair_F1", "b3_F1")


def show(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict) and "pair_F1" in v:
            print(p + k, {x: v[x] for x in KEYS})
        elif isinstance(v, dict):
            show(v, p + k + ".")


u = r["unsupervised"]
show({k: v for k, v in u.items() if k not in ("supplier_stats", "item_stats")}, "unsup.")
print("sup_stats", u["supplier_stats"])
print("item_stats", u["item_stats"], u["generic_materials"])
show({k: v for k, v in r["alias_on_holdout"].items() if isinstance(v, dict)}, "holdout.")
print({k: v for k, v in r["alias_on_holdout"].items() if not isinstance(v, dict)})
