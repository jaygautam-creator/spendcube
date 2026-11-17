import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.eval.recon_holdout import load_prepared, partial_effect, partial_hidden_holdout

df, cfg = load_prepared()
print("EFFECT", json.dumps(partial_effect(df, cfg)), flush=True)
for seed, row in partial_hidden_holdout(df, cfg).items():
    print("HIDDEN seed", seed, json.dumps(row))
