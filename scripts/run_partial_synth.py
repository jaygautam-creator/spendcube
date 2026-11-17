import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spendcube.eval.synth import partial_synth
for seed in (0, 1, 2):
    partial_synth(3000, seed)
