"""LLM spike: classify the top-N described A ERP units (labels hidden) against the tree. DEV key only."""
import json, sys, time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from spendcube.config.loader import load_config
from spendcube.ingest.canonical import ingest_all
from spendcube.llm.prompts import RESPONSE_SCHEMA, build_system, build_user, unit_line
from spendcube.llm.providers.gemini import make_gemini
from spendcube.normalize.stage import normalize_stage
from spendcube.taxonomy.tree import Taxonomy

model, level, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = load_config(ROOT / "configs/A.yaml", input_dir=str(ROOT / "Cantill_DataML_Task_Kit/company_A"))
tax = Taxonomy.load(cfg.taxonomy)
df = ingest_all(cfg)
df = df[df.source == "erp"].copy()
class _C: cfg = cfg
df = normalize_stage(_C, df)
lab = pd.read_csv(ROOT / "Cantill_DataML_Task_Kit/company_A/labels_A.csv", dtype=str, keep_default_na=False)
df = df.merge(lab, on="record_id", how="left")
df["spend"] = df["spend_rc"]
d = df[~df.desc_blank].copy()
g = d.groupby(["supplier_core", "desc_compact"]).agg(spend=("spend", "sum"), text=("desc_raw", "first"),
        sup=("supplier_raw", "first"), l1=("category_l1", lambda s: Counter(s).most_common(1)[0][0]),
        l2=("category_l2", lambda s: Counter(s).most_common(1)[0][0]),
        l3=("category_l3", lambda s: Counter(s).most_common(1)[0][0]), rows=("spend", "size")).reset_index()
g = g.sort_values("spend", ascending=False).head(n).reset_index(drop=True)
system = build_system(tax, cfg.industry_hint)
lines = [unit_line(i + 1, r.sup, "", r.text) for i, r in g.iterrows()]
user = build_user(lines)
prov = make_gemini("GEMINI_API_KEY_DEV")
res = prov.complete_json(system, user, RESPONSE_SCHEMA, model, max_tokens=8192, thinking=level, timeout=180)
print(f"model={model} thinking={level} units={n} latency={res.latency_s}s usage={res.usage} err={res.error_kind}:{(res.error or '')[:150]}")
if not res.parsed:
    print("RAW:", res.raw_text[:300]); sys.exit(1)
rows = res.parsed.get("r", [])
valid, ok = 0, {1: 0, 2: 0, 3: 0}
seen = set()
for r in rows:
    if len(r) == 4 and 1 <= r[0] <= n and 1 <= r[1] <= len(tax) and r[0] not in seen:
        valid += 1; seen.add(r[0])
        pred = tax.leaves[r[1] - 1].path
        row = g.iloc[r[0] - 1]
        truth = (row.l1, row.l2, row.l3)
        for k in (1, 2, 3):
            ok[k] += int(tuple(pred[:k]) == truth[:k])
sp = g.spend.abs().sum()
print(f"valid entries {valid}/{n} (raw {len(rows)}); acc L1={ok[1]/n:.2%} L2={ok[2]/n:.2%} L3={ok[3]/n:.2%}")
# spend-weighted L3
w = 0.0
for r in rows:
    if len(r) == 4 and 1 <= r[0] <= n and 1 <= r[1] <= len(tax):
        row = g.iloc[r[0] - 1]
        w += abs(row.spend) * (tuple(tax.leaves[r[1] - 1].path) == (row.l1, row.l2, row.l3))
print(f"spend-weighted L3 acc {w/sp:.2%}; prompt chars sys={len(system)} user={len(user)}")
