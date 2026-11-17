# EVAL_LOG (numbers only; dates in IST)

## 2026-11-17 zero-label A (labels hidden), 300 units = top-150 by spend + 150 random tail, dev key
Clean variant: all 300 answered by gemini-3.5-flash-lite (primary) and gemini-3.1-flash-lite (second), prompt v2 with compiled definitions (definitions compiled by 3.5-flash-lite: strong Flash quota was exhausted on the dev project; arbiter fell back to 3.1-flash-lite, so the "arbiter" row is NOT a strong-model arbiter).

| metric | L1 | L2 | L3 |
|---|---|---|---|
| 3.5-lite, count-weighted (300 units) | 76.7% | 58.3% | 45.0% |
| 3.5-lite, spend-weighted | 92.1% | 88.2% | 76.2% |
| 3.1-lite, spend-weighted | 92.1% | 89.0% | 77.0% |
| top-150 units only, 3.5-lite (count) | 85.3% | 80.0% | 66.7% |
| agreed subset (70% of units), spend-weighted | 93.9% | 90.7% | 82.0% |
| disagreed subset (30%), spend-weighted | 71.4% | 60.4% | 9.9% |
| 2-model + 3.1 arbiter, spend-weighted | 92.6% | 88.8% | 77.6% |

No-definitions variant (240 of 300 units answered by 3.5-lite, 60 by 3.1-lite fallback after 503s; Gemma answered the whole second opinion): L3 spend-weighted 77.7% on the 240 pure units; no measurable difference from the definitions variant (n=300, noise ~5 pts). Tokens: ~2.9k in / ~0.6k out per 60-unit call; 503 "high demand" on 3.8/4-Hydrogen/3.6/Gemma were frequent.
Unit spend here is the P0 stub-normalised spend (net, ZAR) for A.

## 2026-11-17 zero-label B (seed labels hidden), 300 units = top-150 by true USD spend + 150 random among seeded described units, dev key
Only described, included seed lines are eligible (seeds contain no blank-description lines, so blank-description accuracy is NOT measured here). Spend = P1 USD spend of the whole unit.
Clean variant (all 300 answered by 3.5-flash-lite primary / 3.1-flash-lite second; definitions compiled by 3.5-flash-lite):

| metric | L1 | L2 | L3 |
|---|---|---|---|
| 3.5-lite, count-weighted | 96.3% | 94.7% | 90.0% |
| 3.5-lite, spend-weighted | 99.8% | 97.0% | 96.9% |
| 3.1-lite, spend-weighted | 99.7% | 96.9% | 96.6% |
| top-150 units only (count) | 99.3% | 97.3% | 97.3% |
| agreed subset (96% of units), spend-weighted | 100% | 97.2% | 97.2% |

Models agree on 96% of units here (vs 70% on A). nodefs variant contaminated by fallbacks (Groq/Gemma answered the second opinion), not comparable.

## 2026-11-17 first full LLM runs (dev key, supervised + LLM cascade), timing/usage
- B full (41,028 lines, 15,226 units, 10,187 sent to the LLM): 260 requests, 880k tokens in / 151k out, 18.9 min wall, validated. 36% of spend fell to confidence-0.05 fallback before the blank-unit tiers were added (fixed after).
- A full (65,549 rows, 9,471 units, 850 to the LLM): 30 requests, 89k in / 17k out, validated.
- End-to-end A2-sim (18% id-gap-run label holdout, no LLM): L1 99.87% / L2 99.71% / L3 99.71% (spend-weighted L3 99.51%); reliability: conf>=0.95 -> 99.9% accurate (6,812 of 6,856 rows).

## 2026-11-17 B end-to-end with real LLM (dev key), 18% id-gap-run seed holdout (487 included rows, described seed lines only)
All: L1 99.15% / L3 98.83% spend-weighted (count 95.9% / 93.4%). Supervised-answered (96.9% of spend): L3 99.69%. LLM-answered (2.3% of spend, 78 rows): L1 72.0%, L3 69.2%. Reliability (L3): conf>=0.95 -> 99.7% (353 rows); 0.85-0.95 -> 97.9% (48); 0.7-0.85 -> 65.9% (85, overconfident).

## 2026-11-17 dress rehearsal, B treated as unseen (profile -> unedited draft config -> run)
profile 3.3 s, run 5.7 s (no LLM), validation OK on the first try. Untouched draft vs hand config: 1,068 vs 2,485 excluded lines and USD 338.5M vs 329.3M (+2.8%): the missing intercompany/tax exclusion rules are the human step that matters; everything else (dates, FX table, ids, columns) was inferred correctly.

## 2026-11-17 blind-day quota budget (estimate from measured runs; empty cache assumed)
Measured B full run (41k lines): 260 requests, 880k tokens in / 151k out, of which Flash-Lite 3.5 = 230 req (46% of its 500/day), Flash-Lite 3.1 = 19, Gemma = 9. Peak throughput ~0.23 req/s, far under 15 RPM/250k TPM limits.
Estimate for one key: B2 ~260 + C (50k lines, ~15-20k units) ~300-400 + A2 ~30 => ~600-700 requests against 500 (3.5-lite) + 500 (3.1-lite) = 1,000/day pooled => 60-70%, but 3.5-lite alone would exceed 100%. Policy: run C first (50 points), then B2, then A2; the router fails over 3.5 -> 3.1 automatically when the ledger cap is hit; cap new-model requests at 70% of RPD per entry in config; strong Flash models (20/day) reserved for taxonomy compile + arbitration only.

## 2026-11-17 blank-description accuracy (description blanked on labelled units, all other context kept; labels of an id-gap holdout hidden)
Label-trained mode, spend-weighted, before (supervised top-1 only) -> after (`vote_first`: supplier/context key votes weighted by spend, context-only model for the remainder):
| Company | blank units | L1 | L3 |
|---|---|---|---|
| A | 436 (6,692 rows) | 98.3% -> 99.98% | 57.7% -> 82.6% |
| B | 193 (498 rows) | 94.4% -> 95.4% | 72.7% -> 76.9% |
Votes alone cover 95.6% of A units and only 48.7% of B (opaque vendor ids), where the context-only model fills the rest (B ctx_model share: L3 41.6%). Merged as the default (`classification.blank`, A and B). C-like mode (no labels; LLM-labelled described units as the training set) is NOT yet reliable: the interim run answered through local Ollama (L3 15-37% vs truth) and is being rerun on the dev key; do not quote it.

## 2026-11-17 blank tier, C-like mode (no labels; top-900 described units by spend LLM-labelled on dev key), spend-weighted
| | A | B |
|---|---|---|
| LLM labels vs truth, L3 (ceiling) | 36.6% | 76.7% |
| blank units L1 / L3 | 74.1 / 42.4 | 94.8 / 74.3 |
| blank units L2 | 54.5 | 92.7 |
| context-only model alone L1 / L3 | 74.0 / 26.9 | 94.9 / 46.4 |
Voting lifts L3 (A 26.9->42.4, B 46.4->74.3) at no L1 cost. A is bounded by its LLM-label quality (top-spend units are hardest), not the blank tier. Old context-only-LLM tier not re-measured in this mode.

## 2026-11-17 BLOCKER reconciliation: A no-label LLM L3 36.6% (blank fork, top 900) vs 76.2% (zero-label, top 150 + 150)
Different metrics, partly different models; no truth-join or cache-key defect found.
1. **36.6% was count-weighted over 900 units** (`np.mean(ok3)`); 76.2% is spend-weighted over 300 units. Same zero-label run, count-weighted: L3 45.0% (top-150 only: 66.7% count / 76.4% spend). The harness now reports both (`llm_label_L3_vs_truth_sw`) and a per-entry breakdown.
2. **Who answered (blank run).** Flash-Lite 3.5 had hit its 500/day cap, so 3.1-lite answered 840 units, Groq gpt-oss-120b 60; Gemma 4 returned 503 "high demand" on every call; no nulls. Zero-label used 3.5-lite for 240-300 units + 3.1-lite for 60.
3. **Re-scored from cache, spend-weighted, 900 units:** L1 83.7%, **L3 62.3%** (count 39.6%). By entry: 3.1-lite 840 units L3 66.9% sw / 41.8% count; **Groq 60 units L3 8.2% (8.3% count)**. Re-asking those 60 on 3.1-lite gives 30% count-weighted, so Groq (gpt-oss-120b) is a poor failover on this taxonomy; those are low-spend tail units.
4. **Prompt.** Both go through `LLMClassifier` with supplier, GL, context columns and text. The blank harness passes no compiled definitions; zero-label nodefs vs defs is 75.8 vs 76.2 spend-weighted, so definitions do not explain it. Cache key = sha256(provider|model|PROMPT_VERSION|system|user|thinking): no cross-variant contamination.
5. **Truth/leaf mapping.** Both score path prefixes against the per-unit majority leaf (A has 3 levels, L3 = leaf); not the cause.
Remaining gap 62.3 vs 76.2: 3.1-lite instead of 3.5-lite (~-1 to -5 points on the same 300 units), the 750 extra tail units (lower spend, harder opaque part descriptions), and Groq. **Not yet done:** re-score the 900 on Flash-Lite 3.5 with the standard prompt (3.5-lite dev quota is at its 500/day cap until Wed 13:30 IST).
Consequence: A's "C-like blank-unit L1/L3 74.1/42.4" in the table above used those weaker labels (spend-weighted LLM-label L3 62.3%, not 36.6%); treat A's blank-tier C-like numbers as a lower bound until re-run on 3.5-lite.
Two real defects found on the way, both fixed (D-44): a thread race that built duplicate Gemini clients ("client has been closed" failures), and Gemma 4 `503`s (failover only).
**Blocker resolved (cross-check by rank, A, LLM labels vs truth, from cache, no standard-prompt model change):** top-150 units L3 76.5% spend-weighted (68.0% count), units 150-300 54.0% / 53.3%, units 300-900 33.5% / 34.0%; all 900: 66.2% spend-weighted, 39.6% count. The zero-label top-150 (same task, same units) gave 76.9% / 66.7%, so the two harnesses agree where they overlap. The "76.2%" headline is a top-150-dominated sample; A's accuracy falls steeply with unit rank (lower-spend units are opaque part descriptions). Keep quoting A's no-label L3 as **~76% on the top-spend units, ~66% over the top 900, ~46% count-weighted**. Still to do on Wed after 13:30 IST: the same 900 on Flash-Lite 3.5 with the standard prompt.

## 2026-11-17 B demand baseline (holdout e2e, empty LLM cache, B seeds as labels, job budget off, dev key)
Resumed after one aborted pass, so counts add up: this pass 190 requests (3.1-lite 168, Gemma 19, Groq 3; 3.5-lite was capped) + ~65 in the aborted pass = **~255 requests** for 41k lines (15.2k units, 10.3k LLM candidates after supervised accepts 2.8k): matches the earlier 260 estimate, ~6.2 requests per 1,000 lines. Holdout (487 rows) L1/L3 spend-weighted 99.0/98.7; supervised-answered (96.9% of spend) L3 99.7; LLM-answered (3.1% of spend) L3 64.9-80.3%. Memory/self-training demand reduction (target B2 + A2 <= 150) is NOT measured yet: needs `scripts/b2_sim.py` (about 250 + ~80 requests) after the Wed 13:30 IST reset.

## 2026-11-17 confidence calibration (D-48), out-of-sample (leave-one-run-out; runs: A holdout, B seeds, B holdout with LLM, zero-label A and B LLM units)
Supervised tier: ECE 0.0187 -> 0.0176; bands after the map track accuracy (0.85-0.95 band 94.7% right, >=0.95 band 99.9%). **Shipped** as `configs/_calibration.json` (supervised only, knots, no data).
LLM tier: raw bands are overconfident (0.85-0.95 right 30%; 0.7-0.85 right 69%) but the miscalibration is company-specific (A vs B), so a pooled map made held-out ECE worse (0.349 -> 0.362). **Not shipped**; the LLM tier keeps raw confidence plus the failover caps (D-47). Interpret LLM-tier confidence as a ranking, not a probability.

## 2026-11-17 budget -> accuracy curves (cache replay, holdout e2e; router counts every pool answer against the budget in its own priority order; past it the supervised/propagation/blank/offline tiers answer)
B (seeds as labels, 487 held-out rows, 80 of them LLM candidates), spend-weighted L1 / L3 and count-weighted L3:
| LLM budget (requests) | L1_sw | L3_sw | L3_count |
|---|---|---|---|
| 0 | 98.43 | 97.79 | 90.14 |
| 25 | 99.04 | 98.69 | 91.58 |
| 50 | 99.02 | 98.70 | 91.79 |
| 100 | 99.03 | 98.72 | 93.43 |
| 140 | 99.03 | 98.73 | 94.25 |
| unlimited (~255) | 99.03 | 98.73 | 94.05 |
Knee at ~25 requests for spend-weighted accuracy (+0.9 pt); count-weighted keeps improving to ~100. A2-sim (labels_A, 6,856 held-out rows, only 561 LLM-candidate units): budget 0 -> L3_sw 99.51; 5 / 10 / unlimited (10 requests) -> 99.43; the 15 LLM-answered rows are 0.4% right (spend share 0.12%): the curve is flat and the LLM slightly hurts. Defaults set from this (D-50): A2 max 10, B2 max 60, C the rest (630 on an empty ledger).

## 2026-11-17 Groq / Gemma root cause (from cached responses, no new calls)
Groq gpt-oss-120b, two 60-unit batches: L3 23% and 8% (14/60, 5/60); answers collapse onto 6 and 3 distinct leaves (Flash-Lite 16-21), 34/60 and 50/60 on one catch-all leaf; agreement with Flash-Lite 23% / 7%; self-confidence 81 / 69 (Flash-Lite 89-91). Prompt ~1.4k tokens, input 4.1k, outputs complete and in range: not truncation, not id mismatch, not JSON repair; id shifts of +-1 explain 12/60 and 10/60 at most, no better than chance. Gemma 4: 78 requests, 22 ok on the dev day, every failure `503 UNAVAILABLE ... high demand`; the same model id answers small calls, so it is capacity, not config. Chain before/after: D-49.

## 2026-11-17 LLM-tier confidence mapping (minimax across A and B; 300 A units, 380 B units; raw ECE A 0.364, B 0.100)
Grid over three- and four-piece constant maps and a cap: best is conf < 0.6 -> 0.2, else 0.65 (the four-piece search collapses to the same map): worst-company ECE 0.1027 (A 0.1027, B 0.0974) vs raw worst 0.364; beats raw on both. A pure cap (best c = 0.625) does not (A 0.1445, B 0.1224 vs 0.100 raw on B). Shipped in `configs/_calibration.json` (llm group). Cost: no ranking resolution above 0.6.

## 2026-11-17 error analysis, A, 50 costliest wrong units of the zero-label sample (counts only)
By level: wrong L1 18, right L1 wrong L2 13, right L2 wrong L3 19. By cause (manual): missing context 22 (a generic two-word phrase or a bare part name with no machine reference; only supplier or context could decide), defensible-but-different 23 (leaf-granularity conventions: a generic leaf named like its parent vs a specific one, contract-structure leaves, overhaul/repair variants, near-synonym leaves), clear model error 5 (explicit text ignored). Conclusion: 44% of the costly errors need the supplier's other lines as context (supports the supplier-coherence hint, item 5b); 46% are convention differences a prompt rule cannot resolve without labels (the specific-vs-generic choice goes both ways in the data), so no blanket prompt rule.

## 2026-11-17 B2 demand by cache replay (no quota): date split of the finished full B run, first 75% of posting dates = build (28,893 rows, 11,482 units), last 25% = sim (9,632 rows, 4,657 units)
- Units of the sim part that needed the LLM in the full run: 3,108; already in the build part's LLM-answered memory (exact 209, loose 1): **6.8% of units, 16.8% of their spend**. B's units rarely recur across time, so memory lookup alone trims B2 demand by about 5%: requests for the sim part ~57 without memory, ~54 with it (estimate: 60 units per request + second opinion on the top 300). The whole file cost ~255.
- Self-training lever (supervised model also fed the build part's own confident answers, LLM rows weighted 0.3 x conf): accepts 1,480 of the 3,108 (47.6%) at p >= 0.8, agreeing 69.8% with the LLM labels of the full run; requests ~57 -> ~31. Accuracy against truth is NOT established (only 1 held-out seed row falls in range), so under the measured-before/after rule the lever is **off by default** (`classification.memory.train: false`); memory lookup stays on (it returns the same answers the cache would).
- Reading for blind day: a B2 about a quarter of B's size needs ~55 requests, inside the B2 cap of 60; a B2 as large as B needs ~255 and the cap (60, knee ~25 on the curve) decides.

## 2026-11-17 LLM confidence tiebreak and A2 budget
Map now output = band value + 0.05 x (raw - band mid), clipped [0.05, 0.95], monotone in raw within each band. Worst-company ECE 0.117 (A 0.109, B 0.117) <= 0.12. A2 LLM budget set to 0 (curve above); B2 max 60 reserve 10; C reserve 60 (640 on an empty ledger).

## 2026-11-17 outage path measurement (A zero-label top-300 sample, spend-weighted)
Offline fallback alone (n=300): L1 75.9, L2 31.0, L3 25.4. Local qwen3:4b (first 60 units): L1 94.1, L2 45.3, L3 41.7 vs offline on the same 60: 81.4 / 34.4 / 28.1. Ollama 1.24 s/unit (74.5 s for 60), so about 970 units fit in the 20-minute cap. Policy shipped (D-51): Ollama for top-spend units within the cap, offline for the rest.
Takeover-session follow-up (2026-11-18): the requested full top-300 Ollama rerun could not be performed. The installed `ollama` binary aborts during MLX/Metal startup (no Metal device available), and the workspace sandbox denies localhost access to the local API. Thus only the existing same-60 comparison is measured; D-51 remains supported by that sample, while full-300 L1/L2/L3 and seconds/unit remain unmeasured here.

## 2026-11-18 blind-day timing dress run (mock LLM sleeping 8 s per call at real RPM limits, budgets C 640 / B2 60 / A2 0; synthetic C 50,000 lines, real-size B 41,028 and A 65,549 rows; sequential C -> B2 -> A2)
| job | rows | wall-clock 1x | wall-clock 3x slowed mock | validation |
|---|---|---|---|---|
| C | 50,000 | 23.8 min | 35.0 min | OK |
| B2 | 41,028 | 3.0 min | 4.7 min | OK |
| A2 | 65,549 | 0.3 min | 0.3 min | OK |
| total | | 27.1 min | 40.1 min | all OK |
Target (all validated within 60 min of starting C) met at 1x and at 3x. Deadline guard: with `deadline_minutes` forced to 6 and the mock slowed 3x, C stopped the LLM after 2,880 units, filled the tail with lower tiers and wrote a valid 50,000-row file in 6.1 min. C is the long pole (rate-limited LLM calls); B2 and A2 are compute-bound.

## 2026-11-18 real throughput correction (full B LLM run; 260 requests in 18.9 min)
Effective end-to-end throughput = 260 / 18.9 = **13.76 requests/min** (0.229 requests/s; 4.36 wall minutes per 60 requests). Applying this measured rate to C's 640-request budget gives **46.5 min** for C's LLM work, above the 40-minute threshold; this replaces the 23.8-min 1x mock dress-run estimate. Adding dress-run B2 (3.0 min) and A2 (0.3 min) gives about **49.8 min sequential**, within the 60-minute target with ~10 min margin. This is an effective batch throughput, not isolated model latency. Live per-call latency and batch size are now collected to run-local `llm_call_metrics.csv`.

## 2026-11-18 D-54 two-model timing projection (mock, no API)
The classifier already submits batches concurrently with four workers. Router test load with the mock confirmed both pool members can serve simultaneously, each under its own RPM limiter, for at least 1.7x single-member throughput. Applying the measured one-model effective rate (13.76 requests/min) to 640 requests: **pool OFF 46.5 min; pool ON 23.3 min** for C's LLM budget (two equal-capacity 15-RPM models). Pipeline compute and request mix are unchanged, so the prior 3.3 min B2/A2 estimate implies **49.8 min OFF vs 26.6 min ON** total. This is a rate-based mock projection; not a full 50k-row end-to-end dress run. Config default remains OFF pending T4a accuracy.
