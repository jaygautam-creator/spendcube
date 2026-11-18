# EVAL_LOG (numbers only; dates in IST)

## 2026-11-18 final architecture-vs-brief round (G1-G6, aggregate)
- G1 A2 date split: combined q75 date 2023-11-01; 48,664 A2 rows, 16,885 history rows. Compared with full A on shared IDs: inclusion/exclusion agreement 100% in all four outcome codes; included spend delta $0.00. Tool duplicate recall: history twin 77/77; A2 ERP twin 14,243/14,243; re-export/no ERP twin 1,879/1,879. ERP A2 entity pairwise and item scores were 1.0; L1/L3 spend-weighted were 1.0 because this exact A config resolves labels directly on the scored rows. These classification/entity numbers are label-leaked and are not valid blind-accuracy evidence.
- G2 unsupervised A dedup: supplier without catch-all pairwise F1 1.000, B-cubed F1 1.000; item without catch-all pairwise F1 0.994, B-cubed F1 0.985. Including supplier catch-all reduced pairwise F1 to 0.182 (B-cubed 0.643); dominant error: false splits. Synthetic supplier baseline pairwise F1 0.909/B-cubed 0.933; vendor threshold 50→75 produced pairwise F1 0.952/B-cubed 0.968 with no false splits, but does not fix catch-all fragmentation. Threshold change proposed only, not applied.
- G3 prompt estimate, 4-level ragged tree, 60-unit batch, ~100-character definitions/leaf (chars/3 router estimator): 150 leaves = 4,457 tokens/request without definitions (26.7% of one 250k TPM at 15 RPM), 9,471 with definitions (56.8%); 400 = 6,974 (41.8%) / 20,404 (122.4%); 900 = 12,007 (72.0%) / 42,271 (253.6%). Two independent 15-RPM models each fit TPM without definitions at 900 leaves (72% each); definitions are skipped above 80%. Synthetic 50k mock run with configured 15 RPM/250k TPM (limiter waiting bypassed only for mock): 50,000 rows validated in 46.0 s (44.9 s classify), exact taxonomy paths for all rows; mock selected L3-stop leaves, with L4 blank for all rows. Run report: 834 requests, 6,595,216 input tokens; per-request estimate 10,905 without definitions and 40,905 with, so definitions skipped; batch remained 60. Live rate-bound estimate at 60 units/request and 30 aggregate RPM is 27.8 min, excluding arbitration and software overhead.
- G4 B exact signature: 2,004 groups, 5,708 later copies, $13,778,429.33 repeat spend; B seeds retain these as real included spend. Dedup stays OFF; the profiler always recommends OFF unless the brief says the file was exported twice, falling back to vendor_id when supplier_name is unmapped. G5 B remains 2,487 suggested rows across 7 rules after restricting JIB to billing/invoice/charge/statement phrases.
- G6 blind-day order B2 → A2 → C: B2 top 100 gives ~25 disagreements / 1 arbiter request; A2 top 100 is inert under zero request budget; C top 300 gives ~75 disagreements / 2 requests. Total ~3 3.8-flash requests, within 20 RPD, with Hydrogen/3.6 fallbacks; B measured +0.84 spend-weighted L3 on top-100 disagreements.

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
The classifier already submits batches concurrently with four workers. Router test load with the mock confirmed both pool members can serve simultaneously, each under its own RPM limiter, for at least 1.7x single-member throughput. Applying the measured one-model effective rate (13.76 requests/min) to 640 requests: **pool OFF 46.5 min; pool ON 23.3 min** for C's LLM budget (two equal-capacity 15-RPM models). Pipeline compute and request mix are unchanged, so the prior 3.3 min B2/A2 estimate implies **49.8 min OFF vs 26.6 min ON** total. This is a rate-based mock projection; not a full 50k-row end-to-end dress run. T4a supports enabling the pool by default.

## 2026-11-18 D-56 blind-day mock rehearsal
Ran the human sequence using the generated profiler config, temporary 25% A/B slices, and the synthetic C fixture: `blind_day.sh ab runs/rehearsal`, then `blind_day.sh c runs/rehearsal 2026-11-18T02:20`, then summary; `OUT=runs/rehearsal/out MODE=dev SPENDCUBE_FORCE_MOCK=1`. B2 (10,257 rows), A2 (16,387 rows), and C (3,000 rows) all validated; summary = **GO**. C classification took 122.2 s; full B2/A2/C sequence was about 2m50s. Deadline remained unhit because C finished before 02:20. The rehearsal found and fixed summary report matching for suffixed A2/B2 outputs and added the `OUT` override so rehearsal outputs stay outside `submission/`.

## 2026-11-18 T4a — same 900 A units, prompt v2, no definitions
The top-900 described, labelled A units were scored independently by 3.5-lite and 3.1-lite; spend rank is from the same sample. Values are L1/L2/L3 (spend-weighted; count-weighted):

| Spend rank | 3.5-lite SW | 3.1-lite SW | 3.5-lite count | 3.1-lite count |
|---|---:|---:|---:|---:|
| 1–900 | 86.18 / 75.37 / 66.32 | 86.88 / 78.04 / 66.44 | 74.11 / 55.89 / 45.56 | 72.89 / 57.00 / 42.78 |
| 1–150 | 92.40 / 84.72 / 76.68 | 93.49 / 87.90 / 77.43 | 86.67 / 74.67 / 68.00 | 88.00 / 79.33 / 68.67 |
| 151–300 | 74.25 / 58.99 / 49.46 | 75.15 / 59.89 / 49.68 | 74.00 / 58.00 / 48.67 | 75.33 / 59.33 / 48.67 |
| 301–900 | 70.15 / 50.14 / 37.38 | 69.14 / 52.09 / 34.93 | 71.00 / 50.67 / 39.17 | 68.50 / 50.83 / 34.83 |

3.1 is 0.12 points above 3.5 on overall spend-weighted L3 and 0.75 points above it on the top 150, meeting the within-3-point rule in both required comparisons. D-59 enables the pool by default. Ledger delta for this T4a attempt sequence (including interrupted/retried attempts): 3.5-lite 28 requests (14 successful, 14 errors incl. one 429); 3.1-lite 30 (22 successful, 8 errors). The evaluator's default arbiter leg initially called 3.8; repeated 503 disabled it. Those arbitration answers are excluded from the model-only comparison. Other attempted entries: 3.8 11 (6 successful, 5 errors), 3.6 5 errors, Hydrogen 5 errors. No model exceeded its daily guardrail (3.5 <400, 3.1 <500).

## 2026-11-18 T4b — supplier-rich A sample
Selected all 4,692 units across the 50 highest-spend suppliers (93.7% of described-unit spend). 3.5-lite and 3.1-lite each answered 3,720; repeated 503s left 972 unanswered, scored as misses. Lite disagreement set: 1,705 units. The 3.8 arbiter answered 280 before its daily cap; 3.8 calls were capped by the ledger after T4a had used 11 of 20 requests. Single Lite L3 SW/count was 64.23/24.96%; after available 3.8 arbitration it was 65.82/25.45% (+1.59/+0.49 points), with L1 SW 85.93%→87.27%. On top-100 spend units (20 Lite disagreements), L3 SW/count moved 78.82/72.00%→79.66/73.00% (+0.84/+1.00), and L1 SW did not fall (93.86%→94.56%): retain arbitration for the top 100 in C. Supplier coherence found 98 candidates and re-asked all 98 on 3.1-lite; 96 changed. It moved L3 SW/count 65.82/25.45%→65.49/26.98% (-0.33/+1.53), while L1 SW rose 87.27%→87.51%; keep coherence OFF because spend-weighted L3 fell.

The 3.8 ledger reached 21 requests (one over its configured 20 RPD; concurrent requests raced the non-atomic local RPD check). T4b therefore has limited 3.8 coverage and is not a full-sample arbitration estimate. T4a+T4b ledger totals: 3.5-lite 100, 3.1-lite 93, 3.8 21, 3.6 5, Hydrogen 5. No Lite threshold was crossed (3.5 <400, 3.1 <500). `Router` now reserves per-entry RPD atomically across worker threads; `tests/test_parallel_pool.py` covers the concurrent cap.

## 2026-11-18 T4c — prompt freeze
T4b changed no prompt wording. `PROMPT_VERSION` remains **v2** for the final A/B outputs (D-61).

## 2026-11-18 T4d — final A/B outputs, PROMPT_VERSION v2
Both outputs validated with original row counts: A 65,549, B 41,028. A reused cached LLM answers for its 850 candidate units; B ran with its 60-request budget and the enabled pool. The previously tracked files were the P0 placeholder outputs (every row included, one leaf, confidence 0.01) and had never been regenerated, so nearly every row differs; this is the first real tracked output. Check against measured totals: B included spend 329.25M USD equals the P1 total minus exclusions (349.1M - 19.86M), with 2,485 NON_PROCUREMENT rows as hand-tuned; A excludes 21,798 DUPLICATE, 315 NOT_APPROVED, 188 CANCELLED rows (included spend 3.00B ZAR).

Ledger after T4d (Pacific 2026-11-18): 3.5-lite 110 requests, 3.1-lite 110, 3.8-flash 21, 3.6-flash 8, Hydrogen 9. Lite caps remain below stop levels (400/500). Blind key was not used.
# X1 spend concentration (2026-11-18)

Computed offline from normalized, described, non-excluded zero-label units with positive absolute included spend. The supplier key is `supplier_core`, falling back to `vendor_id` when absent (B has no `supplier_core`). A: 16,256 units, $4,659,829,135.96 included spend, 199 suppliers, median 17 units/supplier. B: 12,899 units, $214,870,878.47 included spend, 2,134 suppliers, median 1 unit/supplier.

| Company | Top 50 | Top 100 | Top 200 | Top 500 | Top 1,000 | Top 2,000 |
|---|---:|---:|---:|---:|---:|---:|
| A | 30.56% | 40.74% | 51.52% | 65.78% | 76.16% | 85.87% |
| B | 40.41% | 51.67% | 63.46% | 78.43% | 88.23% | 95.59% |

Prior 3.8-flash bakeoff reference: T4b's 50-supplier sample had lite-only spend-weighted L3 64.23%, rising to 65.82% (+1.59 points) after available 3.8 arbitration; L1 rose 85.93% to 87.27%. On the top-100 disagreement subset, spend-weighted L3 rose 78.82% to 79.66% (+0.84), with L1 rising 93.86% to 94.56%. Coverage was limited: 280 of 1,705 disagreements arbitrated after 3.8 had already spent 11 of 20 daily requests during T4a; this is not a full-sample estimate. Supplier coherence lowered spend-weighted L3 0.33 points and remains off.

## X2 head-accuracy grid (partial; 2026-11-18)

Live dev-key experiment used the same top-150 labeled-unit sample per company in zero-label mode; labels were used only after inference for scoring. Network DNS resolution failed during the Hydrogen arm, so no B arms or complete strong-model comparisons could be obtained. A's completed 3.5-lite arms:

| Arm | L1 count / spend | L2 count / spend | L3 count / spend | Requests | Tokens/request | Seconds/request | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3.5 minimal, batch 60 | 86.00% / 92.18% | 74.67% / 84.70% | 68.00% / 76.69% | 1 (2 cache hits) | 3,091 | 6.73 | 150/150 |
| 3.5 low, batch 60 | 86.00% / 92.00% | 78.67% / 87.53% | 68.67% / 77.95% | 3 | 4,355 | 5.87 | 150/150 |
| 3.5 minimal, batch 15 | 85.33% / 91.62% | 79.33% / 87.49% | 71.33% / 78.73% | 10 | 2,002 | 5.16 | 150/150 |

The Hydrogen arm stalled on DNS and was interrupted after the dev ledger recorded 5 failed request attempts (daily total 14/20, 6 remaining). 3.6 was not attempted. No requests used the blind key. These A-only results are insufficient to choose a head routing tier or to enable X3; no config default changed. The A small-batch spend-weighted L3 gain over baseline is +2.04 points, but the required B non-regression check is missing.

## X2 completed + X4 live rehearsal (2026-11-18, run by the architect session; Codex sandbox had no network)
X2 B top-150 zero-label (spend-weighted L3): 3.5-lite batch 60 minimal **95.75%** (current), batch 60 thinking low 88.74%,
batch 15 minimal 74.40% (batch 15 low: rate-limited, discarded). A's +2.0 batch-15 gain does not hold on B: keep batch 60 / minimal.
Strong models, top-150: gemini-3.6-flash A L3_sw 78.87% vs lite 76.69% (+2.2, one sample); on B only 80/150 answered within
its 5 RPM. Hydrogen: repeated server failures (0/150). Decision D-64: no head tier (gain within sample noise, 20 RPD / 5 RPM
capacity, late risk); 3.8 arbitration on top-100 disagreements stays.

X4 live rehearsal, B treated as an unseen company (lines + tree only; fresh LLM cache and unit memory; dev key; pool ON):
profile 3 s; config edits ~3 min (industry hint, PO number column, llm_context, cache paths); `blind_day.sh c` 613 s wall
(classify 612 s), 229 requests, 41,028 rows, validator OK, deadline not hit, GO. Exclusions 2,490 rows / 19.87M USD vs
hand-tuned 2,485 / 19.86M. Scored afterwards against the 2,550 included seed rows: L1/L2/L3 spend-weighted
**98.55 / 93.34 / 92.48%**, count-weighted 94.59 / 93.96 / 88.51%; include agreement 99.97%.
