# DECISIONS.md

What we found in the data, the ambiguities in the brief, and what we decided. Facts: `DATA_FINDINGS.md` [F§n]. Design: `MASTER_ARCHITECTURE.md`.
Status legend: **Decided** · **Provisional** (decide by measurement during build) · **Open**.
Implementation must keep this file current. Any change to a decision gets a dated line under it.

---

## A. Interpretation of the brief and evaluation

**D-00 Reading of scoring.** Supplier/item keys are scored on *grouping*, not names. Classification on C is per level (L1–L4), weighted by spend. B2 categories are scored only on "clear-cut lines". B2 is not scored on supplier/item dedup. *Decided.* Consequences:
- C classification accuracy on high-spend lines is the single most valuable thing (§8.4 of the architecture).
- B seeds have no blank-description lines [F§6], so "clear-cut" very likely means described lines. Blank lines still get a best-effort category via propagation.

**D-01 Golden supplier catch-all (`<catch-all supplier key>`) and item catch-all (`<catch-all item key>`).** The client's golden supplier master lumps 122 real suppliers into one key, and the item master lumps 192 materials [F§3]. A2 is scored against these golden groupings. *Decided:* for A, learn alias maps from `labels_A.csv` (supplier core → key, vendor_id → key, material → key, text → key) and emit the golden keys where they resolve. Unseen names/materials get our own keys; we do not guess the catch-all for unseen names. Why: scoring is against the client's truth, and labels are explicitly given "to build and test your matching, de-duplication". For C (no labels) we produce real-world groupings. Alternative rejected: pure name clustering on A, which would split the catch-all into ~120 groups and lose most pairwise recall on A2.
  - **2026-11-17 (P3):** on an 18% run-structured holdout, supplier pairwise F1 with alias maps = 0.9985 (1.000 without the catch-all bucket); item pairwise F1 = 0.997 without the item catch-all, 0.964 with it. The gap is 25 holdout rows whose catch-all materials never appear in the other 82%. Of holdout rows with an unseen non-blank material only 7% (26/364) belong to the catch-all, so guessing it would be wrong 93% of the time: we keep not guessing. Alias-off baseline on the same holdout: supplier F1 0.185 with the catch-all (1.0 without), item F1 0.568 (0.993 without).

**D-02 History-aware matching for A2.** 20% of A's tool rows reference POs that are absent from A's ERP file [F§1]. Invoice↔receipt pairs were split across A/A2. *Decided:* when running A2, A's original files are loaded as `reference` sources. A2 tool rows are matched against ERP(A ∪ A2), and re-exports are detected across tool(A ∪ A2). Reference rows are never emitted. A2 tool rows matching an A record become `DUPLICATE`. Why: "count once" is a company-level property; the truth is generated at company level (pairs cross the split); and if no cross-split pairs exist the feature is a no-op. Risk: if the truth were built per-split, these rows would be scored wrong. The count of `history` matches is logged so the effect is visible. Config switch: `reconciliation.history.enabled`.
  - **2026-11-17 (P2):** implemented as `reconciliation.history` (`enabled: auto`): in an A2 run the stage loads A's original files from `{kit_dir}` as reference rows and never emits them; in a build run it detects the history equals the run's own files and turns itself off. Pseudo-A2 test on real A (18% of POs held out) agreed with the full-data run on 99.99% of 9,343 tool rows.

**D-03 Exclusion precedence.** CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE. A cancelled tool invoice that matches an ERP receipt is labelled CANCELLED, not DUPLICATE. Its ERP receipt still counts: goods were received, and ERP has no status. *Decided.* Why: status is intrinsic to the record, and DUPLICATE presupposes a record that would otherwise be counted. Affects ~50 A rows.

**D-03b Excluded rows** carry `supplier_key` and `item_key` (as in the template's DUPLICATE row), spend 0, blank l1–l4, and `confidence` = confidence of the exclusion decision. *Decided.*

**D-15 Always predict down to a leaf.** The brief says "to its lowest level". Per-level scoring gives partial credit at higher levels, so abstaining at a parent never helps. *Decided.*

## B. Amounts, tax, currency, dates

**D-04 VAT rate parsing (A tool).** Accept `15`, `15%`, `0.15`. A value >1 is a percentage. Blank → 15% (rule). **`0` → 0% (zero-rated), not "blank"**: with net=gross, 1,353 of 1,580 such rows match an ERP amount exactly; treating them as 15% would match only 66 [F§4]. *Decided.*

**D-05 Negative and placeholder amounts.** Negative ERP values (434, returns) and negative tool lines (credit notes) are included as negative spend; they reduce totals. ZAR 0.01 placeholder lines (509) are included at face value. *Decided.* Why: net spend should reflect returns, and there's no rule to exclude them. Open risk: the scorer might treat credit notes as CANCELLED; we accept that risk.

**D-06 B line-rate plausibility.** The brief says to use `currency_rate`. But 64 EUR/CHF/ARS lines of JDE entity CO180 carry the entity's MXN rate (~18.6 per USD), which would undervalue genuine EUR amounts ~20× [F§5]. *Decided:* use the line rate when `line_rate / table_rate(currency, month)` is within [0.5, 2.0]. Otherwise use the FX table for the invoice month, flagged `FX_RATE_IMPLAUSIBLE`. Every other line rate sits within 0.95–1.03 of the table, so the band only catches gross errors.

**D-07 USD lines with a rate.** 1,279 USD lines carry an MXN-level rate. Rule: "USD lines are already in USD". *Decided:* ignore the rate for USD lines. Generalized: lines in the reporting currency always use rate 1.

**D-08 FX month selection and gaps.** For a non-USD line without a usable rate, use the invoice-date month. If the invoice date is impossible or that month is missing from the table, try the posting-date month, then the nearest available month for the currency (e.g. CAD 2022-05…07, AED gaps). If still nothing: `static_rates` in config, else spend 0 with flag `FX_MISSING`. *Decided.*

**D-09 Impossible dates.** B invoice dates in 2018/2020/2027 against 2024 postings. Dates outside [median_year − 3, median_year + 1] of the column are flagged `DATE_IMPOSSIBLE` and treated as missing for FX and matching. The record is still counted. Tool dates are 65% blank; matching never requires dates. *Decided.*
  - **2026-11-17 (P1):** window is `[year(q02)-1, year(q98)+1]` of each date column (config `valid_years` overrides), not median-3/+1, because A's genuine 2021-2024 spread sits on the edge of a median window. B: flags 26 invoice dates, no posting dates.

**D-10 Day-first.** A's `nn/nn/yyyy` dates are dd/mm (2,050 values have day >12; none have month >12). Auto-detection per column uses the same evidence test; config can force it. *Decided.*

**D-11 Blank/unparseable amounts** (3 B rows, currency `RCN`, blank amount) → spend 0, included, flagged `AMOUNT_BLANK`. *Decided.*

## C. What counts as spend

**D-12 B intercompany.** GL descriptions matching `IC\d+`, `I/C Payable`, `IC Payables` → NON_PROCUREMENT. Vendor propagation also marks every row of a vendor whose rows are ≥90% intercompany (75 IC vendors; 2,039/2,040 rows are already on IC GLs [F§5]). *Decided.* Implemented as config rules plus generic learned rules from seeds, not code.

**D-13 Tax lines.** Lines whose *description* is a tax posting (state/local, sales, GST, accrued sales-and-use tax posting labels) → NON_PROCUREMENT. Lines that merely mention tax but buy a service (service lines that mention tax, e.g. a tax consultant, a blanket PO for tax work, a payroll-garnishment service) stay included. *Decided.* The anchored regex lives in the B config; the LLM `spend_class` covers C.
  - **2026-11-17 (P1):** learned exclusion rules use only gl_desc/gl_account/vendor_id by default; context columns (plant, company code, source system) are opt-in: enabling them lifted B exclusions from 2,485 rows (USD 19.9m) to 12,698 (USD 76.2m) because seeds are non-uniformly sampled. Default measured against seeds: precision 1.0, recall 1.0 (450/450). B excluded spend USD 19.86m matches DATA_FINDINGS.

**D-14 B duplicates, reversals and recurring invoices.** Seeds keep reversal pairs and zero-amount lines as included. Recurring monthly invoices (doc type PR) are real. 1,066 exact-duplicate rows exist, but the seeds show no DUPLICATE code. *Decided:* no DUPLICATE detection in B (`intra_source_dedup.enabled: false`). Reversals stay included and net to zero. *Provisional for C:* the profiler reports exact-duplicate counts, and we switch dedup on in C's config only if the evidence looks like re-exports (same invoice and line number, consecutive ids).

**D-17 LLM spend verdict.** The LLM may mark a line NON_PROCUREMENT only for explicit classes (tax, intercompany, payroll/people, financing/interest, revenue/customer, royalty/partner billing) and only with confidence ≥0.85. It never overrides a config INCLUDE rule. *Provisional:* the threshold gets tuned on B seeds with seed-derived rules disabled (a measure of C-like behaviour).

**D-18 ERP lines with cancelled/rejected invoices.** ERP receipts always count (no status in ERP; "count the ERP record"). *Decided.*

## D. Matching and de-duplication (A)

**D-19 Matching evidence.**
- Primary: same PO plus exact net amount (tolerance 0.02 + 0.1%).
- Without PO: same supplier core plus exact net amount.
- Text similarity is computed on compacted (space/punct-free) text and used for tie-breaks only, because OCR noise makes raw text unreliable.
- One-to-one assignment, so bulk items with repeated receipts don't all collapse onto one ERP line.
  - **2026-11-17 (R2 re-test on the scored quantity, DUPLICATE status; supersedes the earlier "OFF" note, architect decision):** the no-PO supplier+amount route is ON in A (`core_route.enabled: true`; code default stays off for other companies). Guards: runs only for tool rows the PO route could not place, only on ERP lines the PO route left unclaimed (one-to-one), ignores |net|<=0.01, requires dates within 120 days when both exist. Hold-out (PO blanked on 18% of PO-bearing tool rows, 3 seeds, truth = full-PO run flags): duplicate-status precision 0.89-0.90 (count), 0.90-0.92 (ZAR), recall 0.97-0.98; route OFF recall 0.13. Correct rows ~1.4k -> ~3.7k, correct ZAR ~100M -> ~283M. On the real 5,101 no-PO rows: 4,198 flagged (ZAR 318.7M) vs 563 with it off. Caveat: the hold-out blanks rows that do have a PO in the data; true no-PO invoices may behave differently.

*Decided.* Thresholds: *Provisional* (synthetic injection test).

**D-21 Re-exports.** Same supplier core, gross and quantity, plus (same invoice digits, or similar text and dates ≤45 days apart). All copies except one canonical are DUPLICATE. If any copy matches ERP, all copies are DUPLICATE. Canonical = best status, then the lowest id. *Decided.*
  - **2026-11-17 (P2):** re-exports need same supplier core, gross, quantity AND invoice digits, plus desc similarity >=0.85 and dates <=45 days when known. The invoice-free link and orphan rescue without equal invoice digits are off (all 49 real-A rescues had different invoice numbers). Canonical copy takes the best PO among siblings. Result on real A: 2,256 re-exports, 15,723 ERP matches, 10,541 tool rows still counted.

**D-22 Partial/aggregated invoices** (same PO, same unit price, different quantity). *Provisional:* a bounded subset-sum matcher sits behind a flag, enabled only if synthetic and audit checks show it adds precision. Otherwise these rows count as tool-only spend.
  - **2026-11-17 (R2):** partial-quantity capacity matching is ON (`reconciliation.partial_capacity`): same PO + supplier core, unit net price within 0.5%, cumulative quantity within the receipt's remaining capacity -> DUPLICATE, capacity decremented. Real A: +184 rows (ZAR 11.8M), 0 existing flags changed, pseudo-A2 agreement 1.0000. Synthetic: incremental precision ~0.98, split-part recall 0.88-0.96; hidden-receipt hold-out ~3.7% false flags on partner-less rows. Truth treatment of partial invoices unknown (open question 6). R7 aggregate matching stays off (+15 rows).

**D-23 Item keys for rows without a material number.** Use a description cluster key. Blank description with no material → `U:<supplier_key>|<gl>`. Golden item keys are blank for such ERP rows, so these rows are presumably unscored on A. *Decided.*

**D-24 Generic material ids** (M1000032, MAT-1000007: one id with hundreds of texts) are not item identity. *Decided.*

**D-25 Supplier names.** Remove whitespace entirely after legal-form stripping (reproduces the golden merges of names differing only by whitespace). Fuzzy merging is deliberately conservative: false merges of distinct suppliers (e.g. two names sharing a long prefix) are worse than leaving a rare variant split. *Decided.*
  - **2026-11-17 (P3):** fuzzy merging is guarded: no edit between two names may touch a digit that survives OCR folding (`DEPOT 12` != `DEPOT 21`). Fuzzy merges fire 0 times on real A (whitespace + legal-suffix stripping already collapse every variant); unsupervised tool-name precision 1.0, recall 0.9999 without the catch-all. **Item fuzzy merging is strict by default** (`entities.items.fuzzy_mode`): the architecture's ratio>=0.93 rule joined `1800MM` and `1500MM` because OCR folding (5->S, 8->B) hides digit differences; tool-item pairwise precision was 0.841 loose vs 0.979 strict. `merge_materials_by_text` stays off (ERP item precision 0.9997 -> 0.974 for little recall).

**D-26 B supplier keys** = `V:<vendor_id>`, global across ERPs (no names exist; 74 ids appear in multiple systems). *Decided.*

## E. Classification

**D-27 A/B use labels; C uses none.** A: supervised on labels_A (99.7% holdout). B: supervised on seeds plus LLM few-shot with retrieved seeds, and the LLM for low-confidence units. C: LLM-first. *Decided.*
  - **2026-11-17 (P4a):** supervised model = TF-IDF word 1-2 + char_wb 3-5 + categorical tokens -> LR (C=15) with temperature calibration. A2-sim on A (18% id-gap-run holdout): L3 99.75% by rows, 99.94% spend-weighted with all features (text only 91.2%; text+supplier 98.5%). Label lookup covers 72% of rows at 99.5%. B seeds 5-fold CV (random rows, optimistic): L3 95.8% text-only vs 91.8% with all features, so for small B-style label sets extra feature groups slightly hurt; not tuned yet.

**D-28 LLM provider: free only (user decision, 2026-11-17).** No paid API. *Decided:* a provider-agnostic router with an ordered chain:
1. **Groq** (primary; e.g. `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, spread across models to multiply the per-model daily token caps);
2. **Cerebras** (optional; free-trial status unclear);
3. **Ollama local** (private, unlimited, slow);
4. offline TF-IDF fallback.

**Gemini** is opt-in only (D-35). A **Claude/Anthropic** adapter stays dormant behind the same interface. *Provisional:* the model order comes from a bakeoff on the zero-label rehearsal (accuracy first, then tokens per unit). Free-tier limits are read live from rate-limit headers; the config seeds are only starting values.

**D-36 Distillation instead of "LLM everything".** Free daily token caps (~200k tokens/day/model on Groq's published Developer figures; free-plan numbers may be lower) can't cover ~15k units × ~150 tokens per unit plus prompts. *Decided:*
- the LLM labels **anchor units** in spend order up to 95% spend coverage (≤2,500 units), plus 300 diversity medoids of the tail;
- a local TF-IDF+LR model trained on those LLM answers (and on labels, where they exist) classifies the tail;
- low-confidence tail units go back to the LLM only if quota remains.

Why: spend concentration [F§8] means about 1.3–2.1k anchors cover 95% of spend, and the C score is spend-weighted. Expected ≈105 calls and ≈0.5M tokens per 50k lines.

**D-28b Gemini free tier as primary — *if the user accepts D-35* (2026-11-17).** Measured on the user's AI Studio account: Flash-Lite 3.5/3.1 at 15 RPM / 250K TPM / 500 RPD each; Flash 3.8 / 4 Hydrogen / 3.6 / 3 / 2.5 at 5 RPM / 250K TPM / 20 RPD each; Gemma 4 31B/26B at 30 RPM / 16K TPM / 14.4K RPD; Pro models unavailable. *Decided (D-35 option (a) accepted 2026-11-17):* the strongest Flash models handle taxonomy compilation and the top-spend units, Flash-Lite handles all remaining described units, and Gemma/Groq/Ollama take overflow. Distillation becomes the safety net, not the main path. Without the D-35 opt-in, fall back to the Groq + Ollama + distillation plan (D-28, D-36).

  - **2026-11-17 (implementation spike, resolved):** the first attempt returned 403 "free tier not available for this project"; on retry later the same day the dev key worked (cause outside the code; blind key never touched). Exact ids resolved with `models.list`: `gemini-3.8-flash`, `gemini-4-hydrogen`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemma-4-31b-it` (also visible, unused: `gemini-3.7-flash`, `gemini-4-argon`, `gemini-4-carbon`). `response_json_schema` and `ThinkingConfig(thinking_level)` work; Gemma rejects thinking levels (adapter retries without, then remembers).
  - **Spike batch (dev key, labels hidden, top-60 spend described A ERP units, 40-leaf tree with no definitions, compact `n|path` format, all-integer `{"r":[[uid,leaf,conf,class]]}`):** valid-JSON 60/60 on every model that answered; ~2.8k tokens in, ~640 out per call; no 429s.

    | model (thinking) | latency | L1 | L2 | L3 | L3 spend-wtd |
    |---|---|---|---|---|---|
    | 3.8 Flash (low) | 6s | 90.0 | 83.3 | 66.7 | 77.8 |
    | 3.8 Flash (medium) | 31s (+4.8k thinking tok) | 91.7 | 86.7 | 68.3 | 79.3 |
    | 3.5 Flash-Lite (minimal) | 19s* | 88.3 | 85.0 | 68.3 | 79.6 |
    | 3.1 Flash-Lite (minimal) | 3s | 85.0 | 80.0 | 65.0 | 77.9 |
    | Gemma 4 31B | 46s | 86.7 | 81.7 | 63.3 | 77.2 |
    | 3.6 Flash | 503 "high demand" | | | | |
    | 4 Hydrogen | 504 deadline at 90s (flaky) | | | | |

    *latency varies a lot run to run (1.3s-19s). Reading: on this sample the cheap Flash-Lite models equal the strong Flash model, so the top-spend tier's advantage must come from taxonomy definitions, second opinions and thinking, not raw model strength. 4 Hydrogen and 3.6 Flash are unreliable today: keep them late in the chains and always set a client timeout (adapter default 60-180s). n=60, so differences under ~5 points are noise. Not tuned on these errors (no prompt hill-climbing).

**D-28c Routing simplification (architect, 2026-11-17, from the spike measurements).** `gemini-3.5-flash-lite` is the primary for ALL classification roles (top-spend and bulk); `gemini-3.1-flash-lite` is the second opinion (agreement on the top 300 units, widen to all units if quota allows); `gemini-3.8-flash` only for taxonomy compilation and for arbitrating disagreements; `gemma-4-31b-it` is overflow; `gemini-4-hydrogen` and `gemini-3.6-flash` are last-resort spares; then Groq, then Ollama. Default thinking minimal/low. Hard client timeout on every call. Supersedes the role split in D-28b/§8.4 (the strong Flash models no longer carry the top-spend tier). *Decided.*

**D-38 Escalation: free → capped paid → cross-vendor (user request 2026-11-17).**
1. Gemini free tier (blind-day project) is primary.
2. A separately billed Gemini project (`GEMINI_API_KEY_PAID`) is used only on free-quota exhaustion/persistent 429s, and for an optional quality boost on the top 300 spend units with the strongest paid model. Hard cap `max_paid_usd` (default $5/run).
3. Groq free is the fallback for a Google-wide outage (a different vendor).
4. Ollama local, then the offline method.

Paid-project traffic falls under Google's paid data terms (no training use). xAI "Grok" is not used (paid, no advantage). *Decided.* *Provisional:* whether the quality boost actually helps is decided by the bakeoff (paid best model vs free 3.8 Flash on the top-300 units of A/B zero-label). If it adds under 1 point spend-weighted, set `quality_boost_top_n: 0`.

**D-37 Quota hygiene.** Gemini RPD resets at 00:00 Pacific = 13:30 IST, so Thu 10:00–12:00 IST shares a quota day with Wed 13:30 IST onward. Use a dedicated blind-day Google Cloud project/key, untouched until Thursday. There's a token ledger per provider/model (rolling 24 h). No heavy LLM experiments on the blind-day key after Wed 10:00 IST. Use separate dev and blind-day keys/projects if the provider allows. *Decided.*

**D-29 Blank descriptions** → supplier spend-weighted category distribution, then GL distribution, then an LLM call with supplier/GL context only. *Decided.*

**D-30 Industry hint.** A free-text `industry_hint` goes in config (written for C at blind time from the brief's description and profiling). It is configuration, not code. *Decided.*
  - **2026-11-17 (P5):** `spendcube profile` drafts the C config from column names (EN/DE/FR/ES/PT/IT) plus content, suggests a multilingual status map, flags non-procurement candidates from GL text/descriptions/suppliers and warns about silent taxonomy misparses. A human still confirms amount basis, static FX rates and exclusion rules. Synthetic C drill (European decimals, German statuses, EUR/CHF with static FX, 4-level ragged path tree): exclusions reproduced exactly, spend within USD 0.06 of truth. Bugs it found were fixed: Swiss `1'234.56` numbers, `*parent*`-named adjacency columns, code-column-first wide/path trees; `taxonomy.columns` override added.

**D-31 Unit of classification** = unique (supplier, description, material, GL, context). Units are processed in descending spend order so that budget pressure only ever hurts the low-spend tail. *Decided.*

**D-39 Cascade implementation (2026-11-17).** Per unit (supplier, description, GL, context), in spend order: (1) rows that carry their own label keep it (`label_lookup`, conf 0.97); (2) supervised label model accepted at p >= 0.80; (3) LLM for the remaining described units (`top_spend` role for the first 1,000 units, `bulk` after; two-model agreement + arbiter on the top 300; conf = 0.5 x self-reported + 0.5 x agreement); (4) distillation: a supervised model trained on the LLM answers (+ labels, + compiled leaf definitions) fills units the LLM could not reach (conf = p x 0.85); (5) supplier then GL propagation (share >= 0.6, conf <= 0.6); (6) for still-unresolved/blank units: the label-trained supervised top-1 when labels exist (B: opaque vendor ids make an LLM useless), else a context-only LLM call (conf <= 0.5); (7) offline TF-IDF similarity to leaf paths/definitions (conf <= 0.4). LLM `spend_class` != procurement at conf >= 0.85 becomes NON_PROCUREMENT unless a config include-rule protects the row (D-17). In A2/B2 runs the label-trained model is trained from the kit files (`kit_dir`), not the run's input. *Decided.*

**D-40 Free-tier behaviour observed (2026-11-17).** Gemini returns frequent 503 "high demand" on every model (30% of Flash-Lite 3.5 calls in the B run); policy: one jittered 2-8 s retry, then fail over along the role chain. Strong Flash models (20 RPD) were exhausted on the dev project during the afternoon, so strong-model arbitration/compile could not be measured on dev; they reset at 13:30 IST. Per-entry request counts come from a persistent quota ledger (`.cache/llm_ledger.sqlite`), so a re-run does not re-spend and `GEMINI_API_KEY` (blind) shows 0 requests. *Decided.*

**D-41 Blank-description units (2026-11-17).** Resolved by `classify/blank.py` before any fallback: a spend-weighted vote over supplier/vendor/GL context keys (share >= 0.6) wins; a categorical-only supervised model fills the rest; confidence <= 0.6. Training signal = the run's own labelled or LLM-labelled described units. Measured by blanking labelled lines (id-gap holdout): label mode spend-weighted L1/L3 A 98.3/57.7 -> 99.98/82.6, B 94.4/72.7 -> 95.4/76.9; vote right ~99% where it fires. C-like mode (900 top-spend units LLM-labelled): blank-unit L1/L3 A 74.1/42.4 (limited by A's LLM labels, 36.6% L3), B 94.8/74.3. Holdout leaves suppliers/GLs seen in training, so label-mode numbers are optimistic for an unseen company. *Decided.*

**D-42 Per-job LLM request budget (2026-11-17).** Enforced in `llm/router.py`, not just planned. Blind-day jobs run sequentially C -> B2 -> A2 (separate invocations sharing the quota ledger). Each job's budget = `share` x (`util_cap` x RPD minus today's ledger count), summed over the budget pool (the two Flash-Lites). Defaults: util_cap 0.7, C share 0.6, B 0.5 of what is left, A 1.0 of what is left (`classification.llm.budget`, per-company override). When spent, classification roles stop (`chain()` returns nothing) and supervised/propagation/offline tiers finish the run; `compile` and `arbiter` are exempt (they use the strong models' own 20/day caps). Never blocks and never raises; the output is always valid (`tests/test_budget.py`). `doctor [--blind]` prints remaining RPD per model (ledger read, no API call) and the per-job budgets. With an empty blind ledger: C 420, B2 140, A2 140 requests. Evals build their router without a budget. *Decided.*

**D-43 Profiler suggests exclusion rules (2026-11-17).** `profile` emits COMMENTED `NON_PROCUREMENT` rules with row counts and spend (raw currency) for template-like lines: a short leading phrase repeated >= 3 times (>= 1 for GL descriptions) in intercompany / tax / payroll / bank / interest / other non-spend groups; phrases with service words (consult, services, audit, software...) are vetoed so "Tax Consultant"-style lines stay in. `--accept-suggestions` (or deleting the leading `# `) is the only human step. B dress rehearsal with suggestions uncommented: 2,487 vs 2,485 excluded (hand config), USD gap -0.000% (the unedited draft: 0 excluded, +6.0%). A: no suggestions (none needed); the synthetic C finds its planted rules. `exclusion_rules:` with every rule commented parses as an empty list. *Decided.*

**D-44 Gemini client race (2026-11-17).** `GeminiProvider._client()` created clients without a lock; concurrent first calls built duplicates, the loser was garbage-collected and closed ("Cannot send a request, as the client has been closed"), failing a few early calls and silently pushing answers down the chain. Fixed with a lock. Gemma 4 (31B) additionally returns frequent `503 UNAVAILABLE` (high demand) on the free tier: it is a failover, not a dependable tier. *Fixed.*

**D-45 Unit memory (2026-11-17).** `classify/memory.py`: LLM answers (conf >= 0.7) of earlier runs on the same taxonomy hash are stored per unit (exact key text|supplier|GL|context, looser key text|supplier) in `.cache/unit_memory.sqlite` and answered before any LLM request (`method=memory`, conf <= 0.9). The LLM response cache is keyed on whole-batch prompts, so a new extract (B2/A2) with a different unit mix would miss it; memory answers individual units. A different company's taxonomy never sees it. *Decided; request saving measured in EVAL_LOG.*

**D-46 Supplier coherence (2026-11-17), experimental, OFF.** `classify/coherence.py` re-asks the LLM about low-confidence units that disagree with a clear supplier majority (spend-weighted share >= 0.5 over >= 3 units), passing the majority path as a hint in the unit's context. Only enabled if the supplier-rich measurement (`eval coherence`) improves spend-weighted L3; result in EVAL_LOG. *Pending measurement (dev quota resets Wed 13:30 IST).*

**D-47 Entry confidence caps (2026-11-17).** Failover models answer at capped confidence (`llm.entry_conf_cap`: Groq 0.5, Ollama 0.4, Gemma 0.7) because they measured far weaker (Groq gpt-oss-120b 8.3% L3 vs 30% for Flash-Lite 3.1 on the same 60 units). Labels unchanged. *Decided.*

**D-48 Confidence calibration (2026-11-17).** `classify/calibration.py`: per-tier isotonic map raw -> P(L3 path correct), fitted by `scripts/fit_calibration.py` on held-out runs, stored as knots in `configs/_calibration.json` (no data), applied at the end of the cascade (`cat_conf_raw` keeps the original). Finding: the supervised tier is already calibrated (>= 0.95 band 99.9% right, ECE 0.018); the LLM tier is **company-dependent** (the "both models agree" band is 82% right on B and 28-35% on A) so a pooled map cannot be validated out-of-sample across companies: leave-one-company-out ECE 0.343 -> 0.341. Numbers and decision in EVAL_LOG. *See EVAL_LOG.*

**D-49 Classification chain (2026-11-17).** Only models shown to beat the non-LLM tiers classify: Flash-Lite 3.5/3.1, Flash 3.6 and Hydrogen. Groq gpt-oss-120b (collapses onto one catch-all leaf: L3 8-23% on two 60-unit batches, 7-23% agreement with Flash-Lite, confidence 69-81 regardless), Gemma 4 (28% of calls succeed, 503 "high demand"; accuracy unmeasured) and local qwen3:4b (L3 15-37%) moved to `outage_chain`, used only when `allow_outage_models: true`. Root cause for Groq is model behaviour, not truncation/ids/repair (prompt ~1.4k tokens, outputs complete, ids in range, +-1 shifts do not help). Before: bulk = lite35, lite31, gemma4, flash36, hydrogen, groq, ollama. After: lite35, lite31, flash36, hydrogen. *Decided.*

**D-50 Budgets rebalanced by measured curves (2026-11-17).** Budget = min(max_requests, share x (remaining pool - reserve)). Defaults: C share 1.0 with reserve 70 (630 on an empty ledger, computed from what is left, C runs first), B2 max 60 (reserve 10), A2 max 10. Curves in EVAL_LOG: B spend-weighted L3 knee at ~25 requests (97.8 -> 98.7), A flat (the LLM does not help, 99.5 -> 99.4). Replaces D-42's 0.6/0.5/1.0 shares. *Decided.*

## F. Process and delivery

**D-20 Config locations.** A/B configs live in `configs/` and use `{input_dir}` (CLI `--input-dir`) so A2/B2 run with unchanged configs. The C config is written at blind time to `submission/C.yaml` (the brief: "push only the C config and the three output files into `submission`"). *Decided.*

**D-32 Labels and training at run time.** Models are trained at run time from the kit (seconds), never committed, since pickles embed data. LLM cache lives in `.cache/` (gitignored); build-time runs warm it for A2/B2. *Decided.*

**D-33 Candidate name** = `jaygautam`. Files: `jaygautam__A.csv`, `jaygautam__B.csv`, then `jaygautam__A2.csv`, `jaygautam__B2.csv`, `jaygautam__C.csv`. *Decided (user, 2026-11-17).*

**D-34 Keys and the "run without our keys" rule.** The tool reads `GROQ_API_KEY` (optionally `CEREBRAS_API_KEY`, `GEMINI_API_KEY`) from the environment or `.env` (gitignored). With no key and no Ollama daemon, it runs the offline fallback and says so. APPROACH.md documents how to get a free Groq key, how to install Ollama with the chosen model, and the cost: $0, plus tokens per 1,000 lines. *Decided.* **Open: the user must create a Groq key and read that account's actual free-plan limits.**

**D-35 Confidentiality vs free tiers.** The brief says the data is confidential and for this task only. Google's terms let the **Gemini API's unpaid quota** use prompts and responses to improve Google products, with possible human review. Mistral's free plan may train on inputs. *Decided:* Gemini and Mistral adapters are **disabled by default**; enabling Gemini needs an explicit `allow_training_use: true` in config, which is the user's call. Groq is believed not to train on API data (verify). Ollama keeps data on the machine and is always allowed. **Alternative that keeps Gemini's capacity without the training-use clause:** enable billing on the Gemini project. Paid usage isn't used for product improvement, and Flash-Lite at this volume (~1–2M tokens per run) should cost cents. This is the user's choice (they asked for free).
  - **2026-11-17 — user decision: option (a).** Use the Gemini free tier and accept Google's unpaid-tier data terms. `allow_training_use: true` is set in all configs, and Gemini is the primary provider (D-28b). APPROACH.md must state this openly. Mitigation: send only the fields needed for classification (description, supplier name, GL text, context columns), never whole rows or amounts beyond the rounded spend magnitude.

## Unresolved questions (cannot be answered from the materials)

1. Does the A2 truth treat a tool record matching an A-period ERP line as DUPLICATE? (D-02 bets yes.)
2. How is supplier/item grouping scored: pairwise F1, B-cubed, or purity? Spend- or count-weighted? We optimize pairwise and B-cubed F1 jointly and report both.
3. Will A2/B2 file names differ from the originals? Mitigated by glob patterns and the `--input-dir` override.
4. Will B2 include an FX file? Mitigated by an FX search path that falls back to the kit's table.
5. C's format and depth are unknown. Mitigated by the profiler, a flexible config, and taxonomy format auto-detection.
6. Is a "partial invoice" a DUPLICATE in the truth (D-22)?

**D-51 Automatic outage path (2026-11-17).** If no Gemini classification entry is usable (all 5xx/dead/quota 0/no key) and the job budget is not merely spent, the router switches to the local model (`llm.outage.chain`) for top-spend units, for at most `outage.max_minutes` (20), logs a loud warning and `outage_switch` in the report; the remaining units use the offline fallback. Measured on A's top-300 zero-label sample: offline fallback L1/L2/L3 spend-weighted 75.9/31.0/25.4; local qwen3:4b (60 units) 94.1/45.3/41.7 vs offline 81.4/34.4/28.1 on the same 60; 1.24 s/unit. Groq stays out of classification. *Decided.*

**D-52 Mock safety and latency audit (2026-11-18).** Blind runs and `doctor --blind` refuse to start whenever any `SPENDCUBE_FORCE_MOCK*` or `SPENDCUBE_MOCK_*` variable is present. Runs record `mock_active` in the run report and print a stderr banner when a mock is active. Every live provider call records model, elapsed wall milliseconds, batch unit count, and input/output tokens in a run-local `llm_call_metrics.csv`. Full B throughput (260 requests / 18.9 min = 13.76 requests/min) implies 46.5 min for C's 640-request budget, replacing the 23.8-min mock estimate. *Decided.*

**D-53 Parallel Flash-Lite pool (2026-11-18).** Router supports an opt-in concurrent classifier pool across Flash-Lite 3.5/3.1, with model-specific RPM/TPM/RPD limiters and the existing shared job request budget. Failure/rate limiting one model removes only that member for the run and dispatches the request to its peer. `parallel_pool.enabled` remains false until T4a establishes 3.1 is within 3 spend-weighted L3 points of 3.5; otherwise use 3.1 only on the spend tail. *Implemented; default pending T4a.*

**D-53 Blind-day scheduling and absolute cutoff (2026-11-18).** B2/A2 run first via `blind_day.sh ab`; C runs separately after the human validates its config, with its request budget drawn from the post-B2/A2 ledger. C's `--deadline-at YYYY-MM-DDTHH:MM` is an Asia/Kolkata-aware per-run cutoff; it is absent by default, while `deadline_minutes` remains a relative safety guard. The report records deadline cause and completed LLM units; the summary reports output validation, exclusions, tier spend, request counts, and GO/NO-GO. *Decided.*

**D-54 Parallel Flash-Lite dispatch (2026-11-18).** An opt-in pool dispatches bulk/top-spend batches across the two Flash-Lite entries using per-model RPM/TPM/RPD limiters, prioritizing the shortest estimated limiter wait and then the lowest daily request count. All pool and normal-chain cache keys are checked before dispatch. Calls already run concurrently in `LLMClassifier.classify` (four workers). The request budget is atomically shared across pool members; each actual provider attempt consumes one reservation. Repeated 503/timeout disables only that model and iterative fallback tries the remaining pool and then normal chain. `parallel_pool.enabled` remains false until T4a compares model accuracy on the same 900 units. *Decided.*
