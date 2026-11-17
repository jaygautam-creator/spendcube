# RISK_ANALYSIS.md

Likelihood (L) and Impact (I): H/M/L. Impact is measured against blind-test points. Each risk lists its root cause, mitigation (where in the design) and the residual risk. Ordered roughly by expected point loss.

---

## 1. Classification of C (50 pts, the largest exposure)

**R-01 Weak zero-shot accuracy on C's unseen tree.** L:M I:H
- *Root cause*: cryptic and abbreviated text; a tree whose names read badly as labels (A has leaves named like contract terms or "unclassified" buckets); industry jargon; lexical methods hit 10–30% [F§7].
- *Mitigation*:
  - LLM classifier with the full tree;
  - taxonomy compilation (leaf definitions from the company's own descriptions);
  - industry hint;
  - spend-tiered routing to the strongest model;
  - self-consistency on the top ~300 units;
  - supplier context ("other descriptions from this supplier").
- *Validation*: zero-label rehearsal on A and B.
- *Residual*: C may be harder than A/B. The rehearsal score is an optimistic estimate.

**R-02 Errors on a handful of huge lines dominate the spend-weighted score.** L:M I:H
- *Root cause*: 55–60 units = 50% of spend [F§8].
- *Mitigation*: these units always get `model_top`, effort ≥medium, 2–3 votes, and full context. Check that the run report lists the top 50 units with their categories for a 5-minute human sanity read during the blind window. Fixes are possible only via config (industry hint, rules); no code edits.
- *Residual*: a systematically ambiguous tree split (e.g. "maintenance services" vs "spares") can still go wrong.

**R-03 Inconsistent labels across batches** (the same kind of item lands in different leaves). L:M I:M
- *Mitigation*: compiled taxonomy in the cached system prompt; units sorted so similar descriptions share batches (sort by supplier, then description within a spend tier); a post-pass that unifies near-duplicate descriptions (same item_key) to their spend-weighted majority leaf when the majority is ≥0.7.

**R-04 Blank or meaningless descriptions** (B: 40.7% of lines, ~37% of spend). L:H I:M
- *Mitigation*: supplier/GL propagation, then an LLM with supplier and GL context. *Residual*: low accuracy on those lines. B2 probably doesn't score them ("clear-cut"), but C might, by spend.

**R-05 Overfitting to A/B** (prompt tweaks, rules or thresholds that only help A/B). L:M I:H
- *Mitigation*: no company text in code (CI grep test); company specifics only in configs; generic prompt instructions only; a synthetic C fixture with a different domain; the zero-label rehearsal is reported but not hill-climbed on individual errors.

**R-06 Ragged or deeper tree, odd tree file format** (path strings, adjacency list, duplicate names, trailing spaces, commas in names). L:M I:H (one bad parse ruins all of C)
- *Mitigation*: `Taxonomy` supports wide/path/adjacency with auto-detection; exact raw strings are preserved; leaf ids; validator checks every output path exists in the tree; the profiler prints the parsed tree summary (leaves per depth) for a human check at blind time.

**R-07 The LLM misjudges spend vs non-spend** (e.g. "tax consultant" excluded, or royalties missed). L:M I:M
- *Mitigation*: explicit spend_class enum; threshold ≥0.85; config rules take precedence; tuning on B seeds with rules disabled.

## 2. A2 matching and exclusions (30 pts)

**R-10 False negatives in tool↔ERP matching** (double counting). L:M I:H
- *Root cause*: missing PO (18%), OCR'd text, reformatted PO, partial or aggregated invoices, re-exports claiming the same ERP line.
- *Mitigation*: amount-centric scoring on compacted text; supplier+amount route; orphan rescue (R6); history matching (R8); optional aggregate matching (R7); synthetic-injection recall measurement.

**R-11 False positives** (a tool-only invoice marked DUPLICATE because the same supplier and amount occur, e.g. stone deliveries at identical prices). L:M I:M
- *Mitigation*: one-to-one assignment; quantity and po_item in the score; supplier+amount route requires amount exactness; injection test with fabricated tool-only lines measures precision.

**R-12 History-matching assumption is wrong** (D-02). L:L–M I:M
- *Mitigation*: the switch is in config and history matches are counted in the report. *Residual*: a judgment call that cannot be validated before the test.

**R-13 VAT misparse** (`0` vs blank, `15` vs `0.15`) turns every amount wrong by 15%. L:L I:H
- *Mitigation*: unit tests per encoding; the matched-amount rate per VAT encoding is a diagnostic in the report.

**R-14 A2 supplier/item keys diverge from golden** (new names, the catch-all). L:M I:M
- *Mitigation*: label alias maps; normalization reproduces the golden merges; tool names resolved through ERP matches. *Residual*: unseen names in A2 that belong to the golden catch-all can't be inferred (D-01).

## 3. B2 amounts and exclusions (20 pts)

**R-20 FX traps** (USD with rate, wrong-currency rates, missing months, inverted conventions). L:M I:H (USD per line is scored)
- *Mitigation*: D-06/D-07/D-08 chain; unit tests on the exact observed patterns; report of fx_method counts and the rate/table ratio distribution.

**R-21 Missing FX file in B2 input.** L:M I:H
- *Mitigation*: the FX search path includes the kit file; months beyond the table use the nearest month.

**R-22 Seed-labelling imperfections teach wrong exclusions.** L:M I:L
- *Mitigation*: learned rules need support ≥5 and purity ≥95%; config rules dominate.

## 4. Operational risks during the 2-hour window

**R-30 API failures, rate limits, overload, slow responses.** L:M I:H
- *Mitigation*:
  - retries with backoff and adaptive concurrency;
  - bisect-on-bad-batch;
  - sqlite cache (resume on re-run with zero re-spend);
  - deadline controller that fills the remainder with fallback;
  - top-spend-first scheduling, so any truncation hits the tail;
  -   - a pre-flight `spendcube doctor` that checks the key, model access and a 1-call latency.
- *Residual*: a full outage leaves the offline fallback (much lower C score).

**R-31 Run takes too long** (C bigger than expected, slow model). L:M I:H
- *Mitigation*: unit dedup; measured throughput in P6; `deadline_minutes` default 75; start A2/B2 immediately at 10:00 in parallel with C config writing. A2/B2 are mostly cache hits and supervised, so they're cheap.

**R-32 Writing the C config takes too long or contains errors.** L:M I:H
- *Mitigation*: `spendcube profile --draft-config` guesses the mapping, lists status values, candidate non-procurement GL/desc values and number/date formats; `C.example.yaml` is annotated; `--limit 2000 --no-llm` smoke run in <1 min; a rehearsed runbook with timings (do a full dress rehearsal on Wed using B as "C").

**R-33 Crash on unexpected input** (encoding, delimiter, Excel file, an empty column, duplicate ids, a giant field). L:M I:H
- *Mitigation*: robust readers; every row-level failure becomes a flag, never an exception; stage-level try/except that degrades (e.g. matching disabled) and logs instead of aborting; output is always written and validated. Fuzz tests on the synthetic fixture.

**R-34 Memory/CPU blow-up from fuzzy matching.** L:L I:M
- *Mitigation*: blocking with block-size caps; rapidfuzz cdist per block; no global pairwise operations.

**R-35 Non-determinism between runs** (the LLM gives different answers on a re-run by the graders). L:H I:L
- *Mitigation*: cache keyed by payload; deterministic unit order; documented in APPROACH. Graders re-running without the cache get a similar but not identical result.

## 4b. Free-LLM-specific risks (added v2)

**R-50 Free quota exhausted or lower than assumed on blind day.** L:M I:H
- *Mitigation*:
  - distillation (≈105 calls per 50k lines);
  - multi-model chain plus Ollama spill-over;
  - quota ledger and `doctor` check at 09:00;
  - no heavy dev runs on the blind-day key after Wed 10:00;
  - A2/B2 mostly served from cache and supervised models.
- *Residual*: if every remote entry is exhausted, C runs on local 4B quality on the tail, but anchors were already scheduled first.

**R-51 Weaker models than frontier Claude** (gpt-oss/Qwen-class open models). L:H I:M
- *Mitigation*: taxonomy compilation; cross-model self-consistency on the top 200 units; strict client-side validation; bakeoff-chosen order. The A/B zero-label rehearsal measures the real accuracy. *Residual*: a few points lower on cryptic abbreviations.

**R-52 Provider API or model churn** (model renamed or retired, limits changed, JSON-schema support differs). L:M I:M
- *Mitigation*: model names only in config; JSON parsing tolerant (extract the first JSON object, then validate); repair retry; `doctor` lists the available models at 09:00 so the config can be adjusted (config change only).

**R-53 Confidentiality breach via a free tier that trains on data.** L:M (if Gemini or Mistral is enabled) I:H
- *Mitigation*: D-35 defaults; opt-in flag; documented in APPROACH.

**R-54 Ollama too slow or memory-starved on 8 GB.** L:M I:L
- *Mitigation*: a 4B-class model only; one concurrent request; used for spill-over, not the primary.

## 5. Submission and process

**R-40 Confidential data committed to the public repo.** L:M I:H (disqualifying, plus breach)
- *Mitigation*:
  - `.gitignore` written before any `git init` (done);
  - a pre-commit hook script that blocks commits of `*.csv` outside `outputs/` and `submission/`, plus the known data file names;
  - LLM cache and run dirs are ignored;
  - test fixtures are synthetic only;
  - `git ls-files` audit in the freeze checklist.
- Note: output CSVs contain supplier names only in keys, and the brief requires those outputs.

**R-41 A code change after the 10:00 freeze** (disqualification). L:L I:H
- *Mitigation*: everything tunable lives in config; tag `freeze` at the deadline; blind-time commits touch only `submission/`; a `git diff --stat freeze..HEAD` check before pushing.

**R-42 Wrong file names / columns / encoding.** L:L I:H
- *Mitigation*: the validator enforces the template header order and naming `<name>__A2.csv` etc.; a runbook check.

**R-43 Paid API keys unavailable to graders.** L:H I:L
- *Mitigation*: offline fallback path; APPROACH.md states how to plug in a key and the measured cost.

**R-44 Python 3.14 wheel or install issues** on the graders' or our machine. L:M I:M
- *Mitigation*: pin versions known to have 3.11–3.14 wheels; README with `python -m venv` steps; avoid optional heavy dependencies in the core path.

## 6. Compound failures

- **The API degrades in the middle of C.** The deadline controller and cache keep the top-spend units classified. The tail falls back. The output is still valid.
- **The C tree is parsed wrong and the LLM classifies against garbage.** The profiler's tree summary is checked by a human before the full run (runbook step), and the validator aborts if any l-path is not in the tree.
- **An unknown status vocabulary plus a wrong default** (everything included, or everything NOT_APPROVED). The profiler lists status values and the run report shows exclusion rates. A sanity threshold warns if more than 30% of rows are excluded by status.

## 7. Remaining accepted risks
D-01 catch-all for unseen A2 names; D-02 history semantics; D-05 credit notes; D-22 partial invoices; R-04 blank-description accuracy; the gap between rehearsal and C accuracy.
