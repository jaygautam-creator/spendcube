# IMPLEMENTATION_PLAN.md

Build window: **now (Tue 17 Nov) → Thu 19 Nov 10:00 IST freeze.** Roughly 1.5 working days. The plan is ordered so that **a valid, scoreable submission exists at the end of every phase**. Each phase lists deliverables and exit criteria; do not start the next phase until the exit criteria hold, except where marked parallel.

Golden rule: **at all times, `python -m spendcube run --config configs/A.yaml` and `configs/B.yaml` must produce valid output files.**

---

## P0 — Skeleton that produces valid files (target: 3–4 h)
Deliverables:
- Repo init (the `.gitignore` already exists; verify with `git status` that no kit file shows up). `pyproject.toml` / `requirements.txt` pinned: pandas, numpy, scikit-learn, rapidfuzz, pyyaml, pydantic, httpx, openpyxl, pytest (+ optional `groq`, `google-genai`, only if their adapters need them).
- `config/schema.py` + `loader.py`: pydantic models for §3 of the architecture, env/placeholder/glob resolution, CLI overrides.
- `ingest/` readers + canonical mapping; `record_id` prefixes; reference role.
- `taxonomy/tree.py` with wide/path/adjacency formats, leaf ids, exact-string paths.
- `output/writer.py` + `validator.py` + `report.py` skeleton.
- `pipeline.py` + `cli.py run|validate`. Classification placeholder: most frequent leaf per company with conf 0.01.
- `configs/A.yaml`, `configs/B.yaml` (columns mapped; rules may be empty).
Exit:
- Both A and B outputs pass the validator, with one row per input record (65,549 for A, 41,028 for B).
- The pre-commit data-guard script is installed.
- A unit test fails if any kit taxonomy string appears in `spendcube/`.

## P1 — Money, dates, exclusions (target: 3 h) — parallel with P2 once P0 is done
- `normalize/numbers.py, tax.py, dates.py, fx.py, text.py` per architecture §5.2, D-04…D-11.
- `exclusions/` status map, ordered rules, learned rules (from labels with include flags), vendor propagation, precedence (D-03).
- B config: IC and tax rules (D-12, D-13).
Tests (synthetic): every VAT encoding; `1.234,56`; parentheses negatives; dd/mm vs mm/dd detection; FX chain including the USD-with-rate case, the implausible-rate case, the missing-month case and nearest month.
Exit:
- B exclusion precision and recall vs seeds ≥0.97 / ≥0.95 (report the numbers).
- B total USD reported, plus fx_method counts.
- A tool net amounts match ERP values for the known pairs (the diagnostic from [F§4] reproduces about 95%).

## P2 — Reconciliation for A (target: 4 h) — parallel with P1
- `reconcile/` R1–R6 (+R8 history) per architecture §6; R7 aggregate behind a flag (implement last, only if time allows).
- `eval/synth.py`: synthetic tool-copy generator (VAT encodings, PO formats, OCR noise, spacing/case, dropped PO 18%, re-exports 8%, date jitter) plus fabricated tool-only lines; `eval matching-synth` reports precision and recall.
Exit:
- Synthetic recall ≥0.97 and precision ≥0.99.
- On real A: report counts (re-exports, erp matches, tool-only, excluded by status). Sanity: roughly 2.4k+ re-exports, about 19k matched, about 7k tool-only (prototype numbers; improvements should *reduce* "PO in ERP but unmatched", currently about 1.4k).
- Manual audit of 30 matches and 30 non-matches logged in DECISIONS.

## P3 — Entity resolution (target: 3 h)
- `entities/suppliers.py`, `items.py`, `legal_forms.py` per §7. Label alias maps. Linking secondary to the primary item master. Generic-material detection.
- `eval entities`: pairwise P/R/F1 and B-cubed on A ERP (unsupervised mode, aliases off) and on tool rows via matched-ERP pseudo truth; also with aliases on (A2 mode).
Exit:
- Unsupervised supplier pairwise precision ≥0.98 on A tool names → ERP-golden (excluding the catch-all effect; report both with and without the catch-all).
- With aliases on, A ERP supplier and item grouping ≈ golden (F1 ≥0.99).

## P4 — Classification (target: 6–8 h, the core)
Order inside P4:
1. `classify/units.py` + `lookup.py` + `supervised.py`, plus the inheritance path. **Exit 4a**: A2-sim ≥98% L3 (spend-weighted); B seeds 5-fold CV reported.
2. `llm/providers/{groq,cerebras,ollama,gemini,mock}.py`, `router.py` (chain, limiters, quota ledger), `cache.py`, `prompts.py` (token-frugal), `schemas.py` (client-side validation). Deadline controller. `spendcube doctor`.
3. `taxonomy/compile.py`: leaf definitions call, cached.
4. `classify/llm.py` (anchor selection, diversity medoids, adaptive batching, few-shot retrieval when labels exist, cross-model self-consistency), `distill.py`, `propagate.py`, `fallback.py`, `cascade.py`, `confidence.py`.
5. `eval zero-label`: A ERP with labels hidden (stratified 2,000-unit sample + the top 300 spend units for the iteration loop; full run once at the end); B vs seeds with seeds hidden.
6. **Bakeoff** (D-28): run on the **dev** Gemini project only (20 RPD on Flash models: plan it; ~400 units at 60/call = 7 calls per model). Every chain model (Gemini 3.8 Flash, 4 Hydrogen, 3.5/3.1 Flash-Lite, Gemma 4 31B, Groq gpt-oss-120b, qwen3.8-27b, gpt-oss-20b; Cerebras if available; 2–3 Ollama 4B-class tags) on the same 400-unit sample (top spend + stratified), A and B zero-label. Measure accuracy, tokens per unit, failure/repair rate and seconds per unit. Set the chain order. Also measure the **quality boost**: the paid strongest model vs free 3.8 Flash on the top-300 units, ≤$1 of paid budget (D-38). Also measure **distillation loss**: tail accuracy of the distilled model vs the LLM on held-out tail units. **Budget this against the quota ledger**; it is the single biggest quota consumer before blind day.

**Exit 4b**:
- Zero-label A and B accuracy reported (L1/L2/L3, count- and spend-weighted).
- A full A and B run with the LLM finishes; cost and time are logged.

Rough expectations, to judge whether something is broken (not targets to overfit): zero-label L1 ≥85% spend-weighted; L3 ≥65–75%.

## P5 — Generalization drills (target: 3 h)
- `profile/profiler.py` + `draft_config.py`: column role guessing (amount/date/currency/status/supplier/description/GL by name and content), distinct status values, number/date format detection, candidate non-procurement values (keyword scan in GL desc/description: tax, VAT, GST, IVA, intercompany, IC, I/C, payroll, salary, wage, interest, loan, dividend, royalty, revenue, rebate, refund, bank charge, fee, donation, penalty, fine), duplicate-row stats, currency list vs FX availability, a taxonomy summary.
- `configs/C.example.yaml` heavily commented.
- `tests/fixtures/synthC/`: a synthetic company (different industry, 4-level ragged tree in *path* format, `;` delimiter, European decimals, `dd.mm.yyyy`, non-English statuses, gross + tax column, two currencies without line rates and a static fx). End-to-end test with a mocked LLM.
- **Dress rehearsal**: treat B as unseen. Hide seeds, start from the profiler's draft config, time yourself writing it, run, validate. Record the timings in the runbook.
Exit: the synthetic C run passes the validator; the rehearsal completes in under 60 min of wall clock including config writing.

## P6 — Performance and robustness (target: 2 h)
- 50k scale test (architecture §11.7); profile hot spots; check block caps.
- Failure injection: free quota exhausted (mock 429 storm on Gemini free → paid escalation → $ cap hit → Groq), Google outage (all Gemini 5xx → Groq), API key missing, 429 storm (mock), timeouts, a bad JSON batch, kill and resume (cache reuse), deadline reached at 50%.
Exit: 50k lines < 50 min with the LLM (extrapolated from measured per-call latency × calls / concurrency); every failure-injection case still yields valid outputs.

## P7 — Deliverables and freeze (target: 2 h, must start by Wed 22:00 IST at the latest)
- Final full runs: `outputs/<name>__A.csv`, `outputs/<name>__B.csv` (validated).
- README.md (setup, key, the one command, blind runbook), APPROACH.md (one page: pipeline, matching/dedup, unseen-tree method, measured zero-label accuracy on A, runtime, LLM cost per 1,000 lines, AI tools used), DECISIONS.md (final).
- Freeze checklist:
  1. `git ls-files` shows no data files; grep for kit supplier names in tracked non-output files.
  2. Fresh clone → venv → run on A with `--limit 500` works.
  3. `spendcube doctor` passes with the key.
  4. Push the repo public and submit the form.
  5. `git tag freeze` before 10:00 IST Thu (aim for 09:00).

---

## Dependencies

```
P0 ──► P1 ──┐
   └─► P2 ──┼─► P3 ──► P4 ──► P5 ──► P6 ──► P7
            │          ▲
            └──────────┘ (P4.2 LLM client can start right after P0, in parallel)
```

Independent work packages (suitable for parallel subagents if Sonnet chooses): normalization (P1); reconciliation + synth generator (P2); LLM client/cache/budget with mock (P4.2); taxonomy loader and compile (P0/P4.3); profiler (P5). Shared contract: the canonical frame (architecture §4). Freeze the column names in `spendcube/schema_cols.py` at the start of P0.

## Interface contracts that must not drift
- Canonical frame column names and meanings (§4).
- `Taxonomy`: `leaves -> list[Leaf(id, path: tuple[str,...], depth)]`, `path_padded(leaf_id, n=4)`, `hash`.
- `LLMClient.classify(units: list[UnitPayload], taxonomy_ctx, tier) -> list[UnitResult(uid, leaf_id, confidence, spend_class, rationale)]`; it never raises on per-batch failure (returns `None` results plus an error count).
- `Stage` signature: `(ctx, df) -> df`; never drops rows; adds documented columns only.
- Output columns exactly: `record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence`.

## Testing pyramid
- Unit (pytest, synthetic): parsers, FX chain, VAT, dates, legal-form stripping, taxonomy formats, validator, cascade logic with a mocked LLM.
- Integration: synthetic C end to end; A/B with `--no-llm --limit 3000`.
- Evaluation (real data, not CI): `spendcube eval all`, written to `runs/eval_<ts>/metrics.json`. Keep a short table of results over time in `EVAL_LOG.md` (gitignored if it quotes data; numbers only are fine to commit).

## Benchmarks to report (APPROACH.md)
A zero-label L1/L2/L3 (count- and spend-weighted); B zero-label vs seeds; A2-sim supervised; supplier/item F1; synthetic matching P/R; runtime per 25k and 50k lines; LLM $ per 1,000 lines (top/tail models).

## Blind-test runbook (Thu 19 Nov, IST) — rehearse on Wed
| Time | Action |
|---|---|
| Wed 13:30 | Gemini quota day for the blind test starts: blind-day key untouched from here |
| 09:00 | Freeze tag pushed; `spendcube doctor` (keys, models reachable, **remaining free quota per chain entry**); Ollama running with the model pulled; cache warm |
| 10:00 | Download the data into `blind/` (gitignored). Inspect file names |
| 10:05 | Start A2: `run --config configs/A.yaml --input-dir blind/A2 --suffix A2 --out submission/` and B2 likewise (background, logs to runs/) |
| 10:05–10:25 | `profile --file blind/C/<file> --taxonomy blind/C/<tree> --draft-config submission/C.yaml`; edit the mapping, status map, rules, industry hint |
| 10:25 | Smoke: `run --config submission/C.yaml --no-llm --limit 2000`; validate; check the tree summary and exclusion rates |
| 10:35 | Full C run (deadline guard 60 min) |
| ~11:20 | Validate all three; read the report's top-50 spend units; sanity-check totals and exclusion rates |
| 11:35 | `git add submission/ && git commit && git push`; verify on GitHub. Confirm `git diff freeze..HEAD -- spendcube/` is empty |
| 12:00 | Hard deadline |

Fallback: if the C run is still going at 11:30, the deadline controller should already have filled the tail. If it hasn't finished, re-run with `--no-llm` (cache hits for completed units still apply) and submit.
