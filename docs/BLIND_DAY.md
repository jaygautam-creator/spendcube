# Blind-day runbook (Thu 19 Nov 2026, freeze at 10:00 IST; the blind keys are touched only here)

At 10:00 IST, start `scripts/blind_day.sh ab blind` in terminal 1. Meanwhile profile the C files and write/review
`submission/C.yaml`; once it is ready, run `scripts/blind_day.sh c blind` in terminal 2. Run `scripts/blind_day.sh summary`
for a later status check. These commands run in a human-operated normal terminal, not a sandbox, because Gemini calls
need network access. Update the default C deadline only if the blind-day calendar changes.

## 0. 09:00 IST: doctor (the only thing allowed to touch `GEMINI_API_KEY` before the runs)
```sh
python -m spendcube doctor --blind          # one tiny call per model; prints remaining RPD per model and the 3 job budgets
```
Expect (empty ledger): C 640, B2 50, A2 0 requests (pool = both Flash-Lites under the 70% cap = 700; C keeps a 60 reserve). A model marked FAIL is skipped automatically by the router.
If `doctor` shows < 2 healthy Flash-Lite entries, leave `classification.llm.budget.share` alone: the router already
finishes on the supervised/propagation/offline tiers when a budget or a model runs out.

## 1. C prep (in parallel with B2/A2)
```sh
python -m spendcube profile --file blind/C/<lines> --taxonomy blind/C/<tree> --draft-config submission/C.yaml
$EDITOR submission/C.yaml        # fix TODOs; uncomment the `# SUGGESTED` exclusion rules that apply
python -m spendcube run --config submission/C.yaml --no-llm --limit 2000 --out /tmp/smoke      # < 1 min sanity
```

### C config checklist (found in the 18 Nov live rehearsal; ~3 min of edits)
1. `industry_hint`: one line from the tree's L1/L2 names and the top suppliers/descriptions.
2. `po_reference`: the profiler can pick a *type* column (e.g. `po_type`); map the PO **number** column.
3. `context` / `llm_context`: the profiler may suggest none. Give the LLM 2-4 low-cardinality business columns
   (document type, PO type, supplier type, GL description is already used). Never ids or amounts.
4. `amount_basis`: net unless a tax column or gross wording exists; if gross set the tax rate column.
5. `status`: map it if any status/approval column exists (CANCELLED / NOT_APPROVED); else leave it out.
6. `reporting_currency: USD` (brief); FX file glob if a rate table ships with C.
7. SUGGESTED exclusion rules: read each one. They are start-anchored template phrases (tax, intercompany, ...);
   delete any that hits real supplier spend.
8. Keep `include: ["../configs/_llm.yaml"]` (path is relative to `submission/`); do not set `classification.llm` keys
   other than `enabled` (the include is deep-merged).
9. Smoke: `--no-llm --limit 2000`, then the full `blind_day.sh c` run.

From the 18 Nov read-only audit (synthetic C variants; no crashes, these need a config edit):
10. `record_id`: if the line key has an unusual name (e.g. `invoice_line_key`) the draft may fall back to row numbers.
    Map the real line-id column: output ids must equal the file's line ids.
11. Dates: if the profiler reports ambiguous day/month, look at the raw values; set `dayfirst: false` for US
    month-first files (it defaults to day-first).
12. A tax **amount** column (money, not %) maps to `tax_amount`, never `tax_rate`; set `amount_basis: gross`.
13. Currency codes in lower case are not detected by the profiler; map the `currency` column by hand (the run
    upper-cases codes). Non-USD lines with no rate and no FX file get `FX_MISSING` and 0 spend: add `fx.static_rates`.
14. Excel with title rows above the header is handled (the header row is found automatically; a warning says so).

## 2. At 10:00: run B2/A2 and C as separate commands
The `ab` command runs B2 (max 60 LLM requests) then A2 (0 requests). The `c` command validates the completed config,
computes its LLM budget from the remaining ledger, and defaults to an absolute `2026-11-19T11:20` IST cutoff.
At the cutoff, the pipeline finishes lower tiers, writes output, and validates it; `deadline_minutes` remains a secondary guard.

Sanity before the full run: validation line OK, excluded share plausible, no `FX_MISSING`, taxonomy sample paths look real.

## 3. B2/A2 budgets
B2 is capped at 60 requests and A2 at 0 (`max_requests` in the configs; the curves show B's knee at 25-60 and A flat: the LLM hurts A2). Recurring units are
answered from `.cache/unit_memory.sqlite` (built by the A/B build runs; same taxonomy only) before any request.

## 4. After each run
`python -m spendcube validate --file submission/jaygautam__<X>.csv ...` is run automatically; check `runs/<ts>_<X>/report.md`
for: budget spent, LLM requests, methods share, flags. Never edit code after 10:00 IST.

## 5. Commit and push (public repo, branch `main`)
Only `submission/` files are committed after the freeze (the last code commit is before Thu 10:00 IST). Never merge or cherry-pick from
`local-history` or any `worktree-*` branch.
```sh
git add submission/jaygautam__C.csv submission/jaygautam__B2.csv submission/jaygautam__A2.csv submission/C.yaml
git commit -m "blind-day submission outputs"
git push                       # .githooks/pre-push must pass; never --no-verify
```
Push before 12:00 IST. If the hook blocks, fix the cause (a data string, key pattern, or extra tracked file) and push again.

## 6. Outage behaviour
If every Gemini entry fails or quota is 0 the run switches by itself to the local model (D-51; top-spend units within 20 minutes, offline
fallback for the rest) and writes `outage_switch` and a loud warning in the report. No flag is needed. Groq is never used for classification.

### Mock rehearsal (local only)
The checked-in script supports `OUT=...` to keep outputs away from `submission/`, plus `MODE=dev` and `PY=...` overrides.
The 2026-11-18 rehearsal used 25% A/B slices and the synthetic C fixture; all three files validated and the summary returned GO.
