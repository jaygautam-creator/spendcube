# Blind-day runbook (Thu 19 Nov 2026, freeze at 10:00 IST; the blind keys are touched only here)

At 10:00 IST, start `scripts/blind_day.sh ab blind` in terminal 1. Meanwhile profile the C files and write/review
`submission/C.yaml`; once it is ready, run `scripts/blind_day.sh c blind` in terminal 2. Run `scripts/blind_day.sh summary`
for a later status check. These commands run in a human-operated normal terminal, not a sandbox, because Gemini calls
need network access. Update the default C deadline only if the blind-day calendar changes.

## 0. 09:00 IST: doctor (the only thing allowed to touch `GEMINI_API_KEY` before the runs)
```sh
python -m spendcube doctor --blind          # one tiny call per model; prints remaining RPD per model and the 3 job budgets
```
Expect (empty ledger): C 630, B2 60, A2 10 requests (C takes everything except the 70 kept for B2 and A2). A model marked FAIL is skipped automatically by the router.
If `doctor` shows < 2 healthy Flash-Lite entries, leave `classification.llm.budget.share` alone: the router already
finishes on the supervised/propagation/offline tiers when a budget or a model runs out.

## 1. C prep (in parallel with B2/A2)
```sh
python -m spendcube profile --file blind/C/<lines> --taxonomy blind/C/<tree> --draft-config submission/C.yaml
$EDITOR submission/C.yaml        # fix TODOs; uncomment the `# SUGGESTED` exclusion rules that apply
python -m spendcube run --config submission/C.yaml --no-llm --limit 2000 --out /tmp/smoke      # < 1 min sanity
## 2. At 10:00: run B2/A2 and C as separate commands
The `ab` command runs B2 (max 60 LLM requests) then A2 (0 requests). The `c` command validates the completed config,
computes its LLM budget from the remaining ledger, and defaults to an absolute `2026-11-19T11:20` IST cutoff.
At the cutoff, the pipeline finishes lower tiers, writes output, and validates it; `deadline_minutes` remains a secondary guard.
```
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

## 5. Outage behaviour
If every Gemini entry fails or quota is 0 the run switches by itself to the local model (D-51; top-spend units within 20 minutes, offline
fallback for the rest) and writes `outage_switch` and a loud warning in the report. No flag is needed. Groq is never used for classification.
