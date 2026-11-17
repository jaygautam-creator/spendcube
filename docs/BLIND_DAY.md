# Blind-day runbook (Thu 19 Nov 2026, freeze at 10:00 IST; the blind keys are touched only here)

Order is fixed: **C (50 points) -> B2 -> A2**, strictly one after the other (the quota ledger carries each job's spend
to the next one). Only a new config may change; no code.

## 0. 09:00 IST: doctor (the only thing allowed to touch `GEMINI_API_KEY` before the runs)
```sh
python -m spendcube doctor --blind          # one tiny call per model; prints remaining RPD per model and the 3 job budgets
```
Expect (empty ledger): C 630, B2 60, A2 10 requests (C takes everything except the 70 kept for B2 and A2). A model marked FAIL is skipped automatically by the router.
If `doctor` shows < 2 healthy Flash-Lite entries, leave `classification.llm.budget.share` alone: the router already
finishes on the supervised/propagation/offline tiers when a budget or a model runs out.

## 1. C (unseen company, 120-minute window)
```sh
python -m spendcube profile --file blind/C/<lines> --taxonomy blind/C/<tree> --draft-config submission/C.yaml
$EDITOR submission/C.yaml        # fix TODOs; uncomment the `# SUGGESTED` exclusion rules that apply
python -m spendcube run --config submission/C.yaml --no-llm --limit 2000 --out /tmp/smoke      # < 1 min sanity
python -m spendcube run --config submission/C.yaml --llm-mode blind --out submission
```
Sanity before the full run: validation line OK, excluded share plausible, no `FX_MISSING`, taxonomy sample paths look real.

## 2. B2 then A2 (unchanged configs, new input)
```sh
python -m spendcube run --config configs/B.yaml --input-dir blind/B2 --suffix B2 --llm-mode blind --out submission
python -m spendcube run --config configs/A.yaml --input-dir blind/A2 --suffix A2 --llm-mode blind --out submission
```
B2 is capped at 60 requests and A2 at 0 (`max_requests` in the configs; the curves show B's knee at 25-60 and A flat: the LLM hurts A2). Recurring units are
answered from `.cache/unit_memory.sqlite` (built by the A/B build runs; same taxonomy only) before any request.

## 3. After each run
`python -m spendcube validate --file submission/jaygautam__<X>.csv ...` is run automatically; check `runs/<ts>_<X>/report.md`
for: budget spent, LLM requests, methods share, flags. Never edit code after 10:00 IST.

## One-shot alternative
`scripts/blind_day.sh blind` runs the three jobs above in order (C, B2, A2); a failure in one is logged and the next still runs.

## 4. Commit and push (public repo, branch `main`)
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
