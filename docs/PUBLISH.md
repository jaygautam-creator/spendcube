# Publishing to GitHub (run these yourself; nothing here is automated)

The repository must be PUBLIC. It must contain code, configs, docs and our own outputs only, never task data.

## 1. Pre-push checklist (all must pass)
```sh
git status --short                                   # nothing unexpected staged or untracked
git ls-files | grep -Ei '\.(csv|xlsx|xls|parquet|jsonl|zip|pkl|joblib|sqlite|db)$'
#   expect only outputs/*.csv, submission/*.csv, tests/fixtures/** (synthetic)
git ls-files | grep -Ei 'Cantill_DataML_Task_Kit|taxonomy_|labels_|seed_labels|\.env($|\.)' | grep -v '^\.env\.example$'
#   expect no output
.venv/bin/python -m pytest -q tests/test_no_data_in_docs.py tests/test_guards.py   # data guard: must pass, not skip (kit present)
git ls-files -z | xargs -0 grep -InE '(AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{20,}|sk-[0-9A-Za-z]{20,}|(API_KEY|SECRET|TOKEN)=[^[:space:]]{8,})' | grep -v '\.env\.example'
#   secrets scan: expect no output
git log -p --all | grep -InE 'AIza[0-9A-Za-z_-]{30,}|gsk_[0-9A-Za-z]{20,}'   # history too: expect no output
scripts/fresh_clone_test.sh "$PWD" "$PWD/Cantill_DataML_Task_Kit"          # fresh clone + tests + A smoke run
```
If anything prints, stop and fix it before pushing (the public history is a fresh single commit, section 2a, so nothing old can leak).

## 2. (SUPERSEDED by 2a, do not use) Create the public repo and push the local history
```sh
gh auth status
gh repo create <your-user>/spendcube --public --source=. --remote=origin --description "Spend cube classification pipeline"
git push -u origin main
```
## 2a. Agreed plan: push a FRESH single-commit history (the local history is never rewritten or pushed)
The local `main` history stays as is. The public repo gets one commit made on an orphan branch, after the leak scan passes
on the tree that will be committed:
```sh
git checkout --orphan main                           # new branch, no parents; working tree KEPT (not `git switch --orphan`, which empties it)
git add -A                                           # .gitignore keeps the kit, .env, caches and runs out
git status --short | grep -Ei '\.(csv|xlsx|xls|parquet|jsonl|zip|pkl|sqlite|db)$' | grep -Ev '^A  (outputs|submission|tests/fixtures)/' && echo "STOP: data staged"
.venv/bin/python scripts/leak_scan.py                 # prints nothing when clean
.venv/bin/python -m pytest -q tests/test_no_data_in_docs.py tests/test_guards.py
git commit -m "spendcube: code freeze snapshot"
gh repo create <your-user>/spendcube --public --source=. --remote=origin --description "Spend cube classification pipeline"
git push -u origin main
```
Because the pushed history is a single commit, the history scans above are not needed for it; the tree scan is. Tag the
freeze on this commit (section 3) and, on blind day, add the submission commit on top of `main` (section 4).

## 3. Set the freeze tag (Thu 19 Nov 2026, before 10:00 IST, on the final code commit)
```sh
git tag -a freeze -m "Code freeze 19 Nov 2026 10:00 IST"
git push origin freeze
git rev-parse freeze                                   # note this SHA in the submission
```
## 4. Blind-day push: submission files only
After generating `jaygautam__A2.csv`, `jaygautam__B2.csv`, `jaygautam__C.csv` into `submission/`:
```sh
git diff freeze..HEAD -- spendcube/                    # MUST print nothing (no code changes after the freeze)
git diff freeze..HEAD --stat -- configs/               # configs may only change if the task allows; ideally empty
git add submission/jaygautam__A2.csv submission/jaygautam__B2.csv submission/jaygautam__C.csv
git status --short                                     # only those files staged
git commit -m "Blind-day submission outputs"
git diff freeze..HEAD --name-only                      # only submission/ paths (plus docs if intended)
git push origin main
```
Never use `git add -A` / `git add .`: the kit is symlinked/present in the working tree. The pre-commit hook (`git config core.hooksPath .githooks`) blocks data files outside `outputs/`, `submission/`, `tests/fixtures/` and scans staged `.md/.yaml/.py` for supplier and taxonomy names.

## 5. Status (done 17 Nov 2026)
Public repo created from a fresh orphan `main` (the local pre-publish history is kept only on the local branch
`local-history` and is never pushed: 36 of its 52 commits contain data strings). From here on every commit on `main` is
pushed in order; `.githooks/pre-push` blocks any push with tracked data files, key patterns, or failing leak tests.
