# spendcube

A config-driven spend-analytics pipeline. It ingests a company's raw purchase files and category tree, cleans
amounts, dates and tax, decides what counts as spend (with reason codes), reconciles two source systems so each
purchase counts once, de-duplicates suppliers and items, and classifies every line to the deepest level of the tree
with a confidence. **A new company needs a new config file, not new code.**

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+ (developed on 3.13)
pip install -r requirements.txt
cp .env.example .env                                    # then fill in keys (all optional, see below)
```

No task data is in this repository. Put the company files wherever you like and point the config at them.

## The one command

```bash
python -m spendcube run --config configs/A.yaml            # -> outputs/jaygautam__A.csv
python -m spendcube run --config configs/B.yaml            # -> outputs/jaygautam__B.csv
```

Useful flags: `--input-dir DIR` (where the raw files are; A2/B2 re-use the unchanged A/B configs), `--suffix A2`,
`--out DIR`, `--no-llm` (fully offline), `--limit N` (smoke test), `--llm-mode dev|blind`.

Other commands: `validate` (re-check an output file), `doctor` (keys/models/quota), `profile` (profile a raw file and
draft a config), `eval zero-label|e2e|a2-sim|b-seeds|entities|matching-synth|blank|coherence`.

**Blind-day quota (D-42).** Jobs run one after another (C, then B2, then A2). Each job has a request budget taken from the
remaining free quota of the two Flash-Lite models (C takes all but 70 requests; B2 is capped at 60 and A2 at 10); when it is spent the
run finishes on the supervised/propagation/offline tiers. `python -m spendcube doctor --blind` prints the remaining
requests per model and the three budgets. Recurring units are answered from `.cache/unit_memory.sqlite` (same taxonomy only).

Output: one row per input record, in input order, columns
`record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence`. Every run ends with a
validated file (a stage that fails is logged and skipped, never fatal). Run artefacts (log, `report.md`, top
spend units) go to `runs/<timestamp>_<company>/` (git-ignored).

## A new company (the blind-test recipe)

```bash
python -m spendcube profile --file data/C/lines.csv --taxonomy data/C/tree.csv --draft-config submission/C.yaml
$EDITOR submission/C.yaml        # fix the TODOs: amount basis, static FX, status map, industry_hint; uncomment the
                                 # `# SUGGESTED` exclusion rules that apply (each shows row count and spend)
python -m spendcube run --config submission/C.yaml --no-llm --limit 2000   # smoke test (<1 min)
python -m spendcube run --config submission/C.yaml --out submission/        # full run
```

`configs/C.example.yaml` documents every key.

## LLM providers (all free)

Classification of an unseen tree uses an LLM; everything else is deterministic or supervised. Providers sit behind one
interface and are tried in a configured order (`configs/_llm.yaml`):

1. **Gemini free tier** (primary). `GEMINI_API_KEY_DEV` for development, `GEMINI_API_KEY` for the blind-day project
   (`--llm-mode blind`). Flash-Lite 3.5 is the workhorse, 3.1 the second opinion, 3.8 Flash compiles the taxonomy and
   arbitrates disagreements. **Disclosure:** the free tier lets Google use prompts for product improvement (data terms
   accepted by the author, D-35). Only description, supplier name, GL text and context columns are sent, never whole rows.
2. **Groq free** (`GROQ_API_KEY`), a different vendor, used if Gemini is unavailable.
3. **Ollama** (local, private, no key): `ollama pull qwen3:4b`; used automatically when its daemon is up.
4. **Offline fallback** (no key, no Ollama): TF-IDF similarity to the category paths plus supplier/GL propagation,
   clearly logged, confidence capped at 0.4.

`python -m spendcube doctor` prints key presence (set/missing, never values), checks the model ids and does one tiny
call per entry. Responses are cached (`.cache/`, git-ignored) so re-runs cost no quota.

## Tests

```bash
python -m pytest            # unit + synthetic end-to-end (mock LLM) + failure injection
```

Test fixtures are synthetic. A pre-commit hook (`.githooks/pre-commit`, `scripts/data_guard.py`) refuses data files
outside `outputs/`, `submission/` and `tests/fixtures/`; install with `git config core.hooksPath .githooks`.

## Layout

`spendcube/` ingest → normalize → exclusions → reconcile → entities → classify (cascade) → output; `llm/` providers,
router, limiter, quota ledger, cache; `configs/` per company; see `APPROACH.md` and `DECISIONS.md`.
