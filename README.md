# spendcube

Config-driven spend cube pipeline: ingest and normalize purchase data, exclude non-spend with reason codes, reconcile and deduplicate records, resolve suppliers/items, classify into a run-time taxonomy, and validate one output row per input record.

## Setup

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; leave keys blank for offline/local operation
```

No task data is committed. Point each config at local files or pass `--input-dir`.

## Run

```sh
python -m spendcube run --config configs/A.yaml
python -m spendcube run --config configs/B.yaml
```

The default output paths are `outputs/jaygautam__A.csv` and `outputs/jaygautam__B.csv`. Useful options include `--input-dir DIR`, `--suffix A2`, `--out DIR`, `--limit N`, `--no-llm`, and `--llm-mode dev|blind`. Without API keys or an available Ollama model, classification falls back to the offline tier.

## Keys and providers

Keys are optional and read from `.env` or the environment. `GEMINI_API_KEY_DEV` is for development; blind runs select `GEMINI_API_KEY` with `--llm-mode blind`. See `.env.example`. `python -m spendcube doctor --list-only` checks key presence/model metadata without generation calls; ordinary `doctor` makes small provider checks. No paid key is required.

## Evaluation and validation

```sh
python -m spendcube validate --config configs/A.yaml --output outputs/jaygautam__A.csv
python -m spendcube eval a2-sim --config configs/A.yaml
python -m spendcube eval b-seeds --config configs/B.yaml
python -m pytest -q
```

## Blind-day runbook

See [docs/BLIND_DAY.md](docs/BLIND_DAY.md). On the blind day, run `scripts/blind_day.sh ab <blind_dir>` first, prepare and review `submission/C.yaml`, then run `scripts/blind_day.sh c <blind_dir> [deadline]`. Run these commands in a normal terminal with network access. `scripts/freeze_check.sh` is the pre-tag check.

## Layout

- `spendcube/`: CLI, ingestion, normalization, matching, entity resolution, classification and validation.
- `configs/`: company configs and shared model/calibration settings.
- `tests/`: unit and synthetic end-to-end tests; fixtures contain synthetic data only.
- `scripts/`: blind-day, data-safety, freeze and evaluation helpers.
- `docs/`: architecture, decisions and operating runbooks.
