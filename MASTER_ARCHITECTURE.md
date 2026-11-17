# MASTER_ARCHITECTURE.md — Spend Cube Pipeline

Status: **architecture v2** (Opus architect, 2026-11-17; v2 = free LLM providers + distillation, see §8.4). Implementation: Sonnet session. Facts referenced as [F§n] point to `DATA_FINDINGS.md`. Decisions referenced as [D-nn] point to `DECISIONS.md`.

---

## 0. What wins this task

Score = A2 (30) + B2 (20) + C (50), produced by a **frozen** tool. The configuration is the only thing we can change after 10:00 IST Thursday. Three facts set the design:

1. **C is half the score**, classified into an unseen tree (≤4 levels) with no labels, weighted by spend. Lexical zero-shot gets ~10–30% [F§7]. The C classifier has to be an LLM that understands procurement language, abbreviations and many languages.
2. **Spend is extremely concentrated** [F§8]: ~60 units carry 50% of spend and ~500 carry 80%. Accuracy on the top few hundred units decides most of the C classification score. The design spends its strongest reasoning there.
3. **Code freezes before we ever see C.** Anything we might need to tune for C must be reachable from config: column mapping, formats, status vocabulary, exclusion rules, dedup toggles, industry hint, model, effort and budget. The tool must also degrade gracefully, never crash, and always emit a valid file.

A2 and B2 are mostly *data engineering correctness* (matching, VAT, FX, exclusions, keys) plus supervised classification. Labels make that almost solved: 99.7% on A [F§7].

**Design principle: deterministic where the rule is known; supervised where labels exist; LLM where knowledge is needed; propagation where structure carries information; and a validator that guarantees the contract.**

---

## 1. System overview

```
                         ┌──────────────────────────── config (YAML) ────────────────────────────┐
                         │ sources, column map, formats, status map, rules, labels, llm, output   │
                         └───────────────┬────────────────────────────────────────────────────────┘
                                         ▼
 files ──► [1 Ingest] ──► [2 Normalize] ──► [3 Status & Rule Exclusions] ──► [4 Reconcile & Dedup]
           readers,        amounts, dates,     status map, GL/desc/vendor      intra-source re-exports,
           canonical map   tax, FX, text       rules, learned rules            cross-source matching,
                                                                                history reference
                                         │
                                         ▼
                 [5 Entity Resolution]  suppliers → supplier_key, items → item_key
                                         │
                                         ▼
                 [6 Classification]  units → tiers: inherit → label lookup → supervised
                                     → LLM (spend-tiered) → propagation → offline fallback
                                     (+ LLM spend verdict feeds NON_PROCUREMENT)
                                         │
                                         ▼
                 [7 Assemble & Validate]  one row per input record, exact tree names,
                                          contract checks, run report, metrics
```

Every stage is a function `stage(ctx: RunContext, df: pd.DataFrame) -> pd.DataFrame`. Each one adds columns to the **canonical frame** and never drops rows. Stage outputs are checkpointed to `runs/<run_id>/<stage>.parquet` (gitignored), so a failed run can resume and every decision can be audited.

---

## 2. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Language | Python ≥3.11 (tested on local 3.14) | pandas/sklearn ecosystem; fastest path |
| Frames | pandas (string dtype on read; `dtype=str, keep_default_na=False`) | 50k rows is small; never let pandas guess types (leading zeros, "NA" strings) |
| Fuzzy | rapidfuzz | C-speed ratios, `process.cdist` for blocks |
| ML | scikit-learn: TF-IDF (word 1–2 + char_wb 3–5) + LogisticRegression, CalibratedClassifier when needed | 99.7% on A; trains in seconds; no GPU |
| LLM | **Free providers behind one interface**: Groq (primary), Cerebras (optional), Ollama local (private fallback), Gemini (opt-in, D-35); `httpx` + provider SDKs only where needed | User requires free access; zero-shot needs real language understanding; quota-aware router + distillation keep calls ≈100 per 50k lines |
| Config | YAML + pydantic v2 validation | Readable, typed, errors early |
| Cache | sqlite (stdlib) under `.cache/` | Resumable, deterministic re-runs, B2/A2 reuse build-time answers |
| CLI | `python -m spendcube <cmd>` (argparse or typer) | One command |
| Optional | sentence-transformers (multilingual-e5-small) for offline fallback only; Ollama daemon for local LLM | Not required; guarded by import / health check |

Deliberately avoided: heavy deep-learning dependencies in the critical path (8 GB RAM laptop, install risk), vector databases, orchestration frameworks, agent loops. The workload is batch classification and does not need an agent.

---

## 3. Configuration model (the only post-freeze lever)

One YAML per company. Paths support `${ENV:-default}` and `{input_dir}` / `{kit_dir}` placeholders and globs. CLI flags override `input_dir` and individual files. A2/B2 then run with **unchanged** A/B configs, and C's config can live in `submission/C.yaml` [D-20].

```yaml
company: C
candidate_name: jaygautam       # output file prefix -> jaygautam__C.csv
output_suffix: C
reporting_currency: USD
industry_hint: "Regional hospital group: clinical consumables, pharma, facilities"  # free text for LLM
taxonomy:
  file: "{input_dir}/taxonomy_C*.csv"
  format: auto            # wide | path | adjacency | auto
  path_separator: " > "   # for format=path
sources:
  - name: lines
    role: primary         # primary | secondary (secondary may duplicate primary)
    file: "{input_dir}/*lines*.csv"
    read: {delimiter: auto, encoding: auto, sheet: null}
    record_id: {column: line_id, prefix: ""}       # A uses prefix "ERP:" / "TOOL:"
    columns:
      supplier_name: supplier
      vendor_id: vendor_no
      description: [item_description, line_text]  # concatenated in order, blanks skipped
      material_id: item_code
      quantity: qty
      uom: unit
      amount: amount
      amount_basis: net          # net | gross
      tax_rate: vat_pct          # optional
      tax_amount: null           # optional (gross - tax)
      default_tax_rate: 0.15     # used when basis=gross and rate blank
      currency: curr             # or currency_constant: USD
      fx_rate: rate              # optional, per-line
      fx_rate_convention: units_per_reporting   # or reporting_per_unit
      invoice_date: {column: inv_date, dayfirst: true}
      posting_date: {column: post_date}
      status: status
      po_reference: po
      invoice_number: invoice_no
      gl_account: gl
      gl_description: gl_text
      cost_centre: cc
      context: [plant, material_group]  # extra columns shown to the LLM / features
    number_format: auto         # or {decimal: ",", thousands: "."}
    status_map:                 # matched case-insensitively after trim
      INCLUDE: [paid, approved, posted]
      CANCELLED: [cancelled, canceled, void, reversed]
      NOT_APPROVED: [rejected, on hold, submitted, disputed, draft, pending]
      default: INCLUDE          # unknown values: INCLUDE | NOT_APPROVED  (logged)
    intra_source_dedup: {enabled: false, keys: [invoice_number, supplier, amount, quantity, description]}
fx:
  files: ["{input_dir}/fx*.csv"]
  plausibility_band: [0.5, 2.0]   # line rate / table rate outside band -> use table
  static_rates: {}                # emergency per-currency rates written at blind time
exclusion_rules:                  # evaluated in order; first hit wins
  - {code: NON_PROCUREMENT, field: gl_description, regex: '(?i)^(IC\d+|I/C\b|IC Payables)'}
  - {code: NON_PROCUREMENT, field: description, regex: '(?i)^\s*(sales|state & local|gst input|accrued sales & use)\s+tax\b'}
vendor_propagation: {enabled: true, min_share: 0.9, min_rows: 3, codes: [NON_PROCUREMENT]}
labels: null                      # or {file, join_on, columns: {l1.., supplier_key, item_key, include, exclusion_reason}}
reconciliation: null              # A only (see §6)
classification:
  llm:
    enabled: true
    allow_training_use: true        # REQUIRED to enable Gemini free tier (D-35) - user's decision
    roles:                          # which chain entries serve which job (ids verified via `spendcube doctor` / models.list)
      compile:   [gemini_flash_38, gemini_4_hydrogen]
      top_spend: [gemini_flash_38, gemini_4_hydrogen, gemini_flash_36]
      bulk:      [gemini_flash_lite_35, gemini_flash_lite_31, gemma4_31b, groq_gptoss120b, ollama_local]
      second_opinion: [gemini_4_hydrogen, gemini_flash_lite_31, gemma4_31b]
      quality_boost: [gemini_paid_best]          # optional: top-N spend units on the strongest PAID model
    escalation:                     # free first; paid only on exhaustion/failure or for quality_boost
      paid_enabled: true            # active only if GEMINI_API_KEY_PAID is set
      max_paid_usd: 5.0             # hard cap per run, enforced by the token ledger
      quality_boost_top_n: 300      # 0 disables the boost
      on_free_exhausted: [gemini_paid_bulk, groq_gptoss120b, gemma4_31b, ollama_local]
      on_google_outage:  [groq_gptoss120b, groq_qwen, ollama_local]   # 5xx/connection failures across Gemini entries
    prices_usd_per_mtok: {}         # filled from Gemini's current pricing page at build time (input/output per model)
    chain:                          # seeds from user's AI Studio free tier (2026-11-17); live headers/429s win
      - {id: gemini_flash_38,      provider: gemini, model: "<gemini-3.8-flash id>",      rpm: 5,  tpm: 250000, rpd: 20,    key_env: GEMINI_API_KEY}
      - {id: gemini_4_hydrogen,    provider: gemini, model: "<gemini-4-hydrogen id>",     rpm: 5,  tpm: 250000, rpd: 20,    key_env: GEMINI_API_KEY}
      - {id: gemini_flash_36,      provider: gemini, model: "<gemini-3.6-flash id>",      rpm: 5,  tpm: 250000, rpd: 20,    key_env: GEMINI_API_KEY}
      - {id: gemini_flash_lite_35, provider: gemini, model: "<gemini-3.5-flash-lite id>", rpm: 15, tpm: 250000, rpd: 500,   key_env: GEMINI_API_KEY}
      - {id: gemini_flash_lite_31, provider: gemini, model: "<gemini-3.1-flash-lite id>", rpm: 15, tpm: 250000, rpd: 500,   key_env: GEMINI_API_KEY}
      - {id: gemma4_31b,           provider: gemini, model: "<gemma-4-31b id>",           rpm: 30, tpm: 16000,  rpd: 14400, key_env: GEMINI_API_KEY}
      - {id: gemini_paid_best,     provider: gemini, model: "<strongest paid Gemini id, e.g. Pro-class>", billing: paid, key_env: GEMINI_API_KEY_PAID, optional: true}
      - {id: gemini_paid_bulk,     provider: gemini, model: "<flash-lite id>",            billing: paid, key_env: GEMINI_API_KEY_PAID, optional: true}
      - {id: groq_gptoss120b,      provider: groq,   model: openai/gpt-oss-120b, optional: true, key_env: GROQ_API_KEY}
      - {id: groq_qwen,            provider: groq,   model: qwen/qwen3.8-27b,    optional: true, key_env: GROQ_API_KEY}
      - {id: ollama_local,         provider: ollama, model: "qwen3:4b", base_url: "http://localhost:11434", optional: true}
    quota_reset: "00:00 America/Los_Angeles"   # Gemini daily reset = 13:30 IST
    limits_note: "rpm/tpm/rpd/tpd are seeds; live values from rate-limit headers win. Verify on the account limits page."
    anchor_spend_coverage: 1.0      # with Gemini Flash-Lite capacity: LLM labels ALL described units (spend order)
    max_anchor_units: 20000         # distillation only fills what quota/deadline could not reach
    diversity_units: 300            # k-means medoids of the tail, so the distilled model sees its vocabulary
    tail_llm_threshold: 0.45        # distilled p below this -> LLM if quota remains
    self_consistency_top_n: 200     # cross-model second opinion on the biggest units
    batch_min: 15
    batch_max: 60                   # Flash-Lite: RPD-bound, so bigger batches save requests
    max_desc_chars: 160
    deadline_minutes: 60            # after this: stop LLM, fill with distilled/propagation/fallback
    llm_spend_verdict: true         # LLM may flag NON_PROCUREMENT (conf >= 0.85 only)
  supervised: {enabled: auto}      # on when labels present
  propagation: {enabled: true}
```

**Rule: if a behaviour could plausibly need to differ for C, it has a config key with a sensible default.** Sonnet must apply this rule whenever it implements a heuristic threshold.

---

## 4. Canonical frame (the core data contract)

All stages read and write this frame. Columns are added progressively; names are fixed.

| Column | Type | Stage | Meaning |
|---|---|---|---|
| `record_id` | str | 1 | output id (`ERP:A000002`, `TOOL:TA000001`, `B000001`) |
| `source` | str | 1 | config source name |
| `source_role` | str | 1 | primary / secondary / reference (history; never output) |
| `row_order` | int | 1 | original order, for stable output |
| `raw` | dict/str | 1 | (optional) original row json for audit |
| `supplier_raw`, `vendor_id` | str | 1 | |
| `desc_raw` | str | 1 | concatenated description fields |
| `material_id` | str | 1 | |
| `quantity` | float | 2 | |
| `amount_raw` | str | 1 | |
| `amount` | float | 2 | as stated (gross or net) |
| `tax_rate` | float | 2 | fraction (0.15) |
| `net_local` | float | 2 | net of tax, document currency |
| `currency` | str | 1/2 | ISO-ish, upper |
| `fx_rate_used`, `fx_method` | float, str | 2 | `identity / line / table_invoice_month / table_posting_month / table_nearest / static / missing` |
| `spend_rc` | float | 2 | net spend in reporting currency (before exclusion zeroing) |
| `invoice_date`, `posting_date` | date | 2 | parsed; NaT if impossible |
| `po_number`, `po_item`, `invoice_no_norm` | str | 2 | extracted/normalized |
| `gl_account`, `gl_desc`, `cost_centre`, `context` | str | 1 | |
| `desc_norm`, `desc_compact`, `desc_blank` | str, str, bool | 2 | see §5.4 |
| `supplier_norm`, `supplier_core` | str | 2 | see §7 |
| `status_raw`, `status_class` | str | 1/3 | INCLUDE / CANCELLED / NOT_APPROVED |
| `excl_code`, `excl_detail`, `excl_conf` | str, str, float | 3,4,6 | final exclusion decision |
| `dup_group`, `match_ref`, `match_type`, `match_score` | str, str, str, float | 4 | `erp`, `reexport`, `history`, `none` |
| `supplier_key`, `supplier_conf` | str, float | 5 | |
| `item_key`, `item_conf` | str, float | 5 | |
| `unit_id` | str | 6 | classification unit |
| `leaf_id`, `cat_conf`, `cat_method` | str, float, str | 6 | `inherit / label_lookup / supervised / llm / llm_sc / propagate_supplier / propagate_gl / fallback` |
| `l1..l4` | str | 7 | exact tree names, blank below leaf depth |
| `flags` | str | all | pipe-joined quality flags (`DATE_UNPARSEABLE`, `FX_RATE_IMPLAUSIBLE`, `AMOUNT_BLANK`, ...) |

Output mapping (stage 7): `record_id, include (excl_code empty→1), exclusion_reason (excl_code), supplier_key, item_key, spend (spend_rc if include else 0; rounded 2dp), l1..l4 (blank if excluded), confidence`. The confidence column is `cat_conf` when the record is included and `excl_conf` when it is excluded.

---

## 5. Stage design

### 5.1 Ingest
- Readers for CSV (delimiter sniffing `, ; | \t`; encoding attempts utf-8-sig → utf-8 → cp1252 → latin-1), XLSX (openpyxl), Parquet, JSON lines. Read everything as `str`.
- Column mapping by config. Columns missing from the file but named in config are a **hard error**; columns present but unmapped are carried into `raw`. A mapped column that is entirely blank produces a warning.
- `record_id` = prefix + id column. If ids are duplicated within a source, append `#n` and flag it (the output must still have one row per input record).
- `reference` sources (history: A's original files when running A2) are ingested with `source_role=reference`. They take part in matching and in label/alias learning but are never emitted.

### 5.2 Normalize
- **Numbers**: one robust parser. It handles `1,234.56`, `1.234,56` (config or auto per column, decided by majority pattern), `(123.45)`, `123.45-`, currency symbols, spaces and NBSP. A failed parse gives NaN plus `AMOUNT_UNPARSEABLE`, and spend falls to 0 (record still output).
- **Tax**: if basis=gross, `net = gross / (1 + rate)`. Rate parsing accepts `15`, `15%`, `0.15`, ` 15 % `. A value >1 is read as a percentage. Blank uses `default_tax_rate`. `0` stays 0% [D-04]. If `tax_amount` is configured, `net = gross - tax_amount` takes precedence.
- **Dates**: try ISO, then configured `dayfirst`, then a list (`%d/%m/%Y, %d-%b-%y, %d.%m.%Y, %Y%m%d, %m/%d/%Y`), then Excel serials. Validity window = config `valid_years` (default: [posting_year-3, posting_year+1] relative to the column median). Out-of-window dates become NaT with `DATE_IMPOSSIBLE`; the original is kept in `raw`.
- **FX** (reporting currency R) [D-06, D-07]:
  1. currency == R → rate 1 (`identity`), **ignoring any line rate**.
  2. A line rate exists and `rate / table_rate(currency, month)` is inside the plausibility band → `line`. If there's no table rate to compare with, accept the line rate unless it looks inverted (`1/rate` within band of any known table rate).
  3. Otherwise use the table rate for the invoice month, then the posting month, then the nearest month for that currency, then `static_rates`. With nothing available, spend = 0 and the record is flagged `FX_MISSING` (never silently converted at 1).
  4. `spend_rc = net_local / rate` for convention `units_per_reporting`.
- **Text**: `desc_norm` = NFKC, collapse whitespace, strip trailing ` -` separators, uppercase. `desc_compact` = `desc_norm` with all non-alphanumerics removed. `desc_blank` = empty after normalization or in a configurable null-token list (`NO OBJECT`, `N/A`, `-`, `PRODUCT`, `PARTS`, …). Null-token lines are treated as blank for classification but keep their raw text in `raw`.

### 5.3 Status & rule-based exclusions
- `status_class` from `status_map`. Unknown values go to `default`, each one logged with counts in the run report.
- Ordered `exclusion_rules` (regex/in/equals/prefix over any canonical field).
- **Learned rules from labels** (when labels include `include`/`exclusion_reason`, e.g. B seeds): for each categorical field (`gl_desc`, `gl_account`, `vendor_id`, `document_type`), a value with support ≥5 and ≥95% one exclusion code becomes a rule. Learned rules are written to the run report and ranked below config rules.
- **Vendor propagation**: if ≥`min_share` of a vendor's rows were excluded NON_PROCUREMENT by rules (and min_rows is met), all its rows become NON_PROCUREMENT (catches IC vendors whose GL text is ambiguous).
- The LLM spend verdict (stage 6) can add NON_PROCUREMENT later. It only does so for configured reason classes with confidence ≥0.85, and never overrides a config INCLUDE rule.
- **Precedence** [D-03]: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE. Status is intrinsic to the record; DUPLICATE only applies to records that would otherwise count.

### 5.4 Reconcile & dedup (A's two systems; generic secondary→primary)
Configured under `reconciliation` (see §6 for the A specifics). Four outputs per secondary record: `match_type ∈ {erp, reexport, history, none}`, `match_ref`, `match_score` and `dup_group`.

### 5.5 Entity resolution → §7. 5.6 Classification → §8. 5.7 Assemble → §9.

---

## 6. Reconciliation design (A: tool → ERP)

The problem has three parts: **re-exports inside the tool**, **tool↔ERP matching**, and **matching against history (A's original files) during A2**.

**Step R1 — features.** For each tool row: `po` (regex `45\d{8}`, configurable `po_regex`), `po_item` (`/(\d+)\.\d`), `net_local`, `quantity`, `supplier_core`, `desc_compact`, `invoice_no_norm` (uppercase alphanumerics; digits-only variant too) and `invoice_date`. For each ERP row: `po_number`, `po_item` (int), `value`, `quantity`, `supplier_core`, `desc_compact`.

**Step R2 — re-export clustering (intra-secondary).** Block on `(supplier_core, round(gross,2), quantity)`. Inside a block, link two rows if `invoice_no_norm` digits are equal **or** (`desc_compact` similarity ≥ 0.85 **and** dates are ≤45 days apart or missing). Connected components form `dup_group`. The canonical member is the row with the best status (Paid > Approved > others), then the lowest tool id. Other members become candidate `reexport` duplicates of the canonical row.

**Step R3 — candidate generation (secondary canonical → primary).**
- (a) Same PO (and the same PO item when both are known).
- (b) No PO, or the PO is not in the primary set: same `supplier_core` and |net − value| ≤ tol.
- (c) Optional: same supplier + same `desc_compact` + qty with amount within 1% (OCR'd amounts).

Tolerance: `abs ≤ 0.02 + 0.001·|value|` (configurable).

**Step R4 — scoring.** Use a hand-weighted additive score (no labels exist to train one):

```
amount_exact (≤tol) +4 | amount within 1% +1.5 | qty equal +1 | po_item equal +1
desc_sim (token-sort on compact text, 0..1) ×2 | supplier_core equal +1 | PO via route (a) +2
```

Accept if score ≥ `min_score` (default 5.5, tuned on synthetic injections, §11).

**Step R5 — assignment.** Greedy maximum-score one-to-one on (tool canonical, ERP line). Each ERP line can absorb one *invoice*. Re-export members inherit their canonical's match.

**Step R6 — orphan rescue.** A canonical tool row left unassigned whose best candidate ERP line is already taken by a tool row with the same `(supplier_core, amount, qty)` and desc_sim ≥0.8 is a re-export missed by R2. Mark it `reexport` of that row.

**Step R7 — aggregate matches (optional phase, behind a flag).** Same PO + item:
- several tool lines summing to one ERP value (split invoicing), or
- one tool line equal to the sum of 2–3 ERP lines (combined invoicing), within tolerance.

Bounded subset search (≤3 items, ≤30 candidates per PO). Mark the tool lines `erp` duplicates.

**Step R8 — history.** Repeat R3–R6 against `reference` primary rows (A's original ERP) for still-unmatched A2 tool rows, and R2 across A2 tool + original tool. Matches become `history` and get exclusion DUPLICATE [D-02].

**Outcome:** a tool row with `match_type ∈ {erp, reexport, history}` and `status_class=INCLUDE` gets `excl_code=DUPLICATE`, `excl_conf` = f(score). All ERP rows count. Tool rows that are INCLUDE with `none` count as tool-only spend.

**Inheritance:** a matched tool row inherits `supplier_key`, `item_key` and the category from its ERP counterpart. Re-exports inherit from their canonical row. This keeps groupings consistent for the scorer, and the ERP side has the cleaner text and the labels.

---

## 7. Entity resolution

### 7.1 Suppliers
1. **Normalize**: NFKD → strip accents → uppercase → remove punctuation. Remove legal-form tokens from a configurable multilingual list (PTY, LTD, LIMITED, CC, SA, GMBH, BV, NV, AG, LLC, INC, CORP, CO, LTDA, SRL, SPA, SAS, SL, OY, AB, AS, PLC, KK, "S A DE C V", …) and the token `THE`. `supplier_core` = the result with **all whitespace removed** (this reproduces the golden merges [F§3]).
2. **Exact core groups.**
3. **Vendor-id union**: rows sharing `(source system, vendor_id)` join one supplier, unless their cores are dissimilar (<0.5); that case is flagged as a master-data conflict and not merged.
4. **Conservative fuzzy merge** between core groups. Block by the first 3 characters, then also by a sorted-token key (handles word order). Merge if Jaro-Winkler ≥ 0.95 or (ratio ≥ 0.92 and |len diff| ≤ 3), and both cores are ≥ 5 characters. A generic "OCR equivalence" (0↔O, 1↔I↔L, 5↔S, 8↔B) is applied before comparing. Merges go to a star around the highest-spend group, with no transitive chaining beyond one hop. Every merge is logged with its similarity.
5. **Label alias map** (when labels carry `supplier_key`): core → majority golden key (purity ≥0.9), and vendor_id → golden key. Rows whose core or vendor id resolves get the golden key; this reproduces the catch-all [D-01]. Unresolved cores keep generated keys.
6. When there is no supplier name (B), `supplier_key = "V:" + vendor_id` (global, not system-scoped, since the same ids appear across systems [F§5]).
7. Key format: golden key if mapped, else `S:<canonical core of the highest-spend member>`.

### 7.2 Items
1. **Generic material detection**: a material id is *generic* if it has ≥ `generic_min_texts` (default 8) distinct `desc_compact` values and distinct/rows ≥ 0.2. This catches M1000032 and MAT-1000007. Generic ids are ignored as identity.
2. Non-generic material id → `M:<material_id>`.
3. No usable material id → description key. Text-cluster on `desc_compact` after OCR-equivalence and decimal-separator unification (`2,2M`≡`2.2M`). Exact compact equality forms groups; then a fuzzy merge (ratio ≥ 0.93, len ≥ 10, blocked by first 6 characters). Key `D:<hash(canonical compact)>`.
4. Blank description and no material → `U:<supplier_key>|<gl_account or ''>` [D-16].
5. **Linking to a primary item master**: tool rows matched to ERP inherit the ERP `item_key`. Unmatched tool rows are linked to the ERP item index by exact `desc_compact` (then fuzzy ≥0.92), which gives them that ERP item key. This is generic for any primary/secondary setup.
6. **Label alias map** (labels with item_key): material → golden key; also text → golden key when labels show one text spanning ≥2 materials under a single key (the REPR/TRNSP merges [F§3]).

### 7.3 Evaluation (built into the harness)
- Pairwise precision/recall/F1 and B-cubed against golden keys on A ERP rows (labels hidden when we test the unsupervised path).
- On A tool rows, pseudo-truth comes from matched ERP rows' golden keys. That tests resolution of *noisy typed names and OCR text*, which is exactly the C situation.

---

## 8. Classification

### 8.1 Taxonomy model
- `Taxonomy.load(config)` supports three input formats:
  - **wide**: level columns auto-detected by name pattern `l\d|level|cat` and ordering;
  - **path**: one column with a separator;
  - **adjacency**: id/parent/name.
- Nodes keep their **exact raw strings** for output; comparisons use stripped and casefolded copies. Leaves are nodes with no children; a ragged tree (blank deeper cells) produces shallower leaves. A node that has children and also appears as an explicit row with blank lower levels is still a valid leaf.
- Each leaf gets a stable short id (`T017`), and `path(leaf) -> [l1, l2, l3, l4]`, padded with blanks.
- Duplicate leaf names under different parents are fine because we work with ids, never names.

### 8.2 Classification units
A unit = unique tuple `(supplier_key, desc_compact or '∅', material_id if non-generic, gl_account, context-signature)`. Classification runs on units, and results are broadcast to rows. Units are ordered by **total |spend|**, descending. Expected units: 5–20k per 50k lines. Excluded rows are still classified when cheap (so a later exclusion change is easy) but are blank in output.

### 8.3 Tiered decision cascade (per unit)
1. **Inherit**: a matched secondary row takes its primary's classification (`cat_method=inherit`).
2. **Label lookup** (labels present): exact material id or exact `desc_compact` with label purity ≥0.9 and support ≥2.
3. **Supervised model** (labels present: A full, B seeds): TF-IDF word(1–2) + char_wb(3–5) over the description plus categorical tokens (`SUP_<core>`, `GL_<acct>`, `MT_<material_type>`, `CN_<contract>`, `CC_<cc>`, context columns), LogisticRegression (C≈10–20, class_weight none). Probabilities are calibrated on a holdout. Accept if p ≥ τ (default 0.80, tuned). Rows below τ go to the LLM, with the supervised top-3 passed along as hints.
4. **LLM zero/few-shot on anchors + distillation to the tail** (§8.4). In C, the LLM labels units covering ~95% of spend; a local model trained on those answers labels the rest.
5. **Propagation** for blank or uninformative units:
   - (a) the supplier's spend-weighted category distribution from already-classified units (accept when the top share is ≥0.6; conf = share × 0.9);
   - (b) else the GL account's distribution;
   - (c) else an LLM call with supplier name + GL description + context only (lower confidence cap 0.5).
6. **Offline fallback** (no API key / budget exhausted / deadline reached): similarity between unit text and *enriched* leaf descriptions (TF-IDF, plus optional embeddings), combined with propagation. Confidence is capped at 0.4.

### 8.4 LLM classifier design (free providers, v2 — 2026-11-17)

**Constraint change:** the user requires **free** LLM access, so there's no paid Claude API. Free tiers have small, changing daily caps and differ on data use (DECISIONS D-28, D-35). The LLM layer is therefore **provider-agnostic, quota-aware and frugal**. Its main lever is **distillation**: the LLM labels the high-spend "anchor" units, and a local model trained on those answers labels the long tail.

**Provider plan v3 (2026-11-17, from the user's Google AI Studio free-tier page).** Gemini's free tier is limited by *requests*, not tokens: Flash-Lite allows 15 RPM, 250K TPM and 500 RPD on each of two models, while the strongest Flash-class models allow only 20 RPD each. Allocate by role:

| Role | Models | Volume per 50k-line run |
|---|---|---|
| Taxonomy compilation | Gemini 3.8 Flash (fallback 4 Hydrogen) | 1–2 calls |
| Top-spend units (top ~1,000 units ≈ 85–90% of spend) | 3.8 Flash → 4 Hydrogen → 3.6 Flash, 60 units/call | ~17 calls |
| Cross-model second opinion on the top 300 | 4 Hydrogen / Flash-Lite 3.1 | ~5–6 calls |
| Bulk: every other described unit | Flash-Lite 3.5 + 3.1 in parallel, 60 units/call | ~230 calls (≈1,000 RPD available) |
| Overflow | Gemma 4 31B (14.4K RPD, 16K TPM), Groq, Ollama | as needed |
| Tail fill | distillation + propagation | 0 calls |

**Escalation policy (v4, 2026-11-17).**
- **Free Gemini first.** Escalate to a **separately billed Gemini project** (`GEMINI_API_KEY_PAID`; billing applies project-wide, so it can't be the same project) only when free entries are exhausted or keep returning 429s, or for the optional **quality boost**: the top `quality_boost_top_n` spend units on the strongest paid model, as tie-breaker/second opinion.
- The paid spend is capped by `max_paid_usd`, enforced from the token ledger and configured prices. On reaching the cap, paid entries are marked exhausted.
- **Groq** is the cross-vendor fallback when Gemini as a whole fails (5xx/connection errors on every Gemini entry for >2 min).
- The run report shows calls, tokens and $ per entry.

**Quota-day trap:** Gemini RPD resets at 00:00 Pacific = **13:30 IST**. The blind window (Thu 10:00–12:00 IST) falls in the quota day that starts **Wed 13:30 IST**. Use a separate Google Cloud project for blind day (limits are per project) and keep its key untouched until then. Development and bakeoffs use a different project. Total blind-day need ≈ 250–300 requests (C) + ~30 (A2/B2 cache misses), well under the cap.

**Thinking/reasoning tokens**: request the lowest thinking setting the Gemini model supports for bulk calls (they count against TPM and latency). Allow a moderate setting for compile and top-spend calls. Verify the parameter names in the current `google-genai` docs.

**Provider abstraction** (`spendcube/llm/providers/`). One interface for every backend:
```
LLMProvider.complete_json(system: str, user: str, schema: dict, model: str, max_tokens: int)
    -> ProviderResult(parsed: dict | None, raw_text: str, usage: {in, out}, rate_headers: dict, error: str | None)
```
Adapters, each optional and enabled only when its key or daemon exists:

| Adapter | Use | Notes |
|---|---|---|
| `groq` | **default primary** | OpenAI-style chat completions over HTTPS (`httpx`, or the `groq` SDK). Models from config, e.g. `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`. Read `x-ratelimit-*` headers. Verify JSON-schema support per model; else JSON mode + validation |
| `cerebras` | secondary | Free-trial credits / limits unclear: optional |
| `gemini` | **opt-in only** (D-35) | Google `google-genai` SDK with `response_schema`. Free ("unpaid") quota allows Google to use prompts for product improvement and human review, which conflicts with the confidentiality clause |
| `ollama` | **local, private, unlimited** fallback | `http://localhost:11434/api/chat` with `format=<json schema>`. A small model (≈4B, e.g. a Qwen3/Gemma-class 4B tag chosen by bakeoff) fits an 8 GB M2. Slow but free and offline; also satisfies the brief's "must run without our keys" |
| `anthropic` | dormant | Kept behind the same interface in case a paid key ever appears; not required |
| `mock` | tests, `--no-llm` determinism | |

**Router** (`llm/router.py`):
- Walks an ordered `chain` of `(provider, model)` entries from config.
- Each entry has a limiter (token bucket for RPM/TPM, counters for RPD/TPD) seeded from config and corrected live from rate-limit headers. A **quota ledger** persisted in sqlite counts tokens per provider/model per rolling 24 h, so build-time experiments can't silently eat blind-day quota.
- On 429 it honours `retry-after` if it's under 20 s; otherwise it moves to the next chain entry. Persistent quota exhaustion marks the entry exhausted for the run.
- On a malformed answer it makes one repair retry, then bisects the batch.
- Nothing raises out of the router. Units it can't place get `None` and drop to lower tiers.

**Anchor selection + distillation** (replaces "LLM everything"):
1. Order units by |spend|. **Anchors** = units in spend order until `anchor_spend_coverage` (default 0.95) or `max_anchor_units` (default 2,500), whichever comes first. Add a **diversity sample** of `diversity_units` (default 300): one medoid per k-means cluster over TF-IDF of the remaining (tail) units, so the local model sees the tail's vocabulary.
2. The LLM classifies anchors and the diversity sample. On A/B, measured spend concentration [F§8] means about 1.3–2.1k anchors cover 95% of spend.
3. **Distil**: train the TF-IDF+categorical LR (§8.3 tier 3 features) on the LLM-labelled units. Sample weight = LLM confidence × log(1+spend). Add one synthetic "document" per leaf from the compiled taxonomy definitions so every leaf has support. Predict the tail. Confidence = calibrated p × 0.85.
4. Tail units where the distilled model's p is below `tail_llm_threshold` go back to the LLM **only if quota remains**, in spend order.
5. Blank or uninformative units use propagation (§8.3 tier 5).

When labels exist (A/B), the distilled training set is labels ∪ LLM anchors, and the LLM is called only for units the supervised model is unsure about.

**Taxonomy compilation** (one call per tree, cached; strongest model in the chain). It produces per-leaf scope definitions (≤25 words), typical items and abbreviations, and sibling disambiguation, using the tree, `industry_hint` and a sample of the company's own descriptions (top 150 by spend + 100 most frequent). This output becomes the system prompt for every classification batch and the leaf documents for distillation.

**Batch prompt, token-frugal by design** (free-tier TPM is the binding constraint):
- Leaves are numbered `1..N`, one compact line each: `17|L1 > L2 > L3 > L4|definition`.
- Unit text is truncated to 160 characters. Supplier, GL and context are included only when non-empty. "Other descriptions from this supplier" (≤2) only for short or blank texts.
- Batch size is adaptive: `floor((TPM_budget − prompt_tokens) / per_unit_tokens)`, clamped to 15–40.
- Output is a compact JSON array: `{"r":[[uid, leaf_no, conf_0_100, spend_class_code], ...]}`. It's validated client-side (leaf_no in range, uid set exact, codes in enum). Provider JSON-schema enforcement is used where supported.
- Reasoning models: request low reasoning effort where the provider exposes it, because hidden reasoning tokens count against the token budget.

**Cross-model self-consistency** for the top `self_consistency_top_n` (default 200) spend units: a second answer from a *different* model in the chain, since model diversity beats resampling one model. On disagreement a third model or the strongest available model gets both candidates and decides. Confidence comes from agreement.

**Spend verdict**: the same call returns `spend_class` (procurement / tax / intercompany / payroll_people / financing_interest / revenue_customer / royalty_partner / other_np). It's applied only at confidence ≥85 and never against a config INCLUDE rule (D-17).

**Cache**: sqlite, key = `sha256(provider|model|prompt_version|taxonomy_hash|unit_payload)`, written before use; resume-safe. Build-time runs warm A/B units, so A2/B2 consume very little quota on blind day.

**Volume estimate (C ≈ 50k lines ≈ 15k units)**: about 2.8k anchor+diversity units in batches of about 35 gives ~80 calls. Add compile (1–2), self-consistency (~8), low-confidence tail (~10) and blank context (~5) for **≈105 calls × ≈4–5k tokens ≈ 0.45–0.55M tokens**. At the Groq Developer-plan figures observed on 2026-11-17 (~200k TPD and 8k TPM per model; free-plan numbers may be lower and **must be read from the account's limits page**), that means spreading across 2–3 Groq models and/or Cerebras. Ollama local covers the remainder; at roughly 20–30 output tok/s on an M2 it does ~1k units in ~15–25 min. **LLM cost: $0.**

**Throughput** is bounded by TPM, not latency: 8k TPM/model ≈ 1.5–2 calls/min/model. With 3 Groq models in parallel that's ≈5 calls/min, so ~105 calls take ≈20–25 min. The deadline controller (`deadline_minutes`) still applies; anchors are scheduled in spend order, so truncation only ever hits the tail.

**Quota hygiene (operational)**: separate keys/projects for *dev* and *blind day* if the provider allows. No heavy LLM experiments on the blind-day key after Wed 10:00 IST (rolling 24 h windows). `spendcube doctor` prints remaining quota per chain entry from headers or the ledger.

### 8.5 Confidence
- Inherit: primary conf × 0.98.
- Lookup: purity.
- Supervised: calibrated p.
- LLM: `0.5 × self-reported + 0.5 × agreement signal`, where agreement = self-consistency vote share, or supervised-hint agreement, or 0.7 if no signal. Clip to [0.05, 0.99].
- Propagation: share × 0.9 (cap 0.6).
- Fallback: ≤0.4.

The goal is a monotone relation to accuracy, checked with reliability bins in the eval harness.

---

## 9. Assemble, validate, report

- Every input record (primary + secondary, never reference) appears **exactly once**, in input order. Checks:
  - `include ∈ {0,1}`; `exclusion_reason` blank iff include=1, otherwise in the 4 codes;
  - `spend = 0` when excluded; `l1..l4` exactly equal to tree strings and blank below leaf depth or when excluded;
  - `supplier_key` and `item_key` non-empty for every row; `confidence ∈ [0,1]`.
- The file is named `<candidate_name>__<suffix>.csv`, UTF-8 and comma-delimited, with every field quoted when needed (category names contain commas).
- `spendcube validate --config ... --output ...` re-runs the checks standalone (used in the blind runbook).
- `runs/<id>/report.md`: counts by stage, exclusion counts by code, spend totals (gross in, net, excluded), FX methods, flags, unknown status values, learned rules, supplier and item merge stats, classification method mix, LLM tokens, $ and latency, plus the timings.

---

## 10. Generalization strategy (C)

| Unknown in C | How the design copes |
|---|---|
| Column names / file shape | config mapping + `spendcube profile` drafts a config (guesses roles by name and content patterns) |
| Number/date formats | auto-detection per column + config override; flags, never crash |
| Status vocabulary | config `status_map`; profiler lists distinct values; unknowns logged |
| Gross vs net, tax columns | `amount_basis`, `tax_rate`, `tax_amount`, `default_tax_rate` |
| Currencies without rates | FX chain + `static_rates` in config |
| Non-procurement patterns | config regex rules (profiler surfaces candidate GL/desc values containing tax/IC/payroll/interest/royalty/… keywords), vendor propagation, LLM spend_class |
| Unseen tree, 4 levels, ragged | `Taxonomy` with leaf ids; LLM sees full paths; compiled leaf definitions |
| Industry jargon / languages | LLM + taxonomy compilation from the company's own descriptions + `industry_hint` |
| Blank descriptions | supplier/GL propagation |
| Supplier variants | normalization + conservative fuzzy + vendor-id union |
| No labels | cascade starts at the LLM tier; no component *requires* labels |
| Scale (≤50k) | unit dedup + spend ordering + budget controller |

**Anti-overfitting guardrails**
- A test greps `src/` for any taxonomy string from the kit files and fails if one is found.
- Every company-specific regex lives in `configs/A.yaml` or `configs/B.yaml`.
- The "C rehearsal" (§11) runs A and B with labels disabled and generic configs.

---

## 11. Validation strategy (what we measure, and what we must not fool ourselves with)

1. **Zero-label rehearsal (C proxy)**: classify A ERP with labels hidden, and B against seeds with seeds hidden. Report L1/L2/L3 accuracy, both count- and spend-weighted. This is the number APPROACH.md must report ("accuracy you measured on A with labels hidden"). Caveat: our prompts were designed while looking at A/B, so expect C to come in somewhat lower. Do not tune prompt wording on A's specific errors; tune only generic instructions.
2. **A2 simulation**: hold out 18% of ERP ids in short runs (mimicking [F§1]) together with the tool rows whose PO lands in the held-out set. Train on the rest and evaluate classification and keys on the holdout.
3. **B seeds**: 5-fold CV for supervised, plus zero-shot. Also report exclusion precision/recall on seeds.
4. **Matching (no ground truth)**:
   - (a) **synthetic injection**: take 1,000 ERP lines, generate tool-like copies with the generators observed in [F§4] (VAT encodings, PO formats, OCR noise, spacing, re-exports, dropped POs). Mix in 300 fabricated tool-only lines and measure precision and recall;
   - (b) manual audit of 50 random matches and 50 random non-matches;
   - (c) stability: the share of tool rows whose PO is in the ERP but who stay unmatched should fall after each improvement.
5. **Entity metrics**: §7.3.
6. **Synthetic C fixture**: a small hand-made company with a 4-level ragged tree, semicolon delimiter, European decimals, `dd.mm.yyyy`, a status column in another language, gross amounts with a tax column, two currencies without a rate column, and supplier variants. It runs in CI end to end with the LLM mocked.
7. **Performance**: run a 50k-row synthetic scale-up (A tool+ERP concatenated with jittered copies) with the LLM enabled on a 10% sample and mocked for the rest. Measure wall clock and peak memory.

---

## 12. Repository layout

```
.
├── README.md                 # setup + the one command
├── APPROACH.md               # one page (required)
├── DECISIONS.md              # required
├── MASTER_ARCHITECTURE.md  DATA_FINDINGS.md  RISK_ANALYSIS.md  IMPLEMENTATION_PLAN.md
├── pyproject.toml / requirements.txt (pinned)
├── configs/  A.yaml  B.yaml  C.example.yaml
├── spendcube/
│   ├── cli.py  pipeline.py  context.py (RunContext: config, logger, timers, run dir)
│   ├── config/ schema.py loader.py
│   ├── ingest/ readers.py canonical.py
│   ├── normalize/ numbers.py dates.py tax.py fx.py text.py
│   ├── exclusions/ status.py rules.py learned.py propagation.py
│   ├── reconcile/ features.py reexport.py match.py aggregate.py
│   ├── entities/ suppliers.py items.py legal_forms.py
│   ├── taxonomy/ tree.py compile.py
│   ├── classify/ units.py lookup.py supervised.py llm.py propagate.py fallback.py cascade.py confidence.py
│   ├── llm/ client.py cache.py budget.py prompts.py schemas.py
│   ├── output/ writer.py validator.py report.py
│   ├── profile/ profiler.py draft_config.py
│   └── eval/ metrics.py harness.py synth.py
├── tests/  (pytest; synthetic fixtures only, under tests/fixtures/)
├── outputs/       # <name>__A.csv, <name>__B.csv (committed at build deadline)
└── submission/    # blind test: C.yaml + <name>__A2.csv, __B2.csv, __C.csv
```

CLI:
```
python -m spendcube run      --config configs/A.yaml [--input-dir DIR] [--out DIR] [--suffix A2] [--no-llm] [--limit N]
python -m spendcube profile  --file PATH [--taxonomy PATH] [--draft-config OUT.yaml]
python -m spendcube validate --config CFG --output CSV
python -m spendcube eval     <zero-label|a2-sim|b-seeds|entities|matching-synth|all> --config CFG
```

---

## 13. Performance budget (50k lines, laptop: 8 GB RAM, 8 cores)

| Stage | Target |
|---|---|
| Ingest + normalize | < 30 s |
| Rules + reconciliation (blocked) | < 2 min |
| Entities (blocked fuzzy) | < 2 min |
| Supervised train + predict | < 2 min |
| LLM (≈105 calls, TPM-bound across 3 Groq models + Ollama spill) | 20–35 min |
| Assemble + validate | < 30 s |
| **Total** | **< 50 min**, hard deadline guard at 75 min of LLM |

Peak memory < 2 GB. No O(n²) comparisons anywhere: every fuzzy step is blocked, with an explicit block-size cap and a logged overflow.
