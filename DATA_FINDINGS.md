# DATA_FINDINGS.md — What the supplied materials actually contain

Measured facts from profiling the kit (Opus architect session, 2026-11-17). Every number below was computed from the files; nothing is guessed. Other documents reference this one as the factual baseline. **These facts are about A and B only. Nothing here may be hard-coded into the pipeline; it informs configs, defaults and tests.**

Kit location (gitignored): `Cantill_DataML_Task_Kit/`

| File | Rows | Notes |
|---|---|---|
| company_A/erp_goods_receipts.csv | 37,029 | 26 cols, SAP goods receipts, ZAR net |
| company_A/invoice_tool_export.csv | 28,520 | 11 cols, AP tool, ZAR gross |
| company_A/labels_A.csv | 37,029 | one row per ERP line (`ERP:<line_id>`); no tool labels, no include flag |
| company_A/taxonomy_A.csv | 40 | 5 L1 / 24 L2 / 40 L3 |
| company_B/invoice_lines.csv | 41,028 | 28 cols, 8 ERPs (JDE+SAP), 25 currencies |
| company_B/seed_labels.csv | 3,000 | include, exclusion_reason, L1–L3 |
| company_B/fx_rates_usd.csv | 683 | currency, month (YYYY-MM), units_per_usd |
| company_B/taxonomy_B.csv | 40 | 9 L1 / 25 L2 / 40 L3 (one L2 contains commas, quoted) |
| output_template.csv | 2 | `record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence` |

---

## 1. Blind-set construction (inferred)

- **A ERP line_ids run from A000002 to A045000, but only 37,029 exist.** 7,971 ids (17.7%) are missing. They're spread evenly across the range in short runs (gap sizes: 1×2604, 2×697, 3×257…). **Almost certainly A2's ERP records are these held-out ids.**
- A tool ids are contiguous (TA000001–TA028520). B ids are contiguous (B000001–B041028). A2 and B2 tool and B records will therefore have new ids. Do not assume an id range.
- 24.5% of tool rows carry a PO that is not in the current ERP file: 3,402 POs, ~5.7k rows. PO numbers interleave with ERP POs, so these are very likely invoices for the held-out (A2) receipts. **Implication: invoice↔receipt pairs were not kept inside one split, so A2 tool rows may reference A's ERP receipts.** This is the basis for history-aware matching (DECISIONS D-02).

## 2. Company A — ERP goods receipts

- **Two extract shapes are mixed in one file.** 24,370 rows (66%) lack `posting_date`, `vendor_id`, `uom`, `po_uom`, `site_code`, `unit_cost_zar`, but still have `supplier_name`, `site_name` and `value_zar`. The other 12,659 rows are complete. Never rely on `vendor_id` or `posting_date` being present.
- `value_zar` is always present and numeric. It's already in ZAR, even where `document_currency` is EUR (13), CHF (2) or MZN (1). 434 values are negative (returns/reversals); 509 are exactly 0.01 (placeholder value). Sum = ZAR 2.553bn.
- `spend_type`: Material 17,959, Service 11,232, Free Text 5,551, Catalogue 2,287.
- `material_number` is blank on 15,816 rows. Those are exactly the rows whose golden `item_key` is blank.
- Material short texts vary only by decimal comma vs point (e.g. `WD:2,2M` vs `WD:2.2M`; 69 materials have >1 text). `M1000032` is a **free-text placeholder material with 622 distinct texts**.
- Texts are SAP-abbreviated (`<abbreviated material text 1>`, `<text 2>`, `<text 3>`), and some are meaningless (`No Object` 914, `<truncated text>` 806).
- No status column. Every ERP line carries a category label, so all ERP lines are treated as countable spend.
- 193 distinct `supplier_name` values; 94 vendor_ids (complete rows only).

## 3. Company A — labels (golden keys)

- **Categories**: every ERP line has L1–L3. Spend by L1: L1 categories P ZAR 1.21bn, Q 0.53bn, R 0.47bn, S 0.18bn, T 0.17bn. Leaf sizes range from 2 rows (leaf X, ZAR 4.4m) to 3,014 rows.
- **Supplier keys (70)**:
  - `<catch-all supplier key>` is a **catch-all bucket**. It holds 122 distinct supplier names (supplier P, supplier Q, supplier R…) and ~20.9k rows. None of those names ever appears under another key. It aligns with the sparse extract: 20,718 of its rows have no vendor_id. Read it as the client master's "unmapped" bucket, not a real supplier.
  - Only two other keys merge names: two pairs of names differing only by whitespace (one by a double space). Both merges come out of "remove whitespace" normalization.
  - 20 keys span 2–3 vendor_ids: one supplier can have several vendor accounts. No vendor_id maps to two keys.
- **Item keys (2,426 non-blank)**: ≈ `material_number` 1:1, with two exceptions:
  - `IG-06a8b34` is a catch-all of 192 materials (incl. M1000032), 2,661 rows.
  - Repair and transport items are merged **by text across material numbers** (`REPR;FAN ELEC;22 KW…` = 155 materials → 1 key; `TRNSP RD,PROD,HAZ;EXPL&DTNTR,…` groups).
  - 25 texts map to >1 item key (the same text on different materials). 95 of 2,426 item keys carry >1 category.

## 4. Company A — invoice tool export

- `invoice_date` blank on 18,660 rows (65%). Three formats: ISO `2024-03-29`, `dd/mm/yyyy` (verified day-first: 2,050 values have first field >12, none have second field >12), `dd-Mon-yy`.
- `po_reference` blank on 5,199 rows. Formats: `4510062123`, `PO 4510…`, `PO#4510…`, `4510…/10.0` (PO/item). Regex `45\d{8}` extracts every non-blank one.
- `gross_amount` has thousands commas (`5,266.63`) and negatives. VAT encodings: `15` 6,920, `0.15` 6,826, `15%` 6,652, blank 6,542, `0` 1,580. **`0` means 0% (zero-rated)**: with net=gross, 1,353 of those rows match an ERP amount; with 15% removed, only 66 do. Blank → 15% (rule).
- Status: Paid 19,837, Approved 8,180, Cancelled 188, On Hold 172, Rejected 143.
- 702 distinct supplier spellings (e.g. one supplier spelled with a stray space, in mixed case, with a legal suffix, or fully compacted). Casefold + strip legal suffixes (PTY/LTD/CC/SA) + **remove all whitespace** gives **186 cores**. Only 7 cores are absent from the ERP, and none of those 7 is a safe fuzzy match to an ERP name (a near-prefix name is a different company).
- Line text is ERP text with **OCR noise**: `DYNAM1TE`, `5MPL`, `1rcnedale`, `Ba1last`, case flips, spaces dropped (`ROPEAHR` vs `ROPE AHR`), truncation.
- **Re-exports** (the same invoice exported 2–3×): same invoice number, gross, quantity and supplier core, usually consecutive tool ids. Supplier spacing, PO format, OCR noise and even the invoice date differ between copies. An exact key (invoice-norm + supplier-core + gross + qty) finds ~2,359 extra copies. Some copies escape it (two different tool rows claiming the same ERP line).
- **Matching evidence**: for tool rows with a PO found in the ERP, 95% have an ERP line on that PO whose net amount (gross/(1+VAT)) matches to the cent. Quantity agrees in 99% of those matches. Bulk items (aggregates sold by weight) produce many receipts on one PO with near-identical amounts, so matching needs an assignment step, not "any match". Without a PO, supplier-core + exact net amount recovers a further ~3.9k matches.
- Prototype result (simple greedy): ~18.9k canonical tool rows matched to ERP, ~7.2k unmatched. Of the unmatched, ~1.4k have a PO that exists in the ERP. These are a mix of partial or aggregated invoices (same unit price, different qty), receipts held out to A2, and re-exports whose twin already took the ERP line.

## 5. Company B — invoice lines

- Sources: B4 JDE 11,000, B7 SAP 11,000, B8 SAP 5,003, B3 4,000, B6 3,500, B2 3,000, B1 2,025, B5 1,500. **No supplier names**, only `vendor_id` (3,995). 74 vendor ids appear in more than one system.
- **No status column.** Signal columns: `gl_account` (52), `gl_account_description` (70: `AP - Other`, `PEUS`, `AP - Metal`, `IC06`, `I/C Payable`, `IC Payables`…), `document_type` (17: PV, RE, CA, PD, Z6…), `po_type`, `supplier_type`, `material_number` (blank on 10,574; `MAT-1000007` is a generic placeholder on 10,484 rows), `plant`, `company_code`.
- **Descriptions**: blank 6,857, whitespace-only 9,838. **40.7% of lines (~37% of spend) have no description.** The rest are multilingual (EN/ES/FR/PT/RU/DE/SR), truncated and ` - `-joined (`Válvula, solenoide, Gas, segur - idad…`), with redaction tokens like `[PERSON]`.
- **Amounts**: 3 blank (currency `RCN`), 833 negative, 918 zero. 177 rows sit in fully reversed groups (+X/−X on the same invoice line).
- 1,066 rows are exact duplicates of another row on every column except line_id. Recurring invoices (doc type PR) repeat the same invoice number monthly. That is real spend.
- **Dates**: all ISO; posting dates 2023-12-31 … 2024-12-29. `invoice_date` has impossible outliers (2018, 2020, 2027); `due_date` likewise; `paid_date` is blank on 10,322 rows.
- **Currency/FX traps**:
  - `currency_rate` is blank on 20,023 rows; 3,343 of them are non-USD.
  - **1,279 USD lines carry a rate (~17.4, the MXN/USD level)** from JDE entities CO180/CO214/CO207/CO134. Per the rule, USD lines are already USD, so the rate must be ignored.
  - **64 EUR/CHF/ARS lines (entity CO180) carry the entity's MXN rate** (~18.6), not their own currency's. The amounts are genuinely EUR (a 12 kW heater at 1,558). The rate is implausible against the FX table by ~20×. Elsewhere, rate/table ratios sit at 0.95–1.03 (1st–99th pct).
  - The FX table covers 30 currencies over 2022-12 … 2025-12 **with gaps** (AED only 8 months). CAD is missing for 2022-05…07, which some invoice dates need.
- **Intercompany**: 75 vendors are "IC" vendors. 2,039 of their 2,040 rows post to IC-type GL descriptions (`IC06`, `IC05`, `I/C Payable`, `IC Payables`, `IC<nn>`). IC is ~USD 19.7m of ~USD 349m.
- **Tax lines**: descriptions that are tax-posting labels (state/local, sales, GST, accrued sales-and-use tax) (~458 lines, USD 0.25m). Not tax: lines that merely mention tax but buy a service are professional services.

## 6. Company B — seed labels

- 2,550 included, 450 excluded, **all NON_PROCUREMENT**. Seeds contain no CANCELLED, NOT_APPROVED or DUPLICATE.
- Excluded seeds are (a) IC GL descriptions, 100% excluded, and (b) tax-description lines on PEUS/PECA/FXW GLs.
- **No seed has a blank description.** The analyst labelled only described lines. This matches "category on clear-cut lines" in B2 scoring.
- Seeds keep reversal-pair rows (3) and zero-amount rows (49) as included.
- Sampling is not uniform: B7 1,061, B4 1,038, B3 327, B2 213, B5 200, B6 104, B8 37, B1 20.
- 39 of 40 leaves appear (one utilities leaf is absent). Count-heavy leaves: leaf X 878, leaf Y 378, leaf Z 311.
- Signal purity for category (weighted): vendor_id 0.78, material_number 0.80, gl_account 0.47, plant 0.55. 452 of 546 vendors in seeds are single-category.

## 7. Classification difficulty (measured baselines)

| Experiment | Result |
|---|---|
| A supervised, random 18% holdout, text only (TF-IDF word+char, LR) | 91.4% L3 acc |
| + supplier name | 98.4% |
| + material_type, spend_type, contract_number, cost_centre | **99.7%** (spend-weighted 99.9%) |
| A material_number lookup | 52.6% coverage, 94.3% accurate |
| **A zero-shot TF-IDF text vs category names** | **10.7% L3 (21% spend-weighted), 33.6% L1** |
| **B zero-shot TF-IDF vs category names** | **29.2% L3, 43.9% L1** |

Conclusions: A2 is solvable almost perfectly by supervised learning on labels_A. Lexical zero-shot is useless for an unseen tree, so C needs language-model reasoning (abbreviation decoding, multilingual text, domain knowledge).

## 8. Spend concentration (drives LLM budget allocation)

| Unique units (supplier+description) | units for 50% spend | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| A ERP: 5,458 units | 55 | 363 | 781 | 1,322 | 2,712 |
| B: 15,475 units | 59 | 480 | 1,159 | 2,107 | 5,164 |

C is scored spend-weighted. **Getting the top ~500 units right is worth more than everything else combined.**
