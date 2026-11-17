# Spendcube approach

Spendcube turns configured purchase files and a run-time taxonomy into one validated row per source record. The pipeline reads text fields, normalizes dates, amounts, tax and FX, assigns exclusion reasons, reconciles records, resolves suppliers/items, classifies included lines, and broadcasts unit-level results back to rows. Each stage retains rows; the final validator checks ids, tree paths, confidence and output shape.

## Config, money and exclusions

A new company needs a new YAML config, not new code. `spendcube profile` reads an unseen file and tree, guesses column roles, number/date formats, status values and currencies, and writes a draft config with commented, counted suggestions for exclusion rules (tax, intercompany, payroll, bank charges and similar). On B, uncommenting the suggestions reproduced the hand-tuned exclusions. Amounts are parsed per configured format, tax is removed where amounts are gross, and FX uses the line rate, else the invoice-month table rate, else the nearest month (flagged). Exclusions follow a fixed precedence: status (CANCELLED / NOT_APPROVED), then rules (NON_PROCUREMENT), then reconciliation (DUPLICATE), each with a reason code.

## Matching and entities

For two-source reconciliation, exact purchase-order evidence is strongest: a configured PO regex extracts the order and item, then net amount and supplier constrain one-to-one matches. Re-export duplicates are grouped using invoice/reference, date, supplier, amount and description signals. A history reference lets A2 lines match prior A records without emitting those reference rows. If PO evidence is absent, a constrained supplier-plus-amount core route is available; partial-quantity matching checks unit price and cumulative quantity capacity. Suppliers use label aliases where available, otherwise normalized names and guarded fuzzy/vendor joins. Items use material codes and normalized descriptions, suppressing generic descriptions before fuzzy links.

## Classification

A new tree is loaded from the config at run time; no category strings are embedded in code. With labels, a supervised TF-IDF classifier handles confident units. For a new company without labels, the spend-ordered cascade uses the configured LLM chain, supplier/context propagation, then an offline TF-IDF path fallback. Units are deduplicated by supplier, description, material, GL and context before classification. Confidence is calibrated by tier; the LLM value is a ranking signal with entry caps, not a universal probability. Flash-Lite pooling is available but defaults off until the same-unit quality comparison supports enabling it.

## Measured results

<!-- T4 --> A zero-label sample (300 units: top 150 by spend plus 150 sampled tail units) reached 3.5-lite L1/L2/L3 accuracy of 76.7/58.3/45.0% by count and 92.1/88.2/76.2% spend-weighted. The same sample's 3.1-lite spend-weighted L1/L2/L3 was 92.1/89.0/77.0%. A separate 900-unit zero-label run scored 66.2% spend-weighted L3 and 39.6% count-weighted L3. T4 will replace these figures with a same-900 comparison.

B seed holdout (487 included rows) reached L1/L3 99.15/98.83% spend-weighted and 95.9/93.4% count-weighted. A2-sim supervised-only holdout (6,856 rows) scored 99.87/99.71/99.71% count-weighted L1/L2/L3 and 99.51% spend-weighted L3.

A full B run processed 41,028 lines in 18.9 minutes with 260 LLM requests. That is about 6.2 requests per 1,000 lines. At 640 C requests, measured single-model throughput implies 46.5 minutes of LLM work; the two-model pool projection is 23.3 minutes. Mock rehearsal on the synthetic C fixture validated 3,000 rows in 122 seconds. <!-- T4 -->

## Tools and cost

The free tier costs $0 per 1,000 lines, with request volume depending on how many units the supervised and memory tiers resolve; the measured B baseline was about 6.2 requests per 1,000 input lines. AI assistance included Claude for architecture/review and Sonnet and OpenAI Codex for implementation. Gemini free tier is an optional classifier provider inside the tool. With no key and no local model, the pipeline uses the offline fallback. No task data is required in the repository.
