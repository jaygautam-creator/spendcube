# Spendcube approach

Spendcube turns configured purchase files and a run-time taxonomy into one validated row per source record. The pipeline reads text fields, normalizes dates, amounts, tax and FX, assigns exclusion reasons, reconciles records, resolves suppliers/items, classifies included lines, and broadcasts unit-level results back to rows. Each stage retains rows; the final validator checks ids, tree paths, confidence and output shape.

## Config, money and exclusions

A new company needs a new YAML config, not new code. `spendcube profile` reads an unseen file and tree, guesses column roles, number/date formats, status values and currencies, and writes a draft config with commented, counted suggestions for exclusion rules (tax, intercompany, payroll, bank charges and similar). On B, uncommenting the suggestions reproduced the hand-tuned exclusions. Amounts are parsed per configured format, tax is removed where amounts are gross, and FX uses the line rate, else the invoice-month table rate, else the nearest month (flagged). Exclusions follow a fixed precedence: status (CANCELLED / NOT_APPROVED), then rules (NON_PROCUREMENT), then reconciliation (DUPLICATE), each with a reason code.

## Matching and entities

For two-source reconciliation, exact purchase-order evidence is strongest: a configured PO regex extracts the order and item, then net amount and supplier constrain one-to-one matches. Re-export duplicates are grouped using invoice/reference, date, supplier, amount and description signals. A history reference lets A2 lines match prior A records without emitting those reference rows. If PO evidence is absent, a constrained supplier-plus-amount core route is available; partial-quantity matching checks unit price and cumulative quantity capacity. Suppliers use label aliases where available, otherwise normalized names and guarded fuzzy/vendor joins. Items use material codes and normalized descriptions, suppressing generic descriptions before fuzzy links.

## Classification

A new tree is loaded from the config at run time; no category strings are embedded in code. With labels, a supervised TF-IDF classifier handles confident units. For a new company without labels, the spend-ordered cascade uses the configured LLM chain, supplier/context propagation, then an offline TF-IDF path fallback. Units are deduplicated by supplier, description, material, GL and context before classification. Confidence is calibrated by tier; the LLM value is a ranking signal with entry caps, not a universal probability. Flash-Lite pooling is enabled: 3.1-lite was within 3 points of 3.5-lite on spend-weighted L3 for the same 900 A units overall and in the top-150 spend band. Supplier coherence remains off; 3.8-flash arbitration is limited to top-100 disagreements.

## Measured results

On the same 900 A zero-label units, 3.5-lite spend-weighted L1/L2/L3 was 86.18/75.37/66.32%; 3.1-lite was 86.88/78.04/66.44%. Their count-weighted L3 was 45.56% and 42.78%. On the top 150 units, spend-weighted L3 was 76.68% vs 77.43%, supporting the pool default. In the top-50 supplier sample, coherence lowered spend-weighted L3 by 0.33 points and stays off. 3.8-flash arbitration raised it by 1.59 points over the available disagreement set; on the top-100 spend units it raised L3 by 0.84 without lowering L1, so C uses arbitration only for top-100 disagreements.

Matching and de-duplication on A: about 15.7k tool lines matched to ERP receipts and about 2.3k re-exports collapsed (tool net amounts agree with ERP on 95% of known pairs); supplier pairwise F1 0.998 and item F1 0.997 against the client masters. B exclusions matched the seed include flags with precision and recall 1.0.

B seed holdout (487 included rows) reached L1/L3 99.15/98.83% spend-weighted and 95.9/93.4% count-weighted. A2-sim supervised-only holdout (6,856 rows) scored 99.87/99.71/99.71% count-weighted L1/L2/L3 and 99.51% spend-weighted L3.

A full B run processed 41,028 lines in 18.9 minutes with 260 LLM requests. That is about 6.2 requests per 1,000 lines. At 640 C requests, measured single-model throughput implies 46.5 minutes of LLM work; the two-model pool projection is 23.3 minutes. A live rehearsal treating B as unseen (no labels, cold cache) scored 98.6/92.5% spend-weighted L1/L3 in 10.2 minutes.

## Tools and cost

The free tier costs $0 per 1,000 lines, with request volume depending on how many units the supervised and memory tiers resolve; the measured B baseline was about 6.2 requests per 1,000 input lines. AI assistance included Claude for architecture/review and Sonnet and OpenAI Codex for implementation. Gemini free tier is an optional classifier provider inside the tool. With no key and no local model, the pipeline uses the offline fallback. No task data is required in the repository.
