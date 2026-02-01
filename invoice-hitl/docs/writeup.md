# Invoice Extraction & Human-in-the-Loop Strategy (Local Prototype)

## 1. System Design

I treated this as a pragmatic extraction + triage problem, not a “train a model” problem. The deliverable is a runnable prototype that:
- takes invoice documents as input,
- outputs extracted fields in JSON,
- produces a confidence value for each field,
- flags invoices/fields for human review.
- A second part of the code which uses Doctr and Tkinter UI for allowing the Human In The Loop (HITL) intervention

### Key Challenges:
1. Partial or missing labels
    - Pipeline works without labels:
        - extraction + confidence + routing do not depend on ground truth.
    - If labels exist, evaluation is done optionally (non-blocking).
    - This matches real-world setups where only some invoices are annotated.
2. Noisy OCR
- I addressed OCR noise with multiple layers:
    - image preprocessing to stabilize OCR (resize, threshold).
    - token/avg confidence from OCR used as an uncertainty signal.
    - field-level sanity checks:
    - reject placeholder patterns,
    - reject “OCR blob” outputs for entity fields (too many invoice keywords),
    - require evidence substring for LLM outputs,
    - validate IBAN checksum.
- Result: noise lowers confidence and increases routing, rather than producing confidently wrong answers.
3. Layout variability
    - Rule extraction uses anchor keywords rather than fixed positions.
    - For entity fields that are layout-dependent, LLM fallback helps.
    - Evidence-based gating keeps it robust even when layout shifts.
4. Small data constraints
- No model training required.
- Rules + validators are data-light and can be improved iteratively.
- Confidence scoring is heuristic but explainable and tunable with thresholds.
- This is appropriate for a prototype where data is limited.

### Why LLM?
1. Role in the system
    - Only to fill missing entity fields after rules + guardrails.
    - Must return JSON in a strict schema with evidence.
    - Output accepted only if evidence is found in OCR text.
2. Why appropriate here
    - Adds resilience on variable layouts for “soft” fields (names/addresses).
    - Keeps critical fields (IBAN, totals, dates, invoice number) primarily rule and validation driven.
3. Limitations / reliability
    - LLM can still output wrong entities or over-select long spans.
    - OCR text itself may be wrong; evidence matching doesn’t guarantee correctness.
    - That’s why:
        - confidence is conservative,
        - failures route to human,
        - per-field “untrusted” tagging exists.
4. Cost considerations
    - Using Ollama locally avoids per-request API cost and privacy issues.
    - Tradeoff is latency on local hardware, especially if the model is large.
    - System minimizes calls by only invoking LLM when required.



### Pipeline: 

![Architecture](docs/images/Cloudfactory_task_architecture.png)
Part 1:
1. **Preprocess** (resize, grayscale, mild denoise)
2. **OCR** (Tesseract) → text + word confidence
3. **Extraction** (hybrid):
   - **Rules** (regex + anchors) for structured fields
   - **Local LLM** (Ollama) fallback for ambiguous fields and conflicts
4. **Validation + normalization** (IBAN checksum, numeric/date parsing)
5. **Confidence score** (triage)
6. **HitL routing decision** (field-level + invoice-level)
7. **Output JSON** with reasons for routing

Part 2:
1. **Doctr OCR** Using OCR model to extract
2. **HITL** Humans can enter missing json field values manually and save the updated json.

Part 1 was dockerized to mimic a production workflow. Part 2 was more of a quick implementation to implement HITL. 







Why OCR-first hybrid:
- With partial labels + tiny dataset, training/fine-tuning a doc model (e.g., Donut/LayoutLM) is risky for a time-boxed prototype.
- Rules are high precision for structured fields (IBAN/dates/totals).
- LLM acts as a selective fallback to improve robustness on messy layouts.


## 2. Key Decisions 

### Extraction Strategy

#### Part 1:
##### Rules (high precision)
- IBAN: regex + checksum validation
- Dates: anchor-based extraction with German date parsing
- Total: anchor-based amount parsing with German separators
- Invoice number: anchor-based patterns + fallback
- Phone: regex

##### LLM fallback (local via Ollama)
Used only for:
- company_name, company_address, bank_name
and only if the rule-based result is missing/weak.
The prompt requires:
- strict JSON output
- evidence quotes from OCR text
- null for unknown fields

This reduces hallucination risk and supports auditability.

#### Part 2 (Default precision from Doctr)
- Used the default extraction strategy used by the Doctr package


###  Confidence / Uncertainty
#### Part 1.
Confidence is used as a triage score (not a calibrated probability).
Signals:
- OCR confidence over evidence tokens
- Validation pass/fail (IBAN checksum, numeric/date parsing)
- Method prior (validated rules > rules > LLM)
- Conflict penalties (e.g., multiple total candidates)

#### Part 2:
- Doctr's confidence is used.
- The average confidence on the Tkinter app is the average of the fields successfully populated. 


## 3. Human-in-the-Loop Routing
### Part 1:
Field-level routing:
- route if missing, validation fails, or confidence < threshold[field]
Invoice-level routing:
- route if any critical field (IBAN/total/invoice_number) is routed
- route if average OCR quality is low
- route if too many fields are missing

Every routed decision includes reason codes (e.g., failed_iban_checksum, low_ocr_quality).

### Part 2
- The images which are labeled as routed to HITL are passed through this code segment
- Here, the Tkinter UI allows the annotators to manually enter the missing JSON field values and save the updated JSON. 
- Human corrections can be stored in a training dataset and used to retrain the model.

## 4. Evaluation (Partial Labels)
Evaluation is performed only where ground-truth exists:
- exact match for canonical fields
- tolerance for totals
- fuzzy token match for name/address/bank fields
We also measure routing tradeoffs:
- auto-accept rate vs accuracy on auto-accepted subset

## 5. Limitations
- OCR quality limits extraction
- Multiple totals (net/gross/subtotal) can cause ambiguity
- LLM outputs can be unreliable without strong evidence gating
- No model training/finetuning taking place

## 6. Future Plan (Production with APIs)
To scale:
- Use more data to train the OCR models
- Replace Ollama with an API LLM for fallback cases only
- Use HITL to updated missing JSON field values using Tkinter app
- Store corrected JSONs for further re-training of models
- Calibrate confidence using a labeled validation set (isotonic/temperature scaling)
- Monitor drift (layout/vendor changes) and re-train or update prompts/validators
- Add audit logs, rate limits, and cost controls (LLM budget per invoice)

