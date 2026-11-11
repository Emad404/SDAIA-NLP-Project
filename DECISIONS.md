# DECISIONS.md: why each model and setting was chosen

Rule used throughout: **choices are made on validation data only**; the frozen test split was opened once, after every choice below was fixed (`scripts/s08_final.py`, lock file `artifacts/final_run.lock`). Where the result disappointed, it is reported as such. Numbers come from `reports/*.json`; the generated tables are in `RESULTS.md` and `BENCHMARKS.md`.

## D0. Data handling

| Decision | Reason |
|---|---|
| Split grouped by `citizen_id` (train / val / frozen test) | The same citizen writes several messages; a random split would leak style and topics across splits and inflate every score. |
| Frozen-test guard in code (`BAYAN_FINAL=1`, SHA-256 of split ids, lock file) | `get_part` refuses to return test rows otherwise, so no training or selection script can touch them by accident. |
| One preprocessing module (`preproc.py`, version `1.0.0`), **PII masked first**, then normalisation | Masking before Unicode/Arabic normalisation means digits in any script (Arabic-Indic, Latin) are caught while they still look like digits; the version string is stored in every model's metadata and checked at server start-up. |
| Arabic model input uses a separate model-specific step (`model_input`) after the shared preprocessing | Keeps the shared module identical everywhere while letting a model family (XLM-R vs CamelBERT) have its own tokenizer-friendly form. |
| Case augmentation on **raw** text before masking | Lower/upper-casing after masking would turn `<PHONE>` into `<phone>` and break the placeholder. |

Known data limits that qualify every number: topic labels are ≈ 3–4% noisy and sentiment ≈ 8%; dialect labels are synthetic; NER sentences are templated; many messages are generic follow-ups with no topic cue.

## D1. Topic classifier

Candidates (validation macro-F1): XLM-R base on all data 0.8778; **XLM-R base + English case augmentation ("teacher", `topic_xlmr_all_ca`) 0.8817**; XLM-R large 0.8834; 4-layer student 0.8754; **3-layer student 0.8778**; XLM-R / CamelBERT Arabic-only models 0.864–0.871 on Arabic rows; TF-IDF word 0.7953, char 0.8024.

1. **Fine-tuned transformer over TF-IDF.** The brief asks for ≥ 8 points over TF-IDF. Validation margin: +8.6 (word) / +7.9 (char) for the teacher. On the frozen test the shipped student reaches +8.08 / +8.21 with 95% CI [6.86, 9.36] / [6.96, 9.56]. The point estimate clears 8; the interval does not, so this requirement is *met as a point estimate, not established*.
2. **Case-augmented teacher instead of plain XLM-R.** English UPPER-CASE messages were the weak spot of the invariance suite. Adding upper- and lower-cased copies of 35% of English training rows cost nothing on validation (+0.4 pt, 0.8817 vs 0.8778). The model still flips on 5.7% of UPPER-CASE inputs (invariance 0.943, below the 0.95 bar for that single test; the mean over all ten invariance tests is 0.990); this is reported, not hidden.
3. **XLM-R large rejected for serving.** +0.2 pt overall, but it was not served: the base-size teacher already has p99 = 141.6 ms at INT8 batch 1 on 2 vCPUs, and large is several times bigger. It is the only model that significantly helped the Gulf slice (+1.1 pt, 95% CI [0.19, 2.24]); that is a size effect, not a dialect effect.
4. **Ship a 3-layer distilled student (`topic_xlmr_l3_kd`), keep the teacher as accuracy reference.** The p99 ≤ 40 ms requirement dominates. Student vs teacher on validation: −0.4 pt (0.8778 vs 0.8817); on test 0.8777 (fp32) / 0.8768 (INT8) vs 0.8799. The 4-layer student (0.8754) is no better and slower, so the 3-layer won on both axes. A skeptic's reading: the "≥ 8 points" margin is thinner with the student than with the teacher (+7.5 vs +7.9 on validation against the char baseline), and the test CI includes values below 8. We chose to meet the latency requirement and say so.
5. **Dialect-aware variant: not shipped.** The brief expects a dialect-aware model to win the Gulf slice. On validation Gulf rows CamelBERT-DA was −0.2 pt (CI −2.4 to +1.7) and the Gulf-fine-tuned XLM-R −1.4 pt against the generalist. With dialect labels that are synthetic and a generalist that already scores 0.8765 on Gulf, there is nothing to win. `dialect_aware` is `null` in `serving_config.json`; the router code path stays, covered by tests, off by default. This is a **requirement not met**.

## D2. Quantisation and runtime (decided with a rule fixed beforehand)

Rule: adopt INT8 only if the accuracy tax on the model's primary metric is ≤ 1.0 pt (embedder: ≤ 2.0 pt) **and** p50 improves ≥ 1.25×; otherwise keep fp32. (`scripts/s06c_quant_decisions.py`)

| Model | Tax | Speed-up | Decision |
|---|---|---|---|
| Topic student (ONNX Runtime dynamic INT8) | 0.02 pt | 1.43× | **INT8 default**, fp32 ONNX kept as rollback (`BAYAN_PRECISION=fp32`), fp32 canary reference stored |
| NER | 1.26 pt | 2.88× | fp32 (tax above limit) |
| QA | 14.62 pt | 1.54× | fp32 (tax far above limit) |
| Query embedder (LaBSE) | 2.69 pt MRR | 2.45× | fp32 (tax above limit) |

Serving settings: **1 ONNX thread, no micro-batch wait (`max_wait_ms = 0`), server and load generator on separate cores.** Why: ONNX INT8 cost per item does not shrink with batch size on 2 vCPUs (≈ 10.6 ms at batch 1 vs ≈ 15 ms/item at batch ≥ 4), so waiting to batch only adds latency; one thread per process avoids two threads fighting the load generator. Measured effect on the 3-layer student at saturation: p99 590 ms (2 threads, 2 ms wait, unpinned) → 249 ms (1 thread, 0 wait, pinned); at 16 req/s p99 55 ms (1 thread, no wait, unpinned) → 34 ms (pinned).

**Latency verdict: partly met.** p99 = 34.0 ms at 16 users with 1 s mean think time (≈ 16 req/s); 43.9 ms at ≈ 33 req/s; 68 ms at 60 req/s; 249 ms in a closed loop at saturation (≈ 90 req/s). The fp32 rollback is slower (p99 67.9 ms at 16 req/s). The test ran on a shared 2-vCPU Colab VM; the lab CPU may behave differently. If the course grader defines "16 concurrent users" as a closed loop with no think time, **this target is not met** on this hardware.

## D3. Sentiment

`sent_xlmr_all` (XLM-R base, all languages): validation 0.9219 vs TF-IDF char 0.9171 (word 0.8725) and CamelBERT-DA (Arabic only) 0.9083. Test: 0.9185 [0.907, 0.929] vs TF-IDF char 0.9171 [0.906, 0.928]. The transformer is **not meaningfully better** than a character n-gram model here; it is kept because it is at least as good and shares the tokenizer/runtime. Not quantised or distilled (no latency requirement on the analyze endpoint). Behavioural limitation: minimum-functionality pass rate 0.75 (18/24) vs the 0.90 target, left unfixed; sentiment labels are ≈ 8% noisy, which caps what more training would show.

## D4. NER

Two candidates, same data: plain XLM-R token tagger (validation entity-F1 0.9902, mean clitic pass-rate 0.888, worst clitic 0.7915) and the same with **clitic augmentation** (F1 1.000, mean 0.9996, worst 0.9985). The augmented one was chosen because the brief's clitic tests (و، ب، ل، ف and their pairs attached to entity names) are exactly where the plain model failed. Test: entity-F1 1.000, every clitic prefix 672/672. Caveat: the corpus is templated, so a perfect score reflects the templates, not real Arabic NER. Run in fp32 (INT8 tax 1.26 pt).

## D5. Question answering

`qa_xlmr_squad2_ft`: XLM-R fine-tuned on SQuAD-2-style data (answerable and unanswerable questions) with a **null-margin threshold** tuned on validation, so the model abstains when the no-answer score beats the best span by that margin. Test split: answerable exact-match 0.50, F1 0.77, null accuracy 0.917 (11/12). Smoke set: the file shipped with the data has 12 items, not the 20 the brief mentions; 12/12 strict exact match. INT8 was rejected (−14.6 pt). Honest reading: spans are right in meaning but often off by a token or two (EM 0.50 vs F1 0.77).

## D6. Similar-case search

Bi-encoder comparison on the dev queries picked **LaBSE** (`reports/search_dev_biencoders.json` has the other candidates). FAISS index `cases_v1` with a manifest (embedder name, dimension, hash) checked at start-up, plus stored probe vectors proving the embedder reproduces the index. Stage 2 re-scores the top-300 with **anchor bonuses** (district 0.1, month 0.4, day 0.4) because case descriptions carry those facts. A cross-encoder re-ranker was evaluated and **lost on dev**, so it is not shipped. The no-answer threshold (1.197 on the stage-2 score) comes from dev. Test: recall@10 1.000, MRR@10 0.992; dense-only 0.908 / 0.786. Caveat: the anchors work because the corpus is templated with exact dates and districts; real queries will not be this clean, so treat 0.99 as an upper bound. The dev set (75 queries) is small.

## D7. PII

Rule-based masking inside preprocessing (phone, national ID, email; Arabic-Indic and Latin digits). Requirement: 100% recall, measured on `pii_test_set` (60 cases): 60/60. Known gap found while debugging the start-up canaries: digit runs of 11 or more characters are not recognised as a phone or ID and are left unmasked. This is a limitation of the rules, not of the test set; it is listed in the model card and Future Improvements.

## D8. Skew canaries and rollback

Start-up refuses to serve (unless `BAYAN_CANARY_WARN_ONLY=1`) when preprocessing version or golden pairs, PII cases, any model's training-time preprocessing version, label order, classifier outputs against the stored fp32 reference (tolerance 0.005 for fp32, 0.15 for INT8 because INT8 can move probabilities), NER spans, or the FAISS manifest/embedder differ from what was trained. The INT8 tolerance (0.15, versus 0.005 for fp32) is deliberately loose because INT8 moves probabilities (largest fp32-vs-INT8 gap on the validation set: 0.197 on a few borderline rows); the canary therefore also requires identical argmax labels on the fixed reference texts. An early run aborted every benchmark server because one canary text had an 11-digit phone typo; the canary was fixed and the PII gap in D7 recorded.

## D9. Evaluation protocol

Bootstrap 95% CIs for every headline number; paired bootstrap for model-vs-model differences; slices by language, dialect, length bucket and class (small slices are flagged `small_sample`); behavioural suite (invariance tests and minimum-functionality tests, validation only); 209 validation errors of the shipped-family teacher categorised by hand-reading (the reading and categorisation were done by the AI assistant that built the project, not by the student: sample-check before relying on the categories).

## D10. Requirement scorecard (summary)

| Requirement | Status |
|---|---|
| Classifier ≥ 8 pts over TF-IDF | point estimate met (+8.08 / +8.21), CI lower bound 6.9: not established |
| Dialect-aware variant wins on Gulf | **not met**; not shipped |
| PII recall 100% | met on 60 cases; 11+ digit gap documented |
| NER F1 ≥ 0.80, clitic tests | met (templated data) |
| QA with honest nulls, ≥ 17/20 smoke | 12/12 on the 12-item smoke file shipped; test EM 0.50 |
| Search recall@10 ≥ 0.80, MRR@10 ≥ 0.70 | met (optimistic) |
| Behavioural suite ≥ 95% invariance, ≥ 90% MFT | topic met (mean invariance 0.990, MFT 0.958); **sentiment MFT 0.75 not met** |
| HTTP p99 ≤ 40 ms at 16 users | **partly met** (34 ms at ≈ 16 req/s; not at ≥ 33 req/s) |
| Optimisation ladder, quantisation decision per model, fp32 rollback, startup canaries | done |
| DECISIONS, model cards (6), sliced metrics, ≥ 100 errors with top-3 fixes | done |
