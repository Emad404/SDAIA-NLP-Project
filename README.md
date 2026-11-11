# Bayan: bilingual (Arabic / English) citizen-feedback NLP service

Capstone for the SDAIA Academy **NLP with Transformers** course (Labs 1–7 assembled into one service).
Course material and the reference organisation repo: <https://github.com/SDAIAAcademy>

Bayan takes one citizen message and returns what a government service desk needs: the **topic**, the **sentiment**, the **entities** (people, places, services), a **PII-masked copy** of the text, and the **most similar past cases**. It also answers questions over the service documents with an honest *"no answer"* when the documents do not say.

> **Read this first (honest status).** Everything below was measured, not assumed. Most targets are met on the frozen test set, but three are only partly met or not met: the HTTP p99 ≤ 40 ms target holds only at light load, the topic margin over TF-IDF clears +8 points as a point estimate but its confidence interval does not, and the sentiment behavioural suite is at 75% (target 90%). See [Results](#5-results--evaluation) and [DECISIONS.md](DECISIONS.md). The data are synthetic and templated, so several scores (NER F1 = 1.0, search MRR = 0.99) are optimistic and say little about real traffic.

---

## 1. Overview

| Capability | Endpoint | Model |
|---|---|---|
| Topic (8 classes) + dialect-aware routing hook | `POST /v1/classify`, `/v1/analyze` | 3-layer distilled XLM-R student, ONNX INT8 (fp32 rollback) |
| Sentiment (3 classes) | `/v1/analyze` | fine-tuned XLM-R |
| Entity extraction (PERSON / LOCATION / SERVICE), Arabic clitic-safe | `/v1/analyze` | fine-tuned XLM-R token tagger (clitic augmentation) |
| PII masking (phone, national ID, email) | every endpoint (preprocessing v1.0.0) | rule-based, 100% recall on the 60-case set |
| Similar past cases | `/v1/analyze` (`similar_cases`) | LaBSE + versioned FAISS index + anchor re-scoring |
| Question answering over service documents | `POST /v1/ask` | XLM-R fine-tuned on SQuAD-2-style data, null-margin threshold |
| Batch analysis | `POST /v1/analyze/batch` | same pipeline, up to 64 messages |
| Health and skew canaries | `GET /health`, `GET /v1/canaries` | start-up asserts, see §4 |

Extensions chosen (the brief asks for at least one): **dialect router** (built and evaluated, *not shipped*, see Results), **batch endpoint**, **QA over documents**.

## 2. Problem

Citizen feedback arrives in Arabic (MSA, Gulf, Egyptian, Levantine) and English, often with phone numbers, national IDs and emails inside the text. The service must (a) never leak that PII into logs or models, (b) route the message to the right team, (c) surface similar earlier cases, (d) stay fast enough to sit behind a web form (p99 ≤ 40 ms at 16 concurrent users on a lab CPU), and (e) say what it does *not* know. The brief also requires one preprocessing module shared by training and serving, a frozen test set touched once, sliced metrics with confidence intervals, a behavioural test suite, hand-read error analysis, model cards, and an optimisation ladder with the accuracy cost of each step.

## 3. Dataset & Model

**Data** (the course's synthetic Bayan corpora, not redistributed here; point `BAYAN_DATA` at the folder): `bayan_feedback_clean` (topic + sentiment + dialect label per message), `bayan_ner_4000`, `bayan_qa_600` + `qa_smoke_set`, `bayan_cases` + `bayan_queries` (retrieval), `bayan_service_docs`, `pii_test_set`.
Known data properties that shape every conclusion: topic labels are roughly 3–4% noisy and sentiment roughly 8%; dialect labels are synthetic; NER sentences are templated; many messages are generic follow-ups with no topic cue.

**Splits.** Grouped by `citizen_id` so one citizen never appears in two splits; the test split is guarded in code (`BAYAN_FINAL=1`, SHA-256 of the split, lock file) and was read **once**, by `scripts/s08_final.py`. All model, threshold and precision choices were made on validation data.

**Models shipped** (full justification in [DECISIONS.md](DECISIONS.md), one model card each in `model_cards/`):

| Artefact | Choice | Why in one line |
|---|---|---|
| Topic | `topic_xlmr_l3_kd`: 3-layer XLM-R student distilled from a case-augmented teacher, ONNX INT8 | −0.4 pt F1 vs the teacher on validation, ≈ 3.5× faster (INT8, batch 1); teacher kept as accuracy reference |
| Sentiment | `sent_xlmr_all` (XLM-R) | +0.5 pt over best TF-IDF on validation, CI overlaps (a weak win) |
| NER | `ner_xlmr_clitic_aug` | clitic augmentation lifted worst-case clitic pass-rate from 0.79 to 0.9985 on validation |
| QA | `qa_xlmr_squad2_ft` + null margin threshold | answers or abstains; INT8 rejected (−14.6 pt) |
| Search | LaBSE, FAISS `cases_v1`, anchor re-scoring (district / month / day) | dense-only recall@10 0.91 → 1.00 with anchors; cross-encoder lost on dev so not shipped |

## 4. Workflow / Architecture

```
                 raw text
                    │
        preprocess v1.0.0  (PII masked FIRST, then Unicode/Arabic normalisation)   ← one module, used in training AND serving
                    │
   ┌────────────────┼──────────────────────────────┬──────────────────────┐
 language detect   topic classifier (ONNX INT8)   sentiment        NER tagger (clitic-aware)
                    │                                                   │
                    └──────── MicroBatcher (batch ≤ 16, wait 0 ms) ─────┘
                                      │
                  LaBSE embed → FAISS cases_v1 → top-300 → anchor re-score → no-answer threshold → similar_cases
                                      │
              QA: question × each service doc → span with null margin → answer | "no answer"
```

Pipeline stages in the repo (`scripts/`), each writing JSON to `reports/` and artefacts to `artifacts/`:
`s01_prepare` (clean, grouped split, TF-IDF baselines) → `s02_train_cls` (topic / sentiment) → `s03_train_ner` → `s04_train_qa` → `s05_search` (bi-encoder comparison, index, re-rank decision, no-answer threshold) → `s06_optimize` (ONNX fp32/INT8 ladder) + `s06b_distill` (students) → `s07_select` (selection on validation, serving config, canary reference) → `s07b_behaviour_errors` → `s07e_finalize_config` + `s06c_quant_decisions` → `s07c_http_bench` → `s10_error_analysis` → `s08_final` (the one test run) → `s09_docs` (RESULTS.md, BENCHMARKS.md, model cards).

**Skew canaries (asserted at start-up; any failure aborts the server unless `BAYAN_CANARY_WARN_ONLY=1`):** preprocessing version and golden input/output pairs; PII masking cases; every model's training-time preprocessing version equals the serving version; label order; classifier outputs vs a stored fp32 reference (tolerance depends on precision); NER reference spans; FAISS manifest, embedder name, dimension and re-embedding of probe vectors.

**Rollback:** `make serve-fp32` (or `BAYAN_PRECISION=fp32`) serves the full-precision ONNX graph of the same model.

## 5. Results & Evaluation

All test numbers are from the single frozen-test run (2 385 feedback rows, 75 retrieval queries, 84 QA items, 598 NER sentences). Full tables: [RESULTS.md](RESULTS.md), serving numbers: [BENCHMARKS.md](BENCHMARKS.md).

| Requirement | Result | Verdict |
|---|---|---|
| Topic beats TF-IDF by ≥ 8 macro-F1 pts | shipped INT8 student 0.8768 vs TF-IDF word 0.796 / char 0.795; **+8.08 / +8.21 pts**, 95% CI [6.86, 9.36] / [6.96, 9.56] | Point estimate **met**; CI lower bound is below 8, so "≥ 8" is not established with confidence. The un-distilled teacher scores 0.8799. |
| Dialect-aware variant wins on the Gulf slice | on validation Gulf rows CamelBERT-DA is −0.2 pt vs the generalist (95% CI −2.4 to +1.7) and the Gulf-fine-tuned XLM-R is −1.4 pt (point estimates); neither wins | **Not met.** No dialect-aware model was shipped; the router stays as an off-by-default hook. Only the much larger XLM-R-large helped Gulf (+1.1 pt, CI [0.19, 2.24]) and it is far too slow. |
| PII masking 100% recall | 60/60 cases, 0 leaks | Met. Known gap found during canary debugging: digit runs of 11+ characters are not masked. |
| NER entity-F1 ≥ 0.80, clitic tests green | F1 1.000 (ar 1.000, en 1.000); all 7 clitic prefixes pass 672/672 | Met, but templated data: treat as an upper bound. |
| Extractive QA with honest nulls | smoke file shipped with the data has **12** items (brief says 20): 12/12 strict exact match; test split: answerable EM 0.50 / F1 0.77, null accuracy 0.917 | Met on the tiny smoke set; test EM shows span boundaries are the weak point. |
| Search recall@10 ≥ 0.80, MRR@10 ≥ 0.70 | recall@10 1.000, MRR@10 0.992 (dense-only stage 1: 0.908 / 0.786) | Met, but the anchors exploit templated dates/districts: optimistic. No-answer accuracy 10/10, false no-answer 0/65. |
| Sentiment | 0.9185 vs TF-IDF char 0.9171 | Tied: CI overlap. Do not claim a transformer win here. |
| Behavioural suite (validation) | topic MFT 0.958; topic invariance mean 0.990 (min 0.943: UPPER-CASE English); sentiment invariance 1.00; **sentiment MFT 0.75** | Topic met. **Sentiment MFT not met** (target 90%), left unfixed and documented. |
| HTTP p99 ≤ 40 ms, 16 concurrent users | INT8 student, 1 ORT thread, server and load generator on separate cores of a 2-vCPU Colab VM: see table below | **Met only at ≈ 16 req/s offered load; not met above.** |
| ≥ 100 hand-read errors, top-3 fixes | all 209 validation topic errors categorised in [error_analysis.md](reports/error_analysis.md) | Met. The categorisation was done by the AI assistant that built the project: please spot-check a sample. |

**HTTP latency (16 users, `POST /v1/classify`, INT8 default, ms):**

| Offered load (mean think time / user) | req/s | p50 | p95 | p99 | p99 ≤ 40 ms |
|---|---|---|---|---|---|
| 1000 ms | 16.3 | 12.2 | 25.1 | **34.0** | yes |
| 500 ms | 33.0 | 14.5 | 32.6 | 43.9 | no |
| 250 ms | 60.1 | 21.8 | 52.0 | 68.0 | no |
| 0 ms (closed loop, saturation) | 89.8 | 172.3 | 226.4 | 248.7 | no |
| fp32 rollback, 1000 ms | 16.0 | 20.0 | 47.6 | 67.9 | no |

Reading it honestly: model-only INT8 latency at batch 1 is 10.5 ms p50 / 39.7 ms p99, so a 40 ms p99 leaves almost no headroom; two vCPUs shared with a Python load generator make tails worse. The "lab CPU" in the course may differ; rerun `make bench` there. The unpinned first attempt (2 threads, 2 ms batching wait) was far worse (p99 ≈ 590 ms at saturation); the history is in BENCHMARKS.md.

**Optimisation ladder (validation macro-F1 and batch-1 latency, 3-layer student):** torch fp32 pad-128 0.8778 / p50 66 ms → dynamic padding 29.5 ms → ONNX fp32 15.0 ms (identical predictions) → ONNX INT8 10.5 ms, F1 0.8776 (−0.02 pt). INT8 was kept only because its pre-registered rule passed (tax ≤ 1.0 pt and ≥ 1.25× faster). Quantisation decisions for the other models: NER fp32 (−1.26 pt), QA fp32 (−14.6 pt), query embedder fp32 (−2.69 pt MRR).

**Sliced metrics (test, topic macro-F1 with 95% CI):** Arabic 0.871 [0.853, 0.891], English 0.882 [0.864, 0.899]; Gulf 0.860, MSA 0.884, Levantine 0.928 (n = 109), Egyptian 0.830 (n = 136, wide CI); long messages 0.909, medium 0.816, short ≤ 6 words 0.751 (n = 42, flagged small sample). Per-class and confusion tables are in `reports/`.

**What the errors say (validation, 209 topic errors, 87.2% accuracy):** most are generic follow-ups with no topic cue and low model confidence (irreducible without context), about 37 are label noise or a hint line that contradicts the label, 9 are compound messages, 4 are boundary cases and only about 4 are clear model misses. Top-3 fixes: abstain / triage below a confidence threshold, audit the labels, and add conversation context plus multi-label output.

## 6. Technologies

Python 3.13, PyTorch, Hugging Face Transformers (XLM-R base/large teacher, CamelBERT variants evaluated), sentence-transformers (LaBSE), FAISS, ONNX / ONNX Runtime (fp32 + dynamic INT8), scikit-learn (TF-IDF baselines), seqeval, FastAPI + Uvicorn, pytest (76 tests), Google Colab (T4 GPU for training, 2-vCPU CPU for the latency tests).

## 7. How to Run

```bash
git clone <this repo> && cd bayan_project
make setup                                  # pip install -r requirements.txt
export BAYAN_DATA=/path/to/the/course/Data  # the Bayan CSV/JSONL files
make test                                   # 76 unit tests
```

Trained artefacts (models, ONNX graphs, FAISS index, canary references) are **not** in git (multi-GB); they are in the Google Drive folder `SDAIA NLP/Bayan_Project` or can be regenerated:

```bash
make reproduce        # data → train → train-extra → search → select → behaviour → optimize → finalize → bench → errors
make final            # the ONE frozen-test evaluation (refuses to run twice)
make docs             # RESULTS.md, BENCHMARKS.md, model_cards/
```

Serve (canaries run first; start-up aborts on any skew):

```bash
make serve                  # INT8 default, http://localhost:8000   (docs at /docs)
make serve-fp32             # rollback
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/analyze -H 'content-type: application/json' \
     -d '{"text":"الكهرباء مقطوعة في حي النخيل منذ الصباح، رقم جوالي 0551234567"}'
curl -s -X POST localhost:8000/v1/ask -H 'content-type: application/json' -d '{"question":"How long does a permit take?"}'
```

`make serve` expects `artifacts/` (or `BAYAN_WORK=<dir with models/ and artifacts/>`). To run end to end on Colab use `notebooks/Bayan_Colab_Runner.ipynb`.

## 8. Future Improvements

1. **Latency:** the 40 ms p99 at higher load needs more cores or a smaller/faster encoder (further layer pruning, int8 with static calibration, ORT with intra-op threads per core, several worker processes); re-measure on the actual lab CPU.
2. **Labels:** audit the ≈ 37 noisy / hint-conflicting validation labels and re-split; re-train with noise-robust loss.
3. **Abstain and triage:** reject low-confidence topics to a human queue (selective-prediction table in `reports/error_analysis.md`).
4. **Context and multi-label:** conversation history for follow-ups; multi-label topics for compound messages.
5. **Sentiment:** 6 of the 24 minimum-functionality cases fail (`reports/behaviour_val.json` lists them); add targeted data and re-test.
6. **PII:** mask arbitrary-length digit runs and spelled-out numbers; the 60-case set is small.
7. **Real data:** every number here comes from synthetic, templated corpora; validate on real, de-identified citizen messages before any deployment.
8. **Dialect model:** a genuinely dialect-aware win may need real dialect data rather than synthetic labels.

---
---

## 9. Team

- [Abdulaziz Alamri](https://github.com/AbdulazizAmri)
- [Emad Alaskar](https://github.com/Emad404)
