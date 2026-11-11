# Bayan — bilingual citizen-feedback NLP service.   `make serve` starts the API (needs trained artifacts, see README).
PY        ?= python
export PYTHONPATH := $(CURDIR)/src
ARTIFACTS ?= $(CURDIR)/artifacts
export BAYAN_WORK      ?= $(ARTIFACTS)
export BAYAN_ARTIFACTS ?= $(ARTIFACTS)/artifacts
export BAYAN_MODELS    ?= $(ARTIFACTS)/models
PORT      ?= 8000
PRECISION ?= int8          # int8 = shipped default | fp32 = rollback

.PHONY: setup test data train train-extra search select behaviour optimize finalize bench errors final docs reproduce serve serve-int8 serve-fp32 clean-lock
setup:
	pip install -r requirements.txt
test:
	$(PY) -m pytest -q tests
data:                       ## stage 1: clean, split (frozen test), baselines, tokenizer audit
	$(PY) scripts/s01_prepare.py
train:                      ## stages 2-4: topic/sentiment classifiers, NER, QA
	bash scripts/run_s02.sh && bash scripts/run_s03_s04.sh
train-extra:                ## stages 2b-2d: case-augmented teacher, NER clitic-aug, distilled students, Gulf adaptation
	bash scripts/train_extra.sh
search:                     ## stage 5: bi-encoder comparison, FAISS index, re-ranker, no-answer threshold
	$(PY) scripts/s05_search.py
select:                     ## stage 7a: model selection on validation, serving config, canary reference
	$(PY) scripts/s07_select.py
behaviour:                  ## stage 7b: invariance + minimum-functionality tests, validation error sheet
	$(PY) scripts/s07b_behaviour_errors.py
MODELS ?= topic_xlmr_all_ca topic_xlmr_l4_kd topic_xlmr_l3_kd      # teacher, 4-layer and 3-layer (shipped) students
optimize:                   ## stage 6: ONNX / INT8 ladder with accuracy tax (per classifier) + quantisation decisions
	for m in $(MODELS); do $(PY) scripts/s06_optimize.py --model $$m || exit 1; done
	$(PY) scripts/s06c_quant_decisions.py
finalize:                   ## stage 7e: ship the 3-layer student (INT8 default, fp32 rollback), 1 ORT thread, no batching wait
	$(PY) scripts/s07e_finalize_config.py && $(PY) scripts/s06c_quant_decisions.py
bench:                      ## stage 7c: HTTP p50/p95/p99, 16 concurrent users, server and load generator pinned to separate cores
	$(PY) scripts/s07c_http_bench.py --precision int8 --pin --threads 1 --max-wait-ms 0 --seconds 25 --think-ms-list 0,250,500,1000 --tag final_int8
	$(PY) scripts/s07c_http_bench.py --precision fp32 --pin --threads 1 --max-wait-ms 0 --seconds 25 --think-ms-list 500,1000 --tag final_fp32
errors:                     ## stage 10: categorised validation error analysis (needs the hand annotation to match the model)
	$(PY) scripts/s10_error_analysis.py
reproduce: data train train-extra search select behaviour optimize finalize bench errors   ## everything up to (not including) the frozen-test run
final:                      ## stage 8: the ONE frozen-test evaluation (lock file prevents silent re-runs)
	$(PY) scripts/s08_final.py
docs:
	$(PY) scripts/s09_docs.py
serve:                      ## start the API with skew canaries asserted at start-up
	BAYAN_PRECISION=$(PRECISION) $(PY) -m uvicorn bayan.service:app_factory --factory --host 0.0.0.0 --port $(PORT)
serve-int8:
	$(MAKE) serve PRECISION=int8
serve-fp32:                 ## rollback: full-precision ONNX graph
	$(MAKE) serve PRECISION=fp32
