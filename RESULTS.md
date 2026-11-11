# Bayan: results (auto-generated from reports/*.json)

Selection (models, thresholds, precision) was made on VALIDATION data only. The frozen test split was evaluated once by `scripts/s08_final.py` (lock file `final_run.lock`). Every test number carries a 95% bootstrap CI.

## Requirement scorecard (frozen test)

| Requirement | Measured | Verdict |
|---|---|---|
| Topic classifier beats topic_tfidf_word by >= 8 macro-F1 points | +8.08 pts, CI [6.86, 9.36] | MET (CI lower bound >= 8: False) |
| Topic classifier beats topic_tfidf_char by >= 8 macro-F1 points | +8.21 pts, CI [6.96, 9.56] | MET (CI lower bound >= 8: False) |
| PII masking recall = 100% (60-case set) | recall 1.0 (0 leaks) | MET |
| NER entity-F1 >= 0.80 | 1.0 (ar 1.0, en 1.0) | MET |
| NER clitic tests | pass rates و:1.0, ب:1.0, ل:1.0, ف:1.0, وب:1.0, ول:1.0, فب:1.0 | min 1.0 |
| QA smoke set >= 17/20 (strict exact match; the smoke file shipped with the data has 12 items, not 20, so the bar is applied as >= 85%) | 12/12 = 100% (lenient F1>=0.5: 12) | MET (tiny set: treat as a smoke test only) |
| Search recall@10 >= 0.80 | 1.000 [1.000, 1.000] | MET |
| Search MRR@10 >= 0.70 | 0.992 [0.977, 1.000] | MET |
| HTTP p99 <= 40 ms, 16 users, int8 [saturation (no think time)] | p99 248.65 ms, p50 172.29 ms, 89.8 rps | NOT MET |
| HTTP p99 <= 40 ms, 16 users, int8 [mean think 250 ms/user] | p99 68.02 ms, p50 21.84 ms, 60.1 rps | NOT MET |
| HTTP p99 <= 40 ms, 16 users, int8 [mean think 500 ms/user] | p99 43.85 ms, p50 14.51 ms, 33.0 rps | NOT MET |
| HTTP p99 <= 40 ms, 16 users, int8 [mean think 1000 ms/user] | p99 33.98 ms, p50 12.19 ms, 16.3 rps | MET |
| HTTP p99 <= 40 ms, 16 users, fp32 [mean think 500 ms/user] | p99 95.16 ms, p50 26.06 ms, 32.0 rps | NOT MET |
| HTTP p99 <= 40 ms, 16 users, fp32 [mean think 1000 ms/user] | p99 67.91 ms, p50 19.99 ms, 16.0 rps | NOT MET |

## Topic classifier (test)

| System | macro-F1 [95% CI] | accuracy |
|---|---|---|
| topic_tfidf_word | 0.796 [0.7805, 0.8117] | 0.7979 |
| topic_tfidf_char | 0.7947 [0.779, 0.811] | 0.7958 |
| topic_general_fp32_torch | 0.8777 [0.8641, 0.8902] | 0.8725 |
| topic_general_ort_fp32 | 0.8777 [0.8641, 0.8902] | 0.8725 |
| topic_general_ort_int8 | 0.8768 [0.8634, 0.8897] | 0.8717 |
| topic_reference_teacher_fp32_torch | 0.8799 [0.8669, 0.8929] | 0.8721 |
| topic_shipped_system | 0.8768 [0.8634, 0.8897] | 0.8717 |

## Sentiment (test)

| System | macro-F1 [95% CI] |
|---|---|
| sentiment_tfidf_word | 0.8806 [0.8665, 0.8931] |
| sentiment_tfidf_char | 0.9171 [0.9058, 0.9281] |
| sentiment_model | 0.9185 [0.9068, 0.9292] |

### slices topic test

| slice          | value         |    n |   macro_f1 |   ci_low |   ci_high |   accuracy | small_sample   |
|:---------------|:--------------|-----:|-----------:|---------:|----------:|-----------:|:---------------|
| language       | ar            | 1147 |     0.8713 |   0.8528 |    0.891  |     0.8666 | False          |
| language       | en            | 1238 |     0.8816 |   0.8641 |    0.8985 |     0.8764 | False          |
| dialect_region | Egyptian      |  136 |     0.8299 |   0.7552 |    0.8871 |     0.8235 | False          |
| dialect_region | English       | 1238 |     0.8816 |   0.8641 |    0.8985 |     0.8764 | False          |
| dialect_region | Gulf          |  606 |     0.8596 |   0.8322 |    0.8829 |     0.8614 | False          |
| dialect_region | Levantine     |  109 |     0.9278 |   0.8753 |    0.9645 |     0.9174 | False          |
| dialect_region | MSA           |  296 |     0.8843 |   0.8413 |    0.9181 |     0.8784 | False          |
| len_bucket     | long(>12w)    | 1512 |     0.9087 |   0.8932 |    0.9231 |     0.9074 | False          |
| len_bucket     | medium(7-12w) |  831 |     0.8164 |   0.7916 |    0.8413 |     0.8111 | False          |
| len_bucket     | short(<=6w)   |   42 |     0.7506 |   0.5897 |    0.8701 |     0.7857 | True           |

### per class topic test

| class            |   precision |   recall |     f1 |   support |
|:-----------------|------------:|---------:|-------:|----------:|
| account_access   |      0.9427 |   0.843  | 0.8901 |       293 |
| appointments     |      0.6896 |   0.9482 | 0.7985 |       328 |
| billing          |      0.7886 |   0.9405 | 0.8579 |       353 |
| digital_services |      0.9776 |   0.7649 | 0.8583 |       285 |
| outage           |      0.9255 |   0.8671 | 0.8954 |       301 |
| permits          |      0.9492 |   0.8127 | 0.8757 |       299 |
| transport        |      0.9788 |   0.8817 | 0.9277 |       262 |
| waste            |      0.9291 |   0.8939 | 0.9112 |       264 |

### slices sentiment test

| slice          | value         |    n |   macro_f1 |   ci_low |   ci_high |   accuracy | small_sample   |
|:---------------|:--------------|-----:|-----------:|---------:|----------:|-----------:|:---------------|
| language       | ar            | 1147 |     0.9198 |   0.9041 |    0.9351 |     0.9207 | False          |
| language       | en            | 1238 |     0.9169 |   0.8993 |    0.9332 |     0.9216 | False          |
| dialect_region | Egyptian      |  136 |     0.9329 |   0.884  |    0.9688 |     0.9338 | False          |
| dialect_region | English       | 1238 |     0.9169 |   0.8993 |    0.9332 |     0.9216 | False          |
| dialect_region | Gulf          |  606 |     0.9242 |   0.9019 |    0.9447 |     0.9257 | False          |
| dialect_region | Levantine     |  109 |     0.8581 |   0.7846 |    0.9154 |     0.8624 | False          |
| dialect_region | MSA           |  296 |     0.9269 |   0.8967 |    0.9547 |     0.9257 | False          |
| len_bucket     | long(>12w)    | 1512 |     0.9234 |   0.9094 |    0.9368 |     0.9279 | False          |
| len_bucket     | medium(7-12w) |  831 |     0.8944 |   0.8703 |    0.9182 |     0.9097 | False          |
| len_bucket     | short(<=6w)   |   42 |     0.616  |   0.5578 |    0.9741 |     0.9048 | True           |

## Behavioural suite (validation)

| Suite | pass rate |
|---|---|
| mft_topic | 0.958 (n=120) |
| mft_topic_core | 0.958 (n=96) |
| mft_sentiment | 0.750 (n=24) |
| invariance_topic invariance mean / min | 0.990 / 0.943 |
| invariance_sentiment invariance mean / min | 1.000 / 1.000 |
