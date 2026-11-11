# Model card: Similar-case retrieval (bi-encoder + FAISS + stage-2 anchor re-scorer)

- Artefact: `sentence-transformers/LaBSE`
- Intended use: Returns the most similar historical cases (cross-language) or an explicit no-answer.
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Test: {'n_test_queries': 75, 'n_answerable': 65, 'dense_stage1_only': {'recall@10': [0.9077, 0.8308, 0.9692], 'mrr@10': [0.7855, 0.6933, 0.8696]}, 'two_stage_anchor_rescore': {'recall@10': [1.0, 1.0, 1.0], 'mrr@10': [0.9923, 0.9769, 1.0]}, 'no_answer': {'threshold': 1.1966615128517157, 'accuracy_on_no_answer_queries': 1.0, 'false_no_answer_rate_on_answerable': 0.0, 'n_no_answer': 10}, 'shipped_stage2': 'anchor_rescore', 'meets_recall@10>=0.80': True, 'meets_mrr@10>=0.70': True}
- Dev stage comparison: {'dense_only': {'recall@10': 0.8461538461538461, 'mrr@10': 0.7485042735042734}, 'cross_encoders': {'mmarco_mminilm': {'checkpoint': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'recall@10': 0.8153846153846154, 'mrr@10': 0.6651282051282051, 'seconds_for_dev_queries': 2.1}, 'bge_reranker_base': {'checkpoint': 'BAAI/bge-reranker-base', 'recall@10': 0.7846153846153846, 'mrr@10': 0.7206593406593407, 'seconds_for_dev_queries': 6.5}}, 'anchor_stage2': {'weights': {'district': 0.1, 'month': 0.4, 'day': 0.4}, 'recall@10': 1.0, 'mrr@10': 1.0, 'adopted': True, 'case_anchor_coverage': 1.0}}
- Config: {'embedder_name': 'labse', 'embedder': 'sentence-transformers/LaBSE', 'qprefix': '', 'pprefix': '', 'index_dir': 'indexes/cases_v1', 'stage2': 'anchor_rescore', 'anchor_weights': {'district': 0.1, 'month': 0.4, 'day': 0.4}, 'pool': 300, 'use_rerank': False, 'reranker': None, 'no_answer_threshold': 1.1966615128517157, 'no_answer_score_source': 'stage-2 top-1 score (dense cosine + anchor bonus)', 'dev_balanced_accuracy_no_answer': 1.0, 'dev_n': 75}

## Limitations (hand-written)

- Stage 2 exploits a property of THIS corpus: every case ends with a district and a date, and queries often repeat them. On a corpus without such anchors, stage 2 does nothing and quality falls back to the dense retriever.
- The stage-2 weights were tuned on 65 development queries; the test queries (75) were scored once.
- The district lexicon covers the 17 districts in the case base; a query naming an unknown district gets no district bonus.
- Off-the-shelf cross-encoder re-rankers (mMARCO mMiniLM, bge-reranker-base) did NOT beat the dense retriever on development queries and are therefore not shipped.
- The no-answer threshold separates 'out of domain' queries well but was tuned on only about 15 dev no-answer queries.

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
