# Model card: Extractive QA with honest nulls

- Artefact: `qa_xlmr_squad2_ft`
- Intended use: Answers a question from a supplied service document or abstains (null answer) when the document does not contain it.
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Test split: {'answerable_em': 0.5, 'answerable_f1': 0.7746, 'null_acc': 0.9167, 'n_answerable': 72, 'n_null': 12, 'score': 0.8456}
- Smoke 20: {'n': 12, 'correct_strict_EM': 12, 'correct_lenient_F1>=0.5': 12, 'meets_17_of_20': False}

## Limitations (hand-written)

- The abstain threshold is tuned on 64 validation questions; at that sample size it is imprecise.
- Questions are answered from the single context passed in; the service does not retrieve documents for the user (the /v1/ask endpoint takes a doc_id).
- Contexts are 3-6 short facts; long documents would need chunking that is not implemented.

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
