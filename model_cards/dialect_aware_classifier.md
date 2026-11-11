# Model card: Dialect-aware topic classifier (Gulf variant) + dialect router

- Artefact: `None`
- Intended use: Optional second classifier used only for Arabic messages the dialect router predicts as Gulf.
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Gulf slice paired comparison (test): None
- Validation comparison: {'dialect_aware': {'candidates': ['topic_camelbert_da_ar', 'topic_xlmr_gulf_ft'], 'chosen': 'topic_camelbert_da_ar', 'wins_val_point_estimate': False, 'significant_95': False}}

## Limitations (hand-written)

- The router itself is a char-n-gram linear model with roughly 77% validation accuracy over four dialects, so a fifth or more of Arabic messages may go to the wrong branch.
- Whether this variant beats the generalist is a statistical question with small samples; read the paired CI in RESULTS.md, not only the point estimate.
- If the CI includes zero the honest conclusion is 'no detectable difference', and routing adds complexity without proven benefit.

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
