# Model card: Entity extractor (PERSON, LOCATION, SERVICE)

- Artefact: `ner_xlmr_clitic_aug`
- Intended use: BIO token tagger returning entity spans with character offsets.
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Test: {'n_test_sentences': 598, 'entity_f1': 1.0, 'precision': 1.0, 'recall': 1.0, 'per_type': {'LOCATION': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 490.0}, 'PERSON': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 445.0}, 'SERVICE': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 529.0}}, 'by_language': {'ar': 1.0, 'en': 1.0}, 'clitic_tests': {'و': {'pass_rate': 1.0, 'n': 672}, 'ب': {'pass_rate': 1.0, 'n': 672}, 'ل': {'pass_rate': 1.0, 'n': 672}, 'ف': {'pass_rate': 1.0, 'n': 672}, 'وب': {'pass_rate': 1.0, 'n': 672}, 'ول': {'pass_rate': 1.0, 'n': 672}, 'فب': {'pass_rate': 1.0, 'n': 672}}, 'meets_0.80': True}
- Validation choice: {'candidates': {'ner_xlmr': {'entity_f1': 0.9902, 'clitic_mean': 0.8877571428571429, 'clitic_min': 0.7915}, 'ner_xlmr_clitic_aug': {'entity_f1': 1.0, 'clitic_mean': 0.9995714285714286, 'clitic_min': 0.9985}}, 'chosen': 'ner_xlmr_clitic_aug'}

## Limitations (hand-written)

- Only three entity types exist; organisations, dates, amounts and IDs are not extracted (PII is masked by the preprocessing module instead).
- The dataset is highly templated, so entity-F1 near 1.0 mostly shows the model learned the templates; do not expect that on free text.
- Arabic clitic robustness (و ب ل ف وب ول فب prefixes) was tested with synthetic prefixing of gold entities; real morphology is richer than that test.

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
