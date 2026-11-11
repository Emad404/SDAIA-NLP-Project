# Model card: Sentiment classifier

- Artefact: `sent_xlmr_all`
- Intended use: Negative / neutral / positive.
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Test: {'sentiment_tfidf_word': {'macro_f1': 0.8806, 'ci': [0.8665, 0.8931]}, 'sentiment_tfidf_char': {'macro_f1': 0.9171, 'ci': [0.9058, 0.9281]}, 'sentiment_model': {'macro_f1': 0.9185, 'ci': [0.9068, 0.9292]}, 'model_minus_tfidf_char_points': {'diff': 0.14, 'ci': [-0.08, 0.38]}}
- Validation: {'sent_xlmr_all': 0.9219, 'sent_camelbert_da_ar': 0.9083, 'tfidf_char': 0.9171, 'tfidf_word': 0.8725}

## Limitations (hand-written)

- About 8% of training labels are noisy.
- On our hand-written minimum-functionality cases the model labelled explicit emotion statements such as 'I am angry that nobody answered my calls' as neutral: it seems to key on dataset style (a described problem) rather than on emotion words.
- Sarcasm, negation and intensity were not systematically tested beyond a few dozen hand-written cases (see behavioural suite).

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
