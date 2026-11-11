# Model card: Topic classifier (generalist)

- Artefact: `topic_xlmr_l3_kd`
- Intended use: Routes a message to one of 8 service topics (appointments, billing, waste, permits, outage, account_access, digital_services, transport).
- Preprocessing: `bayan.preproc` v1.0.0 (single shared module; the model owns its Arabic profile)
- Data: Data are the course's synthetic Bayan corpora (templated messages, synthetic dialect labels, a few percent label noise). No number in this card has been validated on real citizen traffic.

## Measured

- Test macro-F1 (shipped system): {'macro_f1': 0.8768, 'ci': [0.8634, 0.8897], 'accuracy': 0.8717}
- fp32 / int8 test macro-F1: {'macro_f1': 0.8777, 'ci': [0.8641, 0.8902], 'accuracy': 0.8725} / {'macro_f1': 0.8768, 'ci': [0.8634, 0.8897], 'accuracy': 0.8717}
- Validation table: see `reports/topic_selection_val.json`
- Optimisation ladder: see BENCHMARKS.md
- Quantisation decision: tax 0.02 pt (limit 1.0), speedup 1.43x (need 1.25x) -> default int8; fp32 kept as rollback (BAYAN_PRECISION=fp32)

## Limitations (hand-written)

- The corpus is templated and short (3-39 words); accuracy on long, multi-issue or code-switched messages is untested.
- Messages mentioning two topics are labelled with the first one only; the model cannot return a second topic.
- It leans on keywords: in our hand-written minimum-functionality set it labelled 'My electricity bill is much higher than usual' as outage and 'I need a permit to renovate my house' as waste.
- About 3-4% of training labels are noisy by construction, so the ceiling on this data is below 100% and small differences between models are inside noise.
- Gulf-dialect behaviour is measured against SYNTHETIC dialect labels; it says little about real Gulf speech.
- Upper-cased English input was the weakest invariance check on the first model (about 80% unchanged predictions); the shipped model is trained with case augmentation, but other surface changes (typos, emoji, transliterated Arabic) were not tested.

## Rollback

Serve with `BAYAN_PRECISION=fp32` (default) for the unquantised reference; skew canaries are asserted at start-up and the service refuses to start if they fail.
