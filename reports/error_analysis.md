# Error analysis: topic classifier (topic_xlmr_all_ca), validation split

All **209 validation errors** (of 1634 rows, accuracy 0.872) were read one by one and each was assigned a cause. The frozen test split was not used. **Who did the reading:** the AI assistant that built this project read and categorised the errors; the human owner should spot-check a sample (the full annotated list is `error_analysis_topic_val.csv`). Categories are judgements, not ground truth.

| cause | n | share | median conf | what it is |
|---|---:|---:|---:|---|
| label_conflict | 25 | 12% | 0.983 | Label looks wrong (text clearly describes another topic, often the one the model predicted) |
| hint_conflict | 12 | 6% | 0.954 | Message ends with an explicit topic hint ('It is about X' / 'الموضوع يخص X') that contradicts the label |
| compound | 9 | 4% | 0.986 | Two topics in one message (label = the first one; model picks the louder or the second) |
| boundary | 4 | 2% | 0.975 | Genuinely ambiguous boundary between neighbouring topics (billing/transport, billing/digital_services, appointments/permits) |
| model_miss | 4 | 2% | 0.66 | Topical cue present but the model missed it (a real model error) |
| generic_no_topic_cue | 155 | 74% | 0.168 | Generic follow-up / complaint with NO topic words ('still waiting for a reply', 'please escalate'); the text alone cannot determine the topic |

## What the errors tell us

1. **The largest bucket (155 errors, 74%) is messages that contain no topic information at all** (e.g. "Still waiting for a reply", "Please escalate this to a manager", "تم تحويلي من قسم إلى آخر"). No model can recover the topic from such text; the classifier falls back to the most frequent class (appointments, billing, outage, waste) at very low confidence (median 0.168). Only 3 of them have confidence above 0.5.
2. **Label noise explains another 37 errors (18%).** The data README says about 3% of topic labels are noisy; 37 of 1634 validation rows is 2.3%, which is consistent with that statement. In these rows the model is confident (median confidence 0.983) and, by my reading, usually right. They put a ceiling under validation accuracy that no model change will lift.
3. **Compound messages (9) and genuinely ambiguous boundaries (4)** are a modelling-scope issue: one label is forced on a multi-topic text (label = first topic).
4. **Only about 4 errors (2%) are clear model misses with a topical cue in the text.** The "errors" number therefore overstates how much is fixable by modelling.
5. Gulf messages make up 25% of errors vs 25% of the validation rows; Arabic 56% vs 54%. Emoji appear in 58% of errors vs 58% of rows. Orthographic noise (elongation, diacritics, tatweel) is not the driver.

## Top-3 fixes (ranked by expected effect)

1. **Abstain and route to a human when confidence is low.** Because the largest error bucket carries no topic information, the right behaviour is "topic unclear, needs more detail / triage", not a guess. Selective prediction on validation:

| min confidence | coverage | accuracy on answered | abstained |
|---:|---:|---:|---:|
| 0.0 | 100.0% | 0.872 | 0 |
| 0.3 | 89.7% | 0.953 | 168 |
| 0.4 | 89.6% | 0.954 | 170 |
| 0.5 | 88.8% | 0.961 | 183 |
| 0.7 | 87.9% | 0.964 | 197 |

   (Validation-set numbers, thresholds not tuned on test. A threshold is a product decision about how much triage load is acceptable.)

2. **Audit the labels in the training data**, starting with the rows where a confident model disagrees with the label; the two label-related buckets above are a ready-made review list. Cleaning them would improve both training and the honesty of the metric. This is upstream of the model and cheap.
3. **Use context the text lacks, and handle multi-topic messages.** For follow-up messages ("still waiting") the previous case or the channel is the only usable signal: link a follow-up to the citizen's earlier ticket. For compound messages, split into clauses and return a ranked list or multi-label output instead of one forced label.

## Examples (first three per category)

**label_conflict**

- row 0 [ar/MSA] true `billing` → pred `permits` (conf 0.98): مرة أخرى؟ أريد تجديد رخصتي التجارية. أرجو إصلاح هذا. ؟؟؟
- row 18 [en/English] true `account_access` → pred `appointments` (conf 0.97):  Update: My slot disappeared the night before. And they asked for the same deed copy three times. All good now, thanks. 🙏.... 
- row 27 [en/English] true `appointments` → pred `account_access` (conf 0.99): Honestly, I opened the link from the email and it says it has expired. This is not acceptable. !!!

**hint_conflict**

- row 32 [ar/Gulf] true `billing` → pred `account_access` (conf 0.95): ملاحظة سريعة: الرد الذي وصلني لم يكننن مفيدا الموضوع يخص الحساب. شكرا مقدما.
- row 35 [ar/Gulf] true `billing` → pred `account_access` (conf 0.93): محبط فعلا. ياخي الرد الذي وصلني لم يكن مفيدا الموضوع يخص الحساب. هذه المرة الثالثة التي أبلغ فيها. 😞
- row 55 [ar/Gulf] true `billing` → pred `account_access` (conf 0.95): تذكير: والله الرد الذي وصلني لم يكن مفيدا الموضوع يخص الحساب. ما يصير كذا. كل شيء تــمام الآن، شكرا. !

**compound**

- row 16 [ar/MSA] true `transport` → pred `waste` (conf 0.99): محبط فعلا. الحاويات ممتلئة قرب الحديقة، وأيضا تم إغلاق حسابي بعد عدة محاولات !!!
- row 58 [en/English] true `transport` → pred `waste` (conf 0.99): Question: Stray cats are tearing open the rubbish because the bin is full. And the licence expires next month and nothing is moving
- row 82 [ar/Gulf] true `digital_services` → pred `billing` (conf 0.97): مرة أخرى؟ تم خصم 50 ريال مرتين من فاتورتي، و رُفض الإذن دون ذكر السبب ؟؟؟ رقم هويتي 1441585417

**boundary**

- row 33 [ar/Gulf] true `transport` → pred `billing` (conf 0.93): غيرُ مِقبولِ: خصمت قيمة الرِحلة مُن بطِاُقتي ولم تفتح الُبَوابة ما يصير كذا ؟؟؟
- row 70 [ar/Gulf] true `appointments` → pred `transport` (conf 0.98): مرة أخرى؟ معلومات المسار خاطئة في التطبيق وربي 😡
- row 114 [ar/Gulf] true `digital_services` → pred `billing` (conf 0.97): سؤال: المجموع في الكشف لا يطابق ما ذكره الموظف وربي 🙂

**model_miss**

- row 3 [ar/Gulf] true `billing` → pred `permits` (conf 0.54): ياِخي ما وصلنُي اِلمبِلغ الَمسترَد 150 رَيالّ بعد مرور 4 أُيام اُلله يعافيِكم. هذه الّمرة الثالثة اَلتي أَبلغ فيها. 😞 جّوالي 0588132440
- row 19 [ar/Gulf] true `digital_services` → pred `appointments` (conf 0.70): مرة أخرى؟ لست راضيا عن الخدمة الموضوع يخص التطبيق. وربي 😤
- row 31 [ar/Gulf] true `billing` → pred `permits` (conf 0.66): مرة  أخرى، يا جماعة خرج الَماَل من حَسابي يومّ الجمُعة لكن النظام يظهره غير مسدد ما يصير كذا !!!

**generic_no_topic_cue**

- row 1 [ar/Egyptian] true `transport` → pred `appointments` (conf 0.16): تتتحديث: يا جماعة تم تحويلي من قسم إلى آخر من فضلكم. أقدر سرعة المساعدة! 👍
- row 2 [en/English] true `outage` → pred `appointments` (conf 0.16): unacceptable: still waiting for a reply 😡
- row 4 [ar/MSA] true `outage` → pred `appointments` (conf 0.16): للعلم، ما زلت أنتظر الرد

## Limits of this analysis

- Single annotator (an AI), no inter-annotator agreement. The label-conflict judgements in particular should be spot-checked by a human, preferably an Arabic speaker for the Arabic rows.
- It covers validation errors of the general model only. The sentiment, NER and QA errors are summarised in RESULTS.md and the behaviour report (the sentiment MFT failures are listed in `behaviour_val.json`).
- The annotation is tied to this model's exact predictions; a retrained model needs a fresh reading.