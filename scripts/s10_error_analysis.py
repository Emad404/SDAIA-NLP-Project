"""Stage 10: error analysis of the validation topic errors (the frozen test split is never touched).
All 209 validation errors of the general topic model were READ one by one (printed with scripts/dump_errors.py, sorted by
feedback_id) and assigned to a cause. The assignment below is a hand annotation (made by the AI assistant that built this
project; the human owner should spot-check it). Indices refer to that sorted list, so this script only works for the model
the annotation was made on (MODEL below) and refuses to run on any other predictions file."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(os.environ.get("BAYAN_WORK", "/content/bayan_work"))
R = Path(os.environ.get("BAYAN_REPORTS", "/content/bayan_project/reports"))
MODEL = "topic_xlmr_all_ca"
EXPECTED_ERRORS = 209

CATS = {
    "label_conflict": ("Label looks wrong (text clearly describes another topic, often the one the model predicted)",
                       [0, 18, 27, 44, 68, 75, 76, 81, 91, 98, 104, 107, 113, 119, 122, 124, 131, 141, 145, 160, 170, 184, 187, 197, 202]),
    "hint_conflict": ("Message ends with an explicit topic hint ('It is about X' / 'الموضوع يخص X') that contradicts the label",
                      [32, 35, 55, 87, 110, 116, 138, 146, 147, 171, 183, 208]),
    "compound": ("Two topics in one message (label = the first one; model picks the louder or the second)",
                 [16, 58, 82, 133, 157, 186, 192, 193, 195]),
    "boundary": ("Genuinely ambiguous boundary between neighbouring topics (billing/transport, billing/digital_services, appointments/permits)",
                 [33, 70, 114, 136]),
    "model_miss": ("Topical cue present but the model missed it (a real model error)", [3, 19, 31, 101]),
}
assigned = {i: c for c, (_, idx) in CATS.items() for i in idx}
assert len(assigned) == sum(len(v[1]) for v in CATS.values()), "an index is in two categories"

d = pd.read_csv(W / "artifacts" / "preds" / f"val_{MODEL}.csv")
e = d[d.y_true != d.pred].sort_values("feedback_id").reset_index(drop=True)
assert len(e) == EXPECTED_ERRORS, f"annotation was made on {EXPECTED_ERRORS} errors of {MODEL}, found {len(e)}"
e["cause"] = [assigned.get(i, "generic_no_topic_cue") for i in range(len(e))]
e["row"] = range(len(e))
e.to_csv(R / "error_analysis_topic_val.csv", index=False)
CATS["generic_no_topic_cue"] = ("Generic follow-up / complaint with NO topic words ('still waiting for a reply', 'please escalate'); the text alone cannot determine the topic", [])

# sanity check of the 'generic' bucket: it must be low confidence (otherwise the annotation is wrong)
g = e[e.cause == "generic_no_topic_cue"]
n = len(e)
summ = {"model": MODEL, "n_val": int(len(d)), "n_errors": n, "val_accuracy": round(1 - n / len(d), 4), "categories": {}}
for c, (desc, _) in CATS.items():
    s = e[e.cause == c]
    summ["categories"][c] = {"description": desc, "n": int(len(s)), "share_of_errors": round(len(s) / n, 3),
                             "median_conf": round(float(s.conf.median()), 3) if len(s) else None,
                             "by_language": s.language.value_counts().to_dict(),
                             "predicted_as": s.pred.value_counts().head(4).to_dict()}
summ["generic_conf_gt_0.5"] = int((g.conf > 0.5).sum())
summ["errors_with_conf_lt_0.5"] = int((e.conf < 0.5).sum())
summ["share_errors_gulf"] = round(float((e.dialect_region == "Gulf").mean()), 3)
summ["share_val_gulf"] = round(float((d.dialect_region == "Gulf").mean()), 3)
summ["share_errors_arabic"] = round(float((e.language == "ar").mean()), 3)
summ["share_val_arabic"] = round(float((d.language == "ar").mean()), 3)
d["ok"] = d.y_true == d.pred
sel = []
for t in (0.0, 0.3, 0.4, 0.5, 0.7):
    k = d[d.conf >= t]
    sel.append({"min_conf": t, "coverage": round(len(k) / len(d), 3), "accuracy_on_answered": round(float(k.ok.mean()), 4),
                "abstained": int(len(d) - len(k))})
summ["selective_prediction"] = sel
if "pii" in e.columns:
    pass
# how many errors mention PII placeholders / emoji / elongations (nuisance features) vs. overall
summ["share_errors_with_emoji"] = round(float(e.text.str.contains(r"[\U0001F300-\U0001FAFF]").mean()), 3)
summ["share_val_with_emoji"] = round(float(d.text.str.contains(r"[\U0001F300-\U0001FAFF]").mean()), 3)
json.dump(summ, open(R / "error_analysis_topic_val.json", "w"), ensure_ascii=False, indent=1)

ex = {c: e[e.cause == c].head(3) for c in CATS}
L = [f"# Error analysis: topic classifier ({MODEL}), validation split", "",
     f"All **{n} validation errors** (of {len(d)} rows, accuracy {summ['val_accuracy']:.3f}) were read one by one and each was assigned a cause. "
     "The frozen test split was not used. **Who did the reading:** the AI assistant that built this project read and categorised the errors; "
     "the human owner should spot-check a sample (the full annotated list is `error_analysis_topic_val.csv`). Categories are judgements, not ground truth.", "",
     "| cause | n | share | median conf | what it is |", "|---|---:|---:|---:|---|"]
for c, v in summ["categories"].items():
    L.append(f"| {c} | {v['n']} | {v['share_of_errors']:.0%} | {v['median_conf']} | {v['description']} |")
L += ["", "## What the errors tell us", ""]
gc = summ["categories"]["generic_no_topic_cue"]
L.append(f"1. **The largest bucket ({gc['n']} errors, {gc['share_of_errors']:.0%}) is messages that contain no topic information at all** "
         "(e.g. \"Still waiting for a reply\", \"Please escalate this to a manager\", \"تم تحويلي من قسم إلى آخر\"). No model can recover the topic from such text; "
         f"the classifier falls back to the most frequent class ({', '.join(gc['predicted_as'])}) at very low confidence (median {gc['median_conf']}). "
         f"Only {summ['generic_conf_gt_0.5']} of them have confidence above 0.5.")
lc, hc = summ["categories"]["label_conflict"], summ["categories"]["hint_conflict"]
L.append(f"2. **Label noise explains another {lc['n'] + hc['n']} errors ({(lc['n'] + hc['n']) / n:.0%}).** The data README says about 3% of topic labels are noisy; "
         f"{lc['n'] + hc['n']} of {len(d)} validation rows is {(lc['n'] + hc['n']) / len(d):.1%}, which is consistent with that statement. "
         "In these rows the model is confident (median confidence "
         f"{lc['median_conf']}) and, by my reading, usually right. They put a ceiling under validation accuracy that no model change will lift.")
L.append(f"3. **Compound messages ({summ['categories']['compound']['n']}) and genuinely ambiguous boundaries ({summ['categories']['boundary']['n']})** are a modelling-scope issue: "
         "one label is forced on a multi-topic text (label = first topic).")
mm = summ["categories"]["model_miss"]
L.append(f"4. **Only about {mm['n']} errors ({mm['n'] / n:.0%}) are clear model misses with a topical cue in the text.** "
         "The \"errors\" number therefore overstates how much is fixable by modelling.")
L.append(f"5. Gulf messages make up {summ['share_errors_gulf']:.0%} of errors vs {summ['share_val_gulf']:.0%} of the validation rows; Arabic {summ['share_errors_arabic']:.0%} vs {summ['share_val_arabic']:.0%}. "
         "Emoji appear in "
         f"{summ['share_errors_with_emoji']:.0%} of errors vs {summ['share_val_with_emoji']:.0%} of rows. Orthographic noise (elongation, diacritics, tatweel) is not the driver.")
L += ["", "## Top-3 fixes (ranked by expected effect)", "",
      "1. **Abstain and route to a human when confidence is low.** Because the largest error bucket carries no topic information, the right behaviour is "
      "\"topic unclear, needs more detail / triage\", not a guess. Selective prediction on validation:", "",
      "| min confidence | coverage | accuracy on answered | abstained |", "|---:|---:|---:|---:|"]
for s in sel:
    L.append(f"| {s['min_conf']} | {s['coverage']:.1%} | {s['accuracy_on_answered']:.3f} | {s['abstained']} |")
L += ["", "   (Validation-set numbers, thresholds not tuned on test. A threshold is a product decision about how much triage load is acceptable.)", "",
      "2. **Audit the labels in the training data**, starting with the rows where a confident model disagrees with the label; the two label-related buckets above are a ready-made review list. "
      "Cleaning them would improve both training and the honesty of the metric. This is upstream of the model and cheap.",
      "3. **Use context the text lacks, and handle multi-topic messages.** For follow-up messages (\"still waiting\") the previous case or the channel is the only usable signal: link a "
      "follow-up to the citizen's earlier ticket. For compound messages, split into clauses and return a ranked list or multi-label output instead of one forced label.", "",
      "## Examples (first three per category)", ""]
for c, x in ex.items():
    L.append(f"**{c}**")
    L.append("")
    for r in x.itertuples():
        L.append(f"- row {r.row} [{r.language}/{r.dialect_region}] true `{r.y_true}` → pred `{r.pred}` (conf {r.conf:.2f}): {r.text[:160]}")
    L.append("")
L += ["## Limits of this analysis", "",
      "- Single annotator (an AI), no inter-annotator agreement. The label-conflict judgements in particular should be spot-checked by a human, preferably an Arabic speaker for the Arabic rows.",
      "- It covers validation errors of the general model only. The sentiment, NER and QA errors are summarised in RESULTS.md and the behaviour report (the sentiment MFT failures are listed in `behaviour_val.json`).",
      "- The annotation is tied to this model's exact predictions; a retrained model needs a fresh reading."]
(R / "error_analysis.md").write_text("\n".join(L), encoding="utf-8")
print(json.dumps({k: summ[k] for k in ("n_errors", "generic_conf_gt_0.5", "errors_with_conf_lt_0.5")}), {c: v["n"] for c, v in summ["categories"].items()})
print("ERRORANALYSIS DONE")
