"""Stage 7a: model selection on VALIDATION data only. (1) generalist vs dialect-aware topic models, Gulf-slice paired bootstrap,
(2) dialect classifier for the router, (3) sentiment choice, (4) write serving_config.json + canary reference + public case table.
No frozen-test labels are read."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from bayan import config as C
from bayan.data import load_cases, load_docs, load_feedback, load_ner
from bayan.infer import SeqClassifier
from bayan.metrics import bootstrap_ci, macro_f1, paired_bootstrap_diff
from bayan.preproc import PREPROC_VERSION, preprocess
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

C.ensure_dirs()
P = C.ART_DIR / "preds"
df = load_feedback()
splits = load_splits()
tr, va = get_part(df, splits, "train"), get_part(df, splits, "val")
rd = lambda n: pd.read_csv(P / f"val_{n}.csv").set_index("feedback_id")  # noqa: E731
sel = {}

# ---------------- 1. topic: generalist vs dialect-aware, Gulf slice ----------------
names = [n for n in ("topic_xlmr_all", "topic_xlmr_all_ca", "topic_xlmr_large_all", "topic_xlmr_l4_kd", "topic_xlmr_l3_kd", "topic_camelbert_da_ar", "topic_camelbert_mix_ar", "topic_xlmr_gulf_ft", "topic_xlmr_ar", "topic_xlmr_en")
         if (P / f"val_{n}.csv").exists()]
preds = {n: rd(n) for n in names}
ar_ids = va[va.language == "ar"].feedback_id.values
gulf_ids = va[(va.language == "ar") & (va.dialect_region == "Gulf")].feedback_id.values
print("val ar", len(ar_ids), "gulf", len(gulf_ids))
tab = {}
for n, d in preds.items():
    ids = d.index.values
    row = {"overall_val_macro_f1": round(float(macro_f1(d.y_true, d.pred)), 4), "n": int(len(d))}
    for nm, sub in (("arabic", ar_ids), ("gulf", gulf_ids)):
        sub = [i for i in sub if i in set(ids)]
        if sub:
            dd = d.loc[sub]
            pt, lo, hi = bootstrap_ci(dd.y_true.values, dd.pred.values, n_boot=1000)
            row[f"{nm}_macro_f1"] = round(pt, 4)
            row[f"{nm}_ci"] = [round(lo, 4), round(hi, 4)]
    tab[n] = row
    print(n, row)
gen = "topic_xlmr_all"
if "topic_xlmr_all_ca" in preds and tab["topic_xlmr_all_ca"]["overall_val_macro_f1"] >= tab["topic_xlmr_all"]["overall_val_macro_f1"] - 0.005:
    gen = "topic_xlmr_all_ca"      # case-augmented generalist (fixes the UPPER-CASE invariance failure) unless it costs > 0.5 pt on val
print("generalist:", gen)
comparisons = {}
for n in names:
    if n in (gen, "topic_xlmr_en", "topic_xlmr_l4_kd", "topic_xlmr_l3_kd"):
        continue
    for nm, sub in (("gulf", gulf_ids), ("arabic", ar_ids)):
        a, b = preds[n].loc[sub], preds[gen].loc[sub]
        d_, lo_, hi_ = paired_bootstrap_diff(a.y_true.values, a.pred.values, b.pred.values, n_boot=2000)
        diff = {"diff": round(d_, 4), "ci": [round(lo_, 4), round(hi_, 4)], "n": int(len(sub))}
        comparisons[f"{n}_minus_{gen}_on_{nm}"] = diff
        print(f"{n} - {gen} on {nm}:", diff)

# dialect-aware candidates (declared up-front): CAMeLBERT-DA (dialect-pretrained) and the Gulf-adapted XLM-R. Shipped = better Gulf point estimate on VAL, if > 0.
da_cands = [n for n in ("topic_camelbert_da_ar", "topic_xlmr_gulf_ft") if f"{n}_minus_{gen}_on_gulf" in comparisons]
da = max(da_cands, key=lambda n: comparisons[f"{n}_minus_{gen}_on_gulf"]["diff"]) if da_cands else "topic_camelbert_da_ar"
gulf_key = f"{da}_minus_{gen}_on_gulf"
da_wins = gulf_key in comparisons and comparisons[gulf_key]["diff"] > 0
da_sig = da_wins and comparisons[gulf_key]["ci"][0] > 0
print("dialect-aware candidates:", {n: comparisons[f"{n}_minus_{gen}_on_gulf"] for n in da_cands})
print("chosen dialect-aware variant:", da, "| wins on Gulf (val point estimate):", da_wins, "| significant at 95%:", da_sig)
sel["dialect_aware"] = {"candidates": da_cands, "chosen": da, "wins_val_point_estimate": bool(da_wins), "significant_95": bool(da_sig)}

save_json({"table": tab, "paired_vs_generalist": comparisons, "selection": sel}, C.REPORTS_DIR / "topic_selection_val.json")

# ---------------- 2. dialect classifier for the router ----------------
ar_tr, ar_va = tr[tr.language == "ar"], va[va.language == "ar"]
dclf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True), LinearSVC())
dclf.fit(ar_tr.text.map(preprocess), ar_tr.dialect_region)
dp = dclf.predict(ar_va.text.map(preprocess))
dacc = float((dp == ar_va.dialect_region.values).mean())
gulf_f1 = float(macro_f1((ar_va.dialect_region == "Gulf").astype(int), (dp == "Gulf").astype(int)))
print("dialect clf val acc", round(dacc, 4), "Gulf-vs-rest F1", round(gulf_f1, 4))
joblib.dump(dclf, C.ART_DIR / "dialect_clf.joblib")

# routed topic system on val: Arabic&Gulf(pred) -> DA model, everything else -> generalist
if da_wins:
    routed = preds[gen].pred.copy()
    gp = ar_va.feedback_id[(dp == "Gulf")].values
    routed.loc[gp] = preds[da].pred.loc[gp]
    pt, lo, hi = bootstrap_ci(preds[gen].y_true.values, routed.loc[preds[gen].index].values, n_boot=1000)
    print("routed system overall val macro-F1", round(pt, 4), [round(lo, 4), round(hi, 4)])
    sel["routed_val_macro_f1"] = [round(pt, 4), round(lo, 4), round(hi, 4)]

# ---------------- 3. sentiment choice (transformer vs TF-IDF) ----------------
sent = {}
for n in ("sent_xlmr_all", "sent_camelbert_da_ar"):
    if (P / f"val_{n}.csv").exists():
        d = rd(n)
        sent[n] = round(float(macro_f1(d.y_true, d.pred)), 4)
sent["tfidf_char"] = round(float(macro_f1(*[pd.read_csv(P / "val_tfidf_char_sentiment.csv")[c] for c in ("y_true", "pred")])), 4)
sent["tfidf_word"] = round(float(macro_f1(*[pd.read_csv(P / "val_tfidf_word_sentiment.csv")[c] for c in ("y_true", "pred")])), 4)
print("sentiment val macro-F1:", sent)
save_json(sent, C.REPORTS_DIR / "sentiment_selection_val.json")

# ---------------- 4. serving config, canary reference, public tables ----------------
# NER choice (validation only): prefer the clitic-augmented model if entity-F1 is not worse by > 0.005 and its mean clitic pass-rate is higher
ner_cands = {}
for nm in ("ner_xlmr", "ner_xlmr_clitic_aug"):
    rp = C.REPORTS_DIR / f"train_{nm}.json"
    if (C.MODELS_DIR / nm).exists() and rp.exists():
        m = json.load(open(rp))
        cl = m.get("clitic_val", {})
        ner_cands[nm] = {"entity_f1": m.get("val_entity_f1", m.get("val", {}).get("entity_f1")),
                         "clitic_mean": float(np.mean([v["pass_rate"] for v in cl.values()])) if cl else None,
                         "clitic_min": float(min(v["pass_rate"] for v in cl.values())) if cl else None}
print("NER candidates (val)", ner_cands)
ner_name = "ner_xlmr"
if "ner_xlmr_clitic_aug" in ner_cands and "ner_xlmr" in ner_cands:
    a_, b_ = ner_cands["ner_xlmr_clitic_aug"], ner_cands["ner_xlmr"]
    if a_["entity_f1"] is not None and b_["entity_f1"] is not None and a_["entity_f1"] >= b_["entity_f1"] - 0.005 and a_["clitic_mean"] > b_["clitic_mean"]:
        ner_name = "ner_xlmr_clitic_aug"
save_json({"candidates": ner_cands, "chosen": ner_name}, C.REPORTS_DIR / "ner_selection_val.json")
ner_dir = C.MODELS_DIR / ner_name
qa_names = [p.name for p in C.MODELS_DIR.glob("qa_*")]
qa_val = {n: json.load(open(C.MODELS_DIR / n / "bayan_meta.json"))["val"]["score"] for n in qa_names}
best_qa = max(qa_val, key=qa_val.get) if qa_val else None
print("QA val scores", qa_val, "-> chosen", best_qa)
serving_topic_general = gen      # the single model that is exported / quantised / benchmarked
cfg = {"service_version": "1.0.0", "preproc_version": PREPROC_VERSION, "threads": 2,
       "topic": {"general": serving_topic_general, "dialect_aware": da if da_wins else None, "route_dialects": ["Gulf"] if da_wins else [],
                 "default_precision": "fp32"},      # switched to int8 only if s06 shows an acceptable accuracy tax
       "sentiment": max((k for k in sent if k.startswith("sent_")), key=sent.get), "ner": ner_name if ner_dir.exists() else None,
       "qa": best_qa, "search": (C.ART_DIR / "search_config.json").exists()}
save_json(cfg, C.ART_DIR / "serving_config.json")
print("serving_config", cfg)

cases = load_cases()
cases[["case_id", "description"]].to_csv(C.ART_DIR / "cases_public.csv", index=False)
docs = load_docs()
save_json([{"doc_id": r.doc_id, "title": r.title, "text": r.text} for r in docs.itertuples()], C.ART_DIR / "service_docs.json")
idx_dir = C.ART_DIR / "indexes" / "cases_v1"
if idx_dir.exists():
    save_json([preprocess(t) for t in cases.description.head(64)], idx_dir / "probe_texts.json")

# canary reference: 24 high-confidence validation messages per classifier from the fp32 torch reference model
ref = {"classifiers": {}}
rng = np.random.RandomState(C.SEED)
cand_general = [n for n in (serving_topic_general, "topic_xlmr_l4_kd", "topic_xlmr_l3_kd") if (C.MODELS_DIR / n).exists()]
for key, name in ([("general", serving_topic_general)] + [(f"general:{n}", n) for n in cand_general]
                  + [("dialect_aware", da if da_wins else None), ("sentiment", cfg["sentiment"])]):
    if not name:
        continue
    clf = SeqClassifier(C.MODELS_DIR / name, "torch", device="cpu", threads=2)
    pool = va if clf.meta.get("lang", "all") == "all" else va[va.language == clf.meta["lang"]]
    texts = pool.text.iloc[rng.permutation(len(pool))[:120]].tolist()
    probs = clf.predict_proba(texts)
    keep = np.where(probs.max(1) > 0.9)[0][:24]
    ref["classifiers"][key] = {"texts": [texts[i] for i in keep], "labels": [clf.labels[probs[i].argmax()] for i in keep],
                               "probs": probs[keep].round(6).tolist()}
    print("canary reference", key, len(keep))
if ner_dir.exists():
    from bayan.infer import TokenTagger
    tg = TokenTagger(ner_dir, device="cpu")
    nv = load_ner()
    sents = nv.text.iloc[:200].sample(6, random_state=C.SEED).tolist()
    ref["ner"] = [[t, [(s["type"], s["text"]) for s in tg.extract(t)[1]]] for t in sents]
save_json(ref, C.ART_DIR / "canary_reference.json")
save_json(cfg, C.REPORTS_DIR / "serving_config.json")
print("STAGE7A DONE")
