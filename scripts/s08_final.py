"""Stage 8: the ONE frozen-test evaluation. Reads every frozen split exactly once (lock file prevents silent re-runs).
All choices (models, thresholds, precision, routing) were fixed on validation data before this script ran."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["BAYAN_FINAL"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from bayan import config as C
from bayan.arabic import detect_language
from bayan.baselines import fit_tfidf, predict_tfidf
from bayan.data import load_cases, load_feedback, load_ner, load_pii, load_qa, load_qa_smoke, load_queries
from bayan.infer import QAModel, SeqClassifier, TokenTagger
from bayan.metrics import bootstrap_ci, entity_f1, length_bucket, macro_f1, mrr_at_k, paired_bootstrap_diff, recall_at_k, slice_table
from bayan.ner_utils import CLITICS_AR
from bayan.preproc import preprocess
from bayan.qa_utils import decide, evaluate, exact, f1_score
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

ap = argparse.ArgumentParser()
ap.add_argument("--allow-rerun", action="store_true")
a = ap.parse_args()
lock = C.ART_DIR / "final_run.lock"
if lock.exists() and not a.allow_rerun:
    sys.exit(f"REFUSING: the frozen test set was already evaluated ({lock.read_text()}). Use --allow-rerun only for a documented reason.")
lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))

cfg = json.load(open(C.ART_DIR / "serving_config.json"))
R = {"run_at_utc": lock.read_text(), "serving_config": cfg}
rng = np.random.RandomState(C.SEED)


def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    b = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(x.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


# =========================== topic + sentiment (feedback test) ===========================
df = load_feedback()
splits = load_splits()
tr, te = get_part(df, splits, "train"), get_part(df, splits, "test")
te = te.copy()
te["len_bucket"] = te.text.map(preprocess).map(length_bucket)
print("TEST rows", len(te))
P = {}
# baselines (retrained on TRAIN only)
for task in ("topic", "sentiment"):
    for kind in ("word", "char"):
        vec, clf = fit_tfidf(tr.text.map(preprocess), tr[task], kind)
        P[f"{task}_tfidf_{kind}"] = predict_tfidf(vec, clf, te.text.map(preprocess))
# topic transformer(s)
gen = SeqClassifier(C.MODELS_DIR / cfg["topic"]["general"], "torch")
tp = []
for i in range(0, len(te), 64):
    tp += gen.predict(te.text.tolist()[i:i + 64])[0]
P["topic_general_fp32_torch"] = np.array(tp)
onnx32 = C.ART_DIR / "onnx" / cfg["topic"]["general"] / "model_fp32.onnx"
onnx8 = C.ART_DIR / "onnx" / cfg["topic"]["general"] / "model_int8.onnx"
for nm, pth in (("ort_fp32", onnx32), ("ort_int8", onnx8)):
    if pth.exists():
        c = SeqClassifier(C.MODELS_DIR / cfg["topic"]["general"], "ort", onnx_path=pth, threads=2)
        pp = []
        for i in range(0, len(te), 64):
            pp += c.predict(te.text.tolist()[i:i + 64])[0]
        P[f"topic_general_{nm}"] = np.array(pp)
ref_name = cfg["topic"].get("accuracy_reference")
if ref_name and (C.MODELS_DIR / ref_name).exists():          # the full-size teacher, reported next to the shipped student
    rc_ = SeqClassifier(C.MODELS_DIR / ref_name, "torch")
    rp = []
    for i in range(0, len(te), 64):
        rp += rc_.predict(te.text.tolist()[i:i + 64])[0]
    P["topic_reference_teacher_fp32_torch"] = np.array(rp)
    del rc_
# the shipped system is what the API serves: the default-precision ONNX graph of the general model
served_key = "topic_general_ort_int8" if cfg["topic"].get("default_precision") == "int8" and "topic_general_ort_int8" in P else \
    ("topic_general_ort_fp32" if cfg["topic"].get("default_precision") == "fp32" and "topic_general_ort_fp32" in P else "topic_general_fp32_torch")
print("shipped topic system =", served_key)
R["shipped_topic_key"] = served_key
shipped = P[served_key].copy()
if cfg["topic"].get("dialect_aware"):
    import joblib
    dclf = joblib.load(C.ART_DIR / "dialect_clf.joblib")
    da = SeqClassifier(C.MODELS_DIR / cfg["topic"]["dialect_aware"], "torch")
    is_ar = (te.language == "ar").values
    pred_dialect = np.array([None] * len(te), dtype=object)
    pred_dialect[is_ar] = dclf.predict(te.text[is_ar].map(preprocess))
    gulf_pred = np.array([d == "Gulf" for d in pred_dialect])
    P["topic_dialect_aware_on_pred_gulf"] = None
    ix = np.where(gulf_pred)[0]
    dap = []
    for i in range(0, len(ix), 64):
        dap += da.predict(te.text.iloc[ix[i:i + 64]].tolist())[0]
    shipped[ix] = dap
    # DA variant vs generalist on the GOLD Gulf slice (all gold-Gulf rows, DA applied to all of them)
    gold_gulf = np.where((te.language == "ar") & (te.dialect_region == "Gulf"))[0]
    dg = []
    for i in range(0, len(gold_gulf), 64):
        dg += da.predict(te.text.iloc[gold_gulf[i:i + 64]].tolist())[0]
    y = te.topic.values[gold_gulf]
    d_, lo_, hi_ = paired_bootstrap_diff(y, np.array(dg), P["topic_general_fp32_torch"][gold_gulf], n_boot=2000)
    R["gulf_slice_dialect_aware_vs_general"] = {"n": int(len(gold_gulf)), "dialect_aware_macro_f1": round(float(macro_f1(y, dg)), 4),
                                                "general_macro_f1": round(float(macro_f1(y, P["topic_general_fp32_torch"][gold_gulf])), 4),
                                                "diff": round(d_, 4), "ci": [round(lo_, 4), round(hi_, 4)],
                                                "dialect_aware_wins": bool(d_ > 0), "significant_at_95": bool(lo_ > 0)}
    print("Gulf slice DA vs general", R["gulf_slice_dialect_aware_vs_general"])
    P.pop("topic_dialect_aware_on_pred_gulf")
P["topic_shipped_system"] = shipped
res_topic = {}
for k in [k for k in P if k.startswith("topic_")]:
    pt, lo, hi = bootstrap_ci(te.topic.values, P[k], n_boot=1000)
    res_topic[k] = {"macro_f1": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)], "accuracy": round(float((P[k] == te.topic.values).mean()), 4)}
    print(k, res_topic[k])
for base in ("topic_tfidf_word", "topic_tfidf_char"):
    if "topic_reference_teacher_fp32_torch" in P:
        d_, lo_, hi_ = paired_bootstrap_diff(te.topic.values, P["topic_reference_teacher_fp32_torch"], P[base], n_boot=2000)
        res_topic[f"teacher_minus_{base}"] = {"diff_macro_f1_points": round(100 * d_, 2), "ci_points": [round(100 * lo_, 2), round(100 * hi_, 2)],
                                              "meets_plus_8_points": bool(100 * d_ >= 8), "ci_lower_meets_plus_8": bool(100 * lo_ >= 8)}
    d_, lo_, hi_ = paired_bootstrap_diff(te.topic.values, P["topic_shipped_system"], P[base], n_boot=2000)
    res_topic[f"shipped_minus_{base}"] = {"diff_macro_f1_points": round(100 * d_, 2), "ci_points": [round(100 * lo_, 2), round(100 * hi_, 2)],
                                          "meets_plus_8_points": bool(100 * d_ >= 8), "ci_lower_meets_plus_8": bool(100 * lo_ >= 8)}
    print("shipped -", base, res_topic[f"shipped_minus_{base}"])
R["topic"] = res_topic
tt = te.assign(pred=P["topic_shipped_system"])
sl = slice_table(tt, "topic", "pred", ["language", "dialect_region", "len_bucket"], n_boot=500)
sl.to_csv(C.REPORTS_DIR / "slices_topic_test.csv", index=False)
cls_rows = []
from sklearn.metrics import precision_recall_fscore_support
labs = sorted(te.topic.unique())
pr, rc, f1, sup = precision_recall_fscore_support(te.topic, tt.pred, labels=labs, zero_division=0)
pd.DataFrame({"class": labs, "precision": pr.round(4), "recall": rc.round(4), "f1": f1.round(4), "support": sup}).to_csv(C.REPORTS_DIR / "per_class_topic_test.csv", index=False)
pd.crosstab(tt.topic, tt.pred).to_csv(C.REPORTS_DIR / "confusion_topic_test.csv")
tt[["feedback_id", "language", "dialect_region", "topic", "pred", "text"]].to_csv(C.ART_DIR / "preds" / "test_topic_shipped.csv", index=False)

# sentiment
sc = SeqClassifier(C.MODELS_DIR / cfg["sentiment"], "torch")
sp = []
for i in range(0, len(te), 64):
    sp += sc.predict(te.text.tolist()[i:i + 64])[0]
P["sentiment_model"] = np.array(sp)
res_s = {}
for k in [k for k in P if k.startswith("sentiment_")]:
    pt, lo, hi = bootstrap_ci(te.sentiment.values, P[k], n_boot=1000)
    res_s[k] = {"macro_f1": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)]}
    print(k, res_s[k])
d_, lo_, hi_ = paired_bootstrap_diff(te.sentiment.values, P["sentiment_model"], P["sentiment_tfidf_char"], n_boot=2000)
res_s["model_minus_tfidf_char_points"] = {"diff": round(100 * d_, 2), "ci": [round(100 * lo_, 2), round(100 * hi_, 2)]}
R["sentiment"] = res_s
slice_table(te.assign(pred=P["sentiment_model"]), "sentiment", "pred", ["language", "dialect_region", "len_bucket"], n_boot=500).to_csv(
    C.REPORTS_DIR / "slices_sentiment_test.csv", index=False)

# =========================== PII ===========================
pii = load_pii()
leaks = []
for r in pii.itertuples():
    out = preprocess(r.text)
    exp = str(r.expected_value)
    dig = lambda s: "".join(ch for ch in str(s) if ch.isdigit())  # noqa: E731
    if exp in out or (dig(exp) and dig(exp) in dig(out) and "<" not in out):
        leaks.append([r.text, out])
R["pii"] = {"n_cases": int(len(pii)), "leaks": len(leaks), "recall": round(1 - len(leaks) / len(pii), 4), "details": leaks[:5]}
print("PII recall", R["pii"]["recall"])

# =========================== NER ===========================
ner = TokenTagger(C.MODELS_DIR / cfg["ner"], device="cuda" if os.environ.get("BAYAN_NER_GPU", "1") == "1" else "cpu")
ndf = load_ner()
ns = json.load(open(C.ART_DIR / "ner_splits.json"))
import hashlib
assert hashlib.sha256("|".join(sorted(ns["test"])).encode()).hexdigest() == ns["test_sha256"], "NER test split changed!"
nte = ndf[ndf.sentence_id.isin(set(ns["test"]))]
pred = []
for i in range(0, len(nte), 64):
    pred += ner.tag_words(nte.tokens.tolist()[i:i + 64])
rep = entity_f1(nte.tags.tolist(), pred)
byl = {}
for lang in ("ar", "en"):
    m = (nte.language == lang).values
    byl[lang] = round(entity_f1([t for t, k in zip(nte.tags, m) if k], [p for p, k in zip(pred, m) if k])["f1"], 4)
clit = {}
ar = nte[nte.language == "ar"]
for cl in CLITICS_AR:
    tot = ok = 0
    for toks, tags in zip(ar.tokens, ar.tags):
        for ent in ("LOCATION", "SERVICE", "PERSON"):
            idx = [i for i, t in enumerate(tags) if t == f"B-{ent}"]
            if not idx:
                continue
            new = list(toks)
            new[idx[0]] = cl + toks[idx[0]]
            p = ner.tag_words([new])[0]
            tot += 1
            ok += int(p[idx[0]] == f"B-{ent}")
    clit[cl] = {"pass_rate": round(ok / max(tot, 1), 4), "n": tot}
R["ner"] = {"n_test_sentences": int(len(nte)), "entity_f1": round(rep["f1"], 4), "precision": round(rep["precision"], 4), "recall": round(rep["recall"], 4),
            "per_type": rep["per_type"], "by_language": byl, "clitic_tests": clit, "meets_0.80": bool(rep["f1"] >= 0.80)}
print("NER", R["ner"]["entity_f1"], byl, clit)

# =========================== QA ===========================
qa = QAModel(C.MODELS_DIR / cfg["qa"], device="cuda")
qdf = load_qa()
qs = json.load(open(C.ART_DIR / "qa_splits.json"))
assert hashlib.sha256("|".join(sorted(qs["test"])).encode()).hexdigest() == qs["test_sha256"], "QA test split changed!"
qte = qdf[qdf.id.isin(set(qs["test"]))]
outs = qa.answer(qte.question.tolist(), qte.context.tolist())
r_ = evaluate([o["answer"] for o in outs], qte.answers_text.tolist(), qte.is_impossible.tolist())
R["qa_test_split"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r_.items()}
smoke = load_qa_smoke()
so = qa.answer(smoke.question.tolist(), smoke.context.tolist())
corr_strict = corr_lenient = 0
rows = []
for o, g, imp, q_ in zip(so, smoke.answers_text, smoke.is_impossible, smoke.question):
    if imp:
        ok_s = ok_l = (o["answer"] == "")
    else:
        ok_s = o["answer"] != "" and max(exact(o["answer"], x) for x in g) == 1.0
        ok_l = o["answer"] != "" and max(f1_score(o["answer"], x) for x in g) >= 0.5
    corr_strict += int(ok_s)
    corr_lenient += int(ok_l)
    rows.append({"question": q_, "impossible": bool(imp), "gold": g, "pred": o["answer"], "margin": round(o["margin"], 2), "ok_strict": bool(ok_s)})
R["qa_smoke20"] = {"n": len(smoke), "correct_strict_EM": corr_strict, "correct_lenient_F1>=0.5": corr_lenient, "meets_17_of_20": bool(corr_strict >= 17), "rows": rows}
print("QA smoke", corr_strict, corr_lenient, "/", len(smoke))

# =========================== SEARCH ===========================
from bayan.anchors import CaseAnchors, two_stage
from bayan.search import Embedder, Reranker, dense_all, load_index
scfg = json.load(open(C.ART_DIR / "search_config.json"))
index, case_ids, man = load_index(C.ART_DIR / scfg["index_dir"])
cases = load_cases()
assert list(cases.case_id.astype(str)) == list(case_ids)
anch = CaseAnchors(cases.description.tolist())
qdf2 = load_queries()
qsp = json.load(open(C.ART_DIR / "query_splits.json"))
assert hashlib.sha256("|".join(sorted(qsp["test"])).encode()).hexdigest() == qsp["test_sha256"], "query test split changed!"
qt = qdf2[qdf2.query_id.isin(set(qsp["test"]))].reset_index(drop=True)
emb = Embedder(scfg["embedder"], scfg["qprefix"], scfg["pprefix"], device="cuda")
Q = emb.encode(qt["query"].map(preprocess).tolist(), "query")
use_a = scfg.get("stage2") == "anchor_rescore"
rec_d, mrr_d, rec_a, mrr_a, top1_d, top1_a = [], [], [], [], [], []
for qi, (_, row) in enumerate(qt.iterrows()):
    s_all = dense_all(index, Q[qi:qi + 1])
    dcand = [case_ids[j] for j in np.argsort(-s_all)[:100]]
    top1_d.append(float(s_all.max()))
    idx2, sc2 = two_stage(s_all, row["query"], anch, scfg["anchor_weights"], pool=scfg.get("pool", 300))
    acand = [case_ids[j] for j in idx2]
    top1_a.append(float(sc2[0]))
    if not row.no_answer:
        rec_d.append(recall_at_k(dcand, row.relevant_case_id, 10)); mrr_d.append(mrr_at_k(dcand, row.relevant_case_id, 10))
        rec_a.append(recall_at_k(acand, row.relevant_case_id, 10)); mrr_a.append(mrr_at_k(acand, row.relevant_case_id, 10))
srch = {"n_test_queries": int(len(qt)), "n_answerable": int((~qt.no_answer).sum()),
        "dense_stage1_only": {"recall@10": boot_mean(rec_d), "mrr@10": boot_mean(mrr_d)},
        "two_stage_anchor_rescore": {"recall@10": boot_mean(rec_a), "mrr@10": boot_mean(mrr_a)}}
top = np.array(top1_a if use_a else top1_d)
pred_na = top < scfg["no_answer_threshold"]
srch["no_answer"] = {"threshold": scfg["no_answer_threshold"], "accuracy_on_no_answer_queries": round(float(pred_na[qt.no_answer.values].mean()), 4),
                     "false_no_answer_rate_on_answerable": round(float(pred_na[~qt.no_answer.values].mean()), 4),
                     "n_no_answer": int(qt.no_answer.sum())}
final_rec = srch["two_stage_anchor_rescore" if use_a else "dense_stage1_only"]
srch["shipped_stage2"] = scfg.get("stage2")
srch["meets_recall@10>=0.80"] = bool(final_rec["recall@10"][0] >= 0.80)
srch["meets_mrr@10>=0.70"] = bool(final_rec["mrr@10"][0] >= 0.70)
R["search"] = srch
print("SEARCH", json.dumps(srch)[:700])

save_json(R, C.REPORTS_DIR / "final_test_results.json")
print("STAGE8 DONE")
