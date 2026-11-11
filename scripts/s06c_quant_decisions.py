"""Stage 6c: a quantisation decision for EVERY model in the service, with its accuracy tax measured on VALIDATION / DEV data.
Classifier: from the s06 ladder (ONNX INT8). NER, QA, query-embedder: PyTorch dynamic INT8 (Linear layers) vs fp32, CPU, batch=1.
Rule (fixed before looking at the numbers): adopt INT8 only if the tax on the primary metric is <= 1.0 point (embedder: MRR@10 <= 2.0 points)
AND p50 latency improves by >= 1.25x. Otherwise keep fp32. Cross-encoder re-rankers are not shipped (they lost on dev), so no decision is needed.
Writes reports/quant_decisions.json and updates artifacts/serving_config.json (default_precision, aux_precision)."""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from bayan import config as C
from bayan.anchors import CaseAnchors, two_stage
from bayan.data import load_cases, load_ner, load_qa, load_queries
from bayan.infer import QAModel, TokenTagger
from bayan.metrics import entity_f1, mrr_at_k, recall_at_k
from bayan.preproc import preprocess
from bayan.qa_utils import evaluate
from bayan.search import Embedder, dense_all, load_index
from bayan.train_utils import save_json

torch.set_num_threads(2)
cfgp = C.ART_DIR / "serving_config.json"
cfg = json.load(open(cfgp))
TAX_PTS, SPEEDUP = 1.0, 1.25
dec = {"rule": f"INT8 only if tax <= {TAX_PTS} pt (embedder MRR@10: <= 2.0 pt) and p50 speedup >= {SPEEDUP}x; else fp32", "models": {}}


def q8(m):
    return torch.quantization.quantize_dynamic(copy.deepcopy(m).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def lat(fn, xs, warm=10):
    for x in xs[:warm]:
        fn(x)
    t = []
    for x in xs:
        t0 = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2), "p99_ms": round(float(np.percentile(t, 99)), 2)}


def verdict(name, m32, m8, l32, l8, tax_pts, tol=TAX_PTS):
    sp = l32["p50_ms"] / max(l8["p50_ms"], 1e-9)
    use = bool(tax_pts <= tol and sp >= SPEEDUP)
    dec["models"][name] = {"fp32": {**m32, **l32}, "int8": {**m8, **l8}, "accuracy_tax_points": round(tax_pts, 3), "p50_speedup": round(sp, 2),
                           "decision": "int8" if use else "fp32", "why": f"tax {tax_pts:.2f} pt (limit {tol}), speedup {sp:.2f}x (need {SPEEDUP}x)"}
    print(name, dec["models"][name]["decision"], dec["models"][name]["why"])
    return "int8" if use else "fp32"


aux = {}
# ---- classifier (from the s06 ladder)
g = cfg["topic"]["general"]
op = C.REPORTS_DIR / f"optimize_{g}.json"
if op.exists():
    o = json.load(open(op))
    r1, r2, r3 = o["rungs"]["R1_torch_fp32_dynamic_pad"], o["rungs"]["R2_ort_fp32"], o["rungs"]["R3_ort_int8_dynamic"]
    best32 = min(r1["p50"], r2["p50"])
    tax = -(r3["accuracy_tax_vs_R1"]) * 100
    sp = best32 / max(r3["p50"], 1e-9)
    use = bool(tax <= TAX_PTS and sp >= SPEEDUP)
    dec["models"][f"classifier:{g}"] = {"val_macro_f1_fp32": r1["val_macro_f1"], "val_macro_f1_int8": r3["val_macro_f1"], "accuracy_tax_points": round(tax, 3),
                                        "p50_ms_best_fp32": best32, "p50_ms_int8": r3["p50"], "p50_speedup": round(sp, 2),
                                        "decision": "int8" if use else "fp32", "why": f"tax {tax:.2f} pt (limit {TAX_PTS}), speedup {sp:.2f}x (need {SPEEDUP}x)"}
    cfg["topic"]["default_precision"] = "int8" if use else "fp32"
    cfg["topic"]["quantisation_decision"] = dec["models"][f"classifier:{g}"]["why"] + f" -> default {cfg['topic']['default_precision']}; fp32 kept as rollback (BAYAN_PRECISION=fp32)"
    print("classifier", dec["models"][f"classifier:{g}"]["decision"])

# ---- NER
ndf = load_ner()
ns = json.load(open(C.ART_DIR / "ner_splits.json"))
nva = ndf[ndf.sentence_id.isin(set(ns["val"]))]
tg = TokenTagger(C.MODELS_DIR / cfg["ner"], device="cpu")
words = nva.tokens.tolist()


def ner_f1(t):
    pred = []
    for i in range(0, len(words), 32):
        pred += t.tag_words(words[i:i + 32])
    return round(entity_f1(nva.tags.tolist(), pred)["f1"], 4)


f32 = ner_f1(tg)
lat32 = lat(lambda w: tg.tag_words([w]), words[:150])
m_fp = tg.model
tg.model = q8(m_fp)
f8 = ner_f1(tg)
lat8 = lat(lambda w: tg.tag_words([w]), words[:150])
tg.model = m_fp
aux["ner"] = verdict("ner", {"val_entity_f1": f32}, {"val_entity_f1": f8}, lat32, lat8, (f32 - f8) * 100)

# ---- QA
qdf = load_qa()
qs = json.load(open(C.ART_DIR / "qa_splits.json"))
qva = qdf[qdf.id.isin(set(qs["val"]))]
qa = QAModel(C.MODELS_DIR / cfg["qa"], device="cpu")


def qa_score(m):
    outs = m.answer(qva.question.tolist(), qva.context.tolist())
    return round(evaluate([o["answer"] for o in outs], qva.answers_text.tolist(), qva.is_impossible.tolist())["score"], 4)


s32 = qa_score(qa)
qs_ = list(zip(qva.question, qva.context))[:60]
l32 = lat(lambda x: qa.answer([x[0]], [x[1]]), qs_)
m_fp = qa.model
qa.model = q8(m_fp)
s8 = qa_score(qa)
l8 = lat(lambda x: qa.answer([x[0]], [x[1]]), qs_)
qa.model = m_fp
aux["qa"] = verdict("qa", {"val_score": s32}, {"val_score": s8}, l32, l8, (s32 - s8) * 100)

# ---- query embedder (index stays fp32; only the query encoder is quantised) on DEV answerable queries
scfg = json.load(open(C.ART_DIR / "search_config.json"))
index, case_ids, _ = load_index(C.ART_DIR / scfg["index_dir"])
cases = load_cases()
anch = CaseAnchors(cases.description.tolist())
qd = load_queries()
dev = set(json.load(open(C.ART_DIR / "query_splits.json"))["dev"])
qd = qd[qd.query_id.isin(dev) & ~qd.no_answer].reset_index(drop=True)
emb = Embedder(scfg["embedder"], scfg["qprefix"], scfg["pprefix"], device="cpu")
W = scfg["anchor_weights"]


def dev_metrics(e):
    rec, mrr = [], []
    for raw in qd["query"]:
        v = e.encode([preprocess(raw)], "query")
        idx, _ = two_stage(dense_all(index, v), raw, anch, W, pool=scfg.get("pool", 300))
        ranked = [case_ids[j] for j in idx]
        gold = qd.set_index("query").loc[raw, "relevant_case_id"]
        gold = gold if isinstance(gold, str) else gold.iloc[0]
        rec.append(recall_at_k(ranked, gold, 10))
        mrr.append(mrr_at_k(ranked, gold, 10))
    return {"dev_recall@10": round(float(np.mean(rec)), 4), "dev_mrr@10": round(float(np.mean(mrr)), 4)}


qtexts = [preprocess(x) for x in qd["query"]]
m32 = dev_metrics(emb)
l32 = lat(lambda t: emb.encode([t], "query"), qtexts)
m_fp = emb.model
emb.model = q8(m_fp)
m8 = dev_metrics(emb)
l8 = lat(lambda t: emb.encode([t], "query"), qtexts)
emb.model = m_fp
aux["embedder"] = verdict("embedder_query_encoder", m32, m8, l32, l8, (m32["dev_mrr@10"] - m8["dev_mrr@10"]) * 100, tol=2.0)

cfg["aux_precision"] = aux
save_json(dec, C.REPORTS_DIR / "quant_decisions.json")
save_json(cfg, cfgp)
save_json(cfg, C.REPORTS_DIR / "serving_config.json")
print("aux_precision", aux, "| classifier default:", cfg["topic"].get("default_precision"))
print("STAGE6C DONE")
