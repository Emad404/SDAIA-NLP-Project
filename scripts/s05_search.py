"""Stage 5: similar-case retrieval. Compare bi-encoders (+ TF-IDF yardstick, + hybrid RRF), build the versioned FAISS
index from the DEV winner, compare cross-encoder re-rankers, tune the no-answer threshold on DEV queries only.
Test queries are never scored here (BAYAN_FINAL=1 in the final stage)."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

from bayan import config as C
from bayan.data import load_cases, load_queries
from bayan.metrics import mrr_at_k, recall_at_k
from bayan.preproc import preprocess
from bayan.anchors import CaseAnchors, two_stage, parse
from bayan.search import Embedder, Reranker, build_index, rrf
from bayan.train_utils import save_json, set_seed

C.ensure_dirs()
set_seed(C.SEED)
dev_dev = "cuda" if torch.cuda.is_available() else "cpu"
cases, q = load_cases(), load_queries()
print("cases", len(cases), cases.columns.tolist(), "queries", len(q), "no_answer", int(q.no_answer.sum()))
cases["clean"] = cases.description.map(preprocess)
q["clean"] = q["query"].map(preprocess)
case_ids = cases.case_id.astype(str).tolist()
pos = {c: i for i, c in enumerate(case_ids)}

# ---- query split: stratified by (no_answer, language); test is frozen
strat = (q.no_answer.astype(int).astype(str) + "_" + q.query_language).values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
dev_i, test_i = next(sss.split(q, strat))
qd, qt = q.iloc[dev_i].reset_index(drop=True), q.iloc[test_i]
test_sha = hashlib.sha256("|".join(sorted(qt.query_id)).encode()).hexdigest()
save_json({"dev": qd.query_id.tolist(), "test": qt.query_id.tolist(), "test_sha256": test_sha}, C.ART_DIR / "query_splits.json")
print("dev queries", len(qd), "(answerable", int((~qd.no_answer).sum()), ") test queries", len(qt), "sha", test_sha[:12])
ans = qd[~qd.no_answer].reset_index(drop=True)
print("dev answerable by language:", ans.query_language.value_counts().to_dict())


def evaluate_rank(ranked_ids_per_q, qdf):
    r10 = [recall_at_k(r, c, 10) for r, c in zip(ranked_ids_per_q, qdf.relevant_case_id)]
    m10 = [mrr_at_k(r, c, 10) for r, c in zip(ranked_ids_per_q, qdf.relevant_case_id)]
    return float(np.mean(r10)), float(np.mean(m10))


REUSE = bool(os.environ.get("S05_REUSE"))   # S05_REUSE=1: skip bi-encoder/reranker comparison, reuse saved results + index
if REUSE:
    cfg0 = json.load(open(C.ART_DIR / "search_config.json"))
    results = json.load(open(C.REPORTS_DIR / "search_dev_biencoders.json"))
    rer_res = json.load(open(C.REPORTS_DIR / "search_dev_rerankers.json"))
    best = cfg0["embedder_name"]
    from bayan.search import load_index
    _idx, _ids, _man = load_index(C.ART_DIR / cfg0["index_dir"])
    V = _idx.reconstruct_n(0, _idx.ntotal)
    emb = Embedder(cfg0["embedder"], cfg0["qprefix"], cfg0["pprefix"], device=dev_dev)
    Q = emb.encode(ans.clean.tolist(), "query")
    dense_mrr = results[best]["mrr@10"]
else:
    results, vecs = {}, {}
    CANDS = [("minilm_l12", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
             ("e5_small", "intfloat/multilingual-e5-small", "query: ", "passage: "),
             ("e5_base", "intfloat/multilingual-e5-base", "query: ", "passage: "),
             ("labse", "sentence-transformers/LaBSE", "", ""),
             ("mpnet_multi", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "")]

    # ---- TF-IDF char n-gram yardstick
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    Xc = tf.fit_transform(cases.clean)
    Xq = tf.transform(ans.clean)
    sims = (Xq @ Xc.T).toarray()
    tf_rank = [[case_ids[j] for j in np.argsort(-s)[:100]] for s in sims]
    results["tfidf_char"] = dict(zip(("recall@10", "mrr@10"), evaluate_rank(tf_rank, ans)))
    print("tfidf_char", results["tfidf_char"])

    for name, ckpt, qp, pp in CANDS:
        try:
            t0 = time.time()
            emb = Embedder(ckpt, qp, pp, device=dev_dev)
            V = emb.encode(cases.clean.tolist(), "passage")
            enc_s = time.time() - t0
            Q = emb.encode(ans.clean.tolist(), "query")
            S = Q @ V.T
            ranked = [[case_ids[j] for j in np.argsort(-s)[:100]] for s in S]
            rr = evaluate_rank(ranked, ans)
            hyb = [rrf([r, t])[:100] for r, t in zip(ranked, tf_rank)]
            rh = evaluate_rank(hyb, ans)
            # per query-language breakdown (cross-language is the hard part)
            same = (ans.query_language.values == cases.set_index("case_id").loc[ans.relevant_case_id, "language"].values)
            results[name] = {"checkpoint": ckpt, "recall@10": rr[0], "mrr@10": rr[1], "hybrid_recall@10": rh[0], "hybrid_mrr@10": rh[1],
                             "mrr@10_same_language": float(np.mean([mrr_at_k(r, c, 10) for r, c, s in zip(ranked, ans.relevant_case_id, same) if s])),
                             "mrr@10_cross_language": float(np.mean([mrr_at_k(r, c, 10) for r, c, s in zip(ranked, ans.relevant_case_id, same) if not s])),
                             "index_seconds": round(enc_s, 1), "dim": int(V.shape[1])}
            print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[name].items()})
            vecs[name] = (V, emb)
            del S
            if len(vecs) > 2:   # keep GPU memory in check: retain only the best two so far
                worst = min(vecs, key=lambda n: results[n]["mrr@10"])
                del vecs[worst]
        except Exception as e:  # noqa
            print("SKIP", name, repr(e)[:200])
    save_json(results, C.REPORTS_DIR / "search_dev_biencoders.json")

    best = max((n for n in results if n != "tfidf_char"), key=lambda n: results[n]["mrr@10"])
    print("BEST bi-encoder on DEV:", best, round(results[best]["mrr@10"], 4))
    V, emb = vecs[best]
    manifest = build_index(V, case_ids, C.ART_DIR / "indexes" / "cases_v1",
                           {"embedder": results[best]["checkpoint"], "embedder_name": best, "qprefix": emb.qprefix, "pprefix": emb.pprefix,
                            "index_version": "cases_v1", "text_field": "description (after preprocess())"})
    print("manifest", {k: manifest[k] for k in ("index_version", "n_vectors", "dim", "embedder")})

    # ---- re-rankers (stage 2) on top-30 of the dense list
    Q = emb.encode(ans.clean.tolist(), "query")
    S = Q @ V.T
    first = [np.argsort(-s)[:30] for s in S]
    RER = [("mmarco_mminilm", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"), ("bge_reranker_base", "BAAI/bge-reranker-base")]
    rer_res = {}
    best_rr = None
    for name, ckpt in RER:
        try:
            rr_ = Reranker(ckpt, device=dev_dev)
            t0 = time.time()
            reranked, top1 = [], []
            for qi, cand in zip(ans.clean, first):
                sc = rr_.score(qi, [cases.clean.iloc[j] for j in cand])
                order = np.argsort(-sc)
                reranked.append([case_ids[cand[j]] for j in order])
                top1.append(float(sc[order[0]]))
            rec, mrr = evaluate_rank(reranked, ans)
            rer_res[name] = {"checkpoint": ckpt, "recall@10": rec, "mrr@10": mrr, "seconds_for_dev_queries": round(time.time() - t0, 1)}
            print("rerank", name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rer_res[name].items()})
            if best_rr is None or mrr > rer_res[best_rr]["mrr@10"]:
                best_rr = name
        except Exception as e:  # noqa
            print("SKIP rerank", name, repr(e)[:200])
    save_json(rer_res, C.REPORTS_DIR / "search_dev_rerankers.json")
    dense_mrr = results[best]["mrr@10"]
    use_rerank = bool(best_rr and rer_res[best_rr]["mrr@10"] > dense_mrr + 0.005)
    print("re-rank decision:", "USE " + str(best_rr) if use_rerank else "SKIP (no dev gain)", "dense mrr", round(dense_mrr, 4),
          "best rerank mrr", round(rer_res[best_rr]["mrr@10"], 4) if best_rr else None)

# ---- stage 2: anchor (district/date) re-scorer, weights tuned on DEV answerable queries only
import itertools
anch = CaseAnchors(cases.description.tolist())
print("anchor coverage of case base:", round(anch.coverage, 4), "| unknown districts:", anch.unknown_districts)
assert anch.coverage > 0.99 and not anch.unknown_districts, "district lexicon incomplete: extend bayan.anchors.DISTRICTS"
Sd = Q @ V.T   # dev answerable x all cases


def run_two_stage(S_all, qraw, w):
    ranked = []
    for s_, qr in zip(S_all, qraw):
        idx, _ = two_stage(s_, qr, anch, w)
        ranked.append([case_ids[j] for j in idx])
    return ranked


GRID = [0.0, 0.05, 0.1, 0.2, 0.4]
grid_rows = []
for wd, wm, wy in itertools.product(GRID, GRID, GRID):
    w_ = {"district": wd, "month": wm, "day": wy}
    rec_, mrr_ = evaluate_rank(run_two_stage(Sd, ans["query"].tolist(), w_), ans)
    grid_rows.append({**w_, "recall@10": rec_, "mrr@10": mrr_})
gdf = pd.DataFrame(grid_rows).sort_values(["mrr@10", "recall@10"], ascending=False)
gdf.to_csv(C.REPORTS_DIR / "search_dev_stage2_grid.csv", index=False)
top_ = gdf.iloc[0]
# prefer the SMALLEST weights within 0.005 MRR of the best (less over-fit to 65 dev queries)
near = gdf[gdf["mrr@10"] >= top_["mrr@10"] - 0.005].assign(tot=lambda d: d.district + d.month + d.day).sort_values(["tot", "mrr@10"], ascending=[True, False])
pick = near.iloc[0]
W = {"district": float(pick.district), "month": float(pick.month), "day": float(pick.day)}
rec2, mrr2 = pick["recall@10"], pick["mrr@10"]
print("stage-2 weights", W, "dev recall@10", round(rec2, 4), "mrr@10", round(mrr2, 4), "| dense-only mrr", round(dense_mrr, 4))
use_anchor = bool(mrr2 > dense_mrr + 0.005)
save_json({"dense_only": {"recall@10": results[best]["recall@10"], "mrr@10": dense_mrr},
           "cross_encoders": rer_res, "anchor_stage2": {"weights": W, "recall@10": float(rec2), "mrr@10": float(mrr2), "adopted": use_anchor,
                                                        "case_anchor_coverage": anch.coverage}},
          C.REPORTS_DIR / "search_dev_stage_comparison.json")
use_rerank = False   # off-the-shelf cross-encoders lost to dense on dev (see search_dev_rerankers.json); kept as an optional flag

# ---- no-answer threshold on ALL dev queries (answerable + no-answer), on the FINAL stage-2 top-1 score
Qd = emb.encode(qd.clean.tolist(), "query")
top_scores = []
for qraw, qv in zip(qd["query"].tolist(), Qd):
    s_ = V @ qv
    if use_anchor:
        _, sc_ = two_stage(s_, qraw, anch, W)
        top_scores.append(float(sc_[0]))
    else:
        top_scores.append(float(s_.max()))
top_scores = np.array(top_scores)
y_na = qd.no_answer.values
grid = np.unique(np.quantile(top_scores, np.linspace(0.02, 0.98, 97)))
best_thr, best_bal = None, -1
for t in grid:
    pred_na = top_scores < t
    tpr = (pred_na & y_na).sum() / max(y_na.sum(), 1)
    tnr = (~pred_na & ~y_na).sum() / max((~y_na).sum(), 1)
    bal = (tpr + tnr) / 2
    if bal > best_bal:
        best_bal, best_thr = bal, float(t)
print("no-answer threshold", round(best_thr, 4), "dev balanced acc", round(best_bal, 4))
cfg = {"embedder_name": best, "embedder": results[best]["checkpoint"], "qprefix": emb.qprefix, "pprefix": emb.pprefix,
       "index_dir": "indexes/cases_v1", "stage2": "anchor_rescore" if use_anchor else "none", "anchor_weights": W, "pool": 300,
       "use_rerank": False, "reranker": None, "no_answer_threshold": best_thr,
       "no_answer_score_source": "stage-2 top-1 score (dense cosine + anchor bonus)" if use_anchor else "dense top-1 cosine",
       "dev_balanced_accuracy_no_answer": best_bal, "dev_n": int(len(qd))}
save_json(cfg, C.ART_DIR / "search_config.json")
save_json(cfg, C.REPORTS_DIR / "search_config.json")
print("STAGE5 DONE")
