"""Debug helper: show dev retrieval failures (query, gold case, top-3 retrieved)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from bayan import config as C
from bayan.data import load_cases, load_queries
from bayan.preproc import preprocess
from bayan.anchors import CaseAnchors, two_stage
from bayan.search import Embedder, dense_all, load_index

cfg = json.load(open(C.ART_DIR / "search_config.json"))
index, ids, man = load_index(C.ART_DIR / cfg["index_dir"])
cases = load_cases().set_index("case_id")
q = load_queries()
dev = set(json.load(open(C.ART_DIR / "query_splits.json"))["dev"])
q = q[q.query_id.isin(dev) & ~q.no_answer].reset_index(drop=True)
emb = Embedder(cfg["embedder"], cfg["qprefix"], cfg["pprefix"], device="cuda")
Q = emb.encode(q["query"].map(preprocess).tolist(), "query")
anch = CaseAnchors(cases.reset_index().description.tolist())
n = 0
for qi, r in q.iterrows():
    idx2, sc2 = two_stage(dense_all(index, Q[qi:qi + 1]), r["query"], anch, cfg["anchor_weights"], pool=cfg.get("pool", 300))
    cand = [ids[j] for j in idx2[:10]]
    if r.relevant_case_id in cand[:1]:
        continue
    rank = cand.index(r.relevant_case_id) + 1 if r.relevant_case_id in cand else None
    print(f"--- {r.query_id} [{r.query_language}] rank_of_gold={rank}\nQ: {r['query'][:130]}\nGOLD({cases.loc[r.relevant_case_id,'language']},{cases.loc[r.relevant_case_id,'topic']}): {cases.loc[r.relevant_case_id,'description'][:150]}")
    for c in cand[:2]:
        print(f"TOP({cases.loc[c,'language']},{cases.loc[c,'topic']}): {cases.loc[c,'description'][:150]}")
    n += 1
    if n >= 8:
        break
