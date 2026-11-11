"""Stage 4: extractive QA with honest nulls. Train on fact-grouped TRAIN, tune the null margin on VAL.
Test split frozen (BAYAN_FINAL=1); the 20-case smoke set is read only by the final stage."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan import config as C
from bayan.data import load_qa
from bayan.qa_utils import collate, decide, encode_train, evaluate, span_scores, tune_threshold
from bayan.splits import random_split
from bayan.train_utils import fit, save_json, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--zero_shot", action="store_true", help="evaluate the checkpoint without fine-tuning")
a = ap.parse_args()

C.ensure_dirs()
set_seed(C.SEED)
df = load_qa()
print("rows", len(df), "impossible", int(df.is_impossible.sum()), "docs", df.doc_id.nunique())
df["lang_ar"] = df.question.str.contains(r"[؀-ۿ]")
print("arabic questions", int(df.lang_ar.sum()))
df["group"] = [(f"{r.doc_id}|{r.answers_text[0]}" if (not r.is_impossible and r.answers_text) else f"imp|{r.question}") for r in df.itertuples()]
codes = df.group.map(lambda g: int(hashlib.md5(g.encode()).hexdigest()[:8], 16)).values
tr_i, va_i, te_i = random_split(len(df), (0.70, 0.15, 0.15), C.SEED, groups=codes)
assert not (set(codes[tr_i]) & set(codes[te_i])) and not (set(codes[va_i]) & set(codes[te_i])) and not (set(codes[tr_i]) & set(codes[va_i]))
test_sha = hashlib.sha256("|".join(sorted(df.id.iloc[te_i])).encode()).hexdigest()
save_json({"train": df.id.iloc[tr_i].tolist(), "val": df.id.iloc[va_i].tolist(), "test": df.id.iloc[te_i].tolist(),
           "test_sha256": test_sha}, C.ART_DIR / "qa_splits.json")
tr, va = df.iloc[tr_i], df.iloc[va_i]
print({"train": len(tr), "val": len(va), "test": len(te_i)}, "unique train groups", tr.group.nunique())

tok = AutoTokenizer.from_pretrained(a.ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(a.ckpt)
print("context token length p95:", int(np.percentile([len(tok(c)["input_ids"]) for c in tr.context], 95)))


def val_eval(m, thr=None):
    sc = span_scores(m, tok, va.question.tolist(), va.context.tolist())
    golds, imp = va.answers_text.tolist(), va.is_impossible.tolist()
    if thr is None:
        thr, _ = tune_threshold(sc, golds, imp)
    return evaluate(decide(sc, thr), golds, imp), thr


if not a.zero_shot:
    items = encode_train(tok, tr.question, tr.context, tr.answers_start.tolist(), tr.answers_text.tolist(), tr.is_impossible.tolist())
    print("train items with null/unaligned label:", sum(1 for x in items if x["start_positions"] == 0))

    def batches(ep):
        order = np.random.RandomState(C.SEED + ep).permutation(len(items))
        for i in range(0, len(order), a.bs):
            yield collate(tok, [items[j] for j in order[i:i + a.bs]])

    def eval_fn(m):
        r, thr = val_eval(m)
        return {"score": r["score"], "val_f1": r["answerable_f1"], "val_null_acc": r["null_acc"], "thr": thr}

    info = fit(model, batches, eval_fn, (len(items) + a.bs - 1) // a.bs, epochs=a.epochs, lr=a.lr, patience=3)
else:
    info = {"best_epoch": 0, "history": []}
res, thr = val_eval(model)
print("VAL", a.name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, "null margin thr", thr)
mdir = C.MODELS_DIR / a.name
model.save_pretrained(mdir)
tok.save_pretrained(mdir)
meta = {"name": a.name, "checkpoint": a.ckpt, "zero_shot": a.zero_shot, "null_margin_threshold": thr, "val": res,
        "history": info["history"], "best_epoch": info["best_epoch"], "n_train": len(tr), "n_val": len(va),
        "preproc_version": __import__("bayan.preproc", fromlist=["x"]).PREPROC_VERSION}
save_json(meta, mdir / "bayan_meta.json")
save_json(meta, C.REPORTS_DIR / f"train_{a.name}.json")
print("RESULT", a.name, "val score", round(res["score"], 4))
