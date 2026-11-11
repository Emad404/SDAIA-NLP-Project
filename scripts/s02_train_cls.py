"""Stage 2: fine-tune one sequence classifier (topic or sentiment). Trains on TRAIN, early-stops on VAL macro-F1.
The frozen test split is never read here (get_part refuses it unless BAYAN_FINAL=1)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan import config as C
from bayan.arabic import model_input
from bayan.data import load_feedback
from bayan.metrics import bootstrap_ci, macro_f1
from bayan.splits import get_part, load_splits
from bayan.train_utils import cls_batches, cls_logits, fit, save_json, set_seed, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["topic", "sentiment"])
ap.add_argument("--name", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--lang", default="all", choices=["all", "ar", "en"])
ap.add_argument("--profile", default="xlmr")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--case_aug", type=float, default=0.0, help="fraction of English train rows for which an UPPER- and a lower-cased copy is added (invariance fix)")
a = ap.parse_args()

C.ensure_dirs()
set_seed(a.seed)
df = load_feedback()
splits = load_splits()
tr, va = get_part(df, splits, "train"), get_part(df, splits, "val")
if a.lang != "all":
    tr, va = tr[tr.language == a.lang], va[va.language == a.lang]
labels = sorted(df[a.task].unique())
l2i = {l: i for i, l in enumerate(labels)}

def prep(d):
    return [model_input(t, a.profile) for t in d.text]

Xtr, Xva = prep(tr), prep(va)
ytr, yva = [l2i[l] for l in tr[a.task]], np.array([l2i[l] for l in va[a.task]])
if a.case_aug > 0:                      # augment on RAW text (before PII masking), so placeholders like <PHONE> are never lower-cased
    rs = np.random.RandomState(a.seed)
    en = tr[tr.language == "en"]
    pick = en[rs.rand(len(en)) < a.case_aug]
    for fn in (str.upper, str.lower):
        Xtr += [model_input(fn(t), a.profile) for t in pick.text]
        ytr += [l2i[l] for l in pick[a.task]]
    print("case-augmented rows added:", 2 * len(pick), "-> n_train", len(Xtr))
tok = AutoTokenizer.from_pretrained(a.ckpt)
model = AutoModelForSequenceClassification.from_pretrained(a.ckpt, num_labels=len(labels),
                                                           id2label=dict(enumerate(labels)), label2id=l2i)
spe = (len(Xtr) + a.bs - 1) // a.bs

def eval_fn(m):
    lg = cls_logits(m, tok, Xva, C.MAX_LEN_CLS)
    return {"score": float(macro_f1(yva, lg.argmax(1))), "val_acc": float((lg.argmax(1) == yva).mean())}

print(f"== {a.name}: {a.ckpt} lang={a.lang} n_train={len(Xtr)} n_val={len(Xva)}")
info = fit(model, lambda ep: cls_batches(tok, Xtr, ytr, a.bs, C.TRAIN_MAX_LEN, True, a.seed + ep), eval_fn, spe,
           epochs=a.epochs, lr=a.lr)
lg = cls_logits(model, tok, Xva, C.MAX_LEN_CLS)
prob, pred = softmax(lg), lg.argmax(1)
pt, lo, hi = bootstrap_ci(yva, pred, n_boot=1000)
out = va[["feedback_id", "citizen_id", "language", "dialect_region", "text", a.task]].rename(columns={a.task: "y_true"}).copy()
out["pred"] = [labels[i] for i in pred]
out["conf"] = prob.max(1)
for i, l in enumerate(labels):
    out[f"p_{l}"] = prob[:, i]
(C.ART_DIR / "preds").mkdir(parents=True, exist_ok=True)
out.to_csv(C.ART_DIR / "preds" / f"val_{a.name}.csv", index=False)
mdir = C.MODELS_DIR / a.name
model.save_pretrained(mdir)
tok.save_pretrained(mdir)
meta = {"name": a.name, "task": a.task, "checkpoint": a.ckpt, "lang": a.lang, "profile": a.profile, "labels": labels,
        "preproc_version": __import__("bayan.preproc", fromlist=["x"]).PREPROC_VERSION, "seed": a.seed,
        "epochs_run": len(info["history"]), "best_epoch": info["best_epoch"], "lr": a.lr, "bs": a.bs,
        "max_len_train": C.TRAIN_MAX_LEN, "max_len_infer": C.MAX_LEN_CLS, "case_aug": a.case_aug, "n_train": len(Xtr), "n_val": len(Xva),
        "val_macro_f1": round(pt, 4), "val_ci": [round(lo, 4), round(hi, 4)], "history": info["history"]}
save_json(meta, mdir / "bayan_meta.json")
save_json(meta, C.REPORTS_DIR / f"train_{a.name}.json")
print("RESULT", a.name, "val macro-F1", round(pt, 4), [round(lo, 4), round(hi, 4)])
