"""Stage 2c: dialect-aware variant #2 = Gulf-adapted XLM-R. Starts from the fine-tuned generalist (topic_xlmr_all) and continues training on
ALL Arabic train rows with Gulf rows oversampled x3 (low LR), early-stopping on GULF-validation macro-F1. Test split never read.
(Variant #1 is CAMeLBERT-DA from s02.) Which one is shipped is decided on validation in s07_select.py; the test verdict is in s08."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bayan import config as C
from bayan.arabic import model_input
from bayan.data import load_feedback
from bayan.metrics import bootstrap_ci, macro_f1
from bayan.preproc import PREPROC_VERSION
from bayan.splits import get_part, load_splits
from bayan.train_utils import cls_batches, cls_logits, fit, save_json, set_seed, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="topic_xlmr_all")
ap.add_argument("--name", default="topic_xlmr_gulf_ft")
ap.add_argument("--oversample", type=int, default=3)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
C.ensure_dirs()
set_seed(C.SEED)
bdir = C.MODELS_DIR / a.base
bmeta = json.load(open(bdir / "bayan_meta.json"))
labels, profile = bmeta["labels"], bmeta["profile"]
l2i = {l: i for i, l in enumerate(labels)}
df = load_feedback()
sp = load_splits()
tr, va = get_part(df, sp, "train"), get_part(df, sp, "val")
tr_ar = tr[tr.language == "ar"]
gulf = tr_ar[tr_ar.dialect_region == "Gulf"]
tr_mix = pd.concat([tr_ar] + [gulf] * (a.oversample - 1)).reset_index(drop=True)
va_ar = va[va.language == "ar"]
va_g = va_ar[va_ar.dialect_region == "Gulf"]
print(f"train rows: arabic {len(tr_ar)}, gulf {len(gulf)}, mixed (gulf x{a.oversample}) {len(tr_mix)} | val arabic {len(va_ar)} gulf {len(va_g)}")
tok = AutoTokenizer.from_pretrained(bdir)
model = AutoModelForSequenceClassification.from_pretrained(bdir)
X = [model_input(t, profile) for t in tr_mix.text]
y = [l2i[l] for l in tr_mix.topic]
Xg = [model_input(t, profile) for t in va_g.text]
yg = np.array([l2i[l] for l in va_g.topic])
Xa = [model_input(t, profile) for t in va_ar.text]
ya = np.array([l2i[l] for l in va_ar.topic])
spe = (len(X) + a.bs - 1) // a.bs
base_g = float(macro_f1(yg, cls_logits(model.cuda(), tok, Xg, C.MAX_LEN_CLS).argmax(1)))
print("starting Gulf-val macro-F1 (generalist):", round(base_g, 4))


def eval_fn(m):
    lg = cls_logits(m, tok, Xg, C.MAX_LEN_CLS)
    return {"score": float(macro_f1(yg, lg.argmax(1))), "arabic_val_macro_f1": float(macro_f1(ya, cls_logits(m, tok, Xa, C.MAX_LEN_CLS).argmax(1)))}


info = fit(model, lambda ep: cls_batches(tok, X, y, a.bs, C.TRAIN_MAX_LEN, True, C.SEED + ep), eval_fn, spe, epochs=a.epochs, lr=a.lr, patience=2)
lg = cls_logits(model, tok, Xa, C.MAX_LEN_CLS)
prob, pred = softmax(lg), lg.argmax(1)
pt, lo, hi = bootstrap_ci(ya, pred, n_boot=1000)
out = va_ar[["feedback_id", "citizen_id", "language", "dialect_region", "text", "topic"]].rename(columns={"topic": "y_true"}).copy()
out["pred"] = [labels[i] for i in pred]
out["conf"] = prob.max(1)
for i, l in enumerate(labels):
    out[f"p_{l}"] = prob[:, i]
out.to_csv(C.ART_DIR / "preds" / f"val_{a.name}.csv", index=False)
mdir = C.MODELS_DIR / a.name
model.save_pretrained(mdir)
tok.save_pretrained(mdir)
meta = {**bmeta, "name": a.name, "lang": "ar", "adapted_from": a.base, "gulf_oversample": a.oversample, "lr": a.lr, "epochs_run": len(info["history"]),
        "best_epoch": info["best_epoch"], "val_macro_f1": round(pt, 4), "val_ci": [round(lo, 4), round(hi, 4)], "gulf_val_macro_f1_before": round(base_g, 4),
        "gulf_val_macro_f1_after": round(info["best_score"], 4), "preproc_version": PREPROC_VERSION, "history": info["history"]}
save_json(meta, mdir / "bayan_meta.json")
save_json(meta, C.REPORTS_DIR / f"train_{a.name}.json")
print("RESULT", a.name, "arabic val macro-F1", round(pt, 4), "| Gulf-val", round(base_g, 4), "->", round(info["best_score"], 4))
