"""Stage 6b: layer-pruned + distilled student of the served topic classifier (the optimisation ladder's 'smaller model' rung).
Student = the teacher's embeddings + K of its transformer layers (evenly spaced) + classifier head, trained with
 loss = alpha * CE(hard label) + (1-alpha) * T^2 * KL(teacher/T || student/T)
on TRAIN feedback (hard+soft) plus the unlabeled-for-this-purpose case descriptions (soft labels only, teacher logits).
Early stopping on VAL macro-F1. Test split is never read. The accuracy tax vs the teacher is measured on VAL (s06 repeats it)."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from bayan import config as C
from bayan.arabic import model_input
from bayan.data import load_cases, load_feedback
from bayan.metrics import bootstrap_ci, macro_f1
from bayan.preproc import PREPROC_VERSION
from bayan.splits import get_part, load_splits
from bayan.train_utils import cls_logits, save_json, set_seed, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="topic_xlmr_all")
ap.add_argument("--name", default="topic_xlmr_l4_kd")
ap.add_argument("--layers", default="2,5,8,11", help="teacher layer indices copied into the student (in order)")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--alpha", type=float, default=0.3)
ap.add_argument("--temp", type=float, default=2.0)
ap.add_argument("--patience", type=int, default=3)
a = ap.parse_args()
C.ensure_dirs()
set_seed(C.SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tdir = C.MODELS_DIR / a.teacher
tmeta = json.load(open(tdir / "bayan_meta.json"))
profile, labels = tmeta["profile"], tmeta["labels"]
l2i = {l: i for i, l in enumerate(labels)}
keep = [int(x) for x in a.layers.split(",")]

df = load_feedback()
tr, va = get_part(df, load_splits(), "train"), get_part(df, load_splits(), "val")
tok = AutoTokenizer.from_pretrained(tdir)
teacher = AutoModelForSequenceClassification.from_pretrained(tdir).to(dev).eval()
Xtr = [model_input(t, profile) for t in tr.text]
ytr = np.array([l2i[l] for l in tr.topic])
Xva = [model_input(t, profile) for t in va.text]
yva = np.array([l2i[l] for l in va.topic])
Xca = [model_input(t, profile) for t in load_cases().description]          # transfer set (soft labels only)
X = Xtr + Xca
Y = np.concatenate([ytr, -100 * np.ones(len(Xca), dtype=int)])
t0 = time.time()
TL = cls_logits(teacher, tok, X, C.TRAIN_MAX_LEN)      # teacher logits, computed once
tva = cls_logits(teacher, tok, Xva, C.MAX_LEN_CLS)
teacher_f1 = float(macro_f1(yva, tva.argmax(1)))
print(f"teacher val macro-F1 {teacher_f1:.4f}; transfer examples {len(Xca)}; teacher logits in {time.time()-t0:.0f}s")
del teacher
torch.cuda.empty_cache()

# ---- student: copy embeddings + selected layers + head
cfg = AutoConfig.from_pretrained(tdir)
cfg.num_hidden_layers = len(keep)
student = AutoModelForSequenceClassification.from_config(cfg)
tsd = AutoModelForSequenceClassification.from_pretrained(tdir).state_dict()
new = {}
for k, v in tsd.items():
    if ".encoder.layer." in k:
        pre, rest = k.split(".encoder.layer.")
        i, tail = rest.split(".", 1)
        if int(i) in keep:
            new[f"{pre}.encoder.layer.{keep.index(int(i))}.{tail}"] = v
    else:
        new[k] = v
missing = student.load_state_dict(new, strict=False)
print("student layers", len(keep), "copied from", keep, "| missing:", missing.missing_keys[:3], "unexpected:", missing.unexpected_keys[:3])
student.to(dev)
n_par = sum(p.numel() for p in student.parameters())
enc = tok(X, truncation=True, max_length=C.TRAIN_MAX_LEN)
spe = (len(X) + a.bs - 1) // a.bs
opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * spe * a.epochs), spe * a.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
best, best_state, best_ep, bad, hist = -1, None, -1, 0, []
for ep in range(a.epochs):
    student.train()
    order = np.random.RandomState(C.SEED + ep).permutation(len(X))
    losses, t0 = [], time.time()
    for i in range(0, len(X), a.bs):
        ix = order[i:i + a.bs]
        b = dict(tok.pad([{"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j]} for j in ix], return_tensors="pt"))
        b = {k: v.to(dev) for k, v in b.items()}
        tl = torch.tensor(TL[ix]).to(dev)
        y = torch.tensor(Y[ix]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            sl = student(**b).logits
        sl = sl.float()
        kd = F.kl_div(F.log_softmax(sl / a.temp, -1), F.softmax(tl / a.temp, -1), reduction="none").sum(-1) * a.temp ** 2
        hard = F.cross_entropy(sl, y, ignore_index=-100, reduction="none")
        has = (y != -100).float()
        loss = ((1 - a.alpha * has) * kd + a.alpha * has * hard).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(float(loss.detach()))
    lg = cls_logits(student, tok, Xva, C.MAX_LEN_CLS)
    f1 = float(macro_f1(yva, lg.argmax(1)))
    rec = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4), "seconds": round(time.time() - t0, 1), "val_macro_f1": round(f1, 4),
           "agreement_with_teacher": round(float((lg.argmax(1) == tva.argmax(1)).mean()), 4)}
    hist.append(rec)
    print(json.dumps(rec))
    if f1 > best + 1e-6:
        best, best_ep, bad = f1, ep + 1, 0
        best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience:
            print("early stopping")
            break
student.load_state_dict(best_state)
lg = cls_logits(student, tok, Xva, C.MAX_LEN_CLS)
prob, pred = softmax(lg), lg.argmax(1)
pt, lo, hi = bootstrap_ci(yva, pred, n_boot=1000)
import pandas as pd  # noqa: E402
out = va[["feedback_id", "citizen_id", "language", "dialect_region", "text", "topic"]].rename(columns={"topic": "y_true"}).copy()
out["pred"] = [labels[i] for i in pred]
out["conf"] = prob.max(1)
for i, l in enumerate(labels):
    out[f"p_{l}"] = prob[:, i]
out.to_csv(C.ART_DIR / "preds" / f"val_{a.name}.csv", index=False)
mdir = C.MODELS_DIR / a.name
student.save_pretrained(mdir)
tok.save_pretrained(mdir)
meta = {**tmeta, "name": a.name, "distilled_from": a.teacher, "kept_teacher_layers": keep, "n_params_M": round(n_par / 1e6, 1),
        "kd": {"alpha": a.alpha, "temperature": a.temp, "lr": a.lr, "epochs_run": len(hist), "best_epoch": best_ep, "transfer_set": "bayan_cases descriptions (soft labels only)"},
        "val_macro_f1": round(pt, 4), "val_ci": [round(lo, 4), round(hi, 4)], "teacher_val_macro_f1": round(teacher_f1, 4),
        "preproc_version": PREPROC_VERSION, "history": hist}
save_json(meta, mdir / "bayan_meta.json")
save_json(meta, C.REPORTS_DIR / f"train_{a.name}.json")
print("RESULT", a.name, "val macro-F1", round(pt, 4), [round(lo, 4), round(hi, 4)], "teacher", round(teacher_f1, 4), "params(M)", round(n_par / 1e6, 1))
