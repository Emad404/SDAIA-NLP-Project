"""Stage 7b (VALIDATION data only): behavioural suite for topic + sentiment, and the validation error sheet used for hand-reading."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from bayan import config as C
from bayan.behaviour import mft_sent_cases, mft_topic_cases, run_invariance, run_mft
from bayan.data import load_feedback
from bayan.infer import SeqClassifier
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

C.ensure_dirs()
cfg = json.load(open(C.ART_DIR / "serving_config.json"))
df = load_feedback()
va = get_part(df, load_splits(), "val")
topic = SeqClassifier(C.MODELS_DIR / cfg["topic"]["general"], "torch")
sent = SeqClassifier(C.MODELS_DIR / cfg["sentiment"], "torch")


def pred_topic(texts):
    out = []
    for i in range(0, len(texts), 64):
        out += topic.predict(texts[i:i + 64])[0]
    return out


def pred_sent(texts):
    out = []
    for i in range(0, len(texts), 64):
        out += sent.predict(texts[i:i + 64])[0]
    return out


res = {}
mt = run_mft(pred_topic, mft_topic_cases())
ms = run_mft(pred_sent, mft_sent_cases())
print("MFT topic", round(mt["pass_rate"], 3), {k: round(v["pass_rate"], 3) for k, v in mt["by_group"].items()})
print("MFT sentiment", round(ms["pass_rate"], 3), {k: round(v["pass_rate"], 3) for k, v in ms["by_group"].items()})
core = [c for c in mft_topic_cases() if c["group"] != "gulf"]
mt_core = run_mft(pred_topic, core)
print("MFT topic core (EN+MSA)", round(mt_core["pass_rate"], 3))
texts = va.text.tolist()
it, ft = run_invariance(pred_topic, texts)
isn, fs = run_invariance(pred_sent, [t for t in texts if True])
print("INV topic mean", round(it["_mean"], 4), "min", round(it["_min"], 4), {k: round(v["invariance"], 3) for k, v in it.items() if not k.startswith("_")})
print("INV sentiment mean", round(isn["_mean"], 4), "min", round(isn["_min"], 4), {k: round(v["invariance"], 3) for k, v in isn.items() if not k.startswith("_")})
save_json({"mft_topic": {k: v for k, v in mt.items()}, "mft_topic_core": {k: v for k, v in mt_core.items() if k != "failures"},
           "mft_sentiment": ms, "invariance_topic": it, "invariance_topic_failures": ft,
           "invariance_sentiment": isn, "invariance_sentiment_failures": fs}, C.REPORTS_DIR / "behaviour_val.json")

# ---- error sheet (served topic model, validation split)
pv = pd.read_csv(C.ART_DIR / "preds" / f"val_{cfg['topic']['general']}.csv")
err = pv[pv.y_true != pv.pred].sort_values("conf", ascending=False).reset_index(drop=True)
print("val errors", len(err), "of", len(pv), f"({len(err) / len(pv):.1%})")
err[["feedback_id", "language", "dialect_region", "y_true", "pred", "conf", "text"]].to_csv(C.REPORTS_DIR / "errors_topic_val_all.csv", index=False)
print(pd.crosstab(err.y_true, err.pred).to_string())
print("STAGE7B DONE")
