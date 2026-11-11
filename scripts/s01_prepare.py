"""Stage 1: clean data with the ONE preprocessing module, make leak-free splits, TF-IDF baselines,
dialect/length audit and tokenizer (fertility) audit. Touches NO frozen-test labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from bayan import config as C
from bayan.baselines import fit_tfidf, predict_tfidf
from bayan.data import load_feedback
from bayan.metrics import bootstrap_ci, length_bucket, slice_table
from bayan.preproc import PREPROC_VERSION, preprocess
from bayan.splits import get_part, make_feedback_splits, save_splits
from bayan.train_utils import save_json

C.ensure_dirs()
(C.ART_DIR / "preds").mkdir(parents=True, exist_ok=True)

df = load_feedback()
print("rows", len(df), "columns", list(df.columns))
df["text_clean"] = df["text"].map(preprocess)
df["n_words"] = df["text_clean"].str.split().str.len()
df["len_bucket"] = df["text_clean"].map(length_bucket)

splits = make_feedback_splits(df)
save_splits(splits)
df["split"] = "train"
df.loc[df.feedback_id.isin(set(splits["val"])), "split"] = "val"
df.loc[df.feedback_id.isin(set(splits["test"])), "split"] = "test"
df.to_csv(C.ART_DIR / "feedback_clean.csv", index=False)
print({k: len(splits[k]) for k in ("train", "val", "test")}, "test sha", splits["test_sha256"][:12])

# ---------------- data audit (train+val only; the frozen test labels are not inspected) ----------------
dev = df[df.split != "test"]
audit = {
    "preproc_version": PREPROC_VERSION,
    "n_rows": int(len(df)), "split_sizes": {k: int(len(splits[k])) for k in ("train", "val", "test")},
    "test_sha256": splits["test_sha256"],
    "language": df.language.value_counts().to_dict(),
    "arabic_dialect_share_pct": (dev[dev.language == "ar"].dialect_region.value_counts(normalize=True) * 100).round(1).to_dict(),
    "topic_counts_trainval": dev.topic.value_counts().to_dict(),
    "sentiment_counts_trainval": dev.sentiment.value_counts().to_dict(),
    "words_per_message": dev.n_words.describe().round(2).to_dict(),
    "citizens": int(df.citizen_id.nunique()),
    "messages_changed_by_preproc_pct": round(float((df.text != df.text_clean).mean() * 100), 1),
    "pii_placeholders_inserted": {p: int(df.text_clean.str.contains(p, regex=False).sum()) for p in ("<PHONE>", "<NATIONAL_ID>", "<EMAIL>")},
    "residual_long_digit_runs": int(df.text_clean.str.contains(r"\d{9,}").sum()),
}
save_json(audit, C.REPORTS_DIR / "data_audit.json")
print(audit)

# ---------------- TF-IDF baselines (train -> validation) ----------------
tr, va = df[df.split == "train"], df[df.split == "val"]
res = {}
for task in ("topic", "sentiment"):
    for kind in ("word", "char"):
        vec, clf = fit_tfidf(tr.text_clean, tr[task], kind)
        pred = predict_tfidf(vec, clf, va.text_clean)
        point, lo, hi = bootstrap_ci(va[task], pred, n_boot=1000)
        res[f"{task}_{kind}"] = {"val_macro_f1": round(point, 4), "ci": [round(lo, 4), round(hi, 4)]}
        out = va[["feedback_id", "citizen_id", "language", "dialect_region", "len_bucket", "text", task]].copy()
        out = out.rename(columns={task: "y_true"})
        out["pred"] = pred
        out.to_csv(C.ART_DIR / "preds" / f"val_tfidf_{kind}_{task}.csv", index=False)
        print(task, kind, res[f"{task}_{kind}"])
save_json(res, C.REPORTS_DIR / "baseline_tfidf_val.json")

# ---------------- tokenizer / fertility audit (Lab 1 decision evidence) ----------------
from transformers import AutoTokenizer

cands = {"xlm-roberta-base": "XLM-R", "bert-base-multilingual-cased": "mBERT",
         "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT-mix", "CAMeL-Lab/bert-base-arabic-camelbert-da": "CAMeLBERT-DA",
         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "MiniLM-multilingual"}
rows = []
for ckpt, label in cands.items():
    try:
        tok = AutoTokenizer.from_pretrained(ckpt)
    except Exception as e:  # noqa
        print("skip", ckpt, e)
        continue
    for lang in ("ar", "en"):
        texts = tr.loc[tr.language == lang, "text_clean"].head(500).tolist()
        pieces = sum(len(tok.tokenize(t)) for t in texts)
        words = sum(len(t.split()) for t in texts)
        lens = [len(tok(t)["input_ids"]) for t in texts]
        rows.append({"model": label, "checkpoint": ckpt, "language": lang, "fertility": round(pieces / words, 3),
                     "p95_len": int(np.percentile(lens, 95)), "max_len": int(max(lens)), "vocab": len(tok)})
tk = pd.DataFrame(rows)
tk.to_csv(C.REPORTS_DIR / "tokenizer_audit.csv", index=False)
print(tk.to_string())
print("STAGE1 DONE")
