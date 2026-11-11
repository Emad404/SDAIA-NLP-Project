"""Stage 3: NER (PERSON / LOCATION / SERVICE) with XLM-R token classification + clitic robustness tests.
The NER test split is frozen (hash stored) and only read with BAYAN_FINAL=1."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bayan import config as C
from bayan.data import load_ner
from bayan.metrics import entity_f1
from bayan.ner_utils import CLITICS_AR, prep_token, words_with_spans
from bayan.splits import FrozenTestLocked, random_split
from bayan.train_utils import fit, save_json, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="xlm-roberta-base")
ap.add_argument("--name", default="ner_xlmr")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--clitic_aug", type=float, default=0.0, help="prob. that each Arabic entity-start token gets a random clitic prefix in an extra training copy")
a = ap.parse_args()

C.ensure_dirs()
set_seed(C.SEED)
df = load_ner()
print("sentences", len(df), "languages", df.language.value_counts().to_dict())
labels = sorted({t for tags in df.tags for t in tags}, key=lambda t: (t != "O", t))
l2i = {l: i for i, l in enumerate(labels)}
print("labels", labels)
join_ok = float(np.mean([" ".join(tk) == tx for tk, tx in zip(df.tokens, df.text)]))
print("share where ' '.join(tokens)==text:", round(join_ok, 3))

# group by text so duplicate sentences can never straddle splits
groups = df.text.map(lambda t: int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).values
tr_i, va_i, te_i = random_split(len(df), (0.70, 0.15, 0.15), C.SEED, groups=groups)
assert not (set(groups[tr_i]) & set(groups[te_i])) and not (set(groups[va_i]) & set(groups[te_i]))
test_sha = hashlib.sha256("|".join(sorted(df.sentence_id.iloc[te_i])).encode()).hexdigest()
save_json({"train": df.sentence_id.iloc[tr_i].tolist(), "val": df.sentence_id.iloc[va_i].tolist(),
           "test": df.sentence_id.iloc[te_i].tolist(), "test_sha256": test_sha}, C.ART_DIR / "ner_splits.json")
print({"train": len(tr_i), "val": len(va_i), "test": len(te_i)}, "test sha", test_sha[:12])
tr, va = df.iloc[tr_i], df.iloc[va_i]

tok = AutoTokenizer.from_pretrained(a.ckpt)
model = AutoModelForTokenClassification.from_pretrained(a.ckpt, num_labels=len(labels), id2label=dict(enumerate(labels)),
                                                        label2id=l2i)


def encode(words_list, tags_list=None):
    words_list = [[prep_token(w) for w in ws] for ws in words_list]
    enc = tok(words_list, is_split_into_words=True, truncation=True, max_length=C.MAX_LEN_CLS)
    items = []
    for i in range(len(words_list)):
        wid = enc.word_ids(i)
        item = {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "word_ids": wid}
        if tags_list is not None:
            prev, lab = None, []
            for w in wid:
                lab.append(-100 if (w is None or w == prev) else l2i[tags_list[i][w]])
                prev = w
            item["labels"] = lab
        items.append(item)
    return items


def collate(items, with_labels=True):
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), L), tok.pad_token_id)
    att = torch.zeros((len(items), L), dtype=torch.long)
    lab = torch.full((len(items), L), -100)
    for j, x in enumerate(items):
        n = len(x["input_ids"])
        ids[j, :n] = torch.tensor(x["input_ids"])
        att[j, :n] = 1
        if with_labels:
            lab[j, :n] = torch.tensor(x["labels"])
    b = {"input_ids": ids, "attention_mask": att}
    if with_labels:
        b["labels"] = lab
    return b


@torch.no_grad()
def predict_tags(m, words_list, bs=64):
    dev = next(m.parameters()).device
    items = encode(words_list)
    out = []
    m.eval()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = collate(chunk, False)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = m(**{k: v.to(dev) for k, v in b.items()}).logits
        pr = lg.argmax(-1).cpu().numpy()
        for j, x in enumerate(chunk):
            tags, prev = ["O"] * len(words_list[i + j]), None
            for k, w in enumerate(x["word_ids"]):
                if w is not None and w != prev:
                    tags[w] = labels[pr[j, k]]
                prev = w
            out.append(tags)
    return out


aug_tok, aug_tag = [], []
if a.clitic_aug > 0:
    arng = np.random.RandomState(C.SEED + 7)
    for toks, tags, lang in zip(tr.tokens, tr.tags, tr.language):
        if lang != "ar":
            continue
        new, changed = list(toks), False
        for i, t in enumerate(tags):
            if t.startswith("B-") and arng.rand() < a.clitic_aug:
                new[i] = CLITICS_AR[arng.randint(len(CLITICS_AR))] + toks[i]
                changed = True
        if changed:
            aug_tok.append(new)
            aug_tag.append(tags)
    print("clitic-augmented extra training sentences:", len(aug_tok))
tr_items = encode(tr.tokens.tolist() + aug_tok, tr.tags.tolist() + aug_tag)
rng = np.random.RandomState(C.SEED)


def batches(ep):
    order = np.random.RandomState(C.SEED + ep).permutation(len(tr_items))
    for i in range(0, len(order), a.bs):
        yield collate([tr_items[j] for j in order[i:i + a.bs]])


def eval_fn(m):
    pred = predict_tags(m, va.tokens.tolist())
    r = entity_f1(va.tags.tolist(), pred)
    return {"score": r["f1"], "val_entity_f1": r["f1"]}


info = fit(model, batches, eval_fn, (len(tr_items) + a.bs - 1) // a.bs, epochs=a.epochs, lr=a.lr)
pred = predict_tags(model, va.tokens.tolist())
rep = entity_f1(va.tags.tolist(), pred)
print("VAL entity-F1", round(rep["f1"], 4), {k: round(v["f1-score"], 3) for k, v in rep["per_type"].items()})
by_lang = {}
for lang in ("ar", "en"):
    m = (va.language == lang).values
    by_lang[lang] = round(entity_f1([t for t, k in zip(va.tags, m) if k], [p for p, k in zip(pred, m) if k])["f1"], 4)
print("by language", by_lang)

# ---------------- clitic tests (Arabic): attach clitics to the first entity word and require same type ------------
ar = va[va.language == "ar"]
clitic = {}
for c in CLITICS_AR:
    tot = ok = 0
    for toks, tags in zip(ar.tokens, ar.tags):
        for ent in ("LOCATION", "SERVICE", "PERSON"):
            idx = [i for i, t in enumerate(tags) if t == f"B-{ent}"]
            if not idx:
                continue
            i = idx[0]
            new = list(toks)
            new[i] = c + toks[i]
            p = predict_tags(model, [new])[0]
            tot += 1
            ok += int(p[i] == f"B-{ent}")
    clitic[c] = {"pass_rate": round(ok / max(tot, 1), 4), "n": tot}
print("clitic", clitic)
mdir = C.MODELS_DIR / a.name
model.save_pretrained(mdir)
tok.save_pretrained(mdir)
meta = {"name": a.name, "checkpoint": a.ckpt, "labels": labels, "preproc_version": __import__("bayan.preproc", fromlist=["x"]).PREPROC_VERSION,
        "val_entity_f1": round(rep["f1"], 4), "val_per_type": rep["per_type"], "val_by_language": by_lang,
        "clitic_val": clitic, "history": info["history"], "best_epoch": info["best_epoch"], "n_train": len(tr), "n_val": len(va),
        "tokens_join_equals_text_share": round(join_ok, 3), "clitic_aug": a.clitic_aug, "n_train_items": len(tr_items)}
save_json(meta, mdir / "bayan_meta.json")
save_json(meta, C.REPORTS_DIR / f"train_{a.name}.json")
print("STAGE3 DONE")
