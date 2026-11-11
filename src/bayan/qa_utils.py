"""Extractive QA with honest nulls (SQuAD-2 style): shared by training, evaluation and the service."""
import re
import string
from collections import Counter

import numpy as np
import torch


def norm_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) | set("،؟؛"))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = norm_answer(pred).split(), norm_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def exact(pred: str, gold: str) -> float:
    return float(norm_answer(pred) == norm_answer(gold))


def encode_train(tok, questions, contexts, starts, texts, impossible, max_len=384):
    enc = tok(list(questions), list(contexts), truncation="only_second", max_length=max_len, return_offsets_mapping=True)
    sp, ep = [], []
    for i in range(len(questions)):
        offs, seq = enc["offset_mapping"][i], enc.sequence_ids(i)
        if impossible[i]:
            sp.append(0)
            ep.append(0)
            continue
        cs = starts[i][0]
        ce = cs + len(texts[i][0])
        s_tok = e_tok = 0
        for t, (o, sq) in enumerate(zip(offs, seq)):
            if sq != 1:
                continue
            if s_tok == 0 and o[1] > cs:
                s_tok = t
            if o[0] < ce:
                e_tok = t
        if s_tok == 0 or e_tok < s_tok:
            sp.append(0)
            ep.append(0)
        else:
            sp.append(s_tok)
            ep.append(e_tok)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "start_positions": sp[i], "end_positions": ep[i]} for i in range(len(questions))]
    return items


def collate(tok, items):
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), L), tok.pad_token_id)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for j, x in enumerate(items):
        n = len(x["input_ids"])
        ids[j, :n] = torch.tensor(x["input_ids"])
        att[j, :n] = 1
    b = {"input_ids": ids, "attention_mask": att}
    if "start_positions" in items[0]:
        b["start_positions"] = torch.tensor([x["start_positions"] for x in items])
        b["end_positions"] = torch.tensor([x["end_positions"] for x in items])
    return b


@torch.no_grad()
def span_scores(model, tok, questions, contexts, max_len=384, bs=32, max_answer_tokens=40):
    """Returns per-example dict(best_text, best_score, null_score, start_char, end_char). Decision is made by the caller."""
    dev = next(model.parameters()).device
    model.eval()
    res = []
    for i in range(0, len(questions), bs):
        q, c = list(questions[i:i + bs]), list(contexts[i:i + bs])
        enc = tok(q, c, truncation="only_second", max_length=max_len, padding=True, return_offsets_mapping=True,
                  return_tensors="pt")
        offs = enc.pop("offset_mapping").numpy()
        inp = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(**inp)
        sl, el = out.start_logits.float().cpu().numpy(), out.end_logits.float().cpu().numpy()
        for j in range(len(q)):
            seq = np.array([s if s is not None else -1 for s in enc.sequence_ids(j)])
            ctx = seq == 1
            null = float(sl[j, 0] + el[j, 0])
            s_l, e_l = np.where(ctx, sl[j], -1e9), np.where(ctx, el[j], -1e9)
            si, ei = np.argsort(-s_l)[:20], np.argsort(-e_l)[:20]
            best, bs_, be_ = -1e18, 0, 0
            for s in si:
                for e in ei:
                    if e < s or e - s + 1 > max_answer_tokens or not ctx[s] or not ctx[e]:
                        continue
                    sc = float(s_l[s] + e_l[e])
                    if sc > best:
                        best, bs_, be_ = sc, s, e
            a, b = int(offs[j][bs_][0]), int(offs[j][be_][1])
            res.append({"text": c[j][a:b], "start": a, "end": b, "span_score": best, "null_score": null,
                        "margin": best - null})
    return res


def decide(scores, thr):
    return [r["text"] if r["margin"] > thr else "" for r in scores]


def evaluate(preds, golds_list, impossible):
    """golds_list: list[list[str]]; impossible: list[bool]. Returns EM/F1 on answerable and null accuracy."""
    em, f1, nul = [], [], []
    for p, g, imp in zip(preds, golds_list, impossible):
        if imp:
            nul.append(float(p == ""))
        else:
            em.append(max(exact(p, x) for x in g))
            f1.append(max(f1_score(p, x) for x in g))
    out = {"answerable_em": float(np.mean(em)) if em else float("nan"), "answerable_f1": float(np.mean(f1)) if f1 else float("nan"),
           "null_acc": float(np.mean(nul)) if nul else float("nan"), "n_answerable": len(em), "n_null": len(nul)}
    out["score"] = float(np.nanmean([out["answerable_f1"], out["null_acc"]]))
    return out


def tune_threshold(scores, golds_list, impossible, grid=None):
    grid = np.arange(-8, 8.01, 0.25) if grid is None else grid
    best = (-1, 0.0)
    for t in grid:
        r = evaluate(decide(scores, t), golds_list, impossible)
        if r["score"] > best[0] + 1e-9:
            best = (r["score"], float(t))
    return best[1], best[0]
