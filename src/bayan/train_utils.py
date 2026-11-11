"""Small, version-proof PyTorch training helpers (no Trainer API => works on transformers 4.x and 5.x)."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit(model, batch_fn, eval_fn, steps_per_epoch, epochs=3, lr=2e-5, warmup_ratio=0.1, weight_decay=0.01,
        patience=2, grad_clip=1.0, fp16=True, log=print):
    """batch_fn(epoch) -> iterable of dict batches (must include labels / positions).
    eval_fn(model) -> dict with key 'score' (higher is better). Keeps the best epoch (early stopping)."""
    dev = get_device()
    model.to(dev)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    total = max(1, steps_per_epoch * epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total * warmup_ratio), total)
    use_amp = fp16 and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_state, best_epoch, bad, history = -1e9, None, -1, 0, []
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in batch_fn(ep):
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            res = eval_fn(model)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1), **res}
        history.append(rec)
        log(json.dumps(rec))
        if res["score"] > best_score + 1e-6:
            best_score, best_epoch, bad = res["score"], ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("early stopping")
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_score": best_score, "best_epoch": best_epoch, "history": history}


def cls_batches(tok, texts, labels, bs, max_len, shuffle, seed):
    enc = tok(list(texts), truncation=True, max_length=max_len)
    n = len(texts)
    order = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        ix = order[i:i + bs]
        feats = [{"input_ids": enc["input_ids"][j], "attention_mask": enc["attention_mask"][j]} for j in ix]
        b = dict(tok.pad(feats, return_tensors="pt"))
        b["labels"] = torch.tensor([labels[j] for j in ix])
        yield b


@torch.no_grad()
def cls_logits(model, tok, texts, max_len, bs=128):
    dev = next(model.parameters()).device
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    model.eval()
    for i in range(0, len(texts), bs):
        ix = order[i:i + bs]
        b = tok([texts[j] for j in ix], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        b = {"input_ids": b["input_ids"].to(dev), "attention_mask": b["attention_mask"].to(dev)}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(**b).logits
        out[ix] = lg.float().cpu().numpy()
    return out


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
