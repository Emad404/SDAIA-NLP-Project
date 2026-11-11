"""Stage 6: optimisation ladder for the served topic classifier.
rungs: R0 torch fp32 eager (pad to max) -> R1 torch fp32 dynamic padding -> R2 ONNX Runtime fp32 -> R3 ORT INT8 dynamic.
Accuracy tax = VAL macro-F1 change vs R1 (fp32 reference). Benchmark: batch=1, threads pinned, warm-up discarded, real length mix."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from bayan import config as C
from bayan.data import load_feedback
from bayan.infer import SeqClassifier
from bayan.metrics import macro_f1
from bayan.onnx_utils import export_onnx, hardware_info, latency_ms, quantize_int8, size_mb
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model dir name under models/")
ap.add_argument("--task", default="topic")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--lang", default="all")
a = ap.parse_args()
torch.set_num_threads(a.threads)

mdir = C.MODELS_DIR / a.model
df = load_feedback()
va = get_part(df, load_splits(), "val")
if a.lang != "all":
    va = va[va.language == a.lang]
labels = json.load(open(mdir / "bayan_meta.json"))["labels"]
yva = np.array([labels.index(l) for l in va[a.task]])
texts = va.text.tolist()
print("model", a.model, "val n", len(texts), "hardware", hardware_info())

onnx_dir = C.ART_DIR / "onnx" / a.model
fp32, int8 = onnx_dir / "model_fp32.onnx", onnx_dir / "model_int8.onnx"
if not fp32.exists():
    export_onnx(mdir, fp32)
if not int8.exists():
    quantize_int8(fp32, int8)
print("sizes MB:", {"torch_safetensors": size_mb(next(mdir.glob("*.safetensors"))), "onnx_fp32": size_mb(fp32), "onnx_int8": size_mb(int8)})

rng = np.random.RandomState(C.SEED)
mix = [texts[i] for i in rng.permutation(len(texts))[:300]]
rungs = {}


def run_rung(name, clf, pad_max=False):
    pr = clf.prepare(texts)
    # accuracy over the whole val set (batched, for speed)
    P = np.concatenate([clf.predict_proba(pr[i:i + 64], prepared=True) for i in range(0, len(pr), 64)])
    f1 = float(macro_f1(yva, P.argmax(1)))
    prepared_mix = clf.prepare(mix)
    if pad_max:
        orig = clf.tok

        def fn(x):
            b = orig([x], truncation=True, max_length=clf.max_len, padding="max_length", return_tensors="pt")
            with torch.inference_mode():
                return clf.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    else:
        def fn(x):
            return clf.predict_proba([x], prepared=True)
    lat = latency_ms(fn, prepared_mix, warmup=30, repeat=2)
    rungs[name] = {"val_macro_f1": round(f1, 4), **lat}
    print(name, rungs[name])
    return P


ref_clf = SeqClassifier(mdir, "torch", device="cpu", threads=a.threads)
P_ref = run_rung("R0_torch_fp32_pad128", ref_clf, pad_max=True)
P_ref = run_rung("R1_torch_fp32_dynamic_pad", ref_clf)
del ref_clf
c2 = SeqClassifier(mdir, "ort", onnx_path=fp32, threads=a.threads)
P2 = run_rung("R2_ort_fp32", c2)
c3 = SeqClassifier(mdir, "ort", onnx_path=int8, threads=a.threads)
P3 = run_rung("R3_ort_int8_dynamic", c3)
ref = rungs["R1_torch_fp32_dynamic_pad"]["val_macro_f1"]
for k, v in rungs.items():
    v["accuracy_tax_vs_R1"] = round(v["val_macro_f1"] - ref, 4)
agree = {"R2_vs_R1_argmax_agreement": float((P2.argmax(1) == P_ref.argmax(1)).mean()),
         "R3_vs_R1_argmax_agreement": float((P3.argmax(1) == P_ref.argmax(1)).mean()),
         "R2_max_abs_prob_diff": float(np.abs(P2 - P_ref).max()), "R3_max_abs_prob_diff": float(np.abs(P3 - P_ref).max())}
print(agree)
out = {"model": a.model, "threads": a.threads, "hardware": hardware_info(), "val_n": len(texts),
       "sizes_mb": {"onnx_fp32": size_mb(fp32), "onnx_int8": size_mb(int8)}, "rungs": rungs, "parity": agree,
       "benchmark_notes": "batch=1, warm-up 30 requests discarded, 300 real val messages x2 passes, threads pinned, no concurrency (HTTP/16-user test is s07)"}
save_json(out, C.REPORTS_DIR / f"optimize_{a.model}.json")
print("STAGE6 DONE")
