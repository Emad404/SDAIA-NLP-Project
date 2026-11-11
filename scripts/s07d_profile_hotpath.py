"""Where does a /v1/classify request spend its CPU? Times each stage of the hot path in-process (no HTTP), for several batch sizes.
Run on an idle CPU. Usage: s07d_profile_hotpath.py MODEL [precision] [threads]"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from bayan import config as C
from bayan.arabic import detect_language, model_input
from bayan.data import load_feedback
from bayan.infer import SeqClassifier
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

model = sys.argv[1]
prec = sys.argv[2] if len(sys.argv) > 2 else "int8"
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
texts = get_part(load_feedback(), load_splits(), "val").text.tolist()
rng = np.random.RandomState(C.SEED)
mix = [texts[i] for i in rng.permutation(len(texts))[:256]]
clf = SeqClassifier(C.MODELS_DIR / model, "ort", onnx_path=C.ART_DIR / "onnx" / model / f"model_{prec}.onnx", threads=threads)


def T(fn, reps=40):
    fn()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return round(float(np.median(t)), 3)


out = {"model": model, "precision": prec, "threads": threads, "per_batch_ms_median": {}}
for bs in (1, 4, 16):
    b = mix[:bs]
    prep = [model_input(x, clf.profile) for x in b]
    enc = clf.tok(prep, truncation=True, max_length=clf.max_len, padding=True, return_tensors="np")
    feed = {k: enc[k].astype("int64") for k in clf.input_names if k in enc}
    out["per_batch_ms_median"][bs] = {
        "language_detect": T(lambda: [detect_language(x) for x in b]),
        "preprocess_model_input": T(lambda: [model_input(x, clf.profile) for x in b]),
        "tokenize": T(lambda: clf.tok(prep, truncation=True, max_length=clf.max_len, padding=True, return_tensors="np")),
        "ort_run": T(lambda: clf.sess.run(None, feed)),
        "predict_total": T(lambda: clf.predict(b)),
        "padded_len": int(enc["input_ids"].shape[1])}
    r = out["per_batch_ms_median"][bs]
    r["per_request_ms_predict_total"] = round(r["predict_total"] / bs, 3)
print(json.dumps(out, indent=1))
save_json(out, C.REPORTS_DIR / f"hotpath_profile_{model}_{prec}_t{threads}.json")
