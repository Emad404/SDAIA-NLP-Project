"""Inference wrappers used by evaluation, benchmarking and the HTTP service.
Every wrapper owns its preprocessing profile (read from bayan_meta.json) so training and serving cannot drift."""
import json
import os
import time
from pathlib import Path

import numpy as np

from .arabic import model_input
from .ner_utils import bio_to_spans, prep_token, words_with_spans
from .preproc import PREPROC_VERSION, preprocess


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def load_meta(model_dir):
    return json.load(open(Path(model_dir) / "bayan_meta.json"))


class SeqClassifier:
    """backend='torch' (eager) or 'ort' (ONNX Runtime; pass onnx_path of the fp32 or INT8 graph)."""

    def __init__(self, model_dir, backend="torch", onnx_path=None, threads=None, device=None):
        from transformers import AutoTokenizer
        self.dir = Path(model_dir)
        self.meta = load_meta(self.dir)
        self.labels = self.meta["labels"]
        self.profile = self.meta.get("profile", "xlmr")
        self.max_len = self.meta.get("max_len_infer", 128)
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.backend = backend
        self.onnx_path = onnx_path
        if backend == "torch":
            import torch
            from transformers import AutoModelForSequenceClassification
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).to(self.device).eval()
            if threads:
                torch.set_num_threads(threads)
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if threads:
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.sess.get_inputs()]

    def prepare(self, texts):
        return [model_input(t, self.profile) for t in texts]

    def logits(self, texts, prepared=False):
        x = list(texts) if prepared else self.prepare(texts)
        if self.backend == "torch":
            import torch
            b = self.tok(x, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                return self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float().cpu().numpy()
        b = self.tok(x, truncation=True, max_length=self.max_len, padding=True, return_tensors="np")
        feed = {k: b[k].astype("int64") for k in self.input_names if k in b}
        if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        return self.sess.run(None, feed)[0]

    def predict_proba(self, texts, prepared=False):
        return softmax(self.logits(texts, prepared))

    def predict(self, texts):
        p = self.predict_proba(texts)
        return [self.labels[i] for i in p.argmax(1)], p.max(1)


class TokenTagger:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.dir = Path(model_dir)
        self.meta = load_meta(self.dir)
        self.labels = self.meta["labels"]
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(self.dir).to(self.device).eval()

    def tag_words(self, words_list):
        import torch
        words_list = [[prep_token(w) for w in ws] for ws in words_list]
        enc = self.tok(words_list, is_split_into_words=True, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.inference_mode():
            pr = self.model(input_ids=enc["input_ids"].to(self.device),
                            attention_mask=enc["attention_mask"].to(self.device)).logits.argmax(-1).cpu().numpy()
        out = []
        for j, ws in enumerate(words_list):
            tags, prev = ["O"] * len(ws), None
            for k, w in enumerate(enc.word_ids(j)):
                if w is not None and w != prev:
                    tags[w] = self.labels[pr[j, k]]
                prev = w
            out.append(tags)
        return out

    def extract(self, text):
        """text -> list of {type, text, start, end}; offsets refer to preprocess(text)."""
        clean = preprocess(text)
        ws = words_with_spans(clean)
        if not ws:
            return clean, []
        tags = self.tag_words([[w for w, _, _ in ws]])[0]
        spans = bio_to_spans(ws, tags)
        for s in spans:
            s["text"] = clean[s["start"]:s["end"]]
        return clean, spans


class QAModel:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.dir = Path(model_dir)
        self.meta = load_meta(self.dir)
        self.thr = float(self.meta["null_margin_threshold"])
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.dir).to(self.device).eval()

    def answer(self, questions, contexts):
        from .qa_utils import span_scores
        sc = span_scores(self.model, self.tok, [preprocess(q) for q in questions], list(contexts))
        return [{"answer": r["text"] if r["margin"] > self.thr else "", "margin": r["margin"], "start": r["start"], "end": r["end"]}
                for r in sc]


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1000
