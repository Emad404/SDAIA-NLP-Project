"""ONNX export / INT8 quantisation / honest latency benchmarking helpers."""
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def hardware_info() -> dict:
    cpu = ""
    try:
        cpu = [l for l in open("/proc/cpuinfo").read().split("\n") if l.startswith("model name")][0].split(":")[1].strip()
    except Exception:  # noqa
        cpu = platform.processor()
    return {"cpu_model": cpu, "logical_cores": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "platform": platform.platform()}


def export_onnx(model_dir, out_path, max_len=128, opset=17):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float32).eval().cpu()
    enc = tok(["اختبار التصدير export test", "short"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask"]
    args = (enc["input_ids"], enc["attention_mask"])

    class W(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(W(model), args, str(out), input_names=names, output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return out


def quantize_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def size_mb(path) -> float:
    return round(os.path.getsize(path) / 1e6, 1)


def latency_ms(fn, inputs, warmup=30, repeat=1):
    """fn(x) -> any; inputs: list of requests. Warm-up discarded; returns dict with p50/p95/p99/mean (ms) over the real input mix."""
    for x in inputs[:warmup]:
        fn(x)
    ts = []
    for _ in range(repeat):
        for x in inputs:
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "mean": round(float(a.mean()), 2)}
