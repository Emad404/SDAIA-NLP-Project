"""Stage 7e: fix the SHIPPED topic model after the latency work (decision recorded in DECISIONS.md).
General classifier = 3-layer distilled student (topic_xlmr_l3_kd); the full-size case-augmented teacher stays as the accuracy reference.
Serving runtime = 1 ONNX thread per process, no micro-batch wait (measured best on the 2-vCPU lab CPU, see BENCHMARKS.md).
Precision (int8 vs fp32 rollback) is decided afterwards by s06c_quant_decisions.py with a pre-registered rule."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bayan import config as C

p = C.ART_DIR / "serving_config.json"
cfg = json.load(open(p))
cfg["topic"]["accuracy_reference"] = cfg["topic"].get("accuracy_reference") or cfg["topic"]["general"]
cfg["topic"]["general"] = "topic_xlmr_l3_kd"
cfg["threads"] = 1
cfg["max_wait_ms"] = 0
cfg["max_batch"] = 16
cfg["serving_notes"] = ("general=3-layer KD student chosen for latency (teacher is the accuracy reference); 1 ORT thread, no batching wait; "
                        "run server and clients on separate cores for the 16-user benchmark")
json.dump(cfg, open(p, "w"), indent=2)
json.dump(cfg, open(C.REPORTS_DIR / "serving_config.json", "w"), indent=2)
print(json.dumps(cfg))
