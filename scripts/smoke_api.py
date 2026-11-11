"""End-to-end API smoke test with the shipped config (canaries run at start-up through the lifespan). Writes reports/api_smoke.json."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("BAYAN_PRECISION", "int8")
from fastapi.testclient import TestClient

from bayan import config as C
from bayan.service import create_app

out = {}
with TestClient(create_app()) as c:
    h = c.get("/health").json()
    out["health"] = h
    out["canaries_failed"] = [x for x in c.get("/v1/canaries").json()["checks"] if not x["ok"]]
    out["classify_ar"] = c.post("/v1/classify", json={"text": "الكهرباء مقطوعة في حي النخيل منذ الصباح"}).json()
    out["analyze_en"] = c.post("/v1/analyze", json={"text": "My bill is wrong again, please call me on 0551234567", "top_k": 3}).json()
    out["analyze_batch_n"] = len(c.post("/v1/analyze/batch", json={"texts": ["الموعد تأجل", "the bus never came"], "top_k": 2}).json()["results"])
    out["ask_answerable"] = c.post("/v1/ask", json={"question": "How do I reset my password?"}).json()
    out["ask_unanswerable"] = c.post("/v1/ask", json={"question": "What is the capital of Mars?"}).json()
    out["bad_request_status"] = c.post("/v1/classify", json={"text": ""}).status_code
ok = h["status"] == "ok" and h["canaries_ok"] and not out["canaries_failed"] and out["bad_request_status"] == 422
out["all_ok"] = bool(ok)
json.dump(out, open(C.REPORTS_DIR / "api_smoke.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if k in ("all_ok", "canaries_failed", "bad_request_status", "analyze_batch_n") else "...") for k, v in out.items()}))
print("pii_found", out["analyze_en"]["extracted"]["pii_found"], "masked:", out["analyze_en"]["extracted"]["masked_text"])
print("classified:", {k: out["analyze_en"]["classified"].get(k) for k in ("topic", "topic_confidence", "sentiment")})
print("similar_cases:", None if out["analyze_en"]["similar_cases"] is None else out["analyze_en"]["similar_cases"].get("no_answer"))
print("ask_answerable:", out["ask_answerable"]); print("ask_unanswerable:", out["ask_unanswerable"])
