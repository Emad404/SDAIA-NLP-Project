"""Bayan HTTP service (FastAPI). One preprocessing module, PII masked before anything is stored/embedded/returned,
skew canaries asserted at start-up, fp32 rollback via BAYAN_PRECISION=fp32.

Endpoints
  GET  /health            status, versions, precision, canary report
  POST /v1/classify       topic only (the latency-critical path; micro-batched)
  POST /v1/analyze        classified + extracted + similar_cases
  POST /v1/analyze/batch  extension: many messages in one call
  POST /v1/ask            extension: extractive QA over the service documents with honest 'no answer'
"""
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from . import config as C
from .arabic import detect_language
from .canaries import assert_canaries, run_canaries
from .infer import QAModel, SeqClassifier, TokenTagger, load_meta
from .preproc import PREPROC_VERSION, preprocess

PII_TYPES = {"<PHONE>": "phone", "<NATIONAL_ID>": "national_id", "<EMAIL>": "email"}


class MicroBatcher:
    """Collects concurrent requests for up to max_wait_ms (or max_batch) and runs ONE batched model call on a single worker
    thread, so 16 concurrent users never oversubscribe the CPU cores."""

    def __init__(self, fn, max_batch=16, max_wait_ms=2.0):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self.q = asyncio.Queue()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.task = None

    def start(self):
        self.task = asyncio.get_running_loop().create_task(self._run())

    async def submit(self, x):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((x, fut))
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            x, fut = await self.q.get()
            batch = [(x, fut)]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), timeout))
                except asyncio.TimeoutError:
                    break
            try:
                outs = await loop.run_in_executor(self.pool, self.fn, [b[0] for b in batch])
                for (_, f), o in zip(batch, outs):
                    f.set_result(o)
            except Exception as e:  # noqa
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)


def _dyn_int8(model):
    import torch
    return torch.quantization.quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


class Pipeline:
    def __init__(self, art_dir=None, precision=None):
        self.art_dir = Path(art_dir or os.environ.get("BAYAN_ARTIFACTS", C.ART_DIR))
        self.models_dir = Path(os.environ.get("BAYAN_MODELS", C.MODELS_DIR))
        self.cfg = json.load(open(self.art_dir / "serving_config.json"))
        self.precision = (precision or os.environ.get("BAYAN_PRECISION") or self.cfg["topic"].get("default_precision", "fp32")).lower()
        assert self.precision in ("fp32", "int8"), self.precision
        threads = int(os.environ.get("BAYAN_THREADS", self.cfg.get("threads", 2)))
        tcfg = self.cfg["topic"]
        self._topic = {}
        for key in ("general", "dialect_aware"):
            name = tcfg.get(key)
            if key == "general" and os.environ.get("BAYAN_TOPIC_MODEL"):
                name = os.environ["BAYAN_TOPIC_MODEL"]          # benchmark / rollback override (must have an ONNX export + canary reference)
            if not name:
                continue
            onnx = self.art_dir / "onnx" / name / f"model_{self.precision}.onnx"
            self._topic[key] = (SeqClassifier(self.models_dir / name, "ort", onnx_path=onnx, threads=threads)
                                if onnx.exists() else SeqClassifier(self.models_dir / name, "torch", device="cpu", threads=threads))
        self.route_dialects = set(tcfg.get("route_dialects", [])) if not os.environ.get("BAYAN_NO_ROUTE") else set()
        if os.environ.get("BAYAN_NO_ROUTE"):
            self._topic.pop("dialect_aware", None)
        self.dialect_clf = None
        dpath = self.art_dir / "dialect_clf.joblib"
        if dpath.exists():
            import joblib
            self.dialect_clf = joblib.load(dpath)
        self.sentiment = SeqClassifier(self.models_dir / self.cfg["sentiment"], "torch", device="cpu", threads=threads)
        self.ner = TokenTagger(self.models_dir / self.cfg["ner"], device="cpu") if self.cfg.get("ner") else None
        self.qa = QAModel(self.models_dir / self.cfg["qa"], device="cpu") if self.cfg.get("qa") else None
        self.index = self.embedder = self.reranker = None
        self.search_cfg = None
        if self.cfg.get("search"):
            from .search import Embedder, Reranker, load_index
            import pandas as pd
            self.search_cfg = json.load(open(self.art_dir / "search_config.json"))
            self.index_dir = self.art_dir / self.search_cfg["index_dir"]
            self.index, self.case_ids, self.index_manifest = load_index(self.index_dir)
            self.embedder = Embedder(self.search_cfg["embedder"], self.search_cfg["qprefix"], self.search_cfg["pprefix"], device="cpu")
            if self.search_cfg.get("use_rerank"):
                self.reranker = Reranker(self.search_cfg["reranker"], device="cpu")
            cases = pd.read_csv(self.art_dir / "cases_public.csv")
            self.case_text = dict(zip(cases.case_id.astype(str), cases.description))
            from .anchors import CaseAnchors
            assert list(cases.case_id.astype(str)) == list(self.case_ids), "cases_public.csv order != index case_ids order"
            self.anchors = CaseAnchors(cases.description.tolist()) if self.search_cfg.get("stage2") == "anchor_rescore" else None
        ap = self.cfg.get("aux_precision", {})      # per-model quantisation decisions written by scripts/s06c_quant_decisions.py
        if ap.get("ner") == "int8" and self.ner:
            self.ner.model = _dyn_int8(self.ner.model)
        if ap.get("qa") == "int8" and self.qa:
            self.qa.model = _dyn_int8(self.qa.model)
        if ap.get("embedder") == "int8" and self.embedder is not None:
            self.embedder.model = _dyn_int8(self.embedder.model)
        self.docs = None
        dd = self.art_dir / "service_docs.json"
        if dd.exists():
            self.docs = json.load(open(dd))
        self.versions = {"preproc": PREPROC_VERSION, "service": self.cfg.get("service_version", "1.0.0"),
                         "index": getattr(self, "index_manifest", {}).get("index_version")}

    # ---------- introspection used by canaries ----------
    def classifiers(self):
        d = {k: v for k, v in self._topic.items()}
        d["sentiment"] = self.sentiment
        return d

    def model_metas(self):
        m = {k: v.meta for k, v in self.classifiers().items()}
        if self.ner:
            m["ner"] = self.ner.meta
        if self.qa:
            m["qa"] = self.qa.meta
        return m

    def embedder_dim(self):
        return self.embedder.model.get_sentence_embedding_dimension()

    # ---------- inference ----------
    def language(self, text):
        return detect_language(text)

    def dialect(self, text, lang=None):
        if self.dialect_clf is None or (lang or self.language(text)) != "ar":
            return None
        return str(self.dialect_clf.predict([preprocess(text)])[0])

    def route(self, text, lang=None, want_dialect=False):
        """Returns (model_key, dialect). The dialect classifier is only run when routing needs it or the caller asks for it:
        on the latency-critical /v1/classify path with no dialect router shipped it is skipped (it is informational there)."""
        lang = lang or self.language(text)
        needs = bool(self.route_dialects) and "dialect_aware" in self._topic
        d = self.dialect(text, lang) if (needs or want_dialect) else None
        if needs and d in self.route_dialects:
            return "dialect_aware", d
        if lang == "en" and "english" in self._topic:
            return "english", d
        return "general", d

    def classify_batch(self, texts, want_dialect=False):
        langs = [self.language(t) for t in texts]
        routes = [self.route(t, l, want_dialect) for t, l in zip(texts, langs)]
        out = [None] * len(texts)
        for key in set(r[0] for r in routes):
            ix = [i for i, r in enumerate(routes) if r[0] == key]
            labels, conf = self._topic[key].predict([texts[i] for i in ix])
            for i, l, c in zip(ix, labels, conf):
                out[i] = {"topic": l, "topic_confidence": round(float(c), 4), "language": langs[i],
                          "dialect": routes[i][1], "model": key}
        return out

    def search(self, clean_text, k=5, raw_text=None):
        """Stage 1: exact dense cosine over the FAISS index (+ cases whose district/date equal the query's).
        Stage 2: anchor re-scoring (district/month/day) -- or the optional cross-encoder when search_config.use_rerank."""
        from .search import dense_all
        raw = raw_text if raw_text is not None else clean_text
        q = self.embedder.encode([clean_text], "query")
        s_all = dense_all(self.index, q)
        if self.anchors is not None:
            from .anchors import two_stage
            idx, sc = two_stage(s_all, raw, self.anchors, self.search_cfg["anchor_weights"], pool=self.search_cfg.get("pool", 300))
            ranked = [(int(i), float(s)) for i, s in zip(idx[:30], sc[:30])]
        else:
            cand = np.argsort(-s_all)[:30]
            ranked = [(int(i), float(s_all[i])) for i in cand]
        if self.reranker is not None:
            cand = [i for i, _ in ranked]
            sc = self.reranker.score(clean_text, [self.case_text[self.case_ids[i]] for i in cand])
            order = np.argsort(-sc)
            ranked = [(cand[j], float(sc[j])) for j in order]
        top = ranked[0][1]
        no_answer = top < self.search_cfg["no_answer_threshold"]
        res = [] if no_answer else [{"case_id": self.case_ids[i], "score": round(s, 4),
                                     "description": self.case_text[self.case_ids[i]]} for i, s in ranked[:k]]
        return {"no_answer": bool(no_answer), "top_score": round(top, 4), "results": res}

    def analyze(self, text, k=5):
        clean = preprocess(text)
        cls = self.classify_batch([text], want_dialect=True)[0]
        sent, sconf = self.sentiment.predict([text])
        ents = self.ner.extract(text)[1] if self.ner else []
        pii = sorted({PII_TYPES[p] for p in PII_TYPES if p in clean})
        out = {"classified": {**cls, "sentiment": sent[0], "sentiment_confidence": round(float(sconf[0]), 4)},
               "extracted": {"entities": ents, "pii_found": pii, "masked_text": clean},
               "similar_cases": self.search(clean, k, raw_text=text) if self.index is not None else None}
        return out

    def ask(self, question):
        if not (self.qa and self.docs):
            return {"answer": None, "reason": "qa_not_configured"}
        qs, cs = [question] * len(self.docs), [d["text"] for d in self.docs]
        outs = self.qa.answer(qs, cs)
        j = int(np.argmax([o["margin"] for o in outs]))
        o = outs[j]
        if not o["answer"]:
            return {"answer": None, "no_answer": True, "best_margin": round(o["margin"], 3), "threshold": self.qa.thr}
        return {"answer": o["answer"], "doc_id": self.docs[j]["doc_id"], "title": self.docs[j]["title"],
                "margin": round(o["margin"], 3), "no_answer": False}


def create_app(pipe=None):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class TextIn(BaseModel):
        text: str = Field(..., min_length=1, max_length=2000)
        top_k: int = Field(5, ge=1, le=20)

    class BatchIn(BaseModel):
        texts: list[str] = Field(..., min_length=1, max_length=64)
        top_k: int = Field(5, ge=1, le=20)

    class AskIn(BaseModel):
        question: str = Field(..., min_length=3, max_length=500)

    state = {}

    @asynccontextmanager
    async def lifespan(app):
        p = pipe or Pipeline()
        state["pipe"] = p
        state["canaries"] = assert_canaries(p)          # aborts start-up on skew
        state["batcher"] = MicroBatcher(p.classify_batch, max_batch=int(os.environ.get("BAYAN_MAX_BATCH", p.cfg.get("max_batch", 16))),
                                        max_wait_ms=float(os.environ.get("BAYAN_MAX_WAIT_MS", p.cfg.get("max_wait_ms", 2))))
        state["batcher"].start()
        for _ in range(3):                                  # warm-up so the first user never pays graph-compile cost
            p.classify_batch(["warm up", "اختبار"])
        yield

    app = FastAPI(title="Bayan", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        p = state["pipe"]
        return {"status": "ok", "precision": p.precision, "versions": p.versions,
                "canaries_ok": state["canaries"]["ok"], "n_canary_checks": len(state["canaries"]["checks"])}

    @app.get("/v1/canaries")
    def canaries():
        return state["canaries"]

    @app.post("/v1/classify")
    async def classify(body: TextIn):
        return await state["batcher"].submit(body.text)

    @app.post("/v1/analyze")
    def analyze(body: TextIn):
        return state["pipe"].analyze(body.text, body.top_k)

    @app.post("/v1/analyze/batch")
    def analyze_batch(body: BatchIn):
        p = state["pipe"]
        return {"results": [p.analyze(t, body.top_k) for t in body.texts]}

    @app.post("/v1/ask")
    def ask(body: AskIn):
        return state["pipe"].ask(body.question)

    return app


def app_factory():
    return create_app()
