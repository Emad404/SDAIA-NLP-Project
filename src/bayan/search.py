"""Similar-case retrieval: versioned FAISS index + manifest, two-stage (bi-encoder -> cross-encoder) ranking."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .preproc import PREPROC_VERSION, preprocess


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Embedder:
    """Wraps a sentence-transformers model with the e5-style prefixes; always returns L2-normalised float32."""

    def __init__(self, ckpt, qprefix="", pprefix="", device=None, max_seq_len=128):
        from sentence_transformers import SentenceTransformer
        self.ckpt, self.qprefix, self.pprefix = ckpt, qprefix, pprefix
        self.model = SentenceTransformer(ckpt, device=device)
        self.model.max_seq_length = max_seq_len

    def encode(self, texts, kind="passage", bs=128):
        pre = self.pprefix if kind == "passage" else self.qprefix
        v = self.model.encode([pre + t for t in texts], batch_size=bs, convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
        return v.astype("float32")


def build_index(vectors: np.ndarray, case_ids, out_dir, meta: dict):
    import faiss
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "index.faiss"))
    np.save(out / "vectors_probe.npy", vectors[:64])
    json.dump(list(map(str, case_ids)), open(out / "case_ids.json", "w"))
    manifest = {"index_version": meta.get("index_version", "v1"), "n_vectors": int(vectors.shape[0]), "dim": int(vectors.shape[1]),
                "index_type": "IndexFlatIP (exact, L2-normalised => cosine)", "preproc_version": PREPROC_VERSION,
                "built_at_unix": int(time.time()),
                "index_sha256": sha256_file(out / "index.faiss"),
                "case_ids_sha256": hashlib.sha256(json.dumps(list(map(str, case_ids))).encode()).hexdigest(), **meta}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    return manifest


def load_index(index_dir):
    import faiss
    d = Path(index_dir)
    manifest = json.load(open(d / "manifest.json"))
    assert sha256_file(d / "index.faiss") == manifest["index_sha256"], "FAISS index hash mismatch (corrupt or stale index)"
    case_ids = json.load(open(d / "case_ids.json"))
    return faiss.read_index(str(d / "index.faiss")), case_ids, manifest


def dense_all(index, qvec):
    """Exact cosine of one query vector (1, d) to EVERY indexed case, in case-index order (FAISS flat search with k=ntotal)."""
    scores, idx = index.search(qvec, index.ntotal)
    out = np.empty(index.ntotal, dtype="float32")
    out[idx[0]] = scores[0]
    return out


def dense_search(index, qvec, k=100):
    scores, idx = index.search(qvec, k)
    return scores, idx


def rrf(rank_lists, k=60, top=100):
    """Reciprocal-rank fusion of several ranked id lists."""
    score = {}
    for rl in rank_lists:
        for r, i in enumerate(rl):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])][:top]


class Reranker:
    def __init__(self, ckpt, device=None, max_len=192):
        from sentence_transformers import CrossEncoder
        self.ckpt = ckpt
        self.model = CrossEncoder(ckpt, device=device, max_length=max_len)

    def score(self, query, passages, bs=64):
        return np.asarray(self.model.predict([(query, p) for p in passages], batch_size=bs, show_progress_bar=False),
                          dtype="float32")


def case_text(df_row_desc: str) -> str:
    return preprocess(df_row_desc)
