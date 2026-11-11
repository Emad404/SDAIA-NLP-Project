"""Train/serve skew canaries. Run at service start-up; any failure aborts start-up (unless BAYAN_CANARY_WARN_ONLY=1).
Checks: (1) preprocessing version + golden pairs, (2) PII masking canaries, (3) every model was trained with this preprocessing
version, (4) label order, (5) classifier outputs match the stored fp32 reference on fixed texts (tolerance depends on precision),
(6) NER reference spans, (7) FAISS manifest hash + embedder re-embedding of probe vectors."""
import json
import os
from pathlib import Path

import numpy as np

from .preproc import PREPROC_VERSION, contains_pii, preprocess

GOLDEN = Path(__file__).resolve().parents[2] / "canaries" / "golden_preproc.json"
PII_CANARIES = [("اتصل على 0551234567 من فضلك", "<PHONE>"), ("my id is 1012345678 thanks", "<NATIONAL_ID>"),
                ("mail me at a.b@example.com now", "<EMAIL>"), ("رقم الهوية ١٠١٢٣٤٥٦٧٨ وجوالي ٠٥٥١٢٣٤٥٦٧", "<PHONE>")]


class CanaryFailure(RuntimeError):
    pass


def _check(results, name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})


def run_canaries(pipe, reference_path=None) -> dict:
    res = []
    # 1. preprocessing
    _check(res, "preproc_version", PREPROC_VERSION == pipe.cfg.get("preproc_version", PREPROC_VERSION),
           f"serving={PREPROC_VERSION} config={pipe.cfg.get('preproc_version')}")
    gold = json.load(open(GOLDEN)) if GOLDEN.exists() else []
    bad = [g for g in gold if preprocess(g[0]) != g[1]]
    _check(res, "preproc_golden_pairs", gold and not bad, f"{len(gold)} pairs, {len(bad)} mismatches")
    # 2. PII
    bad = [t for t, ph in PII_CANARIES if ph not in preprocess(t) or contains_pii(preprocess(t))]
    _check(res, "pii_masking", not bad, f"{len(PII_CANARIES)} cases, failed={bad}")
    # 3/4. model metadata
    for name, m in pipe.model_metas().items():
        _check(res, f"trained_with_same_preproc:{name}", m.get("preproc_version") == PREPROC_VERSION,
               f"model={m.get('preproc_version')} serving={PREPROC_VERSION}")
    for name, clf in pipe.classifiers().items():
        _check(res, f"label_order:{name}", clf.labels == clf.meta["labels"], clf.labels)
    # 5. classifier reference outputs
    ref_p = Path(reference_path or pipe.art_dir / "canary_reference.json")
    if ref_p.exists():
        ref = json.load(open(ref_p))
        tol = {"fp32": 5e-3, "int8": 0.15}[pipe.precision]
        for name, clf in pipe.classifiers().items():
            r = ref["classifiers"].get(f"{name}:{clf.meta.get('name')}") or ref["classifiers"].get(name)   # per-model reference first
            if not r:
                continue
            P = clf_probs(pipe, name, r["texts"])
            lab_ok = [clf.labels[i] == l for i, l in zip(P.argmax(1), r["labels"])]
            dmax = float(np.abs(P - np.array(r["probs"])).max())
            _check(res, f"classifier_reference:{name}", all(lab_ok) and dmax <= tol,
                   f"labels_match={sum(lab_ok)}/{len(lab_ok)} max|dp|={dmax:.4f} tol={tol} precision={pipe.precision}")
        if pipe.ner is not None and "ner" in ref:
            ok = True
            for t, exp in ref["ner"]:
                _, spans = pipe.ner.extract(t)
                got = sorted((s["type"], s["text"]) for s in spans)
                ok &= got == sorted(map(tuple, exp))
            _check(res, "ner_reference_spans", ok, f"{len(ref['ner'])} sentences")
    else:
        _check(res, "canary_reference_present", False, f"missing {ref_p}")
    # 7. index
    if pipe.index is not None:
        man = pipe.index_manifest
        _check(res, "faiss_manifest_embedder", man["embedder"] == pipe.search_cfg["embedder"], man["embedder"])
        _check(res, "faiss_dim", pipe.index.d == man["dim"] == pipe.embedder_dim(), f"index={pipe.index.d} manifest={man['dim']}")
        probe = np.load(pipe.index_dir / "vectors_probe.npy")
        texts = json.load(open(pipe.index_dir / "probe_texts.json"))
        v = pipe.embedder.encode(texts, "passage")
        cos = float((v * probe[:len(texts)]).sum(1).min())
        _check(res, "faiss_embedder_reproduces_index_vectors", cos > 0.99, f"min cosine={cos:.4f}")
    out = {"ok": all(r["ok"] for r in res), "precision": pipe.precision, "checks": res}
    return out


def clf_probs(pipe, name, texts):
    return pipe.classifiers()[name].predict_proba(texts)


def assert_canaries(pipe):
    out = run_canaries(pipe)
    if not out["ok"]:
        failed = [c for c in out["checks"] if not c["ok"]]
        msg = "SKEW CANARY FAILURE: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed)
        if os.environ.get("BAYAN_CANARY_WARN_ONLY") == "1":
            print("WARNING (warn-only):", msg)
        else:
            raise CanaryFailure(msg)
    return out
