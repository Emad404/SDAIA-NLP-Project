"""Metrics: macro-F1, bootstrap CIs (plain + paired), slice tables, entity-F1, retrieval metrics."""
import numpy as np
import pandas as pd


def _encode(y_true, y_pred):
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    m = {l: i for i, l in enumerate(labels)}
    return np.array([m[str(v)] for v in y_true]), np.array([m[str(v)] for v in y_pred]), len(labels)


def _macro_f1_int(yt, yp, k):
    cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    present = (cm.sum(1) + cm.sum(0)) > 0
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1[present].mean()) if present.any() else 0.0


def macro_f1(y_true, y_pred) -> float:
    yt, yp, k = _encode(y_true, y_pred)
    return _macro_f1_int(yt, yp, k)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true).astype(str) == np.asarray(y_pred).astype(str)))


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """returns (point, lo, hi) for macro-F1 with a percentile bootstrap CI."""
    yt, yp, k = _encode(y_true, y_pred)
    n = len(yt)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.randint(0, n, n)
        vals[b] = _macro_f1_int(yt[ix], yp[ix], k)
    return (_macro_f1_int(yt, yp, k), float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """macro-F1(A) - macro-F1(B) on the SAME resampled rows. returns (diff, lo, hi)."""
    labels = sorted(set(map(str, y_true)) | set(map(str, pred_a)) | set(map(str, pred_b)))
    m = {l: i for i, l in enumerate(labels)}
    yt = np.array([m[str(v)] for v in y_true])
    ya = np.array([m[str(v)] for v in pred_a])
    yb = np.array([m[str(v)] for v in pred_b])
    k, n = len(labels), len(yt)
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.randint(0, n, n)
        d[b] = _macro_f1_int(yt[ix], ya[ix], k) - _macro_f1_int(yt[ix], yb[ix], k)
    point = _macro_f1_int(yt, ya, k) - _macro_f1_int(yt, yb, k)
    return point, float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def length_bucket(text: str) -> str:
    n = len(str(text).split())
    return "short(<=6w)" if n <= 6 else ("medium(7-12w)" if n <= 12 else "long(>12w)")


def slice_table(df: pd.DataFrame, ycol: str, pcol: str, by: list, n_boot: int = 500,
                min_n: int = 50) -> pd.DataFrame:
    rows = []
    for col in by:
        for val, g in df.groupby(col, observed=True):
            point, lo, hi = bootstrap_ci(g[ycol], g[pcol], n_boot=n_boot)
            rows.append({"slice": col, "value": val, "n": len(g), "macro_f1": round(point, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "accuracy": round(accuracy(g[ycol], g[pcol]), 4), "small_sample": len(g) < min_n})
    return pd.DataFrame(rows)


def entity_f1(y_true_tags, y_pred_tags) -> dict:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    rep = classification_report(y_true_tags, y_pred_tags, output_dict=True, zero_division=0)
    return {"f1": float(f1_score(y_true_tags, y_pred_tags)), "precision": float(precision_score(y_true_tags, y_pred_tags)),
            "recall": float(recall_score(y_true_tags, y_pred_tags)),
            "per_type": {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")}
                         for k, v in rep.items() if k in ("PERSON", "LOCATION", "SERVICE")}}


def recall_at_k(ranked, relevant, k=10) -> float:
    return float(relevant in list(ranked)[:k])


def mrr_at_k(ranked, relevant, k=10) -> float:
    for i, r in enumerate(list(ranked)[:k], start=1):
        if r == relevant:
            return 1.0 / i
    return 0.0
