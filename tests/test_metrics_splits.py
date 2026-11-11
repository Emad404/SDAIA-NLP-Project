import numpy as np
import pandas as pd

from bayan.metrics import (bootstrap_ci, macro_f1, mrr_at_k, paired_bootstrap_diff, recall_at_k, slice_table)
from bayan.splits import FrozenTestLocked, get_part, make_feedback_splits
import os
import pytest


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(0)
    y = rng.choice(list("abcd"), 500)
    p = np.where(rng.rand(500) < 0.7, y, rng.choice(list("abcd"), 500))
    assert abs(macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-9


def test_bootstrap_ci_contains_point():
    rng = np.random.RandomState(1)
    y = rng.choice(list("abc"), 300)
    p = np.where(rng.rand(300) < 0.8, y, rng.choice(list("abc"), 300))
    pt, lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo <= pt <= hi


def test_paired_bootstrap_detects_gap():
    rng = np.random.RandomState(2)
    y = rng.choice(list("abc"), 400)
    good = np.where(rng.rand(400) < 0.95, y, "a")
    bad = np.where(rng.rand(400) < 0.60, y, "a")
    d, lo, hi = paired_bootstrap_diff(y, good, bad, n_boot=300)
    assert d > 0 and lo > 0


def test_retrieval_metrics():
    assert recall_at_k(["x", "y", "z"], "z", 3) == 1.0 and recall_at_k(["x", "y", "z"], "z", 2) == 0.0
    assert mrr_at_k(["x", "y", "z"], "y", 10) == 0.5 and mrr_at_k(["x"], "q", 10) == 0.0


def test_slice_table():
    df = pd.DataFrame({"y": list("ababab") * 20, "p": list("ababaa") * 20, "g": ["u", "v"] * 60})
    t = slice_table(df, "y", "p", ["g"], n_boot=50)
    assert set(t.columns) >= {"slice", "value", "n", "macro_f1", "ci_low", "ci_high"} and len(t) == 2


def _fake_feedback(n=2000):
    rng = np.random.RandomState(0)
    topics = ["a", "b", "c", "d"]
    return pd.DataFrame({"feedback_id": [f"F{i}" for i in range(n)], "citizen_id": [f"C{rng.randint(0, 500)}" for _ in range(n)],
                         "topic": rng.choice(topics, n)})


def test_splits_group_by_citizen_and_lock(monkeypatch):
    df = _fake_feedback()
    s = make_feedback_splits(df)
    cit = {k: set(df[df.feedback_id.isin(s[k])].citizen_id) for k in ("train", "val", "test")}
    assert not (cit["train"] & cit["test"]) and not (cit["train"] & cit["val"]) and not (cit["val"] & cit["test"])
    assert abs(len(s["test"]) / len(df) - 0.2) < 0.06
    monkeypatch.delenv("BAYAN_FINAL", raising=False)
    with pytest.raises(FrozenTestLocked):
        get_part(df, s, "test")
    monkeypatch.setenv("BAYAN_FINAL", "1")
    assert len(get_part(df, s, "test")) == len(s["test"])
