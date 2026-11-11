"""Leak-free splits + the frozen-test guard.

* topic/sentiment: stratified by topic, GROUPED by citizen_id (same citizen never in two splits)
* the frozen test set can only be read when BAYAN_FINAL=1 (set by scripts/s08_final.py)
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedShuffleSplit

from .config import ART_DIR, SEED


class FrozenTestLocked(RuntimeError):
    pass


def _sha(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def make_feedback_splits(df: pd.DataFrame, seed: int = SEED) -> dict:
    y, g = df["topic"].values, df["citizen_id"].values
    idx = np.arange(len(df))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pool_idx, test_idx = next(outer.split(idx, y, g))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(inner.split(pool_idx, y[pool_idx], g[pool_idx]))
    train_idx, val_idx = pool_idx[tr_rel], pool_idx[va_rel]
    ids = df["feedback_id"].values
    splits = {"train": list(ids[train_idx]), "val": list(ids[val_idx]), "test": list(ids[test_idx])}
    cit = {k: set(df.loc[df.feedback_id.isin(v), "citizen_id"]) for k, v in splits.items()}
    assert not (cit["train"] & cit["val"]) and not (cit["train"] & cit["test"]) and not (cit["val"] & cit["test"]), \
        "citizen leakage between splits!"
    splits["test_sha256"] = _sha(splits["test"])
    return splits


def save_splits(splits: dict, name: str = "feedback_splits.json"):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_DIR / name, "w") as f:
        json.dump(splits, f)


def load_splits(name: str = "feedback_splits.json") -> dict:
    with open(ART_DIR / name) as f:
        return json.load(f)


def get_part(df: pd.DataFrame, splits: dict, part: str, id_col: str = "feedback_id") -> pd.DataFrame:
    if part == "test" and os.environ.get("BAYAN_FINAL") != "1":
        raise FrozenTestLocked("The frozen test set is locked until the final report (set BAYAN_FINAL=1).")
    if part == "test":
        assert _sha(splits["test"]) == splits["test_sha256"], "frozen test ids changed!"
    return df[df[id_col].isin(set(splits[part]))].copy()


def random_split(n: int, fracs=(0.70, 0.15, 0.15), seed: int = SEED, groups=None, strat=None):
    """index split into train/val/test. groups => keep groups together; strat => stratify."""
    idx = np.arange(n)
    f_test = fracs[2]
    f_val = fracs[1] / (fracs[0] + fracs[1])
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=f_test, random_state=seed)
        pool, te = next(gss.split(idx, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=f_val, random_state=seed + 1)
        tr_rel, va_rel = next(gss2.split(pool, groups=np.asarray(groups)[pool]))
        return pool[tr_rel], pool[va_rel], te
    if strat is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=f_test, random_state=seed)
        pool, te = next(sss.split(idx, strat))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=f_val, random_state=seed + 1)
        tr_rel, va_rel = next(sss2.split(pool, np.asarray(strat)[pool]))
        return pool[tr_rel], pool[va_rel], te
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_te = int(round(n * f_test))
    n_va = int(round(n * fracs[1]))
    return perm[n_te + n_va:], perm[n_te:n_te + n_va], perm[:n_te]
