"""Shared loading, split and fold logic for the extended (v2) experiments.

Reuses the exact split (seed 6102, target-decile stratified 80/20) and five
development folds (seed 42) of run_experiments.py, so every extended result
is comparable with the original development table.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


def logrmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Non-finite values in metric")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    if len(train) != 1460 or len(test) != 1459:
        raise ValueError("Unexpected official dataset dimensions")
    if "SalePrice" in test:
        raise ValueError("Test labels must never be loaded")
    X = train.drop(columns="SalePrice")
    y = np.log(train.SalePrice.to_numpy())
    return train, test, X, y


def split(X, y):
    dev, hold = train_test_split(np.arange(len(X)), test_size=.2, random_state=6102,
                                 stratify=pd.qcut(y, 10, labels=False))
    return dev, hold


def dev_folds(Xd):
    return list(KFold(n_splits=5, shuffle=True, random_state=42).split(Xd))


def outlier_mask(X, y, rule):
    """Rows to drop from a TRAINING set. Never applied to validation rows.

    none      keep everything (v1 behaviour)
    partial2  GrLivArea > 4000 and sale price < $300k (the two large, cheap
              partial sales, Ids 524 and 1299 in the official training file)
    decock    every house above 4000 sq ft, as De Cock (2011) recommends
    """
    g = pd.to_numeric(X["GrLivArea"], errors="coerce").to_numpy()
    if rule == "none":
        return np.zeros(len(X), bool)
    if rule == "partial2":
        return (g > 4000) & (np.exp(y) < 300000)
    if rule == "decock":
        return g > 4000
    raise ValueError(rule)


def cv_oof(make, Xd, yd, folds, drop_rule="none", seed=42):
    """Out-of-fold predictions; outlier rule applies to training folds only."""
    pred = np.full(len(Xd), np.nan)
    fold_scores = []
    for k, (tr, va) in enumerate(folds):
        drop = outlier_mask(Xd.iloc[tr], yd[tr], drop_rule)
        tr_kept = tr[~drop]
        m = make(Xd.iloc[tr_kept], seed + k)
        m.fit(Xd.iloc[tr_kept], yd[tr_kept])
        pred[va] = m.predict(Xd.iloc[va])
        fold_scores.append(logrmse(yd[va], pred[va]))
    assert np.isfinite(pred).all()
    return pred, fold_scores


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
