"""Mean |SHAP| importance of the v2 XGBoost component, fitted on development rows only.

One-hot columns and missing-value indicators are summed back to their source variable,
so each bar is one housing attribute. Engineered variables are flagged.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from extended_common import load, split
from models_v2 import make, GROUPS

ENGINEERED = set(sum(GROUPS.values(), []))


def source_names(prepare):
    names = []
    num_pipe = prepare.named_transformers_["num"]
    num_cols = list(prepare.transformers_[0][2])
    imp = num_pipe.named_steps["impute"]
    names += num_cols
    if imp.indicator_ is not None:
        names += [num_cols[i] for i in imp.indicator_.features_]
    cat_cols = list(prepare.transformers_[1][2])
    enc = prepare.named_transformers_["cat"].named_steps["encode"]
    for c, cats in zip(cat_cols, enc.categories_):
        names += [c] * len(cats)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/extended"))
    a = ap.parse_args()
    train, test, X, y = load(a.data_dir)
    dev, _ = split(X, y)
    Xd = X.iloc[dev].reset_index(drop=True); yd = y[dev]
    m = make("xgb", features={"drop_groups": ("interactions",)}, seed=42, threads=2, depth=3).fit(Xd, yd)
    Z = m[:-1].transform(Xd)
    contrib = m.named_steps["model"].get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
    names = source_names(m.named_steps["prepare"])
    assert len(names) == contrib.shape[1], (len(names), contrib.shape)
    imp = pd.DataFrame(np.abs(contrib), columns=names).T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
    frame = imp.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    frame["engineered"] = frame.feature.isin(ENGINEERED)
    frame.to_csv(a.out / "importance_xgb_v2.csv", index=False)
    print(frame.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
