"""Blend-versus-stack comparison on development OOF predictions (no holdout access).

The combiner is cross-validated over a fixed OOF prediction matrix. This is an
exploratory meta-level diagnostic, NOT fully nested cross-validation: base
models that generated meta-training features may have used the meta-validation
fold, and component selection reused all development folds. It does not remove
all selection bias or provide an independent pipeline performance estimate.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, LinearRegression
from extended_common import load, split, dev_folds, logrmse, write_json

V1 = {"Elastic Net (v1)": "outliers__Elastic Net__none", "SVR (v1)": "outliers__SVR (v1 setting)__none",
      "CatBoost (v1)": "catboost__v1__none", "LightGBM (v1)": "outliers__LightGBM__none"}
V2 = {"Elastic Net + Yeo-Johnson": "encoding__Elastic Net__one-hot + Yeo-Johnson",
      "SVR, standard scaling": "svr__standard__g0.0003__C5.0",
      "CatBoost, no interactions": "catboost__v2__none",
      "XGBoost, no interactions": "sens__xgb__d3"}


def simplex_weights(P, y):
    k = P.shape[1]
    r = minimize(lambda w: np.mean((y - P @ w) ** 2), np.ones(k) / k, method="SLSQP",
                 bounds=[(0, 1)] * k, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                 options={"ftol": 1e-12, "maxiter": 1000})
    if not r.success:
        raise RuntimeError(r.message)
    return r.x / r.x.sum()


COMBINERS = {
    "Equal-weight average": lambda P, y: (lambda Q: Q.mean(1)),
    "Constrained weights (v1 method)": lambda P, y: (lambda w: (lambda Q: Q @ w))(simplex_weights(P, y)),
    "Linear stack, nonnegative": lambda P, y: LinearRegression(positive=True).fit(P, y).predict,
    "Ridge stack": lambda P, y: Ridge(alpha=1e-3).fit(P, y).predict,
}


def evaluate(P, yd, folds):
    rows = {}
    for name, fit in COMBINERS.items():
        meta_cv = np.full(len(yd), np.nan)
        for tr, va in folds:
            meta_cv[va] = fit(P[tr], yd[tr])(P[va])
        rows[name] = {"in_sample": logrmse(yd, fit(P, yd)(P)), "meta_cv": logrmse(yd, meta_cv)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/extended"))
    a = ap.parse_args()
    train, test, X, y = load(a.data_dir)
    dev, _ = split(X, y)
    Xd = X.iloc[dev].reset_index(drop=True); yd = y[dev]; folds = dev_folds(Xd)
    out = []
    for label, cands in (("v1 components", V1), ("v2 components", V2)):
        P = np.column_stack([np.load(a.out / "oof" / f"{k}.npy") for k in cands.values()])
        single = {n: logrmse(yd, P[:, i]) for i, n in enumerate(cands)}
        print(label, json.dumps(single))
        for name, r in evaluate(P, yd, folds).items():
            out.append({"components": label, "combiner": name, **r})
            print(f"  {name:35s} in-sample={r['in_sample']:.5f} meta_cv={r['meta_cv']:.5f}")
        if label == "v2 components":
            # Exploratory selection among combiners on fixed OOF features; this is not unbiased evaluation.
            v2rows = [r for r in out if r["components"] == label]
            chosen = min(v2rows, key=lambda r: r["meta_cv"])["combiner"]
            fitted = simplex_weights(P, yd)
            if chosen == "Equal-weight average":
                w = np.ones(P.shape[1]) / P.shape[1]
            elif chosen == "Constrained weights (v1 method)":
                w = fitted
            else:
                raise RuntimeError(f"Stacking combiner {chosen} selected; extend extended_final.py before use")
            corr = np.corrcoef((P - yd[:, None]).T)
            write_json(a.out / "selection_v2.json", {
                "components": dict(zip(cands, cands.values())), "combiner": chosen,
                "weights": dict(zip(cands, w.tolist())),
                "constrained_weights_reference": dict(zip(cands, fitted.tolist())),
                "single_oof": single, "blend_oof": logrmse(yd, P @ w),
                "residual_correlation_range": [float(corr[np.triu_indices(len(cands), 1)].min()),
                                               float(corr[np.triu_indices(len(cands), 1)].max())],
                "note": "Selected using exploratory meta-level CV on fixed development OOF predictions; not fully nested pipeline evaluation."})
            pd.DataFrame({"Id": Xd.Id, "log_price": yd, **{n: P[:, i] for i, n in enumerate(cands)},
                          "blend": P @ w}).to_csv(a.out / "development_predictions_v2.csv", index=False)
            print("weights", dict(zip(cands, np.round(w, 4))))
    pd.DataFrame(out).to_csv(a.out / "blending.csv", index=False)


if __name__ == "__main__":
    main()
