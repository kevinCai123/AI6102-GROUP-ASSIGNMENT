"""Single locked holdout evaluation of the v2 blend, then final refit and submission.

Components and weights come from selection_v2.json, fixed on development OOF only.
The v1 blend is recomputed with its locked weights for a paired comparison.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from extended_common import load, split, logrmse, write_json
from models import make_model
from models_v2 import make, make_cat

V2_FEATURES = {"drop_groups": ("interactions",)}
V2_FACTORIES = {
    "Elastic Net + Yeo-Johnson": lambda X, s: make("elastic", skew_fix=True, seed=s),
    "SVR, standard scaling": lambda X, s: make("svr", features=V2_FEATURES, scale="standard", skew_fix=True,
                                               seed=s, gamma=0.0003, C=5.0),
    "CatBoost, no interactions": lambda X, s: make_cat(X, features=V2_FEATURES, seed=s, threads=2),
    "XGBoost, no interactions": lambda X, s: make("xgb", features=V2_FEATURES, seed=s, threads=2, depth=3),
}
V1_WEIGHTS = {"elastic_engineered": 0.352647, "svr_engineered": 0.031660,
              "cat_engineered": 0.161085, "lgb_engineered": 0.454608}


def paired_ci(y, a, b, seed=6102, n=3000):
    rng = np.random.default_rng(seed)
    ea, eb = (y - a) ** 2, (y - b) ** 2
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(float(np.sqrt(ea[i].mean()) - np.sqrt(eb[i].mean())))
    return np.quantile(d, [.025, .975]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/extended"))
    ap.add_argument("--stage", choices=["holdout", "final"], required=True)
    a = ap.parse_args()
    sel = json.loads((a.out / "selection_v2.json").read_text())
    names = list(sel["weights"]); w = np.array([sel["weights"][n] for n in names])
    assert set(names) == set(V2_FACTORIES)
    train, test, X, y = load(a.data_dir)
    dev, hold = split(X, y)
    Xd, yd, Xh, yh = X.iloc[dev], y[dev], X.iloc[hold], y[hold]

    if a.stage == "holdout":
        if (a.out / "holdout_v2.json").exists():
            print("v2 holdout already evaluated; refusing repeated access."); return
        tic = time.perf_counter(); pred = {}
        for n in names:
            pred[n] = V2_FACTORIES[n](Xd, 42).fit(Xd, yd).predict(Xh)
        pred["v2 blend"] = np.column_stack([pred[n] for n in names]) @ w
        v1 = {n: make_model(n, Xd, 42, 2).fit(Xd, yd).predict(Xh) for n in ["ridge_basic", *V1_WEIGHTS]}
        pred["Ridge: basic"] = v1["ridge_basic"]
        pred["v1 blend"] = sum(V1_WEIGHTS[n] * v1[n] for n in V1_WEIGHTS)
        scores = {k: logrmse(yh, v) for k, v in pred.items()}
        result = {"n_holdout": len(hold), "scores": scores,
                  "v1_blend_minus_v2_blend_ci95": paired_ci(yh, pred["v1 blend"], pred["v2 blend"]),
                  "baseline_minus_v2_blend_ci95": paired_ci(yh, pred["Ridge: basic"], pred["v2 blend"]),
                  "seconds": time.perf_counter() - tic,
                  "note": "v2 components and weights locked on development OOF before this single v2 holdout access. "
                          "The same holdout was opened once before for v1."}
        frame = pd.DataFrame({"Id": X.Id.iloc[hold].to_numpy(), "log_price": yh, **pred})
        bands = pd.qcut(np.exp(yh), 4, labels=["Lowest quarter", "Second quarter", "Third quarter", "Highest quarter"])
        result["bands"] = {str(b): {"n": int((bands == b).sum()),
                                    **{k: logrmse(yh[bands == b], pred[k][bands == b]) for k in ("Ridge: basic", "v1 blend", "v2 blend")}}
                           for b in bands.categories}
        write_json(a.out / "holdout_v2.json", result)
        frame.to_csv(a.out / "holdout_predictions_v2.csv", index=False)
        print(json.dumps(result, indent=1))
        return

    if not (a.out / "holdout_v2.json").exists():
        raise ValueError("Evaluate the locked v2 holdout first")
    tic = time.perf_counter(); pred = {}
    for n in names:
        pred[n] = V2_FACTORIES[n](X, 42).fit(X, y).predict(test)
    blend = np.column_stack([pred[n] for n in names]) @ w
    sub = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(blend)})
    assert list(sub.columns) == ["Id", "SalePrice"] and len(sub) == 1459
    assert np.array_equal(sub.Id.to_numpy(), test.Id.to_numpy()) and np.isfinite(sub.SalePrice).all() and (sub.SalePrice > 0).all()
    path = a.out / "submission_blend_v2.csv"
    sub.to_csv(path, index=False)
    write_json(a.out / "final_fit_v2.json", {"seconds": time.perf_counter() - tic, "rows": len(sub),
               "prediction_min": float(sub.SalePrice.min()), "prediction_max": float(sub.SalePrice.max()),
               "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    print("wrote", path, sub.SalePrice.describe().round(0).to_dict())


if __name__ == "__main__":
    main()
