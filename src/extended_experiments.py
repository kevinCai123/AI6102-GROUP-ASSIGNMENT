"""Extended (v2) development experiments. Development folds only; the holdout is never read here.

Usage: python src/extended_experiments.py --data-dir data --out results/extended --exp outliers
Experiments: outliers, groups, encoding, svr, sensitivity, catboost
Each experiment writes <out>/<exp>.csv and caches OOF predictions in <out>/oof/<key>.npy.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from extended_common import load, split, dev_folds, cv_oof, logrmse, outlier_mask, write_json
from models_v2 import make, make_cat, GROUPS

PARTIAL2_IDS = (524, 1299)
ENGINEERED = {}
V2_FEATURES = {"drop_groups": ("interactions",)}


def run(key, factory, Xd, yd, folds, rule, out, ids):
    cache = out / "oof" / f"{key}.npy"
    tic = time.perf_counter()
    if cache.exists():
        pred = np.load(cache)
    else:
        pred, _ = cv_oof(factory, Xd, yd, folds, rule)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, pred)
    fold_scores = [logrmse(yd[va], pred[va]) for _, va in folds]
    keep = ~np.isin(ids, PARTIAL2_IDS)
    row = {"key": key, "rule": rule, "oof_rmse": logrmse(yd, pred),
           "oof_rmse_excl_524_1299": logrmse(yd[keep], pred[keep]),
           "fold_sd": float(np.std(fold_scores, ddof=1)), "seconds": round(time.perf_counter() - tic, 1)}
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/extended"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--rule", default="partial2", help="Outlier rule chosen in the outliers experiment")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train, test, X, y = load(a.data_dir)
    dev, hold = split(X, y)
    Xd = X.iloc[dev].reset_index(drop=True); yd = y[dev]; ids = Xd.Id.to_numpy()
    folds = dev_folds(Xd)
    rows = []
    R = lambda key, fac, rule=a.rule, **extra: rows.append({**run(key, fac, Xd, yd, folds, rule, a.out, ids), **extra})

    if a.exp == "outliers":
        found = Xd.Id[outlier_mask(Xd, yd, "decock")].tolist()
        print("development rows above 4000 sq ft:", found)
        models = {
            "Ridge": lambda s: make("ridge", seed=s, alpha=10.0),
            "Elastic Net": lambda s: make("elastic", seed=s),
            "SVR (v1 setting)": lambda s: make("svr", seed=s),
            "LightGBM": lambda s: make("lgb", seed=s, threads=a.threads),
            "XGBoost": lambda s: make("xgb", seed=s, threads=a.threads),
        }
        for name, f in models.items():
            for rule in ("none", "partial2", "decock"):
                R(f"outliers__{name}__{rule}", lambda Xt, s, f=f: f(s), rule, model=name)

    elif a.exp == "groups":
        variants = {"Full engineered set": {}, "No domain handling": {"domain": False},
                    "No fixed log transforms": {"log": False}}
        variants.update({f"No {g.replace('_', ' ')}": {"drop_groups": (g,)} for g in GROUPS})
        for model, kind, extra in (("Ridge", "ridge", {"alpha": 10.0}), ("LightGBM", "lgb", {"threads": a.threads})):
            for name, feats in variants.items():
                R(f"groups__{model}__{name}", lambda Xt, s, k=kind, fe=feats, ex=extra: make(k, features=fe, seed=s, **ex),
                  model=model, variant=name)

    elif a.exp == "encoding":
        for model, kind, extra in (("Ridge", "ridge", {"alpha": 10.0}), ("Elastic Net", "elastic", {})):
            for ordinal in (False, True):
                for skew_fix in (False, True):
                    name = ("ordinal" if ordinal else "one-hot") + (" + Yeo-Johnson" if skew_fix else "")
                    R(f"encoding__{model}__{name}",
                      lambda Xt, s, k=kind, o=ordinal, sk=skew_fix, ex=extra: make(k, features={"ordinal": o}, skew_fix=sk, seed=s, **ex),
                      model=model, variant=name, ordinal=ordinal, skew_fix=skew_fix)

    elif a.exp == "svr":
        # v2 representation: interactions removed (group ablation), Yeo-Johnson on skewed numerics (encoding experiment)
        for scale in ("robust", "standard", "quantile"):
            for gamma in (0.0003, 0.001, 0.003):
                for C in (5.0, 20.0):
                    R(f"svr__{scale}__g{gamma}__C{C}",
                      lambda Xt, s, sc=scale, g=gamma, c=C: make("svr", features=V2_FEATURES, scale=sc, skew_fix=True,
                                                                  seed=s, gamma=g, C=c),
                      scaler=scale, gamma=gamma, C=C)

    elif a.exp == "sensitivity":
        for feats, label in ((ENGINEERED, "with interactions"), (V2_FEATURES, "without interactions")):
            for scale in ("robust", "standard"):
                R(f"sens__ridge_repr__{label}__{scale}", lambda Xt, s, fe=feats, sc=scale: make("ridge", features=fe, scale=sc,
                  skew_fix=True, seed=s, alpha=10.0), model="Ridge", param="representation", value=f"{label}, {scale} scaling")
        for alpha in (1.0, 3.0, 10.0, 30.0, 100.0):
            R(f"sens__ridge__a{alpha}", lambda Xt, s, al=alpha: make("ridge", features=V2_FEATURES, scale="standard",
              skew_fix=True, seed=s, alpha=al), model="Ridge", param="alpha", value=alpha)
        for alpha in (0.0003, 0.0005, 0.001, 0.002, 0.004):
            R(f"sens__elastic__a{alpha}", lambda Xt, s, al=alpha: make("elastic", features=V2_FEATURES, scale="standard",
              skew_fix=True, seed=s, alpha=al), model="Elastic Net", param="alpha", value=alpha)
        for leaves in (4, 8, 16):
            for trees in (1200, 1800, 3000):
                R(f"sens__lgb__l{leaves}__t{trees}", lambda Xt, s, l=leaves, t=trees: make("lgb", features=V2_FEATURES, seed=s,
                  threads=a.threads, leaves=l, trees=t), model="LightGBM", param="leaves,trees", value=f"{leaves},{trees}")
        for depth in (2, 3, 4):
            R(f"sens__xgb__d{depth}", lambda Xt, s, d=depth: make("xgb", features=V2_FEATURES, seed=s, threads=a.threads, depth=d),
              model="XGBoost", param="depth", value=depth)

    elif a.exp == "catboost":
        R(f"catboost__v1__{a.rule}", lambda Xt, s: make_cat(Xt, seed=s, threads=a.threads), model="CatBoost (v1 features)")
        R(f"catboost__v2__{a.rule}", lambda Xt, s: make_cat(Xt, features=V2_FEATURES, seed=s, threads=a.threads), model="CatBoost")

    else:
        raise ValueError(a.exp)
    pd.DataFrame(rows).to_csv(a.out / f"{a.exp}.csv", index=False)


if __name__ == "__main__":
    main()
