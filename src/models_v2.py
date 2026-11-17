"""Configurable pipelines for the extended (v2) experiments.

HousingFeaturesV2 reproduces HousingFeatures("engineered") exactly when all
flags are left at their defaults, and lets single feature groups be removed
for the leave-one-group-out ablation. All learned statistics (medians,
scalers, Yeo-Johnson lambdas, skewed-column selection) are fitted on the
training rows of each fold only.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, RobustScaler, StandardScaler,
                                   QuantileTransformer, PowerTransformer)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from features import CatFrame

GROUPS = {
    "area_totals": ["TotalFloorArea", "TotalPorchArea"],
    "bathrooms": ["TotalBathrooms"],
    "ages": ["AgeAtSale", "YearsSinceRemodel", "GarageAgeAtSale"],
    "indicators": ["WasRemodeled", "Has_TotalBsmtSF", "Has_GarageArea", "Has_Fireplaces",
                   "Has_PoolArea", "Has_2ndFlrSF"],
    "interactions": ["QualityLivingArea", "LivingLotRatio"],
}
LOG_COLUMNS = ["LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
               "MasVnrArea", "TotalFloorArea", "TotalPorchArea"]
QUALITY = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "Absent": 0}
ORDINAL = {
    **{c: QUALITY for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "Absent": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Absent": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "Absent": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "Absent": 0},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
}


class HousingFeaturesV2(TransformerMixin, BaseEstimator):
    def __init__(self, domain=True, drop_groups=(), log=True, ordinal=False):
        self.domain = domain
        self.drop_groups = drop_groups
        self.log = log
        self.ordinal = ordinal

    def fit(self, X, y=None):
        if "SalePrice" in X.columns:
            raise ValueError("Target must be removed before preprocessing.")
        self.output_columns_ = list(self._build(X).columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "output_columns_")
        if "SalePrice" in X.columns:
            raise ValueError("Target leakage")
        return self._build(X).reindex(columns=self.output_columns_)

    def _build(self, X):
        d = X.drop(columns=["Id"], errors="ignore").copy()
        if self.domain:
            for c in ("MSSubClass", "MoSold"):
                d[c] = d[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan).astype(object)
            absent = {
                "GarageCars": ["GarageType", "GarageFinish", "GarageQual", "GarageCond"],
                "TotalBsmtSF": ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
                "Fireplaces": ["FireplaceQu"], "PoolArea": ["PoolQC"], "MasVnrArea": ["MasVnrType"],
            }
            for number, categories in absent.items():
                for c in categories:
                    d[c] = d[c].astype(object)
                    d.loc[d[number].eq(0) & d[c].isna(), c] = "Absent"
            for c in ("Alley", "Fence", "MiscFeature"):
                d[c] = d[c].fillna("Absent").astype(object)

        def numeric(c):
            return pd.to_numeric(d[c], errors="coerce")

        def total(name, columns, weights=None):
            v = d[columns].apply(pd.to_numeric, errors="coerce")
            if weights is not None:
                v = v * np.asarray(weights)
            d[name] = v.sum(axis=1, min_count=len(columns))
        total("TotalFloorArea", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
        total("TotalBathrooms", ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], [1, .5, 1, .5])
        total("TotalPorchArea", ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"])
        for new, a, b in [("AgeAtSale", "YrSold", "YearBuilt"), ("YearsSinceRemodel", "YrSold", "YearRemodAdd"),
                          ("GarageAgeAtSale", "YrSold", "GarageYrBlt")]:
            d[new] = (numeric(a) - numeric(b)).clip(lower=0)
        d["WasRemodeled"] = (numeric("YearRemodAdd") > numeric("YearBuilt")).astype(float)
        for c in ["TotalBsmtSF", "GarageArea", "Fireplaces", "PoolArea", "2ndFlrSF"]:
            d["Has_" + c] = numeric(c).gt(0).astype(float).where(d[c].notna(), np.nan)
        d["QualityLivingArea"] = numeric("OverallQual") * numeric("GrLivArea")
        d["LivingLotRatio"] = numeric("GrLivArea") / numeric("LotArea").replace(0, np.nan)
        if self.log:
            for c in LOG_COLUMNS:
                d[c] = np.log1p(numeric(c).clip(lower=0))
        for g in self.drop_groups:
            d = d.drop(columns=GROUPS[g])
        if self.ordinal:
            for c, mapping in ORDINAL.items():
                d[c] = d[c].map(mapping).astype(float)
        return d


class SkewCorrector(TransformerMixin, BaseEstimator):
    """Yeo-Johnson on numeric columns whose TRAINING-fold skewness exceeds a threshold."""
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        s = np.array([abs(skew(X[:, j])) if np.std(X[:, j]) > 0 else 0.0 for j in range(X.shape[1])])
        self.columns_ = np.where(s > self.threshold)[0]
        self.power_ = PowerTransformer(method="yeo-johnson", standardize=False).fit(X[:, self.columns_]) \
            if len(self.columns_) else None
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        if self.power_ is not None:
            X[:, self.columns_] = self.power_.transform(X[:, self.columns_])
        return X


def scaler(name):
    return {"robust": RobustScaler(), "standard": StandardScaler(),
            "quantile": QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0)}[name]


def encoder(scale="robust", skew_fix=False):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if skew_fix:
        steps.append(("skew", SkewCorrector()))
    steps.append(("scale", scaler(scale)))
    numeric = Pipeline(steps)
    categoric = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
                          ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, make_column_selector(dtype_include="number")),
                              ("cat", categoric, make_column_selector(dtype_exclude="number"))], remainder="drop")


def estimator(kind, seed=42, threads=2, **p):
    if kind == "ridge":
        return Ridge(alpha=p.get("alpha", 10.0))
    if kind == "elastic":
        return ElasticNet(alpha=p.get("alpha", 0.0005), l1_ratio=p.get("l1_ratio", .9), max_iter=50000, tol=1e-5)
    if kind == "svr":
        return SVR(C=p.get("C", 20.0), epsilon=p.get("epsilon", .008), gamma=p.get("gamma", .001))
    if kind == "lgb":
        return LGBMRegressor(n_estimators=p.get("trees", 1800), learning_rate=p.get("lr", .015),
                             num_leaves=p.get("leaves", 8), max_depth=-1, min_child_samples=10, subsample=.8,
                             subsample_freq=1, colsample_bytree=.8, reg_alpha=.01, reg_lambda=.1, verbosity=-1,
                             n_jobs=threads, random_state=seed, deterministic=True, force_col_wise=True)
    if kind == "xgb":
        return XGBRegressor(n_estimators=p.get("trees", 1800), learning_rate=p.get("lr", .02),
                            max_depth=p.get("depth", 3), min_child_weight=2, subsample=.75, colsample_bytree=.75,
                            reg_alpha=.005, reg_lambda=1, objective="reg:squarederror", tree_method="hist",
                            device="cpu", n_jobs=threads, random_state=seed)
    raise ValueError(kind)


def make(kind, features=None, scale="robust", skew_fix=False, seed=42, threads=2, **p):
    features = features or {}
    return Pipeline([("features", HousingFeaturesV2(**features)),
                     ("prepare", encoder(scale, skew_fix)),
                     ("model", estimator(kind, seed, threads, **p))])


def make_cat(Xtrain, features=None, seed=42, threads=2, depth=5, trees=2400, lr=0.025):
    features = features or {}
    shape = HousingFeaturesV2(**features).fit_transform(Xtrain)
    cats = list(shape.select_dtypes(include=["object", "string", "category"]).columns)
    model = CatBoostRegressor(iterations=trees, depth=depth, learning_rate=lr, loss_function="RMSE",
                              l2_leaf_reg=4, random_seed=seed, thread_count=threads, verbose=False,
                              allow_writing_files=False, cat_features=cats, task_type="CPU")
    return Pipeline([("features", HousingFeaturesV2(**features)), ("prepare", CatFrame()), ("model", model)])
