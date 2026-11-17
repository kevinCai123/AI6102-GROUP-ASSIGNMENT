"""Fold-fitted preprocessing for the Kaggle House Prices benchmark.

No target values or test-set statistics are used by these transformations.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HousingFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, mode="basic"):
        self.mode = mode

    def fit(self, X, y=None):
        if "SalePrice" in X.columns:
            raise ValueError("Target must be removed before preprocessing.")
        self.input_columns_ = [c for c in X.columns if c != "Id"]
        built = self._build(X)
        self.output_columns_ = list(built.columns)
        self.categorical_columns_ = list(built.select_dtypes(include=["object", "string", "category"]).columns)
        self.numeric_columns_ = [c for c in built if c not in self.categorical_columns_]
        return self

    def transform(self, X):
        check_is_fitted(self, "input_columns_")
        if "SalePrice" in X.columns:
            raise ValueError("Target leakage: SalePrice is present in features.")
        missing = set(self.input_columns_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        return self._build(X).reindex(columns=self.output_columns_)

    def _build(self, X):
        d = X.drop(columns=["Id"], errors="ignore").copy()
        if self.mode == "basic":
            return d
        if self.mode not in ("domain", "engineered"):
            raise ValueError(self.mode)
        # These integer codes represent categories, not continuous quantities.
        for c in ("MSSubClass", "MoSold"):
            if c in d:
                d[c] = d[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan).astype(object)
        absent = {
            "GarageCars": ["GarageType", "GarageFinish", "GarageQual", "GarageCond"],
            "TotalBsmtSF": ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"],
            "Fireplaces": ["FireplaceQu"],
            "PoolArea": ["PoolQC"],
            "MasVnrArea": ["MasVnrType"],
        }
        for number, categories in absent.items():
            if number in d:
                for c in categories:
                    if c in d:
                        d[c] = d[c].astype(object)
                        d.loc[d[number].eq(0) & d[c].isna(), c] = "Absent"
        for c in ("Alley", "Fence", "MiscFeature"):
            if c in d:
                d[c] = d[c].fillna("Absent").astype(object)
        if self.mode == "domain":
            return d
        def numeric(c):
            return pd.to_numeric(d[c], errors="coerce")
        def total(name, columns, weights=None):
            if all(c in d for c in columns):
                v = d[columns].apply(pd.to_numeric, errors="coerce")
                if weights is not None:
                    v = v * np.asarray(weights)
                d[name] = v.sum(axis=1, min_count=len(columns))
        total("TotalFloorArea", ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
        total("TotalBathrooms", ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], [1,.5,1,.5])
        total("TotalPorchArea", ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"])
        for new, a, b in [("AgeAtSale", "YrSold", "YearBuilt"), ("YearsSinceRemodel", "YrSold", "YearRemodAdd"), ("GarageAgeAtSale", "YrSold", "GarageYrBlt")]:
            if a in d and b in d:
                d[new] = (numeric(a)-numeric(b)).clip(lower=0)
        if "YearRemodAdd" in d and "YearBuilt" in d:
            d["WasRemodeled"] = (numeric("YearRemodAdd") > numeric("YearBuilt")).astype(float)
        for c in ["TotalBsmtSF", "GarageArea", "Fireplaces", "PoolArea", "2ndFlrSF"]:
            if c in d:
                d["Has_"+c] = numeric(c).gt(0).astype(float).where(d[c].notna(), np.nan)
        if "OverallQual" in d and "GrLivArea" in d:
            d["QualityLivingArea"] = numeric("OverallQual") * numeric("GrLivArea")
        if "GrLivArea" in d and "LotArea" in d:
            d["LivingLotRatio"] = numeric("GrLivArea") / numeric("LotArea").replace(0,np.nan)
        # Fixed transforms do not estimate a parameter from held-out observations.
        for c in ["LotArea", "LotFrontage", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "MasVnrArea", "TotalFloorArea", "TotalPorchArea"]:
            if c in d:
                d[c] = np.log1p(numeric(c).clip(lower=0))
        return d


class CatFrame(TransformerMixin, BaseEstimator):
    """Medians learned on training rows only; stable categorical strings."""
    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.cat_ = list(X.select_dtypes(include=["object", "string", "category"]).columns)
        self.num_ = [c for c in X if c not in self.cat_]
        self.medians_ = X[self.num_].median().fillna(0)
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        d = X.reindex(columns=self.columns_).copy()
        d[self.num_] = d[self.num_].fillna(self.medians_).astype(float)
        for c in self.cat_:
            d[c] = d[c].fillna("Unknown").astype(str).astype(object)
        return d
