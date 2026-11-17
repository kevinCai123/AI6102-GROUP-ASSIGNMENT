"""Predeclared CPU model configurations; no GPU dependency."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from features import HousingFeatures, CatFrame


CONFIGS = {
    "ridge_basic": {"kind":"ridge", "features":"basic", "alpha":10.0},
    "ridge_domain": {"kind":"ridge", "features":"domain", "alpha":10.0},
    "ridge_engineered": {"kind":"ridge", "features":"engineered", "alpha":10.0},
    "ridge_engineered_a30": {"kind":"ridge", "features":"engineered", "alpha":30.0},
    "elastic_engineered": {"kind":"elastic", "features":"engineered", "alpha":0.0005},
    "svr_engineered": {"kind":"svr", "features":"engineered", "C":20.0},
    "gbr_huber": {"kind":"gbr", "features":"engineered", "trees":1600},
    "cat_basic": {"kind":"cat", "features":"basic", "depth":5, "trees":2400, "lr":0.025},
    "cat_engineered": {"kind":"cat", "features":"engineered", "depth":5, "trees":2400, "lr":0.025},
    "cat_engineered_d4": {"kind":"cat", "features":"engineered", "depth":4, "trees":3600, "lr":0.025},
    "lgb_engineered": {"kind":"lgb", "features":"engineered", "trees":1800},
    "xgb_engineered": {"kind":"xgb", "features":"engineered", "trees":1800},
}


def make_model(name, X, seed=42, threads=8):
    c = CONFIGS[name]
    features = HousingFeatures(c["features"])
    if c["kind"] == "cat":
        # Column names follow deterministic engineering, without fitting statistics.
        shape = features.fit_transform(X)
        cats = list(shape.select_dtypes(include=["object", "string", "category"]).columns)
        model = CatBoostRegressor(iterations=c["trees"], depth=c["depth"], learning_rate=c["lr"],
            loss_function="RMSE", l2_leaf_reg=4, random_seed=seed, thread_count=threads,
            verbose=False, allow_writing_files=False, cat_features=cats, task_type="CPU")
        return Pipeline([("features", HousingFeatures(c["features"])), ("prepare", CatFrame()), ("model", model)])
    numeric = Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",RobustScaler())])
    categoric = Pipeline([("impute",SimpleImputer(strategy="constant",fill_value="Unknown",keep_empty_features=True)),
                          ("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    encode = ColumnTransformer([
        ("num",numeric,make_column_selector(dtype_include="number")),
        ("cat",categoric,make_column_selector(dtype_exclude="number"))],remainder="drop")
    kind = c["kind"]
    if kind == "ridge": model=Ridge(alpha=c["alpha"])
    elif kind == "elastic": model=ElasticNet(alpha=c["alpha"],l1_ratio=.9,max_iter=50000,tol=1e-5)
    elif kind == "svr": model=SVR(C=c["C"],epsilon=.008,gamma=.001)
    elif kind == "gbr": model=GradientBoostingRegressor(n_estimators=c["trees"],learning_rate=.025,max_depth=3,
        loss="huber",min_samples_leaf=8,max_features=.7,random_state=seed)
    elif kind == "lgb": model=LGBMRegressor(n_estimators=c["trees"],learning_rate=.015,num_leaves=8,
        max_depth=-1,min_child_samples=10,subsample=.8,subsample_freq=1,colsample_bytree=.8,
        reg_alpha=.01,reg_lambda=.1,verbosity=-1,n_jobs=threads,random_state=seed,deterministic=True,force_col_wise=True)
    elif kind == "xgb": model=XGBRegressor(n_estimators=c["trees"],learning_rate=.02,max_depth=3,
        min_child_weight=2,subsample=.75,colsample_bytree=.75,reg_alpha=.005,reg_lambda=1,
        objective="reg:squarederror",tree_method="hist",device="cpu",n_jobs=threads,random_state=seed)
    else: raise ValueError(kind)
    return Pipeline([("features",features),("prepare",encode),("model",model)])
