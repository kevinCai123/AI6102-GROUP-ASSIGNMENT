"""Create reproducible report figures and data tables from measured results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABELS={"ridge_basic":"Ridge: basic", "ridge_domain":"Ridge: domain handling", "ridge_engineered":"Ridge: engineered",
"ridge_engineered_a30":"Ridge: engineered, alpha 30", "elastic_engineered":"Elastic Net: engineered", "svr_engineered":"SVR: engineered",
"gbr_huber":"Gradient boosting: Huber", "cat_basic":"CatBoost: basic", "cat_engineered":"CatBoost: engineered",
"cat_engineered_d4":"CatBoost: depth 4", "lgb_engineered":"LightGBM: engineered", "xgb_engineered":"XGBoost: engineered"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--results",type=Path,default=Path("results"));p.add_argument("--report-dir",type=Path,default=Path("report"))
    p.add_argument("--eda-only",action="store_true",help="Generate development-only data figures without opening holdout results")
    a=p.parse_args();fig=a.report_dir/"figures";fig.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid",context="paper",font_scale=1.05)
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    train=pd.read_csv(a.data_dir/"train.csv")
    manifest=json.loads((a.results/"manifest.json").read_text()); dev=train[train.Id.isin(manifest["development_ids"])]
    fig0,ax=plt.subplots(1,2,figsize=(9.5,3.3))
    ax[0].hist(dev.SalePrice,bins=35,color="#296b8a",alpha=.9);ax[0].set(xlabel="Sale price (USD)",ylabel="Development examples",title="Original target")
    ax[0].ticklabel_format(axis="x",style="sci",scilimits=(0,0))
    ax[1].hist(np.log(dev.SalePrice),bins=35,color="#409780",alpha=.9);ax[1].set(xlabel="Natural log of sale price",ylabel="Development examples",title="Target used for training")
    fig0.tight_layout();fig0.savefig(fig/"target_distribution.pdf",bbox_inches="tight");plt.close(fig0)
    missing=dev.drop(columns=["Id","SalePrice"]).isna().mean().sort_values(ascending=False).head(12).sort_values()*100
    f,ax=plt.subplots(figsize=(8.5,3.5));missing.plot.barh(ax=ax,color="#296b8a")
    ax.set(xlabel="Missing values in development partition (%)",ylabel="",title="Missingness varies strongly across features")
    f.tight_layout();f.savefig(fig/"missingness.pdf",bbox_inches="tight");plt.close(f)
    if a.eda_only:
        print(f"Saved development-only figures to {fig}");return
    scores=pd.read_csv(a.results/"development_results.csv").sort_values("oof_log_rmse",ascending=False)
    f,ax=plt.subplots(figsize=(9.5,4.7)); ax.barh([LABELS.get(x,x) for x in scores.name],scores.oof_log_rmse,color="#296b8a")
    ax.set(xlabel="Development out-of-fold log RMSE (lower is better)",title="Controlled model and feature comparisons")
    for i,v in enumerate(scores.oof_log_rmse):ax.text(v+.001,i,f"{v:.4f}",va="center",fontsize=8)
    ax.set_xlim(0,float(scores.oof_log_rmse.max())*1.13);f.tight_layout();f.savefig(fig/"model_comparison.pdf",bbox_inches="tight");plt.close(f)
    hp=pd.read_csv(a.results/"holdout_predictions.csv");h=json.loads((a.results/"holdout_results.json").read_text())
    f,ax=plt.subplots(1,2,figsize=(9.5,3.8))
    ax[0].scatter(hp.log_price,hp.blend,s=17,alpha=.6,color="#296b8a");lims=[hp.log_price.min()-.1,hp.log_price.max()+.1]
    ax[0].plot(lims,lims,color="#8c5a43",linestyle="--");ax[0].set(xlabel="Observed log price",ylabel="Predicted log price",title="Locked holdout: selected blend")
    ax[1].scatter(np.exp(hp.log_price),hp.blend-hp.log_price,s=17,alpha=.6,color="#409780")
    ax[1].axhline(0,color="#8c5a43",linestyle="--");ax[1].set(xlabel="Observed sale price (USD)",ylabel="Prediction minus observation (log)",title="Residuals across the price range")
    ax[1].ticklabel_format(axis="x",style="sci",scilimits=(0,0));f.tight_layout();f.savefig(fig/"holdout_diagnostics.pdf",bbox_inches="tight");plt.close(f)
    hp["price_band"]=pd.qcut(hp.log_price,4,labels=["Lowest quarter","Second quarter","Third quarter","Highest quarter"])
    rows=[]
    for group,d in hp.groupby("price_band",observed=True):
        rows.append({"price_band":str(group),"n":len(d),"baseline_rmse":float(np.sqrt(np.mean((d.ridge_basic-d.log_price)**2))),"blend_rmse":float(np.sqrt(np.mean((d.blend-d.log_price)**2)))})
    pd.DataFrame(rows).to_csv(a.results/"holdout_price_bands.csv",index=False)
    predcols=[c for c in hp if c not in ["Id","log_price","price_band","blend"]]
    corr=hp[predcols].subtract(hp.log_price,axis=0).corr()
    corr.to_csv(a.results/"holdout_error_correlations.csv")
    cases=hp.assign(absolute_log_error=abs(hp.blend-hp.log_price),prediction=np.exp(hp.blend)).nlargest(5,"absolute_log_error").merge(train,on="Id")
    case_columns=["Id","SalePrice","prediction","absolute_log_error","OverallQual","GrLivArea","Neighborhood","SaleCondition"]
    cases[case_columns].to_csv(a.results/"holdout_largest_errors.csv",index=False)
    dataset={"n_train":len(train),"n_test":len(pd.read_csv(a.data_dir/"test.csv")),"n_features":len(train.columns)-2,
       "n_development":len(dev),"n_holdout":h["n_holdout"],"numeric_features":len(dev.drop(columns=["Id","SalePrice"]).select_dtypes(include="number").columns),
       "categorical_features":len(dev.drop(columns=["Id","SalePrice"]).select_dtypes(exclude="number").columns),
       "development_price_min":int(dev.SalePrice.min()),"development_price_max":int(dev.SalePrice.max()),
       "development_price_median":float(dev.SalePrice.median()),"development_missing_fraction":float(dev.drop(columns=["Id","SalePrice"]).isna().mean().mean()),
       "duplicate_feature_rows":int(train.drop(columns=["Id","SalePrice"]).duplicated().sum()),
       "main_cv_total_seconds":float(scores.seconds.sum())}
    (a.results/"dataset_summary.json").write_text(json.dumps(dataset,indent=2),encoding="utf-8")
    print(json.dumps(dataset,indent=2));print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
