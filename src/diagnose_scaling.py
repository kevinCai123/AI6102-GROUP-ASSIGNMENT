"""Development-only diagnostic for the fixed SVR representation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from features import HousingFeatures
from models import make_model

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--results",type=Path,default=Path("results"));a=p.parse_args()
    manifest=json.loads((a.results/"manifest.json").read_text());tr=pd.read_csv(a.data_dir/"train.csv")
    X=tr.set_index("Id").loc[manifest["development_ids"]].drop(columns="SalePrice")
    features=HousingFeatures("engineered").fit_transform(X);num=features.select_dtypes(include="number")
    iqr=num.quantile(.75)-num.quantile(.25)
    stats=pd.DataFrame({"minimum":num.min(),"maximum":num.max(),"iqr":iqr,"missing_fraction":num.isna().mean()})
    stats=stats[(stats.iqr==0)&(stats.maximum>stats.minimum)]
    stats.to_csv(a.results/"zero_iqr_features.csv",index_label="feature")
    prep=make_model("svr_engineered",X)[:-1];Z=prep.fit_transform(X)
    rng=np.random.default_rng(6102);i=rng.integers(0,len(X),20000);j=rng.integers(0,len(X),20000);mask=i!=j;i,j=i[mask],j[mask]
    kernel=np.exp(-.001*np.square(Z[i]-Z[j]).sum(axis=1))
    result={"n_development":len(X),"n_pairs":len(i),"rbf_gamma":.001,
      "kernel_similarity_quantiles":dict(zip(["q10","q25","q50","q75","q90"],np.quantile(kernel,[.1,.25,.5,.75,.9]).tolist())),
      "fraction_kernel_below_0_01":float((kernel<.01).mean()),
      "interpretation":"Development-only representation diagnostic. Low kernel similarity and zero-IQR features suggest a scaling issue; this is not a causal ablation or evidence that SVR is generally unsuitable."}
    (a.results/"scaling_diagnostic.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))

if __name__=="__main__":main()
