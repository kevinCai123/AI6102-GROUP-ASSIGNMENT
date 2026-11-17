"""Select complementary model families using development OOF labels only.

Run after develop and before holdout. Preserve the original top-four rule as a
development comparison. This refinement never reads holdout targets or errors.
"""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from models import CONFIGS
from run_experiments import logrmse,write_json

FAMILIES={"regularized_linear":{"ridge","elastic"},"kernel":{"svr"},
          "native_categorical_boosting":{"cat"},"one_hot_boosting":{"gbr","lgb","xgb"}}

def optimize(names,out,y):
    pred=np.column_stack([np.load(out/"development_cv"/(name+".npz"))["oof"] for name in names])
    opt=minimize(lambda w:np.mean((y-pred@w)**2),np.ones(len(names))/len(names),method="SLSQP",
        bounds=[(0,1)]*len(names),constraints={"type":"eq","fun":lambda w:w.sum()-1},options={"ftol":1e-12,"maxiter":1000})
    if not opt.success:raise RuntimeError(opt.message)
    w=opt.x/opt.x.sum();return w,pred,logrmse(y,pred@w)

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--results",type=Path,default=Path("results"));a=p.parse_args();out=a.results
    if (out/"holdout_results.json").exists():raise ValueError("Cannot change selection after holdout evaluation")
    if (out/"blend_comparison.json").exists():raise ValueError("Diverse selection is already locked")
    manifest=json.loads((out/"manifest.json").read_text());scores=pd.read_csv(out/"development_results.csv").sort_values("oof_log_rmse")
    train=pd.read_csv(a.data_dir/"train.csv").set_index("Id")
    y=np.log(train.loc[manifest["development_ids"],"SalePrice"].to_numpy())
    if set(scores.name)!=set(CONFIGS):raise ValueError("Finish all declared models before selection")
    previous=json.loads((out/"selection.json").read_text());write_json(out/"selection_top4.json",previous)
    family_best={family:next(n for n in scores.name if CONFIGS[n]["kind"] in kinds) for family,kinds in FAMILIES.items()}
    names=list(family_best.values());weights,pred,score=optimize(names,out,y)
    keep=weights>1e-8;active=[n for n,k in zip(names,keep) if k];active_weights=weights[keep]/weights[keep].sum()
    allnames=scores.name.tolist();allw,_,allscore=optimize(allnames,out,y)
    selection={"method":"Best development performer in each of four complementary pipeline families; convex OOF blend",
      "family_candidates":family_best,"selected_models":active,"weights":active_weights.tolist(),"best_single":scores.name.iloc[0],
      "blend_development_rmse":score,"locked_utc":datetime.now(timezone.utc).isoformat(),
      "selection_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "warning":"The family-diverse selection rule was refined using development results before opening the holdout. Its optimized OOF score is optimistic, not an unbiased estimate."}
    write_json(out/"selection.json",selection)
    comparison={"top_four_score":previous["blend_development_rmse"],"diverse_family_score":score,"all_candidates_score":allscore,
      "all_candidate_weights":dict(zip(allnames,allw.tolist())),"chosen":"diverse_family",
      "rationale":"Preserve complementary regularized-linear, kernel and tree representations without fitting redundant slow CatBoost variants. All decisions use development labels only."}
    write_json(out/"blend_comparison.json",comparison)
    pd.DataFrame({"Id":manifest["development_ids"],"log_price":y,**{n:pred[:,i] for i,n in enumerate(names)},"blend":pred@weights}).to_csv(out/"development_predictions.csv",index=False)
    print(json.dumps(selection,indent=2));print(json.dumps(comparison,indent=2))

if __name__=="__main__":main()
