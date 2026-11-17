"""Reproducible development CV, locked holdout evaluation, and final fitting.

Run `python src/run_experiments.py --help` from the project directory.
The 20% holdout is reserved before model comparison. Do not use it to retune.
"""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"8")
import argparse, hashlib, json, time, platform, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib, psutil
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
from threadpoolctl import threadpool_limits
from models import CONFIGS, make_model


def logrmse(y,p):
    y,p=np.asarray(y),np.asarray(p)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Non-finite values in metric")
    return float(np.sqrt(np.mean((y-p)**2)))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def validate_submission(frame, expected_ids):
    if list(frame.columns)!=["Id","SalePrice"]: raise ValueError("Wrong submission columns")
    if not np.array_equal(frame.Id.to_numpy(),np.asarray(expected_ids)): raise ValueError("Submission Id/order mismatch")
    if frame.Id.duplicated().any(): raise ValueError("Duplicate IDs")
    p=frame.SalePrice.to_numpy()
    if not (np.isfinite(p).all() and (p>0).all()): raise ValueError("Predictions must be finite and positive")


def load_data(data):
    train=pd.read_csv(data/"train.csv")
    test=pd.read_csv(data/"test.csv")
    sample=pd.read_csv(data/"sample_submission.csv")
    if len(train)!=1460 or len(test)!=1459: raise ValueError("Unexpected official dataset dimensions")
    if "SalePrice" in test: raise ValueError("Test labels must never be loaded")
    if train.Id.duplicated().any() or test.Id.duplicated().any(): raise ValueError("Duplicate IDs")
    if set(train.Id)&set(test.Id): raise ValueError("Training and test IDs overlap")
    if not (train.SalePrice>0).all(): raise ValueError("Invalid target")
    if not np.array_equal(sample.Id,test.Id): raise ValueError("Sample/test IDs mismatch")
    X=train.drop(columns="SalePrice")
    if list(X.columns)!=list(test.columns): raise ValueError("Train/test schema mismatch")
    return train,test,X,np.log(train.SalePrice.to_numpy())


def cv(name,X,y,splits,output,seed,threads):
    cache=output/(name+".npz")
    if cache.exists():
        a=np.load(cache)
        meta=json.loads((output/(name+".json")).read_text())
        return a["oof"],meta
    started=time.perf_counter(); pred=np.full(len(X),np.nan); folds=[]
    for k,(tr,va) in enumerate(splits):
        tick=time.perf_counter()
        m=make_model(name,X.iloc[tr],seed+k,threads)
        m.fit(X.iloc[tr],y[tr]); pred[va]=m.predict(X.iloc[va])
        score=logrmse(y[va],pred[va])
        folds.append({"fold":k,"log_rmse":score,"seconds":time.perf_counter()-tick})
        print(f"{name} fold {k+1}/5: {score:.6f} ({folds[-1]['seconds']:.1f}s)",flush=True)
    if not np.isfinite(pred).all(): raise AssertionError("Incomplete OOF coverage")
    result={"name":name,"config":CONFIGS[name],"oof_log_rmse":logrmse(y,pred),
            "mean_fold_log_rmse":float(np.mean([f["log_rmse"] for f in folds])),
            "sd_fold_log_rmse":float(np.std([f["log_rmse"] for f in folds],ddof=1)),
            "seconds":time.perf_counter()-started,"folds":folds}
    np.savez_compressed(cache,oof=pred)
    write_json(output/(name+".json"),result)
    return pred,result


def cv_job(arguments):
    with threadpool_limits(limits=arguments[-1]):
        return cv(*arguments)[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("results"))
    p.add_argument("--stage",choices=["develop","holdout","final"],default="develop")
    p.add_argument("--models",nargs="+",default=list(CONFIGS))
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--workers",type=int,default=1,help="Independent CV processes; total training thread budget is workers times threads")
    args=p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    train,test,X,y=load_data(args.data_dir)
    hashes={n:hashlib.sha256((args.data_dir/n).read_bytes()).hexdigest() for n in ("train.csv","test.csv","sample_submission.csv")}
    dev,hold=train_test_split(np.arange(len(X)),test_size=.2,random_state=6102,
                             stratify=pd.qcut(y,10,labels=False))
    source_hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ("features.py","models.py","run_experiments.py")}
    manifest={"data_sha256":hashes,"source_sha256":source_hashes,"split_seed":6102,"cv_seed":42,"development_ids":train.Id.iloc[dev].tolist(),
        "holdout_ids":train.Id.iloc[hold].tolist(),"model_configs":CONFIGS,"python":platform.python_version(),
        "metric":"RMSE of natural-log SalePrice; lower is better", "threads":args.threads,"workers":args.workers}
    if (out/"manifest.json").exists():
        old=json.loads((out/"manifest.json").read_text())
        for key in ("data_sha256","source_sha256","development_ids","holdout_ids","model_configs"):
            if old[key]!=manifest[key]: raise ValueError(f"Manifest changed: {key}; use a new results directory")
    else: write_json(out/"manifest.json",manifest)
    Xd=X.iloc[dev].reset_index(drop=True); yd=y[dev]
    splits=list(KFold(n_splits=5,shuffle=True,random_state=42).split(Xd))
    fold_ids=np.empty(len(Xd),dtype=int)
    for k,(_,va) in enumerate(splits):fold_ids[va]=k
    pd.DataFrame({"Id":Xd.Id,"fold":fold_ids}).to_csv(out/"development_folds.csv",index=False)
    cvdir=out/"development_cv";cvdir.mkdir(exist_ok=True)
    if args.stage=="develop":
        if (out/"holdout_results.json").exists(): raise ValueError("Holdout already opened; no further development allowed in this run")
        started=time.perf_counter()
        for name in args.models:
            if name not in CONFIGS:raise ValueError(name)
        jobs=[(name,Xd,yd,splits,cvdir,42,args.threads) for name in args.models]
        if args.workers==1:
            for job in jobs:cv_job(job)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(cv_job,jobs))
        write_json(out/"development_execution.json",{"elapsed_seconds":time.perf_counter()-started,"workers":args.workers,"threads_per_worker":args.threads,"cache_reuse_possible":True})
        records=[json.loads(f.read_text()) for f in cvdir.glob("*.json")]
        scores=pd.DataFrame([{k:r[k] for k in ("name","oof_log_rmse","mean_fold_log_rmse","sd_fold_log_rmse","seconds")} for r in records]).sort_values("oof_log_rmse")
        scores.to_csv(out/"development_results.csv",index=False)
        print(scores.to_string(index=False),flush=True)
        names=scores.name.head(4).tolist()
        preds=np.column_stack([np.load(cvdir/(n+".npz"))["oof"] for n in names])
        opt=minimize(lambda w:np.mean((yd-preds@w)**2),np.ones(len(names))/len(names),
                     method="SLSQP",bounds=[(0,1)]*len(names),constraints={"type":"eq","fun":lambda w:w.sum()-1},
                     options={"ftol":1e-12,"maxiter":1000})
        if not opt.success:raise RuntimeError(opt.message)
        weights=opt.x/opt.x.sum()
        selection={"selected_models":names,"weights":weights.tolist(),"best_single":scores.name.iloc[0],
            "blend_development_rmse":logrmse(yd,preds@weights),
            "warning":"Blend weights and model selection use development OOF labels; this is a selection score, not an unbiased performance estimate.",
            "locked_utc":datetime.now(timezone.utc).isoformat()}
        write_json(out/"selection.json",selection)
        pd.DataFrame({"Id":Xd.Id,"log_price":yd,**{n:preds[:,i] for i,n in enumerate(names)},"blend":preds@weights}).to_csv(out/"development_predictions.csv",index=False)
        print("SELECTION",json.dumps(selection),flush=True)
        return
    selection=json.loads((out/"selection.json").read_text())
    names=selection["selected_models"];weights=np.array(selection["weights"])
    if args.stage=="holdout":
        if (out/"holdout_results.json").exists(): print("Holdout already evaluated; refusing repeated access."); return
        evaluate=list(dict.fromkeys(["ridge_basic"]+names))
        hp={};scores={};start=time.perf_counter()
        for name in evaluate:
            m=make_model(name,Xd,42,args.threads)
            with threadpool_limits(limits=args.threads):m.fit(Xd,yd)
            hp[name]=m.predict(X.iloc[hold]);scores[name]=logrmse(y[hold],hp[name])
            print(f"HOLDOUT {name}: {scores[name]:.6f}",flush=True)
        hp["blend"]=np.column_stack([hp[n] for n in names])@weights
        scores["blend"]=logrmse(y[hold],hp["blend"])
        rng=np.random.default_rng(6102);diff=[]
        e0=(y[hold]-hp["ridge_basic"])**2;e1=(y[hold]-hp["blend"])**2
        for _ in range(3000):
            b=rng.integers(0,len(hold),len(hold));diff.append(float(np.sqrt(e0[b].mean())-np.sqrt(e1[b].mean())))
        result={"n_holdout":len(hold),"scores":scores,"baseline_minus_blend_rmse_ci95":np.quantile(diff,[.025,.975]).tolist(),
                "seconds":time.perf_counter()-start,"evaluated_utc":datetime.now(timezone.utc).isoformat(),
                "note":"Models and blend weights locked before this evaluation. No retuning based on this holdout."}
        write_json(out/"holdout_results.json",result)
        pd.DataFrame({"Id":X.Id.iloc[hold].to_numpy(),"log_price":y[hold],**hp}).to_csv(out/"holdout_predictions.csv",index=False)
        print(json.dumps(result),flush=True);return
    if not (out/"holdout_results.json").exists():raise ValueError("Evaluate locked holdout before final fit")
    # After evaluation, all official training rows can be used for the final models.
    modeldir=out/"models";modeldir.mkdir(exist_ok=True)
    submitdir=out/"submissions";submitdir.mkdir(exist_ok=True)
    prediction={};timings={}
    for name in names:
        tic=time.perf_counter(); m=make_model(name,X,42,args.threads)
        with threadpool_limits(limits=args.threads):m.fit(X,y)
        prediction[name]=m.predict(test)
        joblib.dump(m,modeldir/(name+".joblib"),compress=3)
        frame=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(prediction[name])})
        validate_submission(frame,test.Id);frame.to_csv(submitdir/(name+".csv"),index=False)
        timings[name]=time.perf_counter()-tic
        print(f"FINAL {name}: {timings[name]:.1f}s",flush=True)
    final=np.column_stack([prediction[n] for n in names])@weights
    frame=pd.DataFrame({"Id":test.Id,"SalePrice":np.exp(final)})
    validate_submission(frame,test.Id);frame.to_csv(submitdir/"submission_blend.csv",index=False)
    write_json(out/"final_fit.json",{"seconds":timings,"rows":len(frame),"prediction_min":float(frame.SalePrice.min()),
       "prediction_max":float(frame.SalePrice.max()),"peak_working_set_mb":psutil.Process().memory_info().peak_wset/2**20 if hasattr(psutil.Process().memory_info(),"peak_wset") else None,
       "submission_sha256":hashlib.sha256((submitdir/"submission_blend.csv").read_bytes()).hexdigest()})
    print("FINAL SUBMISSION",submitdir/"submission_blend.csv",flush=True)


if __name__=="__main__":main()
