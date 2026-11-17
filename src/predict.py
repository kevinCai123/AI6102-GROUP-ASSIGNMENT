"""Recreate the final submission from saved models without retraining.

Only load joblib files from a source you trust: pickle-based model files execute
Python code while loading.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from run_experiments import validate_submission

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--test-csv",type=Path,required=True)
    p.add_argument("--results",type=Path,default=Path("results"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();test=pd.read_csv(a.test_csv)
    if "SalePrice" in test:raise ValueError("Prediction input must not contain targets")
    selection=json.loads((a.results/"selection.json").read_text())
    predictions=[]
    for name in selection["selected_models"]:
        model=joblib.load(a.results/"models"/(name+".joblib"))
        predictions.append(model.predict(test))
    price=np.exp(np.column_stack(predictions)@np.array(selection["weights"]))
    frame=pd.DataFrame({"Id":test.Id,"SalePrice":price});validate_submission(frame,test.Id)
    a.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output,index=False)
    print(f"Saved {len(frame)} predictions to {a.output}")

if __name__=="__main__":main()
