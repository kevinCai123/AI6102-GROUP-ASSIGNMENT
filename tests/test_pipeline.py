import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import pytest
from features import HousingFeatures,CatFrame
from run_experiments import validate_submission,logrmse
from models import make_model


def test_validation_values_cannot_change_training_medians():
    tr=pd.DataFrame({"LotArea":[10.,20.,np.nan],"Neighborhood":["A","B",None]})
    va=pd.DataFrame({"LotArea":[np.nan,100000.],"Neighborhood":["NeverSeen",None]})
    prep=CatFrame().fit(tr)
    assert prep.transform(va).LotArea.iloc[0]==15.
    assert prep.medians_.LotArea==15.
    assert prep.transform(va).Neighborhood.iloc[0]=="NeverSeen"


def test_submission_rejects_silent_row_permutation_and_invalid_prices():
    ids=[1461,1462]
    with pytest.raises(ValueError):validate_submission(pd.DataFrame({"Id":ids[::-1],"SalePrice":[1.,2.]}),ids)
    with pytest.raises(ValueError):validate_submission(pd.DataFrame({"Id":ids,"SalePrice":[1.,np.inf]}),ids)
    with pytest.raises(ValueError):validate_submission(pd.DataFrame({"Id":ids,"SalePrice":[1.,0.]}),ids)


def test_target_and_id_are_not_features():
    X=pd.DataFrame({"Id":[1,2],"LotArea":[10.,20.]})
    assert "Id" not in HousingFeatures().fit_transform(X)
    with pytest.raises(ValueError): HousingFeatures().fit(X.assign(SalePrice=[100.,200.]))


def test_unseen_category_and_missing_numeric_values_predict():
    X=pd.DataFrame({"Id":range(12),"LotArea":np.arange(12.)+1,"Neighborhood":["A","B"]*6})
    y=np.log(np.arange(12.)*1000+100000)
    m=make_model("ridge_basic",X).fit(X,y)
    val=pd.DataFrame({"Id":[13,14],"LotArea":[np.nan,5000.],"Neighborhood":["New",None]})
    assert np.isfinite(m.predict(val)).all()
    assert logrmse(np.log([100.,200.]),np.log([100.,200.]))==0
