# House Prices prediction (AI6102 team project)

Code, experiment records and Kaggle evidence for our AI6102 team project on Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The project combines housing-specific feature engineering, controlled cross-validation and a small ensemble of complementary regressors. Everything runs on CPU.

Team: Bu Dinglin, Cai Songlin, Wang Yuanzuo. The report was submitted to the course as a PDF; `report/figures/` holds its figures, which the scripts below regenerate.

## Result

| Model and data | Log RMSE |
|---|---|
| Final weighted ensemble, reserved 292-sale holdout | 0.11351 |
| Basic Ridge baseline, same holdout | 0.12730 |
| Final weighted ensemble, Kaggle public score (2 November 2026) | **0.11862** |

On 2 November 2026 the Kaggle account `songlin cai kevin1662` ranked **312 of 4,641** ranked teams (top 6.72%). The competition runs indefinitely with a rolling leaderboard, so the live rank can change.

The final ensemble combines Elastic Net (weight 0.353), SVR (0.032), CatBoost (0.161) and LightGBM (0.455). The weights were fitted on development out-of-fold predictions and locked before the holdout was used. An alternative ensemble that applies all ablation findings with equal weights scored 0.11392 on the holdout and 0.11865 on Kaggle, no better than the final model.

The submitted files are `results/submissions/submission_blend.csv` (final ensemble) and `results/submissions/submission_alternative.csv` (alternative ensemble). Kaggle screenshots, captured page text, exact Singapore submission times and the SHA-256 of both files are in `evidence/2026-11-02_resubmission/` (its JSON lists local paths from the machine that made the submission; the alternative file has since moved, with an unchanged hash). `evidence/2026-11-02/competition_ongoing.png` shows the competition status.

## Repository layout

- `src/features.py`: deterministic housing features and fold-fitted preprocessing.
- `src/models.py`: model configurations for the main comparison.
- `src/run_experiments.py`: development cross-validation, locked holdout evaluation and final refit.
- `src/select_diverse_blend.py`: picks the best model in each family and fits the ensemble weights, using development data only.
- `src/predict.py`: recreates the final submission from the saved models without retraining.
- `src/analyze_results.py`: dataset summaries and four figures (target, missingness, model comparison, holdout diagnostics).
- `src/diagnose_scaling.py`: SVR scaling diagnostic.
- `src/models_v2.py`, `src/extended_common.py`, `src/extended_experiments.py`: ablation and tuning experiments.
- `src/extended_blend.py`: combiner comparison.
- `src/extended_importance.py`: SHAP importance values.
- `src/extended_final.py`: holdout evaluation and refit of the alternative ensemble.
- `src/extended_figures.py`: ablation, importance and sensitivity figures.
- `tests/test_pipeline.py`: tests for leakage boundaries, unseen categories and submission integrity.
- `results/`: main-comparison results, holdout predictions, fitted final models (`models/`) and submissions (`submissions/`).
- `results/extended/`: ablation, tuning and alternative-ensemble results (see its README).
- `evidence/`: dated Kaggle screenshots and records.

## Reproduce

Use Python 3.12 and install `requirements-lock.txt` in a separate environment. Download `train.csv`, `test.csv`, `sample_submission.csv` and `data_description.txt` from the competition's Data page into `data/`. The data is not included here because Kaggle's competition rules do not allow redistributing it; `data/` is ignored by git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
./reproduce.ps1 -DataDir data -Python ./.venv/Scripts/python.exe
```

`reproduce.ps1` runs the tests and the main pipeline (development CV, family-diverse selection, holdout, final refit, figures and SVR diagnostic) into new `results_reproduced/` and `report_reproduced/` folders, leaving the recorded results untouched. It does not submit to Kaggle. The equivalent individual commands are:

```
python src/run_experiments.py --data-dir data --stage develop --workers 3 --threads 2
python src/select_diverse_blend.py --data-dir data
python src/run_experiments.py --data-dir data --stage holdout
python src/run_experiments.py --data-dir data --stage final
```

The ablation, tuning and alternative-ensemble experiments are run with:

```
python src/extended_experiments.py --data-dir data --exp EXP --rule none
    (EXP = outliers, groups, encoding, svr, sensitivity, catboost)
python src/extended_blend.py --data-dir data
python src/extended_importance.py --data-dir data
python src/extended_final.py --data-dir data --stage holdout
python src/extended_final.py --data-dir data --stage final
python src/extended_figures.py
```

Use a new output directory for a fresh run, and do not reuse a development cache after changing feature or model code. The main development run took about 23 minutes on an Intel Core Ultra 7 265K CPU under Windows; the ablation and tuning runs took about 15 minutes on a 2-core Linux machine. XGBoost differs slightly between the two platforms (OOF 0.12603 versus 0.12471); the other models agree closely.

## Evaluation boundary

The 292-sale holdout (20%, stratified by price decile) is never used for preprocessing, model selection or ensemble weights. The ensemble's development score is a selection score; the locked holdout is the independent evaluation. Final models are refitted on all 1,460 labelled sales only after the holdout evaluation. The alternative ensemble was assembled after that evaluation, so its holdout score is a check rather than an independent test.

## References and attribution

The implementation is written for this project. The strategy of housing-specific preprocessing, log-target regression and combining complementary regressors is informed by the public [Serigne notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard), released under Apache 2.0. We do not claim to reproduce its leaderboard result or exact code. Specific differences include preprocessing fitted separately within each fold, a reserved holdout, explicit feature ablations, and blend selection using development out-of-fold predictions.

Libraries used: scikit-learn, CatBoost, XGBoost, LightGBM, NumPy, pandas and SciPy. Generative AI tools assisted with code development, experiment execution and report drafting; the team reviewed the methods and results.
