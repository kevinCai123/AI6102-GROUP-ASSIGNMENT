# Ablation, tuning and alternative-ensemble results

These files back report Sections 5.3-5.9. All experiments use the same 1,168-row development split and five folds as `src/run_experiments.py`; the holdout is only read by `src/extended_final.py --stage holdout`. They were run on Linux with Python 3.13, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1 and CatBoost 1.2.10. The main-comparison scores reproduce to five decimals there except XGBoost (0.12471 on Linux versus 0.12603 on Windows).

| File | Contents |
|---|---|
| `outliers.csv` | Training-fold outlier rules (none / two partial sales / all > 4,000 sq ft) for five models (Table 3) |
| `groups.csv` | Leave-one-feature-group-out ablation for Ridge and LightGBM (Figure 4) |
| `encoding.csv` | One-hot vs ordinal coding, with and without Yeo-Johnson, for Ridge and Elastic Net (Table 4) |
| `svr.csv` | SVR scaler x gamma x C grid (Table 5) |
| `sensitivity.csv` | Hyperparameter sweeps for Ridge, Elastic Net, LightGBM and XGBoost (Figure 6) |
| `catboost.csv` | CatBoost with the engineered features, with and without interaction features |
| `blending.csv` | Combiners on fixed OOF predictions: in-sample score and meta-level cross-validation (Table 6). The column named `nested` holds the meta-level CV score; it is not fully nested evaluation of the base models |
| `importance_xgb_v2.csv` | Mean absolute SHAP per source attribute for the XGBoost model without interaction features (Figure 5) |
| `selection_v2.json` | Components and equal weights of the alternative ensemble |
| `holdout_v2.json` | Holdout scores of the alternative ensemble and its components, paired bootstrap intervals against the final ensemble, and price bands (Tables 7-8) |
| `final_fit_v2.json` | Refit metadata of the alternative ensemble |

The alternative ensemble was assembled after the final ensemble's holdout evaluation, so its holdout score is a check rather than an independent test. The file submitted to Kaggle for it is `../submissions/submission_alternative.csv`; it was regenerated on Windows from the fixed configuration and weights, which changes the holdout score by less than 0.0001.

Reproduce with the commands in the top-level README.
