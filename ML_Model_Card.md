---
tags:
  - tabular-regression
  - xgboost
  - shap
  - wildfire
  - fire-severity
  - protected-areas
  - explanatory-model

---

# Biome-Stratified XGBoost Models for Fire Severity

## Model description

This repository contains two biome-stratified XGBoost regression workflows used to examine associations between fire severity and environmental or landscape variables in matched burned forests.

| Workflow   | Target      | Features                                                     | Analysis unit                                                |
| ---------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Delta-dNBR | `dNBR_diff` | `AGB_diff`, `NDVI_diff`, `LFMC_diff`, `VPD_diff`, `Sc_diff`, `Rd_diff` | One matched protected-area/non-protected-area pair per `Match_ID` |
| dNBR       | `dNBR`      | `AGB`, `NDVI`, `LFMC`, `VPD`, `Sc`, `Rd`                     | One record within either protected areas (`PAs`) or non-protected areas (`nonPAs`) |

Models are fitted independently for biome codes 1, 2, 3, 4, 5, 6, and 12. The implementation uses `xgboost.XGBRegressor` with 500 trees, maximum depth 3, minimum child weight 5, learning rate 0.02, subsample 0.7, column subsample 0.6, `reg_alpha=0.1`, `reg_lambda=3`, `n_jobs=8`, and `random_state=42`.

## Intended use

The workflows are intended for retrospective explanatory analysis within the study population. They quantify model associations between the selected predictors and fire severity or matched fire-severity differences. They are not intended for operational wildfire forecasting, risk management decisions, or predictions beyond the spatial, temporal, and ecological scope of the supplied data.

## Training data

The archive provides analysis-ready data rather than complete upstream source-data processing. `delta_dNBR_datasets.xlsx` contains 24,428 matched records. `dNBR_datasets.xlsx` contains 48,856 records, comprising one `PAs=1` and one `PAs=0` observation for each of the same 24,428 retained `Match_ID` values. The manuscript Methods describes the upstream data sources, matching procedure, dNBR calculation, and variable derivation.

Rows with a missing model feature or target are excluded by the scripts. The supplied modeling variables contain no missing values.

## Training procedure

Within each biome, or biome-by-protection-status group for the dNBR workflow, observations are randomly shuffled and divided by `KFold(n_splits=10, shuffle=True, random_state=42)`. Each fold model is trained on nine folds and evaluated on the held-out fold. Hyperparameters are fixed; no separate validation set or hyperparameter search is used.

## Evaluation

The repository saves out-of-fold predictions and reports Pearson correlation, p-value, and RMSE at fold, subgroup, and overall levels. TreeSHAP values are calculated on held-out observations for each fold. Result tables are stored in each workflow's `Figure_results` directory.

This evaluation is internal random cross-validation only. No external independent test dataset, spatial or temporal holdout, grouped `Match_ID` split, simple baseline comparison, state-of-the-art benchmark comparison, or feature-ablation analysis is provided.

## Limitations and bias considerations

The results may be affected by residual confounding, measurement error, spatial and temporal dependence, matching-selection effects, and unequal sample sizes among biomes. Random row-level cross-validation can be optimistic if related observations occur in different folds. 

## Computing environment

The documented environment is Python 3.11.14 with NumPy 2.4.6, pandas 3.0.1, SciPy 1.16.3, scikit-learn 1.7.2, XGBoost 3.2.0, SHAP 0.51.0, Matplotlib 3.10.7, pygam 0.12.0, statsmodels 0.14.6, and openpyxl 3.1.5. The documented workstation uses Windows 11 Pro 64-bit, an Intel Core Ultra 7 265K CPU, 127.35 GiB RAM, and an NVIDIA GeForce RTX 4060 Ti GPU.

The delta-dNBR workflow processed 24,428 records in 33.22 s; the dNBR workflow processed 48,856 records in 69.84 s. Runtime will vary with hardware and software configuration. Energy use and carbon-emission estimates were not measured.
