# Model Card: Biome-Stratified XGBoost Fire-Severity Models

## Model details

Two XGBoost regression workflows are provided. The delta workflow predicts matched differences in fire severity (`dNBR_diff`) from matched differences in AGB, NDVI, LFMC, VPD, Sc, and Rd. The status-specific workflow predicts fire severity (`dNBR`) from AGB, NDVI, LFMC, VPD, Sc, and Rd separately for protected areas (`PAs`) and non-protected areas (`nonPAs`). Models are fitted separately by forest biome.

Both workflows use `xgboost.XGBRegressor` with histogram trees, 500 estimators, maximum depth 3, minimum child weight 5, learning rate 0.02, subsample 0.7, column subsample 0.6, `reg_alpha=0.1`, `reg_lambda=3`, `n_jobs=8`, and `random_state=42`.

## Intended use

The models quantify explanatory associations between fire severity and the included environmental and landscape variables within the study data. 

## Data and preprocessing

The archive provides analysis-ready matched burned-forest records. The delta dataset contains one row per retained `Match_ID`; the dNBR dataset contains one protected and one non-protected row for every retained `Match_ID`. Models exclude rows with missing predictors or outcome values. The supplied model variables contain no missing values. Upstream source data, matching criteria, dNBR calculation, and variable derivation are described in the manuscript Methods.

## Training and evaluation

Within each biome or biome-by-protection-status subgroup, records are randomly shuffled and divided using 10-fold `KFold` cross-validation with seed 42. Each fold model is trained on nine folds and evaluated on the held-out fold. The archive reports out-of-fold Pearson r, p-value, and RMSE, as well as held-out-fold SHAP values.

This protocol does not include a distinct hyperparameter-validation set, external independent test set, grouped `Match_ID` split, spatial split, temporal split, benchmark comparison, or feature-ablation experiment.

## Interpretability

TreeSHAP values are computed for held-out observations and saved for every subgroup. They describe the contribution of a feature to the fitted model prediction in the study data. SHAP values should be interpreted as model associations, not as causal effects or validated mechanisms.

## Computing environment

The documented environment is Python 3.11.14 with NumPy 2.4.6, pandas 3.0.1, SciPy 1.16.3, scikit-learn 1.7.2, XGBoost 3.2.0, SHAP 0.51.0, Matplotlib 3.10.7, pygam 0.12.0, statsmodels 0.14.6, and openpyxl 3.1.5. The workstation uses Windows 11 Pro 64-bit, an Intel Core Ultra 7 265K CPU, 127.35 GiB RAM, and an NVIDIA GeForce RTX 4060 Ti GPU. Runtime information is saved by the training scripts; energy and carbon estimates are not available.
