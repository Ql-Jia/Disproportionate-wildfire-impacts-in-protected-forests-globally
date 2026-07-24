# Disproportionate-wildfire-impacts-in-protected-forests-globally
This archive contains the analysis-ready data, source code, model outputs, and figure scripts for the XGBoost and SHAP analyses supporting the Figure 2 and S1-7 fire-severity results in *Disproportionate wildfire impacts in protected forests globally*.

## Archive contents

| Directory               | Purpose                                                      | Input data                 | Outcome     | Predictors                                                   | Analysis groups                                 |
| ----------------------- | ------------------------------------------------------------ | -------------------------- | ----------- | ------------------------------------------------------------ | ----------------------------------------------- |
| `01_Delta_dNBR_XGBoost` | Explains matched protected-area minus non-protected-area differences in fire severity. | `delta_dNBR_datasets.xlsx` | `dNBR_diff` | `AGB_diff`, `NDVI_diff`, `LFMC_diff`, `VPD_diff`, `Sc_diff`, `Rd_diff` | One model per biome                             |
| `02_dNBR_XGBoost`       | Explains fire severity separately within protected areas and non-protected areas. | `dNBR_datasets.xlsx`       | `dNBR`      | `AGB`, `NDVI`, `LFMC`, `VPD`, `Sc`, `Rd`                     | One model per biome and protection-status group |

The analyses use forest-biome codes 1, 2, 3, 4, 5, 6, and 12. `Match_ID` identifies a matched protected/non-protected pair. In the dNBR dataset, each retained `Match_ID` has exactly two rows: `PAs=1` and `PAs=0`.

## Installation

The documented environment is Python 3.11.14 with NumPy 2.4.6, pandas 3.0.1, SciPy 1.16.3, scikit-learn 1.7.2, XGBoost 3.2.0, SHAP 0.51.0, Matplotlib 3.10.7, pygam 0.12.0, statsmodels 0.14.6, and openpyxl 3.1.5. The scripts request CUDA; change `device` to `cpu` in the XGBoost scripts when no compatible GPU is available.

## Reproducing the analyses

1. Run `01_Delta_dNBR_XGBoost/01_XGBoost_SHAP_Delta_dNBR.py`.
2. Run `02_dNBR_XGBoost/01_XGBoost_SHAP_dNBR.py`.
3. Run the numbered plotting scripts from their respective directories.

All paths are relative to the script directory. Each training script writes `predictions.csv`, `shap_values.csv`, `evaluation_metrics.csv`, `all_biomes_metrics_summary.csv`, and `runtime_summary.csv` under `Figure_results`. Existing outputs are overwritten when a script is rerun.

## Model specification, splitting, and evaluation

Both analyses use `xgboost.XGBRegressor` with histogram tree construction, 500 trees, maximum depth 3, learning rate 0.02, row subsampling 0.7, column subsampling 0.6, regularization, and seed 42. Each biome or biome-by-protection-status subgroup uses shuffled 10-fold cross-validation (`KFold`, seed 42). Each fold is trained on nine folds and evaluated on its held-out fold; saved predictions and SHAP values are therefore out-of-fold.

Pearson correlation, p-value, and RMSE are reported. SHAP values are calculated on each held-out fold to describe model associations with the response.

## Computational resources

The documented workstation uses Windows 11 Pro 64-bit (10.0.26200), an Intel Core Ultra 7 265K CPU with 20 logical processors, 127.35 GiB RAM, and an NVIDIA GeForce RTX 4060 Ti GPU with 8,188 MiB memory (driver 591.86). The scripts use `n_jobs=8` and save runtime summaries. Energy use and carbon footprint were not measured.

## Availability and contact

The archive is intended to be made available to reviewers and deposited in a public repository after publication. No pretrained model or compiled standalone application is used. Please cite the associated manuscript when using this archive. Contact: Qinglong Jia, University of Electronic Science and Technology of China, Qinglong.Jia@outlook.com.
