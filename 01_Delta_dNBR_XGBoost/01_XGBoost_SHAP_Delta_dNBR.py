"""
Author: Qinglong Jia
Affiliation: University of Electronic Science and Technology of China
Status: Ph.D. candidate researching live fuel moisture content (LFMC) and wildfire prediction
Email: Qinglong.Jia@outlook.com

功能：按Biome执行10折XGBoost回归和SHAP分析，并记录各阶段运行时间。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def format_duration(seconds: float) -> str:
    """将秒数格式化为便于阅读的时分秒。"""

    total_seconds = max(0.0, float(seconds))
    hours, remainder = divmod(total_seconds, 3600.0)
    minutes, seconds_remainder = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds_remainder:05.2f}"
    return f"{int(minutes):02d}:{seconds_remainder:05.2f}"


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """计算皮尔逊相关系数、RMSE和相关性显著性。"""

    correlation, p_value = pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(correlation), rmse, float(p_value)


def create_xgboost_model(model_parameters: dict[str, Any]) -> xgb.XGBRegressor:
    """根据主函数中的参数创建单折XGBoost模型。"""

    return xgb.XGBRegressor(**model_parameters)


def save_biome_outputs(
    biome_directory: Path,
    input_data: np.ndarray,
    input_columns: list[str],
    original_indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    shap_values: np.ndarray,
    fold_results: list[dict[str, Any]],
    overall_metrics: dict[str, float],
    average_metrics: dict[str, float],
) -> float:
    """保存单个Biome的预测、SHAP和交叉验证精度，并返回保存耗时。"""

    save_start = time.perf_counter()
    biome_directory.mkdir(parents=True, exist_ok=True)

    prediction_table = pd.DataFrame(input_data, columns=input_columns, index=original_indices)
    prediction_table["y_true"] = observed
    prediction_table["y_pred"] = predicted
    prediction_table.to_csv(biome_directory / "predictions.csv", index=False)

    shap_table = pd.DataFrame(shap_values, columns=input_columns, index=original_indices)
    shap_table.to_csv(biome_directory / "shap_values.csv", index=False)

    metrics_table = pd.DataFrame(fold_results)
    metrics_table = pd.concat(
        [
            metrics_table,
            pd.DataFrame(
                [
                    {
                        "fold": "overall",
                        "r": overall_metrics["r"],
                        "rmse": overall_metrics["rmse"],
                        "p_value": overall_metrics["p_value"],
                        "n_samples_test": len(observed),
                        "training_seconds": np.nan,
                        "prediction_seconds": np.nan,
                        "shap_seconds": np.nan,
                        "fold_total_seconds": np.nan,
                    }
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "fold": "10-fold avg ± std",
                        "r": f"{average_metrics['r_mean']:.4f} ± {average_metrics['r_std']:.4f}",
                        "rmse": f"{average_metrics['rmse_mean']:.4f} ± {average_metrics['rmse_std']:.4f}",
                        "p_value": "-",
                        "n_samples_test": len(observed),
                        "training_seconds": np.nan,
                        "prediction_seconds": np.nan,
                        "shap_seconds": np.nan,
                        "fold_total_seconds": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metrics_table.to_csv(biome_directory / "evaluation_metrics.csv", index=False)
    return float(time.perf_counter() - save_start)


def analyze_biome(
    full_data: pd.DataFrame,
    biome: int,
    input_columns: list[str],
    output_column: str,
    output_root: Path,
    cross_validation_splits: int,
    cross_validation_seed: int,
    model_parameters: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """完成一个Biome的交叉验证、SHAP、精度计算和保存。"""

    biome_start = time.perf_counter()
    print(f"\n[Biome {biome}] 开始处理", flush=True)
    biome_data = full_data.loc[full_data["Biome"] == biome].copy()

    if biome_data.empty:
        elapsed = float(time.perf_counter() - biome_start)
        print(f"[Biome {biome}] 无数据，跳过；耗时 {format_duration(elapsed)}", flush=True)
        return None, {"Stage": f"Biome_{biome}", "Status": "Skipped_No_Data", "Total_Seconds": elapsed}

    relevant_columns = input_columns + [output_column]
    initial_rows = len(biome_data)
    biome_data = biome_data.dropna(subset=relevant_columns)
    removed_rows = initial_rows - len(biome_data)
    if removed_rows > 0:
        print(f"[Biome {biome}] 删除缺失值记录：{removed_rows}", flush=True)
    if biome_data.empty:
        elapsed = float(time.perf_counter() - biome_start)
        print(f"[Biome {biome}] 删除缺失值后无数据，跳过", flush=True)
        return None, {"Stage": f"Biome_{biome}", "Status": "Skipped_After_DropNA", "Total_Seconds": elapsed}
    if len(biome_data) < cross_validation_splits:
        raise ValueError(
            f"Biome {biome} 只有 {len(biome_data)} 条有效记录，"
            f"不足以执行 {cross_validation_splits} 折交叉验证。"
        )

    input_data = biome_data[input_columns].to_numpy(float)
    observed = biome_data[output_column].to_numpy(float)
    original_indices = biome_data.index.to_numpy()
    predicted = np.zeros_like(observed, dtype=float)
    all_shap_values = np.zeros((len(observed), len(input_columns)), dtype=float)

    kfold = KFold(
        n_splits=cross_validation_splits,
        shuffle=True,
        random_state=cross_validation_seed,
    )
    fold_results: list[dict[str, Any]] = []
    total_training_seconds = 0.0
    total_prediction_seconds = 0.0
    total_shap_seconds = 0.0

    for fold_number, (train_indices, test_indices) in enumerate(kfold.split(input_data), start=1):
        fold_start = time.perf_counter()
        train_data = input_data[train_indices]
        test_data = input_data[test_indices]
        train_target = observed[train_indices]
        test_target = observed[test_indices]

        model = create_xgboost_model(model_parameters)
        training_start = time.perf_counter()
        model.fit(train_data, train_target)
        training_seconds = float(time.perf_counter() - training_start)

        prediction_start = time.perf_counter()
        fold_prediction = model.predict(test_data)
        prediction_seconds = float(time.perf_counter() - prediction_start)
        predicted[test_indices] = fold_prediction

        shap_start = time.perf_counter()
        explainer = shap.TreeExplainer(model)
        fold_shap_values = np.asarray(explainer.shap_values(test_data), dtype=float)
        shap_seconds = float(time.perf_counter() - shap_start)
        all_shap_values[test_indices, :] = fold_shap_values

        correlation, rmse, p_value = evaluate_model(test_target, fold_prediction)
        fold_total_seconds = float(time.perf_counter() - fold_start)
        total_training_seconds += training_seconds
        total_prediction_seconds += prediction_seconds
        total_shap_seconds += shap_seconds
        fold_results.append(
            {
                "fold": fold_number,
                "r": correlation,
                "rmse": rmse,
                "p_value": p_value,
                "n_samples_test": len(test_target),
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                "shap_seconds": shap_seconds,
                "fold_total_seconds": fold_total_seconds,
            }
        )
        print(
            f"[Biome {biome}] Fold {fold_number:02d}/{cross_validation_splits}: "
            f"训练 {format_duration(training_seconds)}，"
            f"预测 {format_duration(prediction_seconds)}，"
            f"SHAP {format_duration(shap_seconds)}，"
            f"本折 {format_duration(fold_total_seconds)}",
            flush=True,
        )

    overall_r, overall_rmse, overall_p_value = evaluate_model(observed, predicted)
    fold_correlations = np.asarray([result["r"] for result in fold_results], dtype=float)
    fold_rmse_values = np.asarray([result["rmse"] for result in fold_results], dtype=float)
    average_metrics = {
        "r_mean": float(np.mean(fold_correlations)),
        "r_std": float(np.std(fold_correlations)),
        "rmse_mean": float(np.mean(fold_rmse_values)),
        "rmse_std": float(np.std(fold_rmse_values)),
    }
    overall_metrics = {
        "r": overall_r,
        "rmse": overall_rmse,
        "p_value": overall_p_value,
    }

    save_seconds = save_biome_outputs(
        biome_directory=output_root / f"Biome_{biome}",
        input_data=input_data,
        input_columns=input_columns,
        original_indices=original_indices,
        observed=observed,
        predicted=predicted,
        shap_values=all_shap_values,
        fold_results=fold_results,
        overall_metrics=overall_metrics,
        average_metrics=average_metrics,
    )
    biome_total_seconds = float(time.perf_counter() - biome_start)
    summary_entry = {
        "Biome": biome,
        "n_samples": len(observed),
        "Pearson_r_mean": average_metrics["r_mean"],
        "Pearson_r_std": average_metrics["r_std"],
        "RMSE_mean": average_metrics["rmse_mean"],
        "RMSE_std": average_metrics["rmse_std"],
        "Pearson_r_overall": overall_r,
        "RMSE_overall": overall_rmse,
        "p_value_overall": overall_p_value,
        "Processing_Seconds": biome_total_seconds,
    }
    runtime_entry = {
        "Stage": f"Biome_{biome}",
        "Status": "Completed",
        "Samples": len(observed),
        "Training_Seconds": total_training_seconds,
        "Prediction_Seconds": total_prediction_seconds,
        "SHAP_Seconds": total_shap_seconds,
        "Saving_Seconds": save_seconds,
        "Total_Seconds": biome_total_seconds,
        "Total_Formatted": format_duration(biome_total_seconds),
    }
    print(
        f"[Biome {biome}] 完成：样本 {len(observed)}，"
        f"总耗时 {format_duration(biome_total_seconds)}",
        flush=True,
    )
    return summary_entry, runtime_entry


def run_analysis(
    input_excel: Path,
    output_root: Path,
    biomes: list[int],
    input_columns: list[str],
    output_column: str,
    cross_validation_splits: int,
    cross_validation_seed: int,
    model_parameters: dict[str, Any],
) -> float:
    """执行全部Biome分析，保存汇总结果并返回总耗时。"""

    total_start = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)

    loading_start = time.perf_counter()
    full_data = pd.read_excel(input_excel)
    loading_seconds = float(time.perf_counter() - loading_start)
    print(
        f"数据读取完成：{len(full_data):,} 行，耗时 {format_duration(loading_seconds)}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = [
        {
            "Stage": "Data_Loading",
            "Status": "Completed",
            "Samples": len(full_data),
            "Training_Seconds": 0.0,
            "Prediction_Seconds": 0.0,
            "SHAP_Seconds": 0.0,
            "Saving_Seconds": 0.0,
            "Total_Seconds": loading_seconds,
            "Total_Formatted": format_duration(loading_seconds),
        }
    ]

    for biome in biomes:
        summary_entry, runtime_entry = analyze_biome(
            full_data=full_data,
            biome=biome,
            input_columns=input_columns,
            output_column=output_column,
            output_root=output_root,
            cross_validation_splits=cross_validation_splits,
            cross_validation_seed=cross_validation_seed,
            model_parameters=model_parameters,
        )
        if summary_entry is not None:
            metric_rows.append(summary_entry)
        runtime_rows.append(runtime_entry)

    global_save_start = time.perf_counter()
    metrics_summary = pd.DataFrame(metric_rows).sort_values("Biome")
    summary_csv = output_root / "all_biomes_metrics_summary.csv"
    metrics_summary.to_csv(summary_csv, index=False)
    global_save_seconds = float(time.perf_counter() - global_save_start)

    total_seconds = float(time.perf_counter() - total_start)
    runtime_rows.append(
        {
            "Stage": "Global_Summary_Saving",
            "Status": "Completed",
            "Samples": len(metrics_summary),
            "Training_Seconds": 0.0,
            "Prediction_Seconds": 0.0,
            "SHAP_Seconds": 0.0,
            "Saving_Seconds": global_save_seconds,
            "Total_Seconds": global_save_seconds,
            "Total_Formatted": format_duration(global_save_seconds),
        }
    )
    runtime_rows.append(
        {
            "Stage": "Entire_Program",
            "Status": "Completed",
            "Samples": len(full_data),
            "Training_Seconds": sum(float(row.get("Training_Seconds", 0.0)) for row in runtime_rows),
            "Prediction_Seconds": sum(float(row.get("Prediction_Seconds", 0.0)) for row in runtime_rows),
            "SHAP_Seconds": sum(float(row.get("SHAP_Seconds", 0.0)) for row in runtime_rows),
            "Saving_Seconds": sum(float(row.get("Saving_Seconds", 0.0)) for row in runtime_rows),
            "Total_Seconds": total_seconds,
            "Total_Formatted": format_duration(total_seconds),
        }
    )
    runtime_csv = output_root / "runtime_summary.csv"
    pd.DataFrame(runtime_rows).to_csv(runtime_csv, index=False)

    print(f"\n所有Biome精度汇总：{summary_csv}", flush=True)
    print(f"运行时间汇总：{runtime_csv}", flush=True)
    print(f"所有处理完成，总耗时：{format_duration(total_seconds)}（{total_seconds:.2f}秒）", flush=True)
    return total_seconds


def main() -> None:
    """主函数：集中设置全部可自定义变量和相对输入输出路径。"""

    script_directory = Path(__file__).resolve().parent
    input_excel = script_directory / "delta_dNBR_datasets.xlsx"
    output_root = script_directory / "Figure_results"
    biomes = [1, 2, 3, 4, 5, 6, 12]
    input_columns = ["AGB_diff", "NDVI_diff", "LFMC_diff", "VPD_diff", "Sc_diff", "Rd_diff"]
    output_column = "dNBR_diff"
    cross_validation_splits = 10
    cross_validation_seed = 42
    model_parameters: dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": 8,
        "max_depth": 3,
        "min_child_weight": 5,
        "n_estimators": 500,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 3,
        "random_state": 42,
    }

    program_start = time.perf_counter()
    try:
        run_analysis(
            input_excel=input_excel,
            output_root=output_root,
            biomes=biomes,
            input_columns=input_columns,
            output_column=output_column,
            cross_validation_splits=cross_validation_splits,
            cross_validation_seed=cross_validation_seed,
            model_parameters=model_parameters,
        )
    except Exception:
        elapsed = float(time.perf_counter() - program_start)
        print(f"程序异常结束，已运行：{format_duration(elapsed)}（{elapsed:.2f}秒）", flush=True)
        raise


if __name__ == "__main__":
    main()
