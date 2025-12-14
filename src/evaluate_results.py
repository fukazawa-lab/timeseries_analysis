# evaluate_results.py

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULT_DIR = "timeseries_analysis/result"

def evaluate_predictions(csv_path: str, output_csv: str = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    required_cols = {"target", "prediction", "flag"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # flag が prediction の行のみ評価
    df_pred = df[df["flag"] == "prediction"]
    if df_pred.empty:
        raise ValueError("No rows with flag == 'prediction'")

    y_true = df_pred["target"].values
    y_pred = df_pred["prediction"].values

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # r2 = r2_score(y_true, y_pred)

    results_df = pd.DataFrame([{
        "File": os.path.basename(csv_path),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        # "MAPE(%)": round(mape, 2),
        # "R2": round(r2, 2),
    }])

    # 画面表示（CSV形式）
    print(results_df.to_csv(index=False))

    # 必要なら保存
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

def evaluate_predictions_all(result_dir: str = RESULT_DIR, output_csv: str = None):
    csv_files = glob.glob(os.path.join(result_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {result_dir}")
        return

    results = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "target" not in df.columns or "prediction" not in df.columns or "flag" not in df.columns:
            continue

        # flagがpredictionの行のみ評価対象
        df_pred = df[df["flag"] == "prediction"]
        if df_pred.empty:
            continue

        y_true = df_pred["target"].values
        y_pred = df_pred["prediction"].values

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)

        results.append({
            "File": os.path.basename(csv_file),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            # "MAPE(%)": round(mape, 2),
            # "R2": round(r2, 2)
        })

    # DataFrameにまとめてコンマ区切りで表示
    results_df = pd.DataFrame(results)
    print(results_df.to_csv(index=False))

    # 必要ならCSVとして書き出し
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # 例
    evaluate_prediction_csv(
        csv_path="result/granite.csv",
        output_csv="result/eval_granite.csv",
    )
