# plot_predictions.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_target_vs_prediction(csv_path: str, algorithm_name: str = "Zero-shot Forecast"):
    """
    CSVから予測と正解値を読み込み、予測開始位置からの推移を可視化する。

    Parameters
    ----------
    csv_path : str
        予測データが保存されたCSVファイルのパス。
    algorithm_name : str
        図のタイトルに表示するアルゴリズム名。
    """
    # === CSV読み込み ===
    df = pd.read_csv(csv_path)

    # === prediction 開始位置 ===
    pred_start_idx = df.index[df["flag"] == "prediction"][0]
    pred_start_row = df.loc[pred_start_idx, "row_index"]

    # === 可視化 ===
    plt.figure(figsize=(12, 5))

    # --- target（全体） ---
    plt.plot(
        df["row_index"],
        df["target"],
        label="target",
        linewidth=3,
        color="black"
    )

    # --- prediction（赤線以降のみ） ---
    df_pred = df[df["row_index"] >= pred_start_row]
    plt.plot(
        df_pred["row_index"],
        df_pred["prediction"],
        label="prediction",
        linewidth=3,
        linestyle="--",
        color="orange"
    )

    # --- 赤い縦線（境界） ---
    plt.axvline(
        x=pred_start_row,
        color="red",
        linestyle=":",
        linewidth=3,
        label="prediction start"
    )

    # === 装飾 ===
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title(f"{algorithm_name}", fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === スクリプト実行用 ===
if __name__ == "__main__":
    csv_file = "ttm_finetuned_models/storage_inflow_discharge_predictions_rolling.csv"
    plot_target_vs_prediction(csv_file, algorithm_name="Granite TTM")
