import numpy as np
import pandas as pd
import os

def generate_dummy_timeseries(
    n_points=1500,
    period=50,
    noise_std=0.5,
    trend=0.0005,
    seed=42,
    out_dir="dataset",
    filename="dummy.csv"
):
    """
    ノイズの乗った周期的な時系列データを生成し、
    1カラム（target）のCSVとして保存する。

    Parameters
    ----------
    n_points : int
        時系列の長さ（1500以上推奨）
    period : int
        周期
    noise_std : float
        ノイズの標準偏差
    trend : float
        緩やかな線形トレンド
    seed : int
        乱数シード
    out_dir : str
        出力ディレクトリ
    filename : str
        出力CSVファイル名
    """

    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    t = np.arange(n_points)

    # 周期成分
    signal = np.sin(2 * np.pi * t / period)

    # ノイズ
    noise = np.random.normal(0, noise_std, n_points)

    # トレンド付き時系列
    data = signal + noise + trend * t

    # DataFrame（1カラムのみ）
    df = pd.DataFrame({
        "target": data
    })

    df.to_csv(out_path, index=False)

    print(f"Saved dummy time series to: {out_path}")
    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    generate_dummy_timeseries()
