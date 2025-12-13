import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(
    data: pd.DataFrame,
    target_columns: list = ["target"],  # デフォルト値
    T: int = 1024,                      # 必要に応じてデフォルト設定
    arima_order=(1, 1, 1),
):

    target = target_columns[0]
    total_len = len(data)
    horizon = total_len - T

    # ===== 出力DF =====
    out_df = data.copy().reset_index(drop=True)
    out_df["prediction_scaled"] = out_df[target]
    out_df["prediction"] = out_df[target]
    out_df["flag"] = "input"
    out_df["roll_id"] = 0
    out_df["row_index"] = range(total_len)

    # ===== 学習データ =====
    train_series = data.loc[:T-1, target]

    # ===== ARIMA fit =====
    model = ARIMA(train_series, order=arima_order)
    result = model.fit()

    # ===== 予測 =====
    forecast = result.forecast(steps=horizon).values

    start = T
    end = T + horizon

    out_df.loc[start:end-1, "prediction_scaled"] = forecast
    out_df.loc[start:end-1, "prediction"] = forecast
    out_df.loc[start:end-1, "flag"] = "prediction"
    out_df.loc[start:end-1, "roll_id"] = 1

    return out_df
