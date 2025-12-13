import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def rolling_forecast_rf(
    data: pd.DataFrame,
    target_column: str = "target",
    T: int = 1250,             # 学習データ長
    context_length: int = 300, # 過去何ステップ使うか
    rolling_step: int = 1,     # ここでは固定
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
):
    """
    RandomForest-based rolling forecast.
    context_length を特徴量、rolling_step=1 を予測する。
    """
    series = data[target_column].values

    # ===== 学習用特徴量とラベル作成 =====
    X_train = []
    y_train = []
    for i in range(T - context_length - rolling_step + 1):
        X_train.append(series[i : i + context_length])
        y_train.append(series[i + context_length : i + context_length + rolling_step])

    X_train = np.array(X_train)  # (n_samples, context_length)
    y_train = np.array(y_train).flatten()  # (n_samples,)

    # ===== RandomForest モデル =====
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # ===== Rolling Forecast =====
    out_df = data.copy().reset_index(drop=True)
    out_df["prediction"] = series
    out_df["flag"] = "input"
    out_df["roll_id"] = 0
    out_df["row_index"] = range(len(out_df))

    rolling_series = series.copy()
    current_T = T
    roll = 1

    while current_T < len(series):
        # 1ステップずつ予測
        context = rolling_series[current_T - context_length : current_T].reshape(1, -1)
        pred = rf.predict(context)[0]

        rolling_series[current_T] = pred

        out_df.loc[current_T, "prediction"] = pred
        out_df.loc[current_T, "flag"] = "prediction"
        out_df.loc[current_T, "roll_id"] = roll

        current_T += 1
        roll += 1

    return out_df
