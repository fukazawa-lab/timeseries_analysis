import os
import tempfile
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, set_seed
from tsfm_public import TimeSeriesPreprocessor, get_datasets, get_model


def rolling_forecast_ttm(
    data: pd.DataFrame,
    target_columns: list = ["target"],      # デフォルト値
    T: int = 1024,                          # 必要に応じてデフォルト設定
    context_length: int = 1024,             # デフォルト値
    prediction_length: int = 96,            # デフォルト値
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2",  # デフォルト値
    batch_size: int = 64,
    seed: int = 42,
):


    assert prediction_length == 96, "Granite TTM は prediction_length=96 固定です"

    set_seed(seed)

    total_len = len(data)
    max_horizon = total_len - T
    n_rolls = int(np.ceil(max_horizon / prediction_length))

    out_df = data.copy().reset_index(drop=True)
    out_df["prediction_scaled"] = out_df[target_columns[0]]
    out_df["prediction"] = out_df[target_columns[0]]
    out_df["flag"] = "input"
    out_df["roll_id"] = 0
    out_df["row_index"] = range(total_len)

    rolling_target = out_df[target_columns[0]].copy()

    # ===== モデル（prediction_length は固定）=====
    model = get_model(
        model_path,
        context_length=context_length,
        prediction_length=prediction_length,
        freq_prefix_tuning=False,
        freq=None,
        prefer_l1_loss=False,
        prefer_longer_context=True,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_eval_batch_size=batch_size,
            seed=seed,
            report_to="none",
        ),
    )

    current_T = T

    for roll in range(n_rolls):
        print(f"\n=== Rolling {roll + 1}/{n_rolls} ===")

        remaining = total_len - current_T
        if remaining <= 0:
            break

        # ★ pred_len は「使う長さ」だけ（モデルには渡さない）
        use_len = min(prediction_length, remaining)

        tsp = TimeSeriesPreprocessor(
            timestamp_column=None,
            id_columns=[],
            target_columns=target_columns,
            control_columns=[],
            context_length=context_length,
            prediction_length=prediction_length,  # ★ 常に 96
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",
        )

        split_config = {
            "train": [0, current_T],
            "valid": [0, current_T],
            "test":  [0, current_T + prediction_length],
        }

        tmp_df = pd.DataFrame({target_columns[0]: rolling_target})

        _, _, dset_test = get_datasets(
            tsp,
            tmp_df,
            split_config,
            use_frequency_token=model.config.resolution_prefix_tuning
        )

        preds = trainer.predict(dset_test)
        pred_scaled_full = preds.predictions[0][-1].squeeze()

        # ★ 余った分だけ切り出す
        pred_scaled = pred_scaled_full[:use_len]

        scaler = tsp.target_scaler_dict["0"]
        pred_original = scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()

        start = current_T
        end = current_T + use_len

        rolling_target.iloc[start:end] = pred_original

        out_df.loc[start:end-1, "prediction_scaled"] = pred_scaled
        out_df.loc[start:end-1, "prediction"] = pred_original
        out_df.loc[start:end-1, "flag"] = "prediction"
        out_df.loc[start:end-1, "roll_id"] = roll + 1

        current_T += use_len

    return out_df

