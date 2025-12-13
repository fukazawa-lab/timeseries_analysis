import torch
import torch.nn as nn
import pandas as pd

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_length: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        """
        x: (B, context_length, 1)
        return: (B, output_length)
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]     # 最終時刻の hidden
        pred = self.fc(last)
        return pred

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, context_length, pred_length):
        self.series = series.values.astype("float32")
        self.context_length = context_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.series) - self.context_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.context_length]
        y = self.series[
            idx + self.context_length :
            idx + self.context_length + self.pred_length
        ]
        return (
            torch.tensor(x).unsqueeze(-1),  # (context, 1)
            torch.tensor(y),               # (pred_length,)
        )

def rolling_forecast_lstm(
    data: pd.DataFrame,
    target_column: str = "target",  # デフォルト値
    T: int = 1250,                  # デフォルト値
    context_length: int = 100,      # デフォルト値
    rolling_step: int = 1,          # デフォルト値（N-step）
    hidden_size: int = 128,         # デフォルト値
    num_layers: int = 2,            # デフォルト値
    dropout: float = 0.2,           # デフォルト値
    epochs: int = 300,              # デフォルト値
    lr: float = 1e-3,
    device: str = "cuda",           # デフォルト値
):
    series = data[target_column]

    # ===== 学習用 Dataset =====
    train_series = series.iloc[:T]
    train_ds = TimeSeriesDataset(
        train_series, context_length, rolling_step
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True
    )

    # ===== LSTM モデル =====
    model = LSTMForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_length=rolling_step,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ===== 学習 =====
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {ep+1}/{epochs} | "
            f"Loss {total_loss / len(train_loader):.4f}"
        )

    # ===== Rolling Forecast =====
    out_df = data.copy().reset_index(drop=True)
    out_df["prediction"] = out_df[target_column]
    out_df["flag"] = "input"
    out_df["roll_id"] = 0
    out_df["row_index"] = range(len(out_df))

    rolling_series = out_df[target_column].copy()
    current_T = T
    roll = 1

    model.eval()
    while current_T < len(out_df):
        remaining = len(out_df) - current_T
        use_len = min(rolling_step, remaining)

        context = rolling_series.iloc[current_T - context_length : current_T]
        x = torch.tensor(
            context.values, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            pred = model(x).cpu().numpy().flatten()

        pred = pred[:use_len]

        start, end = current_T, current_T + use_len
        rolling_series.iloc[start:end] = pred

        out_df.loc[start:end-1, "prediction"] = pred
        out_df.loc[start:end-1, "flag"] = "prediction"
        out_df.loc[start:end-1, "roll_id"] = roll

        current_T += use_len
        roll += 1

    return out_df
