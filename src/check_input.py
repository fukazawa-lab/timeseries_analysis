# check_input.py

import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_and_check_csv(file_path: str) -> pd.DataFrame:
    """
    CSVファイルを読み込み、以下のチェックを行う:
    - UTF-8であること
    - 1カラムのみ
    - カラム内がすべて数値
    - 行数が1500以上
    問題なければDataFrameを返す。問題があればエラーを出して終了。
    正常な場合は関数内でメッセージ出力と可視化も行う。
    """
    # UTF-8で読み込みチェック
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        raise ValueError(f"ファイル '{file_path}' はUTF-8ではありません。")

    # カラム数チェック
    if df.shape[1] != 1:
        raise ValueError(f"ファイル '{file_path}' は1カラムである必要があります。現在 {df.shape[1]} カラムです。")

    col_name = df.columns[0]

    # 数値チェック
    if not pd.api.types.is_numeric_dtype(df[col_name]):
        numeric_series = pd.to_numeric(df[col_name], errors='coerce')
        if numeric_series.isna().any():
            raise ValueError(f"カラム '{col_name}' に数値以外のデータが含まれています。")
        else:
            df[col_name] = numeric_series

    # 行数チェック
    if df.shape[0] < 1500:
        raise ValueError(f"行数が1500未満です。現在 {df.shape[0]} 行です。")

    # 正常メッセージ
    print(f"ファイル '{file_path}' は正常です。行数: {df.shape[0]}, カラム名: {col_name}")

    # データ可視化（折れ線グラフのみ、英語表記）
    plt.figure(figsize=(10, 4))
    plt.plot(df[col_name], color='orange')
    plt.title(f"{col_name} Line Plot")
    plt.xlabel("Row Index")
    plt.ylabel(col_name)
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    file_path = "input.csv"
    try:
        df = load_and_check_csv(file_path)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)
