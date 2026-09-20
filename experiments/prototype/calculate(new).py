import pandas as pd
import numpy as np
import math
import glob
import re
import os

def calculate_rmse_multi(true_dir, pred_dir, env_file):
    """
    計算整個測試集的聚合 RMSE，考慮環境設定過濾超過時間的部分。
    - true_dir: 真值 CSV 資料夾路徑
    - pred_dir: 預測 CSV 資料夾路徑 (submission)
    - env_file: 環境設定 Excel 路徑
    """
    # 載入環境設定 Excel (header=[0,1] 處理多層欄位)
    env_df = pd.read_excel(env_file, header=[0, 1])
    # 清理欄位名稱 (e.g., '第一段_時間 (Hr)' -> '第一段_時間 (Hr)')
    new_cols = []
    for c in env_df.columns:
        part1 = c[0].strip() if "Unnamed" not in c[0] else ""
        part2 = c[1].strip() if "Unnamed" not in c[1] else ""
        new_col = f"{part1}_{part2}" if part1 and part2 else part1 if part1 else part2
        new_cols.append(new_col)
    env_df.columns = new_cols
    env_df.rename(columns={env_df.columns[0]: '日期'}, inplace=True)
    env_df['日期'] = env_df['日期'].astype(str)

    # 計算每個日期的總有效時間 (小時轉分鐘)
    env_dict = {}
    for _, row in env_df.iterrows():
        date = row['日期']
        time_cols = [col for col in row.index if '時間' in col]
        total_hr = sum(row[col] for col in time_cols if pd.notnull(row[col]))
        env_dict[date] = total_hr * 60  # total_min

    # 獲取所有 true 文件，並配對 pred
    true_files = glob.glob(os.path.join(true_dir, "*.csv"))
    total_sum_sq = 0.0
    total_N = 0
    D_count = 0  # |D|

    for true_path in true_files:
        # 提取日期 from 檔名 (e.g., _20200928_)
        match = re.search(r'_(\d{8})_', os.path.basename(true_path))
        if not match:
            print(f"Warning: 無法提取日期 from {true_path}, 跳過。")
            continue
        date = match.group(1)

        # 配對 pred_path
        pred_filename = os.path.basename(true_path)  # 假設檔名相同
        pred_path = os.path.join(pred_dir, pred_filename)
        if not os.path.exists(pred_path):
            print(f"Warning: 無對應 pred 檔案 {pred_path}, 跳過。")
            continue

        # 讀取 CSV
        df_true = pd.read_csv(true_path)
        df_pred = pd.read_csv(pred_path)

        # 確保 Time 是數值 (分鐘)
        if 'Time' in df_true.columns and df_true['Time'].dtype == 'object':
            df_true['Time'] = pd.to_timedelta(df_true['Time']).dt.total_seconds() / 60.0
        if 'Time' in df_pred.columns and df_pred['Time'].dtype == 'object':
            df_pred['Time'] = pd.to_timedelta(df_pred['Time']).dt.total_seconds() / 60.0

        # 過濾超過環境時間的部分
        total_min = env_dict.get(date, float('inf'))  # 如果無設定，用無限大 (不濾)
        if total_min < float('inf'):
            df_true = df_true[df_true['Time'] <= total_min]
            df_pred = df_pred[df_pred['Time'] <= total_min]

        # 檢查筆數是否匹配
        if len(df_true) != len(df_pred):
            print(f"Warning: 檔案 {true_path} 筆數不匹配 (true: {len(df_true)}, pred: {len(df_pred)}), 跳過。")
            continue

        # 從第 101 筆開始 (iloc[100:])
        if len(df_true) <= 100:
            print(f"Warning: 檔案 {true_path} 有效筆數 <=100, 無需計算, 跳過。")
            continue

        true_X = df_true["Disp. X"].iloc[100:].to_numpy()
        true_Z = df_true["Disp. Z"].iloc[100:].to_numpy()
        pred_X = df_pred["Disp. X"].iloc[100:].to_numpy()
        pred_Z = df_pred["Disp. Z"].iloc[100:].to_numpy()

        # 計算該檔的 squared errors
        sq_error = (pred_X - true_X) ** 2 + (pred_Z - true_Z) ** 2
        total_sum_sq += np.sum(sq_error)

        # 累加 N_i (該檔有效總筆數)
        total_N += len(df_true)
        D_count += 1

    if D_count == 0:
        raise ValueError("無有效檔案可計算 RMSE。")

    # 計算 RMSE
    denominator = 2 * (total_N - 100 * D_count)
    if denominator <= 0:
        raise ValueError("無效 denominator (可能總筆數太少)。")
    rmse = math.sqrt(total_sum_sq / denominator)
    return rmse

if __name__ == "__main__":
    # 改成您的資料夾和檔案路徑
    true_dir = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "test_with_answers")
    pred_dir = os.path.join(os.getenv("CNC_OUTPUT_DIR", "outputs"), "submission")
    env_file = os.getenv("CNC_ENV_FILE", os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "env.xlsx"))

    rmse_value = calculate_rmse_multi(true_dir, pred_dir, env_file)
    print(f"模擬測試集 RMSE: {rmse_value:.6f}")