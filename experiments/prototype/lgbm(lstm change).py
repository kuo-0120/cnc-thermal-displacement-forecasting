import os
import glob
import re
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb  # 導入 lightgbm

# 忽略 pandas 相關警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------- 0. 超參數設定 -----------------
SEQUENCE_LENGTH = 100 
SLIDING_WINDOW_SIZE = 15

# LightGBM 的超參數，可以根據需要進行調整
LGB_PARAMS = {
    'objective': 'regression_l1',  # MAE Loss, 對離群值較不敏感
    'metric': 'rmse', 
    'n_estimators': 2000,          # 樹的數量
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,                  # 使用所有可用的 CPU 核心
    'seed': 42,
    'boosting_type': 'gbdt',
}

# --- 路徑設定 ---
TRAIN_DATA_DIR = "./train/"
TEST_DATA_DIR = "./test/"
ENV_SETTINGS_FILE = "./檔案環境設定總表.xlsx"
SUBMISSION_DIR = "./submission/"

# --- 亂數種子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED) 已經不需要

# ----------------- 新增輔助函數：計算 Env_Temp (維持修正後的版本) -----------------
def calculate_env_temp(time_min, temp_desc):
    temp_desc = str(temp_desc).strip()
    
    # 先提取主溫度數字，忽略括號如 (parameter 5)
    main_temp_match = re.search(r'(\d+\.?\d*)', temp_desc)
    if main_temp_match:
        main_temp = float(main_temp_match.group(1))
    else:
        print(f"Warning: Unknown temp_desc '{temp_desc}', using default 25.")
        return 25.0
    
    # 恆溫：如果無其他模式，直接返回主溫度
    if '→' not in temp_desc and '[' not in temp_desc:
        return main_temp
    
    # 暖機：[0→6, 30] 表示 0-360 min 為 30 度，之後為主溫度
    warmup_match = re.search(r'\[0→6,\s*(\d+)\]', temp_desc)
    if warmup_match:
        warmup_temp = float(warmup_match.group(1))
        if time_min <= 360:
            return warmup_temp
        # 暖機結束後，如果還有變溫指令，則讓後續的變溫邏輯處理
        # 否則返回主溫度
        if '→' not in temp_desc:
            return main_temp

    # 變溫：解析多段，如 "15→25→15 (diff +-2°C per 20min)"
    # (此處使用前次已修正好的版本，能夠正確解析複雜變溫)
    if '→' in temp_desc:
        temp_points_str = re.findall(r'(\d+\.?\d+)', temp_desc.split('(')[0])
        if len(temp_points_str) < 2: return main_temp
        
        temp_points = [float(p) for p in temp_points_str]
        segments = [(temp_points[i], temp_points[i+1]) for i in range(len(temp_points) - 1)]

        rate_match = re.search(r'(\d+\.?\d*)\s*°?C?\s*per\s*(\d+)\s*min', temp_desc, re.IGNORECASE)
        if not rate_match: return main_temp

        rate = float(rate_match.group(1))
        per_min = float(rate_match.group(2))
        rate_per_min = rate / per_min

        cum_time = 0
        current_temp = segments[0][0]
        
        time_min_adjusted = time_min - 360 if warmup_match else time_min
        if time_min_adjusted < 0: time_min_adjusted = 0

        for i, (start_temp, end_temp) in enumerate(segments):
            signed_rate = abs(rate_per_min) if end_temp > start_temp else -abs(rate_per_min)
            change_duration = abs(end_temp - start_temp) / abs(signed_rate) if signed_rate != 0 else float('inf')

            if time_min_adjusted <= cum_time + change_duration:
                return current_temp + signed_rate * (time_min_adjusted - cum_time)

            cum_time += change_duration
            current_temp = end_temp

            if i < len(segments) - 1:
                if time_min_adjusted <= cum_time + 60:
                    return current_temp
                cum_time += 60
        
        return current_temp
    
    return main_temp

# ----------------- 1. 資料預處理 (與原版相同) -----------------
def preprocess_dataframe(df, env_settings, file_date, feature_cols_template=None, is_train=False):
    settings_row = env_settings[env_settings['日期'] == file_date]
    if not settings_row.empty and is_train:
        time_cols = [col for col in settings_row.columns if '時間' in col]
        total_hr = sum(settings_row[col].fillna(0).values[0] for col in time_cols)
        total_min = total_hr * 60
        df = df[df['Time'] <= total_min]

    if not settings_row.empty:
        for col in settings_row.columns:
            if col != '日期':
                df.loc[:, col] = settings_row[col].values[0]

    if '環境條件_控溫' in df.columns:
        df['環境條件_控溫'] = df['環境條件_控溫'].astype('category')
        one_hot_df = pd.get_dummies(df['環境條件_控溫'], prefix='控溫模式', dtype=float)
        df = pd.concat([df, one_hot_df], axis=1)
        df.drop(columns=['環境條件_控溫'], inplace=True)

    if '環境條件_溫度' in df.columns:
        temp_desc = df['環境條件_溫度'].iloc[0]
        df['Env_Temp'] = df['Time'].apply(lambda t: calculate_env_temp(t, temp_desc))
        df['環境條件_溫度_cleaned'] = df['Env_Temp']
        df.drop(columns=['環境條件_溫度'], inplace=True)

    stage_cols_to_drop = [col for col in df.columns if '段_' in col]
    df.drop(columns=stage_cols_to_drop, errors='ignore', inplace=True)

    temp_cols = [f'PT{i:02d}' for i in range(1, 14)] + \
                [f'TC{i:02d}' for i in range(1, 9)] + \
                ['Spindle Motor', 'X Motor', 'Z Motor']

    for col in temp_cols:
        if col in df.columns:
            df[f'{col}_mean'] = df[col].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).mean()
            df[f'{col}_std'] = df[col].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).std()

    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    if feature_cols_template:
        for col in feature_cols_template:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_cols_template]
    return df

# ----------------- 2. 建立適用於 LightGBM 的資料集 (修改) -----------------
def create_lgb_dataset(dataframes, feature_cols, target_col, scaler, seq_length=SEQUENCE_LENGTH):
    """
    將時序資料轉換為 LightGBM 適用的 (X, y) 格式。
    X 的每一行都是由 seq_length 個時間步的特徵攤平而來。
    """
    X_list, y_list = [], []
    
    # 移除目標變數本身，避免 data leakage
    features_for_flatten = [col for col in feature_cols if col not in ['Disp. X', 'Disp. Z']]

    for df in tqdm(dataframes, desc=f"Creating LightGBM dataset for {target_col}"):
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
        
        features = df_scaled[feature_cols].values
        labels = df_scaled[target_col].values

        for i in range(len(features) - seq_length):
            sequence_window = features[i:i+seq_length]
            
            feature_indices = [feature_cols.index(f) for f in features_for_flatten]
            flattened_features = sequence_window[:, feature_indices].flatten()

            X_list.append(flattened_features)
            y_list.append(labels[i+seq_length])
            
    return np.array(X_list), np.array(y_list)

# ----------------- 3. 模型結構 (移除 LSTM 模型) -----------------
# (此區塊移除 PyTorch 的 nn.Module, 直接使用 lgb.LGBMRegressor)

# ----------------- 4. 訓練流程 (修改為 LightGBM 訓練) -----------------
def train_model(target_axis, train_dfs, feature_cols, scaler):
    print(f"--- 開始訓練 {target_axis} 軸模型 (LightGBM) ---")

    target_col = f'Disp. {target_axis}'
    
    X_train, y_train = create_lgb_dataset(train_dfs, feature_cols, target_col, scaler)
    
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    
    print(f"開始擬合 {target_axis} 軸模型...")
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=True)])

    print(f"{target_axis} 軸模型訓練完成！")
    return model

# ----------------- 5. 測試與輸出流程 (修改為 LightGBM 預測) -----------------
def test_and_export(model_X, model_Z, scaler, feature_cols, env_settings):
    print("\n--- 開始執行測試集推論與匯出 (自迴歸模式) ---")

    disp_x_idx = feature_cols.index('Disp. X')
    disp_z_idx = feature_cols.index('Disp. Z')

    disp_x_mean = scaler.mean_[disp_x_idx]
    disp_x_std = scaler.scale_[disp_x_idx]
    disp_z_mean = scaler.mean_[disp_z_idx]
    disp_z_std = scaler.scale_[disp_z_idx]
    
    features_for_flatten = [col for col in feature_cols if col not in ['Disp. X', 'Disp. Z']]
    feature_indices = [feature_cols.index(f) for f in features_for_flatten]

    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))

    for f in test_files:
        print(f"正在處理檔案: {os.path.basename(f)}")
        df_original = pd.read_csv(f)
        match = re.search(r'_(\d{8})_', os.path.basename(f))
        if not match: continue
        file_date = match.group(1)

        df_proc = preprocess_dataframe(df_original.copy(), env_settings, file_date, feature_cols, is_train=False)
        df_scaled = scaler.transform(df_proc)

        history_scaled = df_scaled.tolist()
        predictions_X, predictions_Z = [], []
        
        pbar = tqdm(range(SEQUENCE_LENGTH, len(df_proc)), desc="Autoregressive Prediction")
        for i in pbar:
            sequence_window = np.array(history_scaled[i-SEQUENCE_LENGTH:i])
            
            flattened_input = sequence_window[:, feature_indices].flatten().reshape(1, -1)
            
            pred_X_scaled = model_X.predict(flattened_input)[0]
            pred_Z_scaled = model_Z.predict(flattened_input)[0]

            next_features_scaled = history_scaled[i].copy()
            next_features_scaled[disp_x_idx] = pred_X_scaled
            next_features_scaled[disp_z_idx] = pred_Z_scaled
            history_scaled[i] = next_features_scaled
            
            pred_X_final = (pred_X_scaled * disp_x_std) + disp_x_mean
            pred_Z_final = (pred_Z_scaled * disp_z_std) + disp_z_mean
            predictions_X.append(pred_X_final)
            predictions_Z.append(pred_Z_final)

        df_out = df_original.copy()
        # 注意 loc 的索引要對齊 predictions 的長度
        df_out.loc[SEQUENCE_LENGTH:, 'Disp. X'] = predictions_X
        df_out.loc[SEQUENCE_LENGTH:, 'Disp. Z'] = predictions_Z

        outpath = os.path.join(SUBMISSION_DIR, os.path.basename(f))
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        df_out.to_csv(outpath, index=False)
        print(f"已儲存預測結果至 {outpath}")

# ----------------- 6. 主程式 (與原版大部分相同) -----------------
if __name__ == "__main__":
    env_settings = pd.read_excel(ENV_SETTINGS_FILE, header=[0, 1])
    new_cols = []
    for c in env_settings.columns:
        part1 = c[0].strip() if "Unnamed" not in c[0] else ""
        part2 = c[1].strip() if "Unnamed" not in c[1] else ""
        new_col = f"{part1}_{part2}" if part1 and part2 else part1 if part1 else part2
        new_cols.append(new_col)
    env_settings.columns = new_cols
    env_settings.rename(columns={env_settings.columns[0]: '日期'}, inplace=True)
    env_settings['日期'] = env_settings['日期'].astype(str)

    all_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.csv"))
    train_dfs = []
    pbar_load = tqdm(all_files, desc="Loading and Preprocessing Train Data")
    for f in pbar_load:
        match = re.search(r'_(\d{8})_', os.path.basename(f))
        if match:
            df_raw = pd.read_csv(f)
            if 'Time' in df_raw.columns and df_raw['Time'].dtype == 'object':
                 df_raw['Time'] = pd.to_timedelta(df_raw['Time']).dt.total_seconds() / 60.0
            train_dfs.append(preprocess_dataframe(df_raw.copy(), env_settings, match.group(1), is_train=True))

    combined_df = pd.concat(train_dfs, ignore_index=True)
    feature_cols = [col for col in combined_df.columns if col not in ['Time']]
    if 'Disp. X' not in feature_cols: feature_cols.append('Disp. X')
    if 'Disp. Z' not in feature_cols: feature_cols.append('Disp. Z')
    feature_cols.sort()

    processed_train_dfs = []
    for df in train_dfs:
        temp_df = df.copy()
        for col in feature_cols:
            if col not in temp_df.columns:
                temp_df[col] = 0.0
        processed_train_dfs.append(temp_df[feature_cols])

    final_combined_df = pd.concat(processed_train_dfs, ignore_index=True)
    scaler = StandardScaler()
    print("Fitting Scaler on all training data...")
    scaler.fit(final_combined_df[feature_cols])
    print("Scaler fitted.")

    model_X = train_model('X', processed_train_dfs, feature_cols, scaler)
    model_Z = train_model('Z', processed_train_dfs, feature_cols, scaler)

    test_and_export(model_X, model_Z, scaler, feature_cols, env_settings)

    print("\n所有流程執行完畢。")