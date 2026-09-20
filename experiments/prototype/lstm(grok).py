import os
import glob
import re
import random
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 忽略 pandas 相關警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ----------------- 0. 超參數設定 -----------------
SEQUENCE_LENGTH = 100 
SLIDING_WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 路徑設定 ---
TRAIN_DATA_DIR = "./train/"
TEST_DATA_DIR = "./test/"
ENV_SETTINGS_FILE = "./檔案環境設定總表.xlsx"
SUBMISSION_DIR = "./submission/"

# --- 亂數種子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------- 新增輔助函數：計算 Env_Temp (修改版) -----------------
def calculate_env_temp(time_min, temp_desc):
    """
    解析環境溫度描述字串，支援以下格式：
    - 恆溫："25C"、"25"
    - 暖機："[0→6, 30]"
    - 變溫（多段）："15→25→15 (diff +-2°C per 20min)"、"15→25→15 (diff ±2°C per 20 min)"
    - 變溫（單段）："15→25 (rise 2 per 30 min)" 或 "25→15 (fall 1 per 10min)"
    
    備註：
    - 將 "+-2" 與 "±2" 解讀為速率幅度 2（取絕對值），實作上以數值 2 / 週期分鐘。
    - time_min 單位為分鐘。
    """
    # 若 time_min 不是數值（例如是 "HH:MM:SS"），嘗試轉換
    if isinstance(time_min, str):
        try:
            time_min = pd.to_timedelta(time_min).total_seconds() / 60.0
        except Exception:
            pass  # 若轉換失敗，後續比較可能失敗，但不在此處拋錯

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
        if isinstance(time_min, (int, float)):
            return warmup_temp if time_min <= 360 else main_temp
        else:
            return warmup_temp

    # 變溫：解析多段，如 "15→25→15 (diff +-2°C per 20min)"
    segments = []
    arrows = re.findall(r'(\d+)→(\d+)', temp_desc)
    if arrows:
        # 蒐集每一段的 (start, end)
        for start, end in arrows:
            segments.append((float(start), float(end)))
        # 提取變化速率與時間單位，容錯支援 +-, ±, +, - 以及空白
        rate_match = re.search(
            r'(?:diff\s*)?(?:\+\-|±|\+|\-)?\s*(\d+(?:\.\d+)?)\s*°?C?\s*per\s*(\d+)\s*min',
            temp_desc, re.IGNORECASE
        )
        if rate_match:
            rate = float(rate_match.group(1))  # 只取數值大小
            per_min = float(rate_match.group(2))
        else:
            rate, per_min = 1.0, 30.0  # 預設 1 度 / 30 分
        rate_per_min_base = rate / per_min
    else:
        # 單段變溫：如 "15→25 (rise 2 per 30 min)"
        match = re.search(r'(\d+)→(\d+)\s*\((rise|fall)\s*(\d+)\s*per\s*(\d+)\s*min\)', temp_desc, re.IGNORECASE)
        if match:
            start, end = float(match.group(1)), float(match.group(2))
            mode, rate, per_min = match.group(3).lower(), float(match.group(4)), float(match.group(5))
            rate_per_min = rate / per_min if mode == 'rise' else -rate / per_min
            change_dur = abs(end - start) / abs(rate_per_min) if rate_per_min != 0 else 0
            if not isinstance(time_min, (int, float)):
                return end
            if time_min <= change_dur:
                return start + rate_per_min * time_min
            else:
                return end  # 超過後維持上限/下限
        else:
            # 如果無法解析變溫，返回主溫度
            return main_temp

    # 多段變溫：逐段計算，變化後停留 60 min
    cum_time = 0.0
    current_temp = segments[0][0]
    for start, end in segments:
        mode = 'rise' if end > start else 'fall'
        rate_per_min = rate_per_min_base if mode == 'rise' else -rate_per_min_base
        if rate_per_min == 0:
            change_dur = 0
        else:
            change_dur = abs(end - start) / abs(rate_per_min)

        if not isinstance(time_min, (int, float)):
            return end

        if time_min <= cum_time + change_dur:
            return current_temp + rate_per_min * (time_min - cum_time)
        cum_time += change_dur
        current_temp = end
        # 變化完成後停留 1 小時
        if time_min <= cum_time + 60:
            return current_temp
        cum_time += 60

    return current_temp  # 超過所有段後維持最後值

# ----------------- 1. 資料預處理 -----------------
def preprocess_dataframe(df, env_settings, file_date, feature_cols_template=None, is_train=False):
    # 確保 Time 欄位為分鐘數（無論 train 或 test）
    if 'Time' in df.columns and df['Time'].dtype == 'object':
        try:
            df.loc[:, 'Time'] = pd.to_timedelta(df['Time']).dt.total_seconds() / 60.0
        except Exception:
            pass

    settings_row = env_settings[env_settings['日期'] == file_date]
    if not settings_row.empty and is_train:
        time_cols = [col for col in settings_row.columns if '時間' in col]
        total_hr = sum(settings_row[col].fillna(0).values[0] for col in time_cols)
        total_min = total_hr * 60
        df = df[df['Time'] <= total_min]

    if not settings_row.empty:
        for col in settings_row.columns:
            if col != '日期':
                df.loc[:, col] = settings_row[col].values[0]  # 使用 .loc 避免 SettingWithCopyWarning

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

# ----------------- 2. LSTM 資料集 -----------------
class DisplacementDataset(Dataset):
    def __init__(self, dataframes, feature_cols, target_col, scaler, seq_length=SEQUENCE_LENGTH):
        self.X_list = []
        self.y_list = []
        features_for_input = [col for col in feature_cols if col not in ['Disp. X', 'Disp. Z']]

        for df in tqdm(dataframes, desc=f"Creating LSTM dataset for {target_col}"):
            df_scaled = df.copy()
            df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
            
            features = df_scaled[feature_cols].values
            labels = df_scaled[target_col].values

            for i in range(len(features) - seq_length):
                sequence_window = features[i:i+seq_length]
                input_seq = sequence_window[:, [feature_cols.index(f) for f in features_for_input]]
                self.X_list.append(input_seq)
                self.y_list.append(labels[i+seq_length])
        
    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X_list[idx], dtype=torch.float32), torch.tensor(self.y_list[idx], dtype=torch.float32)

# ----------------- 3. LSTM 模型 -----------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()

# ----------------- 4. 訓練流程 -----------------
def train_model(target_axis, train_dfs, feature_cols, scaler):
    print(f"--- 開始訓練 {target_axis} 軸 LSTM 模型 ---")
    target_col = f'Disp. {target_axis}'
    
    dataset = DisplacementDataset(train_dfs, feature_cols, target_col, scaler)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_size = len([col for col in feature_cols if col not in ['Disp. X', 'Disp. Z']])
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")
    
    print(f"{target_axis} 軸 LSTM 模型訓練完成！")
    return model

# ----------------- 5. 測試與輸出流程 -----------------
def test_and_export(model_X, model_Z, scaler, feature_cols, env_settings):
    print("\n--- 開始執行測試集推論與匯出 (自迴歸模式) ---")

    disp_x_idx = feature_cols.index('Disp. X')
    disp_z_idx = feature_cols.index('Disp. Z')

    disp_x_mean = scaler.mean_[disp_x_idx]
    disp_x_std = scaler.scale_[disp_x_idx]
    disp_z_mean = scaler.mean_[disp_z_idx]
    disp_z_std = scaler.scale_[disp_z_idx]
    
    features_for_input = [col for col in feature_cols if col not in ['Disp. X', 'Disp. Z']]
    feature_indices = [feature_cols.index(f) for f in features_for_input]

    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.csv"))

    for f in test_files:
        print(f"正在處理檔案: {os.path.basename(f)}")
        df_original = pd.read_csv(f)
        # 確保 Time 在 test 也轉為分鐘
        if 'Time' in df_original.columns and df_original['Time'].dtype == 'object':
            try:
                df_original.loc[:, 'Time'] = pd.to_timedelta(df_original['Time']).dt.total_seconds() / 60.0
            except Exception:
                pass

        match = re.search(r'_(\d{8})_', os.path.basename(f))
        if not match:
            continue
        file_date = match.group(1)

        df_proc = preprocess_dataframe(df_original.copy(), env_settings, file_date, feature_cols, is_train=False)
        df_scaled = scaler.transform(df_proc)

        history_scaled = df_scaled.tolist()
        predictions_X, predictions_Z = [], []
        
        pbar = tqdm(range(SEQUENCE_LENGTH, len(df_proc)), desc="Autoregressive Prediction")
        for i in pbar:
            sequence_window = np.array(history_scaled[i-SEQUENCE_LENGTH:i])
            input_seq = sequence_window[:, feature_indices]
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred_X_scaled = model_X(input_tensor).item()
                pred_Z_scaled = model_Z(input_tensor).item()

            next_features_scaled = history_scaled[i].copy()
            next_features_scaled[disp_x_idx] = pred_X_scaled
            next_features_scaled[disp_z_idx] = pred_Z_scaled
            history_scaled[i] = next_features_scaled
            
            pred_X_final = (pred_X_scaled * disp_x_std) + disp_x_mean
            pred_Z_final = (pred_Z_scaled * disp_z_std) + disp_z_mean
            predictions_X.append(pred_X_final)
            predictions_Z.append(pred_Z_final)

        df_out = df_original.copy()
        df_out.loc[SEQUENCE_LENGTH:, 'Disp. X'] = predictions_X
        df_out.loc[SEQUENCE_LENGTH:, 'Disp. Z'] = predictions_Z

        outpath = os.path.join(SUBMISSION_DIR, os.path.basename(f))
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        df_out.to_csv(outpath, index=False)
        print(f"已儲存預測結果至 {outpath}")

# ----------------- 6. 主程式 -----------------
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
                try:
                    df_raw.loc[:, 'Time'] = pd.to_timedelta(df_raw['Time']).dt.total_seconds() / 60.0
                except Exception:
                    pass
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