# -*- coding: utf-8 -*-
"""
LSTM_PRO_0825_patched.py  (v0825b)
- 修補 1：標準化與欄位定位支援重複欄名（pd.Index / slice / ndarray），避免 iloc 型別錯誤。
- 修補 2：rollout（驗證）與測試遞推時，只覆寫已存在於 feature_cols 的 lag 欄位，
          並且在取模型輸入視窗時強制 `[..., feature_cols]`，確保輸入維度固定不變。

可以直接覆蓋原檔使用；BASE_DIR 依你的資料夾調整。
"""

import os, re, glob, math, random, warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================= 路徑與參數 =================
BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
TRAIN_DIR = rf"{BASE_DIR}\train"
TEST_DIR  = rf"{BASE_DIR}\test"
VAL_PRED_DIR = rf"{BASE_DIR}\val_pred\lstm_plus"   # 驗證逐列預測輸出
OUT_DIR      = rf"{BASE_DIR}\submission\lstm_plus" # 測試 submission 輸出
ENV_SETTINGS_FILE = rf"{BASE_DIR}\檔案環境設定總表.xlsx"  # 若不存在會自動略過

os.makedirs(VAL_PRED_DIR, exist_ok=True)
os.makedirs(OUT_DIR,      exist_ok=True)

TARGET_COLS = ["Disp. X", "Disp. Z"]
SEED = 42

# 超參數（可依需要調整）
SEQ_LEN    = 64
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.2
LR         = 1e-3
BATCH_SIZE = 256
EPOCHS     = 80
PATIENCE   = 10
FILL_AFTER_N = 100      # 1-based，自第 100 筆起遇空白才補
SLIDING_WINDOW_SIZE = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 小工具 =================

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def is_blank(v):
    return pd.isna(v) or (isinstance(v, str) and v.strip()=="")

def safe_to_minutes(x):
    """將 Time 欄位轉為 minutes（若原本就是數字則原樣返回）。"""
    if pd.api.types.is_numeric_dtype(type(x)):
        try:
            return float(x)
        except Exception:
            return np.nan
    try:
        td = pd.to_timedelta(str(x))
        return td.total_seconds()/60.0
    except Exception:
        m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", str(x).strip())
        if m:
            h = int(m.group(1)); mi = int(m.group(2)); s = int(m.group(3) or 0)
            return (h*3600+mi*60+s)/60.0
        return np.nan

def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum()==0: return float("nan")
    return math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

# ================= 讀檔 =================

def read_all_csv(csv_dir: str) -> List[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"找不到 CSV：{csv_dir}")
    dfs=[]
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        # Time 正規化（轉 minutes）
        if "Time" in df.columns:
            if df["Time"].dtype=="O":
                df["Time"] = df["Time"].map(safe_to_minutes)
            df = df.sort_values("Time").reset_index(drop=True)
        dfs.append(df)
    return dfs

# ================= Excel 設定（可選） =================

def load_env_settings(excel_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(excel_path):
        print(f"[WARN] 找不到環境設定檔：{excel_path}，將略過環境特徵擴充。")
        return None
    env_settings = pd.read_excel(excel_path, header=[0,1])
    # 清理多層欄名
    new_cols = []
    for c in env_settings.columns:
        part1 = c[0].strip() if "Unnamed" not in str(c[0]) else ""
        part2 = c[1].strip() if "Unnamed" not in str(c[1]) else ""
        new_col = f"{part1}_{part2}" if part1 and part2 else (part1 or part2)
        new_cols.append(new_col)
    env_settings.columns = new_cols
    # 第一欄改名為 日期
    env_settings.rename(columns={env_settings.columns[0]: '日期'}, inplace=True)
    env_settings['日期'] = env_settings['日期'].astype(str)
    return env_settings

DATE_PAT = re.compile(r"_(\d{8})_")

def extract_date_from_filename(fname: str) -> Optional[str]:
    m = DATE_PAT.search(fname)
    return m.group(1) if m else None

# ================= Env_Temp 解析 =================

def calculate_env_temp(time_min: float, temp_desc: str) -> float:
    """根據字串描述估算該分鐘的環境溫度。"""
    s = str(temp_desc or "").strip()
    main_temp_match = re.search(r"(\d+\.?\d*)", s)
    main_temp = float(main_temp_match.group(1)) if main_temp_match else 25.0

    if '→' not in s and '[' not in s:
        return main_temp

    # 暖機：[0→6, 30]
    warmup = re.search(r"\[0→(\d+),\s*(\d+)\]", s)
    if warmup:
        hours = float(warmup.group(1)); warmup_temp = float(warmup.group(2))
        return warmup_temp if time_min <= hours*60 else main_temp

    # 多段
    arrows = re.findall(r"(\d+)→(\d+)", s)
    if arrows:
        rate_match = re.search(r"([+\-]?\d+)°?C?\s*per\s*(\d+)min", s)
        if rate_match:
            rate = abs(float(rate_match.group(1)))
            per_min = float(rate_match.group(2))
            rate_per_min = rate / per_min
        else:
            rate_per_min = 1/30  # 預設

        segments = [(float(a), float(b)) for a,b in arrows]
        cum_time = 0.0
        current = segments[0][0]
        for start, end in segments:
            mode_up = end > start
            step = rate_per_min if mode_up else -rate_per_min
            change_dur = abs(end - start) / abs(step)
            if time_min <= cum_time + change_dur:
                return current + step*(time_min - cum_time)
            cum_time += change_dur
            current = end
            # 每段尾端停留 60 分鐘
            if time_min <= cum_time + 60:
                return current
            cum_time += 60
        return current

    # 單段 rise/fall
    one = re.search(r"(\d+)→(\d+)\s*\((rise|fall)\s*(\d+)\s*per\s*(\d+)min\)", s)
    if one:
        start, end = float(one.group(1)), float(one.group(2))
        mode = one.group(3); rate = float(one.group(4)); per_min = float(one.group(5))
        step = rate/per_min if mode=='rise' else -rate/per_min
        change_dur = abs(end - start)/abs(step)
        if time_min <= change_dur:
            return start + step*time_min
        return end

    return main_temp

# ================= 資料前處理 =================

TEMP_BASE_COLS = [f'PT{i:02d}' for i in range(1,14)] + [f'TC{i:02d}' for i in range(1,9)] + [
    'Spindle Motor', 'X Motor', 'Z Motor']


def enrich_with_env(df: pd.DataFrame, env_settings: Optional[pd.DataFrame], file_date: Optional[str], is_train: bool) -> pd.DataFrame:
    d = df.copy()
    if env_settings is not None and file_date is not None:
        row = env_settings[env_settings['日期']==file_date]
        if not row.empty:
            # 修正訓練長度
            if is_train and '時間' in ''.join(row.columns):
                time_cols = [c for c in row.columns if '時間' in c]
                total_hr = float(sum(row[c].fillna(0).values[0] for c in time_cols)) if time_cols else 0.0
                if total_hr>0 and 'Time' in d.columns:
                    d = d[d['Time'] <= total_hr*60].reset_index(drop=True)
            # 設定值 broadcast
            for col in row.columns:
                if col != '日期':
                    d[col] = row[col].values[0]

    # 控溫 one-hot
    if '環境條件_控溫' in d.columns:
        d['環境條件_控溫'] = d['環境條件_控溫'].astype('category')
        oh = pd.get_dummies(d['環境條件_控溫'], prefix='控溫模式', dtype=float)
        d = pd.concat([d.drop(columns=['環境條件_控溫']), oh], axis=1)

    # Env_Temp
    if '環境條件_溫度' in d.columns:
        temp_desc = d['環境條件_溫度'].iloc[0]
        if 'Time' in d.columns:
            d['Env_Temp'] = d['Time'].apply(lambda t: calculate_env_temp(float(t), str(temp_desc)))
        d['環境條件_溫度_cleaned'] = d.get('Env_Temp', np.nan)
        d = d.drop(columns=['環境條件_溫度'])

    # 移除分段欄位
    drop_cols = [c for c in d.columns if '段_' in c]
    if drop_cols:
        d = d.drop(columns=drop_cols, errors='ignore')

    # 感測器 rolling 統計
    for col in TEMP_BASE_COLS:
        if col in d.columns:
            d[f'{col}_mean'] = d[col].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).mean()
            d[f'{col}_std']  = d[col].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).std()

    d = d.bfill().ffill().fillna(0)
    return d


def add_target_lags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'Disp. X' in d.columns:
        d['DispX_lag1'] = d['Disp. X'].shift(1)
        d['DispX_lag2'] = d['Disp. X'].shift(2)
        d['DispX_lag3'] = d['Disp. X'].shift(3)
    else:
        d['DispX_lag1']=np.nan; d['DispX_lag2']=np.nan; d['DispX_lag3']=np.nan
    if 'Disp. Z' in d.columns:
        d['DispZ_lag1'] = d['Disp. Z'].shift(1)
        d['DispZ_lag2'] = d['Disp. Z'].shift(2)
        d['DispZ_lag3'] = d['Disp. Z'].shift(3)
    else:
        d['DispZ_lag1']=np.nan; d['DispZ_lag2']=np.nan; d['DispZ_lag3']=np.nan
    return d


def infer_feature_cols(train_all: pd.DataFrame) -> List[str]:
    base = add_target_lags(train_all)
    cand = [c for c in base.columns if c not in TARGET_COLS and c != '__source_file__']
    feats=[]
    for c in cand:
        loc = base.columns.get_loc(c)
        if isinstance(loc, (list, tuple, np.ndarray, slice, pd.Index)):
            if c in feats:
                continue
            idx = loc.start if isinstance(loc, slice) else int(np.asarray(loc)[0])
            ser = base.iloc[:, idx]
        else:
            ser = base[c]
        if pd.api.types.is_numeric_dtype(ser):
            feats.append(c)
        else:
            tmp = pd.to_numeric(ser, errors='coerce')
            if tmp.notna().sum()>0: feats.append(c)
    feats = list(dict.fromkeys(feats))
    return feats

# === 修正：標準化時安全處理重複欄名（含 pd.Index） ===

def _first_col_idx(df: pd.DataFrame, name: str) -> int:
    loc = df.columns.get_loc(name)
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, (list, tuple, np.ndarray, pd.Index)):
        return int(np.asarray(loc)[0])
    return int(loc)

def _get_first_series(df: pd.DataFrame, name: str) -> pd.Series:
    idx = _first_col_idx(df, name)
    return df.iloc[:, idx]

def _set_first_series(df: pd.DataFrame, name: str, values: pd.Series):
    idx = _first_col_idx(df, name)
    df.iloc[:, idx] = values


def standardize_by_stats(df_list: List[pd.DataFrame], feature_cols: List[str], mean=None, std=None):
    concat = pd.concat(df_list, ignore_index=True)
    # 先確保所有特徵欄存在，若缺則補 NaN；並把第一個同名欄位轉數值
    for c in feature_cols:
        if c not in concat.columns:
            concat[c] = np.nan
        ser = _get_first_series(concat, c)
        ser = pd.to_numeric(ser, errors='coerce')
        _set_first_series(concat, c, ser)

    if mean is None or std is None:
        data_for_stats = {c: _get_first_series(concat, c) for c in feature_cols}
        stats_df = pd.DataFrame(data_for_stats)
        mean = stats_df.mean().astype('float32').fillna(0.0)
        std  = stats_df.std(ddof=0).replace(0,1.0).astype('float32').fillna(1.0)

    out=[]
    for df in df_list:
        d2 = df.copy()
        for c in feature_cols:
            if c not in d2.columns:
                d2[c] = np.nan
            ser = _get_first_series(d2, c)
            ser = pd.to_numeric(ser, errors='coerce')
            ser = ser.fillna(mean[c])
            sd = std[c] if float(std[c])!=0 else 1.0
            ser = (ser - mean[c]) / sd
            ser = ser.replace([np.inf,-np.inf], np.nan).fillna(0.0)
            _set_first_series(d2, c, ser)
        out.append(d2)
    return out, mean, std


def build_sequences_for_train(df: pd.DataFrame, feature_cols: List[str], seq_len: int):
    if not all(t in df.columns for t in TARGET_COLS):
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    yy = df[TARGET_COLS].apply(pd.to_numeric, errors='coerce').values.astype('float32')
    X  = df[feature_cols].values.astype('float32')
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    xs, ys = [], []
    for t in range(seq_len-1, n):
        if not np.isfinite(yy[t]).all():
            continue
        xs.append(X[t-seq_len+1:t+1]); ys.append(yy[t])
    if not xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    return np.stack(xs), np.stack(ys)

# ================= 模型 =================

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.head(out[:,-1,:])

# ================= 驗證 Rollout & 儲存 =================

def rollout_and_save_val_preds(model: nn.Module,
                               val_dfs_raw: List[pd.DataFrame],
                               env_settings: Optional[pd.DataFrame],
                               mean, std,
                               feature_cols: List[str],
                               seq_len: int,
                               fill_after_n: int,
                               device: str,
                               out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    all_px, all_pz, all_yx, all_yz = [], [], [], []

    for df_raw in val_dfs_raw:
        src = df_raw['__source_file__'].iloc[0]
        file_date = extract_date_from_filename(src)
        # 與訓練一致的前處理
        df_proc = enrich_with_env(df_raw, env_settings, file_date, is_train=False)
        df_proc = add_target_lags(df_proc)

        # 標準化
        df_std_list, _, _ = standardize_by_stats([df_proc], feature_cols, mean, std)
        df_std = df_std_list[0]
        # 去重複欄位，保留第一個（standardize_by_stats 已把值寫回第一個欄位）
        df_std = df_std.loc[:, ~pd.Index(df_std.columns).duplicated(keep='first')]
        feat = df_std[feature_cols].copy()  # 僅保留訓練特徵，避免額外欄位

        dispX = pd.to_numeric(df_proc.get('Disp. X', np.nan), errors='coerce').values.astype('float32')
        dispZ = pd.to_numeric(df_proc.get('Disp. Z', np.nan), errors='coerce').values.astype('float32')

        n = len(df_proc)
        start_i = max(seq_len-1, fill_after_n-1)

        idxs, preds_x, preds_z, gts_x, gts_z = [], [], [], [], []
        with torch.no_grad():
            for i in range(start_i, n):
                def _lag(arr,k):
                    j=i-k
                    return None if j<0 or np.isnan(arr[j]) else float(arr[j])
                lag_vals = {
                    'DispX_lag1': _lag(dispX,1), 'DispX_lag2': _lag(dispX,2), 'DispX_lag3': _lag(dispX,3),
                    'DispZ_lag1': _lag(dispZ,1), 'DispZ_lag2': _lag(dispZ,2), 'DispZ_lag3': _lag(dispZ,3),
                }
                if any(v is None for v in lag_vals.values()):
                    continue
                # 只覆寫已存在於 feature_cols 的 lag
                for k, raw_v in lag_vals.items():
                    if k not in feat.columns:
                        continue
                    mu = mean[k]; sd = std[k] if float(std[k])!=0 else 1.0
                    feat.at[i, k] = (raw_v - mu) / sd

                s = i - seq_len + 1
                X_win = feat.iloc[s:i+1][feature_cols].values.astype('float32')
                xb = torch.from_numpy(X_win).unsqueeze(0).to(device)
                pred = model(xb).cpu().numpy().ravel()
                px, pz = float(pred[0]), float(pred[1])

                gtx = pd.to_numeric(df_proc.at[i,'Disp. X'], errors='coerce') if 'Disp. X' in df_proc.columns else np.nan
                gtz = pd.to_numeric(df_proc.at[i,'Disp. Z'], errors='coerce') if 'Disp. Z' in df_proc.columns else np.nan

                preds_x.append(px); preds_z.append(pz)
                gts_x.append(gtx);  gts_z.append(gtz)
                idxs.append(i)

                if np.isnan(dispX[i]): dispX[i]=px
                if np.isnan(dispZ[i]): dispZ[i]=pz

        out_df = pd.DataFrame({
            '__idx__': idxs,
            'y_true_X': gts_x, 'y_pred_LSTM_X': preds_x,
            'y_true_Z': gts_z, 'y_pred_LSTM_Z': preds_z,
        })
        base = src[:-4] if src.lower().endswith('.csv') else src
        out_path = os.path.join(out_dir, f"{base}.csv")
        ensure_parent_dir(out_path)
        out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"[VAL-PRED] save -> {out_path}")

        all_px.append(out_df['y_pred_LSTM_X'].values)
        all_pz.append(out_df['y_pred_LSTM_Z'].values)
        all_yx.append(out_df['y_true_X'].values)
        all_yz.append(out_df['y_true_Z'].values)

    if not all_px:
        print('[VAL-PRED] 無資料。'); return float('nan'), float('nan'), float('nan')
    px = np.concatenate(all_px); pz = np.concatenate(all_pz)
    yx = np.concatenate(all_yx); yz = np.concatenate(all_yz)
    rmse_x = safe_rmse(yx, px); rmse_z = safe_rmse(yz, pz)
    return rmse_x, rmse_z, np.nanmean([rmse_x, rmse_z])

# ================= 主流程 =================

def main():
    print('[STEP] 讀取訓練/測試資料…')
    train_dfs_raw = read_all_csv(TRAIN_DIR)
    test_dfs_raw  = read_all_csv(TEST_DIR)

    env_settings = load_env_settings(ENV_SETTINGS_FILE)

    # 檔名分流 8:2（以檔案為單位，避免資料洩漏）
    files = sorted({df['__source_file__'].iloc[0] for df in train_dfs_raw})
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=SEED)
    train_files, val_files = set(train_files), set(val_files)

    # 資料前處理（環境/rolling/lag）
    train_dfs, val_dfs = [], []
    for df in train_dfs_raw:
        fname = df['__source_file__'].iloc[0]
        fdate = extract_date_from_filename(fname)
        d = enrich_with_env(df, env_settings, fdate, is_train=True)
        d = add_target_lags(d)
        if fname in train_files: train_dfs.append(d)
        else:                    val_dfs.append(d)

    # 特徵欄位推斷（以全部訓練集合）
    train_all = pd.concat(train_dfs, ignore_index=True)
    for t in TARGET_COLS:
        if t not in train_all.columns:
            raise ValueError(f"訓練缺少欄位：{t}")
    feature_cols = infer_feature_cols(train_all)
    print(f"[INFO] 特徵欄位數：{len(feature_cols)}")

    # 標準化（以訓練集統計進行）
    train_dfs_std, mean, std = standardize_by_stats(train_dfs, feature_cols)
    val_dfs_std,   _,   _   = standardize_by_stats(val_dfs,   feature_cols, mean, std)

    # 建立序列
    X_tr_list, y_tr_list = [], []
    for df in train_dfs_std:
        x,y = build_sequences_for_train(df, feature_cols, SEQ_LEN)
        if len(x)>0: X_tr_list.append(x); y_tr_list.append(y)
    X_va_list, y_va_list = [], []
    for df in val_dfs_std:
        x,y = build_sequences_for_train(df, feature_cols, SEQ_LEN)
        if len(x)>0: X_va_list.append(x); y_va_list.append(y)

    if not X_tr_list or not X_va_list:
        raise RuntimeError('序列資料不足，請調整 SEQ_LEN 或確認資料長度。')

    X_tr = np.concatenate(X_tr_list); y_tr = np.concatenate(y_tr_list)
    X_va = np.concatenate(X_va_list); y_va = np.concatenate(y_va_list)
    print(f"[INFO] 訓練樣本：{len(X_tr):,}；驗證樣本：{len(X_va):,}；特徵：{len(feature_cols)}；seq_len：{SEQ_LEN}")

    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 建模
    model = LSTMRegressor(len(feature_cols), HIDDEN, LAYERS, 2, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val=float('inf'); best_state=None; wait=0
    print('[STEP] 訓練 LSTM（融合特徵）…')
    for ep in range(1, EPOCHS+1):
        model.train(); total=0.0
        for xb,yb in train_loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(); pred=model(xb); loss=crit(pred,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()*xb.size(0)
        tr_loss = total/len(train_loader.dataset)

        model.eval(); total=0.0; preds=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE)
                p=model(xb); loss=crit(p,yb); total+=loss.item()*xb.size(0)
                preds.append(p.cpu().numpy())
        va_loss = total/len(val_loader.dataset); preds=np.concatenate(preds)
        rmse_x = safe_rmse(y_va[:,0], preds[:,0]); rmse_z = safe_rmse(y_va[:,1], preds[:,1])
        rmse_avg = np.nanmean([rmse_x, rmse_z])
        print(f"[EPOCH {ep:02d}] train={tr_loss:.6f}  val={va_loss:.6f}  RMSE_X={rmse_x:.4f}  RMSE_Z={rmse_z:.4f}  AVG={rmse_avg:.4f}")

        sch.step(va_loss)
        if va_loss + 1e-12 < best_val:
            best_val = va_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=PATIENCE:
                print(f"[INFO] Early stopping at epoch {ep}."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最終驗證（teacher forcing）
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in val_loader:
            xb=xb.to(DEVICE); preds.append(model(xb).cpu().numpy())
    preds=np.concatenate(preds)
    rmse_x = safe_rmse(y_va[:,0], preds[:,0]); rmse_z = safe_rmse(y_va[:,1], preds[:,1])
    print(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={np.nanmean([rmse_x, rmse_z]):.6f}")

    # Rollout 驗證 & 輸出
    rx, rz, ravg = rollout_and_save_val_preds(
        model, val_dfs, env_settings, mean, std,
        feature_cols, SEQ_LEN, FILL_AFTER_N, DEVICE, VAL_PRED_DIR
    )
    print(f"[VAL-ROLLOUT] RMSE_X={rx:.6f}  RMSE_Z={rz:.6f}  RMSE_avg={ravg:.6f}")

    # ================= 測試集：自迴歸補值 =================
    print('[STEP] 讀取測試資料並遞推補值…')
    test_dfs_proc=[]
    for raw in test_dfs_raw:
        fname = raw['__source_file__'].iloc[0]
        fdate = extract_date_from_filename(fname)
        d = enrich_with_env(raw, env_settings, fdate, is_train=False)
        d = add_target_lags(d)
        test_dfs_proc.append(d)

    test_dfs_std, _, _ = standardize_by_stats(test_dfs_proc, feature_cols, mean, std)

    with torch.no_grad():
        for raw, df_std in zip(test_dfs_raw, test_dfs_std):
            src = raw['__source_file__'].iloc[0]
            df_out = raw.copy()
            for t in TARGET_COLS:
                if t not in df_out.columns: df_out[t]=np.nan

            n=len(df_out)
            if n<SEQ_LEN:
                out_path=os.path.join(OUT_DIR, src)
                ensure_parent_dir(out_path)
                cols=[c for c in df_out.columns if c not in TARGET_COLS and c!='__source_file__']; cols+=TARGET_COLS
                df_out[cols].to_csv(out_path, index=False, encoding='utf-8-sig')
                print(f"[WARN] {src} 長度<{SEQ_LEN}，未補。")
                continue

            # 去重複欄位，保留第一個
            df_std = df_std.loc[:, ~pd.Index(df_std.columns).duplicated(keep='first')]
            feat = df_std[feature_cols].copy()  # 只保留訓練特徵
            dispX = pd.to_numeric(df_out['Disp. X'], errors='coerce').values.astype('float32') if 'Disp. X' in df_out.columns else np.full(n, np.nan, dtype='float32')
            dispZ = pd.to_numeric(df_out['Disp. Z'], errors='coerce').values.astype('float32') if 'Disp. Z' in df_out.columns else np.full(n, np.nan, dtype='float32')

            start_i = max(SEQ_LEN-1, FILL_AFTER_N-1)
            filled=0
            for i in range(start_i, n):
                need_pred = is_blank(df_out.at[i,'Disp. X']) or is_blank(df_out.at[i,'Disp. Z'])

                def get_lag(arr, idx, k):
                    j=idx-k
                    if j<0 or np.isnan(arr[j]): return None
                    return float(arr[j])
                lag_vals = {
                    'DispX_lag1': get_lag(dispX,i,1), 'DispX_lag2': get_lag(dispX,i,2), 'DispX_lag3': get_lag(dispX,i,3),
                    'DispZ_lag1': get_lag(dispZ,i,1), 'DispZ_lag2': get_lag(dispZ,i,2), 'DispZ_lag3': get_lag(dispZ,i,3),
                }
                if any(v is None for v in lag_vals.values()):
                    continue
                # 只覆寫已存在欄位，避免新增維度
                for k, raw_v in lag_vals.items():
                    if k not in feat.columns:
                        continue
                    mu=mean[k]; sd=std[k] if float(std[k])!=0 else 1.0
                    feat.at[i, k]=(raw_v-mu)/sd

                s=i-SEQ_LEN+1
                X_win=feat.iloc[s:i+1][feature_cols].values.astype('float32')
                xb=torch.from_numpy(X_win).unsqueeze(0).to(DEVICE)

                if need_pred:
                    pred=model(xb).cpu().numpy().ravel()
                    px, pz = float(pred[0]), float(pred[1])
                    if is_blank(df_out.at[i,'Disp. X']): df_out.at[i,'Disp. X']=px
                    if is_blank(df_out.at[i,'Disp. Z']): df_out.at[i,'Disp. Z']=pz
                    filled+=1
                    if np.isnan(dispX[i]): dispX[i]=px
                    if np.isnan(dispZ[i]): dispZ[i]=pz
                else:
                    if np.isnan(dispX[i]) and not is_blank(df_out.at[i,'Disp. X']): dispX[i]=float(df_out.at[i,'Disp. X'])
                    if np.isnan(dispZ[i]) and not is_blank(df_out.at[i,'Disp. Z']): dispZ[i]=float(df_out.at[i,'Disp. Z'])

            cols=[c for c in df_out.columns if c not in TARGET_COLS and c!='__source_file__']; cols+=TARGET_COLS
            out_path=os.path.join(OUT_DIR, src)
            ensure_parent_dir(out_path)
            df_out[cols].to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"[SAVE] {out_path} -> 遞推回填 {filled} 筆（自第{FILL_AFTER_N}筆起之空白 Disp）")


if __name__ == '__main__':
    main()
