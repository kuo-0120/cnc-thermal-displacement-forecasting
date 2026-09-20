# -*- coding: utf-8 -*-
"""
LSTM_PRO_solid.py  (v1.0)
單檔完整版（穩定預設值）：
- 預測 Δ(殘差)：y(t)-y(t-1)；推論時累加回絕對值
- Weighted HuberLoss：對 X/Z 加權（預設 X:1.3, Z:0.7）以優先壓 X
- EMA 輸出平滑（rollout 專用；預設 0.1）
- 特徵工程：rolling mean(5) + 少量差分（PT_mean/TC_mean/X Motor/Z Motor）
- 特徵白名單瘦身：只留必要族群，降低噪聲與維度
- 以檔案為單位的 8:2 split；輸出 TF/ROLLOUT RMSE、驗證逐列預測與測試回填
使用：
1) 將 BASE_DIR 改成你的資料根目錄
2) 執行本檔，輸出會在 BASE_DIR 下的 val_pred/submission/metrics
"""

import os, re, glob, math, random, warnings, json
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==================== 路徑與固定參數 ====================
BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
TRAIN_DIR = rf"{BASE_DIR}\\train"
TEST_DIR  = rf"{BASE_DIR}\\test"
ENV_SETTINGS_FILE = os.getenv("CNC_ENV_FILE", os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "env.xlsx"))

VAL_TAG   = "lstm_pro_solid"
VAL_PRED_DIR = rf"{BASE_DIR}\\val_pred\\{VAL_TAG}"
OUT_DIR      = rf"{BASE_DIR}\\submission\\{VAL_TAG}"
METRIC_DIR   = rf"{BASE_DIR}\\metrics"
os.makedirs(VAL_PRED_DIR, exist_ok=True)
os.makedirs(OUT_DIR,      exist_ok=True)
os.makedirs(METRIC_DIR,   exist_ok=True)

TARGET_COLS = ["Disp. X", "Disp. Z"]
SEED = 42

# ==================== 超參數（穩定預設） ====================
SEQ_LEN    = 24
HIDDEN     = 256
LAYERS     = 2
DROPOUT    = 0.2
LR         = 1e-3
BATCH_SIZE = 256
EPOCHS     = 90
PATIENCE   = 10
FILL_AFTER_N = 100
SLIDING_WINDOW_SIZE = 5
DELTA_LOSS_DELTA = 0.5
EMA_ALPHA  = 0.1       # 0=關閉；建議 0.0~0.2
WEIGHT_X   = 1.3       # Weighted Huber：X 權重
WEIGHT_Z   = 0.7       # Weighted Huber：Z 權重

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== 小工具 ====================
def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def is_blank(v):
    return pd.isna(v) or (isinstance(v, str) and v.strip()=="")

def safe_to_minutes(x):
    if pd.api.types.is_numeric_dtype(type(x)):
        try: return float(x)
        except Exception: return np.nan
    try:
        td = pd.to_timedelta(str(x))
        return td.total_seconds()/60.0
    except Exception:
        m = re.match(r"^(\\d{1,2}):(\\d{1,2})(?::(\\d{1,2}))?$", str(x).strip())
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

# ==================== 讀檔 ====================
def read_all_csv(csv_dir: str) -> List[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths: raise FileNotFoundError(f"找不到 CSV：{csv_dir}")
    dfs=[]
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        if "Time" in df.columns:
            if df["Time"].dtype=="O":
                df["Time"] = df["Time"].map(safe_to_minutes)
            df = df.sort_values("Time").reset_index(drop=True)
        dfs.append(df)
    return dfs

# ==================== Env 設定（停用） ====================
def load_env_settings(excel_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(excel_path):
        print(f"[WARN] 找不到環境設定檔：{excel_path}，將略過環境特徵擴充。")
        return None
    try:
        env_settings = pd.read_excel(excel_path, header=[0,1])
    except Exception:
        print(f"[WARN] 無法讀取環境設定檔：{excel_path}，略過。")
        return None
    # 不使用；保留介面以後擴充
    return env_settings

# ==================== 特徵工程 ====================
TEMP_BASE_COLS = [f'PT{i:02d}' for i in range(1,14)] + [f'TC{i:02d}' for i in range(1,9)] + [
    'Spindle Motor', 'X Motor', 'Z Motor']

def enrich_with_env(df: pd.DataFrame, env_settings: Optional[pd.DataFrame], file_date: Optional[str], is_train: bool) -> pd.DataFrame:
    """僅做 rolling mean + 少量差分；環境表格暫不使用。"""
    d = df.copy()

    # 1) 感測器 rolling mean（僅 mean，窗口小以降噪）
    for col in TEMP_BASE_COLS:
        if col in d.columns:
            d[f'{col}_mean'] = d[col].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).mean()

    # 2) 匯總溫度均值
    pt_cols = [c for c in d.columns if re.fullmatch(r'PT\\d{2}', c)]
    tc_cols = [c for c in d.columns if re.fullmatch(r'TC\\d{2}', c)]
    if pt_cols: d['PT_mean'] = d[pt_cols].mean(axis=1)
    if tc_cols: d['TC_mean'] = d[tc_cols].mean(axis=1)

    # 3) 少量差分（強化立即性梯度訊號）
    deriv_targets = [c for c in ['PT_mean','TC_mean','X Motor','Z Motor'] if c in d.columns]
    for col in deriv_targets:
        d[f'd_{col}'] = d[col].diff().fillna(0)
        d[f'd_{col}_mean'] = d[f'd_{col}'].rolling(window=SLIDING_WINDOW_SIZE, min_periods=1).mean()

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
        ser = base[c] if c in base.columns else None
        if ser is None: continue
        if pd.api.types.is_numeric_dtype(ser):
            feats.append(c)
        else:
            tmp = pd.to_numeric(ser, errors='coerce')
            if tmp.notna().sum()>0: feats.append(c)
    feats = list(dict.fromkeys(feats))

    # --- 白名單瘦身 ---
    keep=[]
    for c in feats:
        if c.startswith(('DispX_lag','DispZ_lag')):
            try:
                n = int(re.findall(r'lag(\\d+)', c)[0])
            except Exception:
                n = 0
            if 1 <= n <= 3: keep.append(c); continue
        if c.startswith(('X Motor','Z Motor','Spindle Motor')):
            if ('_mean' in c) or (('_' not in c) or c.endswith(' Motor')):
                keep.append(c); continue
        if c in ('PT_mean','TC_mean') or c.endswith('_mean'):
            if c.startswith(('PT','TC','d_PT','d_TC','d_X Motor','d_Z Motor')):
                keep.append(c); continue
    if keep:
        feats = [x for x in feats if x in keep]
    return feats

def standardize_by_stats(df_list: List[pd.DataFrame], feature_cols: List[str], mean=None, std=None):
    concat = pd.concat(df_list, ignore_index=True)
    for c in feature_cols:
        if c not in concat.columns:
            concat[c] = np.nan
        concat[c] = pd.to_numeric(concat[c], errors='coerce')
    if mean is None or std is None:
        stats_df = concat[feature_cols]
        mean = stats_df.mean().astype('float32').fillna(0.0)
        std  = stats_df.std(ddof=0).replace(0,1.0).astype('float32').fillna(1.0)

    out=[]
    for df in df_list:
        d2 = df.copy()
        for c in feature_cols:
            if c not in d2.columns:
                d2[c] = np.nan
            ser = pd.to_numeric(d2[c], errors='coerce').fillna(mean[c])
            sd = std[c] if float(std[c])!=0 else 1.0
            ser = (ser - mean[c]) / sd
            ser = ser.replace([np.inf,-np.inf], np.nan).fillna(0.0)
            d2[c] = ser
        out.append(d2)
    return out, mean, std

# ==================== 資料：Δ(殘差) ====================
def build_sequences_delta(df: pd.DataFrame, feature_cols: List[str], seq_len: int):
    if not all(t in df.columns for t in TARGET_COLS):
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2)), np.empty((0,2)), np.empty((0,2))
    yy = df[TARGET_COLS].apply(pd.to_numeric, errors='coerce').values.astype('float32')
    X  = df[feature_cols].values.astype('float32')
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2)), np.empty((0,2)), np.empty((0,2))
    xs, y_delta, y_prev, y_abs = [], [], [], []
    for t in range(seq_len-1, n):
        if not np.isfinite(yy[t]).all(): continue
        if not np.isfinite(yy[t-1]).all(): continue
        xs.append(X[t-seq_len+1:t+1])
        y_abs.append(yy[t])
        y_prev.append(yy[t-1])
        y_delta.append(yy[t] - yy[t-1])
    if not xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2)), np.empty((0,2)), np.empty((0,2))
    return np.stack(xs), np.stack(y_delta), np.stack(y_prev), np.stack(y_abs)

# ==================== 模型與 Loss ====================
class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.head(out[:,-1,:])  # 預測 Δy

class HuberLossWeighted(nn.Module):
    def __init__(self, delta=0.5, weight=(1.0,1.0)):
        super().__init__()
        self.delta = float(delta)
        self.weight = torch.tensor(weight, dtype=torch.float32)
    def forward(self, input, target):
        diff = input - target
        abs_diff = torch.abs(diff)
        mask = abs_diff <= self.delta
        loss = torch.where(mask, 0.5*diff*diff, self.delta*(abs_diff - 0.5*self.delta))
        w = self.weight.to(input.device).view(1,-1)
        loss = loss * w
        return loss.mean()

# ==================== 驗證：Rollout（Δ→絕對值） ====================
def rollout_and_save_val_preds(model: nn.Module,
                               val_dfs_raw: List[pd.DataFrame],
                               mean, std,
                               feature_cols: List[str],
                               seq_len: int,
                               fill_after_n: int,
                               device: str,
                               out_dir: str,
                               ema_alpha: float = 0.0):
    os.makedirs(out_dir, exist_ok=True)
    all_px, all_pz, all_yx, all_yz = [], [], [], []

    for df_raw in val_dfs_raw:
        src = df_raw['__source_file__'].iloc[0]
        df_proc = enrich_with_env(df_raw, None, None, is_train=False)
        df_proc = add_target_lags(df_proc)

        df_std_list, _, _ = standardize_by_stats([df_proc], feature_cols, mean, std)
        df_std = df_std_list[0]
        df_std = df_std.loc[:, ~pd.Index(df_std.columns).duplicated(keep='first')]
        feat = df_std[feature_cols].copy()

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
                # 以標準化尺度覆寫 lag 特徵
                for k, raw_v in lag_vals.items():
                    if k not in feat.columns: continue
                    mu = mean[k]; sd = std[k] if float(std[k])!=0 else 1.0
                    feat.at[i, k] = (raw_v - mu) / sd

                s = i - seq_len + 1
                X_win = feat.iloc[s:i+1][feature_cols].values.astype('float32')
                xb = torch.from_numpy(X_win).unsqueeze(0).to(device)
                delta = model(xb).cpu().numpy().ravel()  # Δ
                px = float(dispX[i-1] + delta[0])
                pz = float(dispZ[i-1] + delta[1])

                # EMA 平滑（僅影響 rollout）
                if ema_alpha and i-1 >= 0 and np.isfinite(dispX[i-1]) and np.isfinite(dispZ[i-1]):
                    px = ema_alpha*px + (1-ema_alpha)*dispX[i-1]
                    pz = ema_alpha*pz + (1-ema_alpha)*dispZ[i-1]

                gtx = pd.to_numeric(df_proc.at[i,'Disp. X'], errors='coerce') if 'Disp. X' in df_proc.columns else np.nan
                gtz = pd.to_numeric(df_proc.at[i,'Disp. Z'], errors='coerce') if 'Disp. Z' in df_proc.columns else np.nan

                preds_x.append(px); preds_z.append(pz)
                gts_x.append(gtx);  gts_z.append(gtz)
                idxs.append(i)

                # 遞推所需：將空白點回填為預測，以供後續 lag 使用
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

# ==================== 主流程 ====================
def main():
    print('[STEP] 讀取訓練/測試資料…')
    train_dfs_raw = read_all_csv(TRAIN_DIR)
    test_dfs_raw  = read_all_csv(TEST_DIR)

    env_settings = load_env_settings(ENV_SETTINGS_FILE)  # 目前不使用

    # 以檔案為單位分流（8:2）
    files = sorted({df['__source_file__'].iloc[0] for df in train_dfs_raw})
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=SEED)
    train_files, val_files = set(train_files), set(val_files)

    # 前處理（rolling + 差分 + lag）
    train_dfs, val_dfs = [], []
    for df in train_dfs_raw:
        fname = df['__source_file__'].iloc[0]
        d = enrich_with_env(df, env_settings, None, is_train=True)
        d = add_target_lags(d)
        if fname in train_files: train_dfs.append(d)
        else:                    val_dfs.append(d)

    # 特徵欄位
    train_all = pd.concat(train_dfs, ignore_index=True)
    for t in TARGET_COLS:
        if t not in train_all.columns:
            raise ValueError(f"訓練缺少欄位：{t}")
    feature_cols = infer_feature_cols(train_all)
    print(f"[INFO] 特徵欄位數：{len(feature_cols)}")
    print(f"[INFO] HYPER: SEQ_LEN={SEQ_LEN}  HIDDEN={HIDDEN}  EMA={EMA_ALPHA}  wX={WEIGHT_X} wZ={WEIGHT_Z}")

    # 標準化（以訓練集統計）
    train_dfs_std, mean, std = standardize_by_stats(train_dfs, feature_cols)
    val_dfs_std,   _,   _   = standardize_by_stats(val_dfs,   feature_cols, mean, std)

    # Δ 資料集
    X_tr_list, ytr_d_list, ytr_p_list, ytr_a_list = [], [], [], []
    for df in train_dfs_std:
        x,yd,yp,ya = build_sequences_delta(df, feature_cols, SEQ_LEN)
        if len(x)>0:
            X_tr_list.append(x); ytr_d_list.append(yd); ytr_p_list.append(yp); ytr_a_list.append(ya)
    X_va_list, yva_d_list, yva_p_list, yva_a_list = [], [], [], []
    for df in val_dfs_std:
        x,yd,yp,ya = build_sequences_delta(df, feature_cols, SEQ_LEN)
        if len(x)>0:
            X_va_list.append(x); yva_d_list.append(yd); yva_p_list.append(yp); yva_a_list.append(ya)

    if not X_tr_list or not X_va_list:
        raise RuntimeError('序列資料不足，請調整 SEQ_LEN 或確認資料長度。')

    X_tr = np.concatenate(X_tr_list); y_tr_d = np.concatenate(ytr_d_list); y_tr_p = np.concatenate(ytr_p_list); y_tr_a = np.concatenate(ytr_a_list)
    X_va = np.concatenate(X_va_list); y_va_d = np.concatenate(yva_d_list); y_va_p = np.concatenate(yva_p_list); y_va_a = np.concatenate(yva_a_list)
    print(f"[INFO] 訓練樣本：{len(X_tr):,}；驗證樣本：{len(X_va):,}；特徵：{len(feature_cols)}；seq_len：{SEQ_LEN}")

    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_d), torch.from_numpy(y_tr_p), torch.from_numpy(y_tr_a)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va_d), torch.from_numpy(y_va_p), torch.from_numpy(y_va_a)),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 建模
    model = LSTMRegressor(len(feature_cols), HIDDEN, LAYERS, 2, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = HuberLossWeighted(delta=DELTA_LOSS_DELTA, weight=(WEIGHT_X, WEIGHT_Z))
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val=float('inf'); best_state=None; wait=0
    print('[STEP] 訓練 LSTM（Δ+加權Huber）…')
    for ep in range(1, EPOCHS+1):
        model.train(); total=0.0
        for xb,y_delta,_,_ in train_loader:
            xb=xb.to(DEVICE); y_delta=y_delta.to(DEVICE)
            opt.zero_grad(); pred_delta=model(xb); loss=crit(pred_delta,y_delta); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()*xb.size(0)
        tr_loss = total/len(train_loader.dataset)

        model.eval(); total=0.0; preds_abs=[]
        with torch.no_grad():
            for xb,y_delta,y_prev,_ in val_loader:
                xb=xb.to(DEVICE); y_delta=y_delta.to(DEVICE); y_prev=y_prev.to(DEVICE)
                pred_delta=model(xb)
                loss=crit(pred_delta,y_delta); total+=loss.item()*xb.size(0)
                pred_abs = pred_delta + y_prev  # teacher-forcing 還原絕對量
                preds_abs.append(pred_abs.cpu().numpy())
        va_loss = total/len(val_loader.dataset)
        preds_abs=np.concatenate(preds_abs)
        rmse_x = safe_rmse(y_va_a[:,0], preds_abs[:,0]); rmse_z = safe_rmse(y_va_a[:,1], preds_abs[:,1])
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

    # 最終 teacher-forcing RMSE（絕對量）
    model.eval(); preds_abs=[]
    with torch.no_grad():
        for xb,_,y_prev,_ in val_loader:
            xb=xb.to(DEVICE); y_prev=y_prev.to(DEVICE)
            pred_delta=model(xb); pred_abs = pred_delta + y_prev
            preds_abs.append(pred_abs.cpu().numpy())
    preds_abs=np.concatenate(preds_abs)
    rmse_x = safe_rmse(y_va_a[:,0], preds_abs[:,0]); rmse_z = safe_rmse(y_va_a[:,1], preds_abs[:,1])
    tf_avg = float(np.nanmean([rmse_x, rmse_z]))
    print(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={tf_avg:.6f}")

    # Rollout 驗證
    rx, rz, ravg = rollout_and_save_val_preds(
        model, val_dfs, mean, std,
        feature_cols, SEQ_LEN, FILL_AFTER_N, DEVICE, VAL_PRED_DIR,
        ema_alpha=EMA_ALPHA
    )
    print(f"[VAL-ROLLOUT] RMSE_X={rx:.6f}  RMSE_Z={rz:.6f}  RMSE_avg={ravg:.6f}")

    # 存 metrics
    metric_path = rf"{METRIC_DIR}\\{VAL_TAG}_S{SEQ_LEN}_H{HIDDEN}_E{str(EMA_ALPHA).replace('.','p')}.json"
    metrics = {
        "seq_len": SEQ_LEN, "hidden": HIDDEN, "ema": EMA_ALPHA, "wX": WEIGHT_X, "wZ": WEIGHT_Z,
        "tf_rmse_x": rmse_x, "tf_rmse_z": rmse_z, "tf_rmse_avg": tf_avg,
        "roll_rmse_x": float(rx), "roll_rmse_z": float(rz), "roll_rmse_avg": float(ravg),
        "feature_count": len(feature_cols)
    }
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[METRIC] saved -> {metric_path}")

    # ==================== 測試集：自迴歸補值 ====================
    print('[STEP] 讀取測試資料並遞推補值…')
    test_dfs_proc=[]
    for raw in test_dfs_raw:
        d = enrich_with_env(raw, None, None, is_train=False)
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

            df_std = df_std.loc[:, ~pd.Index(df_std.columns).duplicated(keep='first')]
            feat = df_std[feature_cols].copy()
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
                for k, raw_v in lag_vals.items():
                    if k not in feat.columns: continue
                    mu=mean[k]; sd=std[k] if float(std[k])!=0 else 1.0
                    feat.at[i, k]=(raw_v-mu)/sd

                s=i-SEQ_LEN+1
                X_win=feat.iloc[s:i+1][feature_cols].values.astype('float32')
                xb=torch.from_numpy(X_win).unsqueeze(0).to(DEVICE)

                if need_pred:
                    delta=model(xb).cpu().numpy().ravel()
                    px, pz = float(dispX[i-1] + delta[0]), float(dispZ[i-1] + delta[1])

                    if EMA_ALPHA and i-1 >= 0 and np.isfinite(dispX[i-1]) and np.isfinite(dispZ[i-1]):
                        px = EMA_ALPHA*px + (1-EMA_ALPHA)*dispX[i-1]
                        pz = EMA_ALPHA*pz + (1-EMA_ALPHA)*dispZ[i-1]

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
