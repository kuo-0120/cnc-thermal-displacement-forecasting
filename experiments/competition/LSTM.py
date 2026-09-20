# LSTM.py — 單模型：訓練/驗證 + 驗證逐列輸出 + 測試submission（存到 submission\lstm\）
import os, glob, math, random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ========= 路徑 =========
BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
TRAIN_DIR = rf"{BASE_DIR}\train"
TEST_DIR  = rf"{BASE_DIR}\test"
VAL_PRED_DIR = rf"{BASE_DIR}\val_pred\lstm"       # 驗證逐列預測
OUT_DIR      = rf"{BASE_DIR}\submission\lstm"     # LSTM 的 submission
os.makedirs(VAL_PRED_DIR, exist_ok=True)
os.makedirs(OUT_DIR,      exist_ok=True)
# ========================

TARGET_COLS = ["Disp. X", "Disp. Z"]
SEED = 42

# 可調參數（與你目前設定一致）
SEQ_LEN    = 16
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.2
LR         = 1e-3
BATCH_SIZE = 256
EPOCHS     = 50
PATIENCE   = 8
FILL_AFTER_N = 100  # 1-based

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def is_blank(v):
    return pd.isna(v) or (isinstance(v, str) and v.strip()=="")

def safe_rmse(y_true_col, y_pred_col):
    y_true_col = np.asarray(y_true_col, dtype="float64")
    y_pred_col = np.asarray(y_pred_col, dtype="float64")
    mask = np.isfinite(y_true_col) & np.isfinite(y_pred_col)
    if mask.sum()==0: return float("nan")
    return math.sqrt(mean_squared_error(y_true_col[mask], y_pred_col[mask]))

def read_all_csv(csv_dir: str):
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"找不到 CSV：{csv_dir}")
    dfs=[]
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        if "Time" in df.columns:
            df = df.sort_values("Time").reset_index(drop=True)
        dfs.append(df)
    return dfs

def add_target_lags(df: pd.DataFrame):
    d = df.copy()
    # X lags
    if "Disp. X" in d.columns:
        d["DispX_lag1"] = d["Disp. X"].shift(1)
        d["DispX_lag2"] = d["Disp. X"].shift(2)
        d["DispX_lag3"] = d["Disp. X"].shift(3)
    else:
        d["DispX_lag1"]=np.nan; d["DispX_lag2"]=np.nan; d["DispX_lag3"]=np.nan
    # Z lags
    if "Disp. Z" in d.columns:
        d["DispZ_lag1"] = d["Disp. Z"].shift(1)
        d["DispZ_lag2"] = d["Disp. Z"].shift(2)
        d["DispZ_lag3"] = d["Disp. Z"].shift(3)
    else:
        d["DispZ_lag1"]=np.nan; d["DispZ_lag2"]=np.nan; d["DispZ_lag3"]=np.nan
    return d

def infer_feature_cols(train_all: pd.DataFrame):
    base = add_target_lags(train_all)
    cand = [c for c in base.columns if c not in TARGET_COLS and c != "__source_file__"]
    feats=[]
    for c in cand:
        if pd.api.types.is_numeric_dtype(base[c]): feats.append(c)
        else:
            tmp = pd.to_numeric(base[c], errors="coerce")
            if tmp.notna().sum()>0: feats.append(c)
    return feats

def standardize_by_stats(df_list, feature_cols, mean=None, std=None):
    concat = pd.concat(df_list, ignore_index=True)
    for c in feature_cols:
        if c not in concat.columns: concat[c]=np.nan
        concat[c] = pd.to_numeric(concat[c], errors="coerce")

    if mean is None or std is None:
        mean = concat[feature_cols].mean().astype("float32").fillna(0.0)
        std  = concat[feature_cols].std(ddof=0).replace(0,1.0).astype("float32").fillna(1.0)

    out=[]
    for df in df_list:
        d2 = df.copy()
        for c in feature_cols:
            if c not in d2.columns: d2[c]=np.nan
            d2[c] = pd.to_numeric(d2[c], errors="coerce")
        d2[feature_cols] = d2[feature_cols].fillna(mean)
        d2[feature_cols] = (d2[feature_cols]-mean)/std
        d2[feature_cols] = d2[feature_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out.append(d2)
    return out, mean, std

def build_sequences_for_train(df: pd.DataFrame, feature_cols, seq_len: int):
    if not all(t in df.columns for t in TARGET_COLS):
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    yy = df[TARGET_COLS].apply(pd.to_numeric, errors="coerce").values.astype("float32")
    X  = df[feature_cols].values.astype("float32")
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    xs, ys = [], []
    for t in range(seq_len-1, n):
        if not np.isfinite(yy[t]).all(): continue
        xs.append(X[t-seq_len+1:t+1]); ys.append(yy[t])
    if not xs:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,2))
    return np.stack(xs), np.stack(ys)

class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, out_dim=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.head(out[:,-1,:])

def rollout_and_save_val_preds(model, val_dfs, mean, std, feature_cols, seq_len, fill_after_n, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_px, all_pz, all_yx, all_yz = [], [], [], []
    for df_raw in val_dfs:
        src = df_raw["__source_file__"].iloc[0]
        df_raw2 = add_target_lags(df_raw)
        df_std_list, _, _ = standardize_by_stats([df_raw2], feature_cols, mean, std)
        df_std = df_std_list[0]
        feat = df_std[feature_cols].copy()

        dispX = pd.to_numeric(df_raw2["Disp. X"], errors="coerce").values.astype("float32")
        dispZ = pd.to_numeric(df_raw2["Disp. Z"], errors="coerce").values.astype("float32")
        n = len(df_raw2)
        start_i = max(seq_len-1, fill_after_n-1)

        idxs, preds_x, preds_z, gts_x, gts_z = [], [], [], [], []
        with torch.no_grad():
            for i in range(start_i, n):
                def _lag(arr,k):
                    j=i-k
                    return None if j<0 or np.isnan(arr[j]) else float(arr[j])
                lag_vals = {
                    "DispX_lag1": _lag(dispX,1), "DispX_lag2": _lag(dispX,2), "DispX_lag3": _lag(dispX,3),
                    "DispZ_lag1": _lag(dispZ,1), "DispZ_lag2": _lag(dispZ,2), "DispZ_lag3": _lag(dispZ,3),
                }
                if any(v is None for v in lag_vals.values()):
                    continue
                for k, raw_v in lag_vals.items():
                    mu = mean[k]; sd = std[k] if float(std[k])!=0 else 1.0
                    feat.at[i, k] = (raw_v - mu) / sd
                s = i - seq_len + 1
                X_win = feat.iloc[s:i+1].values.astype("float32")
                xb = torch.from_numpy(X_win).unsqueeze(0).to(device)
                pred = model(xb).cpu().numpy().ravel()
                px, pz = float(pred[0]), float(pred[1])

                gtx = pd.to_numeric(df_raw2.at[i,"Disp. X"], errors="coerce")
                gtz = pd.to_numeric(df_raw2.at[i,"Disp. Z"], errors="coerce")
                preds_x.append(px); preds_z.append(pz)
                gts_x.append(gtx);  gts_z.append(gtz)
                idxs.append(i)
                if np.isnan(dispX[i]): dispX[i]=px
                if np.isnan(dispZ[i]): dispZ[i]=pz

        out_df = pd.DataFrame({
            "__idx__": idxs,
            "y_true_X": gts_x, "y_pred_LSTM_X": preds_x,
            "y_true_Z": gts_z, "y_pred_LSTM_Z": preds_z,
        })
        base = src[:-4] if src.lower().endswith(".csv") else src
        out_path = os.path.join(out_dir, f"{base}.csv")
        ensure_parent_dir(out_path)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[VAL-PRED-LSTM] save -> {out_path}")

        all_px.append(out_df["y_pred_LSTM_X"].values)
        all_pz.append(out_df["y_pred_LSTM_Z"].values)
        all_yx.append(out_df["y_true_X"].values)
        all_yz.append(out_df["y_true_Z"].values)

    if not all_px:
        print("[VAL-PRED-LSTM] 無資料。"); return float("nan"), float("nan"), float("nan")
    px = np.concatenate(all_px); pz = np.concatenate(all_pz)
    yx = np.concatenate(all_yx); yz = np.concatenate(all_yz)
    rmse_x = safe_rmse(yx, px); rmse_z = safe_rmse(yz, pz)
    return rmse_x, rmse_z, np.nanmean([rmse_x, rmse_z])

def main():
    print("[STEP] 讀取訓練資料…")
    train_dfs_all = read_all_csv(TRAIN_DIR)
    train_all = pd.concat(train_dfs_all, ignore_index=True)
    for t in TARGET_COLS:
        if t not in train_all.columns:
            raise ValueError(f"訓練缺少欄位：{t}")

    feature_cols = infer_feature_cols(train_all)
    print(f"[INFO] 特徵欄位數：{len(feature_cols)}")

    files = sorted(train_all["__source_file__"].unique().tolist())
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=SEED)
    train_files, val_files = set(train_files), set(val_files)

    train_dfs = [add_target_lags(df) for df in train_dfs_all if df["__source_file__"].iloc[0] in train_files]
    val_dfs   = [add_target_lags(df) for df in train_dfs_all if df["__source_file__"].iloc[0] in val_files]

    train_dfs_std, mean, std = standardize_by_stats(train_dfs, feature_cols)
    val_dfs_std, _, _        = standardize_by_stats(val_dfs,   feature_cols, mean, std)

    X_tr_list, y_tr_list = [], []
    for df in train_dfs_std:
        x,y = build_sequences_for_train(df, feature_cols, SEQ_LEN)
        if len(x)>0: X_tr_list.append(x); y_tr_list.append(y)
    X_va_list, y_va_list = [], []
    for df in val_dfs_std:
        x,y = build_sequences_for_train(df, feature_cols, SEQ_LEN)
        if len(x)>0: X_va_list.append(x); y_va_list.append(y)

    if not X_tr_list or not X_va_list:
        raise RuntimeError("序列資料不足，調整 SEQ_LEN 或確認資料長度。")

    X_tr = np.concatenate(X_tr_list); y_tr = np.concatenate(y_tr_list)
    X_va = np.concatenate(X_va_list); y_va = np.concatenate(y_va_list)
    print(f"[INFO] 訓練樣本：{len(X_tr):,}；驗證樣本：{len(X_va):,}；特徵：{len(feature_cols)}；seq_len ：{SEQ_LEN}")

    train_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMRegressor(len(feature_cols), HIDDEN, LAYERS, 2, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val=float("inf"); best_state=None; wait=0
    print("[STEP] 訓練 LSTM（含 lag 特徵）…")
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

    if best_state is not None: model.load_state_dict(best_state)

    # teacher forcing 最終驗證
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in val_loader:
            xb=xb.to(DEVICE); preds.append(model(xb).cpu().numpy())
    preds=np.concatenate(preds)
    rmse_x = safe_rmse(y_va[:,0], preds[:,0]); rmse_z = safe_rmse(y_va[:,1], preds[:,1])
    print(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={np.nanmean([rmse_x, rmse_z]):.6f}")
    print("[DONE] 訓練完成。裝置：", DEVICE)

    # ===== 新增：驗證 ROLLOUT 並存逐列預測 =====
    rx, rz, ravg = rollout_and_save_val_preds(model, val_dfs, mean, std, feature_cols, SEQ_LEN, FILL_AFTER_N, DEVICE, VAL_PRED_DIR)
    print(f"[VAL-ROLLOUT] RMSE_X={rx:.6f}  RMSE_Z={rz:.6f}  RMSE_avg={ravg:.6f}")

    # ===== 測試遞推補值（寫到 submission\lstm\）=====
    print("[STEP] 讀取測試資料並遞推補值…")
    test_dfs_raw = read_all_csv(TEST_DIR)
    test_dfs_with_lag = [add_target_lags(df) for df in test_dfs_raw]
    test_dfs_std, _, _ = standardize_by_stats(test_dfs_with_lag, feature_cols, mean, std)
    feature_cols_final = feature_cols

    with torch.no_grad():
        for raw, df_std in zip(test_dfs_raw, test_dfs_std):
            src = raw["__source_file__"].iloc[0]
            df_out = raw.copy()
            for t in TARGET_COLS:
                if t not in df_out.columns: df_out[t]=np.nan

            n=len(df_out)
            if n<SEQ_LEN:
                out_path=os.path.join(OUT_DIR, src)
                ensure_parent_dir(out_path)
                cols=[c for c in df_out.columns if c not in TARGET_COLS and c!="__source_file__"]; cols+=TARGET_COLS
                df_out[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"[WARN] {src} 長度<{SEQ_LEN}，未補。")
                continue

            feat = df_std[feature_cols_final].copy()
            dispX = pd.to_numeric(df_out["Disp. X"], errors="coerce").values.astype("float32")
            dispZ = pd.to_numeric(df_out["Disp. Z"], errors="coerce").values.astype("float32")

            start_i = max(SEQ_LEN-1, FILL_AFTER_N-1)
            filled=0
            for i in range(start_i, n):
                need_pred = is_blank(df_out.at[i,"Disp. X"]) or is_blank(df_out.at[i,"Disp. Z"])

                def get_lag(arr, idx, k):
                    j=idx-k
                    if j<0 or np.isnan(arr[j]): return None
                    return float(arr[j])
                lag_vals = {
                    "DispX_lag1": get_lag(dispX,i,1), "DispX_lag2": get_lag(dispX,i,2), "DispX_lag3": get_lag(dispX,i,3),
                    "DispZ_lag1": get_lag(dispZ,i,1), "DispZ_lag2": get_lag(dispZ,i,2), "DispZ_lag3": get_lag(dispZ,i,3),
                }
                if any(v is None for v in lag_vals.values()):
                    continue
                for k, raw_v in lag_vals.items():
                    mu=mean[k]; sd=std[k] if std[k]!=0 else 1.0
                    feat.at[i, k]=(raw_v-mu)/sd

                s=i-SEQ_LEN+1
                X_win=feat.iloc[s:i+1].values.astype("float32")
                xb=torch.from_numpy(X_win).unsqueeze(0).to(DEVICE)

                if need_pred:
                    pred=model(xb).cpu().numpy().ravel()
                    px, pz = float(pred[0]), float(pred[1])
                    if is_blank(df_out.at[i,"Disp. X"]): df_out.at[i,"Disp. X"]=px
                    if is_blank(df_out.at[i,"Disp. Z"]): df_out.at[i,"Disp. Z"]=pz
                    filled+=1
                    if np.isnan(dispX[i]): dispX[i]=px
                    if np.isnan(dispZ[i]): dispZ[i]=pz
                else:
                    if np.isnan(dispX[i]) and not is_blank(df_out.at[i,"Disp. X"]): dispX[i]=float(df_out.at[i,"Disp. X"])
                    if np.isnan(dispZ[i]) and not is_blank(df_out.at[i,"Disp. Z"]): dispZ[i]=float(df_out.at[i,"Disp. Z"])

            cols=[c for c in df_out.columns if c not in TARGET_COLS and c!="__source_file__"]; cols+=TARGET_COLS
            out_path=os.path.join(OUT_DIR, src)
            ensure_parent_dir(out_path)
            df_out[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[SAVE] {out_path} -> 遞推回填 {filled} 筆（自第{FILL_AFTER_N}筆起之空白 Disp）")

if __name__ == "__main__":
    main()
