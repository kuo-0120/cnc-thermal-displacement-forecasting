# LGBM_X+LSTM_Z_val.py — 驗證集成評分（X←LSTM、Z←LGBM），印 RMSE_X / RMSE_Z / RMSE_avg
import os, glob, math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
DIR_LSTM = rf"{BASE_DIR}\val_pred\lstm"
DIR_LGBM = rf"{BASE_DIR}\val_pred\lgbm"

def safe_rmse(y, p):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum()==0: return float("nan")
    return math.sqrt(mean_squared_error(y[m], p[m]))

def read_csv(path):
    for enc in ["utf-8-sig", "utf-8", "cp950"]:
        try: return pd.read_csv(path, encoding=enc)
        except Exception: continue
    return pd.read_csv(path)

def main():
    files_lstm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LSTM, "*.csv"))}
    files_lgbm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LGBM, "*.csv"))}
    names = sorted(set(files_lstm.keys()) & set(files_lgbm.keys()))
    if not names:
        print("[ERR] 找不到同名驗證檔；請先跑 LSTM.py 與 LGBM.py 讓 val_pred 生成檔案。")
        return

    all_true_x, all_pred_x = [], []
    all_true_z, all_pred_z = [], []

    for name in names:
        df_lstm = read_csv(files_lstm[name])
        df_lgbm = read_csv(files_lgbm[name])

        df = pd.merge(
            df_lstm[["__idx__","y_true_X","y_pred_LSTM_X","y_true_Z","y_pred_LSTM_Z"]],
            df_lgbm[["__idx__","y_true_X","y_pred_LGBM_X","y_true_Z","y_pred_LGBM_Z"]],
            on="__idx__", how="inner", suffixes=("_lstm","_lgbm")
        )

        # 混合規則：X ← LSTM、Z ← LGBM（和你目前策略一致；要改成加權也容易）
        y_true_x = pd.to_numeric(df["y_true_X_lstm"], errors="coerce").values
        y_pred_x = pd.to_numeric(df["y_pred_LSTM_X"], errors="coerce").values
        y_true_z = pd.to_numeric(df["y_true_Z_lstm"], errors="coerce").values
        y_pred_z = pd.to_numeric(df["y_pred_LGBM_Z"], errors="coerce").values

        all_true_x.append(y_true_x); all_pred_x.append(y_pred_x)
        all_true_z.append(y_true_z); all_pred_z.append(y_pred_z)

    all_true_x = np.concatenate(all_true_x); all_pred_x = np.concatenate(all_pred_x)
    all_true_z = np.concatenate(all_true_z); all_pred_z = np.concatenate(all_pred_z)

    rmse_x = safe_rmse(all_true_x, all_pred_x)
    rmse_z = safe_rmse(all_true_z, all_pred_z)
    rmse_avg = np.nanmean([rmse_x, rmse_z])

    print(f"[VAL-ENSEMBLE (X=LSTM, Z=LGBM)] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={rmse_avg:.6f}")

if __name__ == "__main__":
    main()
