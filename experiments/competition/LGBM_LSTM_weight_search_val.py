# LGBM_LSTM_weight_search_val.py
# 在驗證逐列預測上掃描權重，找出 X、Z 各自的最佳 LSTM 權重 (w)，並存檔。
import os, glob, json, math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
DIR_LSTM = rf"{BASE_DIR}\val_pred\lstm"
DIR_LGBM = rf"{BASE_DIR}\val_pred\lgbm"
OUT_JSON = rf"{BASE_DIR}\ensemble_weights.json"

GRID_STEP = 0.05  # 權重步長；0.05 = 掃 21 個點

def safe_rmse(y, p):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum()==0: return float("nan")
    return math.sqrt(mean_squared_error(y[m], p[m]))

def read_csv(path):
    for enc in ["utf-8-sig", "utf-8", "cp950"]:
        try: return pd.read_csv(path, encoding=enc)
        except Exception: pass
    return pd.read_csv(path)

def load_pairs():
    files_lstm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LSTM, "*.csv"))}
    files_lgbm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LGBM, "*.csv"))}
    names = sorted(set(files_lstm) & set(files_lgbm))
    if not names:
        raise SystemExit("[ERR] 找不到同名驗證檔；請先跑 LSTM.py 與 LGBM.py 產生 val_pred 檔。")
    xs_true, xs_lstm, xs_lgbm = [], [], []
    zs_true, zs_lstm, zs_lgbm = [], [], []
    for name in names:
        dfL = read_csv(files_lstm[name])
        dfG = read_csv(files_lgbm[name])
        df = pd.merge(
            dfL[["__idx__","y_true_X","y_pred_LSTM_X","y_true_Z","y_pred_LSTM_Z"]],
            dfG[["__idx__","y_true_X","y_pred_LGBM_X","y_true_Z","y_pred_LGBM_Z"]],
            on="__idx__", how="inner", suffixes=("_lstm","_lgbm")
        )
        xs_true.append(pd.to_numeric(df["y_true_X_lstm"], errors="coerce").values)
        xs_lstm.append(pd.to_numeric(df["y_pred_LSTM_X"], errors="coerce").values)
        xs_lgbm.append(pd.to_numeric(df["y_pred_LGBM_X"], errors="coerce").values)
        zs_true.append(pd.to_numeric(df["y_true_Z_lstm"], errors="coerce").values)
        zs_lstm.append(pd.to_numeric(df["y_pred_LSTM_Z"], errors="coerce").values)
        zs_lgbm.append(pd.to_numeric(df["y_pred_LGBM_Z"], errors="coerce").values)
    return (np.concatenate(xs_true), np.concatenate(xs_lstm), np.concatenate(xs_lgbm),
            np.concatenate(zs_true), np.concatenate(zs_lstm), np.concatenate(zs_lgbm))

def scan_best_weight(y_true, p_lstm, p_lgbm, label="X"):
    best_w, best_rmse = None, float("inf")
    ws = np.round(np.arange(0.0, 1.0 + 1e-9, GRID_STEP), 3)
    for w in ws:
        p = w * p_lstm + (1.0 - w) * p_lgbm
        r = safe_rmse(y_true, p)
        if r < best_rmse:
            best_rmse, best_w = r, float(w)
    print(f"[BEST-{label}] w(LSTM)={best_w:.2f}  -> RMSE_{label}={best_rmse:.6f}")
    return best_w, best_rmse

def main():
    (x_true, x_lstm, x_lgbm, z_true, z_lstm, z_lgbm) = load_pairs()
    wX, rmseX = scan_best_weight(x_true, x_lstm, x_lgbm, label="X")
    wZ, rmseZ = scan_best_weight(z_true, z_lstm, z_lgbm, label="Z")
    rmse_avg = float(np.nanmean([rmseX, rmseZ]))
    print(f"[BEST-AVG] RMSE_avg={rmse_avg:.6f}")

    result = {
        "weight_X_LSTM": wX,     # X 欄 LSTM 權重（LGBM 權重=1-wX）
        "weight_Z_LSTM": wZ,     # Z 欄 LSTM 權重（LGBM 權重=1-wZ）
        "rmse_X": rmseX,
        "rmse_Z": rmseZ,
        "rmse_avg": rmse_avg,
        "grid_step": GRID_STEP,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 最佳權重與成績已寫入：{OUT_JSON}")

if __name__ == "__main__":
    main()
