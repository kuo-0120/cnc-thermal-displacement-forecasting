# LGBM_LSTM_weight_apply.py
# 讀 ensemble_weights.json，將 submission\lstm 與 submission\lgbm 做加權平均，輸出到 submission\ensemble_weighted
import os, glob, json, shutil
import numpy as np
import pandas as pd

BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
DIR_LSTM = rf"{BASE_DIR}\submission\lstm"
DIR_LGBM = rf"{BASE_DIR}\submission\lgbm"
DIR_OUT  = rf"{BASE_DIR}\submission\ensemble_weighted"
WEIGHT_JSON = rf"{BASE_DIR}\ensemble_weights.json"

TARGET_X, TARGET_Z = "Disp. X", "Disp. Z"

def read_csv_safe(p):
    for enc in ["utf-8-sig","utf-8","cp950"]:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: pass
    return pd.read_csv(p)

def main():
    if not os.path.exists(WEIGHT_JSON):
        raise SystemExit(f"[ERR] 找不到 {WEIGHT_JSON}，請先跑 LGBM_LSTM_weight_search_val.py")
    with open(WEIGHT_JSON, "r", encoding="utf-8") as f:
        w = json.load(f)
    wX = float(w.get("weight_X_LSTM", 1.0))
    wZ = float(w.get("weight_Z_LSTM", 0.0))
    os.makedirs(DIR_OUT, exist_ok=True)
    print(f"[USE-WEIGHTS] X: w_LSTM={wX:.2f}, w_LGBM={1-wX:.2f} | Z: w_LSTM={wZ:.2f}, w_LGBM={1-wZ:.2f}")

    files_lstm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LSTM, "*.csv"))}
    files_lgbm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LGBM, "*.csv"))}
    names = sorted(set(files_lstm) | set(files_lgbm))

    merged, copied = 0, 0
    for name in names:
        out_path = os.path.join(DIR_OUT, name)
        if name in files_lstm and name in files_lgbm:
            dfL = read_csv_safe(files_lstm[name])
            dfG = read_csv_safe(files_lgbm[name])
            n = min(len(dfL), len(dfG))
            base = dfL.iloc[:n].copy()
            other= dfG.iloc[:n].copy()
            # 確保欄位存在
            for c in [TARGET_X, TARGET_Z]:
                if c not in base.columns:  base[c]  = np.nan
                if c not in other.columns: other[c] = np.nan

            # 加權：X 欄、Z 欄各用自己的最佳權重
            base[TARGET_X] = wX * pd.to_numeric(base[TARGET_X], errors="coerce") + (1.0 - wX) * pd.to_numeric(other[TARGET_X], errors="coerce")
            base[TARGET_Z] = wZ * pd.to_numeric(base[TARGET_Z], errors="coerce") + (1.0 - wZ) * pd.to_numeric(other[TARGET_Z], errors="coerce")

            base.to_csv(out_path, index=False, encoding="utf-8-sig")
            merged += 1
        elif name in files_lstm:
            shutil.copy2(files_lstm[name], out_path); copied += 1
        else:
            shutil.copy2(files_lgbm[name], out_path); copied += 1

    print(f"[DONE] 加權完成：合併 {merged} 檔；直接複製 {copied} 檔。")
    print(f"[OUT] {DIR_OUT}")

if __name__ == "__main__":
    main()
