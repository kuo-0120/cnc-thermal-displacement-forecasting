# LGBM_X+LSTM_Z.py — 混合輸出（規則：X←LGBM、Z←LSTM），輸出到 submission\ensemble\
import os, glob, shutil
import pandas as pd
import numpy as np

BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
DIR_LSTM = rf"{BASE_DIR}\submission\lstm"
DIR_LGBM = rf"{BASE_DIR}\submission\lgbm"
DIR_OUT  = rf"{BASE_DIR}\submission\ensemble"
os.makedirs(DIR_OUT, exist_ok=True)

TARGET_X = "Disp. X"
TARGET_Z = "Disp. Z"

def read_csv_safe(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp950"]:
        try: return pd.read_csv(path, encoding=enc)
        except Exception: continue
    return pd.read_csv(path)

def merge_one_file(path_lstm: str, path_lgbm: str, out_path: str):
    df_lstm = read_csv_safe(path_lstm)
    df_lgbm = read_csv_safe(path_lgbm)

    # 以 LSTM 檔為基底，對齊長度
    n = min(len(df_lstm), len(df_lgbm))
    base = df_lstm.iloc[:n].copy()
    other = df_lgbm.iloc[:n].copy()

    # 欄位確保存在
    for col in [TARGET_X, TARGET_Z]:
        if col not in base.columns:  base[col]  = np.nan
        if col not in other.columns: other[col] = np.nan

    # 規則：X 用 LGBM、Z 用 LSTM
    x_lgbm = pd.to_numeric(other[TARGET_X], errors="coerce")
    z_lstm = pd.to_numeric(base[TARGET_Z],  errors="coerce")

    base[TARGET_X] = x_lgbm
    base[TARGET_Z] = z_lstm

    base.to_csv(out_path, index=False, encoding="utf-8-sig")

def main():
    files_lstm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LSTM, "*.csv"))}
    files_lgbm = {os.path.basename(p): p for p in glob.glob(os.path.join(DIR_LGBM, "*.csv"))}
    names = sorted(set(files_lstm.keys()) | set(files_lgbm.keys()))
    merged = 0; copied = 0

    for name in names:
        out_path = os.path.join(DIR_OUT, name)
        if name in files_lstm and name in files_lgbm:
            merge_one_file(files_lstm[name], files_lgbm[name], out_path)
            merged += 1
        elif name in files_lstm:
            shutil.copy2(files_lstm[name], out_path); copied += 1
        else:
            shutil.copy2(files_lgbm[name], out_path); copied += 1

    print(f"[DONE] 合併完成：混合 {merged} 檔；直接複製 {copied} 檔。")
    print(f"[OUT] {DIR_OUT}")

if __name__ == "__main__":
    main()
