#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMBD 2025 - Train-only Baseline（內建路徑版）
直接用固定的 train 資料夾路徑，不用手動輸入參數
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ======== 這裡改成你的訓練資料夾路徑 ========
TRAIN_DIR = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "train")
# ===========================================

TARGET_COLS = ["Disp. X", "Disp. Z"]
SEED = 42


def read_all_csv(csv_dir):
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"在資料夾找不到任何 CSV：{csv_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        dfs.append(df)
    return dfs


def infer_feature_cols(df):
    return [c for c in df.columns if c not in TARGET_COLS and c != "__source_file__"]


def make_train_data(train_dir):
    dfs = read_all_csv(train_dir)
    df_all = pd.concat(dfs, ignore_index=True)
    if "Time" in df_all.columns:
        df_all = df_all.sort_values(by=["Time"]).reset_index(drop=True)
    feature_cols = infer_feature_cols(df_all)
    X = df_all[feature_cols].copy()
    y = df_all[TARGET_COLS].copy()
    print(f"[INFO] 合併後資料：{len(df_all):,} 筆，特徵數：{len(feature_cols)} 欄")
    return X, y


def train_and_validate(X, y, model_name="ridge", test_size=0.2, seed=SEED):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if model_name == "ridge":
        model = MultiOutputRegressor(Ridge(alpha=1.0, random_state=seed))
    else:
        model = MultiOutputRegressor(LinearRegression())

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_val_s)

        # RMSE = sqrt(MSE)
    rmse_x = mean_squared_error(y_val["Disp. X"], y_pred[:, 0]) ** 0.5
    rmse_z = mean_squared_error(y_val["Disp. Z"], y_pred[:, 1]) ** 0.5
    rmse_avg = (rmse_x + rmse_z) / 2


    print(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={rmse_avg:.6f}")


if __name__ == "__main__":
    try:
        print("[STEP] 讀取並合併訓練資料…")
        X, y = make_train_data(TRAIN_DIR)
        print("[STEP] 訓練與驗證…")
        train_and_validate(X, y, model_name="ridge")
        print("[DONE] 訓練完成。")
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
