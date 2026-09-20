# XGboost.py － 自動偵測特徵欄位 + 寫死路徑 + 早停相容新舊版
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor

# 1) 寫死你的訓練資料夾路徑
TRAIN_DIR = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "train")

# 目標欄位
TARGET_COLS = ["Disp. X", "Disp. Z"]
SEED = 42

def read_all_csv(csv_dir: str):
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"在資料夾找不到任何 CSV：{csv_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file__"] = os.path.basename(p)
        dfs.append(df)
    return dfs

def infer_feature_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in TARGET_COLS and c != "__source_file__"]
    if not cols:
        raise ValueError("無法推斷特徵欄位，請檢查資料欄位名稱。")
    return cols

def build_xgb(seed: int = SEED):
    # 穩健預設；是否早停交給 fit_with_es 決定
    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
    )

def fit_with_es(model, X_tr, y_tr, X_va, y_va, rounds: int = 50):
    """相容新舊版 xgboost 的 early stopping；都不支援時就不早停。"""
    # 1) 新版：fit(..., early_stopping_rounds=)
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=rounds,
            verbose=False
        )
        return model
    except TypeError:
        pass
    # 2) 中版：callbacks
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[xgb.callback.EarlyStopping(rounds=rounds, save_best=True)],
            verbose=False
        )
        return model
    except TypeError:
        pass
    # 3) 最舊版：不支援早停，直接訓練
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model

if __name__ == "__main__":
    print("[STEP] 讀取並合併訓練資料…")
    dfs = read_all_csv(TRAIN_DIR)
    data = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 合併後資料：{len(data):,} 筆，欄位數：{data.shape[1]}")

    # 檢查必備欄位
    for t in TARGET_COLS:
        if t not in data.columns:
            raise ValueError(f"缺少必要欄位：{t}")

    # 自動推斷特徵欄位
    feature_cols = infer_feature_cols(data)
    print(f"[INFO] 自動偵測特徵欄位數：{len(feature_cols)}")

    X = data[feature_cols].values
    Y = data[TARGET_COLS].values  # shape: (n, 2)

    # 一次切分，確保 X/Z 驗證集一致
    X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    yx_tr, yz_tr = Y_tr[:, 0], Y_tr[:, 1]
    yx_va,  yz_va = Y_va[:, 0], Y_va[:, 1]

    print("[STEP] 訓練 XGBoost（Disp. X）…")
    model_x = build_xgb(seed=SEED)
    model_x = fit_with_es(model_x, X_tr, yx_tr, X_va, yx_va, rounds=50)

    print("[STEP] 訓練 XGBoost（Disp. Z）…")
    model_z = build_xgb(seed=SEED)
    model_z = fit_with_es(model_z, X_tr, yz_tr, X_va, yz_va, rounds=50)

    # 驗證預測與 RMSE（用 sqrt(MSE) 以相容舊版 sklearn）
    yx_pred = model_x.predict(X_va)
    yz_pred = model_z.predict(X_va)
    rmse_x = mean_squared_error(yx_va, yx_pred) ** 0.5
    rmse_z = mean_squared_error(yz_va, yz_pred) ** 0.5
    rmse_avg = (rmse_x + rmse_z) / 2

    best_x = getattr(model_x, "best_iteration", None)
    best_z = getattr(model_z, "best_iteration", None)

    print(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={rmse_avg:.6f}")
    print(f"[INFO] best_iters: X={best_x}, Z={best_z}")
    print("[DONE] 訓練完成。")
