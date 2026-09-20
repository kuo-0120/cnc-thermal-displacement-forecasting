# LGBM.py — 單模型：訓練/驗證 + 驗證逐列輸出 + 測試submission（存到 submission\lgbm\）
import os, glob, math, random, sys, traceback
import numpy as np
import pandas as pd

def log(msg):
    print(msg, flush=True)
    with open("LGBM_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

try:
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    log(f"[ENV] LightGBM OK, version={getattr(lgb,'__version__','?')}")
except Exception as e:
    log("[ERR] 請先安裝 lightgbm：pip install lightgbm")
    raise

# ===== 路徑 =====
BASE_DIR = os.getenv("CNC_DATA_ROOT", "data")
TRAIN_DIR = rf"{BASE_DIR}\train"
TEST_DIR  = rf"{BASE_DIR}\test"
VAL_PRED_DIR = rf"{BASE_DIR}\val_pred\lgbm"       # 驗證逐列預測
OUT_DIR      = rf"{BASE_DIR}\submission\lgbm"     # LGBM 的 submission
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_PRED_DIR, exist_ok=True)

TARGET_X, TARGET_Z = "Disp. X", "Disp. Z"
TARGET_COLS = [TARGET_X, TARGET_Z]
SEED = 42
LAGS = [1,2,3]
FILL_AFTER_N = 100

LGB_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_samples=20,
    random_state=SEED,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 100
RETRAIN_ON_ALL = True

random.seed(SEED); np.random.seed(SEED)

def is_blank(v):
    return (pd.isna(v)) or (isinstance(v,str) and v.strip()=="")

def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return math.sqrt(mean_squared_error(y_true[m], y_pred[m])) if m.sum() else float("nan")

def read_all_csv(csv_dir):
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

def infer_exo_cols(df_all):
    cand = [c for c in df_all.columns if c not in TARGET_COLS and c != "__source_file__"]
    exo=[]
    for c in cand:
        if pd.api.types.is_numeric_dtype(df_all[c]): exo.append(c)
        else:
            arr = pd.to_numeric(df_all[c], errors="coerce")
            if arr.notna().sum()>0: exo.append(c)
    return exo

def add_lag_columns(df, cols, lags):
    d = df.copy()
    for c in cols:
        base = pd.to_numeric(d[c], errors="coerce")
        for k in lags:
            d[f"{c}_lag{k}"] = base.shift(k)
    return d

def numericify(df, cols):
    d=df.copy()
    for c in cols:
        if c not in d.columns: d[c]=np.nan
        d[c]=pd.to_numeric(d[c], errors="coerce")
    return d

def build_supervised_df(df, exo_cols, lags, target):
    d = add_lag_columns(df, exo_cols, lags)
    d = add_lag_columns(d, [TARGET_X, TARGET_Z], lags)
    feat_cols = list(exo_cols) \
        + [f"{c}_lag{k}" for c in exo_cols for k in lags] \
        + [f"{TARGET_X}_lag{k}" for k in lags] \
        + [f"{TARGET_Z}_lag{k}" for k in lags]
    d = numericify(d, feat_cols + TARGET_COLS)
    K = max(lags) if lags else 0
    if K>0: d = d.iloc[K:].copy()
    d = d[~d[target].isna()].copy()
    X_df = d[feat_cols].copy()
    y    = d[target].to_numpy(np.float32)
    return X_df, y, feat_cols, d

def rollout_and_save_val_preds_lgbm(modelX, modelZ, feature_names, exo_cols, lags, val_dfs, fill_after_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    preds_all_x, preds_all_z, gts_all_x, gts_all_z = [], [], [], []
    feat_names_X = getattr(modelX, "feature_name_", None) or feature_names
    feat_names_Z = getattr(modelZ, "feature_name_", None) or feature_names

    for df_raw in val_dfs:
        src = df_raw["__source_file__"].iloc[0]
        df_out = df_raw.copy()
        n = len(df_out)
        if n==0: continue

        dispX = pd.to_numeric(df_out[TARGET_X], errors="coerce").values.astype("float32")
        dispZ = pd.to_numeric(df_out[TARGET_Z], errors="coerce").values.astype("float32")
        start_i = max(max(lags), fill_after_n-1)

        idxs, pxs, pzs, yxs, yzs = [], [], [], [], []
        for i in range(start_i, n):
            row_vals = []
            for c in exo_cols:
                row_vals.append(pd.to_numeric(df_out.at[i, c], errors="coerce"))
                for k in lags:
                    j = i - k
                    row_vals.append(pd.to_numeric(df_out.at[j, c], errors="coerce") if j>=0 else np.nan)

            def lag(arr,k):
                j=i-k
                return np.nan if j<0 or np.isnan(arr[j]) else float(arr[j])
            for k in lags:
                row_vals.append(lag(dispX,k))
            for k in lags:
                row_vals.append(lag(dispZ,k))

            x_df_X = pd.DataFrame([row_vals], columns=feat_names_X)
            x_df_Z = pd.DataFrame([row_vals], columns=feat_names_Z)

            px = modelX.predict(x_df_X)[0]
            pz = modelZ.predict(x_df_Z)[0]

            gtx = pd.to_numeric(df_out.at[i, TARGET_X], errors="coerce")
            gtz = pd.to_numeric(df_out.at[i, TARGET_Z], errors="coerce")
            pxs.append(px); pzs.append(pz)
            yxs.append(gtx); yzs.append(gtz)
            idxs.append(i)

            if np.isnan(dispX[i]): dispX[i] = float(px)
            if np.isnan(dispZ[i]): dispZ[i] = float(pz)

        out_df = pd.DataFrame({
            "__idx__": idxs,
            "y_true_X": yxs, "y_pred_LGBM_X": pxs,
            "y_true_Z": yzs, "y_pred_LGBM_Z": pzs,
        })
        save_path = os.path.join(out_dir, f"{src[:-4] if src.lower().endswith('.csv') else src}.csv")
        out_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        log(f"[VAL-PRED-LGBM] save -> {save_path}")

        preds_all_x.append(out_df["y_pred_LGBM_X"].values)
        preds_all_z.append(out_df["y_pred_LGBM_Z"].values)
        gts_all_x.append(out_df["y_true_X"].values)
        gts_all_z.append(out_df["y_true_Z"].values)

    if not preds_all_x:
        log("[VAL-PRED-LGBM] 無資料。")
        return float("nan"), float("nan"), float("nan")

    px = np.concatenate(preds_all_x); pz = np.concatenate(preds_all_z)
    yx = np.concatenate(gts_all_x);   yz = np.concatenate(gts_all_z)
    rmse_x = safe_rmse(yx, px); rmse_z = safe_rmse(yz, pz)
    log(f"[VAL-ROLLOUT] (per-file saved) RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={(rmse_x+rmse_z)/2:.6f}")
    return rmse_x, rmse_z, (rmse_x+rmse_z)/2

def main():
    log("[STEP] 讀取訓練資料…")
    train_dfs_all = read_all_csv(TRAIN_DIR)
    log(f"[INFO] 訓練 CSV 數量：{len(train_dfs_all)}")
    train_all = pd.concat(train_dfs_all, ignore_index=True)

    for t in TARGET_COLS:
        if t not in train_all.columns:
            raise ValueError(f"訓練資料缺少必要欄位：{t}")

    exo_cols = infer_exo_cols(train_all)
    log(f"[INFO] 外生特徵數：{len(exo_cols)}")

    files = sorted(train_all["__source_file__"].unique().tolist())
    tr_files, va_files = train_test_split(files, test_size=0.2, random_state=SEED)
    tr_files, va_files = set(tr_files), set(va_files)

    tr_dfs = [df for df in train_dfs_all if df["__source_file__"].iloc[0] in tr_files]
    va_dfs = [df for df in train_dfs_all if df["__source_file__"].iloc[0] in va_files]
    log(f"[INFO] 檔案 split -> train:{len(tr_dfs)}  val:{len(va_dfs)}")

    Xtr_X_list, ytr_X_list, Xtr_Z_list, ytr_Z_list = [], [], [], []
    feature_names = None
    for df in tr_dfs:
        Xx_df, yx, feat_names, _ = build_supervised_df(df, exo_cols, LAGS, TARGET_X)
        Xz_df, yz, _, _          = build_supervised_df(df, exo_cols, LAGS, TARGET_Z)
        if len(Xx_df)>0: Xtr_X_list.append(Xx_df); ytr_X_list.append(yx)
        if len(Xz_df)>0: Xtr_Z_list.append(Xz_df); ytr_Z_list.append(yz)
        if feature_names is None: feature_names = feat_names

    if not Xtr_X_list or not Xtr_Z_list:
        raise RuntimeError("訓練資料不足，檢查 LAGS/欄位。")

    Xtr_X = pd.concat(Xtr_X_list, ignore_index=True); ytr_X = np.concatenate(ytr_X_list)
    Xtr_Z = pd.concat(Xtr_Z_list, ignore_index=True); ytr_Z = np.concatenate(ytr_Z_list)

    Xva_X_list, yva_X_list, Xva_Z_list, yva_Z_list = [], [], [], []
    for df in va_dfs:
        Xx_df, yx, _, _ = build_supervised_df(df, exo_cols, LAGS, TARGET_X)
        Xz_df, yz, _, _ = build_supervised_df(df, exo_cols, LAGS, TARGET_Z)
        if len(Xx_df)>0: Xva_X_list.append(Xx_df); yva_X_list.append(yx)
        if len(Xz_df)>0: Xva_Z_list.append(Xz_df); yva_Z_list.append(yz)
    Xva_X = pd.concat(Xva_X_list, ignore_index=True); yva_X = np.concatenate(yva_X_list)
    Xva_Z = pd.concat(Xva_Z_list, ignore_index=True); yva_Z = np.concatenate(yva_Z_list)

    log(f"[INFO] X目標：train={Xtr_X.shape}  val={Xva_X.shape}")
    log(f"[INFO] Z目標：train={Xtr_Z.shape}  val={Xva_Z.shape}")

    log("[STEP] 訓練 LightGBM (Disp.X)…")
    modelX = lgb.LGBMRegressor(**LGB_PARAMS)
    modelX.fit(
        Xtr_X, ytr_X,
        eval_set=[(Xva_X, yva_X)],
        eval_metric="rmse",
        callbacks=[lgb.log_evaluation(period=50), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)]
    )
    best_iter_X = getattr(modelX, "best_iteration_", modelX.n_estimators)

    log("[STEP] 訓練 LightGBM (Disp.Z)…")
    modelZ = lgb.LGBMRegressor(**LGB_PARAMS)
    modelZ.fit(
        Xtr_Z, ytr_Z,
        eval_set=[(Xva_Z, yva_Z)],
        eval_metric="rmse",
        callbacks=[lgb.log_evaluation(period=50), lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)]
    )
    best_iter_Z = getattr(modelZ, "best_iteration_", modelZ.n_estimators)

    # teacher forcing
    pred_va_X = modelX.predict(Xva_X, num_iteration=best_iter_X)
    pred_va_Z = modelZ.predict(Xva_Z, num_iteration=best_iter_Z)
    rmse_x = safe_rmse(yva_X, pred_va_X); rmse_z = safe_rmse(yva_Z, pred_va_Z)
    log(f"[VAL] RMSE_X={rmse_x:.6f}  RMSE_Z={rmse_z:.6f}  RMSE_avg={(rmse_x+rmse_z)/2:.6f}")

    # 驗證 ROLLOUT 並存逐列預測
    rollout_and_save_val_preds_lgbm(modelX, modelZ, feature_names, exo_cols, LAGS, va_dfs, FILL_AFTER_N, VAL_PRED_DIR)

    # 全量重訓（可保留）
    if RETRAIN_ON_ALL:
        log("[STEP] Retrain on ALL training data…")
        Xall_X_list, yall_X_list, Xall_Z_list, yall_Z_list = [], [], [], []
        for df in (tr_dfs + va_dfs):
            Xx_df, yx, _, _ = build_supervised_df(df, exo_cols, LAGS, TARGET_X)
            Xz_df, yz, _, _ = build_supervised_df(df, exo_cols, LAGS, TARGET_Z)
            if len(Xx_df)>0: Xall_X_list.append(Xx_df); yall_X_list.append(yx)
            if len(Xz_df)>0: Xall_Z_list.append(Xz_df); yall_Z_list.append(yz)
        Xall_X = pd.concat(Xall_X_list, ignore_index=True); yall_X = np.concatenate(yall_X_list)
        Xall_Z = pd.concat(Xall_Z_list, ignore_index=True); yall_Z = np.concatenate(yall_Z_list)

        modelX = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": best_iter_X}).fit(Xall_X, yall_X)
        modelZ = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": best_iter_Z}).fit(Xall_Z, yall_Z)
        log("[STEP] Full-train complete.")

    # 測試遞推補值（寫到 submission\lgbm\）
    log("[STEP] 讀取測試資料並遞推補值…")
    test_dfs = read_all_csv(TEST_DIR)
    feat_names_X = getattr(modelX, "feature_name_", None) or feature_names
    feat_names_Z = getattr(modelZ, "feature_name_", None) or feature_names

    for raw in test_dfs:
        src = raw["__source_file__"].iloc[0]
        df_out = raw.copy()
        if TARGET_X not in df_out.columns: df_out[TARGET_X] = np.nan
        if TARGET_Z not in df_out.columns: df_out[TARGET_Z] = np.nan

        dispX = pd.to_numeric(df_out[TARGET_X], errors="coerce").values.astype("float32")
        dispZ = pd.to_numeric(df_out[TARGET_Z], errors="coerce").values.astype("float32")

        n = len(df_out)
        start_i = max(max(LAGS), FILL_AFTER_N-1)
        filled = 0

        for i in range(start_i, n):
            row_vals = []
            for c in exo_cols:
                row_vals.append(pd.to_numeric(df_out.at[i, c], errors="coerce"))
                for k in LAGS:
                    j=i-k
                    row_vals.append(pd.to_numeric(df_out.at[j, c], errors="coerce") if j>=0 else np.nan)
            def lag(arr,k):
                j=i-k
                return np.nan if j<0 or np.isnan(arr[j]) else float(arr[j])
            for k in LAGS:
                row_vals.append(lag(dispX,k))
            for k in LAGS:
                row_vals.append(lag(dispZ,k))

            x_df_X = pd.DataFrame([row_vals], columns=feat_names_X)
            x_df_Z = pd.DataFrame([row_vals], columns=feat_names_Z)

            need_pred = is_blank(df_out.at[i, TARGET_X]) or is_blank(df_out.at[i, TARGET_Z])
            if need_pred:
                px = modelX.predict(x_df_X)[0]
                pz = modelZ.predict(x_df_Z)[0]
                if is_blank(df_out.at[i, TARGET_X]): df_out.at[i, TARGET_X] = float(px)
                if is_blank(df_out.at[i, TARGET_Z]): df_out.at[i, TARGET_Z] = float(pz)
                filled += 1
                if np.isnan(dispX[i]): dispX[i] = float(px)
                if np.isnan(dispZ[i]): dispZ[i] = float(pz)
            else:
                if np.isnan(dispX[i]): dispX[i] = float(pd.to_numeric(df_out.at[i, TARGET_X], errors="coerce"))
                if np.isnan(dispZ[i]): dispZ[i] = float(pd.to_numeric(df_out.at[i, TARGET_Z], errors="coerce"))

        orig_cols = [c for c in raw.columns if c not in ["__source_file__"] + TARGET_COLS]
        save_cols = orig_cols + TARGET_COLS
        out_path = os.path.join(OUT_DIR, src)
        df_out[save_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
        log(f"[SAVE] {out_path} -> 遞推回填 {filled} 筆（自第{FILL_AFTER_N}筆起之空白 Disp）")

if __name__ == "__main__":
    try:
        main()
        log("[DONE] 全流程完成")
    except Exception:
        log("[FATAL] 例外發生，詳細堆疊已寫入 LGBM_log.txt")
        with open("LGBM_log.txt", "a", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(*sys.exc_info())) + "\n")
        raise
