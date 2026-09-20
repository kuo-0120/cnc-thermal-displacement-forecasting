# -*- coding: utf-8 -*-
# 改版說明：
# - 固定驗證抽樣 RANDOM_SEED=20250904 → 分數更穩
# - BACKTEST_B=60、變溫段權重加上 EMA 平滑 + 中性回退 + 權重下限
# - 為 Disp.Z 也加入與 Disp.X 相同的物理白名單 EXOG 穩定化
# - 其他邏輯與 I/O 皆維持原 v9 行為（含 submission 輸出）

r"""
cnc_three_regimes_physical_v9_blend_ar1.py  + submission

重點：
- 維持 v9 模型（暖機、常溫、變溫三模型 + 變溫段 SARIMAX/UCM-LLT 加權融合）。
- 評分沿用「暖機後 + 缺值以上一預測/上一實際回填 + micro RMSE」。
- ✅ 新增 submission：讀取 TEST_DIR，從暖機點後自動回填空白的 Disp.X / Disp.Z，
  並將每檔輸出到 SUBMISSION_DIR，同名 CSV。

使用方法：
1) 調整 Config 中的 TRAIN_DIR / TEST_DIR / SUBMISSION_DIR 路徑。
2) 執行本檔：會先做訓練與驗證列印 RMSE，接著產生 submission 檔到 SUBMISSION_DIR。
"""

import os, glob, warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from math import isfinite

# statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX

VERSION = "v9-blend-LLTAR1-dynfeat+submit+stable"

# ================== 設定 ==================
@dataclass
class Config:
    # === 路徑 ===
    TRAIN_DIR: str = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "train")
    TEST_DIR: str = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "test")
    SUBMISSION_DIR: str = os.path.join(os.getenv("CNC_OUTPUT_DIR", "outputs"), "submission")

    TIME_COL: str  = "Time"
    TARGET_COLS: List[str] = field(default_factory=lambda: ["Disp. X", "Disp. Z"])

    HOLDOUT_K: int = 8
    RANDOM_SEED: Optional[int] = 20250904   # 固定驗證抽樣（原 None）
    HOLDOUT_FILES: Optional[List[str]] = None

    # 評分/暖啟動（同時也是測試回填起點）
    WARMUP_STEPS: int = 100
    CONTEXT_STEPS: int = 100

    # Regime 偵測
    REGIME_ROLL_WIN: int = 60
    CONST_VOL_FACTOR: float = 0.8
    WARM_SLOPE_Q: float = 0.97
    FLAT_SLOPE_Q: float = 0.20
    HIGH_VOL_Q:  float = 0.80

    # EXOG Top-K
    AUTO_EXOG_TOPK_X: int = 28   # X 軸容許多一點
    AUTO_EXOG_TOPK_Z: int = 17
    LAGS: List[int] = field(default_factory=lambda: [1,5,15])

    # SARIMAX 搜尋
    SARIMAX_GRID: List[Tuple[int,int,int]] = field(default_factory=lambda: [
        (1,1,1),(0,1,1),(1,0,1),(2,1,1),(1,1,0),
        (2,1,2),(3,1,1),(1,2,1),(0,1,2)
    ])
    MAXITER: int = 200

    # 暖機（每段）
    WARM_K_GRID: List[float] = field(default_factory=lambda: list(np.logspace(-5, 0, 40)))
    MAX_WARM_PRED_STEPS: int = 80

    # EXOG 清理門檻
    MIN_KEEP_RATIO: float = 0.6  # 欄位在各 run 的可用比例

    # 小回測長度（變溫段融合權重）
    BACKTEST_B: int = 60
    BLEND_EMA: float = 0.6        # 權重 EMA 平滑
    BT_NEUTRAL_EPS: float = 0.05  # rmse 差異小於 5% → 回退中性 0.5/0.5
    MIN_WS_WL: float = 0.25       # 權重下限，避免極端 0/1

    # 其他
    VERBOSE_BLEND: bool = False     # 是否印出 [VAR-BLEND] 權重訊息

CFG = Config()

# ================== 工具 ==================
def read_runs(folder: str, time_col: str = "Time"):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"資料夾內沒有 .csv：{folder}")
    runs = []
    for fp in files:
        df = pd.read_csv(fp, encoding="utf-8", low_memory=False)
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        df["run_id"] = os.path.basename(fp)
        runs.append(df)
    big = pd.concat(runs, ignore_index=True)
    return big, [os.path.basename(x) for x in files]

# ---- 基礎特徵（與你原版一致） ----
def add_physical_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    # 轉溫度特徵（範例：平均、斜率、波動等）
    tc_cols = [c for c in out.columns if c.startswith("TC_")]
    if tc_cols:
        out["TC_mean"] = out[tc_cols].mean(axis=1)
        out["TC_slope"] = out["TC_mean"].diff().fillna(0.0)
        out["TC_vol"] = out["TC_mean"].rolling(10, min_periods=1).std().bfill().fillna(0.0)
        out["Tnorm"] = (out["TC_mean"] - out["TC_mean"].expanding().min()) / (out["TC_mean"].expanding().max() - out["TC_mean"].expanding().min() + 1e-9)
        out["dTC"] = out["TC_mean"].diff().fillna(0.0)
        out["dTC_pos"] = out["dTC"].clip(lower=0.0)
        out["dTC_neg"] = (-out["dTC"].clip(upper=0.0))
        out["cum_dTC_pos"] = out["dTC_pos"].cumsum()
        out["cum_dTC_neg"] = out["dTC_neg"].cumsum()

    # 馬達負載與壓力的例示特徵
    for c in ["X Motor", "Z Motor"]:
        if c not in out.columns:
            out[c] = 0.0
    out["Motor_sum"] = out[["X Motor","Z Motor"]].sum(axis=1)
    out["dMotor"] = out["Motor_sum"].diff().fillna(0.0)
    if "PT_mean" not in out.columns:
        out["PT_mean"] = 0.0
    if "PT02" not in out.columns:
        out["PT02"] = 0.0

    # Lags
    lag_bases = ["TC_mean","TC_slope","TC_vol","Tnorm","dTC","dTC_pos","dTC_neg",
                 "cum_dTC_pos","cum_dTC_neg","X Motor","Z Motor","Motor_sum","dMotor","PT_mean","PT02"]
    for base in lag_bases:
        if base in out.columns:
            for L in CFG.LAGS:
                out[f"{base}_lag{L}"] = out[base].shift(L)
    return out

# ---- Regime 判定 ----
def compute_regime_thresholds(df: pd.DataFrame, roll_win: int, warm_q: float, flat_q: float, vol_q: float):
    # 以 Disp.X 的斜率與波動估門檻
    series = df[["Disp. X","Disp. Z"]].mean(axis=1)
    slope = series.diff().abs().rolling(roll_win, min_periods=1).mean()
    vol = series.rolling(roll_win, min_periods=1).std().fillna(0.0)
    thr_warm = slope.quantile(warm_q)
    thr_flat = slope.quantile(flat_q)
    thr_vol  = vol.quantile(vol_q)
    return float(thr_warm), float(thr_flat), float(thr_vol)

def tag_regime(df: pd.DataFrame, thr_warm: float, thr_flat: float, thr_vol: float) -> pd.Series:
    s = df[["Disp. X","Disp. Z"]].mean(axis=1)
    slope = s.diff().abs().rolling(CFG.REGIME_ROLL_WIN, min_periods=1).mean()
    vol = s.rolling(CFG.REGIME_ROLL_WIN, min_periods=1).std().fillna(0.0)
    reg = np.where(slope > thr_warm, "warmup",
          np.where((slope <= thr_flat) & (vol <= thr_vol*CFG.CONST_VOL_FACTOR), "const", "variable"))
    return pd.Series(reg, index=df.index)

# ---- SARIMAX / UCM-LLT 建模 ----
def fit_sarimax(y: pd.Series, X: Optional[pd.DataFrame], grid: List[Tuple[int,int,int]], maxiter: int=200):
    best = None; best_aic = np.inf
    for (p,d,q) in grid:
        try:
            mod = SARIMAX(endog=y, exog=X, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=maxiter)
            aic = res.aic
            if aic < best_aic:
                best_aic = aic; best = res
        except Exception:
            continue
    return best

def fit_ucm_llt(y: pd.Series, X: Optional[pd.DataFrame]):
    # Local Level + exog
    mod = UnobservedComponents(endog=y, level="local level", exog=X)
    res = mod.fit(disp=False)
    return res

# ---- 暖機線性 + 指數衰退擬合 ----
def fit_warm_prefix_choose_k(prefix: np.ndarray, k_grid: List[float]):
    # 線性 + 指數衰退：y[t] ≈ b0 + b1*t + e^{-k*t}
    T = len(prefix)
    x = np.arange(T)
    y = prefix
    best_k, best_mse, best_b0, best_b1 = None, np.inf, 0.0, 0.0
    for k in k_grid:
        # 簡化：先線性擬合，再加權修正
        A = np.vstack([np.ones(T), x]).T
        try:
            b1_est, b0_est = np.linalg.lstsq(A, y, rcond=None)[0][::-1]
        except Exception:
            b0_est, b1_est = 0.0, 0.0
        y_hat = b0_est + b1_est*x + np.exp(-k*x)
        mse = float(((y - y_hat)**2).mean())
        if mse < best_mse:
            best_mse, best_k, best_b0, best_b1 = mse, k, b0_est, b1_est
    return best_k, best_b0, best_b1

def warm_predict_segment(b0: float, b1: float, k: float, seg_len: int, start_pred: int):
    x = np.arange(seg_len)
    y_hat = b0 + b1*x + np.exp(-k*x)
    return y_hat[start_pred:]

# ---- EXOG 自動挑選（含白名單優先） ----
def auto_exog_for_target(df: pd.DataFrame, candidates: List[str], target: str, k: int,
                         priority_keep: List[str]) -> List[str]:
    y = df[target]
    scores = []
    for c in candidates:
        try: scores.append((c, float(abs(y.corr(df[c])))))
        except Exception: scores.append((c, 0.0))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = [c for c,_ in scores[:k]]
    for p in priority_keep:
        if p in candidates and p not in chosen:
            chosen.append(p)
    return chosen

# ---- EXOG 清理 ----
def clean_exog_train(tr_df: pd.DataFrame, exog_cols: List[str], min_keep_ratio: float = 0.6) -> List[str]:
    exog_cols = [c for c in exog_cols if c in tr_df.columns]
    if not exog_cols: return []
    # 修正：正確的 DataFrame 欄位索引語法（避免把 DataFrame 當 callable）
    tmp = tr_df[["run_id"] + exog_cols].copy()
    tmp[exog_cols] = tmp[exog_cols].replace([np.inf, -np.inf], np.nan)
    def _fill(g):
        g[exog_cols] = g[exog_cols].ffill().bfill()
        return g
    tmp = tmp.groupby("run_id", as_index=False, group_keys=False).apply(_fill)
    miss_ratio = tmp[exog_cols].isna().mean()
    keep = miss_ratio[miss_ratio <= (1 - min_keep_ratio)].index.tolist()
    for c in keep:
        if tmp[c].isna().any():
            med = float(tmp[c].median(skipna=True)); med = 0.0 if not np.isfinite(med) else med
            tmp[c] = tmp[c].fillna(med)
    tr_df[keep] = tmp[keep].astype(float)
    return keep

def clean_exog_apply(X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if X is None: return None
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return X.astype(float)

# ---- Holdout 檔案挑選 ----
def pick_holdout_files(all_files: List[str], k: int, seed: Optional[int], fixed: Optional[List[str]]):
    if fixed:
        missing = [f for f in fixed if f not in all_files]
        if missing: raise ValueError(f"指定的驗證檔不存在：{missing}")
        return sorted(fixed)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    k = min(k, len(all_files))
    return sorted(rng.choice(all_files, size=k, replace=False).tolist())

# ---- 評分 ----
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# ================== 主建模流程 ==================
def train_one_target(tr_df: pd.DataFrame, va_df: pd.DataFrame, target: str, cfg: Config,
                     exog_candidates_all: List[str], topk: int):
    # EXOG 候選
    exog_cols_t = list(exog_candidates_all)

    # 目標特徵白名單（讓 X/Z 都穩定）
    if target == "Disp. X":
        phys_keep = [
            "TC_mean","TC_slope","TC_vol","Tnorm","dTC","dTC_pos","dTC_neg",
            "cum_dTC_pos","cum_dTC_neg",
            "X Motor","Z Motor","Motor_sum","dMotor",
            "PT_mean","PT02"
        ]
        phys_keep_with_lags = set(phys_keep)
        for base in phys_keep:
            for L in cfg.LAGS:
                phys_keep_with_lags.add(f"{base}_lag{L}")
        wl = [c for c in exog_cols_t if c in phys_keep_with_lags]
        if len(wl) == 0:
            candidates_wl = [c for c in exog_candidates_all if c in phys_keep_with_lags]
            if candidates_wl:
                wl = auto_exog_for_target(tr_df, candidates_wl, target, min(topk, len(candidates_wl)), priority_keep=[])
        exog_cols_t = wl if wl else exog_cols_t
        print(f"[INFO] 物理白名單(Disp.X,擴充)後 EXOG = {len(exog_cols_t)} 欄：{exog_cols_t}")

    elif target == "Disp. Z":
        phys_keep = [
            "TC_mean","TC_slope","TC_vol","Tnorm","dTC","dTC_pos","dTC_neg",
            "cum_dTC_pos","cum_dTC_neg",
            "X Motor","Z Motor","Motor_sum","dMotor",
            "PT_mean","PT02"
        ]
        phys_keep_with_lags = set(phys_keep)
        for base in phys_keep:
            for L in cfg.LAGS:
                phys_keep_with_lags.add(f"{base}_lag{L}")
        wl = [c for c in exog_cols_t if c in phys_keep_with_lags]
        if len(wl) == 0:
            candidates_wl = [c for c in exog_candidates_all if c in phys_keep_with_lags]
            if candidates_wl:
                wl = auto_exog_for_target(tr_df, candidates_wl, target, min(topk, len(candidates_wl)), priority_keep=[])
        exog_cols_t = wl if wl else exog_cols_t
        print(f"[INFO] 物理白名單(Disp.Z)後 EXOG = {len(exog_cols_t)} 欄：{exog_cols_t}")

    # 訓練 exog 清理（缺值>40% 的欄剔除）
    exog_cols_t = clean_exog_train(tr_df, exog_cols_t, min_keep_ratio=cfg.MIN_KEEP_RATIO)
    print(f"[INFO] 訓練可用 EXOG（清理後）= {len(exog_cols_t)} 欄：{exog_cols_t}")

    # 準備訓練資料
    sub_c = tr_df[tr_df["regime"]=="const"]; sub_v = tr_df[tr_df["regime"]=="variable"]
    y_c = (sub_c[target] if len(sub_c)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
    X_c = ((sub_c[exog_cols_t] if len(sub_c)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None
    y_v = (sub_v[target] if len(sub_v)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
    X_v = ((sub_v[exog_cols_t] if len(sub_v)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None

    # 建模
    res_const = fit_sarimax(y_c, X_c, cfg.SARIMAX_GRID, maxiter=cfg.MAXITER)
    res_var_smx = fit_sarimax(y_v, X_v, cfg.SARIMAX_GRID, maxiter=cfg.MAXITER)
    res_var_ll  = fit_ucm_llt(y_v, X_v)

    return {
        "res_const": res_const,
        "res_var_smx": res_var_smx,
        "res_var_ll": res_var_ll,
        "exog_cols": exog_cols_t
    }

# ---- 三模型推論（含變溫融合） ----
def predict_run_three_models(res_const, res_var_smx, res_var_ll,
                             y_run: pd.Series, X_run: Optional[pd.DataFrame],
                             reg_series: pd.Series, cfg: Config,
                             verbose_blend: bool = False):
    """
    回傳：y_true_eval, yhat_eval, eval_mask
    （對測試集也可用；eval_mask 會從 WARMUP_STEPS 起為 True）
    """
    n = len(y_run)
    y = y_run.astype(float).fillna(method="ffill")
    X = clean_exog_apply(X_run) if X_run is not None else None
    reg = reg_series.values

    # 連續段偵測
    blocks = []; s = 0
    for i in range(1, n):
        if reg[i] != reg[i-1]:
            blocks.append((s, i-1, reg[i-1])); s = i
    blocks.append((s, n-1, reg[-1]))

    # 評分/回填起點（暖機後）
    yhat = np.full(n, np.nan)
    eval_mask = np.zeros(n, dtype=bool)
    eval_start_global = min(max(1, cfg.WARMUP_STEPS), n-1)
    ctx = cfg.CONTEXT_STEPS

    prev_ws = 0.5  # for EMA smoothing of blend weights

    for (b_start, b_end, b_reg) in blocks:
        start_pred = max(b_start, eval_start_global) if b_start == 0 else max(b_start, eval_start_global)
        if start_pred > b_end:
            continue

        # 上下文
        ctx_start = max(0, start_pred - ctx)
        y_ctx = y.iloc[ctx_start:b_end+1]
        X_ctx = X.iloc[ctx_start:b_end+1] if X is not None else None
        rel_start_ctx = start_pred - ctx_start
        rel_end_ctx   = (b_end - ctx_start)

        if b_reg == "warmup":
            y_prefix = y.iloc[ctx_start:start_pred].values
            if len(y_prefix) >= 5:
                k_sel, b0, b1 = fit_warm_prefix_choose_k(y_prefix, cfg.WARM_K_GRID)
                max_len = min((b_end - start_pred + 1), cfg.MAX_WARM_PRED_STEPS)
                yhat_w = warm_predict_segment(b0, b1, k_sel, seg_len=(b_end - ctx_start + 1), start_pred=rel_start_ctx)[:max_len]
                Lw = len(yhat_w)
                if Lw > 0:
                    yhat[start_pred:start_pred+Lw] = yhat_w
                    eval_mask[start_pred:start_pred+Lw] = True
                rest_start = start_pred + Lw
            else:
                rest_start = start_pred

            if rest_start <= b_end:
                app = res_const.apply(endog=y.iloc[ctx_start:b_end+1],
                                      exog=(X.iloc[ctx_start:b_end+1] if X is not None else None))
                n_eff = int(getattr(app, "nobs", b_end+1-ctx_start))
                if (rest_start - ctx_start) < n_eff:
                    rel_start_ctx2 = rest_start - ctx_start
                    pred = app.get_prediction(start=rel_start_ctx2, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx2)
                    yhat_c = np.asarray(pred.predicted_mean)
                    Lc = len(yhat_c)
                    yhat[rest_start:rest_start+Lc] = yhat_c
                    eval_mask[rest_start:rest_start+Lc] = True

        elif b_reg == "const":
            app = res_const.apply(endog=y.iloc[ctx_start:b_end+1], exog=(X.iloc[ctx_start:b_end+1] if X is not None else None))
            n_eff = int(getattr(app, "nobs", len(y_ctx)))
            if rel_start_ctx < n_eff:
                pred = app.get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx)
                yhat_c = np.asarray(pred.predicted_mean)
                Lc = len(yhat_c)
                yhat[start_pred:start_pred+Lc] = yhat_c
                eval_mask[start_pred:start_pred+Lc] = True

        else:  # variable
            app_smx = res_var_smx.apply(endog=y.iloc[ctx_start:b_end+1], exog=(X.iloc[ctx_start:b_end+1] if X is not None else None))
            app_ll  = res_var_ll .apply(endog=y.iloc[ctx_start:b_end+1], exog=(X.iloc[ctx_start:b_end+1] if X is not None else None))
            n_eff_s = int(getattr(app_smx, "nobs", len(y_ctx)))
            n_eff_l = int(getattr(app_ll,  "nobs", len(y_ctx)))
            n_eff   = min(n_eff_s, n_eff_l)

            # 小回測長度取上下文可用長度
            B = min(cfg.BACKTEST_B, rel_start_ctx)
            has_bt = (B >= 10) and (rel_start_ctx < n_eff)

            if has_bt:
                bt_start = rel_start_ctx - B
                bt_end   = rel_start_ctx - 1
                pred_s_bt = app_smx.get_prediction(start=bt_start, end=bt_end, dynamic=bt_start).predicted_mean
                pred_l_bt = app_ll .get_prediction(start=bt_start, end=bt_end, dynamic=bt_start).predicted_mean
                y_bt      = y_ctx.iloc[bt_start:bt_end+1].values
                rmse_s = float(np.sqrt(np.mean((np.asarray(pred_s_bt) - y_bt)**2)))
                rmse_l = float(np.sqrt(np.mean((np.asarray(pred_l_bt) - y_bt)**2)))
                eps = 1e-6
                ws = 1.0 / (rmse_s + eps)
                wl = 1.0 / (rmse_l + eps)
                ws, wl = ws / (ws + wl), wl / (ws + wl)

                # --- 穩定化：差異小則回退中性；並避免極端，再做 EMA 平滑 ---
                gap = abs(rmse_s - rmse_l) / max(rmse_s, rmse_l)
                if gap < cfg.BT_NEUTRAL_EPS:
                    ws = wl = 0.5
                else:
                    ws = max(cfg.MIN_WS_WL, min(1.0 - cfg.MIN_WS_WL, ws))
                    wl = 1.0 - ws
                # EMA smoothing
                ws = cfg.BLEND_EMA * prev_ws + (1.0 - cfg.BLEND_EMA) * ws
                wl = 1.0 - ws
                prev_ws = ws

                if verbose_blend:
                    print(f"[VAR-BLEND] ws={ws:.2f} wl={wl:.2f} | rmse_s={rmse_s:.4f} rmse_l={rmse_l:.4f} | block=({b_start}-{b_end}) len={b_end-b_start+1}")
            else:
                ws = cfg.BLEND_EMA * prev_ws + (1.0 - cfg.BLEND_EMA) * 0.5
                wl = 1.0 - ws
                prev_ws = ws
                if verbose_blend:
                    print(f"[VAR-BLEND] (no BT) ws={ws:.2f} wl={wl:.2f} | block=({b_start}-{b_end}) len={b_end-b_start+1}")

            if rel_start_ctx < n_eff:
                pred_s = app_smx.get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx).predicted_mean
                pred_l = app_ll .get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx).predicted_mean
                yhat_v = ws*np.asarray(pred_s) + wl*np.asarray(pred_l)
                Lv = len(yhat_v)
                yhat[start_pred:start_pred+Lv] = yhat_v
                eval_mask[start_pred:start_pred+Lv] = True

    # 評分段（暖機後）
    desired_eval = np.zeros(n, dtype=bool)
    desired_eval[eval_start_global:] = eval_mask[eval_start_global:]
    y_true_eval = y.values[desired_eval]
    yhat_eval   = yhat[desired_eval]
    eval_mask   = desired_eval
    # =======================================

    return y_true_eval, yhat_eval, eval_mask

# ================== 主程式 ==================
def main(cfg: Config = CFG):
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    print(f"[VERSION] {VERSION} | WARMUP={cfg.WARMUP_STEPS} | CTX={cfg.CONTEXT_STEPS} | LAGS={cfg.LAGS}")
    big_raw, file_list = read_runs(cfg.TRAIN_DIR, time_col=cfg.TIME_COL)

    thr_warm, thr_flat, thr_vol = compute_regime_thresholds(big_raw, cfg.REGIME_ROLL_WIN,
                                                            cfg.WARM_SLOPE_Q, cfg.FLAT_SLOPE_Q, cfg.HIGH_VOL_Q)
    big_feat = add_physical_features(big_raw, time_col=cfg.TIME_COL)
    big_feat["regime"] = tag_regime(big_feat, thr_warm, thr_flat, thr_vol)

    # EXOG 候選：去掉目標與 regime
    drop_cols2 = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
    exog_candidates_all = [c for c in big_feat.columns if c not in drop_cols2]

    # 切 train / val（固定種子→可重現）
    holdout = pick_holdout_files(file_list, cfg.HOLDOUT_K, cfg.RANDOM_SEED, cfg.HOLDOUT_FILES)
    print(f"[INFO] 共有 {len(file_list)} 檔；驗證抽 {len(holdout)} 檔：")
    for r in holdout: print("  -", r)
    tr_df = big_feat[~big_feat["run_id"].isin(holdout)].copy()
    va_df = big_feat[ big_feat["run_id"].isin(holdout)].copy()

    # 兩個目標各自建模
    models: Dict[str, Dict] = {}
    for tgt, topk in zip(cfg.TARGET_COLS, [cfg.AUTO_EXOG_TOPK_X, cfg.AUTO_EXOG_TOPK_Z]):
        print(f"\n[TRAIN] target = {tgt} | topk={topk}")
        models[tgt] = train_one_target(tr_df, va_df, tgt, cfg, exog_candidates_all, topk)

    # ====== 驗證 RMSE ======
    print("\n[VALID] 開始驗證（暖機後 micro RMSE）…")
    all_res = []
    for run_name in holdout:
        g = va_df[va_df["run_id"]==run_name].reset_index(drop=True)
        reg = g["regime"]
        out = {}
        for tgt in cfg.TARGET_COLS:
            y = g[tgt]
            X = g[models[tgt]["exog_cols"]] if models[tgt]["exog_cols"] else None
            y_true, y_pred, mask = predict_run_three_models(models[tgt]["res_const"], models[tgt]["res_var_smx"], models[tgt]["res_var_ll"],
                                                            y, X, reg, cfg, verbose_blend=cfg.VERBOSE_BLEND)
            out[tgt] = rmse(y_true, y_pred)
        out["run_id"] = run_name
        out["mean_rmse"] = float(np.mean([out[c] for c in cfg.TARGET_COLS]))
        all_res.append(out)
        print(f"  - {run_name} | X={out['Disp. X']:.4f}  Z={out['Disp. Z']:.4f}  mean={out['mean_rmse']:.4f}")

    # Micro RMSE
    micro = {}
    for tgt in cfg.TARGET_COLS:
        seq_true = []; seq_pred = []
        for run_name in holdout:
            g = va_df[va_df["run_id"]==run_name].reset_index(drop=True)
            y = g[tgt]
            X = g[models[tgt]["exog_cols"]] if models[tgt]["exog_cols"] else None
            y_true, y_pred, mask = predict_run_three_models(models[tgt]["res_const"], models[tgt]["res_var_smx"], models[tgt]["res_var_ll"],
                                                            y, X, g["regime"], cfg, verbose_blend=False)
            seq_true.append(y_true); seq_pred.append(y_pred)
        seq_true = np.concatenate(seq_true); seq_pred = np.concatenate(seq_pred)
        micro[tgt] = rmse(seq_true, seq_pred)
    micro_mean = float(np.mean([micro[c] for c in cfg.TARGET_COLS]))
    print("\n[RESULT] micro RMSE：")
    print(f"Disp.X = {micro['Disp. X']:.6f}")
    print(f"Disp.Z = {micro['Disp. Z']:.6f}")
    print(f"Mean   = {micro_mean:.6f}")

    # ====== 產生 submission ======
    print("\n[SUBMISSION] 針對 TEST_DIR 產生回填檔…")
    test_raw, test_files = read_runs(cfg.TEST_DIR, time_col=cfg.TIME_COL)
    test_feat = add_physical_features(test_raw, time_col=cfg.TIME_COL)
    thr_warm2, thr_flat2, thr_vol2 = compute_regime_thresholds(test_raw, cfg.REGIME_ROLL_WIN, cfg.WARM_SLOPE_Q, cfg.FLAT_SLOPE_Q, cfg.HIGH_VOL_Q)
    test_feat["regime"] = tag_regime(test_feat, thr_warm2, thr_flat2, thr_vol2)

    for run_name in sorted(test_files):
        g_raw = test_feat[test_feat["run_id"]==run_name].reset_index(drop=True)
        reg = g_raw["regime"]
        out_cols = {}
        for tgt in cfg.TARGET_COLS:
            y = g_raw[tgt]
            X = g_raw[models[tgt]["exog_cols"]] if models[tgt]["exog_cols"] else None
            y_true, y_pred, mask = predict_run_three_models(models[tgt]["res_const"], models[tgt]["res_var_smx"], models[tgt]["res_var_ll"],
                                                            y, X, reg, cfg, verbose_blend=False)
            # 暖機後位置回填（遇空白才填）
            fill_idx = np.where(mask)[0]
            pred_vals = y_pred
            tgt_col = tgt
            # 只在 NaN 的地方填入預測
            y_out = g_raw[tgt_col].astype(float).values.copy()
            m_nan = np.isnan(y_out)
            m_use = np.zeros_like(m_nan)
            m_use[fill_idx] = True
            to_fill = m_nan & m_use
            y_out[to_fill] = pred_vals[:to_fill.sum()]
            g_raw[tgt_col] = y_out
            out_cols[tgt_col] = y_out

        # 寫檔（保持 Time, run_id, 目標欄位）
        cols = [CFG.TIME_COL, "run_id"]
        cols += cfg.TARGET_COLS
        out_path = os.path.join(cfg.SUBMISSION_DIR, run_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        g_raw[cols].to_csv(out_path, index=False, encoding="utf-8-sig")

        # 簡要統計
        nx = int(np.isnan(test_raw.loc[test_raw["run_id"]==run_name, "Disp. X"]).sum())
        nz = int(np.isnan(test_raw.loc[test_raw["run_id"]==run_name, "Disp. Z"]).sum())
        print(f"[SAVE] {out_path} | 原始空白：X={nx} Z={nz}（已在暖機後區間回填）")

    print(f"\n[DONE] submission 已輸出至：{cfg.SUBMISSION_DIR}")

if __name__ == "__main__":
    main(CFG)
