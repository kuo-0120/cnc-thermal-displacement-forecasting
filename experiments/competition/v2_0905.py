# -*- coding: utf-8 -*-
r"""
v2.py — 三段式（暖機/恆溫/變溫）混合模型 + 一鍵產生 submission

重點：
- 暖機段：指數衰退（可帶線性趨勢），用暖機前綴擬合後從暖機結束點往後外推
- 恆溫段：UCM（Local Linear Trend, 殘差 AR(1)）做平穩微漂移
- 變溫段：SARIMAX 與 UCM-LLT 雙模型，使用「上下文小回測 RMSE 倒數」加權融合
- EXOG：自動以皮爾森相關排序＋物理白名單；對每個 run 做前後填補、以中位數補洞
- 標準化：只對 EXOG 做 z-score（用訓練統計），避免目標被縮放
- 評分：暖機後全區間 RMSE（缺值以「上一個預測→上一個實際」回填）
- ✅ 產生 submission：讀取 --test 目錄內所有 .csv，輸出到 --sub 目錄；檔名原樣

依賴：
    pip install numpy pandas statsmodels
"""
import os, glob, sys, warnings, math, argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


# ==========================
# Config
# ==========================
@dataclass
class Config:
    # 路徑
    TRAIN_DIR: str = os.path.join(os.getenv("CNC_DATA_ROOT", "data"), "train")
    TEST_DIR: Optional[str] = None
    SUBMISSION_DIR: Optional[str] = None

    # 欄位
    TIME_COL: str = "Time"
    TARGET_COLS: List[str] = field(default_factory=lambda: ["Disp. X", "Disp. Z"])

    # 評分 / 預測
    WARMUP_STEPS: int = 100
    CONTEXT_STEPS: int = 100

    # regime 標註（以溫度統計判斷）
    REGIME_ROLL_WIN: int = 60
    CONST_VOL_FACTOR: float = 0.8
    WARM_SLOPE_Q: float = 0.97    # 斜率分位數門檻（大於 → 暖機）
    FLAT_SLOPE_Q: float = 0.20    # 斜率分位數門檻（小於 → 近似恆溫）
    HIGH_VOL_Q: float  = 0.80     # 波動分位數門檻（大於 → 變溫）

    # EXOG Top-K
    AUTO_EXOG_TOPK_X: int = 28
    AUTO_EXOG_TOPK_Z: int = 17
    LAGS: List[int] = field(default_factory=lambda: [1, 5, 15])

    # SARIMAX 搜尋
    SARIMAX_GRID: List[Tuple[int,int,int]] = field(default_factory=lambda: [
        (1,1,1),(0,1,1),(1,0,1),(2,1,1),(1,1,0),(2,1,2),(3,1,1),(1,2,1),(0,1,2)
    ])
    MAXITER: int = 200

    # 暖機（每段）
    WARM_K_GRID: List[float] = field(default_factory=lambda: list(np.logspace(-5, 0, 40)))
    MAX_WARM_PRED_STEPS: int = 80

    # EXOG 清理門檻
    MIN_KEEP_RATIO: float = 0.60

    # 變溫上下文小回測長度
    BACKTEST_B: int = 30

    # 驗證設定
    HOLDOUT_K: int = 8
    RANDOM_SEED: Optional[int] = None
    HOLDOUT_FILES: Optional[List[str]] = None

    # Debug：只對檔名包含此關鍵字者輸出逐點診斷
    DEBUG_RUN_KEYWORD: str = ""


CFG = Config()


# ==========================
# 工具
# ==========================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))


def _tc_cols(df): return [c for c in df.columns if c.upper().startswith("TC")]
def _pt_cols(df): return [c for c in df.columns if c.upper().startswith("PT")]


def read_runs(path: str, time_col: str = "Time") -> Tuple[pd.DataFrame, List[str]]:
    files = sorted([p for p in glob.glob(os.path.join(path, "*.csv"))])
    all_df = []
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        df["run_id"] = os.path.basename(fp)
        all_df.append(df)
    if not all_df:
        raise FileNotFoundError(f"在 {path} 找不到 .csv 檔")
    big = pd.concat(all_df, axis=0, ignore_index=True)
    return big, [os.path.basename(p) for p in files]


def compute_regime_thresholds(big_df: pd.DataFrame, win: int,
                              warm_q: float, flat_q: float, high_vol_q: float) -> Tuple[float,float,float]:
    # 用 TC_mean 的 rolling slope 與 rolling std 做門檻
    df = big_df.copy()
    tcs = _tc_cols(df)
    if not tcs:
        # 若沒有溫度，退化用目標的 rolling
        tcols = [c for c in ["Disp. X","Disp. Z"] if c in df.columns]
        if not tcols:
            raise ValueError("無 TC_* 也無目標欄，無法估門檻")
        Tmean = df[tcols].mean(axis=1)
    else:
        Tmean = df[tcs].mean(axis=1)

    slope = Tmean.diff().rolling(win, min_periods=max(2, win//2)).mean().abs()
    vol   = Tmean.rolling(win, min_periods=max(2, win//2)).std()

    thr_warm = float(np.nanquantile(slope, warm_q))
    thr_flat = float(np.nanquantile(slope, flat_q))
    thr_vol  = float(np.nanquantile(vol,   high_vol_q))
    return thr_warm, thr_flat, thr_vol


def label_regime_for_run(df: pd.DataFrame, win: int, thr_warm: float, thr_flat: float, thr_vol: float,
                         const_vol_factor: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    tcs = _tc_cols(out)
    if tcs:
        Tmean = out[tcs].mean(axis=1)
    else:
        tcols = [c for c in ["Disp. X","Disp. Z"] if c in out.columns]
        Tmean = out[tcols].mean(axis=1) if tcols else pd.Series([0.0]*len(out))

    slope = Tmean.diff().rolling(win, min_periods=max(2, win//2)).mean().abs()
    vol   = Tmean.rolling(win, min_periods=max(2, win//2)).std()

    reg = np.array(["const"] * len(out), dtype=object)
    reg[(slope >= thr_warm)] = "warmup"
    reg[(slope <= thr_flat) & (vol >= thr_vol*const_vol_factor)] = "variable"
    out["regime"] = reg
    return out


# ---- 特徵與標準化 ----
def add_derived_features(df: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    out = df.copy()
    tcs = _tc_cols(out); pts = _pt_cols(out)
    if tcs:
        Tmean = out[tcs].mean(axis=1)
        out["TC_mean"]  = Tmean
        out["TC_slope"] = Tmean.diff().rolling(win, min_periods=max(2, win//2)).mean()
        out["TC_vol"]   = Tmean.rolling(win, min_periods=max(2, win//2)).std()
        # 動態特徵
        out["dTC"] = out["TC_mean"].diff()
        out["dTC_pos"] = out["dTC"].clip(lower=0)
        out["dTC_neg"] = (-out["dTC"].clip(upper=0))
        out["cum_dTC_pos"] = out.groupby("run_id")["dTC_pos"].cumsum()
        out["cum_dTC_neg"] = out.groupby("run_id")["dTC_neg"].cumsum()
        out["Tnorm"] = out["TC_mean"] - out.groupby("run_id")["TC_mean"].transform("first")
    if pts:
        out["PT_mean"] = out[pts].mean(axis=1)
    mcols = [c for c in ["Spindle Motor","X Motor","Z Motor"] if c in out.columns]
    if mcols:
        out["Motor_sum"] = out[mcols].sum(axis=1)
        out["dMotor"] = out["Motor_sum"].diff()
    return out


def add_lag_features_per_run(df: pd.DataFrame, base_cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    if not base_cols: return out
    grouped = out.groupby("run_id", as_index=False, group_keys=False)
    for L in lags:
        def _lag(g):
            for c in base_cols:
                if c in g.columns:
                    g[f"{c}_lag{L}"] = g[c].shift(L)
            return g
        out = grouped.apply(_lag)
    return out


def zscore_fit_transform(tr_df: pd.DataFrame, va_df: pd.DataFrame, exog_cols: List[str]):
    stats = {}
    tr = tr_df.copy(); va = va_df.copy()
    for c in exog_cols:
        if c not in tr.columns: continue
        x = tr[c].astype(float).replace([np.inf,-np.inf], np.nan)
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd == 0: sd = 1.0
        stats[c] = (mu, sd)
        tr[c] = (x - mu) / sd
        if c in va.columns:
            xv = va[c].astype(float).replace([np.inf,-np.inf], np.nan)
            va[c] = (xv - mu) / sd
    return tr, va, stats


def zscore_apply(df: pd.DataFrame, stats: Dict[str, Tuple[float,float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (mu, sd) in stats.items():
        if c in out.columns:
            x = out[c].astype(float).replace([np.inf,-np.inf], np.nan)
            sd = 1.0 if (sd is None or not np.isfinite(sd) or sd == 0.0) else float(sd)
            out[c] = (x - float(mu)) / sd
    return out


def clean_exog_train(tr_df: pd.DataFrame, exog_cols: List[str], min_keep_ratio: float = 0.6) -> List[str]:
    exog_cols = [c for c in exog_cols if c in tr_df.columns]
    if not exog_cols: return []
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
    out = X.copy()
    for c in out.columns:
        out[c] = out[c].replace([np.inf,-np.inf], np.nan)
    out = out.ffill().bfill()
    for c in out.columns:
        if out[c].isna().any():
            med = float(out[c].median(skipna=True)); med = 0.0 if not np.isfinite(med) else med
            out[c] = out[c].fillna(med)
    return out.astype(float)


def auto_exog_for_target(df: pd.DataFrame, candidates: List[str], target: str, k: int,
                         priority_keep: Optional[List[str]] = None) -> List[str]:
    if priority_keep is None: priority_keep = []
    y = df[target].astype(float)
    scores = []
    for c in candidates:
        try:
            sc = float(abs(y.corr(df[c].astype(float))))
        except Exception:
            sc = 0.0
        scores.append((c, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = [c for c,_ in scores[:k]]
    # 物理白名單加回
    for p in priority_keep:
        if p in candidates and p not in chosen:
            chosen.append(p)
    return chosen


# ---- 暖機擬合（用前綴） ----
def fit_warm_prefix_choose_k(y_prefix: np.ndarray, k_grid: List[float]) -> Tuple[float, float, float]:
    """
    模型：y ≈ b0 + b1 * exp(-k * t)，用最小平方找 (b0,b1)；掃 k_grid 取 SSE 最小。
    """
    y = np.asarray(y_prefix, dtype=float)
    n = len(y); t = np.arange(n, dtype=float)
    best = (k_grid[0], np.nan, np.nan, np.inf)
    for k in k_grid:
        X = np.stack([np.ones(n), np.exp(-k*t)], axis=1)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            b0, b1 = float(beta[0]), float(beta[1])
            sse = float(((X @ beta) - y).T @ ((X @ beta) - y))
            if sse < best[3]:
                best = (k, b0, b1, sse)
        except Exception:
            continue
    return best[0], best[1], best[2]


def warm_predict_segment(b0: float, b1: float, k: float, seg_len: int, start_pred: int) -> np.ndarray:
    """
    從段內相對索引 start_pred 起做外推，長度到 seg_len-1。
    """
    t = np.arange(seg_len, dtype=float)
    yfit = b0 + b1 * np.exp(-k * t)
    yhat = yfit[start_pred:]
    return yhat


# ---- 模型訓練 ----
def fit_ucm_const_ll(y: pd.Series, X: Optional[pd.DataFrame], maxiter: int):
    model = UnobservedComponents(endog=y.astype(float).values,
                                 level="local linear trend",
                                 autoregressive=1,
                                 exog=(X.astype(float).values if X is not None else None))
    try:
        res = model.fit(maxiter=maxiter, disp=False)
    except Exception:
        res = model.fit(disp=False)
    return res


def fit_sarimax_variable(y: pd.Series, X: Optional[pd.DataFrame],
                         grid: List[Tuple[int,int,int]], maxiter: int):
    yv = y.astype(float).values
    ex = (X.astype(float).values if X is not None else None)
    best = None
    for (p,d,q) in grid:
        try:
            m = SARIMAX(endog=yv, exog=ex, order=(p,d,q), trend="n", enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(maxiter=maxiter, disp=False)
            aic = float(r.aic) if np.isfinite(r.aic) else np.inf
            if (best is None) or (aic < best[0]):
                best = (aic, r)
        except Exception:
            continue
    if best is None:
        # 後備：ARIMA(1,1,1)
        m = SARIMAX(endog=yv, exog=ex, order=(1,1,1), trend="n",
                    enforce_stationarity=False, enforce_invertibility=False)
        best = (np.inf, m.fit(disp=False))
    return best[1]


# ---- 三模型 + 融合 + 暖啟動 + 上下文小回測 ----
def predict_run_three_models(res_const, res_var_smx, res_var_ll,
                             y_run: pd.Series, X_run: Optional[pd.DataFrame],
                             reg_series: pd.Series, cfg: Config):
    """
    回傳：y_true_eval, yhat_eval, eval_mask
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

    # 評分起點（暖機後）
    yhat = np.full(n, np.nan)
    eval_mask = np.zeros(n, dtype=bool)
    eval_start_global = min(max(1, cfg.WARMUP_STEPS), n-1)
    ctx = cfg.CONTEXT_STEPS
    B = cfg.BACKTEST_B

    for (b_start, b_end, b_reg) in blocks:
        start_pred = max(b_start, eval_start_global) if b_start == 0 else b_start
        if start_pred > b_end:
            continue

        # 上下文
        ctx_start = max(0, start_pred - ctx)
        y_ctx = y.iloc[ctx_start:b_end+1]
        X_ctx = X.iloc[ctx_start:b_end+1] if X is not None else None
        rel_start_ctx = start_pred - ctx_start
        rel_end_ctx   = (b_end - ctx_start)

        if b_reg == "warmup":
            # 用暖機前綴擬合 + 後續用 const 模型補
            y_prefix = y.iloc[ctx_start:start_pred].values
            if len(y_prefix) >= 5:
                k_sel, b0, b1 = fit_warm_prefix_choose_k(y_prefix, cfg.WARM_K_GRID)
                max_len = (b_end - ctx_start + 1)
                yhat_w = warm_predict_segment(b0, b1, k_sel, seg_len=max_len, start_pred=rel_start_ctx)
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
                    pred = app.get_prediction(start=rel_start_ctx2,
                                              end=min(rel_end_ctx, n_eff-1),
                                              dynamic=rel_start_ctx2)
                    yhat_c = np.asarray(pred.predicted_mean)
                    Lc = len(yhat_c)
                    yhat[rest_start:rest_start+Lc] = yhat_c
                    eval_mask[rest_start:rest_start+Lc] = True

        elif b_reg == "const":
            app = res_const.apply(endog=y_ctx, exog=X_ctx)
            n_eff = int(getattr(app, "nobs", len(y_ctx)))
            if rel_start_ctx < n_eff:
                pred = app.get_prediction(start=rel_start_ctx,
                                          end=min(rel_end_ctx, n_eff-1),
                                          dynamic=rel_start_ctx)
                yhat_seg = np.asarray(pred.predicted_mean)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg
                eval_mask[start_pred:start_pred+L] = True

        else:  # variable（變溫）
            app_smx = res_var_smx.apply(endog=y_ctx, exog=X_ctx)
            app_ll  = res_var_ll .apply(endog=y_ctx, exog=X_ctx)
            n_eff_s = int(getattr(app_smx, "nobs", len(y_ctx)))
            n_eff_l = int(getattr(app_ll,  "nobs", len(y_ctx)))
            n_eff   = min(n_eff_s, n_eff_l)

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
            else:
                ws = wl = 0.5

            if rel_start_ctx < n_eff:
                pred_s = app_smx.get_prediction(start=rel_start_ctx,
                                                end=min(rel_end_ctx, n_eff-1),
                                                dynamic=rel_start_ctx).predicted_mean
                pred_l = app_ll .get_prediction(start=rel_start_ctx,
                                                end=min(rel_end_ctx, n_eff-1),
                                                dynamic=rel_start_ctx).predicted_mean
                yhat_seg = ws * np.asarray(pred_s) + wl * np.asarray(pred_l)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg
                eval_mask[start_pred:start_pred+L] = True

    # ====== 暖機後全區間評分，缺值回填 ======
    desired_eval = np.zeros(n, dtype=bool)
    desired_eval[eval_start_global:] = True
    # 先回填
    for i in range(eval_start_global, n):
        if not np.isfinite(yhat[i]):
            prev_pred = yhat[i-1] if i > 0 else np.nan
            if np.isfinite(prev_pred):
                yhat[i] = prev_pred
            else:
                yhat[i] = float(y.iloc[i-1]) if i > 0 else float(y.iloc[0])

    m = desired_eval
    return y.iloc[m].values, yhat[m], m


def predict_run_three_models_full(res_const, res_var_smx, res_var_ll,
                                  y_run: pd.Series, X_run: Optional[pd.DataFrame],
                                  reg_series: pd.Series, cfg: Config):
    """
    回傳：yhat_full（全長，暖機後已回填）、eval_start_global
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

    yhat = np.full(n, np.nan, dtype=float)
    eval_start_global = cfg.WARMUP_STEPS
    ctx = cfg.CONTEXT_STEPS
    B   = cfg.BACKTEST_B

    for (b_start, b_end, b_reg) in blocks:
        start_pred = max(b_start, eval_start_global) if b_start == 0 else b_start
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
                max_len = (b_end - ctx_start + 1)
                yhat_w = warm_predict_segment(b0, b1, k_sel, seg_len=max_len, start_pred=rel_start_ctx)
                Lw = len(yhat_w)
                if Lw > 0:
                    yhat[start_pred:start_pred+Lw] = yhat_w
                rest_start = start_pred + Lw
            else:
                rest_start = start_pred

            if rest_start <= b_end:
                app = res_const.apply(endog=y.iloc[ctx_start:b_end+1],
                                      exog=(X.iloc[ctx_start:b_end+1] if X is not None else None))
                n_eff = int(getattr(app, "nobs", b_end+1-ctx_start))
                if (rest_start - ctx_start) < n_eff:
                    rel_start_ctx2 = rest_start - ctx_start
                    pred = app.get_prediction(start=rel_start_ctx2,
                                              end=min(rel_end_ctx, n_eff-1),
                                              dynamic=rel_start_ctx2)
                    yhat_c = np.asarray(pred.predicted_mean)
                    Lc = len(yhat_c)
                    yhat[rest_start:rest_start+Lc] = yhat_c

        elif b_reg == "const":
            app = res_const.apply(endog=y_ctx, exog=X_ctx)
            n_eff = int(getattr(app, "nobs", len(y_ctx)))
            if rel_start_ctx < n_eff:
                pred = app.get_prediction(start=rel_start_ctx,
                                          end=min(rel_end_ctx, n_eff-1),
                                          dynamic=rel_start_ctx)
                yhat_seg = np.asarray(pred.predicted_mean)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg

        else:
            app_smx = res_var_smx.apply(endog=y_ctx, exog=X_ctx)
            app_ll  = res_var_ll .apply(endog=y_ctx, exog=X_ctx)
            n_eff_s = int(getattr(app_smx, "nobs", len(y_ctx)))
            n_eff_l = int(getattr(app_ll,  "nobs", len(y_ctx)))
            n_eff   = min(n_eff_s, n_eff_l)

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
            else:
                ws = wl = 0.5

            if rel_start_ctx < n_eff:
                pred_s = app_smx.get_prediction(start=rel_start_ctx,
                                                end=min(rel_end_ctx, n_eff-1),
                                                dynamic=rel_start_ctx).predicted_mean
                pred_l = app_ll .get_prediction(start=rel_start_ctx,
                                                end=min(rel_end_ctx, n_eff-1),
                                                dynamic=rel_start_ctx).predicted_mean
                yhat_seg = ws * np.asarray(pred_s) + wl * np.asarray(pred_l)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg

    # 暖機後回填
    for i in range(eval_start_global, n):
        if not np.isfinite(yhat[i]):
            prev_pred = yhat[i-1] if i > 0 else np.nan
            if np.isfinite(prev_pred):
                yhat[i] = prev_pred
            else:
                yhat[i] = float(y.iloc[i-1]) if i > 0 else float(y.iloc[0])
    return yhat, eval_start_global


# ==========================
# 主流程（訓練＋驗證）
# ==========================
def pick_holdout_files(files: List[str], k: int, seed: Optional[int], preset: Optional[List[str]]) -> List[str]:
    if preset:
        return [f for f in preset if f in files]
    rng = np.random.default_rng(seed)
    if k >= len(files): return files[:]
    idx = rng.choice(len(files), size=k, replace=False)
    return [files[i] for i in idx]


def main(cfg: Config):
    print(f"[VERSION] v10-meta-blend-LLTAR1 | WARMUP={cfg.WARMUP_STEPS} | CTX={cfg.CONTEXT_STEPS} | LAGS={cfg.LAGS}")
    # 讀訓練
    big_raw, file_list = read_runs(cfg.TRAIN_DIR, time_col=cfg.TIME_COL)
    print(f"[INFO] 共有 {len(file_list)} 檔；驗證抽 {cfg.HOLDOUT_K} 檔")

    # 門檻
    thr_warm, thr_flat, thr_vol = compute_regime_thresholds(big_raw, cfg.REGIME_ROLL_WIN,
                                                            cfg.WARM_SLOPE_Q, cfg.FLAT_SLOPE_Q, cfg.HIGH_VOL_Q)
    # 標註
    big_reg  = big_raw.groupby("run_id", as_index=False, group_keys=False).apply(
        lambda g: label_regime_for_run(g, cfg.REGIME_ROLL_WIN, thr_warm, thr_flat, thr_vol, cfg.CONST_VOL_FACTOR)
    )
    # 特徵
    big_feat = add_derived_features(big_reg, win=cfg.REGIME_ROLL_WIN)

    # 基礎候選（去掉目標、時間、regime）
    drop_cols = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
    exog_candidates_all = [c for c in big_feat.columns if c not in drop_cols]

    # 加 lag
    big_feat = add_lag_features_per_run(big_feat, exog_candidates_all, cfg.LAGS)

    # holdout
    holdout = pick_holdout_files(file_list, cfg.HOLDOUT_K, cfg.RANDOM_SEED, cfg.HOLDOUT_FILES)
    tr_df = big_feat[~big_feat["run_id"].isin(holdout)].copy()
    va_df = big_feat[ big_feat["run_id"].isin(holdout)].copy()

    # 乾淨 exog + 標準化
    exog_candidates_all = [c for c in big_feat.columns if c not in drop_cols]
    keep_cols = clean_exog_train(tr_df, exog_candidates_all, cfg.MIN_KEEP_RATIO)
    keep_cols = [c for c in keep_cols if c in exog_candidates_all]
    va_df[keep_cols] = clean_exog_apply(va_df[keep_cols])

    tr_df, va_df, zstats = zscore_fit_transform(tr_df, va_df, keep_cols)

    # regime 比例
    cnt = va_df["regime"].value_counts(normalize=True).to_dict()
    print(f"[VAL regime] warmup={cnt.get('warmup',0):.2%}, const={cnt.get('const',0):.2%}, variable={cnt.get('variable',0):.2%}")
    print(f"[INFO] 候選 EXOG 總數（含 lag）= {len(keep_cols)}")

    # 訓練模型
    models_map: Dict[str, tuple] = {}
    exog_map: Dict[str, List[str]] = {}

    contest = {"sse":0.0, "n":0}
    for target in cfg.TARGET_COLS:
        print("\n" + "="*70)
        print(f"[TARGET] {target}")

        topk = cfg.AUTO_EXOG_TOPK_X if target == "Disp. X" else cfg.AUTO_EXOG_TOPK_Z
        priority = [c for c in ["TC_mean","TC_slope","TC_vol","Motor_sum","dMotor","PT_mean","dTC","Tnorm",
                                "dTC_pos","dTC_neg","cum_dTC_pos","cum_dTC_neg"]
                    if c in keep_cols]
        exog_cols_t = auto_exog_for_target(tr_df, keep_cols, target, topk, priority_keep=priority)
        print(f"[INFO] 使用 EXOG（候選）= {len(exog_cols_t)} 欄")

        # 分 regime 的資料（避免資料太少時回退全資料）
        sub_c = tr_df[tr_df["regime"]=="const"]; sub_v = tr_df[tr_df["regime"]=="variable"]
        y_c = (sub_c[target] if len(sub_c)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
        X_c = ((sub_c[exog_cols_t] if len(sub_c)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None
        y_v = (sub_v[target] if len(sub_v)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
        X_v = ((sub_v[exog_cols_t] if len(sub_v)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None

        # 訓練
        print(f"[TRAIN] fit models for {target} ...")
        res_const   = fit_ucm_const_ll(y_c, X_c, cfg.MAXITER)
        res_var_smx = fit_sarimax_variable(y_v, X_v, cfg.SARIMAX_GRID, cfg.MAXITER)
        res_var_ll  = fit_ucm_const_ll(y_v, X_v, cfg.MAXITER)

        models_map[target] = (res_const, res_var_smx, res_var_ll)
        exog_map[target]   = list(exog_cols_t)

        # ===== 驗證：逐檔 RMSE =====
        scores = []
        for run_name, g in va_df.groupby("run_id"):
            X_run = (g[exog_cols_t] if exog_cols_t else None)
            y_true, yhat, eval_mask = predict_run_three_models(res_const, res_var_smx, res_var_ll,
                                                               g[target], X_run, g["regime"], cfg)
            s = rmse(y_true, yhat); scores.append(s)
            contest["sse"] += float(((y_true - yhat)**2).sum()); contest["n"] += len(y_true)
            print(f"[RUN] {run_name} | RMSE={s:.6f}")

        if scores:
            print(f"[RESULT] {target} | mean RMSE = {np.mean(scores):.6f}")
        else:
            print(f"[RESULT] {target} | 無有效驗證檔")

    print("\n=== 賽制版（雙目標合併）平均 RMSE（micro） ===")
    if contest["n"] > 0:
        print(f"RMSE: {(contest['sse']/contest['n'])**0.5:.6f}")
    else:
        print("RMSE: N/A")

    return models_map, exog_map, zstats, (thr_warm, thr_flat, thr_vol)


# ==========================
# Submission 產生
# ==========================
def generate_submission(test_dir: str, sub_dir: str, cfg: Config,
                        models_map: Dict[str, tuple],
                        exog_map: Dict[str, List[str]],
                        zstats: Dict[str, Tuple[float,float]],
                        thr_warm: float, thr_flat: float, thr_vol: float):
    os.makedirs(sub_dir, exist_ok=True)
    files = sorted([p for p in glob.glob(os.path.join(test_dir, "*.csv"))])
    if not files:
        print(f"[SUBMIT] 測試資料夾沒有 .csv：{test_dir}")
        return
    print(f"[SUBMIT] 共有 {len(files)} 檔：")
    for fp in files:
        print("  -", os.path.basename(fp))

    for fp in files:
        try:
            df = pd.read_csv(fp, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)

        run_name = os.path.basename(fp)
        if cfg.TIME_COL in df.columns:
            df = df.sort_values(cfg.TIME_COL).reset_index(drop=True)
        df["run_id"] = run_name

        # 1) 標註 regime
        df_reg  = label_regime_for_run(df, cfg.REGIME_ROLL_WIN, thr_warm, thr_flat, thr_vol, cfg.CONST_VOL_FACTOR)

        # 2) 特徵 + lag
        df_feat = add_derived_features(df_reg, win=cfg.REGIME_ROLL_WIN)
        drop_cols = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
        base_candidates = [c for c in df_feat.columns if c not in drop_cols]
        df_feat = add_lag_features_per_run(df_feat, base_candidates, cfg.LAGS)

        # 3) 標準化（用訓練統計）
        if zstats:
            df_feat = zscore_apply(df_feat, zstats)

        # 4) 逐目標預測，覆寫暖機點之後
        out = df.copy()
        n = len(out)
        for target in cfg.TARGET_COLS:
            if target not in out.columns:
                out[target] = np.nan
            res_tuple = models_map.get(target)
            if res_tuple is None:
                continue
            res_const, res_var_smx, res_var_ll = res_tuple
            exog_cols = [c for c in exog_map.get(target, []) if c in df_feat.columns]
            X_run = (df_feat[exog_cols].astype(float) if exog_cols else None)
            y_run = (df_feat[target].astype(float) if target in df_feat.columns else pd.Series([np.nan]*n))
            yhat_full, eval_start = predict_run_three_models_full(res_const, res_var_smx, res_var_ll,
                                                                  y_run, X_run, df_reg["regime"], cfg)
            start_idx = min(cfg.WARMUP_STEPS, n)
            out.loc[start_idx:, target] = yhat_full[start_idx:]

        # 5) 輸出
        out = out.drop(columns=["run_id"])
        out_path = os.path.join(sub_dir, run_name)
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[SUBMIT] 寫出：{out_path}")


# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=CFG.TRAIN_DIR, help="訓練資料夾（含 train .csv）")
    ap.add_argument("--test",  dest="test_dir", default=None, help="測試資料夾（生成 submission 的來源）")
    ap.add_argument("--sub",   dest="sub_dir",  default=None, help="輸出 submission 的資料夾")
    ap.add_argument("--warmup", type=int, default=CFG.WARMUP_STEPS)
    ap.add_argument("--ctx",    type=int, default=CFG.CONTEXT_STEPS)
    ap.add_argument("--holdout_k", type=int, default=CFG.HOLDOUT_K)
    ap.add_argument("--seed", type=int, default=-1, help="固定驗證抽樣（給 -1 表示不固定）")
    args = ap.parse_args()

    CFG.TRAIN_DIR      = args.train
    CFG.WARMUP_STEPS   = args.warmup
    CFG.CONTEXT_STEPS  = args.ctx
    CFG.TEST_DIR       = args.test_dir
    CFG.SUBMISSION_DIR = args.sub_dir
    CFG.HOLDOUT_K      = args.holdout_k
    CFG.RANDOM_SEED    = (None if args.seed == -1 else int(args.seed))

    # === 訓練 + 驗證 ===
    models_map, exog_map, zstats, thr_tuple = main(CFG)

    # === 產生 submission（若提供 test 與 sub 參數）===
    if CFG.TEST_DIR and CFG.SUBMISSION_DIR:
        print("\n=== 產生 submission ===")
        generate_submission(CFG.TEST_DIR, CFG.SUBMISSION_DIR, CFG,
                            models_map, exog_map, zstats,
                            *thr_tuple)
    else:
        print("\n[SUBMIT] 未提供 --test 或 --sub，略過 submission 輸出。")
