# -*- coding: utf-8 -*-
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
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX

VERSION = "v9-blend-LLTAR1-dynfeat+submit"

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
    RANDOM_SEED: Optional[int] = 42   # None→每次隨機不同；改成整數可重現
    HOLDOUT_FILES: Optional[List[str]] = None

    # 評分/暖啟動（同時也是測試回填起點）
    WARMUP_STEPS: int = 110
    CONTEXT_STEPS: int = 240

    # Regime 偵測
    REGIME_ROLL_WIN: int = 110
    CONST_VOL_FACTOR: float = 0.90
    WARM_SLOPE_Q: float = 0.975
    FLAT_SLOPE_Q: float = 0.30
    HIGH_VOL_Q:  float = 0.85

    # EXOG Top-K
    AUTO_EXOG_TOPK_X: int = 22  # X 軸容許多一點
    AUTO_EXOG_TOPK_Z: int = 16
    LAGS: List[int] = field(default_factory=lambda: [1,5,15])

    # SARIMAX 搜尋
    SARIMAX_GRID: List[Tuple[int,int,int]] = field(default_factory=lambda: [
        (1,1,1),(0,1,1),(2,1,1)
    ])
    MAXITER: int = 300

    # 暖機（每段）
    WARM_K_GRID: List[float] = field(default_factory=lambda: list(np.logspace(-5, 0, 40)))
    MAX_WARM_PRED_STEPS: int = 30

    # EXOG 清理門檻
    MIN_KEEP_RATIO: float = 0.72

    # 變溫上下文小回測長度
    BACKTEST_B: int = 80

    # Debug 與印出
    DEBUG_RUN_KEYWORD: str = ""     # 例如 "20200821"；留空則不輸出逐點診斷
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
    big = pd.concat(runs, axis=0, ignore_index=True)
    return big, [os.path.basename(f) for f in files]

def pick_holdout_files(all_files: List[str], k: int, seed: Optional[int], fixed: Optional[List[str]]):
    if fixed:
        missing = [f for f in fixed if f not in all_files]
        if missing: raise ValueError(f"指定的驗證檔不存在：{missing}")
        return sorted(fixed)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    k = min(k, len(all_files))
    return sorted(rng.choice(all_files, size=k, replace=False).tolist())

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def _tc_cols(df): return [c for c in df.columns if c.upper().startswith("TC")]
def _pt_cols(df): return [c for c in df.columns if c.upper().startswith("PT")]

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

def add_lag_features_per_run(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    def _lag_block(g):
        for c in cols:
            if c not in g.columns: continue
            for L in lags:
                g[f"{c}_lag{L}"] = g[c].shift(L)
        return g
    out = out.groupby("run_id", as_index=False, group_keys=False).apply(_lag_block)
    lag_cols = [f"{c}_lag{L}" for c in cols for L in lags if f"{c}_lag{L}" in out.columns]
    if lag_cols:
        out[lag_cols] = out[lag_cols].fillna(method="ffill")
    return out

def zscore_fit_transform(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float,float]]]:
    stats = {}
    tr = train_df.copy(); va = val_df.copy()
    for c in cols:
        mu = float(tr[c].mean(skipna=True)); sd = float(tr[c].std(skipna=True))
        if not np.isfinite(sd) or sd == 0.0: sd = 1.0
        stats[c] = (mu, sd)
        tr[c] = (tr[c]-mu)/sd
        va[c] = (va[c]-mu)/sd
    return tr, va, stats

def zscore_apply(df: pd.DataFrame, cols: List[str], stats: Dict[str, Tuple[float,float]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            # 若測試檔沒有該欄，就補成 0（= (mean-mean)/std）
            out[c] = 0.0
            continue
        mu, sd = stats.get(c, (0.0, 1.0))
        if not np.isfinite(sd) or sd == 0.0: sd = 1.0
        out[c] = (out[c].astype(float) - mu) / sd
    return out

# ---- Regime 判定 ----
def compute_regime_thresholds(train_df: pd.DataFrame, window: int, q_warm: float, q_flat: float, q_vol: float):
    tcs = _tc_cols(train_df)
    if not tcs: return 0.0, 0.0, 0.0
    T = train_df[tcs].mean(axis=1)
    slope = T.diff().rolling(window, min_periods=max(2, window//2)).mean()
    vol   = T.rolling(window, min_periods=max(2, window//2)).std()
    thr_warm = float(slope.quantile(q_warm))
    thr_flat = float(slope.abs().quantile(q_flat))
    thr_vol  = float(vol.quantile(q_vol))
    return thr_warm, thr_flat, thr_vol

def label_regime_for_run(df_run: pd.DataFrame, window: int,
                         thr_warm: float, thr_flat: float, thr_vol: float,
                         const_vol_factor: float):
    tcs = _tc_cols(df_run)
    if not tcs:
        out = df_run.copy(); out["regime"] = "variable"; return out
    T = df_run[tcs].mean(axis=1)
    slope = T.diff().rolling(window, min_periods=max(2, window//2)).mean()
    vol   = T.rolling(window, min_periods=max(2, window//2)).std()
    warm = (slope > thr_warm)
    const= (slope.abs() <= thr_flat) & (vol <= thr_vol * const_vol_factor)
    reg = np.where(warm.fillna(False), "warmup", np.where(const.fillna(False), "const", "variable"))
    out = df_run.copy(); out["regime"] = reg
    return out

# ---- EXOG 挑選 ----
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

# ================== 模型 ==================
def fit_ucm_const_ll(y: pd.Series, X: Optional[pd.DataFrame], maxiter: int):
    """UCM-LLT with AR(1) residual"""
    try:
        mod = UnobservedComponents(y, level="local linear trend", autoregressive=1, exog=X)
        return mod.fit(disp=False, maxiter=maxiter)
    except Exception:
        mod = UnobservedComponents(y, level="local linear trend", autoregressive=1)
        return mod.fit(disp=False, maxiter=maxiter)

def fit_sarimax_variable(y: pd.Series, X: Optional[pd.DataFrame], grid: List[Tuple[int,int,int]], maxiter: int):
    best_aic, best_res = np.inf, None
    for (p,d,q) in grid:
        try:
            mod = SARIMAX(y, order=(p,d,q), exog=X,
                          enforce_stationarity=False, enforce_invertibility=False,
                          simple_differencing=True, concentrate_scale=True)
            res = mod.fit(disp=False, maxiter=maxiter, method="lbfgs")
            if np.isfinite(res.aic) and (res.aic < best_aic):
                best_aic, best_res = res.aic, res
        except Exception:
            continue
    if best_res is None:
        mod = SARIMAX(y, order=(1,1,1), exog=X,
                      enforce_stationarity=False, enforce_invertibility=False,
                      simple_differencing=True, concentrate_scale=True)
        best_res = mod.fit(disp=False, maxiter=maxiter, method="lbfgs")
    return best_res

# ---- 暖機（每段）----
def _fit_exp_ls(y: np.ndarray, k: float) -> Tuple[float, float]:
    n = len(y); t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), np.exp(-k * t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])

def _sse_exp_ls(y: np.ndarray, k: float) -> Tuple[float, float, float]:
    beta0, beta1 = _fit_exp_ls(y, k)
    n = len(y); t = np.arange(n, dtype=float)
    yhat = beta0 + beta1 * np.exp(-k * t)
    sse = float(((y - yhat)**2).sum())
    return sse, beta0, beta1

def fit_warm_prefix_choose_k(y_prefix: np.ndarray, k_grid: List[float]) -> Tuple[float, float, float]:
    y_prefix = pd.Series(y_prefix).astype(float).fillna(method="ffill").values
    best = (None, None, None, np.inf)
    for k in k_grid:
        sse, b0, b1 = _sse_exp_ls(y_prefix, k)
        if sse < best[3]:
            best = (k, b0, b1, sse)
    if best[0] is None:
        return 0.01, float(y_prefix[-1]), 0.0
    return float(best[0]), float(best[1]), float(best[2])

def warm_predict_segment(beta0: float, beta1: float, k: float, seg_len: int, start_pred: int) -> np.ndarray:
    t = np.arange(seg_len, dtype=float)
    yhat = beta0 + beta1 * np.exp(-k * t)
    return yhat[start_pred:]

# ---- 三模型 + 融合 + 暖啟動 + 上下文小回測 ----
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
            app = res_const.apply(endog=y_ctx, exog=X_ctx)
            n_eff = int(getattr(app, "nobs", len(y_ctx)))
            if rel_start_ctx < n_eff:
                pred = app.get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx)
                yhat_seg = np.asarray(pred.predicted_mean)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg
                eval_mask[start_pred:start_pred+L] = True

        else:  # variable → SARIMAX & LLT 融合
            app_smx = res_var_smx.apply(endog=y_ctx, exog=X_ctx)
            app_ll  = res_var_ll .apply(endog=y_ctx, exog=X_ctx)
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
                if verbose_blend:
                    print(f"[VAR-BLEND] ws={ws:.2f} wl={wl:.2f} | btRMSE SMX={rmse_s:.4f} LLT={rmse_l:.4f} | block=({b_start}-{b_end}) len={b_end-b_start+1}")
            else:
                ws = wl = 0.5
                if verbose_blend:
                    print(f"[VAR-BLEND] (no BT) ws=0.50 wl=0.50 | block=({b_start}-{b_end}) len={b_end-b_start+1}")

            if rel_start_ctx < n_eff:
                pred_s = app_smx.get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx).predicted_mean
                pred_l = app_ll .get_prediction(start=rel_start_ctx, end=min(rel_end_ctx, n_eff-1), dynamic=rel_start_ctx).predicted_mean
                yhat_seg = ws * np.asarray(pred_s) + wl * np.asarray(pred_l)
                L = len(yhat_seg)
                yhat[start_pred:start_pred+L] = yhat_seg
                eval_mask[start_pred:start_pred+L] = True

    # ====== 暖機後全區間，缺值以「上一個預測→上一個實際」回填 ======
    desired_eval = np.zeros(n, dtype=bool)
    desired_eval[eval_start_global:] = True
    for i in range(eval_start_global, n):
        if not np.isfinite(yhat[i]):
            prev_pred = yhat[i-1] if i > 0 else np.nan
            if np.isfinite(prev_pred):
                yhat[i] = prev_pred
            else:
                yhat[i] = float(y.iloc[i-1]) if i > 0 else float(y.iloc[0])

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
    print(f"[REGIME] thr_warm={thr_warm:.4g}, thr_flat={thr_flat:.4g}, thr_vol={thr_vol:.4g}")

    big_reg = big_raw.groupby("run_id", as_index=False, group_keys=False).apply(
        lambda g: label_regime_for_run(g, cfg.REGIME_ROLL_WIN, thr_warm, thr_flat, thr_vol, cfg.CONST_VOL_FACTOR)
    )

    # 特徵 + lag
    big_feat = add_derived_features(big_reg, win=cfg.REGIME_ROLL_WIN)
    drop_cols = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
    base_candidates = [c for c in big_feat.columns if c not in drop_cols]
    big_feat = add_lag_features_per_run(big_feat, base_candidates, cfg.LAGS)

    # 候選 exog（含 lag）
    drop_cols2 = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
    exog_candidates_all = [c for c in big_feat.columns if c not in drop_cols2]

    # 切 train / val（預設隨機，每次不同）
    holdout = pick_holdout_files(file_list, cfg.HOLDOUT_K, cfg.RANDOM_SEED, cfg.HOLDOUT_FILES)
    print(f"[INFO] 共有 {len(file_list)} 檔；驗證抽 {len(holdout)} 檔：")
    for r in holdout: print("  -", r)
    tr_df = big_feat[~big_feat["run_id"].isin(holdout)].copy()
    va_df = big_feat[ big_feat["run_id"].isin(holdout)].copy()

    # 標準化（以訓練統計）
    tr_df[exog_candidates_all] = tr_df[exog_candidates_all].astype(float)
    va_df[exog_candidates_all] = va_df[exog_candidates_all].astype(float)
    tr_df, va_df, zstats = zscore_fit_transform(tr_df, va_df, exog_candidates_all)

    # regime 比例
    cnt = va_df["regime"].value_counts(normalize=True).to_dict()
    print(f"[VAL regime] warmup={cnt.get('warmup',0):.2%}, const={cnt.get('const',0):.2%}, variable={cnt.get('variable',0):.2%}")
    print(f"[INFO] 候選 EXOG 總數（含 lag）= {len(exog_candidates_all)}")

    # 合併 RMSE（micro）
    contest = {"sse":0.0, "n":0}

    # === 針對兩個 target 各自訓練，並保存模型與 EXOG 欄位，用於 submission ===
    trained = {}  # target -> dict(models/exog_cols)
    for target in cfg.TARGET_COLS:
        print("\n" + "="*70)
        print(f"[TARGET] {target}")

        topk = cfg.AUTO_EXOG_TOPK_X if target == "Disp. X" else cfg.AUTO_EXOG_TOPK_Z
        priority = [c for c in ["TC_mean","TC_slope","TC_vol","Motor_sum","PT_mean","dTC","dMotor","Tnorm"] if c in exog_candidates_all]
        exog_cols_t = auto_exog_for_target(tr_df, exog_candidates_all, target, topk, priority_keep=priority)
        print(f"[INFO] 使用 EXOG（候選）= {len(exog_cols_t)} 欄")

        # ★ Disp. X：擴充物理白名單（含動態特徵＋PT02），降低高維不穩
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

        # 訓練 exog 清理（缺值>40% 的欄剔除）
        exog_cols_t = clean_exog_train(tr_df, exog_cols_t, min_keep_ratio=cfg.MIN_KEEP_RATIO)
        print(f"[INFO] 訓練可用 EXOG（清理後）= {len(exog_cols_t)} 欄：{exog_cols_t}")

        # 準備訓練資料
        sub_c = tr_df[tr_df["regime"]=="const"]; sub_v = tr_df[tr_df["regime"]=="variable"]
        y_c = (sub_c[target] if len(sub_c)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
        X_c = ((sub_c[exog_cols_t] if len(sub_c)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None
        y_v = (sub_v[target] if len(sub_v)>=200 else tr_df[target]).astype(float).reset_index(drop=True)
        X_v = ((sub_v[exog_cols_t] if len(sub_v)>=200 else tr_df[exog_cols_t]).astype(float).reset_index(drop=True)) if exog_cols_t else None

        # 最後保險檢查
        for name, Xchk in [("X_const", X_c), ("X_var", X_v)]:
            if Xchk is not None and (np.isinf(Xchk.values).any() or np.isnan(Xchk.values).any()):
                raise ValueError(f"[SANITY] {name} 仍有 NaN/Inf")

        print("[TRAIN] 擬合 const(UCM-LL, AR1) / variable(SARIMAX & UCM-LL, AR1) ...", flush=True)
        res_const   = fit_ucm_const_ll(y_c, X_c, cfg.MAXITER)
        res_var_smx = fit_sarimax_variable(y_v, X_v, cfg.SARIMAX_GRID, cfg.MAXITER)
        res_var_ll  = fit_ucm_const_ll(y_v, X_v, cfg.MAXITER)   # 變溫用 LLT(AR1)
        print("[TRAIN] 完成。", flush=True)

        # 逐檔驗證
        scores = []
        for run_name, g in va_df.groupby("run_id"):
            if len(g) <= cfg.WARMUP_STEPS:
                print(f"[WARN] {run_name} 長度 <= {cfg.WARMUP_STEPS}，無可評分區間，跳過。")
                continue

            y_run = g[target]
            X_run = g[exog_cols_t] if exog_cols_t else None
            reg_s = g["regime"]

            y_true, yhat, eval_mask = predict_run_three_models(res_const, res_var_smx, res_var_ll,
                                                               y_run, X_run, reg_s, cfg,
                                                               verbose_blend=cfg.VERBOSE_BLEND)
            if len(y_true)==0:
                print(f"[WARN] {run_name} 有效樣本不足，跳過。"); continue

            # 逐點診斷（可選）
            if cfg.DEBUG_RUN_KEYWORD and (cfg.DEBUG_RUN_KEYWORD in run_name):
                os.makedirs("debug", exist_ok=True)
                yhat_full = np.full(len(g), np.nan); yhat_full[eval_mask] = yhat
                dbg = g[[cfg.TIME_COL, "regime"]].copy()
                dbg.rename(columns={cfg.TIME_COL: "Time"}, inplace=True)
                dbg["target"] = target
                dbg["y_true"] = g[target].values
                dbg["yhat"]   = yhat_full
                dbg["scored"] = eval_mask.astype(int)
                dbg["err"]    = dbg["yhat"] - dbg["y_true"]
                dbg["abs_err"]= dbg["err"].abs()
                dbg["sq_err"] = dbg["err"]**2
                dbg.to_csv(os.path.join("debug", f"debug_{target}_{run_name.replace('.csv','')}.csv"),
                           index=False, encoding="utf-8-sig")

            s = rmse(y_true, yhat); scores.append(s)
            contest["sse"] += float(((y_true - yhat)**2).sum()); contest["n"] += len(y_true)
            print(f"[RUN] {run_name} | RMSE={s:.6f}")

        if scores:
            print(f"[RESULT] {target} | mean RMSE = {np.mean(scores):.6f}")
        else:
            print(f"[RESULT] {target} | 無有效驗證檔")

        # 保存供 submission 使用
        trained[target] = dict(
            res_const=res_const, res_var_smx=res_var_smx, res_var_ll=res_var_ll,
            exog_cols=exog_cols_t
        )

    print("\n=== 賽制版（雙目標合併）平均 RMSE（micro） ===")
    if contest["n"] > 0:
        print(f"RMSE: {(contest['sse']/contest['n'])**0.5:.6f}")
    else:
        print("RMSE: N/A")

    # =================== 產生 SUBMISSION ===================
    print("\n[STEP] 讀取測試資料並產出 submission…")
    test_raw, test_files = read_runs(cfg.TEST_DIR, time_col=cfg.TIME_COL)

    # 依訓練的門檻標註 regime、建特徵、加 lag、標準化
    test_reg = test_raw.groupby("run_id", as_index=False, group_keys=False).apply(
        lambda g: label_regime_for_run(g, cfg.REGIME_ROLL_WIN, thr_warm, thr_flat, thr_vol, cfg.CONST_VOL_FACTOR)
    )
    test_feat = add_derived_features(test_reg, win=cfg.REGIME_ROLL_WIN)

    drop_cols_t = set([cfg.TIME_COL, "run_id", "regime"] + cfg.TARGET_COLS)
    base_cand_t = [c for c in test_feat.columns if c not in drop_cols_t]
    test_feat = add_lag_features_per_run(test_feat, base_cand_t, cfg.LAGS)

    # 先把所有訓練用的 exog 候選欄位補齊（可能有些在測試檔缺）
    for col in exog_candidates_all:
        if col not in test_feat.columns:
            test_feat[col] = 0.0

    # 標準化（用訓練 zstats）
    test_feat[exog_candidates_all] = test_feat[exog_candidates_all].astype(float)
    test_feat = zscore_apply(test_feat, exog_candidates_all, zstats)

    # 逐檔輸出；兩個 target 分別用各自模型與 exog
    for run_name, g in test_feat.groupby("run_id"):
        # 用原始測試檔來回填目標欄位
        g_raw = test_raw[test_raw["run_id"]==run_name].copy()

        for target in cfg.TARGET_COLS:
            models = trained[target]
            exog_cols = models["exog_cols"]
            X_run = g[exog_cols] if exog_cols else None
            y_run = g_raw[target]
            reg_s = g["regime"]

            # 產生從暖機後的預測序列
            _, yhat, eval_mask = predict_run_three_models(
                models["res_const"], models["res_var_smx"], models["res_var_ll"],
                y_run, X_run, reg_s, cfg, verbose_blend=False
            )

            # 只回填原始為 NaN 且在 eval_mask=True 的位置
            fill_idx = np.where(eval_mask)[0]
            # 對齊 yhat 與 fill_idx
            yhat_full = np.full(len(g_raw), np.nan)
            yhat_full[fill_idx] = yhat

            # 實際回填
            tgt_vals = g_raw[target].astype(float).values
            m_nan = np.isnan(tgt_vals)
            m_fill = m_nan & np.isfinite(yhat_full)
            tgt_vals[m_fill] = yhat_full[m_fill]
            g_raw[target] = tgt_vals

        # 儲存檔案（保留原欄位順序，目標欄位放最後）
        cols = [c for c in g_raw.columns if c not in cfg.TARGET_COLS and c not in ("run_id",)]
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=CFG.TRAIN_DIR, help="訓練資料夾（含 .csv）")
    p.add_argument("--test",  default=CFG.TEST_DIR,  help="測試資料夾（含 .csv）")
    p.add_argument("--out",   default=CFG.SUBMISSION_DIR, help="submission 輸出資料夾")
    p.add_argument("--holdout", nargs="*", default=CFG.HOLDOUT_FILES, help="指定驗證檔名清單（可多個）；不指定則隨機抽取")
    p.add_argument("--seed", type=int, default=CFG.RANDOM_SEED, help="驗證抽樣亂數種子（留空=每次不同）")
    args = p.parse_args()
    CFG.TRAIN_DIR = args.train
    CFG.TEST_DIR  = args.test
    CFG.SUBMISSION_DIR = args.out
    CFG.HOLDOUT_FILES = args.holdout
    CFG.RANDOM_SEED = args.seed
    main(CFG)