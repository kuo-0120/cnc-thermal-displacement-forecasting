# CNC 車床熱位移時間序列預測

預測 CNC 車床刀尖在 X／Z 軸的熱位移。專案整合 LightGBM、LSTM、Ridge、XGBoost、UCM／SARIMAX、遞迴 rollout 與 ensemble 權重搜尋，並使用環境感測特徵描述暖機、常溫與變溫等運轉 regime。

## 問題設定

- 43 個訓練檔的多段時間序列
- 約 25–29 個外生環境與衍生特徵
- 目標：`Disp. X`、`Disp. Z`
- 驗證需模擬測試時的遞迴預測，避免只評估 teacher-forcing 結果

## 已保存結果

| 紀錄來源 | 指標 |
|---|---:|
| 專案實驗紀錄 | validation RMSE 約 2.572 |
| 專案實驗紀錄 | rollout RMSE 約 2.795 |
| `results/metrics/lstm_pro_v2val_S24_H256_E0p1.json` | v2 micro RMSE（X/Z 合併）0.7214 |

上述數值來自不同驗證切分與評分協定，不能直接互相比較。JSON 原始指標保留在 repository 供查核。

## 方法

- lag、rolling statistics 與外生感測特徵
- LSTM 預測位移增量，再以 recursive rollout 還原長期軌跡
- LightGBM／XGBoost／Ridge baseline
- UCM／SARIMAX 對變溫區段建模
- 以 validation prediction 搜尋 X/Z 與模型 ensemble 權重

## 執行

```powershell
$env:CNC_DATA_ROOT = "D:\path\to\cnc-data"
$env:CNC_OUTPUT_DIR = "D:\path\to\outputs"
python -m venv .venv
pip install -r requirements.txt
python experiments/competition/nini_0908_env.py --train "$env:CNC_DATA_ROOT\train" --test "$env:CNC_DATA_ROOT\test" --out "$env:CNC_OUTPUT_DIR"
```

資料目錄預期包含 `train/`、`test/`，環境表可透過 `--env` 或 `CNC_ENV_FILE` 指定。

## 結構

- `experiments/competition/`：主要模型與 ensemble 實驗
- `experiments/prototype/`：早期模型與評分工具
- `results/metrics/`：可重現的驗證指標
- `ensemble_weights.json`：已搜尋的融合權重

原始量測 CSV、submission、虛擬環境與報告中的個人資訊未收錄。
