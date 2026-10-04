# tw-quant：台股選股因子的統計檢定力研究

**結論**：以台股約 20 年的資料，動能、規模、低波動、價值四個文獻因子在事前登記的設計下，全數未通過 t > 3.0 的檢定。這是研究開始前就宣告的預期結果，反映的是檢定力不足，不是因子無效的證明。

## 研究問題

本研究不問「怎麼打敗大盤」，而是問：**在台股可得的資料規模下，選股因子的有效性能否在統計上被驗證？**

事前檢定力分析顯示，16 年樣本的最小可偵測 Sharpe 為 0.71，而文獻因子多在 0.3–0.6。因此研究開始前就宣告：預期得到虛無結果。

## 結果

| 因子 | 探索期 2007–2015（含 2008） | Holdout 2016–2026 | 判定 |
|---|---|---|---|
| 價值 | +8.8%（t = 1.48） | −5.9%（t = −1.21） | 未通過 |
| 低波動 | +4.9%（t = 1.78） | −9.7%（t = −1.79） | 未通過 |
| 規模 | −1.5%（t = −0.19） | −9.6%（t = −1.82） | 未通過 |
| 動能 | −8.6%（t = −1.63） | −9.0%（t = −1.86） | 未通過 |

- 數字為十分位等權組合相對 0050 的年化超額報酬，t 為 Newey-West t 值
- 門檻為 t > 3.0；經 Bonferroni、BH 修正後無一顯著，Deflated Sharpe Ratio 最高 0.34
- 「排除 2008」切法的結論相同，完整結果見 `report.md`（探索期）與 `holdout_report.md`

## 證據時間線：先凍結，再看結果

| 時間（2026-10-04，台灣） | 事件 | 憑證 |
|---|---|---|
| 21:16 | 上傳 protocol、8 個試驗的登記與程式 | commit `2206253` |
| 21:45 | 執行探索期 | `provenance.json` 記錄 `2206253` |
| 21:53 | 上傳探索期結果 | commit `451700f` |
| 21:54 | 執行 holdout（只能一次） | `HOLDOUT_LOCK.json` 記錄 `451700f` |
| 23:34 | 上傳 holdout 結果與鎖檔 | commit `7568b9e` |

三個 commit 都經由 GitHub 網頁上傳，時間由 GitHub 伺服器寫入，無法倒填。

### 自行驗證

```bash
git clone https://github.com/Willie-is-me/tw-quant.git && cd tw-quant
git log --format='%h %cI %s'                  # 三個 commit 的時間
git show 2206253:protocol.md | sha256sum      # dd178744d5c13c40…（探索期執行時的 protocol）
git show 451700f:protocol.md | sha256sum      # 7e194417049e50b4…（holdout 執行時的 protocol）
cat provenance.json holdout_provenance.json   # 執行時間、commit、程式與資料的 SHA-256
for m in universe factors backtest stats run_trial; do python $m.py --self-test; done   # 共 67 項自我測試
```

## 檔案

| 檔案 | 內容 |
|---|---|
| `protocol.md` | 研究協定（事前登記），含修訂記錄（§11）與 holdout 驗證紀錄（§12） |
| `trial_log.md` | 探索期 8 個試驗的事前登記與結果 |
| `universe.py` | point-in-time 資格條件（16 項自我測試） |
| `factors.py` | 四個因子的計算（11 項） |
| `backtest.py` | 回測引擎：VWAP 執行、只交易差額、成本、下市處理（14 項） |
| `stats.py` | Newey-West、Bonferroni／BH、Deflated Sharpe Ratio、stationary bootstrap（14 項） |
| `run_trial.py` | 執行流程與防呆：登記檢查、holdout 鎖、結構乾跑（12 項） |
| `summary.csv`、`report.md`、`monthly.csv`、`picks.csv`、`trades.csv`、`events.csv`、`provenance.json`、`console.log` | 探索期結果 |
| `holdout_*` | holdout 結果 |
| `HOLDOUT_LOCK.json` | holdout 鎖檔（狀態 completed） |

protocol 中提到的 `src/`、`docs/`、`results/` 路徑，在本 repo 中都位於根目錄（網頁上傳時未保留資料夾結構）。

## 重跑

原始資料來自 [FinMind](https://finmindtrade.com/)，不含在 repo 中；各資料檔的 SHA-256 記在 `provenance.json`。有資料後，可以用合成報酬乾跑完整流程（不含任何真實績效）：

```bash
python run_trial.py --phase exploration --data-dir <資料夾> --out results --placebo
```

探索期的 8 列登記已經用掉、holdout 已經上鎖，正式檢定無法重跑，這是設計使然。

## 方法論承諾

- 探索期 2007–2015、holdout 2016–2026，holdout 只驗證一次
- 顯著性門檻 t > 3.0（Harvey, Liu & Zhu 2016），非慣用的 2.0
- 嘗試次數事前登記（N = 8），並列報告 Bonferroni、BH 與 Deflated Sharpe Ratio
- 因子只用文獻已驗證者，不使用均線交叉、MACD、KD
- 不修補個別資料點，一律以 point-in-time 資格條件排除

## 已知限制

- 樣本長度是硬上限，每段檢定期間只有 9–11 年
- holdout 是時間上的保留，不是資訊上的保留
- 下市股以最後成交價出場，對報酬是向上偏誤
- 基準 0050 為市值加權，與等權的因子組合不完全可比

完整清單見 `protocol.md` §10。
