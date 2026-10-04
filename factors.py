#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子計算：protocol.md 第 5 節的四個因子
==================================================================
所有視窗以**選股日 T** 為基準，且只使用 T 之前的資料。
T 之前最後一個交易日記為 t0（= T−1）。

| 因子   | 視窗（相對 T）      | 計算                                   | 取向 |
|--------|--------------------|----------------------------------------|------|
| 動能   | T−273 → T−21       | 還原價累積報酬（跳過最近 21 個交易日） | 高   |
| 低波動 | T−252 → T−1        | 還原價日報酬標準差 × √252              | 低   |
| 規模   | T 前最近有值日     | 未還原收盤價 × 已發行股數              | 低   |
| 價值   | T 前最近有值日     | PBR（官方公布值）                      | 低   |

**動能為何跳過最近一個月**：Jegadeesh & Titman (1993) 的標準做法，
用以避開短期反轉效應。實作上最容易寫錯成「含最近一個月」，
`_test_momentum_skips_recent` 即為此設的迴歸測試。

**十分位的取法**：n 檔中取 floor(n × 0.10 + 0.5) 檔（.5 進位），至少 1 檔。
與 protocol 第 6 節一致。不使用內建 round()，因其為銀行家捨入。

**細節（protocol 第 5 節，2026-10-04 明定）**
- 動能端點（T−273、T−21）當日無成交時，取該日（含）之前最近一個有成交日的
  收盤價（as-of），最多回溯 13 個交易日（= 273 × 5%，與條件 (c) 的容許缺值
  一致）；超過即視為不可計算，由條件 (g) 排除。不使用端點之後的價格
- 低波動以 T−252 至 T−1 共 252 個收盤價計算 251 個日報酬；缺值日前後的報酬不計
- 同值時依股票代號由小到大（穩定排序），結果可完全重現

執行
    python factors.py --self-test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

MOM_START, MOM_END = 273, 21      # T−273 → T−21
VOL_WINDOW = 252                  # T−252 → T−1
ASOF_CAP = 13                     # 動能端點 as-of 最多回溯 13 個交易日（= 273 × 5%）

LOWER_IS_BETTER = {"momentum": False, "lowvol": True,
                   "size": True, "value": True}


def _pos(calendar: pd.DatetimeIndex, T: pd.Timestamp) -> int:
    i = calendar.searchsorted(T)
    if i >= len(calendar) or calendar[i] != T:
        raise KeyError(f"{T} 不在交易日曆中")
    return int(i)


def _asof_row(panel: pd.DataFrame, k: int, cap: int = ASOF_CAP) -> pd.Series:
    """第 k 列（含）之前、最多回溯 cap 列內最近一個有值的列；超過即為缺值。"""
    return panel.iloc[max(0, k - cap): k + 1].ffill().iloc[-1]


def momentum_computable(has: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """動能兩個端點在 as-of 上限內都有成交（供 universe 的條件 (g) 使用）。"""
    i = _pos(has.index, T)
    if i - MOM_START < 0:
        return pd.Series(False, index=has.columns)
    a = has.iloc[max(0, i - MOM_START - ASOF_CAP): i - MOM_START + 1].any()
    b = has.iloc[max(0, i - MOM_END - ASOF_CAP): i - MOM_END + 1].any()
    return a & b


def momentum(close_adj: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    i = _pos(close_adj.index, T)
    if i - MOM_START < 0:
        return pd.Series(np.nan, index=close_adj.columns)
    a = _asof_row(close_adj, i - MOM_START)
    b = _asof_row(close_adj, i - MOM_END)
    return (b / a - 1.0).rename("momentum")


def lowvol(close_adj: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    i = _pos(close_adj.index, T)
    if i - VOL_WINDOW < 0:
        return pd.Series(np.nan, index=close_adj.columns)
    win = close_adj.iloc[i - VOL_WINDOW:i]          # 不含 T
    r = win.pct_change(fill_method=None)
    return (r.std(ddof=1) * np.sqrt(252)).rename("lowvol")


def _as_of(panel, T, name):
    if panel is None:
        return pd.Series(dtype=float, name=name)
    i = _pos(panel.index, T)
    if i == 0:
        return pd.Series(np.nan, index=panel.columns, name=name)
    return panel.iloc[i - 1].rename(name)           # T−1


def size(mcap, T):
    return _as_of(mcap, T, "size")


def value(pbr, T):
    return _as_of(pbr, T, "value")


def compute(panels: dict, T: pd.Timestamp, factor: str) -> pd.Series:
    close = panels["close_adj"]
    if factor == "momentum":
        return momentum(close, T)
    if factor == "lowvol":
        return lowvol(close, T)
    if factor == "size":
        return size(panels.get("mcap"), T)
    if factor == "value":
        return value(panels.get("pbr"), T)
    raise ValueError(f"未知因子：{factor}")


def decile_count(n: int, pct: float = 0.10) -> int:
    """.5 進位，至少 1 檔。不用內建 round()（銀行家捨入 round(40.5)=40）。"""
    return max(1, int(np.floor(n * pct + 0.5)))


def select(values: pd.Series, eligible: pd.Series, factor: str,
           pct: float = 0.10, top_n: int | None = None) -> list[str]:
    """合格股中依因子排序取前 pct（或前 top_n 檔）。"""
    ok = eligible.reindex(values.index).fillna(False).astype(bool)
    v = values[ok].dropna()
    if v.empty:
        return []
    v = v.sort_values(ascending=LOWER_IS_BETTER[factor], kind="mergesort")
    k = top_n if top_n is not None else decile_count(len(v), pct)
    return list(v.index[:k])


def build_portfolios(panels: dict, universe: pd.DataFrame,
                     factors=("momentum", "size", "lowvol", "value"),
                     pct: float = 0.10, top_n: int | None = None) -> pd.DataFrame:
    """universe 為 universe.build() 的輸出（year, T, as_of, factor, stock_id）。"""
    rows = []
    for (y, T, f), grp in universe.groupby(["year", "T", "factor"], sort=True):
        if f not in factors:
            continue
        elig = pd.Series(True, index=pd.Index(grp["stock_id"].unique()))
        vals = compute(panels, pd.Timestamp(T), f)
        picks = select(vals, elig, f, pct, top_n)
        for rank, sid in enumerate(picks, 1):
            rows.append((y, pd.Timestamp(T), f, rank, sid, float(vals.get(sid, np.nan))))
    return pd.DataFrame(rows, columns=["year", "T", "factor",
                                       "rank", "stock_id", "factor_value"])


# ---------------------------------------------------------------- 測試


def _synth(n_stocks=40, n_days=800, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2005-01-03", periods=n_days)
    close = pd.DataFrame(index=cal, columns=[f"{1000+i}" for i in range(n_stocks)],
                         dtype=float)
    for c in close.columns:
        close[c] = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_days)))
    mcap = pd.DataFrame(1e10, index=cal, columns=close.columns)
    pbr = pd.DataFrame(1.5, index=cal, columns=close.columns)
    return {"close_adj": close, "mcap": mcap, "pbr": pbr}, cal


def _t(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _test_momentum_window():
    p, cal = _synth(); i = 700; T = cal[i]
    c = p["close_adj"].copy()
    seg = cal[i - MOM_START:i - MOM_END + 1]
    c.loc[seg, "1005"] *= np.linspace(1, 5, len(seg))
    c.loc[cal[i - MOM_END + 1:], "1005"] *= 5
    return _t("動能：視窗內上漲者排名第一", momentum(c, T).idxmax() == "1005")


def _test_momentum_skips_recent():
    p, cal = _synth(); T = cal[700]
    base = momentum(p["close_adj"], T)
    c = p["close_adj"].copy()
    c.loc[cal[700 - MOM_END + 1:], "1006"] *= 2.0
    return _t("動能：跳過最近 21 日（近月暴漲不影響）",
              np.isclose(base["1006"], momentum(c, T)["1006"]))


def _test_momentum_asof_endpoint():
    """端點無成交：取之前最近一日，不得取之後。"""
    p, cal = _synth(); i = 700; T = cal[i]
    c = p["close_adj"].copy()
    c.iloc[i - MOM_END, c.columns.get_loc("1012")] = np.nan       # T−21 無成交
    c.iloc[i - MOM_START, c.columns.get_loc("1012")] = np.nan     # T−273 無成交
    m = momentum(c, T)["1012"]
    exp = c["1012"].iloc[i - MOM_END - 1] / c["1012"].iloc[i - MOM_START - 1] - 1
    later = c["1012"].iloc[i - MOM_END + 1] / c["1012"].iloc[i - MOM_START + 1] - 1
    # 長期停牌：T−273 之前 14 日以上無成交 -> 不可計算
    c2 = p["close_adj"].copy()
    c2.iloc[i - MOM_START - ASOF_CAP: i - MOM_START + 1, c2.columns.get_loc("1013")] = np.nan
    capped = np.isnan(momentum(c2, T)["1013"])
    comp = momentum_computable(c2.notna(), T)
    return _t("動能：端點無成交時取之前最近一日（不取之後，最多回溯 13 日）",
              np.isclose(m, exp) and not np.isclose(m, later) and capped
              and not comp["1013"] and comp["1012"])


def _test_lowvol():
    p, cal = _synth(); T = cal[700]
    c = p["close_adj"].copy()
    c.loc[cal[700 - VOL_WINDOW:700], "1007"] = np.linspace(50, 51, VOL_WINDOW)
    return _t("低波動：平緩者波動度最小", lowvol(c, T).idxmin() == "1007")


def _test_lowvol_excludes_T():
    p, cal = _synth(); T = cal[700]
    base = lowvol(p["close_adj"], T)
    c = p["close_adj"].copy(); c.loc[cal[700:], "1008"] *= 3.0
    return _t("低波動：T 當日及之後不進入視窗",
              np.isclose(base["1008"], lowvol(c, T)["1008"]))


def _test_size_value_as_of():
    p, cal = _synth(); T = cal[700]
    p["mcap"].loc[:, "1009"] = 1e8
    p["pbr"].loc[:, "1010"] = 0.3
    return _t("規模/價值：取 T−1 的值且最小者可辨識",
              size(p["mcap"], T).idxmin() == "1009"
              and value(p["pbr"], T).idxmin() == "1010")


def _test_as_of_is_before_T():
    p, cal = _synth(); T = cal[700]
    base = size(p["mcap"], T)
    m = p["mcap"].copy(); m.loc[cal[700:], "1011"] = 1.0
    return _t("規模：使用 T−1 而非 T 當日的值",
              np.isclose(base["1011"], size(m, T)["1011"]))


def _test_direction():
    p, cal = _synth(); T = cal[700]
    elig = pd.Series(True, index=p["close_adj"].columns)
    m, v = momentum(p["close_adj"], T), lowvol(p["close_adj"], T)
    tm, tv = select(m, elig, "momentum"), select(v, elig, "lowvol")
    return _t("排序方向：動能取高、低波動取低",
              m[tm].min() >= m.drop(tm).max() and v[tv].max() <= v.drop(tv).min())


def _test_tie_order():
    v = pd.Series([0.8, 0.5, 0.5, 0.5, 0.9], index=["1101", "1104", "1102", "1103", "1105"])
    v = v.sort_index()
    elig = pd.Series(True, index=v.index)
    top2_value = select(v, elig, "value", top_n=2)          # 取低：0.5 有三檔同值
    top2_mom = select(-v, elig, "momentum", top_n=2)        # 取高：同樣三檔同值
    return _t(f"同值依代號由小到大：{top2_value}、{top2_mom}",
              top2_value == ["1102", "1103"] and top2_mom == ["1102", "1103"])


def _test_decile_count():
    cases = [(40, 4), (472, 47), (405, 41), (7, 1), (1, 1), (14, 1), (15, 2)]
    return _t("十分位檔數（.5 進位，至少 1）",
              all(decile_count(n) == k for n, k in cases))


def _test_point_in_time():
    p, cal = _synth(); T = cal[700]
    before = {f: compute(p, T, f) for f in LOWER_IS_BETTER}
    c = p["close_adj"].copy(); c.loc[cal[700:], :] *= 0.1
    m = p["mcap"].copy(); m.loc[cal[700:], :] = 1.0
    b = p["pbr"].copy(); b.loc[cal[700:], :] = 99.0
    after = {f: compute({"close_adj": c, "mcap": m, "pbr": b}, T, f)
             for f in LOWER_IS_BETTER}
    ok = all(np.allclose(before[f].dropna(),
                         after[f].reindex(before[f].dropna().index), equal_nan=True)
             for f in LOWER_IS_BETTER)
    return _t("point-in-time：T 當日及之後的資料不影響任一因子", ok)


def self_test() -> int:
    print("### factors.py 自我驗證 ###\n")
    res = [_test_momentum_window(), _test_momentum_skips_recent(),
           _test_momentum_asof_endpoint(),
           _test_lowvol(), _test_lowvol_excludes_T(),
           _test_size_value_as_of(), _test_as_of_is_before_T(),
           _test_direction(), _test_tie_order(), _test_decile_count(),
           _test_point_in_time()]
    ok = all(res)
    print(f"\n  {sum(res)}/{len(res)} 通過"
          + ("" if ok else " —— 有測試未通過，不可用於真實資料"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    if ap.parse_args().self_test:
        return self_test()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
