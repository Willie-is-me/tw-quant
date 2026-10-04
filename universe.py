#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universe 建構：protocol.md 第 6 節的資格條件
==================================================================
所有條件皆為 point-in-time —— 於選股日 T，僅使用 T 之前的資料
（t0 = T 之前最後一個交易日）。

(0) 母體         四碼普通股：排除 00 開頭（ETF）與 91 開頭（TDR）；
                 t0 當日須有成交
(a) 交易活躍度   T 前 60 個交易日中有成交的日數比例 >= 90%
(b) 流動性       T 前 60 個交易日的日均成交金額（無資料日以 0 計）位於
                 「當日全市場」前 50%，且不低於 500 萬
(c) 資料完整度   T 前 273 個交易日的資料完整度 >= 95%
(d) 歷史長度     T 前至少 273 個交易日
(e) 報酬可信度   T 前 273 個交易日內無「無法解釋的超限報酬」
                 （可解釋 = 除權息日、停牌復牌後首日、資料起點後前 5 個報酬；
                 容差 0.5 個百分點。單日壞值不豁免，見 protocol 3.2）
(f) 資料一致性   T 前 273 個交易日內無 Open 越界
(g) 因子可計算   該因子於 T 有值（PBR <= 0 視為無值；動能端點 as-of
                 最多回溯 13 個交易日，見 factors.py）

**母體 = 四碼普通股**（2026-10-04 補正）：00 開頭為 ETF／受益憑證、
91 開頭為臺灣存託憑證（TDR），在所有條件之前排除，也不進入 (b) 的分母。
真實資料核對：universe 中「從未有 PBR」的 14 檔恰為 8 檔 ETF 與 6 檔 TDR。

**不再使用興櫃名單**（2026-10-04 補正）：TaiwanStockInfo 的 type=emerging
含已轉上市櫃者的歷史列，以「現在的狀態」排除過去年度並非 point-in-time。
興櫃期間一律由 (e)(f) 的行為特徵排除。

**條件 (b) 的分母**（曾實際犯錯）：「全體候選股」= 當日全市場所有滿足
(0)(a)(d)(e)(f) 的普通股，不得以任何已下載或已篩選的子集替代。
(e)(f) 在分母中的角色是「處於興櫃期間」的行為代理（protocol 3.3）：
興櫃不屬上市櫃市場，其成交金額不應拉低上市櫃股票的流動性門檻。
`_test_b_denominator`、`_test_b_emerging_not_in_denominator` 為迴歸測試。

**條件 (e) 不得使用 T 當日的資料**：舊版以「次日是否回補」豁免單日壞值，
既違反 protocol 3.2，又使 t0 的判定取決於 T 的收盤價。已移除，
`_test_e_bad_print` 與 `_test_e_no_lookahead` 為迴歸測試。

make_panels() 的輸出同時供 factors.py 與 backtest.py 使用：
  close_adj  還原收盤價（報酬、動能、低波動）
  close_raw  未還原收盤價（市值、VWAP 換算）
  amount / volume  （VWAP、流動性）

執行
    python universe.py --self-test
    python universe.py --data-dir /content/drive/MyDrive/twmarket --out universe.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factors as F  # noqa: E402  （動能的可計算性與 factors.py 共用同一規則）

LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
LIMIT_BEFORE, LIMIT_AFTER = 0.07, 0.10
LIMIT_TOL = 0.005                 # 容差 0.5 個百分點
SKIP_FIRST_N_DAYS = 6             # 資料起點後前 5 個報酬不計（第 0 列無報酬）

FACTORS = ("momentum", "size", "lowvol", "value")

# 00 開頭：ETF／受益憑證；91 開頭：臺灣存託憑證（TDR）
NON_COMMON = r"^(0|91)"


def is_common(ids) -> np.ndarray:
    """四碼數字且非 ETF／TDR。"""
    s = pd.Index(ids).astype(str)
    return np.asarray(s.str.fullmatch(r"\d{4}") & ~s.str.match(NON_COMMON), dtype=bool)


@dataclass(frozen=True)
class Config:
    liq_lookback: int = 60
    liq_pct: float = 0.50          # 前 50%
    liq_floor: float = 5e6         # 500 萬
    active_min: float = 0.90
    hist_min: int = 273            # 252 + 21
    comp_window: int = 273
    comp_min: float = 0.95
    first_year: int = 2007
    factors: tuple = field(default=FACTORS)


# ---------------------------------------------------------------- 前處理


def price_limit(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return np.where(pd.DatetimeIndex(dates) < LIMIT_CHANGE_DATE,
                    LIMIT_BEFORE, LIMIT_AFTER)


def prepare(adj: pd.DataFrame, div: pd.DataFrame | None) -> pd.DataFrame:
    """在還原價長表上加出 (e)(f) 需要的旗標。每列只用當日與前一列的資料。"""
    d = adj.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("stock_id", sort=False)
    d["seq"] = g.cumcount()
    d["ret"] = g["Close"].pct_change(fill_method=None)

    cal = pd.Index(np.sort(d["Date"].unique()))
    d["rk"] = pd.Series(np.arange(len(cal)), index=cal).reindex(d["Date"]).values
    d["gap"] = d["rk"] - g["rk"].shift(1)

    o, h, l = d["Open"], d["High"], d["Low"]
    d["open_bad"] = ((o > h) | (o < l)).fillna(False).to_numpy(dtype=bool)

    if div is not None and len(div):
        key = set(zip(div["stock_id"].astype(str),
                      pd.to_datetime(div["date"]).dt.normalize()))
        has_div = pd.Series(
            [(s, dt) in key for s, dt in zip(d["stock_id"], d["Date"].dt.normalize())],
            index=d.index)
    else:
        has_div = pd.Series(False, index=d.index)

    lim = price_limit(d["Date"]) + LIMIT_TOL
    over = (d["ret"].abs() > lim).fillna(False)
    d["unexp"] = (over
                  & (d["seq"] >= SKIP_FIRST_N_DAYS)
                  & (d["gap"].fillna(1) <= 1)           # 停牌復牌後首日可解釋
                  & ~has_div)                           # 除權息日可解釋
    return d


def _pivot(df: pd.DataFrame, col: str, agg: str = "last") -> pd.DataFrame:
    return df.pivot_table(index="Date", columns="stock_id", values=col, aggfunc=agg)


def make_panels(adj_flagged: pd.DataFrame,
                raw: pd.DataFrame | None = None,
                per: pd.DataFrame | None = None,
                shares: pd.DataFrame | None = None,
                cfg: Config = Config()) -> dict:
    """把長表轉成 (日期 x 股票) 面板，並預先算好滾動統計。"""
    close = _pivot(adj_flagged, "Close")
    amount = _pivot(adj_flagged, "Amount")
    volume = _pivot(adj_flagged, "Volume")
    has = close.notna()

    cal = close.index
    p = {
        "calendar": cal,
        "close_adj": close,        # factors.py / backtest.py 由此取還原價
        "amount": amount,          # backtest.py 由此算 VWAP
        "volume": volume,
        "has": has,
        "hist": has.cumsum(),
        # (b)：無資料日的成交金額以 0 計（protocol 第 6 節）
        "liq": amount.fillna(0.0).rolling(cfg.liq_lookback,
                                          min_periods=cfg.liq_lookback).mean(),
        "active": (volume.fillna(0) > 0).rolling(
            cfg.liq_lookback, min_periods=cfg.liq_lookback).mean(),
        "comp": has.rolling(cfg.comp_window, min_periods=cfg.comp_window).mean(),
        "unexp_w": _pivot(adj_flagged.assign(
                       _u=adj_flagged["unexp"].astype(float)), "_u", "max")
                     .fillna(0.0).rolling(cfg.comp_window, min_periods=1).sum(),
        "openbad_w": _pivot(adj_flagged.assign(
                       _o=adj_flagged["open_bad"].astype(float)), "_o", "max")
                       .fillna(0.0).rolling(cfg.comp_window, min_periods=1).sum(),
    }

    if raw is not None:
        rr = raw.assign(Date=pd.to_datetime(raw["Date"]),
                        stock_id=raw["stock_id"].astype(str))
        # 不 ffill：VWAP 換算需要「當日」的調整因子
        p["close_raw"] = _pivot(rr, "Close").reindex(index=cal)
    else:
        p["close_raw"] = None

    # 因子值面板（取 T 前最近有值日 -> ffill）
    if shares is not None and len(shares) and p["close_raw"] is not None:
        rawc = p["close_raw"].ffill()
        shr = shares.rename(columns={"date": "Date"})
        shr["Date"] = pd.to_datetime(shr["Date"])
        shr["stock_id"] = shr["stock_id"].astype(str)
        shr_p = _pivot(shr, "NumberOfSharesIssued").reindex(cal).ffill()
        cols = rawc.columns.union(shr_p.columns)
        p["mcap"] = rawc.reindex(columns=cols) * shr_p.reindex(columns=cols)
    else:
        p["mcap"] = None

    if per is not None and len(per):
        pr = per.rename(columns={"date": "Date"})
        pr["Date"] = pd.to_datetime(pr["Date"])
        pr["stock_id"] = pr["stock_id"].astype(str)
        pb = _pivot(pr, "PBR").reindex(cal).ffill()
        p["pbr"] = pb.where(pb > 0)                 # PBR <= 0 視為無值
    else:
        p["pbr"] = None
    return p


# ---------------------------------------------------------------- 資格判定


def eligible(panels: dict, t0: pd.Timestamp,
             factor: str | None = None,
             cfg: Config = Config()) -> pd.Series:
    """回傳 t0 當日各股是否合格的布林 Series。

    factor=None 時只判定 (0)-(f)；指定因子時再加上 (g)。
    """
    has, hist = panels["has"], panels["hist"]
    cols = has.columns

    alive = has.loc[t0] & pd.Series(is_common(cols), index=cols)   # (0)

    cond_d = hist.loc[t0] >= cfg.hist_min                          # (d)
    cond_a = panels["active"].loc[t0].fillna(0) >= cfg.active_min  # (a)

    cond_c = panels["comp"].loc[t0].fillna(0) >= cfg.comp_min    # (c)
    cond_e = panels["unexp_w"].loc[t0].fillna(0) == 0            # (e)
    cond_f = panels["openbad_w"].loc[t0].fillna(0) == 0          # (f)

    # (b) 分母：全市場滿足 (0)(a)(d)(e)(f) 者 —— 見模組說明。
    # (e)(f) 在此是「處於興櫃期間」的行為代理：興櫃不屬上市櫃市場，
    # 其成交金額不參與流動性排序。
    base = alive & cond_a & cond_d
    den = base & cond_e & cond_f
    L = panels["liq"].loc[t0]
    thr = max(L[den].quantile(cfg.liq_pct), cfg.liq_floor) if den.any() else np.inf
    cond_b = L >= thr

    ok = base & cond_b & cond_c & cond_e & cond_f
    if factor is None:
        return ok.fillna(False)

    if factor == "lowvol":
        g = ok                                    # (c)(d) 已保證視窗內有足夠報酬
    elif factor == "momentum":
        cal = has.index
        i0 = cal.get_loc(t0)
        if i0 + 1 >= len(cal):
            return pd.Series(False, index=cols)
        g = ok & F.momentum_computable(has, cal[i0 + 1])   # 端點 as-of 上限 13 日
    elif factor == "size":
        m = panels["mcap"]
        g = ok & (m.loc[t0].reindex(cols).notna() if m is not None else False)
    elif factor == "value":
        v = panels["pbr"]
        g = ok & (v.loc[t0].reindex(cols).notna() if v is not None else False)
    else:
        raise ValueError(f"未知因子：{factor}")
    return g.fillna(False)


def selection_dates(calendar: pd.DatetimeIndex,
                    cfg: Config = Config()) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """每年首個交易日 T，以及其 as_of（T 之前最後一個交易日）。"""
    out = []
    for y in sorted(set(calendar.year)):
        if y < cfg.first_year:
            continue
        days = calendar[calendar.year == y]
        prev = calendar[calendar < days[0]]
        if len(days) == 0 or len(prev) == 0:
            continue
        out.append((y, days[0], prev[-1]))
    return out


def build(panels: dict, cfg: Config = Config()) -> pd.DataFrame:
    """長表：year, T, as_of, factor, stock_id。"""
    rows = []
    for y, T, t0 in selection_dates(panels["calendar"], cfg):
        for f in cfg.factors:
            sel = eligible(panels, t0, f, cfg)
            for sid in sel[sel].index:
                rows.append((y, T, t0, f, sid))
    return pd.DataFrame(rows, columns=["year", "T", "as_of", "factor", "stock_id"])


def summary(u: pd.DataFrame) -> pd.DataFrame:
    t = u.groupby(["year", "factor"]).size().unstack(fill_value=0)
    t["decile"] = np.floor(t.max(axis=1) * 0.1 + 0.5).astype(int)
    return t


# ---------------------------------------------------------------- 合成資料


def _synth(n_stocks: int = 60, n_days: int = 900, seed: int = 0):
    """產生乾淨的合成資料，供各條件測試逐一注入缺陷。

    成交金額隨代號遞增：1000–1029 低於中位數、1030–1059 高於中位數。
    注入缺陷的測試一律使用 1045 以上（流動性充足）的股票，並先確認
    注入前合格 —— 否則測試會因 (b) 而「必然通過」，抓不到任何錯誤。
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2005-01-03", periods=n_days)
    frames = []
    for i in range(n_stocks):
        sid = f"{1000+i}"
        r = rng.normal(0.0002, 0.012, n_days)
        px = 50 * np.exp(np.cumsum(r))
        g = pd.DataFrame({"Date": cal, "Close": px, "stock_id": sid})
        g["Open"] = g.Close * (1 + rng.normal(0, .002, n_days))
        g["High"] = g[["Open", "Close"]].max(axis=1) * 1.004
        g["Low"] = g[["Open", "Close"]].min(axis=1) * 0.996
        # 成交金額直接隨 i 遞增（不經由價格，否則價格漂移會打亂排序）
        g["Amount"] = 1e6 * (i + 1) * rng.uniform(0.9, 1.1, n_days)
        g["Volume"] = g["Amount"] / g["Close"]
        frames.append(g)
    adj = pd.concat(frames, ignore_index=True)
    raw = adj.copy()
    shares = pd.DataFrame([{"date": d, "stock_id": s, "NumberOfSharesIssued": 1e8}
                           for s in adj.stock_id.unique() for d in cal[::20]])
    per = pd.DataFrame([{"date": d, "stock_id": s, "PBR": 1.5}
                        for s in adj.stock_id.unique() for d in cal[::20]])
    return adj, raw, per, shares


def _panels_from(adj, raw=None, per=None, shares=None, cfg=Config()):
    return make_panels(prepare(adj, None), raw, per, shares, cfg)


def _last_t0(panels, cfg=Config()):
    return selection_dates(panels["calendar"], cfg)[-1][2]


def _rows_before(adj, sid, t0, n):
    """sid 在 t0（含）之前的最後 n 列索引。"""
    return adj[(adj.stock_id == sid) & (adj.Date <= t0)].tail(n).index


# ---------------------------------------------------------------- 測試


def _t(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _clean():
    adj, raw, per, shr = _synth()
    p = _panels_from(adj, raw, per, shr)
    t0 = _last_t0(p)
    return adj, raw, per, shr, t0, eligible(p, t0)


def _test_a_active():
    adj, raw, per, shr, t0, e0 = _clean()
    adj.loc[_rows_before(adj, "1045", t0, 60)[:18], ["Volume", "Amount"]] = 0
    e = eligible(_panels_from(adj, raw, per, shr), t0)
    return _t("(a) 活躍度不足者被排除（注入前合格）",
              e0["1045"] and not e["1045"] and e["1050"])


def _test_b_threshold():
    adj, raw, per, shr, t0, e = _clean()
    lo = [s for s in ["1000", "1001", "1002"] if not e.get(s, False)]
    hi = [s for s in ["1057", "1058", "1059"] if e.get(s, False)]
    return _t("(b) 低流動性被排除、高流動性保留", len(lo) == 3 and len(hi) == 3)


def _test_b_denominator():
    """迴歸測試：分母必須是全市場，不得為子集。"""
    adj, raw, per, shr = _synth()
    p_full = _panels_from(adj, raw, per, shr)
    t0 = _last_t0(p_full)
    e_full = eligible(p_full, t0)
    keep = sorted(adj.stock_id.unique())[30:]          # 模擬「以已下載子集為分母」
    p_sub = _panels_from(adj[adj.stock_id.isin(keep)], raw, per, shr)
    e_sub = eligible(p_sub, t0)
    n_full = int(e_full.reindex(keep).fillna(False).sum())
    n_sub = int(e_sub.sum())
    return _t(f"(b) 子集分母會低估合格數（全市場 {n_full} vs 子集 {n_sub}）",
              n_sub < n_full)


def _test_b_emerging_not_in_denominator():
    """興櫃期間的股票（Open 越界）不得拉低上市櫃股票的流動性門檻。"""
    adj, raw, per, shr, t0, e0 = _clean()
    extra = []
    for k in range(20):                         # 20 檔低流動性、具興櫃特徵的股票
        g = adj[adj.stock_id == "1000"].copy()
        g["stock_id"] = f"{7000 + k}"
        g["Open"] = g["Low"] * 0.9               # Open 越界
        extra.append(g)
    adj2 = pd.concat([adj] + extra, ignore_index=True)
    e1 = eligible(_panels_from(adj2, adj2, per, shr), t0)
    return _t("(b) 興櫃特徵股不進入分母（加入 20 檔後普通股合格名單不變）",
              e1.reindex(e0.index).fillna(False).equals(e0)
              and not e1.reindex([f"{7000 + k}" for k in range(20)]).any())


def _test_b_missing_row():
    """迴歸測試：無資料日以成交金額 0 計，不得使 (b) 變成缺值而整檔出局。"""
    adj, raw, per, shr, t0, e0 = _clean()
    adj = adj.drop(_rows_before(adj, "1046", t0, 60)[10:11])
    p = _panels_from(adj, raw, per, shr)
    e = eligible(p, t0)
    return _t(f"(b) 近 60 日缺 1 列仍可合格（流動性 {p['liq'].loc[t0, '1046']/1e6:.1f} 百萬）",
              e0["1046"] and e["1046"] and np.isfinite(p["liq"].loc[t0, "1046"]))


def _test_c_completeness():
    adj, raw, per, shr, t0, e0 = _clean()
    adj = adj.drop(_rows_before(adj, "1047", t0, 273)[:30])   # 視窗前段缺 30 列
    p = _panels_from(adj, raw, per, shr)
    e = eligible(p, t0)
    return _t("(c) 完整度不足者被排除（歷史長度仍足）",
              e0["1047"] and not e["1047"] and p["hist"].loc[t0, "1047"] >= 273)


def _test_d_history():
    adj, raw, per, shr, t0, e0 = _clean()
    keep = _rows_before(adj, "1048", t0, 265)                  # 完整度 265/273 = 97%
    adj = adj[(adj.stock_id != "1048") | adj.index.isin(keep) | (adj.Date > t0)]
    p = _panels_from(adj, raw, per, shr)
    e = eligible(p, t0)
    return _t("(d) 歷史不足者被排除（完整度仍過 95%）",
              e0["1048"] and not e["1048"] and p["comp"].loc[t0, "1048"] >= 0.95)


def _test_e_unexplained():
    adj, raw, per, shr, t0, e0 = _clean()
    row = _rows_before(adj, "1049", t0, 100)[0]
    later = adj[(adj.stock_id == "1049") & (adj.index >= row)].index
    adj.loc[later, ["Open", "High", "Low", "Close"]] *= 1.30   # 不回補的跳空
    e = eligible(_panels_from(adj, raw, per, shr), t0)
    return _t("(e) 無法解釋的超限者被排除", e0["1049"] and not e["1049"])


def _test_e_bad_print():
    """迴歸測試（protocol 3.2）：次日完全回補的單日壞值同樣使該股出局。"""
    adj, raw, per, shr, t0, e0 = _clean()
    row = _rows_before(adj, "1051", t0, 100)[0]
    adj.loc[row, ["Open", "High", "Low", "Close"]] *= 1.30     # 只改一列，次日回補
    e = eligible(_panels_from(adj, raw, per, shr), t0)
    return _t("(e) 單日壞值（次日回補）不豁免", e0["1051"] and not e["1051"])


def _test_e_no_lookahead():
    """t0 的超限是否出局，不得取決於 T 當日是否回補。"""
    adj, raw, per, shr, t0, e0 = _clean()
    row = _rows_before(adj, "1052", t0, 1)[0]                  # t0 當日
    a = adj.copy(); a.loc[row, ["Open", "High", "Low", "Close"]] *= 1.30
    b = adj.copy()
    later = b[(b.stock_id == "1052") & (b.index >= row)].index
    b.loc[later, ["Open", "High", "Low", "Close"]] *= 1.30
    ea = eligible(_panels_from(a, raw, per, shr), t0)
    eb = eligible(_panels_from(b, raw, per, shr), t0)
    return _t("(e) 不使用 T 當日資料（回補與否結果相同）",
              e0["1052"] and ea.equals(eb) and not ea["1052"])


def _test_f_open():
    adj, raw, per, shr, t0, e0 = _clean()
    row = _rows_before(adj, "1053", t0, 50)[0]
    adj.loc[row, "Open"] = adj.loc[row, "Low"] * 0.5
    e = eligible(_panels_from(adj, raw, per, shr), t0)
    return _t("(f) Open 越界者被排除", e0["1053"] and not e["1053"])


def _test_g_factor():
    adj, raw, per, shr = _synth()
    per = per[per.stock_id != "1055"]
    shr = shr[shr.stock_id != "1056"]
    per.loc[per.stock_id == "1054", "PBR"] = 0.0               # PBR <= 0 視為無值
    p = _panels_from(adj, raw, per, shr)
    t0 = _last_t0(p)
    ok = (eligible(p, t0, factor="momentum").get("1055", False)
          and not eligible(p, t0, factor="value").get("1055", False)
          and eligible(p, t0, factor="value").get("1056", False)
          and not eligible(p, t0, factor="size").get("1056", False)
          and eligible(p, t0, factor="momentum").get("1054", False)
          and not eligible(p, t0, factor="value").get("1054", False))
    return _t("(g) 缺因子值者只在該因子被排除；PBR <= 0 視為無值", ok)


def _test_g_momentum_cap():
    """長期停牌後於 T−272 復牌：(c) 仍過，但動能起點超過 as-of 上限 -> 動能不合格。"""
    adj, raw, per, shr, t0, e0 = _clean()
    p0 = _panels_from(adj, raw, per, shr)
    cal = p0["calendar"]
    iT = cal.get_loc(t0) + 1
    gap = cal[iT - F.MOM_START - 300: iT - F.MOM_START + 1]      # 300 日停牌，止於 T−273
    adj = adj[~((adj.stock_id == "1044") & adj.Date.isin(gap))]
    p = _panels_from(adj, raw, per, shr)
    ok = (e0["1044"] and eligible(p, t0, factor="lowvol")["1044"]
          and not eligible(p, t0, factor="momentum")["1044"]
          and p["comp"].loc[t0, "1044"] >= 0.95)
    return _t("(g) 動能端點超過 as-of 上限者只在動能被排除", ok)


NC_IDS = ["0050", "0051", "0052", "0053", "0056",
          "9101", "9103", "9105", "9136", "9151"]


def _test_common_only():
    """迴歸測試：ETF／TDR 不得入選，也不得改變普通股的 (b) 門檻。"""
    adj, raw, per, shr = _synth()
    p0 = _panels_from(adj, raw, per, shr)
    t0 = _last_t0(p0)
    e0 = eligible(p0, t0)
    extra = []
    for k, sid in enumerate(NC_IDS):           # 10 檔流動性最高的非普通股
        g = adj[adj.stock_id == "1059"].copy()
        g["stock_id"] = sid
        g["Amount"] *= 5 + k
        g["Volume"] = g["Amount"] / g["Close"]
        extra.append(g)
    adj2 = pd.concat([adj] + extra, ignore_index=True)
    shr2 = pd.concat([shr, shr[shr.stock_id == "1059"].assign(stock_id="9105")])
    p1 = _panels_from(adj2, adj2, per, shr2)
    e1 = eligible(p1, t0)
    e1s = eligible(p1, t0, factor="size")
    no_nc = not (e1.reindex(NC_IDS).fillna(False).any()
                 or e1s.reindex(NC_IDS).fillna(False).any())
    same = e1.reindex(e0.index).fillna(False).equals(e0)
    common_flags = is_common(["0050", "9105", "9910", "1101", "2881A", "00878"]).tolist()
    return _t("非普通股（00 ETF、91 TDR）排除，且不改變普通股的 (b) 門檻",
              no_nc and same
              and common_flags == [False, False, True, True, False, False])


def _test_point_in_time():
    adj, raw, per, shr = _synth()
    p_a = _panels_from(adj, raw, per, shr)
    t0 = _last_t0(p_a)
    e_a = eligible(p_a, t0)
    fut = adj[(adj.Date > t0)].index                          # 全體股票的未來資料
    adj.loc[fut, ["Open", "High", "Low", "Close"]] *= 0.2
    adj.loc[fut, ["Amount", "Volume"]] = 0
    e_b = eligible(_panels_from(adj, raw, per, shr), t0)
    return _t("point-in-time：未來資料不影響 t0 判定", e_a.equals(e_b))


def _test_panel_keys():
    adj, raw, per, shr = _synth()
    p = _panels_from(adj, raw, per, shr)
    need = {"close_adj", "close_raw", "amount", "volume", "mcap", "pbr"}
    return _t("面板含 factors / backtest 所需欄位", need <= set(p))


def self_test() -> int:
    print("### universe.py 自我驗證 ###\n")
    results = [
        _test_a_active(), _test_b_threshold(), _test_b_denominator(),
        _test_b_emerging_not_in_denominator(), _test_b_missing_row(), _test_c_completeness(), _test_d_history(),
        _test_e_unexplained(), _test_e_bad_print(), _test_e_no_lookahead(),
        _test_f_open(), _test_g_factor(), _test_g_momentum_cap(), _test_common_only(),
        _test_point_in_time(), _test_panel_keys(),
    ]
    ok = all(results)
    print(f"\n  {sum(results)}/{len(results)} 通過"
          + ("" if ok else " —— 有測試未通過，不可用於真實資料"))
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI


def _with_extra(df: pd.DataFrame, data_dir: str, extra: str) -> pd.DataFrame:
    """合併補抓檔（per_extra.parquet / shares_extra.parquet），若存在。"""
    p = os.path.join(data_dir, extra)
    if not os.path.exists(p):
        return df
    out = pd.concat([df, pd.read_parquet(p)], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["stock_id"] = out["stock_id"].astype(str)
    return out.drop_duplicates(["date", "stock_id"], keep="last")


def fetched_ids(data_dir: str) -> set:
    """PBR／股數曾嘗試抓取的股票（原清單 ∪ 補抓清單）。"""
    ids = set()
    for name in ("fetch_universe.csv", "fetch_extra.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            ids |= set(pd.read_csv(p, dtype=str).iloc[:, 0].str.strip())
    return ids


def load_all(data_dir: str):
    L = lambda n: pd.read_parquet(os.path.join(data_dir, n))
    adj = L("twmarket_prices_adj.parquet")
    raw = L("twmarket_prices.parquet")
    per = _with_extra(L("per.parquet"), data_dir, "per_extra.parquet")
    shr = _with_extra(L("shares.parquet"), data_dir, "shares_extra.parquet")
    div = L("dividend_result.parquet")
    for d in (adj, raw):
        d["Date"] = pd.to_datetime(d["Date"]); d["stock_id"] = d["stock_id"].astype(str)
    return adj, raw, per, shr, div


def run(data_dir: str, out: str | None) -> int:
    adj, raw, per, shr, div = load_all(data_dir)
    panels = make_panels(prepare(adj, div), raw, per, shr)
    u = build(panels)
    print(summary(u).to_string())
    if out:
        path = out if os.path.isabs(out) else os.path.join(data_dir, out)
        u.to_parquet(path, index=False)
        print(f"\n已存 {path}（{len(u):,} 列）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/drive/MyDrive/twmarket")
    ap.add_argument("--out", default="universe.parquet")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return run(a.data_dir, a.out)


if __name__ == "__main__":
    sys.exit(main())
