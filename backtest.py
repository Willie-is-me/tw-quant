#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回測引擎：protocol.md 第 6 節（組合）、6.1（持有期間事件）、第 7 節（成本與執行）
==================================================================

執行價（第 7 節）
  VWAP = Amount / Volume 為未還原價。換算至還原基準：
      vwap_adj = VWAP × (close_adj / close_raw)
  不換算就拿未還原 VWAP 對還原收盤價算報酬，會在每次除權息後產生
  虛假的損益。`_test_vwap_basis` 為此設的迴歸測試。

再平衡（第 7 節）
  只交易差額：續留的成分股只買賣到新的等權目標。
  選股日 T 無成交者，該部位暫為現金，順延至首個有成交日買進。

持有期間事件（6.1）
  下市／停止交易  於最後成交日以還原收盤價出場，所得平均分配給其餘
                  仍在交易的持股，計一次賣出與一次買進成本。
                  同日多檔下市時先全部賣出，再一次分配
  次一選股日仍停牌  視同下市，以最後收盤價出場
  短期停牌        續抱，停牌期間報酬為 0（以 ffill 收盤價估值）
  全數出場        持有現金至次一選股日
  本期終點        只結算、不判下市（探索期結果不得取決於 2015 年之後的資料）

  `_test_delisting_loss` 是 protocol 6.1 明文要求的測試：
  一檔持股在持有期間大跌後下市，組合淨值必須反映該損失。

成本（第 7 節）
  主檢定（十分位）  比例成本，不設低消      Costs()
  實務參考（前10檔）資金 100 萬、低消 20 元  Costs.practical()

執行
    python backtest.py --self-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- 成本


@dataclass(frozen=True)
class Costs:
    fee_rate: float = 0.001425 * 0.6     # 0.0855%
    tax_rate: float = 0.001              # 賣出證交稅
    slippage: float = 0.0015             # 買賣各一次
    fee_min: float = 0.0                 # 0 = 不設低消
    capital: float = 1.0                 # 主檢定以 1 為單位即可

    @classmethod
    def practical(cls) -> "Costs":
        return cls(fee_min=20.0, capital=1_000_000.0)

    @classmethod
    def zero(cls) -> "Costs":
        return cls(fee_rate=0.0, tax_rate=0.0, slippage=0.0)

    def _fee(self, amt: float) -> float:
        if amt <= 0:
            return 0.0
        return max(amt * self.fee_rate, self.fee_min)

    def buy_cost(self, amt: float) -> float:
        return 0.0 if amt <= 0 else self._fee(amt) + amt * self.slippage

    def sell_cost(self, amt: float) -> float:
        return 0.0 if amt <= 0 else self._fee(amt) + amt * (self.tax_rate + self.slippage)

    def affordable(self, budget: float) -> float:
        """給定預算（含成本），回傳可買進的部位金額 x：x + buy_cost(x) = budget。"""
        if budget <= 0:
            return 0.0
        x = budget / (1 + self.fee_rate + self.slippage)     # 比例費用區
        if x * self.fee_rate >= self.fee_min:
            return x
        return max((budget - self.fee_min) / (1 + self.slippage), 0.0)  # 低消區


# ---------------------------------------------------------------- 價格


def vwap_adjusted(panels: dict) -> pd.DataFrame:
    """VWAP 換算至還原基準；無成交日為 NaN。"""
    amt, vol = panels["amount"], panels["volume"]
    ca, cr = panels["close_adj"], panels["close_raw"]
    cr = cr.reindex(index=ca.index, columns=ca.columns)
    vwap = amt / vol.where(vol > 0)
    factor = ca / cr
    return (vwap * factor).where(vol > 0)


# ---------------------------------------------------------------- 引擎


def run(panels: dict, picks: dict, costs: Costs = Costs(),
        end: pd.Timestamp | None = None) -> dict:
    """picks: {選股日 T: [stock_id, ...]}，依 T 排序執行。

    回傳 dict：nav（日淨值，起點 = capital）、trades、events。
    """
    cal = panels["calendar"]
    ca = panels["close_adj"]
    cols = {s: j for j, s in enumerate(ca.columns)}
    C = ca.to_numpy(dtype=float)                       # 收盤（含 NaN）
    CF = ca.ffill().to_numpy(dtype=float)              # 估值用（停牌報酬 0）
    VW = vwap_adjusted(panels).reindex(index=cal, columns=ca.columns).to_numpy(dtype=float)
    valid = ~np.isnan(C)
    last_idx = np.where(valid.any(axis=0),
                        len(cal) - 1 - np.argmax(valid[::-1], axis=0), -1)
    data_end = len(cal) - 1

    rebal = sorted(pd.Timestamp(t) for t in picks)
    if not rebal:
        raise ValueError("picks 為空")
    end_i = data_end if end is None else int(cal.searchsorted(pd.Timestamp(end), "right") - 1)
    r_idx = [int(cal.searchsorted(t)) for t in rebal]

    shares: dict[int, float] = {}       # 欄位索引 -> 股數（還原基準）
    pending: dict[int, float] = {}      # 尚未買進的目標金額
    cash = costs.capital
    min_cash = cash
    nav = np.full(len(cal), np.nan)
    trades, events = [], []

    def log(i, j, side, amt, cost, why):
        trades.append((cal[i], ca.columns[j], side, amt, cost, why))

    def buy(i, j, amt, price, why):
        """amt = 部位金額；成本另由現金支付。"""
        nonlocal cash
        if amt <= 0:
            return
        c = costs.buy_cost(amt)
        shares[j] = shares.get(j, 0.0) + amt / price
        cash -= amt + c
        log(i, j, "buy", amt, c, why)

    def sell_all(i, j, price, why):
        nonlocal cash
        amt = shares.pop(j) * price
        c = costs.sell_cost(amt)
        cash += amt - c
        log(i, j, "sell", amt, c, why)
        return amt - c

    for k, ti in enumerate(r_idx):
        if ti > end_i:
            break
        nxt = r_idx[k + 1] if k + 1 < len(r_idx) and r_idx[k + 1] <= end_i else end_i + 1
        new = [cols[s] for s in picks[rebal[k]] if s in cols]

        # ---- 再平衡（只交易差額）----
        # 先處理停牌中的舊持股：視同下市（6.1）
        for j in list(shares):
            if np.isnan(VW[ti, j]):
                sell_all(ti, j, CF[ti, j], "停牌於選股日，視同下市")
                events.append((cal[ti], ca.columns[j], "suspended_at_rebalance"))
        # 未買進的 pending 歸回現金
        pending.clear()

        cur = {j: shares[j] * VW[ti, j] for j in shares}
        W = cash + sum(cur.values())
        tradable = [j for j in new if not np.isnan(VW[ti, j])]
        n = len(new)
        tgt_each = W / n if n else 0.0
        for _ in range(50):                               # 成本與目標的定點迭代，至收斂
            cost = 0.0
            for j in set(cur) | set(new):
                c_v = cur.get(j, 0.0)
                t_v = tgt_each if j in tradable else 0.0
                cost += costs.sell_cost(max(c_v - t_v, 0.0)) + costs.buy_cost(max(t_v - c_v, 0.0))
            prev, tgt_each = tgt_each, ((W - cost) / n if n else 0.0)
            if abs(tgt_each - prev) <= 1e-15 * max(W, 1.0):
                break
        tol = 1e-9 * max(W, 1e-12)                        # 忽略浮點殘渣等級的交易

        for j in list(shares):                            # 先賣
            t_v = tgt_each if j in tradable else 0.0
            c_v = cur[j]
            if c_v > t_v + tol:
                if t_v == 0.0:
                    sell_all(ti, j, VW[ti, j], "再平衡移出")
                else:
                    amt = c_v - t_v
                    c = costs.sell_cost(amt)
                    shares[j] -= amt / VW[ti, j]
                    cash += amt - c
                    log(ti, j, "sell", amt, c, "再平衡減碼")
        for j in tradable:                                # 後買
            c_v = cur.get(j, 0.0)
            if tgt_each > c_v + tol:
                buy(ti, j, tgt_each - c_v, VW[ti, j], "再平衡買進")
        for j in new:
            if j not in tradable:
                pending[j] = tgt_each                     # 順延買進，暫為現金（含成本預算）

        # ---- 持有期間 ----
        for i in range(ti, nxt):
            if i > ti:
                for j in list(pending):                   # 順延買進
                    if not np.isnan(VW[i, j]):
                        budget = min(pending.pop(j), cash)
                        buy(i, j, costs.affordable(budget), VW[i, j], "順延買進")
            # 下市：今日為最後成交日，且不是本期終點（終點只結算、不判下市，
            # 否則探索期結果會取決於終點之後有無資料）
            dead = [j for j in shares if last_idx[j] == i and i < end_i]
            if dead:
                got = 0.0
                for j in dead:                            # 同日多檔下市：先全部賣出
                    got += sell_all(i, j, C[i, j], "下市出場")
                    events.append((cal[i], ca.columns[j], "delisted"))
                alive = [a for a in shares if not np.isnan(C[i, a])]
                if alive:                                 # 再一次分配給其餘持股
                    x = costs.affordable(got / len(alive))
                    for a in alive:
                        buy(i, a, x, C[i, a], "下市所得再分配")
            for j in list(pending):                       # 從未成交即下市：預算留作現金
                if last_idx[j] <= i and i < end_i:
                    pending.pop(j)
                    events.append((cal[i], ca.columns[j], "never_entered"))
            nav[i] = cash + sum(sh * CF[i, j] for j, sh in shares.items())
            min_cash = min(min_cash, cash)

    nav_s = pd.Series(nav, index=cal).dropna()
    return {
        "nav": nav_s,
        "cash_end": cash,
        "min_cash": min_cash,
        "trades": pd.DataFrame(trades, columns=["date", "stock_id", "side",
                                                "amount", "cost", "reason"]),
        "events": pd.DataFrame(events, columns=["date", "stock_id", "event"]),
    }


def benchmark(panels: dict, start: pd.Timestamp, end: pd.Timestamp | None = None,
              sid: str = "0050", capital: float = 1.0) -> pd.Series:
    """0050 買進持有：於 start 以換算後 VWAP 進場，不計成本（第 8 節）。"""
    cal = panels["calendar"]
    vw = vwap_adjusted(panels)[sid]
    ca = panels["close_adj"][sid].ffill()
    i0 = int(cal.searchsorted(pd.Timestamp(start)))
    while np.isnan(vw.iloc[i0]):
        i0 += 1
    nav = capital * ca.iloc[i0:] / vw.iloc[i0]
    if end is not None:
        nav = nav[nav.index <= pd.Timestamp(end)]
    return nav.rename(sid)


def picks_from(portfolios: pd.DataFrame, factor: str) -> dict:
    p = portfolios[portfolios.factor == factor]
    return {pd.Timestamp(T): list(g.sort_values("rank").stock_id)
            for T, g in p.groupby("T")}


# ---------------------------------------------------------------- 測試用合成市場


def _market(paths: dict, n_days=60, start="2010-01-04", factor=None, vol=None):
    """paths: {sid: 價格陣列}；factor: {sid: 調整因子陣列}（還原/未還原）。"""
    cal = pd.bdate_range(start, periods=n_days)
    ca = pd.DataFrame({s: np.asarray(p, float) for s, p in paths.items()}, index=cal)
    fac = pd.DataFrame({s: (factor or {}).get(s, np.ones(n_days)) for s in paths}, index=cal)
    cr = ca / fac
    v = pd.DataFrame({s: (vol or {}).get(s, np.full(n_days, 1000.0)) for s in paths}, index=cal)
    v = v.where(ca.notna(), 0.0)
    amt = cr * v                                    # VWAP（未還原）= 收盤（未還原）
    return {"calendar": cal, "close_adj": ca, "close_raw": cr,
            "amount": amt, "volume": v}


def _t(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _test_equal_weight():
    n = 40
    p = _market({"A": np.linspace(100, 130, n), "B": np.linspace(50, 40, n)}, n)
    T = p["calendar"][0]
    r = run(p, {T: ["A", "B"]}, Costs.zero())
    exp = 0.5 * 130 / 100 + 0.5 * 40 / 50              # 1.05，非對稱以免巧合通過
    return _t(f"等權、無成本：期末淨值 {r['nav'].iloc[-1]:.4f} = 手算 {exp:.4f}",
              np.isclose(r["nav"].iloc[-1], exp))


def _test_delisting_loss():
    """protocol 6.1 明文要求：下市前的大跌必須反映在組合報酬中。"""
    n = 40
    a = np.full(n, 100.0); a[10:20] = np.linspace(100, 20, 10); a[20:] = np.nan
    b = np.full(n, 100.0); b[20:] = np.linspace(100, 110, n - 20)
    p = _market({"A": a, "B": b}, n)
    T = p["calendar"][0]
    r = run(p, {T: ["A", "B"]}, Costs.zero())
    # A 的一半資金跌至 0.2 後於第 19 日出場，再全數轉入 B（第 19 日 B=100）
    exp = (0.5 + 0.5 * 0.2) * 110 / 100
    ev = r["events"]
    return _t(f"下市損失反映於淨值：{r['nav'].iloc[-1]:.4f} = 手算 {exp:.4f}",
              np.isclose(r["nav"].iloc[-1], exp) and (ev.event == "delisted").sum() == 1)


def _test_redistribute_not_cash():
    n = 40
    a = np.full(n, 100.0); a[15:] = np.nan
    b = np.full(n, 100.0); b[15:] = np.linspace(100, 200, n - 15)
    p = _market({"A": a, "B": b}, n)
    T = p["calendar"][0]
    r = run(p, {T: ["A", "B"]}, Costs.zero())
    # 若錯把下市所得當現金，期末為 0.5 + 0.5*2 = 1.5；正確應為 2.0
    return _t(f"下市所得再分配而非持有現金：{r['nav'].iloc[-1]:.4f}（正確 2.0、錯誤 1.5）",
              np.isclose(r["nav"].iloc[-1], 2.0))


def _test_buy_cost():
    n = 10
    p = _market({"A": np.full(n, 100.0), "B": np.full(n, 100.0)}, n)
    T = p["calendar"][0]
    c = Costs()
    r = run(p, {T: ["A", "B"]}, c)
    rate = c.fee_rate + c.slippage                    # 買進成本率 0.2355%
    exp = 1.0 / (1.0 + rate)                          # 部位 x，成本 rate·x，x(1+rate)=1
    return _t(f"買進成本：淨值 {r['nav'].iloc[0]:.6f} = 1/(1+{rate:.4%}) = {exp:.6f}",
              np.isclose(r["nav"].iloc[0], exp, rtol=1e-9))


def _test_cash_invariant():
    """記帳不變式：現金不得為負；完全投資時期末現金應為 0。"""
    n = 60
    a = np.full(n, 100.0); a[20:30] = np.linspace(100, 30, 10); a[30:] = np.nan
    b = np.linspace(100, 130, n); c_ = np.linspace(100, 80, n); d = np.full(n, 100.0)
    p = _market({"A": a, "B": b, "C": c_, "D": d}, n)
    cal = p["calendar"]
    r1 = run(p, {cal[0]: ["A", "B", "C"], cal[40]: ["B", "D"]}, Costs())
    r2 = run(p, {cal[0]: ["A", "B", "C"], cal[40]: ["B", "D"]},
             Costs(fee_min=20.0, capital=100_000.0))
    ok = (r1["min_cash"] > -1e-9 and abs(r1["cash_end"]) < 1e-9
          and r2["min_cash"] > -1e-6 and abs(r2["cash_end"]) < 1e-6)
    return _t(f"現金不變式（比例成本／含低消皆不為負、期末歸零）", ok)


def _test_trade_only_difference():
    n = 30
    p = _market({"A": np.full(n, 100.0), "B": np.full(n, 100.0)}, n)
    cal = p["calendar"]
    c = Costs()
    r = run(p, {cal[0]: ["A", "B"], cal[15]: ["A", "B"]}, c)
    t2 = r["trades"][r["trades"].date == cal[15]]
    return _t(f"續留持股不重複交易（第二次再平衡交易 {len(t2)} 筆）", len(t2) == 0)


def _test_full_turnover_cost():
    n = 30
    p = _market({s: np.full(n, 100.0) for s in "ABCD"}, n)
    cal = p["calendar"]
    c = Costs()
    r = run(p, {cal[0]: ["A", "B"], cal[15]: ["C", "D"]}, c)
    nav1 = r["nav"].iloc[14]
    rate_b = c.fee_rate + c.slippage
    rate_s = c.fee_rate + c.tax_rate + c.slippage
    exp = nav1 * (1 - rate_s) / (1 + rate_b)
    return _t(f"全換手成本：{r['nav'].iloc[15]:.6f} ≈ 手算 {exp:.6f}",
              np.isclose(r["nav"].iloc[15], exp, rtol=1e-4))


def _test_vwap_basis():
    """還原/未還原基準換算：除權息後不得出現虛假損益。"""
    n = 20
    fac = np.ones(n); fac[:10] = 0.5                  # 前 10 日還原價為未還原的一半
    p = _market({"A": np.full(n, 50.0)}, n, factor={"A": fac})
    T = p["calendar"][0]
    r = run(p, {T: ["A"]}, Costs.zero())
    flat = np.allclose(r["nav"].values, 1.0)
    # 若直接用未還原 VWAP（100）買進、以還原收盤（50）估值，會立刻 -50%
    return _t("VWAP 換算至還原基準（無虛假損益）", flat)


def _test_suspended_at_rebalance():
    n = 30
    a = np.full(n, 100.0); a[12:20] = np.nan; a[20:] = 150.0   # 選股日 15 停牌
    p = _market({"A": a, "B": np.full(n, 100.0)}, n)
    cal = p["calendar"]
    r = run(p, {cal[0]: ["A", "B"], cal[15]: ["B"]}, Costs.zero())
    ev = r["events"]
    ok = (ev.event == "suspended_at_rebalance").sum() == 1 and np.isclose(r["nav"].iloc[-1], 1.0)
    return _t("選股日仍停牌：視同下市，以最後收盤出場（不吃到復牌後上漲）", ok)


def _test_deferred_entry():
    n = 20
    vol = np.full(n, 1000.0); vol[:3] = 0.0              # 前 3 日無成交
    a = np.full(n, 100.0); a[3:] = 120.0
    p = _market({"A": a}, n, vol={"A": vol})
    p["close_adj"].iloc[:3, 0] = np.nan                  # 無成交日無收盤
    T = p["calendar"][0]
    r = run(p, {T: ["A"]}, Costs.zero())
    return _t("選股日無成交：順延買進，順延期間為現金（不吃到跳空）",
              np.isclose(r["nav"].iloc[-1], 1.0))


def _test_fee_min():
    n = 5
    p = _market({s: np.full(n, 100.0) for s in "ABCDEFGHIJ"}, n)
    T = p["calendar"][0]
    c = Costs(fee_rate=0.001425 * 0.6, tax_rate=0.001, slippage=0.0,
              fee_min=20.0, capital=100_000.0)          # 每檔 1 萬，手續費 8.55 < 20
    r = run(p, {T: list("ABCDEFGHIJ")}, c)
    buys = r["trades"][r["trades"].side == "buy"]
    return _t(f"低消：每筆手續費 {buys.cost.min():.2f} 元（應為 20）",
              np.allclose(buys.cost, 20.0))


def _test_same_day_delistings():
    """同日兩檔下市：各賣一次，所得一次分配給其餘持股（不得先買進另一檔下市股）。"""
    n = 30
    a = np.full(n, 100.0); a[15:] = np.nan
    b = np.full(n, 100.0); b[15:] = np.nan
    p = _market({"A": a, "B": b, "C": np.full(n, 100.0), "D": np.full(n, 100.0)}, n)
    T = p["calendar"][0]
    r = run(p, {T: list("ABCD")}, Costs())
    t = r["trades"][r["trades"].date == p["calendar"][14]]
    ok = (len(t) == 4 and set(t[t.side == "sell"].stock_id) == {"A", "B"}
          and set(t[t.side == "buy"].stock_id) == {"C", "D"})
    return _t(f"同日兩檔下市：{len(t)} 筆交易（2 賣 2 買）", ok)


def _test_end_not_delisting():
    """本期終點不判下市：結果不得取決於終點之後有無資料。"""
    n = 30
    a1 = np.full(n, 100.0); a1[20:] = np.nan            # 資料止於第 19 日
    a2 = np.full(n, 100.0)                              # 資料持續
    cal_end = None
    res = []
    for a in (a1, a2):
        p = _market({"A": a, "B": np.full(n, 100.0)}, n)
        cal_end = p["calendar"][19]
        res.append(run(p, {p["calendar"][0]: ["A", "B"]}, Costs(), end=cal_end))
    same = np.allclose(res[0]["nav"].values, res[1]["nav"].values)
    return _t("終點當日不判下市（終點之後有無資料，結果相同）",
              same and res[0]["events"].empty)


def _test_point_in_time():
    n = 40
    p1 = _market({"A": np.linspace(100, 130, n), "B": np.linspace(100, 90, n)}, n)
    p2 = _market({"A": np.linspace(100, 130, n), "B": np.linspace(100, 90, n)}, n)
    p2["close_adj"].iloc[25:, 0] *= 3; p2["close_raw"].iloc[25:, 0] *= 3
    p2["amount"].iloc[25:, 0] *= 3
    T = p1["calendar"][0]
    a = run(p1, {T: ["A", "B"]}, Costs())["nav"].iloc[:25]
    b = run(p2, {T: ["A", "B"]}, Costs())["nav"].iloc[:25]
    return _t("point-in-time：之後的價格不影響之前的淨值", np.allclose(a, b))


def self_test() -> int:
    print("### backtest.py 自我驗證 ###\n")
    res = [
        _test_equal_weight(), _test_delisting_loss(), _test_redistribute_not_cash(),
        _test_buy_cost(), _test_cash_invariant(),
        _test_trade_only_difference(), _test_full_turnover_cost(),
        _test_vwap_basis(), _test_suspended_at_rebalance(), _test_deferred_entry(),
        _test_fee_min(), _test_same_day_delistings(), _test_end_not_delisting(),
        _test_point_in_time(),
    ]
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
