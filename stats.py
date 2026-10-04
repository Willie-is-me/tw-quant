#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計檢定：protocol.md 第 8 節與 8.1
==================================================================
本模組只做 protocol 8.1 明文規定的計算，所有常數在此集中定義。
任何與 8.1 不一致之處，以 protocol 為準並修正本模組。

月報酬       NAV(月底) / NAV(上月底) − 1，首月分母為期初資金
超額報酬     e = 組合月報酬 − 基準月報酬（算術差）
主要檢定     t = mean(e) / SE_NW，Bartlett 權重，L = floor(4(n/100)^(2/9))
p 值         雙尾，標準常態
多重比較     Bonferroni、Benjamini-Hochberg（q = 0.05），族 = 該期間的十分位檢定
效果量       年化超額 = mean(e) × 12；IR = mean(e) / sd(e) × √12
DSR          Bailey & López de Prado (2014)，N = 8，V = 同期各試驗月 SR 的樣本變異數
信賴區間     stationary bootstrap，平均區塊 floor(n^(1/3) + .5)，B = 10,000，種子 20261004

執行
    python stats.py --self-test
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

T_THRESHOLD = 3.0          # 事前宣告的顯著性門檻（正向）
PRACTICAL_MIN = 0.02       # 年化超額報酬的實務門檻
FDR_Q = 0.05
N_TRIALS = 8               # 第 8 節：4 因子 × P=1 × 2 切法
BOOT_B = 10_000
BOOT_SEED = 20261004
MONTHS = 12


# ---------------------------------------------------------------- 月報酬


def monthly_returns(nav: pd.Series, base: float) -> pd.Series:
    """日淨值 -> 月報酬（索引為曆月月底）。首月分母為期初資金 base。

    呼叫端須保證 nav 的終點為某月最後一個交易日（protocol 8.1 的終點定義），
    否則最後一個月是不完整月份。
    """
    if nav.empty:
        raise ValueError("nav 為空")
    m = nav.resample("ME").last().dropna()
    prev = m.shift(1)
    prev.iloc[0] = base
    return (m / prev - 1.0).rename(nav.name)


def drop_year(e: pd.Series, year: int) -> pd.Series:
    """排除 2008 切法：刪除該年 1–12 月，其餘不變。"""
    return e[e.index.year != year]


# ---------------------------------------------------------------- 檢定


def nw_lag(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def nw_se(x, lag: int | None = None) -> tuple[float, int]:
    """平均數的 Newey-West 標準誤（Bartlett 權重，分母 n，無小樣本修正）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("樣本數不足")
    L = nw_lag(n) if lag is None else int(lag)
    d = x - x.mean()
    s = d @ d / n
    for j in range(1, L + 1):
        s += 2.0 * (1.0 - j / (L + 1.0)) * (d[j:] @ d[:-j] / n)
    return float(np.sqrt(s / n)), L


def mean_test(e) -> dict:
    e = np.asarray(e, dtype=float)
    se, L = nw_se(e)
    m = float(e.mean())
    t = m / se if se > 0 else np.nan
    return {"n": len(e), "mean_m": m, "se_m": se, "lag": L, "t": t,
            "p": float(2.0 * st.norm.sf(abs(t))) if np.isfinite(t) else np.nan}


def bonferroni(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.minimum(p * len(p), 1.0)


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg 調整後 p 值（step-up）。調整後 <= q 即拒絕。"""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


# ---------------------------------------------------------------- Sharpe 與 DSR


def sharpe(e) -> float:
    e = np.asarray(e, dtype=float)
    s = e.std(ddof=1)
    return float(e.mean() / s) if s > 0 else np.nan


def moments(e) -> tuple[float, float]:
    """樣本偏態與（非超額）峰度，分母 n，不做偏誤修正。"""
    e = np.asarray(e, dtype=float)
    return (float(st.skew(e, bias=True)),
            float(st.kurtosis(e, fisher=False, bias=True)))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """N 個無效策略中最佳者的期望 SR（Bailey & López de Prado 2014, eq. 2）。"""
    g = np.euler_gamma
    z = ((1 - g) * st.norm.ppf(1 - 1.0 / n_trials)
         + g * st.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(var_sr) * z)


def psr(sr: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return float(st.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(den)))


def dsr(e, var_sr: float, n_trials: int = N_TRIALS) -> dict:
    sr = sharpe(e)
    g3, g4 = moments(e)
    sr0 = expected_max_sr(n_trials, var_sr)
    return {"sr_m": sr, "skew": g3, "kurt": g4, "sr0_m": sr0,
            "dsr": psr(sr, sr0, len(e), g3, g4)}


# ---------------------------------------------------------------- Bootstrap


def block_length(n: int) -> int:
    return max(1, int(np.floor(n ** (1.0 / 3.0) + 0.5)))


def stationary_indices(n: int, B: int, mean_block: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994)：區塊長度服從幾何分配，平均 mean_block，環狀接續。"""
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    restart = rng.random((B, n)) < p
    jump = rng.integers(0, n, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], jump[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(e, B: int = BOOT_B, seed: int = BOOT_SEED,
                 level: float = 0.95) -> dict:
    e = np.asarray(e, dtype=float)
    n = len(e)
    rng = np.random.default_rng(seed)
    idx = stationary_indices(n, B, block_length(n), rng)
    x = e[idx]
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    ann = mu * MONTHS
    ir = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.nan) * np.sqrt(MONTHS)
    a = (1 - level) / 2
    q = lambda v: np.nanquantile(v, [a, 1 - a])
    lo_a, hi_a = q(ann)
    lo_i, hi_i = q(ir)
    return {"ci_ann_lo": float(lo_a), "ci_ann_hi": float(hi_a),
            "ci_ir_lo": float(lo_i), "ci_ir_hi": float(hi_i),
            "boot_block": block_length(n), "boot_B": B, "boot_seed": seed}


# ---------------------------------------------------------------- 彙整


def cagr(r: pd.Series) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.prod(1 + r) ** (MONTHS / len(r)) - 1)


def evaluate(rp: pd.Series, rb: pd.Series, B: int = BOOT_B) -> dict:
    """單一檢定：組合與基準的月報酬 -> protocol 8.1 的各項統計量。"""
    if not rp.index.equals(rb.index):
        raise ValueError("組合與基準的月份不一致")
    e = (rp - rb).to_numpy(dtype=float)
    out = mean_test(e)
    out["ann_excess"] = out["mean_m"] * MONTHS
    sr = sharpe(e)
    out["ir"] = sr * np.sqrt(MONTHS)
    out["sr_m"] = sr
    out["skew"], out["kurt"] = moments(e)
    out["cagr_p"], out["cagr_b"] = cagr(rp), cagr(rb)
    out["cagr_diff"] = out["cagr_p"] - out["cagr_b"]
    out["first_month"] = str(rp.index[0].strftime("%Y-%m"))
    out["last_month"] = str(rp.index[-1].strftime("%Y-%m"))
    out.update(bootstrap_ci(e, B=B))
    return out


def verdict(t: float, ann_excess: float) -> str:
    if not np.isfinite(t):
        return "無法計算"
    if t > T_THRESHOLD:
        return "通過" if ann_excess > PRACTICAL_MIN else "顯著但未達實務門檻"
    if t < -T_THRESHOLD:
        return "顯著劣於基準"
    return "不顯著"


def family(rows: list[dict], n_trials: int = N_TRIALS) -> list[dict]:
    """檢定族的多重比較與 DSR。rows 需含 p、sr_m、skew、kurt、n。"""
    p = np.array([r["p"] for r in rows])
    srs = np.array([r["sr_m"] for r in rows])
    var_sr = float(np.var(srs, ddof=1)) if len(rows) > 1 else 0.0
    sr0 = expected_max_sr(n_trials, var_sr)
    for r, pb, pf in zip(rows, bonferroni(p), bh_adjust(p)):
        r["family_size"] = len(rows)
        r["p_bonferroni"] = float(pb)
        r["p_bh"] = float(pf)
        r["bh_reject"] = bool(pf <= FDR_Q)
        r["var_sr"] = var_sr
        r["n_trials"] = n_trials
        r["sr0_m"] = sr0
        r["dsr"] = psr(r["sr_m"], sr0, r["n"], r["skew"], r["kurt"])
        r["verdict"] = verdict(r["t"], r["ann_excess"])
    return rows


# ---------------------------------------------------------------- 測試


def _t(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _test_lag():
    got = {n: nw_lag(n) for n in (96, 108, 128)}
    return _t(f"NW lag：{got}（protocol：96→3、108→4、128→4）",
              got == {96: 3, 108: 4, 128: 4})


def _test_nw_hand():
    x = np.array([0.03, -0.01, 0.02, 0.05, -0.02, 0.01])
    n = len(x); d = x - x.mean()
    g0 = (d ** 2).sum() / n
    g1 = (d[1:] * d[:-1]).sum() / n
    hand = np.sqrt((g0 + 2 * 0.5 * g1) / n)          # L=1：權重 1 − 1/2
    se1, _ = nw_se(x, lag=1)
    se0, _ = nw_se(x, lag=0)
    return _t(f"NW 手算：L=1 {se1:.6f} = {hand:.6f}；L=0 等於 iid（{se0:.6f}）",
              np.isclose(se1, hand) and np.isclose(se0, x.std(ddof=0) / np.sqrt(n)))


def _test_nw_autocorr():
    """正自相關時 NW 標準誤應大於 iid 標準誤（方向檢查）。"""
    rng = np.random.default_rng(1)
    u = rng.normal(0, 1, 5000); x = np.empty_like(u); x[0] = u[0]
    for i in range(1, len(u)):
        x[i] = 0.5 * x[i - 1] + u[i]
    se_nw, L = nw_se(x)
    se_iid = x.std(ddof=0) / np.sqrt(len(x))
    return _t(f"AR(1) φ=0.5：NW SE {se_nw:.4f} > iid {se_iid:.4f}（L={L}）", se_nw > 1.3 * se_iid)


def _test_moments():
    """手算：x = [1,2,3,4,10]，m2=10、m3=36、m4=278.8 -> 偏態 1.13842、峰度 2.788。"""
    g3, g4 = moments([1, 2, 3, 4, 10])
    return _t(f"偏態／峰度（分母 n、非超額）：{g3:.5f}、{g4:.3f}",
              np.isclose(g3, 36 / 10 ** 1.5) and np.isclose(g4, 2.788))


def _test_p_value():
    r = mean_test(np.r_[np.full(50, 0.01), np.full(50, -0.01)] + 0.0)
    p3 = 2 * st.norm.sf(3.0)
    return _t(f"p 值：t=3.0 對應雙尾 {p3:.4f}；均值 0 時 t=0",
              np.isclose(p3, 0.0027, atol=5e-5) and np.isclose(r["t"], 0.0))


def _test_bh():
    a = bh_adjust([0.02, 0.03, 0.2, 0.5])
    b = bh_adjust([0.01, 0.04, 0.03, 0.005])
    ok = (np.allclose(a, [0.06, 0.06, 0.2 * 4 / 3, 0.5])
          and (b <= 0.05).all()
          and np.allclose(bonferroni([0.01, 0.3]), [0.02, 0.6]))
    return _t(f"BH 手算：{np.round(a, 4).tolist()}；Bonferroni", ok)


def _test_threshold_binds():
    """t > 3 必同時通過 N=8 的 Bonferroni 與 BH（protocol 8.1）。"""
    p = [2 * st.norm.sf(3.0001)] + [0.9] * 7
    return _t("t > 3.0 ⇒ 同時通過 Bonferroni 與 BH（N=8）",
              bonferroni(p)[0] < 0.05 and bh_adjust(p)[0] < 0.05)


def _test_dsr_paper():
    """原論文數例：年化 SR 2.5、T=1250 日、偏態 −3、峰度 10、N=100、V=0.5。"""
    k = 250
    sr = 2.5 / np.sqrt(k)
    sr0 = expected_max_sr(100, 0.5 / k)
    d = psr(sr, sr0, 1250, -3.0, 10.0)
    return _t(f"DSR 原論文數例：{d:.4f}（論文 0.9004）", abs(d - 0.9004) < 1e-3)


def _test_expected_max():
    rng = np.random.default_rng(2)
    mc = rng.normal(0, 1, (200_000, 8)).max(axis=1).mean()
    f = expected_max_sr(8, 1.0)
    return _t(f"E[max SR] 近似式 N=8：{f:.3f} vs 模擬 {mc:.3f}（誤差 < 5%）",
              abs(f / mc - 1) < 0.05)


def _sb_var_theory(x, p):
    """Stationary bootstrap 樣本平均數的理論變異數（Politis & Romano 1994, Lemma 1）。"""
    n = len(x); d = x - x.mean()
    c = np.array([d[:n - i] @ d[i:] / n for i in range(n)])     # 一般（非環狀）自共變異
    i = np.arange(1, n)
    b = (1 - i / n) * (1 - p) ** i + (i / n) * (1 - p) ** (n - i)
    return (c[0] + 2 * (b * c[1:]).sum()) / n


def _test_bootstrap():
    rng = np.random.default_rng(3)
    e = rng.normal(0.005, 0.03, 108)
    a = bootstrap_ci(e, B=4000)
    b = bootstrap_ci(e, B=4000)
    ann = e.mean() * 12
    # 索引產生器對照理論變異數（B 大時應吻合）
    idx = stationary_indices(len(e), 50_000, 5, np.random.default_rng(4))
    v_emp = e[idx].mean(axis=1).var()
    v_th = _sb_var_theory(e, 1 / 5)
    restarts = (np.diff(idx, axis=1) % len(e) != 1).mean()
    ok = (a == b and a["ci_ann_lo"] < ann < a["ci_ann_hi"] and a["boot_block"] == 5
          and abs(v_emp / v_th - 1) < 0.03 and abs(restarts * (1 - 1 / len(e)) ** -1 - 0.2) < 0.01)
    return _t(f"bootstrap：可重現、區間含點估計、平均區塊 {1/restarts:.2f}≈5、"
              f"變異數對照理論值 {v_emp/v_th:.3f}", ok)


def _test_monthly_base():
    cal = pd.bdate_range("2007-01-02", "2007-03-30")
    nav = pd.Series(0.997, index=cal)                  # 進場成本 0.3%，其後不動
    nav[cal >= "2007-02-01"] = 0.997 * 1.02
    r = monthly_returns(nav, base=1.0)
    ok = (len(r) == 3 and np.isclose(r.iloc[0], -0.003)
          and np.isclose(r.iloc[1], 0.02) and np.isclose(r.iloc[2], 0.0))
    return _t(f"月報酬：首月分母為期初資金（含進場成本 {r.iloc[0]:.4f}）", ok)


def _test_drop_2008():
    idx = pd.date_range("2007-01-31", "2015-12-31", freq="ME")
    e = pd.Series(0.0, index=idx)
    d = drop_year(e, 2008)
    return _t(f"排除 2008：{len(e)} → {len(d)} 個月，lag {nw_lag(len(e))} → {nw_lag(len(d))}",
              len(e) == 108 and len(d) == 96 and not (d.index.year == 2008).any())


def _test_family():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2007-01-31", periods=108, freq="ME")
    rows = []
    for k in range(8):
        rb = pd.Series(rng.normal(0.008, 0.05, 108), index=idx)
        rp = rb + rng.normal(0.001 * k, 0.02, 108)
        rows.append(evaluate(rp, rb, B=500))
    rows = family(rows)
    ok = (all(0 <= r["dsr"] <= 1 for r in rows)
          and all(r["family_size"] == 8 for r in rows)
          and all(r["p_bonferroni"] >= r["p"] for r in rows)
          and all(r["verdict"] in ("通過", "顯著但未達實務門檻", "顯著劣於基準", "不顯著")
                  for r in rows))
    return _t("檢定族彙整：Bonferroni ≥ 原始 p、DSR ∈ [0,1]、判定字串合法", ok)


def _test_evaluate_alignment():
    idx = pd.date_range("2007-01-31", periods=12, freq="ME")
    try:
        evaluate(pd.Series(0.0, index=idx), pd.Series(0.0, index=idx[1:].append(idx[:1])))
        return _t("組合與基準月份不一致時拋出例外", False)
    except ValueError:
        return _t("組合與基準月份不一致時拋出例外", True)


def self_test() -> int:
    print("### stats.py 自我驗證 ###\n")
    res = [_test_lag(), _test_nw_hand(), _test_nw_autocorr(), _test_moments(), _test_p_value(),
           _test_bh(), _test_threshold_binds(), _test_dsr_paper(),
           _test_expected_max(), _test_bootstrap(), _test_monthly_base(),
           _test_drop_2008(), _test_family(), _test_evaluate_alignment()]
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
