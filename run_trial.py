#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
執行試驗：universe → 因子 → 回測 → 統計（protocol 第 4–8 節）
==================================================================
    python run_trial.py --phase exploration --data-dir DIR
    python run_trial.py --phase holdout --data-dir DIR --i-understand-holdout-is-one-shot
    python run_trial.py --phase holdout --data-dir DIR --placebo     # 結構乾跑（合成報酬）
    python run_trial.py --self-test

期間（protocol 8.1，不得更改）
  exploration  選股日 2007–2015，終點 2015-12-31，切法：含 2008／排除 2008
  holdout      選股日 2016–2026，終點 2026-08-31，資金重新起算

防呆
  共同        protocol.md 必須存在；若在 git repo 中且有未提交的修改，拒跑
  exploration trial_log.md 須有 8 列「（未執行）」的登記（4 因子 × 2 切法，
              每列恰一個因子、一種切法）。真實回測開始前先把這 8 列標為
              「（執行中）」，跑完填入結果，因此同一組登記只能用一次（中途失敗
              亦然）；重跑必須先登記新的列。DSR 的 N = trial_log 的登記總數
  holdout     須在 git repo 中（protocol 已 commit）且已有探索期結果；鎖檔
              HOLDOUT_LOCK.json（protocol 旁與程式旁各一）或 results/holdout
              存在即拒跑。開跑前先以合成報酬跑一次完整流程確認無例外，
              通過後才寫鎖檔並執行真實回測；寫鎖後即使失敗也不解鎖

結構乾跑（--placebo）
  以合成報酬取代真實價格、以亂數取代市值與 PBR，保留真實的交易日／停牌／
  下市結構與資格判定。只輸出檢定結構（月數、檔數），不輸出選股與事件明細。
  不含任何真實績效資訊，不需登記、不上鎖。

輸出（--out，預設 results/<phase>/）
  summary.csv   每個檢定一列：t、p、Bonferroni、BH、DSR、年化超額、IR、CI、判定
  monthly.csv   各試驗的組合、基準與超額月報酬
  picks.csv / trades.csv / events.csv   選股清單、交易明細、下市停牌事件
  report.md     人可讀的結果表
  provenance.json  執行時間、protocol 與程式的 SHA-256
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest as BT   # noqa: E402
import factors as F     # noqa: E402
import stats as S       # noqa: E402
import universe as U    # noqa: E402

FACTORS = ("momentum", "size", "lowvol", "value")
FACTOR_ZH = {"momentum": "動能", "size": "規模", "lowvol": "低波動", "value": "價值"}
BENCH = "0050"
CUT_ALL, CUT_EX08 = "含 2008", "排除 2008"
HOLDOUT_CUT = "全期"
UNRUN = "（未執行）"
RUNNING = "（執行中）"
SRC_FILES = ("universe.py", "factors.py", "backtest.py", "stats.py", "run_trial.py")


@dataclass(frozen=True)
class Phase:
    name: str
    years: tuple[int, int]
    end: str
    cuts: tuple[str, ...]
    expected_n: dict = field(default_factory=dict)


PHASES = {
    "exploration": Phase("exploration", (2007, 2015), "2015-12-31",
                         (CUT_ALL, CUT_EX08), {CUT_ALL: 108, CUT_EX08: 96}),
    "holdout": Phase("holdout", (2016, 2026), "2026-08-31",
                     (HOLDOUT_CUT,), {HOLDOUT_CUT: 128}),
}


# ---------------------------------------------------------------- 來源紀錄


def sha256(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(path: str) -> dict:
    """protocol 所在 repo 的狀態：commit、是否有未提交修改。無 git 時回傳空 dict。"""
    d = os.path.dirname(os.path.abspath(path))
    try:
        head = subprocess.run(["git", "-C", d, "log", "-1", "--format=%H %cI", "--", path],
                              capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", path],
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    if head.returncode != 0:
        return {}
    return {"commit": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}


def provenance(protocol: str) -> dict:
    return {
        "run_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_path": os.path.abspath(protocol),
        "protocol_sha256": sha256(protocol),
        "protocol_git": git_state(protocol),
        "src_sha256": {f: sha256(os.path.join(HERE, f)) for f in SRC_FILES},
    }


def check_protocol(protocol: str, require_git: bool = False) -> dict:
    """protocol 須存在；有 git 時須已 commit 且無修改。holdout 一律要求 git。"""
    if not protocol or not os.path.exists(protocol):
        raise SystemExit(f"拒跑：找不到 protocol（{protocol}）")
    g = git_state(protocol)
    if g.get("dirty"):
        raise SystemExit("拒跑：protocol.md 有未提交的修改，先 commit 並 push")
    if g and not g.get("commit"):
        raise SystemExit("拒跑：protocol.md 尚未 commit")
    if not g:
        if require_git:
            raise SystemExit("拒跑：holdout 必須在 git repo 中執行（protocol.md 須已 commit），"
                             "請先 git clone 研究的 repo")
        print("注意：無法取得 git 狀態。protocol 的 SHA-256 會寫入輸出，"
              "請確認與 GitHub 上已 commit 的版本一致。")
    return g


# ---------------------------------------------------------------- trial_log


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _row_trial(cells: list[str]):
    """表格列 -> (因子, 切法) 或 None。欄位：# | 日期 | 因子/規則 | 參數 | 結果 | 備註。"""
    if len(cells) < 5 or not cells[0].isdigit():
        return None
    fac_cell = cells[2]
    facs = [f for f in FACTORS if re.search(rf"(?<![A-Za-z]){f}(?![A-Za-z])", fac_cell)]
    if len(facs) != 1 or not fac_cell.startswith(facs[0]):
        return ("INVALID", cells[0])
    par = cells[3].replace(" ", "")
    cuts = [c for c in (CUT_ALL, CUT_EX08) if par.startswith("十分位、" + c.replace(" ", ""))]
    if len(cuts) != 1:
        return ("INVALID", cells[0])
    return (facs[0], cuts[0])


def _table_lines(text: str):
    """表格列（略過 ``` 程式碼區塊內的範例）。"""
    fence = False
    for n, line in enumerate(text.split("\n")):
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if not fence and line.startswith("|"):
            yield n, line


def parse_trials(trial_log: str) -> list[dict]:
    """所有合格式的試驗列；格式不符的「未執行」列、重複列號皆拒跑。"""
    if not trial_log or not os.path.exists(trial_log):
        raise SystemExit(f"拒跑：找不到 trial_log（{trial_log}）")
    text = open(trial_log, encoding="utf-8").read()
    rows, bad, seen = [], [], {}
    for n, line in _table_lines(text):
        cells = _cells(line)
        key = _row_trial(cells)
        if key is None:
            continue
        if key[0] == "INVALID":
            if cells[4] == UNRUN:
                bad.append(key[1])
            continue
        if cells[0] in seen:
            raise SystemExit(f"拒跑：trial_log 列號 {cells[0]} 重複")
        seen[cells[0]] = n
        rows.append({"id": cells[0], "key": key, "status": cells[4], "line": n})
    if bad:
        raise SystemExit(f"拒跑：trial_log 第 {', '.join(bad)} 列格式不符"
                         "（因子欄須以單一因子英文名開頭，參數欄須以「十分位、含 2008」"
                         "或「十分位、排除 2008」開頭）")
    return rows


def registered_count(trial_log: str) -> int:
    """已登記的試驗數（不論是否已執行）＝ DSR 的 N（protocol 8.1）。"""
    return len(parse_trials(trial_log))


def check_exploration(trial_log: str) -> dict:
    need = {(f, c) for f in FACTORS for c in (CUT_ALL, CUT_EX08)}
    have = {}
    for r in parse_trials(trial_log):
        if r["status"] != UNRUN:
            continue
        if r["key"] in have:
            raise SystemExit(f"拒跑：trial_log 第 {r['id']} 列與第 {have[r['key']]} 列"
                             f"重複登記 {r['key']}")
        have[r["key"]] = r["id"]
    missing = need - set(have)
    if missing:
        lst = "、".join(f"{f}/{c}" for f, c in sorted(missing))
        raise SystemExit(f"拒跑：trial_log 沒有「{UNRUN}」的登記：{lst}")
    return {k: have[k] for k in need}


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _update_rows(trial_log: str, ids: dict, expect: str, make) -> None:
    by_id = {v: k for k, v in ids.items()}
    lines = open(trial_log, encoding="utf-8").read().split("\n")
    hit = 0
    for n, line in _table_lines("\n".join(lines)):
        cells = _cells(line)
        if cells[0] in by_id and len(cells) >= 5 and cells[4] == expect \
                and _row_trial(cells) == by_id[cells[0]]:
            make(cells, by_id[cells[0]])
            lines[n] = "| " + " | ".join(cells) + " |"
            hit += 1
    if hit != len(ids):
        raise RuntimeError(f"trial_log 只更新了 {hit}/{len(ids)} 列")
    text = "\n".join(lines)
    done = len(re.findall(r"^\|\s*\d+\s*\|.*執行 \|", text, flags=re.M))
    text = re.sub(r"## 累計嘗試次數：.*", lambda m: re.sub(
        r"已執行 \d+", f"已執行 {done}", m.group(0)), text)
    _atomic_write(trial_log, text)


def mark_running(trial_log: str, ids: dict) -> None:
    """真實回測開始前即標記為執行中：中途失敗也不能重複使用這組登記。"""
    today = dt.date.today().isoformat()

    def make(cells, key):
        cells[1] = f"{cells[1]}；{today} 執行"
        cells[4] = RUNNING
    _update_rows(trial_log, ids, UNRUN, make)


def fill_trial_log(trial_log: str, ids: dict, summary: pd.DataFrame) -> None:
    """把結果寫入執行中的列。"""
    res = {(r.factor, r.cut): (f"年化超額 {r.ann_excess * 100:.2f}%、t = {r.t:.2f}、"
                               f"DSR = {r.dsr:.3f}、{r.verdict}")
           for r in summary[summary.portfolio == "decile"].itertuples()}

    def make(cells, key):
        cells[4] = res[key]
    _update_rows(trial_log, ids, RUNNING, make)


# ---------------------------------------------------------------- holdout 鎖


LOCK_NAME = "HOLDOUT_LOCK.json"


def lock_paths(protocol: str, code_dir: str | None = None) -> list[str]:
    """兩把鎖：protocol 旁一把、程式旁一把（換 --protocol 或 --out 都繞不過）。"""
    a = os.path.join(os.path.dirname(os.path.abspath(protocol)), LOCK_NAME)
    b = os.path.join(code_dir or HERE, LOCK_NAME)
    return [a] if os.path.abspath(a) == os.path.abspath(b) else [a, b]


def check_holdout(protocol: str, out_root: str, confirmed: bool,
                  code_dir: str | None = None) -> None:
    if not confirmed:
        raise SystemExit("拒跑：holdout 只能驗證一次，須加 --i-understand-holdout-is-one-shot")
    for p in lock_paths(protocol, code_dir):
        if os.path.exists(p):
            raise SystemExit(f"拒跑：{p} 已存在，holdout 已執行過（或曾中斷）")
    if os.path.exists(os.path.join(out_root, "holdout")):
        raise SystemExit(f"拒跑：{os.path.join(out_root, 'holdout')} 已存在")
    if not os.path.exists(os.path.join(out_root, "exploration", "summary.csv")):
        raise SystemExit(f"拒跑：{out_root} 下尚無探索期結果（exploration/summary.csv）")


def write_lock(protocol: str, info: dict, code_dir: str | None = None) -> None:
    for p in lock_paths(protocol, code_dir):
        with open(p, "x", encoding="utf-8") as f:                # "x"：已存在即失敗
            json.dump(info, f, ensure_ascii=False, indent=2)


def update_lock(protocol: str, code_dir: str | None = None, **kw) -> None:
    for p in lock_paths(protocol, code_dir):
        with open(p, encoding="utf-8") as f:
            info = json.load(f)
        info.update(kw)
        _atomic_write(p, json.dumps(info, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- 流程


def check_fetch_coverage(data_dir: str, universe: pd.DataFrame, years: tuple) -> None:
    """(0)–(f) 合格者都必須抓過 PBR 與股數，否則 (g) 會把「沒抓」誤當「無值」。"""
    y0, y1 = years
    need = set(universe[(universe.year >= y0) & (universe.year <= y1)
                        & (universe.factor == "momentum")].stock_id)
    missing = sorted(need - U.fetched_ids(data_dir))
    if missing:
        raise SystemExit(f"拒跑：{len(missing)} 檔合格股尚未抓取 PBR／股數：{missing}\n"
                         "（先執行補抓，產生 fetch_extra.csv 與 *_extra.parquet）")


def check_end(calendar: pd.DatetimeIndex, end: pd.Timestamp) -> None:
    """終點須為交易日，且為該月最後一個交易日（月份完整，protocol 8.1）。"""
    if end not in calendar:
        raise ValueError(f"終點 {end.date()} 不是交易日")
    i = calendar.get_loc(end)
    if i + 1 >= len(calendar):
        raise ValueError(f"終點 {end.date()} 為資料最後一日，無法確認該月完整")
    if calendar[i + 1].month == end.month:
        raise ValueError(f"終點 {end.date()} 不是該月最後一個交易日")


def run_phase(panels: dict, universe: pd.DataFrame, phase: Phase,
              costs_decile: BT.Costs = BT.Costs(),
              costs_top: BT.Costs = BT.Costs.practical(),
              boot_b: int = S.BOOT_B, check_n: bool = True,
              n_trials: int = S.N_TRIALS) -> dict:
    cal = panels["calendar"]
    end = pd.Timestamp(phase.end)
    check_end(cal, end)
    y0, y1 = phase.years
    u = universe[(universe.year >= y0) & (universe.year <= y1)]
    if u.empty:
        raise ValueError("該期間 universe 為空")

    ports = {"decile": F.build_portfolios(panels, u, FACTORS, pct=0.10),
             "top10": F.build_portfolios(panels, u, FACTORS, top_n=10)}
    starts = sorted(pd.Timestamp(t) for t in u["T"].unique())
    if any(t > end for t in starts):
        raise ValueError("有選股日晚於終點")

    bench_nav = BT.benchmark(panels, starts[0], end, BENCH, capital=1.0)
    rb_full = S.monthly_returns(bench_nav, base=1.0)

    rows, monthly, trades, events = [], [], [], []
    for kind, costs in (("decile", costs_decile), ("top10", costs_top)):
        for fac in FACTORS:
            picks = BT.picks_from(ports[kind], fac)
            res = BT.run(panels, picks, costs, end=end)
            nav = res["nav"]
            if nav.index[0] != starts[0] or nav.index[-1] != end:
                raise ValueError(f"{kind}/{fac} 淨值起訖 {nav.index[0].date()}–"
                                 f"{nav.index[-1].date()} 與期間不符")
            rp_full = S.monthly_returns(nav, base=costs.capital)
            tr = res["trades"].assign(portfolio=kind, factor=fac)
            ev = res["events"].assign(portfolio=kind, factor=fac)
            trades.append(tr); events.append(ev)
            held = ports[kind][ports[kind].factor == fac].groupby("T").size()
            for cut in phase.cuts:
                rp, rb = rp_full, rb_full
                if cut == CUT_EX08:
                    rp, rb = S.drop_year(rp, 2008), S.drop_year(rb, 2008)
                if check_n and len(rp) != phase.expected_n[cut]:
                    raise ValueError(f"{phase.name}/{cut} 月數 {len(rp)}，protocol 規定 "
                                     f"{phase.expected_n[cut]}")
                r = S.evaluate(rp, rb, B=boot_b)
                r.update(phase=phase.name, portfolio=kind, factor=fac, cut=cut,
                         n_trades=len(tr), cost_total=float(tr.cost.sum()) / costs.capital,
                         n_delisted=int((ev.event == "delisted").sum()),
                         n_suspended=int((ev.event == "suspended_at_rebalance").sum()),
                         holdings_min=int(held.min()), holdings_max=int(held.max()))
                rows.append(r)
                monthly.append(pd.DataFrame({
                    "month": rp.index.strftime("%Y-%m"), "phase": phase.name,
                    "portfolio": kind, "factor": fac, "cut": cut,
                    "r_portfolio": rp.values, "r_benchmark": rb.values,
                    "excess": (rp - rb).values}))

    dec = S.family([r for r in rows if r["portfolio"] == "decile"], n_trials=n_trials)
    top = [r for r in rows if r["portfolio"] == "top10"]
    for r in top:
        r["verdict"] = S.verdict(r["t"], r["ann_excess"]) + "（次要，不納入檢定族）"
    summary = pd.DataFrame(dec + top)
    lead = ["phase", "portfolio", "factor", "cut", "n", "first_month", "last_month",
            "ann_excess", "t", "p", "p_bonferroni", "p_bh", "bh_reject", "dsr",
            "ir", "ci_ann_lo", "ci_ann_hi", "ci_ir_lo", "ci_ir_hi", "verdict"]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    picks = pd.concat([p.assign(portfolio=k) for k, p in ports.items()], ignore_index=True)
    return {"summary": summary, "monthly": pd.concat(monthly, ignore_index=True),
            "picks": picks, "trades": pd.concat(trades, ignore_index=True),
            "events": pd.concat(events, ignore_index=True)}


# ---------------------------------------------------------------- 結構乾跑


def placebo_panels(panels: dict, seed: int = 0) -> dict:
    """合成報酬取代真實價格、亂數取代市值與 PBR；保留缺值結構與成交量。

    合成資料與真實報酬、真實因子值皆無關，故結果不含任何績效或選股資訊。
    """
    ca = panels["close_adj"]
    rng = np.random.default_rng(seed)
    lvl = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, ca.shape), axis=0))
    syn = pd.DataFrame(lvl, index=ca.index, columns=ca.columns).where(ca.notna())
    vol = panels["volume"].reindex(index=ca.index, columns=ca.columns)
    p = dict(panels)
    p["close_adj"] = syn
    p["close_raw"] = syn                      # 調整因子 1
    p["amount"] = syn * vol                   # VWAP = 合成收盤
    for k in ("mcap", "pbr"):                 # 只保留「有無值」，數值改為亂數
        if panels.get(k) is not None:
            m = panels[k]
            p[k] = pd.DataFrame(rng.lognormal(0, 1, m.shape), index=m.index,
                                columns=m.columns).where(m.notna())
    return p


def structure_check(panels: dict, universe: pd.DataFrame, phase: Phase) -> pd.DataFrame:
    """以合成資料跑完整流程；任何例外或月數不符都會在此拋出。"""
    out = run_phase(placebo_panels(panels), universe, phase, boot_b=200)
    return out["summary"]


# ---------------------------------------------------------------- 報告


def _pct(x, d=2):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def report_md(out: dict, phase: Phase, prov: dict) -> str:
    s = out["summary"]
    title = (f"# PLACEBO 結構乾跑：{phase.name}（合成報酬，非真實績效）"
             if prov.get("placebo") else f"# {phase.name} 結果")
    L = [title, "",
         f"- 執行時間：{prov['run_at']}",
         f"- protocol SHA-256：`{prov['protocol_sha256']}`",
         f"- protocol git：{prov['protocol_git'] or '（無法取得）'}",
         f"- 期間：選股日 {phase.years[0]}–{phase.years[1]}，終點 {phase.end}",
         f"- 判定門檻：t > {S.T_THRESHOLD}；實務門檻：年化超額 > {S.PRACTICAL_MIN:.0%}", ""]
    for kind, title in (("decile", "十分位（主檢定）"), ("top10", "前 10 檔（實務參考，次要）")):
        part = s[s.portfolio == kind]
        L += [f"## {title}", "",
              "| 因子 | 切法 | n | 年化超額 | 95% CI | t (NW) | p | Bonf. p | BH p | DSR | IR | 判定 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            L.append(f"| {FACTOR_ZH[r.factor]} | {r.cut} | {r.n} | {_pct(r.ann_excess)} | "
                     f"[{_pct(r.ci_ann_lo)}, {_pct(r.ci_ann_hi)}] | {_num(r.t)} | "
                     f"{_num(r.p, 4)} | {_num(r.get('p_bonferroni'), 4)} | "
                     f"{_num(r.get('p_bh'), 4)} | {_num(r.get('dsr'), 3)} | {_num(r.ir)} | "
                     f"{r.verdict} |")
        L.append("")
    d = s[s.portfolio == "decile"].iloc[0]
    L += ["## 計算參數", "",
          f"- 檢定族大小 {int(d.family_size)}；DSR 的 N = {int(d.n_trials)}，"
          f"V[SR] = {d.var_sr:.6f}，SR₀（月）= {d.sr0_m:.4f}",
          f"- Newey-West lag：{sorted(set(s.lag))}；bootstrap 平均區塊 "
          f"{sorted(set(s.boot_block))}、B = {int(d.boot_B)}、種子 {int(d.boot_seed)}", ""]
    return "\n".join(L)


def save(out: dict, phase: Phase, out_dir: str, prov: dict, detail: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    keys = ("summary", "monthly", "picks", "trades", "events") if detail else ("summary",)
    for k in keys:
        out[k].to_csv(os.path.join(out_dir, f"{k}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md(out, phase, prov))
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 資料與主程式


DATA_FILES = ("twmarket_prices_adj.parquet", "twmarket_prices.parquet", "per.parquet",
              "shares.parquet", "dividend_result.parquet", "per_extra.parquet",
              "shares_extra.parquet")


def fingerprint(data_dir: str) -> dict:
    """面板的來源：資料檔與 universe.py 的 SHA-256。"""
    fp = {f: sha256(os.path.join(data_dir, f)) for f in DATA_FILES}
    fp["universe.py"] = sha256(os.path.join(HERE, "universe.py"))
    return fp


def load_panels(data_dir: str, cache: str | None) -> dict:
    """快取的指紋與目前的資料或程式不符時一律重建（避免舊快取悄悄抵銷修正）。"""
    fp = fingerprint(data_dir)
    if cache and os.path.exists(cache):
        obj = pd.read_pickle(cache)
        if isinstance(obj, dict) and obj.get("_fingerprint") == fp:
            return obj["panels"]
        print("面板快取與目前的資料或程式不符，重建中……")
    adj, raw, per, shr, div = U.load_all(data_dir)
    panels = U.make_panels(U.prepare(adj, div), raw, per, shr)
    del adj, raw
    if cache:
        pd.to_pickle({"_fingerprint": fp, "panels": panels}, cache)
    return panels


def main_placebo(a) -> int:
    phase = PHASES[a.phase]
    out_dir = os.path.join(a.out, f"placebo_{phase.name}")
    prov = provenance(a.protocol)
    prov.update(phase=phase.name, placebo=True)
    panels = load_panels(a.data_dir, a.panels_cache)
    universe = U.build(panels)
    out = run_phase(placebo_panels(panels), universe, phase)
    save(out, phase, out_dir, prov, detail=False)
    s = out["summary"]
    print(f"PLACEBO（合成報酬，非真實績效）｜{phase.name}｜"
          f"{len(s)} 個檢定，月數 {sorted(set(s.n))}，|t| 最大 {s.t.abs().max():.2f}")
    print(f"已存 {out_dir}")
    return 0


def main_run(a) -> int:
    phase = PHASES[a.phase]
    out_dir = os.path.join(a.out, phase.name)
    check_protocol(a.protocol, require_git=(phase.name == "holdout"))
    n_trials = max(S.N_TRIALS, registered_count(a.trial_log))
    ids = None
    if phase.name == "exploration":
        ids = check_exploration(a.trial_log)
        if os.path.exists(os.path.join(out_dir, "summary.csv")):
            raise SystemExit(f"拒跑：{out_dir} 已有結果")
    else:
        check_holdout(a.protocol, a.out, a.i_understand_holdout_is_one_shot)

    panels = load_panels(a.data_dir, a.panels_cache)
    universe = U.build(panels)
    check_fetch_coverage(a.data_dir, universe, phase.years)
    print("結構乾跑（合成報酬）……")
    structure_check(panels, universe, phase)
    print("結構乾跑通過")

    prov = provenance(a.protocol)
    prov.update(phase=phase.name, n_trials=n_trials, data_fingerprint=fingerprint(a.data_dir))
    if phase.name == "holdout":
        write_lock(a.protocol, {**prov, "status": "started"})
    else:
        mark_running(a.trial_log, ids)
    out = run_phase(panels, universe, phase, n_trials=n_trials)
    print(report_md(out, phase, prov))                 # 先印出，存檔失敗也不會遺失
    save(out, phase, out_dir, prov)
    if phase.name == "exploration":
        fill_trial_log(a.trial_log, ids, out["summary"])
        print(f"\n已把結果填入 {a.trial_log}（這 8 列已用掉）")
    else:
        update_lock(a.protocol, status="completed",
                    finished_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"已存 {out_dir}")
    return 0


# ---------------------------------------------------------------- 測試


def _synth_market(n_stocks=60, start="2005-01-03", end="2018-12-31", seed=0,
                  same_path=False):
    """合成市場：含 0050。same_path=True 時所有個股與 0050 同一路徑。"""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end)
    n = len(cal)
    base = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    frames, shr, per = [], [], []
    for i in range(n_stocks + 1):
        sid = BENCH if i == n_stocks else f"{1101 + i}"
        px = base.copy() if (same_path or sid == BENCH) else \
            30 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
        g = pd.DataFrame({"Date": cal, "Close": px, "stock_id": sid})
        g["Open"] = g.Close; g["High"] = g.Close * 1.002; g["Low"] = g.Close * 0.998
        g["Amount"] = 2e7 * (1 + i % 7)
        g["Volume"] = g.Amount / g.Close
        frames.append(g)
        if sid != BENCH:
            for d in cal[::20]:
                shr.append((d, sid, 1e7 * (1 + i)))
                per.append((d, sid, 0.5 + (i * 37 % 23) / 10))
    adj = pd.concat(frames, ignore_index=True)
    shares = pd.DataFrame(shr, columns=["date", "stock_id", "NumberOfSharesIssued"])
    pbr = pd.DataFrame(per, columns=["date", "stock_id", "PBR"])
    panels = U.make_panels(U.prepare(adj, None), adj.copy(), pbr, shares)
    return panels, adj


TEST_PHASES = {
    "exploration": Phase("exploration", (2007, 2012), "2012-12-31",
                         (CUT_ALL, CUT_EX08), {CUT_ALL: 72, CUT_EX08: 60}),
    "holdout": Phase("holdout", (2013, 2017), "2017-12-29",
                     (HOLDOUT_CUT,), {HOLDOUT_CUT: 60}),
}


def _t(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def _test_zero_excess():
    """所有個股與 0050 同路徑、零成本 ⇒ 每月超額報酬恆為 0（檢查起點、月界、基期對齊）。"""
    p, _ = _synth_market(same_path=True)
    u = U.build(p)
    out = run_phase(p, u, TEST_PHASES["exploration"], BT.Costs.zero(),
                    BT.Costs.zero(), boot_b=200)
    m = out["monthly"]
    return _t(f"同路徑零成本：超額報酬最大絕對值 {m.excess.abs().max():.1e}",
              m.excess.abs().max() < 1e-12)


def _test_first_month_base():
    """同路徑、主檢定成本：首月超額 = −(1+r_b)·c/(1+c)，其餘月份為 0（首月基期為期初資金）。"""
    p, _ = _synth_market(same_path=True)
    u = U.build(p)
    c = BT.Costs()
    out = run_phase(p, u, TEST_PHASES["exploration"], c, BT.Costs.zero(), boot_b=200)
    m = out["monthly"]
    m = m[(m.portfolio == "decile") & (m.cut == CUT_ALL)]
    rate = c.fee_rate + c.slippage
    first = m[m.month == "2007-01"]
    exp = -(1 + first.r_benchmark) * rate / (1 + rate)
    rest = m[m.month != "2007-01"].excess.abs().max()
    return _t(f"首月含進場成本（{first.excess.iloc[0]:.6f}），其餘月份 {rest:.1e}",
              np.allclose(first.excess, exp) and rest < 1e-12)


def _test_months_and_cuts():
    p, _ = _synth_market()
    u = U.build(p)
    ex = run_phase(p, u, TEST_PHASES["exploration"], boot_b=200)["summary"]
    ho = run_phase(p, u, TEST_PHASES["holdout"], boot_b=200)["summary"]
    a = ex[(ex.portfolio == "decile") & (ex.cut == CUT_ALL)]
    b = ex[(ex.portfolio == "decile") & (ex.cut == CUT_EX08)]
    ok = ((a.n == 72).all() and (b.n == 60).all() and (a.first_month == "2007-01").all()
          and (a.last_month == "2012-12").all() and (ho.first_month == "2013-01").all()
          and (ho.n == 60).all()
          and (ex[ex.portfolio == "decile"].family_size == 8).all()
          and (ho[ho.portfolio == "decile"].family_size == 4).all()
          and ex[ex.portfolio == "top10"].family_size.isna().all())
    return _t("期間與切法：月數、起訖月、檢定族大小（探索 8、holdout 4）", ok)


def _test_phase_isolation():
    """探索期結果不得受終點之後的價格影響（期間層級的 point-in-time）。"""
    p1, adj = _synth_market(seed=3)
    u1 = U.build(p1)
    r1 = run_phase(p1, u1, TEST_PHASES["exploration"], boot_b=200)["summary"]
    adj2 = adj.copy()
    late = adj2.Date > pd.Timestamp("2012-12-31")
    adj2.loc[late, ["Open", "High", "Low", "Close"]] *= 3.0
    p2 = U.make_panels(U.prepare(adj2, None), adj2.copy(), None, None)
    p2["mcap"], p2["pbr"] = p1["mcap"], p1["pbr"]
    u2 = U.build(p2)
    r2 = run_phase(p2, u2, TEST_PHASES["exploration"], boot_b=200)["summary"]
    cols = ["ann_excess", "t", "p", "dsr", "ci_ann_lo", "ci_ann_hi"]
    return _t("探索期不受終點之後的資料影響", np.allclose(r1[cols], r2[cols], equal_nan=True))


def _test_holdings_count():
    p, _ = _synth_market()
    u = U.build(p)
    out = run_phase(p, u, TEST_PHASES["exploration"], boot_b=200)
    pk = out["picks"]
    top = pk[pk.portfolio == "top10"].groupby(["T", "factor"]).size()
    dec = pk[pk.portfolio == "decile"].groupby(["T", "factor"]).size()
    n_univ = u.groupby(["T", "factor"]).size()
    exp = n_univ.map(F.decile_count).reindex(dec.index)
    return _t(f"持股數：前 10 檔皆 10 檔、十分位 = floor(n×0.1+0.5)（{dec.min()}–{dec.max()} 檔）",
              (top == 10).all() and dec.equals(exp.astype(dec.dtype)))


def _test_benchmark_excluded():
    p, _ = _synth_market()
    u = U.build(p)
    return _t("0050 不進入 universe（母體為普通股）", BENCH not in set(u.stock_id))


def _write_log(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| # | 日期 | 因子/規則 | 參數 | 探索期結果 | 備註 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(r + "\n")
        f.write("\n## 累計嘗試次數：已登記 8／已執行 0\n")


def _std_rows():
    rows, k = [], 0
    for f in FACTORS:
        for c in (CUT_ALL, CUT_EX08):
            k += 1
            rows.append(f"| {k} | 2026-10-04 登記 | {f}（{FACTOR_ZH[f]}） | 十分位、{c}；文獻設定 "
                        f"| {UNRUN} | P = 1 |")
    return rows


def _refuses(fn, *a):
    try:
        fn(*a)
        return False
    except SystemExit:
        return True


def _test_trial_log_guard():
    import tempfile
    ok = True
    summ = pd.DataFrame([dict(portfolio="decile", factor=f, cut=c, ann_excess=0.01,
                              t=1.0, dsr=0.5, verdict="不顯著")
                         for f in FACTORS for c in (CUT_ALL, CUT_EX08)])
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "trial_log.md")
        _write_log(log, _std_rows()); ok &= not _refuses(check_exploration, log)
        ok &= registered_count(log) == 8
        _write_log(log, _std_rows()[:7]); ok &= _refuses(check_exploration, log)       # 缺一列
        one = f"| 1 | 2026-10-04 | momentum/size/lowvol/value | 十分位、含 2008 與 排除 2008 | {UNRUN} | |"
        _write_log(log, [one]); ok &= _refuses(check_exploration, log)                 # 一列冒充 8 個
        bad = _std_rows()[:7] + [f"| 8 | 2026-10-04 | value（價值） | 十分位、不含 2008 | {UNRUN} | |"]
        _write_log(log, bad); ok &= _refuses(check_exploration, log)                   # 「不含」不得算「含」
        dup = _std_rows() + [_std_rows()[0].replace("| 1 |", "| 9 |")]
        _write_log(log, dup); ok &= _refuses(check_exploration, log)                   # 重複登記
        dupid = _std_rows() + [_std_rows()[1].replace("| 2 |", "| 1 |")]
        _write_log(log, dupid); ok &= _refuses(check_exploration, log)                 # 列號重複
        fenced = _std_rows()[:7] + ["```", _std_rows()[7], "```"]
        _write_log(log, fenced); ok &= _refuses(check_exploration, log)                # 程式碼區塊內的範例不算
        # 開跑即標記執行中：中途失敗也不能重用；跑完填入結果
        _write_log(log, _std_rows())
        ids = check_exploration(log)
        mark_running(log, ids)
        ok &= _refuses(check_exploration, log)
        fill_trial_log(log, ids, summ)
        text = open(log, encoding="utf-8").read()
        ok &= (_refuses(check_exploration, log) and "已執行 8" in text
               and UNRUN not in text and RUNNING not in text and registered_count(log) == 8)
        # 再登記 8 列新的試驗：可再跑，N 變為 16
        more = [r.replace(f"| {k} |", f"| {k + 8} |", 1)
                for k, r in enumerate(_std_rows(), 1)]
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n".join(more) + "\n")
        ok &= not _refuses(check_exploration, log) and registered_count(log) == 16
    return _t("trial_log 防呆：缺列、冒充、「不含」、重複、程式碼區塊、執行中／已執行皆拒跑；"
              "N 隨登記數增加", ok)


def _test_holdout_guard():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        code = os.path.join(d, "src"); os.makedirs(code)
        proto = os.path.join(d, "protocol.md"); open(proto, "w").write("x")
        out_root = os.path.join(d, "results")
        ch = lambda o, c: check_holdout(proto, o, c, code)
        ok &= _refuses(ch, out_root, True)                              # 無探索期結果
        os.makedirs(os.path.join(out_root, "exploration"))
        open(os.path.join(out_root, "exploration", "summary.csv"), "w").close()
        ok &= _refuses(ch, out_root, False)                             # 未確認
        ok &= not _refuses(ch, out_root, True)                          # 條件齊備
        os.makedirs(os.path.join(out_root, "holdout"))
        ok &= _refuses(ch, out_root, True)                              # 已有 holdout 結果
        os.rmdir(os.path.join(out_root, "holdout"))
        write_lock(proto, {"status": "started"}, code)
        ok &= os.path.exists(os.path.join(code, LOCK_NAME))
        ok &= _refuses(ch, out_root, True)                              # 鎖檔存在
        other = os.path.join(d, "other")                                # 換 --out 也擋
        os.makedirs(os.path.join(other, "exploration"))
        open(os.path.join(other, "exploration", "summary.csv"), "w").close()
        ok &= _refuses(ch, other, True)
        proto2 = os.path.join(d, "copy", "protocol.md")                 # 換 --protocol 也擋
        os.makedirs(os.path.dirname(proto2)); open(proto2, "w").write("x")
        ok &= _refuses(lambda: check_holdout(proto2, out_root, True, code))
        try:
            write_lock(proto, {"status": "again"}, code); ok = False    # 不得覆寫鎖檔
        except FileExistsError:
            pass
        ok &= _refuses(check_protocol, os.path.join(d, "missing.md"))   # protocol 不存在
        ok &= _refuses(lambda: check_protocol(proto, require_git=True)) # holdout 須在 git 中
    return _t("holdout 防呆：無探索結果、未確認、已有結果、鎖檔（換 --out／--protocol 亦擋）、"
              "非 git 皆拒跑", ok)


def _test_cache_fingerprint():
    """面板快取與資料或程式不符時必須重建。"""
    import tempfile
    p, adj = _synth_market(n_stocks=8, end="2008-12-31")
    with tempfile.TemporaryDirectory() as d:
        adj.to_parquet(os.path.join(d, "twmarket_prices_adj.parquet"), index=False)
        adj.to_parquet(os.path.join(d, "twmarket_prices.parquet"), index=False)
        pd.DataFrame({"date": adj.Date, "stock_id": adj.stock_id, "PER": 10.0, "PBR": 1.2,
                      "dividend_yield": 0.0}).to_parquet(os.path.join(d, "per.parquet"))
        pd.DataFrame({"date": adj.Date, "stock_id": adj.stock_id,
                      "NumberOfSharesIssued": 1e8}).to_parquet(os.path.join(d, "shares.parquet"))
        pd.DataFrame({"date": [], "stock_id": []}).to_parquet(
            os.path.join(d, "dividend_result.parquet"))
        cache = os.path.join(d, "panels.pkl")
        p1 = load_panels(d, cache)
        fp1 = pd.read_pickle(cache)["_fingerprint"]
        pd.DataFrame({"date": adj.Date, "stock_id": adj.stock_id, "PER": 10.0, "PBR": 2.5,
                      "dividend_yield": 0.0}).to_parquet(os.path.join(d, "per.parquet"))
        p2 = load_panels(d, cache)                                       # 資料變了 -> 重建
        fp2 = pd.read_pickle(cache)["_fingerprint"]
        pd.to_pickle(p1, cache)                                          # 舊格式（無指紋）
        p3 = load_panels(d, cache)
        ok = fp1 != fp2 and \
            float(p2["pbr"].iloc[-1].dropna().iloc[0]) == 2.5 and \
            float(p3["pbr"].iloc[-1].dropna().iloc[0]) == 2.5 and \
            float(p1["pbr"].iloc[-1].dropna().iloc[0]) == 1.2
    return _t("面板快取：資料改變或舊格式快取時重建，不沿用過期面板", ok)


def _test_fetch_coverage():
    import tempfile
    u = pd.DataFrame({"year": [2016, 2016, 2016], "factor": ["momentum"] * 3,
                      "stock_id": ["1101", "1102", "1103"]})
    with tempfile.TemporaryDirectory() as d:
        pd.Series(["1101", "1102"], name="stock_id").to_csv(
            os.path.join(d, "fetch_universe.csv"), index=False)
        ok = _refuses(check_fetch_coverage, d, u, (2016, 2026))     # 1103 沒抓過
        pd.Series(["1103"], name="stock_id").to_csv(
            os.path.join(d, "fetch_extra.csv"), index=False)
        ok &= not _refuses(check_fetch_coverage, d, u, (2016, 2026))
        ok &= not _refuses(check_fetch_coverage, d, u.iloc[:0], (2007, 2015))
    return _t("PBR／股數抓取涵蓋檢查：有合格股沒抓過即拒跑", ok)


def _test_end_check():
    cal = pd.bdate_range("2015-12-01", "2016-01-29")
    ok = True
    check_end(cal, pd.Timestamp("2015-12-31"))
    for bad in ("2015-12-30", "2016-01-29"):                    # 非月底／資料最後一日
        try:
            check_end(cal, pd.Timestamp(bad)); ok = False
        except ValueError:
            pass
    return _t("終點檢查：須為月底交易日且其後仍有資料", ok)


def _test_placebo():
    p, _ = _synth_market()
    q = placebo_panels(p, seed=1)
    same_nan = (q["close_adj"].isna().equals(p["close_adj"].isna())
                and q["pbr"].isna().equals(p["pbr"].isna()))
    vw = BT.vwap_adjusted(q).to_numpy()
    m = ~np.isnan(vw)
    vwap_ok = np.allclose(vw[m], q["close_adj"].to_numpy()[m])
    p2 = dict(p)
    p2["close_adj"] = p["close_adj"] * 7.0
    p2["mcap"] = p["mcap"] * 3.0; p2["pbr"] = p["pbr"] + 1.0
    q2 = placebo_panels(p2, seed=1)
    indep = all(q2[k].equals(q[k]) for k in ("close_adj", "mcap", "pbr"))
    return _t("placebo：保留缺值結構、VWAP = 合成收盤、與真實價格／市值／PBR 無關",
              same_nan and vwap_ok and indep)


def self_test() -> int:
    print("### run_trial.py 自我驗證（合成資料）###\n")
    res = [_test_trial_log_guard(), _test_holdout_guard(), _test_fetch_coverage(),
           _test_cache_fingerprint(), _test_end_check(),
           _test_benchmark_excluded(), _test_zero_excess(), _test_first_month_base(),
           _test_months_and_cuts(), _test_holdings_count(), _test_phase_isolation(),
           _test_placebo()]
    ok = all(res)
    print(f"\n  {sum(res)}/{len(res)} 通過"
          + ("" if ok else " —— 有測試未通過，不可用於真實資料"))
    return 0 if ok else 1


def _default(*names):
    for n in names:
        p = os.path.normpath(os.path.join(HERE, n))
        if os.path.exists(p):
            return p
    return os.path.normpath(os.path.join(HERE, names[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--data-dir", default="/content/drive/MyDrive/twmarket")
    ap.add_argument("--panels-cache", default=None, help="面板快取（pickle），可省去重建時間")
    ap.add_argument("--out", default=_default("../results", "results"))
    ap.add_argument("--protocol", default=_default("../protocol.md", "protocol.md"))
    ap.add_argument("--trial-log", default=_default("../docs/trial_log.md", "trial_log.md"))
    ap.add_argument("--i-understand-holdout-is-one-shot", action="store_true")
    ap.add_argument("--placebo", action="store_true",
                    help="以合成報酬乾跑完整流程（不含任何真實績效，不需登記）")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.phase:
        ap.error("須指定 --phase 或 --self-test")
    return main_placebo(a) if a.placebo else main_run(a)


if __name__ == "__main__":
    sys.exit(main())
