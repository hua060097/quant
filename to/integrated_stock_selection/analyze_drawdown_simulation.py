# -*- coding: utf-8 -*-
"""
分钟路径回放：用 230 baseline 的 120 笔交易（101 主 + 19 影子），
模拟"drawdown_from_highest ≤ -5% 即卖出（仅 T+2/T+3 生效）"规则的实际效果。

核心问题：
1. 12 笔到期低收益样本中，有多少笔在 T+2/T+3 真的出现过 ≥5% 的从峰值回撤？
2. 14 笔到期高收益样本中，有多少笔曾出现 ≥5% 中途回撤后再创新高（误杀风险）？
3. 47 笔利润保护样本中，新规则会让出场点变好/变坏多少？
4. 把所有 120 笔合起来，新规则相比 230 baseline 的累计收益变化是多少？

输出：
- analyze_drawdown_simulation_per_trade.csv
- analyze_drawdown_simulation_summary.txt
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SELECTIONS = ROOT / "results" / "walkforward_selections_20260618_202018.json"
SHADOW = ROOT / "results" / "walkforward_shadow_skipped_opportunities_20260618_202018.csv"
MINUTE_BASE = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/minute")
OUT_CSV = ROOT / "analyze_drawdown_simulation_per_trade.csv"
OUT_TXT = ROOT / "analyze_drawdown_simulation_summary.txt"

# 模拟参数
DRAWDOWN_THRESHOLD = -0.05  # 用户提议的阈值

PEAK_RE = re.compile(r"峰值(-?\d+\.\d+)%")


def classify_exit(reason: str) -> str:
    if not reason:
        return "未知"
    if "T+2 09:35失败止损" in reason:
        return "T2_0935_硬止损"
    if "T+2 10:00继续走坏" in reason:
        return "T2_1000_软止损"
    if "T+2收盘仍未转正" in reason:
        return "T2_1455_时间止损"
    if "T+3 09:35仍未转正" in reason:
        return "T3_0935_时间止损"
    if "利润保护" in reason:
        return "利润保护"
    if "高峰位保护" in reason:
        return "高峰位保护"
    if "小盈利后延续失败" in reason:
        return "小盈利失败延续"
    if "到期" in reason:
        return "持有到期"
    return "其他"


def load_trades() -> pd.DataFrame:
    rows = []
    raw = json.loads(SELECTIONS.read_text(encoding="utf-8"))
    for win_obj in raw:
        for key, trades in win_obj.get("selections", {}).items():
            if str(key) != "1":
                continue
            for t in trades:
                row = dict(t)
                row["data_source"] = "main"
                rows.append(row)
    main_df = pd.DataFrame(rows)

    sdf = pd.read_csv(SHADOW)
    sdf = sdf[sdf["strategy_type"] == "Top-1"].copy()
    sdf["data_source"] = "shadow"
    sdf = sdf[main_df.columns.intersection(sdf.columns).tolist() + ["data_source"]] if False else sdf
    common = [c for c in main_df.columns if c in sdf.columns]
    sdf = sdf[common]
    main_df = main_df[common]

    df = pd.concat([main_df, sdf], ignore_index=True)
    df["buy_datetime"] = pd.to_datetime(df["buy_datetime"])
    df["sell_datetime"] = pd.to_datetime(df["sell_datetime"])
    df["buy_price"] = pd.to_numeric(df["buy_price"], errors="coerce")
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce")
    df["exit_class"] = df["exit_reason"].apply(classify_exit)
    return df


def load_minute_for_trade(stock_code: str, buy_dt: pd.Timestamp, sell_dt: pd.Timestamp) -> pd.DataFrame:
    """加载 buy_dt~sell_dt 之间的 5 分钟 K 线数据（跨月也能拼接）。"""
    months = set()
    cur = buy_dt.to_pydatetime().date().replace(day=1)
    end = sell_dt.to_pydatetime().date().replace(day=1)
    while cur <= end:
        months.add(cur.strftime("%Y%m"))
        # 下一月份
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    frames = []
    for ym in sorted(months):
        f = MINUTE_BASE / ym / "minute_data.parquet"
        if not f.exists():
            continue
        try:
            sub = pd.read_parquet(
                f,
                filters=[
                    ("datetime", ">=", buy_dt.to_pydatetime()),
                    ("datetime", "<=", sell_dt.to_pydatetime()),
                    ("stock_code", "==", stock_code),
                ],
                columns=["datetime", "stock_code", "open", "high", "low", "close"],
            )
            if not sub.empty:
                frames.append(sub)
        except Exception as e:
            print(f"[WARN] 读取 {f} 失败 {stock_code}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def simulate_one_trade(trade: dict) -> dict:
    """对一笔交易回放分钟路径，计算新规则触发情况。"""
    stock = trade["stock_code"]
    buy_dt = trade["buy_datetime"]
    sell_dt = trade["sell_datetime"]
    buy_price = float(trade["buy_price"])
    actual_sell_price = float(trade["sell_price"])
    actual_return = float(trade["actual_return"])

    bars = load_minute_for_trade(stock, buy_dt, sell_dt)
    if bars.empty or buy_price <= 0:
        return {
            "stock_code": stock,
            "data_source": trade.get("data_source"),
            "buy_date": buy_dt.date(),
            "exit_class": trade.get("exit_class"),
            "actual_return": actual_return,
            "minute_bars_loaded": 0,
            "new_rule_triggered": None,
            "note": "no_minute_data",
        }

    # 标记每根 K 线是 T+1(HD0), T+2(HD1), T+3(HD2) 哪一天
    unique_dates = sorted(bars["datetime"].dt.date.unique())
    date_to_offset = {d: i for i, d in enumerate(unique_dates)}
    bars["holding_day_offset"] = bars["datetime"].dt.date.map(date_to_offset)

    # 跟踪峰值（从买入起，包括 T+1 内）
    peak_price = buy_price
    max_peak_return = 0.0
    peak_dt = None
    max_drawdown = 0.0  # 整个持仓期间的最大回撤
    max_dd_dt = None

    new_exit_dt = None
    new_exit_price = None
    new_exit_return = None
    new_exit_peak = None  # 触发瞬间的 peak

    for _, bar in bars.iterrows():
        # 用 high 更新峰值（保守估计漏抓与误伤都用 high）
        if bar["high"] > peak_price:
            peak_price = bar["high"]
            peak_dt = bar["datetime"]
            max_peak_return = (peak_price - buy_price) / buy_price

        # 用 close 计算当前回撤（与代码一致）
        cur_close = bar["close"]
        dd = (cur_close - peak_price) / peak_price if peak_price > 0 else 0.0
        if dd < max_drawdown:
            max_drawdown = dd
            max_dd_dt = bar["datetime"]

        # 仅 T+2/T+3 生效
        if new_exit_dt is None and bar["holding_day_offset"] in (1, 2):
            if dd <= DRAWDOWN_THRESHOLD:
                new_exit_dt = bar["datetime"]
                new_exit_price = float(cur_close)
                new_exit_return = (cur_close - buy_price) / buy_price
                new_exit_peak = (peak_price - buy_price) / buy_price

    triggered = new_exit_dt is not None
    # 比较：新规则若触发，且时间早于实际卖出 → 用新规则的价格
    new_realized_return = actual_return
    if triggered and new_exit_dt < sell_dt:
        new_realized_return = new_exit_return
        compare_status = "提前触发"
    elif triggered and new_exit_dt >= sell_dt:
        # 实际卖出已经发生在新规则触发之前 → 不影响
        compare_status = "晚于实际"
        new_realized_return = actual_return
    else:
        compare_status = "未触发"

    delta_return = new_realized_return - actual_return

    return {
        "stock_code": stock,
        "data_source": trade.get("data_source"),
        "buy_date": buy_dt.date(),
        "buy_price": buy_price,
        "actual_sell_dt": sell_dt,
        "actual_sell_price": actual_sell_price,
        "actual_return": actual_return,
        "exit_class": trade.get("exit_class"),
        "max_peak_return": max_peak_return,
        "peak_dt": peak_dt,
        "max_drawdown": max_drawdown,  # 整段最深回撤
        "max_dd_dt": max_dd_dt,
        "minute_bars_loaded": len(bars),
        "new_rule_triggered": triggered,
        "new_exit_dt": new_exit_dt,
        "new_exit_price": new_exit_price,
        "new_exit_return": new_exit_return,
        "new_exit_peak_at_trigger": new_exit_peak,
        "compare_status": compare_status,
        "new_realized_return": new_realized_return,
        "delta_return": delta_return,  # 正=新规则赚更多, 负=被误杀
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_trades()
    print(f"[INFO] 共 {len(df)} 笔交易（main {sum(df.data_source=='main')}, shadow {sum(df.data_source=='shadow')}）")

    results = []
    for i, row in df.iterrows():
        if i % 20 == 0:
            print(f"  ... 处理 {i+1}/{len(df)}")
        results.append(simulate_one_trade(row.to_dict()))

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    out = []
    out.append(f"# 新规则模拟报告 (drawdown ≤ {DRAWDOWN_THRESHOLD*100:.0f}% 即卖, T+2/T+3 生效)")
    out.append(f"# 数据：230 baseline 120 笔交易")

    # 排除无数据
    have = res_df[res_df["new_rule_triggered"].notna()].copy()
    no_data = res_df[res_df["new_rule_triggered"].isna()]
    out.append(f"\n有效样本: {len(have)} (无分钟数据: {len(no_data)})")

    triggered = have[have["compare_status"] == "提前触发"]
    not_trig = have[have["compare_status"] == "未触发"]
    late = have[have["compare_status"] == "晚于实际"]
    out.append(f"  提前触发: {len(triggered)} 笔, 累计 delta = {triggered['delta_return'].sum()*100:+.2f}pp")
    out.append(f"  晚于实际: {len(late)} 笔（不影响结果）")
    out.append(f"  未触发  : {len(not_trig)} 笔")

    # 累积净影响
    total_actual = have["actual_return"].sum() * 100
    total_new = have["new_realized_return"].sum() * 100
    out.append(f"\n累计算术收益对比 (n={len(have)}):")
    out.append(f"  baseline   : {total_actual:+.2f}pp (avg {total_actual/len(have):+.3f}%)")
    out.append(f"  新规则模拟  : {total_new:+.2f}pp (avg {total_new/len(have):+.3f}%)")
    out.append(f"  差异       : {total_new - total_actual:+.2f}pp")

    # ★ 按 exit_class 分组
    out.append("\n按 230 出场类别拆分（baseline → 新规则）：")
    for cls, g in have.groupby("exit_class"):
        actual_avg = g["actual_return"].mean() * 100
        new_avg = g["new_realized_return"].mean() * 100
        n_trig = (g["compare_status"] == "提前触发").sum()
        delta = (new_avg - actual_avg) * len(g)
        out.append(
            f"  {cls:<20s}: n={len(g):>3d} | 触发 {n_trig:>3d} | "
            f"actual {actual_avg:+6.2f}% → new {new_avg:+6.2f}% (Δ累计 {delta:+6.2f}pp)"
        )

    # ★ 12 笔到期低收益的具体处理结果
    out.append("\n【关键】持有到期 actual<6% 样本的新规则模拟：")
    expiry_low = have[(have["exit_class"] == "持有到期") & (have["actual_return"] < 0.06)]
    for _, r in expiry_low.sort_values("actual_return").iterrows():
        peak = r["max_peak_return"] * 100
        dd = r["max_drawdown"] * 100
        ar = r["actual_return"] * 100
        if r["compare_status"] == "提前触发":
            ner = r["new_exit_return"] * 100
            out.append(
                f"  {r['stock_code']:<11s} buy={r['buy_date']} | peak={peak:+6.2f}% maxDD={dd:+6.2f}% "
                f"| 实际 {ar:+6.2f}% → 新规则 {ner:+6.2f}% (Δ {ner-ar:+6.2f}pp) ✅"
            )
        else:
            out.append(
                f"  {r['stock_code']:<11s} buy={r['buy_date']} | peak={peak:+6.2f}% maxDD={dd:+6.2f}% "
                f"| 实际 {ar:+6.2f}% (新规则未触发，maxDD<{DRAWDOWN_THRESHOLD*100:.0f}%)"
            )

    # ★ 14 笔到期高收益的"误杀"风险
    out.append("\n【关键】持有到期 actual≥10% 高收益样本的新规则模拟（误杀检查）：")
    expiry_high = have[(have["exit_class"] == "持有到期") & (have["actual_return"] >= 0.10)]
    for _, r in expiry_high.sort_values("actual_return", ascending=False).iterrows():
        peak = r["max_peak_return"] * 100
        dd = r["max_drawdown"] * 100
        ar = r["actual_return"] * 100
        if r["compare_status"] == "提前触发":
            ner = r["new_exit_return"] * 100
            ptrig = r["new_exit_peak_at_trigger"] * 100
            tag = "💥 误杀" if (ner < ar) else "✅ OK"
            out.append(
                f"  {r['stock_code']:<11s} | finalPeak={peak:+6.2f}% maxDD={dd:+6.2f}% "
                f"| 实际 {ar:+6.2f}% → 新规则 {ner:+6.2f}% (触发瞬peak={ptrig:+5.2f}%, Δ {ner-ar:+6.2f}pp) {tag}"
            )
        else:
            out.append(
                f"  {r['stock_code']:<11s} | finalPeak={peak:+6.2f}% maxDD={dd:+6.2f}% "
                f"| 实际 {ar:+6.2f}% (新规则未触发) ✅"
            )

    # ★ 47 笔利润保护：新规则 vs 实际
    out.append("\n【参考】47 笔利润保护样本：新规则 vs 实际：")
    pp = have[have["exit_class"] == "利润保护"]
    actual_pp_mean = pp["actual_return"].mean() * 100
    new_pp_mean = pp["new_realized_return"].mean() * 100
    n_diff = (pp["delta_return"].abs() > 0.001).sum()
    out.append(f"  n={len(pp)}, 改变出场: {n_diff} 笔")
    out.append(f"  实际平均 {actual_pp_mean:+.2f}% → 新规则平均 {new_pp_mean:+.2f}% (Δ {new_pp_mean-actual_pp_mean:+.2f}pp)")
    big_diffs = pp[pp["delta_return"].abs() > 0.005].sort_values("delta_return")
    if len(big_diffs) > 0:
        out.append(f"  Δ>0.5pp 的 {len(big_diffs)} 笔：")
        for _, r in big_diffs.iterrows():
            out.append(
                f"    {r['stock_code']:<11s}: 实际 {r['actual_return']*100:+5.2f}% → 新 {r['new_realized_return']*100:+5.2f}% (Δ {r['delta_return']*100:+5.2f}pp)"
            )

    # ★ 用户特别关注：peak ≥ 9% 但 current 跌破 6% 的"漏洞场景"实测
    out.append("\n【漏洞场景实测】peak≥9% 且 max_drawdown≤-5% 但 ProfitProtect 未触发的样本：")
    leak = have[
        (have["exit_class"] != "利润保护")
        & (have["exit_class"] != "高峰位保护")
        & (have["max_peak_return"] >= 0.09)
        & (have["max_drawdown"] <= -0.05)
    ]
    if len(leak) == 0:
        out.append("  (无样本) - 230 baseline 中没有这种漏洞场景！")
    else:
        for _, r in leak.iterrows():
            ar = r["actual_return"] * 100
            ner = r["new_exit_return"] * 100 if r["new_exit_return"] is not None else float("nan")
            out.append(
                f"  {r['stock_code']:<11s} buy={r['buy_date']} cls={r['exit_class']:<14s}: "
                f"peak={r['max_peak_return']*100:+5.2f}% maxDD={r['max_drawdown']*100:+5.2f}% "
                f"实际 {ar:+5.2f}% → 新规则 {ner:+5.2f}%"
            )

    if len(no_data) > 0:
        out.append(f"\n[WARN] 无分钟数据样本 ({len(no_data)} 笔):")
        for _, r in no_data.iterrows():
            out.append(f"  {r['stock_code']} buy={r['buy_date']}")

    OUT_TXT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n[OK] 明细: {OUT_CSV.name}")
    print(f"[OK] 总结: {OUT_TXT.name}")


if __name__ == "__main__":
    main()
