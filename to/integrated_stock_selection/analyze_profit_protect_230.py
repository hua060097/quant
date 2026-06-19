# -*- coding: utf-8 -*-
"""
分析 230 baseline 日志中 ProfitProtect 触发情况，识别"漏抓"和"误杀"样本。

数据源：
1. results/walkforward_selections_20260618_202018.json    (101 笔实际交易)
2. results/walkforward_shadow_skipped_opportunities_20260618_202018.csv  (19 笔影子)

输出：
- 控制台分类统计
- analyze_profit_protect_230_per_trade.csv   每笔交易明细分类
- analyze_profit_protect_230_summary.txt     文字总结
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).parent
SELECTIONS = ROOT / "results" / "walkforward_selections_20260618_202018.json"
SHADOW = ROOT / "results" / "walkforward_shadow_skipped_opportunities_20260618_202018.csv"
OUT_CSV = ROOT / "analyze_profit_protect_230_per_trade.csv"
OUT_TXT = ROOT / "analyze_profit_protect_230_summary.txt"

PEAK_RE = re.compile(r"峰值(-?\d+\.\d+)%")
DRAWDOWN_RE = re.compile(r"回撤(-?\d+\.\d+)%")
BEARISH_RE = re.compile(r"连续(\d+)根")


def classify_exit(reason: str) -> str:
    """把 exit_reason 文本归类。"""
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
    if "小盈利后延续失败" in reason or "跌破T+1低点" in reason:
        return "小盈利失败延续"
    if "到期" in reason:
        return "持有到期"
    if "硬止损" in reason or "止损" in reason:
        return "其他止损"
    return "其他"


def parse_peak(reason: str) -> Optional[float]:
    m = PEAK_RE.search(reason or "")
    return float(m.group(1)) / 100.0 if m else None


def parse_drawdown(reason: str) -> Optional[float]:
    m = DRAWDOWN_RE.search(reason or "")
    return float(m.group(1)) / 100.0 if m else None


def parse_bearish_count(reason: str) -> Optional[int]:
    m = BEARISH_RE.search(reason or "")
    return int(m.group(1)) if m else None


def load_main_trades() -> pd.DataFrame:
    raw = json.loads(SELECTIONS.read_text(encoding="utf-8"))
    rows = []
    for window_obj in raw:
        win = window_obj.get("window_name", "Window 3")
        sel = window_obj.get("selections", {})
        # selections 的 key 是 strategy_type 数字，比如 '1','2','3','5'
        for key, trades in sel.items():
            if str(key) != "1":
                continue  # 只看 Top-1
            for t in trades:
                row = dict(t)
                row["window_name"] = win
                row["data_source"] = "main"
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def load_shadow_trades() -> pd.DataFrame:
    df = pd.read_csv(SHADOW)
    # 只看 Top-1
    df = df[df["strategy_type"] == "Top-1"].copy()
    df["data_source"] = "shadow"
    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exit_class"] = df["exit_reason"].apply(classify_exit)
    df["peak_return"] = df["exit_reason"].apply(parse_peak)
    df["drawdown_at_exit"] = df["exit_reason"].apply(parse_drawdown)
    df["bearish_count_at_exit"] = df["exit_reason"].apply(parse_bearish_count)
    # 实际收益 (注意 main 和 shadow 列名一致：actual_return)
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce")
    # 峰值 -> 实现收益的吐回比例 (peak - actual)
    df["give_back_pct"] = df.apply(
        lambda r: (r["peak_return"] - r["actual_return"]) if pd.notna(r["peak_return"]) else None,
        axis=1,
    )
    return df


def summarize(df: pd.DataFrame, label: str, out_lines: list) -> None:
    out_lines.append(f"\n========== {label} (n={len(df)}) ==========")
    if len(df) == 0:
        return

    # 1. 出场分类
    counts = df["exit_class"].value_counts()
    avg_ret = df.groupby("exit_class")["actual_return"].mean()
    out_lines.append("\n出场分类分布：")
    for cls, n in counts.items():
        ar = avg_ret.get(cls, float("nan"))
        out_lines.append(f"  {cls:<20s}: {n:>3d} 笔, 平均收益 {ar*100:+6.2f}%")

    # 2. 利润保护触发样本：峰值/回撤/吐回 分布
    pp = df[df["exit_class"] == "利润保护"]
    if len(pp) > 0:
        out_lines.append(f"\n利润保护触发样本 (n={len(pp)})：")
        out_lines.append(
            f"  峰值收益:  mean={pp['peak_return'].mean()*100:+5.2f}%, "
            f"median={pp['peak_return'].median()*100:+5.2f}%, "
            f"min={pp['peak_return'].min()*100:+5.2f}%, "
            f"max={pp['peak_return'].max()*100:+5.2f}%"
        )
        out_lines.append(
            f"  实现收益:  mean={pp['actual_return'].mean()*100:+5.2f}%, "
            f"median={pp['actual_return'].median()*100:+5.2f}%"
        )
        out_lines.append(
            f"  吐回峰值:  mean={pp['give_back_pct'].mean()*100:+5.2f}%, "
            f"median={pp['give_back_pct'].median()*100:+5.2f}%, "
            f"max={pp['give_back_pct'].max()*100:+5.2f}%"
        )
        # 按峰值高低分桶
        bins = [(0.06, 0.09), (0.09, 0.12), (0.12, 0.18), (0.18, 1.0)]
        out_lines.append("  按峰值分桶（实现收益分布）：")
        for lo, hi in bins:
            sub = pp[(pp["peak_return"] >= lo) & (pp["peak_return"] < hi)]
            if len(sub) == 0:
                continue
            out_lines.append(
                f"    [{lo*100:.0f}%, {hi*100:.0f}%): n={len(sub):>2d}, "
                f"actual_mean={sub['actual_return'].mean()*100:+5.2f}%, "
                f"give_back_mean={sub['give_back_pct'].mean()*100:+5.2f}%"
            )

    # 3. 持有到期样本（疑似漏抓）：用 actual_return 区间分桶
    expiry = df[df["exit_class"] == "持有到期"]
    if len(expiry) > 0:
        out_lines.append(f"\n持有到期样本 (n={len(expiry)})：")
        out_lines.append(
            f"  实现收益:  mean={expiry['actual_return'].mean()*100:+5.2f}%, "
            f"median={expiry['actual_return'].median()*100:+5.2f}%, "
            f"min={expiry['actual_return'].min()*100:+5.2f}%, "
            f"max={expiry['actual_return'].max()*100:+5.2f}%"
        )
        ret_bins = [(-1.0, 0.0), (0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 1.0)]
        out_lines.append("  按实现收益分桶：")
        for lo, hi in ret_bins:
            sub = expiry[(expiry["actual_return"] >= lo) & (expiry["actual_return"] < hi)]
            if len(sub) == 0:
                continue
            out_lines.append(
                f"    [{lo*100:+.0f}%, {hi*100:+.0f}%): n={len(sub):>2d}, mean={sub['actual_return'].mean()*100:+5.2f}%"
            )
        # 疑似漏抓：到期且 actual_return < 6% 的（可能峰值 ≥ 9% 但回撤后只剩这点）
        suspect = expiry[expiry["actual_return"] < 0.06].sort_values("actual_return")
        if len(suspect) > 0:
            out_lines.append(f"\n  ⚠️ 持有到期但 actual_return<6% 的样本 (n={len(suspect)})：")
            for _, r in suspect.iterrows():
                out_lines.append(
                    f"    {r['stock_code']:<11s} | buy={str(r['buy_datetime'])[:10]} "
                    f"| actual={r['actual_return']*100:+5.2f}% | exit={r['exit_reason']}"
                )

    # 4. 利润保护吐回最大的 top-N（误杀候选 / 大吐回）
    if len(pp) > 0:
        big_giveback = pp.nlargest(10, "give_back_pct")
        out_lines.append(f"\n  💥 利润保护触发后吐回最大 top10：")
        for _, r in big_giveback.iterrows():
            out_lines.append(
                f"    {r['stock_code']:<11s} | peak={r['peak_return']*100:+5.2f}% "
                f"actual={r['actual_return']*100:+5.2f}% giveback={r['give_back_pct']*100:+5.2f}%"
                f" bearish={r['bearish_count_at_exit']}"
            )

    # 5. 总收益贡献
    if len(df) > 0:
        out_lines.append(f"\n  策略总平均收益: {df['actual_return'].mean()*100:+5.2f}% (n={len(df)})")


def main():
    out_lines = []
    out_lines.append(f"# 230 baseline ProfitProtect 分析报告")
    out_lines.append(f"# 数据源: {SELECTIONS.name}, {SHADOW.name}")

    main_df = annotate(load_main_trades())
    shadow_df = annotate(load_shadow_trades())

    # 主交易和影子分别统计
    summarize(main_df, "主交易 (实际持仓)", out_lines)
    summarize(shadow_df, "影子交易 (满仓跳过的机会)", out_lines)

    # 合并视角：理论上如果资金无限，能抓住的全部 alpha
    combined_cols = [
        "data_source", "stock_code", "buy_datetime", "sell_datetime",
        "buy_price", "sell_price", "actual_return",
        "exit_reason", "exit_class",
        "peak_return", "drawdown_at_exit", "bearish_count_at_exit",
        "give_back_pct", "predicted_return",
    ]
    main_keep = [c for c in combined_cols if c in main_df.columns]
    shadow_keep = [c for c in combined_cols if c in shadow_df.columns]
    combined = pd.concat(
        [main_df[main_keep], shadow_df[shadow_keep]],
        ignore_index=True, sort=False,
    )
    summarize(combined, "合并 (主交易 + 影子)", out_lines)

    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_TXT.write_text("\n".join(out_lines), encoding="utf-8")

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("\n".join(out_lines))
    print(f"\n[OK] 明细已保存: {OUT_CSV.name}")
    print(f"[OK] 总结已保存: {OUT_TXT.name}")


if __name__ == "__main__":
    main()
