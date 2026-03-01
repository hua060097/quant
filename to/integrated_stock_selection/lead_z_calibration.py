#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead_z 自适应阈值校准脚本（面向 A 股 Top-1 日内/隔夜选股）

你给的验证集日志里至少包含“每天 Top-5 候选股票及其预测收益”，
但通常不包含“这一天 Top-1 的真实未来收益 y”与“预测日”的一一对齐表。
因此本脚本支持两种模式：

A) 只有日志：解析出每天 Top-K 的预测分布，并生成可用于实盘的“纯预测侧”自适应阈值：
   - lead 指标：pred_top1 - mean(pred_top2..top5) 等
   - lead_z：对 lead 做 rolling robust z（MAD）
   - threshold：对 lead_z 做 rolling 分位数阈值（过去 N 天的 q），避免拍脑袋 0.8

B) 日志 + 标签文件（推荐）：你额外提供一个 CSV，至少有列：
   day(YYYYMMDD 或 YYYY-MM-DD), y_top1（预测日 Top1 在持有期的真实收益）
   然后脚本会做 walk-forward 校准：只用过去数据选阈值，使得未来段的 Top1 收益/尾部风险更稳。

用法示例：
1) 仅日志（生成阈值方案，不做收益校准）：
   python lead_z_calibration.py --log your.log --out out_dir

2) 日志 + 标签（自动选“阈值分位数 q、rolling 窗口”等）：
   python lead_z_calibration.py --log your.log --label_csv y_top1.csv --out out_dir

输出：
- lead_table.csv：每天的 pred_topK、lead 指标、lead_z、推荐阈值等
- chosen_config.json：建议的在线计算参数（rolling 窗口、阈值分位数等）
"""

import pandas as pd
import numpy as np
from pathlib import Path

import datetime
import re
import string
import time
import math
import json


def _parse_topk_from_log(log_text: str):
    # 形如：🏆 2025-05-19 Top-5候选股票 (18只股票):
    re_block = re.compile(r'🏆\s*(\d{4}-\d{2}-\d{2})\s*Top-(\d+)候选股票')
    # 形如：  1. 301089.SZ: 预测收益=0.019371
    re_line  = re.compile(r'\s*\d+\.\s*(\d{6}\.(?:SZ|SH|BJ)):\s*预测收益=([-\d\.eE]+)')

    daily = {}
    cur_day = None
    cur_k = None

    for ln in log_text.splitlines():
        m = re_block.search(ln)
        if m:
            cur_day = m.group(1)
            cur_k = int(m.group(2))
            daily[cur_day] = []
            continue

        if cur_day is not None:
            m2 = re_line.search(ln)
            if m2:
                daily[cur_day].append((m2.group(1), float(m2.group(2))))
                if len(daily[cur_day]) >= cur_k:
                    cur_day, cur_k = None, None
                continue

            # 某些日志在“保存Top-K预测结果到:”后结束块
            if "保存Top-" in ln:
                cur_day, cur_k = None, None

    # 转成表：每行=一天
    rows = []
    for day, items in sorted(daily.items()):
        preds = [p for _, p in items]
        if len(preds) < 2:
            continue
        top1 = preds[0]
        top2 = preds[1]
        topk_mean = float(np.mean(preds))
        top2to5_mean = float(np.mean(preds[1:])) if len(preds) >= 5 else float(np.mean(preds[1:]))
        topk_std = float(np.std(preds, ddof=0))

        rows.append({
            "day": day,
            "k": len(preds),
            "pred_top1": top1,
            "pred_top2": top2,
            "pred_topk_mean": topk_mean,
            "pred_top2to5_mean": top2to5_mean,
            "pred_topk_std": topk_std,
            # 常用 lead 家族
            "lead_vs_top2": top1 - top2,
            "lead_vs_top2to5mean": top1 - top2to5_mean,
            "lead_vs_topkmean": top1 - topk_mean,
            "lead_ratio_vs_top2to5mean": (top1 - top2to5_mean) / (abs(top2to5_mean) + 1e-9),
            "lead_std_norm": (top1 - top2to5_mean) / (topk_std + 1e-9),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("未能从日志解析出 Top-K 预测块；请确认日志包含“🏆 YYYY-MM-DD Top-5候选股票 ...”段落。")

    return df

def _robust_z_rolling(x: pd.Series, window: int, min_periods: int):
    # rolling MAD z-score
    x = pd.to_numeric(x, errors="coerce").astype("float64")
    med = x.rolling(window, min_periods=min_periods).median()
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median()
    z = (x - med) / (1.4826 * mad + 1e-9)
    return z.clip(-8.0, 8.0)

def _rolling_quantile(x: pd.Series, window: int, min_periods: int, q: float):
    x = pd.to_numeric(x, errors="coerce").astype("float64")
    return x.rolling(window, min_periods=min_periods).quantile(q)

def _load_label_csv(path: str):
    lab = pd.read_csv(path)
    if "day" not in lab.columns:
        raise ValueError("label_csv 必须包含 day 列（YYYYMMDD 或 YYYY-MM-DD）")
    # 兼容 day=20250203 / "2025-02-03"
    s = lab["day"].astype(str).str.strip().str.replace("-", "", regex=False).str.slice(0, 8)
    lab["day"] = pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    # y 列名兼容
    ycol = None
    for c in ["y_top1", "ret_top1", "y", "return"]:
        if c in lab.columns:
            ycol = c
            break
    if ycol is None:
        raise ValueError("label_csv 需要包含 y_top1/ret_top1/y/return 其中之一作为真实收益列")
    lab = lab[["day", ycol]].rename(columns={ycol: "y_top1"})
    lab["y_top1"] = pd.to_numeric(lab["y_top1"], errors="coerce").astype("float64")
    return lab.dropna()

def _walkforward_choose_quantile(df: pd.DataFrame, lead_col: str, z_window: int, q_grid, eval_splits=3):
    """
    在“只有 y_top1 的情况下”做简单的 walk-forward 选 q：
    - 先算 lead_z
    - 用 lead_z 的 rolling 分位数阈值做出手
    - 评价：出手日的 y_top1 均值的 LCB95（保守），并惩罚 CVaR10、MDD
    """
    work = df.sort_values("day").reset_index(drop=True).copy()
    work["lead_z"] = _robust_z_rolling(work[lead_col], window=z_window, min_periods=max(20, z_window//2))

    # 分段：把样本均分为 eval_splits 段
    n = len(work)
    if n < 90:
        # 样本太少就不硬分段，直接用全样本
        split_points = [0, n]
    else:
        split_points = [int(n * i / eval_splits) for i in range(eval_splits)] + [n]

    best = None
    for q in q_grid:
        scores = []
        for si in range(len(split_points)-1):
            a, b = split_points[si], split_points[si+1]
            # 只用过去数据计算阈值：对每个点 t，用 [0..t-1] 的分位数
            # 这里用 rolling quantile + shift(1) 实现“严格因果”
            thr = _rolling_quantile(work["lead_z"], window=60, min_periods=40, q=q).shift(1)
            take = (work["lead_z"] >= thr).astype(bool)

            seg = work.iloc[a:b].copy()
            seg_take = take.iloc[a:b].to_numpy()
            rets = seg.loc[seg_take, "y_top1"].to_numpy()
            if rets.size < 15:
                # 交易太少的 q 没意义，直接记很差
                scores.append(-1e9)
                continue

            mean = float(np.mean(rets))
            std = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
            lcb95 = mean - 1.645 * (std / math.sqrt(rets.size) + 1e-12)

            # CVaR10
            k = max(1, int(rets.size * 0.10))
            cvar10 = float(np.mean(np.sort(rets)[:k]))

            # MDD（复利）
            eq = np.cumprod(1.0 + rets)
            peak = np.maximum.accumulate(eq)
            mdd = float(np.min(eq / (peak + 1e-12) - 1.0))

            score = (
                lcb95
                - 0.50 * abs(cvar10)
                - 0.20 * abs(mdd)
            )
            scores.append(score)

        score_mean = float(np.mean(scores))
        if (best is None) or (score_mean > best["score_mean"]):
            best = {"q": q, "score_mean": score_mean, "scores": scores}

    return best

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="验证集回测日志文件路径")
    ap.add_argument("--label_csv", default=None, help="可选：包含 day 和 y_top1 的CSV（用于阈值校准）")
    ap.add_argument("--out", default="lead_z_out", help="输出目录")
    ap.add_argument("--lead_col", default="lead_vs_top2to5mean", help="使用哪个 lead 指标列")
    ap.add_argument("--z_window", type=int, default=60, help="lead_z rolling window")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    df = _parse_topk_from_log(log_text)
    # 标准化 day
    df["day"] = pd.to_datetime(df["day"], format="%Y-%m-%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    # 计算 lead_z
    df["lead_z"] = _robust_z_rolling(df[args.lead_col], window=args.z_window, min_periods=max(20, args.z_window//2))

    chosen = {
        "lead_col": args.lead_col,
        "z_window": args.z_window,
        # 默认：rolling quantile 阈值（保守一些，不拍脑袋 0.8）
        "thr_window": 60,
        "thr_min_periods": 40,
        "thr_quantile": 0.70,  # 若无标签，先用“过去60天的70分位”作为出手阈值
        "note": "若提供 label_csv，本值会自动校准为更优的分位数。",
    }

    if args.label_csv:
        lab = _load_label_csv(args.label_csv)
        df2 = df.merge(lab, on="day", how="inner")
        if len(df2) < 60:
            print("[WARN] label 样本太少，无法稳定校准；仍输出默认阈值。")
        else:
            q_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
            best = _walkforward_choose_quantile(df2, lead_col=args.lead_col, z_window=args.z_window, q_grid=q_grid)
            if best:
                chosen["thr_quantile"] = float(best["q"])
                chosen["calib_score_mean"] = float(best["score_mean"])
                chosen["calib_scores_by_split"] = [float(x) for x in best["scores"]]

    # 严格因果阈值：用过去窗口分位数并 shift(1)
    df["thr_lead_z"] = _rolling_quantile(df["lead_z"], window=chosen["thr_window"], min_periods=chosen["thr_min_periods"], q=chosen["thr_quantile"]).shift(1)
    df["should_trade"] = (df["lead_z"] >= df["thr_lead_z"]).astype(int)

    out_csv = os.path.join(args.out, "lead_table.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = os.path.join(args.out, "chosen_config.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    print("[OK] 输出完成：")
    print(" -", out_csv)
    print(" -", out_json)
    print("\n推荐在线规则（伪码）：")
    print(f"  lead = {chosen['lead_col']}")
    print(f"  lead_z = rolling_MAD_z(lead, window={chosen['z_window']})")
    print(f"  thr = rolling_quantile(lead_z, window={chosen['thr_window']}, q={chosen['thr_quantile']}) shifted by 1 day")
    print("  trade if lead_z >= thr")

if __name__ == "__main__":
    main()
