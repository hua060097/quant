"""
compare_stage15.py — Stage1.5 Rank 诊断结果对比表

用法:
    python compare_stage15.py --new results/stage15_20260901_032805 --label "111+bucket"

对每个 stage15 目录:
  1. 读取 rank_daily.csv / stock_details_top5.csv
  2. 逐 checkpoint 计算: mean / median / q10 / 拼接MDD(逐折等权曲线min) 
  3. 选择规则(硬门槛): MDD_min > -60% 且 q10 > -10% 的 cp 中取 median 最高者
     (与110基准表"选中cp"口径一致, 无正折门槛)
  4. 在选中 cp 上计算 C率(future_ret<=-0.05) / A率(>=0.07) / A/C / mean / median
  5. 与内嵌基线表合并打印, 并可存 CSV

基线表 (来自历史实验, 见 2026-08-31 汇总):
  110基准  cp400  C17.1  A17.9  A/C1.05  mean0.0121  median0.0057
  113(+3)  cp450  ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_ROWS = [
    {"label": "110基准", "cp": 400, "C": 17.1, "A": 17.9, "AC": 1.05, "mean": 0.0121, "median": 0.0057},
    {"label": "113(+3)", "cp": 450, "C": 14.7, "A": 16.7, "AC": 1.14, "mean": 0.0120, "median": 0.0048},
    {"label": "114(+市场)", "cp": 450, "C": 13.5, "A": 15.5, "AC": 1.15, "mean": 0.0119, "median": 0.0042},
    {"label": "111+fpd", "cp": 450, "C": 16.3, "A": 15.1, "AC": 0.93, "mean": 0.0107, "median": 0.0040},
    {"label": "111+hvnp", "cp": 300, "C": 15.5, "A": 12.7, "AC": 0.82, "mean": 0.0056, "median": 0.0005},
    {"label": "111+hhi", "cp": 200, "C": 15.9, "A": 15.5, "AC": 0.97, "mean": 0.0079, "median": -0.0006},
]

MDD_GATE = -0.60   # 拼接MDD(逐折min) 硬门槛
Q10_GATE = -0.10   # q10 硬门槛


def _fold_equity_mdd(returns: pd.Series) -> float:
    e = np.cumprod(1 + returns.values)
    return float(((e / np.maximum.accumulate(e)) - 1).min())


def analyze_dir(stage15_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """返回 (逐cp明细表, 选中cp行, 指标dict)。"""
    rank_path = stage15_dir / "rank_daily.csv"
    stock_path = stage15_dir / "stock_details_top5.csv"
    if not rank_path.exists() or not stock_path.exists():
        raise FileNotFoundError(f"缺少 rank_daily.csv / stock_details_top5.csv: {stage15_dir}")

    rank = pd.read_csv(rank_path, encoding="utf-8-sig")
    stock = pd.read_csv(stock_path, encoding="utf-8-sig")
    r1 = rank[rank["rank"] == 1].copy()

    rows = []
    for cp in sorted(r1["checkpoint"].unique()):
        sub = r1[r1["checkpoint"] == cp]
        ret = sub["return"].dropna()
        mdds = [_fold_equity_mdd(g.sort_values("date")["return"]) for _, g in sub.groupby("fold_id")]
        rows.append({
            "cp": cp,
            "mean": float(ret.mean()),
            "median": float(ret.median()),
            "q10": float(ret.quantile(0.10)),
            "mdd_min": float(min(mdds)),
            "mdd_mean": float(np.mean(mdds)),
            "n_folds": int(sub["fold_id"].nunique()),
        })
    detail = pd.DataFrame(rows)

    ok = detail[(detail["mdd_min"] > MDD_GATE) & (detail["q10"] > Q10_GATE)]
    if len(ok) == 0:
        ok = detail
    sel = ok.sort_values("median", ascending=False).iloc[0]

    cp = int(sel["cp"])
    st1 = stock[(stock["checkpoint"] == cp) & (stock["rank"] == 1)]
    fr = st1["future_ret"].dropna()
    A = float((fr >= 0.07).mean() * 100)
    C = float((fr <= -0.05).mean() * 100)
    metrics = {
        "cp": cp,
        "C": round(C, 1),
        "A": round(A, 1),
        "AC": round(A / max(C, 1e-9), 2),
        "mean": round(float(sel["mean"]), 4),
        "median": round(float(sel["median"]), 4),
    }
    return detail, sel, metrics


def main():
    ap = argparse.ArgumentParser(description="Stage1.5 结果对比表")
    ap.add_argument("--new", required=True, help="新结果目录, 如 results/stage15_20260901_032805")
    ap.add_argument("--label", default="新运行", help="新运行在表中的显示名")
    ap.add_argument("--baseline", nargs="*", default=[], help="额外参与对比的 stage15 目录(可选, 可多个)")
    ap.add_argument("--out", default="", help="保存CSV路径(可选)")
    args = ap.parse_args()

    new_dir = Path(args.new).expanduser()
    if not new_dir.exists():
        raise FileNotFoundError(f"目录不存在: {new_dir}")

    rows = [dict(r) for r in BASELINE_ROWS]
    print("=" * 88)
    print(f"对比: {args.label} ({new_dir})")
    print("=" * 88)

    detail, sel, m = analyze_dir(new_dir)
    print(f"\n[逐cp明细] {args.label}")
    print(detail.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[选择规则] 硬门槛 MDD_min>{MDD_GATE:.0%} 且 q10>{Q10_GATE:.0%} → 取 median 最高 cp"
          f"\n选中 cp={m['cp']}  (median={m['median']:.4f}, mean={m['mean']:.4f})")
    rows.append({"label": args.label, **m})

    for bd in args.baseline:
        b_dir = Path(bd).expanduser()
        if b_dir.exists():
            _, _, bm = analyze_dir(b_dir)
            rows.append({"label": b_dir.name, **bm})

    df = pd.DataFrame(rows)
    print("\n================ 对比表 ================")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.1f}"))

    best = df[df["label"] != args.label].sort_values("median", ascending=False)
    if len(best):
        top = best.iloc[0]
        diff_median = m["median"] - top["median"]
        verdict = "优于全部基线" if m["median"] > top["median"] else f"不及最佳基线({top['label']}, median {top['median']:.4f}, 差 {diff_median:+.4f})"
        print(f"\n结论: {verdict}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"已保存: {out}")


if __name__ == "__main__":
    main()