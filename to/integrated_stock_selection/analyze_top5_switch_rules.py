import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_top5_json(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for d, v in obj.items():
        row = {"date": pd.to_datetime(d)}
        for k in range(1, 6):
            info = v[f"top{k}"]
            row[f"pred{k}"] = float(info["pred"])
            row[f"ret{k}"] = float(info["real_return"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def max_drawdown(rets) -> float:
    arr = np.asarray(rets, dtype=float)
    if len(arr) == 0:
        return 0.0
    nav = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def summarize(rets) -> dict:
    arr = np.asarray(rets, dtype=float)
    return {
        "mean": float(arr.mean()),
        "win": float((arr > 0).mean()),
        "cum": float(np.prod(1.0 + arr) - 1.0),
        "mdd": max_drawdown(arr),
    }


def run_strategy(df: pd.DataFrame, name: str):
    picks = []

    if name.startswith("fixed_top"):
        k = int(name.replace("fixed_top", ""))
        picks = df[f"ret{k}"].tolist()
    elif name.startswith("top1_vs_top2_mean_"):
        look = int(name.rsplit("_", 1)[1])
        for i in range(len(df)):
            if i < look:
                k = 1
            else:
                m1 = df["ret1"].iloc[i - look : i].mean()
                m2 = df["ret2"].iloc[i - look : i].mean()
                k = 2 if m2 > m1 else 1
            picks.append(df.iloc[i][f"ret{k}"])
    elif name.startswith("pred_gap_top2_thr_"):
        thr = float(name.rsplit("_", 1)[1])
        for _, row in df.iterrows():
            k = 2 if (row["pred1"] - row["pred2"]) <= thr else 1
            picks.append(row[f"ret{k}"])
    elif name.startswith("rolling_mean_argmax_"):
        look = int(name.rsplit("_", 1)[1])
        for i in range(len(df)):
            if i < look:
                k = 1
            else:
                means = {kk: df[f"ret{kk}"].iloc[i - look : i].mean() for kk in range(1, 6)}
                k = max(means, key=means.get)
            picks.append(df.iloc[i][f"ret{k}"])
    elif name.startswith("if_last3_top") and "_gt_top1_choose_top" in name:
        choose = int(name.split("_")[2].replace("top", ""))
        for i in range(len(df)):
            k = 1
            if i >= 3:
                diff = (df[f"ret{choose}"] - df["ret1"]).iloc[i - 3 : i]
                if (diff > 0).all():
                    k = choose
            picks.append(df.iloc[i][f"ret{k}"])
    else:
        raise ValueError(f"Unknown strategy: {name}")

    return summarize(picks)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/daily_top5_details_Window_3_20260320_012750.json")
    df = load_top5_json(path)
    mid = len(df) // 2
    first = df.iloc[:mid].copy()
    second = df.iloc[mid:].copy()

    strategy_names = []
    strategy_names.extend([f"fixed_top{k}" for k in range(1, 6)])
    strategy_names.extend([f"top1_vs_top2_mean_{look}" for look in [3, 5, 10, 20]])
    strategy_names.extend([f"pred_gap_top2_thr_{thr}" for thr in [0.002, 0.005, 0.01]])
    strategy_names.extend([f"rolling_mean_argmax_{look}" for look in [3, 5, 10, 20]])
    strategy_names.extend([f"if_last3_top{k}_gt_top1_choose_top{k}" for k in [2, 3, 4, 5]])

    rows = []
    for name in strategy_names:
        full = run_strategy(df, name)
        f = run_strategy(first, name)
        s = run_strategy(second, name)
        rows.append(
            {
                "name": name,
                "full_mean": full["mean"],
                "full_win": full["win"],
                "full_cum": full["cum"],
                "full_mdd": full["mdd"],
                "first_mean": f["mean"],
                "first_cum": f["cum"],
                "second_mean": s["mean"],
                "second_cum": s["cum"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["full_cum", "full_mean"], ascending=False)
    pd.set_option("display.max_rows", 200)
    print("=== Full Sample Top Strategies ===")
    print(out.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Best First-Half Strategies ===")
    print(out.sort_values("first_cum", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Best Second-Half Strategies ===")
    print(out.sort_values("second_cum", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
