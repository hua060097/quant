import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    rank_all = rec.get("rank_all_top1") or {}
    if len(cls) == 0:
        continue

    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    top5 = cls_by_p[:5]
    if len(top5) == 0:
        continue

    top5_by_rank = sorted(top5, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)

    # A: 分类Top1
    A = cls_by_p[0].get("real_return")

    # D: 纯排序Top1（全市场）
    D = rank_all.get("real_return")

    # P5: 分类Top5等权
    p5_vals = [x.get("real_return") for x in top5 if x.get("real_return") is not None]
    P5 = float(np.mean(p5_vals)) if len(p5_vals) > 0 else None

    # R1 / R2 / R3
    def mean_first_k(arr, k):
        vals = [x.get("real_return") for x in arr[:k] if x.get("real_return") is not None]
        return float(np.mean(vals)) if len(vals) > 0 else None

    R1 = mean_first_k(top5_by_rank, 1)
    R2 = mean_first_k(top5_by_rank, 2)
    R3 = mean_first_k(top5_by_rank, 3)

    rows.append({
        "date": d,
        "A_clsTop1": A,
        "R1_top5_rank_top1": R1,
        "R2_top5_rank_top2eq": R2,
        "R3_top5_rank_top3eq": R3,
        "D_rankall_top1": D,
        "P5_cls_top5eq": P5,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ===== 风险统计函数 =====
def max_drawdown_from_returns(returns):
    """基于日度收益序列构造伪净值曲线，计算最大回撤"""
    x = pd.Series(returns).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return dd.min()

def max_consecutive_losses(returns):
    x = pd.Series(returns).dropna().astype(float).values
    best = 0
    cur = 0
    for r in x:
        if r < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def cvar_left_tail(returns, q=0.1):
    """左尾CVaR：最差q分位内的平均收益"""
    x = pd.Series(returns).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    th = np.quantile(x, q)
    tail = x[x <= th]
    return tail.mean() if len(tail) > 0 else np.nan

def summarize_strategy(name, s):
    x = pd.Series(s).dropna().astype(float)
    if len(x) == 0:
        return None
    out = {
        "count": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "std": x.std(ddof=1),
        "win_rate": (x > 0).mean(),
        "p5": np.quantile(x, 0.05),
        "p10": np.quantile(x, 0.10),
        "p25": np.quantile(x, 0.25),
        "p75": np.quantile(x, 0.75),
        "p90": np.quantile(x, 0.90),
        "p95": np.quantile(x, 0.95),
        "loss_rate": (x < 0).mean(),
        "big_loss_3pct_rate": (x <= -0.03).mean(),
        "big_loss_5pct_rate": (x <= -0.05).mean(),
        "big_loss_8pct_rate": (x <= -0.08).mean(),
        "cvar10": cvar_left_tail(x, 0.10),
        "cvar5": cvar_left_tail(x, 0.05),
        "max_drawdown": max_drawdown_from_returns(x),
        "max_consecutive_losses": max_consecutive_losses(x),
    }
    print(f"\n=== {name} ===")
    for k, v in out.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    return out

strategies = [
    "A_clsTop1",
    "R1_top5_rank_top1",
    "R2_top5_rank_top2eq",
    "R3_top5_rank_top3eq",
    "D_rankall_top1",
    "P5_cls_top5eq",
]

summary = {}
for col in strategies:
    summary[col] = summarize_strategy(col, df[col])

# ===== 重点对比表 =====
comp = pd.DataFrame(summary).T[
    [
        "mean", "median", "win_rate",
        "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
        "cvar10", "cvar5",
        "max_drawdown", "max_consecutive_losses"
    ]
]

print("\n=== Key comparison table ===")
print(comp.to_string(float_format=lambda x: f"{x:.6f}"))

# ===== 两两差值：R1 vs A / R2 / D =====
def diff_stats(name_x, x, name_y, y):
    a = pd.Series(x).dropna().astype(float).values
    b = pd.Series(y).dropna().astype(float).values
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    d = a - b
    print(f"\n=== Diff: {name_x} - {name_y} ===")
    print("mean_diff:", d.mean())
    print("median_diff:", np.median(d))
    print("p5:", np.quantile(d, 0.05))
    print("p10:", np.quantile(d, 0.10))
    print("p90:", np.quantile(d, 0.90))
    print("p95:", np.quantile(d, 0.95))
    print("P(diff>0):", (d > 0).mean())
    print("P(diff<-0.03):", (d < -0.03).mean())
    print("P(diff<-0.05):", (d < -0.05).mean())

diff_stats("R1", df["R1_top5_rank_top1"], "A", df["A_clsTop1"])
diff_stats("R1", df["R1_top5_rank_top1"], "R2", df["R2_top5_rank_top2eq"])
diff_stats("R1", df["R1_top5_rank_top1"], "D", df["D_rankall_top1"])

print("\nDone.")