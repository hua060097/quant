import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json"   # 改成你的文件名


def max_drawdown_from_returns(ret):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def cvar_left_tail(ret, q=0.10):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    th = np.quantile(x, q)
    tail = x[x <= th]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def summarize(name, arr):
    x = pd.Series(arr).dropna().astype(float)
    if len(x) == 0:
        print(f"\n=== {name} ===")
        print("EMPTY")
        return None

    out = {
        "count": len(x),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "win_rate": float((x > 0).mean()),
        "big_loss_3pct_rate": float((x <= -0.03).mean()),
        "big_loss_5pct_rate": float((x <= -0.05).mean()),
        "big_loss_8pct_rate": float((x <= -0.08).mean()),
        "cvar10": cvar_left_tail(x, 0.10),
        "max_drawdown": max_drawdown_from_returns(x),
        "sharpe": float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)),
        "p5": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }

    print(f"\n=== {name} ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    return out


def diff_stats(name_x, x, name_y, y):
    a = pd.Series(x).dropna().astype(float)
    b = pd.Series(y).dropna().astype(float)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].values
    b = b.loc[idx].values
    d = a - b

    print(f"\n=== Diff: {name_x} - {name_y} ===")
    print("count:", len(d))
    print("mean_diff:", float(np.mean(d)))
    print("median_diff:", float(np.median(d)))
    print("P(diff>0):", float(np.mean(d > 0)))
    print("P(diff<-0.03):", float(np.mean(d < -0.03)))
    print("P(diff<-0.05):", float(np.mean(d < -0.05)))
    print("p5:", float(np.quantile(d, 0.05)))
    print("p10:", float(np.quantile(d, 0.10)))
    print("p90:", float(np.quantile(d, 0.90)))
    print("p95:", float(np.quantile(d, 0.95)))


with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for date_str, rec in data.items():
    row = {"date": date_str}

    # baseline
    row["baseline"] = None
    serial_top1 = rec.get("serial_top1", {})
    if isinstance(serial_top1, dict):
        row["baseline"] = serial_top1.get("real_return", None)

    # hard filter
    risk_hard = rec.get("risk_hard_filter", {})
    for k in ["remove_1", "remove_2", "remove_3", "remove_5"]:
        row[f"hard_{k}"] = None
        if isinstance(risk_hard, dict) and k in risk_hard:
            row[f"hard_{k}"] = risk_hard[k].get("real_return", None)

    # soft penalty
    risk_soft = rec.get("risk_soft_penalty", {})
    for lam in ["0.20", "0.50", "1.00"]:
        row[f"soft_{lam}"] = None
        key = f"lambda_{lam}"
        if isinstance(risk_soft, dict) and key in risk_soft:
            row[f"soft_{lam}"] = risk_soft[key].get("real_return", None)

    rows.append(row)

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df = df.set_index("date")

strategies = [
    "baseline",
    "hard_remove_1",
    "hard_remove_2",
    "hard_remove_3",
    "hard_remove_5",
    "soft_0.20",
    "soft_0.50",
    "soft_1.00",
]

summary = {}
for col in strategies:
    summary[col] = summarize(col, df[col])

print("\n=== Key comparison table ===")
comp_cols = [
    "mean", "median", "win_rate",
    "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
    "cvar10", "max_drawdown", "sharpe"
]
comp = pd.DataFrame(summary).T[comp_cols]
print(comp.to_string(float_format=lambda x: f"{x:.6f}"))

# 和 baseline 做对比
for col in strategies[1:]:
    diff_stats(col, df[col], "baseline", df["baseline"])

# 自动挑选最优策略（你可以按自己偏好改权重）
def strategy_score(row):
    return (
        row["mean"]
        + 0.6 * row["median"]
        + 0.05 * row["sharpe"]
        - 0.8 * row["big_loss_5pct_rate"]
        - 0.3 * abs(row["max_drawdown"])
    )

score_table = comp.copy()
score_table["score"] = score_table.apply(strategy_score, axis=1)
score_table = score_table.sort_values("score", ascending=False)

print("\n=== Ranking by composite score ===")
print(score_table[comp_cols + ["score"]].to_string(float_format=lambda x: f"{x:.6f}"))