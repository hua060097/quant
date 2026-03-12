import json
import pandas as pd
import numpy as np


JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"   # 改成你的json文件


# ----------------------------
# 工具函数
# ----------------------------

def calc_stats(arr):

    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return None

    stats = {}

    stats["count"] = len(arr)
    stats["mean"] = np.mean(arr)
    stats["median"] = np.median(arr)
    stats["std"] = np.std(arr)

    stats["win_rate"] = np.mean(arr > 0)

    stats["p5"] = np.percentile(arr, 5)
    stats["p10"] = np.percentile(arr, 10)
    stats["p25"] = np.percentile(arr, 25)
    stats["p75"] = np.percentile(arr, 75)
    stats["p90"] = np.percentile(arr, 90)
    stats["p95"] = np.percentile(arr, 95)

    stats["big_loss_3pct_rate"] = np.mean(arr < -0.03)
    stats["big_loss_5pct_rate"] = np.mean(arr < -0.05)
    stats["big_loss_8pct_rate"] = np.mean(arr < -0.08)

    # CVaR
    stats["cvar10"] = arr[arr <= np.percentile(arr, 10)].mean()
    stats["cvar5"] = arr[arr <= np.percentile(arr, 5)].mean()

    # 最大回撤
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    stats["max_drawdown"] = drawdown.min()

    # 最大连续亏损
    loss = arr < 0
    max_loss_streak = 0
    cur = 0

    for l in loss:
        if l:
            cur += 1
            max_loss_streak = max(max_loss_streak, cur)
        else:
            cur = 0

    stats["max_consecutive_losses"] = max_loss_streak

    return stats


# ----------------------------
# 读取json
# ----------------------------

with open(JSON_PATH, "r") as f:
    data = json.load(f)

rows = []

for d in data:

    r = {}

    r["date"] = d["date"]

    r["rank_top1_ret"] = d.get("rank_all_top1_ret")
    r["rank_top2_ret"] = d.get("rank_all_top2_ret")

    rows.append(r)

df = pd.DataFrame(rows)


# ----------------------------
# Top1 Top2
# ----------------------------

top1 = df["rank_top1_ret"].dropna()
top2 = df["rank_top2_ret"].dropna()

diff = top1 - top2


print("\n====== Top1 ======\n")
for k,v in calc_stats(top1).items():
    print(k,":",v)

print("\n====== Top2 ======\n")
for k,v in calc_stats(top2).items():
    print(k,":",v)

print("\n====== Top1 - Top2 ======\n")
for k,v in calc_stats(diff).items():
    print(k,":",v)


print("\n=== diff probability ===")
print("P(Top1 > Top2):", np.mean(diff > 0))
print("P(Top1 < Top2):", np.mean(diff < 0))
print("P(diff < -0.03):", np.mean(diff < -0.03))
print("P(diff < -0.05):", np.mean(diff < -0.05))