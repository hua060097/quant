import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"
ROLL = 10

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        continue

    # A: 分类Top1
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)
    cls_top1_ret = cls_by_p[0].get("real_return")

    # C: 分类Top5 -> 排序选1
    top5 = cls_by_p[:5]
    top5_by_rank = sorted(top5, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    good_ret = top5_by_rank[0].get("real_return")

    rows.append({
        "date": d,
        "cls_top1_ret": cls_top1_ret,
        "good_ret": good_ret,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# rolling10 of Top5->rank
df["rolling10_good_ret"] = df["good_ret"].rolling(ROLL).mean()

# baseline
baseline_mean = df["good_ret"].mean()

print("Baseline Top5->rank mean:", baseline_mean)

# 搜索阈值
thresholds = np.arange(-0.03, 0.011, 0.005)

results = []

for th in thresholds:
    # 方案1: 差的时候不交易
    ret_skip = []
    for _, row in df.iterrows():
        rr = row["rolling10_good_ret"]
        if pd.isna(rr):
            ret_skip.append(row["good_ret"])   # warmup 先继续交易
        elif rr < th:
            ret_skip.append(0.0)              # 不交易
        else:
            ret_skip.append(row["good_ret"])
    mean_skip = np.mean(ret_skip)

    # 方案2: 差的时候切到分类Top1
    ret_switch = []
    for _, row in df.iterrows():
        rr = row["rolling10_good_ret"]
        if pd.isna(rr):
            ret_switch.append(row["good_ret"])
        elif rr < th:
            ret_switch.append(row["cls_top1_ret"])
        else:
            ret_switch.append(row["good_ret"])
    mean_switch = np.mean(ret_switch)

    results.append({
        "threshold": th,
        "mean_skip": mean_skip,
        "mean_switch": mean_switch,
    })

res = pd.DataFrame(results)
print("\n=== Threshold search ===")
print(res.to_string(index=False))

best_skip = res.loc[res["mean_skip"].idxmax()]
best_switch = res.loc[res["mean_switch"].idxmax()]

print("\n=== Best ===")
print("Best skip threshold:", best_skip["threshold"], "mean:", best_skip["mean_skip"])
print("Best switch threshold:", best_switch["threshold"], "mean:", best_switch["mean_switch"])