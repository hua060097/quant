import json, random
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"
path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"
N_SIM = 2000  # 随机模拟次数（B策略）
TOPK_B = 3
TOPK_C = 5

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

# 固定策略序列（A、C）按日收益
rets_A = []
rets_C = []

# B是随机策略：我们要得到一个“分布”，所以做N_SIM条路径
rets_B_paths = [[] for _ in range(N_SIM)]

for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        continue

    # 按p_cls排序
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    # A: 分类Top1
    A = cls_by_p[0].get("real_return")
    rets_A.append(A)

    # B: 分类Top3随机选1（做N_SIM次）
    topB = cls_by_p[:TOPK_B]
    topB_rets = [x.get("real_return") for x in topB]
    for i in range(N_SIM):
        rets_B_paths[i].append(random.choice(topB_rets))

    # C: 分类Top5 + 排序选1（在Top5里按rank_score最大）
    topC = cls_by_p[:TOPK_C]
    topC_by_rank = sorted(topC, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    C = topC_by_rank[0].get("real_return")
    rets_C.append(C)

def summarize(name, arr):
    arr = np.array(arr, dtype=float)
    print(f"\n{name}: n={len(arr)} mean={arr.mean():.6f} std={arr.std(ddof=1):.6f}")
    for p in [5,10,25,50,75,90,95]:
        print(f"  p{p}: {np.percentile(arr,p):.6f}")

# 汇总A/C
print("=== Strategy A/C (single path) ===")
summarize("A (ClsTop1)", rets_A)
summarize("C (ClsTop5 -> Rank pick 1)", rets_C)

# 汇总B：对每条随机路径算均值，再看均值的分布
B_means = [np.mean(path) for path in rets_B_paths]
print("\n=== Strategy B (ClsTop3 random pick) ===")
summarize("B mean distribution (over simulations)", B_means)

# 也可以顺便看：B在“最差5%/最好5%”路径均值是什么
B_means_sorted = np.sort(np.array(B_means))
print("\nB mean: p5/p50/p95 =", np.percentile(B_means_sorted,5), np.percentile(B_means_sorted,50), np.percentile(B_means_sorted,95))

print("\nDone.")