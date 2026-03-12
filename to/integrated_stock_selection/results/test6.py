import json
import numpy as np
import pandas as pd

# path = "daily_gate_stats_Window_3_gateN80_20260305_225809.json"
# path = "daily_gate_stats_Window_3_gateN5_20260308_233542.json"
path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

# 参数
TOPK_SMALL = 5      # Top5→排序
SEG_LEN = 20        # 时间分段长度（交易日）
ROLL = 10           # rolling 观察
BOOT_ITER = 5000    # bootstrap 次数
BLOCK = 5           # block bootstrap 块长度（保留时间相关性）

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

def pick_top5_rank1(rec, topk=5):
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        return None
    # 先按p_cls取TopK
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)[:topk]
    # 再按rank_score取Top1
    cls_by_rank = sorted(cls_by_p, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    return cls_by_rank[0].get("real_return")

def pick_serial_top1(rec):
    st1 = rec.get("serial_top1") or {}
    return st1.get("real_return")

rows = []
for d in dates:
    rec = data[d]
    r80 = pick_serial_top1(rec)               # 你当前Top80串行（候选80里按rank选1）
    r5  = pick_top5_rank1(rec, TOPK_SMALL)    # Top5→排序选1（基于同一Top80候选子集）
    if r80 is None or r5 is None:
        continue
    rows.append({"date": d, "ret_top80": r80, "ret_top5rank": r5, "diff": r5 - r80})

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
print("days:", len(df), "from", df["date"].iloc[0], "to", df["date"].iloc[-1])

# 1) 全样本对比
print("\n=== Overall (label returns) ===")
print("Top80 mean:", df["ret_top80"].mean())
print("Top5->rank mean:", df["ret_top5rank"].mean())
print("Diff (Top5->rank - Top80) mean:", df["diff"].mean())
print("P(diff>0):", (df["diff"] > 0).mean())

# 2) 分段稳定性（每SEG_LEN天）
segs = []
for i in range(0, len(df), SEG_LEN):
    seg = df.iloc[i:i+SEG_LEN]
    if len(seg) < SEG_LEN:
        continue
    segs.append({
        "start": seg["date"].iloc[0],
        "end": seg["date"].iloc[-1],
        "mean_top80": seg["ret_top80"].mean(),
        "mean_top5rank": seg["ret_top5rank"].mean(),
        "mean_diff": seg["diff"].mean(),
        "p_diff_pos": (seg["diff"] > 0).mean(),
    })

segdf = pd.DataFrame(segs)
print("\n=== Segment stability ({}-day blocks) ===".format(SEG_LEN))
print(segdf[["start","end","mean_top80","mean_top5rank","mean_diff","p_diff_pos"]].to_string(index=False))

# 3) Rolling曲线（帮助你肉眼看“某段时间好/坏”）
df["roll{}_diff".format(ROLL)] = df["diff"].rolling(ROLL).mean()
print("\nrolling diff quantiles (roll{}):".format(ROLL),
      np.percentile(df["roll{}_diff".format(ROLL)].dropna(), [5,10,50,90,95]))

# 4) Block bootstrap：Top5->rank 是否显著优于 Top80
diff = df["diff"].values
n = len(diff)

def block_bootstrap_mean(diff, iters=5000, block=5, seed=42):
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    num_blocks = int(np.ceil(n / block))
    for k in range(iters):
        idx = []
        for _ in range(num_blocks):
            start = rng.integers(0, n - block + 1)
            idx.extend(range(start, start + block))
        idx = idx[:n]
        means[k] = diff[idx].mean()
    return means

boot_means = block_bootstrap_mean(diff, BOOT_ITER, BLOCK)
print("\n=== Block bootstrap (mean diff) ===")
print("boot mean:", boot_means.mean())
print("boot p5/p50/p95:", np.percentile(boot_means, [5,50,95]))
print("Pr(mean_diff>0):", (boot_means > 0).mean())

# 5) “是否被右尾驱动”检查：截尾后再比较（winsorize）
def winsorize(x, p=0.05):
    lo, hi = np.quantile(x, [p, 1-p])
    return np.clip(x, lo, hi)

w = winsorize(diff, 0.05)
print("\n=== Tail sensitivity (winsorize 5%) ===")
print("mean_diff_raw:", diff.mean())
print("mean_diff_winsor:", w.mean())