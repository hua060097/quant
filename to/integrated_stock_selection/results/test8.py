import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

SEG_LEN = 20
BOOT_ITER = 5000
BLOCK = 5
RNG_SEED = 42

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

# -----------------------------
# 构造每日策略收益
# -----------------------------
rows = []

for d in dates:
    rec = data[d]

    cls = rec.get("cls_topn") or []
    rank_all = rec.get("rank_all_top1") or {}

    if len(cls) == 0:
        continue

    # 按分类排序
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    # 按排序排序
    cls_by_rank = sorted(cls, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)

    # A: 分类Top1
    A = cls_by_p[0].get("real_return")

    # B1/B2: 分类Top3/Top5 随机选1（这里只先保存候选池，后面统一模拟）
    top3_rets = [x.get("real_return") for x in cls_by_p[:3] if x.get("real_return") is not None]
    top5_rets = [x.get("real_return") for x in cls_by_p[:5] if x.get("real_return") is not None]

    # C: 分类Top5 -> 排序选1
    top5_by_p = cls_by_p[:5]
    top5_by_rank = sorted(top5_by_p, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
    C = top5_by_rank[0].get("real_return")

    # D: 全市场排序Top1
    D = rank_all.get("real_return")

    # E: 分类Top5等权平均
    E = float(np.mean(top5_rets)) if len(top5_rets) > 0 else None

    rows.append({
        "date": d,
        "A_cls_top1": A,
        "C_top5_rank1": C,
        "D_rankall_top1": D,
        "E_cls_top5_mean": E,
        "top3_rets": top3_rets,
        "top5_rets": top5_rets,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# -----------------------------
# 随机策略模拟
# -----------------------------
rng = np.random.default_rng(RNG_SEED)

def simulate_random_mean(ret_lists, n_iter=5000):
    means = []
    paths = []
    for _ in range(n_iter):
        path = []
        for lst in ret_lists:
            if len(lst) == 0:
                continue
            path.append(rng.choice(lst))
        paths.append(path)
        means.append(np.mean(path))
    return np.array(means), paths

B1_means, B1_paths = simulate_random_mean(df["top3_rets"].tolist(), n_iter=BOOT_ITER)
B2_means, B2_paths = simulate_random_mean(df["top5_rets"].tolist(), n_iter=BOOT_ITER)

# 取随机策略的“中位路径均值”近似作为代表值（仅展示）
B1_repr_mean = np.median(B1_means)
B2_repr_mean = np.median(B2_means)

# -----------------------------
# 工具函数
# -----------------------------
def describe_arr(name, arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    print(f"\n{name}: count={len(arr)} mean={arr.mean():.6f} std={arr.std(ddof=1):.6f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p}: {np.percentile(arr, p):.6f}")

def block_bootstrap_mean_diff(x, y, iters=5000, block=5, seed=123):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    diff = x - y
    n = len(diff)
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    num_blocks = int(np.ceil(n / block))
    for k in range(iters):
        idx = []
        for _ in range(num_blocks):
            s = rng.integers(0, n - block + 1)
            idx.extend(range(s, s + block))
        idx = idx[:n]
        means[k] = diff[idx].mean()
    return means

def segment_compare(name_x, x, name_y, y, dates, seg_len=20):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    dates = np.array(dates)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    dates = dates[mask]

    segs = []
    for i in range(0, len(x), seg_len):
        xs = x[i:i+seg_len]
        ys = y[i:i+seg_len]
        ds = dates[i:i+seg_len]
        if len(xs) < seg_len:
            continue
        segs.append({
            "start": ds[0],
            "end": ds[-1],
            f"mean_{name_x}": xs.mean(),
            f"mean_{name_y}": ys.mean(),
            "mean_diff": (xs - ys).mean(),
            "p_diff_pos": ((xs - ys) > 0).mean(),
        })
    return pd.DataFrame(segs)

# -----------------------------
# 基础统计
# -----------------------------
print("=== Strategy summary (signal-layer / label returns) ===")
describe_arr("A_cls_top1", df["A_cls_top1"].values)
describe_arr("C_top5_rank1", df["C_top5_rank1"].values)
describe_arr("D_rankall_top1", df["D_rankall_top1"].values)
describe_arr("E_cls_top5_mean", df["E_cls_top5_mean"].values)
describe_arr("B1_top3_random_mean_distribution", B1_means)
describe_arr("B2_top5_random_mean_distribution", B2_means)

# -----------------------------
# 关键均值对比
# -----------------------------
A = df["A_cls_top1"].values
C = df["C_top5_rank1"].values
D = df["D_rankall_top1"].values
E = df["E_cls_top5_mean"].values

print("\n=== Key mean comparisons ===")
print("C - A mean =", np.nanmean(C - A))
print("A - D mean =", np.nanmean(A - D))
print("C - D mean =", np.nanmean(C - D))
print("A - E mean =", np.nanmean(A - E))
print("C - E mean =", np.nanmean(C - E))

# B2 用均值分布来比较
print("\nB2(top5 random) mean distribution:")
print("  p5/p50/p95 =", np.percentile(B2_means, [5, 50, 95]))
print("  median mean =", np.median(B2_means))

# -----------------------------
# Bootstrap 显著性
# -----------------------------
boot_CA = block_bootstrap_mean_diff(C, A, iters=BOOT_ITER, block=BLOCK, seed=1)
boot_AD = block_bootstrap_mean_diff(A, D, iters=BOOT_ITER, block=BLOCK, seed=2)
boot_CD = block_bootstrap_mean_diff(C, D, iters=BOOT_ITER, block=BLOCK, seed=3)

print("\n=== Bootstrap significance ===")
print("Pr(mean(C-A) > 0) =", (boot_CA > 0).mean(), "  p5/p50/p95 =", np.percentile(boot_CA, [5,50,95]))
print("Pr(mean(A-D) > 0) =", (boot_AD > 0).mean(), "  p5/p50/p95 =", np.percentile(boot_AD, [5,50,95]))
print("Pr(mean(C-D) > 0) =", (boot_CD > 0).mean(), "  p5/p50/p95 =", np.percentile(boot_CD, [5,50,95]))

# -----------------------------
# 分段稳定性
# -----------------------------
print("\n=== Segment stability: C vs A ===")
seg_CA = segment_compare("C", C, "A", A, df["date"].values, seg_len=SEG_LEN)
print(seg_CA.to_string(index=False))

print("\n=== Segment stability: A vs D ===")
seg_AD = segment_compare("A", A, "D", D, df["date"].values, seg_len=SEG_LEN)
print(seg_AD.to_string(index=False))

print("\n=== Done ===")