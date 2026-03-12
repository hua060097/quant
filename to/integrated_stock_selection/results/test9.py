import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

BOOT_ITER = 5000
BLOCK = 5
SEG_LEN = 20
RNG_SEED = 42

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        continue

    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    def pick_rank_from_topk(k):
        topk = cls_by_p[:k]
        if len(topk) == 0:
            return None
        topk_by_rank = sorted(topk, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)
        return topk_by_rank[0].get("real_return")

    def get_topk_returns(k):
        return [x.get("real_return") for x in cls_by_p[:k] if x.get("real_return") is not None]

    A = cls_by_p[0].get("real_return")  # 分类Top1
    R3 = pick_rank_from_topk(3)
    R5 = pick_rank_from_topk(5)
    R7 = pick_rank_from_topk(7)

    rows.append({
        "date": d,
        "A_clsTop1": A,
        "R3": R3,
        "R5": R5,
        "R7": R7,
        "top3_rets": get_topk_returns(3),
        "top5_rets": get_topk_returns(5),
        "top7_rets": get_topk_returns(7),
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

rng = np.random.default_rng(RNG_SEED)

def simulate_random_mean(ret_lists, n_iter=5000):
    means = []
    for _ in range(n_iter):
        path = []
        for lst in ret_lists:
            if len(lst) == 0:
                continue
            path.append(rng.choice(lst))
        means.append(np.mean(path))
    return np.array(means)

Rand3 = simulate_random_mean(df["top3_rets"].tolist(), BOOT_ITER)
Rand5 = simulate_random_mean(df["top5_rets"].tolist(), BOOT_ITER)
Rand7 = simulate_random_mean(df["top7_rets"].tolist(), BOOT_ITER)

def describe_arr(name, arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    print(f"\n{name}: count={len(arr)} mean={arr.mean():.6f} std={arr.std(ddof=1):.6f}")
    for p in [5,10,25,50,75,90,95]:
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

print("=== Strategy summary ===")
describe_arr("A_clsTop1", df["A_clsTop1"].values)
describe_arr("R3_top3_rank1", df["R3"].values)
describe_arr("R5_top5_rank1", df["R5"].values)
describe_arr("R7_top7_rank1", df["R7"].values)

print("\n=== Random baselines (mean distribution) ===")
describe_arr("Rand3", Rand3)
describe_arr("Rand5", Rand5)
describe_arr("Rand7", Rand7)

print("\n=== Mean comparisons ===")
for x, y, name in [
    ("R3", "A_clsTop1", "R3-A"),
    ("R5", "A_clsTop1", "R5-A"),
    ("R7", "A_clsTop1", "R7-A"),
    ("R5", "R3", "R5-R3"),
    ("R7", "R5", "R7-R5"),
]:
    diff = df[x].values - df[y].values
    print(name, "mean =", np.nanmean(diff))

print("\n=== Bootstrap significance ===")
for x, y, name, seed in [
    ("R3", "A_clsTop1", "R3>A", 1),
    ("R5", "A_clsTop1", "R5>A", 2),
    ("R7", "A_clsTop1", "R7>A", 3),
    ("R5", "R3", "R5>R3", 4),
    ("R7", "R5", "R7>R5", 5),
]:
    boot = block_bootstrap_mean_diff(df[x].values, df[y].values, iters=BOOT_ITER, block=BLOCK, seed=seed)
    print(name, "Pr(mean_diff>0) =", (boot > 0).mean(), " p5/p50/p95 =", np.percentile(boot, [5,50,95]))

print("\n=== Segment stability: R5 vs R3 ===")
print(segment_compare("R5", df["R5"].values, "R3", df["R3"].values, df["date"].values, SEG_LEN).to_string(index=False))

print("\n=== Segment stability: R5 vs A ===")
print(segment_compare("R5", df["R5"].values, "A", df["A_clsTop1"].values, df["date"].values, SEG_LEN).to_string(index=False))