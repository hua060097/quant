import json
import numpy as np
import pandas as pd

path = "daily_gate_stats_Window_3_gateN80_20260308_234514.json"

SEG_LEN = 20
BOOT_ITER = 5000
BLOCK = 5

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

dates = sorted(data.keys())

rows = []
for d in dates:
    rec = data[d]
    cls = rec.get("cls_topn") or []
    if len(cls) == 0:
        continue

    # 按分类排序
    cls_by_p = sorted(cls, key=lambda x: x.get("p_cls", float("-inf")), reverse=True)

    # Top5 by p_cls
    top5 = cls_by_p[:5]
    if len(top5) == 0:
        continue

    # 在 Top5 中按 rank_score 排序
    top5_by_rank = sorted(top5, key=lambda x: x.get("rank_score", float("-inf")), reverse=True)

    # 各种策略
    A_cls_top1 = cls_by_p[0].get("real_return")

    R1 = top5_by_rank[0].get("real_return") if len(top5_by_rank) >= 1 else None

    R2 = None
    if len(top5_by_rank) >= 2:
        vals = [top5_by_rank[0].get("real_return"), top5_by_rank[1].get("real_return")]
        vals = [v for v in vals if v is not None]
        if len(vals) > 0:
            R2 = float(np.mean(vals))

    R3 = None
    if len(top5_by_rank) >= 3:
        vals = [top5_by_rank[0].get("real_return"), top5_by_rank[1].get("real_return"), top5_by_rank[2].get("real_return")]
        vals = [v for v in vals if v is not None]
        if len(vals) > 0:
            R3 = float(np.mean(vals))

    P5_mean = None
    vals = [x.get("real_return") for x in top5 if x.get("real_return") is not None]
    if len(vals) > 0:
        P5_mean = float(np.mean(vals))

    rows.append({
        "date": d,
        "A_cls_top1": A_cls_top1,
        "R1_top5_rank_top1": R1,
        "R2_top5_rank_top2eq": R2,
        "R3_top5_rank_top3eq": R3,
        "P5_cls_top5eq": P5_mean,
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

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

print("=== Strategy summary ===")
for col in ["A_cls_top1", "R1_top5_rank_top1", "R2_top5_rank_top2eq", "R3_top5_rank_top3eq", "P5_cls_top5eq"]:
    describe_arr(col, df[col].values)

print("\n=== Mean comparisons ===")
pairs = [
    ("R2_top5_rank_top2eq", "R1_top5_rank_top1", "R2-R1"),
    ("R3_top5_rank_top3eq", "R2_top5_rank_top2eq", "R3-R2"),
    ("R3_top5_rank_top3eq", "R1_top5_rank_top1", "R3-R1"),
    ("R1_top5_rank_top1", "A_cls_top1", "R1-A"),
    ("R2_top5_rank_top2eq", "A_cls_top1", "R2-A"),
    ("R3_top5_rank_top3eq", "A_cls_top1", "R3-A"),
]
for x, y, name in pairs:
    diff = df[x].values - df[y].values
    print(name, "mean =", np.nanmean(diff))

print("\n=== Bootstrap significance ===")
boot_pairs = [
    ("R2_top5_rank_top2eq", "R1_top5_rank_top1", "R2>R1", 1),
    ("R3_top5_rank_top3eq", "R2_top5_rank_top2eq", "R3>R2", 2),
    ("R3_top5_rank_top3eq", "R1_top5_rank_top1", "R3>R1", 3),
    ("R1_top5_rank_top1", "A_cls_top1", "R1>A", 4),
    ("R2_top5_rank_top2eq", "A_cls_top1", "R2>A", 5),
    ("R3_top5_rank_top3eq", "A_cls_top1", "R3>A", 6),
]
for x, y, name, seed in boot_pairs:
    boot = block_bootstrap_mean_diff(df[x].values, df[y].values, iters=BOOT_ITER, block=BLOCK, seed=seed)
    print(name, "Pr(mean_diff>0) =", (boot > 0).mean(), " p5/p50/p95 =", np.percentile(boot, [5,50,95]))

print("\n=== Segment stability: R2 vs R1 ===")
print(segment_compare("R2", df["R2_top5_rank_top2eq"].values, "R1", df["R1_top5_rank_top1"].values, df["date"].values, SEG_LEN).to_string(index=False))

print("\n=== Segment stability: R3 vs R1 ===")
print(segment_compare("R3", df["R3_top5_rank_top3eq"].values, "R1", df["R1_top5_rank_top1"].values, df["date"].values, SEG_LEN).to_string(index=False))

print("\n=== Done ===")