import json
import numpy as np
import pandas as pd
from itertools import product

JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_015208.json"   # 改成你的路径


# =========================================================
# 基础函数
# =========================================================

def max_drawdown_from_returns(ret):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def annualized_sharpe(ret, periods_per_year=252):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(periods_per_year))


def summarize_arr(arr):
    x = pd.Series(arr).dropna().astype(float)
    if len(x) == 0:
        return None
    q10 = np.quantile(x, 0.10)
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "win_rate": float((x > 0).mean()),
        "big_loss_3pct_rate": float((x <= -0.03).mean()),
        "big_loss_5pct_rate": float((x <= -0.05).mean()),
        "big_loss_8pct_rate": float((x <= -0.08).mean()),
        "cvar10": float(x[x <= q10].mean()) if len(x) >= 10 else np.nan,
        "max_drawdown": max_drawdown_from_returns(x),
        "sharpe": annualized_sharpe(x),
        "p5": float(np.quantile(x, 0.05)),
        "p10": float(q10),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def print_summary(name, arr):
    s = summarize_arr(arr)
    print(f"\n=== {name} ===")
    if s is None:
        print("EMPTY")
        return None
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    return s


def score_strategy(arr):
    """
    综合评分：不是只看mean
    分数越高越好
    """
    s = summarize_arr(arr)
    if s is None:
        return -1e18

    score = (
        2.5 * s["mean"]
        + 0.3 * s["median"]
        + 0.15 * s["win_rate"]
        - 0.8 * s["big_loss_5pct_rate"]
        - 0.8 * s["big_loss_8pct_rate"]
        + 0.25 * s["cvar10"]
        + 0.30 * s["max_drawdown"]   # 注意 max_drawdown 为负，越接近0越好
    )
    return float(score)


# =========================================================
# 读取 JSON 并重建真实交易链路
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

rows = []

for date_str, rec in raw.items():
    cls_topn = rec.get("cls_topn", [])
    serial_top1 = rec.get("serial_top1", {})

    if len(cls_topn) < 5:
        continue

    df = pd.DataFrame(cls_topn).copy()
    for c in ["p_cls", "rank_score", "rank_pct_all", "real_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 严格按真实策略：分类Top5 + 排序Top1
    df_cls_top5 = df.sort_values("p_cls", ascending=False).head(5).reset_index(drop=True)
    df_rank_in_top5 = df_cls_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    chosen = df_rank_in_top5.iloc[0]

    row = {
        "date": date_str,
        "ret": float(chosen["real_return"]),
        "serial_rank_pct_all": pd.to_numeric(serial_top1.get("rank_pct_all", np.nan), errors="coerce"),
        "serial_pred": pd.to_numeric(serial_top1.get("predicted_return", np.nan), errors="coerce"),
        "serial_p_cls": pd.to_numeric(serial_top1.get("p_cls", np.nan), errors="coerce"),
    }
    row["conf_pred_x_cls"] = row["serial_pred"] * row["serial_p_cls"]
    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print("valid_days:", len(df_all))
base_summary = print_summary("baseline_ret", df_all["ret"])
base_score = score_strategy(df_all["ret"])
print(f"\nbaseline_score: {base_score:.6f}")


# =========================================================
# 1) 三档基线仓位（用于对照）
# =========================================================

def apply_three_level_position(df, feature, q30=None, q70=None, positions=(0.4, 0.7, 1.0)):
    sub = df.copy()
    if q30 is None:
        q30 = sub[feature].quantile(0.30)
    if q70 is None:
        q70 = sub[feature].quantile(0.70)

    lo, mid, hi = positions

    def pos_fn(x):
        if x >= q70:
            return hi
        elif x >= q30:
            return mid
        else:
            return lo

    sub["position"] = sub[feature].apply(pos_fn)
    sub["ret_pos"] = sub["ret"] * sub["position"]
    return sub, q30, q70


three_df, q30, q70 = apply_three_level_position(df_all, "serial_rank_pct_all")
three_summary = print_summary("three_level_rank_pct_ret", three_df["ret_pos"])
print(f"three_level_score: {score_strategy(three_df['ret_pos']):.6f}")
print(f"q30={q30:.6f}, q70={q70:.6f}")


# =========================================================
# 2) 反推“分段常数函数”
# =========================================================

# 用固定分桶，不用太细
bucket_edges = [0.0, 0.97, 0.985, 0.993, 0.997, 0.999, 1.000001]
bucket_labels = [f"B{i+1}" for i in range(len(bucket_edges)-1)]

df_seg = df_all.copy()
df_seg["bucket"] = pd.cut(df_seg["serial_rank_pct_all"], bins=bucket_edges, labels=bucket_labels, right=False)

print("\n=== bucket stats by serial_rank_pct_all ===")
bucket_stats = df_seg.groupby("bucket", observed=False)["ret"].agg(["count", "mean", "median", "std"])
bucket_stats["win_rate"] = df_seg.groupby("bucket", observed=False)["ret"].apply(lambda x: (x > 0).mean())
bucket_stats["loss5"] = df_seg.groupby("bucket", observed=False)["ret"].apply(lambda x: (x <= -0.05).mean())
bucket_stats["loss8"] = df_seg.groupby("bucket", observed=False)["ret"].apply(lambda x: (x <= -0.08).mean())
print(bucket_stats.to_string(float_format=lambda x: f"{x:.6f}"))

# 候选仓位：要单调不减
candidate_positions = [0.3, 0.4, 0.5, 0.65, 0.8, 1.0]

all_monotonic_maps = []
for vals in product(candidate_positions, repeat=len(bucket_labels)):
    if all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
        all_monotonic_maps.append(dict(zip(bucket_labels, vals)))

print(f"\nmonotonic_maps_count: {len(all_monotonic_maps)}")

best_map = None
best_score = -1e18
best_arr = None

for mp in all_monotonic_maps:
    tmp = df_seg.copy()
    tmp["position"] = tmp["bucket"].map(mp).astype(float)
    tmp["ret_pos"] = tmp["ret"] * tmp["position"]
    sc = score_strategy(tmp["ret_pos"])
    if sc > best_score:
        best_score = sc
        best_map = mp
        best_arr = tmp["ret_pos"].copy()

print("\n=== best_piecewise_map ===")
for k, v in best_map.items():
    print(k, "->", v)

piecewise_summary = print_summary("piecewise_rank_pct_ret", best_arr)
print(f"piecewise_score: {best_score:.6f}")


# =========================================================
# 3) 反推“连续 sigmoid 仓位函数”
# =========================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def apply_sigmoid_position(df, feature, floor_pos, cap_pos, center, slope):
    tmp = df.copy()
    x = tmp[feature].astype(float)
    z = slope * (x - center)
    tmp["position"] = floor_pos + (cap_pos - floor_pos) * sigmoid(z)
    tmp["ret_pos"] = tmp["ret"] * tmp["position"]
    return tmp

best_sigmoid_params = None
best_sigmoid_score = -1e18
best_sigmoid_arr = None

# 搜索范围不要太大，避免拟合过度
floor_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
cap_grid = [0.85, 0.90, 0.95, 1.00]
center_grid = [0.985, 0.990, 0.993, 0.995, 0.997, 0.998]
slope_grid = [40, 60, 80, 100, 140, 180]

for floor_pos in floor_grid:
    for cap_pos in cap_grid:
        if cap_pos <= floor_pos:
            continue
        for center in center_grid:
            for slope in slope_grid:
                tmp = apply_sigmoid_position(
                    df_all,
                    feature="serial_rank_pct_all",
                    floor_pos=floor_pos,
                    cap_pos=cap_pos,
                    center=center,
                    slope=slope,
                )
                sc = score_strategy(tmp["ret_pos"])
                if sc > best_sigmoid_score:
                    best_sigmoid_score = sc
                    best_sigmoid_params = {
                        "floor_pos": floor_pos,
                        "cap_pos": cap_pos,
                        "center": center,
                        "slope": slope,
                    }
                    best_sigmoid_arr = tmp["ret_pos"].copy()

print("\n=== best_sigmoid_params ===")
print(best_sigmoid_params)
sigmoid_summary = print_summary("sigmoid_rank_pct_ret", best_sigmoid_arr)
print(f"sigmoid_score: {best_sigmoid_score:.6f}")


# =========================================================
# 4) 输出比较表
# =========================================================

compare_rows = []

def add_row(name, arr):
    s = summarize_arr(arr)
    s["strategy"] = name
    s["score"] = score_strategy(arr)
    compare_rows.append(s)

add_row("baseline", df_all["ret"])
add_row("three_level", three_df["ret_pos"])
add_row("piecewise", best_arr)
add_row("sigmoid", best_sigmoid_arr)

cmp = pd.DataFrame(compare_rows)
cmp = cmp[
    [
        "strategy", "mean", "median", "win_rate",
        "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
        "cvar10", "max_drawdown", "sharpe", "score"
    ]
].sort_values("score", ascending=False)

print("\n" + "#" * 120)
print("FINAL COMPARISON")
print("#" * 120)
print(cmp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 5) 输出可直接抄到策略中的连续函数
# =========================================================

print("\n" + "#" * 120)
print("PRACTICAL DEPLOYMENT FORM")
print("#" * 120)

print("Piecewise suggestion:")
for b in bucket_labels:
    print(f"{b}: [{bucket_edges[bucket_labels.index(b)]:.6f}, {bucket_edges[bucket_labels.index(b)+1]:.6f}) -> {best_map[b]:.3f}")

print("\nSigmoid suggestion:")
print(
    f"position = {best_sigmoid_params['floor_pos']:.3f} + "
    f"({best_sigmoid_params['cap_pos']:.3f} - {best_sigmoid_params['floor_pos']:.3f}) * "
    f"sigmoid({best_sigmoid_params['slope']:.3f} * (rank_pct_all - {best_sigmoid_params['center']:.6f}))"
)