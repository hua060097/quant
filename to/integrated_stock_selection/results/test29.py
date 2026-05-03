import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260314_043520.json"   # 改成你的json路径
N_BUCKETS = 5


# =========================================================
# 工具函数
# =========================================================

def max_drawdown_from_returns(ret):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def summarize_arr(arr):
    x = pd.Series(arr).dropna().astype(float)
    if len(x) == 0:
        return None
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "win_rate": float((x > 0).mean()),
        "big_loss_3pct_rate": float((x <= -0.03).mean()),
        "big_loss_5pct_rate": float((x <= -0.05).mean()),
        "big_loss_8pct_rate": float((x <= -0.08).mean()),
        "max_drawdown": max_drawdown_from_returns(x),
        "p5": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
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


def bucket_table(df, feature_col, ret_col="ret", n_buckets=5):
    sub = df[[feature_col, ret_col]].dropna().copy()
    if len(sub) < n_buckets * 5:
        print(f"\n[SKIP] {feature_col}: not enough samples")
        return None

    try:
        sub["bucket"] = pd.qcut(sub[feature_col], q=n_buckets, duplicates="drop")
    except Exception as e:
        print(f"\n[SKIP] {feature_col}: qcut failed -> {e}")
        return None

    g = sub.groupby("bucket", observed=False)[ret_col]

    out = g.agg(["count", "mean", "median", "std"]).copy()
    out["win_rate"] = g.apply(lambda x: (x > 0).mean())
    out["loss3_rate"] = g.apply(lambda x: (x <= -0.03).mean())
    out["loss5_rate"] = g.apply(lambda x: (x <= -0.05).mean())
    out["loss8_rate"] = g.apply(lambda x: (x <= -0.08).mean())
    out["max_drawdown"] = g.apply(max_drawdown_from_returns)

    return out


def monotonic_hint(tbl):
    if tbl is None or len(tbl) < 3:
        return "NA"
    means = tbl["mean"].values
    # 粗略判断是否整体上升
    diffs = np.diff(means)
    pos_ratio = np.mean(diffs > 0)
    neg_ratio = np.mean(diffs < 0)
    if pos_ratio >= 0.75:
        return "LIKELY_POSITIVE_MONOTONIC"
    if neg_ratio >= 0.75:
        return "LIKELY_NEGATIVE_MONOTONIC"
    return "MIXED"


# =========================================================
# 读 JSON
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for date_str, rec in data.items():

    serial_top1 = rec.get("serial_top1", None)
    cls_topn = rec.get("cls_topn", None)

    if not serial_top1 or not cls_topn or len(cls_topn) < 5:
        continue

    row = {"date": date_str}

    # -------------------------
    # 主交易结果（真实 Top5 -> Rank1）
    # -------------------------
    row["ret"] = pd.to_numeric(serial_top1.get("real_return", np.nan), errors="coerce")
    row["serial_pred"] = pd.to_numeric(serial_top1.get("predicted_return", np.nan), errors="coerce")
    row["serial_p_cls"] = pd.to_numeric(serial_top1.get("p_cls", np.nan), errors="coerce")
    row["serial_p_risk"] = pd.to_numeric(serial_top1.get("p_risk", np.nan), errors="coerce")

    row["conditional_score_margin_top1_top2"] = pd.to_numeric(
        rec.get("conditional_score_margin_top1_top2", np.nan), errors="coerce"
    )
    row["conditional_gap"] = pd.to_numeric(rec.get("conditional_gap", np.nan), errors="coerce")
    row["unconditional_gap"] = pd.to_numeric(rec.get("unconditional_gap", np.nan), errors="coerce")

    # -------------------------
    # 从 cls_topn 中重建分类Top5
    # -------------------------
    df = pd.DataFrame(cls_topn).copy()
    for c in ["p_cls", "rank_score", "real_return", "p_risk", "rank_pos_all", "rank_pct_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 按 p_cls 排序的 Top5
    df_cls_top5 = df.sort_values("p_cls", ascending=False).head(5).reset_index(drop=True)

    # 分类 gap
    row["top5_cls_gap_1_2"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[1]["p_cls"])
    row["top5_cls_gap_1_5"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[4]["p_cls"])
    row["top5_cls_std"] = float(df_cls_top5["p_cls"].std(ddof=1))

    # 在分类Top5内部，按 rank_score 排序
    df_rank_in_cls = df_cls_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)

    row["top5_rank_gap_1_2"] = float(df_rank_in_cls.iloc[0]["rank_score"] - df_rank_in_cls.iloc[1]["rank_score"])
    row["top5_rank_gap_1_5"] = float(df_rank_in_cls.iloc[0]["rank_score"] - df_rank_in_cls.iloc[4]["rank_score"])
    row["top5_rank_gap_1_mean_rest"] = float(df_rank_in_cls.iloc[0]["rank_score"] - df_rank_in_cls.iloc[1:]["rank_score"].mean())
    row["top5_rank_std"] = float(df_rank_in_cls["rank_score"].std(ddof=1))

    # top5 候选池本身的整体质量（不用于删票，只分析）
    row["top5_mean_ret"] = float(df_cls_top5["real_return"].mean())
    row["top5_median_ret"] = float(df_cls_top5["real_return"].median())
    row["top5_best_ret"] = float(df_cls_top5["real_return"].max())
    row["top5_worst_ret"] = float(df_cls_top5["real_return"].min())

    # 组合置信度
    row["conf_mul_1"] = row["top5_rank_gap_1_2"] * row["serial_p_cls"]
    row["conf_mul_2"] = row["conditional_score_margin_top1_top2"] * row["serial_pred"]
    row["conf_mul_3"] = row["top5_rank_gap_1_mean_rest"] * row["serial_p_cls"]
    row["conf_mul_4"] = row["top5_cls_gap_1_5"] * row["top5_rank_gap_1_2"]

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print("valid_days:", len(df_all))

print_summary("ret (Top5->Rank1 real return)", df_all["ret"])
print_summary("top5_mean_ret", df_all["top5_mean_ret"])
print_summary("top5_median_ret", df_all["top5_median_ret"])


# =========================================================
# 分析候选仓位信号
# =========================================================

features = [
    "conditional_score_margin_top1_top2",
    "conditional_gap",
    "unconditional_gap",
    "serial_pred",
    "serial_p_cls",
    "serial_p_risk",

    "top5_cls_gap_1_2",
    "top5_cls_gap_1_5",
    "top5_cls_std",

    "top5_rank_gap_1_2",
    "top5_rank_gap_1_5",
    "top5_rank_gap_1_mean_rest",
    "top5_rank_std",

    "conf_mul_1",
    "conf_mul_2",
    "conf_mul_3",
    "conf_mul_4",
]

results = []

for feat in features:
    print("\n" + "=" * 80)
    print(f"FEATURE: {feat}")
    print("=" * 80)

    print_summary(feat, df_all[feat])

    tbl = bucket_table(df_all, feat, ret_col="ret", n_buckets=N_BUCKETS)
    if tbl is not None:
        print("\n[Bucket table: ret by feature quantile]")
        print(tbl.to_string(float_format=lambda x: f"{x:.6f}"))

        hint = monotonic_hint(tbl)
        print("\nmonotonic_hint:", hint)

        bucket_means = tbl["mean"].values
        bucket_loss5 = tbl["loss5_rate"].values
        bucket_loss8 = tbl["loss8_rate"].values

        results.append({
            "feature": feat,
            "monotonic_hint": hint,
            "top_bucket_mean": float(bucket_means[-1]),
            "bottom_bucket_mean": float(bucket_means[0]),
            "spread_mean": float(bucket_means[-1] - bucket_means[0]),
            "top_bucket_loss5": float(bucket_loss5[-1]),
            "bottom_bucket_loss5": float(bucket_loss5[0]),
            "top_bucket_loss8": float(bucket_loss8[-1]),
            "bottom_bucket_loss8": float(bucket_loss8[0]),
        })

# 汇总排序
if results:
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["spread_mean", "feature"], ascending=[False, True]).reset_index(drop=True)

    print("\n" + "#" * 100)
    print("SUMMARY RANKING BY spread_mean (top bucket mean - bottom bucket mean)")
    print("#" * 100)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 可选：简单仓位回测（只做粗分析）
# =========================================================

def simple_position_backtest(df, feature_col):
    sub = df[[feature_col, "ret"]].dropna().copy()
    if len(sub) < 30:
        return None

    q30 = sub[feature_col].quantile(0.30)
    q70 = sub[feature_col].quantile(0.70)

    def pos_fn(x):
        if x <= q30:
            return 0.4
        elif x <= q70:
            return 0.7
        else:
            return 1.0

    sub["pos"] = sub[feature_col].apply(pos_fn)
    sub["ret_pos"] = sub["ret"] * sub["pos"]

    out = {
        "feature": feature_col,
        "raw_mean": float(sub["ret"].mean()),
        "pos_mean": float(sub["ret_pos"].mean()),
        "raw_mdd": max_drawdown_from_returns(sub["ret"]),
        "pos_mdd": max_drawdown_from_returns(sub["ret_pos"]),
        "raw_loss5": float((sub["ret"] <= -0.05).mean()),
        "pos_loss5": float((sub["ret_pos"] <= -0.05).mean()),
        "raw_loss8": float((sub["ret"] <= -0.08).mean()),
        "pos_loss8": float((sub["ret_pos"] <= -0.08).mean()),
    }
    return out


bt_results = []
for feat in features:
    r = simple_position_backtest(df_all, feat)
    if r is not None:
        bt_results.append(r)

if bt_results:
    bt_df = pd.DataFrame(bt_results)
    bt_df["mean_diff"] = bt_df["pos_mean"] - bt_df["raw_mean"]
    bt_df["mdd_improve"] = bt_df["pos_mdd"] - bt_df["raw_mdd"]   # 越大越好（更接近0）
    bt_df["loss5_improve"] = bt_df["pos_loss5"] - bt_df["raw_loss5"]  # 越小越好
    bt_df["loss8_improve"] = bt_df["pos_loss8"] - bt_df["raw_loss8"]

    print("\n" + "#" * 100)
    print("SIMPLE POSITION BACKTEST (0.4 / 0.7 / 1.0 by 30%-70% quantiles)")
    print("#" * 100)
    print(
        bt_df.sort_values(["mean_diff", "mdd_improve"], ascending=[False, False])
             .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )