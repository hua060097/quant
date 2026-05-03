import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_015208.json"   # 改成你的json路径
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
    diffs = np.diff(means)
    pos_ratio = np.mean(diffs > 0)
    neg_ratio = np.mean(diffs < 0)
    if pos_ratio >= 0.75:
        return "LIKELY_POSITIVE_MONOTONIC"
    if neg_ratio >= 0.75:
        return "LIKELY_NEGATIVE_MONOTONIC"
    return "MIXED"


def simple_position_backtest(df, feature_col, higher_is_better=True):
    """
    单因子三档仓位:
    低 -> 0.4
    中 -> 0.7
    高 -> 1.0
    """
    sub = df[[feature_col, "ret"]].dropna().copy()
    if len(sub) < 30:
        return None

    q30 = sub[feature_col].quantile(0.30)
    q70 = sub[feature_col].quantile(0.70)

    def pos_fn(x):
        if higher_is_better:
            if x <= q30:
                return 0.4
            elif x <= q70:
                return 0.7
            else:
                return 1.0
        else:
            if x <= q30:
                return 1.0
            elif x <= q70:
                return 0.7
            else:
                return 0.4

    sub["pos"] = sub[feature_col].apply(pos_fn)
    sub["ret_pos"] = sub["ret"] * sub["pos"]

    return {
        "feature": feature_col,
        "higher_is_better": higher_is_better,
        "raw_mean": float(sub["ret"].mean()),
        "pos_mean": float(sub["ret_pos"].mean()),
        "raw_mdd": max_drawdown_from_returns(sub["ret"]),
        "pos_mdd": max_drawdown_from_returns(sub["ret_pos"]),
        "raw_loss5": float((sub["ret"] <= -0.05).mean()),
        "pos_loss5": float((sub["ret_pos"] <= -0.05).mean()),
        "raw_loss8": float((sub["ret"] <= -0.08).mean()),
        "pos_loss8": float((sub["ret_pos"] <= -0.08).mean()),
    }


def double_factor_position_backtest(df, f1, f2):
    """
    双因子仓位:
    - 两个都高: 1.0
    - 一个高一个中/两个中: 0.8
    - 一个低: 0.5
    - 两个都低: 0.3
    默认假设越大越好
    """
    sub = df[[f1, f2, "ret"]].dropna().copy()
    if len(sub) < 30:
        return None

    f1_q30, f1_q70 = sub[f1].quantile(0.30), sub[f1].quantile(0.70)
    f2_q30, f2_q70 = sub[f2].quantile(0.30), sub[f2].quantile(0.70)

    def level(x, q30, q70):
        if x <= q30:
            return 0
        elif x <= q70:
            return 1
        else:
            return 2

    sub["lv1"] = sub[f1].apply(lambda x: level(x, f1_q30, f1_q70))
    sub["lv2"] = sub[f2].apply(lambda x: level(x, f2_q30, f2_q70))
    sub["lv_sum"] = sub["lv1"] + sub["lv2"]

    def pos_fn(s):
        if s >= 4:
            return 1.0
        elif s >= 2:
            return 0.8
        elif s >= 1:
            return 0.5
        else:
            return 0.3

    sub["pos"] = sub["lv_sum"].apply(pos_fn)
    sub["ret_pos"] = sub["ret"] * sub["pos"]

    return {
        "feature": f"{f1} + {f2}",
        "raw_mean": float(sub["ret"].mean()),
        "pos_mean": float(sub["ret_pos"].mean()),
        "raw_mdd": max_drawdown_from_returns(sub["ret"]),
        "pos_mdd": max_drawdown_from_returns(sub["ret_pos"]),
        "raw_loss5": float((sub["ret"] <= -0.05).mean()),
        "pos_loss5": float((sub["ret_pos"] <= -0.05).mean()),
        "raw_loss8": float((sub["ret"] <= -0.08).mean()),
        "pos_loss8": float((sub["ret_pos"] <= -0.08).mean()),
    }


# =========================================================
# 读取 JSON
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for date_str, rec in data.items():
    serial_top1 = rec.get("serial_top1")
    cls_topn = rec.get("cls_topn")

    if not serial_top1 or not cls_topn or len(cls_topn) < 5:
        continue

    df = pd.DataFrame(cls_topn).copy()
    for c in ["p_cls", "rank_score", "rank_pct_all", "real_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 这里严格按你的真实策略：分类top5 + 排序选top1
    df_cls_top5 = df.sort_values("p_cls", ascending=False).head(5).reset_index(drop=True)
    df_rank_in_top5 = df_cls_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    chosen = df_rank_in_top5.iloc[0]

    row = {"date": date_str}

    # 真实收益（离线评估）
    row["ret"] = float(chosen["real_return"])

    # =====================================================
    # 1) 当天可见的主交易票字段
    # =====================================================
    row["serial_pred"] = pd.to_numeric(serial_top1.get("predicted_return", np.nan), errors="coerce")
    row["serial_score_top1_in_cls"] = pd.to_numeric(serial_top1.get("score_top1_in_cls", np.nan), errors="coerce")
    row["serial_score_top2_in_cls"] = pd.to_numeric(serial_top1.get("score_top2_in_cls", np.nan), errors="coerce")
    row["serial_score_margin_in_cls"] = pd.to_numeric(
        serial_top1.get("score_margin_top1_top2_in_cls", np.nan), errors="coerce"
    )
    row["serial_p_cls"] = pd.to_numeric(serial_top1.get("p_cls", np.nan), errors="coerce")
    row["serial_p_cls_pct"] = pd.to_numeric(serial_top1.get("p_cls_pct", np.nan), errors="coerce")
    row["serial_rank_pos_all"] = pd.to_numeric(serial_top1.get("rank_pos_all", np.nan), errors="coerce")
    row["serial_rank_pct_all"] = pd.to_numeric(serial_top1.get("rank_pct_all", np.nan), errors="coerce")

    # =====================================================
    # 2) top5 候选池结构（全都是真实可用）
    # =====================================================
    row["top5_cls_gap_1_2"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[1]["p_cls"])
    row["top5_cls_gap_1_5"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[4]["p_cls"])
    row["top5_cls_std"] = float(df_cls_top5["p_cls"].std(ddof=1))

    row["top5_rank_gap_1_2"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1]["rank_score"])
    row["top5_rank_gap_1_5"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[4]["rank_score"])
    row["top5_rank_gap_1_mean_rest"] = float(
        df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1:]["rank_score"].mean()
    )
    row["top5_rank_std"] = float(df_rank_in_top5["rank_score"].std(ddof=1))

    # 被选中股票在 top5 内的分类位置
    chosen_code = chosen["code"]
    df_cls_top5 = df_cls_top5.copy()
    df_cls_top5["cls_rank_in_top5"] = np.arange(1, len(df_cls_top5) + 1)
    match = df_cls_top5[df_cls_top5["code"] == chosen_code]
    row["chosen_cls_rank_in_top5"] = float(match.iloc[0]["cls_rank_in_top5"]) if len(match) else np.nan
    row["chosen_cls_rank_pct_in_top5"] = (
        1.0 - (row["chosen_cls_rank_in_top5"] - 1) / 4.0 if pd.notna(row["chosen_cls_rank_in_top5"]) else np.nan
    )

    # 被选中股票相对 top5 池子的 z-score
    top5_rank_mean = float(df_cls_top5["rank_score"].mean())
    top5_rank_std = float(df_cls_top5["rank_score"].std(ddof=1))
    top5_pcls_mean = float(df_cls_top5["p_cls"].mean())
    top5_pcls_std = float(df_cls_top5["p_cls"].std(ddof=1))

    row["chosen_rank_z_in_top5"] = (
        (float(chosen["rank_score"]) - top5_rank_mean) / top5_rank_std
        if top5_rank_std > 1e-12 else np.nan
    )
    row["chosen_pcls_z_in_top5"] = (
        (float(chosen["p_cls"]) - top5_pcls_mean) / top5_pcls_std
        if top5_pcls_std > 1e-12 else np.nan
    )

    # =====================================================
    # 3) 组合信号
    # =====================================================
    row["conf_pred_x_cls"] = row["serial_pred"] * row["serial_p_cls"]
    row["conf_rankgap_x_rankpct"] = row["serial_score_margin_in_cls"] * row["serial_rank_pct_all"]
    row["conf_rankz_x_pclsz"] = (
        row["chosen_rank_z_in_top5"] * row["chosen_pcls_z_in_top5"]
        if pd.notna(row["chosen_rank_z_in_top5"]) and pd.notna(row["chosen_pcls_z_in_top5"])
        else np.nan
    )

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print("valid_days:", len(df_all))
print_summary("ret (Top5 + Rank1 real return)", df_all["ret"])


# =========================================================
# 因子诊断
# =========================================================

feature_signs = {
    "serial_pred": True,
    "serial_score_margin_in_cls": True,
    "serial_p_cls": True,
    "serial_p_cls_pct": True,
    "serial_rank_pct_all": True,

    "top5_cls_gap_1_2": True,
    "top5_cls_gap_1_5": True,
    "top5_cls_std": True,

    "top5_rank_gap_1_2": True,
    "top5_rank_gap_1_5": True,
    "top5_rank_gap_1_mean_rest": True,
    "top5_rank_std": True,

    "chosen_cls_rank_in_top5": False,
    "chosen_cls_rank_pct_in_top5": True,
    "chosen_rank_z_in_top5": True,
    "chosen_pcls_z_in_top5": True,

    "conf_pred_x_cls": True,
    "conf_rankgap_x_rankpct": True,
    "conf_rankz_x_pclsz": True,
}

results = []

for feat, higher_is_better in feature_signs.items():
    if feat not in df_all.columns:
        continue

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
            "higher_is_better": higher_is_better,
            "monotonic_hint": hint,
            "top_bucket_mean": float(bucket_means[-1]),
            "bottom_bucket_mean": float(bucket_means[0]),
            "spread_mean": float(bucket_means[-1] - bucket_means[0]),
            "top_bucket_loss5": float(bucket_loss5[-1]),
            "bottom_bucket_loss5": float(bucket_loss5[0]),
            "top_bucket_loss8": float(bucket_loss8[-1]),
            "bottom_bucket_loss8": float(bucket_loss8[0]),
        })

if results:
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["spread_mean", "feature"], ascending=[False, True]).reset_index(drop=True)

    print("\n" + "#" * 100)
    print("SUMMARY RANKING BY spread_mean (top bucket mean - bottom bucket mean)")
    print("#" * 100)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 单因子动态仓位回测
# =========================================================

bt_results = []
for feat, higher_is_better in feature_signs.items():
    if feat not in df_all.columns:
        continue
    r = simple_position_backtest(df_all, feat, higher_is_better=higher_is_better)
    if r is not None:
        bt_results.append(r)

if bt_results:
    bt_df = pd.DataFrame(bt_results)
    bt_df["mean_diff"] = bt_df["pos_mean"] - bt_df["raw_mean"]
    bt_df["mdd_improve"] = bt_df["pos_mdd"] - bt_df["raw_mdd"]
    bt_df["loss5_improve"] = bt_df["pos_loss5"] - bt_df["raw_loss5"]
    bt_df["loss8_improve"] = bt_df["pos_loss8"] - bt_df["raw_loss8"]

    print("\n" + "#" * 100)
    print("SINGLE-FACTOR POSITION BACKTEST (0.4 / 0.7 / 1.0 by 30%-70% quantiles)")
    print("#" * 100)
    print(
        bt_df.sort_values(["mean_diff", "mdd_improve"], ascending=[False, False])
             .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )


# =========================================================
# 双因子动态仓位回测
# =========================================================

double_candidates = [
    ("serial_rank_pct_all", "conf_pred_x_cls"),
    ("serial_rank_pct_all", "top5_cls_std"),
    ("conf_pred_x_cls", "top5_cls_std"),
    ("serial_rank_pct_all", "chosen_rank_z_in_top5"),
]

double_results = []
for f1, f2 in double_candidates:
    if f1 in df_all.columns and f2 in df_all.columns:
        r = double_factor_position_backtest(df_all, f1, f2)
        if r is not None:
            double_results.append(r)

if double_results:
    ddf = pd.DataFrame(double_results)
    ddf["mean_diff"] = ddf["pos_mean"] - ddf["raw_mean"]
    ddf["mdd_improve"] = ddf["pos_mdd"] - ddf["raw_mdd"]
    ddf["loss5_improve"] = ddf["pos_loss5"] - ddf["raw_loss5"]
    ddf["loss8_improve"] = ddf["pos_loss8"] - ddf["raw_loss8"]

    print("\n" + "#" * 100)
    print("DOUBLE-FACTOR POSITION BACKTEST")
    print("#" * 100)
    print(
        ddf.sort_values(["mean_diff", "mdd_improve"], ascending=[False, False])
           .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )