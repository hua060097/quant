import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_011627.json"   # 改成你的json路径
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
    用 30%-70% 分位做一个非常粗的仓位映射：
    低信号 -> 0.4
    中信号 -> 0.7
    高信号 -> 1.0
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

    row = {"date": date_str}

    # 真实收益（只用于离线评估）
    row["ret"] = pd.to_numeric(serial_top1.get("real_return", np.nan), errors="coerce")

    # =========================
    # 1) serial_top1 当日可见字段
    # =========================
    row["serial_pred"] = pd.to_numeric(serial_top1.get("predicted_return", np.nan), errors="coerce")
    row["serial_score_top1_in_cls"] = pd.to_numeric(serial_top1.get("score_top1_in_cls", np.nan), errors="coerce")
    row["serial_score_top2_in_cls"] = pd.to_numeric(serial_top1.get("score_top2_in_cls", np.nan), errors="coerce")
    row["serial_score_margin_in_cls"] = pd.to_numeric(serial_top1.get("score_margin_top1_top2_in_cls", np.nan), errors="coerce")
    row["serial_p_cls"] = pd.to_numeric(serial_top1.get("p_cls", np.nan), errors="coerce")
    row["serial_p_cls_pct"] = pd.to_numeric(serial_top1.get("p_cls_pct", np.nan), errors="coerce")
    row["serial_rank_pos_all"] = pd.to_numeric(serial_top1.get("rank_pos_all", np.nan), errors="coerce")
    row["serial_rank_pct_all"] = pd.to_numeric(serial_top1.get("rank_pct_all", np.nan), errors="coerce")
    row["serial_p_risk"] = pd.to_numeric(serial_top1.get("p_risk", np.nan), errors="coerce")

    # =========================
    # 2) cls_topn 候选池当日可见结构
    # =========================
    df = pd.DataFrame(cls_topn).copy()
    for c in ["p_cls", "rank_score", "rank_pct_all", "p_risk"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # gate_n 候选池整体统计
    row["pool_size"] = len(df)
    row["pool_pcls_mean"] = float(df["p_cls"].mean())
    row["pool_pcls_std"] = float(df["p_cls"].std(ddof=1))
    row["pool_rank_mean"] = float(df["rank_score"].mean())
    row["pool_rank_std"] = float(df["rank_score"].std(ddof=1))
    row["pool_rank_pct_mean"] = float(df["rank_pct_all"].mean()) if "rank_pct_all" in df.columns else np.nan
    row["pool_rank_pct_std"] = float(df["rank_pct_all"].std(ddof=1)) if "rank_pct_all" in df.columns else np.nan
    row["pool_risk_mean"] = float(df["p_risk"].mean()) if "p_risk" in df.columns else np.nan
    row["pool_risk_std"] = float(df["p_risk"].std(ddof=1)) if "p_risk" in df.columns else np.nan

    # 先按 p_cls 排序，看分类头部结构
    df_by_cls = df.sort_values("p_cls", ascending=False).reset_index(drop=True)
    top5_cls = df_by_cls.head(5).copy()

    row["top5_cls_gap_1_2"] = float(top5_cls.iloc[0]["p_cls"] - top5_cls.iloc[1]["p_cls"])
    row["top5_cls_gap_1_5"] = float(top5_cls.iloc[0]["p_cls"] - top5_cls.iloc[4]["p_cls"])
    row["top5_cls_std"] = float(top5_cls["p_cls"].std(ddof=1))

    # 再在 top5_cls 内看排序结构
    top5_cls_by_rank = top5_cls.sort_values("rank_score", ascending=False).reset_index(drop=True)
    row["top5_rank_gap_1_2"] = float(top5_cls_by_rank.iloc[0]["rank_score"] - top5_cls_by_rank.iloc[1]["rank_score"])
    row["top5_rank_gap_1_5"] = float(top5_cls_by_rank.iloc[0]["rank_score"] - top5_cls_by_rank.iloc[4]["rank_score"])
    row["top5_rank_gap_1_mean_rest"] = float(
        top5_cls_by_rank.iloc[0]["rank_score"] - top5_cls_by_rank.iloc[1:]["rank_score"].mean()
    )
    row["top5_rank_std"] = float(top5_cls_by_rank["rank_score"].std(ddof=1))

    # =========================
    # 3) 分类与排序的一致性 / 冲突度（重点）
    # =========================
    serial_code = serial_top1.get("code", None)

    # serial_top1 在 cls_topn 中按 p_cls 的名次
    df_by_cls["cls_rank_in_pool"] = np.arange(1, len(df_by_cls) + 1)
    match_cls = df_by_cls[df_by_cls["code"] == serial_code]
    row["serial_cls_rank_in_pool"] = float(match_cls.iloc[0]["cls_rank_in_pool"]) if len(match_cls) else np.nan

    # serial_top1 在 cls_topn 中按 rank_score 的名次（理论上应该是1）
    df_by_rank = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df_by_rank["rank_rank_in_pool"] = np.arange(1, len(df_by_rank) + 1)
    match_rank = df_by_rank[df_by_rank["code"] == serial_code]
    row["serial_rank_rank_in_pool"] = float(match_rank.iloc[0]["rank_rank_in_pool"]) if len(match_rank) else np.nan

    # 排序Top1是否同时也是分类Top5成员 / Top10成员 / Top20成员
    row["serial_in_cls_top5"] = int(row["serial_cls_rank_in_pool"] <= 5) if pd.notna(row["serial_cls_rank_in_pool"]) else np.nan
    row["serial_in_cls_top10"] = int(row["serial_cls_rank_in_pool"] <= 10) if pd.notna(row["serial_cls_rank_in_pool"]) else np.nan
    row["serial_in_cls_top20"] = int(row["serial_cls_rank_in_pool"] <= 20) if pd.notna(row["serial_cls_rank_in_pool"]) else np.nan

    # “排序挑中的票”在分类上是否也强
    row["serial_cls_rank_pct_in_pool"] = (
        1.0 - (row["serial_cls_rank_in_pool"] - 1) / max(len(df) - 1, 1)
        if pd.notna(row["serial_cls_rank_in_pool"]) else np.nan
    )

    # 分类Top1 与 serial_top1 的差异
    cls_top1 = df_by_cls.iloc[0]
    row["serial_vs_cls1_pcls_gap"] = float(row["serial_p_cls"] - cls_top1["p_cls"])
    row["serial_vs_cls1_rank_gap"] = float(row["serial_pred"] - cls_top1["rank_score"])

    # serial_top1 的 rank_score 相对池子分位
    row["serial_rank_z_in_pool"] = (
        (row["serial_pred"] - row["pool_rank_mean"]) / row["pool_rank_std"]
        if pd.notna(row["pool_rank_std"]) and row["pool_rank_std"] > 1e-12 else np.nan
    )

    # serial_top1 的 p_cls 相对池子分位
    row["serial_pcls_z_in_pool"] = (
        (row["serial_p_cls"] - row["pool_pcls_mean"]) / row["pool_pcls_std"]
        if pd.notna(row["pool_pcls_std"]) and row["pool_pcls_std"] > 1e-12 else np.nan
    )

    # =========================
    # 4) 组合特征（仍然全是当日可见）
    # =========================
    row["conf_rank_x_cls"] = row["serial_score_margin_in_cls"] * row["serial_p_cls"]
    row["conf_pred_x_cls"] = row["serial_pred"] * row["serial_p_cls"]
    row["conf_rankgap_x_rankpct"] = row["serial_score_margin_in_cls"] * row["serial_rank_pct_all"]
    row["conf_pred_x_rankpct"] = row["serial_pred"] * row["serial_rank_pct_all"]
    row["conf_rankz_x_pclsz"] = row["serial_rank_z_in_pool"] * row["serial_pcls_z_in_pool"]

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print("valid_days:", len(df_all))
print_summary("ret (serial_top1 real return)", df_all["ret"])

# =========================================================
# 要分析的真实可用特征
# =========================================================

feature_signs = {
    # serial top1
    "serial_pred": True,
    "serial_score_top1_in_cls": True,
    "serial_score_top2_in_cls": False,  # Top2越高不一定好，先按反向试
    "serial_score_margin_in_cls": True,
    "serial_p_cls": True,
    "serial_p_cls_pct": True,
    "serial_rank_pct_all": True,
    "serial_p_risk": False,  # 风险分通常越高越差

    # pool structure
    "pool_pcls_std": False,    # 先按“越集中越好”试，std高可能不稳
    "pool_rank_std": False,
    "pool_risk_mean": False,
    "pool_risk_std": False,

    # top5 by cls structure
    "top5_cls_gap_1_2": True,
    "top5_cls_gap_1_5": True,
    "top5_cls_std": True,      # 也可后续反向试

    "top5_rank_gap_1_2": True,
    "top5_rank_gap_1_5": True,
    "top5_rank_gap_1_mean_rest": True,
    "top5_rank_std": True,     # 也可后续反向试

    # consistency / conflict
    "serial_cls_rank_in_pool": False,      # 名次越小越好
    "serial_cls_rank_pct_in_pool": True,
    "serial_vs_cls1_pcls_gap": True,       # 越接近0甚至越高越好
    "serial_vs_cls1_rank_gap": True,
    "serial_rank_z_in_pool": True,
    "serial_pcls_z_in_pool": True,

    # combos
    "conf_rank_x_cls": True,
    "conf_pred_x_cls": True,
    "conf_rankgap_x_rankpct": True,
    "conf_pred_x_rankpct": True,
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
# 粗仓位回测
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
    bt_df["mdd_improve"] = bt_df["pos_mdd"] - bt_df["raw_mdd"]     # 越大越好（更接近0）
    bt_df["loss5_improve"] = bt_df["pos_loss5"] - bt_df["raw_loss5"]  # 越小越好
    bt_df["loss8_improve"] = bt_df["pos_loss8"] - bt_df["raw_loss8"]

    print("\n" + "#" * 100)
    print("SIMPLE POSITION BACKTEST (0.4 / 0.7 / 1.0 by 30%-70% quantiles)")
    print("#" * 100)
    print(
        bt_df.sort_values(["mean_diff", "mdd_improve"], ascending=[False, False])
             .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )