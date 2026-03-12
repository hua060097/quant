import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json"    # 改成你的文件名

PERIODS = [
    ("20250605", "20250912", "P1_20250605_20250912"),
    ("20250913", "20260302", "P2_20250913_20260302"),
]


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


def cvar_left_tail(ret, q=0.10):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    th = np.quantile(x, q)
    tail = x[x <= th]
    return float(tail.mean()) if len(tail) > 0 else np.nan


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
        "cvar10": cvar_left_tail(x, 0.10),
        "max_drawdown": max_drawdown_from_returns(x),
        "sharpe": float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(252)),
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


# =========================================================
# 从 JSON 重建 Top5 baseline / hard_remove_1
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for date_str, rec in data.items():
    row = {"date": date_str}

    cls_topn = rec.get("cls_topn", [])
    if not isinstance(cls_topn, list) or len(cls_topn) == 0:
        rows.append(row)
        continue

    df = pd.DataFrame(cls_topn)

    for c in ["p_cls", "rank_score", "real_return", "p_risk", "rank_pos_all", "rank_pct_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need_cols = {"p_cls", "rank_score", "real_return"}
    if not need_cols.issubset(df.columns):
        rows.append(row)
        continue

    # 分类 Top5
    df_top5 = df.sort_values("p_cls", ascending=False).head(5).copy()

    # baseline: Top5 -> 排序Top1
    df_top5_rank = df_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    if len(df_top5_rank) == 0:
        rows.append(row)
        continue

    base_pick = df_top5_rank.iloc[0]
    row["top5_rank1_code"] = base_pick.get("code", None)
    row["top5_rank1_ret"] = base_pick.get("real_return", np.nan)
    row["top5_rank1_rank_score"] = base_pick.get("rank_score", np.nan)
    row["top5_rank1_p_cls"] = base_pick.get("p_cls", np.nan)
    row["top5_rank1_p_risk"] = base_pick.get("p_risk", np.nan)

    # hard remove 1: 删除 Top5 中 p_risk 最高的那只，再排序选Top1
    if "p_risk" in df_top5.columns and df_top5["p_risk"].notna().sum() >= 2:
        removed_df = (
            df_top5.sort_values(["p_risk", "rank_score"], ascending=[False, False])
                  .head(1)
                  .copy()
        )
        kept_df = (
            df_top5.sort_values(["p_risk", "rank_score"], ascending=[True, False])
                  .head(len(df_top5) - 1)
                  .copy()
        )
        new_pick = kept_df.sort_values("rank_score", ascending=False).iloc[0]
        removed = removed_df.iloc[0]

        row["hard_remove_1_code"] = new_pick.get("code", None)
        row["hard_remove_1_ret"] = new_pick.get("real_return", np.nan)
        row["hard_remove_1_rank_score"] = new_pick.get("rank_score", np.nan)
        row["hard_remove_1_p_cls"] = new_pick.get("p_cls", np.nan)
        row["hard_remove_1_p_risk"] = new_pick.get("p_risk", np.nan)

        row["removed_code"] = removed.get("code", None)
        row["removed_ret"] = removed.get("real_return", np.nan)
        row["removed_rank_score"] = removed.get("rank_score", np.nan)
        row["removed_p_cls"] = removed.get("p_cls", np.nan)
        row["removed_p_risk"] = removed.get("p_risk", np.nan)

        row["change_flag"] = int(row["hard_remove_1_code"] != row["top5_rank1_code"])
        row["ret_diff"] = (
            float(row["hard_remove_1_ret"]) - float(row["top5_rank1_ret"])
            if pd.notna(row["hard_remove_1_ret"]) and pd.notna(row["top5_rank1_ret"])
            else np.nan
        )
    else:
        row["hard_remove_1_code"] = None
        row["hard_remove_1_ret"] = np.nan
        row["hard_remove_1_rank_score"] = np.nan
        row["hard_remove_1_p_cls"] = np.nan
        row["hard_remove_1_p_risk"] = np.nan

        row["removed_code"] = None
        row["removed_ret"] = np.nan
        row["removed_rank_score"] = np.nan
        row["removed_p_cls"] = np.nan
        row["removed_p_risk"] = np.nan

        row["change_flag"] = 0
        row["ret_diff"] = np.nan

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df_all = df_all.set_index("date")


# =========================================================
# 按两个时间段分析
# =========================================================

period_results = []

for start_date, end_date, label in PERIODS:
    sub = df_all.loc[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    print("\n\n" + "=" * 70)
    print(f"{label}  [{start_date} ~ {end_date}]")
    print("=" * 70)

    base_stats = print_summary(f"{label} Top5+Rank1 baseline", sub["top5_rank1_ret"])
    new_stats = print_summary(f"{label} Top5+HardRemove1", sub["hard_remove_1_ret"])

    diff = (sub["hard_remove_1_ret"] - sub["top5_rank1_ret"]).dropna()

    print(f"\n=== {label} hard_remove_1 - baseline ===")
    print("count:", int(diff.notna().sum()))
    print("mean_diff:", float(diff.mean()))
    print("median_diff:", float(diff.median()))
    print("P(diff>0):", float((diff > 0).mean()))
    print("P(diff<-0.03):", float((diff < -0.03).mean()))
    print("P(diff<-0.05):", float((diff < -0.05).mean()))

    changed = sub[sub["change_flag"] == 1].copy()
    unchanged = sub[sub["change_flag"] == 0].copy()

    print(f"\n=== {label} change stats ===")
    print("total_days:", len(sub))
    print("changed_days:", len(changed))
    print("change_rate:", float(changed.shape[0] / len(sub)) if len(sub) > 0 else np.nan)

    if len(changed) > 0:
        print("mean_ret_diff_on_changed_days:", float(changed["ret_diff"].mean()))
        print("median_ret_diff_on_changed_days:", float(changed["ret_diff"].median()))
        print("P(ret_diff>0 | changed):", float((changed["ret_diff"] > 0).mean()))
        print("P(ret_diff<-0.03 | changed):", float((changed["ret_diff"] < -0.03).mean()))
        print("P(ret_diff<-0.05 | changed):", float((changed["ret_diff"] < -0.05).mean()))

        print(f"\n=== {label} changed days profile means ===")
        cols = [
            "top5_rank1_ret", "hard_remove_1_ret", "ret_diff",
            "top5_rank1_rank_score", "hard_remove_1_rank_score",
            "top5_rank1_p_cls", "hard_remove_1_p_cls",
            "top5_rank1_p_risk", "hard_remove_1_p_risk",
            "removed_ret", "removed_rank_score", "removed_p_cls", "removed_p_risk"
        ]
        print(changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

        good_changed = changed[changed["ret_diff"] > 0]
        bad_changed = changed[changed["ret_diff"] <= 0]

        print(f"\n=== {label} good changed days: {len(good_changed)} ===")
        if len(good_changed) > 0:
            print(good_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

        print(f"\n=== {label} bad changed days: {len(bad_changed)} ===")
        if len(bad_changed) > 0:
            print(bad_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    period_results.append({
        "period": label,
        "start": start_date,
        "end": end_date,
        "n_days": len(sub),

        "baseline_mean": base_stats["mean"] if base_stats else np.nan,
        "baseline_median": base_stats["median"] if base_stats else np.nan,
        "baseline_bigloss5": base_stats["big_loss_5pct_rate"] if base_stats else np.nan,
        "baseline_maxdd": base_stats["max_drawdown"] if base_stats else np.nan,
        "baseline_sharpe": base_stats["sharpe"] if base_stats else np.nan,

        "new_mean": new_stats["mean"] if new_stats else np.nan,
        "new_median": new_stats["median"] if new_stats else np.nan,
        "new_bigloss5": new_stats["big_loss_5pct_rate"] if new_stats else np.nan,
        "new_maxdd": new_stats["max_drawdown"] if new_stats else np.nan,
        "new_sharpe": new_stats["sharpe"] if new_stats else np.nan,

        "mean_diff": float(diff.mean()) if len(diff) > 0 else np.nan,
        "median_diff": float(diff.median()) if len(diff) > 0 else np.nan,
        "p_diff_pos": float((diff > 0).mean()) if len(diff) > 0 else np.nan,

        "change_rate": float(changed.shape[0] / len(sub)) if len(sub) > 0 else np.nan,
        "changed_mean_diff": float(changed["ret_diff"].mean()) if len(changed) > 0 else np.nan,
        "changed_p_pos": float((changed["ret_diff"] > 0).mean()) if len(changed) > 0 else np.nan,
    })

# 汇总表
print("\n\n" + "=" * 70)
print("TWO-PERIOD SUMMARY TABLE")
print("=" * 70)

summary_df = pd.DataFrame(period_results)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 简单自动判断
# =========================================================

if len(summary_df) == 2:
    p1 = summary_df.iloc[0]
    p2 = summary_df.iloc[1]

    print("\n\n" + "=" * 70)
    print("AUTO INTERPRETATION")
    print("=" * 70)

    print("1) 分类Top5收益是否下降：")
    print("baseline_mean_change =", float(p2["baseline_mean"] - p1["baseline_mean"]))
    print("baseline_median_change =", float(p2["baseline_median"] - p1["baseline_median"]))
    print("baseline_bigloss5_change =", float(p2["baseline_bigloss5"] - p1["baseline_bigloss5"]))
    print("baseline_maxdd_change =", float(p2["baseline_maxdd"] - p1["baseline_maxdd"]))

    print("\n2) 风控模型作用是否下降：")
    print("mean_diff_change =", float(p2["mean_diff"] - p1["mean_diff"]))
    print("change_rate_change =", float(p2["change_rate"] - p1["change_rate"]))
    print("changed_mean_diff_change =", float(p2["changed_mean_diff"] - p1["changed_mean_diff"]))
    print("changed_p_pos_change =", float(p2["changed_p_pos"] - p1["changed_p_pos"]))

    print("\nInterpretation guide:")
    print("- 若 baseline_mean_change 明显为负，说明分类Top5这条主alpha后段变弱")
    print("- 若 mean_diff_change 明显为负，说明风控模型增益后段变弱")
    print("- 若 change_rate 下降很多，说明后段风控模型介入更少")
    print("- 若 changed_mean_diff / changed_p_pos 下降很多，说明后段即使介入也更难帮到策略")