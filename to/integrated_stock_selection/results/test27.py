import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json" # 改成你的json文件
EPS = 1e-12  # 判断并列风险时的容差
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


# =========================================================
# 从 JSON 逐日解析
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

day_rows = []
changed_pair_rows = []

for date_str, rec in data.items():
    cls_topn = rec.get("cls_topn", [])
    if not isinstance(cls_topn, list) or len(cls_topn) < 5:
        continue

    df = pd.DataFrame(cls_topn).copy()

    need_cols = ["code", "p_cls", "rank_score", "real_return", "p_risk"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        continue

    for c in ["p_cls", "rank_score", "real_return", "p_risk", "rank_pos_all", "rank_pct_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 取分类Top5
    df_top5 = df.sort_values("p_cls", ascending=False).head(5).copy()

    # baseline = Top5内排序Top1
    df_top5_rank = df_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    base_pick = df_top5_rank.iloc[0]

    # risk 排名
    df_risk_desc = df_top5.sort_values(["p_risk", "rank_score"], ascending=[False, False]).reset_index(drop=True)
    risk_max = float(df_risk_desc.iloc[0]["p_risk"])
    risk_second = float(df_risk_desc.iloc[1]["p_risk"])
    risk_min = float(df_risk_desc.iloc[-1]["p_risk"])

    risk_gap_1_2 = risk_max - risk_second
    risk_gap_1_5 = risk_max - risk_min

    # 并列最高风险
    max_mask = np.isclose(df_top5["p_risk"].values, risk_max, atol=EPS, rtol=0.0)
    tie_count_max = int(max_mask.sum())
    unique_highest = int(tie_count_max == 1)

    highest_risk_candidates = df_top5[max_mask].copy().sort_values("rank_score", ascending=False)
    highest_risk_codes = highest_risk_candidates["code"].tolist()
    baseline_is_highest_risk = int(str(base_pick["code"]) in set(map(str, highest_risk_codes)))

    # always remove 1：删掉 risk 排名第一（并列时按 rank_score 最大删）
    removed_row = df_risk_desc.iloc[0]
    keep_df = df_top5[df_top5["code"] != removed_row["code"]].copy()
    keep_pick = keep_df.sort_values("rank_score", ascending=False).iloc[0]

    always_change = int(str(keep_pick["code"]) != str(base_pick["code"]))
    always_ret_diff = float(keep_pick["real_return"] - base_pick["real_return"])

    # 全市场 rank 风险位置（在 cls_topn 里用 rank_pct_all 已经有排序位置，但这里是 risk 的局部信息）
    day_rows.append({
        "date": date_str,

        "baseline_code": str(base_pick["code"]),
        "baseline_ret": float(base_pick["real_return"]),
        "baseline_rank_score": float(base_pick["rank_score"]),
        "baseline_p_cls": float(base_pick["p_cls"]),
        "baseline_p_risk": float(base_pick["p_risk"]),

        "removed_code": str(removed_row["code"]),
        "removed_ret": float(removed_row["real_return"]),
        "removed_rank_score": float(removed_row["rank_score"]),
        "removed_p_cls": float(removed_row["p_cls"]),
        "removed_p_risk": float(removed_row["p_risk"]),

        "replacement_code": str(keep_pick["code"]),
        "replacement_ret": float(keep_pick["real_return"]),
        "replacement_rank_score": float(keep_pick["rank_score"]),
        "replacement_p_cls": float(keep_pick["p_cls"]),
        "replacement_p_risk": float(keep_pick["p_risk"]),

        "risk_max": risk_max,
        "risk_second": risk_second,
        "risk_min": risk_min,
        "risk_gap_1_2": risk_gap_1_2,
        "risk_gap_1_5": risk_gap_1_5,

        "tie_count_max": tie_count_max,
        "unique_highest": unique_highest,
        "baseline_is_highest_risk": baseline_is_highest_risk,

        "always_change": always_change,
        "always_ret_diff": always_ret_diff,
    })

    if always_change == 1:
        changed_pair_rows.append({
            "date": date_str,
            "unique_highest": unique_highest,
            "tie_count_max": tie_count_max,
            "risk_gap_1_2": risk_gap_1_2,
            "risk_gap_1_5": risk_gap_1_5,

            "baseline_code": str(base_pick["code"]),
            "baseline_ret": float(base_pick["real_return"]),
            "baseline_rank_score": float(base_pick["rank_score"]),
            "baseline_p_cls": float(base_pick["p_cls"]),
            "baseline_p_risk": float(base_pick["p_risk"]),

            "removed_code": str(removed_row["code"]),
            "removed_ret": float(removed_row["real_return"]),
            "removed_rank_score": float(removed_row["rank_score"]),
            "removed_p_cls": float(removed_row["p_cls"]),
            "removed_p_risk": float(removed_row["p_risk"]),

            "replacement_code": str(keep_pick["code"]),
            "replacement_ret": float(keep_pick["real_return"]),
            "replacement_rank_score": float(keep_pick["rank_score"]),
            "replacement_p_cls": float(keep_pick["p_cls"]),
            "replacement_p_risk": float(keep_pick["p_risk"]),

            "ret_diff": always_ret_diff,
        })

df_day = pd.DataFrame(day_rows).sort_values("date").reset_index(drop=True)
df_changed = pd.DataFrame(changed_pair_rows).sort_values("date").reset_index(drop=True)

print("total_valid_days:", len(df_day))
print("changed_days:", len(df_changed))


# =========================================================
# 1) Top5 内 risk 分辨率
# =========================================================

print("\n\n================ TOP5 RISK RESOLUTION ================\n")
print_summary("risk_gap_1_2", df_day["risk_gap_1_2"])
print_summary("risk_gap_1_5", df_day["risk_gap_1_5"])

print("\n=== Tie / uniqueness stats ===")
print("unique_highest_rate:", float(df_day["unique_highest"].mean()))
print("tie_rate_any:", float((df_day["tie_count_max"] > 1).mean()))
print("tie_count distribution:")
print(df_day["tie_count_max"].value_counts(dropna=False).sort_index().to_string())

print("\n=== Baseline relation to highest risk ===")
print("baseline_is_highest_risk rate:", float(df_day["baseline_is_highest_risk"].mean()))
print("always_change rate:", float(df_day["always_change"].mean()))
print("Check equality baseline_is_highest_risk vs always_change:")
print(pd.crosstab(df_day["baseline_is_highest_risk"], df_day["always_change"]))


# =========================================================
# 2) 唯一最高 vs 并列最高
# =========================================================

print("\n\n================ UNIQUE vs TIED HIGHEST ================\n")

unique_df = df_day[df_day["unique_highest"] == 1].copy()
tied_df = df_day[df_day["unique_highest"] == 0].copy()

for name, sub in [("unique_highest", unique_df), ("tied_highest", tied_df)]:
    print(f"\n--- {name} ---")
    print("days:", len(sub))
    print("always_change_rate:", float(sub["always_change"].mean()) if len(sub) > 0 else np.nan)
    print_summary(f"{name}_baseline_ret", sub["baseline_ret"])
    print_summary(f"{name}_always_ret_diff", sub["always_ret_diff"])

    changed_sub = sub[sub["always_change"] == 1].copy()
    print(f"{name}_changed_days:", len(changed_sub))
    if len(changed_sub) > 0:
        print("mean_ret_diff_on_changed:", float(changed_sub["always_ret_diff"].mean()))
        print("median_ret_diff_on_changed:", float(changed_sub["always_ret_diff"].median()))
        print("P(diff>0 | changed):", float((changed_sub["always_ret_diff"] > 0).mean()))
        print("P(diff<-0.03 | changed):", float((changed_sub["always_ret_diff"] < -0.03).mean()))
        print("P(diff<-0.05 | changed):", float((changed_sub["always_ret_diff"] < -0.05).mean()))


# =========================================================
# 3) 改票日画像
# =========================================================

print("\n\n================ CHANGED DAYS PROFILE ================\n")

if len(df_changed) > 0:
    print_summary("ret_diff_on_changed_days", df_changed["ret_diff"])

    cols = [
        "baseline_ret", "replacement_ret", "ret_diff",
        "baseline_rank_score", "replacement_rank_score",
        "baseline_p_cls", "replacement_p_cls",
        "baseline_p_risk", "replacement_p_risk",
        "risk_gap_1_2", "risk_gap_1_5"
    ]

    print("\n=== Changed days profile means ===")
    print(df_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Good changed days ===")
    good_changed = df_changed[df_changed["ret_diff"] > 0].copy()
    print("n:", len(good_changed))
    if len(good_changed) > 0:
        print(good_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Bad changed days ===")
    bad_changed = df_changed[df_changed["ret_diff"] <= 0].copy()
    print("n:", len(bad_changed))
    if len(bad_changed) > 0:
        print(bad_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Top positive changed days ===")
    print(
        df_changed.sort_values("ret_diff", ascending=False)
                  .head(20)
                  .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n=== Top negative changed days ===")
    print(
        df_changed.sort_values("ret_diff", ascending=True)
                  .head(20)
                  .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )


# =========================================================
# 4) 两个时间段分析
# =========================================================

print("\n\n================ TWO PERIOD ANALYSIS ================\n")

period_rows = []

for start_date, end_date, label in PERIODS:
    sub = df_day[(df_day["date"] >= start_date) & (df_day["date"] <= end_date)].copy()
    changed_sub = sub[sub["always_change"] == 1].copy()

    print("\n" + "=" * 70)
    print(f"{label} [{start_date} ~ {end_date}]")
    print("=" * 70)

    print("days:", len(sub))
    print("unique_highest_rate:", float(sub["unique_highest"].mean()) if len(sub) > 0 else np.nan)
    print("tie_rate_any:", float((sub["tie_count_max"] > 1).mean()) if len(sub) > 0 else np.nan)
    print("always_change_rate:", float(sub["always_change"].mean()) if len(sub) > 0 else np.nan)

    print_summary(f"{label}_risk_gap_1_2", sub["risk_gap_1_2"])
    print_summary(f"{label}_risk_gap_1_5", sub["risk_gap_1_5"])

    if len(changed_sub) > 0:
        print_summary(f"{label}_ret_diff_on_changed", changed_sub["always_ret_diff"])
        print("P(diff>0 | changed):", float((changed_sub["always_ret_diff"] > 0).mean()))
        print("P(diff<-0.03 | changed):", float((changed_sub["always_ret_diff"] < -0.03).mean()))
        print("P(diff<-0.05 | changed):", float((changed_sub["always_ret_diff"] < -0.05).mean()))

    period_rows.append({
        "period": label,
        "start": start_date,
        "end": end_date,
        "days": len(sub),

        "unique_highest_rate": float(sub["unique_highest"].mean()) if len(sub) > 0 else np.nan,
        "tie_rate_any": float((sub["tie_count_max"] > 1).mean()) if len(sub) > 0 else np.nan,
        "always_change_rate": float(sub["always_change"].mean()) if len(sub) > 0 else np.nan,

        "risk_gap_1_2_mean": float(sub["risk_gap_1_2"].mean()) if len(sub) > 0 else np.nan,
        "risk_gap_1_2_median": float(sub["risk_gap_1_2"].median()) if len(sub) > 0 else np.nan,

        "changed_mean_diff": float(changed_sub["always_ret_diff"].mean()) if len(changed_sub) > 0 else np.nan,
        "changed_p_pos": float((changed_sub["always_ret_diff"] > 0).mean()) if len(changed_sub) > 0 else np.nan,
    })

print("\n=== TWO-PERIOD SUMMARY TABLE ===")
period_df = pd.DataFrame(period_rows)
print(period_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 5) 自动解释
# =========================================================

print("\n\n================ AUTO INTERPRETATION ================\n")
print("1) 如果 tie_rate_any 很高，说明 Top5 内 p_risk 分辨率不足，不适合精细删票。")
print("2) 如果 unique_highest_rate 不高，但 changed_mean_diff 主要来自 unique_highest 子集，说明应考虑'只在唯一最高风险时才删'。")
print("3) 如果 risk_gap_1_2 的均值/中位数很小，说明最高风险与次高风险差异不足，删票决策容易脆弱。")
print("4) 如果 P2 的 unique_highest_rate 或 risk_gap_1_2 明显下降，说明后段风控模型分辨率在变差。")