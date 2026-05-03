import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260314_043520.json"   # 改成你的新json文件
EPS = 1e-12
SEGMENT_SIZE = 20

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


def diff_stats(name_x, x, name_y, y):
    a = pd.Series(x).dropna().astype(float)
    b = pd.Series(y).dropna().astype(float)
    idx = a.index.intersection(b.index)
    a = a.loc[idx].values
    b = b.loc[idx].values
    d = a - b

    print(f"\n=== Diff: {name_x} - {name_y} ===")
    print("count:", len(d))
    print("mean_diff:", float(np.mean(d)))
    print("median_diff:", float(np.median(d)))
    print("P(diff>0):", float(np.mean(d > 0)))
    print("P(diff<-0.03):", float(np.mean(d < -0.03)))
    print("P(diff<-0.05):", float(np.mean(d < -0.05)))
    print("p5:", float(np.quantile(d, 0.05)))
    print("p10:", float(np.quantile(d, 0.10)))
    print("p90:", float(np.quantile(d, 0.90)))
    print("p95:", float(np.quantile(d, 0.95)))


def zscore(arr):
    s = pd.Series(arr).astype(float)
    std = s.std(ddof=1)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


# =========================================================
# 从 JSON 重建策略与 Top5 风险结构
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
changed_rows = []

for date_str, rec in data.items():
    row = {"date": date_str}

    # json 里原 baseline（通常是 gate80 serial_top1）
    serial_top1 = rec.get("serial_top1", {})
    row["json_baseline_ret"] = serial_top1.get("real_return", np.nan)
    row["json_baseline_p_risk"] = serial_top1.get("p_risk", np.nan)

    cls_topn = rec.get("cls_topn", [])
    if not isinstance(cls_topn, list) or len(cls_topn) == 0:
        rows.append(row)
        continue

    df = pd.DataFrame(cls_topn).copy()

    need_cols = ["code", "p_cls", "rank_score", "real_return", "p_risk"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        rows.append(row)
        continue

    for c in ["p_cls", "rank_score", "real_return", "p_risk", "rank_pos_all", "rank_pct_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------------------
    # 1) 重建 Top5 baseline
    # -------------------------------------------
    df_top5 = df.sort_values("p_cls", ascending=False).head(5).copy()
    if len(df_top5) < 5:
        rows.append(row)
        continue

    df_top5_rank = df_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    base_pick = df_top5_rank.iloc[0]

    row["top5_rank1_code"] = str(base_pick["code"])
    row["top5_rank1_ret"] = float(base_pick["real_return"])
    row["top5_rank1_rank_score"] = float(base_pick["rank_score"])
    row["top5_rank1_p_cls"] = float(base_pick["p_cls"])
    row["top5_rank1_p_risk"] = float(base_pick["p_risk"])

    if len(df_top5_rank) >= 2:
        row["top5_score_margin_1_2"] = float(df_top5_rank.iloc[0]["rank_score"] - df_top5_rank.iloc[1]["rank_score"])
    else:
        row["top5_score_margin_1_2"] = np.nan

    # -------------------------------------------
    # 2) Top5 内 risk 结构
    # -------------------------------------------
    df_risk_desc = df_top5.sort_values(["p_risk", "rank_score"], ascending=[False, False]).reset_index(drop=True)
    risk_max = float(df_risk_desc.iloc[0]["p_risk"])
    risk_second = float(df_risk_desc.iloc[1]["p_risk"])
    risk_min = float(df_risk_desc.iloc[-1]["p_risk"])

    row["risk_max"] = risk_max
    row["risk_second"] = risk_second
    row["risk_min"] = risk_min
    row["risk_gap_1_2"] = risk_max - risk_second
    row["risk_gap_1_5"] = risk_max - risk_min

    max_mask = np.isclose(df_top5["p_risk"].values, risk_max, atol=EPS, rtol=0.0)
    tie_count_max = int(max_mask.sum())
    row["tie_count_max"] = tie_count_max
    row["unique_highest"] = int(tie_count_max == 1)

    highest_risk_candidates = df_top5[max_mask].copy().sort_values("rank_score", ascending=False)
    highest_risk_codes = set(map(str, highest_risk_candidates["code"].tolist()))
    row["baseline_is_highest_risk"] = int(row["top5_rank1_code"] in highest_risk_codes)

    # -------------------------------------------
    # 3) hard remove on Top5
    #    remove 1 / 2
    # -------------------------------------------
    for n in [1, 2]:
        col_ret = f"top5_hard_remove_{n}_ret"
        col_code = f"top5_hard_remove_{n}_code"
        col_risk = f"top5_hard_remove_{n}_p_risk"
        col_diff = f"top5_hard_remove_{n}_diff"

        if len(df_top5) <= n:
            row[col_ret] = np.nan
            row[col_code] = None
            row[col_risk] = np.nan
            row[col_diff] = np.nan
        else:
            keep_df = (
                df_top5.sort_values(["p_risk", "rank_score"], ascending=[True, False])
                      .head(len(df_top5) - n)
                      .copy()
            )
            pick = keep_df.sort_values("rank_score", ascending=False).iloc[0]

            row[col_ret] = float(pick["real_return"])
            row[col_code] = str(pick["code"])
            row[col_risk] = float(pick["p_risk"])
            row[col_diff] = float(pick["real_return"] - base_pick["real_return"])

    # -------------------------------------------
    # 4) soft penalty on Top5
    # -------------------------------------------
    risk_z = zscore(df_top5["p_risk"])
    df_soft = df_top5.copy()
    df_soft["p_risk_z"] = risk_z.values

    for lam in [0.05, 0.10, 0.20, 0.50]:
        final_col = f"final_{lam:.2f}"
        ret_col = f"top5_soft_{lam:.2f}_ret"
        code_col = f"top5_soft_{lam:.2f}_code"
        diff_col = f"top5_soft_{lam:.2f}_diff"

        df_soft[final_col] = df_soft["rank_score"] - lam * df_soft["p_risk_z"]
        pick = df_soft.sort_values(final_col, ascending=False).iloc[0]

        row[ret_col] = float(pick["real_return"])
        row[code_col] = str(pick["code"])
        row[diff_col] = float(pick["real_return"] - base_pick["real_return"])

    rows.append(row)

    if row["top5_hard_remove_1_code"] != row["top5_rank1_code"]:
        changed_rows.append({
            "date": date_str,
            "unique_highest": row["unique_highest"],
            "tie_count_max": row["tie_count_max"],
            "risk_gap_1_2": row["risk_gap_1_2"],
            "risk_gap_1_5": row["risk_gap_1_5"],

            "baseline_code": row["top5_rank1_code"],
            "baseline_ret": row["top5_rank1_ret"],
            "baseline_rank_score": row["top5_rank1_rank_score"],
            "baseline_p_cls": row["top5_rank1_p_cls"],
            "baseline_p_risk": row["top5_rank1_p_risk"],

            "replacement_code": row["top5_hard_remove_1_code"],
            "replacement_ret": row["top5_hard_remove_1_ret"],
            "replacement_p_risk": row["top5_hard_remove_1_p_risk"],

            "ret_diff": row["top5_hard_remove_1_diff"],
        })

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True).set_index("date")
df_changed = pd.DataFrame(changed_rows).sort_values("date").reset_index(drop=True)

print("total_valid_days:", len(df_all))
print("changed_days(remove_1 on Top5):", len(df_changed))


# =========================================================
# 1) 策略层比较
# =========================================================

print("\n\n================ STRATEGY LEVEL COMPARISON ================\n")

strategy_cols = [
    "json_baseline_ret",
    "top5_rank1_ret",
    "top5_hard_remove_1_ret",
    "top5_hard_remove_2_ret",
    "top5_soft_0.05_ret",
    "top5_soft_0.10_ret",
    "top5_soft_0.20_ret",
    "top5_soft_0.50_ret",
]

summary = {}
for col in strategy_cols:
    summary[col] = print_summary(col, df_all[col])

print("\n=== Key comparison table ===")
comp_cols = [
    "mean", "median", "win_rate",
    "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
    "cvar10", "max_drawdown", "sharpe"
]
comp = pd.DataFrame(summary).T[comp_cols]
print(comp.to_string(float_format=lambda x: f"{x:.6f}"))

for col in strategy_cols[2:]:
    diff_stats(col, df_all[col], "top5_rank1_ret", df_all["top5_rank1_ret"])

diff_stats("top5_rank1_ret", df_all["top5_rank1_ret"], "json_baseline_ret", df_all["json_baseline_ret"])


# =========================================================
# 2) Top5 风险分辨率
# =========================================================

print("\n\n================ TOP5 RISK RESOLUTION ================\n")
print_summary("risk_gap_1_2", df_all["risk_gap_1_2"])
print_summary("risk_gap_1_5", df_all["risk_gap_1_5"])

print("\n=== Tie / uniqueness stats ===")
print("unique_highest_rate:", float(df_all["unique_highest"].mean()))
print("tie_rate_any:", float((df_all["tie_count_max"] > 1).mean()))
print("tie_count distribution:")
print(df_all["tie_count_max"].value_counts(dropna=False).sort_index().to_string())

print("\n=== Baseline relation to highest risk ===")
print("baseline_is_highest_risk rate:", float(df_all["baseline_is_highest_risk"].mean()))
print("hard_remove_1_change rate:", float((df_all["top5_hard_remove_1_code"] != df_all["top5_rank1_code"]).mean()))
print(pd.crosstab(df_all["baseline_is_highest_risk"], (df_all["top5_hard_remove_1_code"] != df_all["top5_rank1_code"]).astype(int)))


# =========================================================
# 3) 唯一最高 vs 并列最高
# =========================================================

print("\n\n================ UNIQUE vs TIED HIGHEST ================\n")

unique_df = df_all[df_all["unique_highest"] == 1].copy()
tied_df = df_all[df_all["unique_highest"] == 0].copy()

for name, sub in [("unique_highest", unique_df), ("tied_highest", tied_df)]:
    print(f"\n--- {name} ---")
    print("days:", len(sub))
    change_rate = float((sub["top5_hard_remove_1_code"] != sub["top5_rank1_code"]).mean()) if len(sub) > 0 else np.nan
    print("hard_remove_1_change_rate:", change_rate)
    print_summary(f"{name}_baseline_ret", sub["top5_rank1_ret"])
    print_summary(f"{name}_hard_remove_1_diff", sub["top5_hard_remove_1_diff"])

    changed_sub = sub[sub["top5_hard_remove_1_code"] != sub["top5_rank1_code"]].copy()
    print(f"{name}_changed_days:", len(changed_sub))
    if len(changed_sub) > 0:
        print("mean_ret_diff_on_changed:", float(changed_sub["top5_hard_remove_1_diff"].mean()))
        print("median_ret_diff_on_changed:", float(changed_sub["top5_hard_remove_1_diff"].median()))
        print("P(diff>0 | changed):", float((changed_sub["top5_hard_remove_1_diff"] > 0).mean()))
        print("P(diff<-0.03 | changed):", float((changed_sub["top5_hard_remove_1_diff"] < -0.03).mean()))
        print("P(diff<-0.05 | changed):", float((changed_sub["top5_hard_remove_1_diff"] < -0.05).mean()))


# =========================================================
# 4) 改票样本画像
# =========================================================

print("\n\n================ CHANGED DAYS PROFILE ================\n")

if len(df_changed) > 0:
    print_summary("ret_diff_on_changed_days", df_changed["ret_diff"])

    cols = [
        "baseline_ret", "replacement_ret", "ret_diff",
        "baseline_rank_score",
        "baseline_p_cls",
        "baseline_p_risk",
        "replacement_p_risk",
        "risk_gap_1_2", "risk_gap_1_5"
    ]

    print("\n=== Changed days profile means ===")
    print(df_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    good_changed = df_changed[df_changed["ret_diff"] > 0].copy()
    bad_changed = df_changed[df_changed["ret_diff"] <= 0].copy()

    print("\n=== Good changed days ===")
    print("n:", len(good_changed))
    if len(good_changed) > 0:
        print(good_changed[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Bad changed days ===")
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
# 5) 分段稳定性（20天）
# =========================================================

print("\n\n================ SEGMENT STABILITY ================\n")

segments = []
n = len(df_all)

for start in range(0, n, SEGMENT_SIZE):
    seg = df_all.iloc[start:start + SEGMENT_SIZE].copy()
    if len(seg) < max(8, SEGMENT_SIZE // 2):
        continue

    base = seg["top5_rank1_ret"].astype(float)
    hr1 = seg["top5_hard_remove_1_ret"].astype(float)
    diff = hr1 - base

    segments.append({
        "start": seg.index[0],
        "end": seg.index[-1],
        "n": len(seg),
        "mean_base": float(base.mean()),
        "mean_hr1": float(hr1.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "p_diff_pos": float((diff > 0).mean()),
        "base_bigloss5": float((base <= -0.05).mean()),
        "hr1_bigloss5": float((hr1 <= -0.05).mean()),
        "base_maxdd": max_drawdown_from_returns(base),
        "hr1_maxdd": max_drawdown_from_returns(hr1),
        "change_rate": float((seg["top5_hard_remove_1_code"] != seg["top5_rank1_code"]).mean()),
        "unique_highest_rate": float(seg["unique_highest"].mean()),
        "risk_gap_1_2_mean": float(seg["risk_gap_1_2"].mean()),
    })

seg_df = pd.DataFrame(segments)
print(seg_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if len(seg_df) > 0:
    print("\n=== Segment summary ===")
    print("P(segment mean_diff > 0):", float((seg_df["mean_diff"] > 0).mean()))
    print("mean unique_highest_rate:", float(seg_df["unique_highest_rate"].mean()))
    print("mean risk_gap_1_2_mean:", float(seg_df["risk_gap_1_2_mean"].mean()))


# =========================================================
# 6) 两个时间段分析
# =========================================================

print("\n\n================ TWO PERIOD ANALYSIS ================\n")

period_rows = []

for start_date, end_date, label in PERIODS:
    sub = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()
    changed_sub = sub[sub["top5_hard_remove_1_code"] != sub["top5_rank1_code"]].copy()

    print("\n" + "=" * 70)
    print(f"{label} [{start_date} ~ {end_date}]")
    print("=" * 70)

    print_summary(f"{label}_top5_rank1", sub["top5_rank1_ret"])
    print_summary(f"{label}_top5_hard_remove_1", sub["top5_hard_remove_1_ret"])

    print("days:", len(sub))
    print("unique_highest_rate:", float(sub["unique_highest"].mean()) if len(sub) > 0 else np.nan)
    print("tie_rate_any:", float((sub["tie_count_max"] > 1).mean()) if len(sub) > 0 else np.nan)
    print("change_rate:", float((sub["top5_hard_remove_1_code"] != sub["top5_rank1_code"]).mean()) if len(sub) > 0 else np.nan)

    print_summary(f"{label}_risk_gap_1_2", sub["risk_gap_1_2"])
    print_summary(f"{label}_risk_gap_1_5", sub["risk_gap_1_5"])

    if len(changed_sub) > 0:
        print_summary(f"{label}_ret_diff_on_changed", changed_sub["top5_hard_remove_1_diff"])
        print("P(diff>0 | changed):", float((changed_sub["top5_hard_remove_1_diff"] > 0).mean()))
        print("P(diff<-0.03 | changed):", float((changed_sub["top5_hard_remove_1_diff"] < -0.03).mean()))
        print("P(diff<-0.05 | changed):", float((changed_sub["top5_hard_remove_1_diff"] < -0.05).mean()))

    period_rows.append({
        "period": label,
        "start": start_date,
        "end": end_date,
        "days": len(sub),

        "base_mean": float(sub["top5_rank1_ret"].mean()) if len(sub) > 0 else np.nan,
        "hr1_mean": float(sub["top5_hard_remove_1_ret"].mean()) if len(sub) > 0 else np.nan,
        "mean_diff": float((sub["top5_hard_remove_1_ret"] - sub["top5_rank1_ret"]).mean()) if len(sub) > 0 else np.nan,

        "unique_highest_rate": float(sub["unique_highest"].mean()) if len(sub) > 0 else np.nan,
        "tie_rate_any": float((sub["tie_count_max"] > 1).mean()) if len(sub) > 0 else np.nan,
        "change_rate": float((sub["top5_hard_remove_1_code"] != sub["top5_rank1_code"]).mean()) if len(sub) > 0 else np.nan,

        "risk_gap_1_2_mean": float(sub["risk_gap_1_2"].mean()) if len(sub) > 0 else np.nan,
        "risk_gap_1_2_median": float(sub["risk_gap_1_2"].median()) if len(sub) > 0 else np.nan,

        "changed_mean_diff": float(changed_sub["top5_hard_remove_1_diff"].mean()) if len(changed_sub) > 0 else np.nan,
        "changed_p_pos": float((changed_sub["top5_hard_remove_1_diff"] > 0).mean()) if len(changed_sub) > 0 else np.nan,
    })

print("\n=== TWO-PERIOD SUMMARY TABLE ===")
period_df = pd.DataFrame(period_rows)
print(period_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 7) 自动解释
# =========================================================

print("\n\n================ AUTO INTERPRETATION ================\n")
print("1) 先看 Strategy Level：新的风控模型在 Top5+hard_remove_1 上，是否比旧版更好。")
print("2) 看 Top5 Risk Resolution：如果 tie_rate_any 仍然很高、risk_gap_1_2 median 仍接近0，说明局部分辨率问题仍在。")
print("3) 看 Unique vs Tied Highest：如果 unique_highest 子集明显更有效，说明实盘应考虑'仅唯一最高才删'。")
print("4) 看 Two Period：如果 P2 的 unique_highest_rate 上升、risk_gap_1_2 变大，说明新模型后段局部分辨率更好。")