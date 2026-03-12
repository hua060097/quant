import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json"    # 改成你的json路径
SEGMENT_SIZE = 20  # 可改成 20 / 30


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
        return
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


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

    # 数值列
    for c in ["p_cls", "rank_score", "real_return", "p_risk", "rank_pos_all", "rank_pct_all"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    need_cols = {"p_cls", "rank_score", "real_return"}
    if not need_cols.issubset(df.columns):
        rows.append(row)
        continue

    # 原 Top5 baseline
    df_top5 = df.sort_values("p_cls", ascending=False).head(5).copy()
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
    row["top5_rank1_rank_pos_all"] = base_pick.get("rank_pos_all", np.nan)
    row["top5_rank1_rank_pct_all"] = base_pick.get("rank_pct_all", np.nan)

    if len(df_top5_rank) >= 2:
        row["top5_score_margin_1_2"] = float(df_top5_rank.iloc[0]["rank_score"] - df_top5_rank.iloc[1]["rank_score"])
    else:
        row["top5_score_margin_1_2"] = np.nan

    # hard remove 1
    if "p_risk" in df_top5.columns and df_top5["p_risk"].notna().sum() >= 2:
        df_keep = (
            df_top5.sort_values(["p_risk", "rank_score"], ascending=[True, False])
                  .head(len(df_top5) - 1)
                  .copy()
        )

        removed_df = (
            df_top5.sort_values(["p_risk", "rank_score"], ascending=[False, False])
                  .head(1)
                  .copy()
        )

        pick_after_remove = df_keep.sort_values("rank_score", ascending=False).iloc[0]
        removed_row = removed_df.iloc[0]

        row["hard_remove_1_code"] = pick_after_remove.get("code", None)
        row["hard_remove_1_ret"] = pick_after_remove.get("real_return", np.nan)
        row["hard_remove_1_rank_score"] = pick_after_remove.get("rank_score", np.nan)
        row["hard_remove_1_p_cls"] = pick_after_remove.get("p_cls", np.nan)
        row["hard_remove_1_p_risk"] = pick_after_remove.get("p_risk", np.nan)
        row["hard_remove_1_rank_pos_all"] = pick_after_remove.get("rank_pos_all", np.nan)
        row["hard_remove_1_rank_pct_all"] = pick_after_remove.get("rank_pct_all", np.nan)

        row["removed_code"] = removed_row.get("code", None)
        row["removed_ret"] = removed_row.get("real_return", np.nan)
        row["removed_rank_score"] = removed_row.get("rank_score", np.nan)
        row["removed_p_cls"] = removed_row.get("p_cls", np.nan)
        row["removed_p_risk"] = removed_row.get("p_risk", np.nan)
        row["removed_rank_pos_all"] = removed_row.get("rank_pos_all", np.nan)
        row["removed_rank_pct_all"] = removed_row.get("rank_pct_all", np.nan)

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
        row["hard_remove_1_rank_pos_all"] = np.nan
        row["hard_remove_1_rank_pct_all"] = np.nan

        row["removed_code"] = None
        row["removed_ret"] = np.nan
        row["removed_rank_score"] = np.nan
        row["removed_p_cls"] = np.nan
        row["removed_p_risk"] = np.nan
        row["removed_rank_pos_all"] = np.nan
        row["removed_rank_pct_all"] = np.nan

        row["change_flag"] = 0
        row["ret_diff"] = np.nan

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df_all = df_all.set_index("date")

# 丢掉 baseline 空值日期
valid = df_all[df_all["top5_rank1_ret"].notna()].copy()

print_summary("Top5+Rank1 baseline", valid["top5_rank1_ret"])
print_summary("Top5+HardRemove1", valid["hard_remove_1_ret"])

print("\n=== Overall diff: hard_remove_1 - top5_rank1 ===")
d = valid["hard_remove_1_ret"] - valid["top5_rank1_ret"]
print("count:", int(d.notna().sum()))
print("mean_diff:", float(d.mean()))
print("median_diff:", float(d.median()))
print("P(diff>0):", float((d > 0).mean()))
print("P(diff<-0.03):", float((d < -0.03).mean()))
print("P(diff<-0.05):", float((d < -0.05).mean()))


# =========================================================
# 1) 分段稳定性
# =========================================================

print("\n\n================ SEGMENT STABILITY ================\n")

segments = []
n = len(valid)

for start in range(0, n, SEGMENT_SIZE):
    seg = valid.iloc[start:start + SEGMENT_SIZE].copy()
    if len(seg) < max(8, SEGMENT_SIZE // 2):
        continue

    base = seg["top5_rank1_ret"].astype(float)
    new = seg["hard_remove_1_ret"].astype(float)
    diff = new - base

    segments.append({
        "start": seg.index[0],
        "end": seg.index[-1],
        "n": len(seg),
        "mean_base": float(base.mean()),
        "mean_new": float(new.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "p_diff_pos": float((diff > 0).mean()),
        "base_bigloss5": float((base <= -0.05).mean()),
        "new_bigloss5": float((new <= -0.05).mean()),
        "base_maxdd": max_drawdown_from_returns(base),
        "new_maxdd": max_drawdown_from_returns(new),
        "change_rate": float(seg["change_flag"].mean()),
    })

seg_df = pd.DataFrame(segments)
print(seg_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if len(seg_df) > 0:
    print("\n=== Segment summary ===")
    print("mean of mean_diff:", float(seg_df["mean_diff"].mean()))
    print("median of mean_diff:", float(seg_df["mean_diff"].median()))
    print("P(segment mean_diff > 0):", float((seg_df["mean_diff"] > 0).mean()))
    print("mean change_rate:", float(seg_df["change_rate"].mean()))


# =========================================================
# 2) 改票样本画像
# =========================================================

print("\n\n================ CHANGED DAYS PROFILE ================\n")

changed = valid[valid["change_flag"] == 1].copy()
unchanged = valid[valid["change_flag"] == 0].copy()

print("changed_days:", len(changed))
print("unchanged_days:", len(unchanged))

if len(changed) > 0:
    print("\n=== Changed days return diff summary ===")
    print_summary("ret_diff on changed days", changed["ret_diff"])

    cols_profile = [
        "top5_rank1_ret", "hard_remove_1_ret", "ret_diff",
        "top5_rank1_rank_score", "hard_remove_1_rank_score",
        "top5_rank1_p_cls", "hard_remove_1_p_cls",
        "top5_rank1_p_risk", "hard_remove_1_p_risk",
        "top5_rank1_rank_pct_all", "hard_remove_1_rank_pct_all",
        "top5_score_margin_1_2",
        "removed_ret", "removed_rank_score", "removed_p_cls", "removed_p_risk"
    ]

    print("\n=== Changed days profile means ===")
    print(changed[cols_profile].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Changed days profile medians ===")
    print(changed[cols_profile].median(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Good changed days (ret_diff > 0) ===")
    good_changed = changed[changed["ret_diff"] > 0].copy()
    print("n:", len(good_changed))
    if len(good_changed) > 0:
        print(good_changed[cols_profile].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Bad changed days (ret_diff <= 0) ===")
    bad_changed = changed[changed["ret_diff"] <= 0].copy()
    print("n:", len(bad_changed))
    if len(bad_changed) > 0:
        print(bad_changed[cols_profile].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 3) 被删票 vs 替代票 对比
# =========================================================

print("\n\n================ REMOVED vs REPLACEMENT ================\n")

if len(changed) > 0:
    compare_rows = []
    for idx, row in changed.iterrows():
        compare_rows.append({
            "date": idx,
            "removed_code": row["removed_code"],
            "replacement_code": row["hard_remove_1_code"],
            "removed_ret": row["removed_ret"],
            "replacement_ret": row["hard_remove_1_ret"],
            "ret_diff": row["ret_diff"],

            "removed_rank_score": row["removed_rank_score"],
            "replacement_rank_score": row["hard_remove_1_rank_score"],

            "removed_p_cls": row["removed_p_cls"],
            "replacement_p_cls": row["hard_remove_1_p_cls"],

            "removed_p_risk": row["removed_p_risk"],
            "replacement_p_risk": row["hard_remove_1_p_risk"],

            "removed_rank_pct_all": row["removed_rank_pct_all"],
            "replacement_rank_pct_all": row["hard_remove_1_rank_pct_all"],
        })

    cmp_df = pd.DataFrame(compare_rows)

    print("=== Pairwise mean comparison on changed days ===")
    out = {
        "removed_ret_mean": cmp_df["removed_ret"].mean(),
        "replacement_ret_mean": cmp_df["replacement_ret"].mean(),
        "ret_diff_mean": cmp_df["ret_diff"].mean(),

        "removed_rank_score_mean": cmp_df["removed_rank_score"].mean(),
        "replacement_rank_score_mean": cmp_df["replacement_rank_score"].mean(),

        "removed_p_cls_mean": cmp_df["removed_p_cls"].mean(),
        "replacement_p_cls_mean": cmp_df["replacement_p_cls"].mean(),

        "removed_p_risk_mean": cmp_df["removed_p_risk"].mean(),
        "replacement_p_risk_mean": cmp_df["replacement_p_risk"].mean(),

        "removed_rank_pct_mean": cmp_df["removed_rank_pct_all"].mean(),
        "replacement_rank_pct_mean": cmp_df["replacement_rank_pct_all"].mean(),
    }
    for k, v in out.items():
        print(f"{k}: {float(v):.6f}")

    print("\n=== Top positive changed days ===")
    print(
        cmp_df.sort_values("ret_diff", ascending=False)
              .head(15)
              .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\n=== Top negative changed days ===")
    print(
        cmp_df.sort_values("ret_diff", ascending=True)
              .head(15)
              .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )


# =========================================================
# 4) 哪些日期 risk 真正帮了忙
# =========================================================

print("\n\n================ DATES WHERE RISK HELPED ================\n")

if len(changed) > 0:
    helped = changed[changed["ret_diff"] > 0].copy()
    harmed = changed[changed["ret_diff"] <= 0].copy()

    print("helped_days:", len(helped))
    if len(helped) > 0:
        print(
            helped[
                [
                    "top5_rank1_code", "hard_remove_1_code",
                    "top5_rank1_ret", "hard_remove_1_ret", "ret_diff",
                    "top5_rank1_p_risk", "hard_remove_1_p_risk",
                    "top5_rank1_rank_score", "hard_remove_1_rank_score",
                    "top5_score_margin_1_2"
                ]
            ]
            .sort_values("ret_diff", ascending=False)
            .head(30)
            .to_string(float_format=lambda x: f"{x:.6f}")
        )

    print("\nharmed_days:", len(harmed))
    if len(harmed) > 0:
        print(
            harmed[
                [
                    "top5_rank1_code", "hard_remove_1_code",
                    "top5_rank1_ret", "hard_remove_1_ret", "ret_diff",
                    "top5_rank1_p_risk", "hard_remove_1_p_risk",
                    "top5_rank1_rank_score", "hard_remove_1_rank_score",
                    "top5_score_margin_1_2"
                ]
            ]
            .sort_values("ret_diff", ascending=True)
            .head(30)
            .to_string(float_format=lambda x: f"{x:.6f}")
        )


# =========================================================
# 5) 一个简单的“是否值得上真实回测”判断
# =========================================================

print("\n\n================ DECISION HINT ================\n")

base_stats = summarize_arr(valid["top5_rank1_ret"])
new_stats = summarize_arr(valid["hard_remove_1_ret"])
diff = (valid["hard_remove_1_ret"] - valid["top5_rank1_ret"]).dropna()

decision = {
    "mean_improved": new_stats["mean"] > base_stats["mean"],
    "maxdd_improved": new_stats["max_drawdown"] > base_stats["max_drawdown"],  # drawdown越大越好（更接近0）
    "loss8_improved": new_stats["big_loss_8pct_rate"] < base_stats["big_loss_8pct_rate"],
    "changed_days_positive": float((changed["ret_diff"] > 0).mean()) if len(changed) > 0 else np.nan,
    "segment_positive_ratio": float((seg_df["mean_diff"] > 0).mean()) if len(seg_df) > 0 else np.nan
}

for k, v in decision.items():
    print(k, ":", v)

print("\nInterpretation guide:")
print("- 若 mean_improved=True 且 maxdd_improved=True，说明 hard_remove_1 已经值得做真实回测")
print("- 若 changed_days_positive 接近 0.5 但 segment_positive_ratio 较高，说明它可能是少改票、但阶段性有帮助")
print("- 若 top negative changed days 集中在少数极端日期，可以继续做分钟回测确认是否是执行层噪声")