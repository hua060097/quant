import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json"    # 改成你的json路径
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


# =========================================================
# 从 JSON 重建三种策略
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

    # 1) 分类Top5
    df_top5 = df.sort_values("p_cls", ascending=False).head(5).copy()

    # 2) baseline: Top5 -> 排序Top1
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

    # 候选池风险排序
    if "p_risk" in df_top5.columns and df_top5["p_risk"].notna().sum() >= 2:
        # 风险最高的那只
        risk_desc = df_top5.sort_values(["p_risk", "rank_score"], ascending=[False, False]).reset_index(drop=True)
        highest_risk_row = risk_desc.iloc[0]

        row["top5_highest_risk_code"] = highest_risk_row.get("code", None)
        row["top5_highest_risk_ret"] = highest_risk_row.get("real_return", np.nan)
        row["top5_highest_risk_p_risk"] = highest_risk_row.get("p_risk", np.nan)

        # -------------------------------------------------
        # A) always remove 1
        # 永远删 risk 最高的 1 只
        # -------------------------------------------------
        keep_always = (
            df_top5.sort_values(["p_risk", "rank_score"], ascending=[True, False])
                  .head(len(df_top5) - 1)
                  .copy()
        )
        pick_always = keep_always.sort_values("rank_score", ascending=False).iloc[0]

        row["always_code"] = pick_always.get("code", None)
        row["always_ret"] = pick_always.get("real_return", np.nan)
        row["always_p_risk"] = pick_always.get("p_risk", np.nan)
        row["always_change"] = int(row["always_code"] != row["top5_rank1_code"])
        row["always_ret_diff"] = (
            float(row["always_ret"]) - float(row["top5_rank1_ret"])
            if pd.notna(row["always_ret"]) and pd.notna(row["top5_rank1_ret"])
            else np.nan
        )

        # -------------------------------------------------
        # B) guarded remove 1
        # 只有当 baseline Top1 本身就是风险最高那只，才删
        # -------------------------------------------------
        trigger = int(row["top5_rank1_code"] == row["top5_highest_risk_code"])
        row["guard_trigger"] = trigger

        if trigger == 1:
            keep_guarded = keep_always.copy()
            pick_guarded = keep_guarded.sort_values("rank_score", ascending=False).iloc[0]

            row["guarded_code"] = pick_guarded.get("code", None)
            row["guarded_ret"] = pick_guarded.get("real_return", np.nan)
            row["guarded_p_risk"] = pick_guarded.get("p_risk", np.nan)
        else:
            row["guarded_code"] = row["top5_rank1_code"]
            row["guarded_ret"] = row["top5_rank1_ret"]
            row["guarded_p_risk"] = row["top5_rank1_p_risk"]

        row["guarded_change"] = int(row["guarded_code"] != row["top5_rank1_code"])
        row["guarded_ret_diff"] = (
            float(row["guarded_ret"]) - float(row["top5_rank1_ret"])
            if pd.notna(row["guarded_ret"]) and pd.notna(row["top5_rank1_ret"])
            else np.nan
        )

    else:
        row["top5_highest_risk_code"] = None
        row["top5_highest_risk_ret"] = np.nan
        row["top5_highest_risk_p_risk"] = np.nan

        row["always_code"] = row["top5_rank1_code"]
        row["always_ret"] = row["top5_rank1_ret"]
        row["always_p_risk"] = row["top5_rank1_p_risk"]
        row["always_change"] = 0
        row["always_ret_diff"] = 0.0

        row["guard_trigger"] = 0
        row["guarded_code"] = row["top5_rank1_code"]
        row["guarded_ret"] = row["top5_rank1_ret"]
        row["guarded_p_risk"] = row["top5_rank1_p_risk"]
        row["guarded_change"] = 0
        row["guarded_ret_diff"] = 0.0

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
df_all = df_all.set_index("date")

valid = df_all[df_all["top5_rank1_ret"].notna()].copy()


# =========================================================
# 1) 三种策略总体比较
# =========================================================

summary = {}
strategies = {
    "top5_rank1": valid["top5_rank1_ret"],
    "top5_hard_remove_1_always": valid["always_ret"],
    "top5_hard_remove_1_guarded": valid["guarded_ret"],
}

for name, arr in strategies.items():
    summary[name] = print_summary(name, arr)

print("\n=== Key comparison table ===")
comp_cols = [
    "mean", "median", "win_rate",
    "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
    "cvar10", "max_drawdown", "sharpe"
]
comp = pd.DataFrame(summary).T[comp_cols]
print(comp.to_string(float_format=lambda x: f"{x:.6f}"))

diff_stats("always", valid["always_ret"], "top5_rank1", valid["top5_rank1_ret"])
diff_stats("guarded", valid["guarded_ret"], "top5_rank1", valid["top5_rank1_ret"])


# =========================================================
# 2) guarded 触发统计
# =========================================================

print("\n\n================ GUARDED TRIGGER STATS ================\n")
print("total_days:", len(valid))
print("guard_trigger_days:", int(valid["guard_trigger"].sum()))
print("guard_trigger_rate:", float(valid["guard_trigger"].mean()))

triggered = valid[valid["guard_trigger"] == 1].copy()
not_triggered = valid[valid["guard_trigger"] == 0].copy()

if len(triggered) > 0:
    print("\n=== Triggered days baseline vs guarded ===")
    print_summary("triggered baseline", triggered["top5_rank1_ret"])
    print_summary("triggered guarded", triggered["guarded_ret"])

    print("\n=== Triggered diff ===")
    d = triggered["guarded_ret"] - triggered["top5_rank1_ret"]
    print("count:", len(d))
    print("mean_diff:", float(d.mean()))
    print("median_diff:", float(d.median()))
    print("P(diff>0):", float((d > 0).mean()))
    print("P(diff<-0.03):", float((d < -0.03).mean()))
    print("P(diff<-0.05):", float((d < -0.05).mean()))

    print("\n=== Triggered days profile means ===")
    cols = [
        "top5_rank1_ret", "guarded_ret", "guarded_ret_diff",
        "top5_rank1_rank_score", "top5_rank1_p_cls", "top5_rank1_p_risk",
        "top5_highest_risk_p_risk"
    ]
    print(triggered[cols].mean(numeric_only=True).to_string(float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 3) 分段稳定性
# =========================================================

print("\n\n================ SEGMENT STABILITY ================\n")

segments = []
n = len(valid)

for start in range(0, n, SEGMENT_SIZE):
    seg = valid.iloc[start:start + SEGMENT_SIZE].copy()
    if len(seg) < max(8, SEGMENT_SIZE // 2):
        continue

    base = seg["top5_rank1_ret"].astype(float)
    always = seg["always_ret"].astype(float)
    guarded = seg["guarded_ret"].astype(float)

    diff_always = always - base
    diff_guarded = guarded - base

    segments.append({
        "start": seg.index[0],
        "end": seg.index[-1],
        "n": len(seg),

        "mean_base": float(base.mean()),
        "mean_always": float(always.mean()),
        "mean_guarded": float(guarded.mean()),

        "mean_diff_always": float(diff_always.mean()),
        "mean_diff_guarded": float(diff_guarded.mean()),

        "p_pos_always": float((diff_always > 0).mean()),
        "p_pos_guarded": float((diff_guarded > 0).mean()),

        "guard_trigger_rate": float(seg["guard_trigger"].mean()),
        "always_change_rate": float(seg["always_change"].mean()),
        "guarded_change_rate": float(seg["guarded_change"].mean()),
    })

seg_df = pd.DataFrame(segments)
print(seg_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if len(seg_df) > 0:
    print("\n=== Segment summary ===")
    print("P(segment mean_diff_always > 0):", float((seg_df["mean_diff_always"] > 0).mean()))
    print("P(segment mean_diff_guarded > 0):", float((seg_df["mean_diff_guarded"] > 0).mean()))
    print("mean guard_trigger_rate:", float(seg_df["guard_trigger_rate"].mean()))


# =========================================================
# 4) 两个时间段分析
# =========================================================

period_rows = []

for start_date, end_date, label in PERIODS:
    sub = valid.loc[(valid.index >= start_date) & (valid.index <= end_date)].copy()

    print("\n\n" + "=" * 70)
    print(f"{label}  [{start_date} ~ {end_date}]")
    print("=" * 70)

    s_base = print_summary(f"{label} top5_rank1", sub["top5_rank1_ret"])
    s_always = print_summary(f"{label} always", sub["always_ret"])
    s_guarded = print_summary(f"{label} guarded", sub["guarded_ret"])

    d_always = sub["always_ret"] - sub["top5_rank1_ret"]
    d_guarded = sub["guarded_ret"] - sub["top5_rank1_ret"]

    print(f"\n=== {label} always - baseline ===")
    print("mean_diff:", float(d_always.mean()))
    print("P(diff>0):", float((d_always > 0).mean()))
    print("change_rate:", float(sub["always_change"].mean()))

    print(f"\n=== {label} guarded - baseline ===")
    print("mean_diff:", float(d_guarded.mean()))
    print("P(diff>0):", float((d_guarded > 0).mean()))
    print("guard_trigger_rate:", float(sub["guard_trigger"].mean()))
    print("guarded_change_rate:", float(sub["guarded_change"].mean()))

    period_rows.append({
        "period": label,
        "start": start_date,
        "end": end_date,
        "n_days": len(sub),

        "base_mean": s_base["mean"] if s_base else np.nan,
        "always_mean": s_always["mean"] if s_always else np.nan,
        "guarded_mean": s_guarded["mean"] if s_guarded else np.nan,

        "always_mean_diff": float(d_always.mean()) if len(d_always) > 0 else np.nan,
        "guarded_mean_diff": float(d_guarded.mean()) if len(d_guarded) > 0 else np.nan,

        "always_change_rate": float(sub["always_change"].mean()) if len(sub) > 0 else np.nan,
        "guard_trigger_rate": float(sub["guard_trigger"].mean()) if len(sub) > 0 else np.nan,
        "guarded_change_rate": float(sub["guarded_change"].mean()) if len(sub) > 0 else np.nan,
    })

print("\n\n" + "=" * 70)
print("TWO-PERIOD SUMMARY TABLE")
print("=" * 70)
period_df = pd.DataFrame(period_rows)
print(period_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 5) 自动建议
# =========================================================

print("\n\n================ DECISION HINT ================\n")

base_stats = summarize_arr(valid["top5_rank1_ret"])
always_stats = summarize_arr(valid["always_ret"])
guarded_stats = summarize_arr(valid["guarded_ret"])

print("always_mean_improved:", always_stats["mean"] > base_stats["mean"])
print("always_maxdd_improved:", always_stats["max_drawdown"] > base_stats["max_drawdown"])
print("guarded_mean_improved:", guarded_stats["mean"] > base_stats["mean"])
print("guarded_maxdd_improved:", guarded_stats["max_drawdown"] > base_stats["max_drawdown"])

print("\nInterpretation guide:")
print("- 若 guarded 在 mean / maxdd 上都优于 baseline，且接近 always，则 guarded 更抗过拟合")
print("- 若 guarded 明显弱于 always，但仍优于 baseline，可优先上 guarded 做真实回测")
print("- 若 guarded 与 baseline 几乎无差，而 always 明显更强，则说明触发条件过于保守")