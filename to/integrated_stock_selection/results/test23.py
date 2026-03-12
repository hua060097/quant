import json
import numpy as np
import pandas as pd

JSON_PATH = "daily_gate_stats_Window_3_gateN80_20260312_001922.json"   # 改成你的json文件路径


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


def summarize(name, arr):
    x = pd.Series(arr).dropna().astype(float)
    if len(x) == 0:
        print(f"\n=== {name} ===")
        print("EMPTY")
        return None

    out = {
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

    print(f"\n=== {name} ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    return out


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
# 从 JSON 重建策略
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

# 你真正关心的是：
# 1) 原 json baseline = serial_top1（通常 gate_n=80）
# 2) 重建 Top5 + 排序Top1
# 3) Top5 + hard filter
# 4) Top5 + soft penalty
#
# 所以我们全部都在同一个日期层面重建

for date_str, rec in data.items():
    row = {"date": date_str}

    # -----------------------------------------------------
    # A. json里原baseline（通常 gate_n=80 的 serial_top1）
    # -----------------------------------------------------
    serial_top1 = rec.get("serial_top1", {})
    row["json_baseline_ret"] = serial_top1.get("real_return", None)
    row["json_baseline_p_risk"] = serial_top1.get("p_risk", None)

    # -----------------------------------------------------
    # B. 从 cls_topn 重建 “分类Top5 + 排序Top1”
    # -----------------------------------------------------
    cls_topn = rec.get("cls_topn", [])
    if not isinstance(cls_topn, list) or len(cls_topn) == 0:
        rows.append(row)
        continue

    df = pd.DataFrame(cls_topn)

    # 安全转换
    for c in ["p_cls", "rank_score", "real_return", "p_risk"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 先按分类模型 p_cls 取 Top5
    if "p_cls" not in df.columns or "rank_score" not in df.columns or "real_return" not in df.columns:
        rows.append(row)
        continue

    df_cls5 = df.sort_values("p_cls", ascending=False).head(5).copy()

    # baseline: Top5 里按 rank_score 选Top1
    df_cls5_rank = df_cls5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    top5_rank1 = df_cls5_rank.iloc[0]

    row["top5_rank1_ret"] = top5_rank1["real_return"]
    row["top5_rank1_p_risk"] = top5_rank1["p_risk"] if "p_risk" in top5_rank1.index else np.nan
    row["top5_rank1_code"] = top5_rank1["code"] if "code" in top5_rank1.index else None

    # -----------------------------------------------------
    # C. hard filter on Top5
    # 删掉 risk 最高的 n 只，再按 rank_score 选Top1
    # -----------------------------------------------------
    if "p_risk" in df_cls5.columns:
        for n in [1, 2]:
            col_ret = f"top5_hard_remove_{n}_ret"
            col_risk = f"top5_hard_remove_{n}_p_risk"
            col_code = f"top5_hard_remove_{n}_code"

            if len(df_cls5) <= n:
                row[col_ret] = np.nan
                row[col_risk] = np.nan
                row[col_code] = None
                continue

            tmp = df_cls5.copy()
            tmp_keep = tmp.sort_values(["p_risk", "rank_score"], ascending=[True, False]).head(len(tmp) - n).copy()
            tmp_pick = tmp_keep.sort_values("rank_score", ascending=False).iloc[0]

            row[col_ret] = tmp_pick["real_return"]
            row[col_risk] = tmp_pick["p_risk"]
            row[col_code] = tmp_pick["code"] if "code" in tmp_pick.index else None

        # -------------------------------------------------
        # D. soft penalty on Top5
        # final = rank_score - lambda * z(p_risk)
        # -------------------------------------------------
        risk_z = zscore(df_cls5["p_risk"])
        df_soft = df_cls5.copy()
        df_soft["p_risk_z"] = risk_z.values

        for lam in [0.05, 0.10, 0.20, 0.50]:
            final_col = f"final_{lam:.2f}"
            ret_col = f"top5_soft_{lam:.2f}_ret"
            risk_col = f"top5_soft_{lam:.2f}_p_risk"
            code_col = f"top5_soft_{lam:.2f}_code"

            df_soft[final_col] = df_soft["rank_score"] - lam * df_soft["p_risk_z"]
            tmp_pick = df_soft.sort_values(final_col, ascending=False).iloc[0]

            row[ret_col] = tmp_pick["real_return"]
            row[risk_col] = tmp_pick["p_risk"]
            row[code_col] = tmp_pick["code"] if "code" in tmp_pick.index else None
    else:
        for n in [1, 2]:
            row[f"top5_hard_remove_{n}_ret"] = np.nan
            row[f"top5_hard_remove_{n}_p_risk"] = np.nan
            row[f"top5_hard_remove_{n}_code"] = None
        for lam in [0.05, 0.10, 0.20, 0.50]:
            row[f"top5_soft_{lam:.2f}_ret"] = np.nan
            row[f"top5_soft_{lam:.2f}_p_risk"] = np.nan
            row[f"top5_soft_{lam:.2f}_code"] = None

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True).set_index("date")

# =========================================================
# 1) 先比较：json baseline vs 真正的 Top5+排序Top1 baseline
# =========================================================

summary = {}

strategies = [
    "json_baseline_ret",
    "top5_rank1_ret",
    "top5_hard_remove_1_ret",
    "top5_hard_remove_2_ret",
    "top5_soft_0.05_ret",
    "top5_soft_0.10_ret",
    "top5_soft_0.20_ret",
    "top5_soft_0.50_ret",
]

for col in strategies:
    summary[col] = summarize(col, df_all[col])

print("\n=== Key comparison table ===")
comp_cols = [
    "mean", "median", "win_rate",
    "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
    "cvar10", "max_drawdown", "sharpe"
]
comp = pd.DataFrame(summary).T[comp_cols]
print(comp.to_string(float_format=lambda x: f"{x:.6f}"))

# 相对 Top5+排序Top1 baseline 做差值
for col in strategies[2:]:
    diff_stats(col, df_all[col], "top5_rank1_ret", df_all["top5_rank1_ret"])

# json里的 baseline 和我们重建的 top5 baseline 也对比一下
diff_stats("top5_rank1_ret", df_all["top5_rank1_ret"], "json_baseline_ret", df_all["json_baseline_ret"])


# =========================================================
# 2) risk 是否抓住了 baseline 的大亏日
# 这里重点分析你真正关心的 Top5+排序Top1 baseline
# =========================================================

def analyze_big_loss_capture(df, ret_col, risk_col, name):
    sub = df[[ret_col, risk_col]].dropna().copy()
    if len(sub) == 0:
        print(f"\n=== Big-loss capture: {name} ===\nEMPTY")
        return

    sub["bad5"] = (sub[ret_col] <= -0.05).astype(int)
    sub["bad8"] = (sub[ret_col] <= -0.08).astype(int)

    print(f"\n=== Big-loss capture: {name} ===")
    print("count:", len(sub))

    print("\n[bad5]")
    print("bad5 rate:", float(sub["bad5"].mean()))
    print("risk mean on bad5:", float(sub.loc[sub["bad5"] == 1, risk_col].mean()) if (sub["bad5"] == 1).sum() > 0 else np.nan)
    print("risk mean on non-bad5:", float(sub.loc[sub["bad5"] == 0, risk_col].mean()) if (sub["bad5"] == 0).sum() > 0 else np.nan)
    print("risk median on bad5:", float(sub.loc[sub["bad5"] == 1, risk_col].median()) if (sub["bad5"] == 1).sum() > 0 else np.nan)
    print("risk median on non-bad5:", float(sub.loc[sub["bad5"] == 0, risk_col].median()) if (sub["bad5"] == 0).sum() > 0 else np.nan)

    print("\n[bad8]")
    print("bad8 rate:", float(sub["bad8"].mean()))
    print("risk mean on bad8:", float(sub.loc[sub["bad8"] == 1, risk_col].mean()) if (sub["bad8"] == 1).sum() > 0 else np.nan)
    print("risk mean on non-bad8:", float(sub.loc[sub["bad8"] == 0, risk_col].mean()) if (sub["bad8"] == 0).sum() > 0 else np.nan)
    print("risk median on bad8:", float(sub.loc[sub["bad8"] == 1, risk_col].median()) if (sub["bad8"] == 1).sum() > 0 else np.nan)
    print("risk median on non-bad8:", float(sub.loc[sub["bad8"] == 0, risk_col].median()) if (sub["bad8"] == 0).sum() > 0 else np.nan)

    # 按 risk 分位桶看坏日浓度
    try:
        sub["risk_q"] = pd.qcut(sub[risk_col], 5, duplicates="drop")
        g = sub.groupby("risk_q").agg(
            count=(ret_col, "count"),
            mean_ret=(ret_col, "mean"),
            median_ret=(ret_col, "median"),
            bad5_rate=("bad5", "mean"),
            bad8_rate=("bad8", "mean"),
        )
        print("\n[risk quantile buckets]")
        print(g.to_string(float_format=lambda x: f"{x:.6f}"))
    except Exception as e:
        print("qcut failed:", e)


# 分析 Top5 baseline 本身的 risk 捕获能力
analyze_big_loss_capture(df_all, "top5_rank1_ret", "top5_rank1_p_risk", "Top5+Rank1 baseline")

# 分析 json baseline 本身的 risk 捕获能力
analyze_big_loss_capture(df_all, "json_baseline_ret", "json_baseline_p_risk", "JSON serial_top1 baseline")


# =========================================================
# 3) risk 是否真的改变了 Top5 baseline 的选股
# =========================================================

def analyze_pick_change(df, base_code_col, new_code_col, base_ret_col, new_ret_col, name):
    tmp = df[[base_code_col, new_code_col, base_ret_col, new_ret_col]].dropna().copy()
    if len(tmp) == 0:
        print(f"\n=== Pick change: {name} ===\nEMPTY")
        return

    changed = (tmp[base_code_col] != tmp[new_code_col])
    diff = tmp[new_ret_col] - tmp[base_ret_col]

    print(f"\n=== Pick change: {name} ===")
    print("count:", len(tmp))
    print("change_rate:", float(changed.mean()))
    print("mean_diff_all:", float(diff.mean()))
    if changed.sum() > 0:
        print("mean_diff_on_changed_days:", float(diff[changed].mean()))
        print("median_diff_on_changed_days:", float(diff[changed].median()))
        print("P(diff>0 | changed):", float((diff[changed] > 0).mean()))
        print("P(diff<-0.03 | changed):", float((diff[changed] < -0.03).mean()))
        print("P(diff<-0.05 | changed):", float((diff[changed] < -0.05).mean()))
    else:
        print("No changed days.")


analyze_pick_change(
    df_all,
    "top5_rank1_code", "top5_hard_remove_1_code",
    "top5_rank1_ret", "top5_hard_remove_1_ret",
    "Top5 hard remove 1"
)

analyze_pick_change(
    df_all,
    "top5_rank1_code", "top5_hard_remove_2_code",
    "top5_rank1_ret", "top5_hard_remove_2_ret",
    "Top5 hard remove 2"
)

for lam in ["0.05", "0.10", "0.20", "0.50"]:
    analyze_pick_change(
        df_all,
        "top5_rank1_code", f"top5_soft_{lam}_code",
        "top5_rank1_ret", f"top5_soft_{lam}_ret",
        f"Top5 soft {lam}"
    )


# =========================================================
# 4) 自动打分：以 Top5+排序Top1 为主策略 baseline
#    这个打分仍然偏“均衡”，你可以自己调
# =========================================================

def strategy_score(row):
    return (
        row["mean"]
        + 0.6 * row["median"]
        + 0.05 * row["sharpe"]
        - 0.8 * row["big_loss_5pct_rate"]
        - 0.3 * abs(row["max_drawdown"])
    )

score_table = comp.copy()
score_table["score"] = score_table.apply(strategy_score, axis=1)
score_table = score_table.sort_values("score", ascending=False)

print("\n=== Ranking by composite score ===")
print(score_table[comp_cols + ["score"]].to_string(float_format=lambda x: f"{x:.6f}"))