import json
import math
import numpy as np
import pandas as pd

# =========================================================
# 配置
# =========================================================
JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_015208.json"   # 改成你的JSON路径
USE_LOG_RETURN_NAV = False                 # False: 用简单收益累计；True: 用log return近似
DATE_FILTER_START = None                   # 例如 "20250901"
DATE_FILTER_END = None                     # 例如 "20260228"

# 你当前最值得优先研究的两个实时信号
FEATURE_MAIN = "serial_rank_pct_all"
FEATURE_AUX = "conf_pred_x_cls"

# 单因子分位仓位
SINGLE_QUANTILES = (0.30, 0.70)
SINGLE_POSITIONS = (0.4, 0.7, 1.0)

# 双因子四档仓位
DOUBLE_POS_MAP = {
    0: 0.3,   # 两个都低
    1: 0.5,   # 一个中/一个低
    2: 0.8,   # 两个中 或 一个高一个低
    3: 0.8,   # 高+中
    4: 1.0,   # 高+高
}

# 可选：再测试一个“极弱跳过”的规则
SKIP_RULES = [
    # name, condition type
    # "none" 代表不跳过
    ("no_skip", None),
    ("skip_weak_both", "both_low"),
    ("skip_main_very_low", "main_bottom10"),
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


def build_nav(returns, use_log=False):
    x = pd.Series(returns).fillna(0.0).astype(float)
    if use_log:
        nav = np.exp(np.log1p(x).cumsum())
    else:
        nav = (1.0 + x).cumprod()
    return nav


def annualized_sharpe(ret, periods_per_year=252):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(periods_per_year))


def summarize_strategy(name, df, ret_col):
    x = df[ret_col].dropna().astype(float)
    if len(x) == 0:
        return None

    nav = build_nav(x, use_log=USE_LOG_RETURN_NAV)
    peak = nav.cummax()
    dd = nav / peak - 1.0

    out = {
        "strategy": name,
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "win_rate": float((x > 0).mean()),
        "big_loss_3pct_rate": float((x <= -0.03).mean()),
        "big_loss_5pct_rate": float((x <= -0.05).mean()),
        "big_loss_8pct_rate": float((x <= -0.08).mean()),
        "cvar10": float(x[x <= np.quantile(x, 0.10)].mean()) if len(x) >= 10 else np.nan,
        "max_drawdown": float(dd.min()),
        "sharpe": annualized_sharpe(x),
        "final_nav": float(nav.iloc[-1]),
        "p5": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }
    return out


def print_summary_table(rows, title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if not rows:
        print("EMPTY")
        return
    df = pd.DataFrame(rows)
    cols = [
        "strategy", "count", "mean", "median", "win_rate",
        "big_loss_3pct_rate", "big_loss_5pct_rate", "big_loss_8pct_rate",
        "cvar10", "max_drawdown", "sharpe", "final_nav"
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def bucket_table(df, feature_col, ret_col="ret", n_buckets=5):
    sub = df[[feature_col, ret_col]].dropna().copy()
    if len(sub) < n_buckets * 5:
        return None
    try:
        sub["bucket"] = pd.qcut(sub[feature_col], q=n_buckets, duplicates="drop")
    except Exception:
        return None
    g = sub.groupby("bucket", observed=False)[ret_col]
    out = g.agg(["count", "mean", "median", "std"]).copy()
    out["win_rate"] = g.apply(lambda x: (x > 0).mean())
    out["loss3_rate"] = g.apply(lambda x: (x <= -0.03).mean())
    out["loss5_rate"] = g.apply(lambda x: (x <= -0.05).mean())
    out["loss8_rate"] = g.apply(lambda x: (x <= -0.08).mean())
    out["max_drawdown"] = g.apply(max_drawdown_from_returns)
    return out


# =========================================================
# 读取并重建“分类Top5 + 排序Top1”
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

rows = []

for date_str, rec in raw.items():
    if DATE_FILTER_START and date_str < DATE_FILTER_START:
        continue
    if DATE_FILTER_END and date_str > DATE_FILTER_END:
        continue

    cls_topn = rec.get("cls_topn", [])
    serial_top1 = rec.get("serial_top1", {})
    rank_all_top1 = rec.get("rank_all_top1", {})

    if len(cls_topn) < 5:
        continue

    df = pd.DataFrame(cls_topn).copy()
    for c in ["p_cls", "p_cls_pct", "rank_score", "rank_pct_all", "real_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 严格按真实交易链路重建：
    # 1) 分类Top5
    df_cls_top5 = df.sort_values("p_cls", ascending=False).head(5).reset_index(drop=True)
    # 2) Top5里按排序选Top1
    df_rank_in_top5 = df_cls_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    chosen = df_rank_in_top5.iloc[0]

    row = {
        "date": date_str,
        "ret": float(chosen["real_return"]),
        "code": chosen["code"],
    }

    # 直接来自 serial_top1 的当日可见字段
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

    # 全市场排序Top1信息，可做参照
    row["rank_all_top1_pred"] = pd.to_numeric(rank_all_top1.get("predicted_return", np.nan), errors="coerce")
    row["rank_all_top1_ret"] = pd.to_numeric(rank_all_top1.get("real_return", np.nan), errors="coerce")

    # Top5结构
    row["top5_cls_gap_1_2"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[1]["p_cls"])
    row["top5_cls_gap_1_5"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[4]["p_cls"])
    row["top5_cls_std"] = float(df_cls_top5["p_cls"].std(ddof=1))

    row["top5_rank_gap_1_2"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1]["rank_score"])
    row["top5_rank_gap_1_5"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[4]["rank_score"])
    row["top5_rank_gap_1_mean_rest"] = float(
        df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1:]["rank_score"].mean()
    )
    row["top5_rank_std"] = float(df_rank_in_top5["rank_score"].std(ddof=1))

    # chosen 在Top5中的分类位置
    chosen_code = chosen["code"]
    df_cls_top5 = df_cls_top5.copy()
    df_cls_top5["cls_rank_in_top5"] = np.arange(1, len(df_cls_top5) + 1)
    match = df_cls_top5[df_cls_top5["code"] == chosen_code]
    row["chosen_cls_rank_in_top5"] = float(match.iloc[0]["cls_rank_in_top5"]) if len(match) else np.nan
    row["chosen_cls_rank_pct_in_top5"] = (
        1.0 - (row["chosen_cls_rank_in_top5"] - 1) / 4.0 if pd.notna(row["chosen_cls_rank_in_top5"]) else np.nan
    )

    # chosen在Top5内部相对突出程度
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

    # 组合信号
    row["conf_pred_x_cls"] = row["serial_pred"] * row["serial_p_cls"]
    row["conf_rankgap_x_rankpct"] = row["serial_score_margin_in_cls"] * row["serial_rank_pct_all"]
    row["conf_rankz_x_pclsz"] = (
        row["chosen_rank_z_in_top5"] * row["chosen_pcls_z_in_top5"]
        if pd.notna(row["chosen_rank_z_in_top5"]) and pd.notna(row["chosen_pcls_z_in_top5"])
        else np.nan
    )

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

print(f"valid_days: {len(df_all)}")
base_summary = summarize_strategy("baseline_top5_rank1", df_all, "ret")
print_summary_table([base_summary], "BASELINE")


# =========================================================
# 先输出核心因子分桶，方便人工确认
# =========================================================

core_features = [
    "serial_rank_pct_all",
    "conf_pred_x_cls",
    "top5_cls_std",
    "top5_rank_gap_1_5",
    "serial_pred",
    "serial_p_cls",
]

print("\n" + "#" * 120)
print("CORE FEATURE BUCKET DIAGNOSTICS")
print("#" * 120)

for feat in core_features:
    if feat not in df_all.columns:
        continue
    print("\n" + "-" * 100)
    print(f"FEATURE: {feat}")
    tbl = bucket_table(df_all, feat, ret_col="ret", n_buckets=5)
    if tbl is None:
        print("SKIP")
    else:
        print(tbl.to_string(float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 单因子三档仓位
# =========================================================

def single_factor_positions(series, q30, q70, higher_is_better=True, positions=(0.4, 0.7, 1.0)):
    low_pos, mid_pos, high_pos = positions
    pos = []
    for x in series:
        if pd.isna(x):
            pos.append(np.nan)
            continue
        if higher_is_better:
            if x <= q30:
                pos.append(low_pos)
            elif x <= q70:
                pos.append(mid_pos)
            else:
                pos.append(high_pos)
        else:
            if x <= q30:
                pos.append(high_pos)
            elif x <= q70:
                pos.append(mid_pos)
            else:
                pos.append(low_pos)
    return pd.Series(pos, index=series.index)


single_factor_specs = {
    "serial_rank_pct_all": True,
    "conf_pred_x_cls": True,
    "top5_cls_std": True,
    "top5_rank_gap_1_5": True,
    "serial_pred": True,
    "serial_p_cls": False,  # 你这次结果里它更像反向
}

single_rows = []
single_df_map = {}

for feat, higher_is_better in single_factor_specs.items():
    sub = df_all[[feat, "ret", "date", "code"]].dropna().copy()
    if len(sub) < 30:
        continue

    q30 = sub[feat].quantile(SINGLE_QUANTILES[0])
    q70 = sub[feat].quantile(SINGLE_QUANTILES[1])

    sub["position"] = single_factor_positions(
        sub[feat], q30, q70, higher_is_better=higher_is_better, positions=SINGLE_POSITIONS
    )
    sub["ret_pos"] = sub["ret"] * sub["position"]
    sub["skip"] = 0

    s = summarize_strategy(f"single::{feat}", sub, "ret_pos")
    s["feature"] = feat
    s["q30"] = float(q30)
    s["q70"] = float(q70)
    s["higher_is_better"] = higher_is_better
    s["avg_position"] = float(sub["position"].mean())
    s["skip_rate"] = 0.0
    s["mean_diff_vs_base"] = s["mean"] - base_summary["mean"]
    s["mdd_improve_vs_base"] = s["max_drawdown"] - base_summary["max_drawdown"]
    s["loss5_improve_vs_base"] = s["big_loss_5pct_rate"] - base_summary["big_loss_5pct_rate"]
    s["loss8_improve_vs_base"] = s["big_loss_8pct_rate"] - base_summary["big_loss_8pct_rate"]
    single_rows.append(s)
    single_df_map[feat] = sub

print_summary_table(single_rows, "SINGLE-FACTOR POSITION BACKTEST")

if single_rows:
    single_rank = pd.DataFrame(single_rows).sort_values(
        ["mean_diff_vs_base", "mdd_improve_vs_base"], ascending=[False, False]
    )
    print("\n" + "#" * 120)
    print("SINGLE-FACTOR RANKING")
    print("#" * 120)
    cols = [
        "feature", "mean", "mean_diff_vs_base", "max_drawdown", "mdd_improve_vs_base",
        "big_loss_5pct_rate", "loss5_improve_vs_base", "big_loss_8pct_rate", "loss8_improve_vs_base",
        "avg_position", "q30", "q70", "higher_is_better"
    ]
    print(single_rank[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 双因子仓位
# =========================================================

def level_by_quantile(x, q30, q70):
    if pd.isna(x):
        return np.nan
    if x <= q30:
        return 0
    elif x <= q70:
        return 1
    else:
        return 2

def double_factor_positions(df, f1, f2):
    q1_30, q1_70 = df[f1].quantile(0.30), df[f1].quantile(0.70)
    q2_30, q2_70 = df[f2].quantile(0.30), df[f2].quantile(0.70)

    lv1 = df[f1].apply(lambda x: level_by_quantile(x, q1_30, q1_70))
    lv2 = df[f2].apply(lambda x: level_by_quantile(x, q2_30, q2_70))
    lv_sum = lv1 + lv2
    pos = lv_sum.map(DOUBLE_POS_MAP)

    return pos, (q1_30, q1_70, q2_30, q2_70), lv1, lv2

double_candidates = [
    ("serial_rank_pct_all", "conf_pred_x_cls"),
    ("serial_rank_pct_all", "top5_cls_std"),
    ("conf_pred_x_cls", "top5_cls_std"),
    ("serial_rank_pct_all", "top5_rank_gap_1_5"),
]

double_rows = []
double_df_map = {}

for f1, f2 in double_candidates:
    sub = df_all[[f1, f2, "ret", "date", "code"]].dropna().copy()
    if len(sub) < 30:
        continue

    pos, qs, lv1, lv2 = double_factor_positions(sub, f1, f2)
    sub["lv1"] = lv1
    sub["lv2"] = lv2
    sub["position"] = pos
    sub["ret_pos"] = sub["ret"] * sub["position"]
    sub["skip"] = 0

    s = summarize_strategy(f"double::{f1}+{f2}", sub, "ret_pos")
    s["feature_pair"] = f"{f1}+{f2}"
    s["avg_position"] = float(sub["position"].mean())
    s["skip_rate"] = 0.0
    s["mean_diff_vs_base"] = s["mean"] - base_summary["mean"]
    s["mdd_improve_vs_base"] = s["max_drawdown"] - base_summary["max_drawdown"]
    s["loss5_improve_vs_base"] = s["big_loss_5pct_rate"] - base_summary["big_loss_5pct_rate"]
    s["loss8_improve_vs_base"] = s["big_loss_8pct_rate"] - base_summary["big_loss_8pct_rate"]
    s["f1_q30"], s["f1_q70"], s["f2_q30"], s["f2_q70"] = [float(x) for x in qs]
    double_rows.append(s)
    double_df_map[f"{f1}+{f2}"] = sub

print_summary_table(double_rows, "DOUBLE-FACTOR POSITION BACKTEST")

if double_rows:
    double_rank = pd.DataFrame(double_rows).sort_values(
        ["mean_diff_vs_base", "mdd_improve_vs_base"], ascending=[False, False]
    )
    print("\n" + "#" * 120)
    print("DOUBLE-FACTOR RANKING")
    print("#" * 120)
    cols = [
        "feature_pair", "mean", "mean_diff_vs_base", "max_drawdown", "mdd_improve_vs_base",
        "big_loss_5pct_rate", "loss5_improve_vs_base", "big_loss_8pct_rate", "loss8_improve_vs_base",
        "avg_position", "f1_q30", "f1_q70", "f2_q30", "f2_q70"
    ]
    print(double_rank[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 带 skip 的双因子策略
# 以最优主组合为核心：serial_rank_pct_all + conf_pred_x_cls
# =========================================================

best_pair = (FEATURE_MAIN, FEATURE_AUX)
if FEATURE_MAIN in df_all.columns and FEATURE_AUX in df_all.columns:
    sub = df_all[[FEATURE_MAIN, FEATURE_AUX, "ret", "date", "code"]].dropna().copy()
    pos, qs, lv1, lv2 = double_factor_positions(sub, FEATURE_MAIN, FEATURE_AUX)
    sub["lv1"] = lv1
    sub["lv2"] = lv2
    sub["lv_sum"] = sub["lv1"] + sub["lv2"]
    sub["base_position"] = pos

    q1_10 = sub[FEATURE_MAIN].quantile(0.10)

    skip_rows = []

    for rule_name, rule_type in SKIP_RULES:
        tmp = sub.copy()

        if rule_type is None:
            tmp["skip"] = 0
        elif rule_type == "both_low":
            tmp["skip"] = ((tmp["lv1"] == 0) & (tmp["lv2"] == 0)).astype(int)
        elif rule_type == "main_bottom10":
            tmp["skip"] = (tmp[FEATURE_MAIN] <= q1_10).astype(int)
        else:
            raise ValueError(f"Unknown skip rule: {rule_type}")

        tmp["position"] = np.where(tmp["skip"] == 1, 0.0, tmp["base_position"])
        tmp["ret_pos"] = tmp["ret"] * tmp["position"]

        s = summarize_strategy(f"skip::{rule_name}", tmp, "ret_pos")
        s["skip_rule"] = rule_name
        s["avg_position"] = float(tmp["position"].mean())
        s["skip_rate"] = float(tmp["skip"].mean())
        s["mean_diff_vs_base"] = s["mean"] - base_summary["mean"]
        s["mdd_improve_vs_base"] = s["max_drawdown"] - base_summary["max_drawdown"]
        s["loss5_improve_vs_base"] = s["big_loss_5pct_rate"] - base_summary["big_loss_5pct_rate"]
        s["loss8_improve_vs_base"] = s["big_loss_8pct_rate"] - base_summary["big_loss_8pct_rate"]
        skip_rows.append(s)

    print_summary_table(skip_rows, f"DOUBLE-FACTOR + SKIP BACKTEST ({FEATURE_MAIN} + {FEATURE_AUX})")

    if skip_rows:
        skip_rank = pd.DataFrame(skip_rows).sort_values(
            ["mean_diff_vs_base", "mdd_improve_vs_base"], ascending=[False, False]
        )
        print("\n" + "#" * 120)
        print("SKIP RULE RANKING")
        print("#" * 120)
        cols = [
            "skip_rule", "mean", "mean_diff_vs_base", "max_drawdown", "mdd_improve_vs_base",
            "big_loss_5pct_rate", "loss5_improve_vs_base", "big_loss_8pct_rate", "loss8_improve_vs_base",
            "avg_position", "skip_rate"
        ]
        print(skip_rank[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# =========================================================
# 输出最推荐的“可直接抄到实盘规则”的阈值参考
# =========================================================

print("\n" + "#" * 120)
print("PRACTICAL THRESHOLD HINTS")
print("#" * 120)

for feat in ["serial_rank_pct_all", "conf_pred_x_cls", "top5_cls_std", "top5_rank_gap_1_5"]:
    if feat in df_all.columns:
        q10 = df_all[feat].quantile(0.10)
        q30 = df_all[feat].quantile(0.30)
        q50 = df_all[feat].quantile(0.50)
        q70 = df_all[feat].quantile(0.70)
        q90 = df_all[feat].quantile(0.90)
        print(
            f"{feat}: "
            f"q10={q10:.6f}, q30={q30:.6f}, q50={q50:.6f}, q70={q70:.6f}, q90={q90:.6f}"
        )


print("\nDone.")