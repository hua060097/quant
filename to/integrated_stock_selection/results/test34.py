import json
import math
import numpy as np
import pandas as pd

# =========================================================
# 配置
# =========================================================
JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_015208.json"   # 改成你的JSON文件路径
DATE_FILTER_START = None                   # 例如 "20250605"
DATE_FILTER_END = None                     # 例如 "20260302"

# 切分方式
USE_HALF_SPLIT = True                      # True: 前后半段；False: 用指定日期切分
MANUAL_SPLIT_DATE = None                   # 例如 "20250913"，当 USE_HALF_SPLIT=False 时生效

# rolling窗口
ROLLING_WIN = 40

# =========================================================
# 基础函数
# =========================================================

def max_drawdown_from_returns(ret):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) == 0:
        return np.nan
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def annualized_sharpe(ret, periods_per_year=252):
    x = pd.Series(ret).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(mu / sd * np.sqrt(periods_per_year))


def summarize_arr(arr):
    x = pd.Series(arr).dropna().astype(float)
    if len(x) == 0:
        return None
    q10 = np.quantile(x, 0.10) if len(x) >= 10 else np.nan
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "win_rate": float((x > 0).mean()),
        "big_loss_3pct_rate": float((x <= -0.03).mean()),
        "big_loss_5pct_rate": float((x <= -0.05).mean()),
        "big_loss_8pct_rate": float((x <= -0.08).mean()),
        "cvar10": float(x[x <= q10].mean()) if len(x) >= 10 else np.nan,
        "max_drawdown": float(max_drawdown_from_returns(x)),
        "sharpe": float(annualized_sharpe(x)),
        "p5": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
    }


def print_summary(title, arr):
    s = summarize_arr(arr)
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if s is None:
        print("EMPTY")
        return None
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    return s


def bucket_stats(df, feature_col, ret_col="ret", q=5, title=None):
    sub = df[[feature_col, ret_col]].dropna().copy()
    if len(sub) < q * 5:
        print(f"\n[SKIP] {title or feature_col}: not enough samples")
        return None

    try:
        sub["bucket"] = pd.qcut(sub[feature_col], q=q, duplicates="drop")
    except Exception as e:
        print(f"\n[SKIP] {title or feature_col}: qcut failed: {e}")
        return None

    g = sub.groupby("bucket", observed=False)[ret_col]
    out = g.agg(["count", "mean", "median", "std"]).copy()
    out["win_rate"] = g.apply(lambda x: (x > 0).mean())
    out["loss3_rate"] = g.apply(lambda x: (x <= -0.03).mean())
    out["loss5_rate"] = g.apply(lambda x: (x <= -0.05).mean())
    out["loss8_rate"] = g.apply(lambda x: (x <= -0.08).mean())
    out["max_drawdown"] = g.apply(max_drawdown_from_returns)

    print("\n" + "-" * 100)
    print(title or feature_col)
    print(out.to_string(float_format=lambda x: f"{x:.6f}"))

    means = out["mean"].values
    monotonic_up = all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
    monotonic_down = all(means[i] >= means[i + 1] - 1e-12 for i in range(len(means) - 1))
    spread = float(means[-1] - means[0])

    if monotonic_up:
        hint = "LIKELY_POSITIVE_MONOTONIC"
    elif monotonic_down:
        hint = "LIKELY_NEGATIVE_MONOTONIC"
    else:
        hint = "MIXED"

    print(f"monotonic_hint: {hint}")
    print(f"spread_mean(top-bottom): {spread:.6f}")
    return out


def rolling_future_mean(x, win):
    return pd.Series(x).rolling(win).mean()


def safe_pct_rank_desc(values):
    s = pd.Series(values).rank(ascending=False, method="average")
    n = len(values)
    if n <= 1:
        return pd.Series([1.0] * n)
    return 1.0 - (s - 1.0) / n


# =========================================================
# 读取JSON并重建交易链路
# =========================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

day_rows = []
cls_topn_rows = []

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

    for c in ["p_cls", "p_cls_pct", "rank_score", "rank_pos_all", "rank_pct_all", "real_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 按分类选Top5
    df_cls_top5 = df.sort_values("p_cls", ascending=False).head(5).reset_index(drop=True)

    # 在Top5中按排序选Top1
    df_rank_in_top5 = df_cls_top5.sort_values("rank_score", ascending=False).reset_index(drop=True)
    chosen = df_rank_in_top5.iloc[0]

    # Top5明细
    df_cls_top5 = df_cls_top5.copy()
    df_cls_top5["date"] = date_str
    df_cls_top5["cls_rank_in_top5"] = np.arange(1, len(df_cls_top5) + 1)
    df_cls_top5["rank_rank_in_top5"] = df_cls_top5["rank_score"].rank(ascending=False, method="min")
    cls_topn_rows.append(df_cls_top5)

    # 日级字段
    row = {
        "date": date_str,
        "universe_size": pd.to_numeric(rec.get("universe_size", np.nan), errors="coerce"),
        "gate_n": pd.to_numeric(rec.get("gate_n", np.nan), errors="coerce"),

        # 真实策略最终收益：分类Top5 + 排序Top1
        "ret": float(chosen["real_return"]),
        "serial_code": chosen["code"],

        # serial_top1 字段（真实交易链路）
        "serial_pred": pd.to_numeric(serial_top1.get("predicted_return", np.nan), errors="coerce"),
        "serial_score_top1_in_cls": pd.to_numeric(serial_top1.get("score_top1_in_cls", np.nan), errors="coerce"),
        "serial_score_top2_in_cls": pd.to_numeric(serial_top1.get("score_top2_in_cls", np.nan), errors="coerce"),
        "serial_score_margin_in_cls": pd.to_numeric(
            serial_top1.get("score_margin_top1_top2_in_cls", np.nan), errors="coerce"
        ),
        "serial_p_cls": pd.to_numeric(serial_top1.get("p_cls", np.nan), errors="coerce"),
        "serial_p_cls_pct": pd.to_numeric(serial_top1.get("p_cls_pct", np.nan), errors="coerce"),
        "serial_rank_pos_all": pd.to_numeric(serial_top1.get("rank_pos_all", np.nan), errors="coerce"),
        "serial_rank_pct_all": pd.to_numeric(serial_top1.get("rank_pct_all", np.nan), errors="coerce"),

        # 全市场排序Top1，仅作参照
        "rank_all_top1_code": rank_all_top1.get("code", None),
        "rank_all_top1_pred": pd.to_numeric(rank_all_top1.get("predicted_return", np.nan), errors="coerce"),
        "rank_all_top1_ret": pd.to_numeric(rank_all_top1.get("real_return", np.nan), errors="coerce"),

        # JSON已给字段
        "conditional_gap": pd.to_numeric(rec.get("conditional_gap", np.nan), errors="coerce"),
        "unconditional_gap": pd.to_numeric(rec.get("unconditional_gap", np.nan), errors="coerce"),
        "conditional_score_margin_top1_top2": pd.to_numeric(
            rec.get("conditional_score_margin_top1_top2", np.nan), errors="coerce"
        ),
    }

    # Top5结构因子
    row["top5_mean_ret"] = float(df_cls_top5["real_return"].mean())
    row["top5_median_ret"] = float(df_cls_top5["real_return"].median())
    row["top5_cls_gap_1_2"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[1]["p_cls"])
    row["top5_cls_gap_1_5"] = float(df_cls_top5.iloc[0]["p_cls"] - df_cls_top5.iloc[4]["p_cls"])
    row["top5_cls_std"] = float(df_cls_top5["p_cls"].std(ddof=1))

    row["top5_rank_gap_1_2"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1]["rank_score"])
    row["top5_rank_gap_1_5"] = float(df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[4]["rank_score"])
    row["top5_rank_gap_1_mean_rest"] = float(
        df_rank_in_top5.iloc[0]["rank_score"] - df_rank_in_top5.iloc[1:]["rank_score"].mean()
    )
    row["top5_rank_std"] = float(df_rank_in_top5["rank_score"].std(ddof=1))

    # chosen在Top5中的分类位置
    chosen_code = chosen["code"]
    match = df_cls_top5[df_cls_top5["code"] == chosen_code]
    row["chosen_cls_rank_in_top5"] = float(match.iloc[0]["cls_rank_in_top5"]) if len(match) else np.nan
    row["chosen_cls_rank_pct_in_top5"] = (
        1.0 - (row["chosen_cls_rank_in_top5"] - 1) / 4.0 if pd.notna(row["chosen_cls_rank_in_top5"]) else np.nan
    )

    # 组合信号
    row["conf_pred_x_cls"] = row["serial_pred"] * row["serial_p_cls"]
    row["conf_rankgap_x_rankpct"] = row["serial_score_margin_in_cls"] * row["serial_rank_pct_all"]
    row["conf_mul_1"] = row["top5_rank_gap_1_2"] * row["serial_rank_pct_all"]
    row["conf_mul_2"] = row["serial_score_margin_in_cls"] * row["serial_p_cls"]
    row["conf_mul_3"] = row["top5_rank_gap_1_5"] * row["serial_rank_pct_all"]
    row["conf_mul_4"] = row["top5_cls_std"] * row["serial_rank_pct_all"]

    day_rows.append(row)

df_day = pd.DataFrame(day_rows).sort_values("date").reset_index(drop=True)
df_top5 = pd.concat(cls_topn_rows, ignore_index=True) if cls_topn_rows else pd.DataFrame()

print(f"valid_days: {len(df_day)}")
print(f"top5_rows: {len(df_top5)}")

# =========================================================
# 基线概览
# =========================================================

print_summary("BASELINE: Top5 + Rank1 real return", df_day["ret"])
print_summary("TOP5 mean return", df_day["top5_mean_ret"])
print_summary("TOP5 median return", df_day["top5_median_ret"])

# =========================================================
# 分段切分
# =========================================================

if USE_HALF_SPLIT:
    split_idx = len(df_day) // 2
    df_p1 = df_day.iloc[:split_idx].copy()
    df_p2 = df_day.iloc[split_idx:].copy()
    split_label = f"HALF_SPLIT idx={split_idx}"
else:
    if MANUAL_SPLIT_DATE is None:
        raise ValueError("当 USE_HALF_SPLIT=False 时，必须设置 MANUAL_SPLIT_DATE")
    df_p1 = df_day[df_day["date"] < MANUAL_SPLIT_DATE].copy()
    df_p2 = df_day[df_day["date"] >= MANUAL_SPLIT_DATE].copy()
    split_label = f"MANUAL_SPLIT date={MANUAL_SPLIT_DATE}"

print("\n" + "#" * 120)
print("SPLIT INFO")
print("#" * 120)
print(split_label)
print(f"P1 days: {len(df_p1)}, start={df_p1['date'].min()}, end={df_p1['date'].max()}")
print(f"P2 days: {len(df_p2)}, start={df_p2['date'].min()}, end={df_p2['date'].max()}")

# =========================================================
# 1) 排序稳定性分析
# =========================================================

print("\n" + "#" * 120)
print("RANK MODEL STABILITY")
print("#" * 120)

rank_features = [
    "serial_rank_pct_all",
    "serial_pred",
    "serial_score_margin_in_cls",
    "top5_rank_gap_1_2",
    "top5_rank_gap_1_5",
    "top5_rank_gap_1_mean_rest",
    "top5_rank_std",
    "conditional_gap",
    "unconditional_gap",
]

for feat in rank_features:
    if feat in df_day.columns:
        bucket_stats(df_day, feat, ret_col="ret", q=5, title=f"FULL SAMPLE | rank feature = {feat}")

print("\n" + "#" * 120)
print("RANK MODEL STABILITY BY PERIOD")
print("#" * 120)

for feat in ["serial_rank_pct_all", "serial_pred", "top5_rank_gap_1_5", "conditional_gap", "unconditional_gap"]:
    if feat in df_day.columns:
        bucket_stats(df_p1, feat, ret_col="ret", q=5, title=f"P1 | rank feature = {feat}")
        bucket_stats(df_p2, feat, ret_col="ret", q=5, title=f"P2 | rank feature = {feat}")

# =========================================================
# 2) 分类稳定性分析
# =========================================================

print("\n" + "#" * 120)
print("CLS MODEL STABILITY")
print("#" * 120)

cls_features_day_level = [
    "serial_p_cls",
    "serial_p_cls_pct",
    "top5_cls_gap_1_2",
    "top5_cls_gap_1_5",
    "top5_cls_std",
    "chosen_cls_rank_pct_in_top5",
]

for feat in cls_features_day_level:
    if feat in df_day.columns:
        bucket_stats(df_day, feat, ret_col="ret", q=5, title=f"FULL SAMPLE | cls feature = {feat}")

print("\n" + "#" * 120)
print("CLS MODEL STABILITY BY PERIOD")
print("#" * 120)

for feat in ["serial_p_cls", "top5_cls_gap_1_5", "top5_cls_std", "chosen_cls_rank_pct_in_top5"]:
    if feat in df_day.columns:
        bucket_stats(df_p1, feat, ret_col="ret", q=5, title=f"P1 | cls feature = {feat}")
        bucket_stats(df_p2, feat, ret_col="ret", q=5, title=f"P2 | cls feature = {feat}")

# =========================================================
# 3) Top5 横截面稳定性（分类池质量）
# =========================================================

print("\n" + "#" * 120)
print("TOP5 CROSS-SECTION QUALITY")
print("#" * 120)

if len(df_top5) > 0:
    # 先看所有Top5样本本身，分类排位/排序排位与未来收益关系
    bucket_stats(df_top5, "cls_rank_in_top5", ret_col="real_return", q=5, title="TOP5 rows | cls_rank_in_top5 vs real_return")
    bucket_stats(df_top5, "p_cls", ret_col="real_return", q=5, title="TOP5 rows | p_cls vs real_return")
    bucket_stats(df_top5, "rank_score", ret_col="real_return", q=5, title="TOP5 rows | rank_score vs real_return")
    bucket_stats(df_top5, "rank_pct_all", ret_col="real_return", q=5, title="TOP5 rows | rank_pct_all vs real_return")

    # 分前后半段
    top5_p1 = df_top5[df_top5["date"].isin(df_p1["date"])]
    top5_p2 = df_top5[df_top5["date"].isin(df_p2["date"])]

    bucket_stats(top5_p1, "p_cls", ret_col="real_return", q=5, title="TOP5 P1 | p_cls vs real_return")
    bucket_stats(top5_p2, "p_cls", ret_col="real_return", q=5, title="TOP5 P2 | p_cls vs real_return")

    bucket_stats(top5_p1, "rank_score", ret_col="real_return", q=5, title="TOP5 P1 | rank_score vs real_return")
    bucket_stats(top5_p2, "rank_score", ret_col="real_return", q=5, title="TOP5 P2 | rank_score vs real_return")

# =========================================================
# 4) 前后半段核心指标对比
# =========================================================

print("\n" + "#" * 120)
print("P1 / P2 SUMMARY COMPARISON")
print("#" * 120)

sum_rows = []

for name, part in [("FULL", df_day), ("P1", df_p1), ("P2", df_p2)]:
    s = summarize_arr(part["ret"])
    s["part"] = name

    # 核心因子均值
    for feat in [
        "serial_rank_pct_all",
        "serial_pred",
        "serial_p_cls",
        "serial_score_margin_in_cls",
        "top5_rank_gap_1_5",
        "top5_cls_gap_1_5",
        "top5_cls_std",
        "conditional_gap",
        "unconditional_gap",
    ]:
        s[f"{feat}_mean"] = float(part[feat].mean()) if feat in part.columns else np.nan

    sum_rows.append(s)

df_sum = pd.DataFrame(sum_rows)
cols_show = [
    "part", "count", "mean", "median", "win_rate",
    "big_loss_5pct_rate", "big_loss_8pct_rate", "max_drawdown", "sharpe",
    "serial_rank_pct_all_mean", "serial_pred_mean", "serial_p_cls_mean",
    "serial_score_margin_in_cls_mean", "top5_rank_gap_1_5_mean", "top5_cls_gap_1_5_mean",
    "top5_cls_std_mean", "conditional_gap_mean", "unconditional_gap_mean"
]
print(df_sum[cols_show].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# =========================================================
# 5) rolling 稳定性（时间衰减）
# =========================================================

print("\n" + "#" * 120)
print("ROLLING STABILITY")
print("#" * 120)

df_roll = df_day.copy()
df_roll["rolling_ret_mean"] = df_roll["ret"].rolling(ROLLING_WIN).mean()
df_roll["rolling_ret_win"] = df_roll["ret"].rolling(ROLLING_WIN).apply(lambda x: (x > 0).mean(), raw=False)
df_roll["rolling_loss5"] = df_roll["ret"].rolling(ROLLING_WIN).apply(lambda x: (x <= -0.05).mean(), raw=False)

# rolling bucket spread：高rank_pct桶均值 - 低rank_pct桶均值
rolling_spread_rows = []
for i in range(ROLLING_WIN, len(df_day) + 1):
    sub = df_day.iloc[i - ROLLING_WIN:i].copy()
    try:
        sub["bucket"] = pd.qcut(sub["serial_rank_pct_all"], q=5, duplicates="drop")
        means = sub.groupby("bucket", observed=False)["ret"].mean().values
        if len(means) >= 2:
            spread = float(means[-1] - means[0])
        else:
            spread = np.nan
    except Exception:
        spread = np.nan
    rolling_spread_rows.append({"date": df_day.iloc[i - 1]["date"], "rolling_rank_spread": spread})

df_spread = pd.DataFrame(rolling_spread_rows)
df_roll = df_roll.merge(df_spread, on="date", how="left")

print("\nRecent rolling rows:")
show_cols = ["date", "ret", "rolling_ret_mean", "rolling_ret_win", "rolling_loss5", "rolling_rank_spread"]
print(df_roll[show_cols].tail(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nRolling summary:")
for col in ["rolling_ret_mean", "rolling_ret_win", "rolling_loss5", "rolling_rank_spread"]:
    x = df_roll[col].dropna()
    if len(x) == 0:
        continue
    print(
        f"{col}: "
        f"mean={x.mean():.6f}, "
        f"median={x.median():.6f}, "
        f"min={x.min():.6f}, "
        f"max={x.max():.6f}, "
        f"last={x.iloc[-1]:.6f}"
    )

# =========================================================
# 6) 稳定性打分：排序alpha是否衰减、分类alpha是否衰减
# =========================================================

print("\n" + "#" * 120)
print("AUTO INTERPRETATION")
print("#" * 120)

def period_bucket_spread(df, feat, ret_col="ret", q=5):
    sub = df[[feat, ret_col]].dropna().copy()
    if len(sub) < q * 5:
        return np.nan
    try:
        sub["bucket"] = pd.qcut(sub[feat], q=q, duplicates="drop")
        means = sub.groupby("bucket", observed=False)[ret_col].mean().values
        if len(means) < 2:
            return np.nan
        return float(means[-1] - means[0])
    except Exception:
        return np.nan

rank_spread_p1 = period_bucket_spread(df_p1, "serial_rank_pct_all", "ret", 5)
rank_spread_p2 = period_bucket_spread(df_p2, "serial_rank_pct_all", "ret", 5)

cls_spread_p1 = period_bucket_spread(df_p1, "serial_p_cls", "ret", 5)
cls_spread_p2 = period_bucket_spread(df_p2, "serial_p_cls", "ret", 5)

top5rank_spread_p1 = period_bucket_spread(df_p1, "top5_rank_gap_1_5", "ret", 5)
top5rank_spread_p2 = period_bucket_spread(df_p2, "top5_rank_gap_1_5", "ret", 5)

top5cls_spread_p1 = period_bucket_spread(df_p1, "top5_cls_gap_1_5", "ret", 5)
top5cls_spread_p2 = period_bucket_spread(df_p2, "top5_cls_gap_1_5", "ret", 5)

print(f"rank_spread_p1 (serial_rank_pct_all): {rank_spread_p1:.6f}")
print(f"rank_spread_p2 (serial_rank_pct_all): {rank_spread_p2:.6f}")
print(f"rank_spread_change: {(rank_spread_p2 - rank_spread_p1):.6f}")

print(f"cls_spread_p1 (serial_p_cls): {cls_spread_p1:.6f}")
print(f"cls_spread_p2 (serial_p_cls): {cls_spread_p2:.6f}")
print(f"cls_spread_change: {(cls_spread_p2 - cls_spread_p1):.6f}")

print(f"top5rank_spread_p1 (top5_rank_gap_1_5): {top5rank_spread_p1:.6f}")
print(f"top5rank_spread_p2 (top5_rank_gap_1_5): {top5rank_spread_p2:.6f}")
print(f"top5rank_spread_change: {(top5rank_spread_p2 - top5rank_spread_p1):.6f}")

print(f"top5cls_spread_p1 (top5_cls_gap_1_5): {top5cls_spread_p1:.6f}")
print(f"top5cls_spread_p2 (top5_cls_gap_1_5): {top5cls_spread_p2:.6f}")
print(f"top5cls_spread_change: {(top5cls_spread_p2 - top5cls_spread_p1):.6f}")

print("\nInterpretation guide:")
print("1) 若 rank_spread_p2 显著小于 rank_spread_p1，说明排序模型alpha后段衰减。")
print("2) 若 cls_spread_p2 显著小于 cls_spread_p1，说明分类模型alpha后段衰减。")
print("3) 若 top5_rank_gap_1_5 的 spread 后段变差，说明排序模型的“区分度信号”也在衰减。")
print("4) 若 top5_cls_gap_1_5 的 spread 后段变差，说明分类模型Top5池的质量分层在衰减。")
print("5) 若 rolling_rank_spread 后段明显走弱甚至转负，基本可确认排序alpha衰减。")

print("\nDone.")