import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 修改这里
# =========================
JSON_PATH = "daily_gate_stats_Window_3_gateN5_20260315_015208.json"   # 改成你的json路径
ROLLING_WINDOW = 20


# =========================
# 2. 基础函数
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def calc_max_drawdown(ret_series: pd.Series) -> float:
    if len(ret_series) == 0:
        return np.nan
    nav = (1 + ret_series.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def calc_cvar(ret_series: pd.Series, q=0.1) -> float:
    s = ret_series.dropna()
    if len(s) == 0:
        return np.nan
    thresh = s.quantile(q)
    tail = s[s <= thresh]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def calc_sharpe(ret_series: pd.Series) -> float:
    s = ret_series.dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float((s.mean() / std) * math.sqrt(252))


def summarize_series(name, s: pd.Series):
    s = s.dropna()
    out = {
        "name": name,
        "count": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "win_rate": float((s > 0).mean()) if len(s) else np.nan,
        "big_loss_3pct_rate": float((s <= -0.03).mean()) if len(s) else np.nan,
        "big_loss_5pct_rate": float((s <= -0.05).mean()) if len(s) else np.nan,
        "big_loss_8pct_rate": float((s <= -0.08).mean()) if len(s) else np.nan,
        "cvar10": calc_cvar(s, 0.1),
        "max_drawdown": calc_max_drawdown(s),
        "sharpe": calc_sharpe(s),
        "p5": float(s.quantile(0.05)) if len(s) else np.nan,
        "p10": float(s.quantile(0.10)) if len(s) else np.nan,
        "p25": float(s.quantile(0.25)) if len(s) else np.nan,
        "p75": float(s.quantile(0.75)) if len(s) else np.nan,
        "p90": float(s.quantile(0.90)) if len(s) else np.nan,
        "p95": float(s.quantile(0.95)) if len(s) else np.nan,
    }
    return out


def print_summary(title, s: pd.Series):
    d = summarize_series(title, s)
    print("=" * 100)
    print(title)
    print("=" * 100)
    for k, v in d.items():
        if k == "name":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print()


def bucket_table(df, feature_col, ret_col, n_bins=5, title=None):
    sub = df[[feature_col, ret_col]].dropna().copy()
    if len(sub) < max(20, n_bins * 5):
        print(f"[SKIP] {feature_col}: too few rows")
        return None

    try:
        sub["bucket"] = pd.qcut(sub[feature_col], q=n_bins, duplicates="drop")
    except Exception as e:
        print(f"[SKIP] {feature_col}: qcut failed: {e}")
        return None

    g = sub.groupby("bucket", observed=False)[ret_col].agg(
        count="count",
        mean="mean",
        median="median",
        std="std"
    )
    sub2 = sub.copy()
    g["win_rate"] = sub2.groupby("bucket", observed=False)[ret_col].apply(lambda x: (x > 0).mean())
    g["loss3_rate"] = sub2.groupby("bucket", observed=False)[ret_col].apply(lambda x: (x <= -0.03).mean())
    g["loss5_rate"] = sub2.groupby("bucket", observed=False)[ret_col].apply(lambda x: (x <= -0.05).mean())
    g["loss8_rate"] = sub2.groupby("bucket", observed=False)[ret_col].apply(lambda x: (x <= -0.08).mean())
    g["max_drawdown"] = sub2.groupby("bucket", observed=False)[ret_col].apply(calc_max_drawdown)

    means = g["mean"].values
    spread = float(means[-1] - means[0]) if len(means) >= 2 else np.nan

    monotonic_hint = "MIXED"
    if len(means) >= 3:
        diffs = np.diff(means)
        if np.all(diffs >= 0):
            monotonic_hint = "LIKELY_POSITIVE_MONOTONIC"
        elif np.all(diffs <= 0):
            monotonic_hint = "LIKELY_NEGATIVE_MONOTONIC"

    if title:
        print("-" * 100)
        print(title)
    print(g.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"monotonic_hint: {monotonic_hint}")
    print(f"spread_mean(top-bottom): {spread:.6f}")
    print()
    return g, spread, monotonic_hint


# =========================
# 3. 从 JSON 构造日表 / Top5表
# =========================
def build_tables(js):
    day_rows = []
    top5_rows = []

    for dt, item in sorted(js.items(), key=lambda kv: kv[0]):
        cls_topn = item.get("cls_topn", [])
        serial_top1 = item.get("serial_top1", {})
        rank_all_top1 = item.get("rank_all_top1", {})

        if not cls_topn or len(cls_topn) == 0:
            continue
        if not serial_top1:
            continue

        # ---- 分类Top1：cls_topn[0]
        cls_top1 = cls_topn[0]

        # ---- Top2 / Top3 / Top5
        top2 = cls_topn[:2]
        top3 = cls_topn[:3]
        top5 = cls_topn[:5]

        def mean_ret(arr):
            vals = [safe_float(x.get("real_return")) for x in arr]
            vals = [x for x in vals if not np.isnan(x)]
            return np.mean(vals) if vals else np.nan

        def eq_weight_ret(arr):
            return mean_ret(arr)

        # ---- chosen cls rank inside top5: serial_top1 在 cls_topn 中的位置
        chosen_code = serial_top1.get("code")
        chosen_cls_rank = np.nan
        for i, x in enumerate(cls_topn, start=1):
            if x.get("code") == chosen_code:
                chosen_cls_rank = i
                break
        chosen_cls_rank_pct = (chosen_cls_rank - 1) / (len(top5) - 1) if len(top5) > 1 and not np.isnan(chosen_cls_rank) else np.nan

        # ---- top5 internal stats
        pcls_vals = [safe_float(x.get("p_cls")) for x in top5]
        rank_vals = [safe_float(x.get("rank_score")) for x in top5]
        pcls_vals = [x for x in pcls_vals if not np.isnan(x)]
        rank_vals = [x for x in rank_vals if not np.isnan(x)]

        top5_cls_gap_1_2 = np.nan
        top5_cls_gap_1_5 = np.nan
        top5_cls_std = np.nan
        if len(pcls_vals) >= 2:
            top5_cls_gap_1_2 = pcls_vals[0] - pcls_vals[1]
            top5_cls_gap_1_5 = pcls_vals[0] - pcls_vals[-1]
            top5_cls_std = float(np.std(pcls_vals, ddof=0))

        top5_rank_gap_1_2 = np.nan
        top5_rank_gap_1_5 = np.nan
        top5_rank_gap_1_mean_rest = np.nan
        top5_rank_std = np.nan
        if len(rank_vals) >= 2:
            sorted_rank = sorted(rank_vals, reverse=True)
            top5_rank_gap_1_2 = sorted_rank[0] - sorted_rank[1]
            top5_rank_gap_1_5 = sorted_rank[0] - sorted_rank[-1]
            top5_rank_gap_1_mean_rest = sorted_rank[0] - np.mean(sorted_rank[1:])
            top5_rank_std = float(np.std(sorted_rank, ddof=0))

        serial_pred = safe_float(serial_top1.get("predicted_return"))
        serial_p_cls = safe_float(serial_top1.get("p_cls"))
        serial_rank_pct_all = safe_float(serial_top1.get("rank_pct_all"))
        serial_score_margin_in_cls = safe_float(serial_top1.get("score_margin_top1_top2_in_cls"))
        serial_real_ret = safe_float(serial_top1.get("real_return"))

        cls_top1_ret = safe_float(cls_top1.get("real_return"))
        cls_top1_p_cls = safe_float(cls_top1.get("p_cls"))
        cls_top1_rank_score = safe_float(cls_top1.get("rank_score"))
        cls_top1_rank_pct_all = safe_float(cls_top1.get("rank_pct_all"))

        day_rows.append({
            "date": dt,
            "gate_n": item.get("gate_n"),
            "universe_size": item.get("universe_size"),

            # baseline and variants
            "serial_top1_ret": serial_real_ret,          # 你的真实交易链：Top5 + 排序Top1
            "cls_top1_ret": cls_top1_ret,                # 分类Top1
            "top2_eq_ret": eq_weight_ret(top2),
            "top3_eq_ret": eq_weight_ret(top3),
            "top5_eq_ret": eq_weight_ret(top5),
            "top5_mean_ret": mean_ret(top5),
            "top5_median_ret": np.median([safe_float(x.get("real_return")) for x in top5]) if len(top5) else np.nan,

            # serial features
            "serial_pred": serial_pred,
            "serial_p_cls": serial_p_cls,
            "serial_rank_pct_all": serial_rank_pct_all,
            "serial_score_margin_in_cls": serial_score_margin_in_cls,

            # cls-top1 features
            "cls_top1_p_cls": cls_top1_p_cls,
            "cls_top1_rank_score": cls_top1_rank_score,
            "cls_top1_rank_pct_all": cls_top1_rank_pct_all,

            # top5 structural features
            "top5_cls_gap_1_2": top5_cls_gap_1_2,
            "top5_cls_gap_1_5": top5_cls_gap_1_5,
            "top5_cls_std": top5_cls_std,
            "top5_rank_gap_1_2": top5_rank_gap_1_2,
            "top5_rank_gap_1_5": top5_rank_gap_1_5,
            "top5_rank_gap_1_mean_rest": top5_rank_gap_1_mean_rest,
            "top5_rank_std": top5_rank_std,

            # existing json features
            "conditional_gap": safe_float(item.get("conditional_gap")),
            "unconditional_gap": safe_float(item.get("unconditional_gap")),
            "conditional_score_margin_top1_top2": safe_float(item.get("conditional_score_margin_top1_top2")),

            # chosen rank in cls top5
            "chosen_cls_rank_in_top5": chosen_cls_rank,
            "chosen_cls_rank_pct_in_top5": chosen_cls_rank_pct,

            # rank_all_top1
            "rank_all_top1_ret": safe_float(rank_all_top1.get("real_return")),
            "rank_all_top1_pred": safe_float(rank_all_top1.get("predicted_return")),
        })

        # ---- top5 cross-section rows
        for i, row in enumerate(top5, start=1):
            top5_rows.append({
                "date": dt,
                "cls_rank_in_top5": i,
                "cls_rank_pct_in_top5": (i - 1) / 4 if len(top5) == 5 else np.nan,
                "code": row.get("code"),
                "p_cls": safe_float(row.get("p_cls")),
                "rank_score": safe_float(row.get("rank_score")),
                "rank_pct_all": safe_float(row.get("rank_pct_all")),
                "real_return": safe_float(row.get("real_return")),
                "is_serial_top1": int(row.get("code") == chosen_code),
            })

    day_df = pd.DataFrame(day_rows).sort_values("date").reset_index(drop=True)
    top5_df = pd.DataFrame(top5_rows).sort_values(["date", "cls_rank_in_top5"]).reset_index(drop=True)
    return day_df, top5_df


# =========================
# 4. 前后半段分析
# =========================
def split_halves(df):
    n = len(df)
    split_idx = n // 2
    p1 = df.iloc[:split_idx].copy()
    p2 = df.iloc[split_idx:].copy()
    return p1, p2, split_idx


def spread_of_feature(df, feature_col, ret_col, n_bins=5):
    sub = df[[feature_col, ret_col]].dropna()
    if len(sub) < max(20, n_bins * 5):
        return np.nan
    try:
        sub["bucket"] = pd.qcut(sub[feature_col], q=n_bins, duplicates="drop")
    except Exception:
        return np.nan
    g = sub.groupby("bucket", observed=False)[ret_col].mean()
    if len(g) < 2:
        return np.nan
    return float(g.iloc[-1] - g.iloc[0])


# =========================
# 5. rolling
# =========================
def add_rolling(day_df, ret_col="serial_top1_ret", rank_spread_col="serial_rank_pct_all", window=20):
    df = day_df.copy()
    df["rolling_ret_mean"] = df[ret_col].rolling(window).mean()
    df["rolling_ret_win"] = df[ret_col].rolling(window).apply(lambda x: np.mean(np.array(x) > 0), raw=False)
    df["rolling_loss5"] = df[ret_col].rolling(window).apply(lambda x: np.mean(np.array(x) <= -0.05), raw=False)

    def rolling_spread(x_feat, x_ret):
        sub = pd.DataFrame({"f": x_feat, "r": x_ret}).dropna()
        if len(sub) < 10:
            return np.nan
        try:
            sub["b"] = pd.qcut(sub["f"], q=min(5, len(sub)), duplicates="drop")
            g = sub.groupby("b", observed=False)["r"].mean()
            if len(g) < 2:
                return np.nan
            return float(g.iloc[-1] - g.iloc[0])
        except Exception:
            return np.nan

    vals = []
    for i in range(len(df)):
        if i + 1 < window:
            vals.append(np.nan)
            continue
        sub = df.iloc[i + 1 - window:i + 1]
        vals.append(rolling_spread(sub[rank_spread_col], sub[ret_col]))
    df["rolling_rank_spread"] = vals
    return df


# =========================
# 6. 主程序
# =========================
def main():
    js = load_json(JSON_PATH)
    day_df, top5_df = build_tables(js)

    print(f"valid_days: {len(day_df)}")
    print(f"top5_rows: {len(top5_df)}")
    print()

    # -------------------------
    # baseline comparisons
    # -------------------------
    print_summary("BASELINE: Top5 + Rank1 real return", day_df["serial_top1_ret"])
    print_summary("CLS TOP1: classification-only Top1 real return", day_df["cls_top1_ret"])
    print_summary("TOP2 equal-weight", day_df["top2_eq_ret"])
    print_summary("TOP3 equal-weight", day_df["top3_eq_ret"])
    print_summary("TOP5 mean return", day_df["top5_mean_ret"])
    print_summary("TOP5 median return", day_df["top5_median_ret"])

    # -------------------------
    # split info
    # -------------------------
    p1, p2, split_idx = split_halves(day_df)
    print("#" * 116)
    print("SPLIT INFO")
    print("#" * 116)
    print(f"HALF_SPLIT idx={split_idx}")
    print(f"P1 days: {len(p1)}, start={p1['date'].iloc[0]}, end={p1['date'].iloc[-1]}")
    print(f"P2 days: {len(p2)}, start={p2['date'].iloc[0]}, end={p2['date'].iloc[-1]}")
    print()

    # -------------------------
    # rank model stability
    # -------------------------
    print("#" * 116)
    print("RANK MODEL STABILITY")
    print("#" * 116)
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
        bucket_table(day_df, feat, "serial_top1_ret", title=f"FULL SAMPLE | rank feature = {feat}")

    # -------------------------
    # rank stability by period
    # -------------------------
    print("#" * 116)
    print("RANK MODEL STABILITY BY PERIOD")
    print("#" * 116)
    period_rank_feats = [
        "serial_rank_pct_all",
        "serial_pred",
        "top5_rank_gap_1_5",
        "conditional_gap",
        "unconditional_gap",
    ]
    for feat in period_rank_feats:
        bucket_table(p1, feat, "serial_top1_ret", title=f"P1 | rank feature = {feat}")
        bucket_table(p2, feat, "serial_top1_ret", title=f"P2 | rank feature = {feat}")

    # -------------------------
    # cls model stability
    # -------------------------
    print("#" * 116)
    print("CLS MODEL STABILITY")
    print("#" * 116)
    cls_features = [
        "serial_p_cls",
        "top5_cls_gap_1_2",
        "top5_cls_gap_1_5",
        "top5_cls_std",
        "chosen_cls_rank_pct_in_top5",
    ]
    for feat in cls_features:
        bucket_table(day_df, feat, "serial_top1_ret", title=f"FULL SAMPLE | cls feature = {feat}")

    # -------------------------
    # cls stability by period
    # -------------------------
    print("#" * 116)
    print("CLS MODEL STABILITY BY PERIOD")
    print("#" * 116)
    period_cls_feats = [
        "serial_p_cls",
        "top5_cls_gap_1_5",
        "top5_cls_std",
        "chosen_cls_rank_pct_in_top5",
    ]
    for feat in period_cls_feats:
        bucket_table(p1, feat, "serial_top1_ret", title=f"P1 | cls feature = {feat}")
        bucket_table(p2, feat, "serial_top1_ret", title=f"P2 | cls feature = {feat}")

    # -------------------------
    # top5 cross-sectional quality
    # -------------------------
    print("#" * 116)
    print("TOP5 CROSS-SECTION QUALITY")
    print("#" * 116)
    bucket_table(top5_df, "cls_rank_in_top5", "real_return", title="TOP5 rows | cls_rank_in_top5 vs real_return")
    bucket_table(top5_df, "p_cls", "real_return", title="TOP5 rows | p_cls vs real_return")
    bucket_table(top5_df, "rank_score", "real_return", title="TOP5 rows | rank_score vs real_return")
    bucket_table(top5_df, "rank_pct_all", "real_return", title="TOP5 rows | rank_pct_all vs real_return")

    top5_p1 = top5_df[top5_df["date"].isin(set(p1["date"]))]
    top5_p2 = top5_df[top5_df["date"].isin(set(p2["date"]))]

    bucket_table(top5_p1, "p_cls", "real_return", title="TOP5 P1 | p_cls vs real_return")
    bucket_table(top5_p2, "p_cls", "real_return", title="TOP5 P2 | p_cls vs real_return")
    bucket_table(top5_p1, "rank_score", "real_return", title="TOP5 P1 | rank_score vs real_return")
    bucket_table(top5_p2, "rank_score", "real_return", title="TOP5 P2 | rank_score vs real_return")

    # -------------------------
    # direct strategy comparison P1 / P2
    # -------------------------
    print("#" * 116)
    print("DIRECT STRATEGY COMPARISON: CLS TOP1 vs SERIAL TOP1")
    print("#" * 116)
    print_summary("FULL | serial_top1", day_df["serial_top1_ret"])
    print_summary("FULL | cls_top1", day_df["cls_top1_ret"])
    print_summary("P1 | serial_top1", p1["serial_top1_ret"])
    print_summary("P1 | cls_top1", p1["cls_top1_ret"])
    print_summary("P2 | serial_top1", p2["serial_top1_ret"])
    print_summary("P2 | cls_top1", p2["cls_top1_ret"])

    diff_full = day_df["serial_top1_ret"] - day_df["cls_top1_ret"]
    diff_p1 = p1["serial_top1_ret"] - p1["cls_top1_ret"]
    diff_p2 = p2["serial_top1_ret"] - p2["cls_top1_ret"]

    print_summary("FULL | serial_top1 - cls_top1", diff_full)
    print_summary("P1 | serial_top1 - cls_top1", diff_p1)
    print_summary("P2 | serial_top1 - cls_top1", diff_p2)

    # -------------------------
    # summary table
    # -------------------------
    def row_of(df_part, name):
        return {
            "part": name,
            "count": len(df_part),
            "serial_top1_mean": df_part["serial_top1_ret"].mean(),
            "cls_top1_mean": df_part["cls_top1_ret"].mean(),
            "top2_eq_mean": df_part["top2_eq_ret"].mean(),
            "top3_eq_mean": df_part["top3_eq_ret"].mean(),
            "top5_mean_ret": df_part["top5_mean_ret"].mean(),
            "serial_top1_loss5": (df_part["serial_top1_ret"] <= -0.05).mean(),
            "cls_top1_loss5": (df_part["cls_top1_ret"] <= -0.05).mean(),
            "serial_top1_mdd": calc_max_drawdown(df_part["serial_top1_ret"]),
            "cls_top1_mdd": calc_max_drawdown(df_part["cls_top1_ret"]),
            "serial_rank_pct_all_mean": df_part["serial_rank_pct_all"].mean(),
            "serial_pred_mean": df_part["serial_pred"].mean(),
            "serial_p_cls_mean": df_part["serial_p_cls"].mean(),
            "top5_rank_gap_1_5_mean": df_part["top5_rank_gap_1_5"].mean(),
            "top5_cls_gap_1_5_mean": df_part["top5_cls_gap_1_5"].mean(),
            "top5_cls_std_mean": df_part["top5_cls_std"].mean(),
            "conditional_gap_mean": df_part["conditional_gap"].mean(),
            "unconditional_gap_mean": df_part["unconditional_gap"].mean(),
        }

    summary_df = pd.DataFrame([
        row_of(day_df, "FULL"),
        row_of(p1, "P1"),
        row_of(p2, "P2"),
    ])

    print("#" * 116)
    print("P1 / P2 SUMMARY COMPARISON")
    print("#" * 116)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    # -------------------------
    # rolling
    # -------------------------
    roll = add_rolling(day_df, ret_col="serial_top1_ret", rank_spread_col="serial_rank_pct_all", window=ROLLING_WINDOW)

    print("#" * 116)
    print("ROLLING STABILITY")
    print("#" * 116)
    print(f"Recent rolling rows:")
    show_cols = ["date", "serial_top1_ret", "rolling_ret_mean", "rolling_ret_win", "rolling_loss5", "rolling_rank_spread"]
    tmp = roll[show_cols].tail(20).copy()
    tmp = tmp.rename(columns={"serial_top1_ret": "ret"})
    print(tmp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    rs = roll["rolling_rank_spread"].dropna()
    print("Rolling summary:")
    if len(rs):
        print(f"rolling_ret_mean: mean={roll['rolling_ret_mean'].dropna().mean():.6f}, median={roll['rolling_ret_mean'].dropna().median():.6f}, min={roll['rolling_ret_mean'].dropna().min():.6f}, max={roll['rolling_ret_mean'].dropna().max():.6f}, last={roll['rolling_ret_mean'].dropna().iloc[-1]:.6f}")
        print(f"rolling_ret_win: mean={roll['rolling_ret_win'].dropna().mean():.6f}, median={roll['rolling_ret_win'].dropna().median():.6f}, min={roll['rolling_ret_win'].dropna().min():.6f}, max={roll['rolling_ret_win'].dropna().max():.6f}, last={roll['rolling_ret_win'].dropna().iloc[-1]:.6f}")
        print(f"rolling_loss5: mean={roll['rolling_loss5'].dropna().mean():.6f}, median={roll['rolling_loss5'].dropna().median():.6f}, min={roll['rolling_loss5'].dropna().min():.6f}, max={roll['rolling_loss5'].dropna().max():.6f}, last={roll['rolling_loss5'].dropna().iloc[-1]:.6f}")
        print(f"rolling_rank_spread: mean={rs.mean():.6f}, median={rs.median():.6f}, min={rs.min():.6f}, max={rs.max():.6f}, last={rs.iloc[-1]:.6f}")
    print()

    # -------------------------
    # auto interpretation
    # -------------------------
    rank_spread_p1 = spread_of_feature(p1, "serial_rank_pct_all", "serial_top1_ret")
    rank_spread_p2 = spread_of_feature(p2, "serial_rank_pct_all", "serial_top1_ret")

    cls_spread_p1 = spread_of_feature(p1, "serial_p_cls", "serial_top1_ret")
    cls_spread_p2 = spread_of_feature(p2, "serial_p_cls", "serial_top1_ret")

    top5rank_spread_p1 = spread_of_feature(p1, "top5_rank_gap_1_5", "serial_top1_ret")
    top5rank_spread_p2 = spread_of_feature(p2, "top5_rank_gap_1_5", "serial_top1_ret")

    top5cls_spread_p1 = spread_of_feature(p1, "top5_cls_gap_1_5", "serial_top1_ret")
    top5cls_spread_p2 = spread_of_feature(p2, "top5_cls_gap_1_5", "serial_top1_ret")

    print("#" * 116)
    print("AUTO INTERPRETATION")
    print("#" * 116)
    print(f"rank_spread_p1 (serial_rank_pct_all): {rank_spread_p1:.6f}")
    print(f"rank_spread_p2 (serial_rank_pct_all): {rank_spread_p2:.6f}")
    print(f"rank_spread_change: {(rank_spread_p2 - rank_spread_p1):.6f}")
    print()

    print(f"cls_spread_p1 (serial_p_cls): {cls_spread_p1:.6f}")
    print(f"cls_spread_p2 (serial_p_cls): {cls_spread_p2:.6f}")
    print(f"cls_spread_change: {(cls_spread_p2 - cls_spread_p1):.6f}")
    print()

    print(f"top5rank_spread_p1 (top5_rank_gap_1_5): {top5rank_spread_p1:.6f}")
    print(f"top5rank_spread_p2 (top5_rank_gap_1_5): {top5rank_spread_p2:.6f}")
    print(f"top5rank_spread_change: {(top5rank_spread_p2 - top5rank_spread_p1):.6f}")
    print()

    print(f"top5cls_spread_p1 (top5_cls_gap_1_5): {top5cls_spread_p1:.6f}")
    print(f"top5cls_spread_p2 (top5_cls_gap_1_5): {top5cls_spread_p2:.6f}")
    print(f"top5cls_spread_change: {(top5cls_spread_p2 - top5cls_spread_p1):.6f}")
    print()

    print("Interpretation guide:")
    print("1) 重点看 FULL/P1/P2 的 cls_top1_ret 与 serial_top1_ret：")
    print("   - 若 cls_top1 比较稳，而 serial_top1 后段变差，说明排序模型是瓶颈。")
    print("2) 看 serial_rank_pct_all / serial_pred 在 P1/P2 的 spread：")
    print("   - 若 P2 显著小于 P1，说明排序alpha后段衰减。")
    print("3) 看 serial_p_cls 的 spread：")
    print("   - 这是“最终被买股票的分类置信度解释力”，不是分类Top1本身。")
    print("4) 真正判断分类Top1稳不稳，以 cls_top1_ret 的 P1/P2 对比为准。")
    print("5) 若 top5_mean_ret 稳而 serial_top1_ret 不稳，说明 Top5池没坏，坏在池内选票。")


if __name__ == "__main__":
    main()