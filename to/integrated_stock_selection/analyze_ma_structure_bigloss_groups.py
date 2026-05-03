from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

BASELINE_JSON = RESULTS_DIR / "walkforward_metrics_20260430_011835.json"
REMAINING_BIGLOSS_CSV = RESULTS_DIR / "remaining_bigloss_samples_20260430_173039.csv"
DAILY_PARQUET = CACHE_DIR / "stock_features_20221201_20260428.parquet"


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / max(len(x) + len(y) - 2, 1)
    if pooled <= 0:
        return np.nan
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def rank_biserial(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    wins = 0.0
    ties = 0.0
    for xv in x:
        comp = xv - y
        wins += float((comp > 0).sum())
        ties += float((comp == 0).sum())
    total = len(x) * len(y)
    if total == 0:
        return np.nan
    u = wins + 0.5 * ties
    return float((2 * u) / total - 1)


def load_baseline_trades() -> pd.DataFrame:
    data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    trades = pd.DataFrame(data[0]["all_results"]["1"]["trades"])
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"]).dt.normalize()
    trades["actual_return"] = pd.to_numeric(trades["actual_return"], errors="coerce")
    return trades


def load_market_dates() -> pd.DatetimeIndex:
    cols = ["date"]
    dates = pd.read_parquet(DAILY_PARQUET, columns=cols)["date"]
    dates = pd.to_datetime(dates).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def previous_trading_day(buy_dates: pd.Series, market_dates: pd.DatetimeIndex) -> pd.Series:
    values = pd.to_datetime(buy_dates).values.astype("datetime64[ns]")
    md = market_dates.values.astype("datetime64[ns]")
    pos = np.searchsorted(md, values, side="left") - 1
    out = pd.Series(pd.NaT, index=buy_dates.index, dtype="datetime64[ns]")
    valid = pos >= 0
    if valid.any():
        out.loc[valid] = pd.to_datetime(md[pos[valid]])
    return out


def build_groups(trades: pd.DataFrame) -> pd.DataFrame:
    winners = trades.nlargest(16, "actual_return").copy()
    winners["group"] = "winners_top16"

    losers = trades[trades["actual_return"] <= -0.07].copy()
    losers["group"] = "bigloss_le_7"

    remaining = pd.read_csv(REMAINING_BIGLOSS_CSV)
    remaining["prediction_date"] = pd.to_datetime(remaining["prediction_date"])
    remaining["group"] = "remaining_bigloss_after_gate"

    market_dates = load_market_dates()
    winners["prediction_date"] = previous_trading_day(winners["buy_date"], market_dates)
    losers["prediction_date"] = previous_trading_day(losers["buy_date"], market_dates)

    keep_cols = ["stock_code", "prediction_date", "actual_return", "group"]
    winners = winners[keep_cols]
    losers = losers[keep_cols]
    remaining = remaining.rename(columns={"return_pct": "actual_return"})[keep_cols]
    groups = pd.concat([winners, losers, remaining], ignore_index=True)
    groups["prediction_date"] = pd.to_datetime(groups["prediction_date"])
    return groups.drop_duplicates(subset=["stock_code", "prediction_date", "group"])


def build_all_baseline_with_prediction_dates(trades: pd.DataFrame) -> pd.DataFrame:
    market_dates = load_market_dates()
    out = trades.copy()
    out["prediction_date"] = previous_trading_day(out["buy_date"], market_dates)
    return out[["stock_code", "prediction_date", "actual_return"]].dropna(subset=["prediction_date"])


def load_daily_with_ma() -> pd.DataFrame:
    cols = ["date", "stock_code", "open", "high", "low", "close"]
    df = pd.read_parquet(DAILY_PARQUET, columns=cols).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    g = df.groupby("stock_code", sort=False)
    for w in [5, 10, 20, 60]:
        df[f"ma{w}"] = g["close"].transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
        df[f"ma{w}_prev3"] = g[f"ma{w}"].shift(3)
        df[f"ma{w}_prev1"] = g[f"ma{w}"].shift(1)
        df[f"ma{w}_slope3"] = (df[f"ma{w}"] - df[f"ma{w}_prev3"]) / (df[f"ma{w}_prev3"] + 1e-9)
        df[f"ma{w}_slope1"] = (df[f"ma{w}"] - df[f"ma{w}_prev1"]) / (df[f"ma{w}_prev1"] + 1e-9)

    df["close_vs_ma5"] = (df["close"] - df["ma5"]) / (df["ma5"] + 1e-9)
    df["close_vs_ma10"] = (df["close"] - df["ma10"]) / (df["ma10"] + 1e-9)
    df["close_vs_ma20"] = (df["close"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["close_vs_ma60"] = (df["close"] - df["ma60"]) / (df["ma60"] + 1e-9)
    df["ma5_vs_ma10"] = (df["ma5"] - df["ma10"]) / (df["ma10"] + 1e-9)
    df["ma10_vs_ma20"] = (df["ma10"] - df["ma20"]) / (df["ma20"] + 1e-9)
    df["ma20_vs_ma60"] = (df["ma20"] - df["ma60"]) / (df["ma60"] + 1e-9)

    ma_stack = df[["ma5", "ma10", "ma20", "ma60"]]
    short_stack = df[["ma5", "ma10", "ma20"]]
    df["ma_stack_spread"] = (ma_stack.max(axis=1) - ma_stack.min(axis=1)) / (df["close"] + 1e-9)
    df["ma_short_spread"] = (short_stack.max(axis=1) - short_stack.min(axis=1)) / (df["close"] + 1e-9)
    df["ma_bullish_count"] = (
        (df["close"] > df["ma5"]).astype(int)
        + (df["ma5"] > df["ma10"]).astype(int)
        + (df["ma10"] > df["ma20"]).astype(int)
        + (df["ma20"] > df["ma60"]).astype(int)
    )
    df["ma_pair_up_count"] = (
        (df["ma5"] > df["ma10"]).astype(int)
        + (df["ma10"] > df["ma20"]).astype(int)
        + (df["ma20"] > df["ma60"]).astype(int)
    )
    df["weak_close"] = 1.0 - (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
    df["open_to_close"] = (df["close"] - df["open"]) / (df["open"] + 1e-9)
    df["high_to_close_raw"] = (df["high"] - df["close"]) / (df["high"] + 1e-9)
    df["close_below_ma5_above_ma20"] = ((df["close"] < df["ma5"]) & (df["close"] > df["ma20"])).astype(int)
    df["close_below_ma10_above_ma20"] = ((df["close"] < df["ma10"]) & (df["close"] > df["ma20"])).astype(int)
    df["close_below_ma20_above_ma60"] = ((df["close"] < df["ma20"]) & (df["close"] > df["ma60"])).astype(int)
    df["compressed_but_not_bullish"] = (
        (df["ma_short_spread"] < 0.035) & (df["ma_pair_up_count"] <= 1)
    ).astype(int)
    df["high_convergence_failure_raw"] = (
        ((0.035 - df["ma_short_spread"]).clip(lower=0) / 0.035)
        * ((0.25 - df["close_vs_ma5"]).clip(lower=0) / 0.25)
        * (0.5 + df["weak_close"].clip(0, 1))
    )
    return df


def compare_groups(merged: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "close_vs_ma5", "close_vs_ma10", "close_vs_ma20", "close_vs_ma60",
        "ma5_vs_ma10", "ma10_vs_ma20", "ma20_vs_ma60",
        "ma_stack_spread", "ma_short_spread",
        "ma_bullish_count", "ma_pair_up_count",
        "ma5_slope1", "ma5_slope3", "ma10_slope1", "ma10_slope3",
        "ma20_slope1", "ma20_slope3", "ma60_slope1", "ma60_slope3",
        "weak_close", "open_to_close", "high_to_close_raw",
        "close_below_ma5_above_ma20", "close_below_ma10_above_ma20",
        "close_below_ma20_above_ma60", "compressed_but_not_bullish",
        "high_convergence_failure_raw",
    ]
    rows = []
    base_pairs = [
        ("remaining_bigloss_after_gate", "winners_top16"),
        ("remaining_bigloss_after_gate", "bigloss_le_7"),
        ("bigloss_le_7", "winners_top16"),
    ]
    for a, b in base_pairs:
        ga = merged[merged["group"] == a]
        gb = merged[merged["group"] == b]
        for col in metric_cols:
            xa = pd.to_numeric(ga[col], errors="coerce")
            xb = pd.to_numeric(gb[col], errors="coerce")
            rows.append({
                "group_a": a,
                "group_b": b,
                "metric": col,
                "mean_a": float(xa.mean()) if xa.notna().any() else np.nan,
                "mean_b": float(xb.mean()) if xb.notna().any() else np.nan,
                "mean_diff": float(xa.mean() - xb.mean()) if xa.notna().any() and xb.notna().any() else np.nan,
                "cohens_d": cohens_d(xa, xb),
                "rank_biserial": rank_biserial(xa, xb),
            })
    out = pd.DataFrame(rows)
    out["abs_cohens_d"] = out["cohens_d"].abs()
    out = out.sort_values(["group_a", "group_b", "abs_cohens_d"], ascending=[True, True, False])
    return out


def scan_candidate_rules(all_trades_merged: pd.DataFrame, remaining_pairs: set[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    base = all_trades_merged.copy().drop_duplicates(subset=["stock_code", "prediction_date"])
    base["is_bigloss5"] = (base["actual_return"] <= -0.05).astype(int)
    base["is_bigloss7"] = (base["actual_return"] <= -0.07).astype(int)
    base["is_remaining_bigloss"] = [
        (str(code), pd.Timestamp(pred_date)) in remaining_pairs
        for code, pred_date in zip(base["stock_code"], pd.to_datetime(base["prediction_date"]))
    ]

    rules = []
    candidates = [
        ("ma_short_spread", "<", [0.02, 0.025, 0.03, 0.035]),
        ("ma_bullish_count", "<=", [1, 2]),
        ("ma_pair_up_count", "<=", [0, 1]),
        ("close_vs_ma5", "<", [0.0, 0.01, 0.02]),
        ("weak_close", ">", [0.65, 0.75, 0.85]),
        ("high_convergence_failure_raw", ">", [0.10, 0.15, 0.20, 0.25]),
    ]

    def flag(df: pd.DataFrame, col: str, op: str, thr: float) -> pd.Series:
        x = pd.to_numeric(df[col], errors="coerce")
        if op == "<":
            return x < thr
        if op == "<=":
            return x <= thr
        if op == ">":
            return x > thr
        raise ValueError(op)

    base_cum = float(np.prod(1 + base["actual_return"].fillna(0).to_numpy()) - 1)

    for col, op, thresholds in candidates:
        for thr in thresholds:
            mask = flag(base, col, op, thr).fillna(False)
            kept = base.loc[~mask].copy()
            if kept.empty:
                continue
            rules.append({
                "rule": f"{col} {op} {thr}",
                "flagged": int(mask.sum()),
                "flagged_ratio": float(mask.mean()),
                "flagged_mean_return": float(base.loc[mask, "actual_return"].mean()) if mask.any() else np.nan,
                "bigloss5_hit": int(base.loc[mask, "is_bigloss5"].sum()),
                "bigloss7_hit": int(base.loc[mask, "is_bigloss7"].sum()),
                "remaining_bigloss_hit": int(base.loc[mask, "is_remaining_bigloss"].sum()),
                "remaining_bigloss5": int(kept["is_bigloss5"].sum()),
                "remaining_bigloss7": int(kept["is_bigloss7"].sum()),
                "cum_return": float(np.prod(1 + kept["actual_return"].fillna(0).to_numpy()) - 1),
                "cum_delta_vs_base": float((np.prod(1 + kept["actual_return"].fillna(0).to_numpy()) - 1) - base_cum),
            })
    return pd.DataFrame(rules).sort_values(["remaining_bigloss7", "remaining_bigloss5", "flagged_ratio"])


def main() -> None:
    trades = load_baseline_trades()
    groups = build_groups(trades)
    all_trades = build_all_baseline_with_prediction_dates(trades)
    daily = load_daily_with_ma()
    merged = groups.merge(
        daily,
        left_on=["stock_code", "prediction_date"],
        right_on=["stock_code", "date"],
        how="left",
    )
    merged_all = all_trades.merge(
        daily,
        left_on=["stock_code", "prediction_date"],
        right_on=["stock_code", "date"],
        how="left",
    )
    remaining_pairs = {
        (str(row.stock_code), pd.Timestamp(row.prediction_date))
        for row in groups[groups["group"] == "remaining_bigloss_after_gate"][["stock_code", "prediction_date"]].itertuples(index=False)
    }

    diff = compare_groups(merged)
    rules = scan_candidate_rules(merged_all, remaining_pairs)

    ts = "20260430_235500"
    diff_path = RESULTS_DIR / f"ma_structure_group_diff_{ts}.csv"
    sample_path = RESULTS_DIR / f"ma_structure_group_samples_{ts}.csv"
    rules_path = RESULTS_DIR / f"ma_structure_candidate_rules_{ts}.csv"
    diff.to_csv(diff_path, index=False, encoding="utf-8-sig")
    merged.to_csv(sample_path, index=False, encoding="utf-8-sig")
    rules.to_csv(rules_path, index=False, encoding="utf-8-sig")
    print(diff_path)
    print(sample_path)
    print(rules_path)


if __name__ == "__main__":
    main()
