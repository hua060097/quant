from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = ROOT.parent / "local_data"
DAILY_DIR = LOCAL_DATA / "features" / "daily"
RESULTS = ROOT / "results"
FACTORS = ROOT / "factor_lab_data" / "factors.parquet"
TAG = "20260520_210454"

SELECTIONS_FILE = RESULTS / f"walkforward_selections_{TAG}.json"
SHADOW_FILE = RESULTS / f"walkforward_shadow_skipped_opportunities_{TAG}.json"

V3_COLS = [
    "runup_ma5_exhaustion",
    "cpv_factor",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "sector_leadership_persistence_v3",
    "return_20d_intra_z",
    "market_liquidity_change",
    "ulcer_index",
    "momentum_peak_decay_3d",
    "intra_sector_momentum_rank",
    "mkt_dispersion_shock_v2",
    "market_advance_decline_ratio",
    "limit_down_3d",
    "top10_10d_strength_level_v3",
    "buy_sm_amount_rate",
    "prev_non_first_limit_up_performance_percentile_20d",
    "kdj_j_9",
    "score_z_in_day",
    "reg_severe_up_10d_ratio_pct",
    "price_ma5_ratio_raw",
    "ma_ribbon_expansion_3d",
    "ret_3d_vs_10d_accel",
    "tightness_3d",
    "true_turnover_10d",
    "close_below_ma5_after_runup",
]

COMBO_FACTOR_COLS = [
    "reg_severe_up_10d_ratio_pct",
    "return_20d_intra_z",
    "runup_ma5_exhaustion",
    "prev_day_blowoff_risk",
    "cpv_factor",
    "ma_ribbon_expansion_3d",
    "ret_3d_vs_10d_accel",
    "tightness_3d",
    "true_turnover_10d",
    "close_below_ma5_after_runup",
    "price_ma5_ratio_raw",
]


def load_json_records(path: Path) -> pd.DataFrame:
    obj = json.load(open(path, encoding="utf-8"))
    records: list[dict] = []

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if "stock_code" in value and any(
                key in value
                for key in ("actual_return", "profit_loss_pct", "buy_datetime", "date", "trade_date")
            ):
                records.append(value)
            else:
                for item in value.values():
                    walk(item)

    walk(obj)
    return pd.DataFrame(records)


def normalize_events(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        for col in (
            "trade_date",
            "buy_date",
            "selection_date",
            "buy_datetime",
            "shadow_attempt_buy_date",
            "shadow_prediction_date",
        ):
            if col in df.columns:
                df["date"] = df[col]
                break
    if "actual_return" not in df.columns:
        for col in (
            "return",
            "trade_return",
            "future_return",
            "future_return_5d",
            "final_return",
            "realized_return",
            "profit_loss_pct",
        ):
            if col in df.columns:
                df["actual_return"] = df[col]
                break

    keep = [c for c in df.columns if c in {"date", "stock_code", "actual_return", "score_z_in_day"}]
    out = df[keep].copy()
    out["source"] = source
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock_code"] = out["stock_code"].astype(str)
    out["actual_return"] = pd.to_numeric(out["actual_return"], errors="coerce")
    return out.dropna(subset=["date", "stock_code", "actual_return"])


def load_events() -> pd.DataFrame:
    actual = normalize_events(load_json_records(SELECTIONS_FILE), "actual")
    shadow = normalize_events(load_json_records(SHADOW_FILE), "shadow")
    return (
        pd.concat([actual, shadow], ignore_index=True)
        .drop_duplicates(["source", "date", "stock_code", "actual_return"])
        .reset_index(drop=True)
    )


def add_factor_dates(events: pd.DataFrame) -> pd.DataFrame:
    factor_dates = pd.read_parquet(FACTORS, columns=["date"])["date"].drop_duplicates().sort_values()
    trading_days = np.array(pd.to_datetime(factor_dates))
    prev_dates = []
    for date in events["date"]:
        pos = np.searchsorted(trading_days, np.datetime64(pd.Timestamp(date).normalize()), side="left") - 1
        prev_dates.append(pd.Timestamp(trading_days[pos]) if pos >= 0 else pd.NaT)

    out = events.copy()
    out["factor_date"] = prev_dates
    return out


def load_factor_snapshots(events: pd.DataFrame) -> pd.DataFrame:
    schema_cols = set(pq.read_schema(FACTORS).names)
    cols = ["date", "stock_code"] + [c for c in V3_COLS if c in schema_cols]
    factors = pd.read_parquet(FACTORS, columns=cols)
    factors["date"] = pd.to_datetime(factors["date"])
    needed_dates = set(events["factor_date"].dropna())
    factors = factors[factors["date"].isin(needed_dates)]
    factors = factors.rename(columns={"date": "factor_date"})
    return events.merge(factors, on=["factor_date", "stock_code"], how="left")


def load_daily_for_touch(events: pd.DataFrame, lookback_files: int = 20) -> pd.DataFrame:
    needed_dates = sorted(pd.Timestamp(d).normalize() for d in events["factor_date"].dropna().unique())
    if not needed_dates:
        raise RuntimeError("No factor dates available.")

    all_files = []
    for path in DAILY_DIR.glob("*.parquet"):
        try:
            file_date = pd.to_datetime(path.stem, format="%Y%m%d")
        except ValueError:
            continue
        all_files.append((file_date, path))
    all_files.sort(key=lambda x: x[0])

    min_date = min(needed_dates)
    max_date = max(needed_dates)
    start_idx = max(0, next(i for i, (d, _) in enumerate(all_files) if d >= min_date) - lookback_files)
    selected = [(d, p) for d, p in all_files[start_idx:] if d <= max_date]

    frames = []
    for _, path in selected:
        df = pd.read_parquet(path, columns=["date", "stock_code", "high", "low", "close"])
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No daily parquet files found in {DAILY_DIR}")

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["stock_code"] = daily["stock_code"].astype(str)
    daily = daily.drop_duplicates(["stock_code", "date"], keep="last")
    daily = daily.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return daily


def compute_ma5_touch_counts(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily[["date", "stock_code", "high", "low", "close"]].copy()
    grouped = out.groupby("stock_code", sort=False)
    out["ma5"] = grouped["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    out["low_ma5_gap"] = out["low"] / out["ma5"] - 1.0
    out["ma5_touched"] = (
        out["ma5"].notna()
        & out["low"].notna()
        & out["high"].notna()
        & (out["low"] <= out["ma5"])
        & (out["high"] >= out["ma5"])
    ).astype("float32")
    out["ma5_touch_count_5d"] = grouped["ma5_touched"].transform(lambda x: x.rolling(5, min_periods=1).sum())
    out["ma5_touch_count_10d"] = grouped["ma5_touched"].transform(lambda x: x.rolling(10, min_periods=1).sum())
    out["ma5_pullback_depth_10d"] = grouped["low_ma5_gap"].transform(
        lambda x: x.rolling(10, min_periods=5).min()
    )
    out["ma5_air_gap_mean_10d"] = grouped["low_ma5_gap"].transform(
        lambda x: x.clip(lower=0).rolling(10, min_periods=1).mean()
    )
    return out[
        [
            "date",
            "stock_code",
            "ma5",
            "ma5_touched",
            "ma5_touch_count_5d",
            "ma5_touch_count_10d",
            "ma5_pullback_depth_10d",
            "ma5_air_gap_mean_10d",
        ]
    ]


def attach_touch_features(events: pd.DataFrame, touch: pd.DataFrame) -> pd.DataFrame:
    snap = touch.rename(columns={"date": "factor_date"})
    out = events.merge(snap, on=["factor_date", "stock_code"], how="left")
    return add_ma5_break_rebound_accel_risk(out)


def add_ma5_break_rebound_accel_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pullback = pd.to_numeric(out["ma5_pullback_depth_10d"], errors="coerce")
    accel = pd.to_numeric(out["ret_3d_vs_10d_accel"], errors="coerce")
    break_score = 1.0 / (1.0 + np.exp(-((-pullback - 0.04) / 0.015)))
    accel_score = 1.0 / (1.0 + np.exp(-((accel - 4.0) / 0.8)))
    out["ma5_break_rebound_accel_risk"] = (break_score * accel_score).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out


def current_v3_mask(df: pd.DataFrame, include_reg_pct_rule: bool = False) -> pd.Series:
    r1 = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    r2 = (
        ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
        | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
        | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
    )
    r3 = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    r4 = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.80)
        | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67)
    )
    r5 = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    r6 = df["limit_down_3d"] == 1
    r7 = (df["top10_10d_strength_level_v3"] >= 0.70) & (df["buy_sm_amount_rate"] <= -1.25)
    r8 = (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80) & (df["kdj_j_9"] <= 0.75)
    r9 = df.get("score_z_in_day", pd.Series(np.nan, index=df.index)) <= 1.50
    mask = r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9
    if include_reg_pct_rule and "reg_severe_up_10d_ratio_pct" in df.columns:
        mask = mask | (
            (df["reg_severe_up_10d_ratio_pct"] >= 0.80)
            & (df["market_advance_decline_ratio"] <= 0.45)
        )
    return mask.fillna(False)


def bucket_counts(df: pd.DataFrame) -> dict:
    r = df["actual_return"]
    return {
        "n": int(len(df)),
        "mean": float(r.mean()) if len(df) else np.nan,
        "big_loss": int((r <= -0.04).sum()),
        "small_loss": int(((r > -0.04) & (r < 0)).sum()),
        "flat_0_3": int(((r >= 0) & (r < 0.03)).sum()),
        "small_gain_3_8": int(((r >= 0.03) & (r < 0.08)).sum()),
        "big_gain_8": int((r >= 0.08).sum()),
        "big_gain_10": int((r >= 0.10).sum()),
        "actual_big_loss": int(((df["source"] == "actual") & (r <= -0.04)).sum()),
        "actual_big_gain": int(((df["source"] == "actual") & (r >= 0.08)).sum()),
        "shadow_big_loss": int(((df["source"] == "shadow") & (r <= -0.04)).sum()),
        "shadow_big_gain": int(((df["source"] == "shadow") & (r >= 0.08)).sum()),
    }


def eval_rule(df: pd.DataFrame, label: str, mask: pd.Series, baseline: pd.Series) -> dict:
    mask = mask.fillna(False)
    inc = mask & ~baseline
    hit = df[mask]
    inc_rows = df[inc]
    result = {
        "rule": label,
        "standalone_n": int(mask.sum()),
        "overlap": int((mask & baseline).sum()),
    }
    for prefix, values in (("standalone", bucket_counts(hit)), ("inc", bucket_counts(inc_rows))):
        for key, value in values.items():
            result[f"{prefix}_{key}"] = value
    result["score"] = (
        result["inc_big_loss"] * 3
        + result["inc_small_loss"]
        - result["inc_big_gain_8"] * 5
        - result["inc_small_gain_3_8"]
    )
    return result


def scan_rules(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    touch_specs = [
        ("ma5_touch_count_5d", [0, 1, 2, 3, 4, 5]),
        ("ma5_touch_count_10d", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ]
    raw_thresholds = [-2, 0, 1, 2, 3, 5, 7, 9, 10, 12, 15]

    for col, thresholds in touch_specs:
        for threshold in thresholds:
            rows.append(eval_rule(df, f"{col}>={threshold}", df[col] >= threshold, baseline))
            rows.append(eval_rule(df, f"{col}<={threshold}", df[col] <= threshold, baseline))
            for raw_t in raw_thresholds:
                rows.append(
                    eval_rule(
                        df,
                        f"price_ma5_ratio_raw>={raw_t:g} & {col}>={threshold}",
                        (df["price_ma5_ratio_raw"] >= raw_t) & (df[col] >= threshold),
                        baseline,
                    )
                )
                rows.append(
                    eval_rule(
                        df,
                        f"price_ma5_ratio_raw>={raw_t:g} & {col}<={threshold}",
                        (df["price_ma5_ratio_raw"] >= raw_t) & (df[col] <= threshold),
                        baseline,
                    )
                )

    return pd.DataFrame(rows).query("inc_n > 0").sort_values(
        ["score", "inc_big_loss", "inc_big_gain_8", "inc_n"],
        ascending=[False, False, True, True],
    )


def factor_thresholds(df: pd.DataFrame, col: str) -> list[float]:
    values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return []

    qs = values.quantile([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    thresholds = [float(x) for x in qs.tolist() if pd.notna(x)]

    fixed = {
        "reg_severe_up_10d_ratio_pct": [0.70, 0.75, 0.80, 0.85, 0.90],
        "runup_ma5_exhaustion": [0.6, 0.7, 0.8, 0.9],
        "prev_day_blowoff_risk": [1.0, 2.0, 3.0, 5.0],
        "cpv_factor": [-0.8, -0.6, -0.4, -0.2, 0.0],
        "return_20d_intra_z": [0.0, 0.2, 0.4, 0.6, 0.8],
        "price_ma5_ratio_raw": [5.0, 7.0, 9.0, 10.0, 12.0, 15.0],
        "close_below_ma5_after_runup": [0.0, 0.5, 1.0],
    }
    thresholds.extend(fixed.get(col, []))
    return sorted({round(x, 10) for x in thresholds})


def scan_touch_factor_combos(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    touch_specs = [
        ("ma5_touch_count_5d", range(0, 6)),
        ("ma5_touch_count_10d", range(0, 11)),
    ]

    for touch_col, touch_thresholds in touch_specs:
        for touch_threshold in touch_thresholds:
            for touch_op in (">=", "<="):
                touch_mask = (
                    df[touch_col] >= touch_threshold
                    if touch_op == ">="
                    else df[touch_col] <= touch_threshold
                )
                for factor_col in COMBO_FACTOR_COLS:
                    if factor_col not in df.columns:
                        continue
                    for factor_threshold in factor_thresholds(df, factor_col):
                        for factor_op in (">=", "<="):
                            factor_mask = (
                                df[factor_col] >= factor_threshold
                                if factor_op == ">="
                                else df[factor_col] <= factor_threshold
                            )
                            label = (
                                f"{touch_col}{touch_op}{touch_threshold}"
                                f" & {factor_col}{factor_op}{factor_threshold:g}"
                            )
                            stat = eval_rule(df, label, touch_mask & factor_mask, baseline)
                            stat.update(
                                {
                                    "touch_col": touch_col,
                                    "touch_op": touch_op,
                                    "touch_threshold": touch_threshold,
                                    "factor_col": factor_col,
                                    "factor_op": factor_op,
                                    "factor_threshold": factor_threshold,
                                }
                            )
                            rows.append(stat)

    result = pd.DataFrame(rows)
    return result.query("inc_n > 0").sort_values(
        ["score", "inc_big_loss", "inc_big_gain_8", "inc_n"],
        ascending=[False, False, True, True],
    )


def combo_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    touch = (
        df[row["touch_col"]] >= row["touch_threshold"]
        if row["touch_op"] == ">="
        else df[row["touch_col"]] <= row["touch_threshold"]
    )
    factor = (
        df[row["factor_col"]] >= row["factor_threshold"]
        if row["factor_op"] == ">="
        else df[row["factor_col"]] <= row["factor_threshold"]
    )
    return touch & factor


def fixed_bucket_summary(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    specs = [
        (
            "ma5_pullback_depth_10d",
            [-np.inf, -0.08, -0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, np.inf],
            ["<=-8%", "-8~-5%", "-5~-3%", "-3~-2%", "-2~-1%", "-1~0%", "0~1%", "1~2%", ">2%"],
        ),
        (
            "ma5_air_gap_mean_10d",
            [-np.inf, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, np.inf],
            ["=0", "0~0.5%", "0.5~1%", "1~2%", "2~3%", "3~5%", "5~8%", ">8%"],
        ),
        (
            "ma5_break_rebound_accel_risk",
            [-np.inf, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, np.inf],
            ["<=0.05", "0.05~0.10", "0.10~0.20", "0.20~0.35", "0.35~0.50", "0.50~0.70", "0.70~0.85", ">0.85"],
        ),
    ]
    rows = []
    for col, bins, labels in specs:
        bucket = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=True)
        for label in labels:
            mask = bucket == label
            inc = mask & ~baseline
            row = {
                "feature": col,
                "bucket": label,
                "min_value": float(df.loc[mask, col].min()) if mask.any() else np.nan,
                "max_value": float(df.loc[mask, col].max()) if mask.any() else np.nan,
                "baseline_overlap": int((mask & baseline).sum()),
            }
            for prefix, sub in (("all", df[mask]), ("inc", df[inc])):
                counts = bucket_counts(sub)
                for key, value in counts.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def scan_fixed_continuous_rules(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    single_specs = {
        "ma5_pullback_depth_10d": [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02],
        "ma5_air_gap_mean_10d": [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        "ma5_break_rebound_accel_risk": [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85],
    }
    combo_specs = {
        "price_ma5_ratio_raw": [5.0, 7.0, 9.0, 10.0, 12.0],
        "reg_severe_up_10d_ratio_pct": [0.80, 0.85, 0.90, 0.95, 0.98],
        "return_20d_intra_z": [0.4, 0.5, 0.6, 0.7],
        "prev_day_blowoff_risk": [2.0, 3.0, 5.0, 8.0],
        "ret_3d_vs_10d_accel": [3.0, 5.0, 6.0],
    }

    for col, thresholds in single_specs.items():
        for threshold in thresholds:
            rows.append(eval_rule(df, f"{col}>={threshold:g}", df[col] >= threshold, baseline))
            rows.append(eval_rule(df, f"{col}<={threshold:g}", df[col] <= threshold, baseline))

    for ma_col, ma_thresholds in single_specs.items():
        for ma_threshold in ma_thresholds:
            ma_masks = [
                (f"{ma_col}>={ma_threshold:g}", df[ma_col] >= ma_threshold),
                (f"{ma_col}<={ma_threshold:g}", df[ma_col] <= ma_threshold),
            ]
            for factor_col, factor_thresholds in combo_specs.items():
                if factor_col not in df.columns:
                    continue
                for factor_threshold in factor_thresholds:
                    factor_masks = [
                        (f"{factor_col}>={factor_threshold:g}", df[factor_col] >= factor_threshold),
                        (f"{factor_col}<={factor_threshold:g}", df[factor_col] <= factor_threshold),
                    ]
                    for ma_label, ma_mask in ma_masks:
                        for factor_label, factor_mask in factor_masks:
                            rows.append(
                                eval_rule(
                                    df,
                                    f"{ma_label} & {factor_label}",
                                    ma_mask & factor_mask,
                                    baseline,
                                )
                            )

    return pd.DataFrame(rows).query("inc_n > 0").sort_values(
        ["score", "inc_big_loss", "inc_big_gain_8", "inc_n"],
        ascending=[False, False, True, True],
    )


def print_details(
    df: pd.DataFrame,
    label: str,
    mask: pd.Series,
    baseline: pd.Series,
    extra_cols: list[str] | None = None,
) -> None:
    inc = df[mask.fillna(False) & ~baseline].copy()
    cols = [
        "source",
        "date",
        "stock_code",
        "actual_return",
        "factor_date",
        "price_ma5_ratio_raw",
        "ma5_touch_count_5d",
        "ma5_touch_count_10d",
        "ma5_pullback_depth_10d",
        "ma5_air_gap_mean_10d",
        "ma5_break_rebound_accel_risk",
        "ma5_touched",
        "ma5",
    ]
    if extra_cols:
        cols.extend(c for c in extra_cols if c in df.columns and c not in cols)
    print(f"\nDETAIL {label} inc_n={len(inc)}")
    print(inc[cols].sort_values("actual_return").head(20).to_string(index=False))
    bg = inc[inc["actual_return"] >= 0.08].sort_values("actual_return", ascending=False)
    if not bg.empty:
        print("BIG_GAIN_EXAMPLES")
        print(bg[cols].head(10).to_string(index=False))


def main() -> None:
    events = add_factor_dates(load_events())
    events = load_factor_snapshots(events)
    daily = load_daily_for_touch(events)
    touch = compute_ma5_touch_counts(daily)
    events = attach_touch_features(events, touch)

    baseline = current_v3_mask(events, include_reg_pct_rule=False)
    baseline_with_reg = current_v3_mask(events, include_reg_pct_rule=True)

    print(f"events={len(events)}, actual={(events['source'] == 'actual').sum()}, shadow={(events['source'] == 'shadow').sum()}")
    print(f"baseline_r1_r9={bucket_counts(events[baseline])}")
    print(f"baseline_r1_r9_plus_reg_pct={bucket_counts(events[baseline_with_reg])}")

    for baseline_name, base_mask in (
        ("baseline_r1_r9", baseline),
        ("baseline_plus_reg_pct", baseline_with_reg),
    ):
        print("\n" + "=" * 100)
        print(f"SCAN {baseline_name}")
        result = scan_rules(events, base_mask)
        show_cols = [
            "rule",
            "standalone_n",
            "overlap",
            "inc_n",
            "inc_mean",
            "inc_big_loss",
            "inc_small_loss",
            "inc_small_gain_3_8",
            "inc_big_gain_8",
            "inc_big_gain_10",
            "inc_actual_big_loss",
            "inc_actual_big_gain",
            "inc_shadow_big_loss",
            "inc_shadow_big_gain",
            "score",
        ]
        print("\nTOP SCORE")
        print(result[show_cols].head(40).to_string(index=False))

        zero_bg = result[(result["inc_big_gain_8"] == 0) & (result["inc_big_loss"] > 0)]
        print("\nZERO BIG GAIN WITH BIG LOSS")
        if zero_bg.empty:
            print("NONE")
        else:
            print(zero_bg[show_cols].head(30).to_string(index=False))

        if not zero_bg.empty:
            for rule in zero_bg["rule"].head(5):
                if " & " in rule:
                    left, right = rule.split(" & ")
                    raw_t = float(left.split(">=")[1])
                    col, op_threshold = right.split(">=") if ">=" in right else right.split("<=")
                    op = ">=" if ">=" in right else "<="
                    threshold = float(op_threshold)
                    touch_mask = events[col] >= threshold if op == ">=" else events[col] <= threshold
                    mask = (events["price_ma5_ratio_raw"] >= raw_t) & touch_mask
                else:
                    if ">=" in rule:
                        col, threshold = rule.split(">=")
                        mask = events[col] >= float(threshold)
                    else:
                        col, threshold = rule.split("<=")
                        mask = events[col] <= float(threshold)
                print_details(events, rule, mask, base_mask)

        out = RESULTS / f"ma5_touch_rule_scan_{TAG}_{baseline_name}.csv"
        result.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nsaved: {out}")

        combo = scan_touch_factor_combos(events, base_mask)
        combo_out = RESULTS / f"ma5_touch_factor_combo_scan_{TAG}_{baseline_name}.csv"
        combo.to_csv(combo_out, index=False, encoding="utf-8-sig")
        print("\nTOUCH + FACTOR COMBO TOP SCORE")
        print(combo[show_cols].head(50).to_string(index=False))

        combo_zero_bg = combo[(combo["inc_big_gain_8"] == 0) & (combo["inc_big_loss"] > 0)]
        print("\nTOUCH + FACTOR COMBO ZERO BIG GAIN WITH BIG LOSS")
        if combo_zero_bg.empty:
            print("NONE")
        else:
            print(combo_zero_bg[show_cols].head(40).to_string(index=False))
            for _, row in combo_zero_bg.head(5).iterrows():
                print_details(
                    events,
                    row["rule"],
                    combo_mask(events, row),
                    base_mask,
                    extra_cols=[row["factor_col"]],
                )
        print(f"\nsaved: {combo_out}")

        bucket = fixed_bucket_summary(events, base_mask)
        bucket_out = RESULTS / f"ma5_continuous_bucket_summary_{TAG}_{baseline_name}.csv"
        bucket.to_csv(bucket_out, index=False, encoding="utf-8-sig")
        bucket_show = [
            "feature",
            "bucket",
            "all_n",
            "all_mean",
            "all_big_loss",
            "all_big_gain_8",
            "inc_n",
            "inc_mean",
            "inc_big_loss",
            "inc_small_loss",
            "inc_small_gain_3_8",
            "inc_big_gain_8",
            "inc_big_gain_10",
            "baseline_overlap",
        ]
        print("\nMA5 CONTINUOUS FIXED BUCKETS")
        print(bucket[bucket_show].to_string(index=False))
        print(f"\nsaved: {bucket_out}")

        fixed_rules = scan_fixed_continuous_rules(events, base_mask)
        fixed_out = RESULTS / f"ma5_continuous_fixed_rule_scan_{TAG}_{baseline_name}.csv"
        fixed_rules.to_csv(fixed_out, index=False, encoding="utf-8-sig")
        print("\nMA5 CONTINUOUS FIXED RULE TOP SCORE")
        print(fixed_rules[show_cols].head(50).to_string(index=False))

        fixed_zero_bg = fixed_rules[(fixed_rules["inc_big_gain_8"] == 0) & (fixed_rules["inc_big_loss"] > 0)]
        print("\nMA5 CONTINUOUS FIXED RULE ZERO BIG GAIN WITH BIG LOSS")
        if fixed_zero_bg.empty:
            print("NONE")
        else:
            print(fixed_zero_bg[show_cols].head(40).to_string(index=False))
        print(f"\nsaved: {fixed_out}")


if __name__ == "__main__":
    main()
