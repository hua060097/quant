from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import analyze_new_10_factors_risk_filter_114 as base


BASELINE_LOG = Path("stock_selection_test_mode_walkforward_分类模型_原因子95.log")
DEFAULT_LOG = Path("stock_selection_test_mode_walkforward_分类模型_原因子133_optuna.log")
FACTOR_PATH = Path("factor_lab_data/factors.parquet")

NEW_FACTORS = [
    "stock_volume_ratio_20d",
    "high_position_volume_surge",
    "upper_shadow_pressure_5d",
    "lower_shadow_absorption_5d",
    "shadow_reversal_volume_pressure",
    "prior_high_distance_20d",
    "prior_high_proximity_20d",
    "prior_high_volume_contest_20d",
    "market_advance_decline_ratio",
    "market_advance_decline_ratio_percentile_20d_exp",
]

CURRENT_FACTORS = [
    "runup_ma5_exhaustion",
    "top10_10d_strength_level_v3",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "close_below_ma5_after_runup",
    "turnover_z_20",
]

DISPLAY_COLS = [
    "rule",
    "blocked",
    "blocked_rate",
    "blocked_big_loss",
    "blocked_big_gain",
    "blocked_small_loss",
    "blocked_small_gain",
    "big_loss_capture",
    "big_gain_damage",
    "blocked_return_sum",
    "kept_return_sum",
    "score",
]


def load_trades_with_factors(log_path: Path, factor_names: list[str]) -> pd.DataFrame:
    trades = base.parse_trades(log_path)
    factors, available, missing = base.load_factor_slice(
        factor_names,
        trades["buy_date"].min(),
        trades["buy_date"].max(),
    )
    trades = base.attach_prediction_date(trades, factors["date"])
    merged = trades.merge(
        factors,
        left_on=["pred_date", "stock_code"],
        right_on=["date", "stock_code"],
        how="left",
    )
    merged.attrs["available_factors"] = available
    merged.attrs["missing_factors"] = missing
    return merged


def score_rule(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    r = base.eval_mask(df, mask, name)
    # Conservative score: big-gain damage and broad blocking are penalized hard.
    r["score"] = (
        r["blocked_big_loss"] * 4.0
        - r["blocked_big_gain"] * 7.0
        - r["blocked_small_gain"] * 0.8
        - max(0.0, r["blocked_rate"] - 0.30) * 18.0
    )
    return r


def current_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    runup = (df["runup_ma5_exhaustion"] > 0.8) & (df["top10_10d_strength_level_v3"] >= 0.5)
    blowoff = (df["prev_day_blowoff_risk"] >= 4.15) & (df["sector_momentum_accel_rank"] <= 0.41)
    return {
        "runup_branch_default": runup,
        "blowoff_branch_default": blowoff,
        "runup_or_blowoff_default": runup | blowoff,
        "runup_only_no_top10": df["runup_ma5_exhaustion"] > 0.8,
    }


def threshold_specs(df: pd.DataFrame, factors: list[str]) -> list[dict]:
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    specs = []
    for factor in factors:
        if factor not in df.columns:
            continue
        values = df[factor].dropna()
        if len(values) < 30:
            continue
        for q in quantiles:
            threshold = float(values.quantile(q))
            for direction in (">=", "<="):
                specs.append({
                    "factor": factor,
                    "direction": direction,
                    "threshold": threshold,
                    "q": q,
                })
    return specs


def make_mask(df: pd.DataFrame, spec: dict) -> pd.Series:
    if spec["direction"] == ">=":
        return df[spec["factor"]] >= spec["threshold"]
    return df[spec["factor"]] <= spec["threshold"]


def spec_name(spec: dict) -> str:
    return f"{spec['factor']} {spec['direction']} {spec['threshold']:.6g} (q={spec['q']:.2f})"


def as_table(rows: list[dict] | pd.DataFrame, n: int | None = None) -> str:
    df = pd.DataFrame(rows) if isinstance(rows, list) else rows.copy()
    if df.empty:
        return "none"
    if n is not None:
        df = df.head(n)
    keep = [c for c in DISPLAY_COLS if c in df.columns]
    return df[keep].to_string(index=False)


def search_single_rules(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for spec in threshold_specs(df, factors):
        mask = make_mask(df, spec)
        r = score_rule(df, mask, spec_name(spec))
        if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.45:
            r.update(spec)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def search_replacements_for_top10(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    runup = df["runup_ma5_exhaustion"] > 0.8
    for spec in threshold_specs(df, factors):
        mask = runup & make_mask(df, spec)
        r = score_rule(df, mask, f"runup_ma5_exhaustion>0.8 AND {spec_name(spec)}")
        if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.25:
            r.update(spec)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def search_with_default(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    cur = current_masks(df)["runup_or_blowoff_default"]
    rows = []
    specs = threshold_specs(df, factors)
    for spec in specs:
        mask = make_mask(df, spec)
        for op, combo in [("OR", cur | mask), ("AND", cur & mask)]:
            r = score_rule(df, combo, f"runup_or_blowoff {op} {spec_name(spec)}")
            if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.50:
                r.update(spec)
                r["op"] = op
                rows.append(r)

    single_out = pd.DataFrame(rows)
    if not single_out.empty:
        best_specs = single_out.sort_values("score", ascending=False).head(25)[
            ["factor", "direction", "threshold", "q"]
        ].drop_duplicates().to_dict("records")
        for a, b in itertools.combinations(best_specs, 2):
            add_on = make_mask(df, a) & make_mask(df, b)
            r = score_rule(df, cur | add_on, f"runup_or_blowoff OR ({spec_name(a)} AND {spec_name(b)})")
            if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.50:
                rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def pair_search(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    single = search_single_rules(df, factors)
    if single.empty:
        return single
    specs = single.head(40)[["factor", "direction", "threshold", "q"]].drop_duplicates().to_dict("records")
    rows = []
    for a, b in itertools.combinations(specs, 2):
        if a["factor"] == b["factor"]:
            continue
        for op, mask in [
            ("AND", make_mask(df, a) & make_mask(df, b)),
            ("OR", make_mask(df, a) | make_mask(df, b)),
        ]:
            r = score_rule(df, mask, f"({spec_name(a)}) {op} ({spec_name(b)})")
            if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.50:
                rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def overlap_rows(df: pd.DataFrame, candidate_mask: pd.Series) -> list[dict]:
    masks = current_masks(df)
    rows = []
    for name, mask in masks.items():
        if name == "runup_only_no_top10":
            continue
        rows.append(score_rule(df, candidate_mask & mask, f"candidate AND {name}"))
        rows.append(score_rule(df, candidate_mask & ~mask, f"candidate NOT {name}"))
    return rows


def print_detail(df: pd.DataFrame, mask: pd.Series, title: str, factors: list[str], limit: int = 80) -> None:
    cols = ["stock_code", "buy_date", "pred_date", "return_pct", "group"] + factors + ["sell_reason"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + title)
    print(score_rule(df, mask, title))
    print(df.loc[mask, cols].sort_values("return_pct").head(limit).to_string(index=False))


def main() -> None:
    schema = pq.read_schema(FACTOR_PATH)
    all_needed = NEW_FACTORS + CURRENT_FACTORS
    print("factor availability")
    for factor in all_needed:
        print(f"{factor}: {factor in schema.names}")

    df = load_trades_with_factors(BASELINE_LOG, all_needed)
    df_default = load_trades_with_factors(DEFAULT_LOG, all_needed)
    usable_new = [f for f in NEW_FACTORS if f in df.columns and df[f].notna().any()]

    print("\nbaseline log")
    print(f"log={BASELINE_LOG.name}, trades={len(df)}, return_sum={df['return_pct'].sum():.4f}")
    print(df["group"].value_counts().to_string())
    print("\ndefault run log")
    print(f"log={DEFAULT_LOG.name}, trades={len(df_default)}, return_sum={df_default['return_pct'].sum():.4f}")
    print(df_default["group"].value_counts().to_string())

    print("\nusable new/market factors")
    print(usable_new)

    group_summary = base.summarize_groups(df, usable_new)
    print("\nfactor group summary")
    print(group_summary[[
        "factor", "corr", "big_loss_n", "big_gain_n", "big_loss_mean", "big_loss_median",
        "big_gain_mean", "big_gain_median", "small_loss_mean", "small_gain_mean",
        "loss_gain_diff", "nan_rate",
    ]].to_string(index=False))

    print("\ncurrent default semantic rules simulated on baseline")
    current_rows = [score_rule(df, mask, name) for name, mask in current_masks(df).items()]
    print(as_table(current_rows))

    single = search_single_rules(df, usable_new)
    print("\nbest single-factor filters")
    print(as_table(single, 35))

    pairs = pair_search(df, usable_new)
    print("\nbest two-factor filters")
    print(as_table(pairs, 35))

    replacements = search_replacements_for_top10(df, usable_new)
    print("\nbest replacements for top10_10d_strength_level_v3 in runup branch")
    print(as_table(replacements, 40))

    with_default = search_with_default(df, usable_new)
    print("\nbest combinations with current runup_or_blowoff default")
    print(as_table(with_default, 45))

    old_candidate = (
        (df["prior_high_volume_contest_20d"] <= -0.170934)
        & (df["stock_volume_ratio_20d"] >= 1.78077)
    )
    print("\nold threshold candidate after regenerated factors")
    print(as_table([score_rule(
        df,
        old_candidate,
        "prior_high_volume_contest_20d<=-0.170934 AND stock_volume_ratio_20d>=1.78077",
    )]))
    print("\noverlap of old threshold candidate with default branches")
    print(as_table(overlap_rows(df, old_candidate)))

    # Re-optimize only the previously discussed two-factor structure.
    focused = []
    ph_specs = [s for s in threshold_specs(df, ["prior_high_volume_contest_20d"]) if s["direction"] == "<="]
    vol_specs = [s for s in threshold_specs(df, ["stock_volume_ratio_20d"]) if s["direction"] == ">="]
    for a, b in itertools.product(ph_specs, vol_specs):
        mask = make_mask(df, a) & make_mask(df, b)
        r = score_rule(df, mask, f"{spec_name(a)} AND {spec_name(b)}")
        if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.35:
            r.update({"ph_threshold": a["threshold"], "ph_q": a["q"], "vol_threshold": b["threshold"], "vol_q": b["q"]})
            focused.append(r)
    focused_df = pd.DataFrame(focused)
    if not focused_df.empty:
        focused_df = focused_df.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])
    print("\nre-optimized prior_high_volume_contest_20d + stock_volume_ratio_20d")
    print(as_table(focused_df, 25))

    if not focused_df.empty:
        top = focused_df.iloc[0]
        best_mask = (
            (df["prior_high_volume_contest_20d"] <= float(top["ph_threshold"]))
            & (df["stock_volume_ratio_20d"] >= float(top["vol_threshold"]))
        )
        print("\noverlap of re-optimized focused candidate with default branches")
        print(as_table(overlap_rows(df, best_mask)))
        print_detail(
            df,
            best_mask,
            "blocked trades for best focused prior-high-volume candidate",
            [
                "runup_ma5_exhaustion",
                "top10_10d_strength_level_v3",
                "prev_day_blowoff_risk",
                "sector_momentum_accel_rank",
                "prior_high_volume_contest_20d",
                "stock_volume_ratio_20d",
                "market_advance_decline_ratio",
                "market_advance_decline_ratio_percentile_20d_exp",
            ],
        )


if __name__ == "__main__":
    main()
