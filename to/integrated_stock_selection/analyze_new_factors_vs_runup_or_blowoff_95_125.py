import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_new_10_factors_risk_filter_114 as base


LOG95 = Path("stock_selection_test_mode_walkforward_分类模型_原因子95.log")
LOG125 = Path("stock_selection_test_mode_walkforward_分类模型_原因子133_optuna.log")

NEW_FACTORS = [
    "stock_volume_ratio_20d",
    "high_position_volume_surge",
    "upper_shadow_pressure_5d",
    "lower_shadow_absorption_5d",
    "shadow_reversal_volume_pressure",
    "prior_high_distance_20d",
    "prior_high_proximity_20d",
    "prior_high_volume_contest_20d",
]

CURRENT_FACTORS = [
    "runup_ma5_exhaustion",
    "top10_10d_strength_level_v3",
    "prev_day_blowoff_risk",
    "sector_momentum_accel_rank",
    "close_below_ma5_after_runup",
    "turnover_z_20",
]


def load_trades_with_factors(log_path: Path) -> pd.DataFrame:
    trades = base.parse_trades(log_path)
    factors, available, missing = base.load_factor_slice(
        NEW_FACTORS + CURRENT_FACTORS,
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
    return merged


def score_rule(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    r = base.eval_mask(df, mask, name)
    # More conservative than the exploratory script: big-gain damage matters a lot.
    r["score"] = (
        r["blocked_big_loss"] * 4.0
        - r["blocked_big_gain"] * 5.0
        - r["blocked_small_gain"] * 0.8
        - max(0.0, r["blocked_rate"] - 0.30) * 12.0
    )
    return r


def current_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {}
    masks["runup_ma5_only_125_style"] = df["runup_ma5_exhaustion"] > 0.8
    masks["runup_top10_current_semantic"] = (
        (df["runup_ma5_exhaustion"] > 0.8)
        & (df["top10_10d_strength_level_v3"] >= 0.5)
    )
    masks["blowoff_current_semantic"] = (
        (df["prev_day_blowoff_risk"] >= 4.15)
        & (df["sector_momentum_accel_rank"] <= 0.41)
    )
    masks["runup_or_blowoff_current"] = (
        masks["runup_top10_current_semantic"] | masks["blowoff_current_semantic"]
    )
    return masks


def threshold_specs(df: pd.DataFrame, factors: list[str]) -> list[dict]:
    specs = []
    quantiles = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    for f in factors:
        s = df[f]
        valid = s.dropna()
        if len(valid) < 30:
            continue
        for q in quantiles:
            th = float(valid.quantile(q))
            for direction in [">=", "<="]:
                specs.append({"factor": f, "direction": direction, "threshold": th, "q": q})
    return specs


def make_mask(df: pd.DataFrame, spec: dict) -> pd.Series:
    if spec["direction"] == ">=":
        return df[spec["factor"]] >= spec["threshold"]
    return df[spec["factor"]] <= spec["threshold"]


def spec_name(spec: dict) -> str:
    return f"{spec['factor']} {spec['direction']} {spec['threshold']:.6g} (q={spec['q']:.2f})"


def search_top10_replacements(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runup = df["runup_ma5_exhaustion"] > 0.8
    for spec in threshold_specs(df, NEW_FACTORS):
        mask = runup & make_mask(df, spec)
        r = score_rule(df, mask, f"runup_ma5_exhaustion>0.8 AND {spec_name(spec)}")
        if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.35:
            r.update(spec)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def search_additive_filters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cur = current_masks(df)["runup_or_blowoff_current"]
    specs = threshold_specs(df, NEW_FACTORS)
    for spec in specs:
        m = make_mask(df, spec)
        for op, mask in [("OR", cur | m), ("AND", cur & m)]:
            r = score_rule(df, mask, f"runup_or_blowoff {op} {spec_name(spec)}")
            if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.45:
                r.update(spec)
                r["op"] = op
                rows.append(r)
    # Two-factor add-on, restricted to promising specs.
    prelim = pd.DataFrame(rows)
    if not prelim.empty:
        best_specs = prelim.sort_values("score", ascending=False).head(20)[
            ["factor", "direction", "threshold", "q"]
        ].to_dict("records")
        for a, b in itertools.combinations(best_specs, 2):
            ma = make_mask(df, a)
            mb = make_mask(df, b)
            combo = ma & mb
            r = score_rule(df, cur | combo, f"runup_or_blowoff OR ({spec_name(a)} AND {spec_name(b)})")
            if r["blocked_big_loss"] > 0 and r["blocked_rate"] <= 0.45:
                rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "blocked_big_loss", "big_gain_damage"], ascending=[False, False, True])


def print_eval_table(title: str, rows: list[dict]):
    print("\n" + title)
    cols = [
        "rule", "blocked", "blocked_rate", "blocked_big_loss", "blocked_big_gain",
        "blocked_small_loss", "blocked_small_gain", "big_loss_capture", "big_gain_damage",
        "blocked_return_sum", "kept_return_sum", "score",
    ]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


def main():
    df95 = load_trades_with_factors(LOG95)
    df125 = load_trades_with_factors(LOG125)
    print("95 original pool")
    print(f"trades={len(df95)}, return_sum={df95['return_pct'].sum():.4f}")
    print(df95["group"].value_counts().to_string())
    print("\n125 actual filtered run")
    print(f"trades={len(df125)}, return_sum={df125['return_pct'].sum():.4f}")
    print(df125["group"].value_counts().to_string())

    rows = [score_rule(df95, m, n) for n, m in current_masks(df95).items()]
    print_eval_table("Current semantic rules simulated on 95", rows)

    print("\nNew factor group means on 95")
    summary = base.summarize_groups(df95, NEW_FACTORS)
    print(summary[[
        "factor", "corr", "big_loss_mean", "big_gain_mean",
        "small_loss_mean", "small_gain_mean", "loss_gain_diff",
    ]].to_string(index=False))

    repl = search_top10_replacements(df95)
    print("\nBest replacements for top10_10d_strength_level_v3 inside runup branch")
    cols = [
        "rule", "blocked", "blocked_rate", "blocked_big_loss", "blocked_big_gain",
        "blocked_small_loss", "blocked_small_gain", "big_loss_capture", "big_gain_damage",
        "blocked_return_sum", "kept_return_sum", "score",
    ]
    print(repl[cols].head(30).to_string(index=False) if not repl.empty else "none")

    add = search_additive_filters(df95)
    print("\nBest combinations with runup_or_blowoff")
    print(add[cols].head(30).to_string(index=False) if not add.empty else "none")

    # Detail for the most interpretable candidate discussed earlier.
    combo = (
        (df95["prior_high_volume_contest_20d"] <= -0.170934)
        & (df95["stock_volume_ratio_20d"] >= 1.78077)
    )
    detail_cols = [
        "stock_code", "buy_date", "pred_date", "return_pct", "group",
        "runup_ma5_exhaustion", "top10_10d_strength_level_v3",
        "stock_volume_ratio_20d", "prior_high_volume_contest_20d", "sell_reason",
    ]
    print("\nCandidate combo detail: prior_high_volume_contest_20d<=-0.170934 AND stock_volume_ratio_20d>=1.78077")
    print(score_rule(df95, combo, "candidate_combo"))
    print(df95.loc[combo, detail_cols].sort_values("return_pct").to_string(index=False))


if __name__ == "__main__":
    main()
