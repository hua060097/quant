from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TAG = "20260520_210454"
BASE_SCRIPT = ROOT / "scripts" / "analyze_ma5_touch_count_rules.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("ma5_touch_audit", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "R1_runup_cpv": ((df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)).fillna(False),
        "R2_blowoff_sector": (
            ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
            | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
            | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
        ).fillna(False),
        "R3_leadership_return20": (
            (df["sector_leadership_persistence_v3"] <= -0.18)
            & (df["return_20d_intra_z"] <= 0.40)
        ).fillna(False),
        "R4_liquidity_decay": (
            (df["market_liquidity_change"] < 0.94)
            & (
                (df["ulcer_index"] < -0.80)
                | (df["momentum_peak_decay_3d"] > 1.0)
                | (df["intra_sector_momentum_rank"] < 0.67)
            )
        ).fillna(False),
        "R5_dispersion_adr": (
            (df["mkt_dispersion_shock_v2"] < 0.98)
            & (df["market_advance_decline_ratio"] < 0.38)
        ).fillna(False),
        "R6_limit_down": (df["limit_down_3d"] == 1).fillna(False),
        "R7_top10_sm": (
            (df["top10_10d_strength_level_v3"] >= 0.70)
            & (df["buy_sm_amount_rate"] <= -1.25)
        ).fillna(False),
        "R8_nonfirst_kdj": (
            (df["prev_non_first_limit_up_performance_percentile_20d"] >= 0.80)
            & (df["kdj_j_9"] <= 0.75)
        ).fillna(False),
        "R9_score_z": (df.get("score_z_in_day", pd.Series(np.nan, index=df.index)) <= 1.50).fillna(False),
        "R10_reg_pct_adr": (
            (df["reg_severe_up_10d_ratio_pct"] >= 0.80)
            & (df["market_advance_decline_ratio"] <= 0.45)
        ).fillna(False),
    }


def bucket_counts(df: pd.DataFrame) -> dict[str, float | int]:
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
        "actual_big_gain_8": int(((df["source"] == "actual") & (r >= 0.08)).sum()),
        "shadow_big_loss": int(((df["source"] == "shadow") & (r <= -0.04)).sum()),
        "shadow_big_gain_8": int(((df["source"] == "shadow") & (r >= 0.08)).sum()),
    }


def stat_row(label: str, df: pd.DataFrame, mask: pd.Series, baseline: pd.Series | None = None) -> dict:
    mask = mask.fillna(False)
    row = {"label": label, "hits": int(mask.sum())}
    for key, value in bucket_counts(df[mask]).items():
        row[f"hit_{key}"] = value
    if baseline is not None:
        inc = mask & ~baseline
        row["overlap_with_baseline"] = int((mask & baseline).sum())
        row["incremental_hits"] = int(inc.sum())
        for key, value in bucket_counts(df[inc]).items():
            row[f"inc_{key}"] = value
    return row


def attach_close_acceptance(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    snap = daily.rename(columns={"date": "factor_date"})[
        ["factor_date", "stock_code", "high", "low", "close"]
    ].copy()
    out = df.merge(snap, on=["factor_date", "stock_code"], how="left", suffixes=("", "_daily"))
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_vs_ma5"] = out["close"] / out["ma5"] - 1.0
    out["close_position"] = ((out["close"] - out["low"]) / rng).replace([np.inf, -np.inf], np.nan)
    out["close_reject_weak"] = (out["close_vs_ma5"] <= 0.02) | (out["close_position"] <= 0.50)
    out["close_accept_strong"] = (out["close_vs_ma5"] >= 0.04) & (out["close_position"] >= 0.70)
    return out


def attach_raw_accel(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    raw = daily[["date", "stock_code", "close"]].copy()
    raw = raw.sort_values(["stock_code", "date"])
    grouped = raw.groupby("stock_code", sort=False)["close"]
    ret3 = grouped.transform(lambda x: x / x.shift(3) - 1.0)
    ret10 = grouped.transform(lambda x: x / x.shift(10) - 1.0)
    raw["ret_3d_vs_10d_accel_raw"] = (ret3 - ret10 / 3.0).clip(-1.0, 2.0).fillna(0.0)
    raw = raw.rename(columns={"date": "factor_date"})
    return df.merge(
        raw[["factor_date", "stock_code", "ret_3d_vs_10d_accel_raw"]],
        on=["factor_date", "stock_code"],
        how="left",
    )


def market_env_label(df: pd.DataFrame) -> pd.Series:
    adr = df["market_advance_decline_ratio"]
    top10 = df["top10_10d_strength_level_v3"]
    liq = df["market_liquidity_change"]
    env = pd.Series("neutral", index=df.index, dtype="object")
    env[adr <= 0.45] = "weak_adr"
    env[(adr > 0.45) & (top10 >= 0.70)] = "hot_top10"
    env[(adr > 0.60) & (top10 < 0.70)] = "strong_breadth"
    env[liq < 0.94] = env[liq < 0.94] + "_lowliq"
    return env


def group_summary(df: pd.DataFrame, candidate: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["candidate"] = candidate.fillna(False)
    out["baseline"] = baseline.fillna(False)
    out["incremental"] = out["candidate"] & ~out["baseline"]
    out["month"] = out["date"].dt.strftime("%Y-%m")
    midpoint = out["date"].sort_values().iloc[len(out) // 2]
    out["half"] = np.where(out["date"] <= midpoint, "first_half", "second_half")
    out["market_env"] = market_env_label(out)

    rows = []
    for group_col in ["half", "month", "market_env"]:
        for value, sub in out.groupby(group_col, dropna=False):
            cand = sub[sub["candidate"]]
            inc = sub[sub["incremental"]]
            row = {"group_type": group_col, "group": value, "events": int(len(sub))}
            for prefix, frame in (("candidate", cand), ("incremental", inc)):
                for key, val in bucket_counts(frame).items():
                    row[f"{prefix}_{key}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def neighborhood_scan(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    pullback_thresholds = [-0.10, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02]
    accel_thresholds = [3.0, 4.0, 5.0, 5.5, 6.0, 6.5]
    for pullback_t in pullback_thresholds:
        for accel_t in accel_thresholds:
            mask = (
                (df["ma5_pullback_depth_10d"] <= pullback_t)
                & (df["ret_3d_vs_10d_accel"] >= accel_t)
            ).fillna(False)
            label = f"pullback<={pullback_t:g} & accel>={accel_t:g}"
            row = stat_row(label, df, mask, baseline)
            row["pullback_threshold"] = pullback_t
            row["accel_threshold"] = accel_t
            row["score"] = (
                row["inc_big_loss"] * 3
                + row["inc_small_loss"]
                - row["inc_big_gain_8"] * 5
                - row["inc_small_gain_3_8"]
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["score", "inc_big_loss", "inc_big_gain_8", "incremental_hits"],
        ascending=[False, False, True, True],
    )


def raw_accel_scan(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    rows = []
    pullback_thresholds = [-0.06, -0.05, -0.04]
    raw_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
    for pullback_t in pullback_thresholds:
        for raw_t in raw_thresholds:
            mask = (
                (df["ma5_pullback_depth_10d"] <= pullback_t)
                & (df["ret_3d_vs_10d_accel_raw"] >= raw_t)
            ).fillna(False)
            label = f"pullback<={pullback_t:g} & raw_accel>={raw_t:g}"
            row = stat_row(label, df, mask, baseline)
            row["pullback_threshold"] = pullback_t
            row["raw_accel_threshold"] = raw_t
            row["score"] = (
                row["inc_big_loss"] * 3
                + row["inc_small_loss"]
                - row["inc_big_gain_8"] * 5
                - row["inc_small_gain_3_8"]
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["score", "inc_big_loss", "inc_big_gain_8", "incremental_hits"],
        ascending=[False, False, True, True],
    )


def main() -> None:
    m = load_base_module()
    events = m.add_factor_dates(m.load_events())
    events = m.load_factor_snapshots(events)
    daily = m.load_daily_for_touch(events)
    touch = m.compute_ma5_touch_counts(daily)
    events = m.attach_touch_features(events, touch)
    events = attach_close_acceptance(events, daily)
    events = attach_raw_accel(events, daily)

    masks = rule_masks(events)
    base_r1_r9 = pd.concat([masks[k] for k in masks if k != "R10_reg_pct_adr"], axis=1).any(axis=1)
    base_plus_r10 = pd.concat(list(masks.values()), axis=1).any(axis=1)
    candidate = (
        (events["ma5_pullback_depth_10d"] <= -0.05)
        & (events["ret_3d_vs_10d_accel"] >= 6)
    ).fillna(False)

    coverage_rows = []
    for name, mask in masks.items():
        row = stat_row(name, events, mask, candidate)
        row["candidate_overlap"] = int((candidate & mask).sum())
        row["candidate_only_vs_rule"] = int((candidate & ~mask).sum())
        row["rule_only_vs_candidate"] = int((mask & ~candidate).sum())
        coverage_rows.append(row)

    for name, baseline in [("UNION_R1_R9", base_r1_r9), ("UNION_R1_R10", base_plus_r10)]:
        coverage_rows.append(stat_row(name, events, baseline, candidate))

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(RESULTS / f"ma5_pullback_accel_existing_overlap_{TAG}.csv", index=False, encoding="utf-8-sig")

    support_conditions = {
        "candidate_raw": candidate,
        "candidate_close_weak": candidate & events["close_reject_weak"],
        "candidate_not_close_strong": candidate & ~events["close_accept_strong"].fillna(False),
        "candidate_sector_weak": candidate & (events["sector_momentum_accel_rank"] <= 0.50),
        "candidate_sector_not_strong": candidate & (events["sector_momentum_accel_rank"] <= 0.70),
        "candidate_intra_sector_weak": candidate & (events["intra_sector_momentum_rank"] <= 0.67),
        "candidate_close_weak_sector_weak": (
            candidate & events["close_reject_weak"] & (events["sector_momentum_accel_rank"] <= 0.50)
        ),
        "candidate_close_weak_sector_not_strong": (
            candidate & events["close_reject_weak"] & (events["sector_momentum_accel_rank"] <= 0.70)
        ),
        "candidate_return20_not_hot": candidate & (events["return_20d_intra_z"] <= 0.70),
        "candidate_price_ma5_not_extreme": candidate & (events["price_ma5_ratio_raw"] <= 12),
    }
    support_rows = []
    for name, mask in support_conditions.items():
        row = stat_row(name, events, mask, base_plus_r10)
        support_rows.append(row)
    support = pd.DataFrame(support_rows)
    support.to_csv(RESULTS / f"ma5_pullback_accel_support_confirm_{TAG}.csv", index=False, encoding="utf-8-sig")

    grouped = group_summary(events, candidate, base_plus_r10)
    grouped.to_csv(RESULTS / f"ma5_pullback_accel_group_stability_{TAG}.csv", index=False, encoding="utf-8-sig")

    neighborhood = neighborhood_scan(events, base_plus_r10)
    neighborhood.to_csv(
        RESULTS / f"ma5_pullback_accel_neighborhood_{TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_neighborhood = raw_accel_scan(events, base_plus_r10)
    raw_neighborhood.to_csv(
        RESULTS / f"ma5_pullback_raw_accel_neighborhood_{TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    detail_cols = [
        "source",
        "date",
        "stock_code",
        "actual_return",
        "factor_date",
        "ma5_pullback_depth_10d",
        "ret_3d_vs_10d_accel",
        "ret_3d_vs_10d_accel_raw",
        "ma5_break_rebound_accel_risk",
        "close_vs_ma5",
        "close_position",
        "sector_momentum_accel_rank",
        "intra_sector_momentum_rank",
        "return_20d_intra_z",
        "price_ma5_ratio_raw",
        "market_advance_decline_ratio",
        "top10_10d_strength_level_v3",
    ]
    details = events[candidate].copy().sort_values("actual_return")
    details[detail_cols].to_csv(
        RESULTS / f"ma5_pullback_accel_candidate_details_{TAG}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    show_cols = [
        "label",
        "hits",
        "hit_big_loss",
        "hit_big_gain_8",
        "candidate_overlap",
        "candidate_only_vs_rule",
        "rule_only_vs_candidate",
    ]
    print("\nEXISTING OVERLAP")
    print(coverage[[c for c in show_cols if c in coverage.columns]].to_string(index=False))

    support_show = [
        "label",
        "hits",
        "hit_big_loss",
        "hit_small_loss",
        "hit_small_gain_3_8",
        "hit_big_gain_8",
        "incremental_hits",
        "inc_big_loss",
        "inc_small_loss",
        "inc_small_gain_3_8",
        "inc_big_gain_8",
        "inc_big_gain_10",
    ]
    print("\nSUPPORT CONFIRM")
    print(support[support_show].to_string(index=False))

    group_show = [
        "group_type",
        "group",
        "events",
        "candidate_n",
        "candidate_big_loss",
        "candidate_big_gain_8",
        "incremental_n",
        "incremental_big_loss",
        "incremental_small_loss",
        "incremental_big_gain_8",
    ]
    print("\nGROUP STABILITY")
    print(grouped[group_show].to_string(index=False))

    neigh_show = [
        "label",
        "hits",
        "hit_big_loss",
        "hit_big_gain_8",
        "incremental_hits",
        "inc_mean",
        "inc_big_loss",
        "inc_small_loss",
        "inc_small_gain_3_8",
        "inc_big_gain_8",
        "inc_big_gain_10",
        "score",
    ]
    print("\nNEIGHBORHOOD")
    print(neighborhood[neigh_show].head(30).to_string(index=False))

    raw_show = [
        "label",
        "hits",
        "hit_big_loss",
        "hit_big_gain_8",
        "incremental_hits",
        "inc_mean",
        "inc_big_loss",
        "inc_small_loss",
        "inc_small_gain_3_8",
        "inc_big_gain_8",
        "inc_big_gain_10",
        "score",
    ]
    print("\nRAW ACCEL NEIGHBORHOOD")
    print(raw_neighborhood[raw_show].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
