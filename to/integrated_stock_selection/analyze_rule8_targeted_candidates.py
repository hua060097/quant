from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_rule8_remaining_compact_scan import build_df, compact_stats


def qdesc(base: pd.DataFrame, col: str, value: float) -> str:
    s = pd.to_numeric(base[col], errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return "NA"
    return f"pct={float((s <= value).mean()):.2f}"


def evaluate_rule(base: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    st_all = compact_stats(base, mask)
    st_actual = compact_stats(base[base.event_type == "actual"], mask[base.event_type == "actual"])
    st_shadow = compact_stats(base[base.event_type == "shadow"], mask[base.event_type == "shadow"])
    st_live = compact_stats(base[base.event_type == "live_check"], mask[base.event_type == "live_check"])
    return {
        "rule": name,
        **{f"all_{k}": v for k, v in st_all.items()},
        **{f"actual_{k}": v for k, v in st_actual.items()},
        **{f"shadow_{k}": v for k, v in st_shadow.items()},
        **{f"live_{k}": v for k, v in st_live.items()},
    }


def main() -> None:
    df = build_df()
    base = df[~df.current_rule_hit].copy()
    for c in base.columns:
        if c not in ["event_type", "stock_code", "buy_date", "buy_datetime", "factor_date"]:
            base[c] = pd.to_numeric(base[c], errors="ignore")

    print("remaining summary")
    for typ in ["actual", "shadow", "live_check"]:
        sub = base[base.event_type == typ]
        print(typ, compact_stats(sub, pd.Series(True, index=sub.index)))

    # Candidate families, deliberately small and interpretable.
    candidates = []
    add = candidates.append

    # A: high-position chase + retail/small order inflow + overbought KDJ.
    for top10_th in [0.80, 0.85, 0.90, 0.95]:
        for buy_sm_th in [1.0, 2.0, 3.0, 5.0]:
            for kdj_th in [1.0, 1.5, 1.8, 2.0]:
                name = f"top10>={top10_th} & buy_sm>={buy_sm_th} & kdj_j_9>={kdj_th}"
                mask = (
                    (base["top10_10d_strength_level_v3"] >= top10_th)
                    & (base["buy_sm_amount_rate"] >= buy_sm_th)
                    & (base["kdj_j_9"] >= kdj_th)
                )
                add(evaluate_rule(base, name, mask))

    # B: high-position chase + stretched MA5 distance + weak/neutral chain emotion.
    for top10_th in [0.80, 0.90, 0.95]:
        for ma5_th in [4.0, 6.0, 8.0, 9.0]:
            for nf_hi in [0.55, 0.60, 0.70]:
                name = f"top10>={top10_th} & price_ma5_raw>={ma5_th} & non_first_pct20<={nf_hi}"
                mask = (
                    (base["top10_10d_strength_level_v3"] >= top10_th)
                    & (base["price_ma5_ratio_raw"] >= ma5_th)
                    & (base["prev_non_first_limit_up_performance_percentile_20d"] <= nf_hi)
                )
                add(evaluate_rule(base, name, mask))

    # C: weak market breadth/liquidity leftovers.
    for liq_th in [0.86, 0.90, 0.94, 0.96]:
        for adr_th in [0.20, 0.30, 0.45, 0.60]:
            name = f"liq<={liq_th} & adr<={adr_th}"
            mask = (base["market_liquidity_change"] <= liq_th) & (base["market_advance_decline_ratio"] <= adr_th)
            add(evaluate_rule(base, name, mask))

    # D: blowoff extreme but old Rule2 sector threshold missed it.
    for blow_th in [1.2, 1.5, 2.0, 5.0, 8.0]:
        for sector_th in [0.50, 0.70, 0.80, 1.00]:
            name = f"blowoff>={blow_th} & sector_accel<={sector_th}"
            mask = (base["prev_day_blowoff_risk"] >= blow_th) & (base["sector_momentum_accel_rank"] <= sector_th)
            add(evaluate_rule(base, name, mask))

    # E: market dispersion/volatility extreme with poor next-day breadth.
    for disp_th in [1.05, 1.10, 1.15, 1.18]:
        for adr_th in [1.0, 1.5, 2.0]:
            name = f"disp>={disp_th} & adr<={adr_th}"
            mask = (base["mkt_dispersion_shock_v2"] >= disp_th) & (base["market_advance_decline_ratio"] <= adr_th)
            add(evaluate_rule(base, name, mask))

    out = pd.DataFrame(candidates)
    out["score"] = (
        8 * out["all_loss8"]
        + 4 * out["all_loss4"]
        + 6 * out["all_live300687"]
        + out["all_loss0"]
        - 5 * out["all_gain3_8"]
        - 4 * out["all_gain4"]
        - 10 * out["all_gain8"]
        - 0.08 * out["all_n"]
    )
    out = out.sort_values(
        ["score", "all_loss8", "all_loss4", "all_live300687", "all_gain8", "all_gain3_8", "all_n"],
        ascending=[False, False, False, False, True, True, True],
    )
    out.to_csv("results/rule8_remaining_targeted_candidates.csv", index=False, encoding="utf-8-sig")
    show = [
        "score", "rule", "all_n", "all_loss8", "all_loss4", "all_loss0", "all_gain3_8", "all_gain4", "all_gain8", "all_live300687",
        "actual_n", "actual_loss4", "actual_gain8", "shadow_n", "shadow_loss4", "shadow_gain8", "live_live300687", "all_hits",
    ]
    print("\nTOP TARGETED")
    print(out[show].head(80).to_string(index=False))

    print("\nLIVE 300687 factor percentiles within remaining pool")
    live = base[(base.stock_code == "300687.SZ") & (base.buy_date == pd.Timestamp("2026-05-21"))].iloc[0]
    for col in [
        "top10_10d_strength_level_v3", "buy_sm_amount_rate", "kdj_j_9",
        "price_ma5_ratio_raw", "prev_non_first_limit_up_performance_percentile_20d",
        "market_liquidity_change", "market_advance_decline_ratio", "mkt_dispersion_shock_v2",
        "high_to_close_drawdown_pct", "open_to_high_ratio", "stock_volume_ratio_20d",
    ]:
        print(col, float(live[col]), qdesc(base, col, float(live[col])))


if __name__ == "__main__":
    main()
