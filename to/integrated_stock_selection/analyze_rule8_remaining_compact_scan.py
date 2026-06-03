from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_rule8_remaining_loss_filters import (
    BASE_COLS,
    CAND_COLS,
    attach_factors,
    load_actual,
    load_shadow,
    rule_masks,
)


def compact_stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    sub = df[mask.fillna(False)]
    return {
        "n": int(len(sub)),
        "loss8": int((sub["actual_return"] <= -0.08).sum()),
        "loss4": int((sub["actual_return"] <= -0.04).sum()),
        "loss0": int(((sub["actual_return"] > -0.04) & (sub["actual_return"] < 0)).sum()),
        "gain3_8": int(((sub["actual_return"] >= 0.03) & (sub["actual_return"] < 0.08)).sum()),
        "gain4": int((sub["actual_return"] >= 0.04).sum()),
        "gain8": int((sub["actual_return"] >= 0.08).sum()),
        "live300687": int(((sub["stock_code"] == "300687.SZ") & (sub["buy_date"] == pd.Timestamp("2026-05-21"))).sum()),
        "pnl": float(sub["profit_loss"].fillna(0).sum()),
        "hits": ",".join((sub["stock_code"] + "@" + sub["buy_date"].dt.strftime("%Y%m%d")).tolist()[:12]),
    }


def build_df() -> pd.DataFrame:
    events = pd.concat([load_actual(), load_shadow()], ignore_index=True)
    live = pd.DataFrame([{
        "event_type": "live_check",
        "stock_code": "300687.SZ",
        "buy_date": pd.Timestamp("2026-05-21"),
        "buy_datetime": pd.Timestamp("2026-05-21 09:30:00"),
        "actual_return": -0.0982,
        "profit_loss": 0.0,
    }])
    events = pd.concat([events, live], ignore_index=True)
    factor_cols = sorted(set(BASE_COLS + CAND_COLS))
    df = attach_factors(events, factor_cols)
    masks = rule_masks(df)
    current = pd.Series(False, index=df.index)
    for m in masks.values():
        current = current | m.fillna(False)
    df["current_rule_hit"] = current
    return df


def scan_pairs(df: pd.DataFrame) -> pd.DataFrame:
    base = df[~df["current_rule_hit"]].copy()
    thresholds = []
    for col in CAND_COLS:
        if col not in base.columns:
            continue
        s = pd.to_numeric(base[col], errors="coerce")
        if s.notna().sum() < 8:
            continue
        vals = s.dropna()
        for q in [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]:
            th = float(vals.quantile(q))
            thresholds.append((col, "<=", q, th))
            thresholds.append((col, ">=", q, th))

    rows = []
    for i, (c1, op1, q1, th1) in enumerate(thresholds):
        s1 = pd.to_numeric(base[c1], errors="coerce")
        m1 = s1 <= th1 if op1 == "<=" else s1 >= th1
        if int(m1.sum()) < 2:
            continue
        for c2, op2, q2, th2 in thresholds[i + 1:]:
            if c1 == c2:
                continue
            s2 = pd.to_numeric(base[c2], errors="coerce")
            m2 = s2 <= th2 if op2 == "<=" else s2 >= th2
            mask = m1 & m2
            st_all = compact_stats(base, mask)
            if st_all["n"] < 2 or (st_all["loss4"] + st_all["live300687"]) == 0:
                continue
            st_actual = compact_stats(base[base.event_type == "actual"], mask[base.event_type == "actual"])
            st_shadow = compact_stats(base[base.event_type == "shadow"], mask[base.event_type == "shadow"])
            st_live = compact_stats(base[base.event_type == "live_check"], mask[base.event_type == "live_check"])
            rows.append({
                "rule": f"{c1}{op1}{th1:.4g}(q{q1:.1f}) & {c2}{op2}{th2:.4g}(q{q2:.1f})",
                **{f"all_{k}": v for k, v in st_all.items()},
                **{f"actual_{k}": v for k, v in st_actual.items()},
                **{f"shadow_{k}": v for k, v in st_shadow.items()},
                **{f"live_{k}": v for k, v in st_live.items()},
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        8 * out["all_loss8"]
        + 4 * out["all_loss4"]
        + 5 * out["all_live300687"]
        + out["all_loss0"]
        - 5 * out["all_gain3_8"]
        - 4 * out["all_gain4"]
        - 10 * out["all_gain8"]
        - 0.08 * out["all_n"]
    )
    return out.sort_values(
        ["score", "all_loss8", "all_loss4", "all_live300687", "all_gain8", "all_gain3_8", "all_n"],
        ascending=[False, False, False, False, True, True, True],
    )


def main() -> None:
    df = build_df()
    base = df[~df.current_rule_hit].copy()
    print("remaining by type")
    for typ in ["actual", "shadow", "live_check"]:
        sub = base[base.event_type == typ]
        print(typ, compact_stats(sub, pd.Series(True, index=sub.index)))

    print("\nremaining loss rows")
    cols = [
        "event_type", "stock_code", "buy_date", "factor_date", "actual_return",
        "market_liquidity_change", "mkt_dispersion_shock_v2", "market_advance_decline_ratio",
        "top10_10d_strength_level_v3", "buy_sm_amount_rate", "kdj_j_9",
        "prev_non_first_limit_up_performance_percentile_20d", "high_to_close_drawdown_pct",
        "price_ma5_ratio_raw", "open_to_high_ratio", "stock_volume_ratio_20d",
        "ret_3d_vs_10d_accel", "runup_low_break",
    ]
    print(base[base.actual_return <= -0.04][cols].sort_values("actual_return").to_string(index=False))

    pairs = scan_pairs(df)
    pairs.to_csv("results/rule8_remaining_pair_scan.csv", index=False, encoding="utf-8-sig")
    show_cols = [
        "score", "rule", "all_n", "all_loss8", "all_loss4", "all_loss0", "all_gain3_8", "all_gain4", "all_gain8", "all_live300687",
        "actual_n", "actual_loss4", "actual_gain8", "shadow_n", "shadow_loss4", "shadow_gain8", "live_live300687", "all_hits",
    ]
    print("\nTOP PAIRS")
    print(pairs[show_cols].head(60).to_string(index=False))


if __name__ == "__main__":
    main()
