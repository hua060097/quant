import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"

RUNS = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}

SHADOWS = {
    "shadow_20260513": RESULTS / "walkforward_shadow_skipped_opportunities_20260513_230111.csv",
    "shadow_20260520": RESULTS / "walkforward_shadow_skipped_opportunities_20260520_190404.csv",
}

FACTOR_COLS = [
    # Current v3 factors.
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
    # Market / sector candidates.
    "market_60d_new_high_ratio",
    "market_sentiment_ad",
    "market_volatility_context",
    "market_trend_strength",
    "mkt_regime_trend_pressure_v2",
    "mkt_breadth_thrust_v2",
    "mkt_emotion_stability_v2",
    "mkt_ma5_followthrough_v3",
    "mkt_ma5_followthrough_v3_exp",
    "mkt_liquidity_impulse_v2",
    "mkt_sector_dispersion",
    "sector_hotspot_concentration_v3",
    "sector_dispersion",
    "sector_momentum_dispersion_rank",
    "sector_momentum_consistency",
    "sector_relative_strength_vs_market",
    "sector_breadth",
    "sector_rank_change_5d",
    # Stock candidates.
    "money_flow_pressure",
    "money_flow_pressure_pct",
    "net_d5_amount_to_turnover",
    "buy_lg_amount_rate",
    "buy_sm_amount_rate",
    "true_turnover_10d",
    "turnover_z_20",
    "stock_volume_ratio_20d",
    "volume_price_exhaustion",
    "close_position_20d",
    "close_position_60d",
    "max_drawdown_20d",
    "max_rally_10d",
    "max_rally_20d",
    "high_to_close_drawdown",
    "high_to_close_drawdown_pct",
    "pullback_depth_from_10d_high",
    "prior_high_volume_contest_20d",
    "prior_high_proximity_20d",
    "distance_from_52w_high",
    "efficiency_ratio",
    "high_volume_no_gain_3d",
    "high_pos_volume_no_gain",
    "high_position_volume_surge",
    "shadow_reversal_volume_pressure",
    "ma5_ma10_flattening_after_runup",
    "low_break_prev_low_after_runup",
    "runup_low_break",
    "close_below_ma5_after_runup",
    "gap_power",
]


@dataclass(frozen=True)
class Rule:
    name: str
    expr: str
    mask: pd.Series


def load_trades_from_metrics(name, path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data[0]["all_results"]["1"]
    rows = []
    for t in result["trades"]:
        rows.append(
            {
                "dataset": name,
                "stock_code": t["stock_code"],
                "buy_date": t["buy_datetime"][:10],
                "sell_date": t["sell_datetime"][:10],
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
                "exit_reason": t.get("exit_reason", ""),
                "source": "real",
            }
        )
    return rows


def load_shadow(name, path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "dataset": name,
                    "stock_code": r["stock_code"],
                    "buy_date": r["buy_datetime"][:10],
                    "sell_date": r["sell_datetime"][:10],
                    "actual_return": float(r["actual_return"]),
                    "profit_loss": float(r["profit_loss"]),
                    "exit_reason": r.get("exit_reason", ""),
                    "source": "shadow",
                    "shadow_status": r.get("shadow_status", ""),
                    "shadow_exhaustion_blocked": str(r.get("exhaustion_blocked", "")).lower() == "true",
                }
            )
    return rows


def load_all_events():
    rows = []
    for name, path in RUNS.items():
        rows.extend(load_trades_from_metrics(name, path))
    for name, path in SHADOWS.items():
        rows.extend(load_shadow(name, path))
    events = pd.DataFrame(rows)
    events["buy_date"] = pd.to_datetime(events["buy_date"])
    events["sell_date"] = pd.to_datetime(events["sell_date"])
    return events


def enrich_with_factors(events):
    df_all = pd.read_parquet(FACTOR_FILE)
    use_cols = ["date", "stock_code"] + [c for c in FACTOR_COLS if c in df_all.columns]
    df = df_all[use_cols].copy()
    del df_all
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"])

    enriched = []
    grouped = {}
    for stock, stock_df in df.groupby("stock_code", sort=False):
        grouped[stock] = (
            stock_df["date"].to_numpy(dtype="datetime64[ns]"),
            stock_df.reset_index(drop=True),
        )
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is None:
            enriched.append(rec)
            continue
        dates, stock_df = item
        pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left") - 1
        if pos < 0:
            enriched.append(rec)
            continue
        r = stock_df.iloc[int(pos)]
        rec["factor_date"] = r["date"]
        for c in FACTOR_COLS:
            if c in r.index:
                rec[c] = r[c]
        enriched.append(rec)
    out = pd.DataFrame(enriched)
    return out


def current_v3_mask(df):
    r1 = (df["runup_ma5_exhaustion"] > 0.8) & (df["cpv_factor"] <= -0.4)
    r2 = (
        ((df["prev_day_blowoff_risk"] >= 2.0) & (df["sector_momentum_accel_rank"] <= 0.30))
        | ((df["prev_day_blowoff_risk"] > 5.0) & (df["sector_momentum_accel_rank"] <= 0.50))
        | ((df["prev_day_blowoff_risk"] > 1.0) & (df["sector_momentum_accel_rank"] <= 0.15))
    )
    r3 = (df["sector_leadership_persistence_v3"] <= -0.18) & (df["return_20d_intra_z"] <= 0.40)
    r4 = (df["market_liquidity_change"] < 0.94) & (
        (df["ulcer_index"] < -0.8)
        | (df["momentum_peak_decay_3d"] > 1.0)
        | (df["intra_sector_momentum_rank"] < 0.67)
    )
    r5 = (df["mkt_dispersion_shock_v2"] < 0.98) & (df["market_advance_decline_ratio"] < 0.38)
    r6 = df.get("limit_down_3d", pd.Series(False, index=df.index)).fillna(0).astype(float).eq(1)
    return (r1 | r2 | r3 | r4 | r5 | r6).fillna(False)


def manual_rules(df):
    rules = []
    rules.append(
        Rule(
            "R7_ulcer_extreme",
            "ulcer_index <= -2.5",
            (df["ulcer_index"] <= -2.5),
        )
    )
    rules.append(
        Rule(
            "R7_ulcer_or_weak_breadth",
            "ulcer_index <= -2.5 OR adr <= 0.25",
            (df["ulcer_index"] <= -2.5) | (df["market_advance_decline_ratio"] <= 0.25),
        )
    )
    rules.append(
        Rule(
            "R8_weak_breadth_relaxed",
            "adr <= 0.35 & (disp < 1.15 or return20 <= 0.75 or sector_accel <= 0.35)",
            (df["market_advance_decline_ratio"] <= 0.35)
            & (
                (df["mkt_dispersion_shock_v2"] < 1.15)
                | (df["return_20d_intra_z"] <= 0.75)
                | (df["sector_momentum_accel_rank"] <= 0.35)
            ),
        )
    )
    rules.append(
        Rule(
            "R9_liq_boundary_sector",
            "liq <= 0.945 & intra_sector_rank <= 1.6",
            (df["market_liquidity_change"] <= 0.945) & (df["intra_sector_momentum_rank"] <= 1.6),
        )
    )
    rules.append(
        Rule(
            "R10_blowoff_adr",
            "blowoff >= 2.0 & adr <= 0.30",
            (df["prev_day_blowoff_risk"] >= 2.0) & (df["market_advance_decline_ratio"] <= 0.30),
        )
    )
    rules.append(
        Rule(
            "R11_runup_sector_weak",
            "runup > 2.0 & sector_accel <= 0.30 & adr <= 0.55",
            (df["runup_ma5_exhaustion"] > 2.0)
            & (df["sector_momentum_accel_rank"] <= 0.30)
            & (df["market_advance_decline_ratio"] <= 0.55),
        )
    )
    rules.append(
        Rule(
            "R12_ulcer_adr",
            "ulcer <= -1.2 & adr <= 0.70",
            (df["ulcer_index"] <= -1.2) & (df["market_advance_decline_ratio"] <= 0.70),
        )
    )
    return rules


def generated_rules(df):
    factors = [
        "market_advance_decline_ratio",
        "mkt_dispersion_shock_v2",
        "market_liquidity_change",
        "market_60d_new_high_ratio",
        "mkt_breadth_thrust_v2",
        "mkt_liquidity_impulse_v2",
        "mkt_ma5_followthrough_v3",
        "sector_momentum_accel_rank",
        "sector_leadership_persistence_v3",
        "return_20d_intra_z",
        "intra_sector_momentum_rank",
        "ulcer_index",
        "prev_day_blowoff_risk",
        "runup_ma5_exhaustion",
        "cpv_factor",
        "money_flow_pressure_pct",
        "close_position_60d",
        "max_drawdown_20d",
        "max_rally_10d",
        "turnover_z_20",
        "stock_volume_ratio_20d",
    ]
    dirs = {
        "prev_day_blowoff_risk": ">",
        "runup_ma5_exhaustion": ">",
        "turnover_z_20": ">",
        "stock_volume_ratio_20d": ">",
        "max_rally_10d": ">",
    }
    qs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    rules = []
    valid = df[df["source"].eq("real")].copy()
    for i, f1 in enumerate(factors):
        if f1 not in df.columns or valid[f1].notna().sum() < 20:
            continue
        for f2 in factors[i + 1 :]:
            if f2 not in df.columns or valid[f2].notna().sum() < 20:
                continue
            for q1 in qs:
                th1 = valid[f1].quantile(1 - q1 if dirs.get(f1, "<") == ">" else q1)
                m1 = df[f1] >= th1 if dirs.get(f1, "<") == ">" else df[f1] <= th1
                for q2 in qs:
                    th2 = valid[f2].quantile(1 - q2 if dirs.get(f2, "<") == ">" else q2)
                    m2 = df[f2] >= th2 if dirs.get(f2, "<") == ">" else df[f2] <= th2
                    op1 = ">=" if dirs.get(f1, "<") == ">" else "<="
                    op2 = ">=" if dirs.get(f2, "<") == ">" else "<="
                    rules.append(
                        Rule(
                            f"GEN_{f1}_{f2}_{q1:.2f}_{q2:.2f}",
                            f"{f1} {op1} {th1:.4g} & {f2} {op2} {th2:.4g}",
                            (m1 & m2).fillna(False),
                        )
                    )
    return rules


def agg(mask, df):
    rows = df[mask]
    losses = rows[rows["actual_return"] <= -0.05]
    bigwins = rows[rows["actual_return"] >= 0.10]
    positives = rows[rows["actual_return"] > 0]
    return {
        "n": len(rows),
        "loss": len(losses),
        "bigwin": len(bigwins),
        "pos": len(positives),
        "pnl": rows["profit_loss"].sum(),
        "avg_ret": rows["actual_return"].mean() if len(rows) else 0.0,
    }


def evaluate_rule(rule, df):
    add = rule.mask.fillna(False) & ~df["current_v3"]
    combo = rule.mask.fillna(False) | df["current_v3"]
    out = {"name": rule.name, "expr": rule.expr}
    out["add_all"] = agg(add, df)
    out["combo_all"] = agg(combo, df)
    for ds in sorted(df["dataset"].unique()):
        d = df[df["dataset"].eq(ds)]
        out[f"add_{ds}"] = agg(add.loc[d.index], d)
    latest = df[df["dataset"].eq("latest184_v3")]
    recent = latest[latest["buy_date"] >= pd.Timestamp("2026-04-01")]
    out["add_latest184_recent"] = agg(add.loc[recent.index], recent)
    out["score"] = (
        out["add_all"]["loss"] * 4
        + out["add_latest184_recent"]["loss"] * 6
        - out["add_all"]["bigwin"] * 7
        - out["add_all"]["pos"] * 1.5
        - max(0, out["add_all"]["n"] - out["add_all"]["loss"] - out["add_all"]["bigwin"]) * 0.5
    )
    return out


def print_summary(df):
    print("== enriched sample ==")
    for ds in sorted(df["dataset"].unique()):
        d = df[df["dataset"].eq(ds)]
        print(
            "{}: n={} loss<=-5%={} bigwin>=10%={} pos={} avg_ret={:.2%} pnl={:.0f}".format(
                ds,
                len(d),
                (d["actual_return"] <= -0.05).sum(),
                (d["actual_return"] >= 0.10).sum(),
                (d["actual_return"] > 0).sum(),
                d["actual_return"].mean(),
                d["profit_loss"].sum(),
            )
        )
    print("\n== current v3 retrospective hits ==")
    for ds in sorted(df["dataset"].unique()):
        d = df[df["dataset"].eq(ds)]
        a = agg(d["current_v3"], d)
        print(
            "{}: block_n={} loss={} bigwin={} pos={} avg_ret={:.2%} pnl={:.0f}".format(
                ds, a["n"], a["loss"], a["bigwin"], a["pos"], a["avg_ret"], a["pnl"]
            )
        )


def report_rule(res, df, limit=12):
    print("\n-- {} --".format(res["name"]))
    print(res["expr"])
    print(
        "add_all: n={n} loss={loss} bigwin={bigwin} pos={pos} avg_ret={avg_ret:.2%} pnl={pnl:.0f}; "
        "recent184 loss={rloss}, n={rn}".format(
            **res["add_all"],
            rloss=res["add_latest184_recent"]["loss"],
            rn=res["add_latest184_recent"]["n"],
        )
    )
    for ds in sorted(df["dataset"].unique()):
        a = res[f"add_{ds}"]
        if a["n"]:
            print(
                "  {}: n={} loss={} bigwin={} pos={} avg_ret={:.2%} pnl={:.0f}".format(
                    ds, a["n"], a["loss"], a["bigwin"], a["pos"], a["avg_ret"], a["pnl"]
                )
            )
    hit = df[res["mask"] & ~df["current_v3"]].copy() if "mask" in res else pd.DataFrame()
    if hit.empty:
        return
    hit = hit.sort_values(["actual_return", "buy_date"])
    print("  hits:")
    for _, r in hit.head(limit).iterrows():
        print(
            "    {dataset} {date} {code} ret={ret:.2%} pnl={pnl:.0f} "
            "adr={adr:.2f} disp={disp:.2f} liq={liq:.2f} ulcer={ulcer:.2f} "
            "blowoff={blow:.2f} accel={accel:.2f} runup={runup:.2f}".format(
                dataset=r["dataset"],
                date=r["buy_date"].date(),
                code=r["stock_code"],
                ret=r["actual_return"],
                pnl=r["profit_loss"],
                adr=r.get("market_advance_decline_ratio", np.nan),
                disp=r.get("mkt_dispersion_shock_v2", np.nan),
                liq=r.get("market_liquidity_change", np.nan),
                ulcer=r.get("ulcer_index", np.nan),
                blow=r.get("prev_day_blowoff_risk", np.nan),
                accel=r.get("sector_momentum_accel_rank", np.nan),
                runup=r.get("runup_ma5_exhaustion", np.nan),
            )
        )


def main():
    events = load_all_events()
    df = enrich_with_factors(events)
    df["current_v3"] = current_v3_mask(df)
    df["large_loss"] = df["actual_return"] <= -0.05
    df["big_win"] = df["actual_return"] >= 0.10
    print_summary(df)

    print("\n== latest184 recent losses not covered by v3 ==")
    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & df["large_loss"]
        & ~df["current_v3"]
    ].sort_values("profit_loss")
    cols = [
        "stock_code",
        "buy_date",
        "actual_return",
        "profit_loss",
        "market_advance_decline_ratio",
        "mkt_dispersion_shock_v2",
        "market_liquidity_change",
        "ulcer_index",
        "prev_day_blowoff_risk",
        "sector_momentum_accel_rank",
        "return_20d_intra_z",
        "runup_ma5_exhaustion",
        "cpv_factor",
    ]
    print(recent[cols].to_string(index=False))

    rules = manual_rules(df) + generated_rules(df)
    results = []
    mask_by_name = {}
    for rule in rules:
        res = evaluate_rule(rule, df)
        # Keep rules that catch latest v3 leaks and do not kill too many big winners.
        if res["add_all"]["loss"] >= 2 and res["add_latest184_recent"]["loss"] >= 1 and res["add_all"]["bigwin"] <= 2:
            res["mask"] = rule.mask.fillna(False)
            results.append(res)
            mask_by_name[res["name"]] = rule.mask.fillna(False)
    results.sort(key=lambda x: (-x["score"], -x["add_latest184_recent"]["loss"], x["add_all"]["bigwin"], x["add_all"]["n"]))

    print("\n== top candidate add-on rules ==")
    for res in results[:20]:
        print(
            "{score:6.1f} {name:45s} n={n:3d} loss={loss:2d} recent_loss={recent:2d} bigwin={bw:2d} pos={pos:2d} avg={avg:.2%} :: {expr}".format(
                score=res["score"],
                name=res["name"][:45],
                n=res["add_all"]["n"],
                loss=res["add_all"]["loss"],
                recent=res["add_latest184_recent"]["loss"],
                bw=res["add_all"]["bigwin"],
                pos=res["add_all"]["pos"],
                avg=res["add_all"]["avg_ret"],
                expr=res["expr"],
            )
        )

    for res in results[:8]:
        report_rule(res, df)

    # A compact proposed v4: combine the best interpretable rules, then summarize.
    proposed = {
        "R7_ulcer_extreme": next(r for r in manual_rules(df) if r.name == "R7_ulcer_extreme").mask.fillna(False),
        "R8_weak_breadth_relaxed": next(r for r in manual_rules(df) if r.name == "R8_weak_breadth_relaxed").mask.fillna(False),
        "R10_blowoff_adr": next(r for r in manual_rules(df) if r.name == "R10_blowoff_adr").mask.fillna(False),
    }
    add = pd.Series(False, index=df.index)
    for m in proposed.values():
        add |= m
    add &= ~df["current_v3"]
    combo = df["current_v3"] | add
    print("\n== proposed v4 add-on bundle ==")
    print("R7 ulcer<=-2.5 OR R8 adr<=0.35&(disp<1.15|ret20<=0.75|sector_accel<=0.35) OR R10 blowoff>=2&adr<=0.30")
    for ds in sorted(df["dataset"].unique()):
        d = df[df["dataset"].eq(ds)]
        a = agg(add.loc[d.index], d)
        c = agg(combo.loc[d.index], d)
        print(
            "{} add: n={} loss={} bigwin={} pos={} avg={:.2%} pnl={:.0f} | combo: n={} loss={} bigwin={} pos={} avg={:.2%} pnl={:.0f}".format(
                ds,
                a["n"],
                a["loss"],
                a["bigwin"],
                a["pos"],
                a["avg_ret"],
                a["pnl"],
                c["n"],
                c["loss"],
                c["bigwin"],
                c["pos"],
                c["avg_ret"],
                c["pnl"],
            )
        )

    hit = df[add].sort_values(["actual_return", "buy_date"])
    print("\nproposed add-on hit rows:")
    for _, r in hit.iterrows():
        print(
            "{dataset} {date} {code} ret={ret:.2%} pnl={pnl:.0f} adr={adr:.2f} disp={disp:.2f} "
            "liq={liq:.2f} ulcer={ulcer:.2f} blowoff={blow:.2f} accel={accel:.2f} ret20={ret20:.2f}".format(
                dataset=r["dataset"],
                date=r["buy_date"].date(),
                code=r["stock_code"],
                ret=r["actual_return"],
                pnl=r["profit_loss"],
                adr=r.get("market_advance_decline_ratio", np.nan),
                disp=r.get("mkt_dispersion_shock_v2", np.nan),
                liq=r.get("market_liquidity_change", np.nan),
                ulcer=r.get("ulcer_index", np.nan),
                blow=r.get("prev_day_blowoff_risk", np.nan),
                accel=r.get("sector_momentum_accel_rank", np.nan),
                ret20=r.get("return_20d_intra_z", np.nan),
            )
        )


if __name__ == "__main__":
    main()
