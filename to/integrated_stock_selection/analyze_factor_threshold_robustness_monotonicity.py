import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"

RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}


def load_events():
    rows = []
    for ds, path in RUN_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        result = data[0]["all_results"]["1"]
        for t in result["trades"]:
            rows.append(
                {
                    "dataset": ds,
                    "stock_code": t["stock_code"],
                    "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                    "sell_date": pd.Timestamp(t["sell_datetime"][:10]),
                    "actual_return": float(t["actual_return"]),
                    "profit_loss": float(t["profit_loss"]),
                }
            )
    events = pd.DataFrame(rows)
    events["key"] = events["stock_code"] + "|" + events["buy_date"].dt.strftime("%Y-%m-%d")
    return events


def load_factor_frame():
    df = pd.read_parquet(FACTOR_FILE)
    df["date"] = pd.to_datetime(df["date"])
    base = {"date", "stock_code", "stock_name", "name", "sw_l1", "sw_l2", "sw_l3", "industry"}
    numeric_cols = []
    for c in df.columns:
        if c in base:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return df[["date", "stock_code"] + numeric_cols].sort_values(["stock_code", "date"]), numeric_cols


def enrich(events, factors, numeric_cols):
    grouped = {}
    for stock, g in factors.groupby("stock_code", sort=False):
        grouped[stock] = (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
    rows = []
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is not None:
            dates, g = item
            pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left") - 1
            if pos >= 0:
                r = g.iloc[int(pos)]
                rec["factor_date"] = r["date"]
                for c in numeric_cols:
                    rec[c] = r[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def current_v3_mask(df):
    def s(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    return (
        ((s("runup_ma5_exhaustion") > 0.8) & (s("cpv_factor") <= -0.4))
        | (
            ((s("prev_day_blowoff_risk") >= 2.0) & (s("sector_momentum_accel_rank") <= 0.30))
            | ((s("prev_day_blowoff_risk") > 5.0) & (s("sector_momentum_accel_rank") <= 0.50))
            | ((s("prev_day_blowoff_risk") > 1.0) & (s("sector_momentum_accel_rank") <= 0.15))
        )
        | ((s("sector_leadership_persistence_v3") <= -0.18) & (s("return_20d_intra_z") <= 0.40))
        | (
            (s("market_liquidity_change") < 0.94)
            & (
                (s("ulcer_index") < -0.8)
                | (s("momentum_peak_decay_3d") > 1.0)
                | (s("intra_sector_momentum_rank") < 0.67)
            )
        )
        | ((s("mkt_dispersion_shock_v2") < 0.98) & (s("market_advance_decline_ratio") < 0.38))
        | (s("limit_down_3d", 0).fillna(0).astype(float) == 1)
    ).fillna(False)


def dedup_reference(df):
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    x = df.copy()
    x["priority"] = x["dataset"].map(priority)
    x = x.sort_values(["key", "priority"])
    return x.drop_duplicates("key", keep="first").drop(columns=["priority"])


def make_groups(df):
    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
        & ~df["current_v3"]
    ].copy()
    ref = dedup_reference(df)
    hist = ref[~ref["key"].isin(recent["key"])].copy()
    wins = hist[hist["actual_return"] >= 0.10].copy()
    losses = hist[hist["actual_return"] <= -0.05].copy()
    neutral = hist[(hist["actual_return"] > -0.05) & (hist["actual_return"] < 0.10)].copy()
    return recent, wins, losses, neutral, hist


def zdist(a, b):
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(av) < 2 or len(bv) < 3:
        return np.nan
    pooled = np.nanstd(np.r_[av, bv])
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (np.nanmedian(av) - np.nanmedian(bv)) / pooled


def hit_counts(series, th, direction, recent, wins, losses, neutral):
    if direction == "high":
        return {
            "recent": int((recent[series] >= th).sum()),
            "wins": int((wins[series] >= th).sum()),
            "losses": int((losses[series] >= th).sum()),
            "neutral": int((neutral[series] >= th).sum()),
        }
    return {
        "recent": int((recent[series] <= th).sum()),
        "wins": int((wins[series] <= th).sum()),
        "losses": int((losses[series] <= th).sum()),
        "neutral": int((neutral[series] <= th).sum()),
    }


def monotonicity(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3:
        return np.nan
    diffs = np.diff(arr)
    if np.all(diffs >= 0):
        return 1.0
    if np.all(diffs <= 0):
        return 1.0
    same_dir = max(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return float(same_dir)


def analyze_single_factor(c, recent, wins, losses, neutral, hist):
    allv = pd.to_numeric(hist[c], errors="coerce").dropna()
    rv = pd.to_numeric(recent[c], errors="coerce").dropna()
    if len(allv) < 25 or len(rv) < max(2, len(recent) // 2):
        return []
    zw = zdist(rv, wins[c])
    if not np.isfinite(zw):
        return []
    direction = "high" if zw > 0 else "low"
    if direction == "high":
        candidate_thresholds = sorted({float(rv.quantile(q)) for q in [0.0, 0.25, 0.5, 0.75] if pd.notna(rv.quantile(q))})
    else:
        candidate_thresholds = sorted({float(rv.quantile(q)) for q in [0.25, 0.5, 0.75, 1.0] if pd.notna(rv.quantile(q))})

    out = []
    hist_quantiles = allv.quantile(np.linspace(0.05, 0.95, 19))
    for th in candidate_thresholds:
        hc = hit_counts(c, th, direction, recent, wins, losses, neutral)
        if hc["recent"] < 2:
            continue
        pct = float((allv <= th).mean())
        # For high-direction risk, more stringent is higher threshold. For low-direction risk, more stringent is lower threshold.
        if direction == "high":
            grid = [q for q in hist_quantiles.values if q <= max(th, allv.quantile(0.95))]
            grid = sorted(set(float(x) for x in grid if np.isfinite(x)))
            nearby = [x for x in grid if abs((allv <= x).mean() - pct) <= 0.15]
            nearby = sorted(nearby)
        else:
            grid = sorted(set(float(x) for x in hist_quantiles.values if np.isfinite(x)))
            nearby = [x for x in grid if abs((allv <= x).mean() - pct) <= 0.15]
            nearby = sorted(nearby)
        if len(nearby) < 3:
            nearby = sorted(set([float(allv.quantile(q)) for q in np.clip(np.linspace(pct - 0.15, pct + 0.15, 7), 0.01, 0.99)]))

        curve = []
        for x in nearby:
            h = hit_counts(c, x, direction, recent, wins, losses, neutral)
            curve.append(
                {
                    "threshold": x,
                    "pct": float((allv <= x).mean()),
                    **h,
                }
            )
        if direction == "high":
            ordered = sorted(curve, key=lambda r: r["threshold"])
            recent_seq = [r["recent"] for r in ordered]
            win_seq = [r["wins"] for r in ordered]
        else:
            ordered = sorted(curve, key=lambda r: r["threshold"])
            recent_seq = [r["recent"] for r in ordered]
            win_seq = [r["wins"] for r in ordered]
        mono_recent = monotonicity(recent_seq)
        mono_wins = monotonicity(win_seq)
        score = (
            hc["recent"] * 10
            + hc["losses"] * 1.5
            - hc["wins"] * 5
            - hc["neutral"] * 0.8
            + abs(zw) * 2
            + (mono_recent + mono_wins) * 2
        )
        out.append(
            {
                "factor": c,
                "direction": direction,
                "threshold": th,
                "hist_percentile_leq": pct,
                "recent_hits": hc["recent"],
                "win_hits": hc["wins"],
                "hist_loss_hits": hc["losses"],
                "neutral_hits": hc["neutral"],
                "recent_n": len(recent),
                "win_n": len(wins),
                "hist_loss_n": len(losses),
                "recent_median": float(rv.median()),
                "win_median": float(pd.to_numeric(wins[c], errors="coerce").median()),
                "hist_loss_median": float(pd.to_numeric(losses[c], errors="coerce").median()),
                "z_vs_win": float(zw),
                "z_vs_loss": float(zdist(rv, losses[c])),
                "mono_recent": mono_recent,
                "mono_wins": mono_wins,
                "score": score,
                "curve": curve,
            }
        )
    return out


def curve_to_text(curve):
    parts = []
    for r in curve:
        parts.append(
            "p{:.0f}:th={:.4g},R={},W={},L={}".format(
                r["pct"] * 100,
                r["threshold"],
                r["recent"],
                r["wins"],
                r["losses"],
            )
        )
    return " | ".join(parts)


def print_results(results):
    print("\n== Top Single-Factor Robust Thresholds ==")
    show = [
        "factor",
        "direction",
        "threshold",
        "hist_percentile_leq",
        "recent_hits",
        "win_hits",
        "hist_loss_hits",
        "neutral_hits",
        "recent_median",
        "win_median",
        "hist_loss_median",
        "z_vs_win",
        "mono_recent",
        "mono_wins",
        "score",
    ]
    display = pd.DataFrame([{k: r[k] for k in show} for r in results[:50]])
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n== Threshold Neighborhood Curves For Top 20 ==")
    for r in results[:20]:
        op = ">=" if r["direction"] == "high" else "<="
        print(
            f"\n{r['factor']} {op} {r['threshold']:.4g} "
            f"(hist_pct_leq={r['hist_percentile_leq']:.1%}, score={r['score']:.2f}, "
            f"monoR={r['mono_recent']:.2f}, monoW={r['mono_wins']:.2f})"
        )
        print(curve_to_text(r["curve"]))


def main():
    events = load_events()
    factors, numeric_cols = load_factor_frame()
    df = enrich(events, factors, numeric_cols)
    df["current_v3"] = current_v3_mask(df)
    recent, wins, losses, neutral, hist = make_groups(df)

    print("== Inputs ==")
    print(f"numeric_factor_count={len(numeric_cols)}")
    print(f"recent_v3_leak_losses={len(recent)}")
    print(f"historical_big_wins={len(wins)}")
    print(f"historical_big_losses={len(losses)}")
    print(f"historical_neutral={len(neutral)}")

    all_results = []
    for c in numeric_cols:
        if c in {"is_st", "is_new_stock"}:
            continue
        all_results.extend(analyze_single_factor(c, recent, wins, losses, neutral, hist))

    all_results.sort(
        key=lambda r: (
            -r["score"],
            r["win_hits"],
            -r["recent_hits"],
            -r["mono_recent"],
            -r["mono_wins"],
        )
    )
    out = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "curve"} | {"curve": curve_to_text(r["curve"])}
            for r in all_results
        ]
    )
    out.to_csv(RESULTS / "factor_threshold_robustness_monotonicity.csv", index=False, encoding="utf-8-sig")
    print_results(all_results)

    # Pair scan over top robust single factors, using each candidate threshold as-is.
    top = all_results[:80]
    pair_rows = []
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            if a["factor"] == b["factor"]:
                continue
            ma_r = recent[a["factor"]] >= a["threshold"] if a["direction"] == "high" else recent[a["factor"]] <= a["threshold"]
            mb_r = recent[b["factor"]] >= b["threshold"] if b["direction"] == "high" else recent[b["factor"]] <= b["threshold"]
            ma_w = wins[a["factor"]] >= a["threshold"] if a["direction"] == "high" else wins[a["factor"]] <= a["threshold"]
            mb_w = wins[b["factor"]] >= b["threshold"] if b["direction"] == "high" else wins[b["factor"]] <= b["threshold"]
            ma_l = losses[a["factor"]] >= a["threshold"] if a["direction"] == "high" else losses[a["factor"]] <= a["threshold"]
            mb_l = losses[b["factor"]] >= b["threshold"] if b["direction"] == "high" else losses[b["factor"]] <= b["threshold"]
            ma_n = neutral[a["factor"]] >= a["threshold"] if a["direction"] == "high" else neutral[a["factor"]] <= a["threshold"]
            mb_n = neutral[b["factor"]] >= b["threshold"] if b["direction"] == "high" else neutral[b["factor"]] <= b["threshold"]
            rh = int((ma_r & mb_r).sum())
            wh = int((ma_w & mb_w).sum())
            lh = int((ma_l & mb_l).sum())
            nh = int((ma_n & mb_n).sum())
            if rh < 2:
                continue
            score = rh * 12 + lh * 1.5 - wh * 8 - nh * 0.8 + (a["mono_recent"] + b["mono_recent"])
            opa = ">=" if a["direction"] == "high" else "<="
            opb = ">=" if b["direction"] == "high" else "<="
            pair_rows.append(
                {
                    "score": score,
                    "recent_hits": rh,
                    "win_hits": wh,
                    "hist_loss_hits": lh,
                    "neutral_hits": nh,
                    "rule": f"{a['factor']} {opa} {a['threshold']:.4g} & {b['factor']} {opb} {b['threshold']:.4g}",
                    "factor_a": a["factor"],
                    "factor_b": b["factor"],
                    "threshold_a": a["threshold"],
                    "threshold_b": b["threshold"],
                    "pct_a": a["hist_percentile_leq"],
                    "pct_b": b["hist_percentile_leq"],
                    "mono_a": a["mono_recent"],
                    "mono_b": b["mono_recent"],
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(
        ["score", "recent_hits", "win_hits", "neutral_hits"],
        ascending=[False, False, True, True],
    )
    pairs.to_csv(RESULTS / "factor_pair_threshold_robustness_monotonicity.csv", index=False, encoding="utf-8-sig")
    print("\n== Top Pair Rules ==")
    print(pairs.head(40).to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
