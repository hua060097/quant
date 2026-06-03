import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"
MONEYFLOW_FEATURE_FILE = RESULTS / "moneyflow_path_candidate_trade_features.csv"

RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}

CORE_MONEYFLOW = [
    "lg_pos_sm_neg_10d_days",
    "price_up_lg_sm_divergence_10d",
    "sm_outflow_lg_sm_divergence_10d",
    "moneyflow_divergence_10d",
    "lg_minus_sm_10d_mean",
    "price_up_sm_outflow_10d",
    "sm_outflow_strength_10d",
]


def load_events() -> pd.DataFrame:
    rows = []
    for ds, path in RUN_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for t in data[0]["all_results"]["1"]["trades"]:
            rows.append(
                {
                    "dataset": ds,
                    "stock_code": t["stock_code"],
                    "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                    "actual_return": float(t["actual_return"]),
                    "profit_loss": float(t["profit_loss"]),
                }
            )
    out = pd.DataFrame(rows)
    out["key"] = out["stock_code"] + "|" + out["buy_date"].dt.strftime("%Y-%m-%d")
    return out


def load_factor_frame():
    df = pd.read_parquet(FACTOR_FILE)
    df["date"] = pd.to_datetime(df["date"])
    skip = {"date", "stock_code", "stock_name", "name", "sw_l1", "sw_l2", "sw_l3", "industry"}
    numeric_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return df[["date", "stock_code"] + numeric_cols].sort_values(["stock_code", "date"]), numeric_cols


def attach_factor_snapshot(events: pd.DataFrame, factors: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    grouped = {
        stock: (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
        for stock, g in factors.groupby("stock_code", sort=False)
    }
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


def load_moneyflow_features() -> pd.DataFrame:
    if not MONEYFLOW_FEATURE_FILE.exists():
        raise RuntimeError(
            f"Missing {MONEYFLOW_FEATURE_FILE}. Run analyze_moneyflow_path_candidate_factors.py first."
        )
    mf = pd.read_csv(MONEYFLOW_FEATURE_FILE, parse_dates=["buy_date"])
    mf["key"] = mf["stock_code"] + "|" + mf["buy_date"].dt.strftime("%Y-%m-%d")
    keep = ["key"] + [c for c in mf.columns if c not in {"dataset", "stock_code", "buy_date", "actual_return", "profit_loss", "key"}]
    keep = list(dict.fromkeys(keep))
    return mf[keep]


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


def make_groups(df):
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    ref = df.copy()
    ref["priority"] = ref["dataset"].map(priority)
    ref = ref.sort_values(["key", "priority"]).drop_duplicates("key", keep="first").drop(columns=["priority"])
    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
        & ~df["current_v3"]
    ].drop_duplicates("key", keep="first").copy()
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
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.nanmedian(av) - np.nanmedian(bv)) / pooled)


def build_feature_pool(df, recent, wins, losses, hist):
    meta = {
        "dataset",
        "stock_code",
        "buy_date",
        "actual_return",
        "profit_loss",
        "key",
        "factor_date",
        "moneyflow_last_date",
        "moneyflow_days_available",
    }
    rows = []
    for c in df.columns:
        if c in meta or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        rv = pd.to_numeric(recent[c], errors="coerce").dropna()
        hv = pd.to_numeric(hist[c], errors="coerce").dropna()
        if len(rv) < 3 or len(hv) < 35:
            continue
        zw = zdist(recent[c], wins[c])
        if not np.isfinite(zw):
            continue
        direction = "high" if zw > 0 else "low"
        rows.append(
            {
                "feature": c,
                "direction": direction,
                "abs_z_win": abs(zw),
                "z_win": zw,
                "z_loss": zdist(recent[c], losses[c]),
                "recent_median": float(rv.median()),
                "win_median": float(pd.to_numeric(wins[c], errors="coerce").median()),
                "loss_median": float(pd.to_numeric(losses[c], errors="coerce").median()),
            }
        )
    pool = pd.DataFrame(rows).sort_values("abs_z_win", ascending=False)
    # Keep all core moneyflow candidates plus strongest positive/negative base factors.
    keep_features = set([c for c in CORE_MONEYFLOW if c in df.columns])
    keep_features.update(pool.head(80)["feature"].tolist())
    keep_features.update(pool[pool["direction"].eq("low")].head(30)["feature"].tolist())
    pool = pool[pool["feature"].isin(keep_features)].copy()
    return pool


def threshold_candidates(feature, direction, recent, hist):
    rv = pd.to_numeric(recent[feature], errors="coerce").dropna()
    hv = pd.to_numeric(hist[feature], errors="coerce").dropna()
    if rv.empty or hv.empty:
        return []
    qs = [0.0, 0.25, 0.50, 0.75] if direction == "high" else [0.25, 0.50, 0.75, 1.0]
    out = []
    for q in qs:
        th = float(rv.quantile(q))
        if not np.isfinite(th):
            continue
        pct = float((hv <= th).mean())
        # Avoid thresholds that are almost all-market unless paired with a core moneyflow feature.
        if 0.02 <= pct <= 0.98:
            out.append((th, pct))
    return list(dict.fromkeys(out))


def mask(frame, feature, direction, threshold):
    s = pd.to_numeric(frame[feature], errors="coerce")
    return s >= threshold if direction == "high" else s <= threshold


def summarize(frame, hit):
    h = frame[hit.fillna(False)]
    return {
        "n": int(len(h)),
        "loss": int((h["actual_return"] <= -0.05).sum()),
        "bigwin": int((h["actual_return"] >= 0.10).sum()),
        "pos": int((h["actual_return"] > 0).sum()),
        "pnl": float(h["profit_loss"].sum()) if len(h) else 0.0,
        "avg": float(h["actual_return"].mean()) if len(h) else 0.0,
    }


def combo_curve(combo, recent, wins, losses, hist):
    # Move the last condition around its threshold to test neighborhood stability.
    f, direction, th, pct = combo[-1]
    hv = pd.to_numeric(hist[f], errors="coerce").dropna()
    qs = np.clip(np.linspace(pct - 0.12, pct + 0.12, 5), 0.01, 0.99)
    parts = []
    for q in qs:
        x = float(hv.quantile(q))
        test = combo[:-1] + [(f, direction, x, q)]
        mr = combo_mask(recent, test)
        mw = combo_mask(wins, test)
        ml = combo_mask(losses, test)
        parts.append(f"p{q*100:.0f}:R={int(mr.sum())},W={int(mw.sum())},L={int(ml.sum())}")
    return " | ".join(parts)


def combo_mask(frame, combo):
    out = pd.Series(True, index=frame.index)
    for f, direction, th, _pct in combo:
        out &= mask(frame, f, direction, th)
    return out.fillna(False)


def score_combo(combo, recent, wins, losses, neutral, hist):
    mr = combo_mask(recent, combo)
    mw = combo_mask(wins, combo)
    ml = combo_mask(losses, combo)
    mn = combo_mask(neutral, combo)
    sr, sw, sl, sn = summarize(recent, mr), summarize(wins, mw), summarize(losses, ml), summarize(neutral, mn)
    if sr["n"] < 2:
        return None
    score = sr["n"] * 14 + sl["n"] * 2.0 - sw["n"] * 10 - sn["n"] * 1.0
    return {
        "score": score,
        "recent_hits": sr["n"],
        "win_hits": sw["n"],
        "hist_loss_hits": sl["n"],
        "neutral_hits": sn["n"],
        "recent_avg": sr["avg"],
        "win_avg": sw["avg"],
        "loss_avg": sl["avg"],
        "recent_pnl": sr["pnl"],
        "rule": " & ".join(
            f"{f} {'>=' if d == 'high' else '<='} {th:.4g} (p_leq={pct:.2f})"
            for f, d, th, pct in combo
        ),
        "features": "|".join(f for f, *_ in combo),
        "combo": combo,
    }


def main():
    events = load_events()
    factors, numeric_cols = load_factor_frame()
    base = attach_factor_snapshot(events, factors, numeric_cols)
    mf = load_moneyflow_features()
    df = base.merge(mf, on="key", how="left", suffixes=("", "_mf"))
    df["current_v3"] = current_v3_mask(df)
    recent, wins, losses, neutral, hist = make_groups(df)
    print("== Inputs ==")
    print(
        f"events={len(df)}, recent_v3_leak_losses={len(recent)}, "
        f"hist_bigwins={len(wins)}, hist_bigloss={len(losses)}, neutral={len(neutral)}"
    )

    pool = build_feature_pool(df, recent, wins, losses, hist)
    pool.to_csv(RESULTS / "moneyflow_combo_feature_pool.csv", index=False, encoding="utf-8-sig")
    print("\n== Feature pool top 40 ==")
    print(pool.head(40).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    candidates = []
    for _, row in pool.iterrows():
        thresholds = threshold_candidates(row["feature"], row["direction"], recent, hist)
        for th, pct in thresholds:
            candidates.append((row["feature"], row["direction"], th, pct))

    core = [c for c in candidates if c[0] in CORE_MONEYFLOW]
    noncore = [c for c in candidates if c[0] not in CORE_MONEYFLOW]
    results = []

    # Pairs: require at least one moneyflow-path feature.
    for a in core:
        for b in noncore + [x for x in core if x[0] != a[0]]:
            if a[0] == b[0]:
                continue
            res = score_combo([a, b], recent, wins, losses, neutral, hist)
            if res is not None:
                results.append(res)

    # Triples: one core + two noncore among strongest subset.
    strong_noncore = noncore[:35]
    for a in core:
        for b, c in combinations(strong_noncore, 2):
            if len({a[0], b[0], c[0]}) < 3:
                continue
            res = score_combo([a, b, c], recent, wins, losses, neutral, hist)
            if res is not None:
                # Penalize complexity slightly.
                res["score"] -= 3
                results.append(res)

    out = pd.DataFrame([{k: v for k, v in r.items() if k != "combo"} for r in results])
    out = out.sort_values(
        ["score", "recent_hits", "win_hits", "neutral_hits", "hist_loss_hits"],
        ascending=[False, False, True, True, False],
    )
    combo_by_rule = {r["rule"]: r["combo"] for r in results}
    top_rules = set(out.head(120)["rule"])
    out["curve_last"] = [
        combo_curve(combo_by_rule[rule], recent, wins, losses, hist) if rule in top_rules else ""
        for rule in out["rule"]
    ]
    out.to_csv(RESULTS / "moneyflow_path_factor_combo_scores.csv", index=False, encoding="utf-8-sig")
    print("\n== Top combo candidates ==")
    print(out.head(60).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n== Hit details for top 10 ==")
    for _, row in out.head(10).iterrows():
        combo = next(r["combo"] for r in results if r["rule"] == row["rule"])
        hit = combo_mask(df, combo)
        print(f"\n-- {row['rule']} --")
        for ds in sorted(df["dataset"].unique()):
            d = df[df["dataset"].eq(ds)]
            s = summarize(d, hit.loc[d.index])
            print(
                f"  {ds}: n={s['n']} loss={s['loss']} bigwin={s['bigwin']} pos={s['pos']} avg={s['avg']:.2%} pnl={s['pnl']:.0f}"
            )
        for _, r in df[hit].sort_values(["actual_return", "buy_date"]).head(14).iterrows():
            print(
                "    {ds} {date} {code} ret={ret:.2%}".format(
                    ds=r["dataset"],
                    date=r["buy_date"].date(),
                    code=r["stock_code"],
                    ret=r["actual_return"],
                )
            )


if __name__ == "__main__":
    main()
