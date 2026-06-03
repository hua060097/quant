import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
MONEYFLOW_DIR = BASE.parent / "local_data" / "moneyflow_ths"

RUN_FILES = {
    "old162_off": RESULTS / "walkforward_metrics_20260512_194610.json",
    "latest184_v3": RESULTS / "walkforward_metrics_20260520_190404.json",
    "latest185_off": RESULTS / "walkforward_metrics_20260520_210454.json",
}

MF_COLS = [
    "trade_date",
    "ts_code",
    "pct_change",
    "net_amount",
    "net_d5_amount",
    "buy_lg_amount_rate",
    "buy_md_amount_rate",
    "buy_sm_amount_rate",
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


def load_moneyflow(stocks: set[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for path in sorted(MONEYFLOW_DIR.glob("*.parquet")):
        date_text = path.stem
        try:
            d = pd.Timestamp(date_text)
        except Exception:
            continue
        if d < start or d > end:
            continue
        df = pd.read_parquet(path, columns=MF_COLS)
        df = df[df["ts_code"].isin(stocks)].copy()
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No moneyflow data loaded from {MONEYFLOW_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"trade_date": "date", "ts_code": "stock_code"})
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    for c in out.columns:
        if c not in {"date", "stock_code"}:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)
    out["lg_minus_sm"] = out["buy_lg_amount_rate"] - out["buy_sm_amount_rate"]
    out["lg_plus_sm"] = out["buy_lg_amount_rate"] + out["buy_sm_amount_rate"]
    return out


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def max_drawdown_pct_from_returns(pct_changes: np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(pct_changes), errors="coerce").dropna().to_numpy(dtype=float) / 100.0
    if len(vals) == 0:
        return np.nan
    equity = np.cumprod(1.0 + vals)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min() * 100.0)


def compute_features_for_window(w: pd.DataFrame, n: int) -> dict[str, float]:
    tail = w.tail(n)
    rec: dict[str, float] = {}
    for col, short in [
        ("buy_sm_amount_rate", "sm"),
        ("buy_lg_amount_rate", "lg"),
        ("buy_md_amount_rate", "md"),
        ("lg_minus_sm", "lg_minus_sm"),
        ("lg_plus_sm", "lg_plus_sm"),
        ("net_amount", "net"),
        ("net_d5_amount", "net_d5"),
        ("pct_change", "px"),
    ]:
        s = pd.to_numeric(tail[col], errors="coerce")
        rec[f"{short}_{n}d_sum"] = float(s.sum()) if s.notna().any() else np.nan
        rec[f"{short}_{n}d_mean"] = float(s.mean()) if s.notna().any() else np.nan
        rec[f"{short}_{n}d_min"] = float(s.min()) if s.notna().any() else np.nan
        rec[f"{short}_{n}d_max"] = float(s.max()) if s.notna().any() else np.nan
        rec[f"{short}_{n}d_slope"] = slope(s.to_numpy())

    sm = pd.to_numeric(tail["buy_sm_amount_rate"], errors="coerce")
    lg = pd.to_numeric(tail["buy_lg_amount_rate"], errors="coerce")
    px = pd.to_numeric(tail["pct_change"], errors="coerce")
    diff = pd.to_numeric(tail["lg_minus_sm"], errors="coerce")

    rec[f"sm_{n}d_negative_days"] = int((sm < 0).sum())
    rec[f"lg_{n}d_negative_days"] = int((lg < 0).sum())
    rec[f"both_lg_sm_negative_{n}d_days"] = int(((sm < 0) & (lg < 0)).sum())
    rec[f"lg_pos_sm_neg_{n}d_days"] = int(((sm < 0) & (lg > 0)).sum())
    rec[f"px_up_sm_neg_{n}d_days"] = int(((sm < 0) & (px > 0)).sum())
    rec[f"px_up_lg_neg_{n}d_days"] = int(((lg < 0) & (px > 0)).sum())

    rec[f"price_up_sm_outflow_{n}d"] = max(0.0, rec[f"px_{n}d_sum"]) * max(0.0, -rec[f"sm_{n}d_sum"])
    rec[f"price_up_lg_outflow_{n}d"] = max(0.0, rec[f"px_{n}d_sum"]) * max(0.0, -rec[f"lg_{n}d_sum"])
    rec[f"price_up_both_outflow_{n}d"] = max(0.0, rec[f"px_{n}d_sum"]) * max(0.0, -rec[f"lg_plus_sm_{n}d_sum"])
    rec[f"price_up_lg_sm_divergence_{n}d"] = max(0.0, rec[f"px_{n}d_sum"]) * max(0.0, rec[f"lg_minus_sm_{n}d_mean"])
    rec[f"sm_outflow_lg_sm_divergence_{n}d"] = max(0.0, -rec[f"sm_{n}d_sum"]) * max(0.0, rec[f"lg_minus_sm_{n}d_mean"])
    rec[f"sm_outflow_strength_{n}d"] = max(0.0, -rec[f"sm_{n}d_sum"])
    rec[f"lg_outflow_strength_{n}d"] = max(0.0, -rec[f"lg_{n}d_sum"])
    rec[f"both_outflow_strength_{n}d"] = max(0.0, -rec[f"lg_plus_sm_{n}d_sum"])
    rec[f"moneyflow_divergence_{n}d"] = rec[f"lg_minus_sm_{n}d_mean"]
    rec[f"moneyflow_divergence_slope_{n}d"] = rec[f"lg_minus_sm_{n}d_slope"]
    rec[f"px_drawdown_{n}d"] = max_drawdown_pct_from_returns(px.to_numpy())
    rec[f"px_up_and_recent_drawdown_{n}d"] = max(0.0, rec[f"px_{n}d_sum"]) * max(0.0, -rec[f"px_drawdown_{n}d"])
    return rec


def attach_path_features(events: pd.DataFrame, mf: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        stock: (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
        for stock, g in mf.groupby("stock_code", sort=False)
    }
    rows = []
    for _, e in events.iterrows():
        rec = e.to_dict()
        item = grouped.get(e["stock_code"])
        if item is not None:
            dates, g = item
            pos = np.searchsorted(dates, np.datetime64(e["buy_date"]), side="left")
            before = g.iloc[: int(pos)]
            if not before.empty:
                rec["moneyflow_last_date"] = before.iloc[-1]["date"]
                rec["moneyflow_days_available"] = len(before)
                for n in [3, 5, 10]:
                    if len(before) >= min(3, n):
                        rec.update(compute_features_for_window(before, n))
        rows.append(rec)
    return pd.DataFrame(rows)


def make_groups(df: pd.DataFrame):
    priority = {"latest185_off": 0, "old162_off": 1, "latest184_v3": 2}
    ref = df.copy()
    ref["priority"] = ref["dataset"].map(priority)
    ref = ref.sort_values(["key", "priority"]).drop_duplicates("key", keep="first").drop(columns=["priority"])
    recent = df[
        df["dataset"].eq("latest184_v3")
        & (df["buy_date"] >= pd.Timestamp("2026-04-01"))
        & (df["actual_return"] <= -0.05)
    ].copy()
    hist = ref[~ref["key"].isin(recent["key"])].copy()
    wins = hist[hist["actual_return"] >= 0.10].copy()
    losses = hist[hist["actual_return"] <= -0.05].copy()
    neutral = hist[(hist["actual_return"] > -0.05) & (hist["actual_return"] < 0.10)].copy()
    return recent, wins, losses, neutral, hist


def zdist(a: pd.Series, b: pd.Series) -> float:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(av) < 2 or len(bv) < 3:
        return np.nan
    pooled = np.nanstd(np.r_[av, bv])
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.nanmedian(av) - np.nanmedian(bv)) / pooled)


def monotonicity(values: list[int]) -> float:
    if len(values) < 3:
        return np.nan
    diffs = np.diff(np.asarray(values, dtype=float))
    return float(max(np.mean(diffs >= 0), np.mean(diffs <= 0)))


def eval_threshold(feature: str, direction: str, th: float, recent, wins, losses, neutral, hist) -> dict:
    def mask(frame):
        if direction == "high":
            return pd.to_numeric(frame[feature], errors="coerce") >= th
        return pd.to_numeric(frame[feature], errors="coerce") <= th

    mr, mw, ml, mn = mask(recent), mask(wins), mask(losses), mask(neutral)
    allv = pd.to_numeric(hist[feature], errors="coerce").dropna()
    pct = float((allv <= th).mean()) if len(allv) else np.nan

    qs = np.clip(np.linspace(pct - 0.15, pct + 0.15, 7), 0.01, 0.99)
    curve = []
    for q in qs:
        x = float(allv.quantile(q))
        curve.append(
            {
                "pct": q,
                "threshold": x,
                "recent": int(mask_at(recent[feature], direction, x).sum()),
                "wins": int(mask_at(wins[feature], direction, x).sum()),
                "losses": int(mask_at(losses[feature], direction, x).sum()),
            }
        )
    mono_r = monotonicity([c["recent"] for c in curve])
    mono_w = monotonicity([c["wins"] for c in curve])
    return {
        "feature": feature,
        "direction": direction,
        "threshold": th,
        "hist_percentile_leq": pct,
        "recent_hits": int(mr.sum()),
        "win_hits": int(mw.sum()),
        "hist_loss_hits": int(ml.sum()),
        "neutral_hits": int(mn.sum()),
        "recent_median": float(pd.to_numeric(recent[feature], errors="coerce").median()),
        "win_median": float(pd.to_numeric(wins[feature], errors="coerce").median()),
        "hist_loss_median": float(pd.to_numeric(losses[feature], errors="coerce").median()),
        "z_vs_win": zdist(recent[feature], wins[feature]),
        "z_vs_loss": zdist(recent[feature], losses[feature]),
        "mono_recent": mono_r,
        "mono_wins": mono_w,
        "score": int(mr.sum()) * 10 + int(ml.sum()) * 1.5 - int(mw.sum()) * 7 - int(mn.sum()) * 0.7 + (mono_r + mono_w) * 2,
        "curve": " | ".join(
            f"p{c['pct']*100:.0f}:th={c['threshold']:.3g},R={c['recent']},W={c['wins']},L={c['losses']}"
            for c in curve
        ),
    }


def mask_at(series: pd.Series, direction: str, th: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s >= th if direction == "high" else s <= th


def analyze_features(df: pd.DataFrame) -> pd.DataFrame:
    recent, wins, losses, neutral, hist = make_groups(df)
    meta = {
        "dataset",
        "stock_code",
        "buy_date",
        "actual_return",
        "profit_loss",
        "key",
        "moneyflow_last_date",
        "moneyflow_days_available",
    }
    features = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for f in features:
        rv = pd.to_numeric(recent[f], errors="coerce").dropna()
        hv = pd.to_numeric(hist[f], errors="coerce").dropna()
        if len(rv) < 3 or len(hv) < 30:
            continue
        zd = zdist(recent[f], wins[f])
        if not np.isfinite(zd):
            continue
        direction = "high" if zd > 0 else "low"
        qs = [0.0, 0.25, 0.5] if direction == "high" else [0.5, 0.75, 1.0]
        for q in qs:
            th = float(rv.quantile(q))
            row = eval_threshold(f, direction, th, recent, wins, losses, neutral, hist)
            if row["recent_hits"] >= 2:
                rows.append(row)
    out = pd.DataFrame(rows).sort_values(
        ["score", "recent_hits", "win_hits", "neutral_hits"],
        ascending=[False, False, True, True],
    )
    return out


def print_hits(df: pd.DataFrame, feature: str, direction: str, threshold: float):
    mask = mask_at(df[feature], direction, threshold)
    print(f"\n== Hits: {feature} {'>=' if direction == 'high' else '<='} {threshold:.4g} ==")
    for _, r in df[mask].sort_values(["actual_return", "buy_date"]).head(20).iterrows():
        print(
            "{ds} {date} {code} ret={ret:.2%} {feature}={val:.3f}".format(
                ds=r["dataset"],
                date=r["buy_date"].date(),
                code=r["stock_code"],
                ret=r["actual_return"],
                feature=feature,
                val=r[feature],
            )
        )


def main():
    events = load_events()
    start = events["buy_date"].min() - pd.Timedelta(days=30)
    end = events["buy_date"].max()
    mf = load_moneyflow(set(events["stock_code"]), start, end)
    df = attach_path_features(events, mf)
    df.to_csv(RESULTS / "moneyflow_path_candidate_trade_features.csv", index=False, encoding="utf-8-sig")

    recent, wins, losses, neutral, hist = make_groups(df)
    print("== Inputs ==")
    print(f"events={len(df)}, recent_losses={len(recent)}, hist_bigwins={len(wins)}, hist_bigloss={len(losses)}, neutral={len(neutral)}")
    print("\nRecent losses moneyflow path snapshot:")
    snap_cols = [
        "stock_code",
        "buy_date",
        "actual_return",
        "sm_3d_sum",
        "sm_5d_sum",
        "lg_3d_sum",
        "lg_5d_sum",
        "lg_minus_sm_3d_mean",
        "lg_minus_sm_5d_mean",
        "px_3d_sum",
        "px_5d_sum",
        "price_up_sm_outflow_3d",
        "price_up_sm_outflow_5d",
        "sm_outflow_lg_sm_divergence_3d",
    ]
    print(recent[[c for c in snap_cols if c in recent.columns]].sort_values("actual_return").to_string(index=False))

    res = analyze_features(df)
    res.to_csv(RESULTS / "moneyflow_path_candidate_factor_scores.csv", index=False, encoding="utf-8-sig")
    print("\n== Top moneyflow path candidate factors ==")
    show = [
        "feature",
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
    print(res[show].head(50).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n== Curves for top 12 ==")
    for _, r in res.head(12).iterrows():
        print(f"\n{r['feature']} {r['direction']} th={r['threshold']:.4g}, score={r['score']:.2f}")
        print(r["curve"])
        print_hits(df, r["feature"], r["direction"], r["threshold"])


if __name__ == "__main__":
    main()
