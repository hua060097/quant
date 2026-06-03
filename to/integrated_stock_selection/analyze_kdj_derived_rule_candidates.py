import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"
PRICE_FILE = BASE / "cache" / "stock_features_20221201_20260521.parquet"
METRICS_185 = BASE / "results" / "walkforward_metrics_20260520_210454.json"
SHADOW_185 = BASE / "results" / "walkforward_shadow_skipped_opportunities_20260520_210454.csv"

HOT_FACTOR = "prev_non_first_limit_up_performance_percentile_20d"

BASE_FACTOR_COLS = [
    HOT_FACTOR,
    "kdj_j_9",
    "kdj_diff_9",
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
]


def load_events() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = json.loads(METRICS_185.read_text(encoding="utf-8"))
    actual_rows = []
    for t in data[0]["all_results"]["1"]["trades"]:
        actual_rows.append(
            {
                "event_type": "actual",
                "stock_code": t["stock_code"],
                "buy_date": pd.Timestamp(t["buy_datetime"][:10]),
                "buy_datetime": pd.Timestamp(t["buy_datetime"]),
                "actual_return": float(t["actual_return"]),
                "profit_loss": float(t["profit_loss"]),
            }
        )
    actual = pd.DataFrame(actual_rows)

    if SHADOW_185.exists():
        sh = pd.read_csv(SHADOW_185)
        shadow = pd.DataFrame(
            {
                "event_type": "shadow",
                "stock_code": sh["stock_code"].astype(str),
                "buy_date": pd.to_datetime(sh["buy_datetime"].astype(str).str[:10]),
                "buy_datetime": pd.to_datetime(sh["buy_datetime"]),
                "actual_return": pd.to_numeric(sh["actual_return"], errors="coerce"),
                "profit_loss": pd.to_numeric(sh["profit_loss"], errors="coerce"),
            }
        )
    else:
        shadow = pd.DataFrame(columns=actual.columns)

    return actual, shadow, pd.concat([actual, shadow], ignore_index=True, sort=False)


def compute_raw_kdj_features() -> pd.DataFrame:
    cols = ["date", "stock_code", "high", "low", "close"]
    df = pd.read_parquet(PRICE_FILE, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    def calc_one(g: pd.DataFrame) -> pd.DataFrame:
        low9 = g["low"].rolling(9, min_periods=9).min()
        high9 = g["high"].rolling(9, min_periods=9).max()
        rsv = ((g["close"] - low9) / (high9 - low9 + 1e-9)) * 100.0
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame(
            {
                "kdj_k_9_raw": k,
                "kdj_d_9_raw": d,
                "kdj_j_9_raw": j,
                "kdj_diff_9_raw": k - d,
            },
            index=g.index,
        )

    kdj = df.groupby("stock_code", sort=False, group_keys=False).apply(calc_one)
    out = pd.concat([df[["date", "stock_code"]], kdj], axis=1)

    out["kdj_j_9_raw_pct"] = out.groupby("date")["kdj_j_9_raw"].transform(
        lambda x: x.rank(method="average", pct=True)
    )
    out["kdj_diff_9_raw_pct"] = out.groupby("date")["kdj_diff_9_raw"].transform(
        lambda x: x.rank(method="average", pct=True)
    )
    out["kdj_j_9_oversold_raw"] = (out["kdj_j_9_raw"] < 0).astype("float32")
    out["kdj_j_9_low10_raw"] = (out["kdj_j_9_raw"] < 10).astype("float32")
    out["kdj_j_9_overbought_raw"] = (out["kdj_j_9_raw"] > 100).astype("float32")
    out["kdj_j_9_high90_raw"] = (out["kdj_j_9_raw"] > 90).astype("float32")
    return out


def load_base_factors() -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(FACTOR_FILE).schema_arrow.names
    use = ["date", "stock_code"] + [c for c in BASE_FACTOR_COLS if c in names]
    df = pd.read_parquet(FACTOR_FILE, columns=use)
    df["date"] = pd.to_datetime(df["date"])
    return df


def attach_prev_snapshot(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values(["stock_code", "date"])
    grouped = {
        stock: (g["date"].to_numpy(dtype="datetime64[ns]"), g.reset_index(drop=True))
        for stock, g in features.groupby("stock_code", sort=False)
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
                for c in g.columns:
                    if c not in {"date", "stock_code"}:
                        rec[c] = r[c]
        rows.append(rec)
    return pd.DataFrame(rows)


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hot = pd.to_numeric(out[HOT_FACTOR], errors="coerce")
    j_pct = pd.to_numeric(out["kdj_j_9_raw_pct"], errors="coerce")
    diff_pct = pd.to_numeric(out["kdj_diff_9_raw_pct"], errors="coerce")
    j_raw = pd.to_numeric(out["kdj_j_9_raw"], errors="coerce")
    diff_raw = pd.to_numeric(out["kdj_diff_9_raw"], errors="coerce")

    out["kdj_hot_mismatch_j_pct"] = hot * (1.0 - j_pct)
    out["kdj_hot_mismatch_diff_pct"] = hot * (1.0 - diff_pct)
    out["kdj_hot_mismatch_combo_pct"] = hot * (0.65 * (1.0 - j_pct) + 0.35 * (1.0 - diff_pct))
    out["kdj_hot_but_not_top_j"] = ((hot >= 0.8) & (j_pct <= 0.75)).astype("float32")
    out["kdj_hot_but_j_raw_not_strong"] = ((hot >= 0.8) & (j_raw <= 80)).astype("float32")
    out["kdj_hot_but_diff_negative"] = ((hot >= 0.8) & (diff_raw <= 0)).astype("float32")
    out["kdj_hot_but_oversold"] = ((hot >= 0.8) & (j_raw < 0)).astype("float32")
    return out


def current_v3_mask(df: pd.DataFrame) -> pd.Series:
    z = lambda c: pd.to_numeric(df.get(c, pd.Series(np.nan, index=df.index)), errors="coerce")
    r1 = (z("runup_ma5_exhaustion") > 0.8) & (z("cpv_factor") <= -0.4)
    blow = z("prev_day_blowoff_risk")
    sector = z("sector_momentum_accel_rank")
    r2 = ((blow >= 2.0) & (sector <= 0.30)) | ((blow > 5.0) & (sector <= 0.50)) | ((blow > 1.0) & (sector <= 0.15))
    r3 = (z("sector_leadership_persistence_v3") <= -0.18) & (z("return_20d_intra_z") <= 0.40)
    r4 = (z("market_liquidity_change") < 0.94) & (
        (z("ulcer_index") < -0.8)
        | (z("momentum_peak_decay_3d") > 1.0)
        | (z("intra_sector_momentum_rank") < 0.67)
    )
    r5 = (z("mkt_dispersion_shock_v2") < 0.98) & (z("market_advance_decline_ratio") < 0.38)
    r6 = z("limit_down_3d").fillna(0).astype(float) == 1
    r7 = (z("top10_10d_strength_level_v3") >= 0.70) & (z("buy_sm_amount_rate") <= -1.25)
    return (r1 | r2 | r3 | r4 | r5 | r6 | r7).fillna(False)


def stat(df: pd.DataFrame, mask: pd.Series) -> dict:
    mask = pd.Series(mask, index=df.index).fillna(False)
    buckets = {
        "big_loss": df["actual_return"] <= -0.04,
        "small_loss": (df["actual_return"] > -0.04) & (df["actual_return"] < 0),
        "small_gain": (df["actual_return"] >= 0.03) & (df["actual_return"] < 0.08),
        "big_gain8": df["actual_return"] >= 0.08,
        "big_gain10": df["actual_return"] >= 0.10,
    }
    return {
        "hit": int(mask.sum()),
        **{k: int((mask & v).sum()) for k, v in buckets.items()},
        "avg_ret": float(df.loc[mask, "actual_return"].mean()) if mask.any() else np.nan,
        "pnl": float(df.loc[mask, "profit_loss"].sum()) if mask.any() else 0.0,
    }


def scan_rules(df: pd.DataFrame, label: str) -> pd.DataFrame:
    candidates = []
    base = ~df["current_v3"]

    # Explicit human-readable KDJ candidates.
    explicit_rules = {
        "current_z_kdj_j_9 <= 0.75": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9"] <= 0.75),
        "raw_j_pct <= 0.75": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9_raw_pct"] <= 0.75),
        "raw_j <= 80": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9_raw"] <= 80),
        "raw_j <= 50": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9_raw"] <= 50),
        "raw_j < 10": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9_raw"] < 10),
        "raw_j < 0": (df[HOT_FACTOR] >= 0.8) & (df["kdj_j_9_raw"] < 0),
        "raw_diff <= 0": (df[HOT_FACTOR] >= 0.8) & (df["kdj_diff_9_raw"] <= 0),
        "raw_diff_pct <= 0.45": (df[HOT_FACTOR] >= 0.8) & (df["kdj_diff_9_raw_pct"] <= 0.45),
    }
    for name, m in explicit_rules.items():
        s = stat(df, m & base)
        candidates.append({"dataset": label, "rule": name, **s})

    # Score thresholds from event distribution.
    for col, direction in [
        ("kdj_hot_mismatch_j_pct", ">="),
        ("kdj_hot_mismatch_diff_pct", ">="),
        ("kdj_hot_mismatch_combo_pct", ">="),
        ("kdj_j_9_raw_pct", "<="),
        ("kdj_diff_9_raw_pct", "<="),
        ("kdj_j_9_raw", "<="),
        ("kdj_diff_9_raw", "<="),
    ]:
        series = pd.to_numeric(df[col], errors="coerce")
        vals = series[base].dropna()
        if vals.empty:
            continue
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        for q in qs:
            thr = float(vals.quantile(q))
            if direction == ">=":
                m = series >= thr
            else:
                m = series <= thr
            s = stat(df, m & (df[HOT_FACTOR] >= 0.8) & base)
            candidates.append({"dataset": label, "rule": f"{HOT_FACTOR}>=0.8 & {col} {direction} q{q:.2f}({thr:.4g})", **s})

    return pd.DataFrame(candidates)


def main() -> None:
    actual, shadow, all_events = load_events()
    print(f"events: actual={len(actual)} shadow={len(shadow)} all={len(all_events)}")

    print("computing raw KDJ features from price cache...")
    raw_kdj = compute_raw_kdj_features()
    base_factors = load_base_factors()
    features = base_factors.merge(raw_kdj, on=["date", "stock_code"], how="left")

    outputs = []
    enriched = {}
    for label, events in [("actual185", actual), ("shadow185", shadow), ("all185", all_events)]:
        df = attach_prev_snapshot(events, features)
        df = add_candidate_features(df)
        df["current_v3"] = current_v3_mask(df)
        enriched[label] = df
        outputs.append(scan_rules(df, label))

    result = pd.concat(outputs, ignore_index=True)
    for label in ["actual185", "shadow185", "all185"]:
        print(f"\n=== {label} top candidates incremental over current_v3 ===")
        x = result[result["dataset"] == label].copy()
        x = x[(x["hit"] > 0) & (x["big_gain8"] == 0)]
        x = x.sort_values(["big_loss", "hit", "small_loss"], ascending=[False, True, True])
        print(
            x[["rule", "hit", "big_loss", "small_loss", "small_gain", "big_gain8", "big_gain10", "avg_ret", "pnl"]]
            .head(30)
            .to_string(index=False)
        )

    print("\n=== selected comparison ===")
    selected_rules = [
        "current_z_kdj_j_9 <= 0.75",
        "raw_j_pct <= 0.75",
        "raw_j <= 80",
        "raw_j <= 50",
        "raw_j < 10",
        "raw_j < 0",
        "raw_diff <= 0",
        "raw_diff_pct <= 0.45",
    ]
    comp = result[result["rule"].isin(selected_rules)].copy()
    print(
        comp[["dataset", "rule", "hit", "big_loss", "small_loss", "small_gain", "big_gain8", "big_gain10", "avg_ret", "pnl"]]
        .sort_values(["rule", "dataset"])
        .to_string(index=False)
    )

    detail = enriched["all185"]
    best_mask = (detail[HOT_FACTOR] >= 0.8) & (detail["kdj_j_9_raw_pct"] <= 0.75) & (~detail["current_v3"])
    cols = [
        "event_type",
        "stock_code",
        "buy_date",
        "actual_return",
        "profit_loss",
        HOT_FACTOR,
        "kdj_j_9",
        "kdj_j_9_raw",
        "kdj_j_9_raw_pct",
        "kdj_diff_9_raw",
        "kdj_diff_9_raw_pct",
        "factor_date",
    ]
    print("\n=== details for raw_j_pct <= 0.75 candidate, all185 incremental ===")
    print(detail.loc[best_mask, cols].sort_values("actual_return").to_string(index=False))


if __name__ == "__main__":
    main()
