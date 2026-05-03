from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"

BASELINE_JSON = RESULTS_DIR / "walkforward_metrics_20260430_011835.json"
REMAINING_BIGLOSS_CSV = RESULTS_DIR / "remaining_bigloss_samples_20260430_173039.csv"
DAILY_PARQUET = CACHE_DIR / "stock_features_20221201_20260428.parquet"


def load_baseline_trades() -> pd.DataFrame:
    data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    trades = pd.DataFrame(data[0]["all_results"]["1"]["trades"])
    trades["buy_date"] = pd.to_datetime(trades["buy_datetime"]).dt.normalize()
    trades["actual_return"] = pd.to_numeric(trades["actual_return"], errors="coerce")
    return trades


def load_market_dates() -> pd.DatetimeIndex:
    dates = pd.read_parquet(DAILY_PARQUET, columns=["date"])["date"]
    dates = pd.to_datetime(dates).drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def previous_trading_day(buy_dates: pd.Series, market_dates: pd.DatetimeIndex) -> pd.Series:
    values = pd.to_datetime(buy_dates).values.astype("datetime64[ns]")
    md = market_dates.values.astype("datetime64[ns]")
    pos = np.searchsorted(md, values, side="left") - 1
    out = pd.Series(pd.NaT, index=buy_dates.index, dtype="datetime64[ns]")
    valid = pos >= 0
    if valid.any():
        out.loc[valid] = pd.to_datetime(md[pos[valid]])
    return out


def efficiency_ratio(close: pd.Series, n: int = 10) -> pd.Series:
    direction = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    return direction / (volatility + 1e-12)


def compute_kama(close: pd.Series, er_n: int = 10, fast_n: int = 2, slow_n: int = 30) -> pd.Series:
    er = efficiency_ratio(close, er_n).fillna(0.0)
    fast = 2.0 / (fast_n + 1.0)
    slow = 2.0 / (slow_n + 1.0)
    sc = (er * (fast - slow) + slow) ** 2
    kama = np.full(len(close), np.nan, dtype=float)
    values = close.to_numpy(dtype=float)
    if len(values) == 0:
        return pd.Series(kama, index=close.index)
    kama[0] = values[0]
    for i in range(1, len(values)):
        prev = kama[i - 1]
        if np.isnan(prev):
            prev = values[i - 1]
        kama[i] = prev + sc.iloc[i] * (values[i] - prev)
    return pd.Series(kama, index=close.index)


def hma(close: pd.Series, n: int = 16) -> pd.Series:
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    w = np.arange(1, n + 1, dtype=float)
    w_half = np.arange(1, half + 1, dtype=float)
    w_sqrt = np.arange(1, sqrt_n + 1, dtype=float)

    def wma(arr: np.ndarray, weights: np.ndarray) -> float:
        return float(np.dot(arr, weights) / weights.sum())

    wma_half = close.rolling(half).apply(lambda arr: wma(np.asarray(arr), w_half), raw=False)
    wma_full = close.rolling(n).apply(lambda arr: wma(np.asarray(arr), w), raw=False)
    raw = 2 * wma_half - wma_full
    hma_series = raw.rolling(sqrt_n).apply(lambda arr: wma(np.asarray(arr), w_sqrt), raw=False)
    return hma_series


def load_daily_features(stock_codes: list[str]) -> pd.DataFrame:
    cols = ["date", "stock_code", "open", "high", "low", "close"]
    df = pd.read_parquet(DAILY_PARQUET, columns=cols)
    df = df[df["stock_code"].isin(stock_codes)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    out_frames: list[pd.DataFrame] = []
    for stock_code, g in df.groupby("stock_code", sort=False):
        g = g.copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean()

        for w in [5, 10, 20, 60]:
            g[f"ma{w}"] = close.rolling(w, min_periods=max(3, w // 2)).mean()

        g["er10"] = efficiency_ratio(close, 10)
        g["kama10"] = compute_kama(close, 10, 2, 30)
        g["close_to_kama_atr"] = (close - g["kama10"]) / (atr14 + 1e-9)
        g["kama_slope3"] = (g["kama10"] - g["kama10"].shift(3)) / (g["kama10"].shift(3) + 1e-9)

        width = (g[["ma5", "ma10", "ma20"]].max(axis=1) - g[["ma5", "ma10", "ma20"]].min(axis=1)) / (close + 1e-9)
        g["ma_ribbon_width_short"] = width
        g["ma_ribbon_expansion_3d"] = width - width.shift(3)

        pair1 = np.sign(g["ma5"] - g["ma10"])
        pair2 = np.sign(g["ma10"] - g["ma20"])
        twist = (pair1 != pair1.shift(1)).astype(int).fillna(0) + (pair2 != pair2.shift(1)).astype(int).fillna(0)
        g["ma_ribbon_twist_count_10d"] = twist.rolling(10, min_periods=3).sum()

        aligned = ((g["ma5"] > g["ma10"]) & (g["ma10"] > g["ma20"]) & (g["ma20"] > g["ma60"])).astype(int)
        g["ma_alignment_persistence_5d"] = aligned.rolling(5, min_periods=2).mean()

        g["hma16"] = hma(close, 16)
        g["close_to_hma"] = (close - g["hma16"]) / (g["hma16"] + 1e-9)
        g["hma_slope3"] = (g["hma16"] - g["hma16"].shift(3)) / (g["hma16"].shift(3) + 1e-9)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def scan_metric(df: pd.DataFrame, metric: str, remaining_pairs: set[tuple[str, pd.Timestamp]]) -> list[dict]:
    rows = []
    x = pd.to_numeric(df[metric], errors="coerce")
    valid = x.notna()
    base = df.loc[valid].copy()
    x = x.loc[valid]
    if len(base) < 20:
        return rows

    quantiles = [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]
    thresholds = sorted({float(x.quantile(q)) for q in quantiles if x.notna().sum() > 0})
    base_cum = float(np.prod(1 + base["actual_return"].fillna(0).to_numpy()) - 1)
    base["is_bigloss5"] = (base["actual_return"] <= -0.05).astype(int)
    base["is_bigloss7"] = (base["actual_return"] <= -0.07).astype(int)
    base["is_remaining"] = [
        (str(code), pd.Timestamp(pred_date)) in remaining_pairs
        for code, pred_date in zip(base["stock_code"], pd.to_datetime(base["prediction_date"]))
    ]

    for direction in ["<", ">"]:
        for thr in thresholds:
            mask = (x < thr) if direction == "<" else (x > thr)
            flagged = base.loc[mask].copy()
            kept = base.loc[~mask].copy()
            if flagged.empty or kept.empty:
                continue
            rows.append({
                "metric": metric,
                "direction": direction,
                "threshold": float(thr),
                "rule": f"{metric} {direction} {thr:.6f}",
                "flagged": int(mask.sum()),
                "flagged_ratio": float(mask.mean()),
                "flagged_mean_return": float(flagged["actual_return"].mean()),
                "bigloss5_hit": int(flagged["is_bigloss5"].sum()),
                "bigloss7_hit": int(flagged["is_bigloss7"].sum()),
                "remaining_bigloss_hit": int(flagged["is_remaining"].sum()),
                "remaining_bigloss5": int(kept["is_bigloss5"].sum()),
                "remaining_bigloss7": int(kept["is_bigloss7"].sum()),
                "cum_return": float(np.prod(1 + kept["actual_return"].fillna(0).to_numpy()) - 1),
                "cum_delta_vs_base": float((np.prod(1 + kept["actual_return"].fillna(0).to_numpy()) - 1) - base_cum),
            })
    return rows


def main() -> None:
    trades = load_baseline_trades()
    market_dates = load_market_dates()
    trades["prediction_date"] = previous_trading_day(trades["buy_date"], market_dates)
    trades = trades[["stock_code", "prediction_date", "actual_return"]].dropna(subset=["prediction_date"])

    remaining = pd.read_csv(REMAINING_BIGLOSS_CSV)
    remaining["prediction_date"] = pd.to_datetime(remaining["prediction_date"])
    remaining_pairs = {
        (str(r.stock_code), pd.Timestamp(r.prediction_date))
        for r in remaining.itertuples(index=False)
    }

    stock_codes = sorted(trades["stock_code"].astype(str).unique().tolist())
    daily = load_daily_features(stock_codes)
    merged = trades.merge(
        daily,
        left_on=["stock_code", "prediction_date"],
        right_on=["stock_code", "date"],
        how="left",
    )

    metrics = [
        "er10",
        "close_to_kama_atr",
        "kama_slope3",
        "ma_ribbon_width_short",
        "ma_ribbon_expansion_3d",
        "ma_ribbon_twist_count_10d",
        "ma_alignment_persistence_5d",
        "close_to_hma",
        "hma_slope3",
    ]

    rows: list[dict] = []
    for metric in metrics:
        rows.extend(scan_metric(merged, metric, remaining_pairs))

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["remaining_bigloss_hit", "bigloss7_hit", "bigloss5_hit", "flagged_ratio", "cum_return"],
        ascending=[False, False, False, True, False],
    )
    ts = "20260501_001500"
    out_path = RESULTS_DIR / f"professional_ma_prototype_scan_{ts}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_path)


if __name__ == "__main__":
    main()
