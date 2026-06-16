"""Level-aware factor validation for stock, sector, and market factors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
import time

import numpy as np
import pandas as pd


_EPS = 1e-12


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 5 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def _max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna().clip(lower=-0.99)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    return float((1.0 - equity / equity.cummax()).max())


def _daily_cross_section_metrics(panel: pd.DataFrame, factor: str, label: str, min_obs: int) -> Dict[str, Any]:
    work = panel[["date", factor, label]].dropna().copy()
    if work.empty:
        return {}

    rows = []
    bucket_rows = []
    for date, day in work.groupby("date", sort=True):
        unique_count = day[factor].nunique()
        if len(day) < min_obs or unique_count < 2:
            continue
        ic = _safe_corr(day[factor], day[label], "spearman")
        if unique_count < 5:
            low = day.loc[day[factor] == day[factor].min(), label]
            high = day.loc[day[factor] == day[factor].max(), label]
        else:
            ranks = day[factor].rank(pct=True, method="average")
            low = day.loc[ranks <= 0.2, label]
            high = day.loc[ranks >= 0.8, label]
        spread = high.mean() - low.mean() if len(low) and len(high) else np.nan
        rows.append((date, ic, spread, high.mean(), low.mean()))

        if unique_count >= 5:
            try:
                buckets = pd.qcut(day[factor].rank(method="first"), 5, labels=False)
                means = day.groupby(buckets, observed=True)[label].mean()
                if len(means) == 5:
                    bucket_rows.append(means.reindex(range(5)).to_numpy(dtype="float64"))
            except (ValueError, TypeError):
                pass

    daily = pd.DataFrame(rows, columns=["date", "daily_ic", "spread", "high_return", "low_return"])
    bucket_mean = np.nanmean(np.vstack(bucket_rows), axis=0) if bucket_rows else np.full(5, np.nan)
    ic_std = daily["daily_ic"].std()
    spread_std = daily["spread"].std()
    return {
        "observation_count": int(len(work)),
        "valid_period_count": int(len(daily)),
        "daily_ic_mean": float(daily["daily_ic"].mean()),
        "daily_ic_std": float(ic_std),
        "daily_ic_ir": float(daily["daily_ic"].mean() / (ic_std + _EPS)),
        "daily_ic_positive_ratio": float((daily["daily_ic"] > 0).mean()),
        "daily_spread_mean": float(daily["spread"].mean()),
        "daily_spread_std": float(spread_std),
        "daily_spread_ir": float(daily["spread"].mean() / (spread_std + _EPS)),
        "high_group_mean_return": float(daily["high_return"].mean()),
        "low_group_mean_return": float(daily["low_return"].mean()),
        "high_group_win_rate": float((daily["high_return"] > 0).mean()),
        "high_minus_low_max_drawdown": _max_drawdown(daily["spread"]),
        "bucket_1_mean_return": float(bucket_mean[0]),
        "bucket_2_mean_return": float(bucket_mean[1]),
        "bucket_3_mean_return": float(bucket_mean[2]),
        "bucket_4_mean_return": float(bucket_mean[3]),
        "bucket_5_mean_return": float(bucket_mean[4]),
        "bucket_monotonicity": _safe_corr(pd.Series(range(5)), pd.Series(bucket_mean), "spearman"),
    }


def _market_timing_metrics(panel: pd.DataFrame, factor: str, label: str) -> Dict[str, Any]:
    daily = panel.groupby("date", sort=True).agg(
        factor_value=(factor, "median"),
        market_equal_weight_return=(label, "mean"),
        market_median_return=(label, "median"),
        positive_ratio=(label, lambda s: float((s > 0).mean())),
        stock_count=(label, "count"),
    ).dropna(subset=["factor_value", "market_equal_weight_return"])
    if len(daily) < 20 or daily["factor_value"].nunique() < 5:
        return {"observation_count": int(len(daily)), "valid_period_count": int(len(daily))}

    pct = daily["factor_value"].rank(pct=True, method="average")
    low = daily.loc[pct <= 0.2]
    high = daily.loc[pct >= 0.8]
    spread = high["market_equal_weight_return"].mean() - low["market_equal_weight_return"].mean()
    return {
        "observation_count": int(len(daily)),
        "valid_period_count": int(len(daily)),
        "time_series_pearson": _safe_corr(daily["factor_value"], daily["market_equal_weight_return"], "pearson"),
        "time_series_spearman": _safe_corr(daily["factor_value"], daily["market_equal_weight_return"], "spearman"),
        "breadth_spearman": _safe_corr(daily["factor_value"], daily["positive_ratio"], "spearman"),
        "high_state_mean_return": float(high["market_equal_weight_return"].mean()),
        "low_state_mean_return": float(low["market_equal_weight_return"].mean()),
        "high_minus_low_return": float(spread),
        "high_state_win_rate": float((high["market_equal_weight_return"] > 0).mean()),
        "low_state_win_rate": float((low["market_equal_weight_return"] > 0).mean()),
        "high_state_big_loss_rate": float((high["market_equal_weight_return"] <= -0.02).mean()),
        "low_state_big_loss_rate": float((low["market_equal_weight_return"] <= -0.02).mean()),
        "high_state_max_drawdown": _max_drawdown(high["market_equal_weight_return"]),
        "low_state_max_drawdown": _max_drawdown(low["market_equal_weight_return"]),
    }


@dataclass
class FactorLevel:
    level: str
    reason: str
    within_date_unique_median: float
    within_date_unique_p95: float
    within_sector_residual_ratio: float | None


def classify_factor_levels(
    data: pd.DataFrame,
    factor_names: Iterable[str],
    definitions: Mapping[str, Any],
    stock_to_sector: Mapping[str, str],
    progress_callback: Optional[Callable[[str, int, int, str, float], None]] = None,
) -> Dict[str, FactorLevel]:
    factor_names = list(factor_names)
    dates = data["date"]
    sector = data["stock_code"].map(stock_to_sector) if stock_to_sector else pd.Series(np.nan, index=data.index)
    sector_count = int(sector.dropna().nunique())
    result: Dict[str, FactorLevel] = {}
    phase_start = time.perf_counter()
    for index, factor in enumerate(factor_names, 1):
        def report_progress() -> None:
            nonlocal phase_start
            if progress_callback is not None:
                progress_callback("classify", index, len(factor_names), factor, time.perf_counter() - phase_start)
                phase_start = time.perf_counter()

        definition = definitions.get(factor)
        if definition is None:
            for suffix in ("_intra_rank", "_intra_z", "_pct", "_exp"):
                if factor.endswith(suffix):
                    definition = definitions.get(factor[: -len(suffix)])
                    if definition is not None:
                        break
        sample = pd.DataFrame({"date": dates, "sector": sector, "x": data[factor]}).dropna(subset=["date", "x"])
        if sample.empty:
            result[factor] = FactorLevel("invalid", "empty_factor", np.nan, np.nan, None)
            report_progress()
            continue

        global_unique = int(sample["x"].nunique())
        date_unique_stats = sample.groupby("date", sort=False)["x"].nunique()
        date_unique = float(date_unique_stats.median())
        date_unique_p95 = float(date_unique_stats.quantile(0.95))
        if global_unique <= 1:
            result[factor] = FactorLevel("invalid", "constant_factor", date_unique, date_unique_p95, None)
            report_progress()
            continue

        scope = getattr(definition, "calculation_scope", "") if definition is not None else ""
        if scope == "daily_broadcast" or date_unique_p95 <= 1.0:
            reason = "daily_broadcast_metadata" if scope == "daily_broadcast" else "one_value_per_date_p95"
            result[factor] = FactorLevel("market", reason, date_unique, date_unique_p95, None)
            report_progress()
            continue

        requires_sector = bool(getattr(definition, "requires_sector_data", False)) if definition is not None else False
        description = str(getattr(definition, "description", "")) if definition is not None else ""
        sector_name_hint = (
            "_sector_mean" in factor
            or factor.startswith("sector_")
            or factor.startswith("sw2_")
            or "行业均值" in description
            or "所属行业" in description
        )
        residual_ratio = None
        if (requires_sector or sector_name_hint) and stock_to_sector:
            known = sample[sample["sector"].notna()]
            if not known.empty:
                group_unique = known.groupby(["date", "sector"], sort=False)["x"].nunique()
                residual_ratio = float((group_unique > 1).mean())
                if residual_ratio <= 0.10:
                    if requires_sector:
                        reason = "sector_metadata_and_sw2_broadcast_structure"
                    elif sector_name_hint:
                        reason = "sector_name_description_and_sw2_broadcast_structure"
                    else:
                        reason = "sw2_broadcast_structure"
                    result[factor] = FactorLevel("sector", reason, date_unique, date_unique_p95, residual_ratio)
                    report_progress()
                    continue
        reason = "sector_aware_stock_signal" if requires_sector else "stock_cross_section"
        result[factor] = FactorLevel("stock", reason, date_unique, date_unique_p95, residual_ratio)
        report_progress()
    return result


def validate_factors_by_level(
    data: pd.DataFrame,
    factor_names: Iterable[str],
    label: str,
    definitions: Mapping[str, Any],
    stock_to_sector: Mapping[str, str],
    progress_callback: Optional[Callable[[str, int, int, str, float], None]] = None,
) -> Dict[str, Any]:
    factor_names = list(factor_names)
    levels = classify_factor_levels(
        data, factor_names, definitions, stock_to_sector, progress_callback=progress_callback
    )
    results: Dict[str, Dict[str, Any]] = {"stock": {}, "sector": {}, "market": {}, "invalid": {}}

    sector_series = data["stock_code"].map(stock_to_sector) if stock_to_sector else pd.Series(np.nan, index=data.index)
    phase_start = time.perf_counter()
    for index, factor in enumerate(factor_names, 1):
        info = levels[factor]
        common = {
            "factor_level": info.level,
            "classification_reason": info.reason,
            "within_date_unique_median": info.within_date_unique_median,
            "within_date_unique_p95": info.within_date_unique_p95,
            "within_sector_residual_ratio": info.within_sector_residual_ratio,
        }
        if info.level == "invalid":
            metrics = {"observation_count": int(pd.to_numeric(data[factor], errors="coerce").notna().sum())}
        elif info.level == "market":
            metrics = _market_timing_metrics(data, factor, label)
        elif info.level == "sector":
            sector_panel = pd.DataFrame(
                {"date": data["date"], "sector": sector_series, factor: data[factor], label: data[label]}
            ).dropna(subset=["sector"])
            collapsed = sector_panel.groupby(["date", "sector"], sort=True, as_index=False).agg(
                **{factor: (factor, "median"), label: (label, "mean")}
            )
            metrics = _daily_cross_section_metrics(collapsed, factor, label, min_obs=5)
            metrics["sector_observation_count"] = int(len(collapsed))
        else:
            metrics = _daily_cross_section_metrics(data, factor, label, min_obs=30)
        results[info.level][factor] = {**common, **metrics}
        if progress_callback is not None:
            progress_callback("validate", index, len(factor_names), factor, time.perf_counter() - phase_start)
            phase_start = time.perf_counter()

    return {
        "methodology": {
            "stock": "Daily stock cross-sectional Spearman IC and equal-weight quintile spread.",
            "sector": "Collapse to one equal-weight observation per date-sector, then test cross-sector ranking.",
            "market": "Collapse to one observation per date, then test time-series timing and high/low market states.",
            "invalid": "Constant or empty factors are reported but not evaluated.",
            "sector_standard": "SW2 industry mapping from config/sw2_industry_encoding.pkl.",
            "within_sector_residual_ratio": (
                "Share of date-SW2 groups containing more than one factor value. "
                "0 means a sector-broadcast factor; near 1 means a stock-level signal. "
                "null means not applicable because the factor is clearly stock/market level."
            ),
        },
        "sector_mapping": {
            "standard": "SW2",
            "sector_count": int(sector_series.dropna().nunique()),
            "mapped_stock_ratio": float(sector_series.notna().mean()),
        },
        "factor_counts": {level: len(values) for level, values in results.items()},
        "levels": results,
    }
