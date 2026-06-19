"""
Advanced prior-high retest and causal local-KNN prototype factors (v2 fixed).

This file is written for the existing FactorDefinition/register_factor framework.
Historical outcomes are used only after they are fully realized; current and
past factor values never use outcomes unavailable on their scoring date.

Why this v2 exists
------------------
The previous advanced_retest_prototype_factors.py had two practical problems:

1) prior_high_retest_quality_60d rewarded "already touching / reclaiming prior high"
   too aggressively.  This v2 uses the validated pre-second-breakout shapelet as
   its broad-universe base signal, then applies a prior-high context correction:
   confirmed constructive tests add value, while touch-and-reject states subtract
   value.

2) Generic 20-day path KNN matched visually similar paths at different lifecycle
   stages.  That is weak for the short-horizon weighted return target.  This v2
   instead compares the current entry state with causal historical prototypes:
       - state features describe impulse, acceleration, MA position, pullback,
         volatility contraction, close acceptance, and rejection pressure;
       - positive/negative prototypes are formed from the production label
         proxy: T+1 open entry and equal-weight T+2/T+3 typical-price exit;
       - an event enters the prototype bank only after its T+3 outcome is
         fully known, so scoring remains causal;
       - the raw factor is similarity to successful continuation prototypes
         minus similarity to failed/exhausted prototypes.

Factors registered
------------------
1. prior_high_retest_setup_quality_60d_v2
2. pre_breakout_local_knn_similarity_20d_v2
3. pre_breakout_local_knn_quality_20d_v2

Keep these as candidates next to:
    - position_20_60_spread_v2
    - pre_second_breakout_shapelet_corr_20d_v2
Do not replace proven factors until validation and walk-forward model tests agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series
from ..universe import non_st_mask
from .structure_signal_factors_v2 import _pre_second_breakout_shapelet_corr_20d_v2

_EPS = 1e-12
_Z_CLIP = 6.0

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _nan(index: pd.Index | range) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_num(x: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.Series(x, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan).astype("float64")


def _ordered_index(data: pd.DataFrame) -> pd.Index:
    if {"stock_code", "date"}.issubset(data.columns):
        return data.sort_values(["stock_code", "date"], kind="mergesort").index
    return data.index


def _rolling_col(data: pd.DataFrame, col: str, window: int, min_periods: int, agg: str) -> pd.Series:
    if col not in data.columns or "stock_code" not in data.columns:
        return _nan(data.index)
    ordered_idx = _ordered_index(data)
    value = _safe_num(data.loc[ordered_idx, col])
    grouped = value.groupby(data.loc[ordered_idx, "stock_code"], sort=False)
    rolled = grouped.rolling(window, min_periods=min_periods)
    if agg == "max":
        out = rolled.max()
    elif agg == "min":
        out = rolled.min()
    elif agg == "mean":
        out = rolled.mean()
    elif agg == "std":
        out = rolled.std()
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    return out.reset_index(level=0, drop=True).reindex(data.index).astype("float64")


def _shift_by_stock(data: pd.DataFrame, s: pd.Series, periods: int) -> pd.Series:
    if "stock_code" not in data.columns:
        return _nan(data.index)
    ordered_idx = _ordered_index(data)
    ordered_s = s.reindex(ordered_idx)
    out = ordered_s.groupby(data.loc[ordered_idx, "stock_code"], sort=False).shift(periods)
    return out.reindex(data.index).astype("float64")


def _rolling_shifted_col(data: pd.DataFrame, col: str, window: int, min_periods: int, shift_days: int, agg: str) -> pd.Series:
    if col not in data.columns or "stock_code" not in data.columns:
        return _nan(data.index)
    ordered_idx = _ordered_index(data)
    ordered = data.loc[ordered_idx, ["stock_code", col]].copy()
    ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    shifted = ordered.groupby("stock_code", sort=False)[col].shift(shift_days)
    rolled = shifted.groupby(ordered["stock_code"], sort=False).rolling(window, min_periods=min_periods)
    if agg == "max":
        out = rolled.max()
    elif agg == "min":
        out = rolled.min()
    elif agg == "mean":
        out = rolled.mean()
    else:
        raise ValueError(f"Unsupported shifted agg: {agg}")
    return out.reset_index(level=0, drop=True).reindex(data.index).astype("float64")


def _log_return(data: pd.DataFrame, periods: int) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return _nan(data.index)
    close = _safe_num(data["close"])
    prev = _shift_by_stock(data, close, periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(close / prev)
    return out.replace([np.inf, -np.inf], np.nan).astype("float64")


def _daily_pct_rank(
    s: pd.Series,
    dates: pd.Series,
    min_obs: int = 20,
    eligible: pd.Series | None = None,
) -> pd.Series:
    if eligible is None:
        eligible = pd.Series(True, index=s.index)
    eligible = eligible.reindex(s.index).fillna(True)
    work = pd.DataFrame({"v": _safe_num(s).where(eligible).values, "date": dates.values}, index=s.index)
    g = work.groupby("date", sort=False)["v"]
    out = g.rank(pct=True, method="average")
    counts = g.transform("count")
    out = out.where(counts >= min_obs, np.nan)
    return out.where(eligible).astype("float64")


def _daily_zscore(
    s: pd.Series,
    dates: pd.Series,
    min_obs: int = 20,
    eligible: pd.Series | None = None,
) -> pd.Series:
    if eligible is None:
        eligible = pd.Series(True, index=s.index)
    eligible = eligible.reindex(s.index).fillna(True)
    work = pd.DataFrame({"v": _safe_num(s).where(eligible).values, "date": dates.values}, index=s.index)
    g = work.groupby("date", sort=False)["v"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    counts = g.transform("count")
    out = (work["v"] - mean) / (std + _EPS)
    out = out.where(counts >= min_obs, np.nan)
    return out.where(eligible).clip(-_Z_CLIP, _Z_CLIP).astype("float64")


def _daily_zscore_with_fallback(
    s: pd.Series,
    dates: pd.Series,
    min_obs: int = 20,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """Daily zscore, but do not collapse to all NaN when a framework passes thin windows."""
    z = _daily_zscore(s, dates, min_obs=min_obs, eligible=eligible)
    if z.notna().sum() >= max(100, int(0.05 * len(z))):
        return z
    if eligible is None:
        eligible = pd.Series(True, index=s.index)
    eligible = eligible.reindex(s.index).fillna(True)
    v = _safe_num(s).where(eligible)
    med = v.median(skipna=True)
    std = v.std(skipna=True)
    if not np.isfinite(std) or std <= _EPS:
        return v * np.nan
    return ((v - med) / (std + _EPS)).where(eligible).clip(-_Z_CLIP, _Z_CLIP).astype("float64")


def _close_position_extreme(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)
    high_n = _rolling_col(data, "high", window, min_periods, "max")
    low_n = _rolling_col(data, "low", window, min_periods, "min")
    close = _safe_num(data["close"])
    rng = high_n - low_n
    out = (close - low_n) / (rng + _EPS)
    return out.where(rng > _EPS, np.nan).clip(0.0, 1.0).astype("float64")


def _intraday_close_strength(data: pd.DataFrame) -> pd.Series:
    if not {"high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)
    high = _safe_num(data["high"])
    low = _safe_num(data["low"])
    close = _safe_num(data["close"])
    rng = high - low
    out = (close - low) / (rng + _EPS)
    return out.where(rng > _EPS, 0.5).clip(0.0, 1.0).astype("float64")


def _upper_shadow_ratio(data: pd.DataFrame) -> pd.Series:
    if not {"open", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)
    open_ = _safe_num(data["open"])
    high = _safe_num(data["high"])
    low = _safe_num(data["low"])
    close = _safe_num(data["close"])
    rng = high - low
    body_top = np.maximum(open_, close)
    out = (high - body_top) / (rng + _EPS)
    return out.where(rng > _EPS, 0.0).clip(0.0, 1.0).astype("float64")


def _days_since_shifted_rolling_high(data: pd.DataFrame, window: int = 60, shift_days: int = 5) -> pd.Series:
    if not {"stock_code", "date", "high"}.issubset(data.columns):
        return _nan(data.index)
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    high_values = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype="float64")
    date_values = pd.to_datetime(data["date"], errors="coerce").to_numpy()
    # groupby.indices returns row POSITIONS, which is exactly what we need.
    for _, positions in data.groupby("stock_code", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < shift_days + 5:
            continue
        order = np.argsort(date_values[pos], kind="mergesort")
        sp = pos[order]
        h = high_values[sp]
        out = np.full(sp.size, np.nan, dtype="float64")
        for i in range(sp.size):
            end = i - shift_days
            if end < 0:
                continue
            start = max(0, end - window + 1)
            w = h[start : end + 1]
            finite = np.isfinite(w)
            if finite.sum() < max(10, window // 3):
                continue
            vals = np.where(finite, w, -np.inf)
            mx = np.max(vals)
            rel = np.flatnonzero(vals == mx)
            if rel.size:
                peak_i = start + int(rel[-1])
                out[i] = float(i - peak_i)
        result.iloc[sp] = out
    return result.astype("float64")


# -----------------------------------------------------------------------------
# 1. Prior-high retest setup quality v2
# -----------------------------------------------------------------------------


def _prior_high_retest_setup_quality_60d_v2_full(data: pd.DataFrame) -> pd.Series:
    """
    Constructive prior-high retest setup, not chase-after-breakout.

    The previous version rewarded being exactly at / above prior high.  The test
    showed that this is often a negative short-horizon signal.  This v2 instead
    rewards a controlled setup just under or very near the prior high:
      - prior high is measured causally: 60d high shifted by 5 days;
      - price is close to prior high, but large overshoot is penalized;
      - intraday high may test the level, but close should not show heavy upper shadow;
      - volume/amount confirmation is useful, but blow-off volume is not over-rewarded;
      - previous peak should be separated enough to be meaningful;
      - maturity is controlled by current ret20, ret5 and pos20 ranks.
    """
    required = {"stock_code", "date", "open", "high", "low", "close"}
    if not required.issubset(data.columns):
        return _nan(data.index)

    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    eligible = non_st_mask(data)
    close = _safe_num(data["close"])
    high = _safe_num(data["high"])

    prior_high = _rolling_shifted_col(data, "high", window=60, min_periods=30, shift_days=5, agg="max")
    ret1 = _log_return(data, 1)
    vol20 = _rolling_col(pd.DataFrame({"stock_code": data["stock_code"], "date": data["date"], "ret1": ret1}, index=data.index), "ret1", 20, 10, "std")
    vol20 = vol20.clip(lower=0.012, upper=0.18)

    with np.errstate(divide="ignore", invalid="ignore"):
        close_to_ph = np.log(close / prior_high)
        high_to_ph = np.log(high / prior_high)

    close_sigma = close_to_ph / (vol20 + _EPS)
    high_sigma = high_to_ph / (vol20 + _EPS)

    # Constructive zone: close slightly below / near prior high.  Large overshoot
    # is a mature breakout and receives lower setup score.
    near_below = np.exp(-0.5 * np.square(((close_sigma + 0.25) / 0.85).clip(-8.0, 8.0)))
    overshoot_penalty = 1.0 / (1.0 + np.exp((close_sigma - 0.75).clip(-8.0, 8.0)))
    test_component = 1.0 / (1.0 + np.exp((-high_sigma - 0.15).clip(-8.0, 8.0)))

    close_strength = _intraday_close_strength(data)
    upper = _upper_shadow_ratio(data)
    upper3 = _rolling_col(pd.DataFrame({"stock_code": data["stock_code"], "date": data["date"], "u": upper}, index=data.index), "u", 3, 1, "mean")

    # Flow confirmation: middle-high relative flow is better than blow-off top flow.
    flow_col = "amount" if "amount" in data.columns else "volume" if "volume" in data.columns else None
    if flow_col:
        flow = _safe_num(data[flow_col])
        ma20 = _rolling_col(data, flow_col, 20, 10, "mean")
        flow_rel = np.log1p((flow / (ma20 + _EPS)).clip(lower=0.0))
        flow_rank = _daily_pct_rank(flow_rel, dates, min_obs=20, eligible=eligible)
        # Hump: confirms liquidity, but top 5-10% blow-off does not get max score.
        flow_quality = np.sin(np.pi * flow_rank.clip(0.0, 1.0)).clip(0.0, 1.0)
        flow_quality = flow_quality.where(flow_rank.notna(), 0.55)
    else:
        flow_quality = pd.Series(0.55, index=data.index, dtype="float64")

    days_since = _days_since_shifted_rolling_high(data, window=60, shift_days=5)
    interval_rank = _daily_pct_rank(days_since, dates, min_obs=20, eligible=eligible)
    # Prefer middle interval; too recent = noisy chase, too stale = stale resistance.
    interval_quality = np.sin(np.pi * interval_rank.clip(0.0, 1.0)).clip(0.0, 1.0)
    interval_quality = interval_quality.where(interval_rank.notna(), 0.50)

    ret20 = _log_return(data, 20)
    ret5 = _log_return(data, 5)
    pos20 = _close_position_extreme(data, 20, 10)
    maturity = (
        0.42 * _daily_pct_rank(ret20, dates, min_obs=20, eligible=eligible)
        + 0.28 * _daily_pct_rank(ret5, dates, min_obs=20, eligible=eligible)
        + 0.30 * _daily_pct_rank(pos20, dates, min_obs=20, eligible=eligible)
    )
    maturity_control = (1.0 - 0.70 * maturity).clip(0.10, 1.0).where(maturity.notna(), 0.55)

    upper3_clean = upper3.fillna(0.0).clip(0.0, 1.0)
    close_quality = close_strength.clip(0.0, 1.0)

    # A prior high is resistance until the tape confirms otherwise.  The old
    # multiplicative score rewarded proximity itself, so weak closes just below
    # resistance could still receive a high score.  Use a signed score instead:
    # reward controlled tests/reclaims with strong closes, and explicitly
    # penalize touch-and-reject / mature overshoot states.
    constructive_test = (
        near_below
        * test_component
        * (0.20 + 0.80 * close_quality)
        * (1.0 - 0.85 * upper3_clean)
        * (0.35 + 0.65 * flow_quality)
        * (0.40 + 0.60 * interval_quality)
        * maturity_control
    )
    confirmed_reclaim = (
        test_component
        * (1.0 - overshoot_penalty)
        * close_quality
        * (1.0 - upper3_clean)
        * (0.40 + 0.60 * flow_quality)
        * maturity_control
    )
    rejection_risk = (
        test_component
        * (0.25 + 0.75 * upper3_clean)
        * (0.25 + 0.75 * (1.0 - close_quality))
    )
    mature_overshoot_risk = (1.0 - overshoot_penalty) * (1.0 - maturity_control)

    prior_high_overlay = (
        constructive_test
        + 0.65 * confirmed_reclaim
        - 0.90 * rejection_risk
        - 0.45 * mature_overshoot_risk
    )

    # The validated shapelet is a stronger broad-universe base signal than
    # prior-high proximity alone.  Treat the prior-high structure as a contextual
    # correction: reward a constructive shapelet only when the prior-high test is
    # confirmed, and subtract explicit touch-and-reject risk.
    shapelet = _pre_second_breakout_shapelet_corr_20d_v2(data, None)
    prior_high_context = (near_below * test_component).clip(0.0, 1.0)
    rejection_overlay = test_component * (
        0.55 * (1.0 - close_quality)
        + 0.75 * upper3_clean
    )
    raw = (
        shapelet
        + 0.50 * prior_high_context * prior_high_overlay
        - 0.30 * rejection_overlay
    )

    # The factor pipeline already performs daily robust normalization. Returning
    # the signed raw structure avoids a second normalization pass here.
    return raw.replace([np.inf, -np.inf], np.nan).astype("float64")


def _prior_high_retest_setup_quality_60d_v2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _prior_high_retest_setup_quality_60d_v2_full(data)


def _prior_high_retest_setup_quality_60d_v2_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_prior_high_retest_setup_quality_60d_v2_full(data), target_index=target_index)


prior_high_retest_setup_quality_60d_v2 = register_factor(
    FactorDefinition(
        name="prior_high_retest_setup_quality_60d_v2",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=100,
        calculator=_prior_high_retest_setup_quality_60d_v2,
        calculation_scope="target_with_history_window",
        target_calculator=_prior_high_retest_setup_quality_60d_v2_target,
        description="Shapelet基础路径质量的前高场景校正：建设性测试/确认突破加分，触及前高后的弱收盘与上影拒绝风险扣分",
    )
)


# -----------------------------------------------------------------------------
# 2. Causal short-horizon continuation prototype matching v2
# -----------------------------------------------------------------------------

_LOCAL_LOOKBACK_DAYS = 180
_FALLBACK_LOOKBACK_DAYS = 360
_TOPK = 12
_SCORE_CHUNK_ROWS = 1024
_PROTO_QUANTILE = 0.20
_MIN_PROTO_OBS = 30


def _rolling_series_by_stock(data: pd.DataFrame, s: pd.Series, window: int, min_periods: int, agg: str) -> pd.Series:
    ordered_idx = _ordered_index(data)
    ordered_s = _safe_num(s).reindex(ordered_idx)
    grouped = ordered_s.groupby(data.loc[ordered_idx, "stock_code"], sort=False)
    rolled = grouped.rolling(window, min_periods=min_periods)
    if agg == "mean":
        out = rolled.mean()
    elif agg == "max":
        out = rolled.max()
    elif agg == "min":
        out = rolled.min()
    elif agg == "std":
        out = rolled.std()
    else:
        raise ValueError(f"Unsupported series agg: {agg}")
    return out.reset_index(level=0, drop=True).reindex(data.index).astype("float64")


def _forward_weighted_return_and_realized_date(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Match the production short-horizon label: buy T+1 open, exit across T+2/T+3."""
    open_ = _safe_num(data["open"])
    high = _safe_num(data["high"])
    low = _safe_num(data["low"])
    close = _safe_num(data["close"])
    typical = (high + low + close) / 3.0
    buy = _shift_by_stock(data, open_, -1)
    sell2 = _shift_by_stock(data, typical, -2)
    sell3 = _shift_by_stock(data, typical, -3)
    sell = 0.5 * sell2 + 0.5 * sell3
    with np.errstate(divide="ignore", invalid="ignore"):
        outcome = sell / buy - 1.0

    ordered_idx = _ordered_index(data)
    ordered_dates = pd.to_datetime(data.loc[ordered_idx, "date"], errors="coerce")
    realized = ordered_dates.groupby(data.loc[ordered_idx, "stock_code"], sort=False).shift(-3)
    return outcome.replace([np.inf, -np.inf], np.nan).astype("float64"), realized.reindex(data.index)


def _continuation_state_signatures(data: pd.DataFrame, dates: pd.Series) -> tuple[np.ndarray, dict[str, pd.Series]]:
    eligible = non_st_mask(data)
    close = _safe_num(data["close"])
    high = _safe_num(data["high"])
    low = _safe_num(data["low"])
    prev_close = _shift_by_stock(data, close, 1)

    ret3 = _log_return(data, 3)
    ret5 = _log_return(data, 5)
    ret10 = _log_return(data, 10)
    accel = ret3 - 0.30 * ret10
    ma5 = _rolling_col(data, "close", 5, 5, "mean")
    ma10 = _rolling_col(data, "close", 10, 8, "mean")
    ma5_gap = close / (ma5 + _EPS) - 1.0
    ma10_gap = close / (ma10 + _EPS) - 1.0
    close_strength = _intraday_close_strength(data)
    upper3 = _rolling_series_by_stock(data, _upper_shadow_ratio(data), 3, 2, "mean")

    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr5 = _rolling_series_by_stock(data, true_range, 5, 3, "mean")
    atr20 = _rolling_series_by_stock(data, true_range, 20, 10, "mean")
    contraction = atr5 / (atr20 + _EPS)
    high10 = _rolling_col(data, "high", 10, 5, "max")
    pullback = close / (high10 + _EPS) - 1.0
    range5 = _rolling_series_by_stock(data, (high - low) / (prev_close.abs() + _EPS), 5, 3, "mean")
    range20 = _rolling_series_by_stock(data, (high - low) / (prev_close.abs() + _EPS), 20, 10, "mean")
    range_contraction = range5 / (range20 + _EPS)

    raw_features = {
        "ret3": ret3,
        "ret5": ret5,
        "ret10": ret10,
        "accel": accel,
        "ma5_gap": ma5_gap,
        "ma10_gap": ma10_gap,
        "pullback": pullback,
        "close_strength": close_strength,
        "upper3": upper3,
        "atr_contraction": contraction,
        "range_contraction": range_contraction,
    }
    ranked = []
    for value in raw_features.values():
        rank = _daily_pct_rank(value, dates, min_obs=20, eligible=eligible)
        if rank.notna().sum() < max(100, int(0.05 * len(rank))):
            rank = value.where(eligible).rank(pct=True, method="average")
        ranked.append((rank - 0.5).to_numpy(dtype="float32"))
    sig = np.column_stack(ranked).astype("float32")
    norms = np.linalg.norm(sig, axis=1)
    valid = np.isfinite(sig).all(axis=1) & np.isfinite(norms) & (norms > _EPS)
    sig[valid] /= norms[valid, None]
    sig[~valid] = np.nan
    return sig, raw_features


def _build_causal_daily_prototypes(
    signatures: np.ndarray,
    outcomes: pd.Series,
    realized_dates: pd.Series,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_keys = dates.to_numpy(dtype="datetime64[ns]").astype("int64")
    realized_keys = pd.to_datetime(realized_dates, errors="coerce").to_numpy(dtype="datetime64[ns]").astype("int64")
    y = outcomes.to_numpy(dtype="float64")
    valid = np.isfinite(signatures).all(axis=1) & np.isfinite(y) & pd.notna(dates).to_numpy() & pd.notna(realized_dates).to_numpy()

    proto_pos = []
    proto_neg = []
    proto_realized = []
    frame = pd.DataFrame({"date_key": date_keys, "pos": np.arange(len(dates), dtype=np.int64)})
    for _, positions in frame.loc[valid].groupby("date_key", sort=True)["pos"]:
        pos = positions.to_numpy(dtype=np.int64)
        if pos.size < _MIN_PROTO_OBS:
            continue
        values = y[pos]
        q_low, q_high = np.quantile(values, [_PROTO_QUANTILE, 1.0 - _PROTO_QUANTILE])
        winners = pos[values >= q_high]
        losers = pos[values <= q_low]
        if winners.size < 3 or losers.size < 3:
            continue
        win_proto = np.mean(signatures[winners], axis=0)
        lose_proto = np.mean(signatures[losers], axis=0)
        win_norm = float(np.linalg.norm(win_proto))
        lose_norm = float(np.linalg.norm(lose_proto))
        if win_norm <= _EPS or lose_norm <= _EPS:
            continue
        proto_pos.append((win_proto / win_norm).astype("float32"))
        proto_neg.append((lose_proto / lose_norm).astype("float32"))
        proto_realized.append(int(max(np.max(realized_keys[winners]), np.max(realized_keys[losers]))))
    if not proto_pos:
        empty = np.empty((0, signatures.shape[1]), dtype="float32")
        return empty, empty, np.empty(0, dtype=np.int64)
    return np.vstack(proto_pos), np.vstack(proto_neg), np.asarray(proto_realized, dtype=np.int64)


def _local_knn_similarity_full(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    required = {"stock_code", "date", "open", "high", "low", "close"}
    if not required.issubset(data.columns):
        nan = _nan(data.index)
        return nan, nan

    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    current_sigs, state = _continuation_state_signatures(data, dates)
    outcomes, realized_dates = _forward_weighted_return_and_realized_date(data)
    pos_protos, neg_protos, proto_realized_keys = _build_causal_daily_prototypes(
        current_sigs, outcomes, realized_dates, dates
    )
    raw = pd.Series(np.nan, index=data.index, dtype="float64")
    if pos_protos.shape[0] < 5:
        return raw, raw

    date_keys = dates.to_numpy(dtype="datetime64[ns]").astype("int64")
    valid_date_mask = pd.notna(dates).to_numpy()
    unique_date_keys = np.sort(pd.unique(date_keys[valid_date_mask]))
    date_to_rows = {
        int(date_key): np.asarray(positions, dtype=np.int64)
        for date_key, positions in pd.DataFrame({"date_key": date_keys}).groupby("date_key", sort=False).indices.items()
    }

    day_ns = int(pd.Timedelta(days=1).value)
    for date_key in unique_date_keys:
        date_key = int(date_key)
        rows = date_to_rows.get(date_key)
        if rows is None or rows.size == 0:
            continue
        lo = date_key - _LOCAL_LOOKBACK_DAYS * day_ns
        mask = (proto_realized_keys >= lo) & (proto_realized_keys < date_key)
        bank_pos = np.flatnonzero(mask)
        if bank_pos.size < 10:
            lo2 = date_key - _FALLBACK_LOOKBACK_DAYS * day_ns
            bank_pos = np.flatnonzero((proto_realized_keys >= lo2) & (proto_realized_keys < date_key))
        if bank_pos.size < 5:
            continue

        sig = current_sigs[rows]
        valid_rows = np.isfinite(sig).all(axis=1)
        if not valid_rows.any():
            continue
        valid_pos = rows[valid_rows]
        win_bank = pos_protos[bank_pos]
        lose_bank = neg_protos[bank_pos]
        k = min(_TOPK, win_bank.shape[0])
        for start in range(0, valid_pos.size, _SCORE_CHUNK_ROWS):
            chunk_pos = valid_pos[start : start + _SCORE_CHUNK_ROWS]
            win_sims = np.clip(current_sigs[chunk_pos] @ win_bank.T, -1.0, 1.0)
            lose_sims = np.clip(current_sigs[chunk_pos] @ lose_bank.T, -1.0, 1.0)
            win_top = np.partition(win_sims, win_sims.shape[1] - k, axis=1)[:, -k:]
            lose_top = np.partition(lose_sims, lose_sims.shape[1] - k, axis=1)[:, -k:]
            raw.iloc[chunk_pos] = (np.mean(win_top, axis=1) - np.mean(lose_top, axis=1)).astype("float64")

    eligible = non_st_mask(data)
    raw_z = _daily_zscore_with_fallback(raw, dates, min_obs=20, eligible=eligible)

    acceptance = (
        0.45 * state["close_strength"]
        + 0.25 * (1.0 - state["upper3"])
        + 0.15 * (1.0 - _daily_pct_rank(state["atr_contraction"], dates, min_obs=20, eligible=eligible))
        + 0.15 * (1.0 - _daily_pct_rank(state["range_contraction"], dates, min_obs=20, eligible=eligible))
    )
    acceptance_z = _daily_zscore_with_fallback(acceptance, dates, min_obs=20, eligible=eligible)
    exhaustion = (
        0.40 * _daily_pct_rank(state["ret5"], dates, min_obs=20, eligible=eligible)
        + 0.25 * _daily_pct_rank(state["ma5_gap"], dates, min_obs=20, eligible=eligible)
        + 0.20 * _daily_pct_rank(state["upper3"], dates, min_obs=20, eligible=eligible)
        + 0.15 * _daily_pct_rank(state["pullback"], dates, min_obs=20, eligible=eligible)
    )
    exhaustion_z = _daily_zscore_with_fallback(exhaustion, dates, min_obs=20, eligible=eligible)
    shapelet = _pre_second_breakout_shapelet_corr_20d_v2(data, None)
    shapelet_z = _daily_zscore_with_fallback(shapelet, dates, min_obs=20, eligible=eligible)

    quality = (
        raw_z
        + 0.30 * acceptance_z
        - 0.35 * exhaustion_z
        + 0.10 * shapelet_z
    ).clip(-_Z_CLIP, _Z_CLIP)
    return raw_z.astype("float64"), quality.astype("float64")


def _pre_breakout_local_knn_similarity_20d_v2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    raw, _quality = _local_knn_similarity_full(data)
    return raw.astype("float64")


def _pre_breakout_local_knn_similarity_20d_v2_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_pre_breakout_local_knn_similarity_20d_v2(data, _), target_index=target_index)


pre_breakout_local_knn_similarity_20d_v2 = register_factor(
    FactorDefinition(
        name="pre_breakout_local_knn_similarity_20d_v2",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=220,
        calculator=_pre_breakout_local_knn_similarity_20d_v2,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_breakout_local_knn_similarity_20d_v2_target,
        description="短周期因果对比原型相似度v2：贴近T+1开盘买入、T+2/T+3加权退出的成功延续状态，并远离失败衰竭状态",
    )
)


def _pre_breakout_local_knn_quality_20d_v2(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    _raw, quality = _local_knn_similarity_full(data)
    return quality.astype("float64")


def _pre_breakout_local_knn_quality_20d_v2_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_pre_breakout_local_knn_quality_20d_v2(data, _), target_index=target_index)


pre_breakout_local_knn_quality_20d_v2 = register_factor(
    FactorDefinition(
        name="pre_breakout_local_knn_quality_20d_v2",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=220,
        calculator=_pre_breakout_local_knn_quality_20d_v2,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_breakout_local_knn_quality_20d_v2_target,
        description="短周期因果对比原型质量v2：成功/失败原型差异叠加收盘承接、波动收缩与衰竭惩罚",
    )
)
