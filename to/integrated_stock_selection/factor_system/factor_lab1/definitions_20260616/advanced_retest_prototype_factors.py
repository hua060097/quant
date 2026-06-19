"""
Advanced retest and local-prototype structure factors.

This module is designed for the existing FactorDefinition/register_factor framework.
It intentionally does NOT replace the already-good v2 factors immediately.  It adds
new, auditable v3-style candidates that address two practical issues:

1) Prior-high retest / breakout quality
   The previous position_20_60_spread_v2 only says that short-term position is
   improving relative to medium-term position.  That is useful but incomplete.
   A better structural signal should ask:
       - Is price approaching or reclaiming a prior high?
       - Is the close strong relative to the intraday range?
       - Is volume/amount confirming without obvious blow-off?
       - Is upper-shadow supply pressure low?
       - Is the previous peak sufficiently separated in time?
       - Is the current state not already extremely over-extended?

   The factor below expresses this as one continuous score:
       prior_high_retest_quality_60d

2) Causal local prototype matching
   The failed expanding causal prototype was too crude: it mixed long historical
   regimes into a single average path and its path matching was too loose/rigid.
   This module instead builds a recent, lagged historical prototype bank:
       - At each historical event date E, select already-observed strong-but-not-
         extreme trailing 20d movers using cross-sectional ranks, not future labels.
       - For each event, extract the path immediately before the strongest already-
         observed upward segment inside the trailing window.
       - Build one daily prototype from that day's event prepaths.
       - At scoring date T, compare each current 20d path only to daily prototypes
         dated in [T - LOCAL_PROTO_LOOKBACK_DAYS, T - PROTOTYPE_LAG_DAYS].
       - Use top-k local ensemble similarity, not one global prototype.
       - Adjust for current maturity via cross-sectional rank of current ret20,
         position, and upper-shadow pressure, so already-mature/overheated paths
         do not dominate.

No future return label, no full-sample clustering, and no current/future prototype
is used in the prototype factor.  The factor is still experimental; keep it next
to pre_second_breakout_shapelet_corr_20d_v2 and position_20_60_spread_v2 for A/B
validation.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series

_EPS = 1e-12
_Z_CLIP = 6.0

# =============================================================================
# Generic helpers
# =============================================================================


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
        raise ValueError(f"Unsupported rolling aggregation: {agg}")
    return out.reset_index(level=0, drop=True).reindex(data.index).astype("float64")


def _rolling_shifted_col(
    data: pd.DataFrame,
    col: str,
    window: int,
    min_periods: int,
    shift_days: int,
    agg: str,
) -> pd.Series:
    """Rolling aggregation by stock after causal shift, e.g. prior high excluding latest days."""
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
        raise ValueError(f"Unsupported rolling shifted aggregation: {agg}")
    return out.reset_index(level=0, drop=True).reindex(data.index).astype("float64")


def _shift_by_stock(data: pd.DataFrame, s: pd.Series, periods: int) -> pd.Series:
    if "stock_code" not in data.columns:
        return _nan(data.index)
    ordered_idx = _ordered_index(data)
    ordered_s = s.reindex(ordered_idx)
    out = ordered_s.groupby(data.loc[ordered_idx, "stock_code"], sort=False).shift(periods)
    return out.reindex(data.index).astype("float64")


def _log_return(data: pd.DataFrame, periods: int) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return _nan(data.index)
    close = _safe_num(data["close"])
    prev = _shift_by_stock(data, close, periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(close / prev)
    return out.replace([np.inf, -np.inf], np.nan).astype("float64")


def _daily_pct_rank(s: pd.Series, dates: pd.Series, min_obs: int = 30) -> pd.Series:
    work = pd.DataFrame({"v": _safe_num(s).values, "date": dates.values}, index=s.index)
    out = work.groupby("date", sort=False)["v"].rank(pct=True, method="average")
    counts = work.groupby("date", sort=False)["v"].transform("count")
    out = out.where(counts >= min_obs, np.nan)
    return out.astype("float64")


def _daily_zscore(s: pd.Series, dates: pd.Series, min_obs: int = 30) -> pd.Series:
    work = pd.DataFrame({"v": _safe_num(s).values, "date": dates.values}, index=s.index)
    g = work.groupby("date", sort=False)["v"]
    med = g.transform("median")
    q75 = g.transform(lambda x: x.quantile(0.75))
    q25 = g.transform(lambda x: x.quantile(0.25))
    iqr = (q75 - q25).replace(0.0, np.nan)
    out = (work["v"] - med) / (iqr / 1.349 + _EPS)
    counts = g.transform("count")
    out = out.where(counts >= min_obs, np.nan)
    return out.clip(-_Z_CLIP, _Z_CLIP).astype("float64")


def _close_position_extreme(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)
    high_n = _rolling_col(data, "high", window, min_periods, "max")
    low_n = _rolling_col(data, "low", window, min_periods, "min")
    close = _safe_num(data["close"])
    denom = high_n - low_n
    pos = (close - low_n) / (denom + _EPS)
    return pos.where(denom > _EPS, np.nan).clip(0.0, 1.0).astype("float64")


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


# =============================================================================
# 1. Prior-high retest / breakout quality
# =============================================================================

_PRIOR_HIGH_WINDOW = 60
_PRIOR_HIGH_SHIFT = 5
_PRIOR_HIGH_MINP = 30


def _days_since_shifted_rolling_high(data: pd.DataFrame, window: int = 60, shift_days: int = 5) -> pd.Series:
    """Days between current row and the most recent high inside [t-shift-window+1, t-shift]."""
    if not {"stock_code", "date", "high"}.issubset(data.columns):
        return _nan(data.index)
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    date_values = pd.to_datetime(data["date"], errors="coerce").to_numpy()
    high_values = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype="float64")
    for _, positions in data.groupby("stock_code", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < shift_days + 2:
            continue
        order = np.argsort(date_values[pos], kind="mergesort")
        sp = pos[order]
        h = high_values[sp]
        n = sp.size
        out = np.full(n, np.nan, dtype="float64")
        for i in range(n):
            end = i - shift_days
            if end < 0:
                continue
            start = max(0, end - window + 1)
            window_h = h[start : end + 1]
            finite = np.isfinite(window_h)
            if finite.sum() < max(5, min(window, _PRIOR_HIGH_MINP) // 3):
                continue
            # Most recent occurrence of the rolling maximum.
            vals = np.where(finite, window_h, -np.inf)
            max_v = np.nanmax(vals)
            rel_candidates = np.flatnonzero(vals == max_v)
            if rel_candidates.size == 0:
                continue
            peak_idx = start + int(rel_candidates[-1])
            out[i] = float(i - peak_idx)
        result.iloc[sp] = out
    return result.astype("float64")


def _prior_high_retest_quality_60d_full(data: pd.DataFrame) -> pd.Series:
    """
    Prior-high retest / reclaim quality.

    This is a continuous, volatility-normalized structural quality score, not a
    binary breakout rule.  It rewards price touching/reclaiming a meaningful prior
    high with good close location, reasonable volume confirmation, moderate peak
    interval, and low upper-shadow supply pressure.  It penalizes current maturity
    through current ret20/ret5/position ranks instead of using fixed return caps.
    """
    required = {"stock_code", "date", "open", "high", "low", "close"}
    if not required.issubset(data.columns):
        return _nan(data.index)

    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    close = _safe_num(data["close"])
    high = _safe_num(data["high"])

    prior_high = _rolling_shifted_col(
        data,
        "high",
        window=_PRIOR_HIGH_WINDOW,
        min_periods=_PRIOR_HIGH_MINP,
        shift_days=_PRIOR_HIGH_SHIFT,
        agg="max",
    )
    vol20 = _rolling_col(data, "close", 20, 10, "std") / (close.abs() + _EPS)
    vol20 = vol20.clip(lower=0.01, upper=0.20)

    # Distance to prior high in volatility units.  Near/reclaiming prior high has high score.
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_log = np.log(close / prior_high)
        high_touch_log = np.log(high / prior_high)
    dist_sigma = dist_log / (vol20 + _EPS)
    touch_sigma = high_touch_log / (vol20 + _EPS)

    # Smooth proximity to prior high.  It is high near the prior high and remains
    # constructive on slight reclaim, but does not over-reward far-above-prior-high states.
    proximity = np.exp(-0.5 * np.square(dist_sigma.clip(-6.0, 6.0)))
    reclaim_component = 1.0 / (1.0 + np.exp(-touch_sigma.clip(-8.0, 8.0)))

    close_strength = _intraday_close_strength(data)
    upper_shadow = _upper_shadow_ratio(data)
    upper_shadow_3d = _rolling_col(
        pd.DataFrame({
            "stock_code": data["stock_code"],
            "date": data["date"],
            "upper": upper_shadow,
        }, index=data.index),
        "upper",
        3,
        1,
        "mean",
    )

    # Amount/volume confirmation. Prefer amount if available, fallback to volume.
    flow_col = "amount" if "amount" in data.columns else "volume" if "volume" in data.columns else None
    if flow_col is not None:
        flow = _safe_num(data[flow_col])
        flow_ma20 = _rolling_col(data, flow_col, 20, 10, "mean")
        flow_ratio = np.log1p((flow / (flow_ma20 + _EPS)).clip(lower=0.0))
        flow_rank = _daily_pct_rank(flow_ratio, dates, min_obs=50)
    else:
        flow_rank = pd.Series(0.5, index=data.index, dtype="float64")

    # Peak interval: too recent prior high often means noisy chase; too stale may be irrelevant.
    # Use a smooth hump over normalized days-since-high, not a hard day threshold.
    days_since_peak = _days_since_shifted_rolling_high(data, _PRIOR_HIGH_WINDOW, _PRIOR_HIGH_SHIFT)
    interval_x = (days_since_peak / float(_PRIOR_HIGH_WINDOW)).clip(0.0, 1.0)
    interval_score = np.sin(np.pi * interval_x).clip(0.0, 1.0)

    # Current maturity penalty by cross-sectional ranks, no fixed return cap.
    ret20 = _log_return(data, 20)
    ret5 = _log_return(data, 5)
    pos20 = _close_position_extreme(data, 20, 10)
    maturity_rank = (
        0.45 * _daily_pct_rank(ret20, dates, min_obs=50)
        + 0.30 * _daily_pct_rank(pos20, dates, min_obs=50)
        + 0.25 * _daily_pct_rank(ret5, dates, min_obs=50)
    )
    maturity_control = (1.0 - 0.65 * maturity_rank).clip(0.15, 1.0)

    quality_raw = (
        proximity
        * reclaim_component
        * (0.35 + 0.65 * close_strength)
        * (0.35 + 0.65 * flow_rank.fillna(0.5))
        * (0.35 + 0.65 * interval_score.fillna(0.5))
        * (1.0 - 0.75 * upper_shadow_3d.fillna(0.0).clip(0.0, 1.0))
        * maturity_control.fillna(0.5)
    )

    out = _daily_zscore(quality_raw.astype("float64"), dates, min_obs=50)
    return out.astype("float64")


def _prior_high_retest_quality_60d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _prior_high_retest_quality_60d_full(data)


def _prior_high_retest_quality_60d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_prior_high_retest_quality_60d_full(data), target_index=target_index)


prior_high_retest_quality_60d = register_factor(
    FactorDefinition(
        name="prior_high_retest_quality_60d",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=90,
        calculator=_prior_high_retest_quality_60d,
        calculation_scope="target_with_history_window",
        target_calculator=_prior_high_retest_quality_60d_target,
        description="前高触及/突破质量：60日前高距离+收盘强度+量能确认+峰值间隔+上影线压力+当前成熟度控制的连续结构分数",
    )
)


# =============================================================================
# 2. Local causal prototype bank similarity
# =============================================================================

_PATH_WINDOW = 20
_EVENT_WINDOW = 20
_PREPATH_WINDOW = 20
_PROTO_LAG_DAYS = 5
_LOCAL_PROTO_LOOKBACK_DAYS = 90
_TOPK_PROTO = 8
_MIN_DAILY_EVENT_OBS = 20
_MIN_DAILY_SCORE_OBS = 50


def _path_signature_from_close_window(window_close: np.ndarray) -> np.ndarray | None:
    """
    Multi-view path signature for similarity matching.

    Signature components:
      1. detrended cumulative log-price path, length 20
      2. standardized daily log-return path, length 19
      3. drawdown-from-running-max path, length 20

    The final vector is L2-normalized, so dot product is cosine similarity.
    """
    c = np.asarray(window_close, dtype="float64")
    if c.size != _PATH_WINDOW or (not np.isfinite(c).all()) or np.any(c <= 0.0):
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.log(c / c[0])
        rets = np.diff(np.log(c))

    if not np.isfinite(path).all() or not np.isfinite(rets).all():
        return None

    t = np.linspace(-1.0, 1.0, _PATH_WINDOW, dtype="float64")
    centered = path - path.mean()
    slope = float(np.dot(centered, t) / (np.dot(t, t) + _EPS))
    path_resid = centered - slope * t
    path_std = float(np.std(path_resid))
    if not np.isfinite(path_std) or path_std <= _EPS:
        return None
    path_z = path_resid / path_std

    ret_std = float(np.std(rets))
    if not np.isfinite(ret_std) or ret_std <= _EPS:
        ret_z = np.zeros_like(rets)
    else:
        ret_z = (rets - float(np.mean(rets))) / ret_std

    running_max = np.maximum.accumulate(path)
    drawdown = path - running_max
    dd_std = float(np.std(drawdown))
    if not np.isfinite(dd_std) or dd_std <= _EPS:
        dd_z = np.zeros_like(drawdown)
    else:
        dd_z = (drawdown - float(np.mean(drawdown))) / dd_std

    sig = np.concatenate([path_z, 0.65 * ret_z, 0.75 * dd_z]).astype("float64")
    norm = float(np.linalg.norm(sig))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return (sig / norm).astype("float32")


_SIG_DIM = _PATH_WINDOW + (_PATH_WINDOW - 1) + _PATH_WINDOW


def _current_path_signatures(data: pd.DataFrame) -> np.ndarray:
    """Signature ending at each row, in original row order."""
    out = np.full((len(data), _SIG_DIM), np.nan, dtype="float32")
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return out

    close_all = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype="float64")
    date_values = pd.to_datetime(data["date"], errors="coerce").to_numpy()

    for _, positions in data.groupby("stock_code", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < _PATH_WINDOW:
            continue
        order = np.argsort(date_values[pos], kind="mergesort")
        sp = pos[order]
        c = close_all[sp]
        for i in range(_PATH_WINDOW - 1, sp.size):
            sig = _path_signature_from_close_window(c[i - _PATH_WINDOW + 1 : i + 1])
            if sig is not None:
                out[sp[i], :] = sig
    return out


def _event_prepath_signatures(data: pd.DataFrame) -> np.ndarray:
    """
    For each event date E, find the strongest already-observed upward segment
    inside [E-19, E], then extract the 20d path ending immediately before that
    segment starts.  This is causal relative to E.
    """
    out = np.full((len(data), _SIG_DIM), np.nan, dtype="float32")
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return out

    close_all = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype="float64")
    date_values = pd.to_datetime(data["date"], errors="coerce").to_numpy()

    for _, positions in data.groupby("stock_code", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size < (_EVENT_WINDOW + _PREPATH_WINDOW):
            continue
        order = np.argsort(date_values[pos], kind="mergesort")
        sp = pos[order]
        c = close_all[sp]
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.log(c)
        for i in range(_EVENT_WINDOW - 1, sp.size):
            ew_start = i - _EVENT_WINDOW + 1
            ew_end = i
            w = logc[ew_start : ew_end + 1]
            if not np.isfinite(w).all():
                continue
            best_ret = -np.inf
            best_start = None
            min_val = float(w[0])
            min_idx = ew_start
            for j_rel in range(1, w.size):
                v = float(w[j_rel])
                seg_ret = v - min_val
                if seg_ret > best_ret:
                    best_ret = seg_ret
                    best_start = min_idx
                if v < min_val:
                    min_val = v
                    min_idx = ew_start + j_rel
            if best_start is None:
                continue
            pre_end = int(best_start)
            pre_start = pre_end - _PREPATH_WINDOW + 1
            if pre_start < 0:
                continue
            sig = _path_signature_from_close_window(c[pre_start : pre_end + 1])
            if sig is not None:
                out[sp[i], :] = sig
    return out


def _build_daily_local_prototypes(data: pd.DataFrame, event_sigs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build one weighted daily prototype per date from event prepath signatures."""
    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    unique_dates = np.array(sorted(pd.unique(dates.dropna())), dtype="datetime64[ns]")
    if unique_dates.size == 0:
        return unique_dates, np.empty((0, _SIG_DIM), dtype="float32")

    ret20 = _log_return(data, 20)
    ret5 = _log_return(data, 5)
    close_strength = _intraday_close_strength(data)
    upper_shadow = _upper_shadow_ratio(data)
    ret20_rank = _daily_pct_rank(ret20, dates, min_obs=50)
    ret5_rank = _daily_pct_rank(ret5, dates, min_obs=50)
    shadow_rank = _daily_pct_rank(upper_shadow, dates, min_obs=50)

    # Strong-but-not-extreme event weight.  This uses a smooth rank-kernel centered
    # around high-but-not-maximal movers, avoiding fixed return thresholds and
    # reducing mature blow-off events.
    r = ret20_rank.astype("float64")
    strength_kernel = np.exp(-0.5 * np.square((r - 0.88) / 0.10))
    ret5_overheat_control = (1.0 - 0.50 * ret5_rank).clip(0.35, 1.0)
    shadow_control = (1.0 - 0.75 * shadow_rank).clip(0.15, 1.0)
    close_control = (0.25 + 0.75 * close_strength).clip(0.25, 1.0)
    weight = (strength_kernel * ret5_overheat_control * shadow_control * close_control).astype("float64")

    proto_by_date = np.full((unique_dates.size, _SIG_DIM), np.nan, dtype="float32")
    date_to_pos = {d: i for i, d in enumerate(unique_dates)}
    date_values = dates.to_numpy(dtype="datetime64[ns]")

    for d, idx_obj in data.groupby(dates, sort=False).groups.items():
        if pd.isna(d):
            continue
        date_pos = date_to_pos.get(np.datetime64(d))
        if date_pos is None:
            continue
        idx = np.asarray(list(idx_obj), dtype=np.int64)
        sigs = event_sigs[idx]
        valid_sig = np.isfinite(sigs).all(axis=1)
        w = weight.reindex(data.index[idx]).to_numpy(dtype="float64")
        valid = valid_sig & np.isfinite(w) & (w > 0.05)
        if int(valid.sum()) < _MIN_DAILY_EVENT_OBS:
            continue
        ww = w[valid]
        ss = sigs[valid].astype("float64")
        proto = np.average(ss, axis=0, weights=ww)
        norm = float(np.linalg.norm(proto))
        if np.isfinite(norm) and norm > _EPS:
            proto_by_date[date_pos, :] = (proto / norm).astype("float32")
    return unique_dates, proto_by_date


def _local_prototype_similarity_full(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return raw local similarity and maturity-adjusted quality score."""
    required = {"stock_code", "date", "open", "high", "low", "close"}
    if not required.issubset(data.columns):
        nan = _nan(data.index)
        return nan, nan

    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    current_sigs = _current_path_signatures(data)
    event_sigs = _event_prepath_signatures(data)
    proto_dates, protos = _build_daily_local_prototypes(data, event_sigs)

    raw = pd.Series(np.nan, index=data.index, dtype="float64")
    if proto_dates.size == 0 or protos.size == 0:
        return raw, raw

    unique_dates = np.array(sorted(pd.unique(dates.dropna())), dtype="datetime64[ns]")
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    date_to_rows: dict[np.datetime64, np.ndarray] = {}
    for d, idx_obj in data.groupby(dates, sort=False).groups.items():
        if pd.isna(d):
            continue
        date_to_rows[np.datetime64(d)] = np.asarray(list(idx_obj), dtype=np.int64)

    proto_valid = np.isfinite(protos).all(axis=1)
    for d in unique_dates:
        rows = date_to_rows.get(d)
        if rows is None or rows.size == 0:
            continue
        # Use only local, already-available prototypes: [T-lookback, T-lag].
        lo = d - np.timedelta64(_LOCAL_PROTO_LOOKBACK_DAYS, "D")
        hi = d - np.timedelta64(_PROTO_LAG_DAYS, "D")
        mask = (proto_dates >= lo) & (proto_dates <= hi) & proto_valid
        if int(mask.sum()) < 3:
            continue
        bank = protos[mask].astype("float64")
        sig = current_sigs[rows].astype("float64")
        valid = np.isfinite(sig).all(axis=1)
        if int(valid.sum()) < _MIN_DAILY_SCORE_OBS:
            continue
        sims = sig[valid] @ bank.T
        sims = np.clip(sims, -1.0, 1.0)
        k = min(_TOPK_PROTO, sims.shape[1])
        if k <= 0:
            continue
        # Top-k ensemble: less brittle than max, more selective than mean.
        topk = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
        score = np.mean(topk, axis=1)
        raw.iloc[rows[valid]] = score.astype("float64")

    # Raw similarity is useful for diagnostics, but as an alpha we apply a maturity
    # adjustment to avoid already-finished moves.  This is rank-based, not a fixed
    # return threshold.
    ret20 = _log_return(data, 20)
    ret5 = _log_return(data, 5)
    pos20 = _close_position_extreme(data, 20, 10)
    upper = _upper_shadow_ratio(data)

    raw_z = _daily_zscore(raw, dates, min_obs=_MIN_DAILY_SCORE_OBS)
    maturity = (
        0.40 * _daily_pct_rank(ret20, dates, min_obs=_MIN_DAILY_SCORE_OBS)
        + 0.25 * _daily_pct_rank(ret5, dates, min_obs=_MIN_DAILY_SCORE_OBS)
        + 0.20 * _daily_pct_rank(pos20, dates, min_obs=_MIN_DAILY_SCORE_OBS)
        + 0.15 * _daily_pct_rank(upper, dates, min_obs=_MIN_DAILY_SCORE_OBS)
    )
    maturity_z = _daily_zscore(maturity, dates, min_obs=_MIN_DAILY_SCORE_OBS)
    adjusted = (raw_z - 0.35 * maturity_z).clip(-_Z_CLIP, _Z_CLIP)
    return raw.astype("float64"), adjusted.astype("float64")


def _pre_breakout_local_prototype_similarity_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    raw, _adjusted = _local_prototype_similarity_full(data)
    dates = pd.Series(pd.to_datetime(data["date"], errors="coerce"), index=data.index)
    return _daily_zscore(raw, dates, min_obs=_MIN_DAILY_SCORE_OBS).astype("float64")


def _pre_breakout_local_prototype_similarity_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_pre_breakout_local_prototype_similarity_20d(data, _), target_index=target_index)


pre_breakout_local_prototype_similarity_20d = register_factor(
    FactorDefinition(
        name="pre_breakout_local_prototype_similarity_20d",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=160,
        calculator=_pre_breakout_local_prototype_similarity_20d,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_breakout_local_prototype_similarity_20d_target,
        description="局部因果prototype相似度raw版：当前20日多视角路径与T-5以前近历史强势启动前路径库的top-k cosine相似度",
    )
)


def _pre_breakout_local_prototype_quality_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    _raw, adjusted = _local_prototype_similarity_full(data)
    return adjusted.astype("float64")


def _pre_breakout_local_prototype_quality_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return select_target_series(_pre_breakout_local_prototype_quality_20d(data, _), target_index=target_index)


pre_breakout_local_prototype_quality_20d = register_factor(
    FactorDefinition(
        name="pre_breakout_local_prototype_quality_20d",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=160,
        calculator=_pre_breakout_local_prototype_quality_20d,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_breakout_local_prototype_quality_20d_target,
        description="局部因果prototype质量版：prototype相似度经当前20日涨幅/5日涨幅/20日位置/上影线成熟度调整后的结构分数",
    )
)
