"""
Causal adaptive structure factors for A-share short-horizon stock selection.

This module implements two data-driven structure factors under the existing
FactorDefinition/register_factor framework:

1) position_reclaim_residual_20_60
   A threshold-free cross-sectional residual factor.  It estimates the normal
   same-day relation between 20-day position and medium-term state variables
   (60-day position, 20-day return, 20-day volatility), then outputs the daily
   standardized residual.  Positive values mean the stock's short-term position
   is stronger than expected after controlling for medium-term position, recent
   run-up, and volatility.

2) pre_breakout_causal_prototype_similarity_20d
   A causal, label-free historical prototype similarity factor.

   Design goal:
   - Do NOT use future returns or model labels.
   - Do NOT use hand-written shapelet templates.
   - Learn historical "pre-breakout" path prototypes only from events that were
     already observable before the scored date.

   Algorithm overview:
   - For each historical stock-date E, observe only the 20 trading days ending at
     E.  Stocks with stronger *already-realized* trailing 20-day moves receive
     higher cross-sectional weights for that historical date.  This is not a
     future label: at E the move has already happened.
   - For each historical stock-date E, locate the strongest upward segment inside
     the trailing 20-day window, then extract the 20-day path immediately before
     that segment started.  This approximates the "pre-breakout" structure that
     preceded the already-observed strong move.
   - Aggregate these pre-segment paths into one daily contrast prototype by a
     cross-sectional weighted average.  Weights are centered daily return ranks,
     so the prototype describes paths that preceded stronger trailing-20d movers
     relative to weaker ones; no hard top-percentile threshold is used.
   - When scoring date T, use only prototypes from dates <= T - 5 trading days.
     This T-5 lag makes the prototype library strictly historical relative to T.
   - Score the current 20-day path by cosine similarity to the lagged expanding
     historical prototype.
   - Finally residualize that similarity by current 20-day return rank and
     current 20-day position rank on each date.  This removes the component that
     merely says "the stock has already risen a lot", reducing high-maturity /
     already-overextended false positives without hard thresholds.

All calculations are causal with respect to the scored date.  Missing or
insufficient-history observations return NaN; the module does not fabricate
neutral values such as 0 or 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-12
PATH_WINDOW = 20
EVENT_WINDOW = 20
PROTOTYPE_LAG_DAYS = 5
MIN_CROSS_SECTION_OBS = 80
MIN_PROTOTYPE_DAYS = 30


def _nan(index: pd.Index | range) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_num(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.Series(s, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _target_from_full(full: pd.Series, target_index: pd.Index) -> pd.Series:
    return full.loc[target_index].reset_index(drop=True).astype("float64")


def _sort_index(data: pd.DataFrame) -> pd.Index:
    if {"stock_code", "date"}.issubset(data.columns):
        return data.sort_values(["stock_code", "date"], kind="mergesort").index
    return data.index


def _rolling_col(data: pd.DataFrame, col: str, window: int, min_periods: int, agg: str) -> pd.Series:
    if col not in data.columns or "stock_code" not in data.columns:
        return _nan(data.index)
    ordered_idx = _sort_index(data)
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


def _close_position_extreme(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """Close position in rolling high/low range, NaN-preserving, clipped to [0, 1]."""
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)
    high_n = _rolling_col(data, "high", window, min_periods, "max")
    low_n = _rolling_col(data, "low", window, min_periods, "min")
    close = _safe_num(data["close"])
    denom = high_n - low_n
    pos = (close - low_n) / (denom + _EPS)
    pos = pos.where(denom > _EPS, np.nan)
    return pos.clip(0.0, 1.0).astype("float64")


def _log_return(data: pd.DataFrame, periods: int) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return _nan(data.index)
    ordered_idx = _sort_index(data)
    close = _safe_num(data.loc[ordered_idx, "close"])
    log_close = np.log(close.where(close > 0.0))
    prev = log_close.groupby(data.loc[ordered_idx, "stock_code"], sort=False).shift(periods)
    return (log_close - prev).reindex(data.index).astype("float64")


def _rolling_logret_std(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return _nan(data.index)
    ordered_idx = _sort_index(data)
    close = _safe_num(data.loc[ordered_idx, "close"])
    log_close = np.log(close.where(close > 0.0))
    one_day = log_close.groupby(data.loc[ordered_idx, "stock_code"], sort=False).diff()
    out = (
        one_day.groupby(data.loc[ordered_idx, "stock_code"], sort=False)
        .rolling(window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )
    return out.reindex(data.index).astype("float64")


def _daily_pct_rank(s: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"v": _safe_num(s).values, "date": dates.values}, index=s.index)
    return df.groupby("date", sort=False)["v"].rank(pct=True).astype("float64")


def _daily_residual_zscore(
    y: pd.Series,
    x_list: list[pd.Series],
    dates: pd.Series,
    min_obs: int = MIN_CROSS_SECTION_OBS,
) -> pd.Series:
    """
    Daily cross-sectional OLS residual, standardized within each date.

    This is used to remove common state effects without manually specified
    thresholds or fixed weights.
    """
    if len(y) == 0:
        return pd.Series(dtype="float64")

    work = pd.DataFrame({"y": _safe_num(y).values, "date": dates.values}, index=y.index)
    for i, x in enumerate(x_list):
        work[f"x{i}"] = _safe_num(x).reindex(y.index).values

    x_cols = [f"x{i}" for i in range(len(x_list))]
    out = pd.Series(np.nan, index=y.index, dtype="float64")

    for _, idx in work.groupby("date", sort=False).groups.items():
        sub = work.loc[idx, ["y", *x_cols]].dropna()
        if len(sub) < min_obs:
            continue
        yy = sub["y"].to_numpy(dtype="float64")
        xx = sub[x_cols].to_numpy(dtype="float64")
        xx = np.column_stack([np.ones(len(xx), dtype="float64"), xx])
        try:
            beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yy - xx @ beta
        sd = float(np.nanstd(resid))
        if not np.isfinite(sd) or sd <= _EPS:
            continue
        out.loc[sub.index] = (resid - float(np.nanmean(resid))) / sd

    return out.astype("float64")


def _position_reclaim_residual_20_60_full(data: pd.DataFrame) -> pd.Series:
    """
    Threshold-free short-term position reclaim residual.

    y = pct_rank(close_position_20d)
    X = pct_rank(close_position_60d), pct_rank(ret20), pct_rank(vol20)
    output = daily standardized residual.
    """
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)

    dates = pd.Series(data["date"].values, index=data.index)
    pos20 = _close_position_extreme(data, 20, 10)
    pos60 = _close_position_extreme(data, 60, 30)
    ret20 = _log_return(data, 20)
    vol20 = _rolling_logret_std(data, 20, 10)

    y = _daily_pct_rank(pos20, dates)
    x_pos60 = _daily_pct_rank(pos60, dates)
    x_ret20 = _daily_pct_rank(ret20, dates)
    x_vol20 = _daily_pct_rank(vol20, dates)
    return _daily_residual_zscore(y, [x_pos60, x_ret20, x_vol20], dates)


def _normalize_path_from_log_segment(segment: np.ndarray) -> np.ndarray | None:
    """
    Convert a log-close segment into a mean-centered unit path vector.

    The path keeps its slope/shape information; no hand-written detrending is
    applied, because excessive recent run-up is later removed statistically by
    daily residualization against current ret20/position ranks.
    """
    if segment.shape[0] != PATH_WINDOW or not np.all(np.isfinite(segment)):
        return None
    path = segment - segment[0]
    path = path - float(np.mean(path))
    norm = float(np.sqrt(np.sum(path * path)))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return (path / norm).astype("float32")


def _build_current_paths_and_event_prepaths(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Series, pd.Series]:
    """
    Build current 20d paths and historical event pre-breakout paths.

    For each row i of a stock:
    - current_path[i] = normalized 20d path ending at i.
    - event_prepath[i] = normalized 20d path ending immediately before the start
      of the strongest upward segment observed inside [i-19, i].
    - event_ret20[i] = realized trailing 20d log return ending at i.

    All three use only data up to i.
    """
    n = len(data)
    current_paths = np.full((n, PATH_WINDOW), np.nan, dtype="float32")
    event_prepaths = np.full((n, PATH_WINDOW), np.nan, dtype="float32")
    event_ret20 = np.full(n, np.nan, dtype="float64")

    sorted_idx = _sort_index(data)
    sorted_df = data.loc[sorted_idx, ["stock_code", "date", "close"]].copy()
    sorted_dates = pd.Series(sorted_df["date"].values, index=sorted_idx)
    sorted_codes = pd.Series(sorted_df["stock_code"].values, index=sorted_idx)
    close_sorted = _safe_num(sorted_df["close"]).to_numpy(dtype="float64")
    log_close_sorted = np.full_like(close_sorted, np.nan, dtype="float64")
    valid_close = close_sorted > 0.0
    log_close_sorted[valid_close] = np.log(close_sorted[valid_close])

    # groupby.indices gives positions in sorted_df, which correspond to row
    # positions in log_close_sorted/current_paths/event_prepaths.
    groups = sorted_df.groupby("stock_code", sort=False).indices
    for positions in groups.values():
        pos = np.asarray(positions, dtype=np.int64)
        logs = log_close_sorted[pos]
        m = len(pos)
        if m < PATH_WINDOW:
            continue

        for j in range(m):
            row_pos = pos[j]

            if j >= PATH_WINDOW - 1:
                segment = logs[j - PATH_WINDOW + 1 : j + 1]
                norm_path = _normalize_path_from_log_segment(segment)
                if norm_path is not None:
                    current_paths[row_pos, :] = norm_path

            if j >= PATH_WINDOW:
                if np.isfinite(logs[j]) and np.isfinite(logs[j - PATH_WINDOW]):
                    event_ret20[row_pos] = logs[j] - logs[j - PATH_WINDOW]

            # Need a full trailing event window plus a full pre-segment path.
            if j < EVENT_WINDOW - 1 + PATH_WINDOW:
                continue
            event_window = logs[j - EVENT_WINDOW + 1 : j + 1]
            if not np.all(np.isfinite(event_window)):
                continue

            # Strongest already-observed up segment approximation:
            # peak = highest close inside trailing 20d; start = lowest close before
            # that peak.  This is causal and avoids O(window^2) scans.
            peak_rel = int(np.argmax(event_window))
            start_rel = int(np.argmin(event_window[: peak_rel + 1]))
            start_abs = j - EVENT_WINDOW + 1 + start_rel
            pre_start = start_abs - PATH_WINDOW
            pre_end = start_abs
            if pre_start < 0:
                continue
            pre_segment = logs[pre_start:pre_end]
            pre_path = _normalize_path_from_log_segment(pre_segment)
            if pre_path is not None:
                event_prepaths[row_pos, :] = pre_path

    return current_paths, event_prepaths, event_ret20, sorted_idx, sorted_dates, sorted_codes


def _daily_event_rank_weights(event_ret20_sorted: np.ndarray, sorted_dates: pd.Series) -> pd.Series:
    """Daily percentile ranks of already-realized trailing 20d returns."""
    s = pd.Series(event_ret20_sorted, index=sorted_dates.index, dtype="float64")
    return _daily_pct_rank(s, sorted_dates)


def _prototype_by_date(
    event_prepaths: np.ndarray,
    event_rank: pd.Series,
    sorted_dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one contrast prototype for each date.

    For a date D, weights are event_rank - mean(event_rank) among valid rows on D.
    This is a threshold-free cross-sectional contrast: paths that preceded the
    strongest already-realized 20d movers are emphasized relative to weaker ones.
    """
    unique_dates = pd.Index(pd.Series(sorted_dates.values).drop_duplicates().values)
    date_to_pos = {d: i for i, d in enumerate(unique_dates)}
    protos = np.full((len(unique_dates), PATH_WINDOW), np.nan, dtype="float32")

    # Work in sorted-row positional index. sorted_dates.index are original labels;
    # enumerate over groups provides those labels, convert to sorted positions by
    # an auxiliary Series.
    position_of_label = pd.Series(np.arange(len(sorted_dates), dtype=np.int64), index=sorted_dates.index)
    rank_by_label = event_rank.reindex(sorted_dates.index)

    for date_value, labels in sorted_dates.groupby(sorted_dates, sort=False).groups.items():
        sorted_positions = position_of_label.loc[labels].to_numpy(dtype=np.int64)
        ranks = rank_by_label.loc[labels].to_numpy(dtype="float64")
        paths = event_prepaths[sorted_positions]
        valid = np.isfinite(ranks) & np.isfinite(paths).all(axis=1)
        if int(valid.sum()) < MIN_CROSS_SECTION_OBS:
            continue
        ranks_v = ranks[valid]
        weights = ranks_v - float(np.mean(ranks_v))
        denom = float(np.sum(np.abs(weights)))
        if not np.isfinite(denom) or denom <= _EPS:
            continue
        proto = (paths[valid].astype("float64") * weights[:, None]).sum(axis=0) / denom
        norm = float(np.sqrt(np.sum(proto * proto)))
        if not np.isfinite(norm) or norm <= _EPS:
            continue
        protos[date_to_pos[date_value], :] = (proto / norm).astype("float32")

    return unique_dates.to_numpy(), protos


def _lagged_expanding_prototypes(
    protos_by_date: np.ndarray,
    lag_days: int = PROTOTYPE_LAG_DAYS,
    min_proto_days: int = MIN_PROTOTYPE_DAYS,
) -> np.ndarray:
    """Expanding mean of daily prototypes, lagged by lag_days."""
    n_dates = protos_by_date.shape[0]
    out = np.full_like(protos_by_date, np.nan, dtype="float32")
    csum = np.zeros(PATH_WINDOW, dtype="float64")
    count = 0
    cumulative_mean = np.full((n_dates, PATH_WINDOW), np.nan, dtype="float32")

    for i in range(n_dates):
        proto = protos_by_date[i]
        if np.isfinite(proto).all():
            csum += proto.astype("float64")
            count += 1
        if count >= min_proto_days:
            mean_proto = csum / float(count)
            norm = float(np.sqrt(np.sum(mean_proto * mean_proto)))
            if np.isfinite(norm) and norm > _EPS:
                cumulative_mean[i, :] = (mean_proto / norm).astype("float32")

    for i in range(n_dates):
        src = i - lag_days
        if src >= 0 and np.isfinite(cumulative_mean[src]).all():
            out[i, :] = cumulative_mean[src]
    return out


def _pre_breakout_causal_prototype_similarity_20d_full(data: pd.DataFrame) -> pd.Series:
    """
    Causal historical pre-breakout prototype similarity.

    Output is daily residualized cosine similarity, controlling for current 20d
    return rank and current 20d position rank.  This residualization prevents the
    factor from merely selecting stocks that have already risen sharply.
    """
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)

    current_paths, event_prepaths, event_ret20_sorted, sorted_idx, sorted_dates, _ = (
        _build_current_paths_and_event_prepaths(data)
    )
    event_rank = _daily_event_rank_weights(event_ret20_sorted, sorted_dates)
    unique_dates, daily_protos = _prototype_by_date(event_prepaths, event_rank, sorted_dates)
    lagged_protos = _lagged_expanding_prototypes(daily_protos)

    date_to_proto_pos = {d: i for i, d in enumerate(unique_dates)}
    raw_scores_sorted = np.full(len(sorted_idx), np.nan, dtype="float64")

    # Assign scores by date in sorted-row coordinates.
    position_of_label = pd.Series(np.arange(len(sorted_idx), dtype=np.int64), index=sorted_idx)
    for date_value, labels in sorted_dates.groupby(sorted_dates, sort=False).groups.items():
        proto_i = date_to_proto_pos.get(date_value)
        if proto_i is None:
            continue
        proto = lagged_protos[proto_i]
        if not np.isfinite(proto).all():
            continue
        row_positions = position_of_label.loc[labels].to_numpy(dtype=np.int64)
        paths = current_paths[row_positions]
        valid = np.isfinite(paths).all(axis=1)
        if not valid.any():
            continue
        raw_scores_sorted[row_positions[valid]] = paths[valid].astype("float64") @ proto.astype("float64")

    raw_scores = pd.Series(raw_scores_sorted, index=sorted_idx, dtype="float64").reindex(data.index)

    # Remove already-mature components in a threshold-free way.
    dates = pd.Series(data["date"].values, index=data.index)
    ret20 = _log_return(data, 20)
    pos20 = _close_position_extreme(data, 20, 10)
    ret20_rank = _daily_pct_rank(ret20, dates)
    pos20_rank = _daily_pct_rank(pos20, dates)
    residual_score = _daily_residual_zscore(raw_scores, [ret20_rank, pos20_rank], dates)
    return residual_score.astype("float64")


# ============================================================================
# FactorDefinition wrappers
# ============================================================================

def _position_reclaim_residual_20_60(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _position_reclaim_residual_20_60_full(data)


def _position_reclaim_residual_20_60_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    _: FactorContext,
) -> pd.Series:
    return _target_from_full(_position_reclaim_residual_20_60(data, _), target_index)


position_reclaim_residual_20_60 = register_factor(
    FactorDefinition(
        name="position_reclaim_residual_20_60",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=90,
        calculator=_position_reclaim_residual_20_60,
        calculation_scope="target_with_history_window",
        target_calculator=_position_reclaim_residual_20_60_target,
        description=(
            "[-inf,+inf] 20日位置对60日位置、20日涨幅、20日波动率的每日截面回归残差；"
            "刻画短期位置是否相对中期状态异常修复，无人工交易阈值"
        ),
    )
)


def _pre_breakout_causal_prototype_similarity_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _pre_breakout_causal_prototype_similarity_20d_full(data)


def _pre_breakout_causal_prototype_similarity_20d_target(
    data: pd.DataFrame,
    target_index: pd.Index,
    _: FactorContext,
) -> pd.Series:
    return _target_from_full(_pre_breakout_causal_prototype_similarity_20d(data, _), target_index)


pre_breakout_causal_prototype_similarity_20d = register_factor(
    FactorDefinition(
        name="pre_breakout_causal_prototype_similarity_20d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=260,
        calculator=_pre_breakout_causal_prototype_similarity_20d,
        calculation_scope="target_with_history_window",
        target_calculator=_pre_breakout_causal_prototype_similarity_20d_target,
        description=(
            "[-inf,+inf] 因果历史原型相似度：仅用T-5日前已发生的20日强势启动事件，"
            "提取其最大上涨段开始前20日路径形成历史原型；当前20日路径与原型相似度再对当前涨幅/位置残差化，"
            "避免未来标签泄露和已大涨成熟路径污染"
        ),
    )
)


__all__ = [
    "position_reclaim_residual_20_60",
    "pre_breakout_causal_prototype_similarity_20d",
]
