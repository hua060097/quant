"""
Causal adaptive structure factors, fixed production version.

This module is designed for the existing FactorDefinition/register_factor framework.
It contains two causal, label-free structure factors:

1) position_reclaim_residual_20_60
   Daily cross-sectional residual of short-term 20d range position after
   controlling for medium-term 60d position, trailing 20d return and trailing
   20d volatility. This is a threshold-free statistical residual version of
   position reclaim / short-term structure improvement.

2) pre_breakout_causal_prototype_similarity_20d
   Fixed causal historical pre-breakout prototype similarity.

   Important design constraints:
   - No future returns relative to scored date T are used.
   - No supervised label is used.
   - No full-sample clustering/template is used.
   - The prototype used at date T is built only from historical daily prototypes
     dated <= T - 5 trading days.
   - Historical prototypes are learned from *already-realized* trailing 20d
     strong-move events. For each historical event date E, the algorithm looks
     inside [E-19, E] for the strongest already-observed upward segment and
     extracts the 20d path immediately before that segment starts.
   - Current raw similarity is residualized cross-sectionally against current
     trailing 20d return rank and current 20d position rank, so the final score
     is not just a proxy for already-mature high-return/high-position stocks.

Compared with the previous causal prototype implementation, this version fixes
several production risks:
   - Uses explicit sorted-row integer coordinates, avoiding label/position
     confusion in grouped path arrays.
   - Produces a causal expanding prototype with a softer minimum history
     requirement and robust fallback standardization, avoiding all-NaN/all-zero
     outputs when prototype availability is sparse.
   - Keeps NaNs where history is insufficient, but does not silently convert
     the whole factor to a constant zero series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..universe import non_st_mask

_EPS = 1e-12
PATH_WINDOW = 20
EVENT_WINDOW = 20
PROTOTYPE_LAG_DAYS = 5
MIN_CROSS_SECTION_OBS = 50
MIN_PROTOTYPE_DAYS = 5
_Z_CLIP = 6.0


def _nan(index: pd.Index | range) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_num(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.Series(s, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _target_from_full(full: pd.Series, target_index: pd.Index) -> pd.Series:
    return full.loc[target_index].reset_index(drop=True).astype("float64")


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
    ordered_idx = _ordered_index(data)
    close = _safe_num(data.loc[ordered_idx, "close"])
    log_close = np.log(close.where(close > 0.0))
    prev = log_close.groupby(data.loc[ordered_idx, "stock_code"], sort=False).shift(periods)
    return (log_close - prev).reindex(data.index).astype("float64")


def _rolling_logret_std(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    if not {"stock_code", "date", "close"}.issubset(data.columns):
        return _nan(data.index)
    ordered_idx = _ordered_index(data)
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


def _daily_pct_rank(
    s: pd.Series,
    dates: pd.Series,
    eligible: pd.Series | None = None,
) -> pd.Series:
    if eligible is None:
        eligible = pd.Series(True, index=s.index)
    eligible = eligible.reindex(s.index).fillna(True)
    df = pd.DataFrame({"v": _safe_num(s).reindex(s.index).where(eligible).values, "date": dates.values}, index=s.index)
    return df.groupby("date", sort=False)["v"].rank(pct=True).where(eligible).astype("float64")


def _daily_zscore(
    s: pd.Series,
    dates: pd.Series,
    min_obs: int = MIN_CROSS_SECTION_OBS,
    eligible: pd.Series | None = None,
) -> pd.Series:
    if eligible is None:
        eligible = pd.Series(True, index=s.index)
    eligible = eligible.reindex(s.index).fillna(True)
    df = pd.DataFrame({"v": _safe_num(s).where(eligible).values, "date": dates.values}, index=s.index)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    for _, idx in df.groupby("date", sort=False).groups.items():
        sub = df.loc[idx, "v"].dropna()
        if len(sub) < min_obs:
            continue
        mean = float(sub.mean())
        std = float(sub.std(ddof=0))
        if not np.isfinite(std) or std <= _EPS:
            continue
        out.loc[sub.index] = ((sub - mean) / std).clip(-_Z_CLIP, _Z_CLIP)
    return out.astype("float64")


def _daily_residual_zscore(
    y: pd.Series,
    x_list: list[pd.Series],
    dates: pd.Series,
    min_obs: int = MIN_CROSS_SECTION_OBS,
    fallback_to_y_zscore: bool = True,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """
    Daily cross-sectional OLS residual, standardized within each date.

    If a date cannot be residualized because of insufficient valid X/Y rows or
    singular/constant residuals, optionally falls back to daily z-score of y.
    This prevents a useful raw signal from becoming an all-NaN/all-zero factor
    solely due to one overly strict residualization step.
    """
    if len(y) == 0:
        return pd.Series(dtype="float64")

    if eligible is None:
        eligible = pd.Series(True, index=y.index)
    eligible = eligible.reindex(y.index).fillna(True)
    work = pd.DataFrame({"y": _safe_num(y).where(eligible).values, "date": dates.values}, index=y.index)
    for i, x in enumerate(x_list):
        work[f"x{i}"] = _safe_num(x).reindex(y.index).where(eligible).values

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
        out.loc[sub.index] = ((resid - float(np.nanmean(resid))) / sd).clip(-_Z_CLIP, _Z_CLIP)

    if fallback_to_y_zscore:
        missing_dates = out.isna()
        if missing_dates.any():
            y_z = _daily_zscore(y, dates, min_obs=min_obs, eligible=eligible)
            out = out.where(out.notna(), y_z)

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
    eligible = non_st_mask(data)
    pos20 = _close_position_extreme(data, 20, 10)
    pos60 = _close_position_extreme(data, 60, 30)
    ret20 = _log_return(data, 20)
    vol20 = _rolling_logret_std(data, 20, 10)

    y = _daily_pct_rank(pos20, dates, eligible=eligible)
    x_pos60 = _daily_pct_rank(pos60, dates, eligible=eligible)
    x_ret20 = _daily_pct_rank(ret20, dates, eligible=eligible)
    x_vol20 = _daily_pct_rank(vol20, dates, eligible=eligible)
    return _daily_residual_zscore(
        y,
        [x_pos60, x_ret20, x_vol20],
        dates,
        fallback_to_y_zscore=True,
        eligible=eligible,
    )


def _normalize_path_from_log_segment(segment: np.ndarray) -> np.ndarray | None:
    """Convert a log-close segment into a centered unit path vector."""
    if segment.shape[0] != PATH_WINDOW or not np.all(np.isfinite(segment)):
        return None
    path = segment - segment[0]
    path = path - float(np.mean(path))
    norm = float(np.sqrt(np.sum(path * path)))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return (path / norm).astype("float32")


def _sorted_market_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    ordered_idx = data.sort_values(["stock_code", "date"], kind="mergesort").index
    sorted_df = data.loc[ordered_idx, ["stock_code", "date", "close"]].copy()
    sorted_df["_orig_index"] = ordered_idx.to_numpy()
    sorted_df["_spos"] = np.arange(len(sorted_df), dtype=np.int64)
    return sorted_df, ordered_idx


def _build_paths_and_event_prepaths(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Index]:
    """
    Build current paths, event pre-breakout paths and realized trailing returns.

    Arrays are in sorted-row coordinates, never original-index coordinates. This
    avoids index-label vs integer-position confusion.
    """
    sorted_df, ordered_idx = _sorted_market_frame(data)
    n = len(sorted_df)
    current_paths = np.full((n, PATH_WINDOW), np.nan, dtype="float32")
    event_prepaths = np.full((n, PATH_WINDOW), np.nan, dtype="float32")
    event_ret20 = np.full(n, np.nan, dtype="float64")

    close = _safe_num(sorted_df["close"]).to_numpy(dtype="float64")
    logs = np.full(n, np.nan, dtype="float64")
    valid_close = close > 0.0
    logs[valid_close] = np.log(close[valid_close])

    dates_sorted = pd.Series(sorted_df["date"].values, index=np.arange(n, dtype=np.int64))

    groups = sorted_df.groupby("stock_code", sort=False)["_spos"].apply(lambda x: x.to_numpy(dtype=np.int64))
    for spos in groups.values:
        m = len(spos)
        if m < PATH_WINDOW:
            continue
        g_logs = logs[spos]

        for j in range(m):
            abs_pos = spos[j]

            # Current 20d path ending at j, causal for scored row j.
            if j >= PATH_WINDOW - 1:
                cur_seg = g_logs[j - PATH_WINDOW + 1 : j + 1]
                cur_path = _normalize_path_from_log_segment(cur_seg)
                if cur_path is not None:
                    current_paths[abs_pos, :] = cur_path

            # Already-realized trailing 20d return ending at j.
            if j >= PATH_WINDOW:
                if np.isfinite(g_logs[j]) and np.isfinite(g_logs[j - PATH_WINDOW]):
                    event_ret20[abs_pos] = g_logs[j] - g_logs[j - PATH_WINDOW]

            # Need trailing event window plus a complete pre-start 20d path.
            if j < EVENT_WINDOW - 1 + PATH_WINDOW:
                continue
            event_window = g_logs[j - EVENT_WINDOW + 1 : j + 1]
            if not np.all(np.isfinite(event_window)):
                continue

            # Strongest upward segment inside the already-observed 20d event window.
            # O(20^2) is tiny and more precise than only using min-before-max.
            best_gain = -np.inf
            best_start_rel = -1
            for start_rel in range(EVENT_WINDOW - 1):
                future_slice = event_window[start_rel + 1 :]
                if future_slice.size == 0:
                    continue
                peak_rel_after = int(np.argmax(future_slice)) + start_rel + 1
                gain = float(event_window[peak_rel_after] - event_window[start_rel])
                if gain > best_gain:
                    best_gain = gain
                    best_start_rel = start_rel

            if not np.isfinite(best_gain) or best_gain <= 0.0 or best_start_rel < 0:
                continue
            start_abs_local = j - EVENT_WINDOW + 1 + best_start_rel
            pre_start = start_abs_local - PATH_WINDOW
            pre_end = start_abs_local
            if pre_start < 0:
                continue
            pre_seg = g_logs[pre_start:pre_end]
            pre_path = _normalize_path_from_log_segment(pre_seg)
            if pre_path is not None:
                event_prepaths[abs_pos, :] = pre_path

    return current_paths, event_prepaths, event_ret20, sorted_df["_orig_index"].to_numpy(), dates_sorted, ordered_idx


def _daily_event_rank_weights(
    event_ret20: np.ndarray,
    dates_sorted: pd.Series,
    eligible_sorted: pd.Series | None = None,
) -> pd.Series:
    s = pd.Series(event_ret20, index=dates_sorted.index, dtype="float64")
    return _daily_pct_rank(s, dates_sorted, eligible=eligible_sorted)


def _daily_contrast_prototypes(
    event_prepaths: np.ndarray,
    event_rank: pd.Series,
    dates_sorted: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one contrast prototype for every trading date.

    Weight = daily percentile rank of already-realized trailing 20d return -
    daily cross-sectional mean rank. This avoids hard top-percentile thresholds
    while emphasizing paths that preceded the strongest already-observed movers.
    """
    date_values = pd.Series(dates_sorted.values).drop_duplicates().to_numpy()
    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(date_values)}
    protos = np.full((len(date_values), PATH_WINDOW), np.nan, dtype="float32")

    ranks = event_rank.reindex(dates_sorted.index).to_numpy(dtype="float64")
    for date_value, idx in dates_sorted.groupby(dates_sorted, sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=np.int64)
        paths = event_prepaths[positions]
        r = ranks[positions]
        valid = np.isfinite(r) & np.isfinite(paths).all(axis=1)
        if int(valid.sum()) < MIN_CROSS_SECTION_OBS:
            continue
        rv = r[valid]
        weights = rv - float(np.nanmean(rv))
        denom = float(np.sum(np.abs(weights)))
        if not np.isfinite(denom) or denom <= _EPS:
            continue
        proto = (paths[valid].astype("float64") * weights[:, None]).sum(axis=0) / denom
        norm = float(np.sqrt(np.sum(proto * proto)))
        if not np.isfinite(norm) or norm <= _EPS:
            continue
        protos[date_to_pos[pd.Timestamp(date_value)], :] = (proto / norm).astype("float32")

    return date_values, protos


def _lagged_expanding_prototypes(
    daily_protos: np.ndarray,
    lag_days: int = PROTOTYPE_LAG_DAYS,
    min_proto_days: int = MIN_PROTOTYPE_DAYS,
) -> np.ndarray:
    """Expanding normalized average of valid daily prototypes, lagged by lag_days."""
    n_dates = daily_protos.shape[0]
    cumulative = np.full((n_dates, PATH_WINDOW), np.nan, dtype="float32")
    out = np.full((n_dates, PATH_WINDOW), np.nan, dtype="float32")

    csum = np.zeros(PATH_WINDOW, dtype="float64")
    count = 0
    for i in range(n_dates):
        proto = daily_protos[i]
        if np.isfinite(proto).all():
            csum += proto.astype("float64")
            count += 1
        if count >= min_proto_days:
            mean_proto = csum / float(count)
            norm = float(np.sqrt(np.sum(mean_proto * mean_proto)))
            if np.isfinite(norm) and norm > _EPS:
                cumulative[i, :] = (mean_proto / norm).astype("float32")

    for i in range(n_dates):
        src = i - lag_days
        if src >= 0 and np.isfinite(cumulative[src]).all():
            out[i, :] = cumulative[src]
    return out


def _pre_breakout_causal_prototype_similarity_20d_full(data: pd.DataFrame) -> pd.Series:
    """
    Fixed causal historical pre-breakout prototype similarity.

    The raw score is cosine similarity between the current 20d path and a T-5
    lagged expanding historical prototype built from pre-breakout paths before
    already-realized strong 20d moves. Final output is residualized against
    current ret20 rank and pos20 rank to reduce already-overextended positives.
    """
    if not {"stock_code", "date", "high", "low", "close"}.issubset(data.columns):
        return _nan(data.index)

    current_paths, event_prepaths, event_ret20, orig_index_sorted, dates_sorted, _ = _build_paths_and_event_prepaths(data)
    eligible_sorted = pd.Series(
        non_st_mask(data).reindex(orig_index_sorted).fillna(True).to_numpy(),
        index=dates_sorted.index,
    )
    event_rank = _daily_event_rank_weights(event_ret20, dates_sorted, eligible_sorted=eligible_sorted)
    date_values, daily_protos = _daily_contrast_prototypes(event_prepaths, event_rank, dates_sorted)
    lagged_protos = _lagged_expanding_prototypes(daily_protos)

    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(date_values)}
    raw_sorted = np.full(len(orig_index_sorted), np.nan, dtype="float64")

    # Assign raw cosine similarities in sorted-row coordinates.
    for date_value, idx in dates_sorted.groupby(dates_sorted, sort=False).groups.items():
        date_pos = date_to_pos.get(pd.Timestamp(date_value))
        if date_pos is None:
            continue
        proto = lagged_protos[date_pos]
        if not np.isfinite(proto).all():
            continue
        positions = np.asarray(list(idx), dtype=np.int64)
        paths = current_paths[positions]
        valid = np.isfinite(paths).all(axis=1)
        if not valid.any():
            continue
        raw_sorted[positions[valid]] = paths[valid].astype("float64") @ proto.astype("float64")

    raw = pd.Series(raw_sorted, index=orig_index_sorted, dtype="float64").reindex(data.index)

    dates = pd.Series(data["date"].values, index=data.index)
    eligible = non_st_mask(data)
    ret20 = _log_return(data, 20)
    pos20 = _close_position_extreme(data, 20, 10)
    ret20_rank = _daily_pct_rank(ret20, dates, eligible=eligible)
    pos20_rank = _daily_pct_rank(pos20, dates, eligible=eligible)

    # Residualization is useful but should not be able to collapse the entire
    # factor into NaNs/zeros. Fallback keeps the causal raw signal if residual
    # regression cannot be estimated on a particular date.
    out = _daily_residual_zscore(
        raw,
        [ret20_rank, pos20_rank],
        dates,
        fallback_to_y_zscore=True,
        eligible=eligible,
    )
    return out.astype("float64")


# ============================================================================
# FactorDefinition wrappers
# ============================================================================

def _position_reclaim_residual_20_60(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _position_reclaim_residual_20_60_full(data)


def _position_reclaim_residual_20_60_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
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
            "20日位置对60日位置、20日涨幅、20日波动率的每日截面回归残差；"
            "刻画短期位置是否相对中期状态异常修复，无人工交易阈值"
        ),
    )
)


def _pre_breakout_causal_prototype_similarity_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    return _pre_breakout_causal_prototype_similarity_20d_full(data)


def _pre_breakout_causal_prototype_similarity_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
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
            "因果历史原型相似度fixed：仅用T-5日前已发生的20日强势启动事件，"
            "提取最大上涨段开始前20日路径形成历史原型；当前20日路径与原型相似度再对当前涨幅/位置残差化，"
            "避免未来标签泄露和已大涨成熟路径污染"
        ),
    )
)


__all__ = [
    "position_reclaim_residual_20_60",
    "pre_breakout_causal_prototype_similarity_20d",
]
