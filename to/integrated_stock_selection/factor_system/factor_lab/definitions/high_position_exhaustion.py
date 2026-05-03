"""
高位追涨衰竭因子模块（内置归一化版）。

设计目标：
- 刻画“高位是否出现衰竭、放量滞涨、冲高回落、承接破坏”，而不是简单惩罚高位/超买。
- 适配 FactorDefinition / register_factor 架构，与现有因子文件用法保持一致。
- 连续/交互/加速度类因子在文件内部完成稳定化处理，后续训练前不需要再额外 z-score。
- 所有 rolling/shift 均为同股票历史窗口计算，不使用未来数据。

归一化约定：
1. bounded 状态类：输出天然有界值，例如 [0, 1] 或 [-1, 0]。
2. count 类：输出比例，例如近5日次数 / 5、近10日次数 / 10。
3. continuous / interaction / acceleration 类：先做合理 clip，再按交易日横截面 robust z-score，并 clip 到 [-6, 6]。
4. binary 类：保持 0/1。

建议在 definitions/__init__.py 中导入本模块暴露的因子后再参与生成。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

_EPS = 1e-9
_Z_CLIP = 6.0


def _zero(index: pd.Index | range) -> pd.Series:
    return pd.Series(0.0, index=index, dtype="float64")


def _safe_num(s: pd.Series | np.ndarray | float, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return out.astype("float64")
    return pd.Series(s, index=index, dtype="float64").replace([np.inf, -np.inf], np.nan)


def _safe_div(numerator, denominator) -> pd.Series:
    out = numerator / (denominator + _EPS)
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.Series(out).replace([np.inf, -np.inf], np.nan).astype("float64")


def _clip_series(s: pd.Series, lower: float = -10.0, upper: float = 10.0) -> pd.Series:
    return _safe_num(s).clip(lower, upper).astype("float64")


def _daily_robust_zscore(series: pd.Series, date_series: pd.Series, clip: float = _Z_CLIP) -> pd.Series:
    """
    按交易日做横截面 robust z-score。

    使用 median / MAD，MAD 过小时回退到 mean / std。
    输出已经 clip 到 [-clip, clip]，并将 NaN/Inf 填 0。
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype="float64")

    val = _safe_num(series).astype("float64")
    dates = pd.Series(date_series.values, index=series.index)
    df = pd.DataFrame({"val": val.values, "date": dates.values}, index=series.index)

    med = df.groupby("date", sort=False)["val"].transform("median")
    abs_diff = (df["val"] - med).abs()
    mad = pd.DataFrame({"diff": abs_diff.values, "date": dates.values}, index=series.index).groupby(
        "date", sort=False
    )["diff"].transform("median")

    z = (df["val"] - med) / (mad * 1.4826 + _EPS)

    weak_mad = mad < 1e-8
    if weak_mad.any():
        mean = df.groupby("date", sort=False)["val"].transform("mean")
        std = df.groupby("date", sort=False)["val"].transform("std")
        z_fallback = (df["val"] - mean) / (std + _EPS)
        z = z.where(~weak_mad, z_fallback)

    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip).astype("float64")


def _cs_z(data: pd.DataFrame, series: pd.Series, pre_clip: tuple[float, float] | None = None) -> pd.Series:
    """对一个原始因子先可选 clip，再按 date 做横截面 robust z-score。"""
    s = _safe_num(series).reindex(data.index)
    if pre_clip is not None:
        s = s.clip(pre_clip[0], pre_clip[1])
    if "date" not in data.columns:
        return s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-_Z_CLIP, _Z_CLIP).astype("float64")
    return _daily_robust_zscore(s, data["date"], clip=_Z_CLIP).astype("float64")


def _target_from_full(full: pd.Series, target_index: pd.Index) -> pd.Series:
    return full.loc[target_index].reset_index(drop=True).astype("float64")


def _rolling_groupby(
    data: pd.DataFrame,
    value: pd.Series,
    window: int,
    min_periods: int,
    agg: str,
) -> pd.Series:
    """对外部构造的 Series 按 stock_code 做 rolling 聚合。"""
    tmp = pd.Series(value.values, index=data.index, dtype="float64")
    grouped = tmp.groupby(data["stock_code"], sort=False)
    rolled = grouped.rolling(window, min_periods=min_periods)
    if agg == "mean":
        out = rolled.mean()
    elif agg == "max":
        out = rolled.max()
    elif agg == "min":
        out = rolled.min()
    elif agg == "sum":
        out = rolled.sum()
    elif agg == "std":
        out = rolled.std()
    else:
        raise ValueError(f"Unsupported rolling agg: {agg}")
    return out.reset_index(level=0, drop=True).astype("float64")


def _rolling_col(
    data: pd.DataFrame,
    col: str,
    window: int,
    min_periods: int,
    agg: str,
) -> pd.Series:
    if col not in data.columns:
        return _zero(data.index)
    value = _safe_num(data[col]).astype("float64")
    return _rolling_groupby(data, value, window, min_periods, agg)


def _pct_change_by_stock(data: pd.DataFrame, periods: int = 1) -> pd.Series:
    if "close" not in data.columns:
        return _zero(data.index)
    close = _safe_num(data["close"]).astype("float64")
    prev = close.groupby(data["stock_code"], sort=False).shift(periods)
    return _safe_div(close - prev, prev).astype("float64")


def _return_n(data: pd.DataFrame, n: int) -> pd.Series:
    return _pct_change_by_stock(data, n).fillna(0.0).clip(-1.0, 5.0).astype("float64")


def _close_position_raw(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """收盘价在 N 日高低区间中的位置，天然归一化到 [0, 1]。"""
    if not {"high", "low", "close"}.issubset(data.columns):
        return _zero(data.index)
    high_n = _rolling_col(data, "high", window, min_periods, "max")
    low_n = _rolling_col(data, "low", window, min_periods, "min")
    close = _safe_num(data["close"]).astype("float64")
    pos = _safe_div(close - low_n, high_n - low_n)
    return pos.clip(0.0, 1.0).fillna(0.5).astype("float64")


def _upper_shadow_ratio_raw(data: pd.DataFrame) -> pd.Series:
    """上影线 / close，天然非负比例，做上限截断。"""
    if not {"open", "high", "close"}.issubset(data.columns):
        return _zero(data.index)
    open_ = _safe_num(data["open"]).astype("float64")
    high = _safe_num(data["high"]).astype("float64")
    close = _safe_num(data["close"]).astype("float64")
    upper = high - np.maximum(open_, close)
    ratio = _safe_div(pd.Series(upper, index=data.index).clip(lower=0.0), close)
    return ratio.clip(0.0, 0.5).fillna(0.0).astype("float64")


def _close_to_high_raw(data: pd.DataFrame) -> pd.Series:
    """close / high - 1，范围通常为 [-1, 0]。"""
    if not {"high", "close"}.issubset(data.columns):
        return _zero(data.index)
    high = _safe_num(data["high"]).astype("float64")
    close = _safe_num(data["close"]).astype("float64")
    return _clip_series(_safe_div(close, high) - 1.0, -1.0, 0.0).fillna(0.0)


def _high_close_reversal_raw(data: pd.DataFrame) -> pd.Series:
    """(high - close) / (high - low)，天然 [0,1]，越高越冲高回落。"""
    if not {"high", "low", "close"}.issubset(data.columns):
        return _zero(data.index)
    high = _safe_num(data["high"]).astype("float64")
    low = _safe_num(data["low"]).astype("float64")
    close = _safe_num(data["close"]).astype("float64")
    rev = _safe_div(high - close, high - low)
    return rev.clip(0.0, 1.0).fillna(0.0).astype("float64")


def _turnover_source(data: pd.DataFrame) -> pd.Series:
    """
    优先使用换手率，其次使用成交量/成交额。

    你的不同数据源字段可能不同，如有 turnover_rate_f 等字段，可在这里追加。
    """
    for col in ("turnover_rate", "turnover_rate_f", "turnover", "vol", "volume", "amount"):
        if col in data.columns:
            return _safe_num(data[col]).astype("float64")
    return _zero(data.index)


def _rolling_z_by_stock(data: pd.DataFrame, value: pd.Series, window: int, min_periods: int) -> pd.Series:
    """个股自身时间序列 rolling z，先识别“相对自己历史是否异常”。"""
    mean = _rolling_groupby(data, value, window, min_periods, "mean")
    std = _rolling_groupby(data, value, window, min_periods, "std")
    z = _safe_div(value - mean, std)
    return z.clip(-6.0, 6.0).fillna(0.0).astype("float64")


def _turnover_z_20_ts_raw(data: pd.DataFrame) -> pd.Series:
    value = _turnover_source(data)
    return _rolling_z_by_stock(data, value, window=20, min_periods=10)


def _turnover_z_20_norm_full(data: pd.DataFrame) -> pd.Series:
    """成交异常：先时间序列 z，再按日横截面 robust z，最终可直接训练使用。"""
    return _cs_z(data, _turnover_z_20_ts_raw(data), pre_clip=(-6.0, 6.0))


def _high_volume_no_gain_1d_raw(data: pd.DataFrame) -> pd.Series:
    """放量但涨幅不足 2%，原始强度。"""
    turnover_z = _turnover_z_20_ts_raw(data).clip(lower=0.0)
    ret_1d = _pct_change_by_stock(data, 1).fillna(0.0)
    no_gain = turnover_z * np.maximum(0.0, 0.02 - ret_1d)
    return pd.Series(no_gain, index=data.index).clip(0.0, 2.0).astype("float64")


def _high_volume_no_gain_3d_raw(data: pd.DataFrame) -> pd.Series:
    one = _high_volume_no_gain_1d_raw(data)
    return _rolling_groupby(data, one, 3, 1, "mean").clip(0.0, 2.0).fillna(0.0).astype("float64")


def _high_volume_no_gain_3d_norm_full(data: pd.DataFrame) -> pd.Series:
    return _cs_z(data, _high_volume_no_gain_3d_raw(data), pre_clip=(0.0, 2.0))


def _pullback_depth_from_high_raw(data: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """close / N日最高价 - 1，通常 [-1, 0]，已天然尺度化。"""
    if not {"high", "close"}.issubset(data.columns):
        return _zero(data.index)
    high_n = _rolling_col(data, "high", window, min_periods, "max")
    close = _safe_num(data["close"]).astype("float64")
    return _clip_series(_safe_div(close, high_n) - 1.0, -1.0, 0.0).fillna(0.0)


def _gap_up_flag(data: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    if not {"open", "close"}.issubset(data.columns):
        return _zero(data.index)
    open_ = _safe_num(data["open"]).astype("float64")
    prev_close = _safe_num(data["close"]).astype("float64").groupby(data["stock_code"], sort=False).shift(1)
    gap = _safe_div(open_ - prev_close, prev_close)
    return (gap > threshold).astype("float64")


def _upper_shadow_ratio_3d_mean_raw(data: pd.DataFrame) -> pd.Series:
    upper = _upper_shadow_ratio_raw(data)
    return _rolling_groupby(data, upper, 3, 1, "mean").clip(0.0, 0.5).fillna(0.0).astype("float64")


def _close_to_high_3d_mean_raw(data: pd.DataFrame) -> pd.Series:
    cth = _close_to_high_raw(data)
    return _rolling_groupby(data, cth, 3, 1, "mean").clip(-1.0, 0.0).fillna(0.0).astype("float64")


def _ret_3d_vs_10d_accel_raw(data: pd.DataFrame) -> pd.Series:
    ret3 = _return_n(data, 3)
    ret10 = _return_n(data, 10)
    return _clip_series(ret3 - ret10 / 3.0, -1.0, 2.0).fillna(0.0)


def _ret_5d_vs_20d_accel_raw(data: pd.DataFrame) -> pd.Series:
    ret5 = _return_n(data, 5)
    ret20 = _return_n(data, 20)
    return _clip_series(ret5 - ret20 / 4.0, -1.0, 2.0).fillna(0.0)


def _momentum_peak_decay_3d_raw(data: pd.DataFrame) -> pd.Series:
    ret1 = _return_n(data, 1)
    ret3 = _return_n(data, 3)
    return _clip_series(ret1 - ret3 / 3.0, -1.0, 1.0).fillna(0.0)


def _consecutive_up_days_5d_norm_full(data: pd.DataFrame) -> pd.Series:
    ret_1d = _pct_change_by_stock(data, 1).fillna(0.0)
    up = (ret_1d > 0).astype("float64")
    # 归一化到 [0,1]
    return (_rolling_groupby(data, up, 5, 1, "sum") / 5.0).clip(0.0, 1.0).astype("float64")


def _large_up_days_10d_norm_full(data: pd.DataFrame) -> pd.Series:
    ret_1d = _pct_change_by_stock(data, 1).fillna(0.0)
    large = (ret_1d > 0.05).astype("float64")
    # 归一化到 [0,1]
    return (_rolling_groupby(data, large, 10, 1, "sum") / 10.0).clip(0.0, 1.0).astype("float64")


def _gap_up_count_5d_norm_full(data: pd.DataFrame) -> pd.Series:
    gap = _gap_up_flag(data, threshold=0.02)
    # 归一化到 [0,1]
    return (_rolling_groupby(data, gap, 5, 1, "sum") / 5.0).clip(0.0, 1.0).astype("float64")


def _close_below_ma5_after_runup_full(data: pd.DataFrame) -> pd.Series:
    if "close" not in data.columns:
        return _zero(data.index)
    close = _safe_num(data["close"]).astype("float64")
    ma5 = _rolling_col(data, "close", 5, 3, "mean")
    ret_10d = _return_n(data, 10)
    out = ((ret_10d > 0.10) & (close < ma5)).astype("float64")
    return pd.Series(out, index=data.index).astype("float64")


def _low_break_prev_low_after_runup_full(data: pd.DataFrame) -> pd.Series:
    if not {"low", "close"}.issubset(data.columns):
        return _zero(data.index)
    low = _safe_num(data["low"]).astype("float64")
    prev_low = low.groupby(data["stock_code"], sort=False).shift(1)
    ret_10d = _return_n(data, 10)
    out = ((ret_10d > 0.10) & (low < prev_low)).astype("float64")
    return pd.Series(out, index=data.index).astype("float64")


# ==============================================================================
# 第一批：高位位置 / 长上影 / 放量滞涨 / 高位交互
# ==============================================================================
def _close_position_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 已天然归一化到 [0,1]
    return _close_position_raw(data, 20, 10)


def _close_position_20d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_position_20d(data, _), target_index)


close_position_20d = register_factor(
    FactorDefinition(
        name="close_position_20d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=30,
        calculator=_close_position_20d,
        calculation_scope="target_with_history_window",
        target_calculator=_close_position_20d_target,
        description="[0,1] 收盘价在近20日高低区间中的位置，越接近1表示越靠近短线高位",
    )
)


def _close_position_60d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 已天然归一化到 [0,1]
    return _close_position_raw(data, 60, 30)


def _close_position_60d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_position_60d(data, _), target_index)


close_position_60d = register_factor(
    FactorDefinition(
        name="close_position_60d",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=80,
        calculator=_close_position_60d,
        calculation_scope="target_with_history_window",
        target_calculator=_close_position_60d_target,
        description="[0,1] 收盘价在近60日高低区间中的位置，刻画中短周期高位程度",
    )
)


def _upper_shadow_ratio_3d_mean(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 已为比例值，clip 到 [0,0.5]
    return _upper_shadow_ratio_3d_mean_raw(data)


def _upper_shadow_ratio_3d_mean_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_upper_shadow_ratio_3d_mean(data, _), target_index)


upper_shadow_ratio_3d_mean = register_factor(
    FactorDefinition(
        name="upper_shadow_ratio_3d_mean",
        required_columns={"stock_code", "date", "open", "high", "close"},
        lookback_days=10,
        calculator=_upper_shadow_ratio_3d_mean,
        calculation_scope="target_with_history_window",
        target_calculator=_upper_shadow_ratio_3d_mean_target,
        description="[0,0.5] 近3日平均上影线比例，越高说明冲高回落/上方抛压越强",
    )
)


def _close_to_high_3d_mean(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 已为比例值，范围 [-1,0]
    return _close_to_high_3d_mean_raw(data)


def _close_to_high_3d_mean_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_to_high_3d_mean(data, _), target_index)


close_to_high_3d_mean = register_factor(
    FactorDefinition(
        name="close_to_high_3d_mean",
        required_columns={"stock_code", "date", "high", "close"},
        lookback_days=10,
        calculator=_close_to_high_3d_mean,
        calculation_scope="target_with_history_window",
        target_calculator=_close_to_high_3d_mean_target,
        description="[-1,0] 近3日收盘价相对最高价的平均距离，越负说明越常收在日内高点下方",
    )
)


def _turnover_z_20(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 个股时间序列 z 后，再按日横截面 robust z；输出 [-6,6]
    return _turnover_z_20_norm_full(data)


def _turnover_z_20_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_turnover_z_20(data, _), target_index)


turnover_z_20 = register_factor(
    FactorDefinition(
        name="turnover_z_20",
        required_columns={"stock_code", "date"},
        lookback_days=30,
        calculator=_turnover_z_20,
        calculation_scope="target_with_history_window",
        target_calculator=_turnover_z_20_target,
        description="[-6,6] 成交/换手异常：先按个股近20日做时间序列z，再按日横截面robust z",
    )
)


def _high_volume_no_gain_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 原始放量滞涨强度再按日横截面 robust z；输出 [-6,6]
    return _high_volume_no_gain_3d_norm_full(data)


def _high_volume_no_gain_3d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_high_volume_no_gain_3d(data, _), target_index)


high_volume_no_gain_3d = register_factor(
    FactorDefinition(
        name="high_volume_no_gain_3d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=35,
        calculator=_high_volume_no_gain_3d,
        calculation_scope="target_with_history_window",
        target_calculator=_high_volume_no_gain_3d_target,
        description="[-6,6] 近3日放量滞涨强度，已按日横截面robust z标准化",
    )
)


def _pullback_depth_from_10d_high(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 已为比例值，范围 [-1,0]
    return _pullback_depth_from_high_raw(data, 10, 5)


def _pullback_depth_from_10d_high_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_pullback_depth_from_10d_high(data, _), target_index)


pullback_depth_from_10d_high = register_factor(
    FactorDefinition(
        name="pullback_depth_from_10d_high",
        required_columns={"stock_code", "date", "high", "close"},
        lookback_days=20,
        calculator=_pullback_depth_from_10d_high,
        calculation_scope="target_with_history_window",
        target_calculator=_pullback_depth_from_10d_high_target,
        description="[-1,0] 收盘价相对近10日最高价的回撤幅度，越负说明短线高位承接越弱",
    )
)


def _high_pos_upper_shadow_raw(data: pd.DataFrame) -> pd.Series:
    return (_close_position_raw(data, 20, 10) * _upper_shadow_ratio_3d_mean_raw(data)).clip(0.0, 1.0)


def _high_pos_upper_shadow(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 交互项统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _high_pos_upper_shadow_raw(data), pre_clip=(0.0, 1.0))


def _high_pos_upper_shadow_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_high_pos_upper_shadow(data, _), target_index)


high_pos_upper_shadow = register_factor(
    FactorDefinition(
        name="high_pos_upper_shadow",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=30,
        calculator=_high_pos_upper_shadow,
        calculation_scope="target_with_history_window",
        target_calculator=_high_pos_upper_shadow_target,
        description="[-6,6] 高位位置×近3日上影线，已按日横截面robust z标准化",
    )
)


def _high_pos_volume_no_gain_raw(data: pd.DataFrame) -> pd.Series:
    return (_close_position_raw(data, 20, 10) * _high_volume_no_gain_3d_raw(data)).clip(0.0, 2.0)


def _high_pos_volume_no_gain(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 交互项统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _high_pos_volume_no_gain_raw(data), pre_clip=(0.0, 2.0))


def _high_pos_volume_no_gain_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_high_pos_volume_no_gain(data, _), target_index)


high_pos_volume_no_gain = register_factor(
    FactorDefinition(
        name="high_pos_volume_no_gain",
        required_columns={"stock_code", "date", "high", "low", "close"},
        lookback_days=35,
        calculator=_high_pos_volume_no_gain,
        calculation_scope="target_with_history_window",
        target_calculator=_high_pos_volume_no_gain_target,
        description="[-6,6] 高位位置×放量滞涨，已按日横截面robust z标准化",
    )
)


def _high_pos_reversal_volume_raw(data: pd.DataFrame) -> pd.Series:
    pos = _close_position_raw(data, 20, 10)
    upper = _upper_shadow_ratio_3d_mean_raw(data)
    vol_z = _turnover_z_20_ts_raw(data).clip(lower=0.0)
    return (pos * upper * vol_z).clip(0.0, 6.0).astype("float64")


def _high_pos_reversal_volume(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 交互项统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _high_pos_reversal_volume_raw(data), pre_clip=(0.0, 6.0))


def _high_pos_reversal_volume_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_high_pos_reversal_volume(data, _), target_index)


high_pos_reversal_volume = register_factor(
    FactorDefinition(
        name="high_pos_reversal_volume",
        required_columns={"stock_code", "date", "open", "high", "low", "close"},
        lookback_days=35,
        calculator=_high_pos_reversal_volume,
        calculation_scope="target_with_history_window",
        target_calculator=_high_pos_reversal_volume_target,
        description="[-6,6] 高位×上影线×放量，已按日横截面robust z标准化",
    )
)


# ==============================================================================
# 第二批：加速过热 / 跳空追高 / 高位承接破坏
# ==============================================================================
def _ret_3d_vs_10d_accel(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 加速度类统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _ret_3d_vs_10d_accel_raw(data), pre_clip=(-1.0, 2.0))


def _ret_3d_vs_10d_accel_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_ret_3d_vs_10d_accel(data, _), target_index)


ret_3d_vs_10d_accel = register_factor(
    FactorDefinition(
        name="ret_3d_vs_10d_accel",
        required_columns={"stock_code", "date", "close"},
        lookback_days=15,
        calculator=_ret_3d_vs_10d_accel,
        calculation_scope="target_with_history_window",
        target_calculator=_ret_3d_vs_10d_accel_target,
        description="[-6,6] 3日涨幅相对10日平均速度的加速项，已按日横截面robust z标准化",
    )
)


def _ret_5d_vs_20d_accel(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 加速度类统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _ret_5d_vs_20d_accel_raw(data), pre_clip=(-1.0, 2.0))


def _ret_5d_vs_20d_accel_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_ret_5d_vs_20d_accel(data, _), target_index)


ret_5d_vs_20d_accel = register_factor(
    FactorDefinition(
        name="ret_5d_vs_20d_accel",
        required_columns={"stock_code", "date", "close"},
        lookback_days=25,
        calculator=_ret_5d_vs_20d_accel,
        calculation_scope="target_with_history_window",
        target_calculator=_ret_5d_vs_20d_accel_target,
        description="[-6,6] 5日涨幅相对20日平均速度的加速项，已按日横截面robust z标准化",
    )
)


def _momentum_peak_decay_3d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 动量变化类统一按日横截面 robust z；输出 [-6,6]
    return _cs_z(data, _momentum_peak_decay_3d_raw(data), pre_clip=(-1.0, 1.0))


def _momentum_peak_decay_3d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_momentum_peak_decay_3d(data, _), target_index)


momentum_peak_decay_3d = register_factor(
    FactorDefinition(
        name="momentum_peak_decay_3d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=10,
        calculator=_momentum_peak_decay_3d,
        calculation_scope="target_with_history_window",
        target_calculator=_momentum_peak_decay_3d_target,
        description="[-6,6] 1日动量相对3日平均动量的变化，已按日横截面robust z标准化",
    )
)


def _consecutive_up_days_5d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 次数类归一化为 [0,1]
    return _consecutive_up_days_5d_norm_full(data)


def _consecutive_up_days_5d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_consecutive_up_days_5d(data, _), target_index)


consecutive_up_days_5d = register_factor(
    FactorDefinition(
        name="consecutive_up_days_5d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=10,
        calculator=_consecutive_up_days_5d,
        calculation_scope="target_with_history_window",
        target_calculator=_consecutive_up_days_5d_target,
        description="[0,1] 近5日上涨天数占比，刻画短线连续追高拥挤程度",
    )
)


def _large_up_days_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 次数类归一化为 [0,1]
    return _large_up_days_10d_norm_full(data)


def _large_up_days_10d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_large_up_days_10d(data, _), target_index)


large_up_days_10d = register_factor(
    FactorDefinition(
        name="large_up_days_10d",
        required_columns={"stock_code", "date", "close"},
        lookback_days=15,
        calculator=_large_up_days_10d,
        calculation_scope="target_with_history_window",
        target_calculator=_large_up_days_10d_target,
        description="[0,1] 近10日大涨天数占比，默认统计日涨幅超过5%的次数",
    )
)


def _gap_up_count_5d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 次数类归一化为 [0,1]
    return _gap_up_count_5d_norm_full(data)


def _gap_up_count_5d_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_gap_up_count_5d(data, _), target_index)


gap_up_count_5d = register_factor(
    FactorDefinition(
        name="gap_up_count_5d",
        required_columns={"stock_code", "date", "open", "close"},
        lookback_days=10,
        calculator=_gap_up_count_5d,
        calculation_scope="target_with_history_window",
        target_calculator=_gap_up_count_5d_target,
        description="[0,1] 近5日高开超过2%的次数占比，刻画短线追高/情绪拥挤",
    )
)


def _close_below_ma5_after_runup(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 二值状态，保持 0/1
    return _close_below_ma5_after_runup_full(data)


def _close_below_ma5_after_runup_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_close_below_ma5_after_runup(data, _), target_index)


close_below_ma5_after_runup = register_factor(
    FactorDefinition(
        name="close_below_ma5_after_runup",
        required_columns={"stock_code", "date", "close"},
        lookback_days=20,
        calculator=_close_below_ma5_after_runup,
        calculation_scope="target_with_history_window",
        target_calculator=_close_below_ma5_after_runup_target,
        description="0/1：10日涨幅超过10%后收盘跌破MA5，刻画高位承接破坏",
    )
)


def _low_break_prev_low_after_runup(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    # 二值状态，保持 0/1
    return _low_break_prev_low_after_runup_full(data)


def _low_break_prev_low_after_runup_target(data: pd.DataFrame, target_index: pd.Index, _: FactorContext) -> pd.Series:
    return _target_from_full(_low_break_prev_low_after_runup(data, _), target_index)


low_break_prev_low_after_runup = register_factor(
    FactorDefinition(
        name="low_break_prev_low_after_runup",
        required_columns={"stock_code", "date", "low", "close"},
        lookback_days=20,
        calculator=_low_break_prev_low_after_runup,
        calculation_scope="target_with_history_window",
        target_calculator=_low_break_prev_low_after_runup_target,
        description="0/1：10日涨幅超过10%后盘中跌破前低，刻画强势股承接转弱",
    )
)


__all__ = [
    "close_position_20d",
    "close_position_60d",
    "upper_shadow_ratio_3d_mean",
    "close_to_high_3d_mean",
    "turnover_z_20",
    "high_volume_no_gain_3d",
    "pullback_depth_from_10d_high",
    "high_pos_upper_shadow",
    "high_pos_volume_no_gain",
    "high_pos_reversal_volume",
    "ret_3d_vs_10d_accel",
    "ret_5d_vs_20d_accel",
    "momentum_peak_decay_3d",
    "consecutive_up_days_5d",
    "large_up_days_10d",
    "gap_up_count_5d",
    "close_below_ma5_after_runup",
    "low_break_prev_low_after_runup",
]
