"""
市场状态判断因子 v3。

设计目标：
1. 不替换现有 market_context / market_regime_context_v2，只作为增量状态因子使用。
2. 因子专门服务“板块 + 5日线强势股”结构的环境识别：
   - 龙头是否更容易持续
   - 强势股是否沿 MA5 扩散
   - 热点是否集中还是轮动
3. 因子全部按日广播给所有股票，更适合树模型做 regime gating。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor
from ..target_utils import select_target_series


def _to_dt(series: pd.Series) -> pd.Series:
    if str(series.dtype).startswith("datetime64"):
        return series
    txt = series.astype(str).str.replace("-", "", regex=False)
    return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")


def _get_market_data_safe(ctx: FactorContext) -> pd.DataFrame:
    market = ctx.require_market().copy()
    market = market.sort_values("date").reset_index(drop=True)
    market["date"] = _to_dt(market["date"])
    return market


def _broadcast_daily_factor(data: pd.DataFrame, daily_series: pd.Series, fill_value=np.nan) -> pd.Series:
    aligned = daily_series.copy()
    aligned.index = _to_dt(pd.Series(aligned.index))
    out = _to_dt(data["date"]).map(aligned)
    if fill_value is not np.nan:
        out = out.fillna(fill_value)
    return out.astype("float64")


def _get_sector_frame(data: pd.DataFrame, ctx: FactorContext) -> pd.DataFrame:
    sector_map = (ctx.extra or {}).get("stock_to_sw_sector", {})
    df = data[["date", "stock_code", "pct_chg"]].copy()
    df["date"] = _to_dt(df["date"])
    if sector_map:
        df["sector"] = df["stock_code"].map(sector_map).fillna("Unknown")
    else:
        df["sector"] = "All"
    return df


def _safe_turnover_rate(data: pd.DataFrame) -> pd.Series:
    if "turnover_rate" in data.columns:
        return pd.to_numeric(data["turnover_rate"], errors="coerce")
    if "volume" in data.columns and "circ_mv" in data.columns:
        return pd.to_numeric(data["volume"], errors="coerce") / (pd.to_numeric(data["circ_mv"], errors="coerce") + 1e-9)
    return pd.Series(np.nan, index=data.index)


def _mkt_ma5_followthrough_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    大盘沿 MA5 运行的顺滑程度。
    越高表示短期趋势延续越友好，更有利于 Top1 龙头延续。
    """
    market = _get_market_data_safe(ctx)
    close = pd.to_numeric(market["close"], errors="coerce")
    ret = pd.to_numeric(market.get("daily_change_pct", pd.Series(np.nan, index=market.index)), errors="coerce")

    ma5 = close.rolling(5, min_periods=3).mean()
    ma10 = close.rolling(10, min_periods=5).mean()
    vol10 = ret.rolling(10, min_periods=5).std()
    follow = ((close - ma5) / (ma5 + 1e-9)) - 0.5 * ((ma5 - ma10) / (ma10 + 1e-9))
    follow = follow / (vol10 + 1e-6)
    follow.index = market["date"]
    return _broadcast_daily_factor(data, follow)


def _sector_leadership_persistence_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    板块领涨结构的持续性。
    计算今日与前1/2/3日板块收益排名的一致性，越高表示龙头板块更容易持续。
    """
    df = _get_sector_frame(data, ctx)
    sector_ret = df.groupby(["date", "sector"])["pct_chg"].mean().unstack("sector").sort_index()
    rank = sector_ret.rank(axis=1, pct=True)

    vals = []
    idxs = []
    for i in range(len(rank)):
        cur = rank.iloc[i]
        corrs = []
        for lag in (1, 2, 3):
            if i - lag < 0:
                continue
            prev = rank.iloc[i - lag]
            valid = cur.notna() & prev.notna()
            if valid.sum() >= 3:
                corrs.append(cur[valid].corr(prev[valid]))
        vals.append(np.nanmean(corrs) if corrs else np.nan)
        idxs.append(rank.index[i])

    series = pd.Series(vals, index=idxs, dtype="float64")
    return _broadcast_daily_factor(data, series)


def _sector_hotspot_concentration_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    热点集中度：最强板块是否显著强于中位板块。
    越高说明 alpha 更集中在头部龙头，偏利于 Top1。
    """
    df = _get_sector_frame(data, ctx)
    sector_ret = df.groupby(["date", "sector"])["pct_chg"].mean().unstack("sector").sort_index()
    top = sector_ret.max(axis=1)
    med = sector_ret.median(axis=1)
    conc = (top - med).rolling(3, min_periods=1).mean()
    return _broadcast_daily_factor(data, conc)


def _ma5_strength_breadth_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    强势股广度：收在 MA5 上方且当日上涨的股票占比。
    越高表示强势结构扩散到前2~3名，偏利于 Top2/Top3。
    """
    if "close" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    close = pd.to_numeric(data["close"], errors="coerce")
    ma5 = data.groupby("stock_code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    pos = pd.to_numeric(data["pct_chg"], errors="coerce") > 0
    above = close > ma5
    daily_ratio = (above & pos).groupby(_to_dt(data["date"])).mean()
    breadth = daily_ratio.rolling(5, min_periods=3).mean() - daily_ratio.rolling(20, min_periods=10).mean()
    return _broadcast_daily_factor(data, breadth)


def _strong_stock_volume_confirmation_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    强势股量能确认强度：MA5 上方个股的异常换手均值。
    越高表示强势股不只是涨，而且有资金确认。
    """
    if "close" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    ma5 = data.groupby("stock_code")["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    above = pd.to_numeric(data["close"], errors="coerce") > ma5
    turnover = _safe_turnover_rate(data)
    base = data.groupby("stock_code")[turnover.name if turnover.name else "stock_code"].transform("size") if False else None
    to_ma20 = turnover.groupby(data["stock_code"]).transform(lambda x: x.rolling(20, min_periods=10).mean())
    abnormal = (turnover / (to_ma20 + 1e-9)).replace([np.inf, -np.inf], np.nan)
    daily = abnormal.where(above).groupby(_to_dt(data["date"])).mean()
    confirm = daily.rolling(3, min_periods=1).mean()
    return _broadcast_daily_factor(data, confirm)


def _emotion_quality_balance_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    情绪质量：涨停热度与炸板质量的组合。
    越高说明市场愿意给头部强股溢价，偏利于 Top1。
    """
    work = pd.DataFrame({"date": _to_dt(data["date"])})
    for col in ["is_limit_up", "is_limit_down", "is_zhaban"]:
        if col in data.columns:
            work[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        else:
            work[col] = 0.0

    daily = work.groupby("date").agg(
        limit_up=("is_limit_up", "sum"),
        limit_down=("is_limit_down", "sum"),
        zhaban=("is_zhaban", "sum"),
        total=("date", "count"),
    )
    heat = (daily["limit_up"] - daily["limit_down"]) / (daily["total"] + 1e-9)
    quality = 1.0 - daily["zhaban"] / (daily["limit_up"] + daily["zhaban"] + 1e-9)
    score = heat.rolling(3, min_periods=1).mean() * quality.rolling(3, min_periods=1).mean()
    return _broadcast_daily_factor(data, score)


def _rolling_percentile_self(series: pd.Series, window: int = 20, min_periods: int = 10) -> pd.Series:
    """
    将原始时间序列映射为其在过去 window 日自身历史中的分位数。
    适合状态判断：相比直接用原始收益均值，更抗波动率 regime 变化。
    """
    values = []
    for i in range(len(series)):
        cur = series.iloc[i]
        if pd.isna(cur):
            values.append(np.nan)
            continue
        hist = series.iloc[max(0, i - window + 1): i + 1].dropna()
        if len(hist) < min_periods:
            values.append(np.nan)
            continue
        pct = (hist <= cur).mean()
        values.append(float(pct))
    return pd.Series(values, index=series.index, dtype="float64")


def _expanding_percentile_shift1(series: pd.Series, min_periods: int = 20) -> pd.Series:
    values = []
    for i in range(len(series)):
        cur = series.iloc[i]
        if pd.isna(cur):
            values.append(np.nan)
            continue
        hist = series.iloc[:i].dropna()
        if len(hist) < min_periods:
            values.append(np.nan)
            continue
        values.append(float((hist <= cur).mean()))
    return pd.Series(values, index=series.index, dtype="float64")


def _top10_recent3d_strength_factor(
    data: pd.DataFrame,
    lookback_rank_days: int,
) -> pd.Series:
    """
    构造“头部股票近期强度状态因子”：
    1. 每日按 lookback_rank_days 累计涨幅排序，取前10股票
    2. 对这批股票计算其近3日累计涨跌幅均值
    3. 再映射到其过去20日历史中的分位数，得到更稳定的 regime signal
    """
    if "close" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = data[["date", "stock_code", "close"]].copy()
    df["date"] = _to_dt(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    ret_rank = df.groupby("stock_code")["close"].pct_change(lookback_rank_days)
    ret_recent3 = df.groupby("stock_code")["close"].pct_change(3)
    df["ret_rank"] = pd.to_numeric(ret_rank, errors="coerce")
    df["ret_recent3"] = pd.to_numeric(ret_recent3, errors="coerce")

    def _daily_top10_mean(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=["ret_rank", "ret_recent3"])
        if valid.empty:
            return np.nan
        picked = valid.nlargest(10, "ret_rank")
        if picked.empty:
            return np.nan
        return float(picked["ret_recent3"].mean())

    daily_top10_recent3 = df.groupby("date", sort=True).apply(_daily_top10_mean)
    daily_top10_recent3 = daily_top10_recent3.astype("float64")
    state = _rolling_percentile_self(daily_top10_recent3, window=20, min_periods=10)
    return _broadcast_daily_factor(data, state)


def _top10_rank_strength_level_factor(
    data: pd.DataFrame,
    lookback_rank_days: int,
) -> pd.Series:
    """
    构造“头部股票同口径强度水平因子”：
    1. 每日按 lookback_rank_days 累计涨幅排序，取前10股票
    2. 对这批股票计算其 lookback_rank_days 累计涨跌幅均值
    3. 再映射到其过去20日历史中的分位数，得到更稳定的 regime signal

    注意这里是“每天各自的 Top10”形成一个日序列，再做历史分位；
    不是固定 t 日那一批股票回看过去20日。
    """
    if "close" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    df = data[["date", "stock_code", "close"]].copy()
    df["date"] = _to_dt(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    ret_rank = df.groupby("stock_code")["close"].pct_change(lookback_rank_days)
    df["ret_rank"] = pd.to_numeric(ret_rank, errors="coerce")

    def _daily_top10_mean(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=["ret_rank"])
        if valid.empty:
            return np.nan
        picked = valid.nlargest(10, "ret_rank")
        if picked.empty:
            return np.nan
        return float(picked["ret_rank"].mean())

    daily_top10_rank_mean = df.groupby("date", sort=True).apply(_daily_top10_mean)
    daily_top10_rank_mean = daily_top10_rank_mean.astype("float64")
    state = _rolling_percentile_self(daily_top10_rank_mean, window=20, min_periods=10)
    return _broadcast_daily_factor(data, state)


def _top10_rank_members_today_return_factor(
    data: pd.DataFrame,
    lookback_rank_days: int,
) -> pd.Series:
    """
    构造“头部股票当日表现状态因子”：
    1. 每日按 lookback_rank_days 累计涨幅排序，取前10股票
    2. 对这批股票计算其当日涨跌幅均值（不是 lookback 涨幅均值）
    3. 再映射到其过去20日历史中的分位数

    注意：每天的 Top10 成员都重新确定；不是固定 t 日的 Top10 回看历史。
    """
    if "close" not in data.columns and "pct_chg" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    cols = ["date", "stock_code", "close"]
    if "pct_chg" in data.columns:
        cols.append("pct_chg")
    df = data[cols].copy()
    df["date"] = _to_dt(df["date"])
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    ret_rank = df.groupby("stock_code")["close"].pct_change(lookback_rank_days)
    df["ret_rank"] = pd.to_numeric(ret_rank, errors="coerce")
    if "pct_chg" in df.columns:
        df["today_ret"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
    else:
        df["today_ret"] = df.groupby("stock_code")["close"].pct_change(1)

    def _daily_top10_mean(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=["ret_rank", "today_ret"])
        if valid.empty:
            return np.nan
        picked = valid.nlargest(10, "ret_rank")
        if picked.empty:
            return np.nan
        return float(picked["today_ret"].mean())

    daily_top10_today_mean = df.groupby("date", sort=True).apply(_daily_top10_mean)
    daily_top10_today_mean = daily_top10_today_mean.astype("float64")
    state = _rolling_percentile_self(daily_top10_today_mean, window=20, min_periods=10)
    return _broadcast_daily_factor(data, state)


def _mkt_ma5_followthrough_v3_expanding_percentile(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    market = _get_market_data_safe(ctx)
    close = pd.to_numeric(market["close"], errors="coerce")
    ret = pd.to_numeric(market.get("daily_change_pct", pd.Series(np.nan, index=market.index)), errors="coerce")

    ma5 = close.rolling(5, min_periods=3).mean()
    ma10 = close.rolling(10, min_periods=5).mean()
    vol10 = ret.rolling(10, min_periods=5).std()
    follow = ((close - ma5) / (ma5 + 1e-9)) - 0.5 * ((ma5 - ma10) / (ma10 + 1e-9))
    follow = (follow / (vol10 + 1e-6)).astype("float64")
    follow.index = market["date"]
    expanding_pct = _expanding_percentile_shift1(follow, min_periods=20)
    expanding_pct.index = market["date"]
    return _broadcast_daily_factor(data, expanding_pct)


def _top10_10d_recent3_strength_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _top10_recent3d_strength_factor(data, lookback_rank_days=10)


def _top10_5d_recent3_strength_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _top10_recent3d_strength_factor(data, lookback_rank_days=5)


def _top10_3d_recent3_strength_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _top10_recent3d_strength_factor(data, lookback_rank_days=3)


def _top10_10d_strength_level_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _top10_rank_strength_level_factor(data, lookback_rank_days=10)


def _top10_10d_today_return_level_v3(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    return _top10_rank_members_today_return_factor(data, lookback_rank_days=10)


mkt_ma5_followthrough_v3 = register_factor(
    FactorDefinition(
        name="mkt_ma5_followthrough_v3",
        required_columns={"date"},
        requires_market_data=True,
        lookback_days=20,
        calculator=_mkt_ma5_followthrough_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _mkt_ma5_followthrough_v3(d, c), target_index=ti
        ),
        description="市场沿MA5顺滑延续度 v3（偏利于 Top1 龙头延续）",
    )
)

mkt_ma5_followthrough_v3_expanding_percentile = register_factor(
    FactorDefinition(
        name="mkt_ma5_followthrough_v3_expanding_percentile",
        required_columns={"date"},
        requires_market_data=True,
        lookback_days=25,
        calculator=_mkt_ma5_followthrough_v3_expanding_percentile,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _mkt_ma5_followthrough_v3_expanding_percentile(d, c), target_index=ti
        ),
        description="mkt_ma5_followthrough_v3 在历史 expanding 窗口中的分位数(shift1)",
    )
)

sector_leadership_persistence_v3 = register_factor(
    FactorDefinition(
        name="sector_leadership_persistence_v3",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_sector_data=True,
        lookback_days=8,
        calculator=_sector_leadership_persistence_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _sector_leadership_persistence_v3(d, c), target_index=ti
        ),
        description="板块领涨持续性 v3（今日与近3日板块排名一致性）",
    )
)

sector_hotspot_concentration_v3 = register_factor(
    FactorDefinition(
        name="sector_hotspot_concentration_v3",
        required_columns={"stock_code", "date", "pct_chg"},
        requires_sector_data=True,
        lookback_days=5,
        calculator=_sector_hotspot_concentration_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _sector_hotspot_concentration_v3(d, c), target_index=ti
        ),
        description="热点集中度 v3（最强板块相对中位板块的超额）",
    )
)

ma5_strength_breadth_v3 = register_factor(
    FactorDefinition(
        name="ma5_strength_breadth_v3",
        required_columns={"stock_code", "date", "close", "pct_chg"},
        lookback_days=25,
        calculator=_ma5_strength_breadth_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _ma5_strength_breadth_v3(d, c), target_index=ti
        ),
        description="MA5强势广度 v3（收在MA5上方且上涨的股票占比变化）",
    )
)

strong_stock_volume_confirmation_v3 = register_factor(
    FactorDefinition(
        name="strong_stock_volume_confirmation_v3",
        required_columns={"stock_code", "date", "close"},
        optional_columns={"turnover_rate", "volume", "circ_mv"},
        lookback_days=25,
        calculator=_strong_stock_volume_confirmation_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _strong_stock_volume_confirmation_v3(d, c), target_index=ti
        ),
        description="强势股量能确认 v3（MA5上方个股异常换手均值）",
    )
)

emotion_quality_balance_v3 = register_factor(
    FactorDefinition(
        name="emotion_quality_balance_v3",
        required_columns={"date"},
        optional_columns={"is_limit_up", "is_limit_down", "is_zhaban"},
        lookback_days=5,
        calculator=_emotion_quality_balance_v3,
        calculation_scope='daily_broadcast',
        target_calculator=lambda d, ti, c: select_target_series(
            _emotion_quality_balance_v3(d, c), target_index=ti
        ),
        description="情绪质量平衡 v3（涨停热度 x 炸板质量）",
    )
)

top10_10d_recent3_strength_v3 = register_factor(
    FactorDefinition(
        name="top10_10d_recent3_strength_v3",
        required_columns={"stock_code", "date", "close"},
        lookback_days=35,
        calculator=_top10_10d_recent3_strength_v3,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _top10_10d_recent3_strength_v3(d, c), target_index=ti
        ),
        description="10日涨幅前10股票近3日均值的20日历史分位（头部中期强度状态）",
    )
)

top10_5d_recent3_strength_v3 = register_factor(
    FactorDefinition(
        name="top10_5d_recent3_strength_v3",
        required_columns={"stock_code", "date", "close"},
        lookback_days=30,
        calculator=_top10_5d_recent3_strength_v3,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _top10_5d_recent3_strength_v3(d, c), target_index=ti
        ),
        description="5日涨幅前10股票近3日均值的20日历史分位（头部短期强度状态）",
    )
)

top10_3d_recent3_strength_v3 = register_factor(
    FactorDefinition(
        name="top10_3d_recent3_strength_v3",
        required_columns={"stock_code", "date", "close"},
        lookback_days=25,
        calculator=_top10_3d_recent3_strength_v3,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _top10_3d_recent3_strength_v3(d, c), target_index=ti
        ),
        description="3日涨幅前10股票近3日均值的20日历史分位（头部极短期强度状态）",
    )
)

top10_10d_strength_level_v3 = register_factor(
    FactorDefinition(
        name="top10_10d_strength_level_v3",
        required_columns={"stock_code", "date", "close"},
        lookback_days=35,
        calculator=_top10_10d_strength_level_v3,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _top10_10d_strength_level_v3(d, c), target_index=ti
        ),
        description="10日涨幅前10股票10日涨幅均值的20日历史分位（头部中期强度水平）",
    )
)

top10_10d_today_return_level_v3 = register_factor(
    FactorDefinition(
        name="top10_10d_today_return_level_v3",
        required_columns={"stock_code", "date", "close"},
        optional_columns={"pct_chg"},
        lookback_days=35,
        calculator=_top10_10d_today_return_level_v3,
        calculation_scope='stateful_history',
        target_calculator=lambda d, ti, c: select_target_series(
            _top10_10d_today_return_level_v3(d, c), target_index=ti
        ),
        description="10日涨幅前10股票当日涨幅均值的20日历史分位（头部中期当日状态）",
    )
)


__all__ = [
    "mkt_ma5_followthrough_v3",
    "mkt_ma5_followthrough_v3_expanding_percentile",
    "sector_leadership_persistence_v3",
    "sector_hotspot_concentration_v3",
    "ma5_strength_breadth_v3",
    "strong_stock_volume_confirmation_v3",
    "emotion_quality_balance_v3",
    "top10_10d_recent3_strength_v3",
    "top10_5d_recent3_strength_v3",
    "top10_3d_recent3_strength_v3",
    "top10_10d_strength_level_v3",
    "top10_10d_today_return_level_v3",
]
