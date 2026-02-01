"""动量与反转因子（基础+复合）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 0. 辅助工具函数
# ==============================================================================


def _intra_day_zscore(series: pd.Series, date_series: pd.Series) -> pd.Series:
    """
    内部使用的归一化函数：基于 Rank 的 Z-Score 映射
    将数据映射到 Standard Normal Distribution，消除量纲并去极值
    """
    # 1. 计算每日 Rank (0~1)
    ranks = series.groupby(date_series).rank(pct=True).fillna(0.5)
    
    # 2. 映射到 Z-Score (使用 scipy.stats.norm.ppf 的近似或简单线性映射)
    # 为了不引入 scipy 依赖，使用简单的线性映射 + Clip 模拟
    # Rank 0 -> -3, Rank 1 -> 3
    # 这种映射比普通 (x-mean)/std 更稳健
    z_proxy = (ranks - 0.5) * 6.0 # 映射到 [-3, 3] 区间
    
    return z_proxy



# ==============================================================================
# 1. 基础动量/反转因子 (Returns) - Alpha/Beta 分离版
# ==============================================================================

def _make_return_factor(window: int, kind: str = 'alpha') -> callable:
    """
    工厂函数：
    kind='alpha': 返回【原始收益率】 (Raw Return) -> 等待后续全局 Z-Score
    kind='beta':  返回【SW2 行业均值】 (Raw Sector Mean) -> 作为板块热度特征
    """
    def calculator(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
        if 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        # 1. 计算原始收益率
        if window == 0:
            raw_ret = data['pct_chg']
        else:
            grouped = data.groupby('stock_code', sort=False)['close']
            prev_close = grouped.shift(window)
            raw_ret = (data['close'] - prev_close) / (prev_close + 1e-9)

        # 2. 根据类型分流
        if kind == 'alpha':
            # === Alpha: 直接返回原始值 ===
            # 后续统一做 Global Z-Score
            return raw_ret.astype('float64')
            
        else:
            # === Beta: 计算 SW2 行业均值 ===
            # 保留此逻辑，因为"行业均值"本身就是一个独立的特征值
            sector_map = ctx.extra.get('stock_to_sw_sector')
            if sector_map is None:
                return raw_ret.groupby(data['date']).transform('mean')
            
            sectors = data['stock_code'].map(sector_map).fillna('Unknown')
            df = pd.DataFrame({'val': raw_ret.values, 'date': data['date'].values, 'sector': sectors.values}, index=data.index)
            # 计算行业均值
            sector_mean = df.groupby(['date', 'sector'])['val'].transform('mean')
            return sector_mean.astype('float64')

    return calculator



# --- 0日 (pct_chg) ---
pct_chg_intra_z = register_factor(
    FactorDefinition(
        name='pct_chg_intra_z',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=1,
        calculator=_make_return_factor(0, 'alpha'),
        description='当日涨跌幅的行业内Z-Score (相对强弱)',
    )
)


pct_chg_sector_mean = register_factor(
    FactorDefinition(
        name='pct_chg_sector_mean',
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=1,
        calculator=_make_return_factor(0, 'beta'),
        description='当日所属行业的平均涨跌幅 (板块热度)',
    )
)

# --- 3日 ---
return_3d_intra_z = register_factor(
    FactorDefinition(
        name='return_3d_intra_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=5,
        calculator=_make_return_factor(3, 'alpha'),
        description='3日累计涨幅的行业内Z-Score',
    )
)
return_3d_sector_mean = register_factor(
    FactorDefinition(
        name='return_3d_sector_mean',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=5,
        calculator=_make_return_factor(3, 'beta'),
        description='3日累计涨幅的行业均值',
    )
)

# --- 5日 ---
return_5d_intra_z = register_factor(
    FactorDefinition(
        name='return_5d_intra_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_make_return_factor(5, 'alpha'),
        description='5日累计涨幅的行业内Z-Score',
    )
)
return_5d_sector_mean = register_factor(
    FactorDefinition(
        name='return_5d_sector_mean',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_make_return_factor(5, 'beta'),
        description='5日累计涨幅的行业均值',
    )
)

# --- 10日 ---
return_10d_intra_z = register_factor(
    FactorDefinition(
        name='return_10d_intra_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=15,
        calculator=_make_return_factor(10, 'alpha'),
        description='10日累计涨幅的行业内Z-Score',
    )
)
return_10d_sector_mean = register_factor(
    FactorDefinition(
        name='return_10d_sector_mean',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=15,
        calculator=_make_return_factor(10, 'beta'),
        description='10日累计涨幅的行业均值',
    )
)

# --- 20日 ---
return_20d_intra_z = register_factor(
    FactorDefinition(
        name='return_20d_intra_z',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_make_return_factor(20, 'alpha'),
        description='20日累计涨幅的行业内Z-Score',
    )
)
return_20d_sector_mean = register_factor(
    FactorDefinition(
        name='return_20d_sector_mean',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_make_return_factor(20, 'beta'),
        description='20日累计涨幅的行业均值',
    )
)



def _pct_chg_1d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    昨日涨幅（基础因子）

    直接使用昨日涨幅，不进行行业中性化
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    pct_chg_1d = grouped['pct_chg'].shift(1).reset_index(level=0, drop=True)

    return pct_chg_1d.astype('float64')


def _pct_chg_2d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    前日涨幅（基础因子）

    直接使用前日涨幅，不进行行业中性化
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    pct_chg_2d = grouped['pct_chg'].shift(2).reset_index(level=0, drop=True)

    return pct_chg_2d.astype('float64')


def _return_3d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    3日累计收益（基础因子）

    直接使用3日累计收益，不进行行业中性化
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_3d = grouped['close'].pct_change(3).reset_index(level=0, drop=True)
    return_3d_pct = return_3d * 100  # 转换为百分比

    return return_3d_pct.astype('float64')


def _return_5d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    5日累计收益（基础因子）

    直接使用5日累计收益，不进行行业中性化
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_5d = grouped['close'].pct_change(5).reset_index(level=0, drop=True)
    return_5d_pct = return_5d * 100  # 转换为百分比

    return return_5d_pct.astype('float64')


def _return_10d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    10日累计收益（基础因子）

    直接使用10日累计收益，不进行行业中性化
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_10d = grouped['close'].pct_change(10).reset_index(level=0, drop=True)
    return_10d_pct = return_10d * 100  # 转换为百分比

    return return_10d_pct.astype('float64')


def _return_20d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    20日累计收益（基础因子）

    直接使用20日累计收益，不进行行业中性化
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_20d = grouped['close'].pct_change(20).reset_index(level=0, drop=True)
    return_20d_pct = return_20d * 100  # 转换为百分比

    return return_20d_pct.astype('float64')


# ==============================================================================
# 3. 日内反转/强度因子 (Intraday Reversal/Strength)
# ==============================================================================

def _intraday_clv(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    日内相对位置 (Close Location Value, CLV)
    """
    if not {'high', 'low', 'close'}.issubset(data.columns):
        return pd.Series(0.5, index=data.index)
        
    high_low_range = data['high'] - data['low']
    
    # 避免除零
    clv = (data['close'] - data['low']) / (high_low_range + 1e-9)
    
    # 对于一字板 (high == low)，如果涨停设为1，跌停设为0，其他0.5
    is_flat = high_low_range < 1e-4
    # 这里简单处理，一字板给 1.0 (强者恒强)
    clv = np.where(is_flat, 1.0, clv)
    
    return pd.Series(clv, index=data.index).clip(0, 1).astype('float64')


def _trend_acceleration(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    趋势加速度因子（行业中性化）
    """
    if 'close' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_5d = grouped['close'].pct_change(5).reset_index(level=0, drop=True)
    return_20d = grouped['close'].pct_change(20).reset_index(level=0, drop=True)

    # 计算趋势加速度：短期收益与长期收益的差值
    acceleration = (return_5d - return_20d) * 100

    return acceleration.astype('float64')

def _momentum_decay(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    动量衰减因子优化版
    """
    req = ['close', 'pct_chg']
    if not all(c in data.columns for c in req):
        return pd.Series(0.0, index=data.index, dtype='float64')
    
    try:
        # 5日收益率
        ret_5 = data.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change(5)
        )
        
        # 20日收益率
        ret_20 = data.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change(20)
        )
        
        # 20日波动率
        vol_20 = data.groupby('stock_code')['pct_chg'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        
        # 计算因子
        momentum_adjusted = ret_5 - (ret_20 / 4.0)
        factor = momentum_adjusted / (vol_20 + 1e-9)
        
    except Exception as e:
        print(f"Transform方法失败: {e}")
        return pd.Series(0.0, index=data.index)
    
    # 异常值处理
    if len(factor) > 100:
        q_low = factor.quantile(0.01)
        q_high = factor.quantile(0.99)
        factor = factor.clip(q_low, q_high)
    
    return factor.astype('float64')


def _trend_consistency(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    趋势一致性因子（时间序列因子）- 5日版本
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.5, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    up_days = (data['pct_chg'] > 0).astype(float)
    consistency = grouped.apply(
        lambda g: up_days.loc[g.index].rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    return consistency.astype('float64')


def _volatility_structure_alpha(
    data: pd.DataFrame,
    context: FactorContext
) -> pd.Series:
    """
    波动率结构 Alpha（行业中性化）
    """
    required_cols = {'stock_code', 'date', 'pct_chg'}
    if not required_cols.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    # sector_mapping = context.extra.get('stock_to_sw_sector') # 暂不需要sector

    returns = data['pct_chg'] / 100.0

    vol_10 = (
        returns
        .groupby(data['stock_code'])
        .rolling(window=10, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    vol_trend = (
        vol_10
        .groupby(data['stock_code'])
        .diff(5)
    )

    vol_trend_signal = -vol_trend
    raw_alpha = vol_10 * vol_trend_signal
    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)

    return raw_alpha.astype('float64')


def _volume_structure_alpha(
    data: pd.DataFrame,
    context: FactorContext
) -> pd.Series:
    """
    成交结构失衡 Alpha（行业中性化）
    """
    required_cols = {'stock_code', 'date', 'pct_chg', 'volume'}
    if not required_cols.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    returns = data['pct_chg'] / 100.0

    vol_mean_5 = (
        data['volume']
        .groupby(data['stock_code'])
        .rolling(5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    vol_mean_20 = (
        data['volume']
        .groupby(data['stock_code'])
        .rolling(20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    volume_ratio = vol_mean_5 / vol_mean_20

    price_vol_10 = (
        returns
        .groupby(data['stock_code'])
        .rolling(10, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    raw_alpha = volume_ratio - price_vol_10
    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    return raw_alpha.astype('float64')


def _regime_aware_reversal_alpha(
    data: pd.DataFrame,
    context: FactorContext
) -> pd.Series:
    """
    状态感知的短期反转 Alpha（行业中性化）
    """
    required_cols = {'stock_code', 'date', 'pct_chg'}
    if not required_cols.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    returns = data['pct_chg'] / 100.0

    ret_3 = (
        returns
        .groupby(data['stock_code'])
        .rolling(3, min_periods=2)
        .sum()
        .reset_index(level=0, drop=True)
    )

    ret_20 = (
        returns
        .groupby(data['stock_code'])
        .rolling(20, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )

    trend_strength = ret_20.abs()
    raw_alpha = -ret_3 * (1.0 - trend_strength.abs())
    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    return raw_alpha.astype('float64')


regime_aware_reversal_alpha = register_factor(
    FactorDefinition(
        name='regime_aware_reversal_alpha',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=25,
        calculator=_regime_aware_reversal_alpha,
        description='状态感知的短期反转Alpha（趋势弱时反转更有效）',
    )
)

def _trend_pullback_composite_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    [优化版] 趋势门控 + 回撤质量 组合 Alpha
    优化：对子项进行 Rank 化处理，消除量纲差异。
    """
    required = {'close', 'volume'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code')
    close = data['close']
    date = data['date']

    # 1. 趋势门控 (Trend Gate)
    ma20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    ma60 = grouped['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    trend_up = (ma20 > ma60)
    above_ma20 = close > ma20
    ret_20 = grouped['close'].transform(lambda x: x.pct_change(20))
    positive_trend = ret_20 > 0
    trend_gate = (trend_up & above_ma20 & positive_trend).astype(float)

    # 2. 回撤质量 (Pullback Quality) - 内部归一化
    # A. 回撤幅度
    ret_5 = grouped['close'].transform(lambda x: x.pct_change(5))
    pullback_val = ret_5.clip(upper=0) 
    score_depth = _intra_day_zscore(pullback_val, date) 

    # B. 成交量萎缩
    vol_3 = grouped['volume'].transform(lambda x: x.rolling(3, min_periods=2).mean())
    vol_10 = grouped['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    vol_ratio = vol_3 / (vol_10 + 1e-8)
    score_vol = _intra_day_zscore(-vol_ratio, date)

    # C. 支撑位
    ma5 = grouped['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    score_support = (close > ma5).astype(float)

    pullback_quality = score_depth + score_vol + score_support

    # 3. 组合 Alpha
    alpha = trend_gate * pullback_quality

    return alpha.replace([np.inf, -np.inf], np.nan)


def _volume_confirmed_pullback_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    [优化版] 放量上涨后的缩量回撤 Alpha
    优化：对三个不同量纲的子项进行 Z-Score 标准化后再合成。
    """
    required = {'close', 'volume'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code')
    date = data['date']

    # 1. 上涨力度
    ret_5 = grouped['close'].transform(lambda x: x.pct_change(5))
    score_mom = _intra_day_zscore(ret_5, date)

    # 2. 回撤深度
    pullback_ret = grouped['close'].transform(lambda x: x.pct_change(3))
    score_pullback = _intra_day_zscore(-pullback_ret.abs(), date)

    # 3. 量能形态
    vol_5 = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    vol_20 = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    pullback_vol = grouped['volume'].transform(lambda x: x.rolling(3, min_periods=2).mean())
    
    expansion_ratio = vol_5 / (vol_20 + 1e-8)
    score_expansion = _intra_day_zscore(expansion_ratio, date)
    
    contraction_ratio = pullback_vol / (vol_5 + 1e-8)
    score_contraction = _intra_day_zscore(-contraction_ratio, date)

    # 4. 综合 Alpha
    alpha = score_mom + score_contraction + score_expansion + score_pullback

    return alpha.replace([np.inf, -np.inf], np.nan)


def _high_volatility_pullback_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    [优化版] 高波动回踩因子
    """
    grouped = data.groupby('stock_code')
    close = data['close']
    
    ma20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    vol_pct = grouped['close'].transform(lambda x: x.pct_change().rolling(10, min_periods=5).std())
    vol_price = close * vol_pct 
    z_distance = (close - ma20) / (vol_price + 1e-8)
    
    ma20_delay = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean().shift(5))
    trend_score = (ma20 / (ma20_delay + 1e-8)) - 1.0
    trend_mask = trend_score > 0
    
    target_z = -0.5 
    pullback_score = np.exp(-((z_distance - target_z) ** 2) / 2.0)
    
    vol_rank = vol_pct.groupby(data['date']).rank(pct=True)
    
    alpha = trend_score * vol_rank * pullback_score
    alpha = alpha.where(trend_mask, 0)
    
    return alpha.replace([np.inf, -np.inf], np.nan)


# ==============================================================================
# 2. 优化版 CPV：量价趋势相关性 (20日)
# ==============================================================================
def _cpv_factor_optimized(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [优化版] CPV 量价趋势相关性 (20日) - Raw Value
    """
    required_cols = ['close', 'volume']
    if not all(col in data.columns for col in required_cols):
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby('stock_code', sort=False)
    
    log_ret = np.log(data['close']) - np.log(grouped['close'].shift(1))
    log_vol_chg = np.log1p(data['volume']) - np.log1p(grouped['volume'].shift(1))
    
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    log_vol_chg = log_vol_chg.replace([np.inf, -np.inf], np.nan)
    
    temp_df = pd.DataFrame({
        'a': log_ret,
        'b': log_vol_chg,
        'stock_code': data['stock_code']
    }, index=data.index)
    
    def calc_rolling_corr(df):
        return df['a'].rolling(20, min_periods=15).corr(df['b'])

    raw_corr = temp_df.groupby('stock_code', sort=False, group_keys=False).apply(calc_rolling_corr)
    
    if not raw_corr.index.equals(data.index):
        raw_corr = raw_corr.reindex(data.index)
        
    return raw_corr.astype('float64')


cpv_factor = register_factor(
    FactorDefinition(
        name='cpv_factor',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=30, # 需要足够历史计算 20日 rolling
        calculator=_cpv_factor_optimized,
        description='[优化] 20日量价变化相关性',
    )
)

# ==============================================================================
# 4. 优化版：卡夫曼效率动量 (Efficiency Momentum)
# ==============================================================================
def _efficiency_momentum(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [优化] 卡夫曼效率系数 (Kaufman Efficiency Ratio, KER)
    """
    lookback = 10
    grouped = data.groupby('stock_code', sort=False)
    net_change = grouped['close'].diff(lookback).abs()
    daily_change = grouped['close'].diff(1).abs()
    total_path = daily_change.rolling(lookback, min_periods=lookback-2).sum()
    er = net_change / (total_path + 1e-9)
    direction = np.sign(grouped['close'].diff(lookback))
    final_factor = er * direction
    return final_factor

stable_momentum = register_factor(
    FactorDefinition(
        name='stable_momentum', # 保持原名以便替换，实际是效率动量
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=20,
        calculator=_efficiency_momentum,
        description='[优化] 卡夫曼效率动量 (趋势纯度)',
    )
)

high_volatility_pullback_alpha = register_factor(
    FactorDefinition(
        name='high_volatility_pullback_alpha',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=30,
        calculator=_high_volatility_pullback_alpha,
        description='[优化] 高波动趋势回踩因子 (Z-Score Distance)',
    )
)


volume_confirmed_pullback_alpha = register_factor(
    FactorDefinition(
        name='volume_confirmed_pullback_alpha',
        required_columns={'stock_code', 'date', 'close','volume'},
        lookback_days=20,
        calculator=_volume_confirmed_pullback_alpha,
        description='组合残差Alpha（行业 + 动量无法解释的部分）',
    )
)


trend_pullback_composite_alpha = register_factor(
    FactorDefinition(
        name='trend_pullback_composite_alpha',
        required_columns={'stock_code', 'date', 'close','volume'},
        lookback_days=60,
        calculator=_trend_pullback_composite_alpha,
        description='组合残差Alpha（行业 + 动量无法解释的部分）',
    )
)

# 注册因子
pct_chg_1d = register_factor(
    FactorDefinition(
        name='pct_chg_1d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=5,
        calculator=_pct_chg_1d,
        description='昨日涨幅（基础因子）',
    )
)

pct_chg_2d = register_factor(
    FactorDefinition(
        name='pct_chg_2d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=5,
        calculator=_pct_chg_2d,
        description='前日涨幅（基础因子）',
    )
)


return_3d = register_factor(
    FactorDefinition(
        name='return_3d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=8,
        calculator=_return_3d,
        description='3日累计收益（基础因子，百分比形式）',
    )
)

return_5d = register_factor(
    FactorDefinition(
        name='return_5d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_return_5d,
        description='5日累计收益（基础因子，百分比形式）',
    )
)


return_10d = register_factor(
    FactorDefinition(
        name='return_10d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=15,
        calculator=_return_10d,
        description='10日累计收益（基础因子，百分比形式）',
    )
)


return_20d = register_factor(
    FactorDefinition(
        name='return_20d',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=25,
        calculator=_return_20d,
        description='20日累计收益（基础因子，百分比形式）',
    )
)

intraday_clv = register_factor(
    FactorDefinition(
        name='intraday_clv',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=5,
        calculator=_intraday_clv,
        description='动量延续vs反转因子（复合因子）',
    )
)

trend_acceleration = register_factor(
    FactorDefinition(
        name='trend_acceleration',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=25,
        calculator=_trend_acceleration,
        description='趋势加速度因子（行业中性化）',
    )
)

momentum_decay = register_factor(
    FactorDefinition(
        name='momentum_decay',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=25,
        calculator=_momentum_decay,
        description='动量衰减因子（时间序列因子）',
    )
)


trend_consistency = register_factor(
    FactorDefinition(
        name='trend_consistency',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=10,
        calculator=_trend_consistency,
        description='趋势一致性因子（时间序列因子）- 5日版本',
    )
)

volatility_structure_alpha = register_factor(
    FactorDefinition(
        name='volatility_structure_alpha',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=15,
        calculator=_volatility_structure_alpha,
        description='行业中性化的波动率结构Alpha（低波动 + 波动率下降）',
    )
)

volume_structure_alpha = register_factor(
    FactorDefinition(
        name='volume_structure_alpha',
        required_columns={'stock_code', 'date', 'pct_chg', 'volume'},
        lookback_days=25,
        calculator=_volume_structure_alpha,
        description='成交结构失衡Alpha（量增但价格波动未扩散）',
    )
)

__all__ = [
    'pct_chg_intra_z',
    'pct_chg_sector_mean',
    'pct_chg_1d',
    'pct_chg_2d',
    'return_3d_intra_z',
    'return_3d_sector_mean',
    'return_5d_intra_z',
    'return_5d_sector_mean',
    'return_10d_intra_z',
    'return_10d_sector_mean',
    'return_20d_intra_z',
    'return_20d_sector_mean',
    'intraday_clv',
    'momentum_decay',
    'trend_acceleration',
    'trend_consistency',
    'cpv_factor',
    'stable_momentum',
    'high_volatility_pullback_alpha',
    'volume_confirmed_pullback_alpha',
    'trend_pullback_composite_alpha',
    'regime_aware_reversal_alpha',
    'volume_structure_alpha',
    'volatility_structure_alpha'
]
