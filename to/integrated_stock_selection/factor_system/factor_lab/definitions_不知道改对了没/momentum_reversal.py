"""动量与反转因子（基础+复合）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor



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


# ==============================================================================
# High → Close 回撤因子（盘中冲高失败强度）
# ==============================================================================

def _high_to_close_drawdown(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    当日从最高价到收盘价的回撤幅度

    计算：
        (close - high) / high

    含义：
    - 0        : 收在最高点，极强
    - < 0      : 冲高回落
    - 数值越小 : 回落越剧烈（卖压/失败信号）

    典型应用：
    - 炸板识别
    - 冲高出货
    - 回撤风险惩罚
    """
    required = {'high', 'close'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    high = data['high'].astype('float64')
    close = data['close'].astype('float64')

    drawdown = (close - high) / (high + 1e-12)

    return drawdown.replace([np.inf, -np.inf], np.nan).astype('float64')


high_to_close_drawdown = register_factor(
    FactorDefinition(
        name='high_to_close_drawdown',
        required_columns={'stock_code', 'date', 'high', 'close'},
        lookback_days=0,
        calculator=_high_to_close_drawdown,
        description='当日从最高价到收盘价的回撤幅度（盘中冲高失败强度）',
    )
)

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


def _return_5d_cs_rank(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    5日累计收益的日截面排名（降序）。

    含义：
    - 1 表示当日 5 日涨幅最高
    - 200 表示当日 5 日涨幅排在前 200

    适合用于训练样本过滤，例如：
    - 仅保留 return_5d_cs_rank <= 200 的样本
    """
    if 'close' not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    return_5d = grouped['close'].pct_change(5).reset_index(level=0, drop=True)

    rank_df = pd.DataFrame(
        {
            'date': data['date'].values,
            'return_5d': pd.to_numeric(return_5d, errors='coerce').values,
        },
        index=data.index,
    )
    cs_rank = rank_df.groupby('date')['return_5d'].rank(
        ascending=False,
        method='first',
    )
    return cs_rank.astype('float64')


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
    
    逻辑优化：
    原逻辑 (high-low)/close 只是波动率。
    新逻辑 ((close - low) - (high - close)) / (high - low)
    或者简化版: (close - low) / (high - low)
    
    值域 [0, 1]:
    - 1.0: 收在最高 (极强)
    - 0.0: 收在最低 (极弱/日内反转向下)
    - 0.5: 十字星/纠结
    
    这个因子对于预测次日开盘和短期反转非常有效。
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

    计算逻辑：
    return_5d - return_20d
    - 正值：短期趋势强于长期趋势（加速上涨）
    - 负值：短期趋势弱于长期趋势（减速或反转）

    修改说明：
    1. 移除ATR归一化：行业中性化已经消除了系统性风险差异
    2. 保留原始波动率差异：个股波动率是趋势加速度的有效组成部分
    3. 聚焦趋势变化：纯粹衡量趋势的加速/减速现象

    理论依据：
    - 行业中性化已处理行业层面的系统性差异
    - 个股波动率反映股票特有风险，应保留在因子中
    - 趋势加速度的核心在于收益变化的二阶导数，不需要额外归一化
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
    
    解决索引对齐问题，提高计算效率
    """
    req = ['close', 'pct_chg']
    if not all(c in data.columns for c in req):
        return pd.Series(0.0, index=data.index, dtype='float64')
    
    # 方法1：使用transform确保索引一致性（最简洁）
    try:
        # 5日收益率
        ret_5 = data.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change(5)
        )
        
        # 20日收益率
        ret_20 = data.groupby('stock_code')['close'].transform(
            lambda x: x.pct_change(20)
        )
        
        # 20日波动率（使用对数收益率更稳定）
        # 如果pct_chg是百分比形式，需要除以100
        vol_20 = data.groupby('stock_code')['pct_chg'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        
        # 计算因子：短期动量与长期动量调整后的差异
        # 将20日收益率调整到5日尺度：ret_20 / 4
        momentum_adjusted = ret_5 - (ret_20 / 4.0)
        
        # 风险调整
        factor = momentum_adjusted / (vol_20 + 1e-9)
        
    except Exception as e:
        # 如果transform失败，回退到循环方法
        print(f"Transform方法失败，使用循环方法: {e}")
        raise ValueError("Transform方法失败，使用循环方法")
    
    # 异常值处理
    if len(factor) > 100:
        # 使用缩尾处理（winsorization）
        q_low = factor.quantile(0.01)
        q_high = factor.quantile(0.99)
        factor = factor.clip(q_low, q_high)
    
    return factor.astype('float64')


def _trend_consistency(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    趋势一致性因子（时间序列因子）- 5日版本

    计算逻辑：
    5日内上涨天数 / 5
    - 1.0：5天全部上涨（强趋势）
    - 0.0：5天全部下跌（强趋势）
    - 0.5：涨跌各半（无趋势）
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

    核心思想：
    1. 行业内异常低波动率
    2. 波动率正在下降
    """

    # -------- 基础检查 --------
    required_cols = {'stock_code', 'date', 'pct_chg'}
    if not required_cols.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    sector_mapping = context.extra.get('stock_to_sw_sector')
    if sector_mapping is None or len(sector_mapping) == 0:
        raise ValueError("volatility_structure_alpha 计算失败：缺少 SW 行业映射")

    # -------- Step 1：收益率 --------
    returns = data['pct_chg'] / 100.0

    # -------- Step 2：10日波动率（因果） --------
    vol_10 = (
        returns
        .groupby(data['stock_code'])
        .rolling(window=10, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )


    # -------- Step 4：波动率趋势（下降为正） --------
    vol_trend = (
        vol_10
        .groupby(data['stock_code'])
        .diff(5)
    )

    # 波动率下降 ⇒ 正信号
    vol_trend_signal = -vol_trend

    # -------- Step 5：组合 Alpha（连续、可学习） --------
    raw_alpha = vol_10 * vol_trend_signal

    # -------- Step 6：数值清洗 --------
    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)

    return raw_alpha.astype('float64')


def _volume_structure_alpha(
    data: pd.DataFrame,
    context: FactorContext
) -> pd.Series:
    """
    成交结构失衡 Alpha（行业中性化）

    核心：
    成交量放大，但价格波动未同步放大
    """

    required_cols = {'stock_code', 'date', 'pct_chg', 'volume'}
    if not required_cols.issubset(data.columns):
        return pd.Series(0.0, index=data.index, dtype='float64')

    sector_mapping = context.extra.get('stock_to_sw_sector')
    if sector_mapping is None or len(sector_mapping) == 0:
        raise ValueError("volume_structure_alpha 缺少 SW 行业映射")

    # ---- 收益率 ----
    returns = data['pct_chg'] / 100.0

    # ---- 成交量活跃度 ----
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

    # ---- 价格波动 ----
    price_vol_10 = (
        returns
        .groupby(data['stock_code'])
        .rolling(10, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ---- Alpha：量增 + 波动未增 ----
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

    # ---- 短期收益（反转信号）----
    ret_3 = (
        returns
        .groupby(data['stock_code'])
        .rolling(3, min_periods=2)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # ---- 中期趋势强度（状态）----
    ret_20 = (
        returns
        .groupby(data['stock_code'])
        .rolling(20, min_periods=10)
        .sum()
        .reset_index(level=0, drop=True)
    )

    trend_strength = ret_20.abs()

    # ---- 状态感知反转 ----
    raw_alpha = -ret_3 * (1.0 - trend_strength.abs())

    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    return raw_alpha.astype('float64')


def _trend_pullback_composite_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    趋势门控 + 回撤质量 组合 Alpha（A股多头专用，严格因果）
    """

    required = {'close', 'volume'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code')

    close = data['close']
    volume = data['volume']

    # ======================
    # 1. 趋势门控（Trend Gate）
    # ======================
    ma20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    ma60 = grouped['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())

    trend_up = (ma20 > ma60)
    above_ma20 = close > ma20

    ret_20 = grouped['close'].transform(lambda x: x.pct_change(20))
    positive_trend = ret_20 > 0

    trend_gate = (trend_up & above_ma20 & positive_trend).astype(float)

    # ======================
    # 2. 回撤质量（Pullback Quality）
    # ======================
    # 回撤幅度
    ret_5 = grouped['close'].transform(lambda x: x.pct_change(5))
    pullback_depth = -ret_5.clip(upper=0)

    # 成交量萎缩
    vol_3 = grouped['volume'].transform(lambda x: x.rolling(3, min_periods=2).mean())
    vol_10 = grouped['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    volume_contraction = 1.0 - vol_3 / (vol_10 + 1e-8)

    # 回撤仍在支撑上
    ma5 = grouped['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    support_hold = (close > ma5).astype(float)

    pullback_quality = (
        -pullback_depth
        + volume_contraction
        + support_hold
    )

    # ======================
    # 3. 组合 Alpha
    # ======================
    alpha = trend_gate * pullback_quality

    return alpha.replace([np.inf, -np.inf], np.nan)


def _volume_confirmed_pullback_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    放量上涨后的缩量回撤 Alpha（严格因果，日线可用）
    """

    required = {'close', 'volume'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code')

    # --- 放量上涨阶段 ---
    vol_5 = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    vol_20 = grouped['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    volume_expansion = vol_5 / (vol_20 + 1e-8)

    ret_5 = grouped['close'].transform(lambda x: x.pct_change(5))

    # --- 回撤阶段 ---
    pullback_ret = grouped['close'].transform(lambda x: x.pct_change(3))
    pullback_volume = grouped['volume'].transform(lambda x: x.rolling(3, min_periods=2).mean())

    pullback_volume_ratio = pullback_volume / (vol_5 + 1e-8)

    pullback_depth = -pullback_ret.clip(upper=0) / (ret_5.abs() + 1e-8)

    # --- 综合 Alpha ---
    alpha = (
        ret_5
        - pullback_depth
        - pullback_volume_ratio
    )

    return alpha.replace([np.inf, -np.inf], np.nan)




def _high_volatility_pullback_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    [优化版] 高波动回踩因子 (Volatility Adjusted Mean Reversion)
    
    逻辑改进：
    1. 趋势判定：使用 MA20 斜率和价格位置，确保处于上升通道。
    2. 动态回踩：计算价格距离 MA20 的"标准差距离" (Z-Score)，而不是绝对距离。
       这解决了高波动股票天然距离均线远的问题。
    3. 支撑确认：寻找 (Close - MA20) / Volatility 处于 [-1.5, 0.5] 区间的股票。
       即：跌破了均线一点点，或者刚好在均线附近。
    """
    
    # 1. 基础数据准备
    grouped = data.groupby('stock_code')
    close = data['close']
    
    # 2. 计算均线 (MA20 作为生命线，比 MA5 更稳健)
    ma20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    
    # 3. 计算波动率 (10日收益率标准差)
    # 注意：这里我们用 close 计算波动幅度的绝对值代理，为了和距离做除法
    # 更好的方式是用 rolling std of price，或者用 atr。这里沿用收益率波动率但做了转换。
    # 转换逻辑：Volatility_Price ≈ Close * Volatility_Pct
    vol_pct = grouped['close'].transform(lambda x: x.pct_change().rolling(10, min_periods=5).std())
    vol_price = close * vol_pct # 将百分比波动转换为价格波动幅度
    
    # 4. 计算标准化距离 (Z-Score Distance)
    # 含义：当前价格距离 MA20 有几个标准差？
    # 负数代表回踩（在均线下方），正数代表远离（在均线上方）
    # + 1e-8 防止除零
    z_distance = (close - ma20) / (vol_price + 1e-8)
    
    # 5. 构建 Alpha：寻找"良性回踩"
    
    # A. 趋势项 (Trend): 必须是大趋势向上的
    # 简单的判断：MA20 本身在上涨 (MA20 / MA20_delay > 1)
    ma20_delay = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean().shift(5))
    trend_score = (ma20 / (ma20_delay + 1e-8)) - 1.0
    # 只要趋势向上的
    trend_mask = trend_score > 0
    
    # B. 回踩项 (Pullback Score) - 核心优化
    # 我们希望 z_distance 在 -2 到 +0.5 之间
    # 即：跌穿 MA20 两个标准差以内（深洗盘），或者刚到 MA20 上方（浅回调）
    # 使用高斯函数生成得分：距离 -0.5 (轻微跌破) 越近，得分越高
    target_z = -0.5 
    # 这里的 2.0 是带宽，控制对偏离的容忍度。波动率越大，容忍度天然越大(因为分母大)
    pullback_score = np.exp(-((z_distance - target_z) ** 2) / 2.0)
    
    # C. 活跃度项 (Activity): 
    # 我们希望波动率本身是放大的（活跃股），而不是死鱼股
    # 计算波动率的秩 (Rank)，越高越好
    vol_rank = vol_pct.groupby(data['date']).rank(pct=True)
    
    # 6. 综合因子
    # 逻辑：趋势向上 AND (是活跃股) AND (形态完美回踩)
    alpha = trend_score * vol_rank * pullback_score
    
    # 过滤掉趋势向下的噪音
    alpha = alpha.where(trend_mask, 0)
    
    # 再次做一次去极值和填充
    return alpha.replace([np.inf, -np.inf], np.nan)

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



regime_aware_reversal_alpha = register_factor(
    FactorDefinition(
        name='regime_aware_reversal_alpha',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=25,
        calculator=_regime_aware_reversal_alpha,
        description='状态感知的短期反转Alpha（趋势弱时反转更有效）',
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

volatility_structure_alpha = register_factor(
    FactorDefinition(
        name='volatility_structure_alpha',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=15,
        calculator=_volatility_structure_alpha,
        description='行业中性化的波动率结构Alpha（低波动 + 波动率下降）',
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


return_5d_cs_rank = register_factor(
    FactorDefinition(
        name='return_5d_cs_rank',
        required_columns={'stock_code', 'date', 'close'},
        lookback_days=10,
        calculator=_return_5d_cs_rank,
        description='5日累计收益的日截面排名（降序，1=当日最强）',
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




# ==============================================================================
# 2. 优化版 CPV：量价趋势相关性 (20日)
# ==============================================================================
def _cpv_factor_optimized(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    [优化版] CPV 量价趋势相关性 (20日) - Raw Value
    
    逻辑优化：
    1. 使用 Log Diff 替代 pct_change：成交量波动巨大，Log 处理后的相关性更稳健。
    2. 保留 NaN：不填充 0，防止刚上市股票干扰后续的 Global Z-Score 分布。
    3. 移除内部 Z-Score：输出原始相关系数 [-1, 1]，由外部统一标准化。
    """
    required_cols = ['close', 'volume']
    if not all(col in data.columns for col in required_cols):
        return pd.Series(np.nan, index=data.index)

    grouped = data.groupby('stock_code', sort=False)
    
    # 1. 计算对数变化率 (比 pct_change 更适合做相关性分析)
    # Log Return: log(P_t) - log(P_t-1)
    log_ret = np.log(data['close']) - np.log(grouped['close'].shift(1))
    
    # Log Volume Change: log(V_t) - log(V_t-1)
    # 使用 log1p 防止 volume=0 报错
    log_vol_chg = np.log1p(data['volume']) - np.log1p(grouped['volume'].shift(1))
    
    # 2. 清洗数据
    # 将 Inf (除零/爆量) 设为 NaN，而不是 0。让 rolling_corr 跳过这些异常点。
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    log_vol_chg = log_vol_chg.replace([np.inf, -np.inf], np.nan)
    
    # 3. 向量化计算滚动相关系数
    # 构造临时 DataFrame 利用 groupby.corr 加速
    temp_df = pd.DataFrame({
        'a': log_ret,
        'b': log_vol_chg,
        'stock_code': data['stock_code']
    }, index=data.index)
    
    def calc_rolling_corr(df):
        # min_periods=15: 保证至少有15个有效数据点才计算相关性，否则给 NaN
        return df['a'].rolling(20, min_periods=15).corr(df['b'])

    # 计算原始相关系数 [-1, 1]
    raw_corr = temp_df.groupby('stock_code', sort=False, group_keys=False).apply(calc_rolling_corr)
    
    # 4. 返回原始值 (保留 NaN)
    # 不做 fillna(0)，不做 Z-Score。
    # 确保索引对齐
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
    
    原逻辑问题：
    原 stable_momentum (Ret/Vol) 惩罚了高波动，而你的策略需要高波动。
    
    优化逻辑：
    ER = |总位移| / 总路程
       = |Close_t - Close_t-n| / Sum(Abs(Close_i - Close_i-1))
    
    含义：
    - 接近 1：极度流畅的单边趋势（主力控盘极强，不墨迹）。
    - 接近 0：无序震荡（多空分歧大，效率低）。
    - 这个因子不惩罚"暴涨"，只惩罚"反复震荡"，完美契合龙头战法。
    """
    # 1. 计算总位移 (Net Change)
    # 使用 10 日窗口，更贴近短线
    lookback = 10
    
    grouped = data.groupby('stock_code', sort=False)
    
    # change = close_t - close_t-n
    # 这里用 diff 即可，取绝对值表示位移幅度
    net_change = grouped['close'].diff(lookback).abs()
    
    # 2. 计算总路程 (Total Path)
    # 每天涨跌幅的绝对值之和
    daily_change = grouped['close'].diff(1).abs()
    total_path = daily_change.rolling(lookback, min_periods=lookback-2).sum()
    
    # 3. 计算效率 (Efficiency)
    # + 1e-9 防止除零
    er = net_change / (total_path + 1e-9)
    
    # 4. 加入方向修正 (Directional Adjustment)
    # 原始 ER 只有数值没有方向。我们将方向加回去。
    # 如果是下跌趋势的流畅，我们给负分；上涨趋势的流畅，给正分。
    # 用 10日涨跌幅符号来修正
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

__all__ = [
    'high_to_close_drawdown',
    'pct_chg_intra_z',
    'pct_chg_sector_mean',
    'pct_chg_1d',
    'pct_chg_2d',
    'return_3d_intra_z',
    'return_3d_sector_mean',
    'return_5d',
    'return_5d_cs_rank',
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

