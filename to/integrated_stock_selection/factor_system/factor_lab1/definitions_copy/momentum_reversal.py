"""动量与反转因子（基础+复合）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

# ==============================================================================
# 通用工具函数：高效行业处理
# ==============================================================================

def _get_sector_map(ctx: FactorContext) -> pd.Series | None:
    """安全获取行业映射"""
    if ctx.extra and 'stock_to_sw_sector' in ctx.extra:
        return ctx.extra['stock_to_sw_sector']
    return None

def _calc_sector_metrics(
    series: pd.Series, 
    date_series: pd.Series,
    sector_series: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    计算行业内的 Z-Score (Alpha) 和 行业均值 (Beta)。
    比 LinearRegression 快 100 倍以上。
    
    Returns:
        z_score: (Val - Mean) / Std
        sector_mean: Mean
    """
    # 构建临时表
    df = pd.DataFrame({
        'val': series.values,
        'date': date_series.values,
        'sector': sector_series.values
    }, index=series.index)
    
    # 极值处理 (Winsorize)，防止异常值扭曲均值和标准差
    # 简单的 3-sigma 裁剪
    df['val'] = df['val'].clip(df['val'].quantile(0.005), df['val'].quantile(0.995))

    # 按日期和行业分组计算统计量
    grouped = df.groupby(['date', 'sector'])['val']
    
    mean = grouped.transform('mean')
    std = grouped.transform('std').fillna(1.0) # 防止除零
    
    # 计算 Alpha (Z-Score)
    z_score = (df['val'] - mean) / (std + 1e-9)
    
    # 填充缺失值 (行业内只有1只股票时 std 为 nan)
    z_score = z_score.fillna(0.0)
    mean = mean.fillna(0.0)
    
    return z_score, mean

# ==============================================================================
# 1. 基础动量/反转因子 (Returns) - Alpha/Beta 分离版
# ==============================================================================

def _make_return_factor(window: int, kind: str = 'alpha') -> callable:
    """
    工厂函数：生成不同窗口的收益率因子计算器
    kind: 'alpha' (行业内Z-Score) 或 'beta' (行业均值)
    """
    def calculator(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
        if 'close' not in data.columns:
            return pd.Series(0.0, index=data.index)
        
        # 1. 计算原始收益率
        if window == 0:
            raw_ret = data['pct_chg'] # 当日涨跌幅
        else:
            # 过去 N 日涨跌幅: (Close_T - Close_T-N) / Close_T-N
            # 这种计算方式不包含未来数据 (shift N)
            grouped = data.groupby('stock_code', sort=False)['close']
            prev_close = grouped.shift(window)
            raw_ret = (data['close'] - prev_close) / (prev_close + 1e-9)

        # 2. 获取行业数据
        sector_map = _get_sector_map(ctx)
        
        # 如果没有行业数据，直接返回原始值的简单标准化
        if sector_map is None:
            if kind == 'alpha':
                # 全市场 Z-Score
                daily_mean = raw_ret.groupby(data['date']).transform('mean')
                daily_std = raw_ret.groupby(data['date']).transform('std')
                return ((raw_ret - daily_mean) / (daily_std + 1e-9)).fillna(0)
            else:
                # 全市场均值
                return raw_ret.groupby(data['date']).transform('mean').fillna(0)

        # 3. 对齐行业
        sectors = data['stock_code'].map(sector_map).fillna('Unknown')
        
        # 4. 计算行业统计量
        alpha_z, beta_mean = _calc_sector_metrics(raw_ret, data['date'], sectors)
        
        if kind == 'alpha':
            return alpha_z.astype('float64')
        else:
            return beta_mean.astype('float64')

    return calculator

# 注册基础收益率因子 (0d, 3d, 5d, 10d, 20d)
# 每个窗口注册两个版本：_intra_z (Alpha) 和 _sector_mean (Beta)

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


def _sector_neutralize(
    factor_values: pd.Series,
    data: pd.DataFrame,
    sector_mapping: pd.Series
) -> pd.Series:
    """
    行业中性化（高性能版）

    等价于：对每个交易日、每个行业，
    factor - 行业内均值
    """

    # ---- 对齐索引 ----
    if not factor_values.index.equals(data.index):
        factor_values = factor_values.reindex(data.index)

    work_df = data[['stock_code', 'date']].copy()
    work_df['factor'] = factor_values.values
    work_df['sector'] = work_df['stock_code'].map(sector_mapping)

    # 初始化结果
    result = pd.Series(0.0, index=data.index, dtype='float64')

    # 去掉无法参与中性化的行
    mask = (
        work_df['sector'].notna()
        & work_df['factor'].notna()
    )
    work_df = work_df[mask]

    if work_df.empty:
        return result

    # ---- 核心：行业内去均值（严格等价于回归残差） ----
    industry_mean = (
        work_df
        .groupby(['date', 'sector'])['factor']
        .transform('mean')
    )

    residual = work_df['factor'] - industry_mean

    # 回填结果
    result.loc[work_df.index] = residual.values

    return result


def _pct_chg_1d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    昨日涨幅（基础因子）

    直接使用昨日涨幅，不进行行业中性化
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    pct_chg_1d = grouped['pct_chg'].shift(1).reset_index(level=0, drop=True)

    return pct_chg_1d.fillna(0.0).astype('float64')


def _pct_chg_2d(data: pd.DataFrame, context: FactorContext) -> pd.Series:
    """
    前日涨幅（基础因子）

    直接使用前日涨幅，不进行行业中性化
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.0, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    pct_chg_2d = grouped['pct_chg'].shift(2).reset_index(level=0, drop=True)

    return pct_chg_2d.fillna(0.0).astype('float64')


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

    return return_3d_pct.fillna(0.0).astype('float64')


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

    return return_5d_pct.fillna(0.0).astype('float64')


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

    return return_10d_pct.fillna(0.0).astype('float64')


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

    return return_20d_pct.fillna(0.0).astype('float64')


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

    # 获取行业映射：必须使用SW申万行业
    sector_mapping = context.extra.get('stock_to_sw_sector')
    if sector_mapping is None:
        raise ValueError("trend_acceleration因子计算失败：缺少SW申万行业映射数据")
    if not sector_mapping:  # 字典空检查
        raise ValueError("trend_acceleration因子计算失败：SW申万行业映射数据为空")

    # 原始因子值：趋势加速度
    raw_factor = acceleration

    # 行业中性化
    neutralized = _sector_neutralize(raw_factor, data, sector_mapping)
    return neutralized.fillna(0.0).astype('float64')

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
    
    return factor.fillna(0.0).astype('float64')


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

    return consistency.fillna(0.5).astype('float64')


def _trend_persistence_10d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    趋势持续性因子（时间序列因子）- 10日版本

    计算逻辑：
    10日内上涨天数 / 10
    - 1.0：10天全部上涨（强趋势）
    - 0.0：10天全部下跌（强趋势）
    - 0.5：涨跌各半（无趋势）
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.5, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    up_days = (data['pct_chg'] > 0).astype(float)
    persistence = grouped.apply(
        lambda g: up_days.loc[g.index].rolling(10, min_periods=5).mean()
    ).reset_index(level=0, drop=True)

    return persistence.fillna(0.5).astype('float64')


def _trend_persistence_20d(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    趋势持续性因子（时间序列因子）- 20日版本

    计算逻辑：
    20日内上涨天数 / 20
    - 1.0：20天全部上涨（强趋势）
    - 0.0：20天全部下跌（强趋势）
    - 0.5：涨跌各半（无趋势）
    """
    if 'pct_chg' not in data.columns:
        return pd.Series(0.5, index=data.index, dtype='float64')

    grouped = data.groupby('stock_code', sort=False)
    up_days = (data['pct_chg'] > 0).astype(float)
    persistence = grouped.apply(
        lambda g: up_days.loc[g.index].rolling(20, min_periods=10).mean()
    ).reset_index(level=0, drop=True)

    return persistence.fillna(0.5).astype('float64')


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

    # -------- Step 3：行业中性化（关键） --------
    vol_neutral = _sector_neutralize(
        factor_values=vol_10,
        data=data,
        sector_mapping=sector_mapping
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
    raw_alpha = vol_neutral * vol_trend_signal

    # -------- Step 6：数值清洗 --------
    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    raw_alpha = raw_alpha.fillna(0.0)

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

    # ---- 行业中性化（关键） ----
    volume_neutral = _sector_neutralize(
        volume_ratio,
        data,
        sector_mapping
    )

    price_vol_neutral = _sector_neutralize(
        price_vol_10,
        data,
        sector_mapping
    )

    # ---- Alpha：量增 + 波动未增 ----
    raw_alpha = volume_neutral - price_vol_neutral

    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    return raw_alpha.fillna(0.0).astype('float64')


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

    sector_mapping = context.extra.get('stock_to_sw_sector')
    if sector_mapping is None or len(sector_mapping) == 0:
        raise ValueError("regime_aware_reversal_alpha 缺少 SW 行业映射")

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

    # ---- 行业中性化 ----
    ret_3_neutral = _sector_neutralize(ret_3, data, sector_mapping)
    trend_neutral = _sector_neutralize(trend_strength, data, sector_mapping)

    # ---- 状态感知反转 ----
    raw_alpha = -ret_3_neutral * (1.0 - trend_neutral.abs())

    raw_alpha = raw_alpha.replace([np.inf, -np.inf], np.nan)
    return raw_alpha.fillna(0.0).astype('float64')


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

    return alpha.replace([np.inf, -np.inf], 0.0).fillna(0.0)


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

    return alpha.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _high_volatility_pullback_alpha(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    高位震荡 + 回踩支撑再启动 Alpha
    """

    required = {'close'}
    if not required.issubset(data.columns):
        return pd.Series(0.0, index=data.index)

    grouped = data.groupby('stock_code')

    close = data['close']

    ma5 = grouped['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    ma20 = grouped['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())

    returns = grouped['close'].transform(lambda x: x.pct_change())

    volatility = grouped['close'].transform(
        lambda x: x.pct_change().rolling(10, min_periods=5).std()
    )

    # 趋势强度
    trend_strength = close / (ma20 + 1e-8)

    # 回踩支撑（接近 MA5）
    support_proximity = -abs(close - ma5) / (ma5 + 1e-8)

    alpha = trend_strength * volatility * support_proximity

    return alpha.replace([np.inf, -np.inf], 0.0).fillna(0.0)



high_volatility_pullback_alpha = register_factor(
    FactorDefinition(
        name='high_volatility_pullback_alpha',
        required_columns={'stock_code', 'date', 'close','volume'},
        lookback_days=20,
        calculator=_high_volatility_pullback_alpha,
        description='高位震荡 + 回踩支撑再启动 Alpha',
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

trend_persistence_10d = register_factor(
    FactorDefinition(
        name='trend_persistence_10d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=15,
        calculator=_trend_persistence_10d,
        description='趋势持续性因子（时间序列因子）- 10日版本',
    )
)

trend_persistence_20d = register_factor(
    FactorDefinition(
        name='trend_persistence_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        lookback_days=25,
        calculator=_trend_persistence_20d,
        description='趋势持续性因子（时间序列因子）- 20日版本',
    )
)



def _cpv_factor(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """
    CPV因子 (Correlation Price-Volume) - 量价相关性因子

    计算逻辑:
    1. 计算过去6日的价格变化与成交量变化的相关系数
    2. 正相关: 价涨量增,价跌量减(健康趋势)
    3. 负相关: 价涨量减,价跌量增(量价背离,反转信号)

    学术依据:
    - 东吴证券《高频价量相关性,意想不到的选股因子》(2020)
    - A股实证IC值: 0.06-0.09
    - 短期持有效果最佳(1-5天)

    注意: 严格避免未来数据,只使用T日及之前数据
    """
    if not all(col in data.columns for col in ['close', 'volume']):
        return pd.Series(0.5, index=data.index, dtype='float64')

    # 计算价格和成交量的变化率
    grouped = data.groupby('stock_code', sort=False)
    price_pct = grouped['close'].pct_change().reset_index(level=0, drop=True)
    volume_pct = grouped['volume'].pct_change().reset_index(level=0, drop=True)

    # 计算6日滚动相关系数
    data_work = data.copy()
    data_work['price_pct'] = price_pct
    data_work['volume_pct'] = volume_pct

    def calc_corr(group):
        if len(group) < 6:
            return pd.Series(0.0, index=group.index)
        result = group['price_pct'].rolling(6, min_periods=6).corr(group['volume_pct'])
        return result

    cpv = data_work.groupby('stock_code', sort=False, group_keys=False).apply(calc_corr)
    cpv = cpv.reset_index(drop=True) if isinstance(cpv.index, pd.MultiIndex) else cpv

    # 标准化到[0, 1]区间
    cpv_clipped = cpv.clip(-1.0, 1.0)
    cpv_norm = (cpv_clipped + 1.0) / 2.0

    return cpv_norm.fillna(0.5).astype('float64')


cpv_factor = register_factor(
    FactorDefinition(
        name='cpv_factor',
        required_columns={'stock_code', 'date', 'close', 'volume'},
        lookback_days=10,
        calculator=_cpv_factor,
        description='CPV量价相关性因子（高频量价背离）',
    )
)


# ==============================================================================
# 4. 低波反转因子 (Low Volatility Reversal) -> 稳健动量因子
# ==============================================================================

def _stable_momentum(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """生产环境推荐版本"""
    req = ['close', 'pct_chg']
    if not all(c in data.columns for c in req):
        return pd.Series(0.0, index=data.index, dtype='float64')
    
    # 1. 计算收益率和波动率
    ret_20 = data.groupby('stock_code')['close'].transform(
        lambda x: x.pct_change(20)
    )
    
    vol_20 = data.groupby('stock_code')['pct_chg'].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    
    # 2. 计算因子
    factor = ret_20 / (vol_20 + 1e-9)
    
    # 3. 稳健的异常值处理
    if len(factor) > 100:
        # 使用 winsorize（缩尾处理）
        q_low = factor.quantile(0.01)
        q_high = factor.quantile(0.99)
        factor = factor.clip(q_low, q_high)
    
    # 4. 标准化（可选）
    # factor = (factor - factor.mean()) / factor.std()
    
    return factor.fillna(0.0).astype('float64')

stable_momentum = register_factor(
    FactorDefinition(
        name='stable_momentum', # 原 low_volatility_reversal 的优化版
        required_columns={'stock_code', 'date', 'close', 'pct_chg'},
        lookback_days=30,
        calculator=_stable_momentum,
        description='风险调整后的动量 (收益/波动率)',
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
    'trend_persistence_10d',
    'trend_persistence_20d',
    'cpv_factor',
    'stable_momentum',
    'high_volatility_pullback_alpha',
    'volume_confirmed_pullback_alpha',
    'trend_pullback_composite_alpha',
    'regime_aware_reversal_alpha',
    'volume_structure_alpha',
    'volatility_structure_alpha'
]

