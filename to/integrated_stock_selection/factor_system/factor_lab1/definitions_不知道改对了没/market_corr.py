"""
个股与市场交互因子（Market Interaction）。
关注：个股对大盘的敏感度(Beta)和相对大盘的独立性(Alpha/RS)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

def _get_market_data(ctx: FactorContext) -> pd.DataFrame:
    return ctx.require_market().sort_values('date').reset_index(drop=True)

# ==============================================================================
# 1. 滚动 Beta (Rolling Beta) - 市场敏感度
# ==============================================================================
def _rolling_beta(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    60日滚动 Beta
    
    逻辑：
    Cov(Stock_Ret, Market_Ret) / Var(Market_Ret)
    
    含义：
    - > 1: 进攻型，大盘涨它涨更多，大盘跌它跌更惨。
    - < 1: 防御型。
    """
    market = _get_market_data(ctx)
    
    # 映射市场收益
    # 构造 date -> mkt_ret 的 Series
    mkt_map = pd.Series(market['daily_change_pct'].values, index=market['date'])
    
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_map)
    
    # 计算协方差和方差
    grouped = df.groupby('stock_code', sort=False)
    
    # Rolling Covariance
    cov = grouped[['pct_chg', 'mkt_ret']].rolling(60, min_periods=30).cov().unstack().iloc[:, 1]
    
    # Rolling Variance (Market)
    # 注意：所有股票同期的 Market Var 是一样的，可以优化，但直接算也无妨
    var = grouped['mkt_ret'].rolling(60, min_periods=30).var()
    
    beta = cov / (var + 1e-9)
    
    # return beta.reset_index(level=0, drop=True).fillna(1.0).astype('float64')
    return beta.reset_index(level=0, drop=True).astype('float64')

beta_60d = register_factor(
    FactorDefinition(
        name='beta_60d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_market_data=True,
        lookback_days=80,
        calculator=_rolling_beta,
        description='60日市场贝塔系数（系统性风险敏感度）',
    )
)


# ==============================================================================
# 1. 滚动 Beta (Rolling Beta) - 市场敏感度
# ==============================================================================
def _rolling_beta20(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    60日滚动 Beta
    
    逻辑：
    Cov(Stock_Ret, Market_Ret) / Var(Market_Ret)
    
    含义：
    - > 1: 进攻型，大盘涨它涨更多，大盘跌它跌更惨。
    - < 1: 防御型。
    """
    market = _get_market_data(ctx)
    
    # 映射市场收益
    # 构造 date -> mkt_ret 的 Series
    mkt_map = pd.Series(market['daily_change_pct'].values, index=market['date'])
    
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_map)
    
    # 计算协方差和方差
    grouped = df.groupby('stock_code', sort=False)
    
    # Rolling Covariance
    cov = grouped[['pct_chg', 'mkt_ret']].rolling(20, min_periods=15).cov().unstack().iloc[:, 1]
    
    # Rolling Variance (Market)
    # 注意：所有股票同期的 Market Var 是一样的，可以优化，但直接算也无妨
    var = grouped['mkt_ret'].rolling(20, min_periods=15).var()
    
    beta = cov / (var + 1e-9)
    
    # return beta.reset_index(level=0, drop=True).fillna(1.0).astype('float64')
    return beta.reset_index(level=0, drop=True).astype('float64')

beta_20d = register_factor(
    FactorDefinition(
        name='beta_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_market_data=True,
        lookback_days=40,
        calculator=_rolling_beta20,
        description='20日市场贝塔系数（系统性风险敏感度）',
    )
)

# ==============================================================================
# 2. 相对强度 (Relative Strength, RS) - 真正的 Alpha
# ==============================================================================
def _relative_strength_rs(data: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """
    曼斯菲尔德相对强度 (Mansfield Relative Strength) 简化版
    
    逻辑：
    Stock_Close / Market_Close 的 N日 均线乖离。
    或者更简单：CumRet_Stock - CumRet_Market (N日)
    
    这里计算：20日超额收益
    """
    market = _get_market_data(ctx)
    mkt_map = pd.Series(market['daily_change_pct'].values, index=market['date'])
    
    df = data[['date', 'pct_chg', 'stock_code']].copy()
    df['mkt_ret'] = df['date'].map(mkt_map)
    
    # 超额收益
    df['alpha'] = df['pct_chg'] - df['mkt_ret']
    
    # 20日累计超额收益
    rs = df.groupby('stock_code', sort=False)['alpha'].rolling(20, min_periods=10).sum()
    
    # return rs.reset_index(level=0, drop=True).fillna(0.0).astype('float64')
    return rs.reset_index(level=0, drop=True).astype('float64')

relative_strength_20d = register_factor(
    FactorDefinition(
        name='relative_strength_20d',
        required_columns={'stock_code', 'date', 'pct_chg'},
        requires_market_data=True,
        lookback_days=30,
        calculator=_relative_strength_rs,
        description='20日累计超额收益（相对大盘的独立强度）',
    )
)


__all__ = [
    'relative_strength_20d',
    'beta_60d',
    'beta_20d'
]
