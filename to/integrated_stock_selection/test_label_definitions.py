"""
测试不同标签定义下的IC值

对比以下标签定义:
1. future_return_1d_close: T+1开盘买,T+2收盘卖 (当前使用)
2. future_return_1d_close_v2: T日收盘买,T+1收盘卖 (文献常用)
3. future_return_2d_close: T+1开盘买,T+3收盘卖
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入主程序的配置和函数
try:
    from main import (
        _load_factor_cache_data,
        _generate_training_labels_for_all_data,
        ATR_MULTIPLIER_PROFIT,
        ATR_MULTIPLIER_LOSS,
        MAX_HOLDING_DAYS
    )
    logger.info("✅ 成功导入主程序模块")
except ImportError as e:
    logger.error(f"❌ 导入主程序模块失败: {e}")
    logger.error("   请确保在integrated_stock_selection目录下运行此脚本")
    sys.exit(1)


def calculate_alternative_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算不同定义的标签
    
    Args:
        data: 包含基础行情数据的DataFrame
    
    Returns:
        DataFrame: 包含多种标签定义的数据
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 计算不同定义的标签")
    logger.info("=" * 80)
    
    result = data.copy()
    
    # 标签1: future_return_1d_close (T+1开盘买,T+2收盘卖) - 已存在
    logger.info("\n标签1: future_return_1d_close (T+1开盘买,T+2收盘卖)")
    if 'future_return_1d_close' in result.columns:
        logger.info(f"  ✅ 已存在,有效值: {result['future_return_1d_close'].notna().sum()}")
    else:
        logger.error("  ❌ 不存在!")
    
    # 标签2: future_return_1d_close_v2 (T日收盘买,T+1收盘卖)
    logger.info("\n标签2: future_return_1d_close_v2 (T日收盘买,T+1收盘卖)")
    for stock_code in result['stock_code'].unique():
        stock_mask = result['stock_code'] == stock_code
        stock_indices = result[stock_mask].index
        
        close_prices = result.loc[stock_indices, 'close'].values
        
        # 计算T日收盘买,T+1收盘卖的收益
        future_returns = np.full(len(close_prices), np.nan)
        for i in range(len(close_prices) - 1):
            t0_close = close_prices[i]
            t1_close = close_prices[i + 1]
            
            if not np.isnan(t0_close) and t0_close > 0 and not np.isnan(t1_close) and t1_close > 0:
                future_returns[i] = (t1_close - t0_close) / t0_close
        
        result.loc[stock_indices, 'future_return_1d_close_v2'] = future_returns
    
    logger.info(f"  ✅ 计算完成,有效值: {result['future_return_1d_close_v2'].notna().sum()}")
    
    # 标签3: future_return_2d_close (T+1开盘买,T+3收盘卖)
    logger.info("\n标签3: future_return_2d_close (T+1开盘买,T+3收盘卖)")
    for stock_code in result['stock_code'].unique():
        stock_mask = result['stock_code'] == stock_code
        stock_indices = result[stock_mask].index
        
        open_prices = result.loc[stock_indices, 'open'].values
        close_prices = result.loc[stock_indices, 'close'].values
        
        # 计算T+1开盘买,T+3收盘卖的收益
        future_returns = np.full(len(close_prices), np.nan)
        for i in range(len(close_prices) - 3):
            t1_open = open_prices[i + 1]
            t3_close = close_prices[i + 3]
            
            if not np.isnan(t1_open) and t1_open > 0 and not np.isnan(t3_close) and t3_close > 0:
                future_returns[i] = (t3_close - t1_open) / t1_open
        
        result.loc[stock_indices, 'future_return_2d_close'] = future_returns
    
    logger.info(f"  ✅ 计算完成,有效值: {result['future_return_2d_close'].notna().sum()}")
    
    return result


def calculate_ic_for_labels(data: pd.DataFrame, sample_factors: list = None) -> dict:
    """
    计算不同标签定义下的IC值
    
    Args:
        data: 包含因子和标签的DataFrame
        sample_factors: 要测试的因子列表
    
    Returns:
        dict: 各标签定义下的IC值
    """
    if sample_factors is None:
        sample_factors = [
            'pct_chg_0d', 'pct_chg_1d', 'pct_chg_2d',
            'return_5d', 'return_10d',
            'alpha_002', 'alpha_017',
            'cpv_factor', 'low_volatility_reversal',
            'max_rally_20d', 'max_drawdown_20d',
            'volatility_change_rate', 'distance_from_52w_high'
        ]
    
    # 过滤存在的因子
    sample_factors = [f for f in sample_factors if f in data.columns]
    
    # 标签列表
    label_columns = [
        'future_return_1d_close',
        'future_return_1d_close_v2',
        'future_return_2d_close'
    ]
    
    results = {}
    
    for label_col in label_columns:
        if label_col not in data.columns:
            logger.warning(f"⚠️ 标签 {label_col} 不存在,跳过")
            continue
        
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 标签: {label_col}")
        logger.info("=" * 80)
        
        label_results = {}
        
        for factor in sample_factors:
            # 计算IC
            valid_mask = data[factor].notna() & data[label_col].notna()
            if valid_mask.sum() < 100:
                continue
            
            ic = data.loc[valid_mask, factor].corr(data.loc[valid_mask, label_col])
            
            # 计算RankIC
            rank_ic = data.loc[valid_mask, factor].corr(
                data.loc[valid_mask, label_col],
                method='spearman'
            )
            
            label_results[factor] = {
                'IC': ic,
                'RankIC': rank_ic,
                'N': valid_mask.sum()
            }
            
            status = "✅" if ic > 0 else "❌"
            logger.info(f"   {status} {factor:30s} IC={ic:+.4f}   RankIC={rank_ic:+.4f}   N={valid_mask.sum():,}")
        
        # 汇总统计
        ic_values = [v['IC'] for v in label_results.values()]
        if ic_values:
            logger.info(f"\n汇总统计:")
            logger.info(f"   正IC因子: {sum(1 for ic in ic_values if ic > 0)}/{len(ic_values)} ({sum(1 for ic in ic_values if ic > 0)/len(ic_values)*100:.1f}%)")
            logger.info(f"   负IC因子: {sum(1 for ic in ic_values if ic < 0)}/{len(ic_values)} ({sum(1 for ic in ic_values if ic < 0)/len(ic_values)*100:.1f}%)")
            logger.info(f"   平均IC: {np.mean(ic_values):.4f}")
            logger.info(f"   IC标准差: {np.std(ic_values):.4f}")
            logger.info(f"   平均|IC|: {np.mean(np.abs(ic_values)):.4f}")
        
        results[label_col] = label_results
    
    return results


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("🔍 测试不同标签定义下的IC值")
    logger.info("=" * 80)
    
    # 加载因子数据并生成标签
    logger.info("\n步骤1: 加载因子数据")
    data = _load_factor_cache_data(
        include_label_generation_features=True,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if data is None or len(data) == 0:
        logger.error("❌ 因子数据加载失败")
        return
    
    logger.info(f"✅ 因子数据加载成功: {len(data):,} 行")
    
    # 生成标签
    logger.info("\n步骤2: 生成标签")
    labeled_result = _generate_training_labels_for_all_data(
        data,
        atr_multiplier_profit=ATR_MULTIPLIER_PROFIT,
        atr_multiplier_loss=ATR_MULTIPLIER_LOSS,
        max_holding_days=MAX_HOLDING_DAYS,
        debug=False,
        is_training=True,
        label_stats=None,
        winsorize_thresholds=None
    )
    
    if labeled_result is None:
        logger.error("❌ 标签生成失败")
        return
    
    # 解包结果
    if isinstance(labeled_result, tuple):
        data, label_stats = labeled_result
    else:
        data = labeled_result
    
    logger.info(f"✅ 标签生成成功")
    
    # 计算其他标签定义
    logger.info("\n步骤3: 计算其他标签定义")
    data = calculate_alternative_labels(data)
    
    # 计算IC
    logger.info("\n步骤4: 计算不同标签定义下的IC值")
    results = calculate_ic_for_labels(data)
    
    # 对比分析
    logger.info("\n" + "=" * 80)
    logger.info("📊 对比分析")
    logger.info("=" * 80)
    
    # 找出表现最好的标签定义
    for factor in ['alpha_002', 'cpv_factor', 'return_5d', 'return_10d']:
        logger.info(f"\n因子: {factor}")
        for label_col in results.keys():
            if factor in results[label_col]:
                ic = results[label_col][factor]['IC']
                rank_ic = results[label_col][factor]['RankIC']
                logger.info(f"  {label_col:30s} IC={ic:+.4f}   RankIC={rank_ic:+.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 测试完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

