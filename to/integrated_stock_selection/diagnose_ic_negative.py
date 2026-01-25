"""
IC为负值诊断脚本

用于诊断因子IC为负的可能原因:
1. 标签数据质量检查
2. 因子与标签的相关性分析
3. 时间对齐检查
4. 数据泄露检查
5. 因子方向检查

注意: 本系统使用future_return_1d_close作为原始收益率标签
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


def load_factor_and_label_data(start_date='2024-01-01', end_date='2024-12-31'):
    """
    加载因子数据并生成标签

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame: 包含因子和标签的完整数据
    """
    try:
        logger.info("=" * 80)
        logger.info("📊 步骤1: 加载因子数据")
        logger.info("=" * 80)

        # 加载因子数据(包含标签生成所需字段)
        factor_data = _load_factor_cache_data(
            include_label_generation_features=True,
            start_date=start_date,
            end_date=end_date
        )

        if factor_data is None or len(factor_data) == 0:
            logger.error("❌ 因子数据加载失败")
            return None

        logger.info(f"✅ 因子数据加载成功: {len(factor_data):,} 行, {len(factor_data.columns)} 列")
        logger.info(f"   日期范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")
        logger.info(f"   股票数量: {factor_data['stock_code'].nunique()}")

        # 生成标签(主要是为了生成future_return_1d_close)
        logger.info("\n" + "=" * 80)
        logger.info("📊 步骤2: 生成标签(future_return_1d_close)")
        logger.info("=" * 80)
        logger.info(f"   标签类型: future_return_1d_close (T+1开盘买入,T+2收盘卖出)")

        labeled_result = _generate_training_labels_for_all_data(
            factor_data,
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
            return None

        # 解包结果
        if isinstance(labeled_result, tuple):
            data, label_stats = labeled_result
        else:
            data = labeled_result
            label_stats = None

        logger.info(f"✅ 标签生成成功")
        logger.info(f"   总样本: {len(data):,}")

        # 检查future_return_1d_close
        if 'future_return_1d_close' in data.columns:
            logger.info(f"   future_return_1d_close有效值: {data['future_return_1d_close'].notna().sum():,}")
        else:
            logger.error("❌ 标签数据缺少future_return_1d_close字段")
            return None

        return data

    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def check_label_quality(data: pd.DataFrame):
    """检查标签数据质量"""
    logger.info("\n" + "="*80)
    logger.info("📊 步骤3: 标签数据质量检查")
    logger.info("="*80)

    # 使用future_return_1d_close作为标签
    label_col = 'future_return_1d_close'

    if label_col not in data.columns:
        logger.error(f"❌ 标签列 {label_col} 不存在!")
        logger.info(f"   可用列: {data.columns.tolist()}")
        return
    
    # 基础统计
    logger.info(f"\n1. 标签基础统计:")
    logger.info(f"   总样本数: {len(data)}")
    logger.info(f"   NaN数量: {data[label_col].isna().sum()} ({data[label_col].isna().sum()/len(data)*100:.2f}%)")
    logger.info(f"   有效样本: {data[label_col].notna().sum()}")
    
    # 分布统计
    valid_labels = data[label_col].dropna()
    if len(valid_labels) > 0:
        logger.info(f"\n2. 标签分布统计:")
        logger.info(f"   均值: {valid_labels.mean():.6f}")
        logger.info(f"   中位数: {valid_labels.median():.6f}")
        logger.info(f"   标准差: {valid_labels.std():.6f}")
        logger.info(f"   最小值: {valid_labels.min():.6f}")
        logger.info(f"   最大值: {valid_labels.max():.6f}")
        logger.info(f"   25%分位: {valid_labels.quantile(0.25):.6f}")
        logger.info(f"   75%分位: {valid_labels.quantile(0.75):.6f}")
        
        # 正负样本比例
        positive = (valid_labels > 0).sum()
        negative = (valid_labels < 0).sum()
        zero = (valid_labels == 0).sum()
        logger.info(f"\n3. 正负样本分布:")
        logger.info(f"   正收益: {positive} ({positive/len(valid_labels)*100:.2f}%)")
        logger.info(f"   负收益: {negative} ({negative/len(valid_labels)*100:.2f}%)")
        logger.info(f"   零收益: {zero} ({zero/len(valid_labels)*100:.2f}%)")
        
        # 异常值检查
        extreme_positive = (valid_labels > 0.1).sum()
        extreme_negative = (valid_labels < -0.1).sum()
        logger.info(f"\n4. 异常值检查:")
        logger.info(f"   >10%收益: {extreme_positive} ({extreme_positive/len(valid_labels)*100:.2f}%)")
        logger.info(f"   <-10%收益: {extreme_negative} ({extreme_negative/len(valid_labels)*100:.2f}%)")


def check_factor_label_correlation(data: pd.DataFrame, sample_factors: list = None):
    """检查因子与标签的相关性"""
    logger.info("\n" + "="*80)
    logger.info("📊 步骤4: 因子与标签相关性分析")
    logger.info("="*80)

    # 使用future_return_1d_close作为标签
    label_col = 'future_return_1d_close'

    if label_col not in data.columns:
        logger.error(f"❌ 标签列 {label_col} 不存在!")
        return
    
    # 默认检查的因子
    if sample_factors is None:
        sample_factors = [
            'pct_chg_0d', 'pct_chg_1d', 'pct_chg_2d',
            'return_5d', 'return_10d',
            'alpha_001', 'alpha_002', 'alpha_017',
            'cpv_factor', 'low_volatility_reversal',
            'max_drawdown_20d', 'max_rally_20d',
            'volatility_change_rate', 'distance_from_52w_high'
        ]
    
    # 过滤存在的因子
    available_factors = [f for f in sample_factors if f in data.columns]
    
    if not available_factors:
        logger.warning("⚠️ 没有可用的因子进行分析")
        return
    
    logger.info(f"\n检查 {len(available_factors)} 个因子的IC值:\n")
    
    ic_results = []
    for factor in available_factors:
        try:
            # 计算IC
            valid_mask = data[factor].notna() & data[label_col].notna()
            if valid_mask.sum() < 100:
                logger.warning(f"   {factor}: 有效样本不足 ({valid_mask.sum()})")
                continue
            
            ic = data.loc[valid_mask, factor].corr(data.loc[valid_mask, label_col])
            
            # 计算Rank IC
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(
                data.loc[valid_mask, factor],
                data.loc[valid_mask, label_col]
            )
            
            ic_results.append({
                'factor': factor,
                'ic': ic,
                'rank_ic': rank_ic,
                'samples': valid_mask.sum()
            })
            
            # 显示结果
            ic_str = f"{ic:+.4f}"
            rank_ic_str = f"{rank_ic:+.4f}"
            status = "✅" if ic > 0 else "❌"
            logger.info(f"   {status} {factor:30s} IC={ic_str:8s}  RankIC={rank_ic_str:8s}  N={valid_mask.sum():,}")
            
        except Exception as e:
            logger.error(f"   ❌ {factor}: 计算失败 - {e}")
    
    # 汇总统计
    if ic_results:
        ic_values = [r['ic'] for r in ic_results]
        positive_ic = sum(1 for ic in ic_values if ic > 0)
        negative_ic = sum(1 for ic in ic_values if ic < 0)
        
        logger.info(f"\n汇总统计:")
        logger.info(f"   正IC因子: {positive_ic}/{len(ic_results)} ({positive_ic/len(ic_results)*100:.1f}%)")
        logger.info(f"   负IC因子: {negative_ic}/{len(ic_results)} ({negative_ic/len(ic_results)*100:.1f}%)")
        logger.info(f"   平均IC: {np.mean(ic_values):.4f}")
        logger.info(f"   IC标准差: {np.std(ic_values):.4f}")


def check_time_alignment(data: pd.DataFrame):
    """检查时间对齐"""
    logger.info("\n" + "="*80)
    logger.info("📊 步骤5: 时间对齐检查")
    logger.info("="*80)

    # 使用future_return_1d_close作为标签
    label_col = 'future_return_1d_close'

    if label_col not in data.columns or 'pct_chg_0d' not in data.columns:
        logger.error("❌ 缺少必要的列")
        return
    
    # 随机抽取一只股票查看
    sample_stock = data['stock_code'].iloc[0]
    stock_data = data[data['stock_code'] == sample_stock].sort_values('date').head(10)

    logger.info(f"\n示例股票: {sample_stock}")
    logger.info(f"前10个交易日的数据:\n")

    display_cols = ['date', 'pct_chg_0d', label_col]
    if 'open' in stock_data.columns and 'close' in stock_data.columns:
        display_cols.extend(['open', 'close'])

    logger.info(stock_data[display_cols].to_string(index=False))

    # 检查标签是否正确shift
    logger.info(f"\n检查标签计算逻辑:")
    logger.info(f"   本系统使用future_return_1d_close作为标签")
    logger.info(f"   标签计算: (T+2收盘价 - T+1开盘价) / T+1开盘价")
    logger.info(f"   T日因子预测T日标签,T日标签基于T+1开盘买入,T+2收盘卖出的收益")


def check_data_leakage(data: pd.DataFrame):
    """检查数据泄露"""
    logger.info("\n" + "="*80)
    logger.info("📊 步骤6: 数据泄露检查")
    logger.info("="*80)

    if 'pct_chg_0d' not in data.columns:
        logger.error("❌ 缺少pct_chg_0d列")
        return
    
    # 创建T+1日收益率
    data_sorted = data.sort_values(['stock_code', 'date'])
    data_sorted['next_day_return'] = data_sorted.groupby('stock_code')['pct_chg_0d'].shift(-1)

    # 检查T日因子与T+1日收益的相关性
    test_factors = ['pct_chg_0d', 'pct_chg_1d', 'return_5d', 'return_10d']
    test_factors = [f for f in test_factors if f in data_sorted.columns]

    logger.info(f"\n检查因子与T+1日收益的相关性(应该接近0):\n")

    for factor in test_factors:
        valid_mask = data_sorted[factor].notna() & data_sorted['next_day_return'].notna()
        if valid_mask.sum() < 100:
            continue

        ic_leak = data_sorted.loc[valid_mask, factor].corr(
            data_sorted.loc[valid_mask, 'next_day_return']
        )

        status = "✅" if abs(ic_leak) < 0.05 else "⚠️"
        logger.info(f"   {status} {factor:30s} IC={ic_leak:+.4f}")

        if abs(ic_leak) > 0.1:
            logger.warning(f"      ⚠️ 可能存在数据泄露!")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("🔍 IC为负值诊断工具")
    logger.info("   本系统使用future_return_1d_close作为标签")
    logger.info("="*80)

    # 加载因子数据并生成标签
    # 默认分析最近一年的数据
    data = load_factor_and_label_data(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    if data is None:
        logger.error("❌ 数据加载失败,诊断终止")
        return

    # 执行诊断
    check_label_quality(data)
    check_factor_label_correlation(data)
    check_time_alignment(data)
    check_data_leakage(data)

    logger.info("\n" + "="*80)
    logger.info("✅ 诊断完成")
    logger.info("="*80)
    logger.info("\n💡 诊断建议:")
    logger.info("   1. 如果IC值普遍为负,可能是因子方向与标签相反")
    logger.info("   2. 如果IC值接近0,可能是因子与收益无关")
    logger.info("   3. 如果存在数据泄露,需要检查因子计算逻辑")
    logger.info("   4. 本系统使用future_return_1d_close,IC应该在0.05-0.15之间为正常")


if __name__ == '__main__':
    main()

