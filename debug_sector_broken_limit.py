#!/usr/bin/env python3
"""
调试板块炸板股为0的确切原因
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sector_data():
    """加载一个股票较多的板块数据"""
    data_dir = Path("data/sector_analysis")
    
    if not data_dir.exists():
        logger.error("数据目录不存在")
        return None, None
    
    # 查找板块分析文件
    parquet_files = list(data_dir.glob("*_board_analysis.parquet"))
    
    if not parquet_files:
        logger.error("未找到板块分析文件")
        return None, None
    
    # 选择一个文件大小较大的（股票较多的板块）
    largest_file = max(parquet_files, key=lambda f: f.stat().st_size)
    sector_name = largest_file.stem.replace('_board_analysis', '')
    
    logger.info(f"选择板块: {sector_name}, 文件: {largest_file}")
    
    try:
        df = pd.read_parquet(largest_file)
        logger.info(f"板块数据形状: {df.shape}")
        logger.info(f"日期范围: {df['date'].min()} - {df['date'].max()}")
        
        return df, sector_name
    except Exception as e:
        logger.error(f"读取板块数据失败: {e}")
        return None, None

def load_stock_data_for_date(sector_name, date_str):
    """加载指定日期的个股数据"""
    try:
        # 尝试从数据加载器获取数据
        from to.data_loaders import get_stock_data_by_date
        
        stock_data = get_stock_data_by_date(date_str)
        if not stock_data:
            logger.warning(f"未找到日期 {date_str} 的股票数据")
            return None
            
        logger.info(f"日期 {date_str} 加载了 {len(stock_data)} 只股票数据")
        return stock_data
        
    except Exception as e:
        logger.error(f"加载股票数据失败: {e}")
        return None

def analyze_broken_limit_logic(stock_data, date_str):
    """分析炸板逻辑"""
    logger.info(f"=== 分析日期 {date_str} 的炸板逻辑 ===")
    
    if not stock_data:
        logger.warning("股票数据为空")
        return
    
    # 模拟SectorBoardAnalyzer的炸板检测逻辑
    def get_stock_limit_ratios(stock_code):
        """获取股票涨停比例"""
        if 'ST' in stock_code.upper():
            return 0.05, -0.05
        elif stock_code.startswith('688'):  # 科创板
            return 0.20, -0.20
        elif stock_code.startswith('300'):  # 创业板
            return 0.20, -0.20
        elif '.BJ' in stock_code:  # 北交所
            return 0.30, -0.30
        else:
            return 0.10, -0.10
    
    def is_broken_limit(stock_data, stock_code):
        """判断是否炸板"""
        try:
            pct_chg = stock_data.get('pct_chg', stock_data.get('change_percent', 0))
            close = stock_data.get('close', None)
            pre_close = stock_data.get('pre_close', None)
            high = stock_data.get('high', None)
            
            if pre_close is None or high is None or close is None:
                return False, "缺少关键价格数据"
                
            limit_up_ratio, _ = get_stock_limit_ratios(stock_code)
            
            # 计算理论涨停价
            theoretical_limit = pre_close * (1 + limit_up_ratio)
            rounded_limit = round(theoretical_limit, 2)
            
            # 判断条件
            is_high_reached_limit = abs(high - rounded_limit) < 0.005
            is_close_not_limit = abs(close - rounded_limit) >= 0.005
            
            is_broken = is_high_reached_limit and is_close_not_limit
            
            return is_broken, {
                'pct_chg': pct_chg,
                'close': close,
                'pre_close': pre_close,
                'high': high,
                'limit_ratio': limit_up_ratio,
                'theoretical_limit': theoretical_limit,
                'rounded_limit': rounded_limit,
                'high_reached_limit': is_high_reached_limit,
                'close_not_limit': is_close_not_limit
            }
            
        except Exception as e:
            return False, f"计算错误: {e}"
    
    # 分析所有股票
    total_stocks = len(stock_data)
    broken_stocks = []
    limit_up_stocks = []
    analysis_details = []
    
    for stock_code, data in stock_data.items():
        if not isinstance(data, dict):
            continue
            
        is_broken, details = is_broken_limit(data, stock_code)
        
        if is_broken:
            broken_stocks.append(stock_code)
            logger.info(f"发现炸板股: {stock_code}")
            logger.info(f"  详情: {details}")
        
        # 检查是否涨停
        if isinstance(details, dict):
            pct_chg = details.get('pct_chg', 0)
            limit_ratio = details.get('limit_ratio', 0.1)
            
            if abs(pct_chg / 100 - limit_ratio) < 0.002:  # 接近涨停
                limit_up_stocks.append(stock_code)
        
        analysis_details.append({
            'stock_code': stock_code,
            'is_broken': is_broken,
            'details': details
        })
    
    logger.info(f"分析结果:")
    logger.info(f"  总股票数: {total_stocks}")
    logger.info(f"  炸板股数: {len(broken_stocks)}")
    logger.info(f"  涨停股数: {len(limit_up_stocks)}")
    
    if len(broken_stocks) == 0:
        logger.warning("未发现任何炸板股，分析可能的原因:")
        
        # 检查数据完整性
        missing_fields = []
        for stock_code, data in list(stock_data.items())[:10]:  # 检查前10只股票
            required_fields = ['close', 'pre_close', 'high', 'pct_chg']
            for field in required_fields:
                if field not in data:
                    missing_fields.append(f"{stock_code}缺少{field}")
        
        if missing_fields:
            logger.error(f"数据字段缺失问题: {missing_fields[:5]}")  # 只显示前5个
        
        # 检查价格数据的合理性
        price_issues = []
        for stock_code, data in list(stock_data.items())[:10]:
            if all(field in data for field in ['close', 'pre_close', 'high']):
                close = data['close']
                pre_close = data['pre_close']
                high = data['high']
                
                if close <= 0 or pre_close <= 0 or high <= 0:
                    price_issues.append(f"{stock_code}价格数据异常")
                elif high < max(close, pre_close):
                    price_issues.append(f"{stock_code}最高价小于收盘价")
        
        if price_issues:
            logger.error(f"价格数据异常: {price_issues[:5]}")
        
        # 检查是否有接近涨停的股票
        near_limit_stocks = []
        for detail in analysis_details[:20]:  # 检查前20只
            if isinstance(detail['details'], dict):
                details = detail['details']
                pct_chg = details.get('pct_chg', 0)
                limit_ratio = details.get('limit_ratio', 0.1)
                
                if pct_chg / 100 > limit_ratio * 0.8:  # 涨幅超过涨停阈值的80%
                    near_limit_stocks.append({
                        'stock_code': detail['stock_code'],
                        'pct_chg': pct_chg,
                        'limit_ratio': limit_ratio * 100,
                        'details': details
                    })
        
        if near_limit_stocks:
            logger.info(f"发现 {len(near_limit_stocks)} 只接近涨停的股票:")
            for stock in near_limit_stocks[:3]:  # 显示前3只
                logger.info(f"  {stock['stock_code']}: 涨幅{stock['pct_chg']:.2f}%, 涨停线{stock['limit_ratio']:.1f}%")
                logger.info(f"    详情: {stock['details']}")
        else:
            logger.warning("连接近涨停的股票都没有，可能是:")
            logger.warning("1. 该日期市场整体表现较差")
            logger.warning("2. 数据源的涨跌幅计算方式与预期不符")
            logger.warning("3. 价格数据精度问题")
    
    return broken_stocks, analysis_details

def main():
    """主函数"""
    logger.info("=== 开始调试板块炸板股为0的问题 ===")

    # 1. 加载板块数据
    sector_df, sector_name = load_sector_data()
    if sector_df is None:
        logger.error("无法加载板块数据，退出")
        return

    # 2. 检查板块数据中的炸板相关字段
    logger.info("检查板块数据中的炸板相关字段...")
    logger.info(f"板块数据列名: {sector_df.columns.tolist()}")

    if 'broken_limit_up_count' in sector_df.columns:
        broken_stats = sector_df['broken_limit_up_count'].describe()
        logger.info(f"broken_limit_up_count 统计: {broken_stats}")

        # 检查是否有非零值
        non_zero_broken = sector_df[sector_df['broken_limit_up_count'] > 0]
        logger.info(f"有炸板的记录数: {len(non_zero_broken)}")

        if len(non_zero_broken) > 0:
            logger.info("发现有炸板记录，显示前几条:")
            logger.info(non_zero_broken[['date', 'broken_limit_up_count', 'limit_up_count']].head())
        else:
            logger.warning("所有记录的 broken_limit_up_count 都为0")

            # 检查是否有涨停股
            if 'limit_up_count' in sector_df.columns:
                limit_up_stats = sector_df['limit_up_count'].describe()
                logger.info(f"limit_up_count 统计: {limit_up_stats}")

                has_limit_up = sector_df[sector_df['limit_up_count'] > 0]
                logger.info(f"有涨停的记录数: {len(has_limit_up)}")

                if len(has_limit_up) > 0:
                    logger.info("有涨停但无炸板，可能的原因:")
                    logger.info("1. 数据源中没有炸板股票")
                    logger.info("2. 炸板检测逻辑有问题")
                    logger.info("3. 数据处理流程中丢失了炸板信息")

                    # 显示一些有涨停的记录
                    sample_records = has_limit_up[['date', 'limit_up_count', 'broken_limit_up_count']].head()
                    logger.info(f"有涨停的样本记录:\n{sample_records}")
                else:
                    logger.warning("连涨停股都没有，可能是数据源问题")
    else:
        logger.error("板块数据中没有 broken_limit_up_count 字段")

    # 3. 检查其他相关字段
    for field in ['yesterday_broken_limit_premium', 'broken_penalty_score', 'limit_open_count']:
        if field in sector_df.columns:
            stats = sector_df[field].describe()
            logger.info(f"{field} 统计: {stats}")
            non_zero = sector_df[sector_df[field] > 0]
            logger.info(f"{field} 非零记录数: {len(non_zero)}")
        else:
            logger.warning(f"板块数据中没有 {field} 字段")

    logger.info("=== 调试完成 ===")

if __name__ == "__main__":
    main()
