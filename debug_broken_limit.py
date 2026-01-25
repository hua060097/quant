#!/usr/bin/env python3
"""
调试炸板股检测问题的测试脚本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_broken_limit_detection():
    """测试炸板股检测逻辑"""
    
    # 模拟一些测试数据
    test_data = [
        {
            'stock_code': '000001.SZ',
            'pct_chg': 8.5,  # 涨幅8.5%，未达到涨停
            'close': 10.85,
            'pre_close': 10.0,
            'open': 11.0,
            'high': 11.0,  # 最高价达到涨停价11.0
            'low': 10.5
        },
        {
            'stock_code': '688001.SH',  # 科创板，20%涨停
            'pct_chg': 15.0,  # 涨幅15%，未达到涨停
            'close': 11.5,
            'pre_close': 10.0,
            'open': 12.0,
            'high': 12.0,  # 最高价达到涨停价12.0
            'low': 11.0
        },
        {
            'stock_code': '300001.SZ',  # 创业板，20%涨停
            'pct_chg': 18.0,  # 涨幅18%，未达到涨停
            'close': 11.8,
            'pre_close': 10.0,
            'open': 12.0,
            'high': 12.0,  # 最高价达到涨停价12.0
            'low': 11.5
        }
    ]
    
    # 模拟SectorBoardAnalyzer的炸板检测逻辑
    def is_broken_limit(stock_data, stock_code):
        """模拟炸板检测逻辑"""
        try:
            pct_chg = stock_data.get('pct_chg', 0)
            close = stock_data.get('close', None)
            pre_close = stock_data.get('pre_close', None)
            high = stock_data.get('high', None)
            
            if pre_close is None or high is None or close is None:
                return False
                
            # 确定涨停幅度比率
            limit_ratio = 0.10  # 默认10%
            
            # 根据股票类型调整涨停幅度
            if 'ST' in stock_code.upper():
                limit_ratio = 0.05
            elif stock_code.startswith('688'):  # 科创板股票
                limit_ratio = 0.20
            elif stock_code.startswith('300'):  # 创业板股票
                limit_ratio = 0.20
            elif '.BJ' in stock_code:  # 北交所股票
                limit_ratio = 0.30
                
            # 计算理论涨停价
            theoretical_limit = pre_close * (1 + limit_ratio)
            rounded_limit = round(theoretical_limit, 2)
            
            # 如果最高价达到涨停价，但收盘价未达到涨停价，则视为涨停破板
            is_high_reached_limit = abs(high - rounded_limit) < 0.005
            is_close_not_limit = abs(close - rounded_limit) >= 0.005
            
            logger.info(f"股票: {stock_code}")
            logger.info(f"  前收盘: {pre_close}, 最高价: {high}, 收盘价: {close}")
            logger.info(f"  涨停比例: {limit_ratio*100}%, 理论涨停价: {theoretical_limit:.2f}, 四舍五入: {rounded_limit}")
            logger.info(f"  最高价达到涨停: {is_high_reached_limit}, 收盘价未达涨停: {is_close_not_limit}")
            logger.info(f"  是否炸板: {is_high_reached_limit and is_close_not_limit}")
            logger.info("-" * 50)
            
            return is_high_reached_limit and is_close_not_limit
            
        except Exception as e:
            logger.error(f"判断股票 {stock_code} 是否炸板时出错: {e}")
            return False
    
    logger.info("=== 开始测试炸板检测逻辑 ===")
    
    broken_count = 0
    for data in test_data:
        if is_broken_limit(data, data['stock_code']):
            broken_count += 1
    
    logger.info(f"测试完成，检测到 {broken_count} 只炸板股")
    
    return broken_count > 0

def check_real_data():
    """检查真实的板块数据"""
    
    # 尝试读取一个板块的数据文件
    data_dir = Path("data/sector_analysis")
    
    if not data_dir.exists():
        logger.warning("数据目录不存在，跳过真实数据检查")
        return
    
    # 查找板块分析文件
    parquet_files = list(data_dir.glob("*_board_analysis.parquet"))
    
    if not parquet_files:
        logger.warning("未找到板块分析文件，跳过真实数据检查")
        return
    
    # 读取第一个文件
    file_path = parquet_files[0]
    logger.info(f"读取文件: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")
        
        # 检查炸板相关字段
        if 'broken_limit_up_count' in df.columns:
            broken_stats = df['broken_limit_up_count'].describe()
            logger.info(f"broken_limit_up_count 统计:")
            logger.info(broken_stats)
            
            # 查看有炸板的记录
            broken_records = df[df['broken_limit_up_count'] > 0]
            if len(broken_records) > 0:
                logger.info(f"找到 {len(broken_records)} 条有炸板的记录:")
                logger.info(broken_records[['date', 'broken_limit_up_count', 'limit_up_count']].head())
            else:
                logger.warning("所有记录的 broken_limit_up_count 都为0")
        
        # 检查其他相关字段
        for field in ['yesterday_broken_limit_premium', 'broken_penalty_score']:
            if field in df.columns:
                stats = df[field].describe()
                logger.info(f"{field} 统计:")
                logger.info(stats)
        
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")

def main():
    """主函数"""
    logger.info("=== 开始调试炸板股检测问题 ===")
    
    # 1. 测试炸板检测逻辑
    logger.info("1. 测试炸板检测逻辑...")
    logic_works = test_broken_limit_detection()
    
    if logic_works:
        logger.info("✅ 炸板检测逻辑正常工作")
    else:
        logger.warning("❌ 炸板检测逻辑可能有问题")
    
    # 2. 检查真实数据
    logger.info("\n2. 检查真实数据...")
    check_real_data()
    
    logger.info("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()
