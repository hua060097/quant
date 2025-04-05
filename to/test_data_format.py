#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试xtdata.get_market_data返回的数据格式结构
"""

import logging
import os
import sys
import pandas as pd
import numpy as np
from xtquant import xtdata

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/test_data_format.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_data_format")

def test_market_data_structure():
    """测试市场数据的结构"""
    # 设置一些测试参数
    stocks = ['000001.SZ', '600000.SH', '300059.SZ', '002230.SZ', '301628.SZ']  # 包括一些大中小盘股票
    end_date = '20240320'  # 使用一个最近的日期
    start_date = '20240220'  # 一个月的数据足够测试
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    logger.info("开始获取市场数据...")
    
    try:
        # 获取市场数据
        market_data = xtdata.get_market_data(
            field_list=fields,
            stock_list=stocks,
            start_time=start_date,
            end_time=end_date,
            period='1d'  # 日线数据
        )
        
        # 打印市场数据的基本信息
        logger.info(f"市场数据类型: {type(market_data)}")
        logger.info(f"市场数据字段: {list(market_data.keys())}")
        
        # 检查每个字段的数据结构
        if market_data and len(market_data) > 0:
            for field in market_data:
                field_data = market_data[field]
                logger.info(f"\n字段 '{field}' 的详细信息:")
                logger.info(f"  数据类型: {type(field_data)}")
                logger.info(f"  数据形状: {field_data.shape if hasattr(field_data, 'shape') else '未知'}")
                logger.info(f"  索引类型: {type(field_data.index)}")
                logger.info(f"  索引值(前5个): {field_data.index.tolist()[:5]}")
                logger.info(f"  列类型: {type(field_data.columns)}")
                logger.info(f"  列值(前5个): {field_data.columns.tolist()[:5]}")
                
                # 打印数据样例
                logger.info(f"  数据样例(前5行5列):\n{field_data.iloc[:5, :5]}")
                
                # 检查是否有缺失值
                missing_values = field_data.isna().sum().sum()
                logger.info(f"  缺失值数量: {missing_values}")
                
                # 检查各个股票的数据情况
                for stock in stocks:
                    if stock in field_data.index:
                        stock_data = field_data.loc[stock]
                        logger.info(f"  股票 {stock} 的 {field} 数据:")
                        logger.info(f"    数据类型: {type(stock_data)}")
                        logger.info(f"    数据形状: {stock_data.shape if hasattr(stock_data, 'shape') else '未知'}")
                        logger.info(f"    数据点数量: {len(stock_data)}")
                        logger.info(f"    前5个值: {stock_data.values[:5]}")
                        logger.info(f"    非缺失值数量: {stock_data.count()}")
                        logger.info(f"    是否全为NaN: {stock_data.isna().all()}")
                    else:
                        logger.warning(f"  股票 {stock} 在字段 {field} 中不存在")
        else:
            logger.warning("获取的市场数据为空或无效")
        
        # 测试正确的访问方式
        logger.info("\n测试不同的数据访问方式:")
        
        # 方法1: 直接通过dataframe索引访问
        try:
            first_field = list(market_data.keys())[0]
            first_stock = stocks[0]
            if first_stock in market_data[first_field].index:
                method1_data = market_data[first_field].loc[first_stock]
                logger.info(f"方法1 (df.loc[stock]): 成功，数据类型={type(method1_data)}, 长度={len(method1_data)}")
            else:
                logger.warning(f"方法1: 股票 {first_stock} 在 {first_field} 中不存在")
        except Exception as e:
            logger.error(f"方法1访问出错: {e}")
        
        # 方法2: 使用stock作为索引，日期作为列
        try:
            first_field = list(market_data.keys())[0]
            first_stock = stocks[0]
            first_date = market_data[first_field].columns[0]
            if first_stock in market_data[first_field].index:
                method2_data = market_data[first_field].loc[first_stock, first_date]
                logger.info(f"方法2 (df.loc[stock, date]): 成功，数据值={method2_data}")
            else:
                logger.warning(f"方法2: 股票 {first_stock} 在 {first_field} 中不存在")
        except Exception as e:
            logger.error(f"方法2访问出错: {e}")
        
        # 方法3: 转换为多个单股票dataframe
        try:
            stock_dfs = {}
            for stock in stocks:
                stock_data = {}
                for field in fields:
                    if stock in market_data[field].index:
                        stock_data[field] = market_data[field].loc[stock].values
                    else:
                        logger.warning(f"方法3: 股票 {stock} 在字段 {field} 中不存在")
                        break
                
                if len(stock_data) == len(fields):
                    # 使用市场数据的列(日期)作为索引
                    stock_dfs[stock] = pd.DataFrame(stock_data, index=market_data[fields[0]].columns)
                    logger.info(f"方法3 (创建股票DataFrame): 成功，股票={stock}, 形状={stock_dfs[stock].shape}")
                    logger.info(f"方法3 样例数据:\n{stock_dfs[stock].head(3)}")
                else:
                    logger.warning(f"方法3: 股票 {stock} 缺少部分字段数据")
        except Exception as e:
            logger.error(f"方法3访问出错: {e}")
        
        return market_data
    except Exception as e:
        logger.error(f"获取市场数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_valid_stock_check(market_data, stocks, fields):
    """测试检查有效股票的方法"""
    if not market_data:
        logger.error("市场数据为空，无法测试有效股票检查")
        return
    
    logger.info("\n开始测试股票数据有效性检查...")
    valid_stocks = []
    
    # 记录统计信息，用于诊断
    total_checked = 0
    missing_field_count = 0
    insufficient_data_count = 0
    all_nan_count = 0

    required_fields = ['close', 'open', 'high', 'low']
    missing_short_fields = [f for f in required_fields if f not in market_data]
    if missing_short_fields:
        print(f"短期数据缺少必需字段: {missing_short_fields}")
    else:
        print("短期数据包含必需字段")
    
    for stock in stocks:
        total_checked += 1
        is_valid = True
        reason = None
        
        # 检查每个字段
        for field in fields:
            # 检查字段是否存在
            if field not in market_data:
                reason = f"缺少字段 {field}"
                missing_field_count += 1
                is_valid = False
                break
            
            # 检查股票是否在该字段的DataFrame中
            if stock not in market_data[field].index:
                reason = f"在字段 {field} 中不存在"
                missing_field_count += 1
                is_valid = False
                break
            
            # 获取该股票的数据
            stock_data = market_data[field].loc[stock]
            
            # 检查数据点是否足够
            if stock_data.empty or len(stock_data) < 10:  # 至少需要10个数据点
                reason = f"字段 {field} 数据点不足: {len(stock_data) if not stock_data.empty else 0}"
                insufficient_data_count += 1
                is_valid = False
                break
            
            # 检查是否全部为NaN
            if stock_data.isna().all():
                reason = f"字段 {field} 数据全为NaN"
                all_nan_count += 1
                is_valid = False
                break
        
        if is_valid:
            valid_stocks.append(stock)
            logger.info(f"股票 {stock} 数据有效")
        else:
            logger.warning(f"股票 {stock} 数据无效: {reason}")
    
    # 记录统计信息
    logger.info(f"数据可用性检查: 总计 {total_checked} 只股票, 有效 {len(valid_stocks)} 只")
    logger.info(f"无效原因统计: 缺少字段 {missing_field_count}, 数据点不足 {insufficient_data_count}, 全NaN {all_nan_count}")
    
    return valid_stocks

def main():
    """主函数"""
    logger.info("开始测试数据格式...")
    
    # 测试市场数据结构
    market_data = test_market_data_structure()
    
    # 如果成功获取了数据，测试有效股票检查
    if market_data:
        stocks = ['000001.SZ', '600000.SH', '300059.SZ', '002230.SZ', '301628.SZ']
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        valid_stocks = test_valid_stock_check(market_data, stocks, fields)
        
        logger.info(f"有效股票列表: {valid_stocks}")
    
    logger.info("数据格式测试完成")

if __name__ == "__main__":
    main() 