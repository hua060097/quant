#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志数据探索工具 - 辅助分析市场数据格式
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from xtquant import xtdata

# 配置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/data_explorer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_explorer")

def log_dataframe_info(df, name, max_rows=5, max_cols=5):
    """记录DataFrame的详细信息"""
    logger.info(f"--- {name} 信息 ---")
    logger.info(f"类型: {type(df)}")
    
    if isinstance(df, pd.DataFrame):
        logger.info(f"形状: {df.shape}")
        logger.info(f"列: {list(df.columns)[:10]}... (显示前10个)")
        logger.info(f"索引: {list(df.index)[:10]}... (显示前10个)")
        logger.info(f"数据样例 (前{max_rows}行{max_cols}列):\n{df.iloc[:max_rows, :max_cols]}")
        logger.info(f"数据类型:\n{df.dtypes}")
        logger.info(f"缺失值统计: {df.isna().sum().sum()} (总计)")
    elif isinstance(df, pd.Series):
        logger.info(f"长度: {len(df)}")
        logger.info(f"索引: {list(df.index)[:10]}... (显示前10个)")
        logger.info(f"数据样例 (前{max_rows}个):\n{df.iloc[:max_rows]}")
        logger.info(f"数据类型: {df.dtype}")
        logger.info(f"缺失值统计: {df.isna().sum()} (总计)")
    else:
        logger.info(f"对象不是DataFrame或Series: {type(df)}")

def log_market_data(market_data, prefix=""):
    """记录市场数据的结构和内容"""
    if not market_data:
        logger.warning(f"{prefix}市场数据为空")
        return
    
    logger.info(f"{prefix}市场数据类型: {type(market_data)}")
    logger.info(f"{prefix}字段列表: {list(market_data.keys())}")
    
    # 记录每个字段的信息
    for field in market_data:
        logger.info(f"{prefix}字段 '{field}' 信息:")
        log_dataframe_info(market_data[field], f"{prefix}{field}")

def explore_stock_data(stocks, date, lookback_days=30, forward_days=5):
    """探索指定股票在指定日期前后的数据"""
    logger.info(f"开始探索股票数据, 股票={stocks}, 日期={date}, 回溯天数={lookback_days}, 前瞻天数={forward_days}")
    
    # 手动设置开始和结束日期，不依赖交易日历
    try:
        # 解析输入日期
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        
        # 计算大约一个月前的日期（简化处理，不考虑月日有效性）
        prev_month = month - 1
        prev_year = year
        if prev_month < 1:
            prev_month = 12
            prev_year -= 1
        
        # 计算大约5天后的日期（简化处理）
        next_day = min(day + 5, 28)  # 使用28避免月末问题
        
        # 格式化开始和结束日期
        start_date = f"{prev_year:04d}{prev_month:02d}{day:02d}"
        end_date = f"{year:04d}{month:02d}{next_day:02d}"
        
        logger.info(f"使用日期范围: {start_date} 至 {end_date}")
        
        # 获取市场数据
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        logger.info(f"获取市场数据: 字段={fields}, 股票={stocks}")
        market_data = xtdata.get_market_data(
            field_list=fields,
            stock_list=stocks,
            start_time=start_date,
            end_time=end_date,
            period='1d'
        )
        
        log_market_data(market_data, "市场数据: ")
        
        # 检查每只股票的数据
        logger.info("\n分析每只股票的数据可用性:")
        
        for stock in stocks:
            logger.info(f"\n股票 {stock} 数据:")
            valid_fields = []
            
            for field in fields:
                if field in market_data and stock in market_data[field].index:
                    stock_series = market_data[field].loc[stock]
                    valid_data_points = stock_series.count()  # 非NaN值的数量
                    
                    logger.info(f"  字段 {field}: 数据点数量={len(stock_series)}, 有效数据点={valid_data_points}")
                    if valid_data_points > 0:
                        valid_fields.append(field)
                        # 显示部分数据值
                        logger.info(f"  字段 {field} 数据样例: {stock_series.values[:5]}... (前5个)")
                else:
                    logger.warning(f"  字段 {field}: 数据不可用")
            
            logger.info(f"  股票 {stock} 有效字段数: {len(valid_fields)}/{len(fields)}")
        
        # 演示正确的数据处理方法
        logger.info("\n演示正确的数据处理方法:")
        try:
            # 1. 提取数据
            stock_data_dict = {}
            for stock in stocks:
                stock_data = {}
                for field in fields:
                    if field in market_data and stock in market_data[field].index:
                        stock_data[field] = market_data[field].loc[stock].values
                
                if len(stock_data) == len(fields):
                    # 2. 创建DataFrame
                    df = pd.DataFrame(stock_data, index=market_data[fields[0]].columns)
                    stock_data_dict[stock] = df
                    
                    # 3. 计算技术指标示例
                    if len(df) >= 5:
                        df['ma5'] = df['close'].rolling(window=5).mean()
                        logger.info(f"股票 {stock} MA5示例: {df['ma5'].tail(3).values}")
                    
                    # 4. 提取特征示例
                    if date in df.index:
                        current_features = df.loc[date]
                        logger.info(f"股票 {stock} 在日期 {date} 的特征:\n{current_features}")
                    else:
                        logger.warning(f"日期 {date} 不在数据索引中")
                        if len(df) > 0:
                            # 使用最后一个日期作为替代
                            last_date = df.index[-1]
                            logger.info(f"使用最后日期 {last_date} 作为替代:")
                            current_features = df.iloc[-1]
                            logger.info(f"股票 {stock} 在日期 {last_date} 的特征:\n{current_features}")
        except Exception as e:
            logger.error(f"演示数据处理出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return market_data
    
    except Exception as e:
        logger.error(f"探索股票数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def simulate_data_processing(market_data, stocks, date):
    """模拟_extract_features方法的数据处理流程"""
    if not market_data:
        logger.error("无法进行模拟，市场数据为空")
        return
    
    logger.info(f"\n模拟特征提取流程，股票列表={stocks}, 日期={date}")
    
    try:
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        # 1. 检查数据可用性
        valid_stocks = []
        for stock in stocks:
            is_valid = True
            for field in fields:
                if field not in market_data or stock not in market_data[field].index:
                    is_valid = False
                    break
                
                stock_data = market_data[field].loc[stock]
                if stock_data.isna().all() or len(stock_data) < 10:
                    is_valid = False
                    break
                    
            if is_valid:
                valid_stocks.append(stock)
                
        logger.info(f"有效股票: {valid_stocks} ({len(valid_stocks)}/{len(stocks)})")
        
        # 2. 提取特征和标签
        for stock in valid_stocks:
            logger.info(f"\n处理股票 {stock}:")
            
            # 创建单股票数据框
            stock_df = pd.DataFrame()
            for field in fields:
                stock_df[field] = market_data[field].loc[stock]
            
            # 输出数据框信息
            logger.info(f"股票 {stock} 的数据框:")
            log_dataframe_info(stock_df, f"股票 {stock} 数据框")
            
            # 检查日期是否在数据中
            if date in stock_df.index:
                date_idx = stock_df.index.get_loc(date)
                logger.info(f"日期 {date} 在股票 {stock} 数据中的索引: {date_idx}")
                
                # 提取当前日期的特征
                current_features = stock_df.loc[date]
                logger.info(f"当前日期特征: {current_features}")
            else:
                logger.warning(f"日期 {date} 不在股票 {stock} 的数据中")
                
            # 计算简单技术指标
            if len(stock_df) >= 5:
                stock_df['ma5'] = stock_df['close'].rolling(window=5).mean()
                logger.info(f"MA5 (最后3个值): {stock_df['ma5'].tail(3).values}")
                
    except Exception as e:
        logger.error(f"模拟数据处理时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    logger.info("开始数据探索...")
    
    # 设置测试参数
    stocks = ['000001.SZ', '600000.SH', '300059.SZ', '002230.SZ']
    date = '20240320'  # 使用一个最近的日期
    
    # 探索股票数据
    market_data = explore_stock_data(stocks, date)
    
    # 如果成功获取了数据，模拟数据处理
    if market_data:
        simulate_data_processing(market_data, stocks, date)
    
    logger.info("数据探索完成")

if __name__ == "__main__":
    main() 