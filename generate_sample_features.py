#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征生成与测试脚本

演示特征生成、保存和加载的流程，验证优化后的特征处理模块
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 导入相关模块
from to.path_utils import get_data_path, ensure_dir
from to.stock_sectors_mapping import load_mapping, get_stocks_in_sector
from to.feature_loader import load_sector_features
from to.feature_slicer import FeatureSlicer, get_time_slice, get_feature_group, get_window_slice

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sector_features(sector_name, start_date, end_date, stock_count=6):
    """生成板块特征数据
    
    Args:
        sector_name: 板块名称
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
        stock_count: 板块内股票数量
        
    Returns:
        特征DataFrame
    """
    logger.info(f"生成 {sector_name} 板块从 {start_date} 到 {end_date} 的特征数据")
    
    # 生成日期序列
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        # 跳过周末
        if current_dt.weekday() < 5:  # 0-4 对应周一至周五
            dates.append(current_dt.strftime('%Y%m%d'))
        current_dt += timedelta(days=1)
    
    # 创建特征数据
    data = []
    
    # 添加前6天的零数据（用于测试）
    for i in range(6):
        data.append({
            'date': dates[i],
            'open_avg': 0,
            'high_avg': 0,
            'low_avg': 0,
            'close_avg': 0,
            'volume_avg': 0,
            'amount_avg': 0,
            'change_avg': 0,
            'active_count': 0,
            'ma5_avg': 0,
            'ma10_avg': 0,
            'ma20_avg': 0,
            'ma60_avg': 0,
            'price_above_ma5_ratio': 0,
            'price_above_ma20_ratio': 0,
            'total_amount': 0,
            'sector_name': sector_name,
            'stock_count': stock_count
        })
    
    # 为剩余日期生成随机数据
    base_price = 12.5
    for i in range(6, len(dates)):
        # 增加一些随机性但保持趋势
        daily_change = np.random.normal(0.002, 0.008)
        base_price *= (1 + daily_change)
        
        # 添加当天数据
        volume = int(np.random.normal(450000, 100000))
        price_open = base_price * (1 + np.random.normal(0, 0.01))
        price_high = price_open * (1 + abs(np.random.normal(0, 0.02)))
        price_low = price_open * (1 - abs(np.random.normal(0, 0.015)))
        price_close = (price_high + price_low) / 2 + np.random.normal(0, 0.005)
        
        # 计算移动均线
        idx = i - 6  # 真实数据的索引
        
        # 过去5天收盘价的均值
        ma5 = price_close if idx == 0 else sum([data[j]['close_avg'] for j in range(idx-min(5, idx), idx)] + [price_close]) / min(5, idx+1)
        
        # 过去10天收盘价的均值
        ma10 = price_close if idx == 0 else sum([data[j]['close_avg'] for j in range(idx-min(10, idx), idx)] + [price_close]) / min(10, idx+1)
        
        # 过去20天收盘价的均值
        ma20 = price_close if idx == 0 else sum([data[j]['close_avg'] for j in range(idx-min(20, idx), idx)] + [price_close]) / min(20, idx+1)
        
        # 过去60天收盘价的均值（这里用更少的数据模拟）
        ma60 = price_close if idx == 0 else sum([data[j]['close_avg'] for j in range(idx-min(60, idx), idx)] + [price_close]) / min(60, idx+1)
        
        # 当前价格相对于均线的比率
        price_above_ma5_ratio = round(np.random.uniform(0.5, 1.0), 2) if idx < 5 else round((price_close > ma5) * np.random.uniform(0.6, 1.0), 2)
        price_above_ma20_ratio = round(np.random.uniform(0.5, 1.0), 2) if idx < 20 else round((price_close > ma20) * np.random.uniform(0.6, 1.0), 2)
        
        data.append({
            'date': dates[i],
            'open_avg': round(price_open, 1),
            'high_avg': round(price_high, 1),
            'low_avg': round(price_low, 1),
            'close_avg': round(price_close, 1),
            'volume_avg': volume,
            'amount_avg': int(volume * price_close),
            'change_avg': round(daily_change, 3),
            'active_count': int(np.random.randint(3, 7)),
            'ma5_avg': round(ma5, 2),
            'ma10_avg': round(ma10, 2),
            'ma20_avg': round(ma20, 2),
            'ma60_avg': round(ma60, 2),
            'price_above_ma5_ratio': price_above_ma5_ratio,
            'price_above_ma20_ratio': price_above_ma20_ratio,
            'total_amount': int(volume * price_close * stock_count),
            'sector_name': sector_name,
            'stock_count': stock_count
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    return df

def save_sector_features(sector_name, df):
    """保存板块特征数据
    
    Args:
        sector_name: 板块名称
        df: 特征DataFrame
        
    Returns:
        保存的文件路径
    """
    # 构建保存路径
    sector_dir = get_data_path('sector_features') / sector_name
    ensure_dir(sector_dir)
    
    file_path = sector_dir / f"{sector_name}_features.csv"
    
    # 保存到CSV
    df.to_csv(file_path, index=False)
    
    logger.info(f"保存特征数据到 {file_path}，共 {len(df)} 行")
    
    return file_path

def test_feature_loader(sector_name, start_date, end_date):
    """测试特征加载器
    
    Args:
        sector_name: 板块名称
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
    """
    logger.info(f"测试加载 {sector_name} 板块从 {start_date} 到 {end_date} 的特征数据")
    
    # 使用特征加载器加载数据
    df = load_sector_features(sector_name, start_date, end_date)
    
    logger.info(f"成功加载 {len(df)} 行数据")
    logger.info(f"数据范围: {df['date'].min()} - {df['date'].max()}")
    logger.info(f"特征列: {', '.join(df.columns)}")
    
    # 打印前5行数据
    logger.info("前5行数据:")
    print(df.head())

def test_feature_slicer(sector_name, start_date, end_date):
    """测试特征切片器
    
    Args:
        sector_name: 板块名称
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
    """
    logger.info(f"测试特征切片器，切片 {sector_name} 板块的特征数据")
    
    # 创建特征切片器
    slicer = FeatureSlicer()
    
    # 获取时间切片
    df_time = slicer.get_time_slice(sector_name, start_date, end_date)
    logger.info(f"时间切片: {start_date} - {end_date}, 共 {len(df_time)} 行")
    
    # 获取特征分组
    df_basic = slicer.get_feature_group(sector_name, 'basic', start_date, end_date)
    logger.info(f"基本特征分组: 共 {len(df_basic)} 行, {len(df_basic.columns)} 列")
    print("基本特征分组列:", ', '.join(df_basic.columns))
    
    # 获取窗口切片
    target_date = end_date
    window_size = 5
    df_window = slicer.get_window_slice(sector_name, target_date, window_size)
    logger.info(f"窗口切片: 目标日期 {target_date}, 窗口大小 {window_size}, 共 {len(df_window)} 行")
    
    # 打印窗口数据
    print("窗口数据:")
    print(df_window[['date', 'close_avg', 'volume_avg']])
    
    # 测试序列数据生成
    sequences = slicer.get_sequence_data(sector_name, 3, ['close_avg', 'volume_avg'], start_date, end_date)
    logger.info(f"序列数据: 序列长度 3, 共 {len(sequences)} 个序列")
    
    if sequences:
        print("第一个序列:")
        print(sequences[0])

if __name__ == "__main__":
    # 定义测试参数
    sector_name = "TGN人工智能"
    start_date = "20230101"
    end_date = "20230131"
    
    # 生成特征数据
    df = generate_sector_features(sector_name, start_date, end_date)
    
    # 保存特征数据
    save_sector_features(sector_name, df)
    
    # 测试特征加载器
    test_feature_loader(sector_name, "20230110", "20230120")
    
    # 测试特征切片器
    test_feature_slicer(sector_name, "20230110", "20230120")
    
    print("\n所有测试完成!") 