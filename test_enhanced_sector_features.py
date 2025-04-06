#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试增强版板块特征生成

生成并查看增强版板块特征
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from to.sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator

# 模拟get_sector_stocks函数
def mock_get_sector_stocks(sector_name=None):
    """模拟获取板块成分股"""
    sector_stocks = {
        "TGN人工智能": ["000001", "000002", "000003", "000004", "000005", "000006"],
        "TGN金融": ["600000", "600001", "600002", "600003", "600004", "600005"],
    }
    
    if sector_name in sector_stocks:
        return sector_stocks[sector_name]
    return []

# 替代实际的导入
# 由于没有xtquant，我们使用模拟函数
sys.modules['to.stock_sectors_mapping'] = type('', (), {})()
sys.modules['to.stock_sectors_mapping'].get_sector_stocks = mock_get_sector_stocks

# 创建一个简单的数据提供者，用于测试
class SimpleDataProvider:
    def __init__(self):
        self.data_dir = 'data'
        
    def get_daily_data(self, stocks, start_date, end_date):
        """模拟获取股票数据"""
        import numpy as np
        # 创建模拟数据
        data = []
        dates = self._get_dates_between(start_date, end_date)
        
        for date in dates:
            # 生成日期索引，用于生成有趋势的价格
            date_idx = len(data) // len(stocks) if len(stocks) > 0 else 0
            
            for i, stock in enumerate(stocks):
                # 生成更加真实的随机数据
                base_price = 10.0 + (i % 3)  # 不同股票有不同基础价格
                price_trend = 0.02 * date_idx  # 添加时间趋势
                random_factor = np.random.normal(0, 0.05)  # 随机波动
                
                price = base_price * (1 + price_trend + random_factor)
                
                open_price = price * (1 + np.random.normal(0, 0.01))
                high_price = price * (1 + abs(np.random.normal(0, 0.02)))
                low_price = price * (1 - abs(np.random.normal(0, 0.02)))
                close_price = price * (1 + np.random.normal(0, 0.01))
                
                # 确保high >= open/close >= low
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # 成交量和成交额
                volume = np.random.randint(500000, 2000000)
                amount = volume * close_price
                
                # 涨跌幅
                prev_close = base_price if date_idx == 0 else data[-len(stocks)]['close'] if i < len(data)-len(stocks) else base_price
                change = (close_price - prev_close) / prev_close if prev_close > 0 else 0
                
                data.append({
                    'date': date,
                    'stock_code': stock,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'amount': amount,
                    'change': change
                })
        
        return pd.DataFrame(data)
    
    def get_trading_days(self, start_date, end_date):
        """获取交易日列表"""
        return self._get_dates_between(start_date, end_date)
    
    def _get_dates_between(self, start_date, end_date):
        """获取两个日期之间的日期列表"""
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        # 只返回工作日（简化处理）
        result = []
        current = start
        while current <= end:
            # 如果是工作日（周一到周五）
            if current.weekday() < 5:
                result.append(current.strftime("%Y%m%d"))
            current = current.replace(day=current.day + 1)
        
        return result

def main():
    # 要测试的板块
    sectors_to_test = ["TGN人工智能", "TGN金融"]
    
    # 创建数据提供者
    data_provider = SimpleDataProvider()
    
    # 创建板块映射
    sector_mapping = {}
    for sector in sectors_to_test:
        # 使用模拟函数获取股票列表
        stocks = mock_get_sector_stocks(sector_name=sector)
        sector_mapping[sector] = stocks
    
    # 创建输出目录
    output_dir = "enhanced_sector_features_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建增强版特征生成器
    feature_generator = EnhancedSectorFeatureGenerator(
        data_provider, 
        sector_mapping, 
        output_dir,
        weighting_method='equal'
    )
    
    # 测试日期范围
    start_date = "20230101"
    end_date = "20230110"
    
    # 为每个板块生成特征
    for sector in sectors_to_test:
        logger.info(f"为板块 {sector} 生成增强版特征...")
        
        # 为单一日期生成特征
        test_date = "20230110"
        features_df = feature_generator.generate_sector_features_for_date(sector, test_date)
        
        if not features_df.empty:
            logger.info(f"板块 {sector} 的增强版特征列: {features_df.columns.tolist()}")
            logger.info(f"板块 {sector} 的增强版特征:\n{features_df}")
        
        # 生成并保存特征
        success_count = feature_generator.generate_and_save_features(sector, start_date, end_date)
        
        logger.info(f"成功为板块 {sector} 生成了 {success_count} 天的增强版特征")
        
        # 读取并显示生成的特征
        sector_dir = os.path.join(output_dir, sector)
        combined_file = os.path.join(sector_dir, f"{start_date}_{end_date}.csv")
        
        if os.path.exists(combined_file):
            features_df = pd.read_csv(combined_file)
            logger.info(f"板块 {sector} 的增强版特征包含 {len(features_df.columns)} 个列")
            # 只显示特征列名
            logger.info(f"板块 {sector} 的增强版特征列: {features_df.columns.tolist()}")
    
    logger.info("测试完成！")

if __name__ == "__main__":
    main() 