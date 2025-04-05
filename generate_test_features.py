import os
import json
import random
import numpy as np
from datetime import datetime, timedelta

# 创建必要的目录
def create_directories():
    directories = [
        "to/local_data/features/daily",
        "to/local_data/market_status/daily",
        "to/models/TGN人工智能"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

# 生成随机股票代码
def generate_stock_codes(num_stocks=30):
    stock_codes = []
    for i in range(num_stocks):
        if random.random() < 0.6:  # 60% 上海股票
            code = f"60{random.randint(1000, 9999)}.SH"
        else:  # 40% 深圳股票
            code = f"00{random.randint(1000, 9999)}.SZ"
        stock_codes.append(code)
    return stock_codes

# 生成特征数据
def generate_features(stock_code, date_str):
    # 生成随机特征数据
    features = {
        "open": random.uniform(10, 100),
        "high": random.uniform(10, 100),
        "low": random.uniform(10, 100),
        "close": random.uniform(10, 100),
        "volume": random.randint(1000000, 10000000),
        "amount": random.randint(10000000, 100000000),
        "turn": random.uniform(1, 10),
        "pct_chg": random.uniform(-5, 5),
        "ma5": random.uniform(10, 100),
        "ma10": random.uniform(10, 100),
        "ma20": random.uniform(10, 100),
        "vma5": random.randint(1000000, 10000000),
        "vma10": random.randint(1000000, 10000000),
        "vma20": random.randint(1000000, 10000000),
        "momentum_5": random.uniform(-5, 5),
        "momentum_10": random.uniform(-5, 5),
        "momentum_20": random.uniform(-5, 5),
        "rsi_6": random.uniform(0, 100),
        "rsi_12": random.uniform(0, 100),
        "rsi_24": random.uniform(0, 100),
    }
    return features

# 生成指定日期范围的特征数据
def generate_features_for_date_range(start_date_str, end_date_str, stock_codes):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    
    current_date = start_date
    trading_dates = []
    
    # 生成交易日（简化为周一到周五）
    while current_date <= end_date:
        if current_date.weekday() < 5:  # 周一到周五
            trading_dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # 为每个股票生成特征数据
    for stock_code in stock_codes:
        stock_directory = f"to/local_data/features/daily/{stock_code}"
        os.makedirs(stock_directory, exist_ok=True)
        
        for date_str in trading_dates:
            features = generate_features(stock_code, date_str)
            file_path = f"{stock_directory}/{date_str}.json"
            with open(file_path, "w") as f:
                json.dump(features, f, indent=2)
            print(f"生成特征数据: {file_path}")
    
    # 生成市场状态数据
    market_status = {date: {"is_trading_day": True} for date in trading_dates}
    market_status_path = "to/local_data/market_status/daily/market_status.json"
    with open(market_status_path, "w") as f:
        json.dump(market_status, f, indent=2)
    print(f"生成市场状态数据: {market_status_path}")
    
    return trading_dates

# 主函数
if __name__ == "__main__":
    create_directories()
    
    # 生成AI概念股票
    stock_codes = generate_stock_codes(30)
    print(f"生成了30支AI概念股票")
    
    # 生成2023年1月1日到2023年1月10日的数据
    trading_dates = generate_features_for_date_range("20230101", "20230110", stock_codes)
    print(f"生成了{len(trading_dates)}个交易日的数据")
    
    print(f"特征数据路径: {os.path.abspath('to/local_data/features/daily')}")
    print(f"市场状态数据路径: {os.path.abspath('to/local_data/market_status/daily/market_status.json')}") 