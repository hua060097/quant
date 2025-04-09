import sys
import logging
import pandas as pd
import time
from pathlib import Path

# 将上级目录添加到模块搜索路径
sys.path.append(str(Path(__file__).parent))

# 导入advanced_quant_data模块
from advanced_quant_data import ParquetDataStorage

# 配置日志
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_stock_request")

def test_minute_data():
    """使用不同的股票代码测试获取5分钟数据"""
    # 初始化存储对象
    storage = ParquetDataStorage()
    
    # 测试参数 - 2023年数据测试
    test_date_2023 = '20230101'
    end_test_date_2023 = '20230110'
    test_stock_2023 = '000008.SZ'  # 确定存在于2023年分区的股票
    
    # 测试参数 - 2024年数据测试
    test_date_2024 = '20240301'
    end_test_date_2024 = '20240310'
    test_stock_2024 = '000001.SZ'  # 确定存在于2024年分区的股票
    
    # 测试2023年数据
    logger.info(f"=== 测试2023年数据 ===")
    logger.info(f"日期: {test_date_2023} 至 {end_test_date_2023}")
    logger.info(f"股票: {test_stock_2023}")
    
    df_2023 = storage.get_minute_data(
        stock_list=[test_stock_2023], 
        start_date=test_date_2023, 
        end_date=end_test_date_2023, 
        fields=['open', 'high', 'low', 'close', 'volume']
    )
    
    # 检查结果
    if df_2023.empty:
        logger.error("未能获取到2023年数据")
    else:
        logger.info(f"2023年数据获取成功，共 {len(df_2023)} 行")
        logger.info(f"数据列: {df_2023.columns.tolist()}")
        logger.info(f"日期范围: {df_2023['date_str'].min()} 至 {df_2023['date_str'].max()}")
        if 'datetime' in df_2023.columns:
            logger.info(f"时间范围: {df_2023['datetime'].min()} 至 {df_2023['datetime'].max()}")
        
        # 显示前5行
        logger.info("前5行数据:")
        print(df_2023.head(5))
    
    # 测试2024年数据
    logger.info(f"\n=== 测试2024年数据 ===")
    logger.info(f"日期: {test_date_2024} 至 {end_test_date_2024}")
    logger.info(f"股票: {test_stock_2024}")
    
    df_2024 = storage.get_minute_data(
        stock_list=[test_stock_2024], 
        start_date=test_date_2024, 
        end_date=end_test_date_2024, 
        fields=['open', 'high', 'low', 'close', 'volume']
    )
    
    # 检查结果
    if df_2024.empty:
        logger.error("未能获取到2024年数据")
    else:
        logger.info(f"2024年数据获取成功，共 {len(df_2024)} 行")
        logger.info(f"数据列: {df_2024.columns.tolist()}")
        logger.info(f"日期范围: {df_2024['date_str'].min()} 至 {df_2024['date_str'].max()}")
        if 'datetime' in df_2024.columns:
            logger.info(f"时间范围: {df_2024['datetime'].min()} 至 {df_2024['datetime'].max()}")
        
        # 显示前5行
        logger.info("前5行数据:")
        print(df_2024.head(5))

if __name__ == "__main__":
    logger.info("开始测试...")
    test_minute_data()
    logger.info("测试完成") 