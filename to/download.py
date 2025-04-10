import xtquant.xtdata as xtdata
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import numpy as np
import threading

class DownloadProgress:
    def __init__(self):
        self.hasfinished = 0
        self.total_items = 0 # 新增：用于存储总数
        self.is_finished = False # 新增：完成标志
        self._lock = threading.Lock() # Add a lock for thread safety

    def on_progress(self, data):
        with self._lock: # Ensure thread-safe updates
            # 确保data是字典并且包含所需键
            print(f"接收到回调数据: {data}", flush=True)
            if isinstance(data, dict) and 'finished' in data and 'total' in data:
                self.hasfinished = data['finished']
                self.total_items = data['total'] # 更新总数
                print(f"下载进度: {self.hasfinished}/{self.total_items}", flush=True) # Add flush=True
                # time.sleep(1) # 不建议在回调中sleep，会阻塞API内部处理

                # 检查是否完成
                if self.total_items > 0 and self.hasfinished >= self.total_items:
                    if not self.is_finished: # Only print and set once
                        print("下载完成 (回调检测)", flush=True)
                        self.is_finished = True # 设置完成标志
            else:
                # 如果数据格式不符合预期，打印警告
                print(f"警告：接收到意外的回调数据格式: {data}", flush=True)

    def check_finished(self):
         """Thread-safe method to check the finished status."""
         with self._lock:
              return self.is_finished

    def get_progress(self):
         """Thread-safe method to get current progress."""
         with self._lock:
              return self.hasfinished, self.total_items


def download_a_shares_tick_data():
    # 获取沪深A股所有股票代码
    try:
        # Ensure connection
        if not xtdata.get_client().is_connected():
            logger.info("Connecting to xtdata API...")
            xtdata.get_client().connect()
            time.sleep(1) # Allow time for connection

        stock_list = xtdata.get_stock_list_in_sector('沪深A股')
        print(f"获取到的股票代码数量: {len(stock_list)}")
        if not stock_list:
            print("未能获取到股票列表，请检查API连接或权限。")
            return
        # Print first few stocks for confirmation
        print(f"股票列表前几个: {stock_list[:5]}")
    except Exception as e:
        print(f"获取股票列表时出错: {e}")
        return


    INDEX_LIST = [
        '399006.SZ',  # 创业板指数（主要关注）
        '000001.SH',  # 上证指数
        '000688.SH',  # 科创50
        '899050.BJ',  # 北证50
    ]

    # 设置时间范围：2023年6月至今
    start_date = '20220316'
    end_date = '20250317'

    print(f"下载时间范围: {start_date} - {end_date}")

    #初始化下载进度跟踪器
    progress = DownloadProgress()

    stock_list = ['399006.SZ','899050.BJ']
    # 下载tick数据
    print("开始下载沪深A股 5分钟 数据...")
    # Use 5m period as originally intended
    result = xtdata.download_history_data2(
        stock_list=stock_list,
        period='1d',
        start_time=start_date,
        end_time=end_date,
        callback=progress.on_progress,
        incrementally = True
    )
    print(f"下载请求已发出 ({result})，等待下载完成...")

    # 新增：等待下载完成的循环
    start_wait_time = time.time()
    timeout_seconds = 3600 * 2 # Example: 2 hour timeout
    while not progress.check_finished():
        current_has, current_total = progress.get_progress()
        elapsed_time = time.time() - start_wait_time
        print(f"等待中... 进度: {current_has}/{current_total}. 已耗时: {elapsed_time:.1f} 秒", flush=True)
        if elapsed_time > timeout_seconds:
             print(f"下载超时 ({timeout_seconds}秒).")
             break
        time.sleep(5) # Check every 5 seconds

    if progress.check_finished():
        print("主程序检测到下载已完成。")
    else:
         print("主程序退出等待，下载可能未完成或超时。")


    # 测试获取某只股票的tick数据
    test_code = '399006.SZ'  # 贵州茅台
    print(f"验证股票 {test_code} 的5分钟数据...")

    try:
        test_data = xtdata.get_local_data(
            field_list=[], # Empty list to get all available fields
            stock_list=['399006.SZ'],
            period='1d',
            # start_time='20230316093500',
            # end_time='20230317094500'
            start_time='20220316',
            end_time='20230325'
        )

        print(f"验证数据获取成功 : {test_data}.")

        if test_data and 'time' in test_data and not test_data['time'].empty:
             print(f"验证数据获取成功 for {test_code}.")
             # Print shape of one of the dataframes
             if 'close' in test_data:
                  print(f"收盘价数据形状: {test_data['close'].shape}")
                  print(f"收盘价数据示例:\n{test_data['close'].head()}")
             else:
                  print("收盘价数据字段不存在。")
        else:
             print(f"验证数据获取失败或为空 for {test_code}. Result:\n{test_data}")

    except Exception as e:
        print(f"验证数据时出错: {e}")


def download_and_verify_stock_data():
    """下载和验证股票5分钟数据"""
    
    # 确保连接
    if not xtdata.get_client().is_connected():
        print("连接xtdata API...")
        xtdata.connect()
        time.sleep(1)
    
    # 要测试的股票代码列表
    test_stocks = [
        '601222.SH',  # 华电重工
        '605090.SH',  # 家润多
        '600519.SH',  # 贵州茅台
        '000001.SZ',  # 平安银行
        '300059.SZ'   # 东方财富
    ]
    
    # 下载数据（可选，如果已经有数据可以跳过）
    download_data = False  # 设置为True则下载数据，False则跳过下载
    
    if download_data:
        # 设置时间范围
        start_date = '20240101'
        end_date = '20240331'
        
        print(f"开始下载指定股票5分钟数据: {test_stocks}")
        print(f"下载时间范围: {start_date} - {end_date}")
        
        # 初始化下载进度跟踪器
        progress = DownloadProgress()
        
        # 下载5分钟数据
        result = xtdata.download_history_data2(
            stock_list=test_stocks,
            period='5m',
            start_time=start_date,
            end_time=end_date,
            callback=progress.on_progress,
            incrementally=True
        )
        
        print(f"下载请求已发出 ({result})，等待下载完成...")
        
        # 等待下载完成
        start_wait_time = time.time()
        timeout_seconds = 600  # 10分钟超时
        
        while not progress.check_finished():
            current_has, current_total = progress.get_progress()
            elapsed_time = time.time() - start_wait_time
            print(f"等待中... 进度: {current_has}/{current_total}. 已耗时: {elapsed_time:.1f} 秒", flush=True)
            
            if elapsed_time > timeout_seconds:
                print(f"下载超时 ({timeout_seconds}秒).")
                break
                
            time.sleep(5)
        
        if progress.check_finished():
            print("下载已完成。")
        else:
            print("下载可能未完成或超时。")
    
    # 测试获取股票的5分钟数据
    print("\n开始验证多只股票的5分钟数据...")
    
    # 1. 定义正确的时间格式（分钟级数据需要YYYYMMDDHHMMSS格式）
    # 选择最近的交易日，测试两个时间点
    try:
        # 获取最近的交易数据
        start_time = '20240318093000'  # 9:30开盘时间
        end_time = '20240318094500'    # 9:45时间点
        
        print(f"获取时间段: {start_time} - {end_time} 的数据")
        
        # 2. 获取特定字段数据 - 主要关注open字段
        test_data = xtdata.get_market_data(
            field_list=['open', 'close', 'high', 'low', 'volume'],  # 指定需要的字段
            stock_list=test_stocks,
            period='5m',
            start_time=start_time,
            end_time=end_time
        )
        
        # 3. 检查并打印返回的数据
        if test_data and len(test_data) > 0:
            print("\n成功获取到数据，字段包括:", list(test_data.keys()))
            
            # 特别关注open字段
            if 'open' in test_data:
                open_data = test_data['open']
                print("\n各股票开盘价数据:")
                print(open_data)
                
                # 计算每只股票的开盘价均值
                print("\n各股票开盘价均值:")
                for stock in test_stocks:
                    if stock in open_data.index:
                        avg_open = open_data.loc[stock].mean()
                        print(f"{stock}: {avg_open:.2f}")
                
                # 分析开盘价变化
                print("\n开盘价变化分析:")
                for stock in test_stocks:
                    if stock in open_data.index and len(open_data.columns) > 1:
                        first_open = open_data.loc[stock].iloc[0]
                        last_open = open_data.loc[stock].iloc[-1]
                        change = last_open - first_open
                        change_pct = (change / first_open) * 100 if first_open != 0 else 0
                        print(f"{stock}: 起始 {first_open:.2f} -> 结束 {last_open:.2f}, 变化: {change:.2f} ({change_pct:.2f}%)")
            else:
                print("未找到open字段数据")
            
            # 检查数据完整性
            print("\n数据完整性检查:")
            for field in test_data:
                print(f"{field} 字段形状: {test_data[field].shape}")
        else:
            print("未获取到有效数据")
    
    except Exception as e:
        print(f"验证数据时出错: {e}")
        import traceback
        traceback.print_exc()


def download_minute_data(stock_list, start_date=None, end_date=None):
    """
    下载指定股票的5分钟数据
    
    参数:
    stock_list (list): 股票代码列表，格式如 ['600519.SH', '000001.SZ']
    start_date (str): 开始日期，格式为'YYYYMMDD'，默认为一个月前
    end_date (str): 结束日期，格式为'YYYYMMDD'，默认为今天
    
    返回:
    bool: 下载是否成功完成
    """
    # 确保连接
    if not xtdata.get_client().is_connected():
        print("连接xtdata API...")
        xtdata.get_client().connect()
        time.sleep(1)
    
    # 设置默认日期范围（如果未提供）
    if start_date is None:
        # 默认为一个月前
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    if end_date is None:
        # 默认为今天
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"下载股票列表: {stock_list}")
    print(f"下载时间范围: {start_date} - {end_date}")
    
    # 初始化下载进度跟踪器
    progress = DownloadProgress()
    
    # 下载5分钟数据
    result = xtdata.download_history_data2(
        stock_list=stock_list,
        period='5m',
        start_time=start_date,
        end_time=end_date,
        callback=progress.on_progress,
        incrementally=True
    )
    
    print(f"下载请求已发出 ({result})，等待下载完成...")
    
    # 等待下载完成
    start_wait_time = time.time()
    timeout_seconds = 3600  # 1小时超时
    
    while not progress.check_finished():
        current_has, current_total = progress.get_progress()
        elapsed_time = time.time() - start_wait_time
        print(f"等待中... 进度: {current_has}/{current_total}. 已耗时: {elapsed_time:.1f} 秒", flush=True)
        
        if elapsed_time > timeout_seconds:
            print(f"下载超时 ({timeout_seconds}秒).")
            return False
            
        time.sleep(5)
    
    if progress.check_finished():
        print("下载已完成。")
        return True
    else:
        print("下载可能未完成或超时。")
        return False


def get_minute_data(stock_list, field_list=None, start_time=None, end_time=None, days=5):
    """
    获取指定股票的5分钟数据
    
    参数:
    stock_list (list): 股票代码列表，格式如 ['600519.SH', '000001.SZ']
    field_list (list): 需要获取的字段列表，默认为 ['open', 'high', 'low', 'close', 'volume']
    start_time (str): 开始时间，格式为'YYYYMMDDHHMMSS'，默认为days天前的开盘时间
    end_time (str): 结束时间，格式为'YYYYMMDDHHMMSS'，默认为当前时间
    days (int): 如果未提供start_time，则获取最近days天的数据
    
    返回:
    dict: 包含请求字段数据的字典，每个字段对应一个DataFrame
    """
    # 确保连接
    if not xtdata.get_client().is_connected():
        print("连接xtdata API...")
        xtdata.get_client().connect()
        time.sleep(1)
    
    # 设置默认字段列表
    if field_list is None:
        field_list = ['open', 'high', 'low', 'close', 'volume']
    
    # 设置默认时间范围
    if start_time is None:
        # 默认为days天前的开盘时间
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        start_time = f"{start_date}093000"  # 9:30开盘
    
    if end_time is None:
        # 默认为当前时间
        now = datetime.now()
        # 如果当前时间在15:00后，设置为15:00
        if now.hour >= 15:
            end_time = f"{now.strftime('%Y%m%d')}150000"  # 15:00收盘
        else:
            end_time = now.strftime('%Y%m%d%H%M%S')
    
    print(f"获取股票列表: {stock_list}")
    print(f"获取字段列表: {field_list}")
    print(f"获取时间范围: {start_time} - {end_time}")
    
    try:
        # 获取5分钟数据
        data = xtdata.get_market_data(
            field_list=field_list,
            stock_list=stock_list,
            period='5m',
            start_time=start_time,
            end_time=end_time
        )
        
        if data and len(data) > 0:
            print(f"成功获取到数据，字段包括: {list(data.keys())}")
            return data
        else:
            print("未获取到有效数据")
            return None
    
    except Exception as e:
        print(f"获取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_minute_data(data, stock_list, field='open'):
    """
    分析分钟级数据，特别关注特定字段
    
    参数:
    data (dict): 包含各字段数据的字典
    stock_list (list): 股票代码列表
    field (str): 要重点分析的字段，默认为'open'
    
    返回:
    DataFrame: 包含分析结果的DataFrame
    """
    if data is None or field not in data:
        print(f"没有找到{field}字段数据")
        return None
    
    field_data = data[field]
    results = []
    
    print(f"\n===== {field}字段数据分析 =====")
    
    for stock in stock_list:
        if stock in field_data.index:
            stock_data = field_data.loc[stock]
            
            # 基本统计信息
            mean_value = stock_data.mean()
            min_value = stock_data.min()
            max_value = stock_data.max()
            
            # 首尾变化
            first_value = stock_data.iloc[0]
            last_value = stock_data.iloc[-1]
            change = last_value - first_value
            change_pct = (change / first_value) * 100 if first_value != 0 else 0
            
            # 保存结果
            result = {
                '股票代码': stock,
                f'平均{field}': round(mean_value, 2),
                f'最小{field}': round(min_value, 2),
                f'最大{field}': round(max_value, 2),
                f'起始{field}': round(first_value, 2),
                f'结束{field}': round(last_value, 2),
                '变化': round(change, 2),
                '变化百分比(%)': round(change_pct, 2)
            }
            
            results.append(result)
            
            # 打印单只股票的分析结果
            print(f"\n{stock} 分析结果:")
            print(f"  平均{field}: {mean_value:.2f}")
            print(f"  最小{field}: {min_value:.2f}")
            print(f"  最大{field}: {max_value:.2f}")
            print(f"  起始{field}: {first_value:.2f} -> 结束{field}: {last_value:.2f}")
            print(f"  变化: {change:.2f} ({change_pct:.2f}%)")
        else:
            print(f"\n{stock}: 未找到数据")
    
    # 创建结果DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return None


if __name__ == "__main__":
    # Add basic logging setup if needed elsewhere
    import logging
    import threading # Import threading for the lock
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    download_a_shares_tick_data()
    # download_and_verify_stock_data()