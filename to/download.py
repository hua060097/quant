import xtquant.xtdata as xtdata
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import numpy as np
import threading
from data_loaders import get_stock_list
import tushare as ts

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
         



def down_sw_sector():
    """
    获取申万一级行业分类下的股票列表，并保存为JSON格式

    功能：
    1. 获取所有申万一级行业列表
    2. 筛选出以"sw1"开头且不以"加权"结尾的行业
    3. 遍历每个行业，获取对应的股票列表
    4. 保存为{股票代码: 所属行业}的格式到to/local_data/sw_sector/sw_sector.json
    """
    import json

    print("开始获取申万一级行业分类数据...")

    # 下载板块数据
    xtdata.download_sector_data()
    sector_list = xtdata.get_sector_list()


    print(f"获取到总行业数量: {len(sector_list)}")

    # 筛选条件：以"sw1"开头，且不以"加权"结尾
    sw1_sectors = []
    for sector_name in sector_list:
        if sector_name.startswith('SW3') and not sector_name.endswith('加权'):
            sw1_sectors.append(sector_name)

    print(f"筛选出申万一级行业数量: {len(sw1_sectors)}")
    print(f"申万一级行业列表: {sw1_sectors}")

    # 创建股票到行业的映射字典
    stock_sector_mapping = {}
    total_stocks = 0

    # 遍历每个申万一级行业
    for sector_name in sw1_sectors:
        print(f"\n正在处理行业: {sector_name}")

        try:
            # 获取该行业下的股票列表
            stock_list = xtdata.get_stock_list_in_sector(sector_name)

            if stock_list and len(stock_list) > 0:
                print(f"  获取到股票数量: {len(stock_list)}")
                print(f"  股票列表示例: {stock_list[:5]}")

                # 将每只股票映射到该行业
                for stock_code in stock_list:
                    stock_sector_mapping[stock_code] = sector_name

                total_stocks += len(stock_list)
            else:
                print(f"  未获取到股票数据")

        except Exception as e:
            print(f"  处理行业 {sector_name} 时出错: {e}")
            continue

    print(f"\n总计处理股票数量: {total_stocks}")
    print(f"股票到行业映射字典长度: {len(stock_sector_mapping)}")

    # 确保目录存在
    output_dir = 'to/local_data/sw_sector3'
    os.makedirs(output_dir, exist_ok=True)

    # 保存到JSON文件
    output_file = os.path.join(output_dir, 'sw_sector.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stock_sector_mapping, f, ensure_ascii=False, indent=2)

        print(f"\n成功保存股票-行业映射数据到: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file)} 字节")

        # 验证保存的数据
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        print(f"验证：成功读取 {len(loaded_data)} 条股票-行业映射记录")

        # 显示一些示例数据
        sample_items = list(loaded_data.items())[:10]
        print("\n股票-行业映射示例:")
        for stock_code, sector in sample_items:
            print(f"  {stock_code}: {sector}")

        return True

    except Exception as e:
        print(f"\n保存数据时出错: {e}")
        return False
    





def download_a_shares_tick_data(start_date=None, end_date=None, batch_size=500):
    """
    分批下载沪深A股日线数据

    迅投 xtdata.download_history_data2 内部限制每次最多约500只股票，
    超过限制会导致下载极慢或部分失败。此函数将全量股票列表分批下载。

    Args:
        start_date: 开始日期，格式YYYYMMDD，默认 '20260101'
        end_date:   结束日期，格式YYYYMMDD，默认 '20260126'
        batch_size: 每批下载股票数量，默认500
    """
    try:
        # Ensure connection
        if not xtdata.get_client().is_connected():
            logger.info("Connecting to xtdata API...")
            xtdata.get_client().connect()
            time.sleep(1) # Allow time for connection
    except Exception as e:
        print(f"连接API时出错: {e}")
        return

    period = "1d"

    # 设置时间范围：如果未提供参数则使用默认值
    if start_date is None:
        start_date = '20260101'
    if end_date is None:
        end_date = '20260126'

    # 获取股票列表
    test_list = get_stock_list()
    test_list = test_list + ['399006.SZ']
    total_stocks = len(test_list)
    total_batches = (total_stocks + batch_size - 1) // batch_size
    print(f"下载时间范围: {start_date} - {end_date}")
    print(f"总股票: {total_stocks} 只, 每批 {batch_size} 只, 共 {total_batches} 批")

    # 分批下载
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_stocks)
        batch = test_list[start_idx:end_idx]
        batch_num = batch_idx + 1

        print(f"\n{'='*60}")
        print(f"第 {batch_num}/{total_batches} 批: {len(batch)} 只 ({batch[0]} ~ {batch[-1]})")
        print(f"{'='*60}")

        progress = DownloadProgress()
        result = xtdata.download_history_data2(
            stock_list=batch,
            period=period,
            start_time=start_date,
            end_time=end_date,
            callback=progress.on_progress,
            incrementally=True
        )
        print(f"第{batch_num}批下载请求已发出，等待完成...")

        start_wait_time = time.time()
        timeout_seconds = 1800  # 每批30分钟超时

        while not progress.check_finished():
            current_has, current_total = progress.get_progress()
            elapsed = time.time() - start_wait_time
            overall_done = start_idx + min(current_has, current_total)
            print(f"  批次: {current_has}/{current_total}, 总体: {overall_done}/{total_stocks}, 耗时: {elapsed:.0f}s", flush=True)
            if elapsed > timeout_seconds:
                print(f"  第{batch_num}批超时({timeout_seconds}s)，继续下一批...")
                break
            time.sleep(3)

        if progress.check_finished():
            print(f"第{batch_num}批完成。")
        else:
            print(f"第{batch_num}批未完成或超时。")

        # 批次间短暂休息
        if batch_num < total_batches:
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"全部 {total_batches} 批下载完成，共 {total_stocks} 只股票")
    print(f"{'='*60}")

    # 验证数据
    test_code = '399006.SZ'
    print(f"\n验证股票 {test_code} 的 {period} 数据...")
    try:
        test_data = xtdata.get_market_data(
            field_list=[],
            stock_list=[test_code],
            period=period,
            start_time=end_date,
            end_time=end_date,
            dividend_type='front'
        )
        print(f"验证成功: {list(test_data.keys()) if test_data else '空数据'}")
    except Exception as e:
        print(f"验证时出错: {e}")


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


def download_and_check_financial_data():
    """
    下载所有股票（沪深A股和北交所）的财务数据，并验证其中一只股票的财务数据
    包括total_capital和circulating_capital
    """
    import json
    import os
    
    # 确保连接
    if not xtdata.get_client().is_connected():
        print("连接xtdata API...")
        xtdata.connect()
        time.sleep(1)
    
    # 获取沪深A股股票列表
    try:
        xtdata.download_sector_data()
        a_share_list = xtdata.get_stock_list_in_sector('沪深A股')
        print(f"获取到沪深A股数量: {len(a_share_list)}")
    except Exception as e:
        print(f"获取沪深A股列表时出错: {e}")
        a_share_list = []
    
    # 获取北交所股票列表
    bj_stock_list = []
    try:
        # 尝试从bjstock.json读取
        json_path = 'to/local_data/bjstock.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                bj_stock_list = json.load(f)
            print(f"从json文件读取到北交所股票数量: {len(bj_stock_list)}")
        else:
            # 如果文件不存在，尝试使用API获取
            try:
                from data_loaders import get_bj_stock_list
                bj_stock_list = get_bj_stock_list()
                print(f"通过API获取到北交所股票数量: {len(bj_stock_list)}")
            except Exception as e:
                print(f"获取北交所股票列表时出错: {e}")
    except Exception as e:
        print(f"获取北交所股票列表时出错: {e}")
    
    # 合并股票列表
    stock_list = a_share_list + bj_stock_list
    print(f"合计股票数量: {len(stock_list)}")
    
    if not stock_list:
        print("未能获取到股票列表，请检查API连接或权限。")
        return
    
    
    # 创建用于保存结果的列表
    stock_data_list = []
    
    try:
        for stock in stock_list:
            # 获取财务数据
            financial_data = xtdata.get_instrument_detail(stock)
            print(f"股票代码 {stock}， 总股本 {financial_data.get('TotalVolume')}, 流通股本 {financial_data.get('FloatVolume')}")
            
            # 将数据添加到列表中
            stock_data_list.append({
                'stock': stock,
                'total_capital': financial_data.get('TotalVolume'),
                'circulating_capital': financial_data.get('FloatVolume')
            })
        
        # 确保目录存在
        os.makedirs('to/local_data', exist_ok=True)
        
        # 将数据转换为DataFrame并保存为parquet文件
        df = pd.DataFrame(stock_data_list)
        parquet_path = 'to/local_data/stock_capital_data.parquet'
        df.to_parquet(parquet_path, compression='snappy', index=False)
        
        print(f"已成功将{len(stock_data_list)}只股票的总股本和流通股本数据保存至{parquet_path}")
        
        # 验证数据已正确保存
        try:
            verification_df = pd.read_parquet(parquet_path)
            print(f"验证数据：已成功读取{len(verification_df)}条记录")
            print(verification_df.head())
        except Exception as e:
            print(f"验证数据时出错: {e}")
    
    except Exception as e:
        print(f"验证财务数据时出错: {e}")
        import traceback
        traceback.print_exc()


def download_all_stock_ipo_dates():
    """
    获取所有股票的上市日期并保存到JSON文件

    功能：
    1. 获取沪深A股和北交所股票列表
    2. 逐个获取股票的上市日期（OpenDate）
    3. 保存到 to/local_data/ipo_date.json

    返回：
    dict: {股票代码: 上市日期}
    """
    import json
    import os

    print("开始获取所有股票的上市日期...")

    # 确保连接
    if not xtdata.get_client().is_connected():
        print("连接xtdata API...")
        xtdata.connect()
        time.sleep(1)

    # 1. 获取股票列表（参考 download_a_shares_tick_data 的逻辑）
    stock_list = get_stock_list()

    if not stock_list:
        print("未能获取到股票列表")
        return {}

    print(f"获取到 {len(stock_list)} 只股票")

    # 2. 逐个获取上市日期
    ipo_dates = {}
    failed_stocks = []

    for i, stock in enumerate(stock_list, 1):
        try:
            # 获取股票详细信息
            detail = xtdata.get_instrument_detail(stock)

            # 提取上市日期
            if detail and 'OpenDate' in detail:
                ipo_date = detail['OpenDate']
                ipo_dates[stock] = ipo_date

                # 每100只股票打印一次进度
                if i % 100 == 0:
                    print(f"进度: {i}/{len(stock_list)}")

            else:
                print(f"警告: {stock} 没有OpenDate字段")
                failed_stocks.append(stock)

        except Exception as e:
            print(f"错误: {stock} 获取失败 - {e}")
            failed_stocks.append(stock)
            continue

    print(f"\n成功获取 {len(ipo_dates)} 只股票的上市日期")
    if failed_stocks:
        print(f"失败 {len(failed_stocks)} 只股票")

    # 3. 保存到JSON文件
    output_dir = 'to/local_data'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'ipo_date.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ipo_dates, f, ensure_ascii=False, indent=2)

        print(f"成功保存到: {output_file}")

        # 显示前5个示例
        sample_items = list(ipo_dates.items())[:5]
        print("\n示例数据:")
        for stock_code, ipo_date in sample_items:
            print(f"  {stock_code}: {ipo_date}")

        return ipo_dates

    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None


if __name__ == "__main__":
    # Add basic logging setup if needed elsewhere
    import logging
    import argparse
    import threading # Import threading for the lock
    from datetime import datetime

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载A股Tick数据和股票列表数据')
    parser.add_argument('--start', '-s', type=str, default=None,
                        help='开始日期 (YYYYMMDD格式，可选，目前download_a_shares_tick_data不使用此参数)')
    parser.add_argument('--end', '-e', type=str, default=None,
                        help='结束日期 (YYYYMMDD格式，可选，目前download_a_shares_tick_data不使用此参数)')
    args = parser.parse_args()

    print("=== 申万一级行业股票映射下载程序 ===")
    if args.start:
        print(f"开始日期: {args.start}")
    if args.end:
        print(f"结束日期: {args.end}")

    # xtdata.download_sector_data()
    # sector_list = xtdata.get_sector_list()
    # print(f"sector_list:{sector_list}")

    # 调用完善后的down_sw_sector函数
    # success = down_sw_sector()

    # test_data = xtdata.get_market_data(
    #         field_list=[], # Empty list to get all available fields
    #         stock_list=['300689.SZ'],
    #         period='5m',
    #         # start_time='20230316093500',
    #         # end_time='20230317094500'
    #         start_time='20250612',
    #         end_time='20250612',
    #         dividend_type='front'
    #     )
    # print(test_data)
    # ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')
    # df = ts.pro_bar(ts_code='300622.SZ', adj='qfq', start_date='20221001', end_date='20251231')
    # # 只打印 20240813~20240819 日的数据
    # filtered_df = df[(df['trade_date'] >= '20240813') & (df['trade_date'] <= '20240819')]
    # print(filtered_df)

    # 测试单只股票
    # ipoDate = xtdata.get_instrument_detail("000001.SZ")['OpenDate']
    # print(ipoDate)

    # 调用新函数获取所有股票上市日期
    # download_all_stock_ipo_dates()

    # 其他可用函数（注释状态）
    # download_a_shares_tick_data(start_date=args.start, end_date=args.end)
    ts.set_token('gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b')
    pro = ts.pro_api()
    df_st = pro.stock_st(trade_date='20250813')
    print(df_st)

    # download_and_verify_stock_data()
    # download_and_check_financial_data()
    # ts.set_token('gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b')
    # pro = ts.pro_api()

    # #获取浦发银行60000.SH的历史分钟数据

    # df = pro.stk_mins(ts_code='600000.SH', freq='5min', start_date='2023-08-25 09:00:00', end_date='2023-08-25 19:00:00')

    # try:
    #     # Ensure connection
    #     if not xtdata.get_client().is_connected():
    #         logger.info("Connecting to xtdata API...")
    #         xtdata.get_client().connect()
    #         time.sleep(1) # Allow time for connection
    # except Exception as e:
    #     print(f"连接API时出错: {e}")
    # seq =  xtdata.subscribe_whole_quote(['301188.SZ'])
    # xtdata.unsubscribe_quote(seq)
    # while 1:
    #     full_tick = xtdata.get_full_tick(['301188.SZ'])
    #     print(full_tick['301188.SZ']['lastPrice'])  # 最新价
    #     time.sleep(3)  # 每3秒更新
   
    # def on_data(datas):
    #     # datas 是一个字典，格式为 { stock1 : data1, stock2 : data2, ... }
    #     print(datas)  # 打印所有变动的数据
    #     # 例如，获取最新价：datas['301188.SZ']['lastPrice']

    # # 订阅并传入回调函数
    # seq = xtdata.subscribe_whole_quote(['301188.SZ'], callback=on_data)

    # # 进入消息循环，等待并处理回调事件
    # xtdata.run()