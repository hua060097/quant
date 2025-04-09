# import requests

# url = f"https://tsanghi.com/api/fin/stock/XSHG/5min?token=demo&ticker=600519&order=2"
# data = requests.get(url).json()
# print(data)


# import akshare as ak

# stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol='sh600751', period='5', adjust="qfq")
# print(stock_zh_a_minute_df)

import baostock as bs
import pandas as pd
import time
from multiprocessing import Process, Queue, cpu_count
import os
import math
import sys
import logging
import random
import datetime
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('worker_test')

# 添加to目录到路径
sys.path.append("to")

# 导入我们的下载工作函数
from advanced_quant_data import download_minute_data_process_worker

def download_stock_batch(stock_batch, start_date, end_date, result_queue, process_id):
    """每个进程下载一批股票的数据"""
    process_start = time.time()
    
    # 每个进程只登录一次
    print(f"进程 {process_id} (PID: {os.getpid()}) 开始登录")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"进程 {process_id} 登录失败: {lg.error_msg}")
        result_queue.put((None, process_id))
        return
    
    print(f"进程 {process_id} 登录成功，开始下载 {len(stock_batch)} 只股票数据")
    
    # 处理批次中的每只股票
    results = {}
    for stock_code in stock_batch:
        stock_start = time.time()
        print(f"进程 {process_id} 开始下载 {stock_code} 数据")
        
        # 查询数据
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"进程 {process_id} 查询 {stock_code} 失败: {rs.error_msg}")
            continue
        
        # 获取数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 如果有数据，转换为DataFrame
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            results[stock_code] = df
            print(f"进程 {process_id} 成功获取 {stock_code} 数据: {len(df)} 条记录，耗时: {time.time() - stock_start:.2f}秒")
        else:
            print(f"进程 {process_id} 获取 {stock_code} 数据: 无记录，耗时: {time.time() - stock_start:.2f}秒")
    
    # 登出系统（每个进程只登出一次）
    bs.logout()
    process_end = time.time()
    print(f"进程 {process_id} 处理完成，共处理 {len(stock_batch)} 只股票，总耗时: {process_end - process_start:.2f}秒")
    
    # 返回结果
    result_queue.put((results, process_id))

def main():
    # 测试参数
    start_date = '20230101'
    end_date = '20230131'  # 增加测试数据量
    
    # 待测试的股票列表
    stock_list = [
        "sh.600000", "sh.600036", "sh.601398", "sh.601288",
        "sz.000001", "sz.000002", "sz.000063", "sz.000333"
    ]
    
    # 需要下载的字段
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    # 使用较少进程（最多2个）
    num_processes = min(2, cpu_count())
    logger.info(f"CPU核心数: {cpu_count()}, 使用进程数: {num_processes}")
    
    # 先测试单进程顺序下载所有股票
    logger.info("\n===== 单进程顺序下载测试 =====")
    single_start = time.time()
    
    # 登录
    lg = bs.login()
    logger.info(f"单进程登录: {lg.error_code} - {lg.error_msg}")
    
    single_results = {}
    for stock in stock_list:
        logger.info(f"开始下载 {stock} 数据...")
        rs = bs.query_history_k_data_plus(
            stock,
            "date,time,code,open,high,low,close,volume,amount",
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            frequency="5",
            adjustflag="3"
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            single_results[stock] = df
            logger.info(f"获取 {stock} 数据: {len(df)} 条记录")
        else:
            logger.info(f"获取 {stock} 数据: 无记录")
    
    bs.logout()
    single_end = time.time()
    single_time = single_end - single_start
    logger.info(f"单进程总耗时: {single_time:.2f}秒")
    
    # 清空单进程测试结果，释放内存
    single_results.clear()
    
    # 稍作等待，确保与服务器连接已完全断开
    time.sleep(2)
    
    # 将股票列表分成几个批次
    batch_size = math.ceil(len(stock_list) / num_processes)
    stock_batches = [stock_list[i:i + batch_size] for i in range(0, len(stock_list), batch_size)]
    
    # 多进程批量下载测试
    logger.info(f"\n===== 多进程批量下载测试 ({num_processes}个进程) =====")
    multi_start = time.time()
    
    # 创建结果队列
    result_queue = Queue()
    
    # 创建并启动进程
    processes = []
    for i, batch in enumerate(stock_batches):
        p = Process(
            target=download_minute_data_process_worker,
            args=(batch, start_date, end_date, fields, i, result_queue)
        )
        processes.append(p)
        p.start()
        # 减少延迟时间
        time.sleep(0.2)
    
    # 收集结果
    multi_results = {}
    for _ in range(len(stock_batches)):
        try:
            results, process_id = result_queue.get(timeout=300)  # 5分钟超时
            if results is not None:
                multi_results.update(results)
                logger.info(f"已收集进程 {process_id} 的数据: {len(results)} 只股票")
        except Exception as e:
            logger.error(f"获取队列数据时出错: {str(e)}")
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    multi_end = time.time()
    multi_time = multi_end - multi_start
    logger.info(f"多进程总耗时: {multi_time:.2f}秒")
    
    # 计算加速比
    speedup = single_time / multi_time if multi_time > 0 else 0
    logger.info(f"\n===== 性能对比 =====")
    logger.info(f"单进程耗时: {single_time:.2f}秒")
    logger.info(f"多进程耗时: {multi_time:.2f}秒")
    logger.info(f"加速比: {speedup:.2f}倍")
    
    # 检查数据一致性
    logger.info("\n===== 数据完整性检查 =====")
    success_count = len(multi_results)
    logger.info(f"成功下载 {success_count}/{len(stock_list)} 只股票的数据")
    
    # 显示每只股票的数据条数
    for stock, df in multi_results.items():
        logger.info(f"{stock}: {len(df)} 条记录")

if __name__ == "__main__":
    main()
