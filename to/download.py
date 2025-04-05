import xtquant.xtdata as xtdata
from datetime import datetime, timedelta
import time
import os

class DownloadProgress:
    def __init__(self):
        self.hasfinished = 0
        self.total_items = 0 # 新增：用于存储总数
        self.is_finished = False # 新增：完成标志
        self._lock = threading.Lock() # Add a lock for thread safety

    def on_progress(self, data):
        with self._lock: # Ensure thread-safe updates
            # 确保data是字典并且包含所需键
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
    start_date = '20220101'
    end_date = '20230201'

    print(f"下载时间范围: {start_date} - {end_date}")

    #初始化下载进度跟踪器
    progress = DownloadProgress()


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
    test_code = '600051.SH'  # 贵州茅台
    print(f"验证股票 {test_code} 的5分钟数据...")

    try:
        test_data = xtdata.get_market_data(
            field_list=[], # Empty list to get all available fields
            stock_list=[test_code],
            period='1d',
            start_time=start_date,
            end_time=end_date
        )

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