import xtquant.xtdata as xtdata
from datetime import datetime, timedelta
import time
import os

class DownloadProgress:
    def __init__(self):
        self.hasfinished = 0

    def on_progress(self, data):
        self.hasfinished = data['finished']
        print(f"下载进度: {self.hasfinished}/{data['total']}")

def download_a_shares_tick_data():
    # 获取沪深A股所有股票代码
    stock_list = xtdata.get_stock_list_in_sector('沪深A股')
    print(f"获取到的股票代码数量: {len(stock_list)}")
    
    # 设置时间范围：2023年6月至今
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = '20230101'
    print(f"下载时间范围: {start_date} - {end_date}")
    
    # 初始化下载进度跟踪器
    progress = DownloadProgress()
    
    # 下载tick数据
    print("开始下载沪深A股tick数据...")
    result = xtdata.download_history_data2(
        stock_list=stock_list,
        period='5m',  
        start_time=start_date,
        end_time=end_date,
        callback=progress.on_progress,
        incrementally = True
    )

    result1 = xtdata.download_history_data2(
        stock_list=stock_list,
        period='1d',  
        start_time=start_date,
        end_time=end_date,
        callback=progress.on_progress,
        incrementally = True
    )
    print(f"下载请求结果: {result}")
    
    while progress.hasfinished < 2219616:
        print(f"当前进度: {progress.hasfinished}/{2219616}")
        time.sleep(1)  # 每5秒检查一次
    print("下载完成")
    
    # 测试获取某只股票的tick数据
    test_code = '600519.SH'  # 贵州茅台
    print(f"验证股票 {test_code} 的tick数据...")
    
    test_data = xtdata.get_market_data(
        field_list=[],
        stock_list=[test_code],
        period='1d',
        start_time=start_date,
        end_time=end_date
    )
    
    if test_data is not None and len(test_data) > 0:
        # 统计数据
        data_days = len(set([t.split(' ')[0] for t in test_data['time'].values[0]]))
        data_ticks = len(test_data['time'].values[0])
        
        print("测试数据获取成功:")
        print(f"数据天数: {data_days}")
        print(f"总tick数: {data_ticks}")
        
        if data_ticks > 0:
            print(f"第一个tick: 时间={test_data['time'].values[0][0]}, 价格={test_data['price'].values[0][0]}")
            print(f"最后一个tick: 时间={test_data['time'].values[0][-1]}, 价格={test_data['price'].values[0][-1]}")
    else:
        print("测试数据获取失败")

if __name__ == "__main__":
    download_a_shares_tick_data()