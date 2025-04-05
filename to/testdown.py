import xtquant.xtdata as xtdata
from datetime import datetime, timedelta

test_code = '600556.SH'  # 测试股票代码
# 获取最近30个交易日
# 计算近一年的日期范围
end_date =  '20230928'
start_date = '20230401'

class DownloadProgress:
    def __init__(self):
        self.hasfinished = 0

    def on_progress(self, data):
        self.hasfinished = data['finished']
        print(f"下载进度: {self.hasfinished}/{data['total']}")
           
try:
    progress = DownloadProgress()
    xtdata.connect()
    down_result = xtdata.download_history_data2(
        stock_list=[test_code],
        period='1d',
        start_time=start_date,
        end_time=end_date,
        callback=progress.on_progress
    )
    print(down_result)
except Exception as e:
    print(f"下载失败: {e}")


test_data = xtdata.get_market_data(
            field_list=[],
            stock_list=[test_code],
            period='1d',
            start_time=start_date,
            end_time=end_date
            )
    
if test_data is not None:
    print("测试数据获取成功:")
    print(test_data)
else:
    print("测试数据获取失败")             

   
  