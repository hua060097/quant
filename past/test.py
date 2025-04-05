from xtquant import xtdata
import time

# 设定一个标的列表
code_list = ["000001.SZ"]
# 设定获取数据的周期
period = "1d"
print("test1")

# 下载标的行情数据
#if 1:
    ## 为了方便用户进行数据管理，xtquant的大部分历史数据都是以压缩形式存储在本地的
    ## 比如行情数据，需要通过download_history_data下载，财务数据需要通过
    ## 所以在取历史数据之前，我们需要调用数据下载接口，将数据下载到本地
for i in code_list:
    xtdata.download_history_data(i,period=period,incrementally=True) # 增量下载行情数据（开高低收,等等）到本地
    
xtdata.download_financial_data(code_list) # 下载财务数据到本地
xtdata.download_sector_data() # 下载板块数据到本地
    # 更多数据的下载方式可以通过数据字典查询


print("test111")
# 读取本地历史行情数据
history_data = xtdata.get_market_data_ex([],code_list,period=period,count=-1)
print(history_data)
print("=" * 20)