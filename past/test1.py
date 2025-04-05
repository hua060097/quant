import xtquant.xtdata as xtdata

xtdata.download_sector_data()
# 获取板块列表
#print( xtdata.get_sector_list())

# 获取沪深A股板块列表
stock_list = xtdata.get_stock_list_in_sector('TGNDeepSeek概念')
print(stock_list)

# 下载日线历史数据（时间范围需包含目标日期）
xtdata.download_history_data(
    stock_code=['TGNDeepSeek概念.SH'],
    period='1d',  # 日线数据
    start_time='20240101',  # 开始日期
    end_time='20241231'     # 结束日期
)

historical_data = xtdata.get_market_data(
    field_list=[],  # 涨跌幅字段
    stock_list=['TGNDeepSeek概念.SH'],
    period='1d',
    start_time='20241001',  # 目标日期范围
    end_time='20241001',
    count=1  # 仅取1条数据
)
print(historical_data)