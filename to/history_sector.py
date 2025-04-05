import xtquant.xtdata as xtdata

xtdata.download_sector_data()
# 获取板块列表
print( xtdata.get_sector_list())

# 获取沪深A股板块列表
stock_list = xtdata.get_stock_list_in_sector('TGNDeepSeek概念')
print(stock_list)