import time
import tushare as ts
from tushare.pro import client as _ts_client
_ts_client.DataApi._DataApi__http_url = "https://tu.brze.top"  # 这行代码是替换代理，后续所有client的创建都不需要再次申明

pro = ts.pro_api('nEynGpEVcU8TO1E14uDkl6XCIXjma-ThZiFIkTbkL9o')  # 你的专属token
df = pro.stk_mins(ts_code='600000.SH', freq='5min', start_date='2023-08-25 09:00:00', end_date='2023-08-25 19:00:00')
print(f"{df}")

# df = pro.daily(trade_date="20260423")
print("[SDK] columns=%s rows=%d" % (list(df.columns) if df is not None else None,
                                    len(df) if df is not None else 0))
# time.sleep(0.6) #注意控制频次，避免被限流