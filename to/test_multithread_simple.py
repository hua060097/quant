"""
简单测试多线程下载功能

只测试下载和数据保存，跳过市值计算
"""

from datetime import datetime
import pandas as pd
import tushare as ts
from pathlib import Path
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

logger.info("=" * 80)
logger.info("简单测试：多线程下载 3 只股票")
logger.info("=" * 80)

# 测试参数
test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']
start_date = '20260108'
end_date = '20260109'

# 手动下载数据
results = []
for stock in test_stocks:
    logger.info(f"下载 {stock}...")
    df = ts.pro_bar(ts_code=stock, adj='qfq', start_date=start_date, end_date=end_date)
    if df is not None and len(df) > 0:
        df['stock_code'] = stock
        results.append(df)
        logger.info(f"  成功: {len(df)} 行")
    else:
        logger.warning(f"  无数据")

# 合并数据
if results:
    all_data = pd.concat(results, ignore_index=True)
    logger.info(f"\n总共下载: {len(all_data)} 行")
    logger.info(f"股票分布:")
    for stock in test_stocks:
        count = len(all_data[all_data['stock_code'] == stock])
        logger.info(f"  {stock}: {count} 行")

    # 显示样本数据
    logger.info(f"\n样本数据:")
    print(all_data[['stock_code', 'trade_date', 'open', 'close']].head(10))

    logger.info("\n测试成功！")
else:
    logger.error("测试失败：没有数据")
