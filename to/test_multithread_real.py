"""
真实场景测试：多线程下载完整流程

测试内容：
1. 下载少量股票（10只）
2. 使用3个线程
3. 完整的数据处理流程
4. 验证数据正确性
"""

import sys
import os
sys.path.insert(0, 'C:\\Users\\86151\\.trae-cn\\pythons\\path\\to')

from datetime import datetime
import pandas as pd
import tushare as ts
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

logger.info("=" * 80)
logger.info("真实场景测试：多线程下载完整流程")
logger.info("=" * 80)

# 设置token
ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000')

# 测试参数：下载10只股票，2天数据，使用3个线程
test_stocks = [
    '000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '601318.SH',
    '000858.SZ', '002415.SZ', '600519.SH', '000333.SZ', '600887.SH'
]
start_date = '20260108'
end_date = '20260109'
max_workers = 3

logger.info(f"\n测试参数:")
logger.info(f"  股票: {len(test_stocks)} 只")
logger.info(f"  日期: {start_date} ~ {end_date}")
logger.info(f"  线程数: {max_workers}")
logger.info(f"  预期数据: {len(test_stocks)} × 2天 = {len(test_stocks) * 2} 行")

# 导入多线程下载模块
from download_daily_multithread import download_daily_multithread

# 开始下载测试
logger.info(f"\n开始多线程下载...")
start_time = datetime.now()

try:
    success = download_daily_multithread(
        stock_list=test_stocks,
        start_date=start_date,
        end_date=end_date,
        adj='qfq',
        max_workers=max_workers,
        batch_size=20,  # 小批次便于测试
        rows_per_save=20,
        resume=False,  # 不使用断点续传
        progress_file='to/local_data/test_multithread_real_progress.pkl'
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    logger.info(f"\n" + "=" * 80)
    logger.info("测试完成统计")
    logger.info("=" * 80)
    logger.info(f"耗时: {elapsed:.2f} 秒")
    logger.info(f"速度: {len(test_stocks) / elapsed:.2f} 只/秒")

    # 验证下载的数据
    logger.info(f"\n验证下载的数据...")
    data_path = Path('local_data/daily/202601/daily_data.parquet')

    if data_path.exists():
        df = pd.read_parquet(data_path)
        logger.info(f"数据文件存在: {data_path}")
        logger.info(f"总行数: {len(df)}")

        # 检查测试股票的数据
        test_df = df[df['stock_code'].isin(test_stocks)]
        logger.info(f"测试股票数据: {len(test_df)} 行")

        # 检查日期范围
        logger.info(f"日期范围: {test_df['date'].min()} ~ {test_df['date'].max()}")

        # 检查每只股票的记录数
        logger.info(f"\n每只股票的记录数:")
        for stock in test_stocks:
            count = len(test_df[test_df['stock_code'] == stock])
            logger.info(f"  {stock}: {count} 行")

        # 检查是否有重复
        dup_count = test_df.duplicated(subset=['stock_code', 'date_str']).sum()
        if dup_count > 0:
            logger.warning(f"发现 {dup_count} 条重复数据！")
        else:
            logger.info(f"无重复数据 ✓")

        # 显示样本数据
        logger.info(f"\n样本数据:")
        sample = test_df[['stock_code', 'date', 'open', 'close']].head(10)
        print(sample)

        logger.info(f"\n" + "=" * 80)
        if success:
            logger.info("测试成功！多线程下载功能正常")
            logger.info("=" * 80)
        else:
            logger.warning("部分下载失败，请检查日志")
            logger.info("=" * 80)
    else:
        logger.error(f"数据文件不存在: {data_path}")
        logger.error("测试失败！")

except Exception as e:
    logger.error(f"测试过程中出现异常: {e}")
    import traceback
    traceback.print_exc()
