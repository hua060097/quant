"""
测试多线程下载功能

下载少量股票验证多线程功能是否正常
"""

import sys
sys.path.insert(0, 'C:\\Users\\86151\\.trae-cn\\pythons\\path\\to')

from download_daily_multithread import download_daily_multithread
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

logger.info("=" * 80)
logger.info("测试多线程下载功能")
logger.info("=" * 80)

# 测试参数：下载3只股票，2天数据，使用3个线程
test_stocks = ['000001.SZ', '000002.SZ', '600000.SH']
start_date = '20260108'
end_date = '20260109'
max_workers = 3

logger.info(f"\n测试参数:")
logger.info(f"  股票: {len(test_stocks)} 只")
logger.info(f"  日期: {start_date} ~ {end_date}")
logger.info(f"  线程数: {max_workers}")

# 开始测试
success = download_daily_multithread(
    stock_list=test_stocks,
    start_date=start_date,
    end_date=end_date,
    adj='qfq',
    max_workers=max_workers,
    batch_size=10,
    rows_per_save=10,
    resume=False,  # 不使用断点续传
    progress_file='to/local_data/test_multithread_progress.pkl'
)

logger.info("\n" + "=" * 80)
if success:
    logger.info("测试成功！多线程下载功能正常")
else:
    logger.error("测试失败，请检查日志")
logger.info("=" * 80)
