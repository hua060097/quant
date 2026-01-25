"""
测试Tushare的ts.pro_bar是否支持批量下载

测试场景：
1. 单个股票（基准测试）
2. 多个股票用逗号分隔
3. 多个股票用列表
4. 对比性能
"""

import tushare as ts
import time
import pandas as pd
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_stock():
    """测试1：单个股票下载"""
    logger.info("="*80)
    logger.info("测试1：单个股票下载")
    logger.info("="*80)

    pro = ts.pro_api()

    stock_code = "600000.SH"
    start_date = "20250101"
    end_date = "20250110"

    logger.info(f"股票代码: {stock_code}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    start_time = time.time()

    try:
        df = ts.pro_bar(
            ts_code=stock_code,
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        elapsed = time.time() - start_time

        if df is not None and len(df) > 0:
            logger.info(f"✅ 成功下载数据")
            logger.info(f"   数据行数: {len(df)}")
            logger.info(f"   耗时: {elapsed:.3f}秒")
            logger.info(f"   列名: {list(df.columns)}")
            logger.info(f"   前3行数据:\n{df.head(3)}")
            return {'success': True, 'rows': len(df), 'time': elapsed}
        else:
            logger.warning(f"⚠️ 返回数据为空")
            return {'success': False, 'error': 'Empty data', 'time': elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ 失败: {e}")
        return {'success': False, 'error': str(e), 'time': elapsed}


def test_multiple_stocks_comma_separated():
    """测试2：多个股票用逗号分隔"""
    logger.info("="*80)
    logger.info("测试2：多个股票用逗号分隔")
    logger.info("="*80)

    pro = ts.pro_api()

    # 测试3只股票
    stock_codes = "600000.SH,000001.SZ,600519.SH"
    start_date = "20250101"
    end_date = "20250110"

    logger.info(f"股票代码: {stock_codes}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    start_time = time.time()

    try:
        df = ts.pro_bar(
            ts_code=stock_codes,
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        elapsed = time.time() - start_time

        if df is not None and len(df) > 0:
            logger.info(f"✅ 成功下载数据")
            logger.info(f"   数据行数: {len(df)}")

            # 检查是否包含多只股票
            if 'ts_code' in df.columns:
                unique_stocks = df['ts_code'].nunique()
                logger.info(f"   股票数量: {unique_stocks}")
                logger.info(f"   股票列表: {df['ts_code'].unique().tolist()}")

            logger.info(f"   耗时: {elapsed:.3f}秒")
            logger.info(f"   列名: {list(df.columns)}")
            logger.info(f"   前5行数据:\n{df.head(5)}")
            return {'success': True, 'rows': len(df), 'time': elapsed, 'stocks': unique_stocks}
        else:
            logger.warning(f"⚠️ 返回数据为空")
            return {'success': False, 'error': 'Empty data', 'time': elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ 失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        return {'success': False, 'error': str(e), 'time': elapsed}


def test_multiple_stocks_list():
    """测试3：多个股票用列表"""
    logger.info("="*80)
    logger.info("测试3：多个股票用列表")
    logger.info("="*80)

    pro = ts.pro_api()

    # 测试3只股票
    stock_codes = ["600000.SH", "000001.SZ", "600519.SH"]
    start_date = "20250101"
    end_date = "20250110"

    logger.info(f"股票代码: {stock_codes}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    start_time = time.time()

    try:
        df = ts.pro_bar(
            ts_code=stock_codes,  # 传入列表
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        elapsed = time.time() - start_time

        if df is not None and len(df) > 0:
            logger.info(f"✅ 成功下载数据")
            logger.info(f"   数据行数: {len(df)}")

            # 检查是否包含多只股票
            if 'ts_code' in df.columns:
                unique_stocks = df['ts_code'].nunique()
                logger.info(f"   股票数量: {unique_stocks}")
                logger.info(f"   股票列表: {df['ts_code'].unique().tolist()}")

            logger.info(f"   耗时: {elapsed:.3f}秒")
            logger.info(f"   列名: {list(df.columns)}")
            logger.info(f"   前5行数据:\n{df.head(5)}")
            return {'success': True, 'rows': len(df), 'time': elapsed, 'stocks': unique_stocks}
        else:
            logger.warning(f"⚠️ 返回数据为空")
            return {'success': False, 'error': 'Empty data', 'time': elapsed}

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ 失败: {e}")
        logger.error(f"   错误类型: {type(e).__name__}")
        return {'success': False, 'error': str(e), 'time': elapsed}


def test_batch_download_performance():
    """测试4：批量下载性能对比"""
    logger.info("="*80)
    logger.info("测试4：性能对比")
    logger.info("="*80)

    stock_list = ["600000.SH", "000001.SZ", "600519.SH", "000002.SZ", "601318.SH"]
    start_date = "20250101"
    end_date = "20250110"

    logger.info(f"测试股票: {len(stock_list)}只")
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    # 方式1：逐个下载
    logger.info("\n--- 方式1：逐个下载 ---")
    start_time = time.time()
    total_rows_1 = 0

    for stock in stock_list:
        try:
            df = ts.pro_bar(ts_code=stock, adj='qfq', start_date=start_date, end_date=end_date)
            if df is not None:
                total_rows_1 += len(df)
        except Exception as e:
            logger.error(f"  {stock} 失败: {e}")

    elapsed_1 = time.time() - start_time
    logger.info(f"总耗时: {elapsed_1:.3f}秒")
    logger.info(f"总行数: {total_rows_1}")

    # 方式2：批量下载（如果支持）
    logger.info("\n--- 方式2：批量下载（逗号分隔）---")
    start_time = time.time()

    try:
        df_batch = ts.pro_bar(
            ts_code=','.join(stock_list),
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        elapsed_2 = time.time() - start_time

        if df_batch is not None and len(df_batch) > 0:
            total_rows_2 = len(df_batch)
            logger.info(f"✅ 批量下载成功")
            logger.info(f"总耗时: {elapsed_2:.3f}秒")
            logger.info(f"总行数: {total_rows_2}")

            # 性能对比
            if total_rows_1 == total_rows_2:
                speedup = elapsed_1 / elapsed_2 if elapsed_2 > 0 else 0
                logger.info(f"\n性能提升: {speedup:.2f}倍")
                logger.info(f"时间节省: {elapsed_1 - elapsed_2:.3f}秒 ({(1 - elapsed_2/elapsed_1)*100:.1f}%)")
        else:
            logger.warning(f"⚠️ 批量下载失败或不支持")
            elapsed_2 = None
            total_rows_2 = 0

    except Exception as e:
        elapsed_2 = time.time() - start_time
        logger.error(f"❌ 批量下载失败: {e}")
        total_rows_2 = 0

    return {
        'individual_time': elapsed_1,
        'batch_time': elapsed_2,
        'speedup': elapsed_1 / elapsed_2 if elapsed_2 and elapsed_2 > 0 else None
    }


def test_large_batch():
    """测试5：大批量下载（10只股票）"""
    logger.info("="*80)
    logger.info("测试5：大批量下载测试（10只股票）")
    logger.info("="*80)

    # 10只常见股票
    stock_list = [
        "600000.SH", "600519.SH", "601318.SH", "600036.SH", "601166.SH",
        "000001.SZ", "000002.SZ", "000333.SZ", "002594.SZ", "300059.SZ"
    ]

    start_date = "20250101"
    end_date = "20250110"

    logger.info(f"股票数量: {len(stock_list)}只")

    # 测试批量下载
    try:
        start_time = time.time()

        df = ts.pro_bar(
            ts_code=','.join(stock_list),
            adj='qfq',
            start_date=start_date,
            end_date=end_date
        )

        elapsed = time.time() - start_time

        if df is not None and len(df) > 0:
            unique_stocks = df['ts_code'].nunique()
            logger.info(f"✅ 成功下载")
            logger.info(f"   请求数据: {len(stock_list)}只股票")
            logger.info(f"   实际获取: {unique_stocks}只股票")
            logger.info(f"   数据行数: {len(df)}行")
            logger.info(f"   耗时: {elapsed:.3f}秒")
            logger.info(f"   平均每只: {elapsed/len(stock_list):.3f}秒")

            if unique_stocks == len(stock_list):
                logger.info(f"✅ 批量下载完全成功！")
                return {'success': True, 'support_batch': True, 'time': elapsed}
            else:
                logger.warning(f"⚠️ 部分股票缺失")
                return {'success': True, 'support_batch': True, 'partial': True, 'time': elapsed}
        else:
            logger.warning(f"⚠️ 返回数据为空")
            return {'success': False, 'support_batch': False}

    except Exception as e:
        logger.error(f"❌ 失败: {e}")
        return {'success': False, 'support_batch': False, 'error': str(e)}


def main():
    """运行所有测试"""
    logger.info("开始Tushare批量下载测试")
    logger.info("")

    results = {}

    # 测试1：单个股票
    results['test1'] = test_single_stock()
    time.sleep(2)

    # 测试2：逗号分隔
    results['test2'] = test_multiple_stocks_comma_separated()
    time.sleep(2)

    # 测试3：列表
    results['test3'] = test_multiple_stocks_list()
    time.sleep(2)

    # 测试4：性能对比
    results['test4'] = test_batch_download_performance()
    time.sleep(2)

    # 测试5：大批量
    results['test5'] = test_large_batch()

    # 总结
    logger.info("\n" + "="*80)
    logger.info("测试总结")
    logger.info("="*80)

    logger.info(f"1. 单个股票: {'✅' if results['test1']['success'] else '❌'}")
    logger.info(f"2. 逗号分隔: {'✅' if results['test2']['success'] else '❌'}")
    logger.info(f"3. 列表传入: {'✅' if results['test3']['success'] else '❌'}")

    if results['test4']['speedup']:
        logger.info(f"4. 性能提升: {results['test4']['speedup']:.2f}倍")

    logger.info(f"5. 大批量测试: {'✅ 支持批量' if results['test5'].get('support_batch') else '❌ 不支持批量'}")

    # 结论
    logger.info("\n" + "="*80)
    logger.info("结论")
    logger.info("="*80)

    if results['test2']['success'] and results['test2'].get('stocks', 0) > 1:
        logger.info("✅ ts.pro_bar 支持批量下载（逗号分隔）")
        logger.info(f"   可以一次传入多个股票代码，用逗号分隔")
        logger.info(f"   建议每批 {min(100, 50)} 只股票（根据API限制）")
    else:
        logger.info("❌ ts.pro_bar 不支持批量下载")
        logger.info(f"   需要使用其他优化方案（多线程、多进程等）")


if __name__ == "__main__":
    main()
