#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载模块

提供从tushare下载A股市场数据的功能
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tushare配置
TS_TOKEN = "022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000"

# 本地数据路径
PROJECT_ROOT = Path(__file__).parent.resolve()
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"
IPO_DATE_FILE = LOCAL_DATA_DIR / "ipo_date.json"


def download_a_shares_list_date(force_update: bool = False) -> dict:
    """
    获取所有A股股票的上市日期并保存到JSON文件

    数据来源：tushare pro.stock_basic接口
    保存路径：local_data/ipo_date.json
    数据格式：{股票代码: 上市日期(YYYY-MM-DD)}

    Args:
        force_update: 是否强制更新（忽略已存在的文件）

    Returns:
        dict: 股票代码到上市日期的映射字典

    Example:
        >>> ipo_dates = download_a_shares_list_date()
        >>> print(f"获取到 {len(ipo_dates)} 只股票的上市日期")
        >>> print(ipo_dates.get('000001.SZ'))  # 查看平安银行的上市日期
    """
    # 检查文件是否已存在
    if IPO_DATE_FILE.exists() and not force_update:
        logger.info(f"上市日期文件已存在: {IPO_DATE_FILE}")
        logger.info(f"如需更新，请设置 force_update=True")
        try:
            with open(IPO_DATE_FILE, 'r', encoding='utf-8') as f:
                ipo_dates = json.load(f)
            logger.info(f"从本地文件加载了 {len(ipo_dates):,} 只股票的上市日期")
            return ipo_dates
        except Exception as e:
            logger.warning(f"读取本地文件失败: {e}，将重新下载")

    logger.info("=" * 60)
    logger.info("开始下载A股股票上市日期")
    logger.info("=" * 60)

    try:
        import tushare as ts

        # 初始化tushare pro API
        logger.info("正在连接tushare API...")
        pro = ts.pro_api(TS_TOKEN)
        logger.info("[OK] 连接成功")

        # 获取股票基本信息
        logger.info("\n正在获取A股股票基本信息...")
        logger.info("  获取对象: 沪深A股（包括主板、科创板、创业板、北交所）")

        # 获取股票列表（上市状态：L上市 D退市 P暂停上市）
        # list_status: L上市 D退市 P暂停上市
        # exchange: 上交所SSE 深交所SZE 北交所BSE
        df = pro.stock_basic(
            exchange='',
            list_status='L',  # 只获取上市股票
            fields='ts_code,symbol,name,area,industry,list_date,list_status,exchange'
        )

        if df is None or df.empty:
            logger.error("[ERROR] tushare返回空数据")
            return {}

        logger.info(f"[OK] 获取到 {len(df):,} 只股票的基本信息")

        # 过滤A股（沪市：60/68开头，深市：00/30开头）
        logger.info("\n筛选A股股票...")

        # 先检查数据格式
        logger.info(f"  样本数据（前5条）:")
        for idx, row in df.head(5).iterrows():
            logger.info(f"    {row['ts_code']} - {row['name']} - {row['exchange']}")

        # 使用正确的正则表达式：tushare代码格式为 000001.SZ 或 600000.SH
        # 00/30开头是深交所，60/68开头是上交所
        df_a_shares = df[
            (df['exchange'].isin(['SSE', 'SZE'])) &  # 上交所或深交所
            (df['ts_code'].str.match(r'^(00|30|60|68)\d{4}\.(SH|SZ)$'))  # A股代码格式
        ]

        logger.info(f"  沪市主板 (60xxxx): {((df_a_shares['ts_code'].str[:2]=='60')).sum():,} 只")
        logger.info(f"  科创板 (68xxxx): {((df_a_shares['ts_code'].str[:2]=='68')).sum():,} 只")
        logger.info(f"  深市主板 (00xxxx): {((df_a_shares['ts_code'].str[:2]=='00')).sum():,} 只")
        logger.info(f"  创业板 (30xxxx): {((df_a_shares['ts_code'].str[:2]=='30')).sum():,} 只")
        logger.info(f"  合计: {len(df_a_shares):,} 只")

        # 提取股票代码和上市日期
        ipo_dates = {}
        for _, row in df_a_shares.iterrows():
            stock_code = row['ts_code']  # 格式：000001.SZ
            list_date = row['list_date']  # 格式：YYYYMMDD

            # 转换日期格式：YYYYMMDD -> YYYY-MM-DD
            if pd.notna(list_date) and len(str(list_date)) == 8:
                try:
                    list_date_formatted = f"{str(list_date)[:4]}-{str(list_date)[4:6]}-{str(list_date)[6:8]}"
                    ipo_dates[stock_code] = list_date_formatted
                except Exception as e:
                    logger.warning(f"  日期格式转换失败: {stock_code}, list_date={list_date}, error={e}")

        logger.info(f"\n[OK] 成功提取 {len(ipo_dates):,} 只股票的上市日期")

        # 统计上市年份分布
        years = [date[:4] for date in ipo_dates.values()]
        year_counts = pd.Series(years).value_counts().sort_index()
        logger.info("\n上市年份分布:")
        for year, count in year_counts.items():
            logger.info(f"  {year}年: {count:,} 只")

        # 保存到JSON文件
        logger.info(f"\n保存到文件: {IPO_DATE_FILE}")
        IPO_DATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(IPO_DATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(ipo_dates, f, ensure_ascii=False, indent=2)

        logger.info(f"[OK] 成功保存 {len(ipo_dates):,} 只股票的上市日期")

        # 显示示例数据
        logger.info("\n示例数据（前10只）:")
        for i, (stock_code, list_date) in enumerate(list(ipo_dates.items())[:10], 1):
            stock_name = df_a_shares[df_a_shares['ts_code'] == stock_code]['name'].values[0]
            logger.info(f"  {i:2d}. {stock_code} {stock_name:8s} 上市日期: {list_date}")

        logger.info("\n" + "=" * 60)
        logger.info("下载完成！")
        logger.info("=" * 60)

        return ipo_dates

    except ImportError:
        logger.error("[ERROR] 未安装tushare库")
        logger.error("请运行: pip install tushare")
        return {}
    except Exception as e:
        logger.error(f"[ERROR] 下载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def get_stock_ipo_date(stock_code: str) -> str:
    """
    获取单只股票的上市日期

    Args:
        stock_code: 股票代码，如 '000001.SZ'

    Returns:
        str: 上市日期（YYYY-MM-DD格式），未找到返回None

    Example:
        >>> ipo_date = get_stock_ipo_date('000001.SZ')
        >>> print(f"平安银行上市日期: {ipo_date}")
    """
    # 确保数据文件存在
    if not IPO_DATE_FILE.exists():
        logger.warning("上市日期文件不存在，正在下载...")
        download_a_shares_list_date()

    try:
        with open(IPO_DATE_FILE, 'r', encoding='utf-8') as f:
            ipo_dates = json.load(f)

        return ipo_dates.get(stock_code)

    except Exception as e:
        logger.error(f"读取上市日期失败: {e}")
        return None


def load_ipo_dates() -> dict:
    """
    加载所有股票的上市日期

    Returns:
        dict: 股票代码到上市日期的映射

    Example:
        >>> ipo_dates = load_ipo_dates()
        >>> print(f"加载了 {len(ipo_dates)} 只股票的上市日期")
    """
    if not IPO_DATE_FILE.exists():
        logger.warning("上市日期文件不存在，正在下载...")
        return download_a_shares_list_date()

    try:
        with open(IPO_DATE_FILE, 'r', encoding='utf-8') as f:
            ipo_dates = json.load(f)
        logger.info(f"从本地文件加载了 {len(ipo_dates):,} 只股票的上市日期")
        return ipo_dates
    except Exception as e:
        logger.error(f"读取上市日期文件失败: {e}")
        return {}


if __name__ == "__main__":
    """
    测试代码：下载上市日期数据
    """
    print("\n" + "=" * 60)
    print("A股上市日期下载工具")
    print("=" * 60)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"保存路径: {IPO_DATE_FILE}")
    print("=" * 60 + "\n")

    # 下载数据
    ipo_dates = download_a_shares_list_date(force_update=True)

    if ipo_dates:
        print(f"\n[OK] 成功获取 {len(ipo_dates):,} 只股票的上市日期")

        # 测试查询功能
        print("\n测试查询功能:")
        test_stocks = ['000001.SZ', '600000.SH', '688981.SH']
        for stock in test_stocks:
            ipo_date = get_stock_ipo_date(stock)
            if ipo_date:
                print(f"  {stock}: {ipo_date}")
            else:
                print(f"  {stock}: 未找到")
    else:
        print("\n[ERROR] 下载失败")

    print("\n" + "=" * 60)
