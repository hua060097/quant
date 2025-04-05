"""
测试脚本，用于验证数据连接和基本功能
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from xtquant import xtdata
    print("成功导入xtquant模块")
except ImportError as e:
    print(f"错误: 无法导入xtquant模块: {e}")
    sys.exit(1)

def test_connection():
    """测试连接到数据源"""
    try:
        print("尝试连接到数据源...")
        xtdata.connect()
        print("成功连接到数据源")
        return True
    except Exception as e:
        print(f"连接数据源失败: {e}")
        return False

def test_trading_dates():
    """测试获取交易日期"""
    try:
        print("\n测试获取交易日期...")
        today = datetime.now().strftime('%Y%m%d')
        one_month_ago = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
        
        trading_dates = xtdata.get_trading_dates('SH', one_month_ago, today)
        if trading_dates:
            print(f"获取到 {len(trading_dates)} 个交易日期")
            print(f"第一个交易日: {trading_dates[0]}")
            print(f"最后一个交易日: {trading_dates[-1]}")
            return True
        else:
            print("警告: 未获取到交易日期")
            return False
    except Exception as e:
        print(f"获取交易日期失败: {e}")
        return False

def test_market_data():
    """测试获取市场数据"""
    try:
        print("\n测试获取市场数据...")
        today = '20250321'
        one_week_ago = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
        
        # 获取上证指数数据
        data = xtdata.get_market_data(
            field_list=['close', 'open', 'high', 'low', 'volume'],
            stock_list=['000001.SZ'],
            period='1d',
            start_time=one_week_ago,
            end_time=today
        )
        print(data)
        
        if data and isinstance(data, dict):
            print("成功获取市场数据")
            if 'close' in data:
                print(f"收盘价数据类型: {type(data['close'])}")
                return True
            else:
                print("警告: 返回的数据中没有'close'字段")
                print(f"返回的数据包含字段: {list(data.keys())}")
                return False
        else:
            print("警告: 未获取到市场数据或返回格式不正确")
            print(f"返回数据类型: {type(data)}")
            return False
    except Exception as e:
        print(f"获取市场数据失败: {e}")
        return False

def test_stock_list():
    """测试获取股票列表"""
    try:
        print("\n测试获取股票列表...")
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        if stocks:
            print(f"获取到 {len(stocks)} 只股票")
            print(f"前5只股票: {stocks[:5]}")
            return True
        else:
            print("警告: 未获取到股票列表")
            return False
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return False

def test_stock_details():
    """测试获取股票详情"""
    try:
        print("\n测试获取股票详情...")
        # 测试获取平安银行的详情
        stock_code = "000001.SZ"
        details = xtdata.get_instrument_detail(stock_code)
        if details:
            print(f"成功获取股票 {stock_code} 的详情:")
            print(f"股票名称: {details.get('InstrumentName', '未知')}")
            print(f"交易所: {details.get('ExchangeID', '未知')}")
            return True
        else:
            print(f"警告: 未获取到股票 {stock_code} 的详情")
            return False
    except Exception as e:
        print(f"获取股票详情失败: {e}")
        return False

def test_sector_list():
    """测试获取板块列表"""
    try:
        print("\n测试获取板块列表...")
        # 不传参数获取所有板块
        sectors = xtdata.get_sector_list()
        if sectors:
            print(f"获取到 {len(sectors)} 个板块")
            # 筛选TGN开头的板块
            tgn_sectors = [s for s in sectors if s.startswith('TGN')]
            print(f"其中有 {len(tgn_sectors)} 个TGN板块")
            if tgn_sectors:
                print(f"前5个TGN板块: {tgn_sectors[:5]}")
            return True
        else:
            print("警告: 未获取到板块列表")
            return False
    except Exception as e:
        print(f"获取板块列表失败: {e}")
        return False

def test_stocks_in_sector():
    """测试获取板块内股票"""
    try:
        print("\n测试获取板块内股票...")
        # 先获取所有板块
        all_sectors = xtdata.get_sector_list()
        # 筛选出一个存在的板块
        test_sectors = [s for s in all_sectors if s.startswith('TGN')]
        
        if not test_sectors:
            print("找不到以TGN开头的板块，尝试使用默认板块")
            test_sector = "沪深A股"  # 使用一个肯定存在的板块
        else:
            test_sector = test_sectors[0]
        
        print(f"测试获取 {test_sector} 板块内的股票")
        stocks = xtdata.get_stock_list_in_sector(test_sector)
        
        if stocks:
            print(f"获取到 {len(stocks)} 只 {test_sector} 板块内的股票")
            print(f"前5只股票: {stocks[:5]}")
            return True
        else:
            print(f"警告: 未获取到 {test_sector} 板块内的股票")
            return False
    except Exception as e:
        print(f"获取板块内股票失败: {e}")
        return False
  
def run_all_tests():
    """运行所有测试"""
    tests = [
        test_connection,
        test_trading_dates,
        test_market_data,
        test_stock_list,
        test_stock_details,
        test_sector_list,
        test_stocks_in_sector
    ]
    
    results = []
    for test in tests:
        test_name = test.__name__
        print(f"\n----------- 运行测试: {test_name} -----------")
        result = test()
        results.append((test_name, result))
    
    print("\n----------- 测试结果汇总 -----------")
    all_success = True
    for test_name, result in results:
        status = "成功" if result else "失败"
        print(f"{test_name}: {status}")
        if not result:
            all_success = False
    
    if all_success:
        print("\n所有测试都成功通过!")
        return 0
    else:
        print("\n有测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    print("开始测试数据连接和基本功能...")
    exit_code = run_all_tests()
    print("\n测试完成!")
    sys.exit(exit_code) 