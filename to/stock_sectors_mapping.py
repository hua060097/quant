import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import sector_analysis as sa
from functools import lru_cache

def get_tng_sectors():
    """获取所有TGN开头的概念板块"""
    # all_sectors = xtdata.get_sector_list()
    # tng_sectors = [sector for sector in all_sectors if sector.startswith('TGN')]
    tng_sectors = sa.get_tng_sectors()
    return tng_sectors

def get_stock_info():
    """获取所有股票的基本信息"""
    # 获取沪深A股所有股票
    stocks = xtdata.get_stock_list_in_sector('沪深A股')
    
    # 创建股票代码到名称的映射
    stock_code_to_name = {}
    for stock_code in stocks:
        try:
            # 获取股票详细信息
            stock_info = xtdata.get_instrument_detail(stock_code)
            if stock_info and 'InstrumentName' in stock_info:
                stock_code_to_name[stock_code] = stock_info['InstrumentName']
        except Exception as e:
            print(f"获取股票 {stock_code} 信息时出错: {e}")
    
    return stock_code_to_name

def create_stock_sector_mapping():
    """创建股票和TGN概念板块的映射关系"""
    # 连接数据源
    xtdata.connect()
    
    # 获取所有TGN开头的概念板块
    tng_sectors = get_tng_sectors()
    print(f"找到 {len(tng_sectors)} 个TGN开头的概念板块")
    
    # 获取所有股票的名称信息 - 这可能需要一些时间
    print("获取股票基本信息中，这可能需要一些时间...")
    stock_code_to_name = get_stock_info()
    print(f"找到 {len(stock_code_to_name)} 只股票")
    
    # 创建股票到板块的映射
    stock_to_sectors = {}
    # 创建板块到股票的映射
    sector_to_stocks = {}
    
    # 遍历所有TGN概念板块
    for sector in tng_sectors:
        try:
            # 获取板块成分股
            stocks = xtdata.get_stock_list_in_sector(sector)
            if not stocks or len(stocks) == 0:
                print(f"板块 {sector} 没有成分股")
                continue
                
            print(f"板块 {sector} 包含 {len(stocks)} 只股票")
            sector_to_stocks[sector] = stocks
            
            # 更新每个股票所属的板块
            for stock in stocks:
                if stock not in stock_to_sectors:
                    stock_to_sectors[stock] = []
                stock_to_sectors[stock].append(sector)
                
                # 尝试获取股票名称（如果之前没有获取到）
                if stock not in stock_code_to_name:
                    try:
                        stock_info = xtdata.get_instrument_detail(stock)
                        if stock_info and 'InstrumentName' in stock_info:
                            stock_code_to_name[stock] = stock_info['InstrumentName']
                    except:
                        pass
                
        except Exception as e:
            print(f"处理板块 {sector} 时出错: {e}")
    
    # 统计信息
    stocks_with_tgn_sectors = len(stock_to_sectors)
    max_sectors_per_stock = max(len(sectors) for sectors in stock_to_sectors.values()) if stock_to_sectors else 0
    
    print(f"\n统计信息:")
    print(f"总股票数: {len(stock_code_to_name)}")
    print(f"属于TGN概念板块的股票数: {stocks_with_tgn_sectors}")
    print(f"单只股票最多属于 {max_sectors_per_stock} 个TGN概念板块")
    
    # 创建更完整的映射，包含股票名称
    enhanced_stock_to_sectors = {}
    for stock_code, sectors in stock_to_sectors.items():
        stock_name = stock_code_to_name.get(stock_code, "未知")
        enhanced_stock_to_sectors[stock_code] = {
            "name": stock_name,
            "sectors": sectors
        }
    
    # 创建股票名称到代码的映射
    stock_name_to_code = {name: code for code, name in stock_code_to_name.items()}
    
    # 合并所有映射为一个结构
    stock_sectors_data = {
        "stock_to_sectors": enhanced_stock_to_sectors,
        "sector_to_stocks": sector_to_stocks,
        "stock_code_to_name": stock_code_to_name,
        "stock_name_to_code": stock_name_to_code,
        "update_time": datetime.now().strftime('%Y%m%d %H:%M:%S')
    }
    
    return stock_sectors_data

def save_data(data, filename="stock_sectors_mapping.pkl"):
    """保存数据到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已保存到 {filename}")

@lru_cache(maxsize=1)
def load_data(filename=None):
    """从文件加载数据"""
    if filename is None:
        # 使用工程根目录（to目录的父目录）的路径
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_sectors_mapping.pkl")
    
    if not os.path.exists(filename):
        print(f"文件 {filename} 不存在，路径: {filename}")
        return None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载数据，更新时间: {data.get('update_time', '未知')}")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def get_stock_sectors(stock_code=None, stock_name=None, data=None):
    """查询股票所属的TGN概念板块
    
    参数:
        stock_code: 股票代码
        stock_name: 股票名称
        data: 预加载的数据，如果为None则尝试从文件加载
        
    返回:
        股票所属的TGN概念板块列表，如果未找到则返回空列表
    """
    if data is None:
        data = load_data()
        if data is None:
            return []
    
    # 如果提供了股票名称，尝试转换为代码
    if stock_name and not stock_code:
        stock_code = data.get("stock_name_to_code", {}).get(stock_name)
        if not stock_code:
            print(f"未找到股票名称 {stock_name} 对应的代码")
            return []
    
    # 查询股票所属的板块
    if stock_code:
        stock_info = data.get("stock_to_sectors", {}).get(stock_code)
        if stock_info:
            print(f"股票 {stock_code} ({stock_info['name']}) 所属的TGN概念板块:")
            sectors = stock_info["sectors"]
            for i, sector in enumerate(sectors, 1):
                print(f"{i}. {sector}")
            return sectors
        else:
            print(f"股票 {stock_code} 不属于任何TGN概念板块")
            return []
    
    print("请提供股票代码或股票名称")
    return []

def get_sector_stocks(sector_name, data=None):
    """查询TGN概念板块包含的股票
    
    参数:
        sector_name: 板块名称
        data: 预加载的数据，如果为None则尝试从文件加载
        
    返回:
        板块包含的股票列表，如果未找到则返回空列表
    """
    if data is None:
        data = load_data()
        if data is None:
            return []
    
    # 查询板块包含的股票
    stocks = data.get("sector_to_stocks", {}).get(sector_name, [])
    if stocks:
        print(f"板块 {sector_name} 包含 {len(stocks)} 只股票:")
        stock_code_to_name = data.get("stock_code_to_name", {})
        for i, stock_code in enumerate(stocks[:10], 1):  # 只显示前10只
            stock_name = stock_code_to_name.get(stock_code, "未知")
            print(f"{i}. {stock_code} ({stock_name})")
        
        if len(stocks) > 10:
            print(f"... 以及其他 {len(stocks) - 10} 只股票")
            
        return stocks
    else:
        print(f"未找到板块 {sector_name} 或该板块没有成分股")
        return []

def update_mapping():
    """更新股票和TGN概念板块的映射关系"""
    print("开始更新股票和TGN概念板块的映射关系...")
    data = create_stock_sector_mapping()
    save_data(data)
    print("更新完成")
    return data

if __name__ == "__main__":
    # 通过注释/取消注释下面的行来选择要执行的功能
    
    # 1. 更新股票和TGN概念板块的映射关系
    # update_mapping()
    
    # 2. 查询股票所属的TGN概念板块 (可通过代码或名称)
    data = load_data()
    # get_stock_sectors(stock_code="000001.SZ", data=data)  # 通过股票代码查询
    # get_stock_sectors(stock_name="平安银行", data=data)    # 通过股票名称查询
    
    # 3. 查询TGN概念板块包含的股票
    get_sector_stocks("TGN人工智能", data=data) 