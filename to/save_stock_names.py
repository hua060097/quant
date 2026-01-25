import xtquant.xtdata as xtdata
import json
import os
import time
from pathlib import Path

def connect_data_source():
    """连接数据源"""
    try:
        xtdata.connect()
        print("数据源连接成功")
        return True
    except Exception as e:
        print(f"数据源连接失败: {e}")
        return False

def get_stock_list():
    """获取所有A股股票"""
    try:
        from data_loaders import get_bj_stock_list
        bj_stock_list = get_bj_stock_list()
        # 获取沪深A股股票列表
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        stocks = stocks + bj_stock_list
        print(f"获取到 {len(stocks)} 只A股股票")
        return stocks
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

def get_stock_names(stocks):
    """获取股票名称"""
    stock_names_dict = {}
    total = len(stocks)
    
    print(f"开始获取 {total} 只股票的名称...")
    
    for i, stock_code in enumerate(stocks):
        try:
            # 获取股票详细信息
            stock_info = xtdata.get_instrument_detail(stock_code)
            stock_name = stock_info.get('InstrumentName', '未知') if stock_info else '未知'
            stock_names_dict[stock_code] = stock_name
            
            # 每处理100只股票打印一次进度
            if (i + 1) % 100 == 0 or i == total - 1:
                print(f"进度: {i + 1}/{total} ({(i + 1) / total * 100:.2f}%)")
            
            # 每获取50只股票休息一秒，避免请求过于频繁
            if (i + 1) % 50 == 0 and i < total - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"获取股票 {stock_code} 名称失败: {e}")
            stock_names_dict[stock_code] = '未知'
    
    return stock_names_dict

def save_stock_names():
    """保存股票代码和名称的对应关系"""
    # 连接数据源
    if not connect_data_source():
        return False
    
    # 获取股票列表
    stocks = get_stock_list()
    if not stocks:
        return False
    
    # 获取股票名称
    stock_names_dict = get_stock_names(stocks)
    
    # 创建保存目录
    save_dir = Path('to/local_data')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON文件
    save_path = save_dir / 'stock_names.json'
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stock_names_dict, f, ensure_ascii=False, indent=2)
        print(f"股票名称数据已保存到 {save_path}，共 {len(stock_names_dict)} 条记录")
        return True
    except Exception as e:
        print(f"保存股票名称数据失败: {e}")
        return False

if __name__ == "__main__":
    save_stock_names() 