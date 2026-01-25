import xtquant.xtdata as xtdata
import json
import os
from tqdm import tqdm
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

def connect_data_source():
    """连接数据源"""
    try:
        xtdata.connect()
        print("数据源连接成功")
        return True
    except Exception as e:
        print(f"数据源连接失败: {e}")
        return False

def get_bj_stock_list():
    """获取北交所股票代码列表"""
    try:
        # 尝试从本地文件加载北交所股票列表
        project_root = Path(__file__).parent.parent
        bj_stock_list_path = project_root / "to" / "local_data" / "bjstock.json"
        if os.path.exists(bj_stock_list_path):
            with open(bj_stock_list_path, "r") as f:
                bj_stocks = json.load(f)
                print(f"从本地文件加载到 {len(bj_stocks)} 只北交所股票")
                return bj_stocks
        else:
            print("北交所股票列表文件不存在，将返回空列表")
            return []
    except Exception as e:
        print(f"获取北交所股票列表失败: {e}")
        return []

def get_stock_list():
    """获取所有A股股票"""
    try:
        # 直接使用xtdata获取沪深A股股票列表
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        
        # 获取北交所股票
        bj_stock_list = get_bj_stock_list()
        
        # 合并股票列表
        all_stocks = stocks + bj_stock_list
        print(f"获取到 {len(all_stocks)} 只A股股票")
        return all_stocks
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

def get_stock_name(stock_code):
    """获取股票名称"""
    try:
        # 获取股票详细信息
        stock_info = xtdata.get_instrument_detail(stock_code)
        stock_name = stock_info.get('InstrumentName', '未知') if stock_info else '未知'
        return stock_name
    except Exception as e:
        print(f"获取股票 {stock_code} 名称时出错: {e}")
        return '未知'

def generate_stock_names():
    """生成所有股票代码和名称的映射，并保存为JSON文件"""
    # 连接数据源
    if not connect_data_source():
        return
    
    # 获取所有股票列表
    stocks = get_stock_list()
    if not stocks:
        return
    
    # 创建存储目录
    os.makedirs('to/local_data', exist_ok=True)
    
    # 初始化股票名称字典
    stock_names = {}
    
    # 批量获取股票名称
    print("开始获取股票名称...")
    for stock_code in tqdm(stocks):
        stock_name = get_stock_name(stock_code)
        stock_names[stock_code] = stock_name
        # 添加小延迟，避免请求过于频繁
        time.sleep(0.02)
    
    # 保存为JSON文件
    output_file = 'to/local_data/stock_names.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stock_names, f, ensure_ascii=False, indent=4)
    
    print(f"股票名称数据已保存到 {output_file}，共 {len(stock_names)} 只股票")
    
    return stock_names

if __name__ == "__main__":
    generate_stock_names() 