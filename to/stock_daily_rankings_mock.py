import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

# 定义常量
DEFAULT_TOP_N = {
    'ten_day_rise': 60,   # 十日涨幅前60
    'daily_rise': 100,    # 当日涨幅前100
    'daily_fall': 50,     # 当日跌幅前50
    'daily_amount': 50    # 当日成交额前50
}
OUTPUT_FILES = {
    'ten_day_rise': 'ten_day_rise_rankings.pkl',
    'daily_rise': 'daily_rise_rankings.pkl',
    'daily_fall': 'daily_fall_rankings.pkl',
    'daily_amount': 'daily_amount_rankings.pkl'
}

def load_or_create_rankings(ranking_type):
    """加载或创建排名数据"""
    output_file = OUTPUT_FILES[ranking_type]
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                rankings = pickle.load(f)
            print(f"成功加载 {ranking_type} 排名数据，包含 {len(rankings)} 个交易日")
            return rankings
        except Exception as e:
            print(f"加载 {ranking_type} 排名数据失败: {e}")
    
    print(f"创建新的 {ranking_type} 排名数据")
    return {}

def get_mock_df(date_str, ranking_type):
    """生成模拟的排名数据"""
    # 创建一个空的DataFrame
    df = pd.DataFrame()
    
    # 根据排名类型生成不同的数据结构
    if ranking_type in ['ten_day_rise', 'daily_rise']:
        df['code'] = [f"00000{i}.SZ" for i in range(1, 11)]
        df['name'] = [f"股票{i}" for i in range(1, 11)]
        df['sectors'] = [['TGN人工智能', f'板块{i}', f'板块{i+2}'] for i in range(1, 11)]
        df['return_rate'] = np.random.uniform(0.01, 0.2, 10)
        df['amount'] = np.random.uniform(1e7, 1e9, 10)
    elif ranking_type == 'daily_fall':
        df['code'] = [f"60000{i}.SH" for i in range(1, 11)]
        df['name'] = [f"股票{i}" for i in range(1, 11)]
        df['sectors'] = [['TGN人工智能', f'板块{i}', f'板块{i+2}'] for i in range(1, 11)]
        df['return_rate'] = np.random.uniform(-0.1, -0.01, 10)
        df['amount'] = np.random.uniform(1e7, 1e9, 10)
    elif ranking_type == 'daily_amount':
        df['code'] = [f"30000{i}.SZ" for i in range(1, 11)]
        df['name'] = [f"股票{i}" for i in range(1, 11)]
        df['sectors'] = [['TGN人工智能', f'板块{i}', f'板块{i+2}'] for i in range(1, 11)]
        df['return_rate'] = np.random.uniform(-0.05, 0.05, 10)
        df['amount'] = np.random.uniform(1e8, 5e9, 10)
    
    # 添加sectors_str列
    df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x))
    
    return df

def get_date_rankings(date, ranking_type=None, top_n=None):
    """获取特定日期的排名数据（模拟版）"""
    if top_n is None:
        top_n = DEFAULT_TOP_N.get(ranking_type, 10)
    
    if ranking_type is None or ranking_type not in OUTPUT_FILES:
        print("排名类型无效，可选值: ten_day_rise, daily_rise, daily_fall, daily_amount")
        return None
    
    # 尝试从文件加载数据
    try:
        rankings = load_or_create_rankings(ranking_type)
        if date in rankings:
            date_rankings = rankings[date]
            if top_n > 0 and top_n < len(date_rankings):
                date_rankings = date_rankings[:top_n]
            return date_rankings
    except Exception as e:
        print(f"从文件加载排名数据失败: {e}")
    
    # 如果文件中没有数据或加载失败，生成模拟数据
    print(f"为日期 {date} 生成模拟的 {ranking_type} 排名数据")
    return get_mock_df(date, ranking_type)

def get_ten_day_rise_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的十日涨幅排名（模拟版）
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['ten_day_rise']
    
    df = get_mock_df(date, 'ten_day_rise')
    
    if as_dataframe:
        return df
    else:
        # 转换为原始列表格式
        result = []
        for _, row in df.iterrows():
            result.append({
                'code': row['code'],
                'name': row['name'],
                'sectors': row['sectors'],
                'return_rate': row['return_rate'],
                'amount': row['amount']
            })
        return result

def get_daily_rise_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的日涨幅排名（模拟版）
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_rise']
    
    df = get_mock_df(date, 'daily_rise')
    
    if as_dataframe:
        return df
    else:
        # 转换为原始列表格式
        result = []
        for _, row in df.iterrows():
            result.append({
                'code': row['code'],
                'name': row['name'],
                'sectors': row['sectors'],
                'return_rate': row['return_rate'],
                'amount': row['amount']
            })
        return result

def get_daily_fall_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的日跌幅排名（模拟版）
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_fall']
    
    df = get_mock_df(date, 'daily_fall')
    
    if as_dataframe:
        return df
    else:
        # 转换为原始列表格式
        result = []
        for _, row in df.iterrows():
            result.append({
                'code': row['code'],
                'name': row['name'],
                'sectors': row['sectors'],
                'return_rate': row['return_rate'],
                'amount': row['amount']
            })
        return result

def get_daily_amount_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的成交额排名（模拟版）
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_amount']
    
    df = get_mock_df(date, 'daily_amount')
    
    if as_dataframe:
        return df
    else:
        # 转换为原始列表格式
        result = []
        for _, row in df.iterrows():
            result.append({
                'code': row['code'],
                'name': row['name'],
                'sectors': row['sectors'],
                'return_rate': row['return_rate'],
                'amount': row['amount']
            })
        return result 