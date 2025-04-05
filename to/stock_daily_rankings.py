import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from stock_sectors_mapping import load_data as load_sectors_data

# 定义常量
START_DATE = '20230101'  # 起始日期
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

def connect_data_source():
    """连接数据源"""
    try:
        xtdata.connect()
        print("数据源连接成功")
        return True
    except Exception as e:
        print(f"数据源连接失败: {e}")
        return False

def get_trading_dates(start_date, end_date=None):
    """获取交易日历"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        trading_dates = xtdata.get_trading_dates('SH', start_date, end_date)
        trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
        print(f"获取到 {len(trading_dates)} 个交易日")
        return trading_dates
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        return []

def get_stock_list():
    """获取所有A股股票"""
    try:
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        print(f"获取到 {len(stocks)} 只A股股票")
        return stocks
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

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

def save_rankings(rankings, ranking_type):
    """保存排名数据"""
    output_file = OUTPUT_FILES[ranking_type]
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(rankings, f)
        print(f"{ranking_type} 排名数据已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"保存 {ranking_type} 排名数据失败: {e}")
        return False

def calculate_ten_day_returns(stocks, current_date, prev_ten_day):
    """计算十日涨幅"""
    try:
        # 获取历史数据
        raw_data = xtdata.get_market_data(
            stock_list=stocks,
            field_list=['close'],
            period='1d',
            start_time=prev_ten_day,
            end_time=current_date
        )
        
        if raw_data is None or 'close' not in raw_data:
            print(f"获取数据失败")
            return {}
        
        # 计算十日涨幅
        stock_returns = {}
        for stock in stocks:
            if stock in raw_data['close'].index:
                prices = raw_data['close'].loc[stock]
                if len(prices) >= 2 and prev_ten_day in prices.index and current_date in prices.index:
                    curr_price = prices[current_date]
                    prev_price = prices[prev_ten_day]
                    if prev_price > 0:
                        return_rate = (curr_price - prev_price) / prev_price
                        stock_returns[stock] = return_rate
        
        return stock_returns
    except Exception as e:
        print(f"计算十日涨幅时出错: {e}")
        return {}

def calculate_daily_returns_and_amount(stocks, current_date, prev_date):
    """计算当日涨幅和成交额"""
    try:
        # 获取历史数据
        raw_data = xtdata.get_market_data(
            stock_list=stocks,
            field_list=['close', 'amount'],
            period='1d',
            start_time=prev_date,
            end_time=current_date
        )
        
        if raw_data is None or 'close' not in raw_data or 'amount' not in raw_data:
            print(f"获取数据失败")
            return {}, {}
        
        # 计算当日涨幅和成交额
        stock_returns = {}
        stock_amounts = {}
        
        for stock in stocks:
            # 处理涨幅
            if stock in raw_data['close'].index:
                prices = raw_data['close'].loc[stock]
                if len(prices) >= 2 and prev_date in prices.index and current_date in prices.index:
                    curr_price = prices[current_date]
                    prev_price = prices[prev_date]
                    if prev_price > 0:
                        return_rate = (curr_price - prev_price) / prev_price
                        stock_returns[stock] = return_rate
            
            # 处理成交额
            if stock in raw_data['amount'].index:
                amounts = raw_data['amount'].loc[stock]
                if current_date in amounts.index:
                    amount = amounts[current_date]
                    stock_amounts[stock] = amount
        
        return stock_returns, stock_amounts
    except Exception as e:
        print(f"计算当日涨幅和成交额时出错: {e}")
        return {}, {}

def enrich_stock_data(stock_code, return_rate, amount, sectors_data, stock_info=None):
    """丰富股票数据，添加名称和所属概念板块"""
    try:
        if stock_info is None:
            # 获取股票详细信息
            stock_info = xtdata.get_instrument_detail(stock_code)
        
        stock_name = stock_info.get('InstrumentName', '未知') if stock_info else '未知'
        
        # 获取股票所属概念板块
        sectors = []
        if sectors_data and "stock_to_sectors" in sectors_data:
            stock_data = sectors_data["stock_to_sectors"].get(stock_code)
            if stock_data and "sectors" in stock_data:
                sectors = stock_data["sectors"]
        
        return {
            'code': stock_code,
            'name': stock_name,
            'sectors': sectors,
            'return_rate': return_rate,
            'amount': amount if amount is not None else 0
        }
    except Exception as e:
        print(f"丰富股票 {stock_code} 数据时出错: {e}")
        return {
            'code': stock_code,
            'name': '未知',
            'sectors': [],
            'return_rate': return_rate,
            'amount': amount if amount is not None else 0
        }

def get_previous_trading_date(current_date, n_days=1):
    """获取前n个交易日
    
    参数:
        current_date: 当前日期，格式为YYYYMMDD
        n_days: 前n个交易日，默认为1，即前一个交易日
        
    返回:
        前n个交易日，格式为YYYYMMDD，如果获取失败则返回None
    """
    try:
        # 从30天前开始查询，确保能获取到足够多的交易日
        start_date = (datetime.strptime(current_date, '%Y%m%d') - timedelta(days=max(n_days * 3, 30))).strftime('%Y%m%d')
        trading_dates = xtdata.get_trading_dates('SH', start_date, current_date)
        
        if len(trading_dates) <= n_days:
            print(f"无法获取 {current_date} 之前的第{n_days}个交易日")
            return None
        
        # 转换为年月日格式并获取前n个交易日
        trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
        
        if current_date not in trading_dates:
            # 如果当前日期不是交易日，就找最近的前一个交易日
            for date in sorted(trading_dates, reverse=True):
                if date <= current_date:
                    current_date = date
                    break
        
        current_index = trading_dates.index(current_date)
        if current_index >= n_days:
            return trading_dates[current_index - n_days]
        else:
            print(f"无法获取 {current_date} 之前的第{n_days}个交易日")
            return None
    except Exception as e:
        print(f"获取 {current_date} 之前的第{n_days}个交易日时出错: {e}")
        return None

def get_rankings_for_date(date, stocks, sectors_data):
    """获取特定日期的排名数据"""
    print(f"处理日期: {date}")
    
    # 获取前一个交易日
    prev_date = get_previous_trading_date(date, 1)
    if prev_date is None:
        print(f"无法获取 {date} 之前的交易日，跳过该日期")
        return {'ten_day_rise': [], 'daily_rise': [], 'daily_fall': [], 'daily_amount': []}
    
    # 获取十天前交易日
    prev_ten_day = get_previous_trading_date(date, 10)
    if prev_ten_day is None:
        print(f"无法获取 {date} 之前的第十个交易日，跳过十日涨幅计算")
        ten_day_returns = {}
    else:
        # 计算十日涨幅
        ten_day_returns = calculate_ten_day_returns(stocks, date, prev_ten_day)
    
    # 计算当日涨幅和成交额
    daily_returns, daily_amounts = calculate_daily_returns_and_amount(stocks, date, prev_date)
    
    # 创建排名数据
    ten_day_rise_ranking = []
    daily_rise_ranking = []
    daily_fall_ranking = []
    daily_amount_ranking = []
    
    # 股票基本信息缓存
    stock_info_cache = {}
    
    # 处理十日涨幅排名
    if ten_day_returns:
        for stock_code, return_rate in sorted(ten_day_returns.items(), key=lambda x: x[1], reverse=True)[:DEFAULT_TOP_N['ten_day_rise']]:
            amount = daily_amounts.get(stock_code)
            ten_day_rise_ranking.append(enrich_stock_data(stock_code, return_rate, amount, sectors_data, stock_info_cache.get(stock_code)))
    
    # 处理当日涨幅排名
    for stock_code, return_rate in sorted(daily_returns.items(), key=lambda x: x[1], reverse=True)[:DEFAULT_TOP_N['daily_rise']]:
        amount = daily_amounts.get(stock_code)
        daily_rise_ranking.append(enrich_stock_data(stock_code, return_rate, amount, sectors_data, stock_info_cache.get(stock_code)))
    
    # 处理当日跌幅排名
    for stock_code, return_rate in sorted(daily_returns.items(), key=lambda x: x[1])[:DEFAULT_TOP_N['daily_fall']]:
        amount = daily_amounts.get(stock_code)
        daily_fall_ranking.append(enrich_stock_data(stock_code, return_rate, amount, sectors_data, stock_info_cache.get(stock_code)))
    
    # 处理当日成交额排名
    for stock_code, amount in sorted(daily_amounts.items(), key=lambda x: x[1], reverse=True)[:DEFAULT_TOP_N['daily_amount']]:
        return_rate = daily_returns.get(stock_code)
        daily_amount_ranking.append(enrich_stock_data(stock_code, return_rate, amount, sectors_data, stock_info_cache.get(stock_code)))
    
    return {
        'ten_day_rise': ten_day_rise_ranking,
        'daily_rise': daily_rise_ranking,
        'daily_fall': daily_fall_ranking,
        'daily_amount': daily_amount_ranking
    }

def update_all_rankings():
    """更新所有排名数据，从START_DATE到当前日期"""
    # 连接数据源
    if not connect_data_source():
        return
    
    # 获取交易日历
    end_date = datetime.now().strftime('%Y%m%d')
    trading_dates = get_trading_dates(START_DATE, end_date)
    if not trading_dates:
        return
    
    # 获取股票列表
    stocks = get_stock_list()
    if not stocks:
        return
    
    # 加载板块数据
    sectors_data = load_sectors_data()
    if sectors_data is None:
        print("警告: 未能加载板块数据，将不会包含板块信息")
    
    # 加载现有排名数据
    ten_day_rise_rankings = load_or_create_rankings('ten_day_rise')
    daily_rise_rankings = load_or_create_rankings('daily_rise')
    daily_fall_rankings = load_or_create_rankings('daily_fall')
    daily_amount_rankings = load_or_create_rankings('daily_amount')
    
    # 更新排名数据
    for date in trading_dates:
        # 检查是否已有数据
        if (date in ten_day_rise_rankings and 
            date in daily_rise_rankings and 
            date in daily_fall_rankings and 
            date in daily_amount_rankings):
            print(f"日期 {date} 的排名数据已存在，跳过")
            continue
        
        # 获取排名数据
        rankings = get_rankings_for_date(date, stocks, sectors_data)
        
        # 更新排名数据
        if isinstance(rankings, dict):  # 确保返回的数据是字典
            if 'ten_day_rise' in rankings:
                ten_day_rise_rankings[date] = rankings['ten_day_rise']
            if 'daily_rise' in rankings:
                daily_rise_rankings[date] = rankings['daily_rise']
            if 'daily_fall' in rankings:
                daily_fall_rankings[date] = rankings['daily_fall']
            if 'daily_amount' in rankings:
                daily_amount_rankings[date] = rankings['daily_amount']
        else:
            print(f"日期 {date} 获取排名数据失败，返回的数据格式错误")
            continue
        
        # 打印日期的排名情况
        print(f"{date} 排名情况:")
        print(f"十日涨幅前{len(rankings.get('ten_day_rise', []))}名")
        print(f"当日涨幅前{len(rankings.get('daily_rise', []))}名")
        print(f"当日跌幅前{len(rankings.get('daily_fall', []))}名")
        print(f"当日成交额前{len(rankings.get('daily_amount', []))}名")
    
    # 保存排名数据
    save_rankings(ten_day_rise_rankings, 'ten_day_rise')
    save_rankings(daily_rise_rankings, 'daily_rise')
    save_rankings(daily_fall_rankings, 'daily_fall')
    save_rankings(daily_amount_rankings, 'daily_amount')
    
    print("所有排名数据更新完成")

def update_today_rankings():
    """更新当天的排名数据"""
    # 连接数据源
    if not connect_data_source():
        return
    
    # 获取今天的日期
    today = datetime.now().strftime('%Y%m%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    # 检查今天是否为交易日
    trading_dates = xtdata.get_trading_dates('SH', yesterday, today)
    if len(trading_dates) <= 1:
        print(f"{today} 不是交易日或市场未收盘")
        return
    
    # 转换为年月日格式
    trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
    today = trading_dates[-1]  # 获取最新交易日
    
    # 获取股票列表
    stocks = get_stock_list()
    if not stocks:
        return
    
    # 加载板块数据
    sectors_data = load_sectors_data()
    if sectors_data is None:
        print("警告: 未能加载板块数据，将不会包含板块信息")
    
    # 加载现有排名数据
    ten_day_rise_rankings = load_or_create_rankings('ten_day_rise')
    daily_rise_rankings = load_or_create_rankings('daily_rise')
    daily_fall_rankings = load_or_create_rankings('daily_fall')
    daily_amount_rankings = load_or_create_rankings('daily_amount')
    
    # 获取排名数据
    rankings = get_rankings_for_date(today, stocks, sectors_data)
    
    # 更新排名数据
    ten_day_rise_rankings[today] = rankings['ten_day_rise']
    daily_rise_rankings[today] = rankings['daily_rise']
    daily_fall_rankings[today] = rankings['daily_fall']
    daily_amount_rankings[today] = rankings['daily_amount']
    
    # 保存排名数据
    save_rankings(ten_day_rise_rankings, 'ten_day_rise')
    save_rankings(daily_rise_rankings, 'daily_rise')
    save_rankings(daily_fall_rankings, 'daily_fall')
    save_rankings(daily_amount_rankings, 'daily_amount')
    
    print(f"{today} 排名数据更新完成")

def update_date_range(start_date, end_date=None):
    """更新指定日期范围的排名数据"""
    # 连接数据源
    if not connect_data_source():
        return
    
    # 检查日期格式
    try:
        datetime.strptime(start_date, '%Y%m%d')
        if end_date:
            datetime.strptime(end_date, '%Y%m%d')
        else:
            end_date = datetime.now().strftime('%Y%m%d')
    except ValueError:
        print("日期格式错误，请使用YYYYMMDD格式")
        return
    
    # 获取交易日历
    trading_dates = get_trading_dates(start_date, end_date)
    if not trading_dates:
        return
    
    # 获取股票列表
    stocks = get_stock_list()
    if not stocks:
        return
    
    # 加载板块数据
    sectors_data = load_sectors_data()
    if sectors_data is None:
        print("警告: 未能加载板块数据，将不会包含板块信息")
    
    # 加载现有排名数据
    ten_day_rise_rankings = load_or_create_rankings('ten_day_rise')
    daily_rise_rankings = load_or_create_rankings('daily_rise')
    daily_fall_rankings = load_or_create_rankings('daily_fall')
    daily_amount_rankings = load_or_create_rankings('daily_amount')
    
    # 更新排名数据
    for date in trading_dates:
        # 获取排名数据
        rankings = get_rankings_for_date(date, stocks, sectors_data)
        
        # 更新排名数据
        ten_day_rise_rankings[date] = rankings['ten_day_rise']
        daily_rise_rankings[date] = rankings['daily_rise']
        daily_fall_rankings[date] = rankings['daily_fall']
        daily_amount_rankings[date] = rankings['daily_amount']
    
    # 保存排名数据
    save_rankings(ten_day_rise_rankings, 'ten_day_rise')
    save_rankings(daily_rise_rankings, 'daily_rise')
    save_rankings(daily_fall_rankings, 'daily_fall')
    save_rankings(daily_amount_rankings, 'daily_amount')
    
    print(f"日期范围 {start_date} 到 {end_date} 的排名数据更新完成")

def get_date_rankings(date, ranking_type=None, top_n=None):
    """获取特定日期的排名数据"""
    if top_n is None:
        top_n = DEFAULT_TOP_N.get(ranking_type, 10)
    
    if ranking_type is None or ranking_type not in OUTPUT_FILES:
        print("排名类型无效，可选值: ten_day_rise, daily_rise, daily_fall, daily_amount")
        return None
    
    # 加载排名数据
    try:
        with open(OUTPUT_FILES[ranking_type], 'rb') as f:
            rankings = pickle.load(f)
    except Exception as e:
        print(f"加载排名数据失败: {e}")
        return None
    
    if date not in rankings:
        print(f"未找到日期 {date} 的排名数据")
        available_dates = sorted(rankings.keys())
        if available_dates:
            print(f"可用日期范围: {available_dates[0]} 到 {available_dates[-1]}")
        return None
    
    date_rankings = rankings[date]
    if top_n > 0 and top_n < len(date_rankings):
        date_rankings = date_rankings[:top_n]
    
    return date_rankings

def print_date_rankings(date, ranking_type=None, top_n=None):
    """打印特定日期的排名数据"""
    if ranking_type is None or ranking_type not in OUTPUT_FILES:
        print("排名类型无效，可选值: ten_day_rise, daily_rise, daily_fall, daily_amount")
        return
    
    rankings = get_date_rankings(date, ranking_type, top_n)
    if rankings is None:
        return
    
    if ranking_type == 'ten_day_rise':
        print(f"\n{date} 十日涨幅前{len(rankings)}名:")
    elif ranking_type == 'daily_rise':
        print(f"\n{date} 当日涨幅前{len(rankings)}名:")
    elif ranking_type == 'daily_fall':
        print(f"\n{date} 当日跌幅前{len(rankings)}名:")
    elif ranking_type == 'daily_amount':
        print(f"\n{date} 当日成交额前{len(rankings)}名:")
    
    for i, stock in enumerate(rankings, 1):
        sectors_str = ', '.join(stock['sectors'][:3]) if stock['sectors'] else '无'
        if len(stock['sectors']) > 3:
            sectors_str += f" 等{len(stock['sectors'])}个"
        
        if ranking_type in ['ten_day_rise', 'daily_rise', 'daily_fall']:
            print(f"{i}. {stock['code']} ({stock['name']}): {stock['return_rate']:.2%}, 成交额: {stock['amount']/100000000:.2f}亿, 板块: {sectors_str}")
        else:
            print(f"{i}. {stock['code']} ({stock['name']}): 成交额: {stock['amount']/100000000:.2f}亿, 涨跌幅: {stock['return_rate']:.2%}, 板块: {sectors_str}")

def print_all_rankings_for_date(date, top_n=10):
    """打印特定日期的所有排名数据"""
    print(f"\n===== {date} 股票排名数据 =====")
    
    print_date_rankings(date, 'ten_day_rise', top_n)
    print_date_rankings(date, 'daily_rise', top_n)
    print_date_rankings(date, 'daily_fall', top_n)
    print_date_rankings(date, 'daily_amount', top_n)

def get_ten_day_rise_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的十日涨幅排名
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['ten_day_rise']
        
    rankings = get_date_rankings(date, 'ten_day_rise', top_n)
    if rankings is None:
        print(f"未找到日期 {date} 的十日涨幅排名数据")
        return None
    
    if as_dataframe:
        df = pd.DataFrame(rankings)
        # 将sectors列表转换为字符串，便于机器学习处理
        if 'sectors' in df.columns:
            df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x) if x else '')
        return df
    else:
        return rankings

def get_daily_rise_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的日涨幅排名
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_rise']
        
    rankings = get_date_rankings(date, 'daily_rise', top_n)
    if rankings is None:
        print(f"未找到日期 {date} 的日涨幅排名数据")
        return None
    
    if as_dataframe:
        df = pd.DataFrame(rankings)
        # 将sectors列表转换为字符串，便于机器学习处理
        if 'sectors' in df.columns:
            df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x) if x else '')
        return df
    else:
        return rankings

def get_daily_fall_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的日跌幅排名
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_fall']
        
    rankings = get_date_rankings(date, 'daily_fall', top_n)
    if rankings is None:
        print(f"未找到日期 {date} 的日跌幅排名数据")
        return None
    
    if as_dataframe:
        df = pd.DataFrame(rankings)
        # 将sectors列表转换为字符串，便于机器学习处理
        if 'sectors' in df.columns:
            df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x) if x else '')
        return df
    else:
        return rankings

def get_daily_amount_ranking(date, top_n=None, as_dataframe=True):
    """获取某一天的成交额排名
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 返回前多少名，默认为None，表示返回所有排名
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回pandas DataFrame，否则返回原始排名列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N['daily_amount']
        
    rankings = get_date_rankings(date, 'daily_amount', top_n)
    if rankings is None:
        print(f"未找到日期 {date} 的成交额排名数据")
        return None
    
    if as_dataframe:
        df = pd.DataFrame(rankings)
        # 将sectors列表转换为字符串，便于机器学习处理
        if 'sectors' in df.columns:
            df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x) if x else '')
        return df
    else:
        return rankings

def get_all_rankings_for_date(date, top_n=None, as_dataframe=True):
    """获取某一天的所有排名数据
    
    参数:
        date: 日期，格式为YYYYMMDD
        top_n: 字典，指定各排名类型返回的前多少名，默认为None，使用DEFAULT_TOP_N
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        包含四种排名数据的字典，每个排名数据根据as_dataframe参数决定返回DataFrame还是列表
    """
    if top_n is None:
        top_n = DEFAULT_TOP_N
    
    result = {
        'ten_day_rise': get_ten_day_rise_ranking(date, top_n.get('ten_day_rise', DEFAULT_TOP_N['ten_day_rise']), as_dataframe),
        'daily_rise': get_daily_rise_ranking(date, top_n.get('daily_rise', DEFAULT_TOP_N['daily_rise']), as_dataframe),
        'daily_fall': get_daily_fall_ranking(date, top_n.get('daily_fall', DEFAULT_TOP_N['daily_fall']), as_dataframe),
        'daily_amount': get_daily_amount_ranking(date, top_n.get('daily_amount', DEFAULT_TOP_N['daily_amount']), as_dataframe)
    }
    
    return result

def get_date_range_rankings(start_date, end_date=None, ranking_type='ten_day_rise', top_n=None, as_dataframe=True):
    """获取日期范围内的排名数据
    
    参数:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD，默认为None，表示当前日期
        ranking_type: 排名类型，可选值: ten_day_rise, daily_rise, daily_fall, daily_amount
        top_n: 返回前多少名，默认为None，表示使用DEFAULT_TOP_N中的设置
        as_dataframe: 是否返回DataFrame格式，默认为True
        
    返回:
        如果as_dataframe为True，返回包含所有日期数据的DataFrame，
        否则返回字典，键为日期，值为该日期的排名列表
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    if ranking_type not in OUTPUT_FILES:
        print("排名类型无效，可选值: ten_day_rise, daily_rise, daily_fall, daily_amount")
        return None
    
    if top_n is None:
        top_n = DEFAULT_TOP_N[ranking_type]
    
    # 加载排名数据
    try:
        with open(OUTPUT_FILES[ranking_type], 'rb') as f:
            rankings = pickle.load(f)
    except Exception as e:
        print(f"加载排名数据失败: {e}")
        return None
    
    # 筛选日期范围内的数据
    date_range_rankings = {}
    for date in rankings:
        if start_date <= date <= end_date:
            date_rankings = rankings[date]
            if top_n > 0 and top_n < len(date_rankings):
                date_rankings = date_rankings[:top_n]
            date_range_rankings[date] = date_rankings
    
    if not date_range_rankings:
        print(f"未找到日期范围 {start_date} 到 {end_date} 的排名数据")
        return None
    
    if as_dataframe:
        # 将所有日期的数据合并到一个DataFrame
        all_data = []
        for date, date_rankings in date_range_rankings.items():
            for ranking in date_rankings:
                ranking_copy = ranking.copy()
                ranking_copy['date'] = date
                all_data.append(ranking_copy)
        
        df = pd.DataFrame(all_data)
        # 将sectors列表转换为字符串，便于机器学习处理
        if 'sectors' in df.columns:
            df['sectors_str'] = df['sectors'].apply(lambda x: ','.join(x) if x else '')
        
        # 将date列转换为datetime类型
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        return df
    else:
        return date_range_rankings

if __name__ == "__main__":
    # 通过注释/取消注释下面的行来选择要执行的功能
    
    # 1. 更新所有排名数据（从START_DATE到当前日期）
    update_all_rankings()
    
    # 2. 只更新当天的排名数据
    # update_today_rankings()
    
    # 3. 更新指定日期范围的排名数据
    # update_date_range('20230901', '20230930')
    
    # 4. 查看特定日期的排名数据
    # print_all_rankings_for_date('20250313', 10)
    
    # 5. 导出示例：获取某一天的十日涨幅排名并导出为CSV
    # df = get_ten_day_rise_ranking('20230901')
    # if df is not None:
    #     df.to_csv('十日涨幅排名_20230901.csv', index=False, encoding='utf-8-sig')
    #     print("数据已导出到CSV文件")
    # 这里添加一个pass语句避免空代码块导致的缩进错误
    pass 