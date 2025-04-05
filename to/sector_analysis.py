import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import glob
from pathlib import Path

def get_tng_sectors():
    """获取所有TNG概念板块名称，从JSON文件中读取
    
    返回:
        概念板块名称列表
    """
    try:
        import os
        import json
        import logging
        import glob
        from pathlib import Path
        
        # 可能的文件位置列表
        possible_paths = []
        
        # 当前工作目录
        possible_paths.append("tng_sectors_stats.json")
        
        # 当前文件所在目录
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(current_file_dir, "tng_sectors_stats.json"))
        
        # sector_analysis目录
        sector_analysis_dir = "sector_analysis"
        if os.path.exists(sector_analysis_dir):
            # 检查该目录下的tng_sectors_stats开头的文件
            json_pattern = os.path.join(sector_analysis_dir, "tng_sectors_stats*.json")
            json_files = glob.glob(json_pattern)
            if json_files:
                json_files.sort(reverse=True)  # 按名称逆序排序，假设较新的文件名较大
                possible_paths.extend(json_files)
        
        # 绝对路径检查 - 在项目根目录搜索
        project_root = Path(current_file_dir).parent  # to目录的父目录
        for pattern in ["tng_sectors_stats*.json", "**/tng_sectors_stats*.json"]:
            paths = list(project_root.glob(pattern))
            possible_paths.extend([str(p) for p in paths])
        
        # 尝试在sector_analysis目录下搜索 (相对于项目根目录)
        sector_analysis_relpath = os.path.join(project_root, "sector_analysis")
        if os.path.exists(sector_analysis_relpath):
            json_pattern = os.path.join(sector_analysis_relpath, "tng_sectors_stats*.json")
            json_files = glob.glob(json_pattern)
            possible_paths.extend(json_files)
        
        # 打印所有可能路径用于调试
        print(f"搜索TNG板块数据的可能路径: {possible_paths}")
                
        # 尝试打开文件
        json_file = None
        for path in possible_paths:
            if os.path.exists(path):
                json_file = path
                print(f"找到TNG板块数据文件: {json_file}")
                break
        
        if json_file is None:
            # 如果未找到文件，回退到原始方法
            print(f"未找到任何TNG板块数据文件，使用xtdata.get_sector_list()获取")
            import xtquant.xtdata as xtdata
            all_sectors = xtdata.get_sector_list()
            tng_sectors = [sector for sector in all_sectors if sector.startswith('TGN')]
            return tng_sectors
                
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            sector_stats = json.load(f)
            
        # 提取板块名称
        sectors = [item['sector_name'] for item in sector_stats if 'sector_name' in item]
        
        print(f"从JSON文件 {json_file} 加载了 {len(sectors)} 个概念板块")
        return sectors
    except Exception as e:
        # 出现异常时回退到原始方法
        print(f"从JSON文件读取概念板块失败: {e}，使用xtdata.get_sector_list()获取")
        import xtquant.xtdata as xtdata
        all_sectors = xtdata.get_sector_list()
        tng_sectors = [sector for sector in all_sectors if sector.startswith('TGN')]
        return tng_sectors
    
    
def calculate_sector_returns_for_all_dates(sector, start_date, end_date, trading_dates):
    """计算单个板块在所有交易日的涨跌幅"""
    try:
        # 获取板块成分股
        stocks = xtdata.get_stock_list_in_sector(sector)
        if not stocks or len(stocks) == 0:
            print(f"板块 {sector} 没有成分股")
            return None
            
        # 为了计算第一个交易日的涨跌幅，需要获取前一个交易日的数据
        # 获取前一个交易日
        prev_date = xtdata.get_trading_dates('SH', 
                          (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=15)).strftime('%Y%m%d'), 
                          start_date)
        if prev_date and len(prev_date) > 1:
            # 获取前一个交易日的日期
            prev_date = datetime.fromtimestamp(prev_date[-2]/1000).strftime('%Y%m%d')
            actual_start_date = prev_date
        else:
            print(f"无法获取 {start_date} 之前的交易日")
            actual_start_date = start_date
            
        print(f"获取 {sector} 从 {actual_start_date} 到 {end_date} 的数据")
            
        # 获取成分股的历史数据
        try:
            raw_data = xtdata.get_market_data(
                stock_list=stocks,
                field_list=['close'],
                period='1d',
                start_time=actual_start_date,
                end_time=end_date
            )
            
            if raw_data is None or 'close' not in raw_data:
                print(f"板块 {sector} 获取数据失败")
                return None
                
            # 存储每个交易日的板块涨跌幅
            sector_daily_returns = {}
            
            # 遍历所有交易日（除了第一天，因为需要前一天的数据来计算涨跌幅）
            for i, date in enumerate(trading_dates):
                # 跳过获取不到前一天数据的情况
                if i == 0 and actual_start_date == start_date:
                    continue
                    
                # 获取当天日期的索引
                date_obj = datetime.strptime(date, '%Y%m%d')
                
                # 存储所有股票当天的涨跌幅
                stock_returns = {}
                
                # 遍历所有成分股
                for stock in stocks:
                    # 检查股票是否在返回的数据中
                    if stock in raw_data['close'].index:
                        # 获取该股票的收盘价序列
                        prices = raw_data['close'].loc[stock]
                        
                        # 确保该日期有数据
                        if date in prices.index:
                            current_price = prices[date]
                            
                            # 获取前一个交易日数据
                            prev_date = trading_dates[i-1] if i > 0 else actual_start_date
                            if prev_date in prices.index:
                                prev_price = prices[prev_date]
                                
                                # 计算涨跌幅
                                if prev_price > 0:  # 避免除以0
                                    returns = (current_price - prev_price) / prev_price
                                    stock_returns[stock] = returns
                
                # 计算当天板块平均涨跌幅
                if stock_returns:
                    sector_daily_returns[date] = sum(stock_returns.values()) / len(stock_returns)
            
            return sector_daily_returns
            
        except Exception as e:
            print(f"处理板块 {sector} 数据时出错: {e}")
            return None
            
    except Exception as e:
        print(f"计算板块 {sector} 涨跌幅时出错: {e}")
        return None

def main():
    # 连接数据源
    xtdata.connect()
    
    # 获取所有TGN开头的概念板块
    tng_sectors = get_tng_sectors()
    print(f"找到 {len(tng_sectors)} 个TGN开头的概念板块")
    
    # 设置时间范围
    start_date = '20230601'
    end_date = datetime.now().strftime('%Y%m%d')
    print(f"开始日期: {start_date}, 结束日期: {end_date}")  
    
    # 获取交易日历并转换为年月日格式
    trading_dates = xtdata.get_trading_dates('SH', start_date, end_date)
    trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
    print(f"共获取到 {len(trading_dates)} 个交易日")
    
    # 存储每日板块涨跌幅排名
    daily_rankings = {}
    
    # 计算所有板块在所有交易日的涨跌幅
    for sector in tng_sectors:
        print(f"处理板块: {sector}")
        sector_returns = calculate_sector_returns_for_all_dates(sector, start_date, end_date, trading_dates)
        
        if sector_returns:
            # 将每个板块的每日涨跌幅加入到对应日期的排名中
            for date, return_rate in sector_returns.items():
                if date not in daily_rankings:
                    daily_rankings[date] = []
                daily_rankings[date].append((sector, return_rate))
    
    # 对每个交易日的板块涨跌幅进行排序
    for date in daily_rankings:
        daily_rankings[date] = sorted(daily_rankings[date], key=lambda x: x[1], reverse=True)
        
        # 打印当日排名前5的板块
        if daily_rankings[date]:
            print(f"\n{date} 涨幅前5的板块:")
            for i, (sector, return_rate) in enumerate(daily_rankings[date][:5], 1):
                print(f"{i}. {sector}: {return_rate:.2%}")
    
    # 将结果保存到文件
    output_file = 'sector_rankings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(daily_rankings, f)
    
    print(f"结果已保存到 {output_file}")
    
    # 打印一些统计信息
    print("\n统计信息:")
    print(f"总交易日数: {len(trading_dates)}")
    print(f"总板块数: {len(tng_sectors)}")
    
    # 打印最近一天的排名前5的板块
    if daily_rankings:
        latest_date = max(daily_rankings.keys())
        print(f"\n{latest_date} 涨幅前5的板块:")
        for i, (sector, return_rate) in enumerate(daily_rankings[latest_date][:5], 1):
            print(f"{i}. {sector}: {return_rate:.2%}")

def update_today_data():
    """增量更新当天的板块涨跌幅数据"""
    # 连接数据源
    xtdata.connect()
    
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
    
    print(f"更新 {today} 的板块排名数据")
    
    # 获取所有TGN开头的概念板块
    tng_sectors = get_tng_sectors()
    print(f"找到 {len(tng_sectors)} 个TGN开头的概念板块")
    
    # 获取前一个交易日以计算涨跌幅
    prev_trading_day = xtdata.get_trading_dates('SH', 
                         (datetime.strptime(today, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d'), 
                         today)
    if len(prev_trading_day) < 2:
        print("无法获取前一个交易日，无法计算涨跌幅")
        return
    
    prev_trading_day = datetime.fromtimestamp(prev_trading_day[-2]/1000).strftime('%Y%m%d')
    
    # 尝试加载现有数据
    output_file = 'sector_rankings.pkl'
    daily_rankings = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                daily_rankings = pickle.load(f)
            print(f"成功加载现有数据，包含 {len(daily_rankings)} 个交易日")
        except Exception as e:
            print(f"加载现有数据失败: {e}")
    
    # 存储当天板块涨跌幅
    today_rankings = []
    
    # 计算所有板块的当天涨跌幅
    for sector in tng_sectors:
        try:
            # 获取板块成分股
            stocks = xtdata.get_stock_list_in_sector(sector)
            if not stocks or len(stocks) == 0:
                print(f"板块 {sector} 没有成分股")
                continue
                
            # 获取成分股的历史数据
            raw_data = xtdata.get_market_data(
                stock_list=stocks,
                field_list=['close'],
                period='1d',
                start_time=prev_trading_day,
                end_time=today
            )
            
            if raw_data is None or 'close' not in raw_data:
                print(f"板块 {sector} 获取数据失败")
                continue
                
            # 计算当天的涨跌幅
            stock_returns = {}
            
            # 遍历所有成分股
            for stock in stocks:
                # 检查股票是否在返回的数据中
                if stock in raw_data['close'].index:
                    # 获取该股票的收盘价序列
                    prices = raw_data['close'].loc[stock]
                    
                    # 检查是否有今天和前一天的数据
                    if today in prices.index and prev_trading_day in prices.index:
                        current_price = prices[today]
                        prev_price = prices[prev_trading_day]
                        
                        # 计算涨跌幅
                        if prev_price > 0:  # 避免除以0
                            returns = (current_price - prev_price) / prev_price
                            stock_returns[stock] = returns
            
            # 计算当天板块平均涨跌幅
            if stock_returns:
                sector_return = sum(stock_returns.values()) / len(stock_returns)
                today_rankings.append((sector, sector_return))
                print(f"板块 {sector} 当天涨跌幅: {sector_return:.2%}")
            
        except Exception as e:
            print(f"处理板块 {sector} 时出错: {e}")
    
    # 对当天板块涨跌幅进行排序
    today_rankings = sorted(today_rankings, key=lambda x: x[1], reverse=True)
    
    # 更新数据
    daily_rankings[today] = today_rankings
    
    # 保存更新后的数据
    with open(output_file, 'wb') as f:
        pickle.dump(daily_rankings, f)
    
    print(f"已更新 {today} 的排名数据，共 {len(today_rankings)} 个板块")
    
    # 打印当天排名前10的板块
    print(f"\n{today} 涨幅前10的板块:")
    for i, (sector, return_rate) in enumerate(today_rankings[:10], 1):
        print(f"{i}. {sector}: {return_rate:.2%}")

def read_rankings(date=None, top_n=10):
    """读取保存的排名结果，并打印到控制台
    
    参数:
        date: 指定日期，格式为YYYYMMDD，如果为None则显示最新日期
        top_n: 显示前几名，默认为10
    """
    output_file = 'sector_rankings.pkl'
    
    if not os.path.exists(output_file):
        print(f"文件 {output_file} 不存在")
        return
    
    try:
        with open(output_file, 'rb') as f:
            daily_rankings = pickle.load(f)
        
        if not daily_rankings:
            print("没有排名数据")
            return
        
        # 如果未指定日期，则使用最新日期
        if date is None:
            date = max(daily_rankings.keys())
        elif date not in daily_rankings:
            print(f"{date} 没有排名数据")
            print(f"可用日期: {sorted(daily_rankings.keys())}")
            return
        
        rankings = daily_rankings[date]
        
        print(f"\n{date} 涨幅前{top_n}的板块:")
        for i, (sector, return_rate) in enumerate(rankings[:top_n], 1):
            print(f"{i}. {sector}: {return_rate:.2%}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"当日共有 {len(rankings)} 个板块")
        print(f"平均涨跌幅: {sum(r[1] for r in rankings)/len(rankings):.2%}")
        print(f"最大涨幅: {rankings[0][1]:.2%} ({rankings[0][0]})")
        if len(rankings) > 1:
            print(f"最小涨幅: {rankings[-1][1]:.2%} ({rankings[-1][0]})")
        
    except Exception as e:
        print(f"读取数据失败: {e}")

def update_date_range(start_date, end_date):
    """增量更新指定日期范围内的板块涨跌幅数据
    
    参数:
        start_date: 开始日期，格式为YYYYMMDD
        end_date: 结束日期，格式为YYYYMMDD
    """
    # 连接数据源
    xtdata.connect()
    
    # 检查日期格式
    try:
        datetime.strptime(start_date, '%Y%m%d')
        datetime.strptime(end_date, '%Y%m%d')
    except ValueError:
        print("日期格式错误，请使用YYYYMMDD格式")
        return
    
    # 获取交易日历并转换为年月日格式
    trading_dates = xtdata.get_trading_dates('SH', start_date, end_date)
    trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
    
    if not trading_dates:
        print(f"指定日期范围 {start_date} 到 {end_date} 内没有交易日")
        return
    
    print(f"将更新 {start_date} 到 {end_date} 的板块排名数据，共 {len(trading_dates)} 个交易日")
    
    # 获取所有TGN开头的概念板块
    tng_sectors = get_tng_sectors()
    print(f"找到 {len(tng_sectors)} 个TGN开头的概念板块")
    
    # 为了计算第一个交易日的涨跌幅，需要获取前一个交易日的数据
    # 获取前一个交易日
    prev_trading_day = xtdata.get_trading_dates('SH', 
                         (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=15)).strftime('%Y%m%d'), 
                         start_date)
    if len(prev_trading_day) < 2:
        print("无法获取前一个交易日，无法计算涨跌幅")
        return
    
    # 获取前一个交易日的日期
    prev_trading_day = datetime.fromtimestamp(prev_trading_day[-2]/1000).strftime('%Y%m%d')
    
    # 尝试加载现有数据
    output_file = 'sector_rankings.pkl'
    daily_rankings = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                daily_rankings = pickle.load(f)
            print(f"成功加载现有数据，包含 {len(daily_rankings)} 个交易日")
        except Exception as e:
            print(f"加载现有数据失败: {e}")
    
    # 计算所有板块在所有交易日的涨跌幅
    for sector in tng_sectors:
        print(f"处理板块: {sector}")
        
        try:
            # 获取板块成分股
            stocks = xtdata.get_stock_list_in_sector(sector)
            if not stocks or len(stocks) == 0:
                print(f"板块 {sector} 没有成分股")
                continue
                
            # 获取成分股的历史数据（包括前一个交易日的数据）
            raw_data = xtdata.get_market_data(
                stock_list=stocks,
                field_list=['close'],
                period='1d',
                start_time=prev_trading_day,
                end_time=end_date
            )
            
            if raw_data is None or 'close' not in raw_data:
                print(f"板块 {sector} 获取数据失败")
                continue
                
            # 遍历所有交易日
            for i, date in enumerate(trading_dates):
                # 存储所有股票当天的涨跌幅
                stock_returns = {}
                
                # 获取前一个交易日，如果是第一个交易日则使用之前获取的前一个交易日
                if i == 0:
                    prev_date = prev_trading_day
                else:
                    prev_date = trading_dates[i-1]
                
                # 遍历所有成分股
                for stock in stocks:
                    # 检查股票是否在返回的数据中
                    if stock in raw_data['close'].index:
                        # 获取该股票的收盘价序列
                        prices = raw_data['close'].loc[stock]
                        
                        # 检查是否有当天和前一天的数据
                        if date in prices.index and prev_date in prices.index:
                            current_price = prices[date]
                            prev_price = prices[prev_date]
                            
                            # 计算涨跌幅
                            if prev_price > 0:  # 避免除以0
                                returns = (current_price - prev_price) / prev_price
                                stock_returns[stock] = returns
                
                # 计算当天板块平均涨跌幅
                if stock_returns:
                    # 如果日期不在字典中，初始化为空列表
                    if date not in daily_rankings:
                        daily_rankings[date] = []
                    
                    # 计算当天板块平均涨跌幅
                    sector_return = sum(stock_returns.values()) / len(stock_returns)
                    
                    # 添加或更新板块数据
                    # 检查该板块是否已经在当天的排名中
                    existing_sector = next((item for item in daily_rankings[date] if item[0] == sector), None)
                    if existing_sector:
                        # 板块已存在，更新涨跌幅数据
                        daily_rankings[date].remove(existing_sector)
                    
                    # 添加板块涨跌幅数据
                    daily_rankings[date].append((sector, sector_return))
                    print(f"日期 {date}, 板块 {sector} 涨跌幅: {sector_return:.2%}")
            
        except Exception as e:
            print(f"处理板块 {sector} 时出错: {e}")
    
    # 对每个更新的交易日的板块涨跌幅进行排序
    for date in trading_dates:
        if date in daily_rankings:
            daily_rankings[date] = sorted(daily_rankings[date], key=lambda x: x[1], reverse=True)
            
            # 打印当日排名前5的板块
            print(f"\n{date} 涨幅前5的板块:")
            for i, (sector, return_rate) in enumerate(daily_rankings[date][:5], 1):
                print(f"{i}. {sector}: {return_rate:.2%}")
    
    # 保存更新后的数据
    with open(output_file, 'wb') as f:
        pickle.dump(daily_rankings, f)
    
    print(f"已更新 {len(trading_dates)} 个交易日的排名数据")
    print(f"数据已保存到 {output_file}")

if __name__ == "__main__":
    # 通过注释/取消注释下面的行来选择要执行的功能
    
    # 1. 初始化并计算所有历史数据
    #main()
    
    # 2. 只更新当天数据
    #update_today_data()
    
    # 3. 读取最新日期的排名（默认显示前10名）
    #read_rankings()
    
    # 4. 读取指定日期的排名
    #read_rankings('20250317', 10)  # 参数: (日期, 显示数量)
    
    # 5. 更新指定日期范围内的数据
    #update_date_range('20250313', '20250317')  # 参数: (开始日期, 结束日期)
    pass 