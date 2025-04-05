# advanced_quant_model.py
import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import datetime
import time
import warnings
import joblib
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import mmap
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from stock_daily_rankings import get_ten_day_rise_ranking, get_daily_rise_ranking, get_daily_fall_ranking, get_daily_amount_ranking
from stock_sectors_mapping import load_data as load_sectors_data, get_stock_sectors, get_sector_stocks
from sector_analysis import read_rankings as read_sector_rankings

warnings.filterwarnings('ignore')

# 基础配置
START_DATE = '20230601'  # 训练起始日期
TRAINING_DAYS = 30       # 训练使用前N个交易日
MAX_THREADS = min(32, cpu_count())  # 最大线程数
MAX_POSITION = 0.8       # 最大仓位比例
MIN_POSITION = 0.2       # 最小仓位比例
STOP_LOSS = 0.05         # 止损比例
DYNAMIC_STOP_PROFIT = True  # 是否使用动态止盈
STOP_PROFIT_MIN = 0.05   # 最小止盈比例
STOP_PROFIT_MAX = 0.15   # 最大止盈比例
MODEL_DIR = 'models'     # 模型保存目录
RESULT_DIR = 'results'   # 结果保存目录

# 确保目录存在
for dir_path in [MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class AdvancedQuantModel:
    """高级量化回测模型"""
    
    def __init__(self):
        """初始化模型"""
        self.connect_data_source()
        self.sectors_data = None
        self.sector_performance = None
        self.market_data = None
        self.stock_rankings = None
        self.trading_dates = None
        self.models = {}
        self.sector_status = {}
        self.portfolio = {}
        self.position = MIN_POSITION
        self.performance = {
            'daily_returns': [],
            'positions': [],
            'cash': 1000000,  # 初始资金
            'portfolio_value': 1000000,
            'transactions': []
        }
        
    def connect_data_source(self):
        """连接数据源"""
        try:
            xtdata.connect()
            print("数据源连接成功")
        except Exception as e:
            print(f"数据源连接失败: {e}")
            raise ConnectionError("无法连接到数据源")
    
    def load_all_data(self):
        """加载所有必要的数据"""
        print("开始加载数据...")
        
        # 加载板块映射数据
        self.load_sectors_data()
        
        # 获取交易日历
        self.load_trading_dates()
        
        # 加载市场数据
        self.load_market_data()
        
        # 加载板块表现数据
        self.load_sector_performance()
        
        # 加载股票排名数据
        self.load_stock_rankings()
        
        print("数据加载完成")
    
    def load_sectors_data(self):
        """加载板块映射数据"""
        print("加载板块映射数据...")
        self.sectors_data = load_sectors_data()
        if self.sectors_data is None:
            raise ValueError("无法加载板块映射数据，请先运行 stock_sectors_mapping.py 中的 update_mapping() 函数")
    
    def load_trading_dates(self):
        """获取交易日历"""
        print("获取交易日历...")
        end_date = datetime.now().strftime('%Y%m%d')
        trading_dates = xtdata.get_trading_dates('SH', START_DATE, end_date)
        self.trading_dates = [datetime.fromtimestamp(ts/1000).strftime('%Y%m%d') for ts in trading_dates]
        print(f"获取到 {len(self.trading_dates)} 个交易日")
    
    def load_market_data(self):
        """加载市场数据，包括指数和技术指标"""
        print("加载市场数据...")
        # 获取上证指数、深证成指、创业板指数据
        indices = ['000001.SH', '399001.SZ', '399006.SZ']
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        # 日线数据
        self.market_data = {}
        market_daily = xtdata.get_market_data(
            field_list=fields,
            stock_list=indices,
            period='1d',
            start_time=START_DATE,
            end_time=datetime.now().strftime('%Y%m%d')
        )
        
        if market_daily is None or len(market_daily) == 0:
            raise ValueError("获取市场日线数据失败")
        
        # 5分钟数据 (最近30个交易日)
        recent_date = self.trading_dates[-min(30, len(self.trading_dates))]
        market_5min = xtdata.get_market_data(
            field_list=fields,
            stock_list=indices,
            period='5m',
            start_time=recent_date,
            end_time=datetime.now().strftime('%Y%m%d')
        )
        
        # 处理数据，计算技术指标
        self.market_data['daily'] = self.process_market_data(market_daily)
        
        if market_5min is not None and len(market_5min) > 0:
            self.market_data['5min'] = self.process_market_data(market_5min)
        else:
            print("警告: 无法获取5分钟市场数据")
            self.market_data['5min'] = {}
    
    def process_market_data(self, data):
        """处理市场数据，计算技术指标"""
        result = {}
        for field in data:
            result[field] = data[field]
        
        # 添加计算的技术指标
        for index in data['close'].index.unique():
            # 获取该指数的收盘价
            if 'close' in data and index in data['close'].index:
                close = data['close'].loc[index]
                
                # 计算MA5, MA10, MA20
                result[f'{index}_MA5'] = close.rolling(5).mean()
                result[f'{index}_MA10'] = close.rolling(10).mean()
                result[f'{index}_MA20'] = close.rolling(20).mean()
                
                # 计算MACD
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                hist = macd - signal
                
                result[f'{index}_MACD'] = macd
                result[f'{index}_MACD_signal'] = signal
                result[f'{index}_MACD_hist'] = hist
                
                # 计算RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                result[f'{index}_RSI'] = rsi
                
                # 计算波动率
                if 'high' in data and 'low' in data:
                    high = data['high'].loc[index]
                    low = data['low'].loc[index]
                    result[f'{index}_volatility'] = ((high - low) / close).rolling(10).mean()
        
        return result
    
    def load_sector_performance(self):
        """加载板块表现数据"""
        print("加载板块表现数据...")
        
        # 尝试加载sector_rankings.pkl文件
        try:
            with open('sector_rankings.pkl', 'rb') as f:
                self.sector_performance = pickle.load(f)
            print(f"成功加载板块表现数据，包含 {len(self.sector_performance)} 个交易日")
        except Exception as e:
            print(f"加载板块表现数据失败: {e}")
            print("请先运行 sector_analysis.py 中的 main() 函数生成板块表现数据")
    
    def load_stock_rankings(self):
        """加载股票排名数据"""
        print("加载股票排名数据...")
        
        # 检查排名文件是否存在
        required_files = [
            'ten_day_rise_rankings.pkl',
            'daily_rise_rankings.pkl',
            'daily_fall_rankings.pkl',
            'daily_amount_rankings.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"以下排名文件不存在: {missing_files}")
            print("请先运行 stock_daily_rankings.py 中的 update_all_rankings() 函数生成排名数据")
            return
        
        # 初始化排名数据结构
        self.stock_rankings = {
            'ten_day_rise': {},
            'daily_rise': {},
            'daily_fall': {},
            'daily_amount': {}
        }
        
        # 加载排名数据
        for ranking_type in self.stock_rankings:
            try:
                with open(f"{ranking_type}_rankings.pkl", 'rb') as f:
                    self.stock_rankings[ranking_type] = pickle.load(f)
                print(f"成功加载 {ranking_type} 排名数据，包含 {len(self.stock_rankings[ranking_type])} 个交易日")
            except Exception as e:
                print(f"加载 {ranking_type} 排名数据失败: {e}")
    
    def prepare_features_for_date(self, date, lookback=30):
        """为指定日期准备特征
        
        参数:
            date: 日期，格式为YYYYMMDD
            lookback: 回溯天数，默认为30
            
        返回:
            板块特征字典，键为板块名称，值为特征DataFrame
        """
        # 获取日期在trading_dates中的索引
        if date not in self.trading_dates:
            print(f"日期 {date} 不是交易日")
            return {}
            
        date_index = self.trading_dates.index(date)
        
        # 确保有足够的历史数据用于特征提取
        if date_index < lookback:
            print(f"日期 {date} 前没有足够的历史数据用于特征提取")
            return {}
        
        # 获取历史日期列表
        history_dates = self.trading_dates[date_index-lookback:date_index]
        
        # 获取所有TGN概念板块
        tng_sectors = [sector for sector in self.sectors_data.get("sector_to_stocks", {}).keys() 
                      if sector.startswith('TGN')]
        
        # 初始化板块特征字典
        sector_features = {}
        
        # 为每个板块准备特征
        for sector in tng_sectors:
            # 获取板块成分股
            stocks = self.sectors_data.get("sector_to_stocks", {}).get(sector, [])
            if not stocks:
                continue
                
            # 收集板块历史表现
            sector_history = []
            for hist_date in history_dates:
                if hist_date in self.sector_performance:
                    # 找到该板块在该日期的排名
                    sector_rank = next((rank for rank, (sector_name, _) in enumerate(self.sector_performance[hist_date]) 
                                     if sector_name == sector), -1)
                    
                    # 找到该板块在该日期的涨跌幅
                    sector_return = next((return_rate for sector_name, return_rate in self.sector_performance[hist_date]) 
                                       if sector_name == sector)
                    
                    sector_history.append({
                        'date': hist_date,
                        'sector_rank': sector_rank,
                        'sector_return': sector_return if sector_return else 0,
                    })
            
            if not sector_history:
                continue
                
            # 转换为DataFrame
            df_sector = pd.DataFrame(sector_history)
            df_sector['date'] = pd.to_datetime(df_sector['date'], format='%Y%m%d')
            df_sector.set_index('date', inplace=True)
            
            # 添加市场数据指标
            for index in ['000001.SH', '399001.SZ', '399006.SZ']:
                # 添加收盘价
                if 'close' in self.market_data['daily'] and index in self.market_data['daily']['close'].index:
                    close_series = self.market_data['daily']['close'].loc[index]
                    for date in df_sector.index:
                        date_str = date.strftime('%Y%m%d')
                        if date_str in close_series.index:
                            df_sector.loc[date, f'{index}_close'] = close_series[date_str]
                
                # 添加MA5, MA10, MA20
                for ma in [5, 10, 20]:
                    ma_key = f'{index}_MA{ma}'
                    if ma_key in self.market_data['daily']:
                        ma_series = self.market_data['daily'][ma_key]
                        for date in df_sector.index:
                            date_str = date.strftime('%Y%m%d')
                            if date_str in ma_series.index:
                                df_sector.loc[date, ma_key] = ma_series[date_str]
                
                # 添加MACD
                for macd_type in ['MACD', 'MACD_signal', 'MACD_hist']:
                    macd_key = f'{index}_{macd_type}'
                    if macd_key in self.market_data['daily']:
                        macd_series = self.market_data['daily'][macd_key]
                        for date in df_sector.index:
                            date_str = date.strftime('%Y%m%d')
                            if date_str in macd_series.index:
                                df_sector.loc[date, macd_key] = macd_series[date_str]
                
                # 添加RSI
                rsi_key = f'{index}_RSI'
                if rsi_key in self.market_data['daily']:
                    rsi_series = self.market_data['daily'][rsi_key]
                    for date in df_sector.index:
                        date_str = date.strftime('%Y%m%d')
                        if date_str in rsi_series.index:
                            df_sector.loc[date, rsi_key] = rsi_series[date_str]
                
                # 添加波动率
                vol_key = f'{index}_volatility'
                if vol_key in self.market_data['daily']:
                    vol_series = self.market_data['daily'][vol_key]
                    for date in df_sector.index:
                        date_str = date.strftime('%Y%m%d')
                        if date_str in vol_series.index:
                            df_sector.loc[date, vol_key] = vol_series[date_str]
            
            # 计算板块自身的技术指标
            # MA5, MA10
            df_sector['sector_MA5'] = df_sector['sector_return'].rolling(5).mean()
            df_sector['sector_MA10'] = df_sector['sector_return'].rolling(10).mean()
            
            # 动量指标 (5日, 10日)
            df_sector['sector_momentum_5d'] = df_sector['sector_return'].pct_change(5)
            df_sector['sector_momentum_10d'] = df_sector['sector_return'].pct_change(10)
            
            # 波动率指标
            df_sector['sector_volatility'] = df_sector['sector_return'].rolling(10).std()
            
            # 计算板块内热门股票的情况
            for ranking_date in history_dates:
                # 十日涨幅榜
                if ranking_date in self.stock_rankings['ten_day_rise']:
                    # 计算板块内有多少股票在十日涨幅榜前60
                    top_stocks = [stock['code'] for stock in self.stock_rankings['ten_day_rise'][ranking_date]]
                    sector_top_count = sum(1 for stock in stocks if stock in top_stocks)
                    
                    # 计算板块在涨幅榜中的占比
                    if top_stocks:
                        dt = pd.to_datetime(ranking_date, format='%Y%m%d')
                        df_sector.loc[dt, 'sector_in_ten_day_rise_count'] = sector_top_count
                        df_sector.loc[dt, 'sector_in_ten_day_rise_ratio'] = sector_top_count / len(top_stocks)
                        
                # 日涨幅榜
                if ranking_date in self.stock_rankings['daily_rise']:
                    top_stocks = [stock['code'] for stock in self.stock_rankings['daily_rise'][ranking_date]]
                    sector_top_count = sum(1 for stock in stocks if stock in top_stocks)
                    
                    if top_stocks:
                        dt = pd.to_datetime(ranking_date, format='%Y%m%d')
                        df_sector.loc[dt, 'sector_in_daily_rise_count'] = sector_top_count
                        df_sector.loc[dt, 'sector_in_daily_rise_ratio'] = sector_top_count / len(top_stocks)
                        
                # 日跌幅榜
                if ranking_date in self.stock_rankings['daily_fall']:
                    top_stocks = [stock['code'] for stock in self.stock_rankings['daily_fall'][ranking_date]]
                    sector_top_count = sum(1 for stock in stocks if stock in top_stocks)
                    
                    if top_stocks:
                        dt = pd.to_datetime(ranking_date, format='%Y%m%d')
                        df_sector.loc[dt, 'sector_in_daily_fall_count'] = sector_top_count
                        df_sector.loc[dt, 'sector_in_daily_fall_ratio'] = sector_top_count / len(top_stocks)
                        
                # 日成交额榜
                if ranking_date in self.stock_rankings['daily_amount']:
                    top_stocks = [stock['code'] for stock in self.stock_rankings['daily_amount'][ranking_date]]
                    sector_top_count = sum(1 for stock in stocks if stock in top_stocks)
                    
                    if top_stocks:
                        dt = pd.to_datetime(ranking_date, format='%Y%m%d')
                        df_sector.loc[dt, 'sector_in_daily_amount_count'] = sector_top_count
                        df_sector.loc[dt, 'sector_in_daily_amount_ratio'] = sector_top_count / len(top_stocks)
            
            # 缺失值处理
            df_sector.fillna(0, inplace=True)
            
            # 将处理后的特征存入字典
            sector_features[sector] = df_sector
        
        return sector_features
    
    def prepare_label_for_sector(self, sector, date, forward_days=5):
        """为指定板块和日期准备标签（未来N天的表现）
        
        参数:
            sector: 板块名称
            date: 日期，格式为YYYYMMDD
            forward_days: 未来天数，默认为5
            
        返回:
            标签，1表示上涨，0表示不变或下跌
        """
        # 获取日期在trading_dates中的索引
        if date not in self.trading_dates:
            print(f"日期 {date} 不是交易日")
            return None
            
        date_index = self.trading_dates.index(date)
        
        # 确保有足够的未来数据用于标签生成
        if date_index + forward_days >= len(self.trading_dates):
            print(f"日期 {date} 后没有足够的未来数据用于标签生成")
            return None
        
        # 获取未来日期
        future_date = self.trading_dates[date_index + forward_days]
        
        # 检查板块在当前日期和未来日期的表现
        current_performance = None
        future_performance = None
        
        if date in self.sector_performance:
            for sector_name, return_rate in self.sector_performance[date]:
                if sector_name == sector:
                    current_performance = return_rate
                    break
        
        if future_date in self.sector_performance:
            for sector_name, return_rate in self.sector_performance[future_date]:
                if sector_name == sector:
                    future_performance = return_rate
                    break
        
        if current_performance is None or future_performance is None:
            return None
        
        # 计算涨跌幅
        growth_rate = (future_performance - current_performance) / (abs(current_performance) + 1e-10)
        
        # 根据涨跌幅生成标签，1表示上涨，0表示不变或下跌
        return 1 if growth_rate > 0.03 else 0  # 3%的阈值
    
    def train_sector_models(self, date, lookback=30, forward_days=5, test_size=0.2):
        """训练板块模型
        
        参数:
            date: 训练截止日期，格式为YYYYMMDD
            lookback: 特征回溯天数，默认为30
            forward_days: 标签未来天数，默认为5
            test_size: 测试集比例，默认为0.2
        """
        print(f"开始训练板块模型，截止日期: {date}")
        
        # 准备特征
        sector_features = self.prepare_features_for_date(date, lookback)
        
        # 获取所有需要训练的历史日期
        if date not in self.trading_dates:
            print(f"日期 {date} 不是交易日")
            return
            
        date_index = self.trading_dates.index(date)
        
        # 确保有足够的历史数据
        if date_index < TRAINING_DAYS:
            print(f"日期 {date} 前没有足够的历史数据用于训练")
            return
        
        # 获取训练日期范围
        training_dates = self.trading_dates[date_index-TRAINING_DAYS:date_index]
        
        # 初始化模型字典
        self.models = {}
        
        # 为每个板块训练模型
        for sector, features in sector_features.items():
            # 准备训练数据
            X_data = []
            y_data = []
            
            for train_date in training_dates:
                # 获取该日期的特征
                dt = pd.to_datetime(train_date, format='%Y%m%d')
                if dt not in features.index:
                    continue
                    
                X_row = features.loc[dt].values
                
                # 获取标签
                label = self.prepare_label_for_sector(sector, train_date, forward_days)
                
                if label is not None:
                    X_data.append(X_row)
                    y_data.append(label)
            
            if len(X_data) < 10 or len(set(y_data)) < 2:
                print(f"板块 {sector} 训练数据不足或标签单一，跳过训练")
                continue
            
            # 转换为numpy数组
            X = np.array(X_data)
            y = np.array(y_data)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型 (使用LightGBM，表现通常优于传统随机森林)
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            
            # 训练
            model.fit(X_train_scaled, y_train)
            
            # 评估
            y_pred = model.predict(X_test_scaled)
            accuracy = (y_pred == y_test).mean()
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"板块 {sector} 模型训练完成:")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            
            # 保存模型相关数据
            self.models[sector] = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'feature_names': features.columns.tolist(),
                'feature_importances': dict(zip(features.columns, model.feature_importances_