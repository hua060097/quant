#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股量化交易策略
集成技术分析、板块轮动、市场情绪和高级机器学习模型
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import warnings
import talib
import akshare as ak
import joblib
import gc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from xtquant import xtdata
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount
import mmap
import tempfile
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')

class MemoryMappedStorage:
    """内存映射存储类，用于处理大数据集"""
    
    def __init__(self, max_memory_size=1024*1024*1024):  # 默认1GB内存限制
        self.max_memory_size = max_memory_size
        self.temp_files = []
        self.storage = {}
    
    def store(self, key, data):
        """存储数据，如果数据过大则使用内存映射"""
        try:
            data_size = data.memory_usage().sum() if hasattr(data, 'memory_usage') else sys.getsizeof(data)
            
            if data_size > self.max_memory_size * 0.1:  # 如果数据大小超过阈值
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                self.temp_files.append(temp_file.name)
                
                # 将数据保存到临时文件
                with open(temp_file.name, 'wb') as f:
                    pickle.dump(data, f)
                
                # 创建内存映射
                with open(temp_file.name, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    self.storage[key] = {'type': 'mmap', 'file': temp_file.name, 'mmap': mm}
            else:
                # 直接存储在内存中
                self.storage[key] = {'type': 'memory', 'data': data}
                
        except Exception as e:
            print(f"存储数据时出错: {e}")
            # 如果出错，直接存储在内存中
            self.storage[key] = {'type': 'memory', 'data': data}
    
    def retrieve(self, key):
        """获取数据"""
        if key not in self.storage:
            return None
            
        item = self.storage[key]
        if item['type'] == 'memory':
            return item['data']
        else:
            # 从内存映射文件加载数据
            with open(item['file'], 'rb') as f:
                return pickle.load(f)
    
    def clear(self):
        """清理所有数据和临时文件"""
        for key, item in self.storage.items():
            if item['type'] == 'mmap':
                item['mmap'].close()
        
        self.storage.clear()
        
        # 删除临时文件
        for file_path in self.temp_files:
            try:
                os.remove(file_path)
            except:
                pass
        
        self.temp_files = []
        gc.collect()  # 强制垃圾回收

class AShareQuantStrategy:
    def __init__(self, start_date='20230601', end_date=None, initial_capital=100000):
        """
        初始化策略
        
        参数:
            start_date: 回测开始日期，格式'YYYYMMDD'
            end_date: 回测结束日期，格式'YYYYMMDD'，默认为当前日期
            initial_capital: 初始资金，默认10万元
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.now().strftime('%Y%m%d')
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 持仓
        self.position_history = []  # 持仓历史
        self.trade_history = []  # 交易历史
        self.daily_returns = []  # 每日收益率
        self.daily_capital = []  # 每日资金
        
        # 仓位管理参数
        self.max_position_ratio = 0.95  # 最大仓位比例
        self.min_position_ratio = 0.2   # 最小仓位比例
        self.current_position_ratio = 0.5  # 当前仓位比例，初始为50%
        
        # 内存映射存储
        self.storage = MemoryMappedStorage()
        
        # 连接数据源
        self._connect_data_source()
        
        # 获取股票列表
        self.stock_list = xtdata.get_stock_list_in_sector('沪深A股')
        print(f"获取到 {len(self.stock_list)} 只股票")
        
        # 获取行业板块信息
        self.industry_info = self._get_industry_info()
        
        # 机器学习模型
        self.ml_models = {}  # 多模型集成
        self.feature_importance = {}  # 特征重要性
        self.scaler = StandardScaler()
        
        # 缓存数据
        self.data_cache = {}
        
    def _connect_data_source(self):
        """连接数据源"""
        try:
            # 直接尝试连接xtquant数据源
            xtdata.connect()
            print("成功连接到xtquant数据源")
            
            # 获取股票列表
            self.stock_list = xtdata.get_stock_list_in_sector('沪深A股')
            print(f"获取到 {len(self.stock_list)} 只股票")
            
            # 获取行业信息
            self.industry_info = self._get_industry_info()
            
            # 获取交易日历
            self.trading_dates = self._get_trading_dates(self.start_date, self.end_date)
            print(f"获取到 {len(self.trading_dates)} 个交易日")
            
            return True
        except Exception as e:
            print(f"连接数据源失败: {e}")
            return False
    
    def _get_industry_info(self):
        """获取股票行业分类信息"""
        try:
            # 使用akshare获取申万行业分类
            industry_data = ak.stock_sector_spot()
            # 转换为字典格式 {股票代码: 行业名称}
            industry_dict = {}
            for _, row in industry_data.iterrows():
                if 'name' in row and 'industry' in row:
                    stock_code = row['name'].split('(')[1].split(')')[0]
                    if stock_code.startswith('6'):
                        stock_code = f"SH{stock_code}"
                    else:
                        stock_code = f"SZ{stock_code}"
                    industry_dict[stock_code] = row['industry']
            return industry_dict
        except Exception as e:
            print(f"获取行业信息失败: {e}")
            # 如果akshare获取失败，尝试从xtquant获取
            try:
                industry_dict = {}
                for stock in self.stock_list:
                    info = xtdata.get_stock_info(stock)
                    if info and 'industry' in info:
                        industry_dict[stock] = info['industry']
                return industry_dict
            except:
                print("无法获取行业信息，将不使用行业因子")
                return {}
    
    def _get_market_sentiment(self, date):
        """获取市场情绪指标"""
        try:
            # 获取上证指数数据作为市场情绪参考
            index_data = xtdata.get_market_data(
                stock_list=['SH000001'],
                period='1d',
                start_time=self._get_previous_date(date, 30),
                end_time=date,
                field_list=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not index_data or 'SH000001' not in index_data:
                return {
                    'market_trend': 0,
                    'market_volatility': 0,
                    'market_volume_ratio': 1,
                    'market_sentiment_score': 0.5  # 中性
                }
            
            df = index_data['SH000001']
            
            # 计算市场趋势（10日涨跌幅）
            if len(df) > 10:
                market_trend = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
            else:
                market_trend = 0
                
            # 计算市场波动率（10日标准差）
            if len(df) > 10:
                returns = df['close'].pct_change().dropna()
                market_volatility = returns.std() * 100
            else:
                market_volatility = 0
                
            # 计算成交量比率（当日成交量/10日平均成交量）
            if len(df) > 10:
                avg_volume = df['volume'].iloc[-10:-1].mean()
                market_volume_ratio = df['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            else:
                market_volume_ratio = 1
            
            # 计算市场情绪得分 (0-1之间，越高越乐观)
            # 考虑趋势、波动率和成交量
            trend_score = min(max((market_trend + 5) / 10, 0), 1)  # 将-5%到+5%映射到0-1
            volatility_score = min(max(1 - market_volatility / 3, 0), 1)  # 波动率越低越好
            volume_score = min(max(market_volume_ratio / 2, 0), 1)  # 成交量比率映射到0-1
            
            # 综合得分 (趋势权重高)
            market_sentiment_score = 0.6 * trend_score + 0.2 * volatility_score + 0.2 * volume_score
            
            # 根据市场情绪调整仓位比例
            self._adjust_position_ratio(market_sentiment_score)
                
            return {
                'market_trend': market_trend,
                'market_volatility': market_volatility,
                'market_volume_ratio': market_volume_ratio,
                'market_sentiment_score': market_sentiment_score
            }
        except Exception as e:
            print(f"获取市场情绪失败: {e}")
            return {
                'market_trend': 0,
                'market_volatility': 0,
                'market_volume_ratio': 1,
                'market_sentiment_score': 0.5  # 中性
            }
    
    def _adjust_position_ratio(self, sentiment_score):
        """根据市场情绪调整仓位比例"""
        # 将情绪得分映射到仓位比例范围
        target_ratio = self.min_position_ratio + sentiment_score * (self.max_position_ratio - self.min_position_ratio)
        
        # 平滑调整，避免剧烈变化
        self.current_position_ratio = 0.7 * self.current_position_ratio + 0.3 * target_ratio
        
        print(f"市场情绪得分: {sentiment_score:.2f}, 目标仓位比例: {self.current_position_ratio:.2%}")
    
    def _get_industry_performance(self, date):
        """获取行业板块表现"""
        try:
            # 使用akshare获取申万一级行业指数表现
            industry_index = ak.sw_index_spot()
            industry_perf = {}
            
            for _, row in industry_index.iterrows():
                if '指数代码' in row and '指数名称' in row and '涨跌幅' in row:
                    code = row['指数代码']
                    name = row['指数名称']
                    change_pct = row['涨跌幅']
                    industry_perf[name] = change_pct
            
            # 获取行业排名
            sorted_industry = sorted(industry_perf.items(), key=lambda x: x[1], reverse=True)
            industry_rank = {ind[0]: idx+1 for idx, ind in enumerate(sorted_industry)}
            
            # 计算行业强度 (0-1之间，越高越强)
            industry_strength = {}
            total_industries = len(industry_perf)
            for ind, rank in industry_rank.items():
                industry_strength[ind] = 1 - (rank - 1) / total_industries if total_industries > 0 else 0.5
            
            return {
                'industry_performance': industry_perf,
                'industry_rank': industry_rank,
                'industry_strength': industry_strength
            }
        except Exception as e:
            print(f"获取行业表现失败: {e}")
            return {
                'industry_performance': {},
                'industry_rank': {},
                'industry_strength': {}
            }
    
    def _get_previous_date(self, date, days=1):
        """获取N天前的日期"""
        date_obj = datetime.datetime.strptime(date, '%Y%m%d')
        prev_date = date_obj - datetime.timedelta(days=days)
        return prev_date.strftime('%Y%m%d')
    
    def _calculate_technical_indicators(self, df):
        """计算技术指标"""
        if df is None or len(df) < 60:
            return None
        
        try:
            # 复制DataFrame以避免修改原始数据
            df = df.copy()
            
            # 计算移动平均线
            df['ma5'] = talib.MA(df['close'].values, timeperiod=5)
            df['ma10'] = talib.MA(df['close'].values, timeperiod=10)
            df['ma20'] = talib.MA(df['close'].values, timeperiod=20)
            df['ma60'] = talib.MA(df['close'].values, timeperiod=60)
            
            # 计算MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            df['macd_hist_change'] = df['macd_hist'].diff()
            
            # 计算RSI
            df['rsi6'] = talib.RSI(df['close'].values, timeperiod=6)
            df['rsi12'] = talib.RSI(df['close'].values, timeperiod=12)
            df['rsi24'] = talib.RSI(df['close'].values, timeperiod=24)
            df['rsi6_change'] = df['rsi6'].diff()
            
            # 计算布林带
            df['upper'], df['middle'], df['lower'] = talib.BBANDS(
                df['close'].values, 
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            
            # 计算布林带位置
            df['bb_position'] = (df['close'] - df['lower']) / (df['upper'] - df['lower'])
            
            # 计算KDJ
            df['k'], df['d'] = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=9,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            df['j'] = 3 * df['k'] - 2 * df['d']
            
            # 计算成交量比率
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # 计算波动率
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * (252 ** 0.5)
            
            # 计算支撑位和压力位
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            df['dist_to_support'] = (df['close'] - df['support']) / df['close']
            df['dist_to_resistance'] = (df['resistance'] - df['close']) / df['close']
            
            # 计算趋势强度
            df['trend_strength'] = abs(df['ma5'] - df['ma20']) / df['ma20']
            
            # 计算动量指标
            df['momentum_1d'] = df['close'].pct_change(1)
            df['momentum_3d'] = df['close'].pct_change(3)
            df['momentum_5d'] = df['close'].pct_change(5)
            df['momentum_10d'] = df['close'].pct_change(10)
            df['momentum_20d'] = df['close'].pct_change(20)
            
            # 计算均线差值
            df['ma5_10_diff'] = (df['ma5'] - df['ma10']) / df['ma10']
            df['ma10_20_diff'] = (df['ma10'] - df['ma20']) / df['ma20']
            df['ma20_60_diff'] = (df['ma20'] - df['ma60']) / df['ma60']
            
            # 计算交叉信号
            df['kdj_golden_cross'] = ((df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1))).astype(int)
            df['kdj_dead_cross'] = ((df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1))).astype(int)
            df['ma5_10_golden_cross'] = ((df['ma5'] > df['ma10']) & (df['ma5'].shift(1) <= df['ma10'].shift(1))).astype(int)
            df['ma5_10_dead_cross'] = ((df['ma5'] < df['ma10']) & (df['ma5'].shift(1) >= df['ma10'].shift(1))).astype(int)
            df['macd_golden_cross'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            df['macd_dead_cross'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
            
            # 计算突破信号
            df['bb_upper_break'] = (df['close'] > df['upper']).astype(int)
            df['bb_lower_break'] = (df['close'] < df['lower']).astype(int)
            df['volume_break'] = (df['volume'] > df['volume'].rolling(window=20).mean() * 2).astype(int)
            
            # 计算未来5天的收益率作为目标变量
            df['future_return'] = df['close'].pct_change(5).shift(-5)
            
            # 创建目标变量（未来5天收益率大于3%为1，否则为0）
            df['target'] = (df['future_return'] > 0.03).astype(int)
            
            # 删除NaN值
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"计算技术指标时出错: {e}")
            return None
    
    def _prepare_features(self, df):
        """准备特征数据"""
        if df is None or len(df) < 30:
            return None
            
        # 选择最后一行数据（当前交易日）
        current = df.iloc[-1].copy()
        
        # 构建特征字典
        features = {}
        
        # 价格动量特征
        features['price_ma5_ratio'] = current['close'] / current['ma5'] if current['ma5'] > 0 else 1
        features['price_ma10_ratio'] = current['close'] / current['ma10'] if current['ma10'] > 0 else 1
        features['price_ma20_ratio'] = current['close'] / current['ma20'] if current['ma20'] > 0 else 1
        features['price_ma60_ratio'] = current['close'] / current['ma60'] if current['ma60'] > 0 else 1
        features['ma5_ma20_ratio'] = current['ma5'] / current['ma20'] if current['ma20'] > 0 else 1
        
        # MACD特征
        features['macd'] = current['macd']
        features['macd_signal'] = current['macd_signal']
        features['macd_hist'] = current['macd_hist']
        features['macd_hist_change'] = current['macd_hist_change']
        
        # RSI特征
        features['rsi6'] = current['rsi6']
        features['rsi12'] = current['rsi12']
        features['rsi24'] = current['rsi24']
        features['rsi6_change'] = current['rsi6_change']
        
        # 布林带特征
        features['bb_position'] = current['bb_position']
        features['bb_width'] = (current['upper'] - current['lower']) / current['middle'] if current['middle'] > 0 else 0
        
        # KDJ特征
        features['k'] = current['k']
        features['d'] = current['d']
        features['j'] = current['j']
        
        # 成交量特征
        features['volume_ratio'] = current['volume_ratio']
        
        # 波动率特征
        features['volatility'] = current['volatility']
        
        # 支撑压力特征
        features['dist_to_support'] = current['dist_to_support']
        features['dist_to_resistance'] = current['dist_to_resistance']
        
        # 趋势特征
        features['trend_strength'] = current['trend_strength']
        
        # 动量特征
        features['momentum_1d'] = current['momentum_1d']
        features['momentum_3d'] = current['momentum_3d']
        features['momentum_5d'] = current['momentum_5d']
        features['momentum_10d'] = current['momentum_10d']
        features['momentum_20d'] = current['momentum_20d']
        
        # 均线差值特征
        features['ma5_10_diff'] = current['ma5_10_diff']
        features['ma10_20_diff'] = current['ma10_20_diff']
        features['ma20_60_diff'] = current['ma20_60_diff']
        
        # 交叉信号特征
        features['kdj_golden_cross'] = current['kdj_golden_cross']
        features['kdj_dead_cross'] = current['kdj_dead_cross']
        features['ma5_10_golden_cross'] = current['ma5_10_golden_cross']
        features['ma5_10_dead_cross'] = current['ma5_10_dead_cross']
        features['macd_golden_cross'] = current['macd_golden_cross']
        features['macd_dead_cross'] = current['macd_dead_cross']
        
        # 突破特征
        features['bb_upper_break'] = current['bb_upper_break']
        features['bb_lower_break'] = current['bb_lower_break']
        features['volume_break'] = current['volume_break']
        
        return features
    
    def _prepare_training_data(self, stock_data_dict):
        """准备训练数据"""
        X = []
        y = []
        feature_names = None
        
        for stock_code, df in stock_data_dict.items():
            if df is None or len(df) < 60:
                continue
                
            # 去除NaN值
            df = df.dropna()
            
            if len(df) < 30:
                continue
                
            # 提取特征和目标
            for i in range(30, len(df) - 10):  # 确保有足够的历史数据和未来数据
                row = df.iloc[i]
                
                # 构建特征字典
                features = {}
                
                # 价格动量特征
                features['price_ma5_ratio'] = row['close'] / row['ma5'] if row['ma5'] > 0 else 1
                features['price_ma10_ratio'] = row['close'] / row['ma10'] if row['ma10'] > 0 else 1
                features['price_ma20_ratio'] = row['close'] / row['ma20'] if row['ma20'] > 0 else 1
                features['price_ma60_ratio'] = row['close'] / row['ma60'] if row['ma60'] > 0 else 1
                features['ma5_ma20_ratio'] = row['ma5'] / row['ma20'] if row['ma20'] > 0 else 1
                
                # MACD特征
                features['macd'] = row['macd']
                features['macd_signal'] = row['macd_signal']
                features['macd_hist'] = row['macd_hist']
                features['macd_hist_change'] = row['macd_hist_change']
                
                # RSI特征
                features['rsi6'] = row['rsi6']
                features['rsi12'] = row['rsi12']
                features['rsi24'] = row['rsi24']
                features['rsi6_change'] = row['rsi6_change']
                
                # 布林带特征
                features['bb_position'] = row['bb_position']
                features['bb_width'] = (row['upper'] - row['lower']) / row['middle'] if row['middle'] > 0 else 0
                
                # KDJ特征
                features['k'] = row['k']
                features['d'] = row['d']
                features['j'] = row['j']
                
                # 成交量特征
                features['volume_ratio'] = row['volume_ratio']
                
                # 波动率特征
                features['volatility'] = row['volatility']
                
                # 支撑压力特征
                features['dist_to_support'] = row['dist_to_support']
                features['dist_to_resistance'] = row['dist_to_resistance']
                
                # 趋势特征
                features['trend_strength'] = row['trend_strength']
                
                # 动量特征
                features['momentum_1d'] = row['momentum_1d']
                features['momentum_3d'] = row['momentum_3d']
                features['momentum_5d'] = row['momentum_5d']
                features['momentum_10d'] = row['momentum_10d']
                features['momentum_20d'] = row['momentum_20d']
                
                # 均线差值特征
                features['ma5_10_diff'] = row['ma5_10_diff']
                features['ma10_20_diff'] = row['ma10_20_diff']
                features['ma20_60_diff'] = row['ma20_60_diff']
                
                # 交叉信号特征
                features['kdj_golden_cross'] = row['kdj_golden_cross']
                features['kdj_dead_cross'] = row['kdj_dead_cross']
                features['ma5_10_golden_cross'] = row['ma5_10_golden_cross']
                features['ma5_10_dead_cross'] = row['ma5_10_dead_cross']
                features['macd_golden_cross'] = row['macd_golden_cross']
                features['macd_dead_cross'] = row['macd_dead_cross']
                
                # 突破特征
                features['bb_upper_break'] = row['bb_upper_break']
                features['bb_lower_break'] = row['bb_lower_break']
                features['volume_break'] = row['volume_break']
                
                # 记录特征名称
                if feature_names is None:
                    feature_names = list(features.keys())
                
                # 添加到训练数据
                X.append(list(features.values()))
                y.append(row['target'])
        
        if not X or not y:
            return None, None, None
            
        return np.array(X), np.array(y), feature_names
    
    def _select_training_stocks(self, date, sample_size=300):
        """
        智能选择训练样本股票
        结合行业分布、近期涨幅和跌幅来选择更有代表性的样本
        
        参数:
            date: 当前日期
            sample_size: 样本大小
        """
        print(f"选择训练样本股票 - 日期: {date}")
        
        # 获取前一个交易日
        prev_date = self._get_previous_date(date, 1)
        # 获取20个交易日前的日期
        start_date = self._get_previous_date(date, 20)
        
        # 获取行业分类信息
        industry_groups = {}
        for stock in self.stock_list:
            industry = self.industry_info.get(stock, '其他')
            if industry not in industry_groups:
                industry_groups[industry] = []
            industry_groups[industry].append(stock)
        
        # 计算每个行业应选取的股票数量（按行业规模比例）
        industry_sample_sizes = {}
        total_stocks = len(self.stock_list)
        for industry, stocks in industry_groups.items():
            # 每个行业至少选择3只股票
            industry_sample_sizes[industry] = max(3, int(len(stocks) / total_stocks * sample_size * 0.6))
        
        # 获取所有股票的近期涨跌幅
        stock_returns = {}
        winners = []  # 涨幅较大的股票
        losers = []   # 跌幅较大的股票
        
        # 批量获取股票数据以提高效率
        batch_size = 100
        for i in range(0, len(self.stock_list), batch_size):
            batch_stocks = self.stock_list[i:i+batch_size]
            print(f"获取股票数据 - 批次: {i}, 股票数量: {len(batch_stocks)}")
            try:
                # 获取这批股票的历史数据
                field_list = ['close']
                batch_data = xtdata.get_market_data(
                    field_list=field_list,
                    stock_list=batch_stocks,
                    period='1d',
                    start_time=start_date,
                    end_time=date
                )
                
                # 计算每只股票的涨跌幅
                for stock_code in batch_stocks:
                    if 'close' not in batch_data or stock_code not in batch_data['close'] or len(batch_data['close'][stock_code]) < 2:
                        continue
                    
                    # 计算20日涨跌幅
                    start_price = batch_data['close'][stock_code][0]
                    print(f"开始价格: {start_price}")
                    end_price = batch_data['close'][stock_code][-1]
                    print(f"结束价格: {end_price}")
                    
                    if start_price <= 0:
                        continue
                        
                    return_rate = (end_price / start_price - 1) * 100
                    stock_returns[stock_code] = return_rate
                    print(f"涨跌幅: {return_rate}")
            except Exception as e:
                print(f"获取股票数据时出错: {e}")
                continue
        
        # 按涨跌幅排序
        sorted_returns = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)
        
        # 选择涨幅最大的20%作为赢家
        winner_count = int(len(sorted_returns) * 0.2)
        winners = [stock for stock, _ in sorted_returns[:winner_count]]

        print(f"赢家数量: {len(winners)}, 赢家: {winners}")
        
        # 选择跌幅最大的20%作为输家
        losers = [stock for stock, _ in sorted_returns[-winner_count:]]
        print(f"输家数量: {len(losers)}, 输家: {losers}")
        
        # 按行业选择股票
        industry_samples = []
        for industry, size in industry_sample_sizes.items():
            if industry in industry_groups and len(industry_groups[industry]) > 0:
                # 从每个行业中随机选择指定数量的股票
                selected = np.random.choice(
                    industry_groups[industry], 
                    min(size, len(industry_groups[industry])), 
                    replace=False
                )
                industry_samples.extend(selected)
        
        # 确保赢家和输家中的一部分被包含在样本中
        # 从赢家中选择20%的样本量
        winner_sample_size = int(sample_size * 0.2)
        if len(winners) > winner_sample_size:
            winner_samples = np.random.choice(winners, winner_sample_size, replace=False)
        else:
            winner_samples = winners
        
        # 从输家中选择20%的样本量
        loser_sample_size = int(sample_size * 0.2)
        if len(losers) > loser_sample_size:
            loser_samples = np.random.choice(losers, loser_sample_size, replace=False)
        else:
            loser_samples = losers
        
        # 合并所有样本
        sample_stocks = list(set(industry_samples + list(winner_samples) + list(loser_samples)))
        
        # 如果样本数量不足，从剩余股票中随机选择
        if len(sample_stocks) < sample_size:
            remaining_stocks = list(set(self.stock_list) - set(sample_stocks))
            if remaining_stocks:
                additional_samples = np.random.choice(
                    remaining_stocks, 
                    min(sample_size - len(sample_stocks), len(remaining_stocks)), 
                    replace=False
                )
                sample_stocks.extend(additional_samples)
        
        print(f"选择了 {len(sample_stocks)} 只样本股票进行训练")
        print(f"其中包含 {len(set(sample_stocks) & set(winners))} 只赢家股票和 {len(set(sample_stocks) & set(losers))} 只输家股票")
        
        return sample_stocks, winners, losers
    
    def _train_ml_model(self, date):
        """训练机器学习模型"""
        print(f"开始训练机器学习模型 - 日期: {date}")
        
        # 智能选择训练样本股票
        sample_stocks, winners, losers = self._select_training_stocks(date, sample_size=300)
        
        if not sample_stocks or len(sample_stocks) < 50:
            print("没有足够的样本股票进行训练")
            return None
        
        # 获取股票数据并计算技术指标
        stock_data_dict = {}
        for stock_code in sample_stocks:
            try:
                # 获取历史数据（使用更长的时间窗口以确保有足够的训练数据）
                daily_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=self._get_previous_date(date, 180),
                    end_time=date,
                    field_list=['open', 'high', 'low', 'close', 'volume', 'amount']
                )
                
                if not daily_data or stock_code not in daily_data or len(daily_data[stock_code]) < 60:
                    continue
                
                # 计算技术指标
                df = self._calculate_technical_indicators(daily_data[stock_code])
                
                if df is not None and len(df) >= 60:
                    stock_data_dict[stock_code] = df
            except Exception as e:
                print(f"处理股票 {stock_code} 数据时出错: {e}")
        
        if len(stock_data_dict) < 50:
            print(f"只有 {len(stock_data_dict)} 只股票有足够的数据，不足以训练模型")
            return None
        
        print(f"使用 {len(stock_data_dict)} 只股票的数据训练模型")
        
        # 准备训练数据
        X, y, feature_names = self._prepare_training_data(stock_data_dict)
        
        if X is None or y is None or len(X) < 100:
            print("没有足够的数据训练模型")
            return None
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集（使用时间序列分割）
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        # 训练多个模型并集成
        models = {}
        
        # 1. 随机森林
        print("训练随机森林模型...")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        models['random_forest'] = {
            'model': rf_model,
            'weight': 0.3,
            'feature_importance': rf_model.feature_importances_
        }
        
        # 2. LightGBM
        print("训练LightGBM模型...")
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
            models['lightgbm'] = {
                'model': lgb_model,
                'weight': 0.4,
                'feature_importance': lgb_model.feature_importances_
            }
        except Exception as e:
            print(f"LightGBM训练失败: {e}")
        
        # 3. XGBoost
        print("训练XGBoost模型...")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
            models['xgboost'] = {
                'model': xgb_model,
                'weight': 0.3,
                'feature_importance': xgb_model.feature_importances_
            }
        except Exception as e:
            print(f"XGBoost训练失败: {e}")
        
        # 评估集成模型
        if len(models) > 0:
            # 计算加权预测
            weighted_pred = np.zeros(len(y_test))
            total_weight = 0
            
            for model_name, model_info in models.items():
                if model_name == 'random_forest':
                    weighted_pred += model_info['weight'] * rf_model.predict_proba(X_test)[:, 1]
                elif model_name == 'lightgbm':
                    weighted_pred += model_info['weight'] * lgb_model.predict_proba(X_test)[:, 1]
                elif model_name == 'xgboost':
                    weighted_pred += model_info['weight'] * xgb_model.predict_proba(X_test)[:, 1]
                total_weight += model_info['weight']
            
            weighted_pred /= total_weight
            
            # 使用0.5作为阈值进行分类
            y_pred = (weighted_pred >= 0.5).astype(int)
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"集成模型评估结果:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            
            # 分析特征重要性
            print("\n特征重要性分析:")
            feature_importance = {}
            
            for model_name, model_info in models.items():
                for i, importance in enumerate(model_info['feature_importance']):
                    if i < len(feature_names):
                        feature = feature_names[i]
                        if feature not in feature_importance:
                            feature_importance[feature] = 0
                        feature_importance[feature] += importance * model_info['weight']
            
            # 归一化特征重要性
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= total_importance
            
            # 显示前10个最重要的特征
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in top_features:
                print(f"{feature}: {importance:.4f}")
            
            # 保存模型和特征重要性
            self.ml_model = {
                'models': models,
                'scaler': scaler,
                'feature_names': feature_names,
                'feature_importance': feature_importance,
                'training_date': date
            }
            
            # 分析赢家和输家的特征差异
            self._analyze_winner_loser_patterns(stock_data_dict, sample_stocks, winners, losers)
            
            return self.ml_model
        else:
            print("所有模型训练失败")
            return None
    
    def _analyze_winner_loser_patterns(self, stock_data_dict, sample_stocks, winners, losers):
        """分析赢家和输家的特征模式差异"""
        print("分析赢家和输家的特征模式...")
        
        # 收集赢家和输家的特征
        winner_features = defaultdict(list)
        loser_features = defaultdict(list)
        
        for stock_code, df in stock_data_dict.items():
            if stock_code in winners:
                # 获取最近的特征
                features = self._prepare_features(df)
                if features:
                    for feature, value in features.items():
                        winner_features[feature].append(value)
            elif stock_code in losers:
                # 获取最近的特征
                features = self._prepare_features(df)
                if features:
                    for feature, value in features.items():
                        loser_features[feature].append(value)
        
        # 计算平均值并比较差异
        feature_diff = {}
        for feature in winner_features:
            if feature in loser_features and len(winner_features[feature]) > 0 and len(loser_features[feature]) > 0:
                winner_avg = np.mean(winner_features[feature])
                loser_avg = np.mean(loser_features[feature])
                
                # 计算差异百分比
                if abs(loser_avg) > 1e-10:  # 避免除以零
                    diff_pct = (winner_avg - loser_avg) / abs(loser_avg) * 100
                else:
                    diff_pct = 0
                
                feature_diff[feature] = {
                    'winner_avg': winner_avg,
                    'loser_avg': loser_avg,
                    'diff_pct': diff_pct
                }
        
        # 显示差异最大的特征
        sorted_diff = sorted(feature_diff.items(), key=lambda x: abs(x[1]['diff_pct']), reverse=True)
        
        print("赢家和输家的关键特征差异:")
        for feature, stats in sorted_diff[:10]:
            print(f"{feature}: 赢家={stats['winner_avg']:.4f}, 输家={stats['loser_avg']:.4f}, 差异={stats['diff_pct']:.2f}%")
        
        # 保存这些模式用于后续选股
        self.winner_loser_patterns = {
            'feature_diff': feature_diff,
            'top_diff_features': [f for f, _ in sorted_diff[:10]]
        }
    
    def _predict_stock_return(self, features, model):
        """使用模型预测股票收益"""
        if model is None or features is None:
            return 0
            
        # 转换特征为numpy数组
        X = np.array(list(features.values())).reshape(1, -1)
        
        # 标准化特征
        X = self.scaler.transform(X)
        
        # 预测概率
        proba = model.predict_proba(X)[0]
        
        # 返回正类的概率
        if len(proba) > 1:
            return proba[1]
        return 0.5
    
    def _select_stocks(self, date, top_n=10):
        """选择股票"""
        print(f"选股 - 日期: {date}")
        
        # 获取前一个交易日
        prev_date = self._get_previous_date(date, 1)
        
        # 获取所有股票的数据
        stock_scores = {}
        
        # 批量处理股票以提高效率
        batch_size = 100
        for i in range(0, len(self.stock_list), batch_size):
            batch_stocks = self.stock_list[i:i+batch_size]
            
            # 获取这批股票的历史数据
            batch_data = self._get_history_data(
                stock_list=batch_stocks,
                period='1d',
                start_time=self._get_previous_date(date, 60),
                end_time=date
            )
            
            # 处理每只股票
            for stock_code in batch_stocks:
                try:
                    if stock_code not in batch_data or len(batch_data[stock_code]) < 30:
                        continue
                    
                    # 计算技术指标
                    df = self._calculate_technical_indicators(batch_data[stock_code])
                    
                    if df is None or len(df) < 30:
                        continue
                    
                    # 获取最新的一行数据
                    latest_data = df.iloc[-1]
                    
                    # 计算技术分析得分
                    tech_score = self._calculate_technical_score(latest_data)
                    
                    # 如果有机器学习模型，使用模型预测
                    ml_score = 0
                    if self.ml_model is not None:
                        ml_score = self._predict_with_ml_model(df)
                    
                    # 获取行业得分
                    industry = self.industry_info.get(stock_code, '其他')
                    industry_score = self.industry_scores.get(industry, 0)
                    
                    # 综合得分（技术分析40%，机器学习40%，行业20%）
                    total_score = tech_score * 0.4 + ml_score * 0.4 + industry_score * 0.2
                    
                    # 记录得分
                    stock_scores[stock_code] = total_score
                except Exception as e:
                    print(f"处理股票 {stock_code} 时出错: {e}")
        
        # 根据得分排序并选择前N只股票
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = [stock for stock, _ in sorted_stocks[:top_n]]
        
        print(f"选择了 {len(selected_stocks)} 只股票")
        return selected_stocks
    
    def _calculate_technical_score(self, df):
        """计算技术分析得分"""
        if df is None or len(df) < 30:
            return 0
            
        # 获取最新数据
        current = df.iloc[-1]
        
        score = 0
        
        # 1. 均线多头排列得分 (20%)
        if current['ma5'] > current['ma10'] > current['ma20'] > current['ma60']:
            score += 0.2
        elif current['ma5'] > current['ma10'] > current['ma20']:
            score += 0.15
        elif current['ma5'] > current['ma10']:
            score += 0.1
        
        # 2. MACD得分 (15%)
        if current['macd'] > current['macd_signal'] and current['macd_hist'] > 0:
            score += 0.15
        elif current['macd'] > current['macd_signal']:
            score += 0.1
        elif current['macd_hist'] > 0:
            score += 0.05
        
        # 3. RSI得分 (15%)
        if 40 < current['rsi6'] < 70:  # 适中的RSI
            score += 0.1
        if current['rsi6'] > current['rsi12'] > current['rsi24']:  # RSI多头排列
            score += 0.05
        
        # 4. 布林带得分 (15%)
        bb_position = current['bb_position']
        if 0.3 < bb_position < 0.7:  # 价格在布林带中间区域
            score += 0.1
        elif 0.7 <= bb_position < 0.9:  # 价格接近上轨但未超出
            score += 0.15
        
        # 5. KDJ得分 (15%)
        if current['j'] > current['k'] > current['d']:  # KDJ金叉
            score += 0.15
        elif current['k'] > current['d']:
            score += 0.1
        
        # 6. 成交量得分 (10%)
        if current['volume_ratio'] > 1.5:  # 成交量明显放大
            score += 0.1
        elif current['volume_ratio'] > 1:
            score += 0.05
        
        # 7. 支撑压力得分 (10%)
        if current['dist_to_resistance'] > 0.05 and current['dist_to_support'] < 0.03:
            score += 0.1  # 接近支撑远离压力
        elif current['dist_to_resistance'] > 0.03:
            score += 0.05  # 远离压力
        
        return score
    
    def _execute_trades(self, date, selected_stocks):
        """执行交易"""
        print(f"执行交易 - 日期: {date}")
        
        # 获取当前持仓
        current_positions = self.positions.copy()
        
        # 计算每只股票的目标仓位
        target_position_value = self.capital / len(selected_stocks) if selected_stocks else 0
        
        # 卖出不在选股列表中的股票
        for stock_code in list(current_positions.keys()):
            if stock_code not in selected_stocks:
                # 获取当前价格
                price_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=date,
                    end_time=date,
                    field_list=['open', 'high', 'low', 'close', 'volume']
                )
                
                if not price_data or stock_code not in price_data:
                    print(f"无法获取 {stock_code} 的价格数据，跳过卖出")
                    continue
                
                price = price_data[stock_code]['close'].iloc[0]
                shares = current_positions[stock_code]
                
                # 卖出股票
                trade_value = price * shares
                self.capital += trade_value
                
                # 记录交易
                self.trade_history.append({
                    'date': date,
                    'stock': stock_code,
                    'action': 'sell',
                    'price': price,
                    'shares': shares,
                    'value': trade_value
                })
                
                print(f"卖出: {stock_code}, 价格: {price}, 数量: {shares}, 金额: {trade_value:.2f}")
                
                # 更新持仓
                del self.positions[stock_code]
        
        # 买入或调整选中的股票
        for stock_code in selected_stocks:
            # 获取当前价格
            price_data = xtdata.get_market_data(
                stock_list=[stock_code],
                period='1d',
                start_time=date,
                end_time=date,
                field_list=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not price_data or stock_code not in price_data:
                print(f"无法获取 {stock_code} 的价格数据，跳过买入")
                continue
            
            price = price_data[stock_code]['close'].iloc[0]
            
            # 计算目标持仓数量
            target_shares = int(target_position_value / price / 100) * 100  # 取整百
            
            # 获取当前持仓数量
            current_shares = current_positions.get(stock_code, 0)
            
            # 计算需要买入或卖出的数量
            shares_diff = target_shares - current_shares
            
            if shares_diff > 0:  # 需要买入
                # 检查资金是否足够
                required_capital = shares_diff * price
                if required_capital > self.capital:
                    # 调整买入数量
                    affordable_shares = int(self.capital / price / 100) * 100
                    shares_diff = affordable_shares
                    required_capital = shares_diff * price
                
                if shares_diff > 0:
                    # 买入股票
                    self.capital -= required_capital
                    
                    # 更新持仓
                    self.positions[stock_code] = current_shares + shares_diff
                    
                    # 记录交易
                    self.trade_history.append({
                        'date': date,
                        'stock': stock_code,
                        'action': 'buy',
                        'price': price,
                        'shares': shares_diff,
                        'value': required_capital
                    })
                    
                    print(f"买入: {stock_code}, 价格: {price}, 数量: {shares_diff}, 金额: {required_capital:.2f}")
            
            elif shares_diff < 0:  # 需要卖出
                # 卖出股票
                shares_to_sell = abs(shares_diff)
                trade_value = shares_to_sell * price
                self.capital += trade_value
                
                # 更新持仓
                self.positions[stock_code] = current_shares - shares_to_sell
                
                # 如果持仓为0，删除该股票
                if self.positions[stock_code] == 0:
                    del self.positions[stock_code]
                
                # 记录交易
                self.trade_history.append({
                    'date': date,
                    'stock': stock_code,
                    'action': 'sell',
                    'price': price,
                    'shares': shares_to_sell,
                    'value': trade_value
                })
                
                print(f"卖出: {stock_code}, 价格: {price}, 数量: {shares_to_sell}, 金额: {trade_value:.2f}")
    
    def _calculate_portfolio_value(self, date):
        """计算当前组合价值"""
        portfolio_value = self.capital
        
        for stock_code, shares in self.positions.items():
            try:
                # 获取当前价格
                price_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=date,
                    end_time=date
                )
                
                if not price_data or stock_code not in price_data:
                    print(f"无法获取 {stock_code} 的价格数据，使用上次价格")
                    # 使用上次交易价格
                    last_trade = None
                    for trade in reversed(self.trade_history):
                        if trade['stock'] == stock_code:
                            last_trade = trade
                            break
                    
                    if last_trade:
                        price = last_trade['price']
                    else:
                        continue
                else:
                    price = price_data[stock_code]['close'].iloc[0]
                
                # 计算股票价值
                stock_value = price * shares
                portfolio_value += stock_value
                
            except Exception as e:
                print(f"计算 {stock_code} 价值时出错: {e}")
        
        return portfolio_value
    
    def _get_trading_dates(self):
        """获取交易日期列表"""
        try:
            # 使用xtdata的get_trading_dates方法获取交易日期
            trading_dates = xtdata.get_trading_dates('SH', self.start_date, self.end_date)
            
            # 转换为日期字符串格式
            trading_dates = [xtdata.timetag_to_datetime(date, '%Y%m%d') for date in trading_dates]
            
            return trading_dates
        except Exception as e:
            print(f"获取交易日期失败: {e}")
            return []
    
    def _get_history_data(self, stock_list, period, start_time, end_time):
        """获取历史数据"""
        try:
            # 使用xtdata的get_market_data方法获取历史数据
            field_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
            data = xtdata.get_market_data(
                stock_list=stock_list,
                period=period,
                start_time=start_time,
                end_time=end_time,
                field_list=field_list
            )
            
            if data is None:
                print(f"获取历史数据返回None")
                return {}
                
            # 转换为字典格式，每个股票对应一个DataFrame
            result = {}
            for stock_code in stock_list:
                if stock_code in data:
                    # 创建DataFrame
                    import pandas as pd
                    df = pd.DataFrame(data[stock_code])
                    result[stock_code] = df
            
            return result
        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return {}
    
    def backtest(self, retrain_interval=20, stock_selection_interval=5, top_n=10):
        """
        执行回测
        
        参数:
            retrain_interval: 重新训练模型的间隔天数
            stock_selection_interval: 选股间隔天数
            top_n: 选择的股票数量
        """
        print(f"开始回测: {self.start_date} 至 {self.end_date}")
        
        # 获取交易日期
        trading_dates = self._get_trading_dates()
        if not trading_dates:
            print("无法获取交易日期，回测终止")
            return
        
        print(f"获取到 {len(trading_dates)} 个交易日")
        
        # 初始化回测数据
        self.capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.trade_history = []
        self.daily_returns = []
        self.daily_capital = []
        
        # 记录初始资金
        self.daily_capital.append({
            'date': trading_dates[0],
            'capital': self.capital,
            'portfolio_value': self.capital
        })
        
        # 训练初始模型
        print("训练初始机器学习模型...")
        train_data = {}
        
        # 随机选择一部分股票进行训练
        train_stocks = np.random.choice(self.stock_list, min(100, len(self.stock_list)), replace=False)
        
        for stock_code in train_stocks:
            try:
                # 获取历史数据
                daily_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=self._get_previous_date(trading_dates[0], 180),  # 使用更长的历史数据
                    end_time=trading_dates[0],
                    field_list=['open', 'high', 'low', 'close', 'volume', 'amount']
                )
                
                if not daily_data or stock_code not in daily_data or len(daily_data[stock_code]) < 60:
                    continue
                
                # 计算技术指标
                df = self._calculate_technical_indicators(daily_data[stock_code])
                
                if df is not None and len(df) >= 60:
                    train_data[stock_code] = df
            except Exception as e:
                print(f"获取 {stock_code} 训练数据时出错: {e}")
        
        # 训练模型
        self.ml_models[trading_dates[0]] = self._train_ml_model(trading_dates[0])
        
        # 执行回测
        last_selection_date = None
        last_retrain_date = None
        selected_stocks = []
        
        for i, date in enumerate(trading_dates):
            print(f"\n处理交易日: {date} ({i+1}/{len(trading_dates)})")
            
            # 是否需要重新训练模型
            if last_retrain_date is None or i - trading_dates.index(last_retrain_date) >= retrain_interval:
                print(f"重新训练模型 - 日期: {date}")
                
                # 训练模型
                self.ml_models[date] = self._train_ml_model(date)
                last_retrain_date = date
            
            # 是否需要重新选股
            if last_selection_date is None or i - trading_dates.index(last_selection_date) >= stock_selection_interval:
                print(f"重新选股 - 日期: {date}")
                selected_stocks = self._select_stocks(date, top_n)
                last_selection_date = date
                
                # 执行交易
                self._execute_trades(date, selected_stocks)
            
            # 记录当日持仓
            self.position_history.append({
                'date': date,
                'positions': self.positions.copy()
            })
            
            # 计算组合价值
            portfolio_value = self._calculate_portfolio_value(date)
            
            # 计算收益率
            if i > 0:
                prev_value = self.daily_capital[-1]['portfolio_value']
                daily_return = (portfolio_value / prev_value) - 1
                self.daily_returns.append({
                    'date': date,
                    'return': daily_return
                })
            
            # 记录资金和组合价值
            self.daily_capital.append({
                'date': date,
                'capital': self.capital,
                'portfolio_value': portfolio_value
            })
            
            print(f"当日组合价值: {portfolio_value:.2f}, 现金: {self.capital:.2f}")
        
        # 回测结束，计算绩效指标
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        """计算绩效指标"""
        if not self.daily_capital or len(self.daily_capital) < 2:
            print("没有足够的数据计算绩效指标")
            return
        
        # 计算总收益率
        initial_value = self.daily_capital[0]['portfolio_value']
        final_value = self.daily_capital[-1]['portfolio_value']
        total_return = (final_value / initial_value) - 1
        
        # 计算年化收益率
        days = len(self.daily_capital) - 1
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 计算最大回撤
        max_drawdown = 0
        peak_value = initial_value
        
        for daily in self.daily_capital:
            if daily['portfolio_value'] > peak_value:
                peak_value = daily['portfolio_value']
            else:
                drawdown = (peak_value - daily['portfolio_value']) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
        
        # 计算夏普比率
        if not self.daily_returns:
            sharpe_ratio = 0
        else:
            returns = [r['return'] for r in self.daily_returns]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            risk_free_rate = 0.03 / 252  # 假设年化无风险利率为3%
            sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # 计算胜率
        if not self.trade_history:
            win_rate = 0
        else:
            profitable_trades = 0
            for i in range(len(self.trade_history)):
                trade = self.trade_history[i]
                if trade['action'] == 'sell':
                    # 查找对应的买入交易
                    buy_price = None
                    for j in range(i-1, -1, -1):
                        prev_trade = self.trade_history[j]
                        if prev_trade['stock'] == trade['stock'] and prev_trade['action'] == 'buy':
                            buy_price = prev_trade['price']
                            break
                    
                    if buy_price is not None and trade['price'] > buy_price:
                        profitable_trades += 1
            
            total_sell_trades = sum(1 for trade in self.trade_history if trade['action'] == 'sell')
            win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0
        
        # 打印绩效指标
        print("\n===== 回测绩效指标 =====")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"胜率: {win_rate:.2%}")
        print(f"交易次数: {len(self.trade_history)}")
        
        # 绘制收益曲线
        self._plot_performance()
    
    def _plot_performance(self):
        """绘制回测绩效图表"""
        if not self.daily_capital:
            print("没有数据可绘制")
            return
        
        plt.figure(figsize=(14, 10))
        
        # 绘制组合价值曲线
        plt.subplot(2, 1, 1)
        dates = [datetime.datetime.strptime(d['date'], '%Y%m%d') for d in self.daily_capital]
        values = [d['portfolio_value'] for d in self.daily_capital]
        
        plt.plot(dates, values, 'b-', label='策略收益')
        
        # 绘制基准收益（上证指数）
        try:
            index_data = xtdata.get_market_data(
                stock_list=['SH000001'],
                period='1d',
                start_time=self.start_date,
                end_time=self.end_date,
                field_list=['open', 'high', 'low', 'close', 'volume']
            )
            
            if index_data and 'SH000001' in index_data:
                index_df = index_data['SH000001']
                index_values = index_df['close'] / index_df['close'].iloc[0] * self.initial_capital
                plt.plot(index_df.index, index_values, 'r-', label='上证指数')
        except Exception as e:
            print(f"绘制基准收益时出错: {e}")
        
        plt.title('策略收益曲线')
        plt.xlabel('日期')
        plt.ylabel('组合价值')
        plt.legend()
        plt.grid(True)
        
        # 绘制每日收益率
        plt.subplot(2, 1, 2)
        if self.daily_returns:
            return_dates = [datetime.datetime.strptime(r['date'], '%Y%m%d') for r in self.daily_returns]
            returns = [r['return'] * 100 for r in self.daily_returns]
            
            plt.bar(return_dates, returns, color=['g' if r > 0 else 'r' for r in returns])
            plt.title('每日收益率 (%)')
            plt.xlabel('日期')
            plt.ylabel('收益率 (%)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.show()
    
    def run_live_trading(self, top_n=10):
        """
        运行实时交易
        
        参数:
            top_n: 选择的股票数量
        """
        print("开始实时交易...")
        
        # 连接交易账户
        trader = XtQuantTrader(
            path="xtquant_config.json",  # 配置文件路径
            session_id=123456,  # 会话ID
            callback=None  # 回调函数
        )
        
        # 等待连接成功
        while not trader.is_connected():
            print("等待连接交易服务器...")
            time.sleep(1)
        
        print("成功连接到交易服务器")
        
        # 获取账户
        accounts = trader.get_accounts()
        if not accounts:
            print("没有可用账户")
            return
        
        # 使用第一个股票账户
        account = None
        for acc in accounts:
            if isinstance(acc, StockAccount):
                account = acc
                break
        
        if account is None:
            print("没有可用的股票账户")
            return
        
        print(f"使用账户: {account.account_id}")
        
        # 获取当前日期
        today = datetime.datetime.now().strftime('%Y%m%d')
        
        # 训练模型
        print("训练机器学习模型...")
        train_data = {}
        
        # 随机选择一部分股票进行训练
        train_stocks = np.random.choice(self.stock_list, min(100, len(self.stock_list)), replace=False)
        
        for stock_code in train_stocks:
            try:
                # 获取历史数据
                daily_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=self._get_previous_date(today, 180),
                    end_time=today
                )
                
                if not daily_data or stock_code not in daily_data or len(daily_data[stock_code]) < 60:
                    continue
                
                # 计算技术指标
                df = self._calculate_technical_indicators(daily_data[stock_code])
                
                if df is not None and len(df) >= 60:
                    train_data[stock_code] = df
            except Exception as e:
                print(f"获取 {stock_code} 训练数据时出错: {e}")
        
        # 训练模型
        self.ml_models[today] = self._train_ml_model(today)
        
        # 选股
        selected_stocks = self._select_stocks(today, top_n)
        
        # 获取当前持仓
        positions = trader.get_positions(account)
        current_positions = {}
        
        for position in positions:
            if position.stock_code in self.stock_list:
                current_positions[position.stock_code] = position.volume
        
        # 计算每只股票的目标仓位
        account_info = trader.get_account_info(account)
        total_assets = account_info.total_assets if account_info else 0
        
        if total_assets <= 0:
            print("无法获取账户资产信息")
            return
        
        target_position_value = total_assets / len(selected_stocks) if selected_stocks else 0
        
        # 卖出不在选股列表中的股票
        for stock_code, shares in current_positions.items():
            if stock_code not in selected_stocks and shares > 0:
                print(f"卖出: {stock_code}, 数量: {shares}")
                
                # 获取当前价格
                price_data = xtdata.get_market_data(
                    stock_list=[stock_code],
                    period='1d',
                    start_time=today,
                    end_time=today,
                    field_list=['open', 'high', 'low', 'close', 'volume']
                )
                
                if not price_data or stock_code not in price_data:
                    print(f"无法获取 {stock_code} 的价格数据，跳过卖出")
                    continue
                
                # 创建卖出委托
                order_id = trader.sell(
                    account=account,
                    stock_code=stock_code,
                    price=0,  # 市价委托
                    volume=shares
                )
                
                if order_id:
                    print(f"卖出委托已提交: {order_id}")
                else:
                    print(f"卖出委托提交失败")
        
        # 买入选中的股票
        for stock_code in selected_stocks:
            # 获取当前价格
            price_data = xtdata.get_market_data(
                stock_list=[stock_code],
                period='1d',
                start_time=today,
                end_time=today,
                field_list=['open', 'high', 'low', 'close', 'volume']
            )
            
            if not price_data or stock_code not in price_data:
                print(f"无法获取 {stock_code} 的价格数据，跳过买入")
                continue
            
            price = price_data[stock_code]['close'].iloc[0]
            
            # 计算目标持仓数量
            target_shares = int(target_position_value / price / 100) * 100  # 取整百
            
            # 获取当前持仓数量
            current_shares = current_positions.get(stock_code, 0)
            
            # 计算需要买入的数量
            shares_to_buy = target_shares - current_shares
            
            if shares_to_buy > 0:
                print(f"买入: {stock_code}, 数量: {shares_to_buy}")
                
                # 创建买入委托
                order_id = trader.buy(
                    account=account,
                    stock_code=stock_code,
                    price=0,  # 市价委托
                    volume=shares_to_buy
                )
                
                if order_id:
                    print(f"买入委托已提交: {order_id}")
                else:
                    print(f"买入委托提交失败")
        
        print("交易执行完成")
    
    def _update_industry_scores(self, date):
        """更新行业评分"""
        print(f"更新行业评分 - 日期: {date}")
        
        # 获取前20个交易日的日期
        start_date = self._get_previous_date(date, 20)
        
        # 获取所有行业的股票列表
        industry_stocks = {}
        for stock_code in self.stock_list:
            industry = self.industry_info.get(stock_code, '其他')
            if industry not in industry_stocks:
                industry_stocks[industry] = []
            industry_stocks[industry].append(stock_code)
        
        # 计算每个行业的平均涨跌幅
        industry_returns = {}
        
        for industry, stocks in industry_stocks.items():
            # 从每个行业中随机选择最多30只股票计算
            sample_size = min(30, len(stocks))
            sample_stocks = np.random.choice(stocks, sample_size, replace=False)
            
            industry_total_return = 0
            valid_stock_count = 0
            
            # 获取这些股票的历史数据
            field_list = ['close']
            industry_data = xtdata.get_market_data(
                field_list=field_list,
                stock_list=sample_stocks,
                period='1d',
                start_time=start_date,
                end_time=date
            )
            
            for stock_code in sample_stocks:
                try:
                    if 'close' not in industry_data or stock_code not in industry_data['close'] or len(industry_data['close'][stock_code]) < 2:
                        continue
                    
                    # 计算20日涨跌幅
                    start_price = industry_data['close'][stock_code][0]
                    end_price = industry_data['close'][stock_code][-1]
                    
                    if start_price <= 0:
                        continue
                        
                    return_rate = (end_price / start_price - 1) * 100
                    industry_total_return += return_rate
                    valid_stock_count += 1
                except Exception as e:
                    print(f"计算股票 {stock_code} 涨跌幅时出错: {e}")
            
            # 计算行业平均涨跌幅
            if valid_stock_count > 0:
                industry_returns[industry] = industry_total_return / valid_stock_count
            else:
                industry_returns[industry] = 0
        
        # 将行业涨跌幅转换为评分（0-1之间）
        min_return = min(industry_returns.values()) if industry_returns else 0
        max_return = max(industry_returns.values()) if industry_returns else 0
        
        range_return = max_return - min_return
        
        # 更新行业评分
        self.industry_scores = {}
        
        for industry, return_rate in industry_returns.items():
            if range_return > 0:
                # 将涨跌幅映射到0-1之间
                score = (return_rate - min_return) / range_return
            else:
                score = 0.5
            
            self.industry_scores[industry] = score
        
        # 打印行业评分
        print("行业评分:")
        sorted_scores = sorted(self.industry_scores.items(), key=lambda x: x[1], reverse=True)
        for industry, score in sorted_scores[:5]:
            print(f"{industry}: {score:.4f}")
        
        return self.industry_scores
    
    def _get_stock_data(self, stock_code, start_date, end_date):
        """获取单只股票的历史数据并计算技术指标"""
        try:
            # 获取历史数据
            field_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
            daily_data = xtdata.get_market_data(
                field_list=field_list,
                stock_list=[stock_code],
                period='1d',
                start_time=start_date,
                end_time=end_date
            )
            
            if not daily_data or 'close' not in daily_data or stock_code not in daily_data['close'] or len(daily_data['close'][stock_code]) < 30:
                return None
            
            # 将数据转换为DataFrame格式
            df = pd.DataFrame({
                'open': daily_data['open'][stock_code],
                'high': daily_data['high'][stock_code],
                'low': daily_data['low'][stock_code],
                'close': daily_data['close'][stock_code],
                'volume': daily_data['volume'][stock_code],
                'amount': daily_data['amount'][stock_code]
            })
            df.index = daily_data['time'][stock_code]
            
            # 计算技术指标
            df = self._calculate_technical_indicators(df)
            
            return df
        except Exception as e:
            print(f"获取股票 {stock_code} 数据时出错: {e}")
            return None

# 示例用法
if __name__ == "__main__":
    # 创建策略实例
    strategy = AShareQuantStrategy(start_date='20230601', end_date='20240301')
    
    # 执行回测
    strategy.backtest(retrain_interval=20, stock_selection_interval=5, top_n=10)
    
    # 如果需要实时交易，取消下面的注释
    # strategy.run_live_trading(top_n=10) 