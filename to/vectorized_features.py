#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量化特征提取模块
用于加速量化模型的特征提取过程，替代传统循环方式
"""

import logging
import numpy as np
import pandas as pd
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings

# 配置日志
import os
os.makedirs('to/log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/vectorized_features.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vectorized_features")

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

class VectorizedFeatureExtractor:
    """向量化特征提取器类，提供高效的特征计算方法"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.feature_functions = {
            # 技术指标
            'price_features': self._calc_price_features,
            'volume_features': self._calc_volume_features,
            'momentum_features': self._calc_momentum_features,
            'volatility_features': self._calc_volatility_features,
            'pattern_features': self._calc_pattern_features,
            
            # 添加其他特征组
            'trend_features': self._calc_trend_features,
            'liquidity_features': self._calc_liquidity_features
        }
        
        # 常用窗口大小
        self.windows = [5, 10, 20, 30, 60]
    
    def extract_features_batch(self, market_data, stocks, date_idx, lookback_days, forward_days=5):
        """
        批量提取特征和标签
        
        参数:
            market_data: 市场数据字典，字段名为键
            stocks: 股票列表
            date_idx: 当前日期在日期列表中的索引
            lookback_days: 回溯天数
            forward_days: 前瞻天数(用于标签计算)
            
        返回:
            features_matrix: 特征矩阵 [n_samples, n_features]
            labels_vector: 标签向量 [n_samples]
            valid_stocks: 有效股票列表
        """
        start_time = time.time()
        
        # 检查数据有效性
        if not market_data or not stocks:
            logger.warning("无效的市场数据或股票列表")
            return None, None, []
        
        # 获取日期列表
        dates = sorted(market_data['close'].index.unique())
        if date_idx >= len(dates):
            logger.error(f"日期索引越界: {date_idx} >= {len(dates)}")
            return None, None, []
        
        current_date = dates[date_idx]
        logger.info(f"计算特征: 日期={current_date}, 股票数量={len(stocks)}")
        
        # 找出有效的股票（有足够的历史数据和未来数据）
        valid_stocks = self._filter_valid_stocks(
            market_data, stocks, dates, date_idx, lookback_days, forward_days
        )
        
        if not valid_stocks:
            logger.warning("没有满足条件的有效股票")
            return None, None, []
        
        logger.info(f"有效股票数量: {len(valid_stocks)} / {len(stocks)}")
        
        # 准备存储所有特征和标签的容器
        all_features = []
        all_labels = []
        final_valid_stocks = []
        
        # 获取所需的日期范围数据
        start_idx = max(0, date_idx - lookback_days)
        end_idx = min(len(dates) - 1, date_idx + forward_days)
        
        date_range = dates[start_idx:end_idx + 1]
        
        # 为每个股票提取特征和标签
        for stock in valid_stocks:
            try:
                # 提取单只股票的数据
                stock_data = self._extract_single_stock_data(market_data, stock, date_range)
                
                if stock_data.empty or len(stock_data) < lookback_days:
                    continue
                
                # 计算特征
                features = self._compute_all_features(stock_data, lookback_days)
                
                if features is None:
                    continue
                
                # 计算标签 (未来收益率)
                label = self._compute_label(
                    market_data, stock, dates, date_idx, forward_days
                )
                
                if label is None:
                    continue
                
                # 添加到结果集
                all_features.append(features)
                all_labels.append(label)
                final_valid_stocks.append(stock)
                
            except Exception as e:
                logger.debug(f"处理股票 {stock} 时出错: {e}")
                continue
        
        if not all_features:
            logger.warning("没有计算出有效特征")
            return None, None, []
        
        # 转换为numpy数组
        features_matrix = np.array(all_features)
        labels_vector = np.array(all_labels)
        
        # 检查数据有效性
        if np.isnan(features_matrix).any() or np.isinf(features_matrix).any():
            logger.warning("特征矩阵包含NaN或Inf值，进行修复")
            features_matrix = np.nan_to_num(features_matrix, nan=0, posinf=0, neginf=0)
        
        elapsed = time.time() - start_time
        logger.info(f"特征计算完成: {features_matrix.shape}, 耗时: {elapsed:.2f}秒")
        
        return features_matrix, labels_vector, final_valid_stocks
    
    def _filter_valid_stocks(self, market_data, stocks, dates, date_idx, lookback_days, forward_days):
        """筛选有足够历史和未来数据的有效股票"""
        valid_stocks = []
        
        # 检查每只股票是否有足够的数据
        for stock in stocks:
            # 历史数据检查
            has_enough_history = True
            for i in range(max(0, date_idx - lookback_days), date_idx + 1):
                if i >= len(dates):
                    has_enough_history = False
                    break
                    
                hist_date = dates[i]
                if stock not in market_data['close'].loc[hist_date]:
                    has_enough_history = False
                    break
            
            # 未来数据检查（用于标签）
            has_future_data = True
            future_idx = min(len(dates) - 1, date_idx + forward_days)
            future_date = dates[future_idx]
            
            if stock not in market_data['close'].loc[future_date]:
                has_future_data = False
            
            # 只有同时满足两个条件的股票才是有效的
            if has_enough_history and has_future_data:
                valid_stocks.append(stock)
        
        return valid_stocks
    
    def _extract_single_stock_data(self, market_data, stock, date_range):
        """提取单只股票在指定日期范围的数据"""
        # 创建一个空的DataFrame用于存储结果
        stock_data = pd.DataFrame(index=date_range)
        
        # 对每个字段提取数据
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if field in market_data:
                # 提取指定股票和日期范围的数据
                field_data = market_data[field].loc[date_range].get(stock, None)
                
                if field_data is not None:
                    stock_data[field] = field_data
        
        # 删除任何缺失值的行
        stock_data = stock_data.dropna()
        
        return stock_data
    
    def _compute_all_features(self, stock_data, lookback_days):
        """计算所有特征"""
        if len(stock_data) < lookback_days:
            return None
        
        # 只使用最近的lookback_days天数据
        df = stock_data.iloc[-lookback_days:]
        
        features = []
        
        # 计算各组特征
        for feature_name, feature_func in self.feature_functions.items():
            try:
                feature_values = feature_func(df)
                if feature_values is not None:
                    features.extend(feature_values)
            except Exception as e:
                logger.debug(f"计算{feature_name}特征时出错: {e}")
                # 在出错时用0填充（保持特征数量一致）
                if feature_name == 'price_features':
                    features.extend([0] * 15)  # 根据实际特征数量调整
                elif feature_name == 'volume_features':
                    features.extend([0] * 10)
                elif feature_name == 'momentum_features':
                    features.extend([0] * 12)
                elif feature_name == 'volatility_features':
                    features.extend([0] * 8)
                elif feature_name == 'pattern_features':
                    features.extend([0] * 6)
                elif feature_name == 'trend_features':
                    features.extend([0] * 10)
                elif feature_name == 'liquidity_features':
                    features.extend([0] * 5)
        
        return features
    
    def _compute_label(self, market_data, stock, dates, date_idx, forward_days):
        """计算标签（未来收益率）"""
        try:
            current_date = dates[date_idx]
            future_idx = min(len(dates) - 1, date_idx + forward_days)
            future_date = dates[future_idx]
            
            # 获取当前价格和未来价格
            current_price = market_data['close'].loc[current_date, stock]
            future_price = market_data['close'].loc[future_date, stock]
            
            # 计算收益率
            return_rate = (future_price / current_price) - 1
            
            # 根据收益率划分类别（可根据需要调整阈值）
            if return_rate > 0.05:  # 5%以上为强势上涨
                return 2
            elif return_rate > 0:    # 0-5%为小幅上涨
                return 1
            elif return_rate > -0.05:  # 0至-5%为小幅下跌
                return 0
            else:                     # -5%以下为强势下跌
                return -1
                
        except Exception as e:
            logger.debug(f"计算标签时出错: {e}")
            return None
    
    # === 各类特征计算函数 ===
    
    def _calc_price_features(self, df):
        """计算价格相关特征，使用向量化操作"""
        price_features = []
        
        # 使用向量化操作计算价格变化率
        for window in self.windows:
            if len(df) >= window:
                # 计算n日收益率
                price_features.append(df['close'].pct_change(window).iloc[-1])
                
                # 相对于N日最高价和最低价的位置
                n_day_high = df['high'].rolling(window).max().iloc[-1]
                n_day_low = df['low'].rolling(window).min().iloc[-1]
                price_position = (df['close'].iloc[-1] - n_day_low) / (n_day_high - n_day_low) if n_day_high > n_day_low else 0.5
                price_features.append(price_position)
        
        # 计算OHLC价差特征
        price_features.append((df['high'].iloc[-1] / df['low'].iloc[-1]) - 1)  # 高低价差率
        price_features.append((df['close'].iloc[-1] / df['open'].iloc[-1]) - 1)  # 收盘开盘价差率
        price_features.append(df['close'].iloc[-1] / df['close'].mean() - 1)  # 收盘价相对均价偏离
        
        return price_features
    
    def _calc_volume_features(self, df):
        """计算成交量相关特征，使用向量化操作"""
        volume_features = []
        
        # 异常值处理 - 防止除零错误
        df_safe = df.copy()
        df_safe['volume'] = df_safe['volume'].replace(0, np.nan)
        df_safe['volume'] = df_safe['volume'].fillna(df_safe['volume'].median())
        
        # 成交量变化率
        for window in self.windows[:3]:  # 使用较小的窗口
            if len(df) >= window:
                # 计算n日成交量变化率
                volume_features.append(df_safe['volume'].pct_change(window).iloc[-1])
                
                # 计算n日成交量相对于均值的比率
                volume_ma = df_safe['volume'].rolling(window).mean().iloc[-1]
                volume_features.append(df_safe['volume'].iloc[-1] / volume_ma if volume_ma > 0 else 1)
        
        # 成交量波动性特征
        if len(df) >= 10:
            volume_std = df_safe['volume'][-10:].std()
            volume_mean = df_safe['volume'][-10:].mean()
            volume_features.append(volume_std / volume_mean if volume_mean > 0 else 0)
        else:
            volume_features.append(0)
            
        # 价量相关性特征
        if len(df) >= 10:
            price_changes = df['close'].pct_change().iloc[-10:]
            volume_changes = df_safe['volume'].pct_change().iloc[-10:]
            price_volume_corr = price_changes.corr(volume_changes)
            volume_features.append(price_volume_corr if not np.isnan(price_volume_corr) else 0)
        else:
            volume_features.append(0)
        
        return volume_features
    
    def _calc_momentum_features(self, df):
        """计算动量相关特征，使用向量化操作"""
        momentum_features = []
        
        # RSI指标 (相对强弱指数)
        for window in [6, 14]:
            if len(df) >= window + 1:
                # 计算价格变化
                delta = df['close'].diff()
                
                # 分离上涨和下跌
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # 计算平均上涨和下跌
                avg_gain = gain.rolling(window).mean().iloc[-1]
                avg_loss = loss.rolling(window).mean().iloc[-1]
                
                # 计算相对强度
                rs = avg_gain / avg_loss if avg_loss > 0 else 1000
                
                # 计算RSI
                rsi = 100 - (100 / (1 + rs))
                momentum_features.append(rsi)
            else:
                momentum_features.append(50)  # 默认中性值
        
        # MACD指标 (异同移动平均线)
        if len(df) >= 26:
            # 计算EMA
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            
            # MACD线
            macd_line = ema12 - ema26
            
            # 信号线
            signal_line = macd_line.ewm(span=9).mean()
            
            # MACD直方图
            macd_histogram = macd_line - signal_line
            
            # 提取最新值作为特征
            momentum_features.append(macd_line.iloc[-1])
            momentum_features.append(signal_line.iloc[-1])
            momentum_features.append(macd_histogram.iloc[-1])
        else:
            momentum_features.extend([0, 0, 0])
        
        # 增加KDJ指标
        if len(df) >= 9:
            low_min = df['low'].rolling(9).min()
            high_max = df['high'].rolling(9).max()
            
            # 计算随机值K
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            # 计算K的移动平均D
            d = k.rolling(3).mean()
            # 计算D的移动平均J
            j = 3 * k - 2 * d
            
            momentum_features.append(k.iloc[-1])
            momentum_features.append(d.iloc[-1])
            momentum_features.append(j.iloc[-1])
        else:
            momentum_features.extend([50, 50, 50])
        
        return momentum_features
    
    def _calc_volatility_features(self, df):
        """计算波动性相关特征，使用向量化操作"""
        volatility_features = []
        
        # 计算不同窗口的波动率
        for window in self.windows[:3]:  # 使用较小的窗口
            if len(df) >= window:
                # 日收益率
                returns = df['close'].pct_change().iloc[-window:]
                
                # 标准差（波动率）
                volatility = returns.std()
                volatility_features.append(volatility)
                
                # 偏度 - 分布的不对称性
                skew = returns.skew()
                volatility_features.append(0 if np.isnan(skew) else skew)
            else:
                volatility_features.extend([0, 0])
        
        # ATR (Average True Range) - 平均真实范围
        if len(df) >= 14:
            # 计算三种价差
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            # 取最大值作为真实范围
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # 计算ATR
            atr = tr.rolling(14).mean().iloc[-1]
            volatility_features.append(atr)
            
            # 相对ATR (ATR/收盘价)
            relative_atr = atr / df['close'].iloc[-1]
            volatility_features.append(relative_atr)
        else:
            volatility_features.extend([0, 0])
        
        return volatility_features
    
    def _calc_pattern_features(self, df):
        """计算形态特征，使用向量化操作"""
        pattern_features = []
        
        if len(df) < 3:
            return [0] * 6
        
        # 最近N天的价格
        close_prices = df['close'].values
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # 计算烛台特征
        
        # 1. 最近一天的实体比例 (实体/整体范围)
        last_body = abs(close_prices[-1] - open_prices[-1])
        last_range = high_prices[-1] - low_prices[-1]
        body_ratio = last_body / last_range if last_range > 0 else 0
        pattern_features.append(body_ratio)
        
        # 2. 最近一天是否是阳线
        is_bullish = close_prices[-1] > open_prices[-1]
        pattern_features.append(1 if is_bullish else 0)
        
        # 3. 上影线比例
        upper_shadow = high_prices[-1] - max(close_prices[-1], open_prices[-1])
        upper_shadow_ratio = upper_shadow / last_range if last_range > 0 else 0
        pattern_features.append(upper_shadow_ratio)
        
        # 4. 下影线比例
        lower_shadow = min(close_prices[-1], open_prices[-1]) - low_prices[-1]
        lower_shadow_ratio = lower_shadow / last_range if last_range > 0 else 0
        pattern_features.append(lower_shadow_ratio)
        
        # 5. 三连阳/阴
        if len(df) >= 3:
            last_3_bullish = sum(1 for i in range(1, 4) if i <= len(df) and df['close'].iloc[-i] > df['open'].iloc[-i])
            pattern_features.append(last_3_bullish / 3)  # 最近三天中阳线的比例
        else:
            pattern_features.append(0.5)
        
        # 6. 十字星形态
        is_doji = body_ratio < 0.1  # 实体很小
        pattern_features.append(1 if is_doji else 0)
        
        return pattern_features
    
    def _calc_trend_features(self, df):
        """计算趋势相关特征，使用向量化操作"""
        trend_features = []
        
        if len(df) < 20:
            return [0] * 10
        
        # 计算不同周期的移动平均线
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma10 = df['close'].rolling(10).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma30 = df['close'].rolling(30).mean().iloc[-1] if len(df) >= 30 else df['close'].mean()
        
        # 当前价格
        current_price = df['close'].iloc[-1]
        
        # 价格相对于不同MA的位置
        trend_features.append(current_price / ma5 - 1)
        trend_features.append(current_price / ma10 - 1)
        trend_features.append(current_price / ma20 - 1)
        trend_features.append(current_price / ma30 - 1)
        
        # MA交叉特征
        trend_features.append(1 if ma5 > ma10 else 0)  # 金叉/死叉
        trend_features.append(1 if ma10 > ma20 else 0)
        
        # 方向性指标
        # 自动计算方向性指标ADX
        if len(df) >= 14:
            # 真实范围TR
            tr = pd.DataFrame()
            tr['HL'] = df['high'] - df['low']
            tr['HC'] = abs(df['high'] - df['close'].shift())
            tr['LC'] = abs(df['low'] - df['close'].shift())
            tr['TR'] = tr[['HL', 'HC', 'LC']].max(axis=1)
            
            # 方向性移动DM
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
            minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
            
            # 方向性指标DI
            smoothing_period = 14
            tr14 = tr['TR'].rolling(smoothing_period).sum()
            plus_di14 = 100 * plus_dm.rolling(smoothing_period).sum() / tr14
            minus_di14 = 100 * minus_dm.rolling(smoothing_period).sum() / tr14
            
            # 方向性指数ADX
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            adx = dx.rolling(smoothing_period).mean()
            
            trend_features.append(plus_di14.iloc[-1])
            trend_features.append(minus_di14.iloc[-1])
            trend_features.append(adx.iloc[-1])
        else:
            trend_features.extend([0, 0, 0])
        
        # 添加趋势强度特征
        price_std = df['close'].rolling(20).std().iloc[-1]
        price_mean = df['close'].rolling(20).mean().iloc[-1]
        trend_features.append(price_std / price_mean if price_mean > 0 else 0)  # 变异系数
        
        return trend_features
    
    def _calc_liquidity_features(self, df):
        """计算流动性相关特征，使用向量化操作"""
        liquidity_features = []
        
        if len(df) < 5:
            return [0] * 5
        
        # 检查df是否包含amount字段
        has_amount = 'amount' in df.columns
        
        # 1. 最近5天的平均成交额
        if has_amount:
            avg_amount = df['amount'][-5:].mean()
            liquidity_features.append(avg_amount)
        else:
            # 如果没有amount字段，使用close * volume作为近似
            avg_amount = (df['close'] * df['volume'])[-5:].mean()
            liquidity_features.append(avg_amount)
        
        # 2. 换手率相关特征（这需要额外数据如流通股本，此处用成交量变化近似）
        volume_change = df['volume'].pct_change(5).iloc[-1]
        liquidity_features.append(volume_change)
        
        # 3. 最近5天的成交量波动
        volume_std = df['volume'][-5:].std()
        volume_mean = df['volume'][-5:].mean()
        volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
        liquidity_features.append(volume_cv)
        
        # 4. 价格波动对成交量的敏感度
        if len(df) >= 10:
            # 计算价格变化率和成交量变化率
            price_changes = df['close'].pct_change()[-10:]
            volume_changes = df['volume'].pct_change()[-10:]
            
            # 价格变化率与成交量变化率的比值
            valid_indices = ~(price_changes.isna() | volume_changes.isna() | (price_changes == 0))
            
            if valid_indices.any():
                price_volume_sensitivity = (volume_changes[valid_indices] / price_changes[valid_indices]).mean()
                liquidity_features.append(price_volume_sensitivity if not np.isnan(price_volume_sensitivity) else 0)
            else:
                liquidity_features.append(0)
        else:
            liquidity_features.append(0)
        
        # 5. 流动性溢价 (假设高成交量意味着高流动性)
        recent_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        liquidity_premium = recent_volume / avg_volume if avg_volume > 0 else 1
        liquidity_features.append(liquidity_premium)
        
        return liquidity_features

# 创建一个函数用于集成到原始的特征提取流程中
def create_vectorized_extractor():
    """创建并返回向量化特征提取器实例"""
    return VectorizedFeatureExtractor()

# 测试代码
if __name__ == "__main__":
    # 创建一个示例市场数据结构进行测试
    print("向量化特征提取模块加载成功") 