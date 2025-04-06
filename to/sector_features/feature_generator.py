#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成器模块

计算板块的各种特征，包括价格指标、成交量指标、技术指标等。
"""

import os
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
# 导入技术指标计算类
from to.sector_features.technical_indicators import TechnicalIndicators

try:
    from to.path_utils import get_data_path, ensure_dir
except ImportError:
    # 如果导入失败，提供简单的替代实现
    def get_data_path(relative_path=None):
        from pathlib import Path
        data_root = Path('to') / 'local_data'
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

logger = logging.getLogger(__name__)

class SectorFeatureGenerator:
    """板块特征生成器
    
    生成板块的各种基础特征，包括价格均值、成交量、成交额、活跃股票数等。
    """
    
    def __init__(self, data_provider, sector_mapping, output_dir=None, 
                weighting_method='equal'):
        """初始化特征生成器
        
        Args:
            data_provider: 数据提供者，需要实现get_daily_data和get_trading_days方法
            sector_mapping: 板块映射字典 {板块名称: [股票列表]}
            output_dir: 特征输出目录，默认使用get_data_path('sector_features')
            weighting_method: 权重计算方法，可选值有equal(等权重)、market_cap(市值加权)等
        """
        self.data_provider = data_provider
        self.sector_mapping = sector_mapping
        # 使用path_utils来获取正确的路径
        self.output_dir = output_dir if output_dir is not None else get_data_path('sector_features')
        self.weighting_method = weighting_method
        
        # 创建输出目录
        ensure_dir(self.output_dir)
        
        logger.info(f"初始化板块特征生成器，权重方法: {weighting_method}, 输出目录: {self.output_dir}")
        
    def _get_sector_stocks(self, sector):
        """获取板块成分股
        
        Args:
            sector: 板块名称
            
        Returns:
            股票列表
        """
        # 从sector_mapping中获取
        stocks = self.sector_mapping.get(sector, [])
        
        # 如果为空，尝试从stock_sectors_mapping中获取
        if not stocks:
            try:
                # 兼容性代码，允许从外部获取股票列表
                from to.stock_sectors_mapping import get_sector_stocks
                stocks = get_sector_stocks(sector_name=sector)
            except Exception as e:
                logger.error(f"获取板块 {sector} 的成分股失败: {str(e)}")
                stocks = []
                
        logger.debug(f"板块 {sector} 包含 {len(stocks)} 只股票")
        return stocks
        
    def _get_weights(self, data, method='equal'):
        """计算权重
        
        Args:
            data: DataFrame，包含要计算权重的数据
            method: 权重计算方法，可选值有equal、market_cap等
            
        Returns:
            权重Series
        """
        if method == 'equal':
            # 等权重
            return pd.Series(1.0 / len(data), index=data.index)
            
        elif method == 'market_cap':
            # 市值加权
            if 'market_cap' not in data.columns:
                logger.warning("数据中没有市值列，无法进行市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            # 计算市值权重
            market_cap = data['market_cap']
            total_market_cap = market_cap.sum()
            
            if total_market_cap == 0:
                logger.warning("总市值为0，无法进行市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            return market_cap / total_market_cap
            
        elif method == 'float_market_cap':
            # 流通市值加权
            if 'float_market_cap' not in data.columns:
                logger.warning("数据中没有流通市值列，无法进行流通市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            # 计算流通市值权重
            float_market_cap = data['float_market_cap']
            total_float_market_cap = float_market_cap.sum()
            
            if total_float_market_cap == 0:
                logger.warning("总流通市值为0，无法进行流通市值加权，将使用等权重")
                return pd.Series(1.0 / len(data), index=data.index)
                
            return float_market_cap / total_float_market_cap
            
        else:
            logger.warning(f"未知的权重计算方法 {method}，将使用等权重")
            return pd.Series(1.0 / len(data), index=data.index)
            
    def _calculate_weighted_average(self, data, column, weights):
        """计算加权平均值
        
        Args:
            data: DataFrame，包含要计算加权平均的数据
            column: 列名
            weights: 权重Series
            
        Returns:
            加权平均值
        """
        if column not in data.columns:
            logger.warning(f"数据中没有列 {column}，无法计算加权平均值")
            return 0
            
        # 计算加权平均
        weighted_values = data[column] * weights
        return weighted_values.sum()
        
    def _calculate_basic_features(self, sector_data, date, stocks):
        """计算基础特征
        
        Args:
            sector_data: DataFrame，包含板块成分股数据
            date: 日期字符串或Timestamp对象
            stocks: 股票列表
            
        Returns:
            特征字典
        """
        features = {
            'date': date,
            'open_avg': 0,
            'high_avg': 0,
            'low_avg': 0,
            'close_avg': 0,
            'volume_avg': 0,
            'amount_avg': 0,
            'change_avg': 0,
            'active_count': 0,  # 有交易的股票数量
            'total_amount': 0,  # 总成交额
            'sector_name': '',  # 板块名称
            'stock_count': len(stocks)  # 股票数量
        }
        
        if sector_data.empty:
            return features
            
        # 计算有效数据（有交易的股票）
        active_data = sector_data[sector_data['volume'] > 0].copy()
        features['active_count'] = len(active_data)
        
        if features['active_count'] == 0:
            return features
            
        # 计算权重
        weights = self._get_weights(active_data, self.weighting_method)
        
        # 如果数据中没有change列，则计算涨跌幅
        if 'change' not in active_data.columns:
            logger.info(f"数据中没有change列，将根据当日close和前一交易日close计算")
            try:
                # 将日期转换为标准格式字符串
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime("%Y%m%d")
                elif isinstance(date, datetime):
                    date_str = date.strftime("%Y%m%d")
                else:
                    date_str = str(date)
                
                # 获取前一个交易日
                date_obj = pd.to_datetime(date_str) if not isinstance(date, pd.Timestamp) else date
                # 向前推7天，足够找到前一个交易日
                prev_start_date = (date_obj - pd.Timedelta(days=7)).strftime("%Y%m%d")
                
                # 获取交易日历
                trading_days = self.data_provider.get_trading_days(prev_start_date, date_str if isinstance(date_str, str) else date_obj.strftime("%Y%m%d"))
                
                # 处理trading_days可能是DataFrame的情况
                if isinstance(trading_days, pd.DataFrame):
                    # 确定DataFrame中存储日期的列
                    if 'date' in trading_days.columns:
                        date_col = 'date'
                    elif 'date_str' in trading_days.columns:
                        date_col = 'date_str'
                    else:
                        # 找不到日期列，使用第一列
                        date_col = trading_days.columns[0]
                    
                    # 排序并获取所有交易日期
                    trading_days_list = trading_days[date_col].sort_values().tolist()
                    
                    # 找到当前日期在列表中的位置
                    current_date_index = -1
                    
                    # 标准化当前日期格式，移除所有分隔符
                    date_str_normalized = date_str.replace('-', '').replace('/', '')
                    
                    # 记录一下日期格式，帮助调试
                    if len(trading_days_list) > 0:
                        logger.debug(f"交易日历中第一个日期格式: {trading_days_list[0]}, 类型: {type(trading_days_list[0])}")
                        logger.debug(f"当前查找的日期: {date_str}, 标准化后: {date_str_normalized}")
                    
                    # 遍历所有日期，尝试各种格式匹配
                    for i, td in enumerate(trading_days_list):
                        # 尝试多种标准化格式进行比较
                        td_str = str(td)
                        td_normalized = td_str.replace('-', '').replace('/', '')
                        
                        # 检查是否匹配(支持多种格式)
                        if (td_normalized == date_str_normalized or 
                            td_str == date_str or
                            td_str.split()[0].replace('-', '') == date_str):
                            current_date_index = i
                            logger.debug(f"找到日期匹配: 索引={i}, 日期值={td_str}")
                            break
                    
                    # 如果找不到精确匹配，尝试查找最接近的日期
                    if current_date_index == -1:
                        try:
                            # 将当前日期转换为datetime对象
                            target_dt = pd.to_datetime(date_str_normalized, format='%Y%m%d')
                            
                            # 将所有交易日转换为datetime对象
                            trading_dts = [pd.to_datetime(td) if not isinstance(td, pd.Timestamp) else td 
                                           for td in trading_days_list]
                            
                            # 找到最接近但不超过目标日期的交易日
                            closest_i = -1
                            for i, dt in enumerate(trading_dts):
                                if dt <= target_dt and (closest_i == -1 or dt > trading_dts[closest_i]):
                                    closest_i = i
                            
                            if closest_i != -1:
                                current_date_index = closest_i
                                logger.info(f"没有找到精确的日期匹配 {date_str}，使用最接近的交易日: {trading_days_list[current_date_index]}")
                        except Exception as e:
                            logger.error(f"尝试查找最接近的交易日时出错: {e}")
                    
                    if current_date_index > 0:
                        # 获取前一个交易日
                        prev_date_value = trading_days_list[current_date_index - 1]
                        # 确保前一交易日格式正确
                        if isinstance(prev_date_value, pd.Timestamp):
                            prev_date = prev_date_value.strftime('%Y%m%d')
                        else:
                            prev_date = str(prev_date_value).replace('-', '').replace('/', '')
                        
                        logger.debug(f"找到前一交易日: {prev_date}")
                    else:
                        logger.warning(f"无法在交易日历中找到日期 {date_str} 的位置或前一交易日，无法计算涨跌幅")
                        active_data['change'] = 0
                        return features
                else:
                    # 如果是列表，直接获取前一个交易日
                    if len(trading_days) <= 1:
                        logger.warning(f"无法获取前一个交易日，设置涨跌幅为0")
                        active_data['change'] = 0
                        return features
                    
                    # 找到当前日期的索引
                    current_date_index = -1
                    for i, td in enumerate(trading_days):
                        td_str = str(td).replace('-', '')
                        if td_str == date_str:
                            current_date_index = i
                            break
                    
                    if current_date_index > 0:
                        prev_date = str(trading_days[current_date_index - 1]).replace('-', '')
                    else:
                        logger.warning(f"无法在交易日历中找到日期 {date_str}，使用最后两个日期")
                        if len(trading_days) >= 2:
                            # 假设最后一个是当前日期，倒数第二个是前一日
                            prev_date = str(trading_days[-2]).replace('-', '')
                        else:
                            logger.warning(f"交易日历中没有足够的日期，设置涨跌幅为0")
                            active_data['change'] = 0
                
                # 获取前一日的收盘价数据
                prev_data = self.data_provider.get_daily_data(
                    stocks, 
                    prev_date, 
                    prev_date
                )
                
                if prev_data.empty:
                    logger.warning(f"前一交易日 {prev_date} 的数据为空，设置涨跌幅为0")
                    active_data['change'] = 0
                else:
                    # 创建股票代码到前收盘价的映射
                    prev_close_dict = {row['stock_code']: row['close'] 
                                      for _, row in prev_data.iterrows()}
                    
                    # 计算涨跌幅 (%)
                    def calc_change(row):
                        prev_close = prev_close_dict.get(row['stock_code'], 0)
                        if prev_close > 0:
                            return (row['close'] - prev_close) / prev_close * 100
                        return 0
                    
                    active_data['change'] = active_data.apply(calc_change, axis=1)
                    logger.info(f"成功为 {len(active_data)} 只股票计算涨跌幅")
            except Exception as e:
                logger.error(f"计算涨跌幅时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                active_data['change'] = 0
        
        # 计算加权平均值
        for column in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']:
            feature_name = f"{column}_avg"
            features[feature_name] = self._calculate_weighted_average(active_data, column, weights)
            
        # 计算总成交额
        features['total_amount'] = active_data['amount'].sum()
        
        return features
        
    def _calculate_technical_features(self, sector_data, date, features):
        """计算技术指标特征
        
        Args:
            sector_data: DataFrame，包含板块成分股数据
            date: 日期字符串
            features: 现有特征字典
            
        Returns:
            更新后的特征字典
        """
        try:
            # 检查是否有足够的数据计算技术指标
            if len(sector_data) == 0:
                return features
                
            # 创建临时DataFrame用于计算技术指标
            temp_data = pd.DataFrame({
                'date': pd.to_datetime(sector_data['date']),
                'open': sector_data['open_avg'],
                'high': sector_data['high_avg'],
                'low': sector_data['low_avg'],
                'close': sector_data['close_avg'],
                'volume': sector_data['volume_avg'],
                'amount': sector_data['amount_avg']
            }).sort_values('date')
            
            # 计算MA
            ma_windows = [5, 10, 20, 60]
            for window in ma_windows:
                ma = TechnicalIndicators.calculate_ma(temp_data, window, 'close')
                features[f'ma{window}_avg'] = ma.iloc[-1] if not ma.empty else 0
                
            # 计算价格相对于MA的位置
            ma_windows_for_position = [5, 20]
            for window in ma_windows_for_position:
                position = TechnicalIndicators.calculate_price_ma_position(temp_data, window, 'close')
                features[f'price_above_ma{window}_ratio'] = position
                
            # 计算趋势强度
            trend_strength = TechnicalIndicators.calculate_sector_trend_strength(temp_data, 10, 'close')
            features['trend_strength'] = trend_strength
            
            # 计算其他技术指标
            # MACD
            macd_result = TechnicalIndicators.calculate_macd(temp_data)
            features['macd'] = macd_result['macd'].iloc[-1] if not macd_result['macd'].empty else 0
            features['macd_signal'] = macd_result['signal'].iloc[-1] if not macd_result['signal'].empty else 0
            features['macd_hist'] = macd_result['histogram'].iloc[-1] if not macd_result['histogram'].empty else 0
            
            # RSI
            rsi = TechnicalIndicators.calculate_rsi(temp_data, 14, 'close')
            features['rsi14'] = rsi.iloc[-1] if not rsi.empty else 50
            
            # KDJ
            kdj_result = TechnicalIndicators.calculate_kdj(temp_data)
            features['kdj_k'] = kdj_result['k'].iloc[-1] if not kdj_result['k'].empty else 50
            features['kdj_d'] = kdj_result['d'].iloc[-1] if not kdj_result['d'].empty else 50
            features['kdj_j'] = kdj_result['j'].iloc[-1] if not kdj_result['j'].empty else 50
            
            # 布林带
            bb_result = TechnicalIndicators.calculate_bollinger_bands(temp_data)
            features['bb_middle'] = bb_result['middle'].iloc[-1] if not bb_result['middle'].empty else 0
            features['bb_upper'] = bb_result['upper'].iloc[-1] if not bb_result['upper'].empty else 0
            features['bb_lower'] = bb_result['lower'].iloc[-1] if not bb_result['lower'].empty else 0
            features['bb_width'] = (bb_result['upper'].iloc[-1] - bb_result['lower'].iloc[-1]) / bb_result['middle'].iloc[-1] if not bb_result['middle'].empty and bb_result['middle'].iloc[-1] != 0 else 0
            
            # 计算近期涨跌比例
            recent_window = 5
            if len(temp_data) >= recent_window:
                recent_data = temp_data.iloc[-recent_window:]
                up_days = sum(recent_data['close'].pct_change() > 0)
                features['up_days_ratio'] = up_days / recent_window
                
        except Exception as e:
            logger.error(f"计算板块的技术指标特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
        return features
        
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """为指定日期生成板块特征
        
        Args:
            sector: 板块名称
            date: 日期字符串，格式为YYYYMMDD
            preloaded_data: 预加载的数据，如果为None则自动加载
            
        Returns:
            特征DataFrame
        """
        try:
            # 获取板块成分股
            stocks = self._get_sector_stocks(sector)
            
            if not stocks:
                logger.warning(f"板块 {sector} 没有找到成分股，无法生成特征")
                return pd.DataFrame()
                
            # 获取成分股数据
            if preloaded_data is not None:
                sector_data = preloaded_data
            else:
                try:
                    sector_data = self.data_provider.get_daily_data(stocks, date, date)
                except Exception as e:
                    logger.error(f"获取板块 {sector} 在日期 {date} 的数据失败: {str(e)}")
                    return pd.DataFrame()
                    
            if sector_data.empty:
                logger.warning(f"板块 {sector} 在日期 {date} 没有数据")
                return pd.DataFrame({"date": [date], "sector_name": [sector], "stock_count": [len(stocks)]})
                
            # 1. 计算基础特征
            features = self._calculate_basic_features(sector_data, date, stocks)
            features['sector_name'] = sector
            
            # 2. 获取历史数据，用于计算技术指标
            try:
                # 计算30天前的日期
                date_obj = datetime.strptime(date, "%Y%m%d")
                start_date = (date_obj - timedelta(days=90)).strftime("%Y%m%d")
                
                # 获取历史数据
                historical_data = self.data_provider.get_daily_data(stocks, start_date, date)
                
                if not historical_data.empty:
                    # 按日期分组计算每日特征
                    daily_features = []
                    for d, group in historical_data.groupby('date'):
                        day_features = self._calculate_basic_features(group, d, stocks)
                        day_features['sector_name'] = sector
                        daily_features.append(day_features)
                        
                    # 创建历史特征DataFrame
                    historical_features = pd.DataFrame(daily_features)
                    
                    # 3. 计算技术指标特征
                    features = self._calculate_technical_features(historical_features, date, features)
                    
            except Exception as e:
                logger.error(f"计算板块 {sector} 的技术指标特征时出错: {str(e)}")
                # 继续处理，仅使用基础特征
                
            # 转换为DataFrame
            features_df = pd.DataFrame([features])
            
            logger.info(f"成功为板块 {sector} 在日期 {date} 生成特征")
            return features_df
            
        except Exception as e:
            logger.error(f"生成板块 {sector} 在日期 {date} 的特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    def generate_and_save_features(self, sector, start_date, end_date):
        """生成并保存一段时间的板块特征
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            成功生成特征的天数
        """
        try:
            # 获取交易日列表
            trading_days = self.data_provider.get_trading_days(start_date, end_date)
            
            if not trading_days:
                logger.warning(f"在 {start_date} 到 {end_date} 之间没有交易日")
                return 0
                
            # 创建板块目录
            sector_dir = os.path.join(self.output_dir, sector)
            ensure_dir(sector_dir)
            
            # 检查已存在的特征文件
            existing_files = os.listdir(sector_dir) if os.path.exists(sector_dir) else []
            existing_dates = set([f.split('.')[0] for f in existing_files if f.endswith('.csv')])
            
            # 生成特征
            all_features = []
            success_count = 0
            latest_features_df = None
            
            for date in trading_days:
                # 如果特征已存在且不需要重新生成，则跳过
                if date in existing_dates:
                    logger.debug(f"板块 {sector} 在日期 {date} 的特征已存在，跳过")
                    success_count += 1
                    continue
                    
                # 生成特征
                features_df = self.generate_sector_features_for_date(sector, date)
                
                if not features_df.empty:
                    # 添加到结果列表
                    all_features.append(features_df)
                    
                    # 保存单日特征
                    output_file = os.path.join(sector_dir, f"{date}.csv")
                    features_df.to_csv(output_file, index=False)
                    
                    logger.info(f"保存板块 {sector} 在日期 {date} 的特征到 {output_file}")
                    success_count += 1
                    
                    # 记录最新日期的特征
                    latest_features_df = features_df
                    
            # 如果有生成新特征，则保存合并特征
            if all_features:
                combined_features = pd.concat(all_features, ignore_index=True)
                
                # 1. 保存合并CSV特征
                combined_output_file = os.path.join(sector_dir, f"{start_date}_{end_date}.csv")
                combined_features.to_csv(combined_output_file, index=False)
                logger.info(f"保存板块 {sector} 在 {start_date} 到 {end_date} 的合并CSV特征到 {combined_output_file}")
                
                # 2. 保存Parquet格式的特征文件
                try:
                    # 构建Parquet文件路径
                    parquet_file = os.path.join(sector_dir, f"{sector}_features.parquet")
                    
                    # 检查是否有已存在的Parquet文件
                    if os.path.exists(parquet_file):
                        try:
                            # 读取现有文件
                            existing_df = pd.read_parquet(parquet_file)
                            
                            # 合并新旧数据，保留最新的数据
                            all_df = pd.concat([existing_df, combined_features], ignore_index=True)
                            
                            # 确保日期列存在并转换为日期类型
                            if 'date' in all_df.columns:
                                all_df['date'] = pd.to_datetime(all_df['date'])
                                
                                # 删除重复的日期，保留最新的数据
                                all_df = all_df.drop_duplicates(subset=['date'], keep='last')
                                
                                # 按日期排序
                                all_df = all_df.sort_values('date')
                            
                            # 保存完整数据
                            all_df.to_parquet(parquet_file, index=False)
                            logger.info(f"更新板块 {sector} 的Parquet特征文件 {parquet_file}，共 {len(all_df)} 行")
                        except Exception as e:
                            logger.error(f"读取和合并现有Parquet文件时出错: {e}")
                            # 如果出错，直接保存新数据
                            combined_features.to_parquet(parquet_file, index=False)
                            logger.info(f"创建板块 {sector} 的新Parquet特征文件 {parquet_file}，共 {len(combined_features)} 行")
                    else:
                        # 直接保存新数据
                        combined_features.to_parquet(parquet_file, index=False)
                        logger.info(f"创建板块 {sector} 的新Parquet特征文件 {parquet_file}，共 {len(combined_features)} 行")
                except Exception as e:
                    logger.error(f"保存Parquet格式特征时出错: {e}")
                
                # 3. 保存最新日期的特征
                if latest_features_df is not None:
                    latest_file = os.path.join(sector_dir, f"{sector}_latest.csv")
                    latest_features_df.to_csv(latest_file, index=False)
                    logger.info(f"保存板块 {sector} 的最新特征到 {latest_file}")
                
            return success_count
            
        except Exception as e:
            logger.error(f"生成并保存板块 {sector} 在 {start_date} 到 {end_date} 的特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return 0