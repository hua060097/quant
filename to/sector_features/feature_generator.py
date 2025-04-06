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
            data: 包含数据的DataFrame
            column: 要计算的列名
            weights: 权重列名
            
        Returns:
            加权平均值
        """
        if data.empty:
            return 0.0
        
        # 移除任何权重为NaN或无穷的行
        valid_data = data.copy()
        valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna(subset=[weights, column])
        
        if valid_data.empty:
            return 0.0
            
        # 计算加权平均
        try:
            result = np.average(valid_data[column], weights=valid_data[weights])
            # 四舍五入到两位小数
            return round(result, 2)
        except Exception as e:
            logger.error(f"计算加权平均值时出错: {e}")
            return 0.0
        
    def _calculate_basic_features(self, sector_data, date, stocks):
        """
        计算板块的基本特征
        
        Args:
            sector_data: 板块股票的行情数据
            date: 日期
            stocks: 板块的股票列表
            
        Returns:
            包含基本特征的DataFrame
        """
        if sector_data is None or sector_data.empty:
            logger.warning(f"日期 {date} 没有有效数据")
            return None
            
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in required_columns:
            if col not in sector_data.columns:
                logger.error(f"数据缺少必要的列: {col}")
                return None
        
        # 筛选出有效的数据（收盘价大于0）
        valid_data = sector_data[sector_data['close'] > 0].copy()
        
        if valid_data.empty:
            logger.warning(f"日期 {date} 没有有效数据（收盘价>0）")
            return None
            
        # 如果没有change_percent列，尝试计算
        if 'change_percent' not in valid_data.columns:
            logger.info(f"数据源中没有change_percent列，将计算该值")
            try:
                # 获取前一个交易日的数据来计算涨跌幅
                prev_date = self._get_previous_trading_date(date)
                if prev_date:
                    logger.info(f"获取到前一个交易日: {prev_date}")
                    prev_data = self._load_sector_stocks(stocks, prev_date)
                    if prev_data is not None and not prev_data.empty:
                        # 创建前一日收盘价的映射
                        prev_close_map = dict(zip(prev_data['stock_code'], prev_data['close']))
                        
                        # 添加前一日收盘价
                        valid_data['prev_close'] = valid_data['stock_code'].map(prev_close_map)
                        
                        # 计算涨跌幅，并处理缺失值
                        valid_data['change_percent'] = valid_data.apply(
                            lambda x: round((x['close'] / x['prev_close'] - 1) * 100, 2) 
                            if pd.notna(x.get('prev_close', None)) and x.get('prev_close', 0) > 0 
                            else 0, 
                            axis=1
                        )
                        logger.info(f"成功计算涨跌幅，样例: {valid_data['change_percent'].head()}")
                    else:
                        logger.warning(f"无法获取前一交易日数据，将使用默认涨跌幅0")
                        valid_data['change_percent'] = 0
                else:
                    logger.warning(f"无法获取前一交易日，将使用默认涨跌幅0")
                    valid_data['change_percent'] = 0
            except Exception as e:
                logger.error(f"计算涨跌幅时出错: {e}")
                valid_data['change_percent'] = 0
        
        # 计算其他可能缺失的列
        if 'turnover_rate' not in valid_data.columns:
            valid_data['turnover_rate'] = 0  # 使用默认值
            
        if 'amplitude' not in valid_data.columns:
            # 振幅 = (最高价 - 最低价) / 收盘价 * 100%
            valid_data['amplitude'] = ((valid_data['high'] - valid_data['low']) / valid_data['close'] * 100).round(2)
            
        # 计算特征
        features = {}
        
        # 日期
        features['date'] = date
        features['date_str'] = date.strftime('%Y%m%d') if isinstance(date, (pd.Timestamp, datetime)) else str(date)
        
        # 股票数量
        features['stock_count'] = len(valid_data)
        
        # 使用valid_data计算平均值，保留两位小数
        features['close'] = round(valid_data['close'].mean(), 2)
        features['open'] = round(valid_data['open'].mean(), 2)
        features['high'] = round(valid_data['high'].mean(), 2)
        features['low'] = round(valid_data['low'].mean(), 2)
        features['volume'] = round(valid_data['volume'].mean(), 2)
        features['amount'] = round(valid_data['amount'].mean(), 2)
        features['change_percent'] = round(valid_data['change_percent'].mean(), 2)
        
        if 'turnover_rate' in valid_data.columns:
            features['turnover_rate'] = round(valid_data['turnover_rate'].mean(), 2)
        else:
            features['turnover_rate'] = 0
            
        if 'amplitude' in valid_data.columns:
            features['amplitude'] = round(valid_data['amplitude'].mean(), 2)
        else:
            features['amplitude'] = round(((valid_data['high'] - valid_data['low']) / valid_data['close'] * 100).mean(), 2)
        
        # 计算波动性指标
        features['volatility'] = round(valid_data['change_percent'].std(), 2)
        
        # 计算涨跌家数
        features['up_count'] = len(valid_data[valid_data['change_percent'] > 0])
        features['down_count'] = len(valid_data[valid_data['change_percent'] < 0])
        features['flat_count'] = len(valid_data[valid_data['change_percent'] == 0])
        
        # 涨停和跌停计数 (假设涨跌停为9.8%)
        features['limit_up_count'] = len(valid_data[valid_data['change_percent'] >= 9.8])
        features['limit_down_count'] = len(valid_data[valid_data['change_percent'] <= -9.8])
        
        # 创建DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
        
    def _calculate_technical_features(self, sector_data, date, features):
        """计算技术指标特征
        
        利用技术指标计算类，计算各种技术指标，如MA、MACD、RSI等
        
        Args:
            sector_data: 板块股票的行情数据
            date: 当前日期
            features: 包含基本特征的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 如果没有历史数据，无法计算技术指标
        if sector_data is None or sector_data.empty:
            return features
            
        # 使用板块的均价数据计算技术指标
        try:
            # 创建技术指标计算器
            indicator = TechnicalIndicators()
            
            # 构建用于计算指标的价格数据
            # 这里我们使用的是板块均价，不是单个股票
            price_data = pd.DataFrame({
                'close': [features['close'].values[0]],
                'open': [features['open'].values[0]],
                'high': [features['high'].values[0]],
                'low': [features['low'].values[0]],
                'volume': [features['volume'].values[0]]
            })
            
            # 检查是否能获取足够的历史数据来计算技术指标
            # 对于实际应用，应该从数据源获取历史数据
            # 这里简化处理，使用当天数据
            
            # 计算移动平均线 MA5, MA10, MA20, MA30, MA60
            # 在实际应用中，这需要至少60天的历史数据
            # 这里我们模拟计算，填充一些默认值
            
            # 为了演示，我们简单地设置一些默认值
            ma_values = {
                'ma5': features['close'].values[0],
                'ma10': features['close'].values[0],
                'ma20': features['close'].values[0],
                'ma30': features['close'].values[0],
                'ma60': features['close'].values[0]
            }
            
            # 添加MA值到特征中
            for ma_key, ma_value in ma_values.items():
                features[ma_key] = ma_value
                
            # 在这里可以添加更多技术指标的计算
            # 例如MACD, RSI, Bollinger Bands等
            
            return features
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return features
        
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """计算指定日期的板块特征
        
        Args:
            sector: 板块名称
            date: 日期
            preloaded_data: 预加载的数据，如果提供，则不会再次加载数据
            
        Returns:
            tuple: (成功标志, 特征DataFrame)，如果失败则第二个元素为None
        """
        try:
            # 转换日期为pandas datetime
            if isinstance(date, str):
                if len(date) == 8 and date.isdigit():  # 判断是否为YYYYMMDD格式
                    date_obj = pd.to_datetime(
                        datetime.strptime(date, '%Y%m%d')
                    )
                else:
                    date_obj = pd.to_datetime(date)
            else:
                date_obj = pd.to_datetime(date)
                
            # 获取板块股票
            stocks = self._get_sector_stocks(sector)
            if not stocks:
                logger.warning(f"板块 {sector} 没有成分股")
                return False, None
                
            # 加载数据
            if preloaded_data is not None:
                sector_data = preloaded_data
            else:
                # 加载这个板块所有股票在指定日期的数据
                date_str = date_obj.strftime('%Y%m%d')
                sector_data = self._load_sector_stocks(stocks, date_str)
                
            if sector_data is None or sector_data.empty:
                logger.warning(f"日期 {date} 没有数据，板块: {sector}")
                return False, None
                
            # 过滤有效数据
            sector_data = sector_data.replace([np.inf, -np.inf], np.nan)
            sector_data = sector_data.dropna(subset=['close'])
            
            if sector_data.empty:
                logger.warning(f"日期 {date} 没有有效数据，板块: {sector}")
                return False, None
            
            # 计算基本特征
            try:
                basic_features = self._calculate_basic_features(sector_data, date_obj, stocks)
                
                if basic_features is None:
                    logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错")
                    return False, None
                    
                # 转换为Series以兼容其他功能（如果是DataFrame）
                if isinstance(basic_features, pd.DataFrame):
                    if not basic_features.empty:
                        # 使用第一行数据
                        basic_features_row = basic_features.iloc[0]
                    else:
                        logger.warning(f"基本特征DataFrame为空，板块: {sector}, 日期: {date}")
                        return False, None
                else:
                    # 已经是Series
                    basic_features_row = basic_features
                
                # 添加板块名称
                basic_features_row['sector'] = sector
                
                # 确保date列的类型一致
                if 'date' in basic_features_row:
                    basic_features_row['date'] = pd.to_datetime(basic_features_row['date'])
                
                # 创建返回的DataFrame
                features_df = pd.DataFrame([basic_features_row])
                
                # 计算技术指标
                features_df = self._calculate_technical_features(sector_data, date_obj, features_df)
                
                # 将所有可能的浮点数特征转换为最多两位小数
                numeric_cols = features_df.select_dtypes(include=['float', 'int']).columns
                for col in numeric_cols:
                    features_df[col] = features_df[col].round(2)
                
                return True, features_df
                
            except Exception as e:
                logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错: {str(e)}")
                logger.error(traceback.format_exc())
                return False, None
                
        except Exception as e:
            logger.error(f"为行业 {sector} 日期 {date} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
            
    def generate_and_save_features(self, sector, start_date, end_date):
        """生成并保存板块特征
        
        Args:
            sector: 板块名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            表示操作是否成功的布尔值
        """
        try:
            logger.info(f"正在生成板块 {sector} 从 {start_date} 到 {end_date} 的特征数据")
            
            # 获取交易日列表
            trading_days = self.data_provider.get_trading_days(start_date, end_date)
            
            # 确保trading_days是列表类型
            if isinstance(trading_days, pd.DataFrame):
                if 'date' in trading_days.columns:
                    trading_days = trading_days['date'].tolist()
                else:
                    trading_days = trading_days.index.tolist()
            
            if not trading_days:
                logger.warning(f"从 {start_date} 到 {end_date} 没有交易日")
                return False
                
            # 创建一个空的DataFrame来存储所有日期的特征
            all_features = []
            
            # 对每个交易日生成特征
            for day in trading_days:
                # 转换日期格式
                if isinstance(day, pd.Timestamp):
                    date_str = day.strftime('%Y%m%d')
                else:
                    date_str = str(day)
                    
                logger.debug(f"正在计算板块 {sector} 在日期 {date_str} 的特征")
                
                # 生成特征并保存
                success, features = self.generate_sector_features_for_date(sector, date_str)
                
                if success and features is not None and not features.empty:
                    all_features.append(features)
                else:
                    logger.warning(f"板块 {sector} 在日期 {date_str} 没有生成有效特征")
            
            if not all_features:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 没有生成任何特征")
                return False
                
            # 合并所有日期的特征
            features_df = pd.concat(all_features, ignore_index=True)
            
            if features_df.empty:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 没有有效特征")
                return False
                
            # 对所有列做类型转换和四舍五入
            for col in features_df.columns:
                if col not in ['date', 'sector', 'date_str']:
                    try:
                        # 尝试转换为浮点数并四舍五入到两位小数
                        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                        if not pd.isna(features_df[col]).all():
                            features_df[col] = features_df[col].round(2)
                    except Exception as e:
                        logger.warning(f"无法将列 {col} 转换为数值型: {e}")
            
            # 按日期排序
            if 'date' in features_df.columns:
                features_df = features_df.sort_values('date')
                
            # 保存特征
            self._save_features(sector, features_df)
            
            logger.info(f"成功为板块 {sector} 生成并保存了 {len(features_df)} 天的特征数据")
            return True
            
        except Exception as e:
            logger.error(f"为板块 {sector} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _save_features(self, sector, features_df):
        """保存行业特征数据
        
        Args:
            sector: 行业代码
            features_df: 特征DataFrame
            
        Returns:
            None
        """
        try:
            # 确保输出目录存在
            sector_dir = os.path.join(self.output_dir, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            # 确保所有浮点数列都四舍五入到两位小数
            numeric_cols = features_df.select_dtypes(include=['float64']).columns
            for col in numeric_cols:
                features_df[col] = features_df[col].round(2)
            
            # 保存为Parquet文件
            parquet_path = os.path.join(sector_dir, f"{sector}_features.parquet")
            features_df.to_parquet(parquet_path, index=False)
            logger.info(f"已保存行业 {sector} 特征数据到 {parquet_path}，共 {len(features_df)} 行")
            
            # 保存最新特征为CSV文件
            if not features_df.empty:
                # 确保日期列为正确格式
                if 'date' in features_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                        features_df['date'] = pd.to_datetime(features_df['date'])
                    
                # 获取最新日期的数据
                latest_date = features_df['date'].max()
                latest_features = features_df[features_df['date'] == latest_date].copy()
                
                if not latest_features.empty:
                    latest_csv_path = os.path.join(sector_dir, f"{sector}_latest.csv")
                    latest_features.to_csv(latest_csv_path, index=False, encoding='utf-8')
                    logger.info(f"已保存行业 {sector} 最新日期 {latest_date.strftime('%Y-%m-%d')} 的特征数据到 {latest_csv_path}")
        
        except Exception as e:
            logger.error(f"保存行业 {sector} 特征数据时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_previous_trading_date(self, date):
        """
        获取指定日期的前一个交易日
        
        Args:
            date: 当前日期，可以是字符串、datetime或Timestamp
            
        Returns:
            前一个交易日，格式与输入相同
        """
        try:
            # 统一转换为datetime对象进行处理
            if isinstance(date, str):
                if len(date) == 8 and date.isdigit():  # YYYYMMDD格式
                    date_obj = datetime.strptime(date, '%Y%m%d')
                else:
                    # 尝试解析其他格式
                    try:
                        date_obj = datetime.strptime(date, '%Y-%m-%d')
                    except ValueError:
                        try:
                            date_obj = pd.to_datetime(date).to_pydatetime()
                        except:
                            logger.error(f"无法解析日期格式: {date}")
                            return None
            elif isinstance(date, pd.Timestamp):
                date_obj = date.to_pydatetime()
            elif isinstance(date, datetime):
                date_obj = date
            else:
                logger.error(f"不支持的日期类型: {type(date)}")
                return None
                
            # 获取所有交易日
            try:
                trading_days = self.data_provider.get_trading_days(
                    start_date=(date_obj - timedelta(days=30)).strftime('%Y%m%d'),
                    end_date=date_obj.strftime('%Y%m%d')
                )
                
                # 确保trading_days是列表或数组
                if isinstance(trading_days, pd.DataFrame):
                    if 'date' in trading_days.columns:
                        trading_days = trading_days['date'].tolist()
                    else:
                        trading_days = list(trading_days.index)
                
                # 排序交易日
                trading_days = sorted(trading_days)
                
                # 将date_obj转换为相同格式进行比较
                date_str = date_obj.strftime('%Y%m%d')
                
                # 查找当前日期的索引
                for i, day in enumerate(trading_days):
                    # 转换交易日为字符串格式以便比较
                    day_str = day.strftime('%Y%m%d') if hasattr(day, 'strftime') else str(day)
                    if day_str == date_str:
                        # 找到当前日期，返回前一个交易日
                        if i > 0:
                            prev_day = trading_days[i-1]
                            
                            # 与输入类型保持一致
                            if isinstance(date, str):
                                if hasattr(prev_day, 'strftime'):
                                    return prev_day.strftime('%Y%m%d')
                                else:
                                    return str(prev_day)
                            else:
                                return prev_day
                        else:
                            logger.warning(f"日期 {date} 是第一个交易日，没有前一个交易日")
                            return None
                
                logger.warning(f"在交易日历中找不到日期 {date}")
                return None
                
            except Exception as e:
                logger.error(f"获取交易日历时出错: {e}")
                return None
                
        except Exception as e:
            logger.error(f"获取前一个交易日时出错: {e}")
            return None
            
    def _load_sector_stocks(self, stocks, date):
        """
        加载指定日期的板块股票数据
        
        Args:
            stocks: 股票列表
            date: 日期
            
        Returns:
            包含股票数据的DataFrame
        """
        try:
            # 将date转换为字符串格式
            if isinstance(date, (datetime, pd.Timestamp)):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
                
            # 调用data_provider获取数据
            # 注意：使用 stocks 而不是 stock_codes 作为参数名
            data = self.data_provider.get_daily_data(
                stocks,
                date_str,
                date_str
            )
            
            if data is not None and not data.empty:
                return data
            else:
                logger.warning(f"日期 {date_str} 的数据为空")
                return None
                
        except Exception as e:
            logger.error(f"加载板块股票数据时出错: {e}")
            return None