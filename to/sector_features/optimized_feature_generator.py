#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版板块特征生成器

融合了高性能处理技术和增强特征计算功能：
1. 使用内存映射、数据压缩和增量更新等性能优化
2. 多进程和向量化技术加速特征计算
3. 包含基础和增强板块特征

适用于大规模板块特征生成处理。
"""

import os
import gc
import sys
import time
import logging
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入基础特征生成器和增强特征生成器
try:
    from to.sector_features.enhanced_feature_generator import EnhancedSectorFeatureGenerator
    from to.sector_features.technical_indicators import TechnicalIndicators
except ImportError:
    logger.error("无法导入特征生成器模块，请检查路径")
    raise

# 尝试导入性能优化模块
try:
    from to.path_utils import get_data_path, ensure_dir
    from to.memory_map_manager import get_memory_map_manager
    from to.feature_compress import optimize_dtypes, save_features, load_features
    from to.feature_update import (
        get_feature_update_manager, 
        update_features_incrementally,
        merge_feature_files
    )
except ImportError:
    logger.warning("未能导入性能优化模块，将使用基本IO功能")
    
    # 定义简化版功能
    def get_data_path(relative_path=None):
        data_root = Path('to') / 'local_data'
        if relative_path:
            return data_root / relative_path
        return data_root
    
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path
    
    def optimize_dtypes(df):
        return df
    
    def save_features(df, file_path):
        df.to_csv(file_path)
        return True
    
    def load_features(file_path):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return pd.DataFrame()

# 尝试导入业务逻辑模块
try:
    from to.stock_sectors_mapping import get_sector_stocks
except ImportError:
    logger.error("无法导入板块映射模块，请检查路径")
    raise

# 尝试使用Numba进行加速
try:
    from numba import jit, prange
    
    @jit(nopython=True, parallel=True)
    def calculate_ma_fast(values, window):
        """使用Numba加速的移动平均计算"""
        n = len(values)
        result = np.full(n, np.nan)
        
        for i in prange(window - 1, n):
            result[i] = np.mean(values[i - window + 1:i + 1])
            
        return result
    
    @jit(nopython=True)
    def calculate_ema_fast(values, span):
        """使用Numba加速的指数移动平均计算"""
        n = len(values)
        result = np.full(n, np.nan)
        alpha = 2.0 / (span + 1.0)
        
        # 初始化第一个有效值
        for i in range(n):
            if not np.isnan(values[i]):
                result[i] = values[i]
                break
        
        # 计算EMA
        for i in range(i + 1, n):
            if not np.isnan(values[i]):
                result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
            else:
                result[i] = result[i - 1]
                
        return result
    
    @jit(nopython=True)
    def calculate_macd_fast(values, fast_span=12, slow_span=26, signal_span=9):
        """使用Numba加速的MACD指标计算"""
        ema_fast = calculate_ema_fast(values, fast_span)
        ema_slow = calculate_ema_fast(values, slow_span)
        
        macd = ema_fast - ema_slow
        signal = calculate_ema_fast(macd, signal_span)
        hist = macd - signal
        
        return macd, signal, hist
    
    @jit(nopython=True, parallel=True)
    def calculate_rsi_fast(values, window=14):
        """使用Numba加速的RSI指标计算"""
        n = len(values)
        deltas = np.zeros(n)
        result = np.full(n, np.nan)
        
        # 计算价格变化
        for i in range(1, n):
            deltas[i] = values[i] - values[i-1]
        
        # 计算RSI
        up_sum = np.zeros(n)
        down_sum = np.zeros(n)
        
        for i in range(1, n):
            if deltas[i] > 0:
                up_sum[i] = up_sum[i-1] + deltas[i]
                down_sum[i] = down_sum[i-1]
            elif deltas[i] < 0:
                up_sum[i] = up_sum[i-1]
                down_sum[i] = down_sum[i-1] - deltas[i]
            else:
                up_sum[i] = up_sum[i-1]
                down_sum[i] = down_sum[i-1]
            
            if i >= window:
                avg_up = (up_sum[i] - up_sum[i-window]) / window
                avg_down = (down_sum[i] - down_sum[i-window]) / window
                
                if avg_down == 0:
                    result[i] = 100.0
                else:
                    rs = avg_up / avg_down
                    result[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return result
    
    HAS_NUMBA = True
    logger.info("使用Numba加速特征计算")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba未安装，使用标准计算方法")

# 定义工作进程数
DEFAULT_MAX_WORKERS = max(1, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 4

class OptimizedEnhancedSectorFeatureGenerator(EnhancedSectorFeatureGenerator):
    """优化版增强特征生成器
    
    融合高性能处理和丰富特征计算能力：
    1. 继承了增强特征生成器的特征计算功能
    2. 加入了内存映射、数据压缩等性能优化
    3. 支持多进程和向量化加速
    """
    
    def __init__(self, 
                data_provider, 
                sector_mapping, 
                output_dir=None,
                weighting_method='equal', 
                ranking_data_path=None, 
                top_n=100,
                max_workers=DEFAULT_MAX_WORKERS,
                enable_compression=True,
                batch_size=30):
        """初始化优化版特征生成器
        
        Args:
            data_provider: 数据提供者，需要实现get_daily_data和get_trading_days方法
            sector_mapping: 板块映射字典 {板块名称: [股票列表]}
            output_dir: 特征输出目录，默认使用get_data_path('sector_features')
            weighting_method: 权重计算方法，可选值为equal, market_cap等
            ranking_data_path: 排名数据路径，如果为None则不加载排名数据
            top_n: 排名前N的股票数量，用于计算排名占比
            max_workers: 最大工作进程数，默认为CPU核心数-1
            enable_compression: 是否启用数据压缩，默认为True
            batch_size: 每批处理的日期数量，默认为30
        """
        # 使用get_data_path设置默认输出目录
        if output_dir is None:
            output_dir = get_data_path('sector_features')
            
        super().__init__(
            data_provider=data_provider,
            sector_mapping=sector_mapping,
            output_dir=output_dir,
            weighting_method=weighting_method,
            ranking_data_path=ranking_data_path,
            top_n=top_n
        )
        
        # 性能优化相关配置
        self.max_workers = max_workers
        self.enable_compression = enable_compression
        self.batch_size = batch_size
        
        # 尝试初始化内存映射和特征更新管理器
        try:
            self.memory_map_manager = get_memory_map_manager()
            self.feature_update_manager = get_feature_update_manager(self.output_dir)
            self.has_optimization = True
        except Exception as e:
            logger.warning(f"初始化性能优化组件失败: {str(e)}，将使用基本IO功能")
            self.has_optimization = False
    
    def _apply_numba_acceleration(self, price_series, feature_type, **kwargs):
        """应用Numba加速的特征计算
        
        Args:
            price_series: 价格Series
            feature_type: 特征类型，如'ma', 'ema', 'macd', 'rsi'等
            **kwargs: 附加参数
            
        Returns:
            计算结果，根据特征类型不同而不同
        """
        if not HAS_NUMBA:
            # 回退到标准计算
            indicator = TechnicalIndicators()
            if feature_type == 'ma':
                return indicator.calculate_ma(pd.DataFrame({'close': price_series}), **kwargs)
            elif feature_type == 'ema':
                return indicator.calculate_ema(pd.DataFrame({'close': price_series}), **kwargs)
            elif feature_type == 'rsi':
                return indicator.calculate_rsi(pd.DataFrame({'close': price_series}), **kwargs)
            elif feature_type == 'macd':
                macd_result = indicator.calculate_macd(pd.DataFrame({'close': price_series}), **kwargs)
                return macd_result['macd'], macd_result['signal'], macd_result['histogram']
            
            return None
            
        # 使用Numba加速计算
        values = price_series.to_numpy(dtype=np.float64)
        
        if feature_type == 'ma':
            window = kwargs.get('window', 5)
            result = calculate_ma_fast(values, window)
            return pd.Series(result, index=price_series.index)
            
        elif feature_type == 'ema':
            window = kwargs.get('window', 12)
            result = calculate_ema_fast(values, window)
            return pd.Series(result, index=price_series.index)
            
        elif feature_type == 'rsi':
            window = kwargs.get('window', 14)
            result = calculate_rsi_fast(values, window)
            return pd.Series(result, index=price_series.index)
            
        elif feature_type == 'macd':
            fast_window = kwargs.get('fast_window', 12)
            slow_window = kwargs.get('slow_window', 26)
            signal_window = kwargs.get('signal_window', 9)
            
            macd, signal, hist = calculate_macd_fast(
                values, fast_window, slow_window, signal_window
            )
            
            return (
                pd.Series(macd, index=price_series.index),
                pd.Series(signal, index=price_series.index),
                pd.Series(hist, index=price_series.index)
            )
            
        return None
    
    def _calculate_technical_features(self, sector_data, date, features):
        """计算技术指标特征，使用加速版本
        
        Args:
            sector_data: 板块数据DataFrame
            date: 日期字符串
            features: 已有特征字典
            
        Returns:
            更新后的特征字典
        """
        try:
            # 检查是否有足够的数据计算技术指标
            if len(sector_data) < 60:  # 至少需要60天的数据
                logger.warning(f"数据长度不足以计算技术指标，当前长度: {len(sector_data)}")
                return features
            
            # 使用加速版本计算移动平均线
            close_series = sector_data['close_avg']
            
            # 计算MA
            for window in [5, 10, 20, 60]:
                ma = self._apply_numba_acceleration(close_series, 'ma', window=window)
                features[f'ma{window}_avg'] = float(ma.iloc[-1]) if not ma.empty else np.nan
            
            # 计算EMA
            for window in [12, 26]:
                ema = self._apply_numba_acceleration(close_series, 'ema', window=window)
                features[f'ema{window}_avg'] = float(ema.iloc[-1]) if not ema.empty else np.nan
            
            # 计算RSI
            rsi = self._apply_numba_acceleration(close_series, 'rsi', window=14)
            features['rsi14_avg'] = float(rsi.iloc[-1]) if not rsi.empty else np.nan
            
            # 计算MACD
            macd, signal, hist = self._apply_numba_acceleration(
                close_series, 'macd', 
                fast_window=12, slow_window=26, signal_window=9
            )
            features['macd_avg'] = float(macd.iloc[-1]) if not macd.empty else np.nan
            features['macd_signal_avg'] = float(signal.iloc[-1]) if not signal.empty else np.nan
            features['macd_hist_avg'] = float(hist.iloc[-1]) if not hist.empty else np.nan
            
            # 计算价格相对于MA的位置
            for window in [5, 20]:
                ma_series = self._apply_numba_acceleration(close_series, 'ma', window=window)
                if not ma_series.empty and not pd.isna(ma_series.iloc[-1]) and ma_series.iloc[-1] > 0:
                    above_ratio = sum(close_series > ma_series) / len(close_series)
                    features[f'price_above_ma{window}_ratio'] = float(above_ratio)
                    
                    # 价格与MA的距离百分比
                    ma_latest = ma_series.iloc[-1]
                    close_latest = close_series.iloc[-1]
                    distance_pct = (close_latest - ma_latest) / ma_latest if ma_latest > 0 else 0
                    features[f'price_ma{window}_distance_pct'] = float(distance_pct)
                else:
                    features[f'price_above_ma{window}_ratio'] = 0.5
                    features[f'price_ma{window}_distance_pct'] = 0.0
            
            # 添加趋势强度
            # 使用最近20天的数据计算趋势强度
            recent_data = sector_data.iloc[-20:]
            price_changes = recent_data['close_avg'].pct_change().dropna()
            
            if len(price_changes) > 0:
                positive_changes = sum(price_changes > 0)
                negative_changes = sum(price_changes < 0)
                total_changes = positive_changes + negative_changes
                
                if total_changes > 0:
                    trend_strength = (positive_changes - negative_changes) / total_changes
                    features['trend_strength'] = float(trend_strength)
                else:
                    features['trend_strength'] = 0.0
            else:
                features['trend_strength'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return features
    
    def generate_sector_features_for_date(self, sector, date, preloaded_data=None):
        """为特定日期生成板块特征
        
        Args:
            sector: 板块名称
            date: 目标日期
            preloaded_data: 预加载的数据字典，可选
            
        Returns:
            包含特征的DataFrame
        """
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 调用父类方法生成增强特征
            features_df = super().generate_sector_features_for_date(sector, date, preloaded_data)
            
            # 计算执行时间并记录日志
            elapsed_time = time.time() - start_time
            logger.info(f"板块 {sector} 在日期 {date} 的特征生成完成，耗时: {elapsed_time:.2f}秒")
            
            return features_df
            
        except Exception as e:
            logger.error(f"为板块 {sector} 在日期 {date} 生成特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _split_date_range(self, start_date, end_date):
        """将日期范围拆分为多个小批次
        
        Args:
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            日期批次列表，每个元素是(batch_start, batch_end)元组
        """
        try:
            # 获取交易日列表
            trading_days = self.data_provider.get_trading_days(start_date, end_date)
            
            if not trading_days:
                logger.warning(f"日期范围 {start_date} - {end_date} 内没有交易日")
                return []
                
            # 按照batch_size将日期分组
            date_batches = []
            for i in range(0, len(trading_days), self.batch_size):
                batch = trading_days[i:i + self.batch_size]
                if batch:
                    date_batches.append((batch[0], batch[-1]))
            
            return date_batches
            
        except Exception as e:
            logger.error(f"分割日期范围时出错: {str(e)}")
            return [(start_date, end_date)]  # 回退到一个批次
    
    def _worker_process_sector(self, sector, batch_start, batch_end):
        """工作进程处理单个板块的特征生成
        
        Args:
            sector: 板块名称
            batch_start: 批次开始日期
            batch_end: 批次结束日期
            
        Returns:
            (成功标志, 成功生成的特征数量)
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            logger.info(f"工作进程开始处理板块 {sector}，日期范围: {batch_start} - {batch_end}")
            
            # 获取板块股票
            stocks = self._get_sector_stocks(sector)
            if not stocks:
                logger.warning(f"板块 {sector} 没有股票")
                return False, 0
                
            # 确保输出目录存在
            sector_dir = os.path.join(self.output_dir, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            # 获取交易日列表
            trading_days = self.data_provider.get_trading_days(batch_start, batch_end)
            if not trading_days:
                logger.warning(f"日期范围 {batch_start} - {batch_end} 内没有交易日")
                return False, 0
                
            # 预加载数据
            # 添加向前的缓冲区以便计算技术指标
            buffer_days = 100
            buffer_start_date = None
            try:
                # 尝试获取更早的日期
                start_dt = datetime.strptime(batch_start, '%Y%m%d')
                buffer_start = start_dt - timedelta(days=buffer_days)
                buffer_start_date = buffer_start.strftime('%Y%m%d')
            except:
                pass
                
            # 获取股票数据
            stock_data = self.data_provider.get_daily_data(
                stocks, 
                buffer_start_date or batch_start, 
                batch_end
            )
            
            if stock_data.empty:
                logger.warning(f"板块 {sector} 在日期范围 {batch_start} - {batch_end} 没有数据")
                return False, 0
                
            # 预加载数据字典
            preloaded_data = {'stock_data': stock_data}
            
            # 为每个交易日生成特征
            features_list = []
            success_count = 0
            
            for date in trading_days:
                features_df = self.generate_sector_features_for_date(sector, date, preloaded_data)
                
                if not features_df.empty:
                    features_list.append(features_df)
                    success_count += 1
            
            # 合并所有特征数据
            if features_list:
                combined_features = pd.concat(features_list)
                
                # 保存特征数据
                output_file = os.path.join(sector_dir, f"{batch_start}_{batch_end}.csv")
                
                if self.enable_compression and self.has_optimization:
                    # 优化数据类型
                    combined_features = optimize_dtypes(combined_features)
                    save_features(combined_features, output_file)
                else:
                    # 使用标准CSV保存
                    combined_features.to_csv(output_file, index=True)
                    
                # 记录结束时间和日志
                elapsed_time = time.time() - start_time
                logger.info(f"工作进程成功处理板块 {sector}，日期范围: {batch_start} - {batch_end}，" 
                           f"成功: {success_count}/{len(trading_days)}，耗时: {elapsed_time:.2f}秒")
                
                return True, success_count
            else:
                logger.warning(f"板块 {sector} 在日期范围 {batch_start} - {batch_end} 没有生成特征")
                return False, 0
                
        except Exception as e:
            logger.error(f"处理板块 {sector} 时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False, 0
    
    def generate_and_save_features(self, sector, start_date, end_date):
        """生成并保存板块特征数据
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            成功生成特征的数量
        """
        try:
            # 记录开始时间
            start_time = time.time()
            
            logger.info(f"开始为板块 {sector} 生成特征，日期范围: {start_date} - {end_date}")
            
            # 确保输出目录存在
            sector_dir = os.path.join(self.output_dir, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            # 将日期范围分割为多个批次
            date_batches = self._split_date_range(start_date, end_date)
            if not date_batches:
                logger.warning(f"日期范围 {start_date} - {end_date} 没有有效的交易日")
                return 0
            
            # 单进程处理
            if self.max_workers <= 1 or len(date_batches) <= 1:
                logger.info(f"使用单进程模式处理板块 {sector}")
                total_success = 0
                
                for batch_start, batch_end in date_batches:
                    success, count = self._worker_process_sector(sector, batch_start, batch_end)
                    if success:
                        total_success += count
                
                # 合并所有批次的特征文件
                self._merge_feature_files(sector)
                
                # 记录结束时间和日志
                elapsed_time = time.time() - start_time
                logger.info(f"完成板块 {sector} 特征生成，成功: {total_success}，耗时: {elapsed_time:.2f}秒")
                
                return total_success
            
            # 多进程处理
            logger.info(f"使用多进程模式处理板块 {sector}，进程数: {self.max_workers}，批次数: {len(date_batches)}")
            total_success = 0
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有批次的任务
                futures = []
                for batch_start, batch_end in date_batches:
                    future = executor.submit(self._worker_process_sector, sector, batch_start, batch_end)
                    futures.append(future)
                
                # 处理结果
                for future in as_completed(futures):
                    try:
                        success, count = future.result()
                        if success:
                            total_success += count
                    except Exception as e:
                        logger.error(f"获取任务结果时出错: {str(e)}")
            
            # 合并所有批次的特征文件
            self._merge_feature_files(sector)
            
            # 清理内存
            gc.collect()
            
            # 记录结束时间和日志
            elapsed_time = time.time() - start_time
            logger.info(f"完成板块 {sector} 特征生成，总成功: {total_success}，总耗时: {elapsed_time:.2f}秒")
            
            return total_success
            
        except Exception as e:
            logger.error(f"生成板块 {sector} 特征时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return 0
    
    def _merge_feature_files(self, sector):
        """合并板块的特征文件
        
        Args:
            sector: 板块名称
        """
        try:
            sector_dir = os.path.join(self.output_dir, sector)
            if not os.path.exists(sector_dir):
                logger.warning(f"板块 {sector} 的目录不存在")
                return
            
            # 获取所有CSV文件
            csv_files = [f for f in os.listdir(sector_dir) if f.endswith('.csv') and '_' in f]
            if not csv_files:
                logger.warning(f"板块 {sector} 没有特征文件可合并")
                return
            
            # 尝试使用优化版合并
            if self.has_optimization:
                logger.info(f"使用优化版合并板块 {sector} 的 {len(csv_files)} 个特征文件")
                try:
                    merge_feature_files(sector_dir, f"{sector}_all.csv")
                    logger.info(f"成功合并板块 {sector} 的特征文件")
                    return
                except Exception as e:
                    logger.warning(f"优化版合并失败: {str(e)}，使用标准合并")
            
            # 标准合并方法
            logger.info(f"使用标准方法合并板块 {sector} 的 {len(csv_files)} 个特征文件")
            dfs = []
            
            for file in csv_files:
                file_path = os.path.join(sector_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    # 确保日期列格式正确
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
            
            if dfs:
                # 合并所有DataFrame
                combined = pd.concat(dfs)
                
                # 去除重复行
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                
                # 保存合并后的文件
                combined.to_csv(os.path.join(sector_dir, f"{sector}_all.csv"))
                logger.info(f"成功合并板块 {sector} 的特征文件，共 {len(combined)} 行")
            else:
                logger.warning(f"没有成功加载板块 {sector} 的特征文件")
                
        except Exception as e:
            logger.error(f"合并板块 {sector} 特征文件时出错: {str(e)}")
            logger.error(traceback.format_exc()) 