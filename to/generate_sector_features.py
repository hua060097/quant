#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
板块特征生成模块 - 高性能版

使用内存映射、数据压缩和增量更新技术，高效生成板块特征数据。
模块集成了三个优化模块：
1. 内存映射管理 - 高效读取大文件
2. 特征数据压缩 - 优化存储空间和内存占用
3. 特征增量更新 - 避免重复计算，提高效率

支持断点续传、多进程并行处理和自动资源优化。
"""

import os
import gc
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable

# 设置导入路径，确保能够找到模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入自定义优化模块
try:
    # 尝试直接导入
    from path_utils import get_data_path, ensure_dir
    from memory_map_manager import get_memory_map_manager
    from feature_compress import optimize_dtypes, save_features, load_features
    from feature_update import (
        get_feature_update_manager, 
        update_features_incrementally,
        merge_feature_files,
        get_feature_info
    )
except ImportError:
    # 尝试相对导入
    try:
        from .path_utils import get_data_path, ensure_dir
        from .memory_map_manager import get_memory_map_manager
        from .feature_compress import optimize_dtypes, save_features, load_features
        from .feature_update import (
            get_feature_update_manager, 
            update_features_incrementally,
            merge_feature_files,
            get_feature_info
        )
    except ImportError:
        logger.error("无法导入自定义模块，请检查路径设置")
        raise

# 导入业务逻辑相关模块
try:
    # 尝试直接导入
    from data_loaders import get_daily_data, get_trading_days
    from stock_sectors_mapping import get_sector_stocks, get_tng_sectors
except ImportError:
    # 尝试相对导入
    try:
        from .data_loaders import get_daily_data, get_trading_days
        from .stock_sectors_mapping import get_sector_stocks, get_tng_sectors
    except ImportError:
        logger.error("无法导入业务逻辑模块，请检查路径设置")
        raise

# 设置日志
logging.basicConfig(
    level=logging.INFO,  # 改回INFO以减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 特征生成参数
DEFAULT_START_DATE = "20180101"
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_MAX_WORKERS = 4
DEFAULT_BATCH_SIZE = 30
DEFAULT_MIN_RECORDS_PER_FILE = 250  # 单个特征文件的最小记录数

# Numba JIT加速的技术指标计算函数
try:
    from numba import jit, prange
    import numpy as np
    
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
    logger.info("使用Numba加速技术指标计算")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba未安装，使用pandas计算技术指标")


class SectorFeatureManager:
    """板块特征管理类
    
    负责生成和管理板块特征数据，集成了内存映射、数据压缩和增量更新功能。
    
    Attributes:
        output_dir: 输出目录路径
        memory_map_manager: 内存映射管理器
        feature_update_manager: 特征更新管理器
        max_workers: 最大工作进程数
        batch_size: 每批处理的日期数量
    """
    
    def __init__(
        self, 
        output_dir: Optional[Union[str, Path]] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_records_per_file: int = DEFAULT_MIN_RECORDS_PER_FILE
    ):
        """初始化板块特征管理器
        
        Args:
            output_dir: 特征输出目录，如果为None则使用默认目录
            max_workers: 最大工作进程数
            batch_size: 每批处理的日期数量
            min_records_per_file: 特征文件的最小记录数，用于文件合并优化
        """
        self.output_dir = Path(output_dir) if output_dir else get_data_path('sector_features')
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
        
        # 初始化内存映射管理器
        self.memory_map_manager = get_memory_map_manager()
        
        # 初始化特征更新管理器
        self.feature_update_manager = get_feature_update_manager(self.output_dir)
        
        # 设置处理参数
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.min_records_per_file = min_records_per_file
        
        # 缓存统计信息
        self._feature_info_cache = {}
    
    def calculate_sector_features_for_date(self, date: str, sector: str) -> pd.DataFrame:
        """计算特定日期的板块特征
        
        Args:
            date: 目标日期，字符串格式，如"20230101"
            sector: 板块名称
            
        Returns:
            板块特征DataFrame
        """
        try:
            # 获取板块内的股票
            stocks = get_sector_stocks(sector, date)
            
            if not stocks:
                logger.warning(f"日期 {date} 板块 {sector} 没有股票")
                return pd.DataFrame()
            
            # 获取股票的日线数据
            stock_data = get_daily_data(stocks, date, date)
            
            if stock_data.empty:
                logger.warning(f"日期 {date} 板块 {sector} 没有行情数据")
                return pd.DataFrame()
            
            # 计算板块特征
            stock_count = len(stocks)
            
            # 计算板块内股票的平均价格和交易量
            open_avg = stock_data['open'].mean()
            high_avg = stock_data['high'].mean()
            low_avg = stock_data['low'].mean()
            close_avg = stock_data['close'].mean()
            volume_avg = stock_data['volume'].mean()
            amount_avg = stock_data['amount'].mean() if 'amount' in stock_data.columns else 0
            
            # 计算涨跌幅
            if 'change' in stock_data.columns:
                change_avg = stock_data['change'].mean()
            else:
                change_avg = 0
            
            # 计算活跃股票数量
            active_count = len(stock_data[stock_data['volume'] > 0])
            
            # 计算均线特征
            ma5_avg = stock_data['ma5'].mean() if 'ma5' in stock_data.columns else 0
            ma10_avg = stock_data['ma10'].mean() if 'ma10' in stock_data.columns else 0
            ma20_avg = stock_data['ma20'].mean() if 'ma20' in stock_data.columns else 0
            ma60_avg = stock_data['ma60'].mean() if 'ma60' in stock_data.columns else 0
            
            # 计算股价位于均线上方的比例
            if 'close' in stock_data.columns and 'ma5' in stock_data.columns:
                price_above_ma5_ratio = (stock_data['close'] > stock_data['ma5']).mean()
            else:
                price_above_ma5_ratio = 0
            
            if 'close' in stock_data.columns and 'ma20' in stock_data.columns:
                price_above_ma20_ratio = (stock_data['close'] > stock_data['ma20']).mean()
            else:
                price_above_ma20_ratio = 0
            
            # 计算总成交额
            total_amount = stock_data['amount'].sum() if 'amount' in stock_data.columns else 0
            
            # 计算成交量变化率
            if 'volume' in stock_data.columns and 'volume_ma5' in stock_data.columns and stock_data['volume_ma5'].mean() > 0:
                volume_change_ratio = (stock_data['volume'] / stock_data['volume_ma5']).mean()
            else:
                volume_change_ratio = 1.0
                
            # 计算量价相关性（使用近期数据）
            price_volume_correlation = 0.0
            
            # 计算涨跌停比例
            if 'is_limit_up' in stock_data.columns:
                limit_up_ratio = stock_data['is_limit_up'].mean()
            else:
                limit_up_ratio = 0
                
            if 'is_limit_down' in stock_data.columns:
                limit_down_ratio = stock_data['is_limit_down'].mean()
            else:
                limit_down_ratio = 0
            
            # 创建特征DataFrame
            features = pd.DataFrame({
                'date': [date],
                'open_avg': [open_avg],
                'high_avg': [high_avg],
                'low_avg': [low_avg],
                'close_avg': [close_avg],
                'volume_avg': [volume_avg],
                'amount_avg': [amount_avg],
                'change_avg': [change_avg],
                'active_count': [active_count],
                'ma5_avg': [ma5_avg],
                'ma10_avg': [ma10_avg],
                'ma20_avg': [ma20_avg],
                'ma60_avg': [ma60_avg],
                'price_above_ma5_ratio': [price_above_ma5_ratio],
                'price_above_ma20_ratio': [price_above_ma20_ratio],
                'total_amount': [total_amount],
                'volume_change_ratio': [volume_change_ratio],
                'price_volume_correlation': [price_volume_correlation],
                'limit_up_ratio': [limit_up_ratio],
                'limit_down_ratio': [limit_down_ratio],
                'sector_name': [sector],
                'stock_count': [stock_count]
            })
            
            # 使用数据类型优化
            features = optimize_dtypes(features)
            
            return features
        except Exception as e:
            logger.error(f"处理日期 {date} 板块 {sector} 时出错: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标
        
        使用Numba加速的技术指标计算函数增加板块特征。
        
        Args:
            df: 板块特征DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        if df.empty or 'close_avg' not in df.columns:
            return df
        
        # 确保日期是升序排列的
        df = df.sort_values('date')
        
        # 使用内存优化
        df = optimize_dtypes(df)
        
        # 计算板块的技术指标
        close_values = df['close_avg'].values
        
        # 检查是否可以使用Numba加速
        try:
            from numba import jit
            using_numba = True
            logger.debug("将使用Numba加速技术指标计算")
        except ImportError:
            using_numba = False
            logger.debug("Numba未安装，将使用pandas计算技术指标")
        
        if using_numba and HAS_NUMBA:
            # 使用Numba加速的指标计算
            try:
                # 计算移动平均
                df['ma5'] = calculate_ma_fast(close_values, 5)
                df['ma10'] = calculate_ma_fast(close_values, 10)
                df['ma20'] = calculate_ma_fast(close_values, 20)
                df['ma60'] = calculate_ma_fast(close_values, 60)
                
                # 计算MACD
                macd, signal, hist = calculate_macd_fast(close_values)
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = hist
                
                # 计算RSI
                df['rsi'] = calculate_rsi_fast(close_values)
                
            except Exception as e:
                logger.error(f"Numba加速计算技术指标失败: {e}，使用pandas计算")
                using_numba = False
        
        if not using_numba or not HAS_NUMBA:
            # 使用pandas计算技术指标
            # 1. 计算移动平均
            df['ma5'] = df['close_avg'].rolling(window=5).mean()
            df['ma10'] = df['close_avg'].rolling(window=10).mean()
            df['ma20'] = df['close_avg'].rolling(window=20).mean()
            df['ma60'] = df['close_avg'].rolling(window=60).mean()
            
            # 2. 计算EMA
            df['ema12'] = df['close_avg'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close_avg'].ewm(span=26, adjust=False).mean()
            
            # 3. 计算MACD
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 4. 计算RSI
            delta = df['close_avg'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # 5. 计算布林带
        df['bollinger_mid'] = df['close_avg'].rolling(window=20).mean()
        df['bollinger_std'] = df['close_avg'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_mid'] + 2 * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_mid'] - 2 * df['bollinger_std']
        
        # 6. 计算VWAP(成交量加权平均价)
        if 'volume_avg' in df.columns:
            df['vwap'] = (df['close_avg'] * df['volume_avg']).rolling(window=20).sum() / df['volume_avg'].rolling(window=20).sum()
        
        # 7. 计算动量指标
        df['momentum'] = df['close_avg'] / df['close_avg'].shift(10) - 1
        
        # 8. 计算波动率
        df['volatility'] = df['close_avg'].rolling(window=20).std() / df['close_avg'].rolling(window=20).mean()
        
        # 再次优化数据类型
        df = optimize_dtypes(df)
        
        return df

    def process_date(self, date: Union[str, pd.Timestamp], sector: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """处理单个日期的板块特征
        
        Args:
            date: 目标日期，格式为YYYYMMDD或pandas.Timestamp
            sector: 板块名称
            
        Returns:
            (是否成功, 特征DataFrame)
        """
        try:
            # 检查是否为特殊值'date_str'，是则直接跳过
            if date == 'date_str':
                logger.warning(f"跳过特殊值'date_str'，板块: {sector}")
                return False, None
                
            # 将Timestamp转换为字符串格式
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y%m%d')
                logger.debug(f"将Timestamp {date} 转换为字符串格式 {date_str}")
            else:
                date_str = str(date)
            
            # 生成板块特征
            df = self.calculate_sector_features_for_date(date_str, sector)
            
            if df.empty:
                logger.warning(f"日期 {date} 板块 {sector} 没有特征数据")
                return False, None
            
            # 添加日期和板块名称
            if 'date' not in df.columns:
                df['date'] = date_str
            if 'sector_name' not in df.columns:
                df['sector_name'] = sector
            
            # 保存日期特征文件
            success, file_path = self.save_features(df, sector, date_str)
            
            return success, df
        except Exception as e:
            logger.error(f"处理日期 {date} 板块 {sector} 特征时出错: {e}")
            return False, None

    def process_sector(
        self, 
        sector: str, 
        start_date: str, 
        end_date: str,
        force_update: bool = False,
        batch_size: Optional[int] = None
    ) -> Tuple[str, List[str]]:
        """处理单个板块特征生成
        
        Args:
            sector: 板块名称
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            force_update: 是否强制重新生成
            batch_size: 每批处理的日期数量
            
        Returns:
            (板块名称, 处理成功的日期列表)
        """
        batch_size = batch_size or self.batch_size
        processed_dates = []
        
        try:
            # 获取日期列表
            logger.debug(f"请求交易日历: {start_date} - {end_date}")
            date_range = get_trading_days(start_date, end_date)
            logger.debug(f"获取到交易日历，类型: {type(date_range)}, 长度: {len(date_range) if hasattr(date_range, '__len__') else '未知'}")
            
            if isinstance(date_range, pd.DataFrame):
                logger.warning(f"date_range是DataFrame，尝试转换为列表")
                if 'date' in date_range.columns:
                    date_range = date_range['date'].tolist()
                else:
                    date_range = list(date_range.index) if hasattr(date_range, 'index') else []
            
            # 安全检查空列表 - 使用len()而不是直接的布尔值检查
            if not hasattr(date_range, '__len__') or len(date_range) == 0:
                logger.warning(f"板块 {sector} 在日期范围 {start_date} - {end_date} 内没有交易日")
                return sector, []
            
            # 获取最后一个交易日，用于生成CSV
            try:
                last_trading_day = date_range[-1]
                logger.debug(f"最后一个交易日: {last_trading_day}")
            except (IndexError, TypeError) as e:
                logger.error(f"获取最后一个交易日失败: {e}, date_range类型: {type(date_range)}")
                last_trading_day = end_date
            
            # 确定需要处理的日期
            if not force_update:
                # 增量更新，只处理未生成特征的日期
                try:
                    logger.debug(f"请求增量更新日期: {sector}, {start_date} - {end_date}")
                    from feature_update import update_features_incrementally
                    dates_to_process = update_features_incrementally(sector, start_date, end_date, self.output_dir)
                    logger.debug(f"获取到增量更新日期，类型: {type(dates_to_process)}, 内容预览: {str(dates_to_process)[:100]} [...] (如果太长)")
                    
                    # 确保dates_to_process是列表类型
                    if isinstance(dates_to_process, pd.DataFrame):
                        logger.warning(f"update_features_incrementally返回了DataFrame而不是列表，尝试转换")
                        if 'date' in dates_to_process.columns:
                            dates_to_process = dates_to_process['date'].tolist()
                        else:
                            dates_to_process = list(dates_to_process.index) if hasattr(dates_to_process, 'index') else []
                    
                    # 如果增量更新日期为空，我们使用所有日期（为了演示）
                    if len(dates_to_process) == 0:
                        logger.warning(f"增量更新未返回任何日期，使用所有交易日")
                        dates_to_process = date_range
                        
                except (ImportError, AttributeError) as e:
                    # 如果feature_update模块不可用，处理所有日期
                    logger.warning(f"增量更新功能不可用({e})，将处理板块 {sector} 的所有日期")
                    dates_to_process = date_range
            else:
                # 强制更新，处理所有日期
                logger.debug(f"强制更新模式，处理所有日期")
                dates_to_process = date_range
            
            # 安全检查 - 确保dates_to_process是列表或可迭代对象，而不是DataFrame
            logger.debug(f"dates_to_process类型: {type(dates_to_process)}")
            if isinstance(dates_to_process, pd.DataFrame):
                logger.warning(f"dates_to_process仍然是DataFrame，将其转换为列表")
                try:
                    if 'date' in dates_to_process.columns:
                        dates_to_process = dates_to_process['date'].tolist()
                    else:
                        dates_to_process = dates_to_process.index.tolist() if hasattr(dates_to_process.index, 'tolist') else []
                    logger.debug(f"转换后dates_to_process类型: {type(dates_to_process)}, 长度: {len(dates_to_process)}")
                except Exception as e:
                    logger.error(f"转换dates_to_process失败: {e}")
                    dates_to_process = []
            
            # 使用len()检查是否为空而不是直接使用布尔值
            try:
                dates_count = len(dates_to_process)
                if dates_count == 0:
                    logger.info(f"板块 {sector} 在日期范围内所有日期的特征都已生成")
                    return sector, []
                logger.info(f"板块 {sector} 需要处理 {dates_count} 个日期")
            except Exception as e:
                logger.error(f"检查dates_to_process长度失败: {e}, 类型: {type(dates_to_process)}")
                return sector, []
            
            # 分批处理日期
            all_features = []
            last_day_features = None
            start_time = time.time()
            
            for i in range(0, len(dates_to_process), batch_size):
                batch_dates = dates_to_process[i:i+batch_size]
                
                # 过滤无效日期值
                filtered_batch_dates = []
                for date_item in batch_dates:
                    if date_item == 'date_str':
                        logger.warning(f"从批处理中移除特殊值'date_str'，板块: {sector}")
                        continue
                    if not isinstance(date_item, (str, pd.Timestamp)):
                        logger.warning(f"从批处理中移除非字符串/Timestamp类型: {date_item}, 类型: {type(date_item)}, 板块: {sector}")
                        continue
                    filtered_batch_dates.append(date_item)
                
                batch_results = []
                
                for date in filtered_batch_dates:
                    success, df = self.process_date(date, sector)
                    if success and df is not None:
                        batch_results.append(df)
                        processed_dates.append(date)
                        
                        # 如果是最后一个交易日，保存它的特征用于生成CSV
                        if date == last_trading_day:
                            last_day_features = df
                
                # 保存批次结果
                if len(batch_results) > 0:  # 使用len()检查是否为空
                    # 合并批次特征
                    batch_df = pd.concat(batch_results, ignore_index=True)
                    all_features.append(batch_df)
                    
                    # 输出进度
                    logger.debug(f"板块 {sector} 已处理 {len(processed_dates)}/{len(dates_to_process)} 个日期")
            
            # 合并所有特征
            if len(all_features) > 0:  # 使用len()检查是否为空
                # 合并为一个DataFrame
                all_df = pd.concat(all_features, ignore_index=True)
                
                # 排序并去重
                all_df = all_df.sort_values('date').drop_duplicates(subset=['date'])
                
                # 计算技术指标
                all_df = self.add_technical_indicators(all_df)
                
                # 保存完整特征文件为Parquet
                sector_dir = Path(self.output_dir) / sector
                sector_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存Parquet文件
                parquet_path = sector_dir / f"{sector}.parquet"
                from feature_compress import save_features as save_features_func
                save_features_func(
                    all_df, 
                    parquet_path, 
                    format='parquet',
                    compress=True,
                    optimize=True
                )
                
                # 只为最后一个交易日生成CSV文件
                if last_day_features is not None:
                    csv_path = sector_dir / f"{sector}_latest.csv"
                    from feature_compress import save_features as save_csv
                    save_csv(last_day_features, csv_path, format='csv', compress=False)
                    logger.debug(f"为最后一个交易日 {last_trading_day} 生成CSV文件: {csv_path}")
            
            # 记录耗时
            time_spent = time.time() - start_time
            logger.info(f"完成板块 {sector} 的特征数据处理，耗时 {time_spent:.2f} 秒，处理了 {len(processed_dates)} 个日期")
            
            return sector, processed_dates
        
        except Exception as e:
            logger.error(f"处理板块 {sector} 特征时出错: {e}")
            return sector, processed_dates
    
    def split_sectors_by_worker(self, sectors: List[str], max_workers: Optional[int] = None) -> List[List[str]]:
        """将板块按照工作进程数量分割
        
        按照均衡负载的原则分配板块到不同工作进程。
        
        Args:
            sectors: 板块列表
            max_workers: 最大工作进程数，None则使用默认值
            
        Returns:
            分组后的板块列表
        """
        if not sectors:
            return []
        
        # 确保max_workers有效
        if max_workers is None or max_workers <= 0:
            max_workers = self.max_workers
            # 如果self.max_workers也是None，设置一个合理的默认值
            if max_workers is None:
                max_workers = 4  # 默认使用4个工作进程
        
        # 如果只有一个进程或板块数小于进程数，直接返回
        if max_workers <= 1 or len(sectors) <= 1:
            return [sectors]
        
        # 调整工作进程数，不超过板块数
        actual_workers = min(max_workers, len(sectors))
        
        # 初始化每个工作进程的负载
        worker_loads = [0] * actual_workers
        sector_groups = [[] for _ in range(actual_workers)]
        
        # 按照字母表顺序排序板块，使分配更确定
        sorted_sectors = sorted(sectors)
        
        # 使用贪心算法分配板块
        for sector in sorted_sectors:
            # 找到负载最小的工作进程
            min_load_worker = worker_loads.index(min(worker_loads))
            # 分配板块
            sector_groups[min_load_worker].append(sector)
            # 更新负载（简单地使用板块数量作为负载）
            worker_loads[min_load_worker] += 1
        
        # 过滤掉空组
        return [group for group in sector_groups if group]
    
    def process_date_range(
        self, 
        start_date: str, 
        end_date: str,
        sectors: Optional[List[str]] = None,
        force_update: bool = False,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_format: str = 'parquet'  # 添加输出格式参数
    ) -> Dict[str, List[str]]:
        """处理日期范围内的所有板块特征
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            sectors: 板块列表，None则使用所有板块
            force_update: 是否强制重新生成
            max_workers: 最大工作进程数，None则使用默认值
            batch_size: 每批处理的日期数量，None则使用默认值
            output_format: 输出格式，'parquet'或'csv'，默认为'parquet'
            
        Returns:
            处理结果字典，键为板块名称，值为处理的日期列表
        """
        # 设置默认值
        if max_workers is None:
            max_workers = self.max_workers
        if max_workers is None:
            max_workers = 4  # 默认使用4个工作进程
            
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            batch_size = 30  # 默认每批处理30个日期
        
        # 设置输出格式
        self._output_format = output_format
        
        # 获取要处理的板块列表
        if sectors is None:
            try:
                from stock_sectors_mapping import get_tng_sectors
                sectors = get_tng_sectors()
            except (ImportError, AttributeError):
                logger.warning("无法导入get_all_sectors函数，使用默认板块列表")
                # 使用默认板块列表
                sectors = get_tng_sectors()
        
        if not sectors:
            logger.warning("没有要处理的板块")
            return {}
        
        logger.info(f"开始处理日期范围 {start_date} - {end_date} 的板块特征")
        logger.info(f"将处理 {len(sectors)} 个板块的特征数据")
        
        # 记录当前内存使用
        self.log_memory_usage()
        
        # 根据进程数划分板块
        sector_groups = self.split_sectors_by_worker(sectors, max_workers)
        
        # 使用多进程处理
        results = {}
        
        if max_workers > 1 and len(sectors) > 1:
            # 多进程处理
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for sector_group in sector_groups:
                    for sector in sector_group:
                        future = executor.submit(
                            self.process_sector,
                            sector=sector,
                            start_date=start_date,
                            end_date=end_date,
                            force_update=force_update,
                            batch_size=batch_size
                        )
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        sector, processed_dates = future.result()
                        results[sector] = processed_dates
                    except Exception as e:
                        logger.error(f"处理板块特征时出错: {e}")
        else:
            # 单进程处理
            for sector in sectors:
                sector, processed_dates = self.process_sector(
                    sector=sector,
                    start_date=start_date,
                    end_date=end_date,
                    force_update=force_update,
                    batch_size=batch_size
                )
                results[sector] = processed_dates
        
        # 记录最终内存使用
        self.log_memory_usage()
        
        # 返回处理结果
        return results
    
    def optimize_storage(self):
        """优化存储结构
        
        合并小文件，重新组织存储结构以提高访问效率。
        """
        logger.info("开始优化板块特征存储")
        
        # 获取所有板块目录
        sector_dirs = [d for d in Path(self.output_dir).glob("*") if d.is_dir() and not d.name.startswith('.')]
        
        for sector_dir in sector_dirs:
            sector = sector_dir.name
            
            # 查找日期文件
            date_files = list(sector_dir.glob(f"{sector}_*.parquet"))
            if not date_files:
                date_files = list(sector_dir.glob(f"{sector}_20*.parquet"))  # 查找以日期命名的parquet文件
            
            if len(date_files) <= 1:
                continue
            
            logger.info(f"合并板块 {sector} 的 {len(date_files)} 个小文件")
            
            # 加载所有文件
            dfs = []
            for file_path in date_files:
                try:
                    if file_path.suffix.lower() == '.parquet':
                        from feature_compress import load_features as load_features_func
                        df = load_features_func(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"加载文件 {file_path} 失败: {e}")
            
            if len(dfs) == 0:  # 使用len()而不是直接用dfs作为布尔值
                continue
            
            # 合并所有数据
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # 去重并排序
            if 'date' in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=['date']).sort_values('date')
            
            # 保存合并后的Parquet文件
            parquet_path = sector_dir / f"{sector}.parquet"
            
            try:
                # 保存Parquet格式
                from feature_compress import save_features as save_features_func
                success = save_features_func(merged_df, parquet_path, format='parquet', compress=True, optimize=True)
                
                if success:
                    # 保存最后一个交易日的CSV文件
                    try:
                        # 获取最后一个日期的数据
                        if 'date' in merged_df.columns:
                            last_date = merged_df['date'].max()
                            last_day_data = merged_df[merged_df['date'] == last_date]
                            
                            if not last_day_data.empty:
                                csv_path = sector_dir / f"{sector}_latest.csv"
                                from feature_compress import save_features as save_csv
                                save_csv(last_day_data, csv_path, format='csv', compress=False)
                                logger.debug(f"为最后一个交易日 {last_date} 生成CSV文件: {csv_path}")
                    except Exception as e:
                        logger.warning(f"生成最后一个交易日的CSV文件失败: {e}")
                    
                    # 删除原始文件
                    for file_path in date_files:
                        try:
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"删除文件 {file_path} 失败: {e}")
            except Exception as e:
                logger.error(f"合并板块 {sector} 的特征文件失败: {e}")
    
    def log_memory_usage(self):
        """记录内存使用情况"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        logger.info(f"内存使用: {memory_info.rss / (1024 * 1024):.2f} MB")

    def save_features(self, df: pd.DataFrame, sector: str, date: Optional[str] = None) -> Tuple[bool, Path]:
        """保存板块特征数据
        
        Args:
            df: 特征DataFrame
            sector: 板块名称
            date: 可选的日期标记，用于增量更新
            
        Returns:
            (是否保存成功, 保存的文件路径)
        """
        # 获取输出格式
        output_format = getattr(self, '_output_format', 'parquet')
        
        sector_dir = Path(self.output_dir) / sector
        sector_dir.mkdir(parents=True, exist_ok=True)
        
        if date:
            # 按日期保存单独的文件
            file_name = f"{sector}_{date}.{output_format}"
        else:
            # 保存完整特征
            file_name = f"{sector}.{output_format}"
        
        file_path = sector_dir / file_name
        
        try:
            from feature_compress import save_features as save_features_func
            success = save_features_func(
                df, 
                file_path, 
                format=output_format,
                compress=True,
                optimize=True
            )
            
            if success:
                logger.debug(f"成功保存板块 {sector} 的特征数据到 {file_path}")
                return True, file_path
            else:
                logger.error(f"保存板块 {sector} 的特征数据失败")
                return False, file_path
        except Exception as e:
            logger.error(f"保存板块 {sector} 的特征数据时发生错误: {e}")
            return False, file_path

    def load_features(self, sector: str) -> pd.DataFrame:
        """加载板块特征数据
        
        Args:
            sector: 板块名称
            
        Returns:
            板块特征DataFrame
        """
        sector_dir = Path(self.output_dir) / sector
        
        # 首先尝试加载Parquet文件
        parquet_path = sector_dir / f"{sector}.parquet"
        if parquet_path.exists():
            try:
                from feature_compress import load_features as load_features_func
                df = load_features_func(parquet_path)
                if not df.empty:
                    logger.debug(f"已从Parquet文件加载板块 {sector} 的特征数据")
                    return df
            except Exception as e:
                logger.warning(f"加载Parquet文件 {parquet_path} 失败: {e}")
        
        # 如果Parquet文件不存在或加载失败，尝试加载CSV文件
        csv_path = sector_dir / f"{sector}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    logger.debug(f"已从CSV文件加载板块 {sector} 的特征数据")
                    return df
            except Exception as e:
                logger.warning(f"加载CSV文件 {csv_path} 失败: {e}")
        
        # 如果都不存在，加载所有日期文件
        try:
            # 查找所有日期文件
            date_files = list(sector_dir.glob(f"{sector}_*.parquet"))
            
            if not date_files:
                date_files = list(sector_dir.glob(f"{sector}_*.csv"))
            
            if date_files:
                # 加载所有日期文件
                dfs = []
                for file_path in date_files:
                    try:
                        if file_path.suffix.lower() == '.parquet':
                            from feature_compress import load_features as load_features_func
                            df = load_features_func(file_path)
                        else:
                            df = pd.read_csv(file_path)
                        
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        logger.warning(f"加载文件 {file_path} 失败: {e}")
                
                if len(dfs) > 0:  # 使用len()检查而不是直接用dfs作为布尔值
                    # 合并所有数据框
                    df = pd.concat(dfs, ignore_index=True)
                    # 去重并排序
                    if 'date' in df.columns:
                        df = df.drop_duplicates(subset=['date']).sort_values('date')
                    logger.debug(f"已从 {len(date_files)} 个日期文件加载板块 {sector} 的特征数据")
                    return df
        except Exception as e:
            logger.warning(f"加载板块 {sector} 的日期文件失败: {e}")
        
        # 返回空DataFrame
        logger.warning(f"未找到板块 {sector} 的特征数据")
        return pd.DataFrame()


def generate_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sectors: Optional[List[str]] = None,
    force_update: bool = False,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_format: str = 'parquet'  # 指定输出格式
) -> Dict[str, List[str]]:
    """生成板块特征
    
    Args:
        start_date: 开始日期，格式为YYYYMMDD，None则使用当前日期
        end_date: 结束日期，格式为YYYYMMDD，None则使用当前日期
        sectors: 板块列表，None则使用所有板块
        force_update: 是否强制重新生成
        max_workers: 最大工作进程数，None则使用默认值
        batch_size: 每批处理的日期数量，None则使用默认值
        output_dir: 输出目录，None则使用默认目录
        output_format: 输出格式，'parquet'或'csv'，默认为'parquet'
        
    Returns:
        处理结果字典，键为板块名称，值为处理的日期列表
    """
    # 创建特征管理器
    feature_manager = SectorFeatureManager(
        output_dir=output_dir,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # 处理日期范围
    return feature_manager.process_date_range(
        start_date=start_date,
        end_date=end_date,
        sectors=sectors,
        force_update=force_update,
        max_workers=max_workers,
        batch_size=batch_size,
        output_format=output_format  # 传递输出格式
    )


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成板块特征数据')
    
    # 添加支持连字符和下划线的参数
    parser.add_argument('--start-date', '--start_date', type=str, help='开始日期，格式为YYYYMMDD')
    parser.add_argument('--end-date', '--end_date', type=str, help='结束日期，格式为YYYYMMDD')
    parser.add_argument('--sectors', type=str, help='板块列表，用逗号分隔')
    parser.add_argument('--force-update', '--force_update', action='store_true', help='强制重新生成所有特征')
    parser.add_argument('--max-workers', '--max_workers', type=int, help='最大工作进程数')
    parser.add_argument('--batch-size', '--batch_size', type=int, help='每批处理的日期数量')
    parser.add_argument('--output-dir', '--output_dir', type=str, help='输出目录')
    parser.add_argument('--use-parquet', '--use_parquet', action='store_true', default=True, 
                        help='使用Parquet格式保存所有特征数据(默认启用)')
    parser.add_argument('--csv-only', '--csv_only', action='store_true', 
                        help='只使用CSV格式保存特征(调试用，不建议用于大量数据)')
    
    args = parser.parse_args()
    
    # 默认使用当前日期
    if not args.start_date or not args.end_date:
        from datetime import datetime, timedelta
        today = datetime.now()
        if not args.end_date:
            args.end_date = today.strftime('%Y%m%d')
        if not args.start_date:
            start_date = today - timedelta(days=30)
            args.start_date = start_date.strftime('%Y%m%d')
    
    # 解析板块列表
    sectors = None
    if args.sectors:
        sectors = [s.strip() for s in args.sectors.split(',')]
    
    # 检查使用的存储格式
    if args.csv_only:
        output_format = 'csv'
        logger.info("将只使用CSV格式保存特征数据(通常用于调试)")
    else:
        output_format = 'parquet'
        logger.info("将使用Parquet格式保存完整特征数据，并为每个板块生成最后一个交易日的CSV文件用于快速查看")
    
    # 生成特征
    result = generate_features(
        start_date=args.start_date,
        end_date=args.end_date,
        sectors=sectors,
        force_update=args.force_update,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_format=output_format
    )
    
    # 输出处理结果
    total_sectors = len(result)
    total_dates = sum(len(dates) for _, dates in result.items())
    logger.info(f"完成所有板块特征处理，共处理了 {total_sectors} 个板块的 {total_dates} 个日期数据")
    
    # 优化存储
    feature_manager = SectorFeatureManager(
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    feature_manager.optimize_storage()


if __name__ == "__main__":
    main()

"""
使用示例:

1. 从项目根目录执行（推荐方式）:
   python -m to.generate_sector_features --start-date 20230101 --end-date 20230331

2. 使用提供的启动脚本:
   python run_sector_features.py --start-date 20230101 --end-date 20230331 --sectors "TGN人工智能,TGN新能源"

3. 或者在to目录内使用相对路径执行:
   cd to
   python generate_sector_features.py --start-date 20230101 --end-date 20230331

4. 使用fiximport解决导入问题:
   ```python
   import fiximport
   from to.generate_sector_features import generate_features
   
   # 然后使用函数
   result = generate_features(start_date="20230101", end_date="20230331")
   ```

5. 在代码中导入时请记住处理路径问题:
   ```python
   import sys
   from pathlib import Path
   
   # 添加项目根目录到sys.path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   
   # 然后导入模块
   from to.generate_sector_features import generate_features
   ```

注意:
- 此模块依赖于to/目录下的其他模块
- 请确保您正确地处理了导入路径问题
- 参考README_PATH.md了解更多关于项目路径使用规范的信息
"""