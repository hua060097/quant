#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票日线特征生成脚本
用于生成股票日线特征并保存到to/local_data/features/daily目录

特征生成流程:
1. 从ParquetDataStorage获取股票日线数据
2. 计算技术指标和其他特征
3. 保存为parquet格式，支持增量更新
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import traceback
import gc

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取路径工具函数
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    data_path = project_root / "to" / "local_data"
    if relative_path:
        return data_path / relative_path
    return data_path

def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# 特征计算函数
def calculate_technical_indicators(df):
    """
    计算技术指标特征
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        添加了技术指标的DataFrame
    """
    # 复制输入数据，避免修改原始数据
    result = df.copy()
    
    # 确保按日期和股票代码排序
    if 'date' in result.columns and 'stock_code' in result.columns:
        result = result.sort_values(['stock_code', 'date'])
    
    # 计算移动平均线
    result['ma5'] = result.groupby('stock_code')['close'].rolling(5).mean().reset_index(0, drop=True)
    result['ma10'] = result.groupby('stock_code')['close'].rolling(10).mean().reset_index(0, drop=True)
    result['ma20'] = result.groupby('stock_code')['close'].rolling(20).mean().reset_index(0, drop=True)
    result['ma60'] = result.groupby('stock_code')['close'].rolling(60).mean().reset_index(0, drop=True)
    result['ma120'] = result.groupby('stock_code')['close'].rolling(120).mean().reset_index(0, drop=True)
    
    # 计算RSI指标
    delta = result.groupby('stock_code')['close'].diff()
    
    # RSI6
    gain6 = delta.where(delta > 0, 0).groupby(result['stock_code']).rolling(window=6).mean().reset_index(0, drop=True)
    loss6 = -delta.where(delta < 0, 0).groupby(result['stock_code']).rolling(window=6).mean().reset_index(0, drop=True)
    rs6 = gain6 / loss6
    result['rsi6'] = 100 - (100 / (1 + rs6))
    
    # RSI12
    gain12 = delta.where(delta > 0, 0).groupby(result['stock_code']).rolling(window=12).mean().reset_index(0, drop=True)
    loss12 = -delta.where(delta < 0, 0).groupby(result['stock_code']).rolling(window=12).mean().reset_index(0, drop=True)
    rs12 = gain12 / loss12
    result['rsi12'] = 100 - (100 / (1 + rs12))
    
    # RSI24
    gain24 = delta.where(delta > 0, 0).groupby(result['stock_code']).rolling(window=24).mean().reset_index(0, drop=True)
    loss24 = -delta.where(delta < 0, 0).groupby(result['stock_code']).rolling(window=24).mean().reset_index(0, drop=True)
    rs24 = gain24 / loss24
    result['rsi24'] = 100 - (100 / (1 + rs24))
    
    # 计算MACD
    result['ema12'] = result.groupby('stock_code')['close'].ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
    result['ema26'] = result.groupby('stock_code')['close'].ewm(span=26, adjust=False).mean().reset_index(0, drop=True)
    result['macd'] = result['ema12'] - result['ema26']
    result['macd_signal'] = result.groupby('stock_code')['macd'].ewm(span=9, adjust=False).mean().reset_index(0, drop=True)
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    # 布林带指标
    result['bb_middle'] = result['ma20']
    result['bollinger_std'] = result.groupby('stock_code')['close'].rolling(20).std().reset_index(0, drop=True)
    result['bb_upper'] = result['bb_middle'] + 2 * result['bollinger_std']
    result['bb_lower'] = result['bb_middle'] - 2 * result['bollinger_std']
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    result['bb_position'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
    
    # 计算ATR
    high_low = result['high'] - result['low']
    high_close = abs(result['high'] - result['close'].shift(1))
    low_close = abs(result['low'] - result['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result['atr14'] = tr.groupby(result['stock_code']).rolling(window=14).mean().reset_index(0, drop=True)
    
    # 计算支撑位和阻力位
    result['support14'] = result.groupby('stock_code')['low'].rolling(14).min().reset_index(0, drop=True)
    result['resistance14'] = result.groupby('stock_code')['high'].rolling(14).max().reset_index(0, drop=True)
    
    # 计算成交量均线
    result['volume_ma20'] = result.groupby('stock_code')['volume'].rolling(20).mean().reset_index(0, drop=True)
    
    # 计算价格变化百分比
    result['change_pct'] = result.groupby('stock_code')['close'].pct_change() * 100
    
    # 计算波动率
    result['volatility20'] = result.groupby('stock_code')['close'].pct_change().rolling(20).std().reset_index(0, drop=True)
    result['volatility60'] = result.groupby('stock_code')['close'].pct_change().rolling(60).std().reset_index(0, drop=True)
    
    # 添加日期字符串列
    if 'date' in result.columns and pd.api.types.is_datetime64_any_dtype(result['date']):
        result['date_str'] = result['date'].dt.strftime('%Y%m%d')
    
    # 删除不需要的中间计算列
    if 'ema12' in result.columns:
        result = result.drop(columns=['ema12', 'ema26', 'bollinger_std'])
    
    # 确保所有列的数据类型正确
    float_cols = [
        'open', 'high', 'low', 'close', 'amount',
        'ma5', 'ma10', 'ma20', 'ma60', 'ma120',
        'rsi6', 'rsi12', 'rsi24',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr14', 'support14', 'resistance14',
        'volume_ma20', 'change_pct', 'volatility20', 'volatility60'
    ]
    
    # 确保所有浮点列为float32
    for col in float_cols:
        if col in result.columns:
            result[col] = result[col].astype('float32')
    
    # 确保volume为int32
    if 'volume' in result.columns:
        result['volume'] = result['volume'].astype('int32')
    
    return result

def optimize_dtypes(df):
    """
    优化DataFrame的数据类型，确保与目标格式一致
    
    Args:
        df: 输入DataFrame
        
    Returns:
        优化后的DataFrame
    """
    result = df.copy()
    
    # 确保这些列是float32类型
    float_cols = [
        'open', 'high', 'low', 'close', 'amount',
        'ma5', 'ma10', 'ma20', 'ma60', 'ma120',
        'rsi6', 'rsi12', 'rsi24',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr14', 'support14', 'resistance14',
        'volume_ma20', 'change_pct', 'volatility20', 'volatility60'
    ]
    
    for col in float_cols:
        if col in result.columns:
            result[col] = result[col].astype('float32')
    
    # 确保volume是int32类型
    if 'volume' in result.columns:
        result['volume'] = result['volume'].astype('int32')
    
    # 确保stock_code和date_str是object类型
    for col in ['stock_code', 'date_str']:
        if col in result.columns:
            result[col] = result[col].astype('object')
    
    return result

# 主要功能类
class StockFeatureGenerator:
    """股票特征生成器类"""
    
    def __init__(self, output_dir=None):
        """
        初始化特征生成器
        
        Args:
            output_dir: 输出目录，默认为to/local_data/features/daily
        """
        # 设置输出目录
        self.output_dir = Path(output_dir) if output_dir else get_data_path('features/daily')
        ensure_dir(self.output_dir)
        
        logger.info(f"特征生成器初始化完成，输出目录: {self.output_dir}")
    
    def get_data_storage(self):
        """获取ParquetDataStorage实例"""
        # 动态导入ParquetDataStorage类
        try:
            # 先尝试使用绝对导入
            sys.path.append(str(get_project_root()))
            from to.advanced_quant_data import ParquetDataStorage
        except ImportError:
            # 如果失败，尝试相对导入
            try:
                from advanced_quant_data import ParquetDataStorage
            except ImportError:
                logger.error("无法导入ParquetDataStorage类，请确保advanced_quant_data.py文件存在")
                return None
        
        # 创建实例
        return ParquetDataStorage()
    
    def get_stock_list(self):
        """获取股票列表"""
        storage = self.get_data_storage()
        if not storage:
            return []
        
        # 获取股票列表
        try:
            stock_list = storage.get_stock_list()
            if stock_list is not None:
                print(f"获取股票列表成功，共 {len(stock_list)} 只股票")
                return stock_list
            else:
                logger.warning("获取股票列表失败")
                return []
        except Exception as e:
            logger.error(f"获取股票列表时出错: {e}")
            return []
    
    def get_trading_days(self, start_date, end_date):
        """
        获取交易日列表
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            
        Returns:
            交易日列表
        """
        storage = self.get_data_storage()
        if not storage:
            return []
        
        try:
            calendar_df = storage.get_trading_calendar(start_date, end_date)
            if calendar_df is not None and not calendar_df.empty:
                if 'date' in calendar_df.columns:
                    dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                else:
                    # 尝试使用第一列作为日期
                    dates = calendar_df.iloc[:, 0].dt.strftime('%Y%m%d').tolist()
                return dates
            else:
                logger.warning("获取交易日历失败")
                return []
        except Exception as e:
            logger.error(f"获取交易日历时出错: {e}")
            return []
    
    def get_daily_data(self, stock_list, date):
        """
        获取指定日期的日线数据
        
        Args:
            stock_list: 股票列表
            date: 日期，格式为YYYYMMDD
            
        Returns:
            DataFrame: 日线数据
        """
        storage = self.get_data_storage()
        if not storage:
            return None
        
        try:
            # 获取日线数据
            # 传入日期前后各120天，确保有足够数据计算指标
            date_obj = datetime.strptime(date, '%Y%m%d')
            start_date = (date_obj - timedelta(days=150)).strftime('%Y%m%d')
            end_date = (date_obj + timedelta(days=60)).strftime('%Y%m%d')
            
            logger.info(f"获取{len(stock_list)}只股票从{start_date}到{end_date}的日线数据")
            df = storage.get_daily_data(stock_list, start_date, end_date)
            
            if df is None or df.empty:
                logger.warning(f"获取日线数据失败")
                return None
            
            logger.info(f"获取到日线数据，形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"获取日线数据时出错: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def process_features_for_date(self, date, stock_list=None):
        """
        处理指定日期的特征
        
        Args:
            date: 日期，格式为YYYYMMDD
            stock_list: 股票列表，默认为None表示所有股票
            
        Returns:
            bool: 是否成功
        """
        # 获取股票列表（如果未提供）
        if stock_list is None:
            stock_list = self.get_stock_list()
            if not stock_list:
                logger.error("无法获取股票列表")
                return False
        
        logger.info(f"开始处理日期 {date} 的特征，共 {len(stock_list)} 只股票")
        
        # 获取日线数据
        daily_data = self.get_daily_data(stock_list, date)
        if daily_data is None:
            logger.error(f"无法获取日期 {date} 的日线数据")
            return False
        
        # 计算技术指标
        try:
            logger.info("开始计算技术指标")
            features_df = calculate_technical_indicators(daily_data)
            logger.info(f"计算技术指标完成，生成 {features_df.shape[1]} 个特征")
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 筛选当前日期的数据
        try:
            # 确保日期列是datetime类型
            if 'date' in features_df.columns and not pd.api.types.is_datetime64_any_dtype(features_df['date']):
                features_df['date'] = pd.to_datetime(features_df['date'])
            
            target_date = pd.to_datetime(date, format='%Y%m%d')
            date_features_df = features_df[features_df['date'] == target_date].copy()
            
            if date_features_df.empty:
                logger.warning(f"日期 {date} 没有数据")
                return False
            
            logger.info(f"筛选出日期 {date} 的数据，共 {len(date_features_df)} 行")
        except Exception as e:
            logger.error(f"筛选日期数据时出错: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 确保所有特征列存在
        required_columns = [
            'open', 'high', 'low', 'close', 'volume', 'amount',
            'ma5', 'ma10', 'ma20', 'ma60', 'ma120',
            'rsi6', 'rsi12', 'rsi24',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr14', 'support14', 'resistance14',
            'volume_ma20', 'change_pct', 'volatility20', 'volatility60',
            'stock_code', 'date_str'
        ]
        
        for col in required_columns:
            if col not in date_features_df.columns:
                logger.warning(f"缺少必需的列: {col}，将添加默认值")
                if col in ['stock_code', 'date_str']:
                    date_features_df[col] = ''
                else:
                    date_features_df[col] = 0.0
        
        # 只保留需要的列
        date_features_df = date_features_df[required_columns]
        
        # 优化数据类型
        try:
            logger.info("优化数据类型")
            date_features_df = optimize_dtypes(date_features_df)
        except Exception as e:
            logger.error(f"优化数据类型时出错: {e}")
            logger.error(traceback.format_exc())
            # 继续执行，不返回False
        
        # 保存特征
        try:
            # 构建输出文件路径
            output_file = self.output_dir / f"{date}.parquet"
            
            # 保存为parquet格式
            date_features_df.to_parquet(output_file, index=False, compression='snappy')
            
            logger.info(f"成功保存日期 {date} 的特征到 {output_file}，共 {len(date_features_df)} 行")
            return True
        except Exception as e:
            logger.error(f"保存特征时出错: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_features(self, start_date, end_date, stock_list=None, force_update=False):
        """
        生成指定日期范围的特征
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            stock_list: 股票列表，默认为None表示所有股票
            force_update: 是否强制更新已存在的特征
            
        Returns:
            bool: 是否全部成功
        """
        # 获取交易日列表
        trading_days = self.get_trading_days(start_date, end_date)
        if not trading_days:
            logger.error(f"无法获取从 {start_date} 到 {end_date} 的交易日列表")
            return False
        
        logger.info(f"将处理 {len(trading_days)} 个交易日")
        
        # 处理每个交易日
        success_count = 0
        for date in trading_days:
            # 检查是否已存在特征文件（如果不强制更新）
            output_file = self.output_dir / f"{date}.parquet"
            if not force_update and output_file.exists():
                logger.info(f"日期 {date} 的特征文件已存在，跳过")
                success_count += 1
                continue
            
            # 处理特征
            logger.info(f"处理日期 {date} 的特征")
            success = self.process_features_for_date(date, stock_list)
            
            if success:
                success_count += 1
            else:
                logger.warning(f"处理日期 {date} 的特征失败")
            
            # 释放内存
            gc.collect()
        
        logger.info(f"特征生成完成，共 {len(trading_days)} 个交易日，成功 {success_count} 个")
        return success_count == len(trading_days)
    
    def check_features(self, date=None):
        """
        检查特征文件
        
        Args:
            date: 日期，格式为YYYYMMDD，如果为None则检查所有特征文件
            
        Returns:
            DataFrame: 特征信息摘要
        """
        if date:
            # 检查指定日期的特征文件
            file_path = self.output_dir / f"{date}.parquet"
            if not file_path.exists():
                logger.warning(f"日期 {date} 的特征文件不存在")
                return None
            
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"文件 {file_path} 包含 {len(df)} 行，{df.shape[1]} 列")
                return df
            except Exception as e:
                logger.error(f"读取特征文件 {file_path} 失败: {e}")
                return None
        else:
            # 检查所有特征文件
            parquet_files = list(self.output_dir.glob("*.parquet"))
            if not parquet_files:
                logger.warning("未找到特征文件")
                return None
            
            # 统计信息
            file_info = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    file_info.append({
                        'date': file_path.stem,
                        'rows': len(df),
                        'columns': df.shape[1],
                        'stocks': df['stock_code'].nunique() if 'stock_code' in df.columns else 'N/A',
                        'size_mb': file_path.stat().st_size / 1024 / 1024
                    })
                except Exception as e:
                    logger.error(f"读取特征文件 {file_path} 失败: {e}")
            
            # 转换为DataFrame并返回
            if file_info:
                info_df = pd.DataFrame(file_info)
                info_df = info_df.sort_values('date')
                return info_df
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成股票日线特征")
    parser.add_argument("--start_date", type=str, help="开始日期，格式为YYYYMMDD")
    parser.add_argument("--end_date", type=str, help="结束日期，格式为YYYYMMDD")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认为to/local_data/features/daily")
    parser.add_argument("--force", action="store_true", help="强制更新已存在的特征")
    parser.add_argument("--check", action="store_true", help="只检查特征，不生成")
    parser.add_argument("--date", type=str, help="检查指定日期的特征，与--check一起使用")
    
    args = parser.parse_args()
    
    # 创建特征生成器
    generator = StockFeatureGenerator(args.output_dir)
    
    if args.check:
        # 检查模式
        if args.date:
            generator.check_features(args.date)
        else:
            info_df = generator.check_features()
            if info_df is not None:
                logger.info(f"特征文件摘要:\n{info_df}")
    else:
        # 生成特征
        if args.start_date and args.end_date:
            generator.generate_features(args.start_date, args.end_date, force_update=args.force)
        else:
            logger.error("请提供开始日期和结束日期")
            parser.print_help()

if __name__ == "__main__":
    main()