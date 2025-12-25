"""
数据加载模块
使用专业的量化交易数据加载逻辑，不使用任何模拟数据
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# 添加路径以导入现有模块
main_sector_path = Path(__file__).parent.parent / "main_sector_stock"
sys.path.insert(0, str(main_sector_path))

# 导入专业的数据加载器
try:
    from ml_sector_strategy import DataLoader as MLDataLoader, DataConfig
    logger.info("成功导入专业量化交易数据加载器")
except ImportError as e:
    logger.error(f"无法导入专业数据加载器: {e}")
    raise ImportError("专业量化交易系统必须使用真实数据，无法导入数据加载器")

class DataLoader:
    """专业量化交易数据加载器 - 直接使用ml_sector_strategy.py的逻辑"""

    def __init__(self):
        # 创建默认配置
        self.config = DataConfig()

        # 直接使用ml_sector_strategy.py的DataLoader
        self.ml_loader = MLDataLoader(self.config)
        self.logger = logger

        self.logger.info("专业量化交易数据加载器初始化完成")
        
    def load_all_data(self, start_date: str, end_date: str):
        """
        加载所有需要的数据
        
        Args:
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            
        Returns:
            dict: 包含所有数据的字典
        """
        self.logger.info(f"开始加载数据: {start_date} - {end_date}")
        
        try:
            # 直接使用缓存数据
            self.logger.info("使用缓存数据加载")

            # 直接从缓存加载数据
            cache_dir = Path("../main_sector_stock/ml_cache")

            # 查找合适的缓存文件
            stock_cache_files = list(cache_dir.glob("stock_features_*.parquet"))
            sector_cache_files = list(cache_dir.glob("sector_features_*.parquet"))

            # 选择包含目标日期范围的缓存文件（可能需要组合多个文件）
            stock_features = None
            sector_features = None

            # 转换目标日期
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # 尝试找到单个包含完整范围的文件
            for cache_file in sorted(stock_cache_files, key=lambda x: x.stat().st_size, reverse=True):
                try:
                    df = pd.read_parquet(cache_file)
                    if not df.empty and 'date' in df.columns:
                        # 检查日期范围
                        df_start = df['date'].min()
                        df_end = df['date'].max()
                        if df_start <= start_dt and df_end >= end_dt:
                            # 过滤到目标日期范围
                            stock_features = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
                            self.logger.info(f"使用股票缓存文件: {cache_file.name}")
                            break
                except Exception as e:
                    self.logger.warning(f"读取股票缓存文件失败 {cache_file}: {e}")
                    continue

            # 如果没有单个文件包含完整范围，尝试组合文件
            if stock_features is None:
                self.logger.info("尝试组合多个缓存文件...")
                stock_dfs = []
                for cache_file in sorted(stock_cache_files, key=lambda x: x.stat().st_size, reverse=True):
                    try:
                        df = pd.read_parquet(cache_file)
                        if not df.empty and 'date' in df.columns:
                            # 检查是否有重叠的日期范围
                            df_start = df['date'].min()
                            df_end = df['date'].max()
                            if df_end >= start_dt and df_start <= end_dt:
                                # 过滤到目标日期范围
                                filtered_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
                                if not filtered_df.empty:
                                    stock_dfs.append(filtered_df)
                                    self.logger.info(f"添加股票缓存文件: {cache_file.name}")
                    except Exception as e:
                        self.logger.warning(f"读取股票缓存文件失败 {cache_file}: {e}")
                        continue

                if stock_dfs:
                    stock_features = pd.concat(stock_dfs, ignore_index=True).drop_duplicates()
                    self.logger.info(f"组合了 {len(stock_dfs)} 个股票缓存文件")

            # 对板块数据使用相同的逻辑
            for cache_file in sorted(sector_cache_files, key=lambda x: x.stat().st_size, reverse=True):
                try:
                    df = pd.read_parquet(cache_file)
                    if not df.empty and 'date' in df.columns:
                        # 检查日期范围
                        df_start = df['date'].min()
                        df_end = df['date'].max()
                        if df_start <= start_dt and df_end >= end_dt:
                            # 过滤到目标日期范围
                            sector_features = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
                            self.logger.info(f"使用板块缓存文件: {cache_file.name}")
                            break
                except Exception as e:
                    self.logger.warning(f"读取板块缓存文件失败 {cache_file}: {e}")
                    continue

            # 如果没有单个文件包含完整范围，尝试组合文件
            if sector_features is None:
                self.logger.info("尝试组合多个板块缓存文件...")
                sector_dfs = []
                for cache_file in sorted(sector_cache_files, key=lambda x: x.stat().st_size, reverse=True):
                    try:
                        df = pd.read_parquet(cache_file)
                        if not df.empty and 'date' in df.columns:
                            # 检查是否有重叠的日期范围
                            df_start = df['date'].min()
                            df_end = df['date'].max()
                            if df_end >= start_dt and df_start <= end_dt:
                                # 过滤到目标日期范围
                                filtered_df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
                                if not filtered_df.empty:
                                    sector_dfs.append(filtered_df)
                                    self.logger.info(f"添加板块缓存文件: {cache_file.name}")
                    except Exception as e:
                        self.logger.warning(f"读取板块缓存文件失败 {cache_file}: {e}")
                        continue

                if sector_dfs:
                    sector_features = pd.concat(sector_dfs, ignore_index=True).drop_duplicates()
                    self.logger.info(f"组合了 {len(sector_dfs)} 个板块缓存文件")

            if stock_features is None or sector_features is None:
                raise ValueError("未找到合适的缓存数据")

            # 预处理数据
            stock_features = self._preprocess_stock_features(stock_features)
            sector_features = self._preprocess_sector_features(sector_features)

            self.logger.info(f"缓存数据加载完成: 个股 {len(stock_features)} 条, 板块 {len(sector_features)} 条")

            return {
                'stock_features': stock_features,
                'sector_features': sector_features,
                'market_features': None
            }

        except Exception as e:
            self.logger.error(f"缓存数据加载失败: {e}")
            raise


    
    def _preprocess_stock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理个股特征数据"""
        if df is None or df.empty:
            return df

        # 基础清理
        df = df.copy()

        # 清理异常的future_return_2d数据
        df = self._clean_future_returns(df)

        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # 处理无穷值
        df = df.replace([np.inf, -np.inf], 0)

        self.logger.info(f"个股特征预处理完成: {len(df)} 条记录")
        return df

    def _clean_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理异常的future_return_2d数据"""
        if 'future_return_2d' not in df.columns or 'volume' not in df.columns:
            return df

        cleaned_count = 0

        # 识别每只股票的上市日期并清理上市前的异常数据
        for stock_code in df['stock_code'].unique():
            stock_mask = df['stock_code'] == stock_code
            stock_data = df[stock_mask].copy().sort_values('date')

            # 找到第一个有交易量的日期
            trading_data = stock_data[stock_data['volume'] > 0]

            if not trading_data.empty:
                listing_date = trading_data['date'].iloc[0]

                # 清理上市前的异常future_return_2d（>100%或<-50%）
                pre_listing_mask = (df['stock_code'] == stock_code) & (df['date'] < listing_date)
                abnormal_mask = pre_listing_mask & (abs(df['future_return_2d']) > 1.0)

                if abnormal_mask.sum() > 0:
                    self.logger.info(f"清理股票 {stock_code} 上市前 {abnormal_mask.sum()} 个异常future_return_2d")
                    df.loc[abnormal_mask, 'future_return_2d'] = 0.0
                    cleaned_count += abnormal_mask.sum()

        if cleaned_count > 0:
            self.logger.info(f"总共清理了 {cleaned_count} 个异常future_return_2d数据")

        return df

    def _preprocess_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理板块特征数据"""
        if df is None or df.empty:
            return df

        # 基础清理
        df = df.copy()

        # 数据质量检查
        self.logger.info("=== 板块数据质量检查 ===")
        self.logger.info(f"数据形状: {df.shape}")
        self.logger.info(f"可用字段: {list(df.columns)}")

        # 检查数值型字段的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.logger.info(f"数值型字段: {list(numeric_cols)}")

        key_features = ['avg_pct_chg', 'limit_up_count', 'volatility', 'volume_ratio_5']
        for feature in key_features:
            if feature in df.columns:
                non_zero = (df[feature] != 0).sum()
                total = len(df)
                self.logger.info(f"{feature}: 非零数据 {non_zero}/{total} ({non_zero/total*100:.1f}%)")
                if non_zero > 0:
                    self.logger.info(f"  范围: {df[feature].min():.4f} ~ {df[feature].max():.4f}")
            else:
                self.logger.warning(f"缺少关键特征: {feature}")

        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # 处理无穷值
        df = df.replace([np.inf, -np.inf], 0)

        self.logger.info(f"板块特征预处理完成: {len(df)} 条记录")
        return df
    
    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """获取交易日期列表"""
        # 使用专业数据加载器的交易日历
        if hasattr(self.ml_loader, 'trading_calendar') and self.ml_loader.trading_calendar:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            valid_dates = []
            for date_str in self.ml_loader.trading_calendar:
                date_obj = pd.to_datetime(date_str)
                if start_dt <= date_obj <= end_dt:
                    valid_dates.append(date_str)

            return valid_dates
        else:
            raise ValueError("无法获取交易日历，专业量化系统必须使用真实交易日历")

    def split_train_test_dates(self, start_date: str, end_date: str, test_ratio: float = 0.2):
        """
        划分训练和测试日期

        Args:
            start_date: 开始日期
            end_date: 结束日期
            test_ratio: 测试集比例

        Returns:
            tuple: (train_start, train_end, test_start, test_end)
        """
        # 使用专业数据加载器的日期划分逻辑
        train_dates, test_dates = self.ml_loader.split_train_test_dates()

        if train_dates and test_dates:
            return train_dates[0], train_dates[-1], test_dates[0], test_dates[-1]
        else:
            raise ValueError("无法划分训练测试日期，专业量化系统必须使用真实交易日历")
