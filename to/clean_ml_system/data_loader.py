"""
数据加载模块
专业量化交易数据加载器，重新实现数据加载逻辑，不依赖外部模块
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """数据配置类 - 重新实现"""
    start_date: str = "20230601"
    end_date: str = "20250328"
    train_ratio: float = 0.9
    stock_features_dir: str = "local_data/features/daily"
    sector_features_dir: str = "local_data/sector_features"
    market_status_dir: str = "local_data/market_status"
    board_limit_dir:str = "local_data/sector_features_board"
    cache_dir: str = "../main_sector_stock/ml_cache"

    # 模型参数
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    random_state: int = 42
    n_jobs: int = -1
    learning_rate: float = 0.1
    test_size: float = 0.2
    cv_folds: int = 5

class DataLoader:
    """专业量化交易数据加载器 - 重新实现版本"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.logger = logger

        # 设置项目根目录
        self.project_root = Path(__file__).parent.parent

        # 设置缓存目录
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置数据目录
        self.stock_features_dir = self._get_data_path(self.config.stock_features_dir)
        self.sector_features_dir = self._get_data_path(self.config.sector_features_dir)
        self.market_status_dir = self._get_data_path(self.config.market_status_dir)
        self.board_limit_dir = self._get_data_path(self.config.board_limit_dir)

        # 交易日历
        self.trading_calendar = None
        self._load_trading_calendar()

        self.logger.info("专业量化交易数据加载器初始化完成")

    def _get_data_path(self, relative_path: str) -> Path:
        """获取数据路径"""
        # 尝试多个可能的路径
        possible_paths = [
            self.project_root / relative_path,
            self.project_root / "to" / relative_path,
            Path(relative_path)
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # 如果都不存在，返回第一个路径（可能需要创建）
        return possible_paths[0]

    def _load_trading_calendar(self):
        """严格使用advanced_quant_data加载交易日历"""
        try:
            import sys
            from pathlib import Path
            # 添加to目录到路径
            project_root = Path(__file__).parent.parent.parent
            to_dir = str(project_root / "to")
            if to_dir not in sys.path:
                sys.path.append(to_dir)
            from advanced_quant_data import ParquetDataStorage
            storage = ParquetDataStorage()
            # 严格使用get_trading_calendar方法
            calendar_df = storage.get_trading_calendar('20220101', '20251231')
            if calendar_df.empty:
                raise Exception("交易日历数据为空")
            if 'date_str' not in calendar_df.columns:
                raise Exception("交易日历缺少date_str列")
            # 提取交易日列表
            self.trading_calendar = calendar_df['date_str'].tolist()
            if not self.trading_calendar:
                raise Exception("交易日历列表为空")
            print(f"IntegratedDataLoader加载交易日历成功: {len(self.trading_calendar)} 个交易日")
            print(f"交易日历范围: {min(self.trading_calendar)} - {max(self.trading_calendar)}")
        except Exception as e:
            error_msg = f"严格加载交易日历失败: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def split_train_test_dates(self) -> Tuple[List[str], List[str]]:
        """按9:1划分训练和预测日期"""
        if not self.trading_calendar:
            raise ValueError("交易日历为空，无法划分日期")

        # 筛选指定日期范围内的交易日
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)

        valid_dates = []
        for date_str in self.trading_calendar:
            date_obj = pd.to_datetime(date_str)
            if start_date <= date_obj <= end_date:
                valid_dates.append(date_str)

        if not valid_dates:
            raise ValueError(f"在{self.config.start_date}至{self.config.end_date}范围内没有交易日")

        # 按9:1划分
        split_point = int(len(valid_dates) * self.config.train_ratio)
        train_dates = valid_dates[:split_point]
        test_dates = valid_dates[split_point:]

        self.logger.info(f"日期划分完成: 训练期{len(train_dates)}天({train_dates[0]}~{train_dates[-1]}), "
                        f"预测期{len(test_dates)}天({test_dates[0]}~{test_dates[-1]})")

        return train_dates, test_dates

    def load_stock_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载个股特征数据 - 重新实现版本"""
        cache_key = f"stock_features_{start_date}_{end_date}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        # 检查缓存
        if cache_file.exists():
            self.logger.info(f"从缓存加载个股特征: {cache_file}")
            return pd.read_parquet(cache_file)

        self.logger.info(f"加载个股特征数据: {start_date} 至 {end_date}")

        # 获取日期范围内的交易日
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        trading_dates = [d for d in self.trading_calendar if start_dt <= pd.to_datetime(d) <= end_dt]

        if not trading_dates:
            raise ValueError(f"在{start_date}至{end_date}范围内没有交易日")

        # 加载每日特征文件 - 修正文件路径格式
        all_features = []
        for date_str in trading_dates:
            # 修正：使用正确的文件名格式 {date}.parquet
            feature_file = self.stock_features_dir / f"{date_str}.parquet"
            if feature_file.exists():
                try:
                    daily_features = pd.read_parquet(feature_file)
                    if not daily_features.empty:
                        daily_features['date'] = pd.to_datetime(date_str)
                        all_features.append(daily_features)
                        self.logger.debug(f"加载{date_str}特征数据: {len(daily_features)}条记录")
                except Exception as e:
                    self.logger.warning(f"读取{date_str}特征文件失败: {e}")
            else:
                self.logger.warning(f"特征文件不存在: {feature_file}")

        if not all_features:
            raise ValueError(f"在{start_date}至{end_date}范围内没有找到有效的特征文件")

        # 合并所有特征
        combined_features = pd.concat(all_features, ignore_index=True)
        self.logger.info(f"成功加载个股特征: {len(combined_features)}条记录, {len(combined_features.columns)}个特征")

        # 保存缓存
        combined_features.to_parquet(cache_file)

        return combined_features

    def load_sector_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载板块特征数据 - 重新实现版本"""
        cache_key = f"sector_features_{start_date}_{end_date}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        # 检查缓存
        if cache_file.exists():
            self.logger.info(f"从缓存加载板块特征: {cache_file}")
            return pd.read_parquet(cache_file)

        self.logger.info(f"加载板块特征数据: {start_date} 至 {end_date}")

        # 获取日期范围内的交易日
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        trading_dates = [d for d in self.trading_calendar if start_dt <= pd.to_datetime(d) <= end_dt]

        if not trading_dates:
            raise ValueError(f"在{start_date}至{end_date}范围内没有交易日")

        # 修正：加载板块特征文件 - 基于ml_sector_strategy.py的正确逻辑
        # 获取所有板块目录
        if not self.sector_features_dir.exists():
            self.logger.warning(f"板块特征目录不存在: {self.sector_features_dir}")
            return pd.DataFrame()

        sector_dirs = [d for d in self.sector_features_dir.iterdir() if d.is_dir()]
        if not sector_dirs:
            self.logger.warning("未找到任何板块特征目录")
            return pd.DataFrame()

        all_sector_data = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        for sector_dir in sector_dirs:
            sector_code = sector_dir.name
            self.logger.debug(f"处理板块目录: {sector_code}")

            # 查找主特征文件 - 修正：使用正确的文件名格式
            main_feature_file = sector_dir / f"{sector_code}_features.parquet"
            if not main_feature_file.exists():
                main_feature_file = sector_dir / f"{sector_code}.parquet"

            if main_feature_file.exists():
                try:
                    sector_df = pd.read_parquet(main_feature_file)
                    self.logger.debug(f"板块{sector_code}原始数据形状: {sector_df.shape}")

                    # 筛选日期范围 - 修正：使用正确的日期列处理逻辑
                    if 'date_str' in sector_df.columns:
                        sector_df['date'] = pd.to_datetime(sector_df['date_str'])
                    elif 'date' in sector_df.columns:
                        sector_df['date'] = pd.to_datetime(sector_df['date'])
                    else:
                        self.logger.warning(f"板块{sector_code}缺少日期列，跳过")
                        continue

                    # 筛选日期范围
                    filtered_df = sector_df[(sector_df['date'] >= start_dt) & (sector_df['date'] <= end_dt)]
                    if not filtered_df.empty:
                        filtered_df = filtered_df.copy()
                        filtered_df['sector_code'] = sector_code

                        # 生成因子计算所需的衍生字段
                        filtered_df = self._generate_sector_derived_features(filtered_df)

                        all_sector_data.append(filtered_df)
                        self.logger.debug(f"板块{sector_code}筛选后数据: {len(filtered_df)}条记录")

                except Exception as e:
                    self.logger.warning(f"读取板块{sector_code}特征文件失败: {e}")
            else:
                self.logger.debug(f"板块{sector_code}特征文件不存在: {main_feature_file}")

        if not all_sector_data:
            self.logger.warning(f"在{start_date}至{end_date}范围内没有找到板块特征数据，返回空DataFrame")
            return pd.DataFrame()

        # 合并所有板块特征
        combined_features = pd.concat(all_sector_data, ignore_index=True)
        self.logger.info(f"成功加载板块特征: {len(combined_features)}条记录, {len(combined_features.columns)}个特征")

        # 保存缓存
        combined_features.to_parquet(cache_file)

        return combined_features

    def _generate_sector_derived_features(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成板块因子计算所需的衍生特征
        只生成必需字段，避免不必要的复杂计算
        """
        df = sector_df.copy()

        # 确保有日期列
        if 'date' not in df.columns:
            self.logger.warning("板块数据缺少date列")
            return df

        # 生成日收益率字段（因子计算必需）
        if 'pct_chg' in df.columns:
            df['return_1d'] = df['pct_chg'].astype(float)
        elif 'close' in df.columns:
            df['return_1d'] = df['close'].pct_change() * 100
        else:
            df['return_1d'] = 0.0

        # 生成别名字段（兼容不同因子定义）
        df['return'] = df['return_1d']
        df['daily_return'] = df['return_1d']

        # 生成多日收益率（动量因子需要）
        if 'close' in df.columns:
            for window in [3, 5, 10, 20]:
                col = f'return_{window}d'
                df[col] = (df['close'] / df['close'].shift(window) - 1.0) * 100
        else:
            # 如果没有close列，设置为0
            for window in [3, 5, 10, 20]:
                df[f'return_{window}d'] = 0.0

        # 填充NaN值
        for col in ['return_1d', 'return', 'daily_return', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(float)

        return df

    def load_market_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载市场状态特征数据，自动扩展日期范围以获取足够的历史数据计算90个交易日阈值

        Args:
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD

        Returns:
            pd.DataFrame: 市场特征数据
        """
        self.logger.info(f"加载市场特征数据: {start_date} 至 {end_date}")

        try:
            # 构建市场状态数据文件路径
            market_file = self.market_status_dir / "daily" / "market_status.json"

            if not market_file.exists():
                self.logger.warning(f"市场状态文件不存在: {market_file}")
                return pd.DataFrame()

            # 读取JSON数据
            with open(market_file, 'r', encoding='utf-8') as f:
                market_data = json.load(f)

            # 扩展日期范围以获取足够的历史数据
            extended_start_date, extended_end_date = self._extend_date_range_for_threshold(
                start_date, end_date, list(market_data.keys())
            )

            # 转换为DataFrame格式
            market_records = []

            for date_str, daily_data in market_data.items():
                # 使用扩展的日期范围
                if extended_start_date <= date_str <= extended_end_date:
                    # 提取市场整体特征
                    market_record = {
                        'date': pd.to_datetime(date_str, format='%Y%m%d'),
                        'date_str': date_str,
                        'market_type': daily_data.get('market_type', 0),
                        'market_description': daily_data.get('market_description', 0),
                        'market_cycle': daily_data.get('market_cycle', 0),
                        'cycle_description': daily_data.get('cycle_description', 0),
                        'trend_strength': daily_data.get('trend_strength', 0.0),
                    }

                    # 提取信号特征
                    signals = daily_data.get('signals', {})
                    market_record.update({
                        'trading_advice': signals.get('trading_advice', ''),
                        'risk_level': signals.get('risk_level', ''),
                        'signal_strength': signals.get('strength', 0)
                    })

                    # 提取个股状态特征（以399006.SZ为代表）
                    individual_status = daily_data.get('individual_status', {})
                    if '399006.SZ' in individual_status:
                        stock_data = individual_status['399006.SZ']

                        # 技术评分特征
                        market_record.update({
                            'trend_score': stock_data.get('trend_score', 0.0),
                            'position_score': stock_data.get('position_score', 0.0),
                            'volume_score': stock_data.get('volume_score', 0.0),
                            'momentum_score': stock_data.get('momentum_score', 0.0),
                            'volatility_score': stock_data.get('volatility_score', 0.0),
                            'structure_score': stock_data.get('structure_score', 0.0),
                            'cycle_score': stock_data.get('cycle_score', 0.0),
                            'total_score': stock_data.get('total_score', 0.0)
                        })

                        # 技术指标特征
                        technical_data = stock_data.get('technical_data', {})
                        market_record.update({
                            'close': technical_data.get('close', 0.0),
                            'prev_close': technical_data.get('prev_close', 0.0),
                            'ma5': technical_data.get('ma5', 0.0),
                            'ma10': technical_data.get('ma10', 0.0),
                            'ma20': technical_data.get('ma20', 0.0),
                            'ma60': technical_data.get('ma60', 0.0),
                            'rsi6': technical_data.get('rsi6', 50.0),
                            'rsi12': technical_data.get('rsi12', 50.0),
                            'macd': technical_data.get('macd', 0.0),
                            'macd_signal': technical_data.get('macd_signal', 0.0),
                            'macd_hist': technical_data.get('macd_hist', 0.0),
                            'kdj_k': technical_data.get('kdj_k', 0.0),
                            'kdj_d': technical_data.get('kdj_d', 0.0),
                            'bb_width': technical_data.get('bb_width', 0.0),
                            'bb_position': technical_data.get('bb_position', 0.0)
                        })

                        # 市场数据特征
                        market_data_info = stock_data.get('market_data', {})
                        market_record.update({
                            'open': market_data_info.get('open', 0.0),
                            'high': market_data_info.get('high', 0.0),
                            'low': market_data_info.get('low', 0.0),
                            'volume': market_data_info.get('volume', 0.0),
                            'daily_change_pct': market_data_info.get('daily_change_pct', 0.0)
                        })

                        # 支撑阻力和盈亏比特征
                        market_record.update({
                            'support': stock_data.get('support', 0.0),
                            'resistance': stock_data.get('resistance', 0.0),
                            'profit_loss_ratio_14': stock_data.get('profit_loss_ratio_14', 1.0),
                            'profit_loss_ratio_30': stock_data.get('profit_loss_ratio_30', 1.0),
                            'change_percent': stock_data.get('change_percent', 0.0)
                        })

                    market_records.append(market_record)

            if not market_records:
                self.logger.warning(f"未找到日期范围 {start_date}-{end_date} 内的市场数据")
                return pd.DataFrame()

            # 转换为DataFrame
            market_df = pd.DataFrame(market_records)

            # 数据质量检查和预处理
            market_df = self._preprocess_market_features(market_df, start_date, end_date)

            # 添加日期范围日志，帮助调试
            if not market_df.empty and 'date' in market_df.columns:
                date_min = market_df['date'].min()
                date_max = market_df['date'].max()
                self.logger.info(f"成功加载市场特征: {len(market_df)}条记录, {len(market_df.columns)}个特征")
                self.logger.info(f"市场数据日期范围: {date_min} 至 {date_max}")
            else:
                self.logger.info(f"成功加载市场特征: {len(market_df)}条记录, {len(market_df.columns)}个特征")

            return market_df

        except Exception as e:
            self.logger.error(f"加载市场特征数据失败: {e}")
            raise

    def _preprocess_market_features(self, market_df: pd.DataFrame,
                                  original_start_date: str, original_end_date: str) -> pd.DataFrame:
        """
        预处理市场特征数据

        Args:
            market_df: 原始市场特征数据

        Returns:
            pd.DataFrame: 预处理后的市场特征数据
        """
        if market_df.empty:
            return market_df

        try:
            self.logger.info("=== 市场数据质量检查 ===")
            self.logger.info(f"数据形状: {market_df.shape}")

            # 确保日期列正确
            if 'date' in market_df.columns:
                market_df['date'] = pd.to_datetime(market_df['date'])

            # 数值型字段列表
            numeric_fields = [
                'market_type', 'market_description', 'market_cycle', 'cycle_description',
                'trend_strength', 'signal_strength', 'trend_score', 'position_score',
                'volume_score', 'momentum_score', 'volatility_score', 'structure_score',
                'cycle_score', 'total_score', 'close', 'prev_close', 'ma5', 'ma10',
                'ma20', 'ma60', 'rsi6', 'rsi12', 'macd', 'macd_signal', 'macd_hist',
                'kdj_k', 'kdj_d', 'bb_width', 'bb_position', 'open', 'high', 'low',
                'volume', 'daily_change_pct', 'support', 'resistance',
                'profit_loss_ratio_14', 'profit_loss_ratio_30', 'change_percent'
            ]

            # 转换数值型字段
            for field in numeric_fields:
                if field in market_df.columns:
                    market_df[field] = pd.to_numeric(market_df[field], errors='coerce')

            # 填充缺失值
            market_df = market_df.fillna(0)

            # 添加市场状态分级字段（使用扩展的数据计算阈值）
            market_df = self._add_market_state_levels(market_df)

            # 不再过滤回原始日期范围，保留扩展后的完整数据
            # 这样可以确保市场数据覆盖股票数据的所有日期（包括扩展的历史数据）
            original_data = market_df.copy()

            # 数据质量统计
            self.logger.info(f"市场数据预处理完成: {len(original_data)} 条记录")
            self.logger.info(f"可用字段: {list(original_data.columns)}")

            # 检查关键字段的数据质量
            key_fields = ['trend_strength', 'total_score', 'close', 'volume']
            for field in key_fields:
                if field in original_data.columns:
                    non_zero_count = (original_data[field] != 0).sum()
                    total_count = len(original_data)
                    self.logger.info(f"{field}: 非零数据 {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.1f}%)")
                    if non_zero_count > 0:
                        field_min = original_data[field].min()
                        field_max = original_data[field].max()
                        self.logger.info(f"  范围: {field_min:.4f} ~ {field_max:.4f}")

            self.logger.info("市场特征预处理完成: {} 条记录".format(len(original_data)))

            return original_data

        except Exception as e:
            self.logger.error(f"市场特征预处理失败: {e}")
            raise

    def _add_market_state_levels(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        为市场数据添加状态分级字段，使用最近90个交易日的动态阈值

        Args:
            market_df: 市场特征数据

        Returns:
            pd.DataFrame: 添加了市场状态分级的数据
        """
        if market_df.empty:
            return market_df

        try:
            # 为每个日期计算基于90个交易日的动态阈值
            market_df = market_df.sort_values('date').reset_index(drop=True)

            # 初始化分级列
            market_df['market_volume_level'] = 'medium'
            market_df['market_change_level'] = 'small_rise'

            for i, row in market_df.iterrows():
                current_date = row['date']

                # 获取用于计算阈值的90个交易日数据
                threshold_data = self._get_threshold_data_for_date(current_date, market_df)

                if threshold_data is not None and len(threshold_data) > 0:
                    # 计算成交量等级
                    if 'volume' in threshold_data.columns and 'volume' in row:
                        volume_level = self._classify_volume_level(row['volume'], threshold_data['volume'])
                        market_df.at[i, 'market_volume_level'] = volume_level

                    # 计算涨跌幅等级
                    if 'daily_change_pct' in threshold_data.columns and 'daily_change_pct' in row:
                        change_level = self._classify_change_level(row['daily_change_pct'], threshold_data['daily_change_pct'])
                        market_df.at[i, 'market_change_level'] = change_level

            self.logger.info("已添加基于90个交易日动态阈值的市场状态分级字段")
            return market_df

        except Exception as e:
            self.logger.error(f"添加市场状态分级失败: {e}")
            # 如果失败，添加默认值
            market_df['market_volume_level'] = 'medium'
            market_df['market_change_level'] = 'small_rise'
            return market_df

    def _get_threshold_data_for_date(self, current_date: pd.Timestamp, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        获取用于计算阈值的90个交易日数据（基于实际交易日，不要求连续）

        Args:
            current_date: 当前日期
            market_df: 完整的市场数据

        Returns:
            用于计算阈值的90个交易日数据
        """
        try:
            # 获取所有可用的交易日期（已排序）
            all_dates = sorted(market_df['date'].unique())
            current_idx = None

            # 找到当前日期在所有交易日中的位置
            for i, date in enumerate(all_dates):
                if date == current_date:
                    current_idx = i
                    break

            if current_idx is None:
                # 如果找不到当前日期，使用所有数据
                self.logger.debug(f"未找到日期 {current_date.strftime('%Y-%m-%d')}，使用所有可用数据")
                return market_df

            # 计算90个交易日的范围
            target_days = 90

            # 优先使用历史数据（包含当前日期）
            start_idx = max(0, current_idx - target_days + 1)
            end_idx = current_idx + 1

            actual_history_days = end_idx - start_idx

            # 如果历史数据不足90个交易日，使用未来数据补充
            if actual_history_days < target_days:
                needed_future_days = target_days - actual_history_days
                future_end_idx = min(len(all_dates), current_idx + 1 + needed_future_days)
                end_idx = future_end_idx

                self.logger.debug(f"日期 {current_date.strftime('%Y-%m-%d')} 历史交易日不足({actual_history_days}天)，"
                                f"使用未来{needed_future_days}个交易日补充")

            # 获取选定的交易日期
            selected_dates = all_dates[start_idx:end_idx]
            threshold_data = market_df[market_df['date'].isin(selected_dates)]

            self.logger.debug(f"日期 {current_date.strftime('%Y-%m-%d')} 使用 {len(threshold_data)} 个交易日计算阈值 "
                            f"(范围: {selected_dates[0].strftime('%Y-%m-%d')} 至 {selected_dates[-1].strftime('%Y-%m-%d')})")

            return threshold_data

        except Exception as e:
            self.logger.warning(f"获取阈值数据失败: {e}")
            return market_df

    def _classify_volume_level(self, volume: float, volume_series: pd.Series) -> str:
        """
        基于90个交易日的成交量数据分类成交量等级

        Args:
            volume: 当前成交量
            volume_series: 用于计算阈值的成交量序列

        Returns:
            成交量等级
        """
        try:
            volume_quantiles = volume_series.quantile([0.2, 0.4, 0.6, 0.8])

            if volume <= volume_quantiles[0.2]:
                return 'very_low'
            elif volume <= volume_quantiles[0.4]:
                return 'medium_low'
            elif volume <= volume_quantiles[0.6]:
                return 'medium'
            elif volume <= volume_quantiles[0.8]:
                return 'medium_high'
            else:
                return 'very_high'

        except Exception as e:
            self.logger.warning(f"成交量分级失败: {e}")
            return 'medium'

    def _classify_change_level(self, change_pct: float, change_series: pd.Series) -> str:
        """
        基于90个交易日的涨跌幅数据分类涨跌幅等级

        Args:
            change_pct: 当前涨跌幅
            change_series: 用于计算阈值的涨跌幅序列

        Returns:
            涨跌幅等级
        """
        try:
            change_quantiles = change_series.quantile([0.05, 0.15, 0.35, 0.65, 0.85, 0.95])

            if change_pct <= change_quantiles[0.05]:
                return 'big_fall'
            elif change_pct <= change_quantiles[0.15]:
                return 'medium_fall'
            elif change_pct <= change_quantiles[0.35]:
                return 'small_fall'
            elif change_pct <= change_quantiles[0.65]:
                return 'flat'
            elif change_pct <= change_quantiles[0.85]:
                return 'small_rise'
            elif change_pct <= change_quantiles[0.95]:
                return 'medium_rise'
            else:
                return 'big_rise'

        except Exception as e:
            self.logger.warning(f"涨跌幅分级失败: {e}")
            return 'small_rise'

    def _extend_date_range_for_threshold(self, start_date: str, end_date: str,
                                       available_dates: List[str]) -> Tuple[str, str]:
        """
        扩展日期范围以获取足够的历史数据计算90个交易日阈值（基于实际交易日）

        Args:
            start_date: 原始开始日期
            end_date: 原始结束日期
            available_dates: 所有可用的交易日期列表

        Returns:
            扩展后的(开始日期, 结束日期)
        """
        try:
            # 过滤并排序可用的交易日期
            valid_dates = [d for d in available_dates if d.isdigit() and len(d) == 8]
            valid_dates.sort()

            if not valid_dates:
                return start_date, end_date

            # 找到原始开始日期在交易日中的位置
            start_idx = None
            for i, date in enumerate(valid_dates):
                if date >= start_date:
                    start_idx = i
                    break

            if start_idx is None:
                # 如果开始日期晚于所有可用交易日，使用最后的交易日
                return valid_dates[-1], end_date

            # 找到原始结束日期在交易日中的位置
            end_idx = len(valid_dates) - 1
            for i, date in enumerate(valid_dates):
                if date > end_date:
                    end_idx = i - 1
                    break

            # 计算需要的历史交易日数量（120个交易日，与股票数据扩展保持一致）
            target_trading_days = 120

            # 向前扩展开始日期以获取足够的历史交易日
            # 修正：应该是 start_idx - target_trading_days，而不是 +1
            extended_start_idx = max(0, start_idx - target_trading_days)
            extended_start_date = valid_dates[extended_start_idx]

            # 检查历史交易日是否足够
            actual_history_days = start_idx - extended_start_idx
            if actual_history_days < target_trading_days:
                # 如果历史交易日不足，向后扩展结束日期
                needed_future_days = target_trading_days - actual_history_days
                extended_end_idx = min(len(valid_dates) - 1, end_idx + needed_future_days)
                extended_end_date = valid_dates[extended_end_idx]

                self.logger.info(f"历史交易日不足({actual_history_days}天)，使用未来{needed_future_days}个交易日补充")
            else:
                extended_end_date = end_date

            self.logger.info(f"交易日范围扩展: {start_date}-{end_date} -> {extended_start_date}-{extended_end_date} "
                           f"(共{len([d for d in valid_dates if extended_start_date <= d <= extended_end_date])}个交易日)")

            return extended_start_date, extended_end_date

        except Exception as e:
            self.logger.warning(f"扩展交易日范围失败: {e}")
            return start_date, end_date

    def load_all_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        加载所有需要的数据 - 包含个股、板块和市场特征

        Args:
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD

        Returns:
            dict: 包含所有数据的字典
                - stock_features: 个股特征数据
                - sector_features: 板块特征数据
                - market_features: 市场特征数据
        """
        self.logger.info(f"开始加载数据: {start_date} - {end_date}")

        try:
            # 加载个股特征数据
            stock_features = self.load_stock_features(start_date, end_date)

            # 加载板块特征数据
            # sector_features = self.load_sector_features(start_date, end_date)

            # 加载市场特征数据
            market_features = self.load_market_features(start_date, end_date)

            # 预处理数据
            stock_features = self._preprocess_stock_features(stock_features)
            # sector_features = self._preprocess_sector_features(sector_features)
            sector_features = None
            # market_features已在load_market_features中预处理

            self.logger.info(f"数据加载完成: 个股 {len(stock_features)} 条, 市场 {len(market_features)} 条")

            return {
                'stock_features': stock_features,
                'sector_features': sector_features,
                'market_features': market_features
            }

        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise


    def _add_board_structure_features_to_df(self, sector_df: pd.DataFrame,
                                          sector_code: str, sector_dir: Path) -> pd.DataFrame:
        """为板块数据添加涨跌停板结构特征 - 基于sector_stock_analysis.py板块分析"""
        try:
            # 查找板块涨跌停分析数据文件（在sector_features_board目录下）
            board_features_dir = self.board_limit_dir / f"{sector_code}"
            board_analysis_file = board_features_dir / f"{sector_code}_board_analysis.parquet"

            if not board_analysis_file.exists():
                self.logger.debug(f"板块{sector_code}涨跌停分析文件不存在: {board_analysis_file}")
                return sector_df

            # 读取板块涨跌停分析数据
            board_df = pd.read_parquet(board_analysis_file)

            # 确保日期列格式一致
            if 'date' in board_df.columns:
                board_df['date'] = pd.to_datetime(board_df['date']).dt.strftime('%Y%m%d')
            elif 'date_obj' in board_df.columns:
                board_df['date'] = pd.to_datetime(board_df['date_obj']).dt.strftime('%Y%m%d')

            # 确保sector_df的日期格式
            if 'date' in sector_df.columns:
                if sector_df['date'].dtype == 'object':
                    # 如果是字符串格式，尝试转换
                    sector_df['date_str'] = sector_df['date']
                else:
                    # 如果是datetime格式，转换为字符串
                    sector_df['date_str'] = pd.to_datetime(sector_df['date']).dt.strftime('%Y%m%d')

            # 定义需要合并的板块涨跌停特征
            board_features = [
                # 基础统计特征
                "stock_count", "up_count", "down_count", "flat_count",
                "avg_pct_chg", "avg_turnover", "avg_amplitude",
                # 涨跌停统计特征
                "limit_up_count", "limit_down_count", "near_limit_up_count",
                "near_limit_down_count", "one_word_limit_up_count",
                "broken_limit_up_count", "limit_open_count",
                # 连板特征
                "max_consecutive_limit_up", "high_elasticity_count",
                "board_counts_1板", "board_counts_2板", "board_counts_3板",
                "board_counts_4板", "board_counts_5板", "board_counts_6板",
                "board_counts_7板", "board_counts_8板", "board_counts_9板",
                "board_counts_10板及以上",
                # 溢价特征
                "yesterday_limit_up_premium", "yesterday_max_limit_up_premium",
                "yesterday_broken_limit_premium",
                # 市值分布特征
                "market_cap_distribution_large", "market_cap_distribution_medium",
                "market_cap_distribution_small",
                # 连板分布特征
                "limit_board_distribution_large", "limit_board_distribution_medium",
                "limit_board_distribution_small",
                # 综合评价特征
                "heat_index"
            ]

            # 选择需要的特征列
            available_features = [f for f in board_features if f in board_df.columns]
            if not available_features:
                self.logger.warning(f"板块{sector_code}涨跌停分析文件中未找到预期特征")
                return sector_df

            # 准备合并用的数据
            merge_cols = ['date'] + available_features
            board_merge_df = board_df[merge_cols].copy()

            # 合并数据
            merged_df = sector_df.merge(
                board_merge_df,
                left_on='date_str',
                right_on='date',
                how='left',
                suffixes=('', '_board')
            )

            # 删除重复的date列
            if 'date_board' in merged_df.columns:
                merged_df = merged_df.drop('date_board', axis=1)

            # 填充缺失值
            for feature in available_features:
                if feature in merged_df.columns:
                    if feature in ['avg_pct_chg', 'avg_turnover', 'avg_amplitude',
                                 'yesterday_limit_up_premium', 'yesterday_max_limit_up_premium',
                                 'yesterday_broken_limit_premium', 'heat_index']:
                        merged_df[feature] = merged_df[feature].fillna(0.0)
                    else:
                        merged_df[feature] = merged_df[feature].fillna(0)

            self.logger.debug(f"成功为板块{sector_code}添加{len(available_features)}个涨跌停特征")
            return merged_df

        except Exception as e:
            self.logger.warning(f"为板块{sector_code}添加涨跌停特征失败: {e}")
            return sector_df

    def _calculate_derived_features(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """计算缺失的衍生特征"""
        try:
            df = sector_df.copy()

            # 1. 计算参与度特征
            if all(col in df.columns for col in ['up_count', 'down_count', 'stock_count']):
                # 计算up_ratio和down_ratio
                df['up_ratio'] = df['up_count'] / df['stock_count'].replace(0, 1)
                df['down_ratio'] = df['down_count'] / df['stock_count'].replace(0, 1)

                # 计算flat_ratio
                if 'flat_count' in df.columns:
                    df['flat_ratio'] = df['flat_count'] / df['stock_count'].replace(0, 1)
                else:
                    df['flat_ratio'] = 1 - df['up_ratio'] - df['down_ratio']

                # 计算participation_entropy（信息熵）
                def calculate_entropy(row):
                    ratios = [row['up_ratio'], row['down_ratio'], row['flat_ratio']]
                    ratios = [r for r in ratios if r > 0]  # 过滤掉0值
                    if not ratios:
                        return 0
                    entropy = -sum(r * np.log2(r) for r in ratios)
                    return entropy

                df['participation_entropy'] = df.apply(calculate_entropy, axis=1)

            # 2. 计算资金流动特征
            if 'amount' in df.columns:
                # 计算fund_flow_momentum（资金流动动量）
                df['amount_ma5'] = df.groupby('sector_code')['amount'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
                df['fund_flow_momentum'] = df['amount'] / df['amount_ma5'].replace(0, 1)

                # 计算volume_intensity（成交量密集度）
                if 'volume' in df.columns and 'volatility_20d' in df.columns:
                    df['volume_intensity'] = df['volume'] / df['volatility_20d'].replace(0, 1)
                elif 'volume' in df.columns:
                    # 如果没有volatility_20d，使用volume的标准差作为替代
                    df['volume_std'] = df.groupby('sector_code')['volume'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
                    df['volume_intensity'] = df['volume'] / df['volume_std'].replace(0, 1)

            # 3. 计算涨停质量特征
            if all(col in df.columns for col in ['limit_up_count', 'stock_count']):
                df['limit_up_ratio'] = df['limit_up_count'] / df['stock_count'].replace(0, 1)

            if all(col in df.columns for col in ['limit_down_count', 'stock_count']):
                df['limit_down_ratio'] = df['limit_down_count'] / df['stock_count'].replace(0, 1)

            if all(col in df.columns for col in ['one_word_limit_up_count', 'limit_up_count']):
                df['one_word_limit_ratio'] = df['one_word_limit_up_count'] / df['limit_up_count'].replace(0, 1)

            if all(col in df.columns for col in ['broken_limit_up_count', 'limit_up_count']):
                total_limits = df['limit_up_count'] + df['broken_limit_up_count']
                df['broken_ratio'] = df['broken_limit_up_count'] / total_limits.replace(0, 1)

            # 4. 计算综合指标
            if all(col in df.columns for col in ['avg_pct_chg', 'limit_up_ratio']):
                # 板块动量评分
                df['sector_momentum_score'] = df['avg_pct_chg'] * (1 + df['limit_up_ratio'] * 10)

            if all(col in df.columns for col in ['avg_pct_chg', 'avg_turnover', 'avg_amplitude']):
                # 板块强度指标
                df['sector_strength'] = (
                    df['avg_pct_chg'] *
                    (1 + df['avg_turnover']/100) *
                    (1 + df['avg_amplitude']/100)
                )

            # 5. 计算技术指标特征（如果基础数据可用）
            if 'close' in df.columns:
                # 计算移动平均线比率
                for short, long in [(5, 10), (5, 20), (10, 20), (20, 60), (60, 120)]:
                    short_ma_col = f'ma{short}'
                    long_ma_col = f'ma{long}'
                    ratio_col = f'ma{short}_{long}_ratio'

                    if short_ma_col in df.columns and long_ma_col in df.columns:
                        df[ratio_col] = df[short_ma_col] / df[long_ma_col].replace(0, 1)
                    elif 'close' in df.columns:
                        # 如果没有预计算的MA，现场计算
                        df[short_ma_col] = df.groupby('sector_code')['close'].rolling(short, min_periods=1).mean().reset_index(0, drop=True)
                        df[long_ma_col] = df.groupby('sector_code')['close'].rolling(long, min_periods=1).mean().reset_index(0, drop=True)
                        df[ratio_col] = df[short_ma_col] / df[long_ma_col].replace(0, 1)

            # 6. 处理无穷值和NaN值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # 替换无穷值
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # 填充NaN值
                df[col] = df[col].fillna(0)

            self.logger.debug(f"成功计算衍生特征，新增特征数: {len(df.columns) - len(sector_df.columns)}")
            return df

        except Exception as e:
            self.logger.warning(f"计算衍生特征失败: {e}")
            return sector_df



    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期列表"""
        if not self.trading_calendar:
            raise ValueError("交易日历为空")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        valid_dates = []
        for date_str in self.trading_calendar:
            date_obj = pd.to_datetime(date_str)
            if start_dt <= date_obj <= end_dt:
                valid_dates.append(date_str)

        return valid_dates

    def _preprocess_stock_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理个股特征数据"""
        if df is None or df.empty:
            return df

        # 基础清理
        df = df.copy()

        # 清理异常的future_return_2d数据
        # df = self._clean_future_returns(df)

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

    def load_stock_sector_mapping(self) -> Optional[Dict[str, List[str]]]:
        """
        加载股票-板块映射数据

        Returns:
            Dict[str, List[str]]: 股票代码到板块列表的映射，格式为 {"股票代码": ["板块1", "板块2", ...]}
            如果加载失败返回 None
        """
        try:
            # 构建JSON文件路径
            json_file_path = self.project_root / "local_data" / "sector_mapping" / "stock_sector_mapping.json"

            if not json_file_path.exists():
                self.logger.warning(f"股票板块映射文件不存在: {json_file_path}")
                return None

            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)

            # 验证数据格式
            if not isinstance(mapping_data, dict):
                self.logger.error("股票板块映射文件格式错误：根对象不是字典")
                return None

            # 验证数据内容
            valid_mapping = {}
            for stock_code, sectors in mapping_data.items():
                if isinstance(sectors, list) and all(isinstance(s, str) for s in sectors):
                    valid_mapping[stock_code] = sectors
                else:
                    self.logger.warning(f"股票 {stock_code} 的板块数据格式错误，已跳过")

            self.logger.info(f"✅ 股票板块映射加载成功: {len(valid_mapping)} 只股票")
            return valid_mapping

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON文件解析失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"加载股票板块映射失败: {e}")
            return None
