#!/usr/bin/env python3
"""
完整涨跌停数据收集器 (基于Tushare官方文档)
获取A股每日涨跌停、炸板数据情况，包含所有输出参数
数据来源：https://tushare.pro/document/2?doc_id=298

支持的数据类型：
- U: 涨停数据
- D: 跌停数据
- Z: 炸板数据

完整输出参数：
trade_date, ts_code, industry, name, close, pct_chg, amount, limit_amount,
float_mv, total_mv, turnover_ratio, fd_amount, first_time, last_time,
open_times, up_stat, limit_times, limit
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

class CompleteLimitDataCollector:
    """
    完整涨跌停数据收集器
    基于Tushare官方文档：https://tushare.pro/document/2?doc_id=298
    """

    def __init__(self, token='gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b'):
        """初始化收集器"""
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()

        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 数据保存路径
        self.data_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # API调用间隔（避免频率限制）
        self.api_interval = 0.3  # 300ms间隔，更保守

        # 支持的涨跌停类型
        self.limit_types = {
            'U': '涨停',
            'D': '跌停',
            'Z': '炸板'
        }

        # 完整的输出字段（基于官方文档）
        self.output_fields = [
            'trade_date',      # 交易日期
            'ts_code',         # 股票代码
            'industry',        # 所属行业
            'name',           # 股票名称
            'close',          # 收盘价
            'pct_chg',        # 涨跌幅
            'amount',         # 成交额
            'limit_amount',   # 板上成交金额(跌停专用)
            'float_mv',       # 流通市值
            'total_mv',       # 总市值
            'turnover_ratio', # 换手率
            'fd_amount',      # 封单金额
            'first_time',     # 首次封板时间
            'last_time',      # 最后封板时间
            'open_times',     # 炸板次数/开板次数
            'up_stat',        # 涨停统计
            'limit_times',    # 连板数
            'limit'           # 涨跌停标识
        ]

        self.logger.info("完整涨跌停数据收集器初始化完成")
        self.logger.info(f"支持数据类型: {self.limit_types}")
        self.logger.info(f"输出字段数: {len(self.output_fields)}")
    
    def get_complete_limit_data_by_date_range(self, start_date='20230101', end_date='20250623'):
        """
        按日期范围获取完整的涨跌停数据（包含涨停、跌停、炸板）

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            tuple: (涨停数据, 跌停数据, 炸板数据)
        """
        self.logger.info(f"开始获取完整涨跌停数据: {start_date} 到 {end_date}")

        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        # 存储各类型数据
        all_data = {
            'U': [],  # 涨停
            'D': [],  # 跌停
            'Z': []   # 炸板
        }

        # 按月份分批获取数据
        current_date = start_dt
        while current_date <= end_dt:
            # 计算当月的结束日期
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1, day=1)

            month_end = min(next_month - timedelta(days=1), end_dt)

            month_start_str = current_date.strftime('%Y%m%d')
            month_end_str = month_end.strftime('%Y%m%d')

            self.logger.info(f"获取 {month_start_str} 到 {month_end_str} 的数据")

            # 获取三种类型的数据
            for limit_type, type_name in self.limit_types.items():
                try:
                    data = self._get_limit_data_safe(limit_type, month_start_str, month_end_str)
                    if data is not None and not data.empty:
                        all_data[limit_type].append(data)
                        self.logger.info(f"{type_name}数据: {len(data)} 条")

                    time.sleep(self.api_interval)

                except Exception as e:
                    self.logger.error(f"获取{type_name}数据失败 {month_start_str}-{month_end_str}: {e}")
                    time.sleep(1)

            # 移动到下个月
            current_date = next_month

        # 合并各类型数据
        combined_data = {}
        for limit_type, type_name in self.limit_types.items():
            if all_data[limit_type]:
                combined_data[limit_type] = pd.concat(all_data[limit_type], ignore_index=True)
                self.logger.info(f"总{type_name}数据: {len(combined_data[limit_type])} 条")
            else:
                combined_data[limit_type] = pd.DataFrame()

        return combined_data['U'], combined_data['D'], combined_data['Z']
    
    def _get_limit_data_safe(self, limit_type, start_date, end_date, max_retries=5):
        """
        安全获取涨跌停数据（带重试机制）

        Args:
            limit_type: 'U' 涨停, 'D' 跌停, 'Z' 炸板
            start_date: 开始日期
            end_date: 结束日期
            max_retries: 最大重试次数
        """
        for retry in range(max_retries):
            try:
                # 使用完整字段列表获取数据
                data = self.pro.limit_list_d(
                    limit_type=limit_type,
                    start_date=start_date,
                    end_date=end_date,
                    fields=','.join(self.output_fields)
                )

                print(f"dada:{data}")

                # 验证数据完整性
                if data is not None and not data.empty:
                    missing_fields = set(self.output_fields) - set(data.columns)
                    if missing_fields:
                        self.logger.warning(f"缺失字段: {missing_fields}")

                return data

            except Exception as e:
                self.logger.warning(f"获取{self.limit_types.get(limit_type, limit_type)}数据失败 (重试 {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    time.sleep(10 ** retry)  # 指数退避
                else:
                    self.logger.error(f"获取数据最终失败: {e}")
                    return None
    
    def process_and_save_complete_data(self, limit_up_data, limit_down_data, limit_zhaban_data):
        """
        处理和保存完整的涨跌停数据（包含炸板）- 增量更新模式

        增量更新逻辑：
        1. 如果新数据日期与原文件重复，用新数据覆盖原数据
        2. 如果新数据日期不在原文件中，增量添加到原文件
        3. 涨停、跌停、炸板、合并数据都采用增量更新

        Args:
            limit_up_data: 涨停数据
            limit_down_data: 跌停数据
            limit_zhaban_data: 炸板数据
        """
        self.logger.info("开始处理和保存完整涨跌停数据（增量更新模式）")

        processed_data = {}

        # 处理涨停数据（增量更新）
        if not limit_up_data.empty:
            processed_data['up'] = self._process_limit_data(limit_up_data, 'up')
            up_file = self.data_dir / "limit_up_data.parquet"

            # 增量更新保存
            final_up_data = self._incremental_update(up_file, processed_data['up'], 'up')
            processed_data['up'] = final_up_data

            self.logger.info(f"涨停数据已增量更新到: {up_file} ({len(final_up_data)}条)")

            # 🆕 打印涨停数据文件的日期范围
            self._print_data_date_range(up_file, "limit_up_data.parquet")

            # 按年份保存（增量更新）
            self._save_by_year_incremental(final_up_data, 'up')

        # 处理跌停数据（增量更新）
        if not limit_down_data.empty:
            processed_data['down'] = self._process_limit_data(limit_down_data, 'down')
            down_file = self.data_dir / "limit_down_data.parquet"

            # 增量更新保存
            final_down_data = self._incremental_update(down_file, processed_data['down'], 'down')
            processed_data['down'] = final_down_data

            self.logger.info(f"跌停数据已增量更新到: {down_file} ({len(final_down_data)}条)")

            # 🆕 打印跌停数据文件的日期范围
            self._print_data_date_range(down_file, "limit_down_data.parquet")

            # 按年份保存（增量更新）
            self._save_by_year_incremental(final_down_data, 'down')

        # 处理炸板数据（增量更新）
        if not limit_zhaban_data.empty:
            processed_data['zhaban'] = self._process_limit_data(limit_zhaban_data, 'zhaban')
            zhaban_file = self.data_dir / "limit_zhaban_data.parquet"

            # 增量更新保存
            final_zhaban_data = self._incremental_update(zhaban_file, processed_data['zhaban'], 'zhaban')
            processed_data['zhaban'] = final_zhaban_data

            self.logger.info(f"炸板数据已增量更新到: {zhaban_file} ({len(final_zhaban_data)}条)")

            # 🆕 打印炸板数据文件的日期范围
            self._print_data_date_range(zhaban_file, "limit_zhaban_data.parquet")

            # 按年份保存（增量更新）
            self._save_by_year_incremental(final_zhaban_data, 'zhaban')

        # 合并保存所有数据（增量更新）
        if processed_data:
            all_data_list = list(processed_data.values())
            if all_data_list:
                all_data = pd.concat(all_data_list, ignore_index=True)
                all_file = self.data_dir / "all_limit_data.parquet"

                # 增量更新合并数据
                final_all_data = self._incremental_update(all_file, all_data, 'all')

                self.logger.info(f"合并数据已增量更新到: {all_file} ({len(final_all_data)}条)")

                # 🆕 打印all_limit_data.parquet的日期范围
                self._print_data_date_range(all_file, "all_limit_data.parquet")

                return final_all_data

        return pd.DataFrame()
    
    def _process_limit_data(self, data, limit_type):
        """
        处理涨跌停数据，标准化字段格式并添加增强特征

        Args:
            data: 原始数据
            limit_type: 'up', 'down', 'zhaban'
        """
        processed = data.copy()

        # 标准化日期格式
        processed['trade_date'] = pd.to_datetime(processed['trade_date'], format='%Y%m%d')

        # 处理时间字段（转换为标准时间格式）
        if 'first_time' in processed.columns:
            processed['first_time_formatted'] = processed['first_time'].apply(self._format_time)
        if 'last_time' in processed.columns:
            processed['last_time_formatted'] = processed['last_time'].apply(self._format_time)

        # 添加涨跌停类型标识
        processed['limit_type'] = limit_type

        # 处理缺失值（基于官方文档说明）
        processed['fd_amount'] = processed['fd_amount'].fillna(0)  # 封单金额
        processed['limit_amount'] = processed['limit_amount'].fillna(0)  # 板上成交金额
        processed['open_times'] = processed['open_times'].fillna(0)  # 炸板/开板次数
        processed['limit_times'] = processed['limit_times'].fillna(0)  # 连板数
        processed['turnover_ratio'] = processed['turnover_ratio'].fillna(0)  # 换手率

        # 计算增强特征（基于A股涨跌停制度特点）
        processed = self._add_enhanced_features(processed, limit_type)

        return processed

    def _add_enhanced_features(self, data, limit_type):
        """
        添加基于A股涨跌停制度的增强特征
        参考顶级量化论文和策略
        """
        # 1. 涨跌停强度特征
        data['limit_strength'] = self._calculate_limit_strength(data, limit_type)

        # 2. 时间特征
        data['time_to_limit_minutes'] = self._calculate_time_to_limit(data)
        data['time_efficiency_score'] = self._calculate_time_efficiency(data)

        # 3. 资金特征
        data['capital_intensity'] = self._calculate_capital_intensity(data)
        data['turnover_efficiency'] = self._calculate_turnover_efficiency(data)

        # 4. 稳定性特征
        data['stability_score'] = self._calculate_stability_score(data)
        data['board_quality'] = self._calculate_board_quality(data, limit_type)

        # 5. 市值特征
        data['mv_factor'] = self._calculate_mv_factor(data)
        data['float_ratio'] = self._calculate_float_ratio(data)

        # 6. 连板特征
        data['continuity_factor'] = self._calculate_continuity_factor(data)
        data['momentum_score'] = self._calculate_momentum_score(data, limit_type)

        return data
    
    def _format_time(self, time_str):
        """格式化时间字符串"""
        if pd.isna(time_str) or time_str is None:
            return None
        
        try:
            time_str = str(int(time_str))
            if len(time_str) == 6:
                return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            elif len(time_str) == 5:
                return f"0{time_str[0]}:{time_str[1:3]}:{time_str[3:5]}"
            else:
                return time_str
        except:
            return None
    
    def _calculate_limit_strength(self, data, limit_type):
        """
        计算涨跌停强度
        基于封单金额与成交额的比例，考虑不同类型的特点
        """
        if limit_type == 'up':
            # 涨停：封单金额/成交额
            strength = np.where(
                data['amount'] > 0,
                data['fd_amount'] / data['amount'],
                0
            )
        elif limit_type == 'down':
            # 跌停：板上成交金额/成交额（如果有的话）
            strength = np.where(
                data['amount'] > 0,
                data['limit_amount'] / data['amount'],
                0
            )
        else:  # 炸板
            # 炸板：基于炸板次数的反向强度
            strength = np.where(
                data['open_times'] > 0,
                1 / (1 + data['open_times']),
                0.5
            )

        return np.clip(strength, 0, 10)

    def _calculate_time_to_limit(self, data):
        """计算到达涨跌停的时间（分钟）"""
        def time_to_minutes(time_str):
            if pd.isna(time_str) or time_str is None:
                return None
            try:
                time_str = str(int(time_str))
                if len(time_str) == 6:
                    hours = int(time_str[:2])
                    minutes = int(time_str[2:4])
                    seconds = int(time_str[4:6])
                    # 从9:30开始计算
                    total_minutes = (hours - 9) * 60 + minutes - 30 + seconds/60
                    return max(0, total_minutes)
                elif len(time_str) == 5:
                    hours = int(time_str[0])
                    minutes = int(time_str[1:3])
                    seconds = int(time_str[3:5])
                    total_minutes = (hours - 9) * 60 + minutes - 30 + seconds/60
                    return max(0, total_minutes)
            except:
                pass
            return None

        return data['first_time'].apply(time_to_minutes)

    def _calculate_time_efficiency(self, data):
        """
        计算时间效率评分
        越早涨停评分越高
        """
        time_minutes = data['time_to_limit_minutes']
        # 240分钟为全天交易时长
        efficiency = np.where(
            pd.notna(time_minutes),
            np.clip(1 - time_minutes / 240, 0, 1),
            0.5
        )
        return efficiency

    def _calculate_capital_intensity(self, data):
        """
        计算资金强度
        基于成交额与流通市值的比例
        """
        intensity = np.where(
            data['float_mv'] > 0,
            data['amount'] / data['float_mv'],
            0
        )
        return np.clip(intensity, 0, 1)

    def _calculate_turnover_efficiency(self, data):
        """
        计算换手率效率
        高换手率通常表示更强的资金关注
        """
        # 换手率标准化到0-1区间
        efficiency = np.clip(data['turnover_ratio'] / 20, 0, 1)  # 20%换手率为满分
        return efficiency

    def _calculate_stability_score(self, data):
        """
        计算稳定性评分
        基于炸板次数，炸板越少越稳定
        """
        stability = np.where(
            data['open_times'] == 0,
            1.0,  # 无炸板满分
            np.clip(1 - data['open_times'] / 10, 0, 1)  # 10次炸板为0分
        )
        return stability

    def _calculate_board_quality(self, data, limit_type):
        """
        计算板块质量评分
        综合考虑多个因素
        """
        if limit_type == 'up':
            # 涨停质量 = 时间效率 * 0.3 + 稳定性 * 0.4 + 资金强度 * 0.3
            quality = (
                data['time_efficiency_score'] * 0.3 +
                data['stability_score'] * 0.4 +
                data['capital_intensity'] * 0.3
            )
        elif limit_type == 'down':
            # 跌停质量主要看稳定性
            quality = data['stability_score']
        else:  # 炸板
            # 炸板质量主要看炸板后的恢复能力
            quality = 1 - data['stability_score']  # 炸板越多"质量"越高（从炸板角度）

        return np.clip(quality, 0, 1)

    def _calculate_mv_factor(self, data):
        """
        计算市值因子
        中小市值通常更容易涨停
        """
        # 使用流通市值，对数变换后标准化
        mv_log = np.log(data['float_mv'] + 1)
        mv_factor = 1 / (1 + mv_log / 20)  # 市值越小因子越大
        return np.clip(mv_factor, 0, 1)

    def _calculate_float_ratio(self, data):
        """
        计算流通比例
        流通市值/总市值
        """
        ratio = np.where(
            data['total_mv'] > 0,
            data['float_mv'] / data['total_mv'],
            0
        )
        return np.clip(ratio, 0, 1)

    def _calculate_continuity_factor(self, data):
        """
        计算连板因子
        连板数越多因子越高，但有递减效应
        """
        # 使用对数函数避免线性增长
        continuity = np.log(data['limit_times'] + 1) / np.log(10)  # 10连板为满分
        return np.clip(continuity, 0, 1)

    def _calculate_momentum_score(self, data, limit_type):
        """
        计算动量评分
        综合涨跌幅和连板情况
        """
        if limit_type == 'up':
            # 涨停动量 = 涨幅 * 连板因子
            momentum = np.abs(data['pct_chg']) / 10 * (1 + data['continuity_factor'])
        elif limit_type == 'down':
            # 跌停动量 = 跌幅 * 连板因子
            momentum = np.abs(data['pct_chg']) / 10 * (1 + data['continuity_factor'])
        else:  # 炸板
            # 炸板动量主要看炸板强度
            momentum = data['open_times'] / 10

        return np.clip(momentum, 0, 2)
    
    def _incremental_update(self, file_path, new_data, data_type):
        """
        增量更新数据文件

        Args:
            file_path: 文件路径
            new_data: 新数据
            data_type: 数据类型（用于日志）

        Returns:
            合并后的完整数据
        """
        try:
            # 🔧 修复：创建新数据的副本，避免 SettingWithCopyWarning
            new_data_copy = new_data.copy()

            # 检查文件是否存在
            if file_path.exists():
                # 读取原有数据
                old_data = pd.read_parquet(file_path)
                self.logger.info(f"读取原有{data_type}数据: {len(old_data)}条")

                # 确保日期格式一致（使用 .loc 避免警告）
                if 'trade_date' in old_data.columns:
                    old_data.loc[:, 'trade_date'] = pd.to_datetime(old_data['trade_date'])
                if 'trade_date' in new_data_copy.columns:
                    new_data_copy.loc[:, 'trade_date'] = pd.to_datetime(new_data_copy['trade_date'])

                # 获取新数据的日期范围
                new_dates = set(new_data_copy['trade_date'].dt.date)
                self.logger.info(f"新数据日期范围: {min(new_dates)} 到 {max(new_dates)}, 共{len(new_dates)}个交易日")

                # 删除原数据中与新数据日期重复的记录
                old_data_filtered = old_data[~old_data['trade_date'].dt.date.isin(new_dates)].copy()
                removed_count = len(old_data) - len(old_data_filtered)

                if removed_count > 0:
                    self.logger.info(f"覆盖原有数据: {removed_count}条（日期重复）")

                # 合并数据
                merged_data = pd.concat([old_data_filtered, new_data_copy], ignore_index=True)

                # 按日期排序
                merged_data = merged_data.sort_values('trade_date').reset_index(drop=True)

                added_count = len(new_data_copy) - removed_count
                self.logger.info(f"增量添加新数据: {added_count}条")
                self.logger.info(f"合并后总数据量: {len(merged_data)}条")

            else:
                # 文件不存在，直接使用新数据副本
                merged_data = new_data_copy
                self.logger.info(f"文件不存在，创建新文件: {len(merged_data)}条")

            # 保存合并后的数据
            merged_data.to_parquet(file_path, index=False)

            return merged_data

        except Exception as e:
            self.logger.error(f"增量更新{data_type}数据失败: {e}")
            # 出错时直接保存新数据
            new_data_copy = new_data.copy()
            new_data_copy.to_parquet(file_path, index=False)
            return new_data_copy

    def _print_data_date_range(self, file_path, file_name):
        """
        打印数据文件的日期范围信息

        Args:
            file_path: 数据文件路径
            file_name: 文件名称（用于日志）
        """
        try:
            if file_path.exists():
                # 读取数据文件
                data = pd.read_parquet(file_path)

                if not data.empty and 'trade_date' in data.columns:
                    # 确保日期格式正确
                    data['trade_date'] = pd.to_datetime(data['trade_date'])

                    # 获取最早和最晚日期
                    min_date = data['trade_date'].min()
                    max_date = data['trade_date'].max()
                    total_records = len(data)
                    total_trading_days = data['trade_date'].dt.date.nunique()

                    # 打印日期范围信息
                    self.logger.info(f"[{file_name}] 数据概况:")
                    self.logger.info(f"   最早日期: {min_date.strftime('%Y-%m-%d')}")
                    self.logger.info(f"   最晚日期: {max_date.strftime('%Y-%m-%d')}")
                    self.logger.info(f"   总交易日数: {total_trading_days:,} 天")
                    self.logger.info(f"   总记录数: {total_records:,} 条")

                    # 计算数据覆盖天数
                    date_range_days = (max_date - min_date).days + 1
                    coverage_ratio = total_trading_days / date_range_days * 100 if date_range_days > 0 else 0
                    self.logger.info(f"   数据覆盖度: {coverage_ratio:.1f}% ({total_trading_days}/{date_range_days} 交易日)")

                else:
                    self.logger.warning(f"⚠️ {file_name} 文件存在但为空或缺少trade_date字段")

            else:
                self.logger.warning(f"⚠️ {file_name} 文件不存在")

        except Exception as e:
            self.logger.error(f"❌ 读取 {file_name} 日期范围信息失败: {e}")

    def _save_by_year_incremental(self, data, limit_type):
        """按年份增量保存数据"""
        for year in data['trade_date'].dt.year.unique():
            year_data = data[data['trade_date'].dt.year == year]
            year_file = self.data_dir / f"limit_{limit_type}_data_{year}.parquet"

            # 使用增量更新逻辑
            self._incremental_update(year_file, year_data, f"{year}年{limit_type}")

    def _save_by_year(self, data, limit_type):
        """按年份保存数据（已废弃，保留用于兼容性）"""
        for year in data['trade_date'].dt.year.unique():
            year_data = data[data['trade_date'].dt.year == year]
            year_file = self.data_dir / f"limit_{limit_type}_data_{year}.parquet"
            year_data.to_parquet(year_file, index=False)
            self.logger.info(f"{year}年{limit_type}数据已保存: {len(year_data)}条")
    
    def generate_data_summary(self):
        """生成数据摘要报告"""
        summary_file = self.data_dir / "data_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("涨跌停数据摘要报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 检查各个文件
            for file_path in self.data_dir.glob("*.parquet"):
                try:
                    data = pd.read_parquet(file_path)
                    f.write(f"文件: {file_path.name}\n")
                    f.write(f"  记录数: {len(data)}\n")
                    f.write(f"  字段数: {len(data.columns)}\n")
                    f.write(f"  日期范围: {data['trade_date'].min()} 到 {data['trade_date'].max()}\n")
                    f.write(f"  字段列表: {list(data.columns)}\n\n")
                except Exception as e:
                    f.write(f"文件: {file_path.name} - 读取失败: {e}\n\n")
        
        self.logger.info(f"数据摘要报告已生成: {summary_file}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='完整涨跌停数据收集器')
    parser.add_argument('--start_date', '-s', type=str, default='20260112',
                        help='开始日期 (YYYYMMDD格式，默认: 20260112)')
    parser.add_argument('--end_date', '-e', type=str, default='20260113',
                        help='结束日期 (YYYYMMDD格式，默认: 20260113)')
    args = parser.parse_args()

    print("[开始] 完整涨跌停数据收集...")
    print("[数据类型] 涨停(U) + 跌停(D) + 炸板(Z)")
    print(f"[时间范围] {args.start_date} - {args.end_date}")
    print("[输出字段] 18个完整字段 + 10个增强特征")

    start_date = args.start_date
    end_date = args.end_date
    collector = CompleteLimitDataCollector()

    # 获取完整数据（包含炸板）
    limit_up_data, limit_down_data, limit_zhaban_data = collector.get_complete_limit_data_by_date_range(
        start_date=start_date,#20250930
        end_date=end_date#20251021
    )

    # 处理和保存数据
    all_data = collector.process_and_save_complete_data(limit_up_data, limit_down_data, limit_zhaban_data)

    # 生成摘要报告
    collector.generate_data_summary()

    # 显示统计信息
    if not all_data.empty:
        print(f"\n[数据收集统计]:")
        print(f"  总记录数: {len(all_data)}")
        type_counts = all_data['limit_type'].value_counts()
        for limit_type, count in type_counts.items():
            print(f"  {limit_type}: {count} 条")

        print(f"\n[增强特征统计]:")
        enhanced_features = [
            'limit_strength', 'time_to_limit_minutes', 'time_efficiency_score',
            'capital_intensity', 'turnover_efficiency', 'stability_score',
            'board_quality', 'mv_factor', 'float_ratio', 'continuity_factor', 'momentum_score'
        ]
        for feature in enhanced_features:
            if feature in all_data.columns:
                mean_val = all_data[feature].mean()
                print(f"  {feature}: 均值={mean_val:.3f}")

    print("\n[完成] 完整涨跌停数据收集完成！")

if __name__ == "__main__":
    main()
