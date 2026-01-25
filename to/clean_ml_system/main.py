#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化的机器学习选股系统主程序
使用最新的子状态选股规则训练和预测架构
自动执行训练、预测和评估的完整流程
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any
import pickle
import hashlib

from data_loader import DataLoader
from advanced_sector_classifier import AdvancedSectorClassifier
from professional_stock_selection_system import ProfessionalStockSelectionSystem, DailySelectionResult

# 设置日志
from pathlib import Path

# 配置详细的日志系统
def setup_logging():
    """设置详细的日志系统"""
    # 创建日志目录
    log_dir = Path("to/main_sector_stock/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )

    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台处理器 - 显示INFO及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件处理器 - 记录所有级别到文件
    file_handler = logging.FileHandler(
        log_dir / f"ml_sector_strategy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 错误文件处理器 - 只记录ERROR及以上级别
    error_handler = logging.FileHandler(
        log_dir / f"ml_sector_strategy_errors_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    return logging.getLogger(__name__)

# 设置日志
logger = setup_logging()

class OptimizedMLSystem:
    """优化的机器学习选股系统"""

    def __init__(self):
        self.logger = logger

        # 初始化各模块
        self.data_loader = DataLoader()
        self.state_classifier = AdvancedSectorClassifier()
        self.selection_system = ProfessionalStockSelectionSystem()

        self.logger.info("优化的机器学习选股系统初始化完成")

    def split_date_range(self, start_date: str, end_date: str, train_ratio: float = 0.9):
        """
        按比例分割日期范围

        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            train_ratio: 训练数据比例

        Returns:
            tuple: (train_start, train_end, test_start, test_end)
        """
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        total_days = (end_dt - start_dt).days
        train_days = int(total_days * train_ratio)

        train_end_dt = start_dt + timedelta(days=train_days)
        test_start_dt = train_end_dt + timedelta(days=1)

        train_start = start_date
        train_end = train_end_dt.strftime('%Y%m%d')
        test_start = test_start_dt.strftime('%Y%m%d')
        test_end = end_date

        self.logger.info(f"数据分割结果:")
        self.logger.info(f"  训练期: {train_start} - {train_end} ({train_days}天)")
        self.logger.info(f"  测试期: {test_start} - {test_end} ({total_days - train_days}天)")

        return train_start, train_end, test_start, test_end

    def run_complete_analysis(self, start_date: str = "20230601", end_date: str = "20250331",
                            train_ratio: float = 0.9) -> dict:
        """
        运行完整的训练和预测分析

        Args:
            start_date: 总开始日期 (YYYYMMDD)
            end_date: 总结束日期 (YYYYMMDD)
            train_ratio: 训练数据比例

        Returns:
            dict: 包含所有结果的字典
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 开始运行优化的机器学习选股系统")
            self.logger.info(f"总分析期间: {start_date} - {end_date}")
            self.logger.info(f"训练比例: {train_ratio:.1%}")
            self.logger.info("=" * 80)

            # 分割日期
            train_start, train_end, test_start, test_end = self.split_date_range(
                start_date, end_date, train_ratio
            )

            # 阶段1: 训练最优选股规则
            self.logger.info("\n" + "=" * 60)
            self.logger.info("阶段1: 训练最优选股规则")
            self.logger.info("=" * 60)

            training_results = self._train_selection_rules(train_start, train_end)

            if not training_results['success']:
                return {
                    'success': False,
                    'error': f"训练失败: {training_results['error']}"
                }

            # 阶段2: 预测和评估
            self.logger.info("\n" + "=" * 60)
            self.logger.info("阶段2: 预测和评估")
            self.logger.info("=" * 60)

            prediction_results = self._run_prediction_analysis(test_start, test_end)

            if not prediction_results['success']:
                return {
                    'success': False,
                    'error': f"预测失败: {prediction_results['error']}"
                }

            # 阶段3: 生成综合报告
            self.logger.info("\n" + "=" * 60)
            self.logger.info("阶段3: 生成综合报告")
            self.logger.info("=" * 60)

            final_report = self._generate_comprehensive_report(
                training_results, prediction_results,
                train_start, train_end, test_start, test_end
            )

            # 保存结果
            self._save_results(final_report)

            self.logger.info("=" * 80)
            self.logger.info("🎉 系统运行完成！")
            self.logger.info("=" * 80)

            return {
                'success': True,
                'training_results': training_results,
                'prediction_results': prediction_results,
                'final_report': final_report
            }

        except Exception as e:
            self.logger.error(f"系统运行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _train_selection_rules(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """训练选股规则 - 添加缓存机制"""
        try:
            # 🔥 新增：检查训练规则缓存
            cache_key = f"train_selection_rules_{start_date}_{end_date}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
            cache_file = f"cache/train_selection_rules_{cache_hash}.pkl"

            # 创建缓存目录
            os.makedirs("cache", exist_ok=True)

            # 检查缓存是否存在
            if os.path.exists(cache_file):
                self.logger.info(f"从缓存加载训练规则结果: {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)

                    # 恢复选股系统状态
                    if 'system_file' in cached_result and os.path.exists(cached_result['system_file']):
                        self.selection_system.load_system(cached_result['system_file'])
                        self.logger.info(f"缓存加载成功，恢复选股系统: {cached_result['system_file']}")
                        return cached_result
                    else:
                        self.logger.warning("缓存中的系统文件不存在，重新训练")
                except Exception as e:
                    self.logger.warning(f"缓存加载失败: {e}，重新训练")

            self.logger.info("开始训练选股规则...")

            # 加载训练数据
            data = self.data_loader.load_alll_data(start_date, end_date)
            sector_features = data['sector_features']
            stock_features = data['stock_features']
            market_features = data['market_features']

            self.logger.info(f"板块特征: {len(sector_features)} 条记录")
            self.logger.info(f"个股特征: {len(stock_features)} 条记录")

            # 板块状态分类 - 添加缓存机制
            self.logger.info("进行板块状态分类...")
            sector_states = self._classify_sectors_with_cache(sector_features, market_features, start_date, end_date)

            # 准备训练数据 - 添加缓存机制
            self.logger.info("整合训练数据...")
            training_data = self._prepare_training_data_with_cache(sector_states, stock_features, start_date, end_date)

            if not training_data:
                return {
                    'success': False,
                    'error': '训练数据准备失败'
                }

            # 训练所有子状态的选股规则
            self.logger.info("训练所有子状态的选股规则...")
            all_rules = self.selection_system.train_all_sub_state_rules(training_data, "advanced")

            if not all_rules:
                return {
                    'success': False,
                    'error': '没有训练出有效的选股规则'
                }

            # 选择最佳2个子状态
            self.logger.info("选择效果最好的2个子状态...")
            best_sub_states = all_rules.get('best_sub_states', [])

            # 保存训练好的系统
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            system_file = f"optimized_system_{timestamp}.pkl"
            self.selection_system.save_system(system_file)

            # 🔥 新增：构建返回结果并保存到缓存
            result = {
                'success': True,
                'all_rules': all_rules,
                'best_sub_states': best_sub_states,
                'system_file': system_file,
                'training_period': f"{start_date} - {end_date}"
            }

            # 保存到缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                self.logger.info(f"训练规则结果已缓存: {cache_file}")
            except Exception as e:
                self.logger.warning(f"缓存保存失败: {e}")

            return result

        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _prepare_training_data(self, sector_states: pd.DataFrame,
                             stock_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """准备训练数据 - 内存优化版本，支持大数据量处理"""
        training_data = {}

        # 确保日期字段一致
        if 'date' not in stock_features.columns and 'date_str' in stock_features.columns:
            stock_features['date'] = stock_features['date_str']

        # 统计子状态
        sub_state_counts = sector_states['sub_state'].value_counts()
        self.logger.info(f"发现 {len(sub_state_counts)} 个子状态")

        # 预先按日期分组股票数据以提高效率
        self.logger.info("预处理股票数据...")
        stock_by_date = {}
        unique_dates = stock_features['date'].unique()

        # 分批处理日期以节省内存
        batch_size = 50  # 每次处理50个日期
        for i in range(0, len(unique_dates), batch_size):
            batch_dates = unique_dates[i:i+batch_size]
            for date in batch_dates:
                daily_stocks = stock_features[stock_features['date'] == date]
                if not daily_stocks.empty:
                    # 保留所有必要的列，包括个股和板块特征（不进行特征过滤）
                    # 🔥 修复：包含所有数值类型，不仅仅是float64/float32/int64/int32
                    essential_cols = ['stock_code', 'date', 'future_return_1d', 'future_return_2d'] + [
                        col for col in daily_stocks.columns
                        if col not in ['stock_code', 'date', 'future_return_1d', 'future_return_2d'] and
                        pd.api.types.is_numeric_dtype(daily_stocks[col])  # 包含所有数值类型
                    ]  # 保留所有数值特征，包括float16, uint8, uint32等

                    stock_by_date[date] = daily_stocks[essential_cols].copy()

        self.logger.info(f"预处理完成，共 {len(stock_by_date)} 个交易日")

        # 为每个子状态准备训练数据 - 使用流式处理避免内存溢出
        for sub_state in sub_state_counts.index:
            self.logger.info(f"准备子状态 {sub_state} 的数据...")

            # 获取该子状态的板块
            state_sectors = sector_states[sector_states['sub_state'] == sub_state]

            if state_sectors.empty:
                continue

            # 使用流式处理收集数据，避免一次性加载所有数据到内存
            valid_data = self._collect_sub_state_data_streaming(
                state_sectors, stock_by_date, sub_state
            )

            if valid_data is not None and len(valid_data) >= 100:
                training_data[sub_state] = valid_data
                self.logger.info(f"  {sub_state}: {len(valid_data)} 个有效样本")
            else:
                sample_count = len(valid_data) if valid_data is not None else 0
                self.logger.warning(f"  {sub_state}: 样本不足 ({sample_count})")

        # 清理临时数据
        del stock_by_date

        self.logger.info(f"训练数据准备完成，共 {len(training_data)} 个子状态有数据")

        return training_data

    def _classify_sectors_with_cache(self, sector_features: pd.DataFrame, market_features: pd.DataFrame,
                                    start_date: str, end_date: str) -> pd.DataFrame:
        """带缓存的板块状态分类"""
        # 生成缓存文件名
        cache_key = f"classify_sectors_{start_date}_{end_date}_{len(sector_features)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = f"cache/classify_sectors_{cache_hash}.pkl"

        # 创建缓存目录
        os.makedirs("cache", exist_ok=True)

        # 检查缓存是否存在
        if os.path.exists(cache_file):
            self.logger.info(f"从缓存加载板块状态分类结果: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    sector_states = pickle.load(f)
                self.logger.info(f"缓存加载成功，共 {len(sector_states)} 条记录")
                return sector_states
            except Exception as e:
                self.logger.warning(f"缓存加载失败: {e}，重新计算")

        # 执行板块状态分类
        self.logger.info("执行板块状态分类...")
        sector_states = self.state_classifier.classify_sectors(sector_features, market_features)

        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(sector_states, f)
            self.logger.info(f"板块状态分类结果已缓存: {cache_file}")
        except Exception as e:
            self.logger.warning(f"缓存保存失败: {e}")

        return sector_states

    def _prepare_training_data_with_cache(self, sector_states: pd.DataFrame, stock_features: pd.DataFrame,
                                        start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """带缓存的训练数据准备"""
        # 生成缓存文件名
        cache_key = f"prepare_training_data_{start_date}_{end_date}_{len(sector_states)}_{len(stock_features)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = f"cache/prepare_training_data_{cache_hash}.pkl"

        # 创建缓存目录
        os.makedirs("cache", exist_ok=True)

        # 检查缓存是否存在
        if os.path.exists(cache_file):
            self.logger.info(f"从缓存加载训练数据准备结果: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    training_data = pickle.load(f)
                self.logger.info(f"缓存加载成功，共 {len(training_data)} 个子状态")
                return training_data
            except Exception as e:
                self.logger.warning(f"缓存加载失败: {e}，重新计算")

        # 执行训练数据准备
        self.logger.info("执行训练数据准备...")
        training_data = self._prepare_training_data(sector_states, stock_features)

        # 保存到缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(training_data, f)
            self.logger.info(f"训练数据准备结果已缓存: {cache_file}")
        except Exception as e:
            self.logger.warning(f"缓存保存失败: {e}")

        return training_data

    def predict_with_saved_model(self, model_file: str = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """使用保存的模型进行独立预测 - 自动获取最新训练模型"""
        try:
            # 🔥 修改：自动获取最新训练模型
            if model_file is None:
                model_file = self._get_latest_trained_model()
                if model_file is None:
                    return {
                        'success': False,
                        'error': "未找到训练好的模型，请先运行训练模式"
                    }

            self.logger.info(f"加载保存的模型: {model_file}")

            # 加载保存的系统
            if not self.selection_system.load_system(model_file):
                return {
                    'success': False,
                    'error': f"无法加载模型文件: {model_file}"
                }

            # 运行分子状态预测分析
            return self._run_sub_state_prediction_analysis(start_date, end_date, model_file)

        except Exception as e:
            self.logger.error(f"使用保存模型预测失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_latest_trained_model(self) -> str:
        """获取最新训练的模型文件"""
        try:
            # 🔥 方法1：从缓存中获取最新的训练结果
            cache_dir = "cache"
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.startswith("train_selection_rules_") and f.endswith(".pkl")]
                if cache_files:
                    # 按修改时间排序，获取最新的缓存文件
                    latest_cache = max(cache_files, key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)))
                    cache_path = os.path.join(cache_dir, latest_cache)

                    try:
                        with open(cache_path, 'rb') as f:
                            cached_result = pickle.load(f)

                        if 'system_file' in cached_result and os.path.exists(cached_result['system_file']):
                            self.logger.info(f"从缓存获取最新模型: {cached_result['system_file']}")
                            return cached_result['system_file']
                    except Exception as e:
                        self.logger.warning(f"读取缓存失败: {e}")

            # 🔥 方法2：查找最新的系统文件
            system_files = [f for f in os.listdir(".") if f.startswith("optimized_system_") and f.endswith(".pkl")]
            if system_files:
                latest_system = max(system_files, key=lambda f: os.path.getmtime(f))
                self.logger.info(f"找到最新系统文件: {latest_system}")
                return latest_system

            self.logger.warning("未找到任何训练好的模型文件")
            return None

        except Exception as e:
            self.logger.error(f"获取最新模型失败: {e}")
            return None

    def _collect_sub_state_data_streaming(self, state_sectors: pd.DataFrame,
                                        stock_by_date: dict, sub_state: str) -> pd.DataFrame:
        """
        流式收集子状态数据 - 只从对应板块中选择股票训练
        修复训练/预测不一致问题：训练时也只从子状态对应的板块中选择股票

        Args:
            state_sectors: 该子状态的板块数据
            stock_by_date: 按日期分组的股票数据
            sub_state: 子状态名称

        Returns:
            合并后的有效数据（只包含对应板块的股票）
        """
        import gc

        # 获取该子状态对应的所有板块代码
        target_sectors = set(state_sectors['sector_code'].unique())
        self.logger.info(f"子状态 {sub_state} 对应板块: {len(target_sectors)} 个")

        # 估算数据量并设置限制
        total_sector_dates = len(state_sectors)
        estimated_records = total_sector_dates * 200  # 估算每个板块日期约200只股票（更准确的估算）
        max_records = 1000000  # 100万条记录的硬限制

        # 如果估算数据量过大，进行预采样
        if estimated_records > max_records:
            sample_ratio = max_records / estimated_records
            state_sectors = state_sectors.sample(frac=sample_ratio, random_state=42)
            self.logger.warning(f"子状态 {sub_state} 数据量过大，预采样比例: {sample_ratio:.3f}")

        # 使用更小的批次大小
        batch_size = 10  # 每批10个板块日期
        all_valid_data = []
        max_batch_records = 100000  # 每批最多10万条记录

        # 将板块数据分批
        sector_batches = [state_sectors.iloc[i:i+batch_size]
                         for i in range(0, len(state_sectors), batch_size)]

        for batch_idx, sector_batch in enumerate(sector_batches):
            self.logger.debug(f"  处理批次 {batch_idx+1}/{len(sector_batches)}")

            batch_stocks = []
            batch_record_count = 0

            for _, sector_row in sector_batch.iterrows():
                sector_date = sector_row['date']
                sector_code = sector_row['sector_code']

                # 从预处理的数据中获取当日股票数据
                if sector_date in stock_by_date:
                    daily_stocks = stock_by_date[sector_date]

                    if not daily_stocks.empty:
                        # 🔥 关键修改：只选择属于当前板块的股票，统一使用_filter_stocks_by_sector方法
                        sector_stocks = self._filter_stocks_by_sector(daily_stocks, sector_code)

                        if sector_stocks.empty:
                            continue

                        # 检查是否会超过批次限制
                        if batch_record_count + len(sector_stocks) > max_batch_records:
                            # 对当前板块股票进行采样
                            sample_size = max_batch_records - batch_record_count
                            if sample_size > 0:
                                sector_stocks = sector_stocks.sample(n=min(sample_size, len(sector_stocks)), random_state=42)
                            else:
                                break  # 跳出当前批次

                        # 添加子状态和板块标记
                        sector_stocks_copy = sector_stocks.copy()
                        sector_stocks_copy['sub_state'] = sub_state
                        sector_stocks_copy['sector_date'] = sector_date
                        sector_stocks_copy['target_sector'] = sector_code

                        batch_stocks.append(sector_stocks_copy)
                        batch_record_count += len(sector_stocks_copy)

            if batch_stocks:
                try:
                    # 合并当前批次数据
                    batch_combined = pd.concat(batch_stocks, ignore_index=True)

                    # 检查目标变量并清理数据
                    target_col = 'future_return_1d'
                    if target_col in batch_combined.columns:
                        batch_valid = batch_combined.dropna(subset=[target_col])

                        if len(batch_valid) > 0:
                            # 如果批次数据仍然过大，进行采样
                            if len(batch_valid) > max_batch_records:
                                batch_valid = batch_valid.sample(n=max_batch_records, random_state=42)
                                self.logger.info(f"批次数据过大，采样到 {max_batch_records} 条记录")

                            all_valid_data.append(batch_valid)

                except MemoryError:
                    self.logger.warning(f"批次 {batch_idx+1} 内存不足，跳过")
                    continue
                finally:
                    # 清理批次数据
                    del batch_stocks
                    if 'batch_combined' in locals():
                        del batch_combined
                    if 'batch_valid' in locals():
                        del batch_valid
                    gc.collect()

        # 最终合并所有有效数据 - 使用更安全的方式
        if all_valid_data:
            try:
                # 检查总数据量
                total_rows = sum(len(df) for df in all_valid_data)

                if total_rows > max_records:
                    # 对每个批次进行等比例采样
                    target_rows = max_records
                    sample_ratio = target_rows / total_rows

                    sampled_data = []
                    for df in all_valid_data:
                        if len(df) > 0:
                            sample_size = max(1, int(len(df) * sample_ratio))
                            sampled_df = df.sample(n=sample_size, random_state=42)
                            sampled_data.append(sampled_df)

                    all_valid_data = sampled_data
                    self.logger.info(f"总数据量过大，采样到约 {target_rows} 条记录")

                final_data = pd.concat(all_valid_data, ignore_index=True)

                # 清理临时数据
                del all_valid_data
                gc.collect()

                return final_data

            except MemoryError:
                self.logger.error(f"最终合并仍然内存不足，返回空DataFrame")
                return pd.DataFrame()
        else:
            return None

    def _filter_stocks_by_sector(self, daily_stocks: pd.DataFrame, target_sector: str) -> pd.DataFrame:
        """
        根据板块代码过滤股票 - 参考professional_stock_selection_system的_get_stocks_in_sector方法

        Args:
            daily_stocks: 当日股票数据
            target_sector: 目标板块代码

        Returns:
            属于目标板块的股票数据
        """
        try:
            # 修正：导入股票板块映射函数 - 使用正确的相对路径
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from data_loaders import get_stock_list_in_sector

            # 获取板块内的股票列表
            sector_stocks = get_stock_list_in_sector(target_sector)

            if not sector_stocks:
                self.logger.debug(f"板块 {target_sector} 没有找到成分股")
                return pd.DataFrame()

            # 过滤出在当日数据中存在的股票
            filtered_stocks = daily_stocks[daily_stocks['stock_code'].isin(sector_stocks)]

            if not filtered_stocks.empty:
                self.logger.debug(f"- _filter_stocks_by_sector - 板块 {target_sector} 找到 {len(filtered_stocks)} 只股票")
                return filtered_stocks
            else:
                self.logger.debug(f"板块 {target_sector} 在当日数据中没有找到股票")
                return pd.DataFrame()

        except Exception as e:
            self.logger.warning(f"过滤板块 {target_sector} 股票时出错: {e}")
            return pd.DataFrame()

    def _run_sub_state_prediction_analysis(self, start_date: str, end_date: str, model_file: str) -> Dict[str, Any]:
        """运行分子状态预测分析 - 复用train模式的评估逻辑"""
        try:
            self.logger.info("开始分子状态预测分析...")

            # 复用train模式的预测分析逻辑
            prediction_results = self._run_prediction_analysis(start_date, end_date)

            if not prediction_results.get('success', False):
                return prediction_results

            # 转换为predict模式的结果格式，保持与train模式一致的数据结构
            sub_state_performance = prediction_results.get('sub_state_performance', {})

            # 转换子状态结果格式，添加predict模式需要的字段
            sub_state_results = {}
            for sub_state, perf in sub_state_performance.items():
                sub_state_results[sub_state] = {
                    'selections': perf.get('selections', 0),
                    'wins': perf.get('wins', 0),
                    'win_rate': perf.get('win_rate', 0.0),
                    'cumulative_return': perf.get('total_return', 0.0),  # 映射total_return到cumulative_return
                    'max_drawdown': perf.get('max_drawdown', 0.0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0.0),
                    'annualized_return': self._calculate_annualized_return_from_total(
                        perf.get('total_return', 0.0),
                        prediction_results.get('total_trading_days', 252)
                    ),
                    'trading_days': perf.get('trading_days', 0),
                    'selected_stocks': perf.get('selected_stocks', [])
                }

            # 构建整体摘要
            overall_summary = {
                'total_selections': prediction_results.get('total_selections', 0),
                'total_wins': prediction_results.get('total_wins', 0),
                'overall_win_rate': prediction_results.get('overall_win_rate', 0.0),
                'cumulative_return': prediction_results.get('cumulative_return', 0.0),
                'max_drawdown': prediction_results.get('max_drawdown', 0.0),
                'sharpe_ratio': prediction_results.get('sharpe_ratio', 0.0),
                'annualized_return': prediction_results.get('annualized_return', 0.0),
                'trading_days': prediction_results.get('total_trading_days', 0)
            }

            final_results = {
                'success': True,
                'model_file': model_file,
                'prediction_period': prediction_results.get('prediction_period', f"{start_date} - {end_date}"),
                'best_sub_states': prediction_results.get('best_sub_states', []),
                'sub_state_results': sub_state_results,
                'overall_summary': overall_summary,
                'detailed_selections': prediction_results.get('detailed_daily_records', [])
            }

            # 保存详细结果到JSON文件
            self._save_prediction_results(final_results, model_file)

            return final_results

        except Exception as e:
            self.logger.error(f"分子状态预测分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_annualized_return_from_total(self, total_return: float, trading_days: int) -> float:
        """从总收益计算年化收益率"""
        if trading_days == 0:
            return 0.0

        years = trading_days / 252  # 假设一年252个交易日
        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def _calculate_annualized_return(self, cumulative_return: float, trading_days: int) -> float:
        """计算年化收益率"""
        if trading_days <= 0 or cumulative_return <= -1:
            return 0.0
        years = trading_days / 252.0
        if years <= 0:
            return 0.0
        return (1 + cumulative_return) ** (1 / years) - 1

    def _calculate_max_drawdown(self, daily_returns: List[float]) -> float:
        """计算最大回撤"""
        if not daily_returns:
            return 0.0

        cumulative_returns = []
        cumulative = 1.0
        for ret in daily_returns:
            cumulative *= (1 + ret)
            cumulative_returns.append(cumulative)

        if not cumulative_returns:
            return 0.0

        running_max = cumulative_returns[0]
        max_drawdown = 0.0

        for value in cumulative_returns:
            if value > running_max:
                running_max = value
            drawdown = (running_max - value) / running_max
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """计算夏普比率"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0

        import numpy as np
        returns_array = np.array(daily_returns)

        if np.std(returns_array) == 0:
            return 0.0

        # 假设无风险利率为3%年化
        risk_free_rate = 0.03 / 252  # 日无风险利率
        excess_returns = returns_array - risk_free_rate

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _run_prediction_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """运行预测分析"""
        try:
            self.logger.info("加载预测数据...")

            # 加载预测数据
            data = self.data_loader.load_all_data(start_date, end_date)
            sector_features = data['sector_features']
            stock_features = data['stock_features']
            market_features = data['market_features']

            self.logger.info(f"板块特征: {len(sector_features)} 条记录")
            self.logger.info(f"个股特征: {len(stock_features)} 条记录")

            # 板块状态分类
            self.logger.info("进行板块状态分类...")
            sector_states = self.state_classifier.classify_sectors(sector_features, market_features)

            # 统计目标子状态出现频率
            target_states_count = {}
            if self.selection_system.best_sub_states:
                for state in self.selection_system.best_sub_states:
                    count = (sector_states['sub_state'] == state).sum()
                    target_states_count[state] = count
                    self.logger.info(f"目标子状态 {state}: {count} 次出现")
            else:
                self.logger.warning("没有找到最佳子状态，将使用所有可用子状态")

            # 确保日期字段一致
            if 'date' not in stock_features.columns and 'date_str' in stock_features.columns:
                stock_features['date'] = stock_features['date_str']

            # 按日期进行预测
            dates = sorted(sector_states['date'].unique())
            daily_results = []
            detailed_daily_records = []
            total_selections = 0
            total_wins = 0
            cumulative_return = 0.0
            daily_returns = []

            # 分子状态统计
            sub_state_stats = {}
            for state in self.selection_system.best_sub_states:
                sub_state_stats[state] = {
                    'selections': 0,
                    'wins': 0,
                    'total_return': 0.0,
                    'daily_returns': [],
                    'trading_days': 0,
                    'selected_stocks': []
                }

            # 🔥 修改：分别回测两种算法
            self.logger.info("=" * 60)
            self.logger.info("开始分离算法回测")
            self.logger.info("=" * 60)

            # 分别回测两种算法
            algorithms = ['multitask_attention', 'genetic_rl']
            algorithm_results = {}

            for algorithm in algorithms:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"开始回测算法: {algorithm}")
                self.logger.info(f"{'='*50}")

                algorithm_results[algorithm] = self._backtest_single_algorithm(
                    sector_states, stock_features, dates, algorithm
                )

            # 对比分析两种算法
            self._compare_algorithm_performance(algorithm_results)

            # 添加成功标志
            algorithm_results['success'] = True
            return algorithm_results

        except Exception as e:
            self.logger.error(f"预测分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _backtest_single_algorithm(self, sector_states: pd.DataFrame,
                                 stock_features: pd.DataFrame,
                                 dates: list,
                                 algorithm_type: str) -> Dict[str, Any]:
        """回测单个算法"""

        daily_results = []
        detailed_daily_records = []
        total_selections = 0
        total_wins = 0
        cumulative_return = 0.0
        daily_returns = []

        # 分子状态统计 - 🔥 修复：动态收集实际使用的子状态
        sub_state_stats = {}

        self.logger.info(f"开始 {algorithm_type} 算法逐日预测，共 {len(dates)} 个交易日")

        for i, date in enumerate(dates):
            if i % 10 == 0:  # 每10天报告一次进度
                self.logger.info(f"{algorithm_type} 预测进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")

            # 🔥 使用指定算法预测
            selected_stocks = self.selection_system.predict_and_select_stocks(
                sector_states, stock_features, str(date), algorithm_type=algorithm_type
            )

            # 获取当日板块状态分布
            daily_sector_states = sector_states[sector_states['date'] == str(date)]
            sector_state_summary = daily_sector_states['sub_state'].value_counts().to_dict()

            if selected_stocks:
                    # 获取实际收益
                    actual_returns = {}
                    for candidate in selected_stocks:
                        stock_code = candidate.stock_code
                        stock_data = stock_features[
                            (stock_features['stock_code'] == stock_code) &
                            (stock_features['date'] == str(date))
                        ]

                        if not stock_data.empty and 'future_return_1d' in stock_data.columns:
                            actual_return = stock_data['future_return_1d'].iloc[0]
                            if pd.notna(actual_return):
                                actual_returns[stock_code] = actual_return
                            else:
                                actual_returns[stock_code] = 0.0
                        else:
                            actual_returns[stock_code] = 0.0

                    # 评估当日收益
                    day_results = self.selection_system.evaluate_final_performance(selected_stocks, actual_returns)

                    # 计算当日收益率（等权重）
                    daily_return = day_results['avg_return']
                    daily_returns.append(daily_return)
                    cumulative_return += daily_return

                    # 按子状态分别统计 - 🔥 修复：动态创建子状态统计
                    for stock in selected_stocks:
                        sub_state = stock.sub_state

                        # 动态创建子状态统计
                        if sub_state not in sub_state_stats:
                            sub_state_stats[sub_state] = {
                                'selections': 0,
                                'wins': 0,
                                'total_return': 0.0,
                                'daily_returns': [],
                                'trading_days': 0,
                                'selected_stocks': []
                            }

                        sub_state_stats[sub_state]['selections'] += 1
                        sub_state_stats[sub_state]['selected_stocks'].append({
                            'date': str(date),
                            'stock_code': stock.stock_code,
                            'predicted_return': stock.predicted_return,
                            'actual_return': actual_returns.get(stock.stock_code, 0.0)
                        })

                        actual_ret = actual_returns.get(stock.stock_code, 0.0)
                        sub_state_stats[sub_state]['total_return'] += actual_ret
                        sub_state_stats[sub_state]['daily_returns'].append(actual_ret)

                        if actual_ret > 0:
                            sub_state_stats[sub_state]['wins'] += 1

                    # 更新子状态交易日统计
                    active_sub_states = set(stock.sub_state for stock in selected_stocks)
                    for sub_state in active_sub_states:
                        if sub_state in sub_state_stats:
                            sub_state_stats[sub_state]['trading_days'] += 1

                    # 详细记录
                    detailed_record = {
                        'date': str(date),
                        'selected_stocks': day_results['selected_stocks'],
                        'sector_states': sector_state_summary,
                        'daily_return': daily_return,
                        'cumulative_return': cumulative_return,
                        'win_count': day_results['win_count'],
                        'total_count': day_results['total_count'],
                        'daily_win_rate': day_results['win_rate'],
                        'market_condition': self._assess_market_condition(daily_sector_states)
                    }
                    detailed_daily_records.append(detailed_record)

                    # 简化的日结果
                    day_summary = {
                        'date': str(date),
                        'daily_return': daily_return,
                        'win_rate': day_results['win_rate'],
                        'selected_count': len(selected_stocks)
                    }
                    daily_results.append(day_summary)

                    # 累计统计
                    total_selections += len(selected_stocks)
                    total_wins += day_results['win_count']
            else:
                # 没有选股的日子
                daily_returns.append(0.0)
                detailed_record = {
                    'date': str(date),
                    'selected_stocks': [],
                    'sector_states': sector_state_summary,
                    'daily_return': 0.0,
                    'cumulative_return': cumulative_return,
                    'win_count': 0,
                    'total_count': 0,
                    'daily_win_rate': 0.0,
                    'market_condition': self._assess_market_condition(daily_sector_states)
                }
                detailed_daily_records.append(detailed_record)

            # 计算正确的年化收益率
            trading_days = len([r for r in daily_returns if r != 0])  # 实际交易天数
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
            sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0

            # 年化收益率（修正计算）
            if trading_days > 0 and avg_daily_return > -1:
                # 使用累计收益率进行年化
                total_return = cumulative_return
                days_ratio = 252 / trading_days  # 年化倍数
                annualized_return = (1 + total_return) ** days_ratio - 1
            else:
                annualized_return = 0

            # 最大回撤
            cumulative_returns = np.cumsum(daily_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

            # 计算子状态分别的性能指标
            sub_state_performance = {}
            for sub_state, stats in sub_state_stats.items():
                if stats['selections'] > 0:
                    win_rate = stats['wins'] / stats['selections']
                    avg_return = stats['total_return'] / stats['selections']
                    volatility_sub = np.std(stats['daily_returns']) if len(stats['daily_returns']) > 1 else 0
                    sharpe_sub = avg_return / volatility_sub if volatility_sub > 0 else 0

                    # 计算最大回撤
                    if len(stats['daily_returns']) > 0:
                        cumulative_sub = np.cumsum(stats['daily_returns'])
                        running_max_sub = np.maximum.accumulate(cumulative_sub)
                        drawdowns_sub = cumulative_sub - running_max_sub
                        max_drawdown_sub = np.min(drawdowns_sub) if len(drawdowns_sub) > 0 else 0
                    else:
                        max_drawdown_sub = 0

                    sub_state_performance[sub_state] = {
                        'selections': stats['selections'],
                        'wins': stats['wins'],
                        'win_rate': win_rate,
                        'total_return': stats['total_return'],
                        'avg_return': avg_return,
                        'volatility': volatility_sub,
                        'sharpe_ratio': sharpe_sub,
                        'max_drawdown': abs(max_drawdown_sub),
                        'trading_days': stats['trading_days'],
                        'selected_stocks': stats['selected_stocks']
                    }
                else:
                    sub_state_performance[sub_state] = {
                        'selections': 0,
                        'wins': 0,
                        'win_rate': 0,
                        'total_return': 0,
                        'avg_return': 0,
                        'volatility': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'trading_days': 0,
                        'selected_stocks': []
                    }



        # 返回单个算法的回测结果
        return {
            'algorithm': algorithm_type,
            'total_trading_days': len(dates),
            'active_trading_days': trading_days,
            'total_selections': total_selections,
            'total_wins': total_wins,
            'overall_win_rate': total_wins / total_selections if total_selections > 0 else 0.0,
            'cumulative_return': cumulative_return,
            'annualized_return': self._calculate_annualized_return(cumulative_return, trading_days),
            'max_drawdown': self._calculate_max_drawdown(daily_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'daily_returns': daily_returns,
            'sub_state_performance': sub_state_performance,  # 🔥 修复：使用正确的子状态性能数据
            'daily_results': daily_results,
            'detailed_records': detailed_daily_records
        }

    def _compare_algorithm_performance(self, algorithm_results: Dict[str, Dict[str, Any]]) -> None:
        """对比分析两种算法的性能"""

        self.logger.info("\n" + "=" * 80)
        self.logger.info("算法性能对比分析")
        self.logger.info("=" * 80)

        # 提取关键指标
        comparison_data = []
        for algo_name, results in algorithm_results.items():
            comparison_data.append({
                'algorithm': algo_name,
                'total_return': results['cumulative_return'],
                'annualized_return': results['annualized_return'],
                'win_rate': results['overall_win_rate'],
                'max_drawdown': results['max_drawdown'],
                'sharpe_ratio': results['sharpe_ratio'],
                'total_selections': results['total_selections'],
                'active_days': results['active_trading_days']
            })

        # 打印对比表格
        self.logger.info(f"{'算法名称':<20} {'总收益率':<12} {'年化收益率':<12} {'胜率':<8} {'最大回撤':<12} {'夏普比率':<10} {'选股次数':<8}")
        self.logger.info("-" * 90)

        for data in comparison_data:
            self.logger.info(
                f"{data['algorithm']:<20} "
                f"{data['total_return']:<12.2%} "
                f"{data['annualized_return']:<12.2%} "
                f"{data['win_rate']:<8.2%} "
                f"{data['max_drawdown']:<12.2%} "
                f"{data['sharpe_ratio']:<10.3f} "
                f"{data['total_selections']:<8d}"
            )

        # 确定最佳算法
        if len(comparison_data) >= 2:
            best_return_algo = max(comparison_data, key=lambda x: x['total_return'])
            best_sharpe_algo = max(comparison_data, key=lambda x: x['sharpe_ratio'])
            best_winrate_algo = max(comparison_data, key=lambda x: x['win_rate'])

            self.logger.info("\n" + "=" * 50)
            self.logger.info("最佳算法分析:")
            self.logger.info(f"最高收益: {best_return_algo['algorithm']} ({best_return_algo['total_return']:.2%})")
            self.logger.info(f"最佳夏普: {best_sharpe_algo['algorithm']} ({best_sharpe_algo['sharpe_ratio']:.3f})")
            self.logger.info(f"最高胜率: {best_winrate_algo['algorithm']} ({best_winrate_algo['win_rate']:.2%})")

        # 子状态性能对比
        self._compare_sub_state_performance(algorithm_results)

    def _compare_sub_state_performance(self, algorithm_results: Dict[str, Dict[str, Any]]) -> None:
        """增强的子状态性能对比分析"""

        self.logger.info("\n" + "=" * 80)
        self.logger.info("详细子状态性能对比分析")
        self.logger.info("=" * 80)



        # 收集所有子状态
        all_sub_states = set()
        for results in algorithm_results.values():
            if 'sub_state_performance' in results:
                all_sub_states.update(results['sub_state_performance'].keys())

        if not all_sub_states:
            self.logger.warning("未找到子状态性能数据")
            return

        # 为每个子状态创建详细对比
        for sub_state in sorted(all_sub_states):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"子状态: {sub_state}")
            self.logger.info(f"{'='*60}")

            # 收集该子状态下所有算法的数据
            sub_state_data = []
            for algo_name, results in algorithm_results.items():
                if ('sub_state_performance' in results and
                    sub_state in results['sub_state_performance']):

                    stats = results['sub_state_performance'][sub_state]

                    # 确保所有必要字段存在
                    selections = stats.get('selections', 0)
                    wins = stats.get('wins', 0)
                    total_return = stats.get('total_return', 0.0)
                    win_rate = stats.get('win_rate', wins / selections if selections > 0 else 0)
                    avg_return = stats.get('avg_return', total_return / selections if selections > 0 else 0)
                    sharpe_ratio = stats.get('sharpe_ratio', 0.0)
                    max_drawdown = stats.get('max_drawdown', 0.0)
                    trading_days = stats.get('trading_days', 0)

                    # 计算年化收益率
                    if trading_days > 0 and avg_return > -1:
                        annualized_return = (1 + avg_return) ** (252 / trading_days) - 1
                    else:
                        annualized_return = 0.0

                    sub_state_data.append({
                        'algorithm': algo_name,
                        'selections': selections,
                        'wins': wins,
                        'win_rate': win_rate,
                        'total_return': total_return,
                        'avg_return': avg_return,
                        'annualized_return': annualized_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'trading_days': trading_days
                    })

            if not sub_state_data:
                self.logger.info("该子状态无有效数据")
                continue

            # 打印详细对比表格
            self.logger.info(f"{'算法':<20} {'选股次数':<8} {'胜率':<8} {'总收益率':<12} {'年化收益率':<12} {'夏普比率':<10} {'最大回撤':<12}")
            self.logger.info("-" * 90)

            for data in sub_state_data:
                self.logger.info(
                    f"{data['algorithm']:<20} "
                    f"{data['selections']:<8d} "
                    f"{data['win_rate']:<8.2%} "
                    f"{data['total_return']:<12.3f} "
                    f"{data['annualized_return']:<12.2%} "
                    f"{data['sharpe_ratio']:<10.3f} "
                    f"{data['max_drawdown']:<12.2%}"
                )

            # 子状态内算法优劣分析
            if len(sub_state_data) >= 2:
                best_return = max(sub_state_data, key=lambda x: x['total_return'])
                best_winrate = max(sub_state_data, key=lambda x: x['win_rate'])
                best_sharpe = max(sub_state_data, key=lambda x: x['sharpe_ratio'])
                min_drawdown = min(sub_state_data, key=lambda x: x['max_drawdown'])

                self.logger.info(f"\n{sub_state} 最佳表现:")
                self.logger.info(f"  最高收益: {best_return['algorithm']} ({best_return['total_return']:.3f})")
                self.logger.info(f"  最高胜率: {best_winrate['algorithm']} ({best_winrate['win_rate']:.2%})")
                self.logger.info(f"  最佳夏普: {best_sharpe['algorithm']} ({best_sharpe['sharpe_ratio']:.3f})")
                self.logger.info(f"  最小回撤: {min_drawdown['algorithm']} ({min_drawdown['max_drawdown']:.2%})")

        # 跨子状态算法综合对比
        self._cross_sub_state_algorithm_comparison(algorithm_results, all_sub_states)

    def _cross_sub_state_algorithm_comparison(self, algorithm_results: Dict[str, Dict[str, Any]],
                                            all_sub_states: set) -> None:
        """跨子状态算法综合对比分析"""

        self.logger.info(f"\n{'='*80}")
        self.logger.info("跨子状态算法综合对比分析")
        self.logger.info(f"{'='*80}")

        # 收集每个算法在所有子状态上的综合表现
        algorithm_summary = {}

        for algo_name, results in algorithm_results.items():
            if 'sub_state_performance' not in results:
                continue

            sub_state_perf = results['sub_state_performance']

            # 计算算法在所有子状态上的加权平均指标
            total_selections = 0
            total_wins = 0
            total_return = 0.0
            weighted_sharpe = 0.0
            max_drawdown_overall = 0.0
            active_sub_states = 0

            sub_state_details = []

            for sub_state in all_sub_states:
                if sub_state in sub_state_perf:
                    stats = sub_state_perf[sub_state]
                    selections = stats.get('selections', 0)

                    if selections > 0:  # 只统计有交易的子状态
                        wins = stats.get('wins', 0)
                        sub_return = stats.get('total_return', 0.0)
                        sharpe = stats.get('sharpe_ratio', 0.0)
                        drawdown = stats.get('max_drawdown', 0.0)
                        win_rate = stats.get('win_rate', 0.0)

                        total_selections += selections
                        total_wins += wins
                        total_return += sub_return
                        weighted_sharpe += sharpe * selections  # 按选股次数加权
                        max_drawdown_overall = max(max_drawdown_overall, drawdown)
                        active_sub_states += 1

                        sub_state_details.append({
                            'sub_state': sub_state,
                            'selections': selections,
                            'win_rate': win_rate,
                            'return': sub_return,
                            'sharpe': sharpe
                        })

            # 计算综合指标
            overall_win_rate = total_wins / total_selections if total_selections > 0 else 0
            avg_return_per_selection = total_return / total_selections if total_selections > 0 else 0
            weighted_avg_sharpe = weighted_sharpe / total_selections if total_selections > 0 else 0

            algorithm_summary[algo_name] = {
                'total_selections': total_selections,
                'overall_win_rate': overall_win_rate,
                'total_return': total_return,
                'avg_return_per_selection': avg_return_per_selection,
                'weighted_sharpe': weighted_avg_sharpe,
                'max_drawdown': max_drawdown_overall,
                'active_sub_states': active_sub_states,
                'sub_state_details': sub_state_details
            }

        # 打印综合对比表格
        self.logger.info(f"{'算法':<20} {'总选股':<8} {'综合胜率':<10} {'总收益':<12} {'平均收益':<12} {'加权夏普':<10} {'最大回撤':<12} {'活跃子状态':<10}")
        self.logger.info("-" * 110)

        for algo_name, summary in algorithm_summary.items():
            self.logger.info(
                f"{algo_name:<20} "
                f"{summary['total_selections']:<8d} "
                f"{summary['overall_win_rate']:<10.2%} "
                f"{summary['total_return']:<12.3f} "
                f"{summary['avg_return_per_selection']:<12.4f} "
                f"{summary['weighted_sharpe']:<10.3f} "
                f"{summary['max_drawdown']:<12.2%} "
                f"{summary['active_sub_states']:<10d}"
            )

        # 算法优劣势分析
        if len(algorithm_summary) >= 2:
            self.logger.info(f"\n{'='*60}")
            self.logger.info("算法优劣势分析")
            self.logger.info(f"{'='*60}")

            best_overall_return = max(algorithm_summary.items(), key=lambda x: x[1]['total_return'])
            best_win_rate = max(algorithm_summary.items(), key=lambda x: x[1]['overall_win_rate'])
            best_sharpe = max(algorithm_summary.items(), key=lambda x: x[1]['weighted_sharpe'])
            min_drawdown = min(algorithm_summary.items(), key=lambda x: x[1]['max_drawdown'])
            most_active = max(algorithm_summary.items(), key=lambda x: x[1]['active_sub_states'])

            self.logger.info(f"最高总收益: {best_overall_return[0]} ({best_overall_return[1]['total_return']:.3f})")
            self.logger.info(f"最高胜率: {best_win_rate[0]} ({best_win_rate[1]['overall_win_rate']:.2%})")
            self.logger.info(f"最佳夏普比率: {best_sharpe[0]} ({best_sharpe[1]['weighted_sharpe']:.3f})")
            self.logger.info(f"最小回撤: {min_drawdown[0]} ({min_drawdown[1]['max_drawdown']:.2%})")
            self.logger.info(f"最活跃算法: {most_active[0]} (活跃于{most_active[1]['active_sub_states']}个子状态)")

            # 详细的子状态表现对比
            self.logger.info(f"\n{'='*60}")
            self.logger.info("各算法子状态详细表现")
            self.logger.info(f"{'='*60}")

            for algo_name, summary in algorithm_summary.items():
                self.logger.info(f"\n{algo_name} 子状态表现:")
                if summary['sub_state_details']:
                    # 按收益排序
                    sorted_details = sorted(summary['sub_state_details'],
                                          key=lambda x: x['return'], reverse=True)

                    for detail in sorted_details:
                        self.logger.info(
                            f"  {detail['sub_state']:<25}: "
                            f"选股{detail['selections']:>3}次, "
                            f"胜率{detail['win_rate']:>6.1%}, "
                            f"收益{detail['return']:>8.3f}, "
                            f"夏普{detail['sharpe']:>6.2f}"
                        )
                else:
                    self.logger.info("  无活跃子状态")

    def _assess_market_condition(self, daily_sector_states: pd.DataFrame) -> str:
        """评估市场状况"""
        if daily_sector_states.empty:
            return "unknown"

        # 统计各主状态的比例
        main_states = daily_sector_states['main_state'].value_counts()
        total_sectors = len(daily_sector_states)

        if total_sectors == 0:
            return "unknown"

        # 计算各状态占比
        rising_ratio = main_states.get('rising', 0) / total_sectors
        retreat_ratio = main_states.get('retreat', 0) / total_sectors
        oscillation_ratio = main_states.get('oscillation', 0) / total_sectors
        differentiation_ratio = main_states.get('differentiation', 0) / total_sectors

        # 判断市场状况
        if rising_ratio >= 0.5:
            return "bull_market"
        elif retreat_ratio >= 0.5:
            return "bear_market"
        elif differentiation_ratio >= 0.4:
            return "differentiation_market"
        elif oscillation_ratio >= 0.4:
            return "oscillation_market"
        else:
            return "mixed_market"

    def _get_best_algorithm_performance(self, algorithm_results: Dict) -> Dict[str, Any]:
        """从算法结果中选择最佳性能的算法数据"""
        if not algorithm_results or 'success' not in algorithm_results:
            return {}

        # 移除success标志，获取算法结果
        algo_data = {k: v for k, v in algorithm_results.items() if k != 'success'}

        if not algo_data:
            return {}

        # 如果只有一个算法，直接返回
        if len(algo_data) == 1:
            return list(algo_data.values())[0]

        # 多个算法时，选择夏普比率最高的
        best_algo = None
        best_sharpe = float('-inf')

        for algo_name, results in algo_data.items():
            sharpe = results.get('sharpe_ratio', 0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_algo = results

        return best_algo if best_algo else {}

    def _generate_comprehensive_report(self, training_results: Dict, prediction_results: Dict,
                                     train_start: str, train_end: str,
                                     test_start: str, test_end: str) -> Dict[str, Any]:
        """生成综合报告"""

        self.logger.info("📊 生成综合性能报告")

        # 训练阶段摘要
        training_summary = {
            'period': f"{train_start} - {train_end}",
            'total_rules_trained': len(training_results.get('all_rules', {})),
            'best_sub_states': training_results.get('best_sub_states', []),
            'system_file': training_results.get('system_file', '')
        }

        # 🔥 修复：从算法结果中提取最佳性能指标
        best_algorithm_results = self._get_best_algorithm_performance(prediction_results)

        # 预测阶段摘要
        prediction_summary = {
            'period': f"{test_start} - {test_end}",
            'total_trading_days': best_algorithm_results.get('total_trading_days', 0),
            'active_trading_days': best_algorithm_results.get('active_trading_days', 0),
            'total_selections': best_algorithm_results.get('total_selections', 0),
            'overall_win_rate': best_algorithm_results.get('overall_win_rate', 0),
            'cumulative_return': best_algorithm_results.get('cumulative_return', 0),
            'avg_daily_return': best_algorithm_results.get('avg_daily_return', 0),
            'annualized_return': best_algorithm_results.get('annualized_return', 0),
            'volatility': best_algorithm_results.get('volatility', 0),
            'sharpe_ratio': best_algorithm_results.get('sharpe_ratio', 0),
            'max_drawdown': best_algorithm_results.get('max_drawdown', 0)
        }

        # 最佳子状态性能
        best_states_performance = {}
        if training_results.get('all_rules'):
            for state in training_results['best_sub_states']:
                if state in training_results['all_rules']:
                    rule = training_results['all_rules'][state]
                    best_states_performance[state] = {
                        'win_rate': rule.win_rate,
                        'avg_return': rule.avg_return,
                        'sharpe_ratio': rule.sharpe_ratio,
                        'max_drawdown': rule.max_drawdown,
                        'sample_count': rule.sample_count
                    }

        # 生成最终报告
        final_report = {
            'system_info': {
                'system_name': 'OptimizedMLSystem',
                'version': '2.0',
                'generation_time': datetime.now().isoformat(),
                'total_analysis_period': f"{train_start} - {test_end}"
            },
            'training_summary': training_summary,
            'prediction_summary': prediction_summary,
            'best_states_performance': best_states_performance,
            'key_metrics': {
                'final_win_rate': prediction_summary['overall_win_rate'],
                'final_cumulative_return': prediction_summary['cumulative_return'],
                'final_annualized_return': prediction_summary['annualized_return'],
                'final_sharpe_ratio': prediction_summary['sharpe_ratio'],
                'final_max_drawdown': prediction_summary['max_drawdown'],
                'selection_efficiency': prediction_summary['active_trading_days'] / prediction_summary['total_trading_days'] if prediction_summary['total_trading_days'] > 0 else 0
            },
            'detailed_daily_records': prediction_results.get('detailed_daily_records', [])
        }

        # 打印关键指标
        self.logger.info("=" * 60)
        self.logger.info("📈 关键性能指标")
        self.logger.info("=" * 60)
        self.logger.info(f"最佳子状态: {training_summary['best_sub_states']}")
        self.logger.info(f"总胜率: {prediction_summary['overall_win_rate']:.2%}")
        self.logger.info(f"累计收益: {prediction_summary['cumulative_return']:.4f}")
        self.logger.info(f"年化收益率: {prediction_summary['annualized_return']:.2%}")
        self.logger.info(f"夏普比率: {prediction_summary['sharpe_ratio']:.3f}")
        self.logger.info(f"最大回撤: {prediction_summary['max_drawdown']:.2%}")
        self.logger.info(f"选股活跃度: {final_report['key_metrics']['selection_efficiency']:.1%}")

        # 子状态分别性能
        self.logger.info("=" * 60)
        self.logger.info("📊 子状态分别性能分析")
        self.logger.info("=" * 60)

        # 🔥 修复：从最佳算法结果中获取子状态性能
        sub_state_perf = best_algorithm_results.get('sub_state_performance', {})
        for sub_state, perf in sub_state_perf.items():
            if perf['selections'] > 0:
                self.logger.info(f"🔸 {sub_state}:")
                self.logger.info(f"  选股次数: {perf['selections']}")
                self.logger.info(f"  胜率: {perf['win_rate']:.2%}")
                self.logger.info(f"  平均收益: {perf['avg_return']:.4f}")
                self.logger.info(f"  累计收益: {perf['total_return']:.4f}")
                self.logger.info(f"  夏普比率: {perf['sharpe_ratio']:.3f}")
                self.logger.info(f"  最大回撤: {perf['max_drawdown']:.2%}")
                self.logger.info(f"  交易日数: {perf['trading_days']}")
                self.logger.info("")
            else:
                self.logger.info(f"🔸 {sub_state}: 无选股记录")

        # 子状态对比分析
        self.logger.info("=" * 60)
        self.logger.info("🔍 子状态对比分析")
        self.logger.info("=" * 60)

        active_states = {k: v for k, v in sub_state_perf.items() if v['selections'] > 0}
        if len(active_states) >= 2:
            # 按夏普比率排序
            sorted_states = sorted(active_states.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)

            self.logger.info("按夏普比率排序:")
            for i, (state, perf) in enumerate(sorted_states, 1):
                self.logger.info(f"{i}. {state}: 夏普比率 {perf['sharpe_ratio']:.3f}, 胜率 {perf['win_rate']:.2%}, 累计收益 {perf['total_return']:.4f}")

            # 分析差异
            best_state = sorted_states[0]
            worst_state = sorted_states[-1]

            self.logger.info(f"\n最佳子状态 {best_state[0]} vs 最差子状态 {worst_state[0]}:")
            self.logger.info(f"  夏普比率差异: {best_state[1]['sharpe_ratio'] - worst_state[1]['sharpe_ratio']:.3f}")
            self.logger.info(f"  胜率差异: {best_state[1]['win_rate'] - worst_state[1]['win_rate']:.2%}")
            self.logger.info(f"  收益差异: {best_state[1]['total_return'] - worst_state[1]['total_return']:.4f}")
        else:
            self.logger.info("活跃子状态不足2个，无法进行对比分析")

        return final_report

    def _save_results(self, final_report: Dict[str, Any]):
        """保存结果到文件"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存JSON报告
            report_file = f"final_analysis_report_{timestamp}.json"

            # 转换为可序列化的格式
            serializable_report = self._make_json_serializable(final_report)

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"分析报告已保存到: {report_file}")

        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")

    def _save_prediction_results(self, results: Dict[str, Any], model_file: str):
        """保存预测结果到JSON文件"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = os.path.splitext(os.path.basename(model_file))[0]

            # 保存详细结果 - 复用train模式的保存逻辑
            detailed_file = f"prediction_detailed_{model_name}_{timestamp}.json"
            serializable_results = self._make_json_serializable(results)

            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"详细预测结果已保存到: {detailed_file}")

        except Exception as e:
            self.logger.error(f"保存预测结果失败: {e}")

    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # numpy标量
            return obj.item()
        else:
            return obj


def main():
    """主函数 - 支持训练和独立预测模式"""
    import argparse

    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                       help='运行模式: train(训练) 或 predict(预测)')
    parser.add_argument('--model-file', type=str,
                       help='预测模式下使用的模型文件路径')
    parser.add_argument('--start-date', type=str, default='20230601',
                       help='开始日期 (格式: YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default='20250331',
                       help='结束日期 (格式: YYYYMMDD)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='训练数据比例 (仅训练模式)')

    args = parser.parse_args()

    system = OptimizedMLSystem()

    if args.mode == 'train':
        # 训练模式
        print("🚀 开始训练模式...")
        results = system.run_complete_analysis(
            start_date=args.start_date,
            end_date=args.end_date,
            train_ratio=args.train_ratio
        )

        if results['success']:
            print("\n🎉 训练完成！")
            print(f"模型文件: {results['training_results']['system_file']}")
            print(f"最佳子状态: {results['final_report']['training_summary']['best_sub_states']}")
            print(f"总胜率: {results['final_report']['prediction_summary']['overall_win_rate']:.2%}")
            print(f"累计收益: {results['final_report']['prediction_summary']['cumulative_return']:.4f}")
            print(f"年化收益率: {results['final_report']['prediction_summary']['annualized_return']:.2%}")
            print(f"夏普比率: {results['final_report']['prediction_summary']['sharpe_ratio']:.3f}")
        else:
            print(f"\n❌ 训练失败: {results['error']}")
            return 1

    elif args.mode == 'predict':
        # 预测模式 - 自动获取最新训练模型
        print("🔮 开始预测模式...")

        # 🔥 修改：自动获取最新训练模型，不需要手动指定
        model_file = args.model_file if args.model_file else None

        results = system.predict_with_saved_model(
            model_file=model_file,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if results['success']:
            print("\n🎉 预测完成！")
            print(f"预测期间: {results['prediction_period']}")
            print(f"最佳子状态: {results['best_sub_states']}")

            # 整体表现
            overall = results['overall_summary']
            print(f"\n📊 整体表现:")
            print(f"  总选股次数: {overall['total_selections']}")
            print(f"  总胜率: {overall['overall_win_rate']:.2%}")
            print(f"  累计收益: {overall['cumulative_return']:.4f}")
            print(f"  最大回撤: {overall['max_drawdown']:.2%}")
            print(f"  年化收益率: {overall['annualized_return']:.2%}")
            print(f"  夏普比率: {overall['sharpe_ratio']:.3f}")

            # 分子状态表现
            print(f"\n🔍 分子状态表现:")
            for sub_state, perf in results['sub_state_results'].items():
                if perf['selections'] > 0:
                    print(f"\n  📈 {sub_state}:")
                    print(f"    选股次数: {perf['selections']}")
                    print(f"    胜率: {perf['win_rate']:.2%}")
                    print(f"    累计收益: {perf['cumulative_return']:.4f}")
                    print(f"    最大回撤: {perf['max_drawdown']:.2%}")
                    print(f"    年化收益率: {perf['annualized_return']:.2%}")
                    print(f"    夏普比率: {perf['sharpe_ratio']:.3f}")
                    print(f"    交易日数: {perf['trading_days']}")
                else:
                    print(f"\n  📈 {sub_state}: 无选股记录")

            print(f"\n💾 详细结果已保存到JSON文件")
        else:
            print(f"\n❌ 预测失败: {results['error']}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
