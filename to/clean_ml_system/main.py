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

from data_loader import DataLoader
from advanced_sector_classifier import AdvancedSectorClassifier
from professional_stock_selection_system import ProfessionalStockSelectionSystem, DailySelectionResult

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class OptimizedMLSystem:
    """优化的机器学习选股系统"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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
        """训练选股规则"""
        try:
            self.logger.info("准备训练数据...")

            # 加载训练数据
            data = self.data_loader.load_all_data(start_date, end_date)
            sector_features = data['sector_features']
            stock_features = data['stock_features']

            self.logger.info(f"板块特征: {len(sector_features)} 条记录")
            self.logger.info(f"个股特征: {len(stock_features)} 条记录")

            # 板块状态分类
            self.logger.info("进行板块状态分类...")
            sector_states = self.state_classifier.classify_sectors(sector_features)

            # 准备训练数据
            self.logger.info("整合训练数据...")
            training_data = self._prepare_training_data(sector_states, stock_features)

            if not training_data:
                return {
                    'success': False,
                    'error': '训练数据准备失败'
                }

            # 训练所有子状态的选股规则
            self.logger.info("训练所有子状态的选股规则...")
            all_rules = self.selection_system.train_all_sub_state_rules(training_data)

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

            return {
                'success': True,
                'all_rules': all_rules,
                'best_sub_states': best_sub_states,
                'system_file': system_file,
                'training_period': f"{start_date} - {end_date}"
            }

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
                    # 只保留必要的列以节省内存
                    essential_cols = ['stock_code', 'date', 'future_return_2d'] + [
                        col for col in daily_stocks.columns
                        if col not in ['stock_code', 'date', 'future_return_2d'] and
                        daily_stocks[col].dtype in ['float64', 'float32', 'int64', 'int32']
                    ][:20]  # 只保留前20个数值特征

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

    def predict_with_saved_model(self, model_file: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """使用保存的模型进行独立预测 - 分子状态详细评估"""
        try:
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

    def _collect_sub_state_data_streaming(self, state_sectors: pd.DataFrame,
                                        stock_by_date: dict, sub_state: str) -> pd.DataFrame:
        """
        流式收集子状态数据，避免内存溢出 - 增强版

        Args:
            state_sectors: 该子状态的板块数据
            stock_by_date: 按日期分组的股票数据
            sub_state: 子状态名称

        Returns:
            合并后的有效数据
        """
        import gc

        # 估算数据量并设置限制
        total_sectors = len(state_sectors)
        estimated_records = total_sectors * 5000  # 估算每个板块日期约5000只股票
        max_records = 5000000  # 500万条记录的硬限制

        # 如果估算数据量过大，进行预采样
        if estimated_records > max_records:
            sample_ratio = max_records / estimated_records
            state_sectors = state_sectors.sample(frac=sample_ratio, random_state=42)
            self.logger.warning(f"子状态 {sub_state} 数据量过大，预采样比例: {sample_ratio:.3f}")

        # 使用更小的批次大小
        batch_size = 5  # 减少到每批5个板块日期
        all_valid_data = []
        max_batch_records = 200000  # 每批最多20万条记录

        # 将板块数据分批
        sector_batches = [state_sectors.iloc[i:i+batch_size]
                         for i in range(0, len(state_sectors), batch_size)]

        for batch_idx, sector_batch in enumerate(sector_batches):
            self.logger.debug(f"  处理批次 {batch_idx+1}/{len(sector_batches)}")

            batch_stocks = []
            batch_record_count = 0

            for _, sector_row in sector_batch.iterrows():
                sector_date = sector_row['date']

                # 从预处理的数据中获取当日股票数据
                if sector_date in stock_by_date:
                    daily_stocks = stock_by_date[sector_date]

                    if not daily_stocks.empty:
                        # 检查是否会超过批次限制
                        if batch_record_count + len(daily_stocks) > max_batch_records:
                            # 对当前日期数据进行采样
                            sample_size = max_batch_records - batch_record_count
                            if sample_size > 0:
                                daily_stocks = daily_stocks.sample(n=sample_size, random_state=42)
                            else:
                                break  # 跳出当前批次

                        # 添加子状态标记
                        daily_stocks_copy = daily_stocks.copy()
                        daily_stocks_copy['sub_state'] = sub_state
                        daily_stocks_copy['sector_date'] = sector_date

                        batch_stocks.append(daily_stocks_copy)
                        batch_record_count += len(daily_stocks_copy)

            if batch_stocks:
                try:
                    # 合并当前批次数据
                    batch_combined = pd.concat(batch_stocks, ignore_index=True)

                    # 检查目标变量并清理数据
                    target_col = 'future_return_2d'
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

    def _run_prediction_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """运行预测分析"""
        try:
            self.logger.info("加载预测数据...")

            # 加载预测数据
            data = self.data_loader.load_all_data(start_date, end_date)
            sector_features = data['sector_features']
            stock_features = data['stock_features']

            self.logger.info(f"板块特征: {len(sector_features)} 条记录")
            self.logger.info(f"个股特征: {len(stock_features)} 条记录")

            # 板块状态分类
            self.logger.info("进行板块状态分类...")
            sector_states = self.state_classifier.classify_sectors(sector_features)

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

            self.logger.info(f"开始逐日预测，共 {len(dates)} 个交易日")

            for i, date in enumerate(dates):
                if i % 10 == 0:  # 每10天报告一次进度
                    self.logger.info(f"预测进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")

                # 预测选股
                selected_stocks = self.selection_system.predict_and_select_stocks(
                    sector_states, stock_features, str(date)
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

                        if not stock_data.empty and 'future_return_2d' in stock_data.columns:
                            actual_return = stock_data['future_return_2d'].iloc[0]
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

                    # 按子状态分别统计
                    for stock in selected_stocks:
                        sub_state = stock.sub_state
                        if sub_state in sub_state_stats:
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
                    for sub_state in sub_state_stats:
                        if any(stock.sub_state == sub_state for stock in selected_stocks):
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

            # 汇总结果
            prediction_results = {
                'success': True,
                'prediction_period': f"{start_date} - {end_date}",
                'total_trading_days': len(dates),
                'active_trading_days': trading_days,
                'total_selections': total_selections,
                'total_wins': total_wins,
                'overall_win_rate': total_wins / total_selections if total_selections > 0 else 0,
                'cumulative_return': cumulative_return,
                'avg_daily_return': avg_daily_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'avg_selections_per_day': total_selections / trading_days if trading_days > 0 else 0,
                'target_states_frequency': target_states_count,
                'sub_state_performance': sub_state_performance,  # 新增子状态分别性能
                'daily_summary': daily_results,
                'detailed_daily_records': detailed_daily_records,
                'best_sub_states': self.selection_system.best_sub_states
            }

            return prediction_results

        except Exception as e:
            self.logger.error(f"预测分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

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

        # 预测阶段摘要
        prediction_summary = {
            'period': f"{test_start} - {test_end}",
            'total_trading_days': prediction_results.get('total_trading_days', 0),
            'active_trading_days': prediction_results.get('active_trading_days', 0),
            'total_selections': prediction_results.get('total_selections', 0),
            'overall_win_rate': prediction_results.get('overall_win_rate', 0),
            'cumulative_return': prediction_results.get('cumulative_return', 0),
            'avg_daily_return': prediction_results.get('avg_daily_return', 0),
            'annualized_return': prediction_results.get('annualized_return', 0),
            'volatility': prediction_results.get('volatility', 0),
            'sharpe_ratio': prediction_results.get('sharpe_ratio', 0),
            'max_drawdown': prediction_results.get('max_drawdown', 0)
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

        sub_state_perf = prediction_results.get('sub_state_performance', {})
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
        # 预测模式
        if not args.model_file:
            print("❌ 预测模式需要指定 --model-file 参数")
            return 1

        print(f"🔮 开始预测模式，使用模型: {args.model_file}")
        results = system.predict_with_saved_model(
            model_file=args.model_file,
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
