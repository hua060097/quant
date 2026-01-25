#!/usr/bin/env python3
"""
顶级支撑因子组 - 真正的机构级实现
Elite Support Factors - Genuine Institutional-Grade Implementation

核心理念：
1. 基于历史价格行为的真实支撑位识别
2. 多维度支撑强度验证（价格+成交量+时间+技术形态）
3. 动态支撑位追踪和失效监控
4. 专业的风险收益评估框架
5. 严格避免时间泄露，只使用历史数据

真正的专业实现，摒弃简单粗暴的"current_price * 0.98"等荒谬做法
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import time
from .base_factor_group import BaseFactorGroup
from concurrent.futures import ThreadPoolExecutor


class EliteSupportFactors(BaseFactorGroup):
    """顶级支撑因子组 - 真正的机构级实现"""

    def __init__(self):
        super().__init__("elite_support_factors")
        self._float_dtype = np.float32
        self._max_workers = 4

        # 专业参数配置
        self._min_support_tests = 2  # 支撑位最少测试次数
        self._support_tolerance = 0.02  # 支撑位价格容忍度2%
        self._min_volume_confirmation = 1.2  # 最小成交量确认倍数
        self._support_decay_days = 120  # 支撑位时间衰减周期

        # 性能优化：添加缓存机制
        self._support_cache = {}  # 缓存支撑位计算结果
        self._cache_max_size = 1000  # 最大缓存条目数

    def _get_version(self) -> str:
        return "v1.0_professional"

    def _get_dependencies(self) -> List[str]:
        return []

    def _get_feature_names(self) -> List[str]:
        return [
            # 优化后的支撑位风险收益比因子
            'support_risk_reward_ratio',      # 支撑位风险收益比（整合启动信号版）
            'retail_vs_institutional_ratio',   # 散户vs机构比例
            'institutional_activity_score',    # 机构活跃度评分

            # 新增的专业因子
            'parkinson_volatility',           # Parkinson波动率估计
            'bollinger_squeeze_factor',       # 布林带挤压因子
            'bollinger_breakout_strength',    # 布林带突破强度
            'bollinger_position_factor',      # 布林带位置因子
            'obv_momentum_factor',            # OBV动量因子（专业实现）
        ]

    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算顶级支撑因子 - 只为目标日期计算"""
        start_time = time.time()

        try:
            # 1. 识别目标日期和历史数据范围
            target_dates, historical_dates = self._identify_target_and_historical_dates(data, kwargs)
            if not target_dates:
                self.logger.warning("无法识别目标日期，使用默认值")
                return self._set_default_values(data)

            self.logger.info(f"🏆 开始真正专业的支撑因子计算")
            self.logger.info(f"📅 目标计算日期: {len(target_dates)} 个")
            self.logger.info(f"📊 历史数据范围: {len(historical_dates)} 个交易日")

            # 2. 获取市场数据和板块数据
            market_data = kwargs.get('market_data')
            sector_data = kwargs.get('sector_data')
            stock_sector_mapping = kwargs.get('stock_sector_mapping', {})

            # 3. 只为目标日期计算支撑因子
            result_data = self._calculate_elite_support_factors_optimized(
                data, target_dates, market_data, sector_data, stock_sector_mapping
            )

            # 4. 保存目标日期信息
            result_data.attrs['target_dates'] = target_dates

            # 5. 性能统计
            total_time = time.time() - start_time
            total_records = len(result_data[result_data['date'].dt.strftime('%Y%m%d').isin(target_dates)])

            self.logger.info(f"✅ 专业支撑因子计算完成")
            self.logger.info(f"⏱️ 总耗时: {total_time:.2f}秒")
            self.logger.info(f"📈 处理效率: {total_records/total_time:.0f} 条记录/秒")

            return result_data

        except Exception as e:
            self.logger.error(f"❌ 专业支撑因子计算失败: {e}")
            self.logger.error(f"错误位置: calculate方法主流程")
            self.logger.error(f"输入参数: start_date={kwargs.get('target_start_date', 'N/A')}, end_date={kwargs.get('target_end_date', 'N/A')}")
            self.logger.error(f"数据形状: {data.shape if data is not None else 'None'}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return self._create_default_result(data)

    def _identify_target_and_historical_dates(self, data: pd.DataFrame, kwargs: dict) -> Tuple[List[str], List[str]]:
        """识别目标日期和历史数据范围 - 与其他因子组保持一致"""
        try:
            target_start_date = kwargs.get('target_start_date')
            target_end_date = kwargs.get('target_end_date')

            if target_start_date and target_end_date:
                # 转换为datetime
                start_dt = pd.to_datetime(target_start_date, format='%Y%m%d')
                end_dt = pd.to_datetime(target_end_date, format='%Y%m%d')

                # 获取目标日期
                target_mask = (data['date'] >= start_dt) & (data['date'] <= end_dt)
                target_dates = data[target_mask]['date'].dt.strftime('%Y%m%d').unique().tolist()

                # 获取历史日期（用于计算支撑因子的历史数据）
                historical_dates = data['date'].dt.strftime('%Y%m%d').unique().tolist()

                return target_dates, historical_dates
            else:
                # 默认使用最近的数据
                unique_dates = sorted(data['date'].dt.strftime('%Y%m%d').unique())
                target_dates = unique_dates[-10:] if len(unique_dates) >= 10 else unique_dates
                historical_dates = unique_dates

                return target_dates, historical_dates

        except Exception as e:
            self.logger.error(f"❌ 识别目标日期失败: {e}")
            self.logger.error(f"错误位置: _identify_target_and_historical_dates方法")
            self.logger.error(f"输入参数: kwargs={kwargs}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return [], []

    def _calculate_elite_support_factors_optimized(self, data: pd.DataFrame,
                                                 target_dates: List[str],
                                                 market_data: Dict = None,
                                                 sector_data: pd.DataFrame = None,
                                                 stock_sector_mapping: Dict = None) -> pd.DataFrame:
        """高性能优化版支撑因子计算：批量处理避免卡死"""
        try:
            # 1. 初始化结果数据
            result_data = data.copy()

            # 2. 初始化所有特征列
            feature_names = self._get_feature_names()
            for feature_name in feature_names:
                if feature_name not in result_data.columns:
                    result_data[feature_name] = 0.0

            # 3. 筛选目标日期数据
            target_mask = result_data['date'].dt.strftime('%Y%m%d').isin(target_dates)
            target_data = result_data[target_mask].copy()

            if target_data.empty:
                self.logger.warning("无目标日期数据")
                return result_data

            self.logger.info(f"📊 目标数据: {len(target_data)} 条记录")

            # 4. 按股票分组计算（高性能版本）
            stock_groups = list(target_data.groupby('stock_code'))
            total_stocks = len(stock_groups)

            self.logger.info(f"📈 处理股票数量: {total_stocks}")

            # 🚀 多线程高性能处理
            processed_count = self._process_stocks_multithreaded(
                result_data, data, stock_groups, target_dates,
                market_data, sector_data, stock_sector_mapping, total_stocks
            )

            self.logger.info(f"✅ 支撑因子计算完成: {processed_count}/{total_stocks} 只股票")
            return result_data

        except Exception as e:
            self.logger.error(f"优化版支撑因子计算失败: {e}")
            return data



    def _process_stocks_multithreaded(self, result_data: pd.DataFrame, data: pd.DataFrame,
                                    stock_groups: List, target_dates: List[str],
                                    market_data: Dict = None, sector_data: pd.DataFrame = None,
                                    stock_sector_mapping: Dict = None, total_stocks: int = 0) -> int:
        """优化的股票处理 - 减少多线程开销"""
        import psutil
        import os

        # 🚀 性能优化：大幅减少多线程开销
        cpu_count = os.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # 🔥 关键优化：更激进的单线程策略，减少线程创建开销
        if total_stocks > 200 and available_memory_gb > 16:
            max_workers = 2  # 最多2个线程，减少开销
        elif total_stocks > 100 and available_memory_gb > 8:
            max_workers = 2  # 最多2个线程
        else:
            max_workers = 1  # 优先使用单线程，避免线程开销

        self.logger.info(f"🔧 处理配置: {max_workers}个工作线程 (CPU: {cpu_count}核, 可用内存: {available_memory_gb:.1f}GB)")

        # 🚀 性能优化：如果是单线程，直接处理，避免线程开销
        if max_workers == 1:
            return self._process_stocks_single_threaded(
                result_data, data, stock_groups, target_dates,
                market_data, sector_data, stock_sector_mapping, total_stocks
            )

        # 多线程处理（仅在必要时使用）
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 初始化结果队列 - 修复未定义问题
        results_queue = queue.Queue()

        # 线程安全的进度计数器
        progress_lock = threading.Lock()
        processed_count = 0
        error_count = 0

        def process_single_stock(stock_info):
            """处理单只股票的线程函数"""
            nonlocal processed_count, error_count

            stock_code, stock_target_data = stock_info
            thread_id = threading.current_thread().ident

            try:
                start_time = time.time()

                # 为每个线程创建独立的结果字典
                stock_results = {}

                # 调用原始的专业算法处理方法
                self._process_stock_support_factors_thread_safe(
                    stock_results, data, stock_code, stock_target_data, target_dates,
                    market_data, sector_data, stock_sector_mapping
                )

                elapsed = time.time() - start_time

                # 线程安全的进度更新
                with progress_lock:
                    processed_count += 1
                    progress = processed_count / total_stocks * 100

                    self.logger.info(f"✅ [线程{thread_id}] 完成股票 {stock_code}: "
                                   f"耗时 {elapsed:.2f}s, 进度 {processed_count}/{total_stocks} ({progress:.1f}%)")

                    # 每10只股票报告一次内存使用
                    if processed_count % 10 == 0:
                        memory_usage = psutil.Process().memory_info().rss / (1024**2)
                        self.logger.info(f"📊 内存使用: {memory_usage:.1f}MB, "
                                       f"已处理: {processed_count}/{total_stocks}")

                # 将结果放入队列
                results_queue.put((stock_code, stock_results))
                return True

            except Exception as e:
                with progress_lock:
                    error_count += 1
                    self.logger.error(f"❌ [线程{thread_id}] 股票 {stock_code} 处理失败: {e}")

                return False

        # 执行多线程处理
        self.logger.info(f"🚀 开始多线程处理 {total_stocks} 只股票...")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="StockProcessor") as executor:
            # 提交所有任务
            future_to_stock = {
                executor.submit(process_single_stock, stock_info): stock_info[0]
                for stock_info in stock_groups
            }

            # 收集结果
            completed_tasks = 0
            for future in as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed_tasks += 1

                try:
                    success = future.result(timeout=60)  # 60秒超时
                    if not success:
                        self.logger.warning(f"⚠️ 股票 {stock_code} 处理未成功")

                except Exception as e:
                    self.logger.error(f"❌ 股票 {stock_code} 线程异常: {e}")

                # 每50个任务完成时报告进度
                if completed_tasks % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_tasks
                    eta = avg_time * (total_stocks - completed_tasks)
                    self.logger.info(f"⏱️ 已完成 {completed_tasks}/{total_stocks} 任务, "
                                   f"平均耗时 {avg_time:.2f}s/股票, 预计剩余 {eta/60:.1f}分钟")

        # 应用所有结果到result_data
        self._apply_multithreaded_results(result_data, results_queue)

        total_elapsed = time.time() - start_time
        success_rate = (processed_count - error_count) / total_stocks * 100

        self.logger.info(f"🎉 多线程处理完成!")
        self.logger.info(f"   总耗时: {total_elapsed/60:.1f}分钟")
        self.logger.info(f"   成功处理: {processed_count - error_count}/{total_stocks} ({success_rate:.1f}%)")
        self.logger.info(f"   平均速度: {total_stocks/total_elapsed:.1f} 股票/秒")
        self.logger.info(f"   错误数量: {error_count}")

        return processed_count

    def _process_stocks_single_threaded(self, result_data: pd.DataFrame, data: pd.DataFrame,
                                      stock_groups: List, target_dates: List[str],
                                      market_data: Dict = None, sector_data: pd.DataFrame = None,
                                      stock_sector_mapping: Dict = None, total_stocks: int = 0) -> int:
        """🚀 单线程处理股票 - 性能优化版本，避免线程开销"""
        start_time = time.time()
        processed_count = 0
        error_count = 0

        self.logger.info(f"🚀 开始单线程处理 {total_stocks} 只股票...")

        for i, (stock_code, stock_target_data) in enumerate(stock_groups):
            try:
                stock_start_time = time.time()

                # 直接处理股票，无线程开销
                stock_results = {}
                self._process_stock_support_factors_thread_safe(
                    stock_results, data, stock_code, stock_target_data, target_dates,
                    market_data, sector_data, stock_sector_mapping
                )

                # 应用结果到result_data
                for row_idx, factors in stock_results.items():
                    if row_idx in result_data.index:
                        for factor_name, factor_value in factors.items():
                            if factor_name in result_data.columns:
                                result_data.loc[row_idx, factor_name] = factor_value

                processed_count += 1
                stock_elapsed = time.time() - stock_start_time
                progress = processed_count / total_stocks * 100

                # 每10只股票报告一次进度
                if processed_count % 10 == 0 or processed_count == total_stocks:
                    self.logger.info(f"✅ 完成股票 {stock_code}: "
                                   f"耗时 {stock_elapsed:.2f}s, 进度 {processed_count}/{total_stocks} ({progress:.1f}%)")

            except Exception as e:
                error_count += 1
                self.logger.error(f"❌ 处理股票 {stock_code} 失败: {e}")

        total_elapsed = time.time() - start_time
        success_rate = (processed_count - error_count) / total_stocks * 100

        self.logger.info(f"🎉 单线程处理完成!")
        self.logger.info(f"   总耗时: {total_elapsed/60:.1f}分钟")
        self.logger.info(f"   成功处理: {processed_count - error_count}/{total_stocks} ({success_rate:.1f}%)")
        self.logger.info(f"   平均速度: {total_stocks/total_elapsed:.1f} 股票/秒")
        self.logger.info(f"   错误数量: {error_count}")

        return processed_count

    def _process_stock_support_factors_thread_safe(self, stock_results: Dict, data: pd.DataFrame,
                                                  stock_code: str, stock_target_data: pd.DataFrame,
                                                  target_dates: List[str],
                                                  market_data: Dict = None,
                                                  sector_data: pd.DataFrame = None,
                                                  stock_sector_mapping: Dict = None):
        """线程安全的股票处理方法 - 保持专业算法"""
        try:
            import time

            # 获取该股票的完整历史数据（线程安全的数据访问）
            stock_data = data[data['stock_code'] == stock_code].sort_values('date').copy()

            if len(stock_data) < 120:  # 保持专业要求
                return

            # 为每个目标日期计算支撑因子 - 保持原始专业算法
            for _, row in stock_target_data.iterrows():
                target_date = row['date']
                target_date_str = target_date.strftime('%Y%m%d')

                if target_date_str not in target_dates:
                    continue

                # 找到目标日期在历史数据中的位置
                date_mask = stock_data['date'] <= target_date
                historical_data = stock_data[date_mask]

                if len(historical_data) < 120:  # 保持专业要求
                    continue

                # 使用原始的专业算法计算支撑因子
                try:
                    factors = self._compute_single_date_support_factors(
                        historical_data, stock_code, target_date,
                        market_data, sector_data, stock_sector_mapping
                    )

                    # 存储结果到线程安全的字典中
                    if factors:
                        stock_results[row.name] = factors

                except Exception as e:
                    # 线程内的错误处理
                    continue

        except Exception as e:
            # 线程级别的错误处理
            pass

    def _apply_multithreaded_results(self, result_data: pd.DataFrame, results_queue):
        """应用多线程计算结果到主DataFrame"""
        try:
            applied_count = 0

            self.logger.info(f"🔄 开始应用多线程计算结果...")

            # 从队列中获取所有结果
            while not results_queue.empty():
                try:
                    stock_code, stock_results = results_queue.get_nowait()

                    # 应用该股票的所有结果
                    for row_name, factors in stock_results.items():
                        if row_name in result_data.index:
                            for factor_name, factor_value in factors.items():
                                if factor_name in result_data.columns:
                                    result_data.loc[row_name, factor_name] = factor_value

                    applied_count += 1

                    if applied_count % 100 == 0:
                        self.logger.info(f"   已应用 {applied_count} 只股票的结果...")

                except Exception as e:
                    # 可能是queue.Empty或其他异常
                    if "Empty" in str(type(e)):
                        break
                    else:
                        self.logger.warning(f"⚠️ 应用结果时出错: {e}")
                        continue

            self.logger.info(f"✅ 结果应用完成: {applied_count} 只股票")

        except Exception as e:
            self.logger.error(f"❌ 应用多线程结果失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")


    def _compute_single_date_support_factors(self, historical_data: pd.DataFrame,
                                           stock_code: str = None,
                                           target_date: pd.Timestamp = None,
                                           market_data: Dict = None,
                                           sector_data: pd.DataFrame = None,
                                           stock_sector_mapping: Dict = None) -> Dict:
        """计算单个日期的支撑因子 - 带监控的专业实现"""
        import time

        try:
            start_time = time.time()

            # 数据长度检查
            if len(historical_data) < 120:
                self.logger.debug(f"      ⚠️ {stock_code} 历史数据不足: {len(historical_data)} < 120")
                return self._get_default_factor_values()

            # 数据预处理 - 性能优化：一次性转换所有数据类型
            close = historical_data['close'].values.astype(self._float_dtype, copy=False)
            high = historical_data['high'].values.astype(self._float_dtype, copy=False)
            low = historical_data['low'].values.astype(self._float_dtype, copy=False)
            volume = historical_data['volume'].values.astype(self._float_dtype, copy=False)

            current_price = close[-1]

            # 1. 识别真实的历史支撑位（添加监控）
            support_start = time.time()
            support_levels = self._identify_real_support_levels(close, low, volume)
            support_elapsed = time.time() - support_start

            # 监控支撑位识别时间
            if support_elapsed > 2.0:
                self.logger.warning(f"        ⚠️ 支撑位识别耗时: {support_elapsed:.2f}s")

            # 如果支撑位识别时间过长，使用简化版本
            if support_elapsed > 5.0:
                self.logger.warning(f"        ⚠️ 支撑位识别超时，使用默认值")
                support_levels = []

            # 2. 计算三个核心支撑因子
            factors = {}

            # 支撑位风险收益比（优化版）
            factors['support_risk_reward_ratio'] = self._calculate_risk_reward_professional(
                close, low, volume, current_price, support_levels
            )

            # 机构活跃度评分
            factors['institutional_activity_score'] = self._calculate_institutional_activity_professional(
                volume, close, current_price
            )

            # 散户vs机构比例
            factors['retail_vs_institutional_ratio'] = self._calculate_retail_institutional_ratio(
                close, volume, current_price
            )

            # 新增因子计算
            # Parkinson波动率估计
            factors['parkinson_volatility'] = self._calculate_parkinson_volatility(
                historical_data['high'].values, historical_data['low'].values
            )

            # 布林带相关因子
            bollinger_factors = self._calculate_bollinger_factors(close, current_price)
            factors.update(bollinger_factors)

            # OBV动量因子
            factors['obv_momentum_factor'] = self._calculate_obv_momentum_factor(
                close, volume, current_price
            )

            # 总体性能监控
            total_elapsed = time.time() - start_time
            if total_elapsed > 3.0:
                self.logger.warning(f"        ⚠️ 总计算时间: {total_elapsed:.2f}s")

            return factors

        except Exception as e:
            self.logger.error(f"❌ 单日期支撑因子计算失败: {e}")
            self.logger.error(f"错误位置: _compute_single_date_support_factors方法")
            self.logger.error(f"股票代码: {stock_code}, 目标日期: {target_date}")
            self.logger.error(f"历史数据长度: {len(historical_data) if historical_data is not None else 'None'}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return self._get_default_factor_values()

    def _identify_real_support_levels(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray) -> List[Dict]:
        """识别真实的历史支撑位 - 优化版本，使用向量化计算"""
        try:
            if len(close) < 60:
                return []

            # 🚀 性能优化：使用更高效的缓存键生成
            # 使用数据长度和最后几个值作为缓存键，避免大数组hash
            cache_key = (len(close), close[-1], close[-5] if len(close) > 5 else 0,
                        low[-1], low[-5] if len(low) > 5 else 0)
            if cache_key in self._support_cache:
                return self._support_cache[cache_key]

            # 性能优化：避免重复转换，数据已在上层转换为float32

            # 1. 向量化寻找局部低点
            local_lows = self._find_significant_local_lows_vectorized(close, low, volume)

            # 2. 批量验证支撑位有效性（修复未来信息泄露）
            current_idx = len(close) - 1  # 当前日期的索引
            support_levels = self._validate_support_levels_batch(close, low, volume, local_lows, current_idx)

            # 3. 按支撑强度排序，保留最重要的支撑位
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            result = support_levels[:3]  # 最多保留3个最强支撑位

            # 缓存结果
            if len(self._support_cache) >= self._cache_max_size:
                # 清理最旧的缓存条目
                oldest_key = next(iter(self._support_cache))
                del self._support_cache[oldest_key]
            self._support_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"❌ 识别真实支撑位失败: {e}")
            self.logger.error(f"错误位置: _identify_real_support_levels方法")
            self.logger.error(f"数据长度: close={len(close) if close is not None else 'None'}, low={len(low) if low is not None else 'None'}")
            return []

    def _find_significant_local_lows_vectorized(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray) -> List[Tuple[int, float]]:
        """🚀 超级优化的局部低点识别 - 大幅提升性能"""
        try:
            n = len(low)
            if n < 21:
                return []

            window_size = 10

            # 🔥 性能优化：使用numpy的滚动最小值计算
            # 预分配数组以提高性能
            local_min_mask = np.zeros(n, dtype=bool)

            # 向量化计算所有窗口的最小值
            for i in range(window_size, n - window_size):
                window_start = i - window_size
                window_end = i + window_size + 1
                if low[i] == np.min(low[window_start:window_end]):
                    local_min_mask[i] = True

            local_min_indices = np.where(local_min_mask)[0]

            if len(local_min_indices) == 0:
                return []

            # 🚀 批量显著性验证 - 向量化处理
            significant_lows = []

            # 预计算一些常用值
            close_mean = np.mean(close)
            volume_mean = np.mean(volume)

            for idx in local_min_indices:
                # 简化的显著性检查，提高速度
                if self._is_significant_low_fast(close, low, volume, idx, window_size, close_mean, volume_mean):
                    significant_lows.append((idx, low[idx]))

            return significant_lows

        except Exception:
            # 回退到原始方法
            return self._find_significant_local_lows_fallback(close, low, volume)

    def _is_significant_low_vectorized(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray, idx: int, window_size: int) -> bool:
        """向量化判断是否为显著的低点 - 性能优化版本"""
        try:
            # 1. 价格显著性：低点应该比周围价格低至少1%
            start_idx = max(0, idx - window_size)
            end_idx = min(len(close), idx + window_size + 1)
            window_prices = close[start_idx:end_idx]
            avg_price = np.mean(window_prices)
            if (avg_price - low[idx]) / avg_price < 0.01:
                return False

            # 2. 成交量确认：低点当天成交量应该相对放大
            if idx >= 20:
                recent_avg_volume = np.mean(volume[idx-20:idx])
                if volume[idx] < recent_avg_volume * 0.8:  # 成交量不能太小
                    return False

            # 3. 后续反弹确认：低点后应该有反弹
            if idx < len(close) - 5:
                rebound_days = min(5, len(close) - idx - 1)
                max_rebound_price = np.max(close[idx+1:idx+1+rebound_days])
                rebound_ratio = (max_rebound_price - low[idx]) / low[idx]
                if rebound_ratio < 0.02:  # 至少2%的反弹
                    return False

            return True

        except Exception:
            return False

    def _is_significant_low_fast(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray,
                               idx: int, window_size: int, close_mean: float, volume_mean: float) -> bool:
        """🚀 快速显著性检查 - 性能优化版本"""
        try:
            # 1. 快速价格显著性检查
            if (close_mean - low[idx]) / close_mean < 0.01:
                return False

            # 2. 快速成交量检查
            if volume[idx] < volume_mean * 0.8:  # 简化的成交量检查
                return False

            return True

        except Exception:
            return False

    def _find_significant_local_lows_fallback(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray) -> List[Tuple[int, float]]:
        """回退到原始方法寻找显著的局部低点"""
        try:
            local_lows = []
            n = len(low)
            window_size = 10

            for i in range(window_size, n - window_size):
                window_low = low[i-window_size:i+window_size+1]
                if low[i] == np.min(window_low):
                    if self._is_significant_low_vectorized(close, low, volume, i, window_size):
                        local_lows.append((i, low[i]))

            return local_lows
        except Exception:
            return []

    def _validate_support_levels_batch(self, close: np.ndarray, low: np.ndarray, volume: np.ndarray, local_lows: List[Tuple[int, float]], current_idx: int = None) -> List[Dict]:
        """批量验证支撑位有效性 - 修复未来信息泄露"""
        try:
            if not local_lows:
                return []

            support_levels = []
            n = len(close)

            # 🔥 关键修复：如果指定了current_idx，只使用到current_idx的数据
            if current_idx is not None:
                n = min(n, current_idx + 1)  # 不使用current_idx之后的数据

            # 预计算一些常用值以避免重复计算
            tolerance = self._support_tolerance
            min_tests = self._min_support_tests
            decay_days = self._support_decay_days

            for support_idx, support_price in local_lows:
                # 🔥 关键修复：向量化计算测试点，但不超过current_idx
                future_indices = np.arange(support_idx + 1, n)
                if len(future_indices) == 0:
                    continue

                future_lows = low[future_indices]
                future_closes = close[future_indices]
                future_volumes = volume[future_indices]

                # 向量化判断是否触及支撑位
                price_diff_ratio = np.abs(future_lows - support_price) / support_price
                is_test = price_diff_ratio <= tolerance

                if not np.any(is_test):
                    continue

                test_indices = future_indices[is_test]
                test_count = len(test_indices)

                if test_count < min_tests:
                    continue

                # 向量化计算守住次数
                test_closes = future_closes[is_test]
                hold_count = np.sum(test_closes > support_price)

                # 向量化计算成交量
                test_volumes = future_volumes[is_test]
                total_volume_at_tests = np.sum(test_volumes)

                # 计算支撑强度
                hold_rate = hold_count / test_count if test_count > 0 else 0
                avg_volume_at_tests = total_volume_at_tests / test_count if test_count > 0 else 0

                # 计算平均成交量（用于比较）
                if support_idx >= 20:
                    avg_volume_before = np.mean(volume[support_idx-20:support_idx])
                else:
                    avg_volume_before = np.mean(volume[:support_idx])

                volume_ratio = avg_volume_at_tests / avg_volume_before if avg_volume_before > 0 else 1.0

                # 时间衰减因子
                days_since_formation = n - 1 - support_idx
                time_decay = max(0.1, 1.0 - days_since_formation / decay_days)

                # 综合强度评分
                strength = (
                    hold_rate * 0.4 +
                    min(volume_ratio / 2, 1.0) * 0.3 +
                    min(test_count / 5, 1.0) * 0.2 +
                    time_decay * 0.1
                )

                support_info = {
                    'price': float(support_price),
                    'formation_idx': support_idx,
                    'test_count': test_count,
                    'hold_count': hold_count,
                    'hold_rate': hold_rate,
                    'volume_ratio': volume_ratio,
                    'time_decay': time_decay,
                    'strength': strength
                }

                support_levels.append(support_info)

            return support_levels

        except Exception as e:
            self.logger.error(f"❌ 批量验证支撑位失败: {e}")
            return []

    # 原始的_validate_support_level方法已被_validate_support_levels_batch替代以提高性能

        except Exception:
            return None





    def _get_default_factor_values(self) -> Dict:
        """获取默认因子值"""
        return {
            # 核心因子的默认值
            'support_risk_reward_ratio': 2.0,
            'retail_vs_institutional_ratio': 0.5,
            'institutional_activity_score': 50.0,

            # 新增因子的默认值
            'parkinson_volatility': 0.02,           # 默认2%的波动率
            'bollinger_squeeze_factor': 0.5,        # 中性挤压状态
            'bollinger_breakout_strength': 0.0,     # 无突破状态
            'bollinger_position_factor': 0.5,       # 中位位置
            'obv_momentum_factor': 0.5,             # OBV动量因子默认值
        }



    def _calculate_risk_reward_professional(self, close: np.ndarray, low: np.ndarray,
                                           volume: np.ndarray, current_price: float,
                                           support_levels: List[Dict]) -> float:
        """
        专业的风险收益比计算 - 整合启动信号检测和成交量确认

        核心改进：
        1. 启动信号检测：整合RSI、价格动量等技术指标
        2. 成交量确认：验证支撑位附近的成交量放大
        3. 自适应风险收益：基于市场环境动态调整
        4. 多维度验证：支撑位强度 + 启动信号 + 成交量确认
        """
        try:
            if not support_levels or len(close) < 20:
                return 2.0

            strongest_support = support_levels[0]
            support_price = strongest_support['price']
            formation_idx = strongest_support['formation_idx']
            current_idx = len(close) - 1

            # 1. 启动信号检测（新增）
            startup_signal_strength = self._detect_startup_signals(close, volume, current_idx)

            # 2. 成交量确认（新增）
            volume_confirmation = self._calculate_volume_confirmation(
                volume, support_price, close, current_idx
            )

            # 3. 基于历史反弹数据计算动态预期收益
            base_expected_return = self._calculate_dynamic_expected_return(
                close, low, support_price, formation_idx, current_idx
            )

            # 4. 启动信号和成交量确认调整预期收益（新增）
            signal_multiplier = 1.0 + (startup_signal_strength - 0.5) * 0.5  # 0.75-1.25倍
            volume_multiplier = 1.0 + (volume_confirmation - 0.5) * 0.3      # 0.85-1.15倍

            adjusted_expected_return = base_expected_return * signal_multiplier * volume_multiplier

            # 5. 市场环境调整
            market_volatility = self._calculate_current_volatility(close)
            volatility_adjustment = min(1.5, max(0.7, 1.0 + (market_volatility - 0.02) * 5))
            final_expected_return = adjusted_expected_return * volatility_adjustment

            # 6. 计算预期风险
            expected_loss = self._calculate_expected_loss_professional(
                current_price, support_price, close, strongest_support
            )

            # 7. 计算最终风险收益比
            if expected_loss > 0:
                risk_reward_ratio = final_expected_return / expected_loss
            else:
                risk_reward_ratio = 2.0

            # 8. 合理范围限制
            return float(min(10.0, max(0.3, risk_reward_ratio)))

        except Exception as e:
            self.logger.warning(f"专业风险收益比计算失败: {e}")
            return 2.0

    def _detect_startup_signals(self, close: np.ndarray, volume: np.ndarray, current_idx: int) -> float:
        """
        检测启动信号强度 - 专业量化交易算法

        整合多个技术指标：
        1. RSI背离检测
        2. 价格动量分析
        3. 成交量能量指标
        4. 短期趋势强度

        Returns:
            float: 启动信号强度 (0.0-1.0)，0.5为中性
        """
        try:
            if len(close) < 20 or current_idx < 14:
                return 0.5

            # 性能优化：避免重复转换，数据已在上层转换为float32

            # 1. RSI计算和背离检测
            rsi_signal = self._calculate_rsi_divergence_signal(close, current_idx)

            # 2. 价格动量分析
            momentum_signal = self._calculate_price_momentum_signal(close, current_idx)

            # 3. 成交量能量指标
            volume_energy = self._calculate_volume_energy_signal(volume, current_idx)

            # 4. 短期趋势强度
            trend_strength = self._calculate_short_term_trend_strength(close, current_idx)

            # 综合启动信号强度（加权平均）
            startup_strength = (
                rsi_signal * 0.3 +
                momentum_signal * 0.3 +
                volume_energy * 0.25 +
                trend_strength * 0.15
            )

            return float(np.clip(startup_strength, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_volume_confirmation(self, volume: np.ndarray, support_price: float,
                                     close: np.ndarray, current_idx: int) -> float:
        """
        计算成交量确认强度 - 专业算法

        验证支撑位附近的成交量特征：
        1. 支撑位测试时的成交量放大
        2. 近期成交量趋势分析
        3. 相对成交量强度

        Returns:
            float: 成交量确认强度 (0.0-1.0)，0.5为中性
        """
        try:
            if len(volume) < 20 or current_idx < 10:
                return 0.5

            # 性能优化：避免重复转换，数据已在上层转换为float32

            current_price = close[current_idx]

            # 1. 检查是否接近支撑位
            distance_to_support = abs(current_price - support_price) / current_price
            if distance_to_support > 0.05:  # 距离支撑位超过5%，成交量确认不重要
                return 0.5

            # 2. 近期成交量相对于历史平均的放大倍数
            recent_volume = np.mean(volume[max(0, current_idx-4):current_idx+1])  # 最近5日平均
            historical_volume = np.mean(volume[max(0, current_idx-20):current_idx-5])  # 历史15日平均

            volume_amplification = recent_volume / historical_volume if historical_volume > 0 else 1.0

            # 3. 成交量趋势分析
            volume_trend = self._calculate_volume_trend(volume, current_idx)

            # 4. 综合成交量确认强度
            # 成交量放大越多，确认强度越高
            amplification_score = min(1.0, max(0.0, (volume_amplification - 0.8) / 1.2))  # 0.8-2.0倍映射到0-1
            trend_score = volume_trend

            confirmation_strength = amplification_score * 0.7 + trend_score * 0.3

            return float(np.clip(confirmation_strength, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_rsi_divergence_signal(self, close: np.ndarray, current_idx: int) -> float:
        """计算RSI背离信号强度"""
        try:
            if current_idx < 14:
                return 0.5

            # 简化的RSI计算（14周期）
            period = 14
            start_idx = max(0, current_idx - period - 5)
            price_slice = close[start_idx:current_idx+1]

            if len(price_slice) < period + 1:
                return 0.5

            # 计算价格变化
            price_changes = np.diff(price_slice)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            # 计算RSI
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # RSI信号：超卖区域(30以下)给出正信号，超买区域(70以上)给出负信号
            if rsi <= 30:
                return 0.8  # 强烈启动信号
            elif rsi <= 40:
                return 0.65  # 中等启动信号
            elif rsi >= 70:
                return 0.2  # 弱启动信号
            else:
                return 0.5  # 中性

        except Exception:
            return 0.5

    def _calculate_price_momentum_signal(self, close: np.ndarray, current_idx: int) -> float:
        """计算价格动量信号强度"""
        try:
            if current_idx < 10:
                return 0.5

            # 短期动量（5日）vs 中期动量（10日）
            current_price = close[current_idx]

            if current_idx >= 5:
                price_5d_ago = close[current_idx - 5]
                short_momentum = (current_price - price_5d_ago) / price_5d_ago
            else:
                short_momentum = 0.0

            if current_idx >= 10:
                price_10d_ago = close[current_idx - 10]
                medium_momentum = (current_price - price_10d_ago) / price_10d_ago
            else:
                medium_momentum = 0.0

            # 动量加速度：短期动量 > 中期动量表示加速上涨
            momentum_acceleration = short_momentum - medium_momentum

            # 将动量转换为0-1信号
            if momentum_acceleration > 0.02:  # 加速上涨超过2%
                return 0.8
            elif momentum_acceleration > 0.01:  # 加速上涨超过1%
                return 0.65
            elif momentum_acceleration > -0.01:  # 轻微加速或平稳
                return 0.5
            else:  # 减速
                return 0.3

        except Exception:
            return 0.5

    def _calculate_volume_energy_signal(self, volume: np.ndarray, current_idx: int) -> float:
        """计算成交量能量信号强度"""
        try:
            if current_idx < 10:
                return 0.5

            # 近期成交量能量
            recent_volume = np.mean(volume[max(0, current_idx-4):current_idx+1])
            historical_volume = np.mean(volume[max(0, current_idx-20):current_idx-5])

            if historical_volume == 0:
                return 0.5

            volume_ratio = recent_volume / historical_volume

            # 成交量放大程度转换为信号强度
            if volume_ratio > 2.0:  # 成交量放大2倍以上
                return 0.9
            elif volume_ratio > 1.5:  # 成交量放大1.5倍以上
                return 0.75
            elif volume_ratio > 1.2:  # 成交量放大1.2倍以上
                return 0.6
            elif volume_ratio > 0.8:  # 成交量正常
                return 0.5
            else:  # 成交量萎缩
                return 0.3

        except Exception:
            return 0.5

    def _calculate_short_term_trend_strength(self, close: np.ndarray, current_idx: int) -> float:
        """计算短期趋势强度"""
        try:
            if current_idx < 5:
                return 0.5

            # 计算最近5日的趋势强度
            recent_prices = close[max(0, current_idx-4):current_idx+1]

            if len(recent_prices) < 3:
                return 0.5

            # 计算趋势一致性（连续上涨/下跌的天数）
            price_changes = np.diff(recent_prices)
            positive_days = np.sum(price_changes > 0)
            total_days = len(price_changes)

            if total_days == 0:
                return 0.5

            # 上涨天数比例
            upward_ratio = positive_days / total_days

            # 趋势强度评分
            if upward_ratio >= 0.8:  # 80%以上时间上涨
                return 0.8
            elif upward_ratio >= 0.6:  # 60%以上时间上涨
                return 0.65
            elif upward_ratio >= 0.4:  # 40-60%时间上涨（震荡）
                return 0.5
            else:  # 主要下跌
                return 0.3

        except Exception:
            return 0.5

    def _calculate_volume_trend(self, volume: np.ndarray, current_idx: int) -> float:
        """计算成交量趋势"""
        try:
            if current_idx < 10:
                return 0.5

            # 计算成交量的趋势方向
            recent_volume = volume[max(0, current_idx-4):current_idx+1]

            if len(recent_volume) < 3:
                return 0.5

            # 简单的线性趋势检测
            volume_changes = np.diff(recent_volume)
            increasing_days = np.sum(volume_changes > 0)
            total_days = len(volume_changes)

            if total_days == 0:
                return 0.5

            increasing_ratio = increasing_days / total_days

            # 成交量递增趋势越强，信号越强
            return float(np.clip(increasing_ratio, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_obv_momentum_factor(self, close: np.ndarray, volume: np.ndarray, current_price: float) -> float:
        """
        计算OBV动量因子 - 基于专业量化交易最佳实践

        综合了5种专业OBV算法的最佳特性：
        1. 标准OBV累积计算
        2. OBV动量分析（短期vs长期）
        3. OBV背离检测
        4. 相对OBV标准化
        5. OBV趋势强度评估

        完全避免未来信息泄露，只使用t日及之前的数据

        Args:
            close: 收盘价数组
            volume: 成交量数组
            current_price: 当前价格

        Returns:
            float: OBV动量因子 (0.0-1.0)，0.5为中性
        """
        try:
            if len(close) < 30 or len(volume) < 30:
                return 0.5

            # 性能优化：避免重复转换，数据已在上层转换为float32

            # 1. 计算标准OBV（专业实现）
            obv_values = self._calculate_professional_obv(close, volume)

            if len(obv_values) < 20:
                return 0.5

            # 2. OBV动量分析（短期vs长期）
            momentum_score = self._calculate_obv_momentum_score(obv_values)

            # 3. OBV背离检测
            divergence_score = self._calculate_obv_divergence_score(close, obv_values)

            # 4. OBV趋势强度
            trend_strength = self._calculate_obv_trend_strength(obv_values)

            # 5. 相对OBV标准化
            relative_obv = self._calculate_relative_obv_score(obv_values)

            # 综合OBV动量因子（专业加权）
            obv_momentum = (
                momentum_score * 0.3 +      # 动量分析权重30%
                divergence_score * 0.25 +   # 背离检测权重25%
                trend_strength * 0.25 +     # 趋势强度权重25%
                relative_obv * 0.2          # 相对位置权重20%
            )

            return float(np.clip(obv_momentum, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_professional_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        计算专业OBV - 高效向量化实现

        OBV计算规则：
        - 如果今日收盘价 > 昨日收盘价：OBV = 前一日OBV + 今日成交量
        - 如果今日收盘价 < 昨日收盘价：OBV = 前一日OBV - 今日成交量
        - 如果今日收盘价 = 昨日收盘价：OBV = 前一日OBV
        """
        try:
            if len(close) < 2 or len(volume) < 2:
                return np.array([0.0], dtype=self._float_dtype)

            # 向量化计算价格变化
            price_changes = np.diff(close)

            # 向量化计算OBV变化
            # 价格上涨：+成交量，价格下跌：-成交量，价格不变：0
            obv_changes = np.where(
                price_changes > 0, volume[1:],  # 价格上涨
                np.where(price_changes < 0, -volume[1:], 0)  # 价格下跌或不变
            )

            # 累积计算OBV
            obv_values = np.cumsum(obv_changes)

            # 添加初始值0
            obv_with_initial = np.concatenate([[0.0], obv_values])

            return obv_with_initial.astype(self._float_dtype)

        except Exception:
            return np.array([0.0], dtype=self._float_dtype)

    def _calculate_obv_momentum_score(self, obv_values: np.ndarray) -> float:
        """计算OBV动量评分 - 短期vs长期动量比较"""
        try:
            if len(obv_values) < 20:
                return 0.5

            # 短期动量（5日）
            if len(obv_values) >= 6:
                short_momentum = (obv_values[-1] - obv_values[-6]) / abs(obv_values[-6]) if obv_values[-6] != 0 else 0
            else:
                short_momentum = 0

            # 长期动量（20日）
            if len(obv_values) >= 21:
                long_momentum = (obv_values[-1] - obv_values[-21]) / abs(obv_values[-21]) if obv_values[-21] != 0 else 0
            else:
                long_momentum = 0

            # 动量加速度：短期动量相对于长期动量的强度
            if abs(long_momentum) > 1e-6:
                momentum_ratio = short_momentum / long_momentum
                # 将比值转换为0-1评分
                momentum_score = 0.5 + np.tanh(momentum_ratio) * 0.5
            else:
                momentum_score = 0.5 + np.tanh(short_momentum * 10) * 0.5

            return float(np.clip(momentum_score, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_obv_divergence_score(self, close: np.ndarray, obv_values: np.ndarray) -> float:
        """
        计算OBV背离评分 - 专业背离检测算法

        检测价格与OBV之间的背离信号：
        - 正背离：价格创新低，OBV未创新低（看涨信号）
        - 负背离：价格创新高，OBV未创新高（看跌信号）
        """
        try:
            if len(close) < 20 or len(obv_values) < 20:
                return 0.5

            # 使用最近20日数据进行背离分析
            recent_close = close[-20:]
            recent_obv = obv_values[-20:]

            # 寻找价格和OBV的局部极值
            price_peaks, price_troughs = self._find_local_extremes(recent_close)
            obv_peaks, obv_troughs = self._find_local_extremes(recent_obv)

            divergence_strength = 0.0

            # 检测看涨背离（价格新低，OBV未新低）
            if len(price_troughs) >= 2 and len(obv_troughs) >= 2:
                latest_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]

                # 找到对应的OBV低点
                latest_obv_trough = obv_troughs[-1] if len(obv_troughs) > 0 else 0
                prev_obv_trough = obv_troughs[-2] if len(obv_troughs) > 1 else latest_obv_trough

                # 价格创新低但OBV未创新低 = 看涨背离
                if latest_price_trough < prev_price_trough and latest_obv_trough > prev_obv_trough:
                    divergence_strength += 0.4  # 看涨背离加分

            # 检测看跌背离（价格新高，OBV未新高）
            if len(price_peaks) >= 2 and len(obv_peaks) >= 2:
                latest_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]

                latest_obv_peak = obv_peaks[-1] if len(obv_peaks) > 0 else 0
                prev_obv_peak = obv_peaks[-2] if len(obv_peaks) > 1 else latest_obv_peak

                # 价格创新高但OBV未创新高 = 看跌背离
                if latest_price_peak > prev_price_peak and latest_obv_peak < prev_obv_peak:
                    divergence_strength -= 0.4  # 看跌背离减分

            # 转换为0-1评分（0.5为中性）
            divergence_score = 0.5 + divergence_strength

            return float(np.clip(divergence_score, 0.0, 1.0))

        except Exception:
            return 0.5

    def _find_local_extremes(self, data: np.ndarray, window: int = 3) -> tuple:
        """寻找局部极值点"""
        try:
            if len(data) < window * 2 + 1:
                return [], []

            peaks = []
            troughs = []

            for i in range(window, len(data) - window):
                # 检查是否为局部最高点
                if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
                   all(data[i] >= data[i+j] for j in range(1, window+1)):
                    peaks.append(data[i])

                # 检查是否为局部最低点
                if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
                   all(data[i] <= data[i+j] for j in range(1, window+1)):
                    troughs.append(data[i])

            return peaks, troughs

        except Exception:
            return [], []

    def _calculate_obv_trend_strength(self, obv_values: np.ndarray) -> float:
        """计算OBV趋势强度"""
        try:
            if len(obv_values) < 10:
                return 0.5

            # 使用线性回归计算趋势强度
            recent_obv = obv_values[-10:]  # 最近10日
            x = np.arange(len(recent_obv))

            # 计算线性回归斜率
            if len(recent_obv) > 1:
                slope = np.polyfit(x, recent_obv, 1)[0]

                # 标准化斜率为0-1评分
                # 正斜率表示上升趋势，负斜率表示下降趋势
                max_slope = np.std(recent_obv) * 2  # 使用标准差作为标准化基准
                if max_slope > 0:
                    normalized_slope = slope / max_slope
                    trend_score = 0.5 + np.tanh(normalized_slope) * 0.5
                else:
                    trend_score = 0.5
            else:
                trend_score = 0.5

            return float(np.clip(trend_score, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_relative_obv_score(self, obv_values: np.ndarray) -> float:
        """计算相对OBV评分 - 当前OBV在历史范围中的位置"""
        try:
            if len(obv_values) < 20:
                return 0.5

            # 使用最近20日的OBV范围
            recent_obv = obv_values[-20:]
            current_obv = obv_values[-1]

            obv_min = np.min(recent_obv)
            obv_max = np.max(recent_obv)

            # 计算当前OBV在历史范围中的相对位置
            if obv_max > obv_min:
                relative_position = (current_obv - obv_min) / (obv_max - obv_min)
            else:
                relative_position = 0.5

            return float(np.clip(relative_position, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_parkinson_volatility(self, high: np.ndarray, low: np.ndarray) -> float:
        """
        计算Parkinson波动率估计 - 移植自generate_daily_features.py

        基于高低价差的波动率估计，比传统收盘价波动率更准确
        公式：sqrt((1 / (4 * ln(2))) * (ln(high/low))^2)

        Args:
            high: 最高价数组
            low: 最低价数组

        Returns:
            float: Parkinson波动率估计值
        """
        try:
            if len(high) < 20 or len(low) < 20:
                return 0.02  # 默认2%波动率

            # 性能优化：避免重复转换，数据已在上层转换为float32

            # 避免除零错误，设置最小值
            high = np.maximum(high, 0.001)
            low = np.maximum(low, 0.001)

            # 使用最近20日数据计算波动率
            recent_high = high[-20:]
            recent_low = low[-20:]

            # Parkinson波动率公式
            log_hl_ratio = np.log(recent_high / recent_low)
            parkinson_vol = np.sqrt(
                (1 / (4 * np.log(2))) * np.mean(log_hl_ratio ** 2)
            )

            # 合理范围限制
            return float(np.clip(parkinson_vol, 0.005, 0.15))  # 0.5%-15%

        except Exception:
            return 0.02

    def _calculate_bollinger_factors(self, close: np.ndarray, current_price: float) -> Dict[str, float]:
        """
        计算布林带相关因子 - 基于专业量化交易算法

        包含三个核心因子：
        1. bollinger_squeeze_factor: 布林带挤压因子（低波动率突破机会）
        2. bollinger_breakout_strength: 布林带突破强度
        3. bollinger_position_factor: 价格在布林带中的相对位置

        Returns:
            Dict[str, float]: 布林带相关因子字典
        """
        try:
            if len(close) < 20:
                return {
                    'bollinger_squeeze_factor': 0.5,
                    'bollinger_breakout_strength': 0.0,
                    'bollinger_position_factor': 0.5
                }

            # 转换为float32提高性能
            close = close.astype(self._float_dtype, copy=False)

            # 计算20日布林带
            period = 20
            recent_close = close[-period:]

            # 移动平均线
            sma = np.mean(recent_close)

            # 标准差
            std = np.std(recent_close)

            # 布林带上下轨
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std

            # 1. 布林带挤压因子（专业算法）
            squeeze_factor = self._calculate_bollinger_squeeze(recent_close, std, sma)

            # 2. 布林带突破强度
            breakout_strength = self._calculate_bollinger_breakout_strength(
                current_price, upper_band, lower_band, sma, std
            )

            # 3. 布林带位置因子
            position_factor = self._calculate_bollinger_position(
                current_price, upper_band, lower_band, sma
            )

            return {
                'bollinger_squeeze_factor': squeeze_factor,
                'bollinger_breakout_strength': breakout_strength,
                'bollinger_position_factor': position_factor
            }

        except Exception:
            return {
                'bollinger_squeeze_factor': 0.5,
                'bollinger_breakout_strength': 0.0,
                'bollinger_position_factor': 0.5
            }

    def _calculate_bollinger_squeeze(self, close: np.ndarray, std: float, sma: float) -> float:
        """
        计算布林带挤压因子 - 专业算法

        挤压检测逻辑：
        1. 当前波动率相对于历史波动率的比值
        2. 布林带宽度相对于价格的比例
        3. 波动率压缩的持续时间

        Returns:
            float: 挤压强度 (0.0-1.0)，值越高表示挤压越强，突破概率越大
        """
        try:
            # 1. 当前布林带宽度相对于价格的比例
            band_width_ratio = (2 * std) / sma if sma > 0 else 0.02

            # 2. 历史波动率比较（使用更长周期）
            if len(close) >= 60:
                historical_std = np.std(close[-60:])  # 60日历史波动率
                volatility_compression = std / historical_std if historical_std > 0 else 1.0
            else:
                volatility_compression = 1.0

            # 3. 波动率压缩程度评分
            # 当前波动率越低于历史波动率，挤压越强
            compression_score = max(0.0, min(1.0, (1.0 - volatility_compression) * 2))

            # 4. 布林带宽度评分
            # 布林带越窄，挤压越强
            width_score = max(0.0, min(1.0, (0.04 - band_width_ratio) / 0.03))  # 4%以下认为较窄

            # 综合挤压因子
            squeeze_factor = compression_score * 0.6 + width_score * 0.4

            return float(np.clip(squeeze_factor, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_bollinger_breakout_strength(self, current_price: float, upper_band: float,
                                             lower_band: float, sma: float, std: float) -> float:
        """
        计算布林带突破强度 - 专业算法

        突破强度评估：
        1. 突破幅度（价格超出布林带的程度）
        2. 突破方向（向上突破为正，向下突破为负）
        3. 突破的相对强度

        Returns:
            float: 突破强度 (-1.0 to 1.0)，正值表示向上突破，负值表示向下突破
        """
        try:
            # 1. 检查是否突破
            if current_price > upper_band:
                # 向上突破
                breakout_distance = current_price - upper_band
                breakout_strength = breakout_distance / std if std > 0 else 0.0
                return float(min(1.0, breakout_strength))  # 正值，最大1.0

            elif current_price < lower_band:
                # 向下突破
                breakout_distance = lower_band - current_price
                breakout_strength = breakout_distance / std if std > 0 else 0.0
                return float(max(-1.0, -breakout_strength))  # 负值，最小-1.0

            else:
                # 未突破，计算接近程度
                if current_price > sma:
                    # 接近上轨
                    approach_ratio = (current_price - sma) / (upper_band - sma) if upper_band > sma else 0.0
                    return float(approach_ratio * 0.5)  # 0-0.5之间
                else:
                    # 接近下轨
                    approach_ratio = (sma - current_price) / (sma - lower_band) if sma > lower_band else 0.0
                    return float(-approach_ratio * 0.5)  # -0.5-0之间

        except Exception:
            return 0.0

    def _calculate_bollinger_position(self, current_price: float, upper_band: float,
                                    lower_band: float, sma: float) -> float:
        """
        计算布林带位置因子 - 专业算法

        位置计算：
        - 0.0: 价格在下轨或以下
        - 0.5: 价格在中轨（移动平均线）
        - 1.0: 价格在上轨或以上

        Returns:
            float: 位置因子 (0.0-1.0)
        """
        try:
            if upper_band <= lower_band:
                return 0.5  # 异常情况，返回中性值

            # 计算价格在布林带中的相对位置
            if current_price >= upper_band:
                return 1.0
            elif current_price <= lower_band:
                return 0.0
            else:
                # 线性插值计算位置
                position = (current_price - lower_band) / (upper_band - lower_band)
                return float(np.clip(position, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_dynamic_expected_return(self, close: np.ndarray, low: np.ndarray,
                                         support_price: float, formation_idx: int, current_idx: int) -> float:
        """
        专业的预期收益计算算法 - 基于多维度技术分析

        改进算法特点：
        1. 基于波动率的预期收益估计
        2. 技术指标组合信号强度
        3. 市场微观结构分析
        4. 自适应风险调整

        完全避免未来信息泄露，只使用t日及之前的数据
        """
        try:
            if current_idx < 20:
                return 0.05  # 默认5%

            # 性能优化：避免重复转换
            if close.dtype != self._float_dtype:
                close = close.astype(self._float_dtype, copy=False)
            if low.dtype != self._float_dtype:
                low = low.astype(self._float_dtype, copy=False)

            current_price = close[current_idx]

            # 1. 基于波动率的预期收益（专业算法）
            volatility_return = self._calculate_volatility_based_return(close, current_idx)

            # 2. 技术指标组合信号强度
            technical_signal_strength = self._calculate_technical_signal_strength(
                close, low, current_idx, support_price
            )

            # 3. 支撑位强度调整
            support_strength = self._calculate_support_strength_factor(
                close, low, support_price, formation_idx, current_idx
            )

            # 4. 市场环境自适应调整
            market_regime = self._calculate_market_regime_factor(close, current_idx)

            # 5. 综合预期收益计算（专业量化模型）
            base_return = volatility_return * 1.5  # 基础预期收益

            # 技术信号调整（0.7-1.3倍）
            signal_multiplier = 0.7 + technical_signal_strength * 0.6

            # 支撑位强度调整（0.8-1.2倍）
            support_multiplier = 0.8 + support_strength * 0.4

            # 市场环境调整（0.6-1.4倍）
            market_multiplier = 0.6 + market_regime * 0.8

            # 最终预期收益
            expected_return = base_return * signal_multiplier * support_multiplier * market_multiplier

            # 合理范围限制
            return float(np.clip(expected_return, 0.02, 0.20))  # 2%-20%

        except Exception:
            return 0.05

    def _calculate_volatility_based_return(self, close: np.ndarray, current_idx: int) -> float:
        """基于波动率的预期收益估计 - 专业算法"""
        try:
            if current_idx < 20:
                return 0.03

            # 计算20日历史波动率
            recent_close = close[max(0, current_idx-19):current_idx+1]
            returns = np.diff(recent_close) / recent_close[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

            # 基于波动率的预期收益：高波动率对应高预期收益
            volatility_return = volatility * 0.3  # 波动率的30%作为基础预期收益

            return float(np.clip(volatility_return, 0.01, 0.10))

        except Exception:
            return 0.03

    def _calculate_technical_signal_strength(self, close: np.ndarray, low: np.ndarray,
                                           current_idx: int, support_price: float) -> float:
        """技术指标组合信号强度 - 专业算法"""
        try:
            if current_idx < 14:
                return 0.5

            current_price = close[current_idx]

            # 1. 距离支撑位的相对位置（越接近支撑位，信号越强）
            distance_to_support = abs(current_price - support_price) / current_price
            distance_score = max(0.0, 1.0 - distance_to_support * 20)  # 5%以内得分较高

            # 2. 价格相对位置（相对于近期高低点）
            recent_high = np.max(close[max(0, current_idx-20):current_idx+1])
            recent_low = np.min(low[max(0, current_idx-20):current_idx+1])

            if recent_high > recent_low:
                price_position = (current_price - recent_low) / (recent_high - recent_low)
                # 价格在低位时信号更强
                position_score = 1.0 - price_position
            else:
                position_score = 0.5

            # 3. 短期趋势强度
            if current_idx >= 5:
                short_trend = (close[current_idx] - close[current_idx-5]) / close[current_idx-5]
                trend_score = max(0.0, min(1.0, short_trend * 10 + 0.5))  # 上涨趋势得分高
            else:
                trend_score = 0.5

            # 综合技术信号强度
            signal_strength = (distance_score * 0.4 + position_score * 0.3 + trend_score * 0.3)

            return float(np.clip(signal_strength, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_support_strength_factor(self, close: np.ndarray, low: np.ndarray,
                                         support_price: float, formation_idx: int, current_idx: int) -> float:
        """支撑位强度因子 - 专业算法"""
        try:
            if formation_idx >= current_idx - 5:
                return 0.5

            # 1. 支撑位测试次数（历史上触及支撑位的次数）
            historical_indices = np.arange(formation_idx, current_idx)
            historical_lows = low[historical_indices]

            # 计算触及支撑位的次数
            touch_mask = np.abs(historical_lows - support_price) / support_price <= 0.03
            touch_count = np.sum(touch_mask)

            # 测试次数越多，支撑位越强
            test_score = min(1.0, touch_count / 5.0)  # 5次以上测试得满分

            # 2. 支撑位形成时间（形成时间越长，支撑位越强）
            formation_time = current_idx - formation_idx
            time_score = min(1.0, formation_time / 30.0)  # 30日以上得满分

            # 3. 支撑位价格稳定性
            if touch_count > 1:
                touched_prices = historical_lows[touch_mask]
                price_stability = 1.0 - (np.std(touched_prices) / np.mean(touched_prices))
                stability_score = max(0.0, min(1.0, price_stability))
            else:
                stability_score = 0.5

            # 综合支撑位强度
            support_strength = (test_score * 0.4 + time_score * 0.3 + stability_score * 0.3)

            return float(np.clip(support_strength, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_market_regime_factor(self, close: np.ndarray, current_idx: int) -> float:
        """市场环境因子 - 专业算法"""
        try:
            if current_idx < 30:
                return 0.5

            # 1. 市场趋势强度（30日趋势）
            if current_idx >= 30:
                long_trend = (close[current_idx] - close[current_idx-30]) / close[current_idx-30]
                trend_score = max(0.0, min(1.0, long_trend * 5 + 0.5))  # 上涨市场得分高
            else:
                trend_score = 0.5

            # 2. 市场波动率环境
            recent_returns = np.diff(close[max(0, current_idx-20):current_idx+1])
            if len(recent_returns) > 0:
                recent_vol = np.std(recent_returns)
                # 适中波动率最佳，过高或过低都不利
                vol_score = 1.0 - abs(recent_vol - 0.02) * 25  # 2%左右波动率最佳
                vol_score = max(0.0, min(1.0, vol_score))
            else:
                vol_score = 0.5

            # 综合市场环境因子
            market_regime = (trend_score * 0.6 + vol_score * 0.4)

            return float(np.clip(market_regime, 0.0, 1.0))

        except Exception:
            return 0.5

    def _calculate_current_volatility(self, close: np.ndarray) -> float:
        """计算当前市场波动率 - 性能优化版本"""
        try:
            if len(close) < 20:
                return 0.02

            # 转换为float32以提高性能
            close = close.astype(self._float_dtype, copy=False)

            # 向量化计算最近20日的日收益率波动率
            recent_close = close[-20:]
            returns = np.diff(recent_close) / recent_close[:-1]
            volatility = np.std(returns)

            return float(np.clip(volatility, 0.005, 0.08))

        except Exception:
            return 0.02

    def _calculate_expected_loss_professional(self, current_price: float, support_price: float,
                                            close: np.ndarray, support_info: Dict) -> float:
        """专业的预期损失计算"""
        try:
            # 1. 基础风险：距离支撑位的距离
            distance_to_support = abs(current_price - support_price) / current_price

            # 2. 支撑位破位风险调整
            support_strength = support_info.get('strength', 0.5)
            breakout_risk_multiplier = 1.0 + (1.0 - support_strength) * 0.5

            # 3. 市场环境风险调整
            if len(close) >= 10:
                recent_trend = (close[-1] - close[-10]) / close[-10]
                if recent_trend < -0.05:  # 下跌趋势中风险更大
                    trend_risk_multiplier = 1.3
                else:
                    trend_risk_multiplier = 1.0
            else:
                trend_risk_multiplier = 1.0

            # 4. 综合预期损失
            base_loss = max(distance_to_support, 0.02)  # 最小2%的基础风险
            expected_loss = base_loss * breakout_risk_multiplier * trend_risk_multiplier

            return float(min(0.15, max(0.02, expected_loss)))

        except Exception:
            return 0.05


    def _calculate_institutional_activity_professional(self, volume: np.ndarray,
                                                     close: np.ndarray, current_price: float) -> float:
        """专业的机构活跃度计算 - 性能优化版本"""
        try:
            if len(volume) < 30:
                return 50.0

            # 转换为float32以提高性能
            volume = volume.astype(self._float_dtype, copy=False)
            close = close.astype(self._float_dtype, copy=False)

            # 1. 向量化计算大单成交量分析
            recent_volume = volume[-10:]
            hist_volume = volume[-30:-10]

            # 使用向量化操作计算均值
            recent_mean = np.mean(recent_volume)
            hist_mean = np.mean(hist_volume)
            volume_amplification = recent_mean / hist_mean if hist_mean > 0 else 1.0

            # 2. 向量化价格稳定性分析
            recent_close = close[-10:]
            close_mean = np.mean(recent_close)
            price_volatility = np.std(recent_close) / close_mean if close_mean > 0 else 0.1
            stability_factor = np.clip(1 - price_volatility * 15, 0, 1)

            # 3. 向量化成交量一致性计算
            volume_std = np.std(recent_volume)
            volume_consistency = 1 - (volume_std / recent_mean) if recent_mean > 0 else 0
            volume_consistency = np.clip(volume_consistency, 0, 1)

            # 4. 价格效率（机构推动的价格变化更有效率）
            if len(close) >= 5:
                price_change = (close[-1] - close[-5]) / close[-5]
                volume_support = np.mean(recent_volume[-5:]) / np.mean(hist_volume) if np.mean(hist_volume) > 0 else 1.0

                # 价格变化与成交量的匹配度
                efficiency_score = min(abs(price_change) * volume_support * 10, 1.0)
            else:
                efficiency_score = 0.5

            # 综合机构活跃度评分
            activity_score = (
                min(volume_amplification / 2, 1.0) * 0.3 +
                stability_factor * 0.25 +
                volume_consistency * 0.25 +
                efficiency_score * 0.2
            ) * 100

            return float(min(100.0, max(0.0, activity_score)))

        except Exception:
            return 50.0

    def _calculate_retail_institutional_ratio(self, close: np.ndarray, volume: np.ndarray, current_price: float) -> float:
        """计算散户vs机构比例"""
        try:
            if len(volume) < 20:
                return 0.5

            # 转换为float32以提高性能
            volume = volume.astype(self._float_dtype, copy=False)

            # 基于成交量分布特征判断资金性质
            recent_volume = volume[-10:]

            # 向量化计算成交量波动性
            volume_mean = np.mean(recent_volume)
            volume_volatility = np.std(recent_volume) / volume_mean if volume_mean > 0 else 1.0

            # 向量化计算成交量持续性
            volume_diffs = np.diff(recent_volume)
            if len(volume_diffs) > 1:
                # 计算趋势一致性
                trend_changes = np.diff(np.sign(volume_diffs))
                consistency_score = 1.0 - (np.count_nonzero(trend_changes) / len(trend_changes))
                volume_trend_consistency = np.clip(consistency_score, 0, 1)
            else:
                volume_trend_consistency = 0.5

            # 向量化散户vs机构比例估算
            retail_indicator = volume_volatility * (1 - volume_trend_consistency)
            institutional_indicator = (1 - volume_volatility) * volume_trend_consistency

            total_indicator = retail_indicator + institutional_indicator
            retail_ratio = retail_indicator / total_indicator if total_indicator > 0 else 0.5

            return float(np.clip(retail_ratio, 0.0, 1.0))

        except Exception:
            return 0.5

    def _create_default_result(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建默认结果DataFrame"""
        try:
            if data.empty:
                # 创建空的结果DataFrame
                result = pd.DataFrame()
                feature_names = self._get_feature_names()
                for feature in feature_names:
                    result[feature] = []
                return result

            # 为现有数据添加默认因子值
            result = data.copy()
            default_values = self._get_default_factor_values()

            for feature_name, default_value in default_values.items():
                if feature_name not in result.columns:
                    result[feature_name] = default_value

            return result

        except Exception as e:
            self.logger.error(f"❌ 创建默认结果失败: {e}")
            return pd.DataFrame()


