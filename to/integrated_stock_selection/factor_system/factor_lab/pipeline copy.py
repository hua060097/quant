"""因子流水线。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Tuple
import warnings

import pandas as pd
import numpy as np
import logging
from scipy import stats
from .base import FactorContext
from .registry import registry, FactorRegistry


def _validate_factor_quality(factor_series: pd.Series, factor_name: str, logger, working_df: pd.DataFrame = None):
    """
    因子质量检查函数

    检查因子是否存在以下质量问题：
    1. 常数列（所有值相同）
    2. 缺失数据过多
    3. 异常值过多

    Args:
        factor_series: 因子数据Series
        factor_name: 因子名称
        logger: 日志记录器
        working_df: 包含stock_code和date的DataFrame（用于定位极端值）

    Raises:
        ValueError: 当因子质量严重问题时抛出异常并终止程序
    """
    import sys

    # 🛡️ 防御性编程：确保factor_name是有效字符串
    if factor_name is None or not isinstance(factor_name, str):
        factor_name = 'unknown'

    try:
        # 基础统计信息
        total_count = len(factor_series)
        nan_count = factor_series.isna().sum()
        valid_count = total_count - nan_count
        valid_ratio = valid_count / total_count if total_count > 0 else 0

        logger.info(f"[质量检查] 因子 {factor_name}: 总数={total_count}, 有效={valid_count}, 缺失={nan_count}, 有效率={valid_ratio:.4f}")

        # 检查1: 缺失数据过多
        if valid_ratio < 0.5:  # 有效数据少于50%
            error_msg = f"❌ 因子 {factor_name} 缺失数据过多! 有效率仅{valid_ratio:.4f} (< 0.50)"
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif valid_ratio < 0.8:  # 有效数据少于80%，发出警告
            logger.warning(f"⚠️  因子 {factor_name} 缺失数据较多: 有效率{valid_ratio:.4f} (< 0.80)")

        # 如果没有有效数据，直接报错
        if valid_count == 0:
            error_msg = f"❌ 因子 {factor_name} 没有任何有效数据!"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 获取有效数据
        valid_data = factor_series.dropna()

        # 检查2: 常数列
        unique_count = valid_data.nunique()
        if unique_count == 1:
            # 完全常数列
            constant_value = valid_data.iloc[0]
            error_msg = f"❌ 因子 {factor_name} 是完全常数列! 所有值={constant_value}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif unique_count <= 5:
            # 几乎常数列，很少的不同值
            logger.warning(f"⚠️  因子 {factor_name} 不同值过少: 仅{unique_count}个不同值")
            logger.info(f"    这些值: {sorted(valid_data.unique())[:10]}{'...' if unique_count > 10 else ''}")

        # 检查3: 方差为0（另一种常数列检测）
        if valid_data.var() == 0:
            error_msg = f"❌ 因子 {factor_name} 方差为0，是常数列!"
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif valid_data.var() < 1e-10:
            logger.warning(f"⚠️  因子 {factor_name} 方差过小: {valid_data.var():.2e}")

        # 检查4: 极端值过多（超过99.9%分位数的值）
        q99_9 = valid_data.quantile(0.999)
        q0_1 = valid_data.quantile(0.001)
        extreme_high_count = (valid_data > q99_9).sum()
        extreme_low_count = (valid_data < q0_1).sum()
        extreme_ratio = (extreme_high_count + extreme_low_count) / valid_count

        if extreme_ratio > 0.01:  # 超过1%的数据是极端值
            logger.warning(f"⚠️  因子 {factor_name} 极端值过多: {extreme_ratio:.4f}")
            logger.info(f"    >99.9%分位数: {extreme_high_count}个值")
            logger.info(f"    <0.1%分位数: {extreme_low_count}个值")

        # 检查5: 日期顺序检查（确保数据按时间从早到晚正确排序）
        if working_df is not None and 'date' in working_df.columns:
            logger.info(f"[日期顺序检查] 开始检查因子 {factor_name} 的日期顺序...")

            try:
                # 提取所有日期并去重
                all_dates = working_df['date'].dropna().unique()

                # 转换为日期类型（如果还不是）
                if len(all_dates) > 0:
                    # 尝试排序日期
                    sorted_dates = sorted(all_dates)

                    # 检查原始日期序列是否已经是升序
                    is_sorted = all(all_dates[i] <= all_dates[i+1] for i in range(len(all_dates)-1))

                    if not is_sorted:
                        # 找到第一个逆序的位置
                        for i in range(len(all_dates)-1):
                            if all_dates[i] > all_dates[i+1]:
                                error_msg = (
                                    f"❌ 因子 {factor_name} 的日期顺序错误! "
                                    f"发现日期逆序: 位置{i}的日期{all_dates[i]} > 位置{i+1}的日期{all_dates[i+1]}\n"
                                    f"正确的日期顺序应该是从早到晚，例如: 20230601 → 20230602 → ... → 20251228\n"
                                    f"请检查数据源是否正确排序!"
                                )
                                logger.error(error_msg)
                                logger.error(f"    前10个日期: {all_dates[:10]}")
                                logger.error(f"    后10个日期: {all_dates[-10:]}")
                                logger.error(f"    日期范围: {all_dates.min()} ~ {all_dates.max()}")
                                raise ValueError(error_msg)

                    # 检查是否有重复的日期（这不一定是错误，但需要警告）
                    date_counts = working_df['date'].value_counts()
                    duplicate_dates = date_counts[date_counts > 1]
                    if len(duplicate_dates) > 0:
                        logger.warning(f"⚠️  因子 {factor_name} 存在重复日期: {len(duplicate_dates)}个日期出现多次")
                        logger.info(f"    重复最多的前5个日期: {duplicate_dates.head(5).to_dict()}")

                    # 输出日期范围信息
                    logger.info(f"[日期顺序检查] 因子 {factor_name} 日期顺序正确 ✓")
                    logger.info(f"    日期范围: {all_dates.min()} ~ {all_dates.max()}")
                    logger.info(f"    总交易日数: {len(all_dates)}")

            except Exception as e:
                if isinstance(e, ValueError):
                    # 重新抛出ValueError，终止程序
                    raise
                else:
                    # 其他异常记录但继续
                    logger.warning(f"⚠️  日期顺序检查时发生异常: {e}")

        # 输出基本统计信息用于参考
        logger.info(f"[质量检查] 因子 {factor_name} 通过基本质量检查")
        logger.info(f"    基本统计: 均值={valid_data.mean():.6f}, 标准差={valid_data.std():.6f}")
        logger.info(f"    范围: [{valid_data.min():.6f}, {valid_data.max():.6f}]")
        logger.info(f"    分位数: 25%={valid_data.quantile(0.25):.6f}, 50%={valid_data.median():.6f}, 75%={valid_data.quantile(0.75):.6f}")

        # 🆕 特殊检查：pct_chg_1d因子的极端值详情
        if factor_name == 'pct_chg_1d':
            logger.info(f"    [pct_chg_1d特殊检查] 开始分析极端值...")

            # 定义极端值阈值（超过1000%或低于-1000%）
            extreme_threshold_high = 1000.0
            extreme_threshold_low = -1000.0

            extreme_high_mask = valid_data > extreme_threshold_high
            extreme_low_mask = valid_data < extreme_threshold_low

            extreme_high_count = extreme_high_mask.sum()
            extreme_low_count = extreme_low_mask.sum()

            if extreme_high_count > 0 or extreme_low_count > 0:
                logger.warning(f"    [pct_chg_1d特殊检查] 发现极端值! 高值>{extreme_threshold_high}: {extreme_high_count}个, 低值<{extreme_threshold_low}: {extreme_low_count}个")

                # 尝试获取股票和日期信息
                try:
                    if working_df is not None and len(working_df) == len(valid_data):
                        logger.warning(f"    [pct_chg_1d特殊检查] 使用working_df定位极端值...")

                        # 获取极端值的数据
                        extreme_mask = extreme_high_mask | extreme_low_mask
                        extreme_data = valid_data[extreme_mask]

                        # 记录前10个最极端的值
                        logger.warning(f"    [pct_chg_1d特殊检查] 最极端的10个值:")
                        extreme_indices = extreme_data.index

                        for i, idx in enumerate(extreme_indices[:10]):
                            stock_code = working_df.iloc[idx]['stock_code'] if 'stock_code' in working_df.columns else 'unknown'
                            date = working_df.iloc[idx]['date'] if 'date' in working_df.columns else 'unknown'
                            value = extreme_data.iloc[i]
                            logger.warning(f"      {i+1}. 股票={stock_code}, 日期={date}, pct_chg_1d={value:.6f}")

                        # 特别检查1913.573332这个目标值
                        target_value = 1913.573332
                        target_mask = abs(valid_data - target_value) < 0.001
                        target_count = target_mask.sum()

                        if target_count > 0:
                            logger.warning(f"    [pct_chg_1d特殊检查] *** 找到目标异常值 {target_value} ***")
                            target_indices = valid_data[target_mask].index
                            for idx in target_indices:
                                stock_code = working_df.iloc[idx]['stock_code'] if 'stock_code' in working_df.columns else 'unknown'
                                date = working_df.iloc[idx]['date'] if 'date' in working_df.columns else 'unknown'
                                logger.warning(f"      *** 股票={stock_code}, 日期={date}, pct_chg_1d={target_value:.6f} ***")
                        else:
                            # 查找最接近的值
                            closest_diff = abs(valid_data - target_value).min()
                            if closest_diff < 100:  # 如果有比较接近的值
                                closest_idx = abs(valid_data - target_value).argmin()
                                closest_value = valid_data.iloc[closest_idx]

                                stock_code = working_df.iloc[closest_idx]['stock_code'] if 'stock_code' in working_df.columns else 'unknown'
                                date = working_df.iloc[closest_idx]['date'] if 'date' in working_df.columns else 'unknown'

                                logger.warning(f"    [pct_chg_1d特殊检查] 未找到精确目标值，但找到最接近的值:")
                                logger.warning(f"      股票={stock_code}, 日期={date}, pct_chg_1d={closest_value:.6f}, 差异={closest_diff:.6f}")
                    else:
                        logger.warning(f"    [pct_chg_1d特殊检查] 无法获取working_df信息来定位股票和日期")
                        logger.warning(f"    working_df状态: {working_df is not None}, 长度匹配: {working_df is not None and len(working_df) == len(valid_data) if working_df else 'N/A'}")

                except Exception as e:
                    logger.error(f"    [pct_chg_1d特殊检查] 分析极端值时出错: {e}")
                    import traceback
                    logger.error(f"    [pct_chg_1d特殊检查] 错误详情: {traceback.format_exc()}")
            else:
                logger.info(f"    [pct_chg_1d特殊检查] 未发现超过±{extreme_threshold_high}的极端值")

    except ValueError:
        # 重新抛出ValueError，让上层程序终止
        raise
    except Exception as e:
        # 其他异常也要记录，但可能不终止程序
        error_msg = f"❌ 因子 {factor_name} 质量检查过程中发生异常: {str(e)}"
        logger.error(error_msg)
        # 对于检查过程中的异常，也终止程序以确保数据质量
        raise ValueError(error_msg)



@dataclass(slots=True)
class PipelineResult:
    factors: List[str]
    data: pd.DataFrame
    clipping_report: Optional[dict] = None


class FactorPipeline:
    """基于因子注册中心的计算流水线。"""

    def __init__(self, registry_obj: FactorRegistry | None = None) -> None:
        self.registry = registry_obj or registry

    def available_factors(self) -> List[str]:
        return self.registry.names()

    def compute(
        self,
        factor_names: Iterable[str],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        *,
        market_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[pd.DataFrame] = None,
        context_extra: Optional[dict] = None,
        batch_callback: Optional[callable] = None,
        batch_size: int = 10,
        normalization_config: Optional[dict] = None,
        existing_factors: Optional[set] = None,
            ) -> PipelineResult:
        factor_list = list(dict.fromkeys(factor_names))
        if not factor_list:
            raise ValueError("未指定任何因子。")

        # 🆕 断点续算：过滤掉已存在的因子
        if existing_factors:
            original_count = len(factor_list)
            factor_list = [f for f in factor_list if f not in existing_factors]
            skipped_count = original_count - len(factor_list)
            if skipped_count > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🔄 断点续算：跳过 {skipped_count} 个已存在的因子，剩余 {len(factor_list)} 个因子需要计算")

            if not factor_list:
                logger.info("✅ 所有因子已存在，无需计算")
                # 返回空结果
                return PipelineResult(factors=[], data=pd.DataFrame(columns=['stock_code', 'date']), clipping_report={})

        working = data.copy()
        working['date'] = pd.to_datetime(working['date'])
        working = working.sort_values(['stock_code', 'date']).reset_index(drop=True)

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        target_mask = (working['date'] >= start_ts) & (working['date'] <= end_ts)

        extra = dict(context_extra or {})
        extra.setdefault('target_start_date', start_ts)
        extra.setdefault('target_end_date', end_ts)
        extra['target_mask'] = target_mask

        ctx = FactorContext(
            market_data=_prepare_optional_df(market_data),
            sector_data=_prepare_optional_df(sector_data),
            extra=extra,
        )

        computed = {}
        total_factors = len(factor_list)
        import logging
        logger = logging.getLogger(__name__)

        for idx, name in enumerate(factor_list, 1):
            logger.info(f"正在计算因子 [{idx}/{total_factors}]: {name}")
            definition = self.registry.get(name)
            series = definition.run(working, ctx)
            if len(series) != len(working):
                raise ValueError(
                    f"因子{name}返回的长度{len(series)}与输入数据{len(working)}不匹配。"
                )
            # 🛡️ 确保因子Series保持正确的name属性
            if hasattr(series, 'name') and series.name is not None:
                # 如果原Series有name，保持不变
                computed[name] = pd.Series(series, name=series.name).astype('float64')
            else:
                # 如果原Series没有name，使用因子名称
                computed[name] = pd.Series(series, name=name).astype('float64')

            logger.info(f"✓ 因子 {name} 计算完成")

            # 🆕 分批保存机制：每计算batch_size个因子就触发一次回调
            if batch_callback and idx % batch_size == 0:
                batch_factors = factor_list[idx - batch_size:idx]
                batch_result = working[['stock_code', 'date']].copy()
                for batch_name in batch_factors:
                    batch_result[batch_name] = computed[batch_name].values

                # 过滤到目标日期范围
                batch_filtered = batch_result.loc[target_mask].copy()
                batch_filtered.sort_values(['date', 'stock_code'], inplace=True)
                batch_filtered.reset_index(drop=True, inplace=True)

                # 🆕 因子质量检查：在target_mask过滤后检查目标日期范围内的数据质量
                for batch_name in batch_factors:
                    factor_series = batch_filtered[batch_name]
                    _validate_factor_quality(factor_series, batch_name, logger, batch_filtered)

                # 调用回调函数保存批次数据
                batch_callback(batch_filtered, batch_factors)
                logger.info(f"💾 已保存批次 [{idx - batch_size + 1}-{idx}] 共 {len(batch_factors)} 个因子")

        # 🆕 处理最后一批不足batch_size的因子
        if batch_callback:
            last_batch_start = (total_factors // batch_size) * batch_size
            if last_batch_start < total_factors:
                remaining_factors = factor_list[last_batch_start:]
                batch_result = working[['stock_code', 'date']].copy()
                for batch_name in remaining_factors:
                    batch_result[batch_name] = computed[batch_name].values

                batch_filtered = batch_result.loc[target_mask].copy()
                batch_filtered.sort_values(['date', 'stock_code'], inplace=True)
                batch_filtered.reset_index(drop=True, inplace=True)

                # 🆕 因子质量检查：在target_mask过滤后检查目标日期范围内的数据质量
                for batch_name in remaining_factors:
                    factor_series = batch_filtered[batch_name]
                    _validate_factor_quality(factor_series, batch_name, logger, batch_filtered)

                batch_callback(batch_filtered, remaining_factors)
                logger.info(f"💾 已保存最后批次 [{last_batch_start + 1}-{total_factors}] 共 {len(remaining_factors)} 个因子")

        # 🔧 优化：早期过滤到目标日期范围，避免对历史数据做不必要的后处理
        # 这一步在所有因子计算完成后、后处理（winsorize、归一化）之前执行
        # 原因：历史数据仅用于计算窗口（如MA60），不需要被归一化和保存
        logger.info(f"🔧 性能优化：早期过滤到目标日期范围")
        logger.info(f"   目标日期: {start_date} - {end_date}")
        logger.info(f"   原始数据量: {len(working)} 条记录")

        # 过滤到目标日期范围
        # 🔧 关键：立即 reset_index，保持整个流程索引连续
        # 同时需要为 sector_interaction.py 等因子保持原始索引的引用
        working_target = working[target_mask].reset_index(drop=True)
        computed_target = {}
        for name, series in computed.items():
            # 过滤并立即重置索引，保持索引连续
            filtered_series = series[target_mask].reset_index(drop=True)
            computed_target[name] = filtered_series

        logger.info(f"   过滤后数据量: {len(working_target)} 条记录")
        logger.info(f"   ✅ 节省后处理: {len(working) - len(working_target)} 条历史记录")
        logger.info(f"   ✅ 索引状态: 连续 RangeIndex(0, 1, ..., {len(working_target)-1})")

        # 初始化裁剪报告
        clipping_report = {}

        # 应用后处理流程：winsorize + 归一化
        # 注意：在两阶段保存架构中，归一化在阶段2统一处理，这里只在非批次保存模式下使用
        if normalization_config:
            logger.info("开始应用后处理流程：winsorize + 归一化")
            # 从normalization_config中获取智能裁剪设置
            enable_intelligent_clipping = normalization_config.get('enable_intelligent_clipping', True)
            # 🔧 优化：只对目标日期范围的数据进行后处理
            computed_target, clipping_report = self._apply_postprocessing(computed_target, working_target, normalization_config, logger, enable_intelligent_clipping)
            # 更新因子列表（包含归一化后的因子）
            factor_list = list(computed_target.keys())
            logger.info(f"后处理完成，最终因子数量: {len(factor_list)}")

        # 🔧 优化：使用已过滤的数据
        result = working_target[['stock_code', 'date']].copy()
        for name in factor_list:
            result[name] = computed_target[name].values

        # 🔧 优化：不再需要再次过滤（已经提前过滤了）
        filtered = result.copy()
        filtered.sort_values(['date', 'stock_code'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)

        # 🆕 非分批保存模式下的因子质量检查：在target_mask过滤后检查目标日期范围内的数据质量
        if not batch_callback:  # 只有在非分批保存模式下才需要这里的质量检查
            logger.info("开始进行非分批保存模式下的因子质量检查...")
            for name in factor_list:
                factor_series = filtered[name]
                _validate_factor_quality(factor_series, name, logger, filtered)
            logger.info("非分批保存模式下的因子质量检查完成")

        return PipelineResult(factors=factor_list, data=filtered, clipping_report=clipping_report)

    def _apply_postprocessing(
        self,
        computed: dict,
        data: pd.DataFrame,
        config: dict,
        logger,
        enable_intelligent_clipping: bool = True
    ) -> tuple[dict, dict]:
        """
        应用后处理流程：针对不同类型因子采用不同处理策略

        处理策略：
        - original_factors: Inf处理 + 缺失值填充 (跳过winsorize)
        - 其他因子: 完整处理 (Inf → winsorize → 填充缺失值 → 归一化)

        Args:
            computed: 原始因子字典 {factor_name: pd.Series}
            data: 包含 date 列的数据框
            config: 归一化配置
                - zscore_factors: 需要 z-score 归一化的因子列表
                - rankpct_factors: 需要 rankpct 归一化的因子列表
                - original_factors: 原始因子列表（仅Inf处理+缺失值填充）
            logger: 日志记录器

        Returns:
            处理后的因子字典（包含原始因子 + 归一化因子）
        """
        # 🆕 添加数据结构检查日志
        logger.info(f"[POSTPROCESSING] 开始后处理流程")
        logger.info(f"[POSTPROCESSING] 输入数据形状: data={data.shape}")
        logger.info(f"[POSTPROCESSING] data列: {list(data.columns)}")
        logger.info(f"[POSTPROCESSING] 输入因子数量: {len(computed)}")
        logger.info(f"[POSTPROCESSING] 输入因子列表: {list(computed.keys())[:10]}...")

        # 检查stock_code列是否存在（行业内处理需要）
        if 'stock_code' in data.columns:
            logger.info(f"[POSTPROCESSING] stock_code列存在: {len(data['stock_code'].unique())}个唯一值")
        else:
            logger.error(f"[POSTPROCESSING] 错误: data中缺少stock_code列，行业内处理将失败")

        # 检查SW行业映射
        sw_sector_mapping = config.get('sw_sector_mapping', {})
        logger.info(f"[POSTPROCESSING] SW行业映射状态: {len(sw_sector_mapping)}个股票映射")

        # 获取因子类型配置
        original_factors = set(config.get('original_factors', []))
        intra_sector_rank_factors = set(config.get('intra_sector_rank_factors', []))
        intra_sector_zscore_factors = set(config.get('intra_sector_zscore_factors', []))

        # 分别处理不同类型的因子
        processed = {}

        # 处理原始因子（跳过winsorize）
        original_computed = {name: series for name, series in computed.items() if name in original_factors}
        regular_computed = {name: series for name, series in computed.items() if name not in original_factors}

        if original_computed:
            logger.info(f"步骤1: 处理 {len(original_computed)} 个原始因子（直接使用原始值，不做额外处理）")
            for factor_name, series in original_computed.items():
                processed[factor_name] = self._process_original_factor(factor_name, series, data, logger)

        if regular_computed:
            logger.info(f"步骤2: 处理 {len(regular_computed)} 个常规因子（完整处理流程）")
            # 对常规因子进行完整的4步处理
            # 步骤1: 替换 Inf 为 NaN
            logger.info("  步骤2.1: 替换 Inf 为 NaN")
            cleaned = self._replace_inf_with_nan(regular_computed, logger)

            # # 步骤2: 传统每日 cross-sectional winsorize
            logger.info("  步骤2.2: 应用每日 cross-sectional winsorize")
            winsorized = self._daily_winsorize(cleaned, data, logger)
            clipping_report = {}

            # # 步骤3: 填充缺失值（每日截面中位数）
            # logger.info("  步骤2.3: 填充缺失值（每日截面中位数）")
            filled = self._fill_missing_with_daily_median(winsorized, data, logger)

            # 合并到processed字典
            processed.update(filled)

        # 步骤3: 归一化处理
        logger.info("步骤3: 应用归一化处理")
        normalized = self._normalize_factors(processed, data, config, logger, enable_intelligent_clipping)

        # 返回归一化结果和裁剪报告
        return normalized, clipping_report if 'clipping_report' in locals() else {}

    def _process_original_factor(self, factor_name: str, series: pd.Series, data: pd.DataFrame, logger) -> pd.Series:
        """
        原始因子处理：最小化处理（仅Inf处理）

        适用于：
        - 已经是rank类型的因子（不会受益于winsorize，已有完善的缺失值处理）
        - 需要保持原始数值尺度的因子
        - 因子内部已经有完善的Inf和缺失值处理逻辑

        改进：
        - 至少处理Inf值，防止模型训练失败
        - 对于相关系数类因子，填充为0（中性值）

        Args:
            factor_name: 因子名称
            series: 因子值Series
            data: 包含date列的数据框（保留参数以保持接口一致性）
            logger: 日志记录器

        Returns:
            原始因子Series（仅处理Inf，保持其他不变）
        """
        # 🔧 改进：至少处理Inf值，防止模型训练失败
        processed = series.replace([np.inf, -np.inf], np.nan)

        # 🔧 对于相关系数类因子，填充为0（中性值）
        # 包括：beta, corr, correlation等关键词
        correlation_keywords = ['beta', 'corr', 'correlation', 'coskewness']
        if any(keyword in factor_name for keyword in correlation_keywords):
            processed = processed.fillna(0.0)

        # 其他情况保持原样（因子内部已有缺失值处理）
        # 理由：
        # 1. rank因子内部已有完善的fillna(0.5)处理
        # 2. 这些因子基本不会产生Inf值（排名结果范围固定）
        # 3. 额外处理可能扭曲因子原有的含义和数值分布

        return processed

    def _fill_missing_with_daily_median_single_factor(self, series: pd.Series, data: pd.DataFrame, logger) -> pd.Series:
        """
        为单个因子填充缺失值（每日截面中位数）

        Args:
            series: 因子值Series，索引与data一致
            data: 包含date列的数据框
            logger: 日志记录器

        Returns:
            填充后的因子Series
        """
        result = series.copy()

        # 确保数据索引一致
        if not result.index.equals(data.index):
            result = result.reindex(data.index)

        # 按日期分组填充
        data_with_factor = data[['date']].copy()
        data_with_factor['factor'] = result.values

        # 计算每日截面中位数
        daily_medians = data_with_factor.groupby('date')['factor'].median()

        # 填充缺失值
        for date, median in daily_medians.items():
            mask = (data_with_factor['date'] == date) & data_with_factor['factor'].isna()
            if mask.any() and not pd.isna(median):
                result.loc[mask] = median

        return result

    def _replace_inf_with_nan(self, computed: dict, logger) -> dict:
        """
        替换 Inf 为 NaN

        Args:
            computed: 因子字典
            logger: 日志记录器

        Returns:
            替换后的因子字典
        """
        result = {}
        total_inf_count = 0

        for factor_name, series in computed.items():
            # 统计 Inf 数量
            inf_mask = np.isinf(series)
            inf_count = inf_mask.sum()
            total_inf_count += inf_count

            # 替换 Inf 为 NaN
            cleaned_series = series.replace([np.inf, -np.inf], np.nan)
            result[factor_name] = cleaned_series

        logger.info(f"  替换了 {total_inf_count} 个 Inf 值为 NaN")
        return result

    def _fill_missing_with_daily_median(
        self,
        computed: dict,
        data: pd.DataFrame,
        logger
    ) -> dict:
        """
        使用每日截面中位数填充缺失值

        Args:
            computed: 因子字典
            data: 包含 date 列的数据框
            logger: 日志记录器

        Returns:
            填充后的因子字典
        """
        result = {}
        total_filled_count = 0

        for factor_name, series in computed.items():
            df = pd.DataFrame({
                'value': series.values,
                'date': data['date'].values
            })

            # 统计缺失值数量
            missing_count = df['value'].isna().sum()

            # 按日期分组，使用每日中位数填充
            df['filled'] = df.groupby('date')['value'].transform(
                lambda x: x.fillna(x.median())
            )

            # 如果某天所有值都是 NaN，则填充为 0
            df['filled'] = df['filled'].fillna(0)

           # 🛡️ 确保Series保持正确的name属性
            result[factor_name] = pd.Series(df['filled'].values, index=series.index, name=series.name)
            total_filled_count += missing_count

        logger.info(f"  填充了 {total_filled_count} 个缺失值（使用每日截面中位数）")
        return result

    # ==================== 改进版智能裁剪算法 v2.0 辅助函数 ====================

    def _calculate_basic_stats(self, data: pd.Series) -> dict:
        """计算基础统计信息"""
        return {
            'count': len(data),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'median': float(data.median())
        }

    def _analyze_distribution_shape(self, data: pd.Series, basic_stats: dict) -> dict:
        """分析分布形状（偏态、峰度等）"""
        import scipy.stats as stats

        # 计算偏态和峰度，处理小样本情况
        if len(data) >= 8:
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
        else:
            # 小样本时使用更稳健的估计
            skewness = 0.0
            kurtosis = 0.0

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_right_skewed': skewness > 0,
            'is_left_skewed': skewness < 0,
            'skewness_severity': abs(skewness),
            'is_heavy_tailed': kurtosis > 3,
            'kurtosis_severity': max(0, kurtosis - 3)
        }

    def _analyze_extreme_values(self, data: pd.Series, shape_analysis: dict) -> dict:
        """分析极端值情况"""
        basic_stats = self._calculate_basic_stats(data)

        # 使用IQR方法检测极端值
        iqr = basic_stats['q75'] - basic_stats['q25']
        lower_bound = basic_stats['q25'] - 1.5 * iqr
        upper_bound = basic_stats['q75'] + 1.5 * iqr

        extreme_mask = (data < lower_bound) | (data > upper_bound)
        extreme_count = extreme_mask.sum()
        extreme_ratio = extreme_count / len(data)

        # 更严格的极端值检测（3倍IQR）
        strict_lower = basic_stats['q25'] - 3 * iqr
        strict_upper = basic_stats['q75'] + 3 * iqr
        strict_extreme_mask = (data < strict_lower) | (data > strict_upper)
        strict_extreme_count = strict_extreme_mask.sum()
        strict_extreme_ratio = strict_extreme_count / len(data)

        return {
            'extreme_count': int(extreme_count),
            'extreme_ratio': float(extreme_ratio),
            'strict_extreme_count': int(strict_extreme_count),
            'strict_extreme_ratio': float(strict_extreme_ratio),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'has_extreme_values': extreme_ratio > 0,
            'has_severe_extremes': strict_extreme_ratio > 0.01
        }

    def _detect_boundary_behavior(self, data: pd.Series) -> dict:
        """检测边界聚集行为"""
        # 检查是否有大量值聚集在边界
        q01 = data.quantile(0.01)
        q99 = data.quantile(0.99)

        lower_boundary_count = (data <= q01).sum()
        upper_boundary_count = (data >= q99).sum()

        lower_boundary_ratio = lower_boundary_count / len(data)
        upper_boundary_ratio = upper_boundary_count / len(data)

        # 检查是否自然的理论边界（如0下界）
        natural_lower_bound = None
        if data.min() >= 0 and lower_boundary_ratio > 0.02:
            natural_lower_bound = 0

        has_boundary_cluster = (lower_boundary_ratio > 0.02) or (upper_boundary_ratio > 0.02)

        if has_boundary_cluster:
            boundary_type = 'both' if (lower_boundary_ratio > 0.02 and upper_boundary_ratio > 0.02) else \
                          'lower' if lower_boundary_ratio > 0.02 else 'upper'
        else:
            boundary_type = 'none'

        return {
            'lower_boundary_ratio': float(lower_boundary_ratio),
            'upper_boundary_ratio': float(upper_boundary_ratio),
            'has_boundary_cluster': has_boundary_cluster,
            'boundary_type': boundary_type,
            'natural_lower_bound': natural_lower_bound,
            'q01': float(q01),
            'q99': float(q99)
        }

    def _identify_factor_type(self, factor_name: str) -> dict:
        """识别因子类型和特性"""
        import re

        # 🛡️ 防御性编程：确保factor_name不为None
        if factor_name is None or factor_name == '':
            factor_name = 'unknown'

        factor_patterns = {
            'correlation': {
                'pattern': r'corr.*|correlation',
                'natural_skewness': 'right',
                'boundary_behavior': 'none',
                'sensitivity': 'high',
                'preferred_method': 'rankpct'
            },
            'convergence': {
                'pattern': r'convergence|ma_.*_score|distance',
                'natural_skewness': 'left',
                'boundary_behavior': 'lower_bound',
                'sensitivity': 'high',
                'preferred_method': 'rankpct'
            },
            'momentum': {
                'pattern': r'momentum|return_.*|pct_chg',
                'natural_skewness': 'slight_right',
                'boundary_behavior': 'none',
                'sensitivity': 'medium',
                'preferred_method': 'zscore'
            },
            'volatility': {
                'pattern': r'volatility|std|var',
                'natural_skewness': 'right',
                'boundary_behavior': 'lower_bound',
                'sensitivity': 'medium',
                'preferred_method': 'rankpct'
            },
            'ratio': {
                'pattern': r'ratio|_ma_.*|_to_.*',
                'natural_skewness': 'right',
                'boundary_behavior': 'lower_bound',
                'sensitivity': 'medium',
                'preferred_method': 'rankpct'
            },
            'rank': {
                'pattern': r'rank_.*',
                'natural_skewness': 'none',
                'boundary_behavior': 'none',
                'sensitivity': 'low',
                'preferred_method': 'rankpct'
            }
        }

        for factor_type, config in factor_patterns.items():
            if re.search(config['pattern'], factor_name, re.IGNORECASE):
                return {
                    'type': factor_type,
                    **config
                }

        return {
            'type': 'unknown',
            'pattern': r'.*',
            'natural_skewness': 'none',
            'boundary_behavior': 'none',
            'sensitivity': 'medium',
            'preferred_method': 'zscore'
        }

    def _get_adaptive_thresholds(self, shape_analysis: dict, extreme_analysis: dict,
                                boundary_analysis: dict, factor_type: dict, sample_size: int) -> dict:
        """获取自适应裁剪阈值"""
        # 基础阈值
        base_thresholds = {
            'clip_upper': 3.0,
            'clip_lower': -3.0,
            'extreme_ratio_tolerance': 0.03,
            'skewness_tolerance': 1.5,
            'kurtosis_tolerance': 3.0
        }

        # 样本大小调整
        if sample_size >= 100:
            sample_multiplier = 1.0  # 大样本：标准阈值
        elif sample_size >= 30:
            sample_multiplier = 0.9   # 中样本：稍微收紧
        else:
            sample_multiplier = 0.8   # 小样本：更收紧

        # 分布形状调整
        shape_adjustments = {
            'clip_upper': 1.0,
            'clip_lower': 1.0
        }

        # 偏态调整（更温和的策略）
        skewness = abs(shape_analysis['skewness'])
        if skewness > 2.0:  # 严重偏态
            if shape_analysis['is_right_skewed']:
                shape_adjustments['clip_upper'] *= 0.9  # 右偏时上阈值收紧
            else:
                shape_adjustments['clip_lower'] *= 0.9  # 左偏时下阈值收紧
        elif skewness > 1.0:  # 中等偏态
            if shape_analysis['is_right_skewed']:
                shape_adjustments['clip_upper'] *= 0.95
            else:
                shape_adjustments['clip_lower'] *= 0.95

        # 峰度调整
        kurtosis_excess = shape_analysis['kurtosis_severity']
        if kurtosis_excess > 2.0:  # 严重厚尾
            shape_adjustments['clip_upper'] *= 0.9
            shape_adjustments['clip_lower'] *= 0.9
        elif kurtosis_excess > 1.0:  # 中等厚尾
            shape_adjustments['clip_upper'] *= 0.95
            shape_adjustments['clip_lower'] *= 0.95

        # 极端值调整
        if extreme_analysis['has_severe_extremes']:
            shape_adjustments['clip_upper'] *= 0.9
            shape_adjustments['clip_lower'] *= 0.9

        # 边界聚集调整
        boundary_adjustments = {
            'clip_upper': 1.0,
            'clip_lower': 1.0
        }

        if boundary_analysis['has_boundary_cluster']:
            if boundary_analysis['boundary_type'] == 'lower':
                # 下边界聚集：下阈值应该更宽松
                boundary_adjustments['clip_lower'] *= 1.2
            elif boundary_analysis['boundary_type'] == 'upper':
                # 上边界聚集：上阈值应该更宽松
                boundary_adjustments['clip_upper'] *= 1.2
            elif boundary_analysis['boundary_type'] == 'both':
                # 双边界聚集：两侧都放宽
                boundary_adjustments['clip_upper'] *= 1.15
                boundary_adjustments['clip_lower'] *= 1.15

        # 因子类型调整
        type_adjustments = {
            'clip_upper': 1.0,
            'clip_lower': 1.0
        }

        if factor_type['sensitivity'] == 'high':
            # 高敏感因子：更宽松的阈值
            type_adjustments['clip_upper'] *= 1.1
            type_adjustments['clip_lower'] *= 1.1

        # 综合调整
        final_upper = (base_thresholds['clip_upper'] *
                      sample_multiplier *
                      shape_adjustments['clip_upper'] *
                      boundary_adjustments['clip_upper'] *
                      type_adjustments['clip_upper'])

        final_lower = (base_thresholds['clip_lower'] *
                      sample_multiplier *
                      shape_adjustments['clip_lower'] *
                      boundary_adjustments['clip_lower'] *
                      type_adjustments['clip_lower'])

        # 限制在合理范围内
        final_upper = max(2.0, min(4.0, final_upper))
        final_lower = min(-2.0, max(-4.0, final_lower))

        return {
            'clip_upper': final_upper,
            'clip_lower': final_lower,
            'sample_multiplier': sample_multiplier,
            'shape_adjustments': shape_adjustments,
            'boundary_adjustments': boundary_adjustments,
            'type_adjustments': type_adjustments
        }

    def _apply_graduated_clipping(self, data: pd.Series, thresholds: dict) -> pd.Series:
        """应用渐进式裁剪"""
        upper_threshold = thresholds['clip_upper']
        lower_threshold = thresholds['clip_lower']

        # 渐进式裁剪：使用软截断
        clipped_data = data.copy()

        # 对超出阈值的值进行软处理
        upper_mask = data > upper_threshold
        lower_mask = data < lower_threshold

        if upper_mask.any():
            # 对于超出上阈值的值，使用渐进式衰减而不是硬截断
            excess = data[upper_mask] - upper_threshold
            decay_factor = 0.8  # 衰减因子
            clipped_data[upper_mask] = upper_threshold + excess * decay_factor

        if lower_mask.any():
            # 对于低于下阈值的值，使用渐进式衰减
            excess = lower_threshold - data[lower_mask]
            decay_factor = 0.8
            clipped_data[lower_mask] = lower_threshold - excess * decay_factor

        # 最终硬截断，确保在合理范围内
        return clipped_data.clip(lower_threshold, upper_threshold)

    def _normalize_factors(
        self,
        computed: dict,
        data: pd.DataFrame,
        config: dict,
        logger,
        enable_intelligent_clipping: bool = True
    ) -> dict:
        """
        归一化处理

        Args:
            computed: 因子字典（已经过后处理）
            data: 包含 date 列的数据框
            config: 归一化配置
                - zscore_factors: 需要 z-score 归一化的因子列表
                - rankpct_factors: 需要 rankpct 归一化的因子列表
                - original_factors: 原始因子列表（不进行归一化）
            logger: 日志记录器

        Returns:
            归一化后的因子字典
        """
        normalized = {}

        zscore_factors = set(config.get('zscore_factors', []))
        rankpct_factors = set(config.get('rankpct_factors', []))
        original_factors = set(config.get('original_factors', []))
        intra_sector_rank_factors = set(config.get('intra_sector_rank_factors', []))
        intra_sector_zscore_factors = set(config.get('intra_sector_zscore_factors', []))

        zscore_count = 0
        rankpct_count = 0
        intra_sector_rank_count = 0
        intra_sector_zscore_count = 0

        # 🆕 添加详细的行业内处理debug日志（改为info级别确保能看见）
        logger.info(f"[NORMALIZE] 开始归一化处理流程")
        logger.info(f"[NORMALIZE] 输入因子数量: {len(computed)}")
        logger.info(f"[NORMALIZE] Z-score因子: {len(zscore_factors)}个")
        logger.info(f"[NORMALIZE] Rankpct因子: {len(rankpct_factors)}个")
        logger.info(f"[NORMALIZE] Original因子: {len(original_factors)}个")
        logger.info(f"[NORMALIZE] Intra-sector rank因子: {len(intra_sector_rank_factors)}个")
        logger.info(f"[NORMALIZE] Intra-sector z-score因子: {len(intra_sector_zscore_factors)}个")

        # 获取SW行业映射（用于行业内处理）
        sw_sector_mapping = config.get('sw_sector_mapping', {})

        logger.info(f"[NORMALIZE] SW行业映射状态: {len(sw_sector_mapping) if sw_sector_mapping else 0}个股票映射")

        if not sw_sector_mapping:
            logger.warning(f"[NORMALIZE] ⚠️ SW行业映射为空，所有intra-sector因子将回退到常规归一化")
        else:
            logger.info(f"[NORMALIZE] ✅ SW行业映射加载成功")
            logger.info(f"[NORMALIZE] SW行业映射示例: {dict(list(sw_sector_mapping.items())[:3])}")

        # 🆕 显示配置的intra sector因子列表
        if intra_sector_rank_factors:
            logger.info(f"[NORMALIZE] 配置的intra-sector rank因子: {list(intra_sector_rank_factors)}")
        if intra_sector_zscore_factors:
            logger.info(f"[NORMALIZE] 配置的intra-sector z-score因子: {list(intra_sector_zscore_factors)}")

        for factor_name, series in computed.items():
            # 保留原始因子（已经过相应的后处理）
            normalized[factor_name] = series

            # 跳过original_factors的归一化
            if factor_name in original_factors:
                  continue

            # 判断是否需要归一化
            need_zscore = factor_name in zscore_factors
            need_rankpct = factor_name in rankpct_factors
            need_intra_sector_rank = factor_name in intra_sector_rank_factors
            need_intra_sector_zscore = factor_name in intra_sector_zscore_factors

            # 🆕 添加因子判断的详细日志
            processing_info = []
            if need_zscore:
                processing_info.append("Z-score")
            if need_rankpct:
                processing_info.append("Rank")
            if need_intra_sector_rank:
                processing_info.append("Intra-Rank")
            if need_intra_sector_zscore:
                processing_info.append("Intra-Z")

            if processing_info:
                logger.info(f"[FACTOR] {factor_name} -> {', '.join(processing_info)}")

            # z-score 归一化
            if need_zscore:
                try:
                    logger.info(f"[DEBUG-ZSCORE] 开始 z-score 归一化: {factor_name}")
                    logger.info(f"[DEBUG-ZSCORE]   series 类型: {type(series)}, 长度: {len(series)}")
                    logger.info(f"[DEBUG-ZSCORE]   data 类型: {type(data)}, shape: {data.shape}")
                    logger.info(f"[DEBUG-ZSCORE]   前5个值: {series.head().tolist()}")

                    zscore_name = f"{factor_name}_z"
                    normalized[zscore_name] = self._zscore_normalize_with_clipping(series, data, enable_intelligent_clipping)
                    logger.info(f"  [OK] 生成 z-score 归一化因子: {zscore_name}")
                    zscore_count += 1
                except Exception as e:
                    logger.error(f"[ERROR] z-score 归一化失败: {factor_name}")
                    logger.error(f"[ERROR] 错误类型: {type(e).__name__}")
                    logger.error(f"[ERROR] 错误信息: {str(e)}")
                    import traceback
                    logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                    raise

            # rankpct 归一化
            if need_rankpct:
                rankpct_name = f"{factor_name}_rank"
                normalized[rankpct_name] = self._rankpct_normalize(series, data)
                logger.info(f"  [OK] 生成 rankpct 归一化因子: {rankpct_name}")
                rankpct_count += 1

            # 行业内排名归一化
            if need_intra_sector_rank:
                intra_rank_name = f"{factor_name}_intra_rank"
                logger.info(f"  [ATTEMPT] 开始行业内排名归一化: {intra_rank_name}")
                if sw_sector_mapping:
                    try:
                        normalized[intra_rank_name] = self._intra_sector_rank_normalize(series, data, sw_sector_mapping)
                        logger.info(f"  [OK] 生成行业内排名归一化因子: {intra_rank_name}")
                        intra_sector_rank_count += 1
                    except Exception as e:
                        logger.error(f"  [FAILED] 行业内排名归一化失败: {factor_name}, 错误: {e}")
                        import traceback
                        logger.error(f"  [TRACEBACK] {traceback.format_exc()}")
                        # 回退到常规排名归一化
                        normalized[intra_rank_name] = self._rankpct_normalize(series, data)
                        logger.info(f"  [FALLBACK] 使用常规排名归一化: {intra_rank_name}")
                        rankpct_count += 1
                else:
                    logger.warning(f"  [SKIP] 行业内排名归一化跳过: {factor_name} (缺少SW行业映射)")
                    # 回退到常规排名归一化
                    normalized[intra_rank_name] = self._rankpct_normalize(series, data)
                    logger.info(f"  [FALLBACK] 使用常规排名归一化: {intra_rank_name}")
                    rankpct_count += 1

            # 行业内z-score归一化
            if need_intra_sector_zscore:
                intra_zscore_name = f"{factor_name}_intra_z"
                logger.info(f"  [ATTEMPT] 开始行业内z-score归一化: {intra_zscore_name}")
                if sw_sector_mapping:
                    try:
                        normalized[intra_zscore_name] = self._intra_sector_zscore_normalize_with_clipping(series, data, sw_sector_mapping, enable_intelligent_clipping)
                        logger.info(f"  [OK] 生成行业内z-score归一化因子: {intra_zscore_name}")
                        intra_sector_zscore_count += 1
                    except Exception as e:
                        logger.error(f"  [FAILED] 行业内z-score归一化失败: {factor_name}, 错误: {e}")
                        import traceback
                        logger.error(f"  [TRACEBACK] {traceback.format_exc()}")
                        # 回退到常规z-score归一化
                        normalized[intra_zscore_name] = self._zscore_normalize_with_clipping(series, data, enable_intelligent_clipping)
                        logger.info(f"  [FALLBACK] 使用常规z-score归一化: {intra_zscore_name}")
                        zscore_count += 1
                else:
                    logger.warning(f"  [SKIP] 行业内z-score归一化跳过: {factor_name} (缺少SW行业映射)")
                    # 回退到常规z-score归一化
                    normalized[intra_zscore_name] = self._zscore_normalize_with_clipping(series, data, enable_intelligent_clipping)
                    logger.info(f"  [FALLBACK] 回退到常规z-score归一化: {intra_zscore_name}")
                    zscore_count += 1

        logger.info(f"归一化完成 - 原始因子: {len(computed)}, 最终因子: {len(normalized)} "
                   f"(z-score: {zscore_count}, rankpct: {rankpct_count}, intra-sector-rank: {intra_sector_rank_count}, intra-sector-zscore: {intra_sector_zscore_count})")
        return normalized


    def _daily_winsorize(
        self,
        computed: dict,
        data: pd.DataFrame,
        logger,
        lower: float = 0.01,
        upper: float = 0.99
    ) -> dict:
        """
        每日 cross-sectional winsorize

        Args:
            computed: 因子字典
            data: 包含 date 列的数据框
            logger: 日志记录器
            lower: 下分位数（默认 1%）
            upper: 上分位数（默认 99%）

        Returns:
            winsorize 后的因子字典
        """
        result = {}
        for factor_name, series in computed.items():
            try:
                df = pd.DataFrame({
                    'value': series.values,
                    'date': data['date'].values
                })

                # 按日期分组进行 winsorize
                df['winsorized'] = df.groupby('date')['value'].transform(
                    lambda x: x.clip(
                        lower=x.quantile(lower) if not x.isna().all() else np.nan,
                        upper=x.quantile(upper) if not x.isna().all() else np.nan
                    )
                )

                result[factor_name] = pd.Series(df['winsorized'].values, index=series.index)
            except Exception as e:
                logger.error(f"[WINSORIZE] {factor_name}: winsorize处理失败 - {e}")
                # 回退：返回原始值
                result[factor_name] = series

        logger.info(f"  Winsorize 完成，处理了 {len(result)} 个因子")
        return result

    def _zscore_normalize(self, series: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        每日 cross-sectional z-score 归一化 使用中位数(Median)和MAD替代均值和标准差，彻底免疫异常值。

        Args:
            series: 因子序列
            data: 包含 date 列的数据框

        Returns:
            z-score 归一化后的序列
        """

        def calc_mad_zscore(x):
            med = x.median()
            # 1.4826 确保了在正态分布下 MAD 与标准差量纲一致
            mad = (x - med).abs().median()
            if mad == 0: return x - med 
            return (x - med) / (mad * 1.4826)

        df = pd.DataFrame({'val': series.values, 'date': data['date'].values})
        # 先做稳健标准化，这样异常值只会变成很大的Z-score（如50），而不会压缩正常值（如1.2）
        df['zscore'] = df.groupby('date')['val'].transform(calc_mad_zscore)
        return pd.Series(df['zscore'].values, index=series.index, name=series.name)


    def _zscore_normalize_with_clipping(self, series: pd.Series, data: pd.DataFrame,
                                      enable_intelligent_clipping: bool = True) -> pd.Series:
        """
        每日 cross-sectional z-score 归一化 + 改进的智能裁剪 v2.0

        新特性：
        1. 三层检测机制：基础检测 → 深度分析 → 因子类型感知
        2. 自适应阈值调整：根据样本大小和分布特征动态调整
        3. 渐进式裁剪策略：避免突然的严格截断
        4. 边界聚集识别：识别并适当处理自然的边界聚集
        5. 因子类型感知：考虑不同类型因子的自然特性

        Args:
            series: 因子序列
            data: 包含 date 列的数据框
            enable_intelligent_clipping: 是否启用智能裁剪

        Returns:
            z-score 归一化并裁剪后的序列
        """
      
        # 先进行z-score归一化
        zscore_series = self._zscore_normalize(series, data)

        if not enable_intelligent_clipping:
            # 使用传统裁剪
            return zscore_series.clip(-3, 3)

       
        def daily_intelligent_clip_v2_series(group_series):
            """每日智能裁剪函数"""

            # 获取当日非NaN数据
            valid_data = group_series.dropna() 

            if len(valid_data) < 5:
                return group_series.clip(-3, 3)

            # 🆕 改进版智能裁剪
            try:
                # 1. 基础统计信息
                basic_stats = self._calculate_basic_stats(valid_data)

                # 2. 分布特征分析
                shape_analysis = self._analyze_distribution_shape(valid_data, basic_stats)

                # 3. 极端值分析
                extreme_analysis = self._analyze_extreme_values(valid_data, shape_analysis)

                # 4. 边界聚集检测
                boundary_analysis = self._detect_boundary_behavior(valid_data)

                # 5. 因子类型识别
                factor_name = getattr(series, 'name', 'unknown')
                # 🛡️ 防御性编程：确保factor_name是有效字符串
                if factor_name is None or not isinstance(factor_name, str):
                    factor_name = 'unknown'
                factor_type = self._identify_factor_type(factor_name)

                # 6. 获取自适应阈值
                adaptive_thresholds = self._get_adaptive_thresholds(
                    shape_analysis, extreme_analysis, boundary_analysis,
                    factor_type, len(valid_data)
                )

                # 7. 应用渐进式裁剪
                clipped_values = self._apply_graduated_clipping(valid_data, adaptive_thresholds)

                # 8. 将结果映射回原始数据结构
                result = group_series.copy()
                result.loc[valid_data.index] = clipped_values.values

                return result

            except Exception as e:
                # 如果分析失败，回退到标准裁剪
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"智能裁剪失败，回退到标准裁剪: {e}")
                return group_series.clip(-3, 3)

        # 应用改进的每日智能裁剪
        # df['clipped_zscore'] = df.groupby('date').apply(daily_intelligent_clip_v2).reset_index(level=0, drop=True)
        df = pd.DataFrame({
                'zscore': zscore_series.values,
                'date': data['date'].values
            }, index=series.index)

        # 使用 transform 对单列进行处理，传入函数的是 Series
        final_values = df.groupby('date')['zscore'].transform(daily_intelligent_clip_v2_series)
        # 🛡️ 确保Series保持正确的name属性
        return pd.Series(final_values.values, index=series.index, name=series.name)

    def _rankpct_normalize(self, series: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        每日 cross-sectional rankpct 归一化，映射到 [-1, 1]

        Args:
            series: 因子序列
            data: 包含 date 列的数据框

        Returns:
            rankpct 归一化后的序列（范围 [-1, 1]）
        """
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values
        })

        # 按日期分组进行 rankpct 归一化，映射到 [-1, 1]
        df['rankpct'] = df.groupby('date')['value'].transform(
            lambda x: 2 * x.rank(pct=True) - 1
        )

        # 🛡️ 确保Series保持正确的name属性
        return pd.Series(df['rankpct'].values, index=series.index, name=series.name)

    def _intra_sector_rank_normalize(self, series: pd.Series, data: pd.DataFrame, sw_sector_mapping: dict) -> pd.Series:
        """
        每日行业内排名归一化，映射到 [-1, 1]

        在每个交易日，对同一SW行业内的股票进行排名，消除行业间差异

        Args:
            series: 因子序列
            data: 包含 date 和 stock_code 列的数据框
            sw_sector_mapping: SW行业映射字典 {stock_code: sector_code}

        Returns:
            行业内排名归一化后的序列（范围 [-1, 1]）
        """
        logger = logging.getLogger(__name__)
    
        # 检查必需的列
        if 'stock_code' not in data.columns:
            logger.error(f"[INTRA-RANK] 错误: data中缺少stock_code列")
            # 🛡️ 确保Series保持正确的name属性
            return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)

        # 构建包含股票信息的数据框
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values,
            'stock_code': data['stock_code'].values
        })
      
        # 添加SW行业信息
        df['sw_sector'] = df['stock_code'].map(sw_sector_mapping)
        mapped_count = df['sw_sector'].notna().sum()
      
        # 移除没有SW行业映射的股票
        df = df.dropna(subset=['sw_sector'])
        
        if df.empty:
            # 如果没有有效数据，返回0值序列
            logger.warning(f"[INTRA-RANK] 没有有效数据，返回0值序列")
            # 🛡️ 确保Series保持正确的name属性
            return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)

        # 按日期和SW行业分组进行排名归一化
        def intra_sector_rank(x):
            """行业内排名函数"""
            if len(x) <= 1:
                # 如果行业内只有1只股票或为空，返回0
                return pd.Series([0.0] * len(x), index=x.index)
            # 排名映射到 [-1, 1]
            ranks = 2 * x.rank(pct=True) - 1
            return ranks

        df['intra_rank'] = df.groupby(['date', 'sw_sector'])['value'].transform(intra_sector_rank)

        # 对于没有行业映射的股票，设置为0
        result_series = pd.Series(np.zeros(len(series)), index=series.index)
        valid_indices = df.index
        result_series.iloc[valid_indices] = df['intra_rank'].values

        return result_series

    def _intra_sector_zscore_normalize(self, series: pd.Series, data: pd.DataFrame, sw_sector_mapping: dict) -> pd.Series:
        """
        每日行业内z-score归一化

        在每个交易日，对同一SW行业内的股票进行z-score标准化，消除行业间差异

        Args:
            series: 因子序列
            data: 包含 date 和 stock_code 列的数据框
            sw_sector_mapping: SW行业映射字典 {stock_code: sector_code}

        Returns:
            行业内z-score归一化后的序列
        """
        logger = logging.getLogger(__name__)
      
        # 检查必需的列
        if 'stock_code' not in data.columns:
            logger.error(f"[INTRA-ZSCORE] 错误: data中缺少stock_code列")
            # 🛡️ 确保Series保持正确的name属性
            return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)

        # 构建包含股票信息的数据框
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values,
            'stock_code': data['stock_code'].values
        })
        
        # 添加SW行业信息
        df['sw_sector'] = df['stock_code'].map(sw_sector_mapping)
        mapped_count = df['sw_sector'].notna().sum()
        
        # 移除没有SW行业映射的股票
        df = df.dropna(subset=['sw_sector'])
        
        if df.empty:
            # 如果没有有效数据，返回0值序列
            logger.warning(f"[INTRA-ZSCORE] 没有有效数据，返回0值序列")
            # 🛡️ 确保Series保持正确的name属性
            return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)

        # 按日期和SW行业分组进行z-score标准化
        def intra_sector_zscore(x):
            """行业内z-score函数"""
            if len(x) <= 1 or x.std() == 0:
                # 如果行业内只有1只股票、为空或标准差为0，返回0
                return pd.Series([0.0] * len(x), index=x.index)
            # z-score标准化
            zscores = (x - x.mean()) / x.std()
            return zscores

        df['intra_zscore'] = df.groupby(['date', 'sw_sector'])['value'].transform(intra_sector_zscore)

        # 对于没有行业映射的股票，设置为0
        result_series = pd.Series(np.zeros(len(series)), index=series.index)
        valid_indices = df.index
        result_series.iloc[valid_indices] = df['intra_zscore'].values

        return result_series

    def _intra_sector_zscore_normalize_with_clipping(self, series: pd.Series, data: pd.DataFrame,
                                                   sw_sector_mapping: dict, enable_intelligent_clipping: bool = True) -> pd.Series:
        """
        每日行业内z-score归一化 + 智能裁剪

        Args:
            series: 因子序列
            data: 包含 date 和 stock_code 列的数据框
            sw_sector_mapping: SW行业映射字典 {stock_code: sector_code}
            enable_intelligent_clipping: 是否启用智能裁剪

        Returns:
            行业内z-score归一化并裁剪后的序列
        """
        logger = logging.getLogger(__name__)

        # 先进行行业内z-score归一化
        intra_zscore_series = self._intra_sector_zscore_normalize(series, data, sw_sector_mapping)

        if not enable_intelligent_clipping:
            # 使用传统裁剪
            return intra_zscore_series.clip(-3, 3)

        # 使用智能裁剪
        df = pd.DataFrame({
            'intra_zscore': intra_zscore_series.values,
            'date': data['date'].values,
            'stock_code': data['stock_code'].values
        })

        # 添加SW行业信息
        df['sw_sector'] = df['stock_code'].map(sw_sector_mapping)

        # 移除没有SW行业映射的股票
        df = df.dropna(subset=['sw_sector'])

        if df.empty:
            logger.warning("[INTRA-ZSCORE-CLIP] 没有有效数据，返回裁剪后的序列")
            return intra_zscore_series.clip(-3, 3)

        # 应用行业内智能裁剪 - 使用transform避免索引问题
        def intra_sector_intelligent_clip(group):
            """行业内智能裁剪函数"""
            valid_data = group.dropna()

            if len(valid_data) < 5:  # 行业内样本太少，使用标准裁剪
                return group.clip(-3, 3)

            # 计算分布特征
            skewness = float(valid_data.skew())
            kurtosis = float(valid_data.kurtosis())

            # 检测极端值比例
            extreme_ratio = float((np.abs(valid_data) > 3).mean())

            # 行业内因子使用更保守的策略（样本量更小）
            if extreme_ratio > 0.03 or kurtosis > 4:  # 极端值过多或严重厚尾
                # 使用更保守的阈值
                clip_threshold = 2.5
            elif extreme_ratio > 0.015 or abs(skewness) > 0.8:  # 中等问题
                # 稍微收紧
                clip_threshold = 2.8
            else:
                # 标准阈值，但行业内使用稍微保守的值
                clip_threshold = 2.9

            # 行业内因子更严格地处理偏态
            if skewness > 0.3:  # 右偏，上阈值收紧
                upper_threshold = clip_threshold * 0.85
                lower_threshold = clip_threshold
            elif skewness < -0.3:  # 左偏，下阈值收紧
                upper_threshold = clip_threshold
                lower_threshold = clip_threshold * 0.85
            else:
                upper_threshold = clip_threshold
                lower_threshold = clip_threshold

            # 对于行业内样本很少的情况，进一步收紧
            if len(valid_data) < 10:
                upper_threshold *= 0.9
                lower_threshold *= 0.9

            return group.clip(lower_threshold, upper_threshold)

        # 使用transform来避免索引兼容性问题
        df['clipped_intra_zscore'] = df.groupby(['date', 'sw_sector'])['intra_zscore'].transform(intra_sector_intelligent_clip)

        # 重新构建结果序列
        result_series = pd.Series(np.zeros(len(series)), index=series.index)
        valid_indices = df.index
        result_series.iloc[valid_indices] = df['clipped_intra_zscore'].values

        # 对于没有行业映射的股票，应用标准裁剪
        unmapped_mask = result_series == 0
        if unmapped_mask.any():
            # 使用loc而不是iloc来避免整数索引错误
            unmapped_indices = result_series.index[unmapped_mask]
            result_series.loc[unmapped_indices] = intra_zscore_series.loc[unmapped_indices].clip(-3, 3)

        return result_series


def _prepare_optional_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None

    # 🆕 修复：处理嵌套的市场数据结构，转换为平面DataFrame
    if isinstance(df, dict):
        # 从嵌套结构提取市场数据
        records = []
        for date_str, data_dict in df.items():
            if 'individual_status' in data_dict and '399006.SZ' in data_dict['individual_status']:
                market_info = data_dict['individual_status']['399006.SZ'].get('market_data', {})
                records.append({
                    'date': date_str,
                    'daily_change_pct': market_info.get('daily_change_pct', 0.0),
                    'volume': market_info.get('volume', 0.0)
                })

        if records:
            prepared = pd.DataFrame(records)
        else:
            return None
    elif isinstance(df, pd.DataFrame):
        prepared = df.copy()
    else:
        return None

    # 标准化日期格式
    if 'date' in prepared.columns:
        prepared['date'] = pd.to_datetime(prepared['date'])

    return prepared


__all__ = ["FactorPipeline", "PipelineResult"]
