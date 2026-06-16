"""
因子流水线 (LightGBM 专用 / NaN 友好版)。

核心变更：
1. 彻底移除"填充缺失值"步骤：保留 NaN，让 LightGBM 自动处理。
2. 彻底移除"预先 Winsorize"步骤：保护龙头股的极端数值。
3. 归一化逻辑：仅保留 Inf 清洗和稳健 Z-Score，确保数据纯度。
4. 质量检查：放宽对缺失值的容忍度，不再因 NaN 过多而报错。
5. 修复参数匹配：更新 _apply_postprocessing 签名以接收 enable_intelligent_clipping。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import pandas as pd
import numpy as np
import logging
from .base import FactorContext
from .registry import registry, FactorRegistry


def _expanding_percentile_from_values(values: pd.Series) -> pd.Series:
    """Return the percentile rank of each observation within its expanding history."""
    result = pd.Series(index=values.index, dtype="float32")
    clean_values = values.astype(float)
    history = []
    history_index = []
    for idx, value in clean_values.items():
        if pd.isna(value):
            result.loc[idx] = np.nan
            continue
        history.append(float(value))
        history_index.append(idx)
        rank_pct = pd.Series(history, index=history_index).rank(method="average", pct=True).iloc[-1]
        result.loc[idx] = float(rank_pct)
    return result.astype("float32")


def _validate_factor_quality(factor_series: pd.Series, factor_name: str, logger, working_df: pd.DataFrame = None):
    """
    因子质量检查函数 (宽松版)
    仅做记录和警告，不再因缺失值阻断流程。
    """
    if factor_name is None: factor_name = 'unknown'

    try:
        total_count = len(factor_series)
        nan_count = factor_series.isna().sum()
        valid_count = total_count - nan_count
        valid_ratio = valid_count / total_count if total_count > 0 else 0

        logger.info(f"[质量检查] 因子 {factor_name}: 有效率={valid_ratio:.4f} (NaN={nan_count})")

        # 检查1: 全空检查 (这是唯一需要报错的)
        if valid_count == 0:
            logger.error(f"❌ 因子 {factor_name} 全是空值！请检查计算逻辑。")
            return

        # 检查2: 常数列 (有效值完全相同)
        valid_data = factor_series.dropna()
        if valid_data.nunique() <= 1:
            logger.warning(f"⚠️  因子 {factor_name} 有效值单一 (常数列)")

        # 检查3: 极端值 (Inf)
        inf_count = np.isinf(factor_series).sum()
        if inf_count > 0:
            logger.warning(f"⚠️  因子 {factor_name} 包含 {inf_count} 个 Inf 值 (将在后续步骤清洗)")

    except Exception as e:
        logger.warning(f"⚠️ 因子 {factor_name} 质量检查异常: {e}")

@dataclass(slots=True)
class PipelineResult:
    factors: List[str]
    data: pd.DataFrame
    clipping_report: Optional[dict] = None


class FactorPipeline:
    """LightGBM 专用因子流水线 (保留 NaN)"""

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
        return_factor_data: bool = True,
    ) -> PipelineResult:
        factor_list = list(dict.fromkeys(factor_names))
        if not factor_list:
            raise ValueError("未指定任何因子。")

        # 1. 过滤已存在因子
        if existing_factors:
            factor_list = [f for f in factor_list if f not in existing_factors]
            if not factor_list:
                return PipelineResult(factors=[], data=pd.DataFrame(), clipping_report={})

        # 2. 准备数据
        working = data.copy()
        working['date'] = pd.to_datetime(working['date'])
        working = working.sort_values(['stock_code', 'date']).reset_index(drop=True)

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        target_mask = (working['date'] >= start_ts) & (working['date'] <= end_ts)
        target_rows = working.loc[target_mask].copy()
        target_index = target_rows.index
        target_index_data = target_rows[['stock_code', 'date']].reset_index(drop=True)

        extra = dict(context_extra or {})
        extra['target_start_date'] = start_ts
        extra['target_end_date'] = end_ts
        
        # 准备 Context
        ctx = FactorContext(
            market_data=_prepare_optional_df(market_data),
            extra=extra,
        )

        computed_target = {}
        logger = logging.getLogger(__name__)
        # 3. 计算因子
        for idx, name in enumerate(factor_list, 1):
            logger.info(f"计算因子 [{idx}/{len(factor_list)}]: {name}")
            try:
                definition = self.registry.get(name)
                target_series = self._compute_factor_target_series(
                    definition,
                    working,
                    target_index,
                    ctx,
                )
                computed_target[name] = target_series.astype('float32')
                
            except Exception as e:
                logger.error(f"❌ 因子 {name} 计算失败: {e}")
                # 失败时填 NaN，不阻断流程
                computed_target[name] = pd.Series(np.nan, index=range(len(target_rows)), name=name, dtype='float32')

            # 批次回调 (可选)
            if batch_callback and idx % batch_size == 0:
                batch_factors = factor_list[idx-batch_size:idx]
                self._handle_batch(
                    target_index_data,
                    batch_factors,
                    computed_target,
                    batch_callback,
                )
                if not return_factor_data and normalization_config is None:
                    for factor_name in batch_factors:
                        computed_target.pop(factor_name, None)

        # 处理剩余批次
        if batch_callback:
            remaining = len(factor_list) % batch_size
            if remaining > 0:
                remaining_factors = factor_list[-remaining:]
                self._handle_batch(
                    target_index_data,
                    remaining_factors,
                    computed_target,
                    batch_callback,
                )
                if not return_factor_data and normalization_config is None:
                    for factor_name in remaining_factors:
                        computed_target.pop(factor_name, None)

        # 原始因子已由批次回调落盘，直接返回索引，避免再复制完整 target_rows。
        if not return_factor_data and normalization_config is None:
            target_index_data.sort_values(['date', 'stock_code'], inplace=True)
            return PipelineResult(
                factors=factor_list,
                data=target_index_data,
                clipping_report={},
            )

        # 4. 后处理 (核心修改处)
        # 只保留目标日期的数据进行后处理，节省内存
        working_target = target_rows.reset_index(drop=True)
        del target_rows
        
        logger.info("开始后处理流程 (Inf清洗 -> 归一化)...")
        if normalization_config:
            # 获取配置中的 enable_intelligent_clipping，默认为 True
            enable_smart_clip = normalization_config.get('enable_intelligent_clipping', True)
            
            computed_target, report = self._apply_postprocessing(
                computed_target, 
                working_target, 
                normalization_config, 
                logger,
                enable_smart_clip # [修正] 显式传递参数
            )
        else:
            report = {}

        # 5. 组装结果。批次回调已持久化原始因子时，调用方可只取轻量索引，
        # 避免在阶段切换期间继续持有一整张原始因子表。
        if not return_factor_data:
            result = target_index_data
            result.sort_values(['date', 'stock_code'], inplace=True)
            return PipelineResult(
                factors=factor_list,
                data=result,
                clipping_report=report,
            )

        result = working_target[['stock_code', 'date']].copy()
        for name, series in computed_target.items():
            result[name] = series.values

        # 最终排序
        result.sort_values(['date', 'stock_code'], inplace=True)
        
        return PipelineResult(factors=list(computed_target.keys()), data=result, clipping_report=report)

    def _compute_factor_target_series(self, definition, working, target_index, ctx):
        scope = getattr(definition, 'calculation_scope', 'full_history')
        if scope in {'target_cross_section', 'target_with_history_window', 'daily_broadcast'}:
            target_series = definition.run_target(working, target_index, ctx)
        else:
            target_series = definition.run_target(working, target_index, ctx)

        return self._coerce_target_series(target_series, target_index, definition.name)

    def _coerce_target_series(self, series, target_index, factor_name):
        target_len = len(target_index)
        if isinstance(series, pd.Series):
            if len(series) == target_len and series.index.equals(target_index):
                return series.reset_index(drop=True).rename(factor_name)
            if len(series) == target_len and list(series.index) == list(target_index):
                return series.reset_index(drop=True).rename(factor_name)
            if len(series) == target_len and not series.index.equals(target_index):
                return pd.Series(series.to_numpy(), index=range(target_len), name=factor_name)

        if hasattr(series, '__len__') and len(series) == target_len:
            return pd.Series(series, index=range(target_len), name=factor_name)

        raise ValueError(
            f"因子{factor_name} target-only结果长度不匹配: expected={target_len}, got={len(series)}"
        )

    def _handle_batch(self, working_target, factors, computed, callback):
        """处理分批保存"""
        batch_df = working_target[['stock_code', 'date']].copy()
        for f in factors:
            batch_df[f] = computed[f].values
        callback(batch_df, factors)

    def _apply_postprocessing(
        self,
        computed: dict,
        data: pd.DataFrame,
        config: dict,
        logger,
        enable_intelligent_clipping: bool = True, # [修正] 添加此参数以匹配 commands.py 的调用
        expanding_history: pd.DataFrame | None = None,
    ) -> tuple[dict, dict]:
        """
        后处理流程 (极简版)
        
        逻辑：
        1. 替换 Inf -> NaN
        2. 归一化 (Z-Score)
        * 不再填充缺失值
        * 不再做预先 Winsorize
        """
        normalized = {}
        zscore_factors = set(config.get('zscore_factors', []))
        rankpct_factors = set(config.get('rankpct_factors', []))
        expanding_factors = set(config.get('expanding_factors', []))
        expanding_anchor_start = config.get('expanding_anchor_start')
        intra_sector_rank_factors = set(config.get('intra_sector_rank_factors', []))
        intra_sector_zscore_factors = set(config.get('intra_sector_zscore_factors', []))
        sw_sector_mapping = config.get('sw_sector_mapping', {})

        # 这里的 enable_intelligent_clipping 由参数直接传入，不再从 config 读取
        # 保持与 commands.py 的控制一致

        # 单遍清洗 + 归一化，避免同时持有 processed 与 normalized 两份完整因子表。
        for name, series in computed.items():
            s = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')

            if name in config.get('original_factors', []):
                normalized[name] = s
            elif name in zscore_factors or (
                name not in rankpct_factors
                and name not in expanding_factors
                and name not in intra_sector_rank_factors
                and name not in intra_sector_zscore_factors
            ):
                normalized[name] = self._zscore_normalize_with_clipping(
                    s, data, enable_intelligent_clipping
                )

            if name in rankpct_factors:
                normalized[f"{name}_pct"] = self._rankpct_normalize(s, data)

            if name in expanding_factors:
                normalized[f"{name}_exp"] = self._expanding_percentile_normalize(
                    s,
                    data,
                    factor_name=name,
                    history_data=expanding_history,
                    anchor_start=expanding_anchor_start,
                )

            if name in intra_sector_rank_factors:
                if sw_sector_mapping:
                    normalized[f"{name}_intra_rank"] = self._intra_sector_rank_normalize(s, data, sw_sector_mapping)
                else:
                    normalized[f"{name}_intra_rank"] = self._rankpct_normalize(s, data)

            if name in intra_sector_zscore_factors:
                if sw_sector_mapping:
                    normalized[f"{name}_intra_z"] = self._intra_sector_zscore_normalize_with_clipping(
                        s, data, sw_sector_mapping, enable_intelligent_clipping
                    )
                else:
                    normalized[f"{name}_intra_z"] = self._zscore_normalize_with_clipping(
                        s, data, enable_intelligent_clipping
                    )

        return normalized, {}

    def _zscore_normalize_with_clipping(
        self, 
        series: pd.Series, 
        data: pd.DataFrame,
        enable_clip: bool = True
    ) -> pd.Series:
        """
        稳健 Z-Score + 宽阈值裁剪 (保留 NaN)
        
        逻辑：
        1. 使用 Median/MAD 进行标准化 (抗干扰)。
        2. Clip 到 [-6, 6] (保留龙头特征，去除数据错误)。
        3. 绝不填充 NaN。
        """
        # 1. 构造临时 DF
        df = pd.DataFrame({'val': series.values, 'date': data['date'].values}, index=series.index)
        
        # 2. 计算 Median
        # transform 会自动忽略 NaN
        medians = df.groupby('date')['val'].transform('median')
        
        # 3. 计算 MAD
        abs_diff = (df['val'] - medians).abs()
        # 复用索引构造临时 DF (为了效率)
        temp_diff = pd.DataFrame({'diff': abs_diff.values, 'date': data['date'].values}, index=series.index)
        mads = temp_diff.groupby('date')['diff'].transform('median')
        
        # 4. 计算 Z-Score
        # 1.4826 是正态分布调整系数
        robust_std = mads * 1.4826
        zscore = (df['val'] - medians) / (robust_std + 1e-9)
        
        # 5. 极值处理 (MAD=0 的情况)
        # 如果某天所有股票数值一样 (MAD=0)，Z-Score 会炸
        # 此时回退到普通 Mean/Std，或者直接设为 0
        mask_zero_mad = (mads < 1e-6)
        if mask_zero_mad.any():
            # 尝试回退到 Mean/Std
            means = df.groupby('date')['val'].transform('mean')
            stds = df.groupby('date')['val'].transform('std')
            z_fallback = (df['val'] - means) / (stds + 1e-9)
            zscore = zscore.where(~mask_zero_mad, z_fallback)
        
        # 6. 裁剪 (Clipping)
        # 保留 NaN，只对有效值裁剪
        if enable_clip:
            # 放宽到 6.0，保护龙头股的极端收益
            return zscore.clip(-6.0, 6.0).astype('float32')
        else:
            return zscore.astype('float32')



    def _rankpct_normalize(self, series: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Daily cross-sectional percentile in [0, 1]."""
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values,
        }, index=series.index)
        rank_pct = df.groupby('date')['value'].transform(
            lambda x: x.rank(method='average', pct=True)
        )
        return pd.Series(rank_pct.values, index=series.index, name=series.name).astype('float32')

    def _expanding_percentile_normalize(
        self,
        series: pd.Series,
        data: pd.DataFrame,
        factor_name: str | None = None,
        history_data: pd.DataFrame | None = None,
        anchor_start: str | None = None,
    ) -> pd.Series:
        """
        Expanding percentile in [0, 1].

        Intended for market/date-level factors. The factor is collapsed to a single
        daily value via median and then broadcast back to all rows of that date.
        """
        df = pd.DataFrame({
            'value': series.values,
            'date': pd.to_datetime(data['date'].values),
        }, index=series.index)
        current_daily_values = (
            df.groupby('date', as_index=True)['value']
            .median()
            .sort_index()
        )
        daily_values = current_daily_values

        if (
            history_data is not None
            and not history_data.empty
            and factor_name
            and factor_name in history_data.columns
        ):
            hist = history_data[['date', factor_name]].copy()
            hist['date'] = pd.to_datetime(hist['date'])
            hist = hist.dropna(subset=[factor_name])
            if not hist.empty:
                hist_daily_values = (
                    hist.groupby('date', as_index=True)[factor_name]
                    .median()
                    .sort_index()
                )
                daily_values = pd.concat([hist_daily_values, current_daily_values])
                daily_values = daily_values[~daily_values.index.duplicated(keep='last')].sort_index()

        if anchor_start:
            anchor_ts = pd.to_datetime(anchor_start)
            daily_values = daily_values[daily_values.index >= anchor_ts]

        daily_exp = _expanding_percentile_from_values(daily_values)
        mapped = df['date'].map(daily_exp.to_dict())
        return pd.Series(mapped.values, index=series.index, name=series.name).astype('float32')

    def _intra_sector_rank_normalize(self, series: pd.Series, data: pd.DataFrame, sw_sector_mapping: dict) -> pd.Series:
        """Daily sector-neutral percentile in [0, 1]."""
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values,
            'stock_code': data['stock_code'].values,
        }, index=series.index)
        df['sw_sector'] = df['stock_code'].map(sw_sector_mapping)
        valid_mask = df['sw_sector'].notna()
        result = pd.Series(np.nan, index=series.index, dtype='float32')
        if valid_mask.any():
            ranked = (
                df.loc[valid_mask]
                .groupby(['date', 'sw_sector'])['value']
                .transform(lambda x: x.rank(method='average', pct=True))
            )
            result.loc[valid_mask] = ranked.astype('float32').values
        return result

    def _intra_sector_zscore_normalize(self, series: pd.Series, data: pd.DataFrame, sw_sector_mapping: dict) -> pd.Series:
        """Daily sector-neutral zscore."""
        df = pd.DataFrame({
            'value': series.values,
            'date': data['date'].values,
            'stock_code': data['stock_code'].values,
        }, index=series.index)
        df['sw_sector'] = df['stock_code'].map(sw_sector_mapping)
        valid_mask = df['sw_sector'].notna()
        result = pd.Series(np.nan, index=series.index, dtype='float32')
        if valid_mask.any():
            grouped = df.loc[valid_mask].groupby(['date', 'sw_sector'])['value']
            means = grouped.transform('mean')
            stds = grouped.transform('std').replace(0, np.nan)
            zscores = ((df.loc[valid_mask, 'value'] - means) / (stds + 1e-9)).clip(-6.0, 6.0)
            result.loc[valid_mask] = zscores.astype('float32').values
        return result

    def _intra_sector_zscore_normalize_with_clipping(
        self,
        series: pd.Series,
        data: pd.DataFrame,
        sw_sector_mapping: dict,
        enable_intelligent_clipping: bool = True,
    ) -> pd.Series:
        intra_zscore = self._intra_sector_zscore_normalize(series, data, sw_sector_mapping)
        if enable_intelligent_clipping:
            return intra_zscore.clip(-6.0, 6.0).astype('float32')
        return intra_zscore.astype('float32')


def _prepare_optional_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    if isinstance(df, dict):
        # 处理嵌套字典数据
        records = []
        for date_str, data_dict in df.items():
            # 这里需要根据实际数据结构调整，假设是简单的扁平结构
            pass 
        return pd.DataFrame(records) if records else None
        
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    return None

__all__ = ["FactorPipeline", "PipelineResult"]
