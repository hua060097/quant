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

        extra = dict(context_extra or {})
        extra['target_start_date'] = start_ts
        extra['target_end_date'] = end_ts
        
        # 准备 Context
        ctx = FactorContext(
            market_data=_prepare_optional_df(market_data),
            sector_data=_prepare_optional_df(sector_data),
            extra=extra,
        )

        computed = {}
        logger = logging.getLogger(__name__)

        # 3. 计算因子
        for idx, name in enumerate(factor_list, 1):
            logger.info(f"计算因子 [{idx}/{len(factor_list)}]: {name}")
            try:
                definition = self.registry.get(name)
                series = definition.run(working, ctx)
                
                # 统一转为 float64 并赋予正确的 name
                computed[name] = pd.Series(series, name=name, index=working.index).astype('float64')
                
            except Exception as e:
                logger.error(f"❌ 因子 {name} 计算失败: {e}")
                # 失败时填 NaN，不阻断流程
                computed[name] = pd.Series(np.nan, index=working.index, name=name)

            # 批次回调 (可选)
            if batch_callback and idx % batch_size == 0:
                self._handle_batch(working, target_mask, factor_list[idx-batch_size:idx], computed, batch_callback)

        # 处理剩余批次
        if batch_callback:
            remaining = len(factor_list) % batch_size
            if remaining > 0:
                self._handle_batch(working, target_mask, factor_list[-remaining:], computed, batch_callback)

        # 4. 后处理 (核心修改处)
        # 只保留目标日期的数据进行后处理，节省内存
        working_target = working[target_mask].reset_index(drop=True)
        computed_target = {k: v[target_mask].reset_index(drop=True) for k, v in computed.items()}
        
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

        # 5. 组装结果
        result = working_target[['stock_code', 'date']].copy()
        for name, series in computed_target.items():
            result[name] = series.values

        # 最终排序
        result.sort_values(['date', 'stock_code'], inplace=True)
        
        return PipelineResult(factors=list(computed_target.keys()), data=result, clipping_report=report)

    def _handle_batch(self, working, mask, factors, computed, callback):
        """处理分批保存"""
        batch_df = working.loc[mask, ['stock_code', 'date']].copy()
        for f in factors:
            batch_df[f] = computed[f][mask].values
        callback(batch_df, factors)

    def _apply_postprocessing(
        self,
        computed: dict,
        data: pd.DataFrame,
        config: dict,
        logger,
        enable_intelligent_clipping: bool = True # [修正] 添加此参数以匹配 commands.py 的调用
    ) -> tuple[dict, dict]:
        """
        后处理流程 (极简版)
        
        逻辑：
        1. 替换 Inf -> NaN
        2. 归一化 (Z-Score)
        * 不再填充缺失值
        * 不再做预先 Winsorize
        """
        processed = {}
        
        # 1. 基础清洗：Inf -> NaN
        for name, series in computed.items():
            processed[name] = series.replace([np.inf, -np.inf], np.nan)

        # 2. 归一化
        normalized = {}
        
        # 这里的 enable_intelligent_clipping 由参数直接传入，不再从 config 读取
        # 保持与 commands.py 的控制一致

        for name, series in processed.items():
            # 如果配置了 original_factors，则跳过归一化
            if name in config.get('original_factors', []):
                normalized[name] = series
                continue
                
            # 默认逻辑：全部做稳健 Z-Score + 智能裁剪
            normalized[name] = self._zscore_normalize_with_clipping(
                series, data, enable_intelligent_clipping
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
            return zscore.clip(-6.0, 6.0).astype('float64')
        else:
            return zscore.astype('float64')


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