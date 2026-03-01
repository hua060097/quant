"""Enhanced support / resistance distance factors without ATR normalization (Fixed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

EPS = 1e-8


def _support_resistance_distances_optimized(
    data: pd.DataFrame,
    window: int = 45,
    n_bins: int = 36,
    decay: float = 0.93,
    smooth_sigma: float = 1.25,
    max_lookback: int = 80,
    tau: int = 8,
) -> tuple[pd.Series, pd.Series]:
    """
    综合优化版本：在一个groupby操作中同时计算支撑和阻力距离
    主要优化：
    1. 合并groupby操作，减少数据扫描次数
    2. 共享预计算的中间结果
    3. 向量化距离计算
    """
    def per_stock_combined(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        if n < max(window, max_lookback):
            # 数据不足，返回零距离
            return pd.DataFrame({
                'support_distance': np.zeros(n),
                'resistance_distance': np.zeros(n),
            }, index=group.index)

        # 预处理数据
        avg_price = group[['high', 'low']].mean(axis=1).to_numpy()
        close = group['close'].to_numpy()
        volume = group['volume'].fillna(0).to_numpy()
        lows = group['low'].interpolate(limit_direction='both').ffill().bfill().to_numpy()
        highs = group['high'].interpolate(limit_direction='both').ffill().bfill().to_numpy()

        # 初始化结果数组
        support_distance = np.zeros(n)
        resistance_distance = np.zeros(n)

        # ========== 价格-成交量节点计算 ==========
        # 预计算衰减权重矩阵
        decay_matrix = np.triu(decay ** np.arange(window).reshape(-1, 1) - np.arange(window))

        # 预计算高斯平滑核
        radius = max(1, int(3 * smooth_sigma))
        kernel_x = np.arange(-radius, radius + 1)
        gaussian_kernel = np.exp(-(kernel_x ** 2) / (2 * smooth_sigma ** 2))
        gaussian_kernel /= gaussian_kernel.sum()

        # 初始化支撑阻力位数组
        pv_support = np.full(n, np.nan)
        pv_resistance = np.full(n, np.nan)
        pv_sup_conf = np.zeros(n)
        pv_res_conf = np.zeros(n)

        for i in range(window - 1, n):
            start = i - window + 1
            end = i + 1

            prices_window = avg_price[start:end]
            volumes_window = volume[start:end]
            window_decay = decay_matrix[end - start - 1, :end - start]
            weights = volumes_window * window_decay

            valid_mask = np.isfinite(prices_window) & np.isfinite(weights) & (weights > 0)
            if valid_mask.sum() < 3:
                continue

            prices_clean = prices_window[valid_mask]
            weights_clean = weights[valid_mask]

            price_min = prices_clean.min()
            price_max = prices_clean.max()
            if price_max - price_min < 1e-6 or weights_clean.sum() <= 0:
                continue

            hist, bin_edges = np.histogram(prices_clean, bins=n_bins, range=(price_min, price_max), weights=weights_clean)
            if hist.sum() <= 0:
                continue

            hist = np.convolve(hist.astype(float), gaussian_kernel, mode='same')
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            total_weight = hist.sum()

            price_today = close[i]
            if np.isnan(price_today):
                continue

            below_mask = bin_centers <= price_today
            above_mask = bin_centers >= price_today

            if np.any(below_mask):
                candidates = np.where(below_mask)[0]
                best_idx = candidates[np.argmax(hist[candidates])]
                pv_support[i] = bin_centers[best_idx]
                pv_sup_conf[i] = hist[best_idx] / (total_weight + EPS)

            if np.any(above_mask):
                candidates = np.where(above_mask)[0]
                best_idx = candidates[np.argmax(hist[candidates])]
                pv_resistance[i] = bin_centers[best_idx]
                pv_res_conf[i] = hist[best_idx] / (total_weight + EPS)

        # ========== 摆动点计算 ==========
        low_prom = max(np.nanstd(lows), 1e-3) * 0.2
        high_prom = max(np.nanstd(highs), 1e-3) * 0.2

        low_idx, low_props = find_peaks(-lows, distance=2, prominence=low_prom)
        high_idx, high_props = find_peaks(highs, distance=2, prominence=high_prom)

        low_proms = low_props.get('prominences', np.ones_like(low_idx, dtype=float))
        high_proms = high_props.get('prominences', np.ones_like(high_idx, dtype=float))

        # 预计算时间衰减权重
        time_weights_1d = np.exp(-np.arange(max_lookback + 1) / tau)

        swing_support = np.full(n, np.nan)
        swing_resistance = np.full(n, np.nan)
        swing_sup_conf = np.zeros(n)
        swing_res_conf = np.zeros(n)

        for i in range(n):
            if np.isnan(close[i]):
                continue

            # 摆动低点
            mask_low = (low_idx <= i) & (low_idx >= i - max_lookback)
            if np.any(mask_low):
                idxs = low_idx[mask_low]
                pivots_low = lows[idxs]
                proms_low = low_proms[mask_low]
                recency = i - idxs
                proximity = 1.0 / (np.abs(close[i] - pivots_low) + 1e-4)
                time_weights = time_weights_1d[recency]
                weights = time_weights * proms_low * proximity

                w_sum = weights.sum()
                if w_sum > 0:
                    swing_support[i] = np.dot(pivots_low, weights) / w_sum
                    swing_sup_conf[i] = np.tanh(w_sum)

            # 摆动高点
            mask_high = (high_idx <= i) & (high_idx >= i - max_lookback)
            if np.any(mask_high):
                idxs = high_idx[mask_high]
                pivots_high = highs[idxs]
                proms_high = high_proms[mask_high]
                recency = i - idxs
                proximity = 1.0 / (np.abs(close[i] - pivots_high) + 1e-4)
                time_weights = time_weights_1d[recency]
                weights = time_weights * proms_high * proximity

                w_sum = weights.sum()
                if w_sum > 0:
                    swing_resistance[i] = np.dot(pivots_high, weights) / w_sum
                    swing_res_conf[i] = np.tanh(w_sum)

        # ========== 融合支撑阻力位并计算距离 ==========
        # 支撑位融合
        a = pv_support
        b = swing_support
        wa = np.where(np.isnan(a), 0.0, np.maximum(pv_sup_conf, 1e-4))
        wb = np.where(np.isnan(b), 0.0, np.maximum(swing_sup_conf, 1e-4))

        stacked_levels = np.vstack([np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)])
        stacked_weights = np.vstack([wa, wb])
        weight_sum = stacked_weights.sum(axis=0)

        blended_support = np.divide(
            (stacked_levels * stacked_weights).sum(axis=0),
            weight_sum,
            out=np.full_like(weight_sum, np.nan, dtype=float),
            where=weight_sum > EPS,
        )
        fallback_support = np.where(~np.isnan(a), a, b)
        final_support = np.where(np.isnan(blended_support), fallback_support, blended_support)

        # 阻力位融合
        a = pv_resistance
        b = swing_resistance
        wa = np.where(np.isnan(a), 0.0, np.maximum(pv_res_conf, 1e-4))
        wb = np.where(np.isnan(b), 0.0, np.maximum(swing_res_conf, 1e-4))

        stacked_levels = np.vstack([np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)])
        stacked_weights = np.vstack([wa, wb])
        weight_sum = stacked_weights.sum(axis=0)

        blended_resistance = np.divide(
            (stacked_levels * stacked_weights).sum(axis=0),
            weight_sum,
            out=np.full_like(weight_sum, np.nan, dtype=float),
            where=weight_sum > EPS,
        )
        fallback_resistance = np.where(~np.isnan(a), a, b)
        final_resistance = np.where(np.isnan(blended_resistance), fallback_resistance, blended_resistance)

        # 计算距离（向量化）
        valid_close = np.isfinite(close) & (close > 0)

        support_dist = np.zeros_like(close)
        resistance_dist = np.zeros_like(close)

        valid_support = valid_close & (~np.isnan(final_support))
        valid_resistance = valid_close & (~np.isnan(final_resistance))

        support_dist[valid_support] = (close[valid_support] - final_support[valid_support]) / np.maximum(close[valid_support], EPS)
        resistance_dist[valid_resistance] = (close[valid_resistance] - final_resistance[valid_resistance]) / np.maximum(close[valid_resistance], EPS)

        # 移除固定clip，保留原始百分比距离，让pipeline处理归一化和裁剪
        # support_dist = np.clip(support_dist, -0.35, 0.35)
        # resistance_dist = np.clip(resistance_dist, -0.35, 0.35)

        return pd.DataFrame({
            'support_distance': support_dist,
            'resistance_distance': resistance_dist,
        }, index=group.index)

    result = data.groupby('stock_code', group_keys=False).apply(per_stock_combined)
    return result['support_distance'], result['resistance_distance']


def _support_distance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """优化版本的支撑距离因子"""
    support_dist, _ = _support_resistance_distances_optimized(data)
    return support_dist


def _resistance_distance(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """优化版本的阻力距离因子"""
    _, resistance_dist = _support_resistance_distances_optimized(data)
    return resistance_dist


support_distance = register_factor(
    FactorDefinition(
        name='support_distance',
        required_columns={'stock_code', 'date', 'close', 'high', 'low', 'volume'},
        lookback_days=80,
        calculator=_support_distance,
        description='相对当前价格的支撑贴近度（多源融合）',
    )
)

resistance_distance = register_factor(
    FactorDefinition(
        name='resistance_distance',
        required_columns={'stock_code', 'date', 'close', 'high', 'low', 'volume'},
        lookback_days=80,
        calculator=_resistance_distance,
        description='相对当前价格的压力贴近度（多源融合）',
    )
)

__all__ = ['support_distance', 'resistance_distance']