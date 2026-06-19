"""Debug version of support / resistance distance factors for tracking 300699.SZ on 2023-06-01."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import datetime
import os

from ..base import FactorContext, FactorDefinition
from ..registry import register_factor

EPS = 1e-8

# Debug configuration
DEBUG_STOCK = "300699.SZ"
DEBUG_DATE = "2023-06-01"
DEBUG_LOG_FILE = "support_distance_debug.log"

def write_debug_log(message):
    """Write debug message to log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)  # Also print to console

def _support_resistance_distances_debug(
    data: pd.DataFrame,
    window: int = 45,
    n_bins: int = 36,
    decay: float = 0.93,
    smooth_sigma: float = 1.25,
    max_lookback: int = 80,
    tau: int = 8,
) -> tuple[pd.Series, pd.Series]:
    """
    Debug version with detailed logging for 300699.SZ on 2023-06-01.
    """
    write_debug_log("=" * 80)
    write_debug_log("STARTING SUPPORT_DISTANCE DEBUG CALCULATION")
    write_debug_log(f"Input data shape: {data.shape}")
    write_debug_log(f"Date range: {data['date'].min()} ~ {data['date'].max()}")
    write_debug_log(f"Unique stocks: {data['stock_code'].nunique()}")

    # Clear previous log file
    if os.path.exists(DEBUG_LOG_FILE):
        os.remove(DEBUG_LOG_FILE)

    def per_stock_combined_debug(group: pd.DataFrame) -> pd.DataFrame:
        stock_code = group['stock_code'].iloc[0]
        stock_dates = group['date'].unique()

        write_debug_log(f"\n{'='*60}")
        write_debug_log(f"PROCESSING STOCK: {stock_code}")
        write_debug_log(f"Date range: {stock_dates[0]} ~ {stock_dates[-1]}")
        write_debug_log(f"Total records: {len(group)}")

        # Check if this is our target stock and date
        is_target_stock = stock_code == DEBUG_STOCK
        target_date = pd.to_datetime(DEBUG_DATE)

        if is_target_stock:
            write_debug_log(f"*** TARGET STOCK DETECTED: {DEBUG_STOCK} ***")
            # Log basic data for target stock
            write_debug_log(f"Target stock data preview:")
            for i, (_, row) in enumerate(group.head(10).iterrows()):
                write_debug_log(f"  {i}: {row['date']}, close={row['close']:.6f}, high={row['high']:.6f}, low={row['low']:.6f}, volume={row['volume']}")

        n = len(group)
        if n < max(window, max_lookback):
            write_debug_log(f"  INSUFFICIENT DATA: {n} < max({window}, {max_lookback})")
            result_df = pd.DataFrame({
                'support_distance': np.zeros(n),
                'resistance_distance': np.zeros(n),
            }, index=group.index)

            if is_target_stock and target_date in stock_dates:
                write_debug_log(f"  TARGET DATE RESULT: support_distance=0.000000 (insufficient data)")
            return result_df

        # Log preprocessing data for target stock
        if is_target_stock:
            write_debug_log(f"\n*** PREPROCESSING DATA for {stock_code} ***")
            write_debug_log(f"  Original data shape: {group.shape}")
            write_debug_log(f"  Missing values: close={group['close'].isna().sum()}, high={group['high'].isna().sum()}, low={group['low'].isna().sum()}, volume={group['volume'].isna().sum()}")

        # 预处理数据
        avg_price = group[['high', 'low']].mean(axis=1).to_numpy()
        close = group['close'].to_numpy()
        volume = group['volume'].fillna(0).to_numpy()
        lows = group['low'].interpolate(limit_direction='both').ffill().bfill().to_numpy()
        highs = group['high'].interpolate(limit_direction='both').ffill().bfill().to_numpy()

        # Log preprocessed data for target stock
        if is_target_stock:
            write_debug_log(f"  After interpolation:")
            write_debug_log(f"    lows NaN count: {np.isnan(lows).sum()}")
            write_debug_log(f"    highs NaN count: {np.isnan(highs).sum()}")
            write_debug_log(f"    volume zero count: {np.sum(volume == 0)}")
            write_debug_log(f"  First 10 preprocessed values:")
            for i in range(min(10, len(close))):
                write_debug_log(f"    {i}: date={group.iloc[i]['date']}, close={close[i]:.6f}, avg_price={avg_price[i]:.6f}, low={lows[i]:.6f}, high={highs[i]:.6f}, volume={volume[i]:.0f}")

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
            current_date = group.iloc[i]['date']

            # Log details for target date
            if is_target_stock and current_date == target_date:
                write_debug_log(f"\n*** TARGET DATE PROCESSING: {target_date.date()} (index {i}) ***")
                write_debug_log(f"  Window: start={start}, end={end}")
                write_debug_log(f"  Current price: close={close[i]:.6f}")
                write_debug_log(f"  Window size: {end-start}")

            prices_window = avg_price[start:end]
            volumes_window = volume[start:end]
            window_decay = decay_matrix[end - start - 1, :end - start]
            weights = volumes_window * window_decay

            if is_target_stock and current_date == target_date:
                write_debug_log(f"  Window prices (first 5, last 5): {prices_window[:5].tolist()} ... {prices_window[-5:].tolist()}")
                write_debug_log(f"  Window volumes (first 5, last 5): {volumes_window[:5].tolist()} ... {volumes_window[-5:].tolist()}")
                write_debug_log(f"  Window weights (first 5, last 5): {weights[:5].tolist()} ... {weights[-5:].tolist()}")

            valid_mask = np.isfinite(prices_window) & np.isfinite(weights) & (weights > 0)
            if valid_mask.sum() < 3:
                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  INSUFFICIENT VALID DATA: {valid_mask.sum()} < 3")
                continue

            prices_clean = prices_window[valid_mask]
            weights_clean = weights[valid_mask]

            if is_target_stock and current_date == target_date:
                write_debug_log(f"  Valid data points: {valid_mask.sum()}")
                write_debug_log(f"  Clean prices range: {prices_clean.min():.6f} ~ {prices_clean.max():.6f}")
                write_debug_log(f"  Clean weights sum: {weights_clean.sum():.6f}")

            price_min = prices_clean.min()
            price_max = prices_clean.max()
            if price_max - price_min < 1e-6 or weights_clean.sum() <= 0:
                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  INSUFFICIENT PRICE RANGE or WEIGHTS: price_range={price_max-price_min:.10f}, weight_sum={weights_clean.sum():.6f}")
                continue

            hist, bin_edges = np.histogram(prices_clean, bins=n_bins, range=(price_min, price_max), weights=weights_clean)
            if hist.sum() <= 0:
                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  EMPTY HISTOGRAM: hist.sum()={hist.sum()}")
                continue

            hist = np.convolve(hist.astype(float), gaussian_kernel, mode='same')
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            total_weight = hist.sum()

            if is_target_stock and current_date == target_date:
                write_debug_log(f"  Histogram: bins={n_bins}, range=({price_min:.6f}, {price_max:.6f})")
                write_debug_log(f"  Total weight: {total_weight:.6f}")
                write_debug_log(f"  Histogram peak: {hist.max():.6f} at index {np.argmax(hist)}")

            price_today = close[i]
            if np.isnan(price_today):
                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  CURRENT PRICE IS NAN: {price_today}")
                continue

            below_mask = bin_centers <= price_today
            above_mask = bin_centers >= price_today

            if is_target_stock and current_date == target_date:
                write_debug_log(f"  Price comparison: below_count={below_mask.sum()}, above_count={above_mask.sum()}")
                write_debug_log(f"  Below threshold: max_price_below={bin_centers[below_mask].max() if below_mask.any() else 'None'}")
                write_debug_log(f"  Above threshold: min_price_above={bin_centers[above_mask].min() if above_mask.any() else 'None'}")

            # Support calculation
            if np.any(below_mask):
                candidates = np.where(below_mask)[0]
                best_idx = candidates[np.argmax(hist[candidates])]
                pv_support[i] = bin_centers[best_idx]
                pv_sup_conf[i] = hist[best_idx] / (total_weight + EPS)

                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  SUPPORT CALCULATION:")
                    write_debug_log(f"    Best bin index: {best_idx}")
                    write_debug_log(f"    Support price: {pv_support[i]:.6f}")
                    write_debug_log(f"    Support confidence: {pv_sup_conf[i]:.6f}")
                    write_debug_log(f"    Support hist value: {hist[best_idx]:.6f}")

            # Resistance calculation
            if np.any(above_mask):
                candidates = np.where(above_mask)[0]
                best_idx = candidates[np.argmax(hist[candidates])]
                pv_resistance[i] = bin_centers[best_idx]
                pv_res_conf[i] = hist[best_idx] / (total_weight + EPS)

                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  RESISTANCE CALCULATION:")
                    write_debug_log(f"    Best bin index: {best_idx}")
                    write_debug_log(f"    Resistance price: {pv_resistance[i]:.6f}")
                    write_debug_log(f"    Resistance confidence: {pv_res_conf[i]:.6f}")
                    write_debug_log(f"    Resistance hist value: {hist[best_idx]:.6f}")

        # ========== 摆动点计算 ==========
        if is_target_stock:
            write_debug_log(f"\n*** SWING POINTS CALCULATION for {stock_code} ***")
            write_debug_log(f"  Lows std: {np.nanstd(lows):.6f}")
            write_debug_log(f"  Highs std: {np.nanstd(highs):.6f}")

        low_prom = max(np.nanstd(lows), 1e-3) * 0.2
        high_prom = max(np.nanstd(highs), 1e-3) * 0.2

        if is_target_stock:
            write_debug_log(f"  Low prominence threshold: {low_prom:.10f}")
            write_debug_log(f"  High prominence threshold: {high_prom:.10f}")

        low_idx, low_props = find_peaks(-lows, distance=2, prominence=low_prom)
        high_idx, high_props = find_peaks(highs, distance=2, prominence=high_prom)

        if is_target_stock:
            write_debug_log(f"  Low peaks found: {len(low_idx)} at indices {low_idx[:10].tolist() if len(low_idx) > 0 else []}")
            write_debug_log(f"  High peaks found: {len(high_idx)} at indices {high_idx[:10].tolist() if len(high_idx) > 0 else []}")

        low_proms = low_props.get('prominences', np.ones_like(low_idx, dtype=float))
        high_proms = high_props.get('prominences', np.ones_like(high_idx, dtype=float))

        # 预计算时间衰减权重
        time_weights_1d = np.exp(-np.arange(max_lookback + 1) / tau)

        swing_support = np.full(n, np.nan)
        swing_resistance = np.full(n, np.nan)
        swing_sup_conf = np.zeros(n)
        swing_res_conf = np.zeros(n)

        for i in range(n):
            current_date = group.iloc[i]['date']
            if np.isnan(close[i]):
                continue

            # Log swing calculation for target date
            if is_target_stock and current_date == target_date:
                write_debug_log(f"\n*** SWING POINTS FOR TARGET DATE (index {i}) ***")
                write_debug_log(f"  Current close: {close[i]:.6f}")

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

                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  LOW SWINGS ({len(idxs)} points):")
                    for j, (idx, pivot, prom, recency_val, prox, weight_val) in enumerate(zip(idxs, pivots_low, proms_low, recency, proximity, weights)):
                        write_debug_log(f"    {j}: idx={idx}, pivot={pivot:.6f}, prom={prom:.6f}, recency={recency_val}, prox={prox:.6f}, weight={weight_val:.6f}")

                w_sum = weights.sum()
                if w_sum > 0:
                    swing_support[i] = np.dot(pivots_low, weights) / w_sum
                    swing_sup_conf[i] = np.tanh(w_sum)

                    if is_target_stock and current_date == target_date:
                        write_debug_log(f"    Final swing support: {swing_support[i]:.6f}")
                        write_debug_log(f"    Swing support confidence: {swing_sup_conf[i]:.6f}")

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

                if is_target_stock and current_date == target_date:
                    write_debug_log(f"  HIGH SWINGS ({len(idxs)} points):")
                    for j, (idx, pivot, prom, recency_val, prox, weight_val) in enumerate(zip(idxs, pivots_high, proms_high, recency, proximity, weights)):
                        write_debug_log(f"    {j}: idx={idx}, pivot={pivot:.6f}, prom={prom:.6f}, recency={recency_val}, prox={prox:.6f}, weight={weight_val:.6f}")

                w_sum = weights.sum()
                if w_sum > 0:
                    swing_resistance[i] = np.dot(pivots_high, weights) / w_sum
                    swing_res_conf[i] = np.tanh(w_sum)

                    if is_target_stock and current_date == target_date:
                        write_debug_log(f"    Final swing resistance: {swing_resistance[i]:.6f}")
                        write_debug_log(f"    Swing resistance confidence: {swing_res_conf[i]:.6f}")

        # ========== 融合支撑阻力位并计算距离 ==========
        if is_target_stock:
            write_debug_log(f"\n*** FUSION AND FINAL DISTANCE CALCULATION for {stock_code} ***")

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

        # Log final results for target stock
        if is_target_stock:
            write_debug_log(f"\n*** FINAL RESULTS for {stock_code} ***")
            for i in range(n):
                current_date = group.iloc[i]['date']
                if current_date == target_date:
                    write_debug_log(f"  Target date {target_date.date()} (index {i}):")
                    write_debug_log(f"    Close price: {close[i]:.10f}")
                    write_debug_log(f"    Final support: {final_support[i]:.10f}")
                    write_debug_log(f"    PV support: {pv_support[i]:.10f}")
                    write_debug_log(f"    Swing support: {swing_support[i]:.10f}")
                    write_debug_log(f"    PV support conf: {pv_sup_conf[i]:.10f}")
                    write_debug_log(f"    Swing support conf: {swing_sup_conf[i]:.10f}")
                    write_debug_log(f"    Support distance: {support_dist[i]:.15f}")

                    write_debug_log(f"    Final resistance: {final_resistance[i]:.10f}")
                    write_debug_log(f"    PV resistance: {pv_resistance[i]:.10f}")
                    write_debug_log(f"    Swing resistance: {swing_resistance[i]:.10f}")
                    write_debug_log(f"    PV resistance conf: {pv_res_conf[i]:.10f}")
                    write_debug_log(f"    Swing resistance conf: {swing_res_conf[i]:.10f}")
                    write_debug_log(f"    Resistance distance: {resistance_dist[i]:.15f}")
                    break

        result_df = pd.DataFrame({
            'support_distance': support_dist,
            'resistance_distance': resistance_dist,
        }, index=group.index)

        write_debug_log(f"  Completed processing {stock_code}: {len(result_df)} records")
        return result_df

    write_debug_log("Starting groupby operation...")
    result = data.groupby('stock_code', group_keys=False).apply(per_stock_combined_debug)
    write_debug_log(f"Groupby operation completed, result shape: {result.shape}")

    write_debug_log("\n" + "=" * 80)
    write_debug_log("SUPPORT_DISTANCE DEBUG CALCULATION COMPLETED")
    write_debug_log("=" * 80)

    return result['support_distance'], result['resistance_distance']


def _support_distance_debug(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """Debug version of support distance factor"""
    support_dist, _ = _support_resistance_distances_debug(data)
    return support_dist


def _resistance_distance_debug(data: pd.DataFrame, _: FactorContext) -> pd.Series:
    """Debug version of resistance distance factor"""
    _, resistance_dist = _support_resistance_distances_debug(data)
    return resistance_dist


# Register debug versions (commented out by default to avoid interference)
# Uncomment these lines when you want to run the debug version
# support_distance_debug = register_factor(
#     FactorDefinition(
#         name='support_distance',
#         required_columns={'stock_code', 'date', 'close', 'high', 'low', 'volume'},
#         lookback_days=80,
#         calculator=_support_distance_debug,
#         description='DEBUG: 相对当前价格的支撑贴近度（多源融合）',
#     )
# )
#
# resistance_distance_debug = register_factor(
#     FactorDefinition(
#         name='resistance_distance',
#         required_columns={'stock_code', 'date', 'close', 'high', 'low', 'volume'},
#         lookback_days=80,
#         calculator=_resistance_distance_debug,
#         description='DEBUG: 相对当前价格的压力贴近度（多源融合）',
#     )
# )

__all__ = ['_support_resistance_distances_debug', '_support_distance_debug', '_resistance_distance_debug']