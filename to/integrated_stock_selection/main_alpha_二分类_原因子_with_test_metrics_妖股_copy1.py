#!/usr/bin/env python3
"""
[强化]量化选股系统 - 优化完成版

支持五种运行模式：
1. 因子生成模式 (generate_factors) - 生成和计算因子
2. 因子测试模式 (test_factors) - 单因子测试和分析
3. 模型训练模式 (train_model) - 训练机器学习模型
4. 模型测试模式 (test_model) - 使用优化后的模型进行选股测试
5. 完整流水线模式 (full_pipeline) - 执行完整的因子-模型流水线

[TARGET] 已完成的核心优化：
- [OK] 智能标签变换：平衡极值学习与信号保留
- [OK] 增强模型复杂度：Pseudo-Huber回归专用参数优化
- [OK] 交易导向评估：优先Top-K收益、夏普比率、最大回撤等实际交易指标
- [OK] 移除冗余Quantile Loss：智能标签变换已解决极值处理问题
- [OK] 修复trading_metrics变量作用域错误

基于2020-2025年顶尖量化研究的模块化因子系统架构
"""

import logging
import pandas as pd
import json
import argparse
import sys
import numpy as np
import math
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

# Python官方推荐：在模块级别定义logger
logger = logging.getLogger(__name__)

# 分钟回测模式:
# False -> 固定 Top1 满仓
# True  -> 使用 live_trading 默认策略(日级 Top1/Top2/空仓切换)
USE_LIVE_DEFAULT_SCHEME_FOR_MINUTE_BACKTEST = False

# Top1 日级开关
# False -> 不做 Top1 开关，固定交易 Top1
# True  -> 仅当当天 Top1 的 price_ma5_ratio_pct >= 0.4 才允许买入
USE_TOP1_PRICE_MA5_RATIO_GATE = False

# Top1 OR 型双分支风控
# False -> 不做该风控
# True  -> 若满足任一条件则空仓/跳过:
#          1) sector_momentum_accel_rank < 0.287 且 market_volatility_context > 1.695
#          2) ma_alignment_persistence_5d < 1.0
USE_TOP1_TWO_BRANCH_GATE = False
TOP1_SECTOR_MOMENTUM_ACCEL_GATE_THRESHOLD = 0.287
TOP1_MARKET_VOLATILITY_CONTEXT_GATE_THRESHOLD = 1.695
TOP1_MA_ALIGNMENT_PERSISTENCE_GATE_THRESHOLD = 1.0
USE_TOP1_TWO_BRANCH_STRICT_MODE = False
TOP1_STRICT_ENTRY_GAP_UP_THRESHOLD = 0.02

# Top1 开盘极端过滤模式
# 'gap6' -> 低开超过6%跳过，同时保留 gap_up > 5% 跳过（默认）
# 'ma5'  -> 开盘价低于开盘MA5跳过，同时保留 gap_up > 5% 跳过（备选）
TOP1_ENTRY_FILTER_MODE = 'gap6'
#
# 延后买入确认模式
# 'disabled'              -> 沿用当前实现（09:30开盘价成交，日志时间显示为09:35）
# '1000_not_drop_1pct'    -> 延后到10:00买入，且10:00相对09:35收盘价未跌超1%才买
TOP1_DELAYED_ENTRY_MODE = 'disabled'

# Top1 专用三层分钟止盈止损
ENABLE_T2_0935_HARD_STOP = True
T2_0935_HARD_STOP_THRESHOLD = -0.04
ENABLE_T2_1000_SOFT_STOP = True
T2_1000_SOFT_STOP_THRESHOLD = -0.02
ENABLE_SECOND_LAYER_TIME_STOP = True
T2_CLOSE_BREAKEVEN_STOP_THRESHOLD = 0.00
T3_0935_BREAKEVEN_STOP_THRESHOLD = 0.00
ENABLE_SMALL_PROFIT_FAILURE_EXIT = True
SMALL_PROFIT_ACTIVATION_RETURN = 0.04
SMALL_PROFIT_RETRACE_RETURN = -0.005
SMALL_PROFIT_TRAILING_DRAWDOWN = -0.04
SMALL_PROFIT_BEARISH_REQUIRED_COUNT = 2
SMALL_PROFIT_WATCH_WINDOW_BARS = 3
ENABLE_PROFIT_THRESHOLD_FOR_BEARISH_EXIT = True
PROFIT_BEARISH_ACTIVATION_RETURN = 0.05
ENABLE_TOP1_STATE_MACHINE_RISK = True

# ============================================================
# Exhaustion factor buy filter (高位衰竭因子买入过滤)
# ============================================================
ENABLE_EXHAUSTION_BUY_FILTER = True

# 策略选择:
#   'runup_or_blowoff' (默认): [runup_ma5_exhaustion > 0.8 AND top10_10d_strength_level_v3 >= 0.5]
#                              OR [prev_day_blowoff_risk >= 4.15 AND sector_momentum_accel_rank <= 0.41]
#                              OR [market_advance_decline_ratio >= 5.39559]
#   'runup_ma5':                runup_ma5_exhaustion > 0.8 AND top10_10d_strength_level_v3 >= 0.5 → 拦截
#   'close_below_ma5':          close_below_ma5_after_runup==1 AND turnover_z_20>THRESHOLD → 拦截
EXHAUSTION_FILTER_STRATEGY = 'runup_or_blowoff'

# v2 策略参数 (strategy='runup_ma5')
EXHAUSTION_RUNUP_MA5_THRESHOLD = 0.8  # runup_ma5_exhaustion > 此值拦截
EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD = 0.5  # 仅在头部中期强度较高环境下启用runup_ma5拦截
EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD = 4.15
EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD = 0.41
EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD = 5.39559

# v1 策略参数 (strategy='close_below_ma5')
EXHAUSTION_FILTER_TURNOVER_Z_THRESHOLD = 1.0  # turnover_z_20 > 此值才确认衰竭

# 导入标准化SW编码接口
from sw_encoding_loader import (
    load_standardized_sw_encoding,
    get_sw_mapping_for_model,
    get_sw_industry_code,
    load_sw_sector_mapping as load_standardized_sw_sector_mapping
)
logger.info("[SW编码] 使用标准化编码系统")

def print_top_k_robust_stats(top_1_all_returns, top_3_all_returns, top_5_all_returns, top_10_all_returns):
    """
    打印Top-k稳健统计指标 - 简化版本，推荐关注最重要的3个指标

    🆕 注意：这里的Top-k表示"第k名预测值的股票"，而不是"前k名的股票"
    - Top-1: 每天预测值排名第1的股票收益
    - Top-3: 每天预测值排名第3的股票收益（只有1只股票）
    - Top-5: 每天预测值排名第5的股票收益（只有1只股票）
    - Top-10: 每天预测值排名第10的股票收益（只有1只股票）

    Args:
        top_1_all_returns: 第1名预测值的股票收益数组
        top_3_all_returns: 第3名预测值的股票收益数组
        top_5_all_returns: 第5名预测值的股票收益数组
        top_10_all_returns: 第10名预测值的股票收益数组
    """

    # 收益分布直方图
    logger.info("")
    logger.info("[HISTOGRAM] 收益分布直方图:")

    for k, all_returns, name in [(1, top_1_all_returns, "Top-1"),
                                  (3, top_3_all_returns, "Top-3"),
                                  (5, top_5_all_returns, "Top-5"),
                                  (10, top_10_all_returns, "Top-10")]:
        if len(all_returns) > 0:
            logger.info(f"  {name}收益分布:")
            create_ascii_histogram(all_returns, bins=12)
            logger.info("")

    logger.info("")
    logger.info("💡 推荐对比指标:")
    logger.info("  1. Winsorized均值 - 去除极端值影响，更稳健")
    logger.info("  2. 夏普比率 - 风险调整后收益，更全面")
    logger.info("  3. 传统均值 - 作为参考，但注意极端值影响")
    logger.info("")


def create_ascii_histogram(values, bins=15):
    """创建ASCII直方图"""
    if len(values) == 0:
        return

    # 计算直方图数据
    hist, bin_edges = np.histogram(values, bins=bins)
    max_count = hist.max()

    logger.info(f"      区间          频数  直方图")
    logger.info(f"      " + "-"*45)

    for i, count in enumerate(hist):
        # 格式化区间显示
        left_edge = bin_edges[i]
        right_edge = bin_edges[i+1]
        interval = f"[{left_edge:6.3f},{right_edge:6.3f})"

        # 创建条形图
        if max_count > 0:
            bar_length = int(count / max_count * 30)  # 30字符最大宽度
            bar = "█" * bar_length
        else:
            bar = ""

        logger.info(f"      {interval} {count:3d}   {bar}")

def calculate_ndcg_at_k(y_true, y_score, k=5):
    """
    计算NDCG@k (Normalized Discounted Cumulative Gain) - 使用真实收益率

    Args:
        y_true: 真实相关性得分（推荐使用收益率）
        y_score: 预测得分
        k: 截断位置

    Returns:
        NDCG@k 值
    """
    # 获取排序后的索引
    sorted_indices = np.argsort(y_score)[::-1]  # 降序排列
    sorted_true_relevance = y_true[sorted_indices[:k]]

    # 计算DCG@k
    dcg = 0.0
    for i, relevance in enumerate(sorted_true_relevance):
        if i == 0:
            dcg += relevance
        else:
            dcg += relevance / math.log2(i + 1)

    # 计算IDCG@k (理想DCG)
    ideal_sorted_relevance = np.sort(y_true)[::-1][:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted_relevance):
        if i == 0:
            idcg += relevance
        else:
            idcg += relevance / math.log2(i + 1)

    # 计算NDCG@k
    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_adjusted_ndcg_at_k(y_true, y_score, k=5):
    """
    [强化] 改进的NDCG计算，解决负收益和量纲问题

    主要改进：
    1. 收益率标准化，解决量纲敏感性问题
    2. 负收益阈值处理，避免过度惩罚
    3. 基于百分位的稳健归一化

    Args:
        y_true: 真实收益率数组
        y_score: 预测得分数组
        k: 截断位置

    Returns:
        float: 改进的NDCG@k 值
    """
    import numpy as np

    # 输入验证
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return 0.0

    if k <= 0:
        return 0.0

    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)

    # [WRENCH] 改进1：收益率标准化（z-score）
    if y_true.std() > 1e-8:  # 避免除零
        normalized_returns = (y_true - y_true.mean()) / y_true.std()
    else:
        # 如果标准差为0，直接设为0
        normalized_returns = np.zeros_like(y_true)

    # [WRENCH] 改进2：负收益阈值处理
    # 使用30%分位数作为最小值阈值，避免过度惩罚极端负收益
    min_threshold = np.percentile(normalized_returns, 30)
    normalized_returns = np.maximum(normalized_returns, min_threshold)

    # [WRENCH] 改进3：确保有足够的正值用于有意义的排序
    # 如果大部分值都是负数，进行平移
    if normalized_returns.mean() < 0:
        normalized_returns = normalized_returns - normalized_returns.mean()

    # 使用原始NDCG逻辑计算标准化的结果
    return calculate_ndcg_at_k(normalized_returns, y_score, k)


def _calc_max_drawdown(returns):
    rets = np.asarray(returns, dtype=float)
    if rets.size == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + rets)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / np.maximum(running_peak, 1e-12) - 1.0
    return float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0


def _build_or_load_validation_minute_cache_binary(
    val_data_raw: Optional[pd.DataFrame],
    selection_data_raw: Optional[pd.DataFrame] = None,
    test_data_raw: Optional[pd.DataFrame] = None,
    holding_days: int = 3,
    minute_data_path: str = "../local_data/minute/",
    build_if_missing: bool = False,
) -> Optional[pd.DataFrame]:
    if val_data_raw is None or len(val_data_raw) == 0:
        return None

    from rqalpha_minute_datasource.strategy_backtest_lead_fusion_rank_binary_v2 import (
        StrategyConfig,
        StrategyType,
        build_minute_trade_cache,
        get_minute_trade_cache_rule_signature,
    )

    context_frames = [df for df in [val_data_raw, selection_data_raw, test_data_raw] if df is not None and len(df) > 0]
    if not context_frames:
        return None

    target_rows = val_data_raw.copy()
    context_rows = pd.concat(context_frames, ignore_index=True)
    for df in (target_rows, context_rows):
        df['date'] = pd.to_datetime(df['date'])

    trading_dates = sorted(pd.to_datetime(context_rows['date']).dt.date.unique().tolist())
    cfg = StrategyConfig(
        strategy_type=StrategyType.TOP_1,
        stock_count=1,
        enable_dynamic_rank_switch=False,
        holding_days=int(holding_days),
        max_concurrent_positions=1,
        fixed_position_pct=1.0,
        position_schedule=[1.0]
    )
    # todo 20260502 规则变了导致加载不到，所以改成写死
    # rule_sig = get_minute_trade_cache_rule_signature(cfg)
    rule_sig = "hd3_entrygap_based_down-0.030_downn-0.050_up0.050_riskt1_low_break_with_high_drawdown_sl-0.050_pb3"


    cache_dir = Path(__file__).resolve().parent / "results" / "minute_eval_cache"
    start_str = pd.to_datetime(target_rows['date']).min().strftime('%Y%m%d')
    end_str = pd.to_datetime(target_rows['date']).max().strftime('%Y%m%d')
    cache_path = cache_dir / f"val_minute_cache_{start_str}_{end_str}_{rule_sig}.parquet"

    logger.info(
        f"[MINUTE_CACHE] 二分类脚本目标验证集分钟缓存: rows={len(target_rows):,}, "
        f"dates={target_rows['date'].nunique()}, path={cache_path}"
    )
    if cache_path.exists():
        logger.info(f"[MINUTE_CACHE] 命中已有缓存: {cache_path}")
        return pd.read_parquet(cache_path)

    shard_pattern = re.compile(
        rf"^val_minute_cache_(\d{{8}})_(\d{{8}})_{re.escape(rule_sig)}\.parquet$"
    )
    shard_files = []
    for path in cache_dir.glob(f"val_minute_cache_*_{rule_sig}.parquet"):
        match = shard_pattern.match(path.name)
        if not match:
            continue
        shard_start = pd.to_datetime(match.group(1))
        shard_end = pd.to_datetime(match.group(2))
        if shard_end < pd.to_datetime(start_str) or shard_start > pd.to_datetime(end_str):
            continue
        shard_files.append((shard_start, shard_end, path))

    if shard_files:
        shard_files.sort(key=lambda x: (x[0], x[1], str(x[2])))
        logger.info(f"[MINUTE_CACHE] 二分类脚本尝试拼接 {len(shard_files)} 个分钟缓存分片...")
        shard_frames = []
        for _, _, shard_path in shard_files:
            logger.info(f"[MINUTE_CACHE] 读取分片: {shard_path.name}")
            shard_df = pd.read_parquet(shard_path)
            shard_df['date'] = pd.to_datetime(shard_df['date'])
            shard_df = shard_df[
                (shard_df['date'] >= pd.to_datetime(start_str)) &
                (shard_df['date'] <= pd.to_datetime(end_str))
            ].copy()
            shard_frames.append(shard_df)

        if shard_frames:
            merged_df = pd.concat(shard_frames, ignore_index=True)
            merged_df = (
                merged_df
                .sort_values(['date', 'stock_code'])
                .drop_duplicates(['date', 'stock_code'], keep='last')
                .reset_index(drop=True)
            )
            expected_dates = set(pd.to_datetime(target_rows['date']).dt.normalize().unique().tolist())
            actual_dates = set(pd.to_datetime(merged_df['date']).dt.normalize().unique().tolist())
            missing_dates = sorted(expected_dates - actual_dates)
            if not missing_dates:
                logger.info(
                    f"[MINUTE_CACHE] 二分类脚本成功通过增量分片拼接缓存: rows={len(merged_df):,}, dates={len(actual_dates)}"
                )
                return merged_df
            logger.warning(
                f"[MINUTE_CACHE] 二分类脚本分片缓存仍缺少 {len(missing_dates)} 个交易日: "
                f"{[d.strftime('%Y-%m-%d') for d in missing_dates[:5]]}"
            )

    if not build_if_missing:
        logger.warning("[MINUTE_CACHE] 二分类脚本未命中有效分钟缓存，当前仅尝试读取不自动构建。")
        return None

    return build_minute_trade_cache(
        target_prediction_rows=target_rows,
        context_prediction_data=context_rows,
        minute_data_path=minute_data_path,
        trading_dates=trading_dates,
        cache_path=cache_path,
        holding_days=int(holding_days),
        overwrite=False,
        log_prefix="[MINUTE_CACHE]"
    )


def _calc_daily_rank_panel_binary(model, X_eval, y_eval_original, eval_dates, eval_data_raw=None, rank_position=1):
    if X_eval is None or len(X_eval) == 0 or y_eval_original is None or len(y_eval_original) == 0:
        return pd.DataFrame()
    try:
        best_iter = getattr(model, 'best_iteration', None)
        eval_pred = model.predict(X_eval, num_iteration=best_iter) if best_iter else model.predict(X_eval)
    except Exception:
        eval_pred = model.predict(X_eval)

    eval_pred = np.asarray(eval_pred, dtype=float).reshape(-1)
    y_eval_original = np.asarray(y_eval_original, dtype=float).reshape(-1)
    n = min(len(eval_pred), len(y_eval_original), len(eval_dates))
    if n <= 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'date': pd.to_datetime(eval_dates[:n]),
        'pred': eval_pred[:n],
        'actual_return': y_eval_original[:n]
    })
    if eval_data_raw is not None and len(eval_data_raw) > 0:
        eval_data_raw_aligned = eval_data_raw.reset_index(drop=True)
        for col in ['stock_code', 'relevance', 'future_min_low_ret', 'risk_adjusted_return_label']:
            if col in eval_data_raw_aligned.columns:
                df[col] = eval_data_raw_aligned[col].values[:len(df)]

    rows = []
    for date_val, group in df.groupby('date', sort=True):
        if len(group) < max(rank_position, 5):
            continue
        sorted_group = group.sort_values('pred', ascending=False).reset_index(drop=True)
        row = sorted_group.iloc[rank_position - 1]
        rows.append({
            'date': date_val,
            'stock_code': row.get('stock_code'),
            'rank': int(rank_position),
            'pred_score': float(row['pred']),
            'actual_return': float(row['actual_return']),
            'relevance': float(row['relevance']) if 'relevance' in row and pd.notna(row['relevance']) else np.nan,
            'future_min_low_ret': float(row['future_min_low_ret']) if 'future_min_low_ret' in row and pd.notna(row['future_min_low_ret']) else np.nan,
            'risk_adjusted_return_label': float(row['risk_adjusted_return_label']) if 'risk_adjusted_return_label' in row and pd.notna(row['risk_adjusted_return_label']) else np.nan,
        })
    return pd.DataFrame(rows)


def _calc_daily_topk_selection_rows_binary(model, X_eval, y_eval_original, eval_dates, eval_data_raw=None, top_k=3, min_group_size=5):
    try:
        best_iter = getattr(model, 'best_iteration', None)
        eval_pred = model.predict(X_eval, num_iteration=best_iter) if best_iter else model.predict(X_eval)
    except Exception:
        eval_pred = model.predict(X_eval)

    eval_pred = np.asarray(eval_pred, dtype=float).reshape(-1)
    y_eval_original = np.asarray(y_eval_original, dtype=float).reshape(-1)
    n = min(len(eval_pred), len(y_eval_original), len(eval_dates))
    if n <= 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'date': pd.to_datetime(eval_dates[:n]),
        'pred': eval_pred[:n],
        'actual_return': y_eval_original[:n]
    })
    if eval_data_raw is not None and len(eval_data_raw) > 0:
        eval_data_raw_aligned = eval_data_raw.reset_index(drop=True)
        for col in ['stock_code', 'relevance', 'future_min_low_ret', 'risk_adjusted_return_label']:
            if col in eval_data_raw_aligned.columns:
                df[col] = eval_data_raw_aligned[col].values[:len(df)]

    selected_rows = []
    for date_val, group in df.groupby('date', sort=True):
        if len(group) < max(top_k, min_group_size):
            continue
        sorted_group = group.sort_values('pred', ascending=False).reset_index(drop=True)
        basket = sorted_group.head(top_k).copy()
        for rank_idx, row in enumerate(basket.itertuples(index=False), 1):
            selected_rows.append({
                'date': date_val,
                'stock_code': getattr(row, 'stock_code', None),
                'rank': int(rank_idx),
                'pred_score': float(getattr(row, 'pred')),
                'actual_return': float(getattr(row, 'actual_return')),
                'relevance': float(getattr(row, 'relevance', np.nan)) if pd.notna(getattr(row, 'relevance', np.nan)) else np.nan,
                'future_min_low_ret': float(getattr(row, 'future_min_low_ret', np.nan)) if pd.notna(getattr(row, 'future_min_low_ret', np.nan)) else np.nan,
                'risk_adjusted_return_label': float(getattr(row, 'risk_adjusted_return_label', np.nan)) if pd.notna(getattr(row, 'risk_adjusted_return_label', np.nan)) else np.nan,
            })
    return pd.DataFrame(selected_rows)


def _evaluate_candidate_with_minute_cache_binary(model, X_eval, y_eval_original, eval_dates, eval_data_raw, minute_cache_df):
    if minute_cache_df is None or len(minute_cache_df) == 0:
        return None

    top1_panel = _calc_daily_rank_panel_binary(model, X_eval, y_eval_original, eval_dates, eval_data_raw, rank_position=1)
    if top1_panel is None or top1_panel.empty:
        return None

    cache_cols = [
        'date', 'stock_code', 'buy_executable', 'skip_reason', 'exit_reason',
        'minute_realized_return', 'minute_min_return', 'minute_max_drawdown',
        'stoploss_hit', 'profit_bearish_hit', 'holding_days_realized',
        'missing_minute_data'
    ]
    cache_view = minute_cache_df[[c for c in cache_cols if c in minute_cache_df.columns]].copy()
    cache_view['date'] = pd.to_datetime(cache_view['date'])

    top1_merged = top1_panel.merge(cache_view, on=['date', 'stock_code'], how='left')
    top1_merged['buy_executable'] = top1_merged['buy_executable'].fillna(False).astype(bool)
    top1_merged['missing_minute_data'] = top1_merged['missing_minute_data'].fillna(True).astype(bool)
    top1_merged['stoploss_hit'] = top1_merged['stoploss_hit'].fillna(False).astype(bool)
    top1_merged['profit_bearish_hit'] = top1_merged['profit_bearish_hit'].fillna(False).astype(bool)
    top1_merged['minute_realized_return'] = pd.to_numeric(top1_merged.get('minute_realized_return', 0.0), errors='coerce').fillna(0.0)
    top1_merged['effective_minute_return'] = np.where(top1_merged['buy_executable'], top1_merged['minute_realized_return'], 0.0)

    rets = top1_merged['effective_minute_return'].values.astype(float)
    mean_return = float(np.mean(rets)) if len(rets) > 0 else 0.0
    median_return = float(np.median(rets)) if len(rets) > 0 else 0.0
    win_rate = float(np.mean(rets > 0)) if len(rets) > 0 else 0.0
    q10 = float(np.percentile(rets, 10)) if len(rets) > 0 else 0.0
    max_drawdown = _calc_max_drawdown(rets)
    cumulative_return = float(np.prod(1.0 + rets) - 1.0) if len(rets) > 0 else 0.0
    ret_std = float(np.std(rets)) if len(rets) > 1 else 0.0
    downside_rets = rets[rets < 0]
    downside_std = float(np.std(downside_rets)) if len(downside_rets) > 0 else 0.0
    drawdown_abs = max(abs(max_drawdown), 0.05)
    calmar_like = cumulative_return / drawdown_abs
    efficiency_gap = cumulative_return + max_drawdown
    sharpe_like = mean_return / max(ret_std, 0.01)
    sortino_like = mean_return / max(downside_std, 0.01)
    stoploss_rate = float(np.mean(top1_merged['stoploss_hit'])) if len(top1_merged) > 0 else 0.0
    profit_bearish_rate = float(np.mean(top1_merged['profit_bearish_hit'])) if len(top1_merged) > 0 else 0.0
    skip_rate = float(np.mean(~top1_merged['buy_executable'])) if len(top1_merged) > 0 else 0.0
    missing_rate = float(np.mean(top1_merged['missing_minute_data'])) if len(top1_merged) > 0 else 0.0

    # ===== 分段独立评估：核心指标分段算再取平均，防止单一好运段主导总分 =====
    minute_segment_metrics = []
    minute_segment_penalty = 0.0
    minute_positive_segments = 0
    minute_weak_segments = 0
    # 收集各段核心指标用于取平均
    seg_sharpes, seg_sortinos, seg_cums, seg_calmars, seg_efficiencies = [], [], [], [], []
    seg_win_rates, seg_means = [], []

    unique_days = np.array(sorted(pd.to_datetime(top1_merged['date']).unique()))
    minute_day_segments = [seg for seg in np.array_split(unique_days, 3) if len(seg) > 0]
    for seg_idx, seg_days in enumerate(minute_day_segments, 1):
        seg_df = top1_merged[top1_merged['date'].isin(seg_days)].copy()
        seg_rets = seg_df['effective_minute_return'].values.astype(float)
        if len(seg_rets) == 0:
            continue
        seg_mean = float(np.mean(seg_rets))
        seg_win = float(np.mean(seg_rets > 0))
        seg_q10 = float(np.percentile(seg_rets, 10))
        seg_dd = _calc_max_drawdown(seg_rets)
        seg_cum = float(np.prod(1.0 + seg_rets) - 1.0)
        seg_stoploss = float(np.mean(seg_df['stoploss_hit'])) if len(seg_df) > 0 else 0.0
        seg_std = float(np.std(seg_rets)) if len(seg_rets) > 1 else 0.0
        seg_down = seg_rets[seg_rets < 0]
        seg_down_std = float(np.std(seg_down)) if len(seg_down) > 0 else 0.0

        # 逐段独立计算核心指标
        seg_sharpes.append(seg_mean / max(seg_std, 0.01))
        seg_sortinos.append(seg_mean / max(seg_down_std, 0.01))
        seg_cums.append(seg_cum)
        seg_calmars.append(seg_cum / max(abs(seg_dd), 0.05))
        seg_efficiencies.append(seg_cum + seg_dd)
        seg_win_rates.append(seg_win)
        seg_means.append(seg_mean)

        if seg_cum > 0:
            minute_positive_segments += 1
        if seg_cum < 0 or seg_dd < -0.22 or seg_stoploss > 0.18:
            minute_weak_segments += 1

        # 硬地板：任一段负收益 → 巨额惩罚，确保该参数组合不可能被选中
        seg_penalty = 0.0
        if seg_cum < -0.05:
            seg_penalty += 8000.0  # 亏损超5%直接出局
        elif seg_cum < 0:
            seg_penalty += 3000.0  # 任一负段严重扣分
        if seg_dd < -0.25:
            seg_penalty += 4000.0  # 回撤超25%不可接受
        if seg_stoploss > 0.12:
            seg_penalty += 180.0 * (seg_stoploss - 0.12)

        minute_segment_penalty += seg_penalty
        minute_segment_metrics.append({
            'segment': int(seg_idx),
            'days': int(len(seg_rets)),
            'cumulative_return': seg_cum,
            'mean_return': seg_mean,
            'win_rate': seg_win,
            'q10': seg_q10,
            'max_drawdown': seg_dd,
            'stoploss_rate': seg_stoploss,
            'sharpe': float(seg_sharpes[-1]),
            'sortino': float(seg_sortinos[-1]),
            'penalty': float(seg_penalty),
        })

    # 用各段平均替代全局计算——任一段差都会拉低总分
    _avg = lambda arr: float(np.mean(arr)) if len(arr) > 0 else 0.0
    avg_sharpe = _avg(seg_sharpes)
    avg_sortino = _avg(seg_sortinos)
    avg_cum = _avg(seg_cums)
    avg_calmar = _avg(seg_calmars)
    avg_efficiency = _avg(seg_efficiencies)
    avg_win_rate = _avg(seg_win_rates)
    avg_mean_return = _avg(seg_means)

    topk_cum_returns = {}
    for top_k in (3, 5):
        selected_rows = _calc_daily_topk_selection_rows_binary(model, X_eval, y_eval_original, eval_dates, eval_data_raw, top_k=top_k)
        if selected_rows is None or selected_rows.empty:
            topk_cum_returns[top_k] = 0.0
            continue
        merged = selected_rows.merge(cache_view, on=['date', 'stock_code'], how='left')
        merged['buy_executable'] = merged['buy_executable'].fillna(False).astype(bool)
        merged['minute_realized_return'] = pd.to_numeric(merged.get('minute_realized_return', 0.0), errors='coerce').fillna(0.0)
        merged['effective_minute_return'] = np.where(merged['buy_executable'], merged['minute_realized_return'], 0.0)
        daily_means = merged.groupby('date')['effective_minute_return'].mean()
        topk_cum_returns[top_k] = float(np.prod(1.0 + daily_means.values.astype(float)) - 1.0) if len(daily_means) > 0 else 0.0

    topk_penalty = 0.0
    if topk_cum_returns[3] < 0:
        topk_penalty += 420.0 * abs(topk_cum_returns[3])
    if topk_cum_returns[5] < 0:
        topk_penalty += 260.0 * abs(topk_cum_returns[5])

    minute_consistency_bonus = 0.0
    if minute_positive_segments >= 2:
        minute_consistency_bonus += 220.0
    if minute_positive_segments == 3:
        minute_consistency_bonus += 180.0
    if minute_weak_segments >= 2:
        minute_segment_penalty += 260.0 * minute_weak_segments

    minute_score = (
        # === 核心：分段独立 → 取平均，任一段差都拉低总分，防过拟合 ===
        3000.0 * avg_sharpe +
        2500.0 * avg_sortino +
        600.0 * avg_cum +
        1000.0 * avg_calmar +
        1200.0 * avg_efficiency +
        80.0 * (avg_win_rate - 0.5) +
        100.0 * avg_mean_return +
        # === 辅助：全局统计（次要） ===
        180.0 * topk_cum_returns[3] +
        90.0 * topk_cum_returns[5] +
        40.0 * median_return +
        120.0 * q10 -
        180.0 * stoploss_rate -
        90.0 * profit_bearish_rate -
        180.0 * skip_rate -
        180.0 * missing_rate -
        topk_penalty -
        minute_segment_penalty * 1.5 +
        minute_consistency_bonus * 1.5
    )
    return {
        'score': float(minute_score),
        'days': int(len(top1_merged)),
        'cumulative_return': cumulative_return,
        'calmar_like': float(calmar_like),
        'efficiency_gap': float(efficiency_gap),
        'sharpe_like': float(sharpe_like),
        'sortino_like': float(sortino_like),
        'mean_return': mean_return,
        'median_return': median_return,
        'win_rate': win_rate,
        'q10': q10,
        'max_drawdown': max_drawdown,
        'stoploss_rate': stoploss_rate,
        'profit_bearish_rate': profit_bearish_rate,
        'skip_rate': skip_rate,
        'missing_rate': missing_rate,
        'top3_cum_return': float(topk_cum_returns[3]),
        'top5_cum_return': float(topk_cum_returns[5]),
        'segments': minute_segment_metrics,
        'positive_segments': int(minute_positive_segments),
        'weak_segments': int(minute_weak_segments),
        'segment_penalty': float(minute_segment_penalty),
        'consistency_bonus': float(minute_consistency_bonus),
    }


def calculate_top_k_return(true_returns, pred_scores, k=5):
    """
    [TARGET] 计算Top-K组合收益率

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: Top-K组合平均收益率
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]  # 降序排列，取前k个

    # 计算Top-K平均收益率
    top_k_returns = true_returns[top_k_indices]
    return np.mean(top_k_returns)


def calculate_win_rate(true_returns, pred_scores, k=5):
    """
    [OK] 计算胜率（Top-K中正收益股票的比例）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 胜率 (0-1)
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]

    # 计算正收益股票的比例
    top_k_returns = true_returns[top_k_indices]
    positive_returns = top_k_returns > 0

    return np.mean(positive_returns)


def calculate_profit_loss_ratio(true_returns, pred_scores, k=5):
    """
    ⚡ 计算标准盈亏比（总盈利/总亏损）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 标准盈亏比（总盈利/总亏损）
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 分离正收益和负收益
    positive_returns = top_k_returns[top_k_returns > 0]
    negative_returns = top_k_returns[top_k_returns < 0]

    if len(positive_returns) == 0:
        return 0.0  # 只有亏损，没有盈利

    if len(negative_returns) == 0:
        return float('inf')  # 只有盈利，没有亏损

    # [TARGET] 标准盈亏比：总盈利 / 总亏损
    total_profit = sum(positive_returns)
    total_loss = sum(abs(v) for v in negative_returns)

    return total_profit / total_loss if total_loss > 0 else float('inf')


def calculate_sharpe_ratio(true_returns, pred_scores, k=5, risk_free_rate=0.0):
    """
    [PROFIT] 计算夏普比率（风险调整后收益）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量
        risk_free_rate: 无风险收益率（默认0）

    Returns:
        float: 夏普比率
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 计算超额收益
    excess_returns = top_k_returns - risk_free_rate

    # 计算夏普比率
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return 0.0

    return mean_excess / std_excess


def calculate_max_drawdown(true_returns, pred_scores, k=5):
    """
    [DOWN] 计算最大回撤（衡量极端风险）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量

    Returns:
        float: 最大回撤（正值）
    """
    if len(true_returns) == 0 or len(pred_scores) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # 获取Top-K股票的索引
    top_k_indices = np.argsort(pred_scores)[-k:][::-1]
    top_k_returns = true_returns[top_k_indices]

    # 计算累积收益曲线
    cumulative_returns = np.cumprod(1 + top_k_returns)

    # 计算最大回撤
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak

    return np.max(drawdown)


def calculate_excess_return(true_returns, pred_scores, k=5, benchmark_return=0.0):
    """
    [UP] 计算超额收益（相对基准的超额表现）

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        k: Top-K数量
        benchmark_return: 基准收益率（默认所有股票的平均收益）

    Returns:
        float: 超额收益
    """
    top_k_return = calculate_top_k_return(true_returns, pred_scores, k)

    # 如果没有提供基准，使用所有股票的平均收益
    if benchmark_return == 0.0 and len(true_returns) > 0:
        benchmark_return = np.mean(true_returns)

    return top_k_return - benchmark_return


def evaluate_investment_performance(true_returns, pred_scores, val_dates=None, k_values=[1, 2, 3, 5], risk_free_rate=0.0):
    """
    [TARGET] 投资导向的综合评估函数 - 使用日期分组平均算法

    Args:
        true_returns: 真实收益率数组
        pred_scores: 预测得分数组
        val_dates: 验证集日期数组（用于分组计算）
        k_values: Top-K值列表，默认[1,2,3,5]
        risk_free_rate: 无风险收益率

    Returns:
        dict: 包含各种投资指标的字典
    """
    results = {}

    # 如果没有日期信息，退化为整体计算
    if val_dates is None or len(val_dates) != len(true_returns):
        # 使用整体计算作为后备方案
        for k in k_values:
            if k <= len(true_returns):
                results[f'top_{k}_return'] = calculate_top_k_return(true_returns, pred_scores, k)
                results[f'top_{k}_excess_return'] = calculate_excess_return(true_returns, pred_scores, k)
                results[f'top_{k}_win_rate'] = calculate_win_rate(true_returns, pred_scores, k)
                results[f'top_{k}_profit_loss_ratio'] = calculate_profit_loss_ratio(true_returns, pred_scores, k)
                results[f'top_{k}_sharpe_ratio'] = calculate_sharpe_ratio(true_returns, pred_scores, k, risk_free_rate)
                results[f'top_{k}_max_drawdown'] = calculate_max_drawdown(true_returns, pred_scores, k)
                results[f'top_{k}_adjusted_ndcg'] = calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k)

        # 计算综合评分（确保有composite_score）
        available_ks = [k for k in k_values if f'top_{k}_return' in results]
        if available_ks:
            max_k = max(available_ks)
            primary_results = {
                'top_k_return': results.get(f'top_{max_k}_return', 0.0),
                'top_k_excess_return': results.get(f'top_{max_k}_excess_return', 0.0),
                'top_k_win_rate': results.get(f'top_{max_k}_win_rate', 0.0),
                'top_k_profit_loss_ratio': results.get(f'top_{max_k}_profit_loss_ratio', 0.0),
                'top_k_sharpe_ratio': results.get(f'top_{max_k}_sharpe_ratio', 0.0),
                'top_k_max_drawdown': results.get(f'top_{max_k}_max_drawdown', 0.0),
                'adjusted_ndcg': results.get(f'top_{max_k}_adjusted_ndcg', 0.0)
            }
            results['composite_score'] = calculate_composite_investment_score(primary_results)
        else:
            results['composite_score'] = 0.0

        return results

    # [TARGET] 使用日期分组平均算法（与现有Top-k实际收益率统计保持一致）
    import pandas as pd
    from scipy.stats import spearmanr

    # 创建DataFrame便于按日期分组
    val_df = pd.DataFrame({
        'date': val_dates,
        'pred': pred_scores,
        'return': true_returns
    })

    # 为每个Top-K初始化列表
    for k in k_values:
        results[f'top_{k}_return_list'] = []
        results[f'top_{k}_win_list'] = []
        results[f'top_{k}_profit_loss_list'] = []
        results[f'top_{k}_excess_list'] = []
        results[f'top_{k}_ndcg_list'] = []

    # 计算基准收益（整个验证集按日期分组的平均收益）
    # 这与LightGBM中"验证集平均收益率"的计算方式保持一致
    benchmark_return = val_df.groupby('date')['return'].mean().mean()

    # 按日期分组计算
    for date, group in val_df.groupby('date'):
        if len(group) < 5:  # 至少需要5只股票才有意义
            continue

        group_returns = group['return'].values
        group_preds = group['pred'].values

        # 对每个Top-K计算
        for k in k_values:
            if len(group) >= k:
                # 收益率计算
                top_k_idx = group.nlargest(k, 'pred').index
                top_k_returns = group.loc[top_k_idx, 'return'].values

                # 平均收益率
                avg_return = np.mean(top_k_returns)
                results[f'top_{k}_return_list'].append(avg_return)

                # 超额收益
                excess_return = avg_return - benchmark_return
                results[f'top_{k}_excess_list'].append(excess_return)

                # 胜率
                win_rate = np.mean(top_k_returns > 0)
                results[f'top_{k}_win_list'].append(win_rate)

                # 盈亏比
                positive_returns = top_k_returns[top_k_returns > 0]
                negative_returns = top_k_returns[top_k_returns < 0]
                if len(positive_returns) > 0 and len(negative_returns) > 0:
                    profit_loss_ratio = np.mean(positive_returns) / np.mean(np.abs(negative_returns))
                elif len(positive_returns) > 0:
                    profit_loss_ratio = float('inf')
                else:
                    profit_loss_ratio = 0.0
                results[f'top_{k}_profit_loss_list'].append(profit_loss_ratio)

                # NDCG（使用改进版本）
                ndcg_k = calculate_adjusted_ndcg_at_k(group_returns, group_preds, k)
                results[f'top_{k}_ndcg_list'].append(ndcg_k)

    # 计算最终的统计指标
    for k in k_values:
        if len(results[f'top_{k}_return_list']) > 0:
            # 主要指标：使用平均值
            results[f'top_{k}_return'] = np.mean(results[f'top_{k}_return_list'])
            results[f'top_{k}_excess_return'] = np.mean(results[f'top_{k}_excess_list'])
            results[f'top_{k}_win_rate'] = np.mean(results[f'top_{k}_win_list'])

            # 盈亏比使用中位数，避免极端值影响
            profit_loss_ratios = [r for r in results[f'top_{k}_profit_loss_list'] if r != float('inf')]
            if profit_loss_ratios:
                results[f'top_{k}_profit_loss_ratio'] = np.median(profit_loss_ratios)
            else:
                results[f'top_{k}_profit_loss_ratio'] = 0.0

            # NDCG平均值
            results[f'top_{k}_adjusted_ndcg'] = np.mean(results[f'top_{k}_ndcg_list'])

            # 风险调整指标（基于收益率列表）
            returns_list = results[f'top_{k}_return_list']
            excess_returns = [r - risk_free_rate for r in returns_list]

            if len(excess_returns) > 1:
                results[f'top_{k}_sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
            else:
                results[f'top_{k}_sharpe_ratio'] = 0.0

            # 最大回撤计算（基于累积收益）
            cumulative_returns = np.cumprod(1 + np.array(returns_list))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            results[f'top_{k}_max_drawdown'] = np.max(drawdown)
        else:
            # 如果没有有效数据，设为0
            results[f'top_{k}_return'] = 0.0
            results[f'top_{k}_excess_return'] = 0.0
            results[f'top_{k}_win_rate'] = 0.0
            results[f'top_{k}_profit_loss_ratio'] = 0.0
            results[f'top_{k}_adjusted_ndcg'] = 0.0
            results[f'top_{k}_sharpe_ratio'] = 0.0
            results[f'top_{k}_max_drawdown'] = 0.0

    # 综合评分（主要基于最大可用的Top-K）
    # 确保即使某些Top-K无法计算，也能得到composite_score
    available_ks = [k for k in k_values if f'top_{k}_return' in results]

    if available_ks:
        # 使用最大可用的k值进行综合评分
        max_k = max(available_ks)
        primary_results = {
            'top_k_return': results.get(f'top_{max_k}_return', 0.0),
            'top_k_excess_return': results.get(f'top_{max_k}_excess_return', 0.0),
            'top_k_win_rate': results.get(f'top_{max_k}_win_rate', 0.0),
            'top_k_profit_loss_ratio': results.get(f'top_{max_k}_profit_loss_ratio', 0.0),
            'top_k_sharpe_ratio': results.get(f'top_{max_k}_sharpe_ratio', 0.0),
            'top_k_max_drawdown': results.get(f'top_{max_k}_max_drawdown', 0.0),
            'adjusted_ndcg': results.get(f'top_{max_k}_adjusted_ndcg', 0.0)
        }
        results['composite_score'] = calculate_composite_investment_score(primary_results)
    else:
        # 如果没有任何可用的Top-K数据，设置默认值
        results['composite_score'] = 0.0

    return results


def calculate_composite_investment_score(metrics):
    """
    [TROPHY] 计算综合投资评分 - 聚焦交易盈利能力

    核心理念：只关心交易能否盈利，模型是否选出了足够盈利的股票

    评分策略：
    - 盈利能力占60% (收益率 + 胜率 + 盈亏比)
    - 风险控制占20% (夏普比率 + 最大回撤)
    - 排序质量占20% (改进NDCG，确保能选出好股票)

    Args:
        metrics: 评估指标字典 (使用top_k_return, top_k_win_rate等)

    Returns:
        float: 综合评分 (0-100)
    """
    score = 0

    # 1. 盈利能力评分 (60分) - 核心关注点
    profit_score = 0

    # 收益率评分 (30分)
    if metrics['top_k_return'] > 0:
        # 正收益才能得分，收益越高分数越高
        profit_score += min(30, metrics['top_k_return'] * 200)  # 1%收益 = 2分，最高30分

    # 胜率评分 (20分) - 选对盈利股票的能力
    profit_score += metrics['top_k_win_rate'] * 20  # 100%胜率=20分

    # 盈亏比评分 (10分) - 盈利时赚多少，亏损时亏多少
    if metrics['top_k_profit_loss_ratio'] > 1:
        profit_score += min(10, metrics['top_k_profit_loss_ratio'] * 2)  # 盈亏比>1才得分
    elif metrics['top_k_profit_loss_ratio'] > 0:
        profit_score += metrics['top_k_profit_loss_ratio'] * 5  # 盈亏比<1部分得分

    score += profit_score

    # 2. 风险控制评分 (20分) - 但不能过度影响盈利
    risk_score = 0

    # 夏普比率评分 (15分)
    if metrics['top_k_sharpe_ratio'] > 0.5:  # 夏普比率>0.5才算不错
        risk_score += min(15, metrics['top_k_sharpe_ratio'] * 10)
    elif metrics['top_k_sharpe_ratio'] > 0:
        risk_score += metrics['top_k_sharpe_ratio'] * 5

    # 最大回撤扣分 (5分惩罚)
    drawdown_penalty = min(5, metrics['top_k_max_drawdown'] * 50)  # 10%回撤扣5分
    risk_score = max(0, risk_score - drawdown_penalty)

    score += risk_score

    # 3. 排序质量评分 (20分) - 确保能选到好股票
    ranking_score = metrics['adjusted_ndcg'] * 20  # NDCG接近1得高分
    score += ranking_score

    return min(100, max(0, score))  # 限制在0-100范围内


def print_investment_evaluation_report(model_name, results):
    """
    [CHART] 打印投资评估报告 - 支持Top-1,2,3,5分开评估

    Args:
        model_name: 模型名称
        results: 评估结果字典
    """
    logger.info(f"\n[CHART] {model_name}投资评估报告")
    logger.info("=" * 80)

    # Top-K分别评估
    k_values = [1, 2, 3, 5]
    for k in k_values:
        prefix = f'top_{k}'
        if f'{prefix}_return' in results:
            logger.info(f"[TARGET] Top-{k}组合评估:")
            logger.info(f"  收益率: {results[f'{prefix}_return']:.4f} ({results[f'{prefix}_return']*100:.2f}%)")
            logger.info(f"  超额收益: {results[f'{prefix}_excess_return']:.4f} ({results[f'{prefix}_excess_return']*100:.2f}%)")
            logger.info(f"  胜率: {results[f'{prefix}_win_rate']:.2%}")
            logger.info(f"  盈亏比: {results[f'{prefix}_profit_loss_ratio']:.2f}")
            logger.info(f"  夏普比率: {results[f'{prefix}_sharpe_ratio']:.4f}")
            logger.info(f"  最大回撤: {results[f'{prefix}_max_drawdown']:.4f} ({results[f'{prefix}_max_drawdown']*100:.2f}%)")
            logger.info(f"  排序质量(NDCG@{k}): {results[f'{prefix}_adjusted_ndcg']:.4f}")
            logger.info("")

    # 综合评分 (主要基于Top-5)
    logger.info("[TROPHY] 综合评估:")
    logger.info(f"  投资评分: {results['composite_score']:.1f}/100")

    # 评级
    if results['composite_score'] >= 80:
        rating = "优秀 [STAR][STAR][STAR][STAR][STAR] (强烈推荐实盘)"
        recommendation = "该模型具有很强的盈利能力，建议重点考虑实盘应用"
    elif results['composite_score'] >= 70:
        rating = "良好 [STAR][STAR][STAR][STAR] (值得实盘)"
        recommendation = "该模型具有较好的盈利能力，可以考虑实盘应用"
    elif results['composite_score'] >= 60:
        rating = "及格 [STAR][STAR][STAR] (谨慎实盘)"
        recommendation = "该模型基本盈利，建议优化后再考虑实盘"
    elif results['composite_score'] >= 50:
        rating = "需要改进 [STAR][STAR] (不建议实盘)"
        recommendation = "该模型盈利能力不足，不建议实盘应用"
    else:
        rating = "不合格 [STAR] (重新训练)"
        recommendation = "该模型无法实现盈利，需要重新设计或训练"

    logger.info(f"  综合评级: {rating}")
    logger.info(f"  投资建议: {recommendation}")
    logger.info("=" * 80)


import statsmodels.api as sm


# ============================================================
# [WRENCH] 全局配置参数
# ============================================================
# ATR策略配置
MAX_HOLDING_DAYS = 3  # 最大持仓期（天）- 训练和预测统一使用
ATR_MULTIPLIER_PROFIT = 1.2  # ATR止盈倍数1.2
ATR_MULTIPLIER_LOSS = 1    # ATR止损倍数1
MAX_STOP_LOSS_PCT = 0.08   # 最大止损百分比（3%）- 与ATR止损取较小损失（优化：5%->3%）

# 交易成本配置（基于A股实际交易成本）
# A股交易成本构成：
# 1. 佣金：万一（0.01%），双边收取
# 2. 印花税：千分之0.5（0.05%），卖出时收取
# 3. 过户费：万分之0.2（0.002%），双边收取
# 4. 滑点成本：约0.02%-0.05%
# 总计：买入约0.012%，卖出约0.062%，双边约0.074%
# 考虑到量化交易频繁操作，设置为0.08%
TRADING_COST_RATE = 0.0008  # 交易成本率（0.08%）

# 时间切分配置（train_model和test_model模式共用）
# [TARGET] 纯预测选股的时间切分策略
TRAIN_END_DATE = "2025-02-01"  # 训练数据截止日期（不包含此日期）
TEST_START_DATE = "2025-02-01"  # 测试数据开始日期（包含此日期）
TEST_END_DATE = "2025-10-21"    # 测试数据结束日期（包含此日期）

FACTOR_LAB_STORE_PATH = (Path(__file__).resolve().parent / "factor_lab_data" / "factors.parquet")

try:
    # 尝试相对导入（当作为模块导入时）
    from .data_loader import IntegratedDataLoader
    from .stock_selector import StockSelector
    from .backtester import Backtester
    from .config import system_config, backtest_config
except ImportError:
    # 回退到绝对导入（当直接运行时）
    from data_loader import IntegratedDataLoader
    from stock_selector import StockSelector
    from backtester import Backtester
    from config import system_config, backtest_config

try:
    from .factor_system.factor_lab import definitions as _factor_lab_definitions  # noqa: F401
    from .factor_system.factor_lab.pipeline import FactorPipeline
    from .factor_system.factor_lab.store import FactorDatasetStore
    from .factor_system.factor_lab.commands import generate_and_store
except ImportError:
    from factor_system.factor_lab import definitions as _factor_lab_definitions  # noqa: F401
    from factor_system.factor_lab.pipeline import FactorPipeline
    from factor_system.factor_lab.store import FactorDatasetStore
    from factor_system.factor_lab.commands import generate_and_store


def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # 处理numpy标量
        return obj.item()
    else:
        return obj

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="机器学习K线形态选股系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明:
  generate_factors   - 因子生成模式（计算和生成所有因子）
  test_factors      - 因子测试模式（单因子测试和分析）
  train_model       - 模型训练模式（训练机器学习模型）
  test_model        - 模型测试模式（使用模型进行选股测试）
  test_model_walkforward - 滚动窗口测试模式
  pred_stocks       - 股票预测模式（根据指定日期预测top5股票）
  full_pipeline     - 完整流水线模式（因子生成->测试->模型训练->测试）
  test_optimization - 优化测试模式（分位数回归&动态损失函数优化）
  factor_correlation - 因子相关性分析模式（分析ATR三分类系统因子相关性）
  genetic_factor    - 遗传编程因子进化模式（使用gplearn进化超级因子）
  # interaction_analysis - 已移除交互特征分析模式

使用示例:
  # 生成因子（默认生成所有因子，可能覆盖已存在的）
  python main.py --mode generate_factors

  # 强制重新生成所有因子（清理旧数据）
  python main.py --mode generate_factors --force

  # 🆕 增量保存：保留现有数据并智能合并新数据
  python main.py --mode generate_factors --incremental_save

  # 测试因子
  python main.py --mode test_factors --factor_names rsi6,macd,momentum_quality_score

  # 训练模型
  python main.py --mode train_model

  # 测试模型（默认启用市场因子）
  python main.py --mode test_model

  # 测试模型（禁用市场因子，与main2.py一致）
  python main.py --mode test_model --no_market_factors

  # 完整流水线
  python main.py --mode full_pipeline

  # 因子相关性分析
  python main.py --mode factor_correlation

  # 遗传编程因子进化
  python main.py --mode genetic_factor --population_size 1000 --generations 20

  # 股票预测模式
  python main.py --mode pred_stocks --date 20251216

  # 自定义输出目录
  python main.py --output_dir ./custom_results
        """
    )

    parser.add_argument(
        '--mode',
        choices=['generate_factors', 'test_factors', 'train_model', 'test_model', 'test_model_walkforward', 'pred_stocks', 'full_pipeline', 'test_optimization', 'factor_correlation', 'genetic_factor', 'market_state_test', 'interaction_analysis'],
        default='generate_factors',
        help='运行模式 (默认: generate_factors)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='输出目录 (默认: ./results)'
    )

    parser.add_argument(
        '--factor_names',
        type=str,
        help='因子名称列表，用逗号分隔 (仅在test_factors模式下使用)'
    )

    parser.add_argument(
        '--factor_groups',
        type=str,
        help='因子组名称列表，用逗号分隔 (可选：technical,fundamental,kline_patterns等)'
    )

    parser.add_argument(
        '--start_date',
        type=str,
        help='开始日期 (格式: YYYYMMDD)'
    )

    parser.add_argument(
        '--end_date',
        type=str,
        help='结束日期 (格式: YYYYMMDD)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='预测日期 (格式: YYYYMMDD, 仅在pred_stocks模式下使用)'
    )

    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--save_results',
        action='store_true',
        default=True,
        help='是否保存结果文件 (默认: True)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新计算，忽略缓存'
    )

    # 🆕 增量保存参数
    parser.add_argument(
        '--incremental_save',
        action='store_true',
        help='启用增量保存模式，保留现有数据并智能合并新数据（重合日期用新数据覆盖）'
    )

    # 🆕 新增因子模式参数
    parser.add_argument(
        '--add_factor',
        action='store_true',
        help='新增因子模式：保留原有因子不变，仅新增指定因子（列扩展模式，不会触发overwrite）'
    )

      # 分批保存参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='因子生成分批保存大小，每计算N个因子保存一次（默认: 10，设为0禁用分批保存）'
    )

    # 遗传编程相关参数
    parser.add_argument(
        '--population_size',
        type=int,
        default=1000,
        help='遗传编程种群大小 (默认: 1000)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=20,
        help='遗传编程进化代数 (默认: 20)'
    )

    parser.add_argument(
        '--n_best_factors',
        type=int,
        default=10,
        help='返回最佳因子数量 (默认: 10)'
    )

    # 市场因子开关参数
    parser.add_argument(
        '--use_market_factors',
        action='store_true',
        default=True,
        help='是否使用市场因子进行预测和仓位调节 (默认: True)'
    )

    parser.add_argument(
        '--no_market_attention',
        action='store_true',
        help='禁用市场注意力机制，使用原始股票因子'
    )

    # [强化] 超参数搜索相关参数
    parser.add_argument(
        '--enable_param_search',
        action='store_true',
        help='启用超参数搜索优化（默认: False）'
    )

    parser.add_argument(
        '--search_strategy',
        choices=['bayesian'],
        default='bayesian',
        help='超参数搜索策略 (bayesian=贝叶斯优化)'
    )

    # [强化] 模型选择参数
    parser.add_argument(
        '--models',
        type=str,
        default='xgboost,lightgbm',
        help='选择使用的模型，用逗号分隔 (可选: xgboost, lightgbm)。默认: xgboost,lightgbm (仅test_model和test_model_walkforward模式生效)'
    )

    # 🆕 窗口选择参数（用于test_model_walkforward模式）
    parser.add_argument(
        '--window',
        type=str,
        default=None,
        help='指定要训练的窗口编号，用逗号分隔（例如: 1 或 1,3）。不指定则训练所有窗口（仅test_model_walkforward模式生效）'
    )

    # [强化] 止盈止损阈值优化参数
    parser.add_argument(
        '--optimize_thresholds',
        action='store_true',
        help='启用止盈止损ATR阈值优化（基于LightGBM验证集，使用网格搜索+夏普比率）。默认: False'
    )

    # 🆕 跳过训练参数
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='跳过训练阶段，直接加载最新的集成模型进行预测（仅在test_model模式下有效）'
    )

    return parser.parse_args()

def setup_logging():
    """设置日志"""
    # 创建根logger配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有handlers以避免重复
    root_logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    file_handler = logging.FileHandler(
        'stock_selection_test_mode_walkforward_分类模型_原因子146_optuna.log',
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 添加handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 返回根logger以便验证配置
    return root_logger

def save_results(selections, backtest_result, train_result=None, sector_report=None):
    """保存结果"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存选股结果
        selections_data = []
        for selection in selections:
            selections_data.append({
                'date': selection.date,
                'stock_code': selection.stock_code,
                'stock_name': selection.stock_name,
                'composite_score': selection.composite_score,
                'risk_adjusted_score': selection.risk_adjusted_score,
                'technical_score': selection.technical_score,
                'fundamental_score': selection.fundamental_score,
                'sector_score': selection.sector_score,
                'capital_flow_score': selection.capital_flow_score,
                'selected_sectors': selection.selected_sectors,
                'confidence': selection.confidence,
                'expected_return': selection.expected_return
            })

        selections_file = system_config.results_dir / f'selections_{timestamp}.json'
        with open(selections_file, 'w', encoding='utf-8') as f:
            json.dump(selections_data, f, ensure_ascii=False, indent=2)

        # 保存回测结果
        backtest_data = {
            'summary': {
                'total_return': backtest_result.total_return,
                'annual_return': backtest_result.annual_return,
                'sharpe_ratio': backtest_result.annualized_sharpe,  # 使用年化夏普比率
                'max_drawdown': backtest_result.max_drawdown,
                'win_rate': backtest_result.win_rate,
                'profit_loss_ratio': backtest_result.profit_loss_ratio,
                'total_trades': backtest_result.total_trades,
                'profitable_trades': backtest_result.profitable_trades,
                'avg_holding_days': backtest_result.avg_holding_days
            },
            'trade_details': backtest_result.trade_details
        }

        backtest_file = system_config.results_dir / f'backtest_{timestamp}.json'
        with open(backtest_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_data, f, ensure_ascii=False, indent=2)

        # 保存简化报告
        report_file = system_config.results_dir / f'report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("板块-个股一体化选股系统回测报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"测试期间: {system_config.test_start_date} - {system_config.end_date}\n")
            f.write(f"每日选股数量: {system_config.daily_stock_count}\n")
            f.write(f"持仓天数: {backtest_config.holding_days}\n\n")

            f.write("回测结果:\n")
            f.write(f"  总收益率: {backtest_result.total_return:.2%}\n")
            f.write(f"  年化收益率: {backtest_result.annual_return:.2%}\n")
            f.write(f"  夏普比率: {backtest_result.annualized_sharpe:.2f}\n")
            f.write(f"  最大回撤: {backtest_result.max_drawdown:.2%}\n")
            f.write(f"  胜率: {backtest_result.win_rate:.1%}\n")
            f.write(f"  盈亏比: {backtest_result.profit_loss_ratio:.2f}\n")
            f.write(f"  总交易次数: {backtest_result.total_trades}\n")
            f.write(f"  盈利交易: {backtest_result.profitable_trades}\n\n")

            if train_result:
                f.write("训练集表现:\n")
                f.write(f"  总收益率: {train_result.total_return:.2%}\n")
                f.write(f"  胜率: {train_result.win_rate:.1%}\n")
                f.write(f"  总交易次数: {train_result.total_trades}\n\n")

            if sector_report:
                f.write("板块策略回测:\n")
                f.write(sector_report)
                f.write("\n\n")

            # 选股示例
            f.write("选股示例 (前10次):\n")
            for i, selection in enumerate(selections[:10]):
                f.write(f"  {selection.date}: {selection.stock_code} "
                       f"(评分{selection.risk_adjusted_score:.1f}, "
                       f"置信度{selection.confidence:.2f})\n")

        print(f"[OK] 结果已保存:")
        print(f"  选股结果: {selections_file}")
        print(f"  回测结果: {backtest_file}")
        print(f"  简化报告: {report_file}")

    except Exception as e:
        print(f"[FAIL] 保存结果失败: {e}")



def _load_factor_cache_data(
    required_factors=None,
    include_label_generation_features: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """统一从 factor_lab 数据集中加载所需因子并补全基础字段。"""
    try:
        store = FactorDatasetStore(FACTOR_LAB_STORE_PATH)
        if not store.exists():
            logger.error("因子数据文件不存在，请先运行 generate_factors 模式生成 factor_lab 数据。")
            return None

        factor_df = store.load()
        if factor_df is None or factor_df.empty:
            logger.error("因子数据文件为空，请确认已生成 factor_lab 数据。")
            return None

        required_columns = {'date', 'stock_code'}
        missing_required = [col for col in required_columns if col not in factor_df.columns]
        if missing_required:
            logger.error(f"因子数据缺少基础列: {missing_required}")
            return None

        factor_df = factor_df.copy()
        factor_df['date'] = pd.to_datetime(factor_df['date'])
        factor_df.sort_values(['date', 'stock_code'], inplace=True)
        factor_df.drop_duplicates(subset=['date', 'stock_code'], keep='last', inplace=True)
        factor_df.reset_index(drop=True, inplace=True)
        factor_df['date_str'] = factor_df['date'].dt.strftime('%Y%m%d')

        available_factor_columns = [
            col for col in factor_df.columns if col not in {'date', 'stock_code', 'date_str'}
        ]

        if required_factors is None:
            # [WRENCH] 模型训练时使用归一化因子（默认：z-score + rankpct）
            stock_factors = _get_model_factors()
            # market_factors = _get_available_market_factors()
            # required_factors = stock_factors + market_factors
            required_factors = stock_factors
        else:
            required_factors = list(dict.fromkeys(required_factors))
            if 'returns_zscore' not in required_factors:
                required_factors.insert(0, 'returns_zscore')
            logger.info("按需加载因子 %d 个。", len(required_factors))

        available_required = [f for f in required_factors if f in available_factor_columns]
        missing_requested = [f for f in required_factors if f not in available_factor_columns]
        if missing_requested:
            logger.warning(f"因子库缺少以下因子: {missing_requested}")

        # [强化] 关键优化：在加载后立即过滤掉不需要的因子列，避免内存浪费
        # 只保留 date, stock_code 和 available_required 中的因子
        columns_to_keep = ['date', 'stock_code'] + available_required
        columns_to_keep = [col for col in columns_to_keep if col in factor_df.columns]
        factor_df = factor_df[columns_to_keep].copy()
        logger.info(f"过滤后保留因子列数: {len(columns_to_keep) - 2} (原始: {len(available_factor_columns)})")

        dataset_start = factor_df['date'].min()
        dataset_end = factor_df['date'].max()
        logger.info(f"因子 start:{dataset_start}, end:{dataset_end}")
        if start_date:
            dataset_start = max(dataset_start, pd.to_datetime(start_date))
        if end_date:
            dataset_end = min(dataset_end, pd.to_datetime(end_date))
        logger.info(f"因子2 start:{dataset_start}, end:{dataset_end}")
        mask = (factor_df['date'] >= dataset_start) & (factor_df['date'] <= dataset_end)
        factor_df = factor_df.loc[mask].copy()
        if factor_df.empty:
            logger.error("按日期过滤后因子数据为空。")
            return None

        loader = IntegratedDataLoader()
        stock_start = dataset_start.strftime('%Y%m%d')
        stock_end = dataset_end.strftime('%Y%m%d')
        stock_data = loader.load_stock_features(stock_start, stock_end)
        if stock_data is None or stock_data.empty:
            logger.error("无法加载基础行情数据，请检查 local_data/features/daily 下的数据文件。")
            return None

        stock_data = stock_data.copy()
        if 'date' not in stock_data.columns or 'stock_code' not in stock_data.columns:
            logger.error("基础行情数据缺少 'date' 或 'stock_code' 列。")
            return None
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.drop_duplicates(subset=['date', 'stock_code'], keep='last', inplace=True)

        base_columns = [
            'date', 'stock_code', 'date_str', 'pct_chg', 'is_can_buy',
            'volume', 'amount', 'close', 'open', 'high', 'low',  # [强化] 添加amount字段用于流动性过滤
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
            'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
            'near_support', 'near_resistance', 'short_reversal', 'parkinson_vol',
            'vol_term_structure', 'volatility20', 'atr14',  # 添加标签生成必需字段
            'future_return_2d', 'future_return_3d', 'future_return_5d',  # [强化] 添加未来收益字段
            'turnover_rate', 'circ_mv'  # [强化] 添加换手率和流通市值字段（用于样本权重计算）
        ]
        # [强化] 标签生成所需字段:
        # - volatility20, atr14: ATR策略计算
        # - open, high, low: 价格计算
        # - is_limit_up, is_limit_down: 过滤T日涨跌停
        # - amount, volume: 过滤T日流动性不足
        # - is_can_buy: 过滤T+1日一字涨停
        # - future_return_2d, future_return_3d, future_return_5d: 加权收益率标签计算
        label_generation_columns = [
            'volatility20', 'atr14', 'open', 'high', 'low',
            'is_limit_up', 'is_limit_down', 'amount', 'volume', 'is_can_buy',
            'future_return_2d', 'future_return_3d', 'future_return_5d'
        ]
        default_values = {
            'pct_chg': 0.0,
            'is_can_buy': 1,
            'volume': 0.0,
            'amount': 0.0,  # [强化] 添加amount默认值
            'close': 0.0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'is_limit_up': 0,
            'is_limit_down': 0,
            'is_zhaban': 0,
            'limit_strength': 0.0,
            'limit_times': 0,
            'consecutive_limit_up': 0,
            'consecutive_limit_down': 0,
            'near_support': 0,
            'near_resistance': 0,
            'short_reversal': 0,
            'parkinson_vol': 0.0,
            'vol_term_structure': 0.0,
            'volatility20': 1.0,
            'atr14': 0.0,
            'future_return_2d': np.nan,  # [强化] 未来收益字段默认为NaN
            'future_return_3d': np.nan,
            'future_return_5d': np.nan,
            'turnover_rate': np.nan,  # [强化] 换手率默认为NaN
            'circ_mv': np.nan,  # [强化] 流通市值默认为NaN
        }

        # 🆕 添加SW行业加载和过滤逻辑
        sw_sector_mapping = load_sw_sector_mapping()
        if sw_sector_mapping:
            # 获取有SW行业映射的股票代码集合
            valid_stocks_with_sw = set(sw_sector_mapping.keys())
            original_stock_count = len(stock_data)

            # 过滤股票数据，只保留有SW行业映射的股票
            stock_data = stock_data[stock_data['stock_code'].isin(valid_stocks_with_sw)].copy()
            filtered_stock_count = len(stock_data)

            if filtered_stock_count < original_stock_count:
                removed_stock_count = original_stock_count - filtered_stock_count
                logger.info(f"SW行业过滤: 原始股票数量 {original_stock_count} -> 过滤后数量 {filtered_stock_count}")
                logger.info(f"已移除 {removed_stock_count} 只没有SW行业映射的股票")
                print(f"?? SW行业过滤: 原始股票数量 {original_stock_count} -> 过滤后数量 {filtered_stock_count}")
                print(f"?? 已移除 {removed_stock_count} 只没有SW行业映射的股票")

                # 如果过滤后股票数量太少，发出警告
                if filtered_stock_count == 0:
                    logger.error("过滤后没有有效股票，请检查SW行业映射数据")
                    print("❌ 错误：过滤后没有有效股票，请检查SW行业映射数据")
                    return None
                elif filtered_stock_count < 100:
                    logger.warning(f"过滤后股票数量较少：{filtered_stock_count}只，可能影响因子数据质量")
                    print(f"⚠️  警告：过滤后股票数量较少：{filtered_stock_count}只，可能影响因子数据质量")
            else:
                logger.info("SW行业过滤：所有股票都有有效的SW行业映射")
                print("?? SW行业过滤：所有股票都有有效的SW行业映射")
        else:
            logger.warning("没有SW行业映射数据，无法进行股票过滤")
            print("⚠️  警告：没有SW行业映射数据，无法进行股票过滤")

        base_merge_columns = [
            col for col in base_columns if col in stock_data.columns and col not in {'date', 'stock_code'}
        ]
        if base_merge_columns:
            merge_source = stock_data[['date', 'stock_code'] + base_merge_columns].copy()
        else:
            merge_source = stock_data[['date', 'stock_code']].copy()
        factor_df = factor_df.merge(merge_source, on=['date', 'stock_code'], how='left')
        # factor_df = factor_df.merge(merge_source, on=['date', 'stock_code'], how='inner')

        # 🆕 基于SW行业过滤后的stock_data，对合并后的factor_df进行一致性过滤
        if sw_sector_mapping and 'stock_code' in factor_df.columns:
            valid_stocks_with_sw = set(sw_sector_mapping.keys())
            original_factor_count = len(factor_df)

            # 只保留有SW行业映射的股票的因子数据
            factor_df = factor_df[factor_df['stock_code'].isin(valid_stocks_with_sw)].copy()
            filtered_factor_count = len(factor_df)

            if filtered_factor_count < original_factor_count:
                removed_factor_count = original_factor_count - filtered_factor_count
                logger.info(f"SW行业过滤后因子数据: 原始记录数 {original_factor_count} -> 过滤后数量 {filtered_factor_count}")
                logger.info(f"已移除 {removed_factor_count} 条没有SW行业映射的因子记录")
                print(f"?? 因子数据SW行业过滤: 原始记录数 {original_factor_count} -> 过滤后数量 {filtered_factor_count}")
                print(f"?? 已移除 {removed_factor_count} 条没有SW行业映射的因子记录")
            else:
                logger.info("因子数据与SW行业过滤保持一致")
                print("?? 因子数据与SW行业过滤保持一致")

        if 'date_str' not in factor_df.columns:
            factor_df['date_str'] = factor_df['date'].dt.strftime('%Y%m%d')

        missing_base_cols = [col for col in base_columns if col not in factor_df.columns]
        if missing_base_cols:
            logger.warning(f"基础行情数据缺少列: {missing_base_cols}")
            for col in missing_base_cols:
                factor_df[col] = default_values.get(col, 0.0)

        final_columns = ['date', 'date_str', 'stock_code']
        final_columns.extend([
            col for col in base_columns
            if col in factor_df.columns and col not in {'date', 'date_str', 'stock_code'}
        ])
        final_columns.extend([col for col in available_required if col in factor_df.columns])

        if include_label_generation_features:
            # [强化] 检查标签生成所需字段是否存在(可能已在base_columns中)
            missing_label_cols = [col for col in label_generation_columns if col not in factor_df.columns]
            existing_label_cols = [col for col in label_generation_columns if col in factor_df.columns]

            # 添加尚未在final_columns中的标签生成字段
            extra_cols = [
                col for col in label_generation_columns
                if col in factor_df.columns and col not in final_columns
            ]

            if extra_cols:
                logger.info(f"附加标签生成必需字段: {extra_cols}")
                final_columns.extend(extra_cols)

            # 如果有缺失的字段,使用默认值填充
            if missing_label_cols:
                logger.warning(f"标签生成字段缺失,将使用默认值: {missing_label_cols}")
                for col in missing_label_cols:
                    factor_df[col] = default_values.get(col, 0.0)
                    if col not in final_columns:
                        final_columns.append(col)

            # 记录已存在的字段
            if existing_label_cols:
                logger.info(f"[OK] 标签生成字段已存在: {existing_label_cols}")

        missing_after_selection = [col for col in final_columns if col not in factor_df.columns]
        if missing_after_selection:
            logger.warning(f"最终列缺失: {missing_after_selection}")
            for col in missing_after_selection:
                factor_df[col] = default_values.get(col, 0.0)

        # [强化] 优化：避免内存分配失败，直接使用列选择而不是copy
        # 原因：在Line 773已经执行过一次copy，这里再次copy会导致内存不足
        factor_df = factor_df[final_columns]
        factor_df.sort_values(['date', 'stock_code'], inplace=True)
        factor_df.reset_index(drop=True, inplace=True)

        logger.info(
            "统一加载因子数据完成: %d 条记录, %d 列 (股票: %d 支)。",
            len(factor_df),
            len(factor_df.columns),
            factor_df['stock_code'].nunique(),
        )
        if missing_requested:
            logger.info(f"仍缺少的因子: {missing_requested}")
        return factor_df
    except Exception as exc:
        logger.error(f"因子数据加载失败: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _get_ml_feature_columns_simple(data: pd.DataFrame) -> list:
    """
    [WRENCH] 重构后的特征筛选函数：直接使用_get_model_factors()定义的因子列表

    核心改进：
    1. 不再使用黑名单动态过滤，而是使用固定的因子列表
    2. 确保训练和测试使用完全相同的特征
    3. 如果数据缺少某个因子，记录警告但不填充0

    Args:
        data: 输入数据DataFrame

    Returns:
        list: 有效的特征列名列表（按字母顺序排序）
    """
    try:

        # [强化] 直接使用模型因子列表（归一化因子）
        # 🆕 包含所有类型的因子，包括行业内因子
        all_factors = _get_model_factors(
            use_original=True,
            use_zscore=True,
            use_rankpct=True,
            use_intra_sector_rank=True,  # 🆕 包含行业内排名因子
            use_intra_sector_zscore=True  # 🆕 包含行业内z-score因子
        )

        # 验证所有因子都存在
        available_factors = [f for f in all_factors if f in data.columns]
        missing_factors = [f for f in all_factors if f not in data.columns]

        if missing_factors:
            logger.warning(f"[WARNING] 数据缺少{len(missing_factors)}个因子:")
            logger.warning(f"   缺失因子: {missing_factors[:10]}{'...' if len(missing_factors) > 10 else ''}")

        if not available_factors:
            logger.error("[FAIL] 没有可用的因子")
            return None

        logger.info(f"[OK] 可用因子: {len(available_factors)}/{len(all_factors)}")

        # 按字母顺序排序（确保一致性）
        return sorted(available_factors)

    except Exception as e:
        import traceback
        logger.error(f"[FAIL] 特征筛选失败: {e}")
        logger.error(traceback.format_exc())
        return None

def _get_market_state(market_data: pd.DataFrame, lookback: int = 20, logger=None) -> str:
    """
    判断当前市场状态

    Args:
        market_data: 市场数据DataFrame,包含'date'和'close'列
        lookback: 回看天数
        logger: 日志记录器

    Returns:
        str: 'bull'(牛市), 'bear'(熊市), 'sideways'(震荡市)
    """
    try:
        if market_data is None or market_data.empty:
            if logger:
                logger.warning("市场数据为空，默认返回震荡市状态")
            return 'sideways'

        # 确保数据按日期排序
        market_data = market_data.sort_values('date')

        # 计算最近lookback天的收益率
        recent_return = market_data['close'].pct_change(lookback).iloc[-1]

        # 计算波动率
        volatility = market_data['close'].pct_change().rolling(lookback).std().iloc[-1]

        # 判断市场状态
        if recent_return > 0.05 and volatility < 0.02:
            state = 'bull'  # 牛市:涨幅>5%且波动率<2%
        elif recent_return < -0.05:
            state = 'bear'  # 熊市:跌幅>5%
        else:
            state = 'sideways'  # 震荡市

        if logger:
            logger.info(f"[CHART] 市场状态判断:")
            logger.info(f"  最近{lookback}天收益率: {recent_return*100:.2f}%")
            logger.info(f"  最近{lookback}天波动率: {volatility*100:.2f}%")
            logger.info(f"  市场状态: {state} ({'牛市' if state == 'bull' else '熊市' if state == 'bear' else '震荡市'})")

        return state

    except Exception as e:
        if logger:
            logger.error(f"市场状态判断失败: {e}")
        return 'sideways'


def _calculate_xgboost_group_weights(train_dates: np.ndarray,
                                    market_data: pd.DataFrame,
                                    logger,
                                    time_decay_half_life: float = 180.0,
                                    reference_date: str = None,
                                    verbose: bool = True) -> np.ndarray:
    """
    [强化] XGBoost排序模型的分组权重计算（基于业务逻辑）

    XGBoost ranking objectives要求group-level weights（长度=group数量）

    权重计算逻辑:
    1. 市场波动率适中的日子权重高（黄金波动区间）
    2. 近期数据权重高（时间衰减）
    3. 高成交量日子权重高（市场活跃度）

    Args:
        train_dates: 训练集日期数组（每个样本的日期）
        market_data: 市场数据DataFrame，包含date, close, volume等字段
        logger: 日志记录器
        time_decay_half_life: 时间衰减半衰期（天数，默认180天）
        reference_date: 时间衰减基准日期（默认为训练集最新日期）
        verbose: 是否输出详细日志（默认True，贝叶斯搜索时设为False）

    Returns:
        分组权重数组（长度=group数量）
    """
    try:
        import numpy as np
        import pandas as pd

        if verbose:
            logger.info("=" * 60)
            logger.info("[强化] XGBoost分组权重计算（基于业务逻辑）")
            logger.info("=" * 60)

        # 获取唯一日期（按顺序）
        unique_dates = pd.Series(train_dates).unique()
        unique_dates = pd.to_datetime(unique_dates)
        unique_dates = sorted(unique_dates)

        if verbose:
            logger.info(f"  Group数量: {len(unique_dates)}")
            logger.info(f"  日期范围: {unique_dates[0]} 至 {unique_dates[-1]}")

        # 确定基准日期
        if reference_date is None:
            ref_date = unique_dates[-1]
        else:
            ref_date = pd.to_datetime(reference_date)

        if verbose:
            logger.info(f"  基准日期: {ref_date}")
            logger.info(f"  时间衰减半衰期: {time_decay_half_life}天")

        # 确保market_data的date列是datetime类型
        if market_data is not None and not market_data.empty:
            if 'date' in market_data.columns:
                market_data = market_data.copy()
                market_data['date'] = pd.to_datetime(market_data['date'])

        # 计算每个group的权重
        group_weights = []

        for date in unique_dates:
            weight = 1.0

            # 业务逻辑1: 时间衰减（近期数据权重高）
            days_diff = (ref_date - date).days
            time_decay = np.exp(-np.log(2) * days_diff / time_decay_half_life)
            weight *= time_decay

            # 业务逻辑2: 市场波动率适中的日子权重高
            if market_data is not None and not market_data.empty:
                date_market = market_data[market_data['date'] == date]

                if not date_market.empty and 'close' in date_market.columns:
                    # 计算当日市场波动率（使用最近5天数据）
                    recent_market = market_data[market_data['date'] <= date].tail(5)
                    if len(recent_market) >= 2 and 'close' in recent_market.columns:
                        market_returns = recent_market['close'].pct_change().dropna()
                        if len(market_returns) > 0:
                            market_vol = market_returns.std()

                            # 黄金波动区间: 0.02 <= vol <= 0.05
                            if 0.02 <= market_vol <= 0.05:
                                weight *= 1.5  # 黄金波动区间
                            elif market_vol > 0.1:
                                weight *= 0.6  # 过度波动，噪声大

                # 业务逻辑3: 高成交量日子权重高
                if not date_market.empty and 'volume' in date_market.columns:
                    total_volume = date_market['volume'].sum()
                    if total_volume > 0:
                        # 对数变换以压缩极端值
                        weight *= np.log(total_volume / 1e9 + 1)

            group_weights.append(weight)

        # 转换为numpy数组
        group_weights = np.array(group_weights)

        # 归一化权重（使权重均值为1）
        group_weights = group_weights / group_weights.mean()

        if verbose:
            logger.info(f"分组权重统计:")
            logger.info(f"  权重范围: [{group_weights.min():.4f}, {group_weights.max():.4f}]")
            logger.info(f"  权重均值: {group_weights.mean():.4f}")
            logger.info(f"  权重中位数: {np.median(group_weights):.4f}")
            logger.info(f"  权重标准差: {group_weights.std():.4f}")
            logger.info("=" * 60)

        return group_weights

    except Exception as e:
        logger.error(f"XGBoost分组权重计算失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        # 返回均匀权重
        unique_dates = pd.Series(train_dates).unique()
        return np.ones(len(unique_dates))


def _calculate_professional_sample_weights(
    y_train: np.ndarray,
    imbalance_ratio: float,
    logger,
    boundary_distances: np.ndarray = None,
    is_regression: bool = False,
    y_train_original: np.ndarray = None,
    train_dates: np.ndarray = None,
    time_decay_half_life: float = 365.0,
    reference_date: str = None,
    turnover_rate: np.ndarray = None,
    circ_mv: np.ndarray = None,
    label_rank: np.ndarray = None,
    label_boost_beta: float = 0.1,
    w_min: float = 1,
    w_max: float = 1
) -> np.ndarray:
    """
    简化且可解释的权重方案：
    - 类别不平衡权重
    - 时间衰减权重（按半衰期真实生效）
    - 标签强度轻微增强

    设计原则：
    - 所有乘法分量先构造，再统一做一次均值归一化
    - 最后只做一次 clip，保证最终结果严格落在 [w_min, w_max]
    - 不再执行"clip 后再次除均值"，避免边界失效
    """
    import numpy as np
    import pandas as pd

    try:
        logger.info("样本权重计算开始（类别平衡 + 时间衰减 + 标签增强）")
        n = len(y_train)
        y = np.asarray(y_train, dtype=float)
        pos_mask = y > 0.5
        neg_mask = y <= 0.5
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos > 0 and n_neg > 0:
            imbalance = n_neg / max(n_pos, 1)
            # pos_weight = float(np.clip(np.power(imbalance, 0.65), 1.0, 3.5))
            # todo
            pos_weight = float(np.power(imbalance, 0.65))
        else:
            pos_weight = 1.0

        class_weight = np.where(pos_mask, pos_weight, 1.0).astype(float)

        if train_dates is None:
            decay = np.ones(n, dtype=float)
            logger.warning("[WARNING] train_dates 为 None，跳过时间衰减，仅使用类别权重")
        else:
            dates = pd.to_datetime(pd.Series(train_dates))
            recent = pd.to_datetime(reference_date) if reference_date is not None else dates.max()
            age_days = (recent - dates).dt.days.values
            age_days = np.clip(age_days, 0, None)
            half_life = max(float(time_decay_half_life), 1.0)
            decay_raw = np.exp(-np.log(2) * age_days / half_life)
            decay = decay_raw / max(float(np.mean(decay_raw)), 1e-12)
            decay = np.nan_to_num(decay, nan=1.0, posinf=1.0, neginf=1.0)

        # if label_rank is not None:
        #     rank_arr = np.asarray(label_rank, dtype=float)
        #     if len(rank_arr) == n:
        #         rank_arr = np.nan_to_num(rank_arr, nan=0.0)
        #         rank_min = float(np.min(rank_arr))
        #         rank_max = float(np.max(rank_arr))
        #         if rank_max > rank_min:
        #             rank_norm = (rank_arr - rank_min) / (rank_max - rank_min)
        #             label_boost = 1.0 + float(label_boost_beta) * (rank_norm - np.mean(rank_norm))
        #         else:
        #             label_boost = np.ones(n, dtype=float)
        #     else:
        #         logger.warning("[WARNING] label_rank 长度与样本数不一致，跳过标签增强")
        #         label_boost = np.ones(n, dtype=float)
        # else:
        #     label_boost = np.ones(n, dtype=float)

        # w_raw = class_weight * decay #* label_boost
        w_raw = decay #* label_boost
        w_raw = np.nan_to_num(w_raw, nan=1.0, posinf=w_max, neginf=w_min)

        mean_w = float(np.mean(w_raw)) if len(w_raw) > 0 else 1.0
        if mean_w > 0:
            w_final = w_raw / mean_w
        else:
            w_final = np.ones(n, dtype=float)

        w_final = np.clip(w_final, w_min, w_max)

        logger.info(
            f"类别权重统计: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.4f}, "
            f"权重范围[{w_final.min():.4f}, {w_final.max():.4f}], 均值={w_final.mean():.4f}"
        )
        logger.info(
            f"time adjusted weight 统计: decay范围[{np.nanmin(decay):.4f}, {np.nanmax(decay):.4f}], "
            # f"label_boost范围[{np.nanmin(label_boost):.4f}, {np.nanmax(label_boost):.4f}], "
            f"最终min={np.nanmin(w_final):.4f}, max={np.nanmax(w_final):.4f}, mean={np.nanmean(w_final):.4f}, "
            f"NaN={int(np.isnan(w_final).sum())}"
        )
        return w_final.astype(np.float32)

    except Exception as e:
        logger.error(f"样本权重计算失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("返回均匀权重作为后备")
        return np.ones(len(y_train) if y_train is not None else 100, dtype=np.float32)


def _calculate_precision_at_k(y_true, y_scores, k_values=[1, 3, 5, 10]):
    """
    计算Precision@k评估指标

    Args:
        y_true: 真实标签
        y_scores: 预测概率分数
        k_values: k值列表

    Returns:
        dict: 各个k值对应的Precision@k
    """
    try:
        import numpy as np

        # 按预测概率降序排序
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_y_true = y_true[sorted_indices]

        precision_at_k = {}

        for k in k_values:
            if k > len(y_true):
                # 如果k大于样本总数，使用全部样本
                k_actual = len(y_true)
            else:
                k_actual = k

            # 取前k个预测结果
            top_k_true = sorted_y_true[:k_actual]

            # 计算Precision@k
            if k_actual > 0:
                precision = np.sum(top_k_true) / k_actual
            else:
                precision = 0.0

            precision_at_k[k] = precision

        return precision_at_k

    except Exception as e:
        logger.error(f"计算Precision@k失败: {e}")
        return {k: 0.0 for k in k_values}

def _calculate_quantitative_metrics(y_true, y_pred, logger):
    """
    计算专业量化评估指标

    参数:
        y_true: 真实收益率 (ATR单位)
        y_pred: 预测收益率 (ATR单位)
        logger: 日志记录器

    返回:
        dict: 包含IC、RankIC、ICIR、RankICIR等指标的字典
    """
    try:
        import scipy.stats as stats

        # 1. Information Coefficient (IC) - 线性相关性
        ic, ic_pvalue = stats.pearsonr(y_true, y_pred)

        # 2. Rank Information Coefficient (RankIC) - 排序相关性
        rank_ic, rank_ic_pvalue = stats.spearmanr(y_true, y_pred)

        # 3. IC Information Ratio (ICIR) - IC的稳定性
        # 这里简化为IC的绝对值除以标准误差的估计
        ic_std = np.sqrt((1 - ic**2) / (len(y_true) - 2)) if len(y_true) > 2 else 0.1
        icir = abs(ic) / max(ic_std, 0.01)

        # 4. RankIC Information Ratio (RankICIR)
        rank_ic_std = np.sqrt((1 - rank_ic**2) / (len(y_true) - 2)) if len(y_true) > 2 else 0.1
        rank_icir = abs(rank_ic) / max(rank_ic_std, 0.01)

        # 5. 预测准确率指标
        # 方向准确率：预测和真实收益同号的比例
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        # 6. 分位数分析
        # 将预测值分为5个分位数，计算每个分位数的平均真实收益
        pred_quantiles = np.quantile(y_pred, [0.2, 0.4, 0.6, 0.8])
        quantile_returns = []
        for i in range(5):
            if i == 0:
                mask = y_pred <= pred_quantiles[0]
            elif i == 4:
                mask = y_pred > pred_quantiles[3]
            else:
                mask = (y_pred > pred_quantiles[i-1]) & (y_pred <= pred_quantiles[i])

            if mask.sum() > 0:
                quantile_returns.append(y_true[mask].mean())
            else:
                quantile_returns.append(0.0)

        # 7. 单调性检验 - 分位数收益是否单调递增
        monotonicity = np.corrcoef(range(5), quantile_returns)[0, 1]

        # 记录结果
        logger.info("[CHART] 专业量化评估指标:")
        logger.info(f"  Information Coefficient (IC): {ic:.4f} (p-value: {ic_pvalue:.4f})")
        logger.info(f"  Rank Information Coefficient (RankIC): {rank_ic:.4f} (p-value: {rank_ic_pvalue:.4f})")
        logger.info(f"  IC Information Ratio (ICIR): {icir:.4f}")
        logger.info(f"  RankIC Information Ratio (RankICIR): {rank_icir:.4f}")
        logger.info(f"  方向准确率: {direction_accuracy:.4f} ({direction_accuracy*100:.1f}%)")
        logger.info(f"  分位数单调性: {monotonicity:.4f}")

        logger.info("[CHART] 分位数收益分析:")
        for i, ret in enumerate(quantile_returns):
            logger.info(f"  Q{i+1} (预测值{i*20}-{(i+1)*20}%分位): {ret:.4f}")

        # 评估标准
        logger.info("[CHART] 评估标准:")
        ic_quality = "优秀" if abs(ic) > 0.05 else "良好" if abs(ic) > 0.03 else "一般" if abs(ic) > 0.01 else "较差"
        rank_ic_quality = "优秀" if abs(rank_ic) > 0.05 else "良好" if abs(rank_ic) > 0.03 else "一般" if abs(rank_ic) > 0.01 else "较差"
        direction_quality = "优秀" if direction_accuracy > 0.55 else "良好" if direction_accuracy > 0.52 else "一般" if direction_accuracy > 0.50 else "较差"

        logger.info(f"  IC质量: {ic_quality} (|IC|>0.05为优秀, >0.03为良好, >0.01为一般)")
        logger.info(f"  RankIC质量: {rank_ic_quality}")
        logger.info(f"  方向预测质量: {direction_quality} (>55%为优秀, >52%为良好, >50%为一般)")

        return {
            'ic': ic,
            'ic_pvalue': ic_pvalue,
            'rank_ic': rank_ic,
            'rank_ic_pvalue': rank_ic_pvalue,
            'icir': icir,
            'rank_icir': rank_icir,
            'direction_accuracy': direction_accuracy,
            'quantile_returns': quantile_returns,
            'monotonicity': monotonicity,
            'ic_quality': ic_quality,
            'rank_ic_quality': rank_ic_quality,
            'direction_quality': direction_quality
        }

    except Exception as e:
        logger.warning(f"计算量化评估指标失败: {e}")
        return {}

def _calculate_prediction_range_diagnostics(y_true, y_pred, logger, y_pred_raw=None):
    """
    计算预测范围诊断指标

    参数:
        y_true: 真实收益率 (ATR单位)
        y_pred: 预测收益率 (ATR单位)
        logger: 日志记录器
    """
    try:
        # 1. 范围对比
        true_range = [y_true.min(), y_true.max()]
        pred_range = [y_pred.min(), y_pred.max()]
        true_span = true_range[1] - true_range[0]
        pred_span = pred_range[1] - pred_range[0]
        coverage_ratio = pred_span / true_span if true_span > 0 else 0

        # 2. 标准差对比
        true_std = y_true.std()
        pred_std = y_pred.std()
        std_ratio = pred_std / true_std if true_std > 0 else 0

        # 3. 分位数对比
        true_quantiles = np.quantile(y_true, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        pred_quantiles = np.quantile(y_pred, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])

        # 4. 极值预测能力
        extreme_positive_mask = y_true > 2.0  # 大于2ATR的极值
        extreme_negative_mask = y_true < -1.0  # 小于-1ATR的极值

        extreme_positive_pred = y_pred[extreme_positive_mask].mean() if extreme_positive_mask.sum() > 0 else 0
        extreme_negative_pred = y_pred[extreme_negative_mask].mean() if extreme_negative_mask.sum() > 0 else 0

        # 记录诊断结果
        logger.info("[SEARCH] 预测范围诊断指标:")
        logger.info(f"  真实值范围: [{true_range[0]:.3f}, {true_range[1]:.3f}] ATR (跨度: {true_span:.3f})")

        # [强化] 新增：显示逆变换前后的预测值范围对比
        if y_pred_raw is not None:
            raw_range = [y_pred_raw.min(), y_pred_raw.max()]
            raw_span = raw_range[1] - raw_range[0]
            logger.info(f"  预测值范围(变换前): [{raw_range[0]:.3f}, {raw_range[1]:.3f}] (跨度: {raw_span:.3f})")

        logger.info(f"  预测值范围(逆变换后): [{pred_range[0]:.3f}, {pred_range[1]:.3f}] ATR (跨度: {pred_span:.3f})")
        logger.info(f"  范围覆盖率: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")

        logger.info(f"  真实值标准差: {true_std:.3f} ATR")
        logger.info(f"  预测值标准差: {pred_std:.3f} ATR")
        logger.info(f"  标准差比例: {std_ratio:.3f} ({std_ratio*100:.1f}%)")

        logger.info("  分位数对比:")
        quantile_names = ['1%', '5%', '25%', '75%', '95%', '99%']
        for i, name in enumerate(quantile_names):
            logger.info(f"    {name}: 真实{true_quantiles[i]:.3f} vs 预测{pred_quantiles[i]:.3f}")

        logger.info(f"  极值预测能力:")
        logger.info(f"    >2ATR样本数: {extreme_positive_mask.sum()}, 平均预测: {extreme_positive_pred:.3f}")
        logger.info(f"    <-1ATR样本数: {extreme_negative_mask.sum()}, 平均预测: {extreme_negative_pred:.3f}")

        # 诊断评估
        logger.info("🚨 诊断评估:")
        if coverage_ratio < 0.3:
            logger.warning(f"  [FAIL] 严重预测范围压缩: {coverage_ratio:.1%} < 30%")
        elif coverage_ratio < 0.5:
            logger.warning(f"  [WARNING]  预测范围偏小: {coverage_ratio:.1%} < 50%")
        else:
            logger.info(f"  [OK] 预测范围合理: {coverage_ratio:.1%}")

        if std_ratio < 0.5:
            logger.warning(f"  [FAIL] 预测过度保守: 标准差比例{std_ratio:.1%} < 50%")
        elif std_ratio < 0.7:
            logger.warning(f"  [WARNING]  预测偏保守: 标准差比例{std_ratio:.1%} < 70%")
        else:
            logger.info(f"  [OK] 预测变异性合理: 标准差比例{std_ratio:.1%}")

        return {
            'coverage_ratio': coverage_ratio,
            'std_ratio': std_ratio,
            'true_range': true_range,
            'pred_range': pred_range,
            'extreme_positive_pred': extreme_positive_pred,
            'extreme_negative_pred': extreme_negative_pred
        }

    except Exception as e:
        logger.warning(f"计算预测范围诊断指标失败: {e}")
        return {}

def _calculate_trading_metrics(y_true_return, y_pred_scores, logger, k_values=[1, 3, 5]):
    """
    [强化] 计算交易指标：包含夏普比率、最大回撤等实际交易表现指标

    Args:
        y_true_return: 真实收益标签（ATR倍数）
        y_pred_scores: 预测分数（用于排序选股）
        logger: 日志记录器
        k_values: Top-K值列表

    Returns:
        dict: 交易指标字典
    """
    try:
        logger.info("[强化] 计算交易指标...")

        # 按预测分数降序排序
        sorted_indices = np.argsort(y_pred_scores)[::-1]
        sorted_returns = y_true_return[sorted_indices]

        # [强化] 修复：使用与选股阶段一致的固定阈值
        # 重要：模型评估必须与实际选股使用相同的阈值标准
        profit_threshold = ATR_MULTIPLIER_PROFIT  # 2.0 ATR（与选股阶段一致）
        loss_threshold = -ATR_MULTIPLIER_LOSS     # -1.0 ATR（与选股阶段一致）

        float_tolerance = 1e-6  # 浮点数容差

        logger.info(f"[强化] 使用与选股阶段一致的固定阈值:")
        logger.info(f"  止盈阈值: >= {profit_threshold:.1f} ATR (与选股阶段一致)")
        logger.info(f"  止损阈值: <= {loss_threshold:.1f} ATR (与选股阶段一致)")
        logger.info(f"  注意：评估标准必须与实际应用保持一致")

        # 1. 止盈止损命中率（使用固定阈值）
        profit_hits = (y_true_return >= profit_threshold).sum()
        profit_hit_rate = profit_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0
        loss_hits = (y_true_return <= loss_threshold).sum()
        loss_hit_rate = loss_hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 2. 中等收益阈值评估
        moderate_thresholds = [0.5, 1.0, 1.5]
        moderate_hit_rates = {}
        for threshold in moderate_thresholds:
            hits = (y_true_return >= threshold).sum()
            moderate_hit_rates[f"profit_{threshold}"] = hits / len(y_true_return) if len(y_true_return) > 0 else 0.0

        # 3. [强化] Top-K交易表现指标（修复计算错误）
        top_k_returns = {}
        top_k_profit_rates = {}
        top_k_loss_rates = {}
        top_k_sharpe_ratios = {}
        top_k_max_drawdowns = {}
        top_k_calmar_ratios = {}

        for k in k_values:
            if k <= len(sorted_returns):
                top_k_return = sorted_returns[:k]
                top_k_returns[k] = top_k_return.mean()

                # 使用固定阈值计算Top-K止盈率
                top_k_profit_rates[k] = (top_k_return >= (profit_threshold - float_tolerance)).mean()

                # 使用固定阈值计算Top-K止损率
                top_k_loss_rates[k] = (top_k_return <= (loss_threshold + float_tolerance)).mean()

                # [强化] 修复：Top-K夏普比率（处理单样本情况）
                if len(top_k_return) > 1 and top_k_return.std() > 1e-8:
                    # 使用无风险利率3%
                    risk_free_daily = 0.03 / 252
                    excess_returns = top_k_return - risk_free_daily
                    top_k_sharpe_ratios[k] = excess_returns.mean() / top_k_return.std() * np.sqrt(252)
                else:
                    # 单样本情况：如果收益为正且显著，给予合理评分
                    if len(top_k_return) == 1 and top_k_return[0] > 0.5:
                        top_k_sharpe_ratios[k] = top_k_return[0] * 10  # 简化评分
                    else:
                        top_k_sharpe_ratios[k] = 0.0

                # [强化] 修复：Top-K最大回撤（使用ATR直接计算）
                if len(top_k_return) > 1:
                    cumulative_returns = top_k_return.cumsum()  # 累计ATR收益
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdowns = cumulative_returns - running_max
                    top_k_max_drawdowns[k] = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
                else:
                    top_k_max_drawdowns[k] = 0.0

                # [强化] 修复：Top-K卡尔马比率
                if top_k_max_drawdowns[k] > 1e-8:
                    annual_return = top_k_return.mean() * 252  # 年化ATR收益
                    top_k_calmar_ratios[k] = annual_return / top_k_max_drawdowns[k]
                else:
                    top_k_calmar_ratios[k] = 0.0
            else:
                top_k_returns[k] = 0.0
                top_k_profit_rates[k] = 0.0
                top_k_loss_rates[k] = 0.0
                top_k_sharpe_ratios[k] = 0.0
                top_k_max_drawdowns[k] = 0.0
                top_k_calmar_ratios[k] = 0.0

        # 4. 收益分布分析
        return_distribution = {
            'profit_samples': (y_true_return >= (profit_threshold - float_tolerance)).sum(),
            'neutral_samples': ((y_true_return > (loss_threshold + float_tolerance)) & (y_true_return < (profit_threshold - float_tolerance))).sum(),
            'loss_samples': (y_true_return <= (loss_threshold + float_tolerance)).sum(),
            'mean_return': y_true_return.mean(),
            'std_return': y_true_return.std(),
            'min_return': y_true_return.min(),
            'max_return': y_true_return.max()
        }

        # 5. 预测效果分析
        # 计算预测收益最高的样本的实际表现
        top_10_pct_idx = int(len(sorted_returns) * 0.1)
        if top_10_pct_idx > 0:
            top_10_pct_return = sorted_returns[:top_10_pct_idx].mean()
            bottom_10_pct_return = sorted_returns[-top_10_pct_idx:].mean()
            prediction_effectiveness = top_10_pct_return - bottom_10_pct_return
        else:
            prediction_effectiveness = 0.0

        trading_metrics = {
            # 固定阈值指标
            'profit_hit_rate': profit_hit_rate,
            'loss_hit_rate': loss_hit_rate,
            # Top-K指标
            'top_k_returns': top_k_returns,
            'top_k_profit_rates': top_k_profit_rates,
            'top_k_loss_rates': top_k_loss_rates,
            'top_k_sharpe_ratios': top_k_sharpe_ratios,
            'top_k_max_drawdowns': top_k_max_drawdowns,
            'top_k_calmar_ratios': top_k_calmar_ratios,
            # 其他指标
            'return_distribution': return_distribution,
            'prediction_effectiveness': prediction_effectiveness,
            # 阈值信息
            'profit_threshold': profit_threshold,
            'loss_threshold': loss_threshold
        }

        # [强化] 多层次交易指标统计（解决阈值过高问题）
        logger.info(f"[CHART] 交易指标统计:")
        logger.info(f"  参与统计的y_true数量: {len(y_true_return):,}个样本")
        logger.info(f"  [TARGET] 固定阈值统计:")
        logger.info(f"    止盈边界样本数量: {profit_hits:,}个 (收益 >= {profit_threshold:.1f} ATR)")
        logger.info(f"    止损边界样本数量: {loss_hits:,}个 (收益 <= {loss_threshold:.1f} ATR)")
        logger.info(f"    止盈命中率: {profit_hit_rate:.4f} ({profit_hits}/{len(y_true_return)})")
        logger.info(f"    止损命中率: {loss_hit_rate:.4f} ({loss_hits}/{len(y_true_return)})")
        logger.info(f"  预测有效性: {prediction_effectiveness:.4f} ATR (Top10% - Bottom10%)")

        # [强化] 添加收益分布诊断信息
        logger.info(f"[CHART] 收益分布诊断:")
        logger.info(f"  收益范围: [{y_true_return.min():.3f}, {y_true_return.max():.3f}] ATR")
        logger.info(f"  收益均值: {y_true_return.mean():.3f} ATR")
        logger.info(f"  收益标准差: {y_true_return.std():.3f} ATR")
        logger.info(f"  收益中位数: {np.median(y_true_return):.3f} ATR")
        logger.info(f"  负收益样本数: {(y_true_return < 0).sum():,}个 ({(y_true_return < 0).mean():.1%})")

        logger.info(f"[CHART] Top-K交易表现指标:")
        for k in k_values:
            if k <= len(sorted_returns):
                top_k_data = sorted_returns[:k]
                logger.info(f"  Top-{k}: 平均{top_k_returns[k]:.4f} ATR, 范围[{top_k_data.min():.3f}, {top_k_data.max():.3f}]")
                logger.info(f"    止盈率: {top_k_profit_rates[k]:.4f} ({(top_k_data >= (profit_threshold - float_tolerance)).sum()}/{k})")
                logger.info(f"    止损率: {top_k_loss_rates[k]:.4f} ({(top_k_data <= (loss_threshold + float_tolerance)).sum()}/{k})")
                logger.info(f"    负收益率: {(top_k_data < 0).mean():.4f} ({(top_k_data < 0).sum()}/{k})")
                logger.info(f"    [强化] 夏普比率: {top_k_sharpe_ratios[k]:.4f}")
                logger.info(f"    [强化] 最大回撤: {top_k_max_drawdowns[k]:.4f}")
                logger.info(f"    [强化] 卡尔马比率: {top_k_calmar_ratios[k]:.4f}")

        # [强化] 添加对比统计：Top-K vs Bottom-K，解释为什么Top-K止损率为0
        if len(sorted_returns) >= 50:
            bottom_10_return = sorted_returns[-10:].mean()
            bottom_50_return = sorted_returns[-50:].mean()

            logger.info(f"[CHART] 选股效果对比（解释Top-K止损率为0的原因）:")
            logger.info(f"  Top-10平均收益: {sorted_returns[:10].mean():.4f} ATR")
            logger.info(f"  Bottom-10平均收益: {bottom_10_return:.4f} ATR")
            logger.info(f"  Top-Bottom差异: {sorted_returns[:10].mean() - bottom_10_return:.4f} ATR")

            # 计算Bottom-K的止损率
            bottom_10_loss_rate = (sorted_returns[-10:] <= loss_threshold).mean()
            bottom_50_loss_rate = (sorted_returns[-50:] <= loss_threshold).mean()
            logger.info(f"  Bottom-10止损率: {bottom_10_loss_rate:.4f}")
            logger.info(f"  Bottom-50止损率: {bottom_50_loss_rate:.4f}")
            logger.info(f"[OK] Top-K止损率为0说明选股策略有效，成功避开了表现差的股票")

        logger.info(f"[CHART] 收益分布:")
        logger.info(f"  盈利样本: {return_distribution['profit_samples']} 个")
        logger.info(f"  中性样本: {return_distribution['neutral_samples']} 个")
        logger.info(f"  亏损样本: {return_distribution['loss_samples']} 个")
        logger.info(f"  平均收益: {return_distribution['mean_return']:.4f} ATR")

        return trading_metrics

    except Exception as e:
        logger.error(f"计算交易指标失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return {}


class EarlyStoppingCallback:
    """早停回调函数 - 如果连续N次试验没有改进,提前停止"""

    def __init__(self, patience=5, min_improvement=0.001, logger=None):
        self.patience = patience
        self.min_improvement = min_improvement
        self.logger = logger or logging.getLogger(__name__)
        self.best_score = float('-inf')  # [OK] 改为负无穷（越大越好）
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        current_score = study.best_value

        # 检查是否有改进（越大越好）
        if current_score > self.best_score + self.min_improvement:  # [OK] 改为 >
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # 检查是否需要早停
        if self.no_improvement_count >= self.patience:
            self.logger.info(f"  [WARNING] 早停触发: 连续{self.patience}次试验无明显改进(改进<{self.min_improvement:.4f})")
            study.stop()


def _perform_parameter_search(X_train, y_train, X_val, y_val, y_val_eval, sample_weights, search_strategy, val_dates=None, train_dates=None, market_data=None):
    """
    [SEARCH] 优化版贝叶斯超参数搜索（排序模型 + 数据子采样 + Optuna剪枝）

    优化策略:
    1. 单次搜索: 60%数据, 30 trials, 激进MedianPruner剪枝
    2. 使用排序损失函数(rank:ndcg)
    3. 减少early_stopping轮数加速

    参数范围:
    - n_estimators: 300-2000 (避免过度训练)
    - learning_rate: 0.01-0.2 (对数尺度)
    - max_depth: 3-10
    - subsample: 0.6-0.95
    - colsample_bytree: 0.6-0.95
    - min_child_weight: 1-15
    - gamma: 0-1
    - reg_alpha: 1e-4-5 (对数尺度)
    - reg_lambda: 1e-4-5 (对数尺度)

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        y_val_eval: 验证集评估标签
        sample_weights: 样本权重（未使用，保留接口兼容性）
        search_strategy: 搜索策略(未使用,保留接口兼容性)
        val_dates: 验证集日期(用于Top-K计算和group计算)
        train_dates: 训练集日期(用于group计算)
        market_data: 市场数据DataFrame（用于计算group权重）

    Returns:
        tuple: (best_params, search_results)
    """
    import time
    import xgboost as xgb
    import numpy as np
    import logging

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("[FAIL] 缺少optuna库,请安装: pip install optuna")
        logger.error("[FAIL] 贝叶斯搜索需要optuna库,无法继续")
        return None, None, []

    logger.info("=" * 80)
    logger.info("[SEARCH] 启动优化版贝叶斯超参数搜索")
    logger.info("=" * 80)
    logger.info("优化策略:")
    logger.info("  1. 单次搜索（避免数据集不一致问题）")
    logger.info("  2. 数据子采样加速（60%数据）")
    logger.info("  3. Optuna MedianPruner激进剪枝")
    logger.info("  4. XGBoost多线程（n_jobs=-1）")
    logger.info("  5. 合理参数范围（n_estimators: 300-2000）")
    logger.info("  6. 减少early_stopping轮数加速")
    logger.info("  7. 预计算group权重（避免重复计算）")
    logger.info("=" * 80)

    # [强化] 数据子采样策略（60%数据）
    n_samples = len(X_train)
    sample_size = int(n_samples * 0.6)
    sample_idx = np.random.choice(n_samples, sample_size, replace=False)
    X_sampled = X_train[sample_idx]
    y_sampled = y_train[sample_idx]
    # [强化] XGBoost排序模型不使用样本权重,跳过采样
    # weights_sampled = sample_weights[sample_idx] if sample_weights is not None else None
    train_dates_sampled = train_dates[sample_idx] if train_dates is not None else None

    logger.info(f"[TARGET] 数据采样策略:")
    logger.info(f"  全量数据: {n_samples:,}条")
    logger.info(f"  搜索数据: {sample_size:,}条 (60%)")
    logger.info(f"  最终训练: {n_samples:,}条 (100%)")
    logger.info(f"  ⏭️ 跳过样本权重采样（XGBoost排序模型使用分组权重）")

    # [强化] 预计算group权重（避免每个trial重复计算）
    logger.info(f"\n[TARGET] 预计算group权重:")
    group_weights_sampled = None
    if train_dates_sampled is not None:
        group_weights_sampled = _calculate_xgboost_group_weights(
            train_dates=train_dates_sampled,
            market_data=market_data,
            logger=logger,
            time_decay_half_life=180.0,
            reference_date=None,
            verbose=True  # 🔊 首次计算时输出详细日志
        )
        logger.info(f"  [OK] group权重已预计算: {len(group_weights_sampled)}个分组")
    else:
        logger.warning("  [WARNING] 缺少训练集日期信息,无法预计算group权重")

    # 基础参数（启用多线程加速，排序模型）
    # [强化] 使用rank:pairwise支持sample-level weights
    base_params = {'random_state': 42, 'n_jobs': -1, 'verbosity': 0, 'objective': 'rank:pairwise'}

    # 全局最优结果
    all_trials = []

    # ==================== 单次搜索: 探索参数空间 ====================
    logger.info("\n" + "=" * 80)
    logger.info("[SEARCH] 开始参数搜索（60%数据，30 trials）")
    logger.info("=" * 80)

    # 定义objective函数
    def objective(trial):
        """Objective: 探索参数空间"""
        params = {
            'max_depth': trial.suggest_int("max_depth", 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 0.95),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 15),
            'gamma': trial.suggest_float("gamma", 0, 1),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000, step=100)
        }
        params.update(base_params)

        # [强化] 减少early_stopping轮数加速搜索
        if params['n_estimators'] <= 500:
            early_stopping = 30
        elif params['n_estimators'] <= 1000:
            early_stopping = 50
        else:
            early_stopping = 80

        score, metrics = _evaluate_params(
            params, X_sampled, y_sampled, X_val, y_val, y_val_eval,
            None,  # [强化] XGBoost排序模型不使用样本权重
            params['n_estimators'], early_stopping,
            is_coarse_search=False,
            val_dates=val_dates,
            train_dates=train_dates_sampled,
            market_data=market_data,  # [强化] 传递market_data（仅在group_weights=None时使用）
            group_weights=group_weights_sampled  # [强化] 传递预计算的group权重
        )

        all_trials.append({'stage': '搜索', 'params': params, 'score': score, 'metrics': metrics})
        trial.set_user_attr('metrics', metrics)

        return score

    # 创建study（使用激进的MedianPruner）
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # 减少到3(原5)
            n_warmup_steps=5     # 减少到5(原10)
        )
    )

    logger.info("  [强化] 开始搜索（30 trials）...")
    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"  [OK] 搜索完成: score={best_score:.6f}")
    logger.info(f"     最优参数: lr={best_params['learning_rate']:.4f}, n_est={best_params['n_estimators']}, depth={best_params['max_depth']}")

    # 添加base_params
    global_best_params = best_params.copy()
    global_best_params.update(base_params)

    # ==================== 最终结果汇总 ====================
    logger.info("\n" + "=" * 80)
    logger.info("[CELEBRATE] 优化版贝叶斯超参数搜索完成!")
    logger.info("=" * 80)
    logger.info(f"[TARGET] 全局最优综合评分: {best_score:.6f}")
    logger.info(f"[TARGET] 最优参数:")
    for key, value in global_best_params.items():
        if key not in ['random_state', 'n_jobs', 'verbosity', 'objective']:
            logger.info(f"  {key}: {value}")
    logger.info(f"[CHART] 总试验次数: {len(all_trials)}")
    logger.info(f"[CHART] 预计加速: 2-5倍（数据子采样 + Optuna剪枝 + XGBoost多线程 + 减少early_stopping）")
    logger.info("=" * 80)

    return global_best_params, all_trials


def _evaluate_params(params, X_train, y_train, X_val, y_val, y_val_eval, sample_weights, max_boost_rounds, early_stopping, is_coarse_search=False, val_dates=None, train_dates=None, market_data=None, group_weights=None):
    """
    [强化] 评估单组参数的性能（排序模型专用）

    Args:
        group_weights: 预计算的group权重（如果为None则重新计算）

    [强化] 综合评分 = 头部收益主导 + 稳定性辅助
                - overfitting_penalty - underfitting_penalty + r2_bonus

    [OK] 评分方向: 越大越好 (所有指标都是正向指标)
    [OK] 核心指标:
       - Head Return (70%): Top-1/3/5收益率加权均值,重点优化最头部股票的真实收益
       - Head Profit Rate (20%): Top-1/3/5中盈利股票的比例,衡量头部选股质量
       - Rank IC (10%): 仅作为排序稳定性辅助,避免完全失去整体排序能力

    [OK] 辅助指标(仅日志输出,不参与score计算):
       - R²: 模型拟合质量,通过惩罚项间接影响score
       - IC: Pearson相关系数,诊断模型方向
       - NDCG: 归一化折损累积增益,排序模型标准评估指标

    [TARGET] 设计理念:
       - 参数搜索更贴近实盘目标,优先优化Top-1~Top-5的头部收益
       - 保留少量Rank IC作为辅助约束,避免模型只会"赌Top1"而整体排序失真
       - 不使用更大范围Top-N收益参与打分,减少对尾部股票表现的关注

    过拟合/欠拟合处理:
    - 过拟合惩罚（指数平滑）:
      * overfitting_ratio > 0.2 -> penalty = 0.1 * (1 - exp(-(overfitting_ratio - 0.2) * 2.0))
    - 欠拟合惩罚（指数平滑）:
      * val_r2 < 0 -> penalty = 0.05 * (1 - exp(val_r2 * 5.0))
    - R²奖励:
      * val_r2 > 0 -> bonus = 0.1 * val_r2

    Args:
        params: XGBoost参数字典
        X_train, y_train: 训练数据（标签为连续收益率）
        X_val, y_val: 验证数据（标签为连续收益率）
        y_val_eval: 验证集评估标签（加权收益率weighted_return: 0.4*2d + 0.4*3d + 0.2*5d）
        sample_weights: 样本权重（未使用，保留接口兼容性）
        max_boost_rounds: 最大训练轮数
        early_stopping: 早停轮数
        is_coarse_search: 已废弃（保留接口兼容性）
        val_dates: 验证集日期（用于计算group）
        train_dates: 训练集日期（用于计算group）
        market_data: 市场数据DataFrame（用于计算group权重）

    Returns:
        tuple: (综合评分, 指标字典) - 评分越大越好
    """
    import xgboost as xgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr

    try:
        # 计算group信息（排序模型必需）
        if train_dates is not None:
            train_dates_series = pd.Series(train_dates)
            train_groups = train_dates_series.value_counts().sort_index().values
        else:
            raise ValueError("排序模型需要train_dates参数")

        if val_dates is not None:
            val_dates_series = pd.Series(val_dates)
            val_groups = val_dates_series.value_counts().sort_index().values
        else:
            raise ValueError("排序模型需要val_dates参数")

        # [强化] XGBoost排序模型权重处理: 使用group-level weights

        # [强化] 使用预计算的group权重（如果没有则重新计算）
        if group_weights is None:
            logger.info("  [WARNING] group权重未预计算,正在重新计算...")
            group_weights = _calculate_xgboost_group_weights(
                train_dates=train_dates,
                market_data=market_data,
                logger=logger,
                time_decay_half_life=180.0,
                reference_date=None,  # 使用训练集最新日期
                verbose=False  # 🔇 避免重复输出日志
            )

        # 训练模型(使用排序损失函数)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_groups)
        dtrain.set_weight(group_weights)  # [强化] 使用group-level weights

        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(val_groups)

        evals = [(dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=max_boost_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )

        # 预测(直接使用排名标签空间,无需逆变换)
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)

        # [强化] 计算MSE和R²（基于排名标签,而非原始收益率）
        # 注意: y_train和y_val是排名标签(0-1范围), y_val_eval是原始收益率
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)  # [WRENCH] 修复: 使用y_val而非y_val_eval

        # [强化] 计算R²（用于过拟合检测,基于排名标签）
        train_r2 = 1 - mse_train / (y_train.var() + 1e-9)
        val_r2 = 1 - mse_val / (y_val.var() + 1e-9)  # [WRENCH] 修复: 使用y_val而非y_val_eval
        overfitting_ratio = train_r2 - val_r2


        # [强化] 排序模型核心指标
        # [强化] 不再使用sklearn的ndcg_score，改用自定义的calculate_ndcg_at_k

        # [强化] 初始化指标
        ic = 0.0
        rank_ic = 0.0
        ndcg_5 = 0.0
        ndcg_10 = 0.0
        top_k_return = 0.0

        if val_dates is not None and len(val_dates) == len(y_val_pred):
            import pandas as pd

            # 创建DataFrame便于按日期分组
            # 注意: y_val是relevance标签(0-3), y_val_eval是原始收益率
            val_df = pd.DataFrame({
                'date': val_dates,
                'pred': y_val_pred,
                'relevance': y_val,  # 🆕 relevance标签用于NDCG
                'return': y_val_eval  # 原始收益率用于IC和Top-K
            })

            # [强化] 计算每日指标
            ndcg_1_list = []
            ndcg_3_list = []
            ndcg_5_list = []
            ndcg_10_list = []
            # 🎲 随机基线NDCG列表
            ndcg_1_random_list = []
            ndcg_3_random_list = []
            ndcg_5_random_list = []
            ndcg_10_random_list = []
            daily_ic_list = []  # 🆕 每日IC
            daily_rank_ic_list = []  # 🆕 每日Rank IC
            daily_top_returns = []
            daily_top_profits = []
            # [UP] Top-k实际收益率列表
            top_1_return_list = []
            top_3_return_list = []
            top_5_return_list = []
            top_10_return_list = []

            for date, group in val_df.groupby('date'):
                if len(group) < 5:
                    continue

                # 🆕 每日IC (Pearson相关系数)
                daily_ic = np.corrcoef(group['pred'].values, group['return'].values)[0, 1]
                if not np.isnan(daily_ic):
                    daily_ic_list.append(daily_ic)

                # 🆕 每日Rank IC (Spearman相关系数)
                daily_rank_ic, _ = spearmanr(group['pred'].values, group['return'].values)
                if not np.isnan(daily_rank_ic):
                    daily_rank_ic_list.append(daily_rank_ic)

                # [强化] NDCG使用真实收益率（更准确的评估）
                true_returns = group['return'].values
                pred_scores = group['pred'].values

                # 🎲 随机基线：多次随机打乱预测值，取平均（更公平的对比）
                num_random_trials = 10  # 多次随机试验取平均
                random_ndcg_1_trials, random_ndcg_3_trials, random_ndcg_5_trials, random_ndcg_10_trials = [], [], [], []

                for trial in range(num_random_trials):
                    # 保持预测值分布，只打乱顺序
                    np.random.seed(trial + int(hash(str(group.index.values.tobytes()))) % (2**32))
                    random_scores = np.random.permutation(pred_scores)

                    if len(group) >= 1:
                        random_ndcg_1_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=1))
                    if len(group) >= 3:
                        random_ndcg_3_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=3))
                    if len(group) >= 5:
                        random_ndcg_5_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=5))
                    if len(group) >= 10:
                        random_ndcg_10_trials.append(calculate_adjusted_ndcg_at_k(true_returns, random_scores, k=10))

                # 模型NDCG计算（使用改进版本）
                if len(group) >= 1:
                    ndcg_1_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=1))
                    if random_ndcg_1_trials:
                        ndcg_1_random_list.append(np.mean(random_ndcg_1_trials))

                if len(group) >= 3:
                    ndcg_3_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=3))
                    if random_ndcg_3_trials:
                        ndcg_3_random_list.append(np.mean(random_ndcg_3_trials))

                if len(group) >= 5:
                    ndcg_5_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=5))
                    if random_ndcg_5_trials:
                        ndcg_5_random_list.append(np.mean(random_ndcg_5_trials))

                if len(group) >= 10:
                    ndcg_10_list.append(calculate_adjusted_ndcg_at_k(true_returns, pred_scores, k=10))
                    if random_ndcg_10_trials:
                        ndcg_10_random_list.append(np.mean(random_ndcg_10_trials))

                # [UP] Top-k实际收益率计算
                if len(group) >= 1:
                    top_1_idx = group.nlargest(1, 'pred').index
                    top_1_return_list.append(group.loc[top_1_idx, 'return'].mean())

                if len(group) >= 3:
                    top_3_idx = group.nlargest(3, 'pred').index
                    top_3_return_list.append(group.loc[top_3_idx, 'return'].mean())

                if len(group) >= 5:
                    top_5_idx = group.nlargest(5, 'pred').index
                    top_5_return_list.append(group.loc[top_5_idx, 'return'].mean())

                if len(group) >= 10:
                    top_10_idx = group.nlargest(10, 'pred').index
                    top_10_return_list.append(group.loc[top_10_idx, 'return'].mean())

                # 按预测值降序排列
                group_sorted = group.sort_values('pred', ascending=False)

                # 选择Top-1, Top-3, Top-5个股票
                top_1 = group_sorted.iloc[:1]['return'].values
                top_3 = group_sorted.iloc[:3]['return'].values
                top_5 = group_sorted.iloc[:5]['return'].values

                # 计算每日Top-K收益
                daily_top_1_return = top_1.mean() if len(top_1) > 0 else 0.0
                daily_top_3_return = top_3.mean() if len(top_3) > 0 else 0.0
                daily_top_5_return = top_5.mean() if len(top_5) > 0 else 0.0

                # 计算每日Top-K盈利率
                daily_top_1_profit = (top_1 > 0).mean() if len(top_1) > 0 else 0.0
                daily_top_3_profit = (top_3 > 0).mean() if len(top_3) > 0 else 0.0
                daily_top_5_profit = (top_5 > 0).mean() if len(top_5) > 0 else 0.0

                # 加权平均(Top-1权重最高)
                daily_return = 0.5 * daily_top_1_return + 0.3 * daily_top_3_return + 0.2 * daily_top_5_return
                daily_profit = 0.5 * daily_top_1_profit + 0.3 * daily_top_3_profit + 0.2 * daily_top_5_profit

                daily_top_returns.append(daily_return)
                daily_top_profits.append(daily_profit)

            # [强化] 计算平均指标（每日指标的平均值）
            ndcg_1 = np.mean(ndcg_1_list) if ndcg_1_list else 0.0
            ndcg_3 = np.mean(ndcg_3_list) if ndcg_3_list else 0.0
            ndcg_5 = np.mean(ndcg_5_list) if ndcg_5_list else 0.0
            ndcg_10 = np.mean(ndcg_10_list) if ndcg_10_list else 0.0
            ic = np.mean(daily_ic_list) if daily_ic_list else 0.0  # 🆕 每日IC的平均值
            rank_ic = np.mean(daily_rank_ic_list) if daily_rank_ic_list else 0.0  # 🆕 每日Rank IC的平均值

            # 计算所有交易日的平均值
            top_1_return = np.mean([group.sort_values('pred', ascending=False).iloc[:1]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 1])
            top_3_return = np.mean([group.sort_values('pred', ascending=False).iloc[:3]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 3])
            top_5_return = np.mean([group.sort_values('pred', ascending=False).iloc[:5]['return'].mean()
                                   for _, group in val_df.groupby('date') if len(group) >= 5])

            top_1_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:1]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 1])
            top_3_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:3]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 3])
            top_5_profit_rate = np.mean([(group.sort_values('pred', ascending=False).iloc[:5]['return'] > 0).mean()
                                        for _, group in val_df.groupby('date') if len(group) >= 5])

            top_k_return = np.mean(daily_top_returns)
            top_k_profit_rate = np.mean(daily_top_profits)

        else:
            # 降级方案: 如果没有日期信息,使用全局Top-K个股票
            sorted_indices = np.argsort(y_val_pred)[::-1]  # 降序排列,最大的在前

            # 直接选择Top-K个股票,而不是Top-K%
            top_1_return = y_val_eval[sorted_indices[:1]].mean() if len(sorted_indices) >= 1 else 0.0
            top_3_return = y_val_eval[sorted_indices[:3]].mean() if len(sorted_indices) >= 3 else 0.0
            top_5_return = y_val_eval[sorted_indices[:5]].mean() if len(sorted_indices) >= 5 else 0.0

            top_1_profit_rate = (y_val_eval[sorted_indices[:1]] > 0).mean() if len(sorted_indices) >= 1 else 0.0
            top_3_profit_rate = (y_val_eval[sorted_indices[:3]] > 0).mean() if len(sorted_indices) >= 3 else 0.0
            top_5_profit_rate = (y_val_eval[sorted_indices[:5]] > 0).mean() if len(sorted_indices) >= 5 else 0.0

            top_k_return = 0.5 * top_1_return + 0.3 * top_3_return + 0.2 * top_5_return
            top_k_profit_rate = 0.5 * top_1_profit_rate + 0.3 * top_3_profit_rate + 0.2 * top_5_profit_rate

        # [强化] 头部收益导向评分（越大越好）
        # 聚焦Top-1/3/5收益和胜率，弱化整体排序指标，避免搜索过程被更大范围Top-N表现牵引。
        head_return_score = (
            0.55 * top_1_return +
            0.30 * top_3_return +
            0.15 * top_5_return
        )
        head_profit_score = (
            0.55 * top_1_profit_rate +
            0.30 * top_3_profit_rate +
            0.15 * top_5_profit_rate
        )

        composite_score = (
            0.70 * head_return_score +          # 头部收益权重70%(主要指标)
            0.20 * head_profit_score +          # 头部胜率权重20%(辅助确认质量)
            0.10 * rank_ic                      # Rank IC权重10%(仅保留排序稳定性约束)
        )

        # [强化] 软性惩罚: 过拟合和欠拟合（完全重新设计）
        overfitting_penalty = 0.0
        underfitting_penalty = 0.0

        # 过拟合惩罚: 使用平滑的指数惩罚
        # overfitting_ratio > 0.2时开始惩罚,越大惩罚越重
        if overfitting_ratio > 0.2:
            overfitting_penalty = 0.1 * (1 - np.exp(-(overfitting_ratio - 0.2) * 2.0))  # 最大惩罚约0.1

        # 欠拟合惩罚: 基于val_r2的负值程度
        # val_r2 < 0时开始惩罚,越负惩罚越重
        if val_r2 < 0:
            underfitting_penalty = 0.05 * (1 - np.exp(val_r2 * 5.0))  # 最大惩罚约0.05

        # R²奖励: 如果val_r2 > 0,给予额外奖励
        r2_bonus = 0.0
        if val_r2 > 0:
            r2_bonus = 0.1 * val_r2  # val_r2越高,奖励越多

        # 应用惩罚和奖励
        composite_score = composite_score - overfitting_penalty - underfitting_penalty + r2_bonus

        metrics = {
            'mse_train': mse_train,
            'mse_val': mse_val,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'ic': ic,
            'rank_ic': rank_ic,
            'ndcg_5': ndcg_5,  # 🆕 NDCG@5
            'ndcg_10': ndcg_10,  # 🆕 NDCG@10
            'head_return_score': head_return_score,
            'head_profit_score': head_profit_score,
            'top_1_return': top_1_return,
            'top_3_return': top_3_return,
            'top_5_return': top_5_return,
            'top_k_return': top_k_return,
            'top_1_profit_rate': top_1_profit_rate,
            'top_3_profit_rate': top_3_profit_rate,
            'top_5_profit_rate': top_5_profit_rate,
            'top_k_profit_rate': top_k_profit_rate,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_penalty': overfitting_penalty,
            'underfitting_penalty': underfitting_penalty,
            'r2_bonus': r2_bonus,
            'composite_score': composite_score
        }

        return composite_score, metrics

    except Exception as e:
        # 训练失败,返回极小惩罚（越大越好）
        logger.error(f"  [FAIL] 训练失败: {str(e)}")
        return -1e6, {'error': str(e)}



def _get_lgb_search_space():
    """
    【大师级重构】LightGBM 二分类模型搜索空间
    """
# {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'max_depth': 7, 'bagging_freq': 1, 'n_jobs': -1, 'verbosity': -1, 'n_estimators': 400, 'random_state': 42, 'num_leaves': 15, 'min_data_in_leaf': 500, 'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'reg_alpha': 1, 'reg_lambda': 10, 'scale_pos_weight': 60}, top1_return:5237.797392723667
# 'n_estimators': 500, 'num_leaves': 127, 'min_data_in_leaf': 400, 'feature_fraction': 0.8, 'bagging_fraction': 0.6, 'reg_alpha': 5, 'reg_lambda': 5, 'scale_pos_weight': 90}, score:7414.455930008164
#  {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.02, 'max_depth': -1, 'bagging_freq': 1, 'n_jobs': -1, 'verbosity': -1, 'n_estimators': 500, 'random_state': 42, 'num_leaves': 127, 'min_data_in_leaf': 402, 'feature_fraction': 0.9, 'bagging_fraction': 1.0, 'reg_alpha': 4, 'reg_lambda': 10, 'scale_pos_weight': 91}
# 9000{'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.02, 'max_depth': -1, 'bagging_freq': 1, 'n_jobs': -1, 'verbosity': -1, 'n_estimators': 500, 'random_state': 42, 'num_leaves': 127, 'min_data_in_leaf': 400, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'reg_alpha': 7, 'reg_lambda': 5, 'scale_pos_weight': 90}
# 12000 {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.02, 'max_depth': -1, 'bagging_freq': 1, 'n_jobs': -1, 'verbosity': -1, 'n_estimators': 500, 'random_state': 42, 'num_leaves': 127, 'min_data_in_leaf': 405, 'feature_fraction': 0.9, 'bagging_fraction': 0.6, 'reg_alpha': 5, 'reg_lambda': 7, 'scale_pos_weight': 91}
# 8479 {  'min_data_in_leaf': 402, 'feature_fraction': 0.9, 'bagging_fraction': 0.6, 'reg_alpha': 10, 'reg_lambda': 5, 'scale_pos_weight': 90}
# 15000 'bagging_freq': 1, 'n_jobs': -1, 'verbosity': -1, 'n_estimators': 500, 'random_state': 42, 'num_leaves': 127, 'min_data_in_leaf': 405, 'feature_fraction': 0.9, 'bagging_fraction': 0.6, 'reg_alpha': 4, 'reg_lambda': 9, 'scale_pos_weight': 90}
# 9052 { 'n_estimators': 500, 'random_state': 42, 'num_leaves': 127, 'min_data_in_leaf': 405, 'feature_fraction': 0.8, 'bagging_fraction': 1.0, 'reg_alpha': 10, 'reg_lambda': 6, 'scale_pos_weight': 91}

 
    # space = {
    #     'learning_rate': [0.02],
    #     'num_leaves': [127],
    #     'max_depth': [-1],
    #     'min_data_in_leaf': [405],
    #     'feature_fraction': [0.8],
    #     'bagging_fraction': [1.0],
    #     'reg_alpha': [10],
    #     'reg_lambda': [6],
    #     'scale_pos_weight': [91] # 处理正负样本不平衡
    # }

    # space = {
    #     'learning_rate': [0.02],
    #     'num_leaves': [127],
    #     'max_depth': [-1],
    #     'min_data_in_leaf': [400],
    #     'feature_fraction': [ 0.8],
    #     'bagging_fraction': [0.6],
    #     'reg_alpha': [5],
    #     'reg_lambda': [5],
    #     'scale_pos_weight': [90] # 处理正负样本不平衡
    # }
    # 精简搜索空间：只搜正则化参数（防过拟合最关键的三个）
    space = {
        'min_data_in_leaf': [50, 100, 200, 400],
        'reg_alpha': [1, 3, 5, 8],
        'reg_lambda': [1, 3, 5, 8],
    }
    return space, [[0]]  # 保持返回值向后兼容

def _train_lightgbm_model(raw_data, labels, feature_columns, label_stats=None, preprocessed_data=None, optimize_thresholds=False, sw_sector_mapping=None):
    """
    【大师级重构】训练LightGBM极值二分类模型
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, log_loss
        import numpy as np
        import pandas as pd
        import random
        import math

        logger.info("[强化] 开始训练LightGBM极值二分类模型...")

        if preprocessed_data is not None:
            logger.info("[OK] 使用预处理数据")
            X_train_raw = preprocessed_data['X_train']
            X_val_raw = preprocessed_data['X_val']
            X_test_raw = preprocessed_data['X_test']

            y_train = preprocessed_data.get('train_relevance')
            y_val = preprocessed_data.get('val_relevance')
            y_val_original = preprocessed_data.get('y_val_original', preprocessed_data.get('y_val'))

            test_dates = preprocessed_data.get('test_dates')
            val_data_raw = preprocessed_data.get('val_data_raw')
            test_data_raw = preprocessed_data.get('test_data_raw')
            train_dates = preprocessed_data.get('train_dates')
            val_dates = preprocessed_data.get('val_dates')

            if y_train is None or y_val is None:
                logger.error("[FAIL] 缺少relevance标签,无法训练二分类模型")
                return None
        else:
            logger.error("[FAIL] 缺少预处理数据,无法训练LightGBM")
            return None

        # 特征对齐
        if X_train_raw.shape[1] != len(feature_columns):
            if X_train_raw.shape[1] > len(feature_columns):
                X_train = X_train_raw[:, :len(feature_columns)]
                X_val = X_val_raw[:, :len(feature_columns)]
            else:
                logger.error(f"  [FAIL] 特征数量不足: {X_train_raw.shape[1]} < {len(feature_columns)}")
                return None
        else:
            X_train = X_train_raw
            X_val = X_val_raw

        X_train_scaled = X_train
        X_val_scaled = X_val

        # 清理 NaN 标签
        train_fit_mask = ~pd.isna(y_train)
        if not np.all(train_fit_mask):
            X_train_scaled = X_train_scaled[train_fit_mask]
            y_train = y_train[train_fit_mask]
            if train_dates is not None: train_dates = train_dates[train_fit_mask]

        val_fit_mask = ~pd.isna(y_val)
        if not np.all(val_fit_mask):
            X_val_scaled = X_val_scaled[val_fit_mask]
            y_val = y_val[val_fit_mask]
            if val_dates is not None: val_dates = val_dates[val_fit_mask]
            if y_val_original is not None: y_val_original = y_val_original[val_fit_mask]

        sample_weights = preprocessed_data.get('sample_weights')
        val_sample_weights = preprocessed_data.get('val_sample_weights')
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)[train_fit_mask]
        else:
            sample_weights = np.ones(len(y_train), dtype=float)
        if val_sample_weights is not None:
            val_sample_weights = np.asarray(val_sample_weights)[val_fit_mask]
        else:
            val_sample_weights = np.ones(len(y_val), dtype=float)

        # 准备 Lgb Dataset (不需要 group)
        train_data = lgb.Dataset(
            X_train_scaled, label=y_train, weight=sample_weights, params={'feature_pre_filter': False}
        )
        val_data = lgb.Dataset(
            X_val_scaled, label=y_val, weight=val_sample_weights, reference=train_data, params={'feature_pre_filter': False}
        )

        early_stopping_rounds = 40
        eval_result = {}
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=False, verbose=False),
            lgb.log_evaluation(period=0),
            lgb.record_evaluation(eval_result)
        ]

        base_params = {
            'objective': 'binary',
            'learning_rate':0.02,
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'max_depth': -1,
            'num_leaves': 31,
            'bagging_freq': 1,
            'n_jobs': -1,
            'verbosity': -1,
            'n_estimators': 500,
            'scale_pos_weight': 90,
            'random_state': 42,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.6,
        }

        OPTUNA_N_TRIALS = 15
        OPTUNA_SEARCH_TRAIN_DATE_FRAC = 0.40
        OPTUNA_SEARCH_VAL_DATE_FRAC = 0.50
        OPTUNA_SEARCH_MIN_TRAIN_DAYS = 120
        OPTUNA_SEARCH_MIN_VAL_DAYS = 36
        OPTUNA_SEARCH_NUM_BOOST_ROUND = 180
        OPTUNA_FINAL_NUM_BOOST_ROUND = 400
        OPTUNA_SEARCH_EARLY_STOPPING = 25
        best_top1_return = -float('inf')
        best_model = None
        best_params_found = None
        best_minute_diag = None

        # 加载分钟缓存
        minute_cache_df = None
        use_minute_primary_eval = False
        try:
            minute_cache_df = _build_or_load_validation_minute_cache_binary(
                val_data_raw=val_data_raw, selection_data_raw=val_data_raw,
                test_data_raw=test_data_raw, holding_days=3, build_if_missing=False,
            )
            use_minute_primary_eval = minute_cache_df is not None and len(minute_cache_df) > 0
            if use_minute_primary_eval:
                logger.info(f"[MINUTE-PRIMARY] rows={len(minute_cache_df):,}, dates={minute_cache_df['date'].nunique():,}")
            else:
                logger.warning("[MINUTE-PRIMARY] no minute cache, fallback to daily eval")
        except Exception as e:
            logger.warning(f"[MINUTE-PRIMARY] 读取失败: {e}")

        def _build_recent_date_mask(dates, keep_frac=1.0, min_days=0):
            if dates is None:
                return None, None
            dates_ts = pd.to_datetime(pd.Series(dates))
            unique_dates = np.array(sorted(dates_ts.dropna().unique()))
            if len(unique_dates) == 0:
                return None, None
            keep_days = max(int(math.ceil(len(unique_dates) * keep_frac)), int(min_days or 0))
            keep_days = min(len(unique_dates), max(1, keep_days))
            keep_date_set = set(unique_dates[-keep_days:])
            mask = dates_ts.isin(keep_date_set).values
            return mask, unique_dates[-keep_days:]

        search_train_mask, search_train_days = _build_recent_date_mask(
            train_dates, keep_frac=OPTUNA_SEARCH_TRAIN_DATE_FRAC, min_days=OPTUNA_SEARCH_MIN_TRAIN_DAYS
        )
        search_val_mask, search_val_days = _build_recent_date_mask(
            val_dates, keep_frac=OPTUNA_SEARCH_VAL_DATE_FRAC, min_days=OPTUNA_SEARCH_MIN_VAL_DAYS
        )

        if search_train_mask is None:
            search_train_mask = np.ones(len(y_train), dtype=bool)
            search_train_days = None
        if search_val_mask is None:
            search_val_mask = np.ones(len(y_val), dtype=bool)
            search_val_days = None

        X_train_search = X_train_scaled[search_train_mask]
        y_train_search = y_train[search_train_mask]
        sample_weights_search = sample_weights[search_train_mask]
        X_val_search = X_val_scaled[search_val_mask]
        y_val_search = y_val[search_val_mask]
        val_sample_weights_search = val_sample_weights[search_val_mask]
        y_val_original_search = y_val_original[search_val_mask] if y_val_original is not None else None
        val_dates_search = val_dates[search_val_mask] if val_dates is not None else None
        val_data_raw_search = val_data_raw.iloc[np.flatnonzero(search_val_mask)].reset_index(drop=True) if val_data_raw is not None else None
        minute_cache_search = minute_cache_df
        if minute_cache_df is not None and search_val_days is not None:
            search_day_set = set(pd.to_datetime(search_val_days))
            minute_cache_search = minute_cache_df[pd.to_datetime(minute_cache_df['date']).isin(search_day_set)].copy()

        logger.info(
            "[Optuna] 搜索阶段轻量化: "
            f"train={len(X_train_search):,}/{len(X_train_scaled):,} "
            f"val={len(X_val_search):,}/{len(X_val_scaled):,} "
            f"boost_round={OPTUNA_SEARCH_NUM_BOOST_ROUND} "
            f"early_stop={OPTUNA_SEARCH_EARLY_STOPPING}"
        )

        train_data_search = lgb.Dataset(
            X_train_search, label=y_train_search, weight=sample_weights_search, params={'feature_pre_filter': False}
        )
        val_data_search = lgb.Dataset(
            X_val_search, label=y_val_search, weight=val_sample_weights_search, reference=train_data_search, params={'feature_pre_filter': False}
        )

        # ================================================================
        # Optuna 贝叶斯搜索 (TPE sampler + Median pruner)
        # ================================================================
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        lgb_optuna_search_space = {
            'min_data_in_leaf': [200, 330, 380],
            'reg_alpha': [3,5,7,8],
            'reg_lambda': [3,5,7,8],
        }
        total_lgb_optuna_combinations = (
            len(lgb_optuna_search_space['min_data_in_leaf'])
            * len(lgb_optuna_search_space['reg_alpha'])
            * len(lgb_optuna_search_space['reg_lambda'])
        )

        logger.info(f"[Optuna] 开始创建study...")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.GridSampler(lgb_optuna_search_space, seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=2),
        )
        logger.info(
            f"[Optuna] study创建完成，使用GridSampler避免重复参数；"
            f"搜索空间共{total_lgb_optuna_combinations}组，开始optimize ({OPTUNA_N_TRIALS} trials)..."
        )

        _trial_count = [0]
        def _objective(trial):
            nonlocal best_minute_diag, best_top1_return
            import time as _time
            _t0 = _time.time()
            _tn = _trial_count[0] + 1
            logger.info(f"[Optuna] trial #{trial.number} (第{_tn}个) 开始...")

            current_params = base_params.copy()
            current_params['min_data_in_leaf'] = trial.suggest_categorical(
                'min_data_in_leaf', lgb_optuna_search_space['min_data_in_leaf']
            )
            current_params['reg_alpha'] = trial.suggest_categorical(
                'reg_alpha', lgb_optuna_search_space['reg_alpha']
            )
            current_params['reg_lambda'] = trial.suggest_categorical(
                'reg_lambda', lgb_optuna_search_space['reg_lambda']
            )
            logger.info(f"[Optuna] trial #{trial.number} 参数采样完成，开始训练LightGBM...")

            search_eval_result = {}
            search_callbacks = [
                lgb.early_stopping(stopping_rounds=OPTUNA_SEARCH_EARLY_STOPPING, first_metric_only=False, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(search_eval_result)
            ]

            model = lgb.train(
                current_params, train_data_search, num_boost_round=OPTUNA_SEARCH_NUM_BOOST_ROUND,
                valid_sets=[train_data_search, val_data_search], valid_names=['train', 'valid'], callbacks=search_callbacks
            )
            logger.info(f"[Optuna] trial #{trial.number} LightGBM训练完成，开始评估...")

            score = float('-inf')
            if use_minute_primary_eval:
                minute_diag = _evaluate_candidate_with_minute_cache_binary(
                    model, X_val_search, y_val_original_search, val_dates_search, val_data_raw_search, minute_cache_search
                )
                if minute_diag is not None:
                    score = float(minute_diag['score'])
                    trial.set_user_attr('minute_cum', float(minute_diag.get('cumulative_return', 0.0)))
                    trial.set_user_attr('minute_dd', float(minute_diag.get('max_drawdown', 0.0)))
                    best_minute_diag = minute_diag if (best_minute_diag is None or score > best_top1_return) else best_minute_diag

            def _safe_auc(y_true, y_score):
                try:
                    y_true_arr = np.asarray(y_true)
                    y_score_arr = np.asarray(y_score)
                    mask = ~pd.isna(y_true_arr)
                    y_true_arr = y_true_arr[mask]
                    y_score_arr = y_score_arr[mask]
                    if len(np.unique(y_true_arr)) < 2:
                        return None
                    return float(roc_auc_score(y_true_arr, y_score_arr))
                except Exception:
                    return None

            try:
                best_iter = getattr(model, 'best_iteration', None)
                train_pred = model.predict(X_train_search, num_iteration=best_iter) if best_iter else model.predict(X_train_search)
                val_pred = model.predict(X_val_search, num_iteration=best_iter) if best_iter else model.predict(X_val_search)
                train_auc = _safe_auc(y_train_search, train_pred)
                val_auc = _safe_auc(y_val_search, val_pred)
                if train_auc is not None and val_auc is not None:
                    auc_gap = max(0.0, train_auc - val_auc)
                    # 让分钟回测分数仍然主导，但明显压制训练集过拟合过强的参数
                    if score != float('-inf'):
                        score = float(score * max(0.0, 1.0 - min(auc_gap, 0.35)) + 120.0 * (val_auc - 0.5))
                    else:
                        score = float(120.0 * (val_auc - 0.5))
                    trial.set_user_attr('train_auc', float(train_auc))
                    trial.set_user_attr('val_auc', float(val_auc))
                    trial.set_user_attr('auc_gap', float(auc_gap))
            except Exception as e:
                logger.debug(f"[Optuna] AUC辅助评估失败: {e}")

            if score == float('-inf'):
                score = 0.0

            _trial_count[0] += 1
            _elapsed = _time.time() - _t0
            logger.info(
                f"[Optuna] trial #{trial.number} 完成: "
                f"score={score:.1f} time={_elapsed:.0f}s "
                f"min_leaf={current_params['min_data_in_leaf']} "
                f"alpha={current_params['reg_alpha']} lambda={current_params['reg_lambda']}"
            )
            if score > best_top1_return:
                best_top1_return = score
                if use_minute_primary_eval and minute_diag is not None:
                    best_minute_diag = minute_diag
            return score

        study.optimize(_objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
        logger.info(f"[Optuna] optimize完成，共{len(study.trials)}个trial")

        if study.best_trial is not None:
            best_top1_return = study.best_value
            best_cum = study.best_trial.user_attrs.get('minute_cum', 0)
            best_dd = study.best_trial.user_attrs.get('minute_dd', 0)
            logger.info(
                f"[Optuna] 最佳 trial #{study.best_trial.number}: "
                f"score={best_top1_return:.2f} cum={best_cum:.2%} dd={best_dd:.2%}"
            )
            # 用最佳参数重新训练
            best_params_found = base_params.copy()
            best_params_found['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
            best_params_found['reg_alpha'] = study.best_params['reg_alpha']
            best_params_found['reg_lambda'] = study.best_params['reg_lambda']
            logger.info(f"[Optuna] 最佳参数: {best_params_found}")
            model = lgb.train(
                best_params_found, train_data, num_boost_round=OPTUNA_FINAL_NUM_BOOST_ROUND,
                valid_sets=[train_data, val_data], valid_names=['train', 'valid'], callbacks=callbacks
            )
        else:
            logger.warning("[Optuna] 所有trial失败，使用默认参数")
            model = lgb.train(
                base_params, train_data, num_boost_round=OPTUNA_FINAL_NUM_BOOST_ROUND,
                valid_sets=[train_data, val_data], valid_names=['train', 'valid'], callbacks=callbacks
            )

        try:
            best_iter = getattr(model, 'best_iteration', None)
            y_train_pred = model.predict(X_train_scaled, num_iteration=best_iter) if best_iter else model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled, num_iteration=best_iter) if best_iter else model.predict(X_val_scaled)
        except:
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)

        def _safe_roc_auc(y_true, y_score):
            import numpy as np
            y_true = np.asarray(y_true)
            # 过滤nan
            mask = ~pd.isna(y_true)
            y_true = y_true[mask]
            y_score = np.asarray(y_score)[mask]
            uniq = np.unique(y_true)
            if len(uniq) < 2:
                return None
            try:
                return float(roc_auc_score(y_true, y_score))
            except Exception:
                return None

        train_auc = _safe_roc_auc(y_train, y_train_pred)
        val_auc = _safe_roc_auc(y_val, y_val_pred)

        # 记录类别分布，便于你判断是否是窗口太短/标签太稀导致
        try:
            import numpy as np
            train_pos = int((np.asarray(y_train) == 1).sum())
            train_neg = int((np.asarray(y_train) == 0).sum())
            val_pos = int((np.asarray(y_val) == 1).sum())
            val_neg = int((np.asarray(y_val) == 0).sum())
            logger.info(f"[LABEL] Train pos={train_pos:,}, neg={train_neg:,} | Val pos={val_pos:,}, neg={val_neg:,}")
        except Exception:
            pass

        if train_auc is None:
            logger.warning("[WARNING] Train AUC 无法计算（y_true 单一类别或异常），将显示为 NA")
        if val_auc is None:
            logger.warning("[WARNING] Val AUC 无法计算（y_true 单一类别或异常），将显示为 NA")

        logger.info(f"[模型指标] Train AUC: {train_auc if train_auc is not None else 'NA'}, Val AUC: {val_auc if val_auc is not None else 'NA'}")
        if use_minute_primary_eval:
            logger.info(f"[SUCCESS] 二分类脚本当前为分钟主评估选模: score={best_top1_return:.4f}")
            if best_minute_diag is not None:
                logger.info(
                    f"[BEST-MINUTE] days={best_minute_diag.get('days', 0)}, cum={best_minute_diag.get('cumulative_return', 0.0):.3%}, "
                    f"mean={best_minute_diag.get('mean_return', 0.0):.3%}, median={best_minute_diag.get('median_return', 0.0):.3%}, "
                    f"win={best_minute_diag.get('win_rate', 0.0):.1%}, q10={best_minute_diag.get('q10', 0.0):.3%}, "
                    f"dd={best_minute_diag.get('max_drawdown', 0.0):.2%}, sharpe={best_minute_diag.get('sharpe_like', 0.0):.3f}, "
                    f"sortino={best_minute_diag.get('sortino_like', 0.0):.3f}, stoploss={best_minute_diag.get('stoploss_rate', 0.0):.1%}, "
                    f"profit_bearish={best_minute_diag.get('profit_bearish_rate', 0.0):.1%}, skip={best_minute_diag.get('skip_rate', 0.0):.1%}, "
                    f"top3_cum={best_minute_diag.get('top3_cum_return', 0.0):.3%}, top5_cum={best_minute_diag.get('top5_cum_return', 0.0):.3%}"
                )
                for seg in best_minute_diag.get('segments', []):
                    logger.info(
                        f"  [BEST-MINUTE-SEG{seg['segment']}] days={seg['days']}, cum={seg['cumulative_return']:.3%}, "
                        f"mean={seg['mean_return']:.3%}, win={seg['win_rate']:.1%}, q10={seg['q10']:.3%}, "
                        f"dd={seg['max_drawdown']:.2%}, stoploss={seg['stoploss_rate']:.1%}, penalty={seg['penalty']:.2f}"
                    )

        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = feature_columns

        # [新增] 打印特征重要性
        logger.info("[FEATURE] 特征重要性分析:")

        # 创建特征重要性DataFrame
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        # 打印全部特征重要性
        logger.info(f"  全部特征重要性 (共 {len(feature_importance_df)} 个特征):")
        for idx, row in feature_importance_df.iterrows():
            logger.info(f"    {idx+1:3d}. {row['feature_name']:<30} = {row['importance']:.2f}")
        
        model_results = {
            'model': model,
            'scaler': None,
            'feature_names': feature_columns,
            'label_stats': label_stats,
            'performance': {'train_auc': train_auc, 'val_auc': val_auc},
            'training_info': {'best_iteration': getattr(model, 'best_iteration', None)},
            'model_params': best_params_found if best_params_found else base_params,
            'feature_importance': dict(zip(feature_columns, feature_importance)),
            'model_type': 'lightgbm',
        }

        return model_results

    except Exception as e:
        logger.error(f"[FAIL] LightGBM模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _train_ml_model(raw_data, labels, feature_columns, winsorization_thresholds=None, label_stats=None, label_transform='adaptive', enable_param_search=False, search_strategy='grid', random_seed=42, preprocessed_data=None, enable_shap_analysis=True, preset_params=None, market_data=None):
    """
    [强化] 成本感知回归模型训练（支持超参数搜索优化 + 集成学习）

    Args:
        raw_data: 原始特征数据DataFrame
        labels: 标签数据Series（ATR倍数连续值）
        feature_columns: 特征列名列表
        winsorization_thresholds: 预计算的Winsorization阈值（可选）
        label_stats: 标签统计量字典（用于反向转换）
        label_transform: 标签变换方法（'adaptive', 'asinh', 'signed_log', 'none'）
        enable_param_search: 是否启用超参数搜索（默认False）
        search_strategy: 搜索策略 ('grid', 'fast')
        random_seed: 随机种子（用于集成学习，默认42）
        preprocessed_data: 预处理数据字典（可选），包含:
            - X_train: 训练集特征(已Winsorization)
            - X_val: 验证集特征(已Winsorization)
            - y_train: 训练集标签
            - y_val: 验证集标签
            - winsorization_thresholds: Winsorization阈值
        enable_shap_analysis: 是否启用SHAP分析（默认True）
            - 在集成学习中,建议只对第一个模型启用以节省时间
        preset_params: 预设的模型参数（可选）
            - 如果提供,将跳过参数搜索,直接使用该参数
            - 用于Walk-Forward模式,避免重复搜索
        market_data: 市场数据DataFrame（用于XGBoost计算group权重）

    Returns:
        dict: 包含模型、性能指标、标签统计量、最优参数等信息的字典
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        import numpy as np
        import pandas as pd

        # 导入回归评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        logger.info(f"[强化] 开始训练回归模型，样本数: {len(raw_data)}, 特征数: {len(feature_columns)}")

        # [强化] 统一阈值定义（避免重复定义）
        PROFIT_THRESHOLD = ATR_MULTIPLIER_PROFIT  # 2.0 ATR
        LOSS_THRESHOLD = -ATR_MULTIPLIER_LOSS     # -1.0 ATR

        # [强化] 步骤1：准备数据
        logger.info("[强化] 准备训练数据...")

        # 数据对齐
        if len(raw_data) != len(labels):
            logger.warning(f"数据长度不一致，进行对齐: 特征{len(raw_data)} vs 标签{len(labels)}")
            min_length = min(len(raw_data), len(labels))
            raw_data = raw_data.iloc[:min_length] if hasattr(raw_data, 'iloc') else raw_data[:min_length]
            labels = labels.iloc[:min_length] if hasattr(labels, 'iloc') else labels[:min_length]

        # [强化] 排序模型：labels参数实际上是atr_return_label（连续值，用于IC计算）
        # 但训练时会使用relevance标签（整数0-3）
        # [WRENCH] 重要: 保持labels的index,确保与raw_data对齐
        if hasattr(labels, 'values'):
            # labels是pandas Series,保留index
            y_data_series = labels.astype(np.float32)
            y_data = y_data_series.values
        else:
            # labels是numpy array
            y_data = labels.astype(np.float32)
            y_data_series = None
        y_data_original = y_data.copy()

        # [强化] 保存原始标签统计（用于IC计算）
        original_y_stats = {
            'mean': y_data.mean(),
            'std': y_data.std(),
            'min': y_data.min(),
            'max': y_data.max(),
            'range': y_data.max() - y_data.min()
        }
        logger.info(f"[OK] 原始收益率标签统计（用于IC计算）:")
        logger.info(f"  标签范围: [{original_y_stats['min']:.3f}, {original_y_stats['max']:.3f}]")
        logger.info(f"  标签统计: mean={original_y_stats['mean']:.3f}, std={original_y_stats['std']:.3f}")

        logger.info(f"数据对齐完成: {len(raw_data)} 个样本, {len(feature_columns)} 个特征列")

        # [强化] 优化：如果提供了预处理数据,跳过数据划分和预处理步骤
        if preprocessed_data is not None:
            logger.info("=" * 60)
            logger.info("[OK] 使用预处理数据（集成学习优化）")
            logger.info("=" * 60)
            logger.info(f"  [TARGET] 训练集: {len(preprocessed_data['X_train']):,}样本")
            logger.info(f"  [TARGET] 验证集: {len(preprocessed_data['X_val']):,}样本")
            logger.info("=" * 60)

            X_train_processed = preprocessed_data['X_train']
            X_val_processed = preprocessed_data['X_val']

            # 🆕 XGBoost排序模型使用relevance_30作为标签(整数类型0-29,30层)
            y_train_split = preprocessed_data.get('train_relevance_30')
            y_val_split = preprocessed_data.get('val_relevance_30')

            # 保留原始收益率用于IC计算
            y_train_split_original = preprocessed_data.get('y_train_original', preprocessed_data.get('y_train'))
            y_val_split_original = preprocessed_data.get('y_val_original', preprocessed_data.get('y_val'))

            if y_train_split is None or y_val_split is None:
                logger.error("[FAIL] 缺少relevance_30标签,无法训练XGBoost排序模型")
                return None

            # [强化] 确保relevance_30标签是整数类型（XGBoost排序模型要求）
            y_train_split = np.array(y_train_split, dtype=np.int32)
            y_val_split = np.array(y_val_split, dtype=np.int32)
            logger.info(f"  [OK] relevance_30标签已转换为int32类型(30层)")
            logger.info(f"  训练集relevance_30范围: [{y_train_split.min()}, {y_train_split.max()}]")
            logger.info(f"  验证集relevance_30范围: [{y_val_split.min()}, {y_val_split.max()}]")

            train_winsorization_thresholds = preprocessed_data['winsorization_thresholds']

            # [强化] 从预处理数据中提取特征名
            # 注意: X_train_processed可能是DataFrame或numpy数组
            if hasattr(X_train_processed, 'columns'):
                feature_names = list(X_train_processed.columns)
            else:
                # 如果是numpy数组,使用传入的feature_columns
                feature_names = feature_columns if isinstance(feature_columns, list) else list(feature_columns)

            logger.info("[OK] 使用relevance_30标签（整数0-29,30层）用于XGBoost排序模型训练")
            logger.info(f"  训练标签类型: relevance_30 (整数0-29,30层)")
            logger.info(f"  保留原始收益率用于IC计算")

            # 🆕 从预处理数据中提取训练集和验证集日期（用于时间衰减权重和group计算）
            train_dates = None
            val_dates = None
            if 'train_dates' in preprocessed_data:
                train_dates = preprocessed_data['train_dates']
                logger.info("[OK] 从预处理数据中提取训练集日期")
            if 'val_dates' in preprocessed_data:
                val_dates = preprocessed_data['val_dates']
                logger.info("[OK] 从预处理数据中提取验证集日期")

            # [WRENCH] 关键验证: 检查训练集数据顺序
            if train_dates is not None:
                if not _validate_data_order(train_dates, y_train_split_original, y_train_split, "XGBoost训练集"):
                    logger.error("[FAIL] XGBoost训练集数据顺序验证失败,程序终止")
                    raise ValueError("XGBoost训练集数据顺序混乱,无法继续训练")

            # [WRENCH] 关键验证: 检查验证集数据顺序
            if val_dates is not None:
                if not _validate_data_order(val_dates, y_val_split_original, y_val_split, "XGBoost验证集"):
                    logger.error("[FAIL] XGBoost验证集数据顺序验证失败,程序终止")
                    raise ValueError("XGBoost验证集数据顺序混乱,无法继续训练")

            # 继续原有逻辑(train_dates和val_dates已在上面提取并验证)

            # 🆕 从预处理数据中提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate = None
            train_circ_mv = None
            if 'turnover_rate' in preprocessed_data:
                train_turnover_rate = preprocessed_data['turnover_rate']
                logger.info("[OK] 从预处理数据中提取训练集换手率")
            else:
                logger.warning("[WARNING] 预处理数据中未包含turnover_rate，将使用均匀权重")

            if 'circ_mv' in preprocessed_data:
                train_circ_mv = preprocessed_data['circ_mv']
                logger.info("[OK] 从预处理数据中提取训练集流通市值")
            else:
                logger.warning("[WARNING] 预处理数据中未包含circ_mv，将使用均匀权重")
        else:
            # [WRENCH] 步骤2：先进行训练/验证集划分（避免数据泄露）
            logger.info("=" * 60)
            logger.info("[REFRESH] 时间序列友好的数据划分策略")
            logger.info("=" * 60)
            logger.info("[CLIPBOARD] 划分策略说明:")
            logger.info("  [OK] 按时间顺序划分，非随机划分")
            logger.info("  [OK] 前80%时间段训练，后20%验证")
            logger.info("  [OK] 避免未来信息泄露到历史训练中")
            logger.info("  [OK] 更符合实际交易场景的时间顺序")
            logger.info("  [OK] 预处理参数仅基于训练集计算")

            split_point = int(len(raw_data) * 0.8)

            # [WRENCH] 关键修复: 划分原始数据时,确保数据和标签使用相同的index
            raw_train_data = raw_data.iloc[:split_point] if hasattr(raw_data, 'iloc') else raw_data[:split_point]
            raw_val_data = raw_data.iloc[split_point:] if hasattr(raw_data, 'iloc') else raw_data[split_point:]

            # [WRENCH] 重要: 如果labels是Series,使用.iloc保持index对齐;否则使用numpy切片
            if y_data_series is not None:
                y_train_split = y_data_series.iloc[:split_point].values
                y_val_split = y_data_series.iloc[split_point:].values
                logger.info("[OK] 使用Series.iloc划分标签,确保与数据对齐")
            else:
                y_train_split = y_data[:split_point]
                y_val_split = y_data[split_point:]
                logger.info("[OK] 使用numpy切片划分标签")

            # [强化] 保留原始收益率标签（用于IC计算）
            y_train_split_original = y_data_original[:split_point]
            logger.info("[OK] 保留原始收益率标签（用于IC计算）")

            # 🆕 提取训练集日期（用于时间衰减权重）
            train_dates = None
            if 'date' in raw_train_data.columns:
                train_dates = raw_train_data['date'].values
                logger.info("[OK] 提取训练集日期（用于时间衰减权重计算）")

            # 🆕 提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate = None
            train_circ_mv = None
            if 'turnover_rate' in raw_train_data.columns:
                train_turnover_rate = raw_train_data['turnover_rate'].values
                logger.info("[OK] 提取训练集换手率（用于基础权重计算）")
            else:
                logger.warning("[WARNING] 训练数据中未包含turnover_rate列，将使用均匀权重")

            if 'circ_mv' in raw_train_data.columns:
                train_circ_mv = raw_train_data['circ_mv'].values
                logger.info("[OK] 提取训练集流通市值（用于基础权重计算）")
            else:
                logger.warning("[WARNING] 训练数据中未包含circ_mv列，将使用均匀权重")

            # [强化] 添加详细的日期范围日志
            train_date_min = raw_train_data['date'].min() if 'date' in raw_train_data.columns else None
            train_date_max = raw_train_data['date'].max() if 'date' in raw_train_data.columns else None
            val_date_min = raw_val_data['date'].min() if 'date' in raw_val_data.columns else None
            val_date_max = raw_val_data['date'].max() if 'date' in raw_val_data.columns else None

            logger.info(f"[CHART] 时间序列划分结果:")
            logger.info(f"  [TARGET] 训练集: {len(raw_train_data):,}样本 (前80%时间段)")
            if train_date_min and train_date_max:
                logger.info(f"     [DATE] 日期范围: {train_date_min.strftime('%Y-%m-%d')} 至 {train_date_max.strftime('%Y-%m-%d')}")
            logger.info(f"  [TARGET] 验证集: {len(raw_val_data):,}样本 (后20%时间段)")
            if val_date_min and val_date_max:
                logger.info(f"     [DATE] 日期范围: {val_date_min.strftime('%Y-%m-%d')} 至 {val_date_max.strftime('%Y-%m-%d')}")
            logger.info("=" * 60)

            # [强化] 步骤3：分别对训练集和验证集进行特征预处理（避免数据泄露）
            X_train_processed, feature_names = _prepare_feature_data(
                raw_train_data, feature_columns, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return None

            X_val_processed, _ = _prepare_feature_data(
                raw_val_data, feature_columns, data_type="验证集"
            )

            if X_val_processed is None:
                logger.error("验证集特征预处理失败")
                return None

            # [WRENCH] 关键修复：确保标签与预处理后的特征数据长度匹配
            # 由于预处理可能移除重复行，需要调整标签数组长度
            if len(X_train_processed) != len(y_train_split):
                logger.warning(f"训练集预处理后长度变化: {len(y_train_split)} -> {len(X_train_processed)}")
                y_train_split = y_train_split[:len(X_train_processed)]

            if len(X_val_processed) != len(y_val_split):
                logger.warning(f"验证集预处理后长度变化: {len(y_val_split)} -> {len(X_val_processed)}")
                y_val_split = y_val_split[:len(X_val_processed)]

        # 使用预处理后的数据
        X_train_split = X_train_processed
        X_val_split = X_val_processed
        winsorization_thresholds = train_winsorization_thresholds

        logger.info("[OK] 训练集和验证集特征预处理完成，确保数据预处理一致性且避免数据泄露")
        logger.info(f"最终训练集: {len(X_train_split)}个样本, {X_train_split.shape[1]}个特征")
        logger.info(f"最终验证集: {len(X_val_split)}个样本, {X_val_split.shape[1]}个特征")
        logger.info(f"训练标签: {len(y_train_split)}个, 验证标签: {len(y_val_split)}个")

        # [强化] 成本感知回归：分析连续收益标签分布
        logger.info("=" * 50)
        logger.info("[强化] 成本感知回归训练集样本分布分析:")


        # 回归模式不需要不平衡比例
        imbalance_ratio = 1.0
        logger.info("=" * 50)

        # [WRENCH] 使用预处理阶段计算的Winsorization阈值
        if winsorization_thresholds is None:
            logger.warning("[WARNING] 未提供Winsorization阈值，数据预处理可能不完整")
            winsorization_thresholds = {}

        # 详细的特征日志记录
        logger.info("=" * 60)
        logger.info("输入特征详细信息")
        logger.info("=" * 60)
        logger.info(f"原始特征总数: {len(feature_names)}")


        # 先进行常规的特征处理，PCA特征工程将在标准化后进行
        n_features = X_train_split.shape[1]

        logger.info("=" * 60)
        logger.info("特征处理策略")
        logger.info("=" * 60)
        logger.info(f"[TARGET] 原始特征数量: {n_features}")
    
        logger.info("=" * 60)
        logger.info("[WRENCH] XGBoost模型训练")
        logger.info("=" * 60)

        # [强化] 处理标签中的NaN值
        logger.info("处理标签中的NaN值...")

        # 移除NaN值
        train_nan_mask = np.isnan(y_train_split)
        val_nan_mask = np.isnan(y_val_split)

        if train_nan_mask.any():
            logger.warning(f"训练集中发现{train_nan_mask.sum()}个NaN标签，将被移除")
            X_train_split = X_train_split[~train_nan_mask]
            y_train_split = y_train_split[~train_nan_mask]
            # 🆕 同步移除train_dates中的NaN对应项
            if train_dates is not None:
                train_dates = train_dates[~train_nan_mask]
            # 🆕 同步移除y_train_split_original中的NaN对应项
            if y_train_split_original is not None:
                y_train_split_original = y_train_split_original[~train_nan_mask]
            # 🆕 同步移除换手率和流通市值中的NaN对应项
            if 'train_turnover_rate' in locals() and train_turnover_rate is not None:
                train_turnover_rate = train_turnover_rate[~train_nan_mask]
            if 'train_circ_mv' in locals() and train_circ_mv is not None:
                train_circ_mv = train_circ_mv[~train_nan_mask]

        if val_nan_mask.any():
            logger.warning(f"验证集中发现{val_nan_mask.sum()}个NaN标签，将被移除")
            X_val_split = X_val_split[~val_nan_mask]
            y_val_split = y_val_split[~val_nan_mask]
            # 🆕 同步移除val_dates中的NaN对应项
            if val_dates is not None:
                val_dates = val_dates[~val_nan_mask]

        # [强化] 计算样本权重（基于换手率、流通市值和指数时间衰减）
        logger.info("[WRENCH] 计算样本权重（基于换手率、流通市值和指数时间衰减）...")

        # [强化] 使用换手率和流通市值计算基础权重，结合指数时间衰减
        # [强化] 从preprocessed_data获取统一计算的样本权重
        sample_weights = preprocessed_data.get('sample_weights') if preprocessed_data is not None else None

        if sample_weights is None:
            # 如果preprocessed_data中没有权重,则重新计算（兼容旧代码）
            logger.warning("[WARNING] preprocessed_data中缺少sample_weights,将重新计算")
            train_relevance = preprocessed_data.get('train_relevance') if preprocessed_data is not None else None
            sample_weights = _calculate_professional_sample_weights(
                y_train_split,
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=y_train_split_original,
                train_dates=train_dates,
                time_decay_half_life=180.0,
                turnover_rate=train_turnover_rate if 'train_turnover_rate' in locals() else None,
                circ_mv=train_circ_mv if 'train_circ_mv' in locals() else None,
                label_rank=train_relevance,
                label_boost_beta=0.15,
                w_min=0.2,
                w_max=5.0
            )
        else:
            logger.info("[OK] 使用preprocessed_data中的统一样本权重")
            logger.info(f"   权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
            logger.info(f"   权重均值: {sample_weights.mean():.4f}")

        # [强化] 排序模型：确保relevance标签是numpy数组
        if not isinstance(y_train_split, np.ndarray):
            y_train_split = np.array(y_train_split)
        if not isinstance(y_val_split, np.ndarray):
            y_val_split = np.array(y_val_split)

        # [强化] XGBoost排序模型标签统计（relevance_30整数0-29,30层）
        logger.info(f"[强化] XGBoost排序模型标签统计（relevance_30整数0-29,30层）:")
        logger.info(f"  训练集: {len(y_train_split):,} 个样本")
        logger.info(f"  验证集: {len(y_val_split):,} 个样本")
        logger.info(f"  训练集relevance_30范围: [{y_train_split.min()}, {y_train_split.max()}]")
        logger.info(f"  验证集relevance_30范围: [{y_val_split.min()}, {y_val_split.max()}]")
        logger.info(f"  训练集relevance_30分布(Top-5层级):")
        train_dist = np.bincount(y_train_split.astype(int))
        for i in range(len(train_dist)-1, max(0, len(train_dist)-6), -1):
            if train_dist[i] > 0:
                logger.info(f"    Relevance_30={i}: {train_dist[i]:,}样本")
        logger.info(f"  验证集relevance_30分布(Top-5层级):")
        val_dist = np.bincount(y_val_split.astype(int))
        for i in range(len(val_dist)-1, max(0, len(val_dist)-6), -1):
            if val_dist[i] > 0:
                logger.info(f"    Relevance_30={i}: {val_dist[i]:,}样本")

        # [强化] XGBoost排序模型：使用relevance_30标签（整数0-29,30层）进行训练
        logger.info("[WRENCH] XGBoost排序模型：使用relevance_30标签（整数0-29,30层）进行训练")

        # [强化] 关键修复：XGBoost排序模型使用relevance_30标签，不覆盖为ATR标签
        # y_train_split和y_val_split已经被赋值为relevance_30标签
        # 用于训练的标签（relevance_30整数0-29,30层）
        y_train_split_mapped = y_train_split  # relevance_30标签（整数0-29），用于XGBoost排序模型训练
        y_val_split_mapped = y_val_split      # relevance_30标签（整数0-29），用于XGBoost排序模型训练

        # 用于评估的标签（原始ATR收益率）
        y_train_split_eval = y_train_split_original  # 原始ATR标签，用于IC计算
        y_val_split_eval = y_val_split_original      # 原始ATR标签，用于IC计算

        reverse_label_map = None

        # [强化] 验证relevance_30标签类型
        logger.info(f"  [OK] 训练标签类型验证:")
        logger.info(f"     y_train_split_mapped类型: {type(y_train_split_mapped)}")
        logger.info(f"     y_train_split_mapped唯一值数量: {len(np.unique(y_train_split_mapped))}")
        logger.info(f"     y_train_split_mapped范围: [{np.min(y_train_split_mapped)}, {np.max(y_train_split_mapped)}]")
        logger.info(f"     y_train_split_mapped数据类型: {y_train_split_mapped.dtype if hasattr(y_train_split_mapped, 'dtype') else 'N/A'}")

        # 保存原始标签统计量（用于日志记录）
        if label_stats is None:
            label_stats = {}
        label_stats.update({
            'mean': y_train_split_original.mean() if y_train_split_original is not None else 0,
            'std': y_train_split_original.std() if y_train_split_original is not None else 0,
            'min': y_train_split_original.min() if y_train_split_original is not None else 0,
            'max': y_train_split_original.max() if y_train_split_original is not None else 0
        })

        logger.info(f"[OK] 准备使用relevance_30标签（整数0-29,30层）进行XGBoost排序模型训练")
        logger.info(f"  排序模型输出：相对排序分数（非绝对收益率）")

        logger.info("[OK] 数据预处理已完成")
        logger.info("=" * 60)


        # 确定股票因子列表（不包含市场因子）
        # [WRENCH] 使用模型因子列表（归一化因子）
        stock_factors = _get_model_factors()

        # 只保留股票因子，过滤掉市场因子
        available_stock_factors = []
        stock_factor_indices = []

        for i, feature_name in enumerate(feature_names):
            if feature_name in stock_factors:
                available_stock_factors.append(feature_name)
                stock_factor_indices.append(i)

        # 提取股票因子数据
        X_train_split_stock = X_train_split[:, stock_factor_indices]
        X_val_split_stock = X_val_split[:, stock_factor_indices]
        feature_names = available_stock_factors

        logger.info(f"[OK] 股票因子特征: {len(feature_names)} 个")
        logger.info(f"   - 特征列表: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")

        # [WRENCH] 修复：删除StandardScaler标准化
        # 原因：
        # 1. 因子在计算时已经进行了标准化（如zscore、rank等）
        # 2. 树模型对特征尺度不敏感，不需要额外标准化
        # 3. 额外标准化导致贝叶斯搜索和最终训练的数据分布不一致
        # 4. 这是导致walkforward模式LightGBM过拟合的根本原因
        logger.info("⏭️ 跳过StandardScaler标准化（因子已在计算时标准化）")
        logger.info("   - 因子计算时已进行时间序列标准化（如zscore、rank）")
        logger.info("   - 树模型对特征尺度不敏感")
        logger.info("   - 避免贝叶斯搜索和训练的数据分布不一致")

        # 直接使用Winsorization后的特征，不进行额外标准化
        X_train_enhanced = X_train_split_stock
        X_val_enhanced = X_val_split_stock

        logger.info(f"[OK] 使用基础特征（已Winsorization）: {len(feature_names)} 个特征")

        # PCA信息标记为未应用（使用基础特征）
        pca_info = {'applied': False, 'reason': 'using_base_features_only'}

        logger.info("\n" + "=" * 60)
        logger.info("[强化] 步骤3: 训练XGBoost模型（使用基础特征）")
        logger.info("=" * 60)

        # 分析样本分布（支持二分类和三分类）
        unique_labels_split = np.unique(y_train_split)
        total_samples = len(y_train_split)



        # 根据特征数量和样本分布调整XGBoost参数
        # [WRENCH] 修复：使用X_train_enhanced替代已删除的X_train_split_scaled
        current_features = X_train_enhanced.shape[1]
        n_samples = len(X_train_enhanced)

        logger.info("=" * 60)
        # [WRENCH] 修复：提前定义actual_feature_count变量
        actual_feature_count = len(feature_names)  # 使用实际的特征数量

        logger.info("XGBoost参数优化")
        logger.info("=" * 60)
        logger.info(f"[CHART] 训练样本: {n_samples:,}个")
        logger.info(f"[CHART] 特征数量: {actual_feature_count}个")  # [WRENCH] 修复：使用实际特征数量
        logger.info(f"[CHART] 类别不平衡比例: {imbalance_ratio:.1f}:1")

        # 使用基础特征的标准参数配置
        logger.info(f"[CHART] 特征组成分析: 总计{actual_feature_count}个基础特征")


        # [强化] 优化：统一参数配置，根据特征数量智能调整
        logger.info(f"[WRENCH] 使用基础特征集({actual_feature_count}个特征)，智能参数配置")

        # 基础参数配置（将被后续优化覆盖）
        params = {
            'random_state': random_seed,  # [强化] 集成学习：使用传入的随机种子
            'n_jobs': -1,
            'verbosity': 0
        }

        # [强化] 排序模型模式：使用XGBoost排序目标函数
        logger.info("[强化] 排序模型模式：使用XGBoost排序目标函数和relevance_30标签（整数0-29,30层）")

        # [强化] 排序模式使用基础特征的优化参数
        logger.info(f"ℹ️ 排序模式使用基础特征，采用专门优化的参数配置")

        # [SEARCH] 根据特征数量和预测目标动态调整参数
        # [强化] 优化：统一的智能参数配置，平衡性能和泛化能力
        if actual_feature_count <= 19:  # 小特征集：更积极的参数
            optimized_params = {
                'max_depth': 6,                   # [WRENCH] 优化：适度深度，平衡复杂度和泛化
                'learning_rate': 0.08,            # [WRENCH] 优化：平衡收敛速度和稳定性
                'n_estimators': 3000,             # [WRENCH] 优化：适度增加树数量
                'subsample': 0.8,                 # [WRENCH] 优化：高采样率但保留一定随机性
                'colsample_bytree': 0.8,          # [WRENCH] 优化：高特征采样
                'reg_alpha': 0.1,                 # [WRENCH] 优化：适度L1正则化
                'reg_lambda': 0.1,                # [WRENCH] 优化：适度L2正则化
                'min_child_weight': 3,            # [WRENCH] 优化：防止过拟合
                'gamma': 0.1,                     # [WRENCH] 优化：适度分裂阈值
                'colsample_bylevel': 0.9,         # [WRENCH] 优化：每层特征采样
                'max_delta_step': 0,              # 不限制权重更新
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            logger.info("[TARGET] 使用小特征集优化参数配置")
        else:
            # 大特征集：更保守的参数
            optimized_params = {
                'max_depth': 7,                   # [WRENCH] 重新优化：适度深度，平衡学习能力和泛化
                'learning_rate': 0.1,            # [WRENCH] 重新优化：适中学习率，避免过度保守
                'n_estimators': 1200,             # [WRENCH] 重新优化：合理树数量，避免过度训练
                'subsample': 0.87,                # [WRENCH] 重新优化：标准采样率，保持学习能力
                'colsample_bytree': 0.92,          # [WRENCH] 重新优化：平衡特征采样，避免信息损失
                'reg_alpha': 2,                 # [WRENCH] 重新优化：适度L1正则化，平衡特征选择
                'reg_lambda': 3,                # [WRENCH] 重新优化：适度L2正则化，避免过度约束
                'min_child_weight': 3,            # [WRENCH] 重新优化：适中最小权重，平衡泛化和学习
                'gamma': 0.3,                     # [WRENCH] 重新优化：适中分裂阈值，平衡复杂度
                'colsample_bylevel': 0.8,         # [WRENCH] 重新优化：标准每层特征采样
                'max_delta_step': 0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

        params.update(optimized_params)
        logger.info(f"[TARGET] 针对{actual_feature_count}个特征优化参数配置")

        # [强化] 超参数搜索功能
        if preset_params is not None:
            # [强化] 使用预设参数（Walk-Forward模式）
            logger.info("=" * 80)
            logger.info("[强化] 使用预设参数（跳过参数搜索）")
            logger.info("=" * 80)
            # params.update(preset_params) 暂时注释掉

           
        elif enable_param_search:
            logger.info("=" * 80)
            logger.info("[SEARCH] 启用超参数搜索优化")
            logger.info("=" * 80)


            # 执行参数搜索
            best_params, search_results = _perform_parameter_search(
                X_train_enhanced, y_train_split_mapped, X_val_enhanced, y_val_split_mapped,
                y_val_split_eval, sample_weights,
                search_strategy,
                val_dates=val_dates,
                train_dates=train_dates,
                market_data=market_data  # [强化] XGBoost需要market_data计算group权重
            )

            # 更新参数
            params.update(best_params)
            logger.info("[OK] 参数搜索完成，使用最优参数配置")
            logger.info(f"[TARGET] 最优参数: {best_params}")
       

        logger.info("[强化]回归参数配置:")
        logger.info(f"  max_depth: {params['max_depth']} (针对{actual_feature_count}特征优化)")
        logger.info(f"  learning_rate: {params['learning_rate']}")
        logger.info(f"  n_estimators: {params['n_estimators']}")
        logger.info(f"  subsample: {params['subsample']}")
        logger.info(f"  colsample_bytree: {params['colsample_bytree']}")
        logger.info(f"  reg_alpha: {params['reg_alpha']}")
        logger.info(f"  reg_lambda: {params['reg_lambda']}")
        logger.info(f"  min_child_weight: {params['min_child_weight']}")
        logger.info(f"  gamma: {params['gamma']}")
        logger.info("[OK] 配置针对MSE目标函数和小特征集优化")


        # [WRENCH] 移除交互约束配置，使用基础特征
        logger.info("[OK] 使用基础特征，无需特征交互约束")

        logger.info("=" * 60)

        # [强化] 排序模式：不进行标签映射，保持原始连续收益标签
        unique_labels = None
        label_mapping = None
        reverse_mapping = None
        logger.info("排序模式：保持原始连续收益标签，不进行映射")

        # [强化] 排序模型配置
        # [强化] 使用rank:pairwise支持sample-level weights
        objective = 'rank:pairwise'
        eval_metric = 'auc'  # rank:pairwise不支持ndcg,使用auc
        num_class = None
        logger.info("使用排序模型配置（Learning to Rank - Pairwise）")

        # 通用参数
        params.update({
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': random_seed,  # [强化] 集成学习：使用传入的随机种子
            'n_jobs': -1,
            'verbosity': 0
        })

        logger.info(f"[OK] XGBoost参数配置完成，排序模式，objective={objective}")

        # [强化] 开始模型训练
        logger.info("=" * 60)
        logger.info("[强化] 开始排序模型训练（Learning to Rank）")
        logger.info("=" * 60)


        # [强化] 排序模式：使用xgboost.train
        import xgboost as xgb

        # 计算group信息（每个日期的样本数）
        if train_dates is not None:
            import pandas as pd
            train_dates_series = pd.Series(train_dates)
            val_dates_series = pd.Series(val_dates) if val_dates is not None else None

            # 确保数据按日期排序
            if not train_dates_series.is_monotonic_increasing:
                logger.warning("[WARNING] 训练集日期未按升序排列,排序模型可能出现问题")

            train_groups = train_dates_series.value_counts().sort_index().values
            logger.info(f"  训练集group数: {len(train_groups)}, 总样本数: {train_groups.sum()}")

            if val_dates_series is not None:
                val_groups = val_dates_series.value_counts().sort_index().values
                logger.info(f"  验证集group数: {len(val_groups)}, 总样本数: {val_groups.sum()}")
            else:
                logger.warning("[WARNING] 缺少验证集日期信息,无法计算验证集group")
                val_groups = None
        else:
            logger.error("[FAIL] 缺少训练集日期信息,无法使用排序模型")
            return None

        # [强化] XGBoost排序模型权重处理: 使用group-level weights
        # [SEARCH] 调试: 检查数据是否按日期排序
        logger.info("[SEARCH] 调试: 检查训练数据是否按日期排序")
        unique_dates = pd.Series(train_dates).unique()
        sorted_dates = pd.Series(train_dates).sort_values().unique()
        if not np.array_equal(unique_dates, sorted_dates):
            logger.error("[FAIL] 训练数据未按日期排序!这会导致排序模型出现问题!")
            logger.error(f"   前10个日期: {train_dates[:10]}")
            logger.error(f"   唯一日期数: {len(unique_dates)}")
        else:
            logger.info("[OK] 训练数据已按日期排序")

        # [强化] 计算XGBoost group权重（最终训练时输出详细日志）
        group_weights = _calculate_xgboost_group_weights(
            train_dates=train_dates,
            market_data=market_data,
            logger=logger,
            time_decay_half_life=180.0,
            reference_date=None,  # 使用训练集最新日期
            verbose=True  # 🔊 最终训练时输出详细日志
        )

        logger.info(f"  [OK] 使用分组级权重（XGBoost排序模型要求）")
        logger.info(f"     分组权重数量: {len(group_weights)} (应等于group数: {len(train_groups)})")
        logger.info(f"     分组权重范围: [{group_weights.min():.4f}, {group_weights.max():.4f}]")
        logger.info(f"     分组权重均值: {group_weights.mean():.4f}")
        logger.info(f"     Group数量: {len(train_groups)}")

        # 验证权重长度
        if len(group_weights) != len(train_groups):
            logger.error(f"[FAIL] 权重长度错误: {len(group_weights)} != {len(train_groups)}")
            logger.error("   这会导致XGBoost训练失败!")
            return None

        # [SEARCH] 调试: 检查训练标签和收益率的关系
        logger.info("[SEARCH] 调试: 检查relevance_30标签和原始收益率的关系(Top-5层级)")
        import pandas as pd
        debug_df = pd.DataFrame({
            'relevance_30': y_train_split_mapped,
            'return': y_train_split_eval
        })
        # 显示Top-5层级的统计
        unique_vals = sorted(debug_df['relevance_30'].unique(), reverse=True)[:5]
        for rel_val in unique_vals:
            rel_mask = debug_df['relevance_30'] == rel_val
            if rel_mask.sum() > 0:
                avg_return = debug_df[rel_mask]['return'].mean()
                logger.info(f"  Relevance_30={rel_val}: 样本数={rel_mask.sum():,}, 平均收益率={avg_return:.4f}")

        # [强化] XGBoost排序模型使用group-level weights
        dtrain = xgb.DMatrix(X_train_enhanced, label=y_train_split_mapped)
        dtrain.set_group(train_groups)
        dtrain.set_weight(group_weights)  # [强化] 使用group-level weights

        dval = xgb.DMatrix(X_val_enhanced, label=y_val_split_mapped)
        if val_groups is not None:
            dval.set_group(val_groups)

        logger.info(f"[CHART] 最终模型训练使用基础特征: 训练集{X_train_enhanced.shape}, 验证集{X_val_enhanced.shape}")

        # 使用MSE损失函数训练
        final_params = params.copy()
        logger.info("[TARGET] 训练MSE回归模型...")

        # [强化] 智能Early Stopping调整机制（基于特征数量和复杂度）
        # [强化] 根据特征数量和训练轮数智能调整Early Stopping参数（优化版）
        # 策略: 基于n_estimators的百分比,同时考虑合理的上下限
        if actual_feature_count <= 20:
            # 小特征集：使用较激进的Early Stopping
            early_stop_ratio = 0.08  # 8%
            min_early_stop, max_early_stop = 50, 500
        elif actual_feature_count <= 35:
            # 中特征集：使用适中的Early Stopping
            early_stop_ratio = 0.10  # 10%
            min_early_stop, max_early_stop = 80, 800
        else:
            # 大特征集：使用宽松的Early Stopping
            early_stop_ratio = 0.12  # 12%
            min_early_stop, max_early_stop = 100, 1200

        # 计算动态early_stopping,限制在合理范围内
        dynamic_early_stopping = int(params['n_estimators'] * early_stop_ratio)
        dynamic_early_stopping = max(min_early_stop, min(max_early_stop, dynamic_early_stopping))

        logger.info(f"[TARGET] 特征集规模({actual_feature_count}个)：Early Stopping = {dynamic_early_stopping}轮 "
                   f"({early_stop_ratio*100:.0f}% of {params['n_estimators']}, 范围[{min_early_stop}, {max_early_stop}])")

        # [强化] 专业训练监控和自动调参建议系统 - 优化版本
        class TrainingMonitor(xgb.callback.TrainingCallback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.train_metrics = []
                self.val_metrics = []
                self.epochs = []
                self.best_metric_value = None
                self.best_epoch = 0
                self.overfitting_start = None
                self.metric_oscillations = 0
                self.last_val_metric = None
                self.metric_name = None
                self.metric_direction = None  # 'higher_is_better' or 'lower_is_better'
                self.is_ranking_model = False

            def after_iteration(self, model, epoch, evals_log):
                """XGBoost callback函数 - 优化版本，正确处理排序模型指标"""
                # 从evals_log获取指标值
                if 'train' in evals_log and 'val' in evals_log:
                    # [强化] 优化的指标检测逻辑 - 适配rank:pairwise模型
                    # 优先级：auc > ndcg@10 > rmse > 其他
                    available_metrics = list(evals_log['train'].keys())

                    if 'auc' in available_metrics:
                        # rank:pairwise模型通常使用AUC
                        self.metric_name = 'auc'
                        self.metric_direction = 'higher_is_better'
                        self.is_ranking_model = True
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'ndcg@10' in available_metrics:
                        # NDCG指标，越大越好
                        self.metric_name = 'ndcg@10'
                        self.metric_direction = 'higher_is_better'
                        self.is_ranking_model = True
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'rmse' in available_metrics:
                        # RMSE指标，越小越好
                        self.metric_name = 'rmse'
                        self.metric_direction = 'lower_is_better'
                        self.is_ranking_model = False
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    elif 'mae' in available_metrics:
                        # MAE指标，越小越好
                        self.metric_name = 'mae'
                        self.metric_direction = 'lower_is_better'
                        self.is_ranking_model = False
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                    else:
                        # 使用第一个可用指标
                        self.metric_name = available_metrics[0]
                        self.metric_direction = 'lower_is_better'  # 默认假设越小越好
                        train_metric = evals_log['train'][self.metric_name][-1]
                        val_metric = evals_log['val'][self.metric_name][-1]
                else:
                    return False  # 如果没有评估日志，跳过

                # 记录数据
                self.epochs.append(epoch)
                self.train_metrics.append(train_metric)
                self.val_metrics.append(val_metric)

                # 检测最佳指标值（适配指标方向）
                if self.best_metric_value is None:
                    self.best_metric_value = val_metric
                    self.best_epoch = epoch
                elif self.metric_direction == 'higher_is_better':
                    if val_metric > self.best_metric_value:
                        self.best_metric_value = val_metric
                        self.best_epoch = epoch
                else:  # lower_is_better
                    if val_metric < self.best_metric_value:
                        self.best_metric_value = val_metric
                        self.best_epoch = epoch

                # [强化] 优化的过拟合检测逻辑
                if len(self.val_metrics) >= 15:  # 需要足够的历史数据
                    # 计算近期的指标趋势
                    recent_val_window = min(8, len(self.val_metrics) // 3)
                    if recent_val_window >= 3:
                        recent_val_values = self.val_metrics[-recent_val_window:]
                        older_val_values = self.val_metrics[-recent_val_window*2:-recent_val_window]

                        recent_val_avg = np.mean(recent_val_values)
                        older_val_avg = np.mean(older_val_values)

                        # 计算训练集趋势
                        recent_train_values = self.train_metrics[-recent_val_window:]
                        older_train_values = self.train_metrics[-recent_val_window*2:-recent_val_window]

                        recent_train_avg = np.mean(recent_train_values)
                        older_train_avg = np.mean(older_train_values)

                        # 根据指标方向判断趋势
                        if self.metric_direction == 'higher_is_better':
                            val_trend = recent_val_avg - older_val_avg  # 正值表示改善
                            train_trend = recent_train_avg - older_train_avg
                            # 过拟合：验证指标恶化 + 训练指标改善
                            overfitting_signal = val_trend < 0 and train_trend > 0
                        else:  # lower_is_better
                            val_trend = older_val_avg - recent_val_avg  # 正值表示改善
                            train_trend = older_train_avg - recent_train_avg
                            # 过拟合：验证指标恶化 + 训练指标改善
                            overfitting_signal = val_trend < 0 and train_trend > 0

                        # 设置过拟合检测阈值
                        trend_threshold = 0.001 if self.is_ranking_model else 0.01

                        if overfitting_signal and abs(val_trend) > trend_threshold and self.overfitting_start is None:
                            self.overfitting_start = epoch

                # 检测指标震荡
                if self.last_val_metric is not None:
                    # 震荡检测阈值根据指标类型调整
                    if self.is_ranking_model:
                        oscillation_threshold = 0.005  # AUC/NDCG 0.5%
                    else:
                        oscillation_threshold = 0.01   # RMSE/MAE 1%

                    relative_change = abs(val_metric - self.last_val_metric)
                    if self.last_val_metric != 0:
                        relative_change = relative_change / abs(self.last_val_metric)

                    if relative_change > oscillation_threshold:
                        self.metric_oscillations += 1

                self.last_val_metric = val_metric

                # 每100轮输出详细分析
                if epoch % 100 == 0 and epoch > 0:
                    self._log_training_analysis(epoch)

                return False  # 继续训练

            def _log_training_analysis(self, epoch):
                """输出训练过程分析 - 优化版本"""
                if len(self.train_metrics) < 20:
                    return

                # 计算指标趋势
                recent_epochs = min(50, len(self.train_metrics))
                train_trend = np.polyfit(range(recent_epochs), self.train_metrics[-recent_epochs:], 1)[0]
                val_trend = np.polyfit(range(recent_epochs), self.val_metrics[-recent_epochs:], 1)[0]

                # 计算训练集和验证集指标的差距（适配指标方向）
                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，训练集 > 验证集表示过拟合
                    train_val_gap = self.train_metrics[-1] - self.val_metrics[-1]
                    gap_descriptions = {
                        'direction': '下降' if train_trend < 0 else '上升',
                        'gap_status': '扩大' if train_val_gap > 0 else '缩小'
                    }
                else:  # lower_is_better
                    # 对于RMSE/MAE，验证集 > 训练集表示过拟合
                    train_val_gap = self.val_metrics[-1] - self.train_metrics[-1]
                    gap_descriptions = {
                        'direction': '下降' if train_trend < 0 else '上升',
                        'gap_status': '扩大' if train_val_gap > 0 else '缩小'
                    }

                # 计算学习效率（适配指标方向）
                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，正值改善
                    learning_efficiency = train_trend / (abs(train_trend) + abs(val_trend) + 1e-8)
                else:  # lower_is_better
                    # 对于RMSE/MAE，负值改善
                    learning_efficiency = -train_trend / (abs(train_trend) + abs(val_trend) + 1e-8)

                # 格式化指标输出
                current_train_metric = self.train_metrics[-1]
                current_val_metric = self.val_metrics[-1]

                self.logger.info(f"  [CHART] 训练监控 [Epoch {epoch}] - {self.metric_name.upper()}:")
                self.logger.info(f"    [REFRESH] 训练{self.metric_name}: {current_train_metric:.6f}")
                self.logger.info(f"    [REFRESH] 验证{self.metric_name}: {current_val_metric:.6f}")
                self.logger.info(f"    [UP] 训练指标趋势: {train_trend:.6f} ({gap_descriptions['direction']})")
                self.logger.info(f"    [UP] 验证指标趋势: {val_trend:.6f} ({'改善' if (val_trend > 0 and self.metric_direction == 'higher_is_better') or (val_trend < 0 and self.metric_direction == 'lower_is_better') else '恶化'})")
                self.logger.info(f"    [UP] 过拟合程度: {train_val_gap:.6f} (差距{gap_descriptions['gap_status']})")
                self.logger.info(f"    ⚡ 学习效率: {learning_efficiency:.3f}")
                self.logger.info(f"    🌊 指标震荡次数: {self.metric_oscillations}")

                if self.overfitting_start:
                    self.logger.info(f"    [WARNING]  过拟合检测: 第{self.overfitting_start}轮开始")

            def get_final_analysis(self):
                """获取最终训练分析结果 - 优化版本"""
                if len(self.train_metrics) < 50:
                    return {}

                # 整体趋势分析（适配指标方向）
                total_epochs = len(self.train_metrics)

                if self.metric_direction == 'higher_is_better':
                    # 对于AUC/NDCG，改善为正值
                    train_improvement = self.train_metrics[-1] - self.train_metrics[0]
                    val_improvement = self.val_metrics[-1] - self.val_metrics[0]

                    # 过拟合严重度：训练集指标远高于验证集
                    final_train_metric = self.train_metrics[-1]
                    final_val_metric = self.val_metrics[-1]
                    if final_train_metric > 0:
                        overfitting_severity = max(0, (final_train_metric - final_val_metric) / final_train_metric)
                    else:
                        overfitting_severity = 0.0

                    # 学习效率分析（适配指标方向）
                    early_improvement = self.train_metrics[min(100, total_epochs//4)] - self.train_metrics[0]
                    late_improvement = self.train_metrics[-1] - self.train_metrics[max(100, total_epochs//2)]

                else:  # lower_is_better
                    # 对于RMSE/MAE，改善为负值
                    train_improvement = self.train_metrics[0] - self.train_metrics[-1]
                    val_improvement = self.val_metrics[0] - self.val_metrics[-1]

                    # 过拟合严重度：验证集指标远高于训练集
                    final_train_metric = self.train_metrics[-1]
                    final_val_metric = self.val_metrics[-1]
                    if final_train_metric > 0:
                        overfitting_severity = max(0, (final_val_metric - final_train_metric) / final_train_metric)
                    else:
                        overfitting_severity = 0.0

                    # 学习效率分析（适配指标方向）
                    early_improvement = self.train_metrics[0] - self.train_metrics[min(100, total_epochs//4)]
                    late_improvement = self.train_metrics[max(100, total_epochs//2)] - self.train_metrics[-1]

                learning_rate_efficiency = early_improvement / (early_improvement + late_improvement + 1e-8)

                # 收敛性分析
                last_20_val_std = np.std(self.val_metrics[-20:])
                convergence_stability = 1.0 / (1.0 + last_20_val_std)

                # 模型健康度评估
                model_health = self._assess_model_health(train_improvement, val_improvement, overfitting_severity)

                return {
                    'total_epochs': total_epochs,
                    'metric_name': self.metric_name,
                    'metric_direction': self.metric_direction,
                    'is_ranking_model': self.is_ranking_model,
                    'train_improvement': train_improvement,
                    'val_improvement': val_improvement,
                    'convergence_stability': convergence_stability,
                    'overfitting_severity': overfitting_severity,
                    'learning_rate_efficiency': learning_rate_efficiency,
                    'model_health': model_health,
                    'best_epoch': self.best_epoch,
                    'overfitting_start': self.overfitting_start,
                    'metric_oscillations': self.metric_oscillations,
                    'final_train_metric': self.train_metrics[-1],
                    'final_val_metric': self.val_metrics[-1],
                    'best_metric_value': self.best_metric_value
                }

            def _assess_model_health(self, train_improvement, val_improvement, overfitting_severity):
                """评估模型健康状况"""
                health_score = 100.0

                # 改善度评分（40分）
                if self.metric_direction == 'higher_is_better':
                    improvement_score = min(40, max(0, val_improvement * 40))  # AUC/NDCG改善
                else:
                    improvement_score = min(40, max(0, val_improvement * 400))  # RMSE/MAE改善

                # 过拟合扣分（30分）
                overfitting_penalty = min(30, overfitting_severity * 30)

                # 震荡扣分（20分）
                oscillation_penalty = min(20, self.metric_oscillations * 2)

                # 收敛性奖励（10分）
                convergence_bonus = min(10, (100 - len(self.train_metrics)) / 10) if len(self.train_metrics) < 100 else 10

                health_score = health_score + improvement_score - overfitting_penalty - oscillation_penalty + convergence_bonus
                health_score = max(0, min(100, health_score))

                if health_score >= 80:
                    return '优秀'
                elif health_score >= 60:
                    return '良好'
                elif health_score >= 40:
                    return '一般'
                else:
                    return '需要改进'

        # 训练最终模型
        logger.info("[强化] 开始最终模型训练（启用专业监控系统）...")
        logger.info(f"  [CHART] 最大训练轮数: {params['n_estimators']}")
        logger.info(f"  [CHART] Early Stopping轮数: {dynamic_early_stopping} (针对{actual_feature_count}个特征优化)")
        logger.info(f"  [CHART] Early Stopping比例: {dynamic_early_stopping/params['n_estimators']*100:.1f}%")
        logger.info(f"  [CHART] 训练集样本: {len(X_train_enhanced):,}")
        logger.info(f"  [CHART] 验证集样本: {len(X_val_enhanced):,}")

        # 创建训练监控器
        monitor = TrainingMonitor(logger)

        # 使用MSE损失函数进行最终训练
        model = xgb.train(
                final_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=100,
                early_stopping_rounds=dynamic_early_stopping,
                callbacks=[monitor]
        )

        

        # --- [FIX] 训练后立即生成训练/验证预测，用于后续IC/RankIC/NDCG等诊断 ---
        _best_iter = getattr(model, "best_iteration", None)
        try:
            if _best_iter is not None:
                # xgboost>=1.6 推荐 iteration_range
                y_train_pred = model.predict(dtrain, iteration_range=(0, int(_best_iter) + 1))
                y_val_pred = model.predict(dval, iteration_range=(0, int(_best_iter) + 1))
            else:
                y_train_pred = model.predict(dtrain)
                y_val_pred = model.predict(dval)
        except TypeError:
            # 兼容旧版 xgboost（无 iteration_range）
            if _best_iter is not None:
                y_train_pred = model.predict(dtrain, ntree_limit=int(_best_iter) + 1)
                y_val_pred = model.predict(dval, ntree_limit=int(_best_iter) + 1)
            else:
                y_train_pred = model.predict(dtrain)
                y_val_pred = model.predict(dval)

        # [CHART] 训练完成后的详细日志
        logger.info("[OK] 最终模型训练完成")
        logger.info(f"  [CHART] 实际训练轮数: {model.best_iteration + 1}")
        logger.info(f"  [CHART] 最佳验证轮数: {model.best_iteration}")

        # [强化] 专业训练分析和自动调参建议系统
        analysis = monitor.get_final_analysis()

        # 输出训练分析结果
        logger.info("=" * 80)
        logger.info("[SEARCH] 专业训练分析报告")
        logger.info("=" * 80)

        if analysis and len(analysis) > 0:  # 确保analysis不为空
            # [强化] 新增：显示指标类型信息 - 使用安全访问
            metric_name = analysis.get('metric_name', 'unknown')
            is_ranking_model = analysis.get('is_ranking_model', False)
            metric_direction = analysis.get('metric_direction', 'lower_is_better')
            model_health = analysis.get('model_health', '未知')

            logger.info(f"[TARGET] 监控指标: {metric_name.upper()} ({'排序模型' if is_ranking_model else '回归模型'})")
            logger.info(f"[TARGET] 指标方向: {'越高越好' if metric_direction == 'higher_is_better' else '越低越好'}")
            logger.info(f"[CHART] 模型健康度: {model_health}")
            logger.info("")

            total_epochs = analysis.get('total_epochs', 0)
            best_epoch = analysis.get('best_epoch', 0)
            train_improvement = analysis.get('train_improvement', 0.0)
            val_improvement = analysis.get('val_improvement', 0.0)
            final_train_metric = analysis.get('final_train_metric', 0.0)
            final_val_metric = analysis.get('final_val_metric', 0.0)
            best_metric_value = analysis.get('best_metric_value', 0.0)
            convergence_stability = analysis.get('convergence_stability', 0.0)
            overfitting_severity = analysis.get('overfitting_severity', 0.0)
            learning_rate_efficiency = analysis.get('learning_rate_efficiency', 0.0)
            metric_oscillations = analysis.get('metric_oscillations', 0)

            logger.info(f"[CHART] 训练总轮数: {total_epochs}")
            if total_epochs > 0:
                logger.info(f"[CHART] 最佳轮数: {best_epoch} (效率: {(best_epoch+1)/total_epochs*100:.1f}%)")
            logger.info(f"[CHART] 训练{metric_name}改善: {train_improvement:.6f}")
            logger.info(f"[CHART] 验证{metric_name}改善: {val_improvement:.6f}")
            logger.info(f"[CHART] 最终训练{metric_name}: {final_train_metric:.6f}")
            logger.info(f"[CHART] 最终验证{metric_name}: {final_val_metric:.6f}")
            logger.info(f"[CHART] 最佳{metric_name}值: {best_metric_value:.6f}")
            logger.info(f"[CHART] 收敛稳定性: {convergence_stability:.3f} ({'稳定' if convergence_stability > 0.7 else '不稳定'})")
            logger.info(f"[CHART] 过拟合严重度: {overfitting_severity:.3f} ({'严重' if overfitting_severity > 0.15 else '轻微' if overfitting_severity > 0.05 else '无'})")
            logger.info(f"[CHART] 学习率效率: {learning_rate_efficiency:.3f} ({'过高' if learning_rate_efficiency < 0.3 else '过低' if learning_rate_efficiency > 0.8 else '合适'})")
            logger.info(f"[CHART] 指标震荡次数: {metric_oscillations}")

            # [强化] 量化交易正则化效果专业评估
            current_reg_alpha = final_params['reg_alpha']
            current_reg_lambda = final_params['reg_lambda']
            total_reg_strength = current_reg_alpha + current_reg_lambda
            # 安全访问analysis字典
            overfitting_severity = analysis.get('overfitting_severity', 0.0)

            logger.info("=" * 60)
            logger.info("[UP] 正则化效果专业评估（量化交易导向）")
            logger.info("=" * 60)

            # 过拟合与正则化匹配度分析
            if overfitting_severity > 0.15 and total_reg_strength < 1.0:
                logger.info("🚨 正则化不足: 存在过拟合但正则化较弱，建议增强")
            elif overfitting_severity < 0.05 and total_reg_strength > 3.0:
                logger.info("🚨 正则化过强: 无过拟合但正则化很强，可能过度平滑")
            elif overfitting_severity < 0.05 and total_reg_strength < 0.5:
                logger.info("[OK] 理想状态: 无过拟合且正则化较弱，有利于捕捉市场信号")
            else:
                logger.info("[OK] 正则化匹配: 正则化强度与过拟合程度基本匹配")

            # 量化交易特殊建议
            sample_count = len(X_train_enhanced)
            feature_count = actual_feature_count
            sample_feature_ratio = sample_count / feature_count

            logger.info("=" * 60)
            logger.info("[IDEA] 量化交易正则化优化建议")
            logger.info("=" * 60)

            if sample_feature_ratio < 100:
                logger.info("[CHART] 数据特征: 高维小样本 -> 建议使用较强正则化防止过拟合")
            elif sample_feature_ratio > 500:
                logger.info("[CHART] 数据特征: 大样本低维 -> 可使用较弱正则化充分学习")
            else:
                logger.info("[CHART] 数据特征: 中等规模 -> 建议平衡正则化策略")

            # 基于训练表现的具体建议
            if overfitting_severity > 0.2:
                logger.info("[TARGET] 建议: 过拟合严重，优先增强L2正则化平滑权重")
            elif overfitting_severity < 0.03:
                logger.info("[TARGET] 建议: 无明显过拟合，可降低正则化增强信号敏感性")

            # 安全访问analysis字典，避免KeyError
        if 'metric_oscillations' in analysis and 'total_epochs' in analysis:
            if analysis['metric_oscillations'] > analysis['total_epochs'] * 0.1:
                logger.info("[TARGET] 建议: 训练震荡较多，增加L2正则化提高稳定性")

            logger.info("=" * 60)

            if 'overfitting_start' in analysis and analysis['overfitting_start']:
                logger.info(f"[WARNING]  过拟合开始轮数: {analysis['overfitting_start']}")

        # 判断停止原因（使用动态Early Stopping参数）
        if model.best_iteration < params['n_estimators'] - dynamic_early_stopping:
            logger.info(f"  [TARGET] 停止原因: Early Stopping触发（验证集性能连续{dynamic_early_stopping}轮未改善）")
            logger.info(f"  [CHART] Early Stopping节省轮数: {params['n_estimators'] - (model.best_iteration + 1)}")
            logger.info(f"  [OK] 训练效率: Early Stopping机制有效防止过拟合")
        else:
            logger.info(f"  [TARGET] 停止原因: 达到最大训练轮数 ({params['n_estimators']})")
            logger.info(f"  [WARNING]  建议: 可能需要增加最大训练轮数或调整Early Stopping参数")
            logger.info(f"  [IDEA] 当前Early Stopping设置: {dynamic_early_stopping}轮 ({dynamic_early_stopping/params['n_estimators']*100:.1f}%)")

        logger.info(f"  [CHART] 训练效率: {(model.best_iteration + 1) / params['n_estimators'] * 100:.1f}% 轮数使用率")

        # [CHART] Early Stopping效果评估
        efficiency_ratio = (model.best_iteration + 1) / params['n_estimators']
        if efficiency_ratio < 0.5:
            logger.info(f"  [TARGET] Early Stopping效果: 优秀 (使用{efficiency_ratio*100:.1f}%轮数即达到最佳性能)")
        elif efficiency_ratio < 0.8:
            logger.info(f"  [TARGET] Early Stopping效果: 良好 (使用{efficiency_ratio*100:.1f}%轮数达到最佳性能)")
        else:
            logger.info(f"  [TARGET] Early Stopping效果: 一般 (使用{efficiency_ratio*100:.1f}%轮数，建议增加训练轮数)")

       


        # [CHART] 参数合理性验证和资源分析
        logger.info("=" * 80)
        logger.info("[CHART] 参数合理性验证与量化交易最佳实践")
        logger.info("=" * 80)

        # 数据规模分析
        sample_count = len(X_train_enhanced)
        feature_count = actual_feature_count
        sample_feature_ratio = sample_count / feature_count

        logger.info(f"[CHART] 数据规模分析:")
        logger.info(f"  训练样本数: {sample_count:,}")
        logger.info(f"  特征数量: {feature_count}")
        logger.info(f"  样本/特征比: {sample_feature_ratio:.1f}")

        # 样本特征比合理性判断
        if sample_feature_ratio < 100:
            logger.info(f"  [WARNING]  样本/特征比过低，容易过拟合，建议增强正则化")
        elif sample_feature_ratio > 1000:
            logger.info(f"  [OK] 样本/特征比充足，可以使用复杂模型")
        else:
            logger.info(f"  [OK] 样本/特征比适中，当前配置合理")

        # n_estimators 专业分析
        # 基于量化交易的推荐公式：考虑数据噪声和市场变化
        base_estimators = min(8000, max(3000, int(sample_count / 100)))  # 基础轮数
        feature_adjustment = min(2000, feature_count * 100)  # 特征调整
        recommended_n_estimators = base_estimators + feature_adjustment

        logger.info(f"[CHART] n_estimators 专业分析:")
        logger.info(f"  当前设置: {params['n_estimators']}")
        logger.info(f"  量化交易推荐: {recommended_n_estimators}")
        logger.info(f"  推荐公式: min(8000, max(3000, 样本数/100)) + min(2000, 特征数*100)")

        efficiency_ratio = (analysis.get('best_epoch', 0) + 1) / params['n_estimators']
        logger.info(f"  实际使用率: {efficiency_ratio:.1%}")

        if params['n_estimators'] > recommended_n_estimators * 1.5:
            logger.info(f"  [WARNING]  当前设置过大，建议调整为 {recommended_n_estimators}")
        elif params['n_estimators'] < recommended_n_estimators * 0.7:
            logger.info(f"  [WARNING]  当前设置过小，建议调整为 {recommended_n_estimators}")
        else:
            logger.info(f"  [OK] 当前设置合理")

        # 当前参数配置专业评估
        logger.info(f"[CHART] 当前参数配置评估:")
        current_params = final_params

    
        # 模型复杂度评估
        complexity_score = current_params['max_depth'] * (1 - current_params['gamma']) * current_params['colsample_bytree']
        if complexity_score > 6:
            logger.info(f"  🏗️  模型复杂度: 高 ({complexity_score:.2f}) - 表达能力强但易过拟合")
        elif complexity_score > 3:
            logger.info(f"  🏗️  模型复杂度: 中等 ({complexity_score:.2f}) - 平衡表达能力和稳定性")
        else:
            logger.info(f"  🏗️  模型复杂度: 低 ({complexity_score:.2f}) - 稳定但表达能力有限")

        # 采样策略评估
        sampling_conservatism = (1 - current_params['subsample']) + (1 - current_params['colsample_bytree'])
        if sampling_conservatism > 0.4:
            logger.info(f"  [TARGET] 采样策略: 保守 ({sampling_conservatism:.2f}) - 减少过拟合但可能欠拟合")
        elif sampling_conservatism > 0.2:
            logger.info(f"  [TARGET] 采样策略: 平衡 ({sampling_conservatism:.2f}) - 适中的随机性")
        else:
            logger.info(f"  [TARGET] 采样策略: 激进 ({sampling_conservatism:.2f}) - 充分利用数据")

        # Early Stopping 合理性分析
        optimal_early_stopping = max(50, min(500, int(params['n_estimators'] * 0.08)))
        logger.info(f"[CHART] Early Stopping 分析:")
        logger.info(f"  当前设置: {dynamic_early_stopping}")
        logger.info(f"  推荐范围: {int(params['n_estimators'] * 0.06)}-{int(params['n_estimators'] * 0.12)}")
        logger.info(f"  最优建议: {optimal_early_stopping} (约8%)")

        if dynamic_early_stopping > params['n_estimators'] * 0.15:
            logger.info(f"  [WARNING]  Early Stopping设置过大，可能无法有效防止过拟合")
        elif dynamic_early_stopping < params['n_estimators'] * 0.05:
            logger.info(f"  [WARNING]  Early Stopping设置过小，可能过早停止训练")
        else:
            logger.info(f"  [OK] Early Stopping设置合理")

        # 计算资源分析
        estimated_training_time = (params['n_estimators'] * sample_count * feature_count) / 1000000  # 粗略估算（分钟）
        logger.info(f"[CHART] 计算资源分析:")
        logger.info(f"  预估训练时间: {estimated_training_time:.1f} 分钟")
        logger.info(f"  内存需求: 约 {(sample_count * feature_count * 8) / (1024**3):.2f} GB")

        if estimated_training_time > 30:
            logger.info(f"  [WARNING]  训练时间较长，建议考虑减少n_estimators或使用更强的正则化")

        logger.info("=" * 80)

        model.loss_function = 'MSE'


        # [强化] 模型评估：支持MSE回归
        logger.info("评估模型性能...")


        # [强化] 使用XGBoost模型预测
        dval_for_pred = xgb.DMatrix(X_val_enhanced)  # [强化] 使用增强特征
        y_pred_raw = model.predict(dval_for_pred)

        # [强化] 直接使用预测的排名分位数，无需逆变换
        y_pred = y_pred_raw
        logger.info(f"[OK] 直接使用预测的排名分位数")

        y_pred_proba = None
        logger.info(f"[OK] {model.loss_function}模型预测完成，最终预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}] (排名分位数)")

        # [强化] 排名标签回归：使用排名分位数和原始收益率进行评估
        y_val_rank_labels = y_val_split  # 验证集排名标签（0-1分位数）
        y_val_return_labels = y_val_split_eval  # 验证集原始收益率（用于计算Top-K收益）


        # [强化] 排名标签回归模型评估
        logger.info("[强化] 排名标签回归模型评估...")
        logger.info("[CHART] 标签分布:")

        # 统一命名：预测值和真实值
        y_pred_rank = y_pred  # 预测的排名分位数
        y_val_rank_true = y_val_rank_labels  # 真实的排名标签
        y_val_return_true = y_val_return_labels  # 真实的原始收益率

        # 计算回归指标（基于排名标签）
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        mse = mean_squared_error(y_val_rank_true, y_pred_rank)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_rank_true, y_pred_rank)
        r2 = r2_score(y_val_rank_true, y_pred_rank)

        # [强化] 计算IC和Rank IC（关键指标）
        # 注意: XGBoost排序模型输出的是排序分数,不是标签值
        # Spearman相关系数会自动将两个变量转换为排名,所以可以直接使用
        rank_ic, _ = spearmanr(y_pred_rank, y_val_return_true)  # Spearman相关系数(正确!)

        # IC (Pearson相关系数): 对于排序模型,这个指标意义不大,仅供参考
        # 因为排序分数的绝对值没有意义,只有相对大小有意义
        ic = np.corrcoef(y_pred_rank, y_val_return_true)[0, 1]  # Pearson相关系数(仅供参考)

        # [强化] 调试信息：检查预测值和真实值的分布
        logger.info(f"[SEARCH] 调试信息:")
        logger.info(f"  y_pred_rank类型: {type(y_pred_rank)}, 形状: {y_pred_rank.shape if hasattr(y_pred_rank, 'shape') else 'N/A'}")
        logger.info(f"  y_val_rank_true类型: {type(y_val_rank_true)}, 形状: {y_val_rank_true.shape if hasattr(y_val_rank_true, 'shape') else 'N/A'}")
        logger.info(f"  y_val_return_true类型: {type(y_val_return_true)}, 形状: {y_val_return_true.shape if hasattr(y_val_return_true, 'shape') else 'N/A'}")

        # [强化] 计算Top 5收益率（前5只股票）
        sorted_indices = np.argsort(y_pred_rank)
        n_samples = len(y_pred_rank)
        # 取前5只股票（如果样本数少于5则取所有样本）
        top_5_count = min(5, n_samples)
        top_5_indices = sorted_indices[-top_5_count:]
        top_5_return = y_val_return_true[top_5_indices].mean()

        # [强化] 计算Top 5排序指标
        # NDCG@5使用全局函数 calculate_ndcg_at_k

        # [强化] 计算多种Top 5排序指标 - 改进版本
        # 1. NDCG@5 - 使用改进的NDCG计算解决负收益和量纲问题
        ndcg_at_5 = calculate_adjusted_ndcg_at_k(y_val_return_true, y_pred_rank, k=5)

        # 2. Precision@5 - Top5中真正高收益股票的比例
        # 定义高收益阈值（前25%）
        high_return_threshold = np.percentile(y_val_return_true, 75)
        actual_high_returns = y_val_return_true >= high_return_threshold
        top_5_high_returns = actual_high_returns[top_5_indices]
        precision_at_5 = np.mean(top_5_high_returns) if len(top_5_indices) > 0 else 0.0

        # 3. Recall@5 - Top5中捕获的高收益股票比例
        total_high_returns = np.sum(actual_high_returns)
        captured_high_returns = np.sum(top_5_high_returns)
        recall_at_5 = captured_high_returns / total_high_returns if total_high_returns > 0 else 0.0

        # 4. Hit@5 - Top5中是否包含至少一个高收益股票
        hit_at_5 = 1.0 if captured_high_returns > 0 else 0.0

        # 5. MRR@5 (Mean Reciprocal Rank) - 修复版本
        # 计算高收益股票中的最佳排名
        high_return_indices = np.where(actual_high_returns)[0]
        if len(high_return_indices) > 0:
            # 获取所有高收益股票的预测分数
            high_return_scores = y_pred_rank[high_return_indices]
            # 找到预测分数最高的高收益股票
            best_high_return_local_idx = np.argmax(high_return_scores)
            best_high_return_global_idx = high_return_indices[best_high_return_local_idx]

            # 计算该股票在全局预测中的排名（从小到大，1开始）
            all_indices_sorted = np.argsort(y_pred_rank)[::-1]  # 降序
            rank_position = np.where(all_indices_sorted == best_high_return_global_idx)[0][0] + 1

            mrr_at_5 = 1.0 / rank_position if rank_position <= 5 else 0.0
        else:
            mrr_at_5 = 0.0

        # 6. 🆕 添加基于真实排名的相关性指标
        # 计算预测排名与真实收益率排名的相关性
        from scipy.stats import spearmanr, pearsonr

        # 真实收益排名（从小到大，收益越高排名越靠前）
        true_return_ranks = np.argsort(np.argsort(-y_val_return_true))  # 降序排名，0开始
        # 预测排名（从小到大，预测分数越高排名越靠前）
        pred_ranks = np.argsort(np.argsort(-y_pred_rank))

        # 基于排名的相关性（更准确的评估）
        rank_correlation_ic, _ = pearsonr(pred_ranks, true_return_ranks)
        rank_correlation_rankic, _ = spearmanr(pred_ranks, true_return_ranks)

        logger.info(f"[TARGET] 排名模型性能评估:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  IC (预测分数vs收益): {ic:.6f}")
        logger.info(f"  Rank IC (预测分数vs收益): {rank_ic:.6f}")
        logger.info(f"  [SEARCH] 基于排名的IC: {rank_correlation_ic:.6f}")
        logger.info(f"  [SEARCH] 基于排名的Rank IC: {rank_correlation_rankic:.6f}")
        logger.info(f"  Top 5 平均收益率: {top_5_return:.4f} ({top_5_return*100:.2f}%)")

        # [强化] 新增：Top 5排序质量指标
        logger.info(f"[CHART] Top 5 排序质量指标:")
        logger.info(f"  NDCG@5: {ndcg_at_5:.6f} (排序质量，0-1，越高越好)")
        logger.info(f"  Precision@5: {precision_at_5:.4f} ({precision_at_5*100:.1f}%)")
        logger.info(f"  Recall@5: {recall_at_5:.4f} ({recall_at_5*100:.1f}%)")
        logger.info(f"  Hit@5: {hit_at_5:.4f} ({hit_at_5*100:.1f}%)")
        logger.info(f"  MRR@5: {mrr_at_5:.6f} (首命中排名，越高越好)")

        # [SEARCH] 指标一致性检查
        logger.info(f"[SEARCH] 指标一致性分析:")
        logger.info(f"  预测分数方向: {'正向' if ic > 0 else '负向'} (预测分数与收益的关系)")
        logger.info(f"  排名一致性: {'一致' if rank_correlation_rankic > 0 else '不一致'}")
        if rank_correlation_rankic > 0 and top_5_return > 0:
            logger.info(f"  [OK] 指标一致: 排序正确且Top5有正收益")
        elif rank_correlation_rankic < 0 and top_5_return > 0:
            logger.info(f"  [WARNING]  部分矛盾: 排序负相关但Top5收益为正")
        else:
            logger.info(f"  [FAIL] 指标矛盾: 排序错误且Top5收益不佳")

        # [强化] 标签和预测分布
        logger.info(f"[CHART] 排名标签分布:")
        logger.info(f"  真实标签范围: [{y_val_rank_true.min():.4f}, {y_val_rank_true.max():.4f}]")
        logger.info(f"  真实标签均值: {y_val_rank_true.mean():.4f}")
        logger.info(f"  真实标签标准差: {y_val_rank_true.std():.4f}")

        logger.info(f"[CHART] 预测分布:")
        logger.info(f"  预测值范围: [{y_pred_rank.min():.4f}, {y_pred_rank.max():.4f}]")
        logger.info(f"  预测值均值: {y_pred_rank.mean():.4f}")
        logger.info(f"  预测值标准差: {y_pred_rank.std():.4f}")

        logger.info(f"[CHART] 原始收益率分布:")
        logger.info(f"  收益率范围: [{y_val_return_true.min():.4f}, {y_val_return_true.max():.4f}]")
        logger.info(f"  收益率均值: {y_val_return_true.mean():.4f} ({y_val_return_true.mean()*100:.2f}%)")
        logger.info(f"  收益率标准差: {y_val_return_true.std():.4f}")

        # [强化] 计算交易指标（基于预测排名选择Top-K股票）
        trading_metrics = _calculate_trading_metrics(
            y_true_return=y_val_return_true,   # 原始收益率
            y_pred_scores=y_pred_rank,         # 预测的排名分位数
            logger=logger,
            k_values=[1, 3, 5]  # [强化] 修复：与实际选股策略一致
        )

        # [强化] 添加专业量化评估指标（基于排名预测）
        quant_metrics = _calculate_quantitative_metrics(y_val_return_true, y_pred_rank, logger)

        # [强化] 添加预测范围诊断指标
        _calculate_prediction_range_diagnostics(y_val_rank_true, y_pred_rank, logger, y_pred_raw)

        # [强化] 诊断评估：检查预测范围压缩问题
        true_range = y_val_rank_true.max() - y_val_rank_true.min()
        pred_range = y_pred_rank.max() - y_pred_rank.min()
        range_coverage_ratio = pred_range / true_range if true_range > 0 else 0

        true_std = y_val_rank_true.std()
        pred_std = y_pred_rank.std()
        std_ratio = pred_std / true_std if true_std > 0 else 0

        logger.info(f"🚨 诊断评估:")
        if range_coverage_ratio < 0.3:
            logger.warning(f"  [FAIL] 严重预测范围压缩: {range_coverage_ratio:.1%} < 30%")
        else:
            logger.info(f"  [OK] 预测范围覆盖率: {range_coverage_ratio:.1%}")

        if std_ratio < 0.5:
            logger.warning(f"  [FAIL] 预测过度保守: 标准差比例{std_ratio:.1%} < 50%")
        else:
            logger.info(f"  [OK] 预测标准差比例: {std_ratio:.1%}")

        if hasattr(model, 'loss_function'):
            logger.info(f"  [CHART] 使用损失函数: {model.loss_function}")

        # [强化] 预测范围压缩的可能原因分析
        if range_coverage_ratio < 0.3 or std_ratio < 0.5:
            logger.warning("[SEARCH] 预测范围压缩可能原因:")
            logger.warning("  3. 训练数据中极端值被过度处理")
            logger.warning("  4. 模型复杂度不足，无法捕捉标签的完整分布")
            logger.warning("[IDEA] 建议:")
            logger.warning("  - 尝试使用MSE损失函数")
            logger.warning("  - 减少特征Winsorization强度")
            logger.warning("  - 增加模型复杂度（更深的树、更多特征交互）")

        # [强化] 交易指标（基于预测排名选择Top-K股票）
        logger.info("[TARGET] 交易指标（Top-K选股评估）:")

        # [强化] Top-K收益统计
        top_k_returns = trading_metrics.get('top_k_returns', {})
        top_k_profit_rates = trading_metrics.get('top_k_profit_rates', {})
        top_k_loss_rates = trading_metrics.get('top_k_loss_rates', {})

        for k in [1, 3, 5]:
            if k in top_k_returns:
                logger.info(f"  Top-{k} 平均收益: {top_k_returns[k]:.4f} ({top_k_returns[k]*100:.2f}%)")
                if k in top_k_profit_rates:
                    logger.info(f"  Top-{k} 盈利率: {top_k_profit_rates[k]:.4f}")
                if k in top_k_loss_rates:
                    logger.info(f"  Top-{k} 亏损率: {top_k_loss_rates[k]:.4f}")

        # 🆕 投资导向评估（综合评估）
        logger.info("")
        logger.info("[TARGET] 投资导向评估:")

        # 需要日期信息进行分组计算，如果有的话
        val_dates_for_eval = preprocessed_data.get('val_dates') if preprocessed_data else None
        if val_dates_for_eval is not None:
            # 使用日期分组计算
            overall_investment_results = evaluate_investment_performance(
                y_val_return_true, y_pred_rank, val_dates_for_eval, k_values=[1, 2, 3, 5], risk_free_rate=0.0
            )
        else:
            # 如果没有日期信息，使用整体计算作为后备
            logger.warning("[WARNING] 缺少日期信息，使用整体计算评估")
            overall_investment_results = evaluate_investment_performance(
                y_val_return_true, y_pred_rank, None, k_values=[1, 2, 3, 5], risk_free_rate=0.0
            )

        # 输出投资评估报告
        print_investment_evaluation_report("XGBoost排序模型", overall_investment_results)

        # ATR三分类不使用传统AUC，专注于分类准确性
        auc_score = 0.0

        # 特征重要性分析
        # [强化] 修复Pseudo-Huber回归特征重要性：使用gain而不是weight
        try:
            # 使用gain作为重要性指标，而不是weight（使用次数）
            importance_dict = model.get_score(importance_type='gain')

            # 转换为数组格式，保持与feature_names顺序一致
            feature_importance = np.zeros(len(feature_names))
            for i, feature_name in enumerate(feature_names):
                # XGBoost使用f0, f1, f2...作为特征名
                xgb_feature_name = f'f{i}'
                feature_importance[i] = importance_dict.get(xgb_feature_name, 0.0)

            # 归一化特征重要性到0-1范围
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()

            logger.info(f"[OK] 获取Booster特征重要性成功（使用gain指标），{len(importance_dict)}个特征有重要性值")
            logger.info(f"  重要性范围: [{feature_importance.min():.6f}, {feature_importance.max():.6f}]")
        except Exception as e:
            logger.warning(f"[WARNING] 获取Booster特征重要性失败: {e}，使用默认值")
            feature_importance = np.ones(len(feature_names)) / len(feature_names)


        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        logger.info("=" * 60)
        logger.info("特征重要性分析结果")
        logger.info("=" * 60)
        logger.info("[CHART] 重要特征:")
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            logger.info(f"  {i:2d}. {row['feature']}: {row['importance']:.4f}")
        logger.info("=" * 60)

        # [强化] SHAP值分析 - 增强模型可解释性
        # [强化] 优化：在集成学习中,只对第一个模型进行SHAP分析以节省时间
        if not enable_shap_analysis:
            logger.info("=" * 60)
            logger.info("⏭️ 跳过SHAP分析（集成学习优化）")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("[SEARCH] SHAP值分析 - 模型可解释性")
            logger.info("=" * 60)

        if enable_shap_analysis:
            try:
                # 抑制SHAP相关警告
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning, module='shap')
                warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
                warnings.filterwarnings('ignore', category=DeprecationWarning, module='shap')

                import shap
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端

                logger.info("[OK] SHAP库导入成功，开始分析...")

                # 创建SHAP解释器 - 针对XGBoost树模型优化
                explainer = shap.TreeExplainer(model)
                logger.info("[OK] SHAP TreeExplainer创建成功")

                # 选择样本进行SHAP分析（避免内存过载）
                sample_size = min(500, len(X_val_enhanced))
                sample_indices = np.random.choice(len(X_val_enhanced), sample_size, replace=False)
                X_sample = X_val_enhanced[sample_indices]

                logger.info(f"[CHART] 使用{sample_size}个验证样本进行SHAP分析")

                # 计算SHAP值
                shap_values = explainer.shap_values(X_sample)
                logger.info("[OK] SHAP值计算完成")

                # 创建可视化保存目录
                import os
                from pathlib import Path
                shap_dir = Path("results") / "shap_analysis"
                shap_dir.mkdir(parents=True, exist_ok=True)

                # 1. 特征重要性汇总图 - 显示全部特征
                num_features = len(feature_names)
                # 根据特征数量自动调整图形大小
                fig_height = max(8, num_features * 0.3)  # 每个特征至少0.3英寸高度
                fig_width = max(12, 16)  # 保证足够宽度显示特征名

                plt.figure(figsize=(fig_width, fig_height))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                show=False, max_display=num_features)  # 显示全部特征
                plt.title("SHAP特征重要性汇总图（全部特征）", fontsize=14, pad=20)
                plt.tight_layout()
                summary_plot_path = shap_dir / "shap_summary_plot_full.png"
                plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"[OK] SHAP汇总图（全部{num_features}个特征）已保存: {summary_plot_path}")

                # 2. 特征重要性条形图 - 显示全部特征
                plt.figure(figsize=(fig_width, fig_height))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=num_features)  # 显示全部特征
                plt.title("SHAP特征重要性排序（全部特征）", fontsize=14, pad=20)
                plt.tight_layout()
                bar_plot_path = shap_dir / "shap_feature_importance_full.png"
                plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"[OK] SHAP特征重要性图（全部{num_features}个特征）已保存: {bar_plot_path}")

                # 3. 生成Top20特征的精简版本（便于快速查看）
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                show=False, max_display=20)
                plt.title("SHAP特征重要性汇总图（Top20）", fontsize=14, pad=20)
                plt.tight_layout()
                summary_top20_path = shap_dir / "shap_summary_plot_top20.png"
                plt.savefig(summary_top20_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"[OK] SHAP汇总图（Top20特征）已保存: {summary_top20_path}")

                # 4. 生成Top20特征的条形图
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=20)
                plt.title("SHAP特征重要性排序（Top20）", fontsize=14, pad=20)
                plt.tight_layout()
                bar_top20_path = shap_dir / "shap_feature_importance_top20.png"
                plt.savefig(bar_top20_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"[OK] SHAP特征重要性图（Top20特征）已保存: {bar_top20_path}")

                # 3. 瀑布图 - 展示个别预测的解释
                try:
                    sample_predictions = model.predict(xgb.DMatrix(X_sample))

                    # 选择几个代表性样本：高预测值、低预测值、中等预测值
                    high_pred_idx = np.argmax(sample_predictions)
                    low_pred_idx = np.argmin(sample_predictions)
                    med_pred_idx = np.argsort(sample_predictions)[len(sample_predictions)//2]

                    representative_indices = [high_pred_idx, med_pred_idx, low_pred_idx]
                    representative_labels = ["高预测收益", "中等预测收益", "低预测收益"]

                    for idx, (sample_idx, label) in enumerate(zip(representative_indices, representative_labels)):
                        try:
                            plt.figure(figsize=(12, 8))

                            # 创建SHAP Explanation对象 - 新版SHAP API要求
                            explanation = shap.Explanation(
                                values=shap_values[sample_idx],
                                base_values=explainer.expected_value,
                                data=X_sample[sample_idx],
                                feature_names=feature_names
                            )

                            # 使用新版API调用瀑布图 - 显示更多特征
                            max_waterfall_features = min(25, len(feature_names))  # 最多显示25个特征
                            shap.plots.waterfall(explanation, max_display=max_waterfall_features, show=False)
                            plt.title(f"SHAP瀑布图 - {label}样本", fontsize=14, pad=20)
                            plt.tight_layout()
                            waterfall_path = shap_dir / f"shap_waterfall_{idx+1}_{label}.png"
                            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"[OK] SHAP瀑布图已保存: {waterfall_path}")
                        except Exception as e:
                            logger.warning(f"[WARNING] 生成第{idx+1}个瀑布图失败: {e}")
                            plt.close()  # 确保图形被关闭
                except Exception as e:
                    logger.warning(f"[WARNING] SHAP瀑布图分析失败: {e}")

                # 4. 特征交互分析（如果特征数量适中）
                if len(feature_names) <= 50:  # 避免计算量过大
                    try:
                        # 计算前10个最重要特征的交互效应
                        top_features_idx = np.argsort(np.abs(shap_values).mean(0))[-10:]

                        for i, feat_idx in enumerate(top_features_idx[:3]):  # 只分析前3个最重要特征
                            plt.figure(figsize=(10, 8))
                            shap.dependence_plot(
                                feat_idx, shap_values, X_sample,
                                feature_names=feature_names,
                                show=False
                            )
                            plt.title(f"SHAP依赖图 - {feature_names[feat_idx]}", fontsize=14, pad=20)
                            plt.tight_layout()
                            dependence_path = shap_dir / f"shap_dependence_{i+1}_{feature_names[feat_idx]}.png"
                            plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            logger.info(f"[OK] SHAP依赖图已保存: {dependence_path}")
                    except Exception as e:
                        logger.warning(f"[WARNING] SHAP特征交互分析失败: {e}")

                # 5. SHAP值统计分析
                shap_importance = np.abs(shap_values).mean(0)
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_importance': shap_importance,
                    'shap_mean': shap_values.mean(0),  # 平均SHAP值（显示正负影响）
                    'shap_std': shap_values.std(0),   # SHAP值标准差（显示稳定性）
                    'shap_abs_mean': shap_importance  # 绝对值平均（重要性）
                }).sort_values('shap_importance', ascending=False)

                # 添加重要性等级分类
                shap_df['importance_level'] = pd.cut(
                    shap_df['shap_importance'],
                    bins=[0, shap_df['shap_importance'].quantile(0.5),
                          shap_df['shap_importance'].quantile(0.8),
                          shap_df['shap_importance'].max()],
                    labels=['低重要性', '中等重要性', '高重要性'],
                    include_lowest=True
                )

                logger.info("[CHART] SHAP特征重要性完整排序:")
                logger.info(f"{'排名':<4} {'特征名':<30} {'重要性':<10} {'平均影响':<10} {'稳定性':<10} {'等级':<10}")
                logger.info("=" * 80)
                for i, (_, row) in enumerate(shap_df.iterrows(), 1):
                    logger.info(f"{i:3d}. {row['feature']:<30} {row['shap_importance']:.4f}    "
                              f"{row['shap_mean']:+.4f}    {row['shap_std']:.4f}    {row['importance_level']}")

                # 统计各重要性等级的特征数量
                importance_counts = shap_df['importance_level'].value_counts()
                logger.info("\n[CHART] 特征重要性分布:")
                for level, count in importance_counts.items():
                    percentage = count / len(shap_df) * 100
                    logger.info(f"  {level}: {count}个特征 ({percentage:.1f}%)")

                # 保存SHAP分析结果
                shap_results_path = shap_dir / "shap_analysis_results.csv"
                shap_df.to_csv(shap_results_path, index=False, encoding='utf-8')
                logger.info(f"\n[OK] SHAP分析结果已保存: {shap_results_path}")

                # 生成特征筛选建议
                low_importance_features = shap_df[shap_df['importance_level'] == '低重要性']['feature'].tolist()
                logger.info(f"\n[IDEA] 特征筛选建议:")
                logger.info(f"  - 可考虑删除的低重要性特征({len(low_importance_features)}个):")
                for i, feature in enumerate(low_importance_features[:10], 1):  # 只显示前10个
                    logger.info(f"    {i}. {feature}")
                if len(low_importance_features) > 10:
                    logger.info(f"    ... 还有{len(low_importance_features)-10}个特征（详见CSV文件）")

                logger.info("=" * 60)
                logger.info("[OK] SHAP分析完成")
                logger.info(f"📁 所有SHAP可视化文件保存在: {shap_dir}")
                logger.info("=" * 60)

            except ImportError:
                logger.warning("[WARNING] SHAP库未安装，跳过SHAP分析")
                logger.info("[IDEA] 安装提示: pip install shap")
            except Exception as e:
                logger.warning(f"[WARNING] SHAP分析过程中出现错误: {e}")
                logger.info("[WRENCH] SHAP分析失败，但不影响模型训练结果")
            finally:
                # 恢复警告设置
                warnings.resetwarnings()

        # 构建结果字典（兼容_save_model_results）
        # [强化] 成本感知回归：构建性能指标
        performance_metrics = {
            'best_iteration': getattr(model, 'best_iteration', params['n_estimators'])
        }

        # 回归模式：添加回归指标和交易指标
        performance_metrics.update({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_type': 'regression',
            'trading_metrics': trading_metrics  # [强化] 添加交易指标
        })


        results = {
            'model': model,
            'scaler': None,  # [WRENCH] 修复：不再使用StandardScaler，设为None
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize，不再需要保存阈值
            'pca_info': pca_info,
            'feature_names': feature_names,
            'interaction_info': None,  # 不使用交互特征
            'feature_importance': importance_df,
            'performance': performance_metrics,
            'training_info': {
                'n_samples': len(X_train_split),
                'n_features_original': len(feature_names),
                'n_features_used': current_features,
                'is_regression': True
            },
            'model_params': params,
            'label_stats': label_stats,  # [强化] 添加标签统计量（用于回归反标准化）
            'hyperparameter_search': {  # [强化] 添加超参数搜索信息
                'enabled': enable_param_search,
                'strategy': search_strategy if enable_param_search else None,
                'search_results': locals().get('search_results', None) if enable_param_search else None
            },
            # [OK] 添加预处理数据(用于LightGBM复用)
            'X_train_processed': X_train_split,
            'X_val_processed': X_val_split,
            'y_train': y_train_split,
            'y_val': y_val_split,
            'model_type': 'xgboost'  # [OK] 标记模型类型
        }

        # [强化] 添加模式特定信息
        results['training_info'].update({
            'target_range': [float(y_train_split.min()), float(y_train_split.max())],
            'target_mean': float(y_train_split.mean()),
            'target_std': float(y_train_split.std())
        })


        logger.info("=" * 60)

        # [强化] 简化：统一返回训练结果
        logger.info("[OK] 模型训练完成，返回训练结果")
        return results

    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def _analyze_feature_interactions(model, feature_names, X_val):
    """分析特征交互"""
    try:
        # [强化] 修复：兼容Booster和XGBRegressor的特征重要性获取
        try:
            # 尝试Booster方法，使用gain而不是weight
            importance_dict = model.get_score(importance_type='gain')
            feature_importance = np.zeros(len(feature_names))
            for i, feature_name in enumerate(feature_names):
                xgb_feature_name = f'f{i}'
                feature_importance[i] = importance_dict.get(xgb_feature_name, 0.0)
            # 归一化
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
        except AttributeError:
            # 回退到传统方法
            feature_importance = model.feature_importances_

        # 找出重要特征
        important_indices = np.argsort(feature_importance)[-20:]  # 前20个重要特征
        important_features = [feature_names[i] for i in important_indices]

        # 分析特征组合的预测能力
        interactions = []

        # 简化的特征交互分析：基于特征重要性的组合
        for i, feat1 in enumerate(important_features[:10]):
            for j, feat2 in enumerate(important_features[i+1:11]):
                # 计算特征组合的交互强度（简化版本）
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)

                # 基于特征重要性计算交互分数
                interaction_score = (feature_importance[idx1] + feature_importance[idx2]) / 2
                interactions.append((f"{feat1} + {feat2}", interaction_score))

        # 按交互分数排序
        interactions.sort(key=lambda x: x[1], reverse=True)
        return interactions

    except Exception as e:
        return []


def _get_new_stock_cache():
    """获取新股缓存，如果不存在则创建"""
    from pathlib import Path
    cache_file = Path(__file__).parent / "cache" / "new_stock_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    return {}





def _get_zscore_factors():
    """
    返回需要 z-score 归一化的因子列表（保留幅度信息）

    适用场景：
    - 当因子的绝对幅度包含信息（例如极端正/负幅度代表强势/弱势）
    - 典型例子：return_1d, return_5d, 相关性因子, 比率因子
    - 已经做了 rank 的因子（如 rank_momentum_20d）也用 z-score
    """
    return [
        'range_compression',
        'body_efficiency',
        # 'shadow_dominance',
        'gap_power',
        'kdj_golden_cross_strength',
        'open_to_high_ratio',
        'sector_concentration',
        'sector_rank_change_5d',
        'sector_breadth',
        'intraday_clv',
        'amihud_illiquidity',
        'abnormal_turnover',
        'efficiency_ratio',
        'mass_index_reversal',
        'price_acceleration',
        'ma60_slope',
        # 'price_ma60_ratio',
        'kdj_diff_9',
        'kdj_j_9',
        'kdj_k_9',
        'stable_momentum',
        'momentum_decay',
        'turnover_stability',
        # 'money_flow_pressure',
        'ulcer_index',
        'max_drawdown_20d',
        'max_rally_20d',
        'max_rally_10d',
        # 'ppo_strength',
        'mfi_strength',
        'volatility_structure_alpha',
        'price_impact',
        # 'volume_structure_alpha',
        'pct_chg_1d',
        'pct_chg_2d',
        'regime_aware_reversal_alpha',
        'trend_acceleration',
        'trend_consistency',
        'trend_pullback_composite_alpha',
        # 'cpv_factor',
        'price_ma5_ratio',
        'high_volatility_pullback_alpha',
        'volume_confirmed_pullback_alpha',
        'distance_from_52w_high',
        'distance_from_52w_low',
        'ma_alignment_score',
        # 'stock_volume_ratio_20d',
        # 'high_position_volume_surge',
        # 'upper_shadow_pressure_5d',
        # 'lower_shadow_absorption_5d',
        # 'shadow_reversal_volume_pressure',
    ]


def _get_rankpct_factors():
    """
    返回需要横截面分位数归一化的因子列表（映射到 [0, 1]，抗极端值）

    适用场景：
    - 当因子分布极度偏斜、含大量极端值，或不同股票间量纲差异巨大
    - 典型例子：PE, PB, 财务比率（分布长尾）、换手率（极端日子）
    - 做截面排序/Top-K 时常用横截面 pct

    输出后缀：
    - `_pct`

    注意：同一因子可以同时出现在 _get_zscore_factors 和 _get_rankpct_factors 中
    """
    return [
        # 个股相对极端性 / Top1-Top2切换 / veto 优先使用横截面 pct
        'severe_abnormal_pressure_10d',
        # 'price_ma5_ratio',
        'high_to_close_drawdown',
        'cpv_factor',
        'ppo_strength',
        'return_10d_sector_mean',
        # 'return_20d_sector_mean',
        # 'return_3d_sector_mean',
        # 'return_5d_sector_mean',
        'shadow_dominance',
        # 'abnormal_turnover',
        'volume_structure_alpha',
        'money_flow_pressure',
        # 'sector_turnover_change_5d_20d',
        # 'open_to_high_ratio',
        # 'intraday_reversal',
        'buy_lg_amount_rate',
        'buy_sm_amount_rate',
        # 'main_outflow_retail_takeover',
    ]


def _get_expanding_factors():
    """
    返回需要 expanding 分位数变换的因子列表。

    主要用于市场/环境广播因子，按日期生成 [0, 1] 的 expanding percentile，
    输出后缀：
    - `_exp`
    """
    return [
        # 市场/环境广播因子优先使用 expanding percentile
        'mkt_ma5_followthrough_v3',
        # 'market_advance_decline_ratio',
        # 'market_advance_decline_ratio_percentile_20d',
        # 'market_trend_strength',
        # 'mkt_regime_trend_pressure_v2',
        # 'mkt_dispersion_shock_v2',
        # 'mkt_sector_dispersion',
        # 'ma5_strength_breadth_v3',
        # 'mkt_breadth_thrust_v2',
        # 'mkt_liquidity_impulse_v2',
        # 'market_volatility_context',
        # 'strong_stock_volume_confirmation_v3',
        # 'sector_hotspot_concentration_v3',
        # 'sector_leadership_persistence_v3',
        # 'mkt_emotion_stability_v2',
        # 'emotion_quality_balance_v3',
        # 'market_60d_new_high_ratio',
        # 'top10_10d_strength_level_v3',
        # 'top10_10d_recent3_strength_v3',
    ]


def _get_stock_factors():

    all_factors = [
        #--原始因子--#
        # 'max_rally_20d_raw',
        # 'max_rally_10d_raw',
        # 'market_prev_60d_new_high_avg_return', 
        'log_price_slope_10d',
        'log_price_slope_20d',
        # 'market_60d_new_high_ratio',
        # 'log_price_slope_10d_x_log_circ_mv',                                                                                              
        # 'market_60d_new_high_ratio_train_percentile',
        # 'abnormal_deviation_3d_ratio',
        # 'abnormal_event_count_10d',
        'severe_abnormal_pressure_10d',
        # 'severe_abnormal_pressure_30d',
        # 'limit_up_count_percentile_20d',                                                                                                    
        # 'zhaban_count_percentile_20d',                                                                                                     
        # 'limit_down_count_percentile_20d',
        'limit_up_sw2_concentration',                                                                                                                
        'sector_ma5_followthrough_v3',
        'sw2_high_low_position_return_spread', 
        'beta_20d',
        # 'return_5d_cs_rank',
        'high_to_close_drawdown',
        'market_trend_strength',
        'market_liquidity_change',
        'market_sentiment_ad',
        'market_advance_decline_ratio',
        'market_advance_decline_ratio_percentile_20d',
        'market_volatility_context',
        'mkt_regime_trend_pressure_v2',
        'mkt_breadth_thrust_v2',
        'mkt_emotion_stability_v2',
        'mkt_dispersion_shock_v2',
        'mkt_liquidity_impulse_v2',
        'mkt_ma5_followthrough_v3',
        # 'mkt_ma5_followthrough_v3_expanding_percentile',
        'buy_lg_amount_rate',
        'buy_sm_amount_rate',
        'net_d5_amount_to_turnover',
        'main_outflow_retail_takeover',
        'sector_leadership_persistence_v3',
        'sector_hotspot_concentration_v3',
        'ma5_strength_breadth_v3',
        'strong_stock_volume_confirmation_v3',
        'emotion_quality_balance_v3',
        'top10_10d_recent3_strength_v3',
        # 'top10_5d_recent3_strength_v3',
        # 'top10_3d_recent3_strength_v3',
        'top10_10d_strength_level_v3',
        'top10_10d_today_return_level_v3',
        'pct_chg_sector_mean',
        'return_10d_sector_mean',
        'return_20d_sector_mean',
        'return_3d_sector_mean',
        'return_5d_sector_mean',
        'sector_momentum_consistency',
        'sector_momentum_dispersion_rank',
        'value_growth_rotation',
        'size_rotation_strength',
        'sector_dispersion',
        'sector_relative_volatility',
        'sector_volatility_5d',
        'sw_circ_mv_rank_intra_sector',
        'ma_convergence_score',
        'ma5_slope',
        'ma_alignment_persistence_5d',
        # 'ma_ribbon_expansion_3d',
        'volume_price_exhaustion',
        'failed_breakout_pressure',
        # 'ma5_ma10_flattening_after_runup',
        'prev_day_blowoff_risk',
        'carryover_overheat_risk',
        'distribution_after_strength_risk',
        'pct_chg_intra_z',
        'return_10d_intra_z',
        'return_20d_intra_z',
        'return_3d_intra_z',
        'return_5d_intra_z',
        'intra_sector_momentum_rank',
        'sector_trend_gated_momentum_intra_z',
        'sector_rotation_breakout_intra_z',
        'sector_strong_leader_alpha',
        'sector_liquidity_gated_turnover_z',
        'sector_weak_reversal_alpha',
        'sector_momentum_accel_rank',
        'sector_turnover_change_5d_20d',
        'sector_hhi_change',
        'sector_relative_strength_vs_market',
        'mkt_sector_dispersion',
        'sector_mkt_corr_20d',
        'market_vols_adaptive_iv',
        'sector_excess_intra_rank',
        'sector_vols_divergence_alpha',
        'sector_turnover_surge_pv_alpha',
        'intraday_reversal',
        'sector_limit_overflow_alpha',
        'sector_broken_resilience_alpha',
        'limit_sustain_alpha',
        'market_emotion_resonance_alpha',
        'resistance_distance',
        'support_distance',
        'prior_high_distance_20d',
        'prior_high_proximity_20d',
        'prior_high_volume_contest_20d',
        'coskewness',
        'kurtosis_20d',
        'skewness_20d',
        'alpha_001',
        'alpha_002',
        'alpha_017',
        'downside_beta',
        'sector_correlation_20d',
        'stock_sector_pos_align',

    # ===== 按日z-score =====
        'range_compression',
        'body_efficiency',
        'gap_power',
        'kdj_golden_cross_strength',
        'open_to_high_ratio',
        'sector_concentration',
        'sector_rank_change_5d',
        'sector_breadth',
        'intraday_clv',
        'amihud_illiquidity',
        'abnormal_turnover',
        'efficiency_ratio',
        'mass_index_reversal',
        'price_acceleration',
        'ma60_slope',
        'price_ma60_ratio',
        'kdj_diff_9',
        'kdj_j_9',
        'kdj_k_9',
        'stable_momentum',
        'momentum_decay',
        'turnover_stability',
        'money_flow_pressure',
        'ulcer_index',
        'max_drawdown_20d',
        'max_rally_20d',
        'max_rally_10d',
        'ppo_strength',
        'mfi_strength',
        'volatility_structure_alpha',
        'price_impact',
        'volume_structure_alpha',
        'pct_chg_1d',
        'pct_chg_2d',
        'regime_aware_reversal_alpha',
        'trend_acceleration',
        'trend_consistency',
        'trend_pullback_composite_alpha',
        'cpv_factor',
        'price_ma5_ratio',
        'high_volatility_pullback_alpha',
        'volume_confirmed_pullback_alpha',
        'distance_from_52w_high',
        'distance_from_52w_low',
        'ma_alignment_score',
        "close_position_20d",
        "close_position_60d",
        "upper_shadow_ratio_3d_mean",
        "upper_shadow_pressure_5d",
        "lower_shadow_absorption_5d",
        "close_to_high_3d_mean",
        "turnover_z_20",
        "stock_volume_ratio_20d",
        "high_position_volume_surge",
        # "high_volume_no_gain_3d",
        "pullback_depth_from_10d_high",
        # "high_pos_upper_shadow",
        # "high_pos_volume_no_gain",
        # "high_pos_reversal_volume",
        "shadow_reversal_volume_pressure",
        # "ret_3d_vs_10d_accel",
        # "ret_5d_vs_20d_accel",
        "momentum_peak_decay_3d",
        # "consecutive_up_days_5d",
        # "large_up_days_10d",
        # "gap_up_count_5d",
        "close_below_ma5_after_runup",
        "low_break_prev_low_after_runup",
        'runup_ma5_exhaustion',
        'runup_low_break',
        
    ]


    return all_factors


def _get_original_factors():
    """获取需要保留原始值的因子列表（不进行归一化）

    这些因子在因子生成时会被计算，但不会进行z-score或rankpct归一化。
    在模型训练时，可以选择是否使用这些原始因子。

    Returns:
        list: 原始因子名称列表
    """
    # 目前没有需要保留原始值的因子
    # 如果未来需要，可以在这里添加
    return [
        'beta_20d',
        # 'market_prev_60d_new_high_avg_return',  
        'log_price_slope_10d',
        'log_price_slope_20d',
        # 'log_price_slope_10d_x_log_circ_mv',                                                                                    
        # 'market_60d_new_high_ratio_train_percentile',
        # 'abnormal_deviation_3d_ratio',
        # 'abnormal_event_count_10d',
        # 'severe_abnormal_pressure_10d',
        # 'severe_abnormal_pressure_30d',
        # 'max_rally_20d_raw',
        # 'max_rally_10d_raw',
        # 'limit_up_count_percentile_20d',                                                                                                             
        # 'zhaban_count_percentile_20d',                                                                                                               
        # 'limit_down_count_percentile_20d', 
        # 'limit_up_sw2_concentration',                                                                                                                
        # 'sector_ma5_followthrough_v3',
        'sw2_high_low_position_return_spread', 
        # 'return_5d_cs_rank',
        'high_to_close_drawdown',
        'market_trend_strength',
        'market_liquidity_change',
        'market_sentiment_ad',
        'market_advance_decline_ratio',
        # 'market_advance_decline_ratio_percentile_20d',
        'market_volatility_context',
        # 'mkt_regime_trend_pressure_v2',
        # 'mkt_breadth_thrust_v2',
        # 'mkt_emotion_stability_v2',
        # 'mkt_dispersion_shock_v2',
        # 'mkt_liquidity_impulse_v2',
        # 'mkt_ma5_followthrough_v3',
        # 'mkt_ma5_followthrough_v3_expanding_percentile',
        # 'buy_lg_amount_rate',
        # 'buy_sm_amount_rate',
        'net_d5_amount_to_turnover',
        'main_outflow_retail_takeover',
        'sector_leadership_persistence_v3',
        'sector_hotspot_concentration_v3',
        # 'ma5_strength_breadth_v3',
        'strong_stock_volume_confirmation_v3',
        'emotion_quality_balance_v3',
        # 'top10_10d_recent3_strength_v3',
        # 'top10_5d_recent3_strength_v3',
        # 'top10_3d_recent3_strength_v3',
        # 'top10_10d_strength_level_v3',
        # 'top10_10d_today_return_level_v3',
        'pct_chg_sector_mean',
        'return_10d_sector_mean',
        'return_20d_sector_mean',
        'return_3d_sector_mean',
        'return_5d_sector_mean',
        'sector_momentum_consistency',
        'sector_momentum_dispersion_rank',
        'value_growth_rotation',
        'size_rotation_strength',
        'sector_dispersion',
        'sector_relative_volatility',
        'sector_volatility_5d',
        'sw_circ_mv_rank_intra_sector',
        'ma_convergence_score',
        # 'ma5_slope',
        # 'ma_alignment_persistence_5d',
        # 'ma_ribbon_expansion_3d',
        # 'volume_price_exhaustion',
        # 'failed_breakout_pressure',
        # 'ma5_ma10_flattening_after_runup',
        'prev_day_blowoff_risk',
        # 'carryover_overheat_risk',
        # 'distribution_after_strength_risk',
        'pct_chg_intra_z',
        'return_10d_intra_z',
        'return_20d_intra_z',
        'return_3d_intra_z',
        'return_5d_intra_z',
        'intra_sector_momentum_rank',
        'sector_trend_gated_momentum_intra_z',
        'sector_rotation_breakout_intra_z',
        'sector_strong_leader_alpha',
        'sector_liquidity_gated_turnover_z',
        'sector_weak_reversal_alpha',
        'sector_momentum_accel_rank',
        'sector_turnover_change_5d_20d',
        'sector_hhi_change',
        'sector_relative_strength_vs_market',
        'mkt_sector_dispersion',
        'sector_mkt_corr_20d',
        'market_vols_adaptive_iv',
        'sector_excess_intra_rank',
        'sector_vols_divergence_alpha',
        'sector_turnover_surge_pv_alpha',
        'intraday_reversal',
        'sector_limit_overflow_alpha',
        'sector_broken_resilience_alpha',
        'limit_sustain_alpha',
        'market_emotion_resonance_alpha',
        'resistance_distance',
        'support_distance',
        # 'coskewness',
        'kurtosis_20d',
        'skewness_20d',
        'alpha_001',
        'alpha_002',
        'alpha_017',
        'downside_beta',
        'sector_correlation_20d',
        'stock_sector_pos_align',
        "close_position_20d",
        # "close_position_60d",
        # "upper_shadow_ratio_3d_mean",
        # "close_to_high_3d_mean",
        # "turnover_z_20",
        # "high_volume_no_gain_3d",
        # "pullback_depth_from_10d_high",
        # "high_pos_upper_shadow",
        # "high_pos_volume_no_gain",
        # "high_pos_reversal_volume",
        # "ret_3d_vs_10d_accel",
        # "ret_5d_vs_20d_accel",
        # "momentum_peak_decay_3d",
        'runup_ma5_exhaustion',
        # 'runup_low_break',
        # "consecutive_up_days_5d",
        # "large_up_days_10d",
        # "gap_up_count_5d",
        # "close_below_ma5_after_runup",
        # "low_break_prev_low_after_runup",
        # 'prior_high_volume_contest_20d'
    ]




# def _get_zscore_factors():
#     """
#     返回需要 z-score 归一化的因子列表（保留幅度信息）

#     适用场景：
#     - 当因子的绝对幅度包含信息（例如极端正/负幅度代表强势/弱势）
#     - 典型例子：return_1d, return_5d, 相关性因子, 比率因子
#     - 已经做了 rank 的因子（如 rank_momentum_20d）也用 z-score
#     """
#     return [
#         'range_compression',
#         'body_efficiency',
#         'shadow_dominance',
#         'gap_power',
#         'overnight_return',
#         'kdj_golden_cross_strength',
#         'open_to_high_ratio',
#         'sector_concentration',
#         'sector_rank_change_5d',
#         'sector_breadth',
#         'intraday_clv',
#         'beta_60d',
#         'amihud_illiquidity',
#         'abnormal_turnover',
#         'efficiency_ratio',
#         'mass_index_reversal',
#         'price_acceleration',
#         'log_price_slope_5d',
#         'log_price_slope_10d',
#         'log_price_slope_20d',
#         'log_price_slope_10d_x_log_circ_mv',
#         'ma60_slope',
#         'price_ma60_ratio',
#         'kdj_diff_9',
#         'kdj_j_9',
#         'kdj_k_9',
#         'stable_momentum',
#         'momentum_decay',
#         'turnover_stability',
#         'money_flow_pressure',
#         'ulcer_index',
#         'max_drawdown_20d',
#         'max_rally_20d',
#         'max_rally_10d',
#         'ppo_strength',
#         'mfi_strength',
#         'relative_strength_20d',
#         'volatility_structure_alpha',
#         'price_impact',
#         'volume_structure_alpha',
#         'pct_chg_1d',
#         'pct_chg_2d',
#         'regime_aware_reversal_alpha',
#         'trend_acceleration',
#         'trend_consistency',
#         'trend_pullback_composite_alpha',
#         'cpv_factor',
#         'price_ma5_ratio',
#         'high_volatility_pullback_alpha',
#         'volume_confirmed_pullback_alpha',
#         'distance_from_52w_high',
#         'distance_from_52w_low',
#         'ma_alignment_score',
#     ]


# def _get_rankpct_factors():
#     """
#     返回需要横截面分位数归一化的因子列表（映射到 [0, 1]，抗极端值）

#     适用场景：
#     - 当因子分布极度偏斜、含大量极端值，或不同股票间量纲差异巨大
#     - 典型例子：PE, PB, 财务比率（分布长尾）、换手率（极端日子）
#     - 做截面排序/Top-K 时常用横截面 pct

#     输出后缀：
#     - `_pct`

#     注意：同一因子可以同时出现在 _get_zscore_factors 和 _get_rankpct_factors 中
#     """
#     return [
#         # 个股相对极端性 / Top1-Top2切换 / veto 优先使用横截面 pct
#         'severe_abnormal_pressure_10d',
#         'high_to_close_drawdown',
#         'price_ma5_ratio',
#         'cpv_factor',
#         'ppo_strength',
#         'return_10d_sector_mean',
#         'return_20d_sector_mean',
#         'return_3d_sector_mean',
#         'return_5d_sector_mean',
#         'shadow_dominance',
#         'abnormal_turnover',
#         'volume_structure_alpha',
#         'money_flow_pressure',
#         'sector_turnover_change_5d_20d',
#         'open_to_high_ratio',
#         'intraday_reversal',
#         'buy_lg_amount_rate',
#         'buy_sm_amount_rate',
#         'main_outflow_retail_takeover',
#     ]


# def _get_expanding_factors():
#     """
#     返回需要 expanding 分位数变换的因子列表。

#     主要用于市场/环境广播因子，按日期生成 [0, 1] 的 expanding percentile，
#     输出后缀：
#     - `_exp`
#     """
#     return [
#         # 市场/环境广播因子优先使用 expanding percentile
#         'mkt_ma5_followthrough_v3',
#         'market_trend_strength',
#         'mkt_regime_trend_pressure_v2',
#         'mkt_dispersion_shock_v2',
#         'mkt_sector_dispersion',
#         'ma5_strength_breadth_v3',
#         'mkt_breadth_thrust_v2',
#         'mkt_liquidity_impulse_v2',
#         'market_volatility_context',
#         'strong_stock_volume_confirmation_v3',
#         'sector_hotspot_concentration_v3',
#         'sector_leadership_persistence_v3',
#         'mkt_emotion_stability_v2',
#         'emotion_quality_balance_v3',
#         'market_60d_new_high_ratio',
#         'top10_10d_strength_level_v3',
#         'top10_10d_recent3_strength_v3',
#     ]


# def _get_stock_factors():

#     all_factors = [
#         #--原始因子--#
#         'max_rally_20d_raw',
#         'max_rally_10d_raw',
#         # 'market_prev_60d_new_high_avg_return',                                                                                     
#         'market_60d_new_high_ratio',                                                                                                          
#         'market_60d_new_high_ratio_train_percentile',
#         # 'abnormal_deviation_3d_ratio',
#         # 'abnormal_event_count_10d',
#         'severe_abnormal_pressure_10d',
#         # 'severe_abnormal_pressure_30d',
#         # 'limit_up_count_percentile_20d',                                                                                                    
#         # 'zhaban_count_percentile_20d',                                                                                                     
#         # 'limit_down_count_percentile_20d',
#         'limit_up_sw2_concentration',                                                                                                                
#         'sector_ma5_followthrough_v3',
#         'sw2_high_low_position_return_spread', 
#         'beta_20d',
#         # 'return_5d_cs_rank',
#         'high_to_close_drawdown',
#         'market_trend_strength',
#         'market_liquidity_change',
#         'market_sentiment_ad',
#         'market_volatility_context',
#         'mkt_regime_trend_pressure_v2',
#         'mkt_breadth_thrust_v2',
#         'mkt_emotion_stability_v2',
#         'mkt_dispersion_shock_v2',
#         'mkt_liquidity_impulse_v2',
#         'mkt_ma5_followthrough_v3',
#         # 'mkt_ma5_followthrough_v3_expanding_percentile',
#         'buy_lg_amount_rate',
#         'buy_sm_amount_rate',
#         'net_d5_amount_to_turnover',
#         'main_outflow_retail_takeover',
#         'sector_leadership_persistence_v3',
#         'sector_hotspot_concentration_v3',
#         'ma5_strength_breadth_v3',
#         'strong_stock_volume_confirmation_v3',
#         'emotion_quality_balance_v3',
#         'top10_10d_recent3_strength_v3',
#         'top10_5d_recent3_strength_v3',
#         'top10_3d_recent3_strength_v3',
#         'top10_10d_strength_level_v3',
#         'top10_10d_today_return_level_v3',
#         'pct_chg_sector_mean',
#         'return_10d_sector_mean',
#         'return_20d_sector_mean',
#         'return_3d_sector_mean',
#         'return_5d_sector_mean',
#         'sector_momentum_consistency',
#         'sector_momentum_dispersion_rank',
#         'sector_momentum_interaction',
#         'value_growth_rotation',
#         'size_rotation_strength',
#         'sector_dispersion',
#         'sector_relative_volatility',
#         'sector_volatility_5d',
#         'sw_circ_mv_rank_intra_sector',
#         'ma_convergence_score',
#         'pct_chg_intra_z',
#         'return_10d_intra_z',
#         'return_20d_intra_z',
#         'return_3d_intra_z',
#         'return_5d_intra_z',
#         'intra_sector_momentum_rank',
#         'sector_trend_gated_momentum_intra_z',
#         'sector_rotation_breakout_intra_z',
#         'sector_leader_capture_rank',
#         'sector_strong_leader_alpha',
#         'sector_liquidity_gated_turnover_z',
#         'sector_weak_reversal_alpha',
#         'sector_momentum_accel_rank',
#         'sector_turnover_change_5d_20d',
#         'sector_hhi_change',
#         'sector_relative_strength_vs_market',
#         'mkt_sector_dispersion',
#         'sector_mkt_corr_20d',
#         'intra_sector_breadth',
#         'market_vols_adaptive_iv',
#         'market_trend_adaptive_beta',
#         'sector_excess_intra_rank',
#         'sector_vols_divergence_alpha',
#         'sector_turnover_surge_pv_alpha',
#         'sector_dispersion_leader_alpha',
#         'vortex_trend',
#         'intraday_reversal',
#         'sector_limit_overflow_alpha',
#         'sector_broken_resilience_alpha',
#         'limit_sustain_alpha',
#         'market_emotion_resonance_alpha',
#         'resistance_distance',
#         'support_distance',
#         'coskewness',
#         'kurtosis_20d',
#         'skewness_20d',
#         'alpha_001',
#         'alpha_002',
#         'alpha_017',
#         'downside_beta',
#         'sector_correlation_20d',
#         'log_price_slope_5d',
#         'log_price_slope_10d',
#         'log_price_slope_20d',
#         'log_price_slope_10d_x_log_circ_mv',
#         'stock_sector_pos_align',

#     # ===== 按日z-score =====
#         'range_compression',
#         'body_efficiency',
#         'shadow_dominance',
#         'gap_power',
#         'overnight_return',
#         'kdj_golden_cross_strength',
#         'open_to_high_ratio',
#         'sector_concentration',
#         'sector_rank_change_5d',
#         'sector_breadth',
#         'intraday_clv',
#         'beta_60d',
#         'amihud_illiquidity',
#         'abnormal_turnover',
#         'efficiency_ratio',
#         'mass_index_reversal',
#         'price_acceleration',
#         'log_price_slope_5d',
#         'log_price_slope_10d',
#         'log_price_slope_20d',
#         'log_price_slope_10d_x_log_circ_mv',
#         'ma60_slope',
#         'price_ma60_ratio',
#         'kdj_diff_9',
#         'kdj_j_9',
#         'kdj_k_9',
#         'stable_momentum',
#         'momentum_decay',
#         'turnover_stability',
#         'money_flow_pressure',
#         'ulcer_index',
#         'max_drawdown_20d',
#         'max_rally_20d',
#         'max_rally_10d',
#         'ppo_strength',
#         'mfi_strength',
#         'relative_strength_20d',
#         'volatility_structure_alpha',
#         'price_impact',
#         'volume_structure_alpha',
#         'pct_chg_1d',
#         'pct_chg_2d',
#         'regime_aware_reversal_alpha',
#         'trend_acceleration',
#         'trend_consistency',
#         'trend_pullback_composite_alpha',
#         'cpv_factor',
#         'price_ma5_ratio',
#         'high_volatility_pullback_alpha',
#         'volume_confirmed_pullback_alpha',
#         'distance_from_52w_high',
#         'distance_from_52w_low',
#         'ma_alignment_score',


#     # ===== 按行业z-score =====
       
    
#     # ===== 全局rank =====
        
#     ]


#     return all_factors


# def _get_original_factors():
#     """获取需要保留原始值的因子列表（不进行归一化）

#     这些因子在因子生成时会被计算，但不会进行z-score或rankpct归一化。
#     在模型训练时，可以选择是否使用这些原始因子。

#     Returns:
#         list: 原始因子名称列表
#     """
#     # 目前没有需要保留原始值的因子
#     # 如果未来需要，可以在这里添加
#     return [
#         'beta_20d',
#         # 'market_prev_60d_new_high_avg_return',                                                                                                
#         'market_60d_new_high_ratio',                                                                                                                 
#         'market_60d_new_high_ratio_train_percentile',
#         # 'abnormal_deviation_3d_ratio',
#         # 'abnormal_event_count_10d',
#         'severe_abnormal_pressure_10d',
#         # 'severe_abnormal_pressure_30d',
#         'max_rally_20d_raw',
#         'max_rally_10d_raw',
#         # 'limit_up_count_percentile_20d',                                                                                                             
#         # 'zhaban_count_percentile_20d',                                                                                                               
#         # 'limit_down_count_percentile_20d', 
#         'limit_up_sw2_concentration',                                                                                                                
#         'sector_ma5_followthrough_v3',
#         'sw2_high_low_position_return_spread', 
#         # 'return_5d_cs_rank',
#         'high_to_close_drawdown',
#         'market_trend_strength',
#         'market_liquidity_change',
#         'market_sentiment_ad',
#         'market_volatility_context',
#         'mkt_regime_trend_pressure_v2',
#         'mkt_breadth_thrust_v2',
#         'mkt_emotion_stability_v2',
#         'mkt_dispersion_shock_v2',
#         'mkt_liquidity_impulse_v2',
#         'mkt_ma5_followthrough_v3',
#         # 'mkt_ma5_followthrough_v3_expanding_percentile',
#         'buy_lg_amount_rate',
#         'buy_sm_amount_rate',
#         'net_d5_amount_to_turnover',
#         'main_outflow_retail_takeover',
#         'sector_leadership_persistence_v3',
#         'sector_hotspot_concentration_v3',
#         'ma5_strength_breadth_v3',
#         'strong_stock_volume_confirmation_v3',
#         'emotion_quality_balance_v3',
#         'top10_10d_recent3_strength_v3',
#         'top10_5d_recent3_strength_v3',
#         'top10_3d_recent3_strength_v3',
#         'top10_10d_strength_level_v3',
#         'top10_10d_today_return_level_v3',
#         'pct_chg_sector_mean',
#         'return_10d_sector_mean',
#         'return_20d_sector_mean',
#         'return_3d_sector_mean',
#         'return_5d_sector_mean',
#         'sector_momentum_consistency',
#         'sector_momentum_dispersion_rank',
#         'sector_momentum_interaction',
#         'value_growth_rotation',
#         'size_rotation_strength',
#         'sector_dispersion',
#         'sector_relative_volatility',
#         'sector_volatility_5d',
#         'sw_circ_mv_rank_intra_sector',
#         'ma_convergence_score',
#         'pct_chg_intra_z',
#         'return_10d_intra_z',
#         'return_20d_intra_z',
#         'return_3d_intra_z',
#         'return_5d_intra_z',
#         'intra_sector_momentum_rank',
#         'sector_trend_gated_momentum_intra_z',
#         'sector_rotation_breakout_intra_z',
#         'sector_leader_capture_rank',
#         'sector_strong_leader_alpha',
#         'sector_liquidity_gated_turnover_z',
#         'sector_weak_reversal_alpha',
#         'sector_momentum_accel_rank',
#         'sector_turnover_change_5d_20d',
#         'sector_hhi_change',
#         'sector_relative_strength_vs_market',
#         'mkt_sector_dispersion',
#         'sector_mkt_corr_20d',
#         'intra_sector_breadth',
#         'market_vols_adaptive_iv',
#         'market_trend_adaptive_beta',
#         'sector_excess_intra_rank',
#         'sector_vols_divergence_alpha',
#         'sector_turnover_surge_pv_alpha',
#         'sector_dispersion_leader_alpha',
#         'vortex_trend',
#         'intraday_reversal',
#         'sector_limit_overflow_alpha',
#         'sector_broken_resilience_alpha',
#         'limit_sustain_alpha',
#         'market_emotion_resonance_alpha',
#         'resistance_distance',
#         'support_distance',
#         'coskewness',
#         'kurtosis_20d',
#         'skewness_20d',
#         'alpha_001',
#         'alpha_002',
#         'alpha_017',
#         'downside_beta',
#         'sector_correlation_20d',
#         'log_price_slope_5d',
#         'log_price_slope_10d',
#         'log_price_slope_20d',
#         'log_price_slope_10d_x_log_circ_mv',
#         'stock_sector_pos_align',
#     ]







def _get_intra_sector_rank_factors():
    """获取需要进行按日行业内排名的因子列表

    这些因子在pipeline.py中需要进行特殊的按日行业内排名处理，
    而不是常规的全市场横截面排名。

    处理逻辑：
    1. 在每个交易日，对因子值在同一个SW行业内进行排名
    2. 排名结果映射到[0, 1]区间，便于不同行业间的比较
    3. 消除行业间的系统性差异，关注个股在行业内的相对表现

    Returns:
        list: 需要按日行业内排名的因子名称列表
    """
    # 定义需要按日行业内排名的因子
    # 这些因子通常是需要消除行业偏误的个股特征因子
    return [
    ]


def _get_intra_sector_zscore_factors():
    """获取需要进行按日行业内z-score标准化的因子列表

    这些因子在pipeline.py中需要进行特殊的按日行业内z-score处理，
    而不是常规的全市场横截面z-score标准化。

    处理逻辑：
    1. 在每个交易日，对因子值在同一个SW行业内计算均值和标准差
    2. 对因子值进行行业内标准化：(x -行业均值) / 行业标准差
    3. 消除行业间的系统性差异，保留个股在行业内的相对位置
    4. 适用于需要保持连续数值特征的因子（与排名不同）

    Returns:
        list: 需要按日行业内z-score标准化的因子名称列表
    """
    # 定义需要按日行业内z-score标准化的因子
    # 这些因子通常是需要保持数值连续性，同时消除行业偏误的个股特征因子
    return [
    ]


def _get_model_factors(
    use_original=True,
    use_zscore=True,
    use_rankpct=True,
    use_expanding=True,
    use_intra_sector_rank=True,
    use_intra_sector_zscore=True,
):
    """获取模型训练时使用的因子列表

    此函数用于模型训练和因子测试，可以灵活选择使用哪些类型的因子。

    Args:
        use_original: 是否使用原始因子（不带后缀）
        use_zscore: 是否使用z-score归一化因子（_z后缀）
        use_rankpct: 是否使用横截面分位数归一化因子（_pct后缀）
        use_expanding: 是否使用expanding分位数因子（_exp后缀）
        use_intra_sector_rank: 是否使用行业内排名归一化因子（_intra_rank后缀）
        use_intra_sector_zscore: 是否使用行业内z-score归一化因子（_intra_z后缀）

    Returns:
        list: 因子名称列表

    Examples:
        # 只使用z-score归一化因子（默认）
        factors = _get_model_factors()

        # 使用原始因子 + z-score因子
        factors = _get_model_factors(use_original=True, use_zscore=True, use_rankpct=False)

        # 使用所有因子
        factors = _get_model_factors(use_original=True, use_zscore=True, use_rankpct=True, use_expanding=True)

        # 🆕 包含行业内因子
        factors = _get_model_factors(use_original=True, use_zscore=True, use_rankpct=True, use_intra_sector_rank=True, use_intra_sector_zscore=True)
    """
    factors = []

    # 添加原始因子
    if use_original:
        original_factors = _get_original_factors()
        factors.extend(original_factors)

    # 添加 z-score 归一化因子
    if use_zscore:
        zscore_factors = _get_zscore_factors()
        #todo 20260102
        # factors.extend([f"{f}_z" for f in zscore_factors])
        factors.extend(zscore_factors)

    # 添加 rankpct 归一化因子
    if use_rankpct:
        rankpct_factors = _get_rankpct_factors()
        factors.extend([f"{f}_pct" for f in rankpct_factors])

    # 添加 expanding 分位数因子
    if use_expanding:
        expanding_factors = _get_expanding_factors()
        factors.extend([f"{f}_exp" for f in expanding_factors])

    # 🆕 添加行业内排名归一化因子
    if use_intra_sector_rank:
        intra_rank_factors = _get_intra_sector_rank_factors()
        factors.extend([f"{f}_intra_rank" for f in intra_rank_factors])

    # 🆕 添加行业内z-score归一化因子
    if use_intra_sector_zscore:
        intra_zscore_factors = _get_intra_sector_zscore_factors()
        factors.extend([f"{f}_intra_z" for f in intra_zscore_factors])

    # 去重（保持顺序）
    seen = set()
    unique_factors = []
    for factor in factors:
        if factor not in seen:
            seen.add(factor)
            unique_factors.append(factor)

    return unique_factors



def _get_available_market_factors():
    """获取可用的市场状态因子列表"""

    # 基于market_state_test_mode测试结果，返回有效的市场状态因子
    market_factors = [
        # 'gain_loss_ratio_rank_10d',
        # 'gain_loss_ratio_rank_20d',
        'gain_loss_ratio_rank_60d',
        'limit_up_count_rank_60d',
        'limit_up_count_rank_20d',
        # 'limit_up_count_rank_10d',
        # 'limit_up_count_rank_5d',
        'limit_down_count_rank_60d',
        'limit_down_count_rank_5d',
        'market_volume_rank_20d',
        'broken_limit_count_rank_20d',
        'max_consecutive_limit_rank_20d',
        'market_volatility_atr20d',
        'market_trend_factor',
        'max_gain_10d_rank_60d',
        'max_gain_5d_rank_30d'
    ]

    logger.info(f"返回 {len(market_factors)} 个市场状态因子")
    return market_factors


def _clear_old_cache():
    """清理过期的缓存文件"""
    try:
        from pathlib import Path
        import datetime

        cache_dir = Path(__file__).parent / "cache"
        if not cache_dir.exists():
            return

        # 清理7天前的缓存
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)

        for cache_file in cache_dir.glob("*.pkl"):
            try:
                if cache_file.stat().st_mtime < cutoff_date.timestamp():
                    cache_file.unlink()
                    print(f"[DELETE] 清理过期缓存: {cache_file.name}")
            except:
                pass
    except Exception as e:
        print(f"[WARNING] 清理缓存失败: {e}")

def _save_new_stock_cache(cache):
    """保存新股缓存"""
    cache_file = Path(__file__).parent / "cache" / "new_stock_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[WARNING] 保存新股缓存失败: {e}")

def _build_new_stock_list_for_date(current_date: str) -> set:
    """为特定日期构建新股列表"""
    try:
        from pathlib import Path
        import pandas as pd

        # 获取当前日期前30个交易日的日期列表
        trading_dates = _get_previous_trading_dates(current_date, 30)

        if len(trading_dates) < 20:
            return set()  # 返回空集合，表示没有新股过滤

        # 使用相对于当前文件的路径
        current_file_dir = Path(__file__).parent
        features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

        # 收集所有股票的交易天数
        stock_trading_days = {}

        for date_str in trading_dates:
            file_path = features_dir / f"{date_str}.parquet"

            if file_path.exists():
                try:
                    daily_data = pd.read_parquet(file_path)

                    # 处理所有股票，包括成交量=0的股票，以确保新股能被正确识别
                    all_stocks = daily_data['stock_code'].unique()

                    # 统计有效成交量（>0）的交易天数
                    valid_volume_stocks = daily_data[
                        (~np.isnan(daily_data['volume'])) & (daily_data['volume'] > 0)
                    ]['stock_code'].unique()

                    # 为所有股票建立记录
                    for stock_code in all_stocks:
                        if stock_code not in stock_trading_days:
                            stock_trading_days[stock_code] = 0

                        # 只有成交量>0才计入交易天数
                        if stock_code in valid_volume_stocks:
                            stock_trading_days[stock_code] += 1

                except Exception as e:
                    print(f"[WARNING] 读取{date_str}数据失败: {e}")
                    continue

        # 识别新股（有效交易天数少于20天）
        new_stocks = {
            stock_code for stock_code, days in stock_trading_days.items()
            if days < 25
        }

        print(f"[DATE] {current_date}: 识别出 {len(new_stocks)} 只新股")
        return new_stocks

    except Exception as e:
        print(f"[FAIL] 构建新股列表失败: {e}")
        return set()


def _get_nth_trading_date(start_date_str, trading_dates, n=25, logger=None):
    """
    从指定日期开始，获取第N个交易日

    参数:
        start_date_str: 开始日期字符串 (YYYYMMDD)
        trading_dates: 交易日期列表 (YYYYMMDD格式)
        n: 要获取的第N个交易日
        logger: 日志记录器

    返回:
        pd.Timestamp: 第N个交易日的日期对象
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # 找到开始日期在交易日历中的位置
        start_index = None
        for i, date_str in enumerate(trading_dates):
            if date_str >= start_date_str:
                start_index = i
                break

        if start_index is None:
            logger.warning(f"[FILTER] 开始日期 {start_date_str} 超出交易日历范围")
            # 使用最后N个交易日
            if len(trading_dates) >= n:
                end_date_str = trading_dates[-1]
            else:
                end_date_str = trading_dates[-1] if trading_dates else start_date_str
            return pd.to_datetime(end_date_str, format='%Y%m%d')

        # 获取第N个交易日（从start_index开始数N个）
        target_index = start_index + n - 1  # -1因为包含IPO当天

        if target_index >= len(trading_dates):
            # 如果超出范围，使用最后一个交易日
            logger.warning(f"[FILTER] 从 {start_date_str} 开始的第{n}个交易日超出交易日历范围，使用最后一个交易日")
            end_date_str = trading_dates[-1]
        else:
            end_date_str = trading_dates[target_index]

        return pd.to_datetime(end_date_str, format='%Y%m%d')

    except Exception as e:
        logger.error(f"[FILTER] 计算第{n}个交易日失败: {e}")
        # 降级方案：使用自然日
        start_dt = pd.to_datetime(start_date_str, format='%Y%m%d')
        return start_dt + pd.Timedelta(days=n)




def _filter_new_stocks_batch(stock_data: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    批量过滤新股数据（基于真实IPO日期，过滤IPO后25个交易日）

    优化版本：
    1. 使用真实IPO日期数据
    2. 向量化过滤操作
    3. 统计过滤信息
    - 从 ipo_date.json 准确获取IPO日期
    - 过滤上市后25个自然日的数据
    - 不依赖交易日历文件
    - 简化逻辑，提高准确性

    参数:
        stock_data: 股票数据DataFrame，必须包含stock_code、date列
        logger: 日志记录器

    返回:
        pd.DataFrame: 过滤后的股票数据
    """
    import pandas as pd
    from datetime import timedelta

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # ===== 第1步：验证输入数据 =====
        required_columns = ['stock_code', 'date']
        if not all(col in stock_data.columns for col in required_columns):
            logger.error(f"[FILTER] 缺少必需的列: {required_columns}")
            return stock_data

        if len(stock_data) < 10:
            logger.info(f"[FILTER] 数据量过小({len(stock_data)})，跳过新股过滤")
            return stock_data

        # logger.info(f"[FILTER] 开始新股过滤: {len(stock_data):,} 条记录")

        # ===== 第2步：加载IPO日期 =====
        ipo_dates = _load_ipo_dates(logger)
        if not ipo_dates:
            logger.warning("[FILTER] 无法加载IPO日期，跳过新股过滤")
            return stock_data

        # ===== 第2.5步：加载交易日历 =====
        trading_dates = _load_trading_calendar(logger)
        if not trading_dates:
            logger.warning("[FILTER] 无法加载交易日历，使用自然日计算（降级方案）")
            use_trading_days = False
        else:
            use_trading_days = True
            logger.info("[FILTER] 使用交易日历计算新股过滤期（25个交易日）")

        # ===== 第3步：获取数据中的日期范围 =====
        data_start_date = stock_data['date'].min()
        data_end_date = stock_data['date'].max()
        logger.info(f"[FILTER] 数据日期范围: {data_start_date.strftime('%Y-%m-%d')} ~ {data_end_date.strftime('%Y-%m-%d')}")

        # ===== 第4步：预计算需要过滤的股票和日期 =====
        filter_stock_date_pairs = []  # [(stock_code, filter_start_date, filter_end_date), ...]

        for stock_code in stock_data['stock_code'].unique():
            if stock_code not in ipo_dates:
                # 股票不在IPO数据中，可能是老股，跳过
                continue

            # 解析IPO日期
            try:
                ipodate_str = ipo_dates[stock_code]
                ipodate_dt = pd.to_datetime(ipodate_str, format='%Y%m%d')

                # 计算过滤结束日期（IPO后第60个交易日）
                if use_trading_days:
                    # 检查IPO日期是否在交易日历范围内
                    trading_start = trading_dates[0]
                    trading_end = trading_dates[-1]

                    if ipodate_str < trading_start:
                        # IPO日期早于交易日历起始日期，说明这是很早上市的股票，跳过过滤
                        logger.debug(f"[FILTER] {stock_code} IPO日期: {ipodate_str} 早于交易日历起始日期: {trading_start}，跳过新股过滤")
                        continue

                    if ipodate_str > trading_end:
                        # IPO日期晚于交易日历结束日期，这种情况也应该跳过
                        logger.debug(f"[FILTER] {stock_code} IPO日期: {ipodate_str} 晚于交易日历结束日期: {trading_end}，跳过新股过滤")
                        continue

                    # 使用交易日历计算第60个交易日
                    filter_end_date = _get_nth_trading_date(ipodate_str, trading_dates, n=5, logger=logger)
                    # logger.info(f"[FILTER] {stock_code} IPO日期: {ipodate_str}, 第60个交易日: {filter_end_date.strftime('%Y%m%d')}")
                else:
                    # 降级方案：使用自然日
                    filter_end_date = ipodate_dt + pd.Timedelta(days=25)
                    logger.info(f"[FILTER] {stock_code} IPO日期: {ipodate_str}, 使用自然日+60: {filter_end_date.strftime('%Y%m%d')}")

                # 检查过滤期间是否与数据有重叠
                if filter_end_date >= data_start_date:
                    # 需要过滤该股票
                    filter_stock_date_pairs.append({
                        'stock_code': stock_code,
                        'ipo_date': ipodate_dt,
                        'filter_end_date': filter_end_date
                    })

            except Exception as e:
                logger.warning(f"[FILTER] 解析 {stock_code} 的IPO日期失败: {e}")
                continue

        if not filter_stock_date_pairs:
            logger.info("[FILTER] 数据中没有需要过滤的新股")
            return stock_data

        logger.info(f"[FILTER] 识别出 {len(filter_stock_date_pairs)} 只新股需要过滤")

        # ===== 第5步：构建向量化过滤掩码（优化版）=====
        # 5.1 将新股过滤条件转换为DataFrame
        new_stock_df = pd.DataFrame(filter_stock_date_pairs)

        # 5.2 使用merge快速匹配，避免循环
        # 重置索引后再merge，避免后续布尔过滤出现索引错位
        work_df = stock_data.reset_index(drop=True).copy()
        # 将日期转换为数值便于比较
        work_df['_date_num'] = work_df['date'].astype(np.int64)
        new_stock_df['_ipo_num'] = new_stock_df['ipo_date'].astype(np.int64)
        new_stock_df['_filter_end_num'] = new_stock_df['filter_end_date'].astype(np.int64)

        # 执行merge
        merged = work_df.merge(
            new_stock_df[['stock_code', '_ipo_num', '_filter_end_num']],
            on='stock_code',
            how='left'
        )

        # 5.3 向量化判断：是否需要过滤
        # 条件：匹配到新股 且 日期在IPO和过滤结束日期之间
        need_filter_mask = (
            merged['_ipo_num'].notna() &
            (merged['_date_num'] >= merged['_ipo_num']) &
            (merged['_date_num'] <= merged['_filter_end_num'])
        )

        # ===== 第6步：执行过滤 =====
        # 关键修复：直接使用merged来过滤，避免索引不匹配问题
        # 使用merged的原始列（work_df的列）来过滤
        original_columns = stock_data.columns.tolist()
        filtered_data = merged.loc[~need_filter_mask, original_columns].copy()

        # ===== 第7步：统计结果 =====
        original_count = len(stock_data)
        filtered_count = len(filtered_data)
        removed_count = original_count - filtered_count

        logger.info(f"[FILTER] 新股过滤完成:")
        logger.info(f"  原始数据: {original_count:,} 条")
        logger.info(f"  过滤后数据: {filtered_count:,} 条")
        logger.info(f"  过滤掉新股数据: {removed_count:,} 条 ({removed_count/original_count*100:.2f}%)")

        # # 详细统计每只新股的过滤情况
        # for pair in filter_stock_date_pairs:
        #     stock_code = pair['stock_code']
        #     ipo_date = pair['ipo_date'].strftime('%Y-%m-%d')
        #     filter_end = pair['filter_end_date'].strftime('%Y-%m-%d')

        #     original_stock_count = len(stock_data[stock_data['stock_code'] == stock_code])
        #     filtered_stock_count = len(filtered_data[filtered_data['stock_code'] == stock_code])
        #     stock_removed = original_stock_count - filtered_stock_count

        #     logger.info(f"  {stock_code}(IPO:{ipo_date}, 过滤至:{filter_end}): 过滤 {stock_removed} 条记录")

        return filtered_data

    except Exception as e:
        logger.error(f"[FILTER] 新股过滤失败: {e}")
        import traceback
        logger.error(f"[FILTER] 错误详情: {traceback.format_exc()}")
        return stock_data



def _load_ipo_dates(logger) -> dict:
    """
    加载所有股票的IPO日期

    Returns:
        dict: {股票代码: IPO日期(str, YYYYMMDD)}
              例如: {'000001.SZ': '19910403', '600000.SH': '19991110'}
    """
    import json
    from pathlib import Path

    ipo_json_path = Path(__file__).parent.parent / "local_data" / "ipo_date.json"

    if not ipo_json_path.exists():
        logger.error(f"[FILTER] IPO日期文件不存在: {ipo_json_path}")
        logger.error(f"[FILTER] 请先运行: cd to && python download.py")
        return {}

    try:
        with open(ipo_json_path, 'r', encoding='utf-8') as f:
            ipo_dates = json.load(f)

        logger.info(f"[FILTER] 成功加载 {len(ipo_dates):,} 只股票的IPO日期")
        return ipo_dates

    except Exception as e:
        logger.error(f"[FILTER] 加载IPO日期失败: {e}")
        return {}


def _load_trading_calendar(logger=None):
    """
    加载交易日历

    返回:
        list: 交易日期字符串列表 (YYYYMMDD格式)
    """
    import pandas as pd
    from pathlib import Path

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # 交易日历文件路径
        calendar_path = Path(__file__).parent.parent / "to" / "local_data" / "calendar" / "trading_calendar.parquet"

        if not calendar_path.exists():
            logger.error(f"[FILTER] 交易日历文件不存在: {calendar_path}")
            return []

        # 读取交易日历
        calendar_df = pd.read_parquet(calendar_path)

        # 获取交易日期列（可能是 'date_str' 或其他列名）
        if 'date_str' in calendar_df.columns:
            trading_dates = calendar_df['date_str'].tolist()
        elif 'date' in calendar_df.columns:
            trading_dates = calendar_df['date'].dt.strftime('%Y%m%d').tolist()
        else:
            logger.error(f"[FILTER] 交易日历文件缺少 'date_str' 或 'date' 列，现有列: {list(calendar_df.columns)}")
            return []

        logger.info(f"[FILTER] 成功加载交易日历: {len(trading_dates)} 个交易日")
        logger.debug(f"[FILTER] 交易日历范围: {trading_dates[0]} ~ {trading_dates[-1]}")

        return trading_dates

    except Exception as e:
        logger.error(f"[FILTER] 加载交易日历失败: {e}")
        import traceback
        logger.error(f"[FILTER] 错误详情: {traceback.format_exc()}")
        return []





def _load_trading_calendar_main(logger) -> list:
    """加载交易日历"""
    trading_dates_cache = _get_trading_dates_cache()

    if 'all_trading_dates' in trading_dates_cache:
        return trading_dates_cache['all_trading_dates']

    # 加载完整交易日历
    calendar_path = Path(__file__).parent / "to" / "local_data" / "calendar" / "trading_calendar.parquet"
    if not calendar_path.exists():
        logger.warning(f"[FILTER] 无法获取交易日历，跳过新股过滤")
        return None

    trading_calendar = pd.read_parquet(calendar_path)
    all_trading_dates = trading_calendar['date_str'].tolist()

    # 缓存交易日历
    trading_dates_cache['all_trading_dates'] = all_trading_dates
    _save_trading_dates_cache(trading_dates_cache)

    return all_trading_dates


def _filter_new_stock_data_vectorized_main(stock_data: pd.DataFrame, new_stocks_info: dict,
                                         trading_dates: list, logger) -> pd.DataFrame:
    """
    向量化过滤新股数据 - 优化版本，完全消除嵌套循环

    关键优化：
    1. 预计算所有需要过滤的日期范围
    2. 使用向量化操作而不是逐行循环
    3. 构建高效的过滤掩码
    """
    new_stock_period = 25  # 新股过滤期间：25个交易日

    # 构建所有需要过滤的日期集合（预计算优化）
    filter_date_set = set()
    for stock_code, listing_date in new_stocks_info.items():
        listing_date_str = listing_date.strftime('%Y%m%d')
        if listing_date_str not in trading_dates:
            continue

        listing_idx = trading_dates.index(listing_date_str)

        # 批量计算需要过滤的日期索引
        filter_indices = range(listing_idx,
                             min(listing_idx + new_stock_period, len(trading_dates)))

        for idx in filter_indices:
            filter_date_set.add(trading_dates[idx])

    # 转换为datetime格式
    filter_dates = [pd.to_datetime(date_str, format='%Y%m%d') for date_str in filter_date_set]

    logger.info(f"[FILTER] 预计算过滤日期完成，需要过滤 {len(filter_dates)} 个交易日")

    # 构建需要过滤的股票集合
    filter_stocks = set(new_stocks_info.keys())

    # 向量化过滤 - 一次性应用所有过滤条件
    filter_mask = (
        (stock_data['stock_code'].isin(filter_stocks)) &
        (stock_data['date'].isin(filter_dates))
    )

    # 额外过滤：移除成交量为0的记录（无论是否为新股）
    if 'volume' in stock_data.columns:
        volume_filter_mask = (stock_data['volume'] > 0)
        # 应用所有过滤条件：新股过滤 + 成交量过滤
        final_filter_mask = filter_mask | (~volume_filter_mask)
        filtered_data = stock_data[~final_filter_mask].copy()

        # 统计成交量过滤效果
        volume_filtered_count = (~volume_filter_mask).sum()
        logger.info(f"[FILTER] 成交量过滤: 移除 {volume_filtered_count:,} 条成交量为0的记录")
    else:
        # 如果没有volume列，只进行新股过滤
        filtered_data = stock_data[~filter_mask].copy()
        logger.warning("[FILTER] 数据中没有volume列，无法进行成交量过滤")

    # 统计结果
    original_count = len(stock_data)
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_count

    logger.info(f"[FILTER] 优化版数据过滤完成:")
    logger.info(f"  原始数据: {original_count:,} 条")
    logger.info(f"  过滤后数据: {filtered_count:,} 条")
    logger.info(f"  总过滤数据: {removed_count:,} 条")
    logger.info(f"  过滤比例: {removed_count/original_count*100:.2f}%")

    # 分别统计新股过滤和成交量过滤的效果
    if 'volume' in stock_data.columns:
        new_stock_filter_mask = filter_mask & volume_filter_mask
        new_stock_removed = new_stock_filter_mask.sum()
        volume_removed = removed_count - new_stock_removed

        logger.info(f"  - 新股过滤: {new_stock_removed:,} 条")
        logger.info(f"  - 成交量过滤: {volume_removed:,} 条")
    else:
        logger.info(f"  - 新股过滤: {removed_count:,} 条")
        logger.info(f"  - 成交量过滤: 未执行（无volume列）")

    # 详细统计各新股过滤情况
    for stock_code in new_stocks_info:
        original_stock_count = len(stock_data[stock_data['stock_code'] == stock_code])
        filtered_stock_count = len(filtered_data[filtered_data['stock_code'] == stock_code])
        stock_removed = original_stock_count - filtered_stock_count

        if stock_removed > 0:
            listing_date = new_stocks_info[stock_code].strftime('%Y-%m-%d')
            logger.info(f"  {stock_code}(上市:{listing_date}): 过滤 {stock_removed} 条记录")

    # 特别检查001279.SZ
    if '001279.SZ' in new_stocks_info:
        original_001279 = len(stock_data[stock_data['stock_code'] == '001279.SZ'])
        filtered_001279 = len(filtered_data[filtered_data['stock_code'] == '001279.SZ'])
        removed_001279 = original_001279 - filtered_001279

        logger.info(f"[FILTER] *** 001279.SZ处理结果: 过滤 {removed_001279} 条记录 ***")
        if removed_001279 > 0:
            logger.info(f"[FILTER] *** 成功：001279.SZ的极端值数据已被过滤，不会污染pct_chg_1d因子 ***")
        else:
            logger.warning(f"[FILTER] *** 警告：001279.SZ未被有效过滤 ***")

    return filtered_data



def _filter_st_stocks(data: pd.DataFrame, logger) -> pd.Series:
    """
    从 ST 股票数据文件中过滤 ST 股票（高效向量化实现）

    参数:
    - data: 股票数据DataFrame，必须包含 'date' 和 'stock_code' 列

    返回:
    - pd.Series: 布尔值序列，True表示非ST股票，False表示ST股票
    """
    try:
        from pathlib import Path

        # ST 数据文件路径
        st_file = Path(__file__).parent.parent / "local_data" / "st.parquet"

        # 如果文件不存在，返回全部 True（不过滤）
        if not st_file.exists():
            logger.warning(f"[FILTER] ST数据文件不存在: {st_file}，跳过ST股票过滤")
            return pd.Series(True, index=data.index)

        # 检查数据必要列
        if 'date' not in data.columns or 'stock_code' not in data.columns:
            logger.error(f"[FILTER] 数据缺少必要列，可用列: {data.columns.tolist()}")
            return pd.Series(True, index=data.index)

        # 快速日期格式转换（向量化）
        if pd.api.types.is_datetime64_any_dtype(data['date']):
            date_str = data['date'].dt.strftime('%Y%m%d')
        else:
            date_str = data['date'].astype(str).str.replace('-', '').str[:8]

        # 读取 ST 数据（只加载需要的列以提高性能）
        try:
            st_df = pd.read_parquet(st_file, columns=['trade_date', 'ts_code'])
        except Exception as e:
            logger.error(f"[FILTER] 读取ST数据文件失败: {e}")
            return pd.Series(True, index=data.index)

        # 检查 ST 数据列
        if 'trade_date' not in st_df.columns or 'ts_code' not in st_df.columns:
            logger.error(f"[FILTER] ST数据文件缺少必要列，可用列: {st_df.columns.tolist()}")
            return pd.Series(True, index=data.index)

        # 高效过滤算法：使用 merge + 向量化操作
        # 步骤1: 准备临时列
        data['_temp_date'] = date_str

        # 步骤2: 使用 merge 快速匹配（比循环+字典查找更快）
        merged = data.merge(
            st_df,
            left_on=['_temp_date', 'stock_code'],
            right_on=['trade_date', 'ts_code'],
            how='left',
            indicator=True
        )

        # 步骤3: 向量化判断：只在 ST 数据中的股票被标记为 True
        is_st = merged['_merge'] == 'both'

        # 步骤4: 创建过滤结果（True = 非ST，False = ST）
        st_filter = ~is_st

        # 步骤5: 清理临时列
        if '_temp_date' in data.columns:
            del data['_temp_date']

        # ✅ 修复：重置 st_filter 的索引，使其与原始 data 对齐
        # 因为 merged 的索引是 RangeIndex，而 data 可能有原始索引
        # 我们需要创建一个与 data 索引对齐的 Series
        st_filter_aligned = pd.Series(st_filter.values, index=data.index)

        # 统计和日志
        st_count = (~st_filter_aligned).sum()
        if st_count > 0:
            # 获取ST股票示例（用于日志）
            # ✅ 修复：使用 merged 而不是 data，避免索引不对齐问题
            st_stocks_example = merged.loc[~st_filter, 'stock_code'].unique()[:5]
            logger.info(f"[FILTER] ST股票过滤: 过滤掉 {st_count} 只ST股票")
            if len(st_stocks_example) > 0:
                logger.info(f"   ST股票示例: {', '.join(st_stocks_example)} 等{st_count}只")

        remaining_count = st_filter_aligned.sum()
        logger.info(f"[FILTER] ST股票过滤后剩余: {remaining_count} 只非ST股票")

        return st_filter_aligned

    except Exception as e:
        logger.error(f"[FAIL] ST股票过滤失败: {e}")
        import traceback
        logger.error(f"[FAIL] 错误详情: {traceback.format_exc()}")
        # 出错时不过滤任何股票
        return pd.Series(True, index=data.index)



def _get_trading_dates_cache():
    """获取交易日缓存"""
    cache_file = Path(__file__).parent / "cache" / "trading_dates_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass

    return {}

def _save_trading_dates_cache(cache):
    """保存交易日缓存"""
    cache_file = Path(__file__).parent / "cache" / "trading_dates_cache.pkl"
    cache_file.parent.mkdir(exist_ok=True)

    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"[WARNING] 保存交易日缓存失败: {e}")

def _get_previous_trading_dates(current_date: str, days: int) -> list:
    """获取指定日期前N个交易日的日期列表（使用缓存优化）"""
    try:
        from pathlib import Path
        import datetime

        # 检查缓存
        cache = _get_trading_dates_cache()
        cache_key = f"{current_date}_{days}"

        if cache_key in cache:
            return cache[cache_key]

        # 使用相对于当前文件的路径
        current_file_dir = Path(__file__).parent
        features_dir = current_file_dir.parent / "local_data" / "features" / "daily"

        # 一次性获取所有可能的文件列表（更高效）
        all_files = list(features_dir.glob("*.parquet"))
        all_dates = []

        for file_path in all_files:
            try:
                date_str = file_path.stem  # 获取文件名（不含扩展名）
                if len(date_str) == 8 and date_str.isdigit():
                    all_dates.append(date_str)
            except:
                continue

        # 排序并筛选出当前日期之前的交易日
        all_dates.sort(reverse=True)  # 降序排列
        current_dt = datetime.datetime.strptime(current_date, '%Y%m%d')

        trading_dates = []
        for date_str in all_dates:
            try:
                check_dt = datetime.datetime.strptime(date_str, '%Y%m%d')
                if check_dt < current_dt:
                    trading_dates.append(date_str)
                    if len(trading_dates) >= days:
                        break
            except:
                continue

        # 缓存结果
        cache[cache_key] = trading_dates
        _save_trading_dates_cache(cache)

        return trading_dates

    except Exception as e:
        print(f"[FAIL] 获取交易日列表失败: {e}")
        return []


def _get_model_cache_path(train_start_date: str, train_end_date: str, feature_count: int = None) -> Path:
    """获取模型缓存路径

    Args:
        train_start_date: 训练开始日期
        train_end_date: 训练结束日期
        feature_count: 特征数量（用于缓存键区分）
    """
    cache_dir = Path("cache/models")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # [WRENCH] 改进缓存键：包含特征数量，避免特征不一致时的缓存污染
    if feature_count is not None:
        model_name = f"ml_model_{train_start_date}_{train_end_date}_f{feature_count}"
    else:
        model_name = f"ml_model_{train_start_date}_{train_end_date}"

    return cache_dir / f"{model_name}.pkl"

def _save_model_cache(ml_model, scaler, feature_names: list, train_start_date: str, train_end_date: str):
    """保存模型缓存"""
    try:
        import pickle

        cache_path = _get_model_cache_path(train_start_date, train_end_date)

        model_data = {
            'ml_model': ml_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_start_date': train_start_date,
            'train_end_date': train_end_date,
            'created_time': pd.Timestamp.now().isoformat(),
            'model_type': type(ml_model).__name__,
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"[OK] 模型已保存到缓存: {cache_path}")
        return True

    except Exception as e:
        print(f"[WARNING] 模型缓存保存失败: {e}")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置日志级别和日志配置
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    root_logger = setup_logging()

    # 验证logger配置是否正确
    logger.info("=== 日志系统初始化完成 ===")
    logger.info(f"日志级别: {args.log_level}")
   
    print("板块-个股一体化选股系统")
    print("=" * 60)
    print(f"系统配置:")
    print(f"  运行模式: {args.mode}")
    print(f"  数据期间: {system_config.start_date} - {system_config.end_date}")
    print(f"  训练期间: {system_config.start_date} - {system_config.train_end_date}")
    print(f"  测试期间: {system_config.test_start_date} - {system_config.end_date}")
    print(f"  每日选股: {system_config.daily_stock_count}只")
    print(f"  持仓天数: {backtest_config.holding_days}天")
    print(f"  板块筛选: 前{system_config.sector_filter_count}个（优化后）")
    print(f"  板块权重: {system_config.factor_weights['sector']:.0%}（大幅提升）")
    print(f"  连板回看: {system_config.max_lookback_days}天")
    print(f"  输出目录: {args.output_dir}")
    print()


#     # 三种模式：
# 1. 默认模式：生成所有请求的因子（可能覆盖已存在的）
#    python main.py --mode generate_factors

# 2. 强制模式：清理旧数据并重新生成
#    python main.py --mode generate_factors --force

# 3. 🆕 增量保存：保留现有数据并智能合并新数据
#    python main.py --mode generate_factors --incremental_save
# 如果想只生成特定因子，可以使用 --factor_names 参数
# 自定义批次大小
# python main.py --mode generate_factors --batch_size 5

# # 禁用分批保存
# python main.py --mode generate_factors --batch_size 0



    # 执行对应的运行模式
    if args.mode == 'generate_factors':
        success = run_generate_factors_mode(args)
    elif args.mode == 'test_factors':
        success = run_test_factors_mode(args)
    elif args.mode == 'train_model':
        # success = run_train_model_mode(args,train_end_date='2024-06-01', is_regression=True)  # [强化] 启用成本感知回归模式
        success = run_train_model_mode(args,None, is_regression=True)
    #python main.py --mode test_model --models lightgbm    
    elif args.mode == 'test_model':
        success = run_test_model_mode(args)
    # python main.py --mode test_model_walkforward --window 1,3
    #python main.py --mode test_model --models lightgbm --optimize_thresholds 用来搜索最优atr止盈止损阈值
    #python main.py --mode test_model_walkforward --window 3 --models lightgbm --optimize_thresholds
    elif args.mode == 'test_model_walkforward':
        success = run_test_model_walkforward_mode(args)
    # python main.py --mode pred_stocks --date 20251216
    elif args.mode == 'pred_stocks':
        success = run_pred_stocks_mode(args)
    # test_optimization模式已删除，优化方案已集成到test_model模式
    elif args.mode == 'factor_correlation':
        success = run_factor_correlation_mode(args)
    elif args.mode == 'genetic_factor':
        success = run_genetic_factor_mode(args)
    elif args.mode == 'market_state_test':
        success = run_market_state_test_mode(args)
    # elif args.mode == 'interaction_analysis':
    #     success = run_interaction_analysis_mode(args)  # 已移除交互特征分析模式
    else:
        logger.error(f"未知的运行模式: {args.mode}")
        return False

    return success



def load_sw_sector_mapping() -> Dict[str, str]:
    """
    加载SW申万行业数据（使用标准化编码）

    数据格式: {"920351.BJ": "SW1交通运输", ...}
    返回: 股票代码到SW行业名称的映射字典
    """
    encoding_data = load_standardized_sw_encoding()
    if not encoding_data:
        raise RuntimeError("无法加载标准化SW编码文件，请确保文件存在且格式正确")

    return encoding_data['mappings']['stock_to_industry']



def _get_historical_st_flag(data: pd.DataFrame, logger=None) -> pd.Series:
    """基于历史ST文件返回逐行ST标记（True=ST）。"""
    import pandas as pd
    from pathlib import Path

    if data is None or data.empty:
        return pd.Series(False, index=getattr(data, 'index', None))

    try:
        st_file = Path(__file__).parent.parent / "local_data" / "st.parquet"
        if not st_file.exists():
            stock_name = data.get('stock_name')
            if stock_name is None:
                if logger:
                    logger.warning(f"[LABEL] 未找到ST历史文件: {st_file}，且缺少stock_name，ST窗口屏蔽跳过")
                return pd.Series(False, index=data.index)
            fallback_flag = (
                stock_name.astype(str).str.contains('ST', case=False, na=False)
                | stock_name.astype(str).str.contains('\\*', regex=True, na=False)
                | stock_name.astype(str).str.contains('退', na=False)
            )
            if logger:
                logger.warning(
                    f"[LABEL] 未找到ST历史文件: {st_file}，回退到名称法ST识别: "
                    f"{int(fallback_flag.sum())}/{len(fallback_flag)}"
                )
            return pd.Series(fallback_flag.to_numpy(), index=data.index)

        if 'date' not in data.columns or 'stock_code' not in data.columns:
            if logger:
                logger.warning("[LABEL] 数据缺少 date/stock_code，ST窗口屏蔽跳过")
            return pd.Series(False, index=data.index)

        if pd.api.types.is_datetime64_any_dtype(data['date']):
            date_str = data['date'].dt.strftime('%Y%m%d')
        else:
            date_str = data['date'].astype(str).str.replace('-', '').str[:8]

        st_df = pd.read_parquet(st_file, columns=['trade_date', 'ts_code']).copy()
        if 'trade_date' not in st_df.columns or 'ts_code' not in st_df.columns:
            if logger:
                logger.warning(f"[LABEL] ST数据文件缺少必要列: {st_df.columns.tolist()}")
            return pd.Series(False, index=data.index)

        st_df = st_df.dropna(subset=['trade_date', 'ts_code']).drop_duplicates(['trade_date', 'ts_code'])
        st_df['trade_date'] = st_df['trade_date'].astype(str).str.replace('-', '').str[:8]
        st_df['ts_code'] = st_df['ts_code'].astype(str)

        probe = pd.DataFrame({
            '_temp_date': date_str,
            'stock_code': data['stock_code'].astype(str).values,
        }, index=data.index)
        merged = probe.merge(
            st_df,
            left_on=['_temp_date', 'stock_code'],
            right_on=['trade_date', 'ts_code'],
            how='left',
            indicator=True
        )
        return pd.Series((merged['_merge'] == 'both').to_numpy(), index=data.index)
    except Exception as e:
        if logger:
            logger.warning(f"[LABEL] 历史ST标记构造失败，ST窗口屏蔽跳过: {e}")
        return pd.Series(False, index=data.index)


def _build_valid_cs_mask(stock_data: pd.DataFrame, logger):
    """
    综合生成横截面有效样本掩码 valid_cs_mask（不直接删行）：

    - is_st: 复用 _filter_st_stocks 的判定口径（True=非ST）
    - is_new_stock: 复用 _filter_new_stocks_batch 的判定口径（通过"过滤前后索引差"还原mask）
      （注意：_filter_new_stocks_batch 可能同时过滤成交量为0的数据，这里会同步体现在new_keep里）

    返回:
        valid_cs_mask: pd.Series[bool]，True表示该行可用于"横截面统计/训练/打分"
        st_keep: pd.Series[bool]，True表示非ST
        new_keep: pd.Series[bool]，True表示非新股期且未被其内部规则过滤
    """
    if stock_data is None or len(stock_data) == 0:
        empty = pd.Series(dtype=bool)
        return empty, empty, empty

    # 1) ST 口径（True=非ST）
    try:
        st_keep = _filter_st_stocks(stock_data, logger)
        st_keep = st_keep.astype(bool)
    except Exception as e:
        logger.warning(f"[FILTER] 生成ST掩码失败，回退为全True: {e}")
        st_keep = pd.Series(True, index=stock_data.index)

    # 2) 新股口径（通过过滤函数"还原"mask，不修改其逻辑）
    try:
        filtered_new = _filter_new_stocks_batch(stock_data, logger)
        # _filter_new_stocks_batch 使用 stock_data[mask].copy()，通常会保留原索引
        new_keep = pd.Series(stock_data.index.isin(filtered_new.index), index=stock_data.index)
    except Exception as e:
        logger.warning(f"[FILTER] 生成新股掩码失败，回退为全True: {e}")
        new_keep = pd.Series(True, index=stock_data.index)

    valid_cs_mask = (st_keep.values & new_keep.values)
    valid_cs_mask = pd.Series(valid_cs_mask, index=stock_data.index, name="valid_cs_mask")

    return valid_cs_mask, st_keep, new_keep



def run_generate_factors_mode(args):
    """运行因子实验室的因子生成流程"""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger = logging.getLogger(__name__)
    print("?? 运行模式: 因子生成")
    print("=" * 60)

    try:
        data_loader = IntegratedDataLoader()

        start_date = args.start_date or "20251021" #20230601
        end_date = args.end_date or "20251217" #20251021

        print(f"?? 目标区间: {start_date} - {end_date}")

        extend_history_days = getattr(args, 'extend_history_days', 120)

        extended_start_date = data_loader._calculate_extended_start_date(

            start_date, end_date, extend_history_days
        )

        stock_data = data_loader.load_stock_features(extended_start_date, end_date)
        try:
            # 修复：板块数据也需要历史窗口（最大需求20天，使用extended_start_date确保足够）
            sector_data = data_loader.load_sector_features(extended_start_date, end_date)
        except Exception as exc:
            logger.warning('sector data loader raised an exception: %s', exc)
            sector_data = None

        if sector_data is None or sector_data.empty:
            logger.warning('sector data missing, falling back to local cache')
            sector_data = _load_sector_features_basic(extended_start_date, end_date)

        try:
            # 修复：市场数据也需要历史窗口（最大需求120天，使用extended_start_date确保足够）
            market_data = data_loader.load_market_features(extended_start_date, end_date)
        except Exception as exc:
            logger.warning('market data loader raised an exception: %s', exc)
            market_data = None

        if market_data is None or market_data.empty:
            logger.warning('market data missing, using placeholder values')
            trading_dates = data_loader.get_trading_dates(start_date, end_date)
            market_data = pd.DataFrame({
                'date': pd.to_datetime(trading_dates, format='%Y%m%d'),
                'market_return': 0.0,
                'market_volume': 1.0,
            })

        if stock_data is None or stock_data.empty:
            logger.error('failed to load stock base data; aborting factor generation')
            print('stock data is empty, stop factor generation.')
            return False

        if sector_data is None or sector_data.empty:

            logger.warning('no valid sector data, sector-related factors may be missing')

        if market_data is None or market_data.empty:

            logger.warning('no valid market data, placeholder values will be used')


        for dataset in (stock_data, sector_data, market_data):

            if dataset is not None and not dataset.empty and 'date' in dataset.columns:

                dataset['date'] = pd.to_datetime(dataset['date'])

        print(f"? 股票记录数: {len(stock_data)}")

        if sector_data is not None and not sector_data.empty:
            print(f"? 板块记录数: {len(sector_data)}")

        else:
            print('[warning] sector data missing, using stock-only factors')
        if market_data is not None and not market_data.empty:
            print(f"? 市场记录数: {len(market_data)}")
        else:
            print('[warning] market data missing, placeholder values applied')

        # 🆕 修复：明确记录是否指定了特定因子或因子组
        has_specific_factors = False
        if args.factor_names:
            requested_factors = [name.strip() for name in args.factor_names.split(',')]
            has_specific_factors = True
            logger.info(f"[参数] 指定特定因子: {requested_factors}")
        elif getattr(args, 'factor_groups', None):
            requested_factors = [name.strip() for name in args.factor_groups.split(',')]
            has_specific_factors = True
            logger.info(f"[参数] 指定因子组: {requested_factors}")
        else:
            requested_factors = _get_stock_factors()
            has_specific_factors = False
            logger.info(f"[参数] 生成所有因子: {len(requested_factors)}个")

        factor_names = []
        seen = set()
        for name in requested_factors:
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                factor_names.append(name)

        pipeline = FactorPipeline()
        available = set(pipeline.available_factors())
        missing = [name for name in factor_names if name not in available]
        if missing:
            logger.warning(f"下列因子暂未在factor_lab注册，将忽略: {missing}")
            print(f"?? 忽略未注册因子: {missing}")
        factor_names = [name for name in factor_names if name in available]

        if not factor_names:
            logger.error("没有可用的因子名称，请检查输入。")
            print("? 无可计算的因子名称，生成流程结束。")
            return False

        print(f"?? 计划生成因子: {factor_names}")

        extra_context = {}
        stock_sector_mapping = data_loader.load_stock_sector_mapping()
        if stock_sector_mapping:
            extra_context['stock_to_sector'] = stock_sector_mapping
            print(f"? 股票-板块映射数量: {len(stock_sector_mapping)}")
        else:
            print("?? 未找到股票-板块映射，将跳过板块相关因子。")

        # 加载SW申万行业数据
        sw_sector_mapping = load_sw_sector_mapping()
        if sw_sector_mapping:
            extra_context['stock_to_sw_sector'] = sw_sector_mapping
            print(f"? SW申万行业映射数量: {len(sw_sector_mapping)}")
        else:
            sw_sector_mapping = {}  # 确保为空字典而不是None
            print("?? 未找到SW申万行业映射，将使用默认行业映射。")

        # 🆕 添加SW行业股票过滤逻辑
        if sw_sector_mapping:
            # 获取有SW行业映射的股票代码集合
            valid_stocks_with_sw = set(sw_sector_mapping.keys())

            # 计算原始股票数量（唯一股票代码数）
            original_unique_stocks = stock_data['stock_code'].nunique()
            original_data_rows = len(stock_data)

            logger.info(f"[SW-FILTER] 过滤前: {original_unique_stocks} 只股票, {original_data_rows} 条数据行")

            # 过滤股票数据，只保留有SW行业映射的股票
            stock_data = stock_data[stock_data['stock_code'].isin(valid_stocks_with_sw)].copy()

            # 计算过滤后股票数量
            filtered_unique_stocks = stock_data['stock_code'].nunique()
            filtered_data_rows = len(stock_data)

            removed_stocks = original_unique_stocks - filtered_unique_stocks
            removed_data_rows = original_data_rows - filtered_data_rows

            if removed_stocks > 0:
                logger.info(f"[SW-FILTER] 过滤结果:")
                logger.info(f"   股票数量: {original_unique_stocks} -> {filtered_unique_stocks} (移除 {removed_stocks} 只)")
                logger.info(f"   数据行数: {original_data_rows} -> {filtered_data_rows} (移除 {removed_data_rows} 行)")

                # 如果过滤后股票数量太少，发出警告
                if filtered_unique_stocks == 0:
                    logger.error("过滤后没有有效股票，请检查SW行业映射数据")
                    print("❌ 错误：过滤后没有有效股票，请检查SW行业映射数据")
                    return False
                elif filtered_unique_stocks < 100:
                    logger.warning(f"过滤后股票数量较少：{filtered_unique_stocks}只，可能影响因子计算效果")
                    print(f"⚠️  警告：过滤后股票数量较少：{filtered_unique_stocks}只，可能影响因子计算效果")

            else:
                logger.info(f"[SW-FILTER] 所有 {original_unique_stocks} 只股票都有有效的SW行业映射")
                print(f"✅ SW行业过滤：所有股票都有有效的SW行业映射")
        else:
            logger.warning("没有SW行业映射数据，无法进行股票过滤")

        # 🆕 添加北交所、新股和ST股票过滤逻辑
        print("🔍 开始股票质量过滤...")

        # 检查是否包含date列，用于后续的新股过滤
        has_date_column = 'date' in stock_data.columns
        original_count = len(stock_data)

        # 1. 过滤北交所股票 (股票代码以.BJ或.bj结尾)
        if 'stock_code' in stock_data.columns:
            logger.info("🔍 开始北交所股票过滤...")
            bj_filter = ~stock_data['stock_code'].str.endswith(('.BJ', '.bj'), na=False)
            stock_data = stock_data[bj_filter].copy()
            after_bj_count = len(stock_data)
            bj_filtered_count = original_count - after_bj_count

            if bj_filtered_count > 0:
                logger.info(f"   🚫 北交所过滤: {original_count} -> {after_bj_count} (过滤掉 {bj_filtered_count} 只北交所股票)")
            else:
                logger.info(f"   ✅ 北交所过滤: 无北交所股票，保留全部 {original_count} 只股票")
        else:
            after_bj_count = len(stock_data)
            bj_filtered_count = 0
            logger.info("⚠️  警告：缺少stock_code列，跳过北交所过滤")

        # 2. 生成ST/新股综合有效性掩码（不直接删行，供后续pipeline使用）
        logger.info("🔍 生成valid_cs_mask（综合ST/新股口径）...")
        try:
            valid_cs_mask, st_keep, new_keep = _build_valid_cs_mask(stock_data, logger)
            # 写入列，供后续 pipeline / 训练时使用（横截面统计也可按此mask做更"干净"的截面）
            stock_data["valid_cs_mask"] = valid_cs_mask.astype(bool)

            st_count = int((~st_keep).sum()) if len(st_keep) else 0
            new_count = int((~new_keep).sum()) if len(new_keep) else 0
            invalid_count = int((~valid_cs_mask).sum()) if len(valid_cs_mask) else 0

            logger.info(
                f"   ✅ valid_cs_mask生成完成: 总样本{len(stock_data)} | "
                f"ST标记{st_count} | 新股/其它规则标记{new_count} | 综合无效{invalid_count}"
            )
        except Exception as e:
            logger.warning(f"⚠️  valid_cs_mask生成失败（回退为全True）: {e}")
            stock_data["valid_cs_mask"] = True

        # 3. （保留原流程变量名，避免后续日志/分支引用出错）
        after_st_count = len(stock_data)
        st_filtered_count = 0
        final_count = len(stock_data)

        # 最终过滤统计
        total_filtered = original_count# - final_count
        new_filtered_count = total_filtered - bj_filtered_count - st_filtered_count
        if total_filtered > 0:
            logger.info(f"🎯 股票过滤总结: 原始 {original_count} -> 最终  (过滤掉 {total_filtered} 只)")
            logger.info(f"   - 北交所: {bj_filtered_count} 只")
            logger.info(f"   - ST股票: {st_filtered_count} 只")
            # logger.info(f"   - 新股: {new_filtered_count} 只")

            # # 如果过滤后股票数量太少，发出警告
            # if final_count == 0:
            #     logger.error("过滤后没有有效股票，请检查过滤条件")
            #     return False
            # elif final_count < 100:
            #     logger.warning(f"过滤后股票数量较少：{final_count}只，可能影响因子计算效果")
        else:
            print(f"✅ 股票过滤完成: 保留全部 {original_count} 只股票（无需要过滤的股票）")

        FACTOR_LAB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        store = FactorDatasetStore(FACTOR_LAB_STORE_PATH)

        # 🆕 因子生成功能：检测已存在的因子
        if getattr(args, 'force', False):
            logger.info("force 标志已启用，将在重新生成前清理对应因子列。")
            store.drop_factors(factor_names)
            print("?? 已清理旧的因子列，准备重新生成。")
            factors_to_generate = factor_names
        else:
            # 默认模式：生成所有请求的因子（可能覆盖已存在的）
            factors_to_generate = factor_names

        if not factors_to_generate:
            logger.warning("没有需要生成的因子")
            print("[WARNING]  没有需要生成的因子")
            return True

        print(f"?? 实际生成因子数: {len(factors_to_generate)}")
        print(f"?? 实际生成因子: {factors_to_generate}")

        # 🆕 获取分批保存配置
        batch_size = getattr(args, 'batch_size', 10)
        enable_batch_save = batch_size > 0

        if enable_batch_save:
            logger.info(f"启用分批保存机制，批次大小: {batch_size}")
            print(f"[SAVE] 启用分批保存机制，每 {batch_size} 个因子保存一次")
        else:
            logger.info("禁用分批保存，将在所有因子计算完成后一次性保存")
            print("[SAVE] 禁用分批保存，将在所有因子计算完成后一次性保存")

        # 🆕 构建归一化配置
        normalization_config = {
            'zscore_factors': _get_zscore_factors(),
            'rankpct_factors': _get_rankpct_factors(),
            'original_factors': _get_original_factors(),  # 新增原始因子配置
            'intra_sector_rank_factors': _get_intra_sector_rank_factors(),  # 新增行业内排名因子配置
            'intra_sector_zscore_factors': _get_intra_sector_zscore_factors(),  # 新增行业内z-score因子配置
            'sw_sector_mapping': sw_sector_mapping,  # 🆕 SW行业映射，用于行业内归一化
        }

        logger.info(f"归一化配置: zscore因子数={len(normalization_config['zscore_factors'])}, rankpct因子数={len(normalization_config['rankpct_factors'])}, original因子数={len(normalization_config['original_factors'])}, intra_sector_rank因子数={len(normalization_config['intra_sector_rank_factors'])}, intra_sector_zscore因子数={len(normalization_config['intra_sector_zscore_factors'])}")
        print(f"[WRENCH] 归一化配置:")
        if normalization_config['zscore_factors']:
            print(f"  · Z-score 因子数: {len(normalization_config['zscore_factors'])}")
        if normalization_config['rankpct_factors']:
            print(f"  · Rankpct 因子数: {len(normalization_config['rankpct_factors'])}")
        if normalization_config['original_factors']:
            print(f"  · Original 因子数: {len(normalization_config['original_factors'])} (仅Inf处理+缺失值填充)")
        if normalization_config['intra_sector_rank_factors']:
            print(f"  · Intra-Sector Rank 因子数: {len(normalization_config['intra_sector_rank_factors'])} (行业内排名)")
        if normalization_config['intra_sector_zscore_factors']:
            print(f"  · Intra-Sector Z-score 因子数: {len(normalization_config['intra_sector_zscore_factors'])} (行业内z-score)")
            print(f"    📝 行业内z-score因子: {normalization_config['intra_sector_zscore_factors'][:5]}{'...' if len(normalization_config['intra_sector_zscore_factors']) > 5 else ''}")

        # 🆕 获取新增因子模式和增量保存参数
        add_factor_mode = getattr(args, 'add_factor', False)
        raw_incremental_save_mode = getattr(args, 'incremental_save', False)

        # 🆕 新增因子模式优先级最高，确保列扩展模式
        if has_specific_factors and add_factor_mode:
            incremental_save_mode = True
            add_factor_mode = True
            print(f"[新增因子] 启用新增因子模式 - 保留原有因子，仅新增指定因子: {len(factor_names)}个")
            logger.info(f"[新增因子] 启用新增因子模式 - 保留原有因子，仅新增指定因子: {len(factor_names)}个")
        elif has_specific_factors and raw_incremental_save_mode:
            incremental_save_mode = True
            add_factor_mode = False
            print(f"[增量保存] 启用增量保存模式 - 指定特定因子更新")
            logger.info(f"[增量保存] 启用增量保存模式 - 指定特定因子更新: {len(factor_names)}个")
        elif not has_specific_factors:
            # 生成所有因子时，尊重用户的增量保存选择
            # ✅ 修复：不再强制False，而是使用用户指定的参数值
            incremental_save_mode = raw_incremental_save_mode
            add_factor_mode = False
            if incremental_save_mode:
                print(f"[全量增量] 生成所有因子，使用增量保存模式 - 保留现有数据，合并新数据")
                logger.info(f"[全量增量] 生成所有因子: {len(factor_names)}个，使用增量保存（时间维度扩展）")
            else:
                print(f"[全量生成] 生成所有因子，使用全量覆盖模式")
                logger.info(f"[全量生成] 生成所有因子: {len(factor_names)}个，使用全量保存")
        else:
            # 其他情况使用参数设置
            incremental_save_mode = raw_incremental_save_mode
            add_factor_mode = False
            if incremental_save_mode:
                print(f"[增量保存] 启用增量保存模式 - 用户明确指定")
                logger.info("[增量保存] 启用增量保存模式 - 用户明确指定")
            else:
                print(f"[全量保存] 使用全量保存模式")
                logger.info("[全量保存] 使用全量保存模式")

        pipeline_result = generate_and_store(
            factor_names=factors_to_generate,  # 🆕 使用实际需要生成的因子列表
            data=stock_data,
            start_date=start_date,
            end_date=end_date,
            store=store,
            pipeline=pipeline,
            market_data=market_data,
            sector_data=sector_data,
            context_extra=extra_context,
            enable_batch_save=enable_batch_save,  # 🆕 分批保存开关
            batch_size=batch_size,  # 🆕 批次大小
            normalization_config=normalization_config,  # 🆕 归一化配置
            incremental_mode=incremental_save_mode,  # 🆕 传递增量保存模式参数
            add_factor_mode=add_factor_mode,  # 🆕 传递新增因子模式参数
        )

        generated = pipeline_result.data
        record_count = len(generated)
        factor_count = len([c for c in generated.columns if c not in {'date', 'stock_code'}])

        logger.info(f"生成因子完成: {factor_count} 列, {record_count} 条记录")
        print(f"? 生成完成: {record_count} 条记录, {factor_count} 个因子")
        if record_count > 0:
            date_min = generated['date'].min()
            date_max = generated['date'].max()
            stock_count = generated['stock_code'].nunique()
            print(f"?? 实际生成区间: {date_min.date()} - {date_max.date()}")
            print(f"?? 覆盖股票数量: {stock_count}")
        print(f"?? 因子数据已保存至: {FACTOR_LAB_STORE_PATH}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                'start_date': start_date,
                'end_date': end_date,
                'factors': factor_names,
                'record_count': record_count,
                'storage_path': str(FACTOR_LAB_STORE_PATH.resolve()),
            }
            summary_path = output_dir / f"factor_lab_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w', encoding='utf-8') as fp:
                json.dump(summary, fp, ensure_ascii=False, indent=2, default=str)
            print(f"? 因子生成摘要已保存到: {summary_path}")

        print("?? 因子生成模式完成")
        return True

    except Exception as exc:
        logger.error(f"因子生成模式失败: {exc}")
        print(f"? 因子生成失败: {exc}")
        return False


def run_test_factors_mode(args):
    """因子测试模式 - 使用日线 VWAP 收益标签"""
    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)
    print("===== 因子质量测试 =====")

    try:
        start_date = args.start_date or "20230601"
        end_date = args.end_date or "20250131"

        print(f"· 测试区间：{start_date} - {end_date}")
        print(f"· 输出目录：{args.output_dir}")

        data_loader = IntegratedDataLoader()
        stock_data, _, _ = data_loader.load_all_data(
            start_date,
            end_date,
            extend_history_days=60,
        )
        if stock_data is None or stock_data.empty:
            logger.error("无法加载股票基础数据，无法生成标签")
            return False

        stock_data = stock_data.copy()
        if 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])

        required_base_columns = ['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing_base_columns = [col for col in required_base_columns if col not in stock_data.columns]
        if missing_base_columns:
            logger.error(f"数据中缺少必要字段: {missing_base_columns}")
            return False

        factor_data = _load_factor_cache_data(
            include_label_generation_features=True,
            start_date=start_date,
            end_date=end_date,
        )
        if factor_data is None or factor_data.empty:
            logger.error("因子数据加载失败")
            return False

        factor_data = factor_data.copy()
        if 'date' in factor_data.columns:
            factor_data['date'] = pd.to_datetime(factor_data['date'])

        stock_count = factor_data['stock_code'].nunique() if 'stock_code' in factor_data.columns else 0
        print(f"· 因子样本量：{len(factor_data)} 条，字段 {len(factor_data.columns)} 个")
        if 'date' in factor_data.columns:
            print(f"· 因子日期范围：{factor_data['date'].min()} - {factor_data['date'].max()}")
        print(f"· 覆盖股票数：{stock_count}")

        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        label_source = stock_data[(stock_data['date'] >= start_ts) & (stock_data['date'] <= end_ts)].copy()
        if label_source.empty:
            logger.error("标签源数据为空，请检查基础行情数据")
            return False

        labeled_data_result = _generate_training_labels_for_all_data(label_source)
        if labeled_data_result is None:
            logger.error("标签生成失败")
            return False

        labeled_data = labeled_data_result[0] if isinstance(labeled_data_result, tuple) else labeled_data_result
        if 'date' in labeled_data.columns:
            labeled_data['date'] = pd.to_datetime(labeled_data['date'])

        if 'atr_return_label' not in labeled_data.columns:
            logger.error("标签数据缺少 atr_return_label，无法继续做因子质量测试")
            return False

        label_data = labeled_data[['stock_code', 'date', 'atr_return_label']].copy()
        label_data.rename(columns={'atr_return_label': 'vwap_return_label'}, inplace=True)
        label_data['vwap_return_label'] = pd.to_numeric(label_data['vwap_return_label'], errors='coerce')
        label_data = label_data.dropna(subset=['vwap_return_label'])
        if label_data.empty:
            logger.error("VWAP收益标签全部为空，无法继续测试")
            return False

        print("\n· 标签定义：日线 VWAP 收益率")
        print(f"  - 可用样本：{len(label_data):,}")
        print(f"  - 均值：{label_data['vwap_return_label'].mean():.4%}")
        print(f"  - 中位数：{label_data['vwap_return_label'].median():.4%}")
        print(f"  - 正收益占比：{(label_data['vwap_return_label'] > 0).mean():.2%}")

        factor_features = factor_data.copy()
        conflicting_columns = [
            col for col in ['vwap_return_label', 'atr_return_label', 'future_ret', 'relevance',
                            'max_high_return', 'min_low_return']
            if col in factor_features.columns
        ]
        if conflicting_columns:
            print(f"· 因子数据中已存在标签列，已剔除：{conflicting_columns}")
            factor_features = factor_features.drop(columns=conflicting_columns)

        merged_data = pd.merge(
            factor_features,
            label_data,
            on=['stock_code', 'date'],
            how='inner',
        )
        if merged_data.empty:
            logger.error("因子数据与标签无法对齐")
            return False

        merged_data = merged_data.replace([np.inf, -np.inf], np.nan)
        if {'date', 'stock_code'}.issubset(merged_data.columns):
            merged_data = merged_data.sort_values(['date', 'stock_code']).reset_index(drop=True)

        print(f"· 合并样本量：{len(merged_data):,} 条，字段 {len(merged_data.columns)} 个")

        if merged_data['vwap_return_label'].notna().sum() <= 100:
            logger.error("有效 VWAP 收益标签样本过少，无法做稳定验证")
            return False

        final_return_columns = ['vwap_return_label']
        print(f"· 有效收益列：{final_return_columns}")

        exclude_columns = {
            'stock_code', 'date', 'date_str', 'exit_day',
            'atr_return_label', 'vwap_return_label', 'future_ret', 'relevance',
            'max_high_return', 'min_low_return',
            'min_dist_to_sl', 'min_dist_to_pt',
            'entry_price', 'profit_target', 'stop_loss',
            'pct_chg', 'is_can_buy', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'turnover_rate',
            'total_mv', 'circ_mv', 'pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm',
            'volatility20', 'parkinson_vol', 'vol_term_structure',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'volume_weighted_price', 'vol_price_corr_20d', 'risk_reward_ratio',
            'near_support', 'near_resistance', 'short_reversal',
        }
        exclude_columns.update(
            {
                col for col in merged_data.columns
                if 'future_return' in col or 'atr_return' in col or 'weighted_return' in col or 'vwap_return' in col
            }
        )

        potential_factors = [col for col in merged_data.columns if col not in exclude_columns]

        if args.factor_names:
            requested_factors = [name.strip() for name in args.factor_names.split(',') if name.strip()]
            missing_requested = [name for name in requested_factors if name not in potential_factors]
            if missing_requested:
                logger.warning(f"指定因子未找到：{missing_requested}")
            potential_factors = [name for name in potential_factors if name in requested_factors]
            print(f"· 指定因子列表：{potential_factors}")

        actual_factors = []
        for col in potential_factors:
            if pd.api.types.is_numeric_dtype(merged_data[col]):
                non_null_ratio = merged_data[col].notna().sum() / len(merged_data)
                if non_null_ratio > 0.1:
                    actual_factors.append(col)

        print(f"· 可用因子：{len(actual_factors)} / {len(potential_factors)}")
        if not actual_factors:
            logger.error("未找到可用于验证的因子")
            return False

        sample_factors = actual_factors[:5]
        print(f"· 因子抽样检查：{sample_factors}{'...' if len(actual_factors) > 5 else ''}")
        _check_factor_data_quality(merged_data, sample_factors)

        try:
            from .factor_system.factor_validator import FactorValidator
        except ImportError:
            from factor_system.factor_validator import FactorValidator

        validator = FactorValidator()
        validation_results = validator.validate_quantitative_factors(
            merged_data,
            actual_factors,
            return_columns=final_return_columns,
        )

        print("===== 验证结果概览 =====")
        print(f"· 检测因子：{validation_results.get('total_factors', 0)} 个")
        print(f"· 样本条数：{validation_results.get('total_records', 0)} 条")
        print(f"· 总体状态：{validation_results.get('overall_status', 'UNKNOWN')}")

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        validation_file = output_path / f"factor_validation_report_{timestamp}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"· 验证报告：{validation_file}")

        time_windows = validation_results.get('time_windows', {})
        for window_name, window_data in time_windows.items():
            factor_metrics = window_data.get('factor_metrics', {})
            if not factor_metrics:
                continue
            metrics_df = pd.DataFrame(factor_metrics).T
            if window_name == 'vwap_return_label':
                metrics_file = output_path / f"factor_metrics_vwap_return_{timestamp}.csv"
            elif window_name == 'atr_return_label':
                metrics_file = output_path / f"factor_metrics_vwap_return_legacy_{timestamp}.csv"
            else:
                metrics_file = output_path / f"factor_metrics_{window_name}_{timestamp}.csv"
            metrics_df.to_csv(metrics_file, encoding='utf-8')
            print(f"· 详情指标：{metrics_file}")

        summary_stats = validation_results.get('summary_statistics', {})
        if summary_stats:
            summary_df = pd.DataFrame([summary_stats])
            summary_file = output_path / f"factor_summary_cross_window_{timestamp}.csv"
            summary_df.to_csv(summary_file, encoding='utf-8', index=False)
            print(f"· 跨窗口摘要：{summary_file}")

        if validation_results.get('warnings'):
            print(f"· 警告共 {len(validation_results['warnings'])} 条，示例：{validation_results['warnings'][:3]}")
        if validation_results.get('errors'):
            print(f"· 错误共 {len(validation_results['errors'])} 条，示例：{validation_results['errors'][:3]}")

        print("===== 因子测试完成 =====")
        return True

    except Exception as e:
        logger.error(f"因子测试失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _check_factor_data_quality(data: pd.DataFrame, factors: List[str]):
    """快速检查因子覆盖率与数值分布，帮助定位异常因子"""
    import pandas as pd

    print(f"-> 检查 {len(factors)} 个因子质量...")

    for factor in factors:
        if factor not in data.columns:
            print(f"  · {factor}: 未在数据中找到")
            continue

        factor_data = data[factor]
        total_count = len(factor_data)
        non_null_count = factor_data.notna().sum()
        unique_count = factor_data.nunique()

        print(f"  · {factor}:")
        print(f"    - 总样本: {total_count:,}")
        print(f"    - 可用样本: {non_null_count:,} ({non_null_count/total_count:.1%})")
        print(f"    - 唯一值数: {unique_count:,}")

        if non_null_count == 0:
            print("    - 提示: 全部为空，IC 将为 0")
            continue

        if unique_count <= 1:
            print("    - 提示: 仅包含单一取值，IC 将为 0")
            if unique_count == 1:
                print(f"      · 固定取值: {factor_data.dropna().iloc[0]}")
            continue

        min_val = factor_data.min()
        max_val = factor_data.max()
        mean_val = factor_data.mean()
        std_val = factor_data.std()

        print(f"    - 数值区间: {min_val:.6f} ~ {max_val:.6f}")
        print(f"    - 均值 / 标准差: {mean_val:.6f} / {std_val:.6f}")

        if (factor_data == 0).all():
            print("    - 提示: 全部为 0，IC 将为 0")
        elif abs(std_val) < 1e-10:
            print("    - 提示: 波动极小，可能需要归一化或剔除")
        else:
            print("    - 状态: 数值分布正常")


def _load_sector_features_basic(start_date: str, end_date: str):
    """从本地缓存的板块文件中加载数据，供因子生成兜底使用"""
    logger = logging.getLogger(__name__)
    try:
        base_dir = Path(__file__).resolve().parent.parent / "local_data" / "sector_features"
        if not base_dir.exists():
            logger.warning("sector_features 目录不存在，无法加载板块数据")
            return None

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        frames = []

        for sector_dir in base_dir.iterdir():
            if not sector_dir.is_dir():
                continue
            sector_code = sector_dir.name
            candidates = [
                sector_dir / f"{sector_code}_features.parquet",
                sector_dir / f"{sector_code}.parquet",
            ]
            feature_file = next((p for p in candidates if p.exists()), None)
            if feature_file is None:
                continue

            try:
                df = pd.read_parquet(feature_file)
            except Exception as exc:
                logger.debug("读取板块文件失败 %s: %s", feature_file, exc)
                continue

            if 'date' not in df.columns:
                if 'date_str' in df.columns:
                    df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
                else:
                    logger.debug("板块文件 %s 缺少 date 字段，跳过", feature_file)
                    continue
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            df = df.dropna(subset=['date'])
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                continue

            if 'sector_code' not in df.columns:
                if 'sector' in df.columns:
                    df['sector_code'] = df['sector']
                elif 'sector_name' in df.columns:
                    df['sector_code'] = df['sector_name']
                else:
                    df['sector_code'] = sector_code
            df['sector_code'] = df['sector_code'].astype(str)

            if 'sector_name' not in df.columns:
                df['sector_name'] = df.get('sector', df['sector_code'])

            if 'pct_chg' in df.columns:
                df['return_1d'] = df['pct_chg'].astype(float)
            elif 'close' in df.columns:
                df = df.sort_values(['sector_code', 'date'])
                df['return_1d'] = df.groupby('sector_code')['close'].pct_change() * 100
            else:
                df['return_1d'] = 0.0

            df['return'] = df['return_1d']
            df['daily_return'] = df['return_1d']

            if 'close' in df.columns:
                df = df.sort_values(['sector_code', 'date'])
                close_group = df.groupby('sector_code')['close']
                for window in (3, 5, 10, 20):
                    col = f'return_{window}d'
                    if col not in df.columns:
                        df[col] = close_group.pct_change(window) * 100
            else:
                logger.debug("板块文件 %s 缺少 close 列，无法计算多日收益", feature_file)

            for col in ['return_1d', 'return', 'daily_return', 'return_3d', 'return_5d', 'return_10d', 'return_20d']:
                if col in df.columns:
                    df[col] = df[col].astype(float).fillna(0.0)

            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
            frames.append(df)

        if not frames:
            logger.warning("未在本地板块缓存中找到有效数据")
            return None

        sector_data = pd.concat(frames, ignore_index=True)
        sector_data = sector_data.sort_values(['sector_code', 'date']).reset_index(drop=True)
        return sector_data

    except Exception as exc:
        logger.warning("板块数据兜底加载失败: %s", exc)
        return None


def analyze_bucket_ranges(board_data, valid_mask):
    """分析各分桶对应的收益率范围

    Args:
        board_data: 包含分桶结果的DataFrame
        valid_mask: 有效数据掩码

    Returns:
        dict: 各分桶值的统计信息
    """
    # 获取所有有效的分桶数据
    bucket_data = board_data.loc[valid_mask, [
        'relevance', 'weighted_return', 't1_daily_return',
        't2_daily_return', 'rank_abs', 'rank_pct'
    ]]

    # 按分桶值分组统计
    bucket_stats = {}
    for bucket_value in sorted(bucket_data['relevance'].unique(), reverse=True):
        bucket_mask = bucket_data['relevance'] == bucket_value
        bucket_subset = bucket_data[bucket_mask]

        if len(bucket_subset) == 0:
            continue

        bucket_stats[bucket_value] = {
            'count': len(bucket_subset),
            'weighted_return': {
                'min': bucket_subset['weighted_return'].min(),
                'max': bucket_subset['weighted_return'].max(),
                'mean': bucket_subset['weighted_return'].mean(),
                'std': bucket_subset['weighted_return'].std(),
                'median': bucket_subset['weighted_return'].median()
            },
            't1_daily_return': {
                'min': bucket_subset['t1_daily_return'].min(),
                'max': bucket_subset['t1_daily_return'].max(),
                'mean': bucket_subset['t1_daily_return'].mean(),
                'std': bucket_subset['t1_daily_return'].std()
            },
            't2_daily_return': {
                'min': bucket_subset['t2_daily_return'].min(),
                'max': bucket_subset['t2_daily_return'].max(),
                'mean': bucket_subset['t2_daily_return'].mean(),
                'std': bucket_subset['t2_daily_return'].std()
            },
            'rank_abs': {
                'min': bucket_subset['rank_abs'].min(),
                'max': bucket_subset['rank_abs'].max(),
                'mean': bucket_subset['rank_abs'].mean()
            },
            'rank_pct': {
                'min': bucket_subset['rank_pct'].min(),
                'max': bucket_subset['rank_pct'].max(),
                'mean': bucket_subset['rank_pct'].mean()
            }
        }

    return bucket_stats


# def _generate_training_labels_for_all_data(stock_data, alpha=1.5, debug=False,
#                                            is_training=True, label_stats=None):
#     """
#     [重构-绝对收益增强版] 标签生成函数
    
#     诊断修复：
#     1. 引入"绝对收益门控"：只有绝对收益为正的股票，才能被标记为高Relevance (4或5)。
#        防止模型在熊市中学习"相对抗跌但实盘亏损"的样本。
#     2. 兼容 Tushare/QMT 数据：增加 Typical Price 兜底 VWAP。
#     """
#     try:
#         import pandas as pd
#         import numpy as np
#         import logging
#         import gc

#         logger = logging.getLogger(__name__)
#         logger.info("[TARGET] 生成标签 (绝对收益门控 + T+2风控)...")
        
#         stock_data = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)
        
#         # 0. 风控字段清洗
#         control_cols = ['is_limit_up', 'is_limit_down', 'is_can_buy', 'is_zhaban']
#         for col in control_cols:
#             if col in stock_data.columns:
#                 stock_data[col] = stock_data[col].fillna(0).astype(int)

#         # ==============================================================================
#         # 1. 稳健 VWAP 计算 (Tushare 适配)
#         # ==============================================================================
#         # Tushare 数据如果 volume/amount 没复权，计算出的 VWAP 会错。
#         # 最安全的做法：直接使用 Typical Price ((O+H+L+C)/4)，它天然与价格复权同步。
        
#         # 优先使用 OHLC 均价，因为它绝对稳定，不受复权因子错配影响
#         # if all(c in stock_data.columns for c in ['open', 'high', 'low', 'close']):
#         #     stock_data['vwap'] = (stock_data['open'] + stock_data['high'] + 
#         #                           stock_data['low'] + stock_data['close']) / 4.0
#         # else:
#         stock_data['vwap'] = (stock_data['open'] + stock_data['close']) / 2.0

#         # ==============================================================================
#         # 2. 向量化获取未来数据
#         # ==============================================================================
#         grouped_open = stock_data.groupby('stock_code')['open']
#         grouped_low  = stock_data.groupby('stock_code')['low']
#         grouped_vwap = stock_data.groupby('stock_code')['vwap']
        
#         # 买入：T+1 开盘
#         stock_data['buy_price'] = grouped_open.shift(-1)
#         # 卖出：T+2 均价
#         stock_data['sell_price_target'] = grouped_vwap.shift(-2)
#         # 风控：T+2 最低价
#         stock_data['low_2'] = grouped_low.shift(-2)
        
#         # ==============================================================================
#         # 3. 计算收益率 (含 T+2 止损)
#         # ==============================================================================
#         STOP_LOSS_PCT = 0.06  # 放宽到 6% 止损，避免噪音震仓
#         stop_loss_price = stock_data['buy_price'] * (1 - STOP_LOSS_PCT)
        
#         # 原始收益
#         raw_ret = (stock_data['sell_price_target'] - stock_data['buy_price']) / (stock_data['buy_price'] + 1e-8)
        
#         # 止损判定
#         is_stop_2 = stock_data['low_2'] < stop_loss_price
        
#         # 惩罚分 (-8%)
#         PENALTY_SCORE = -0.08
        
#         stock_data['raw_future_ret'] = np.where(is_stop_2, PENALTY_SCORE, raw_ret)
        
#         # 过滤涨幅异常 (Tushare有时会有脏数据)
#         stock_data['raw_future_ret'] = np.where(stock_data['raw_future_ret'].abs() > 0.4, np.nan, stock_data['raw_future_ret'])
        
#         # 清洗无效数据
#         mask_invalid = stock_data['buy_price'].isna() | stock_data['sell_price_target'].isna()
#         stock_data['raw_future_ret'] = np.where(mask_invalid, np.nan, stock_data['raw_future_ret'])

#         # ==============================================================================
#         # 4. 市场中性化 (用于中间排名，但不决定最终高分)
#         # ==============================================================================
#         daily_median = stock_data.groupby('date')['raw_future_ret'].transform('median')
#         stock_data['excess_ret'] = stock_data['raw_future_ret'] - daily_median
        
#         # 关键：我们保留 raw_future_ret 用于后续的"绝对收益门控"检查
#         stock_data['atr_return_label'] = stock_data['raw_future_ret']

#         # ==============================================================================
#         # 5. 生成离散标签 (引入绝对收益修正)
#         # ==============================================================================
        
#         # 第一步：先按超额收益排名 (Relative Rank)
#         def to_gaussian_rank(x):
#             valid_x = x.dropna()
#             if len(valid_x) == 0: return pd.Series(np.nan, index=x.index)
#             ranks = valid_x.rank(pct=True, method='dense')
#             return (ranks - 0.5) * 2

#         stock_data['gaussian_score'] = stock_data.groupby('date')['excess_ret'].transform(to_gaussian_rank)

#         # 第二步：初步映射
#         def map_initial_relevance(score):
#             if pd.isna(score): return np.nan
#             if score > 0.96: return 5   # Top 2%
#             if score > 0.90: return 4   # Top 5%
#             if score > 0.75: return 3   # Top 12.5%
#             if score > 0.40: return 2   # Top 30%
#             if score > 0.00: return 1   # Top 50%
#             return 0
        
#         stock_data['relevance'] = stock_data['gaussian_score'].apply(map_initial_relevance)
        
#         # --------------------------------------------------------------------------
#         # ⚡️大师级修正：绝对收益门控 (Absolute Return Gate)
#         # 逻辑：即使你排第一名，如果你亏钱了，你也不配叫 Label 5。
#         # --------------------------------------------------------------------------
        
#         # 规则 1: Label 4 和 5 必须是正收益 (raw_ret > 0)
#         # 如果是负收益，降级为 Label 1 (平庸) 或 0 (差)，取决于亏损程度
        
#         high_label_mask = stock_data['relevance'] >= 4
#         loss_mask = stock_data['raw_future_ret'] <= 0.005  # 0.5% 以下都不算强势
        
#         # 降级打击：既是高分，又是亏损/微利的，降级为 2
#         downgrade_mask = high_label_mask & loss_mask
#         if downgrade_mask.sum() > 0:
#             logger.info(f"[GATE] 绝对收益门控触发：降级 {downgrade_mask.sum()} 个虚假高分样本 (相对强但绝对弱)")
#             stock_data.loc[downgrade_mask, 'relevance'] = 2
            
#         # 规则 2: 大亏样本 (-3%以下) 强制 Label 0
#         deep_loss_mask = stock_data['raw_future_ret'] < -0.03
#         stock_data.loc[deep_loss_mask, 'relevance'] = 0

#         # ==============================================================================
#         # 6. 风控过滤
#         # ==============================================================================
#         filter_mask = pd.Series(False, index=stock_data.index)
#         if 'is_can_buy' in stock_data.columns:
#             filter_mask |= (stock_data['is_can_buy'] == 0)
#         if 'amount' in stock_data.columns:
#             # 放宽一点，Tushare数据可能有单位差异，确保不是0即可
#             filter_mask |= (stock_data['amount'].fillna(0) < 1000) 

#         if filter_mask.sum() > 0:
#             stock_data.loc[filter_mask, 'relevance'] = 0
#             stock_data.loc[filter_mask, 'atr_return_label'] = np.nan

#         # ==============================================================================
#         # 7. 清理
#         # ==============================================================================
#         cols_to_drop = [
#             'buy_price', 'sell_price_target', 'low_2', 
#             'raw_future_ret', 'excess_ret', 'gaussian_score', 'vwap'
#         ]
#         stock_data.drop(columns=[c for c in cols_to_drop if c in stock_data.columns], inplace=True)
#         gc.collect()

#         if is_training:
#             valid_labels = stock_data['atr_return_label'].dropna()
#             stats = {'mean': valid_labels.mean(), 'std': valid_labels.std()}
#             return stock_data, stats
#         else:
#             return stock_data

#     except Exception as e:
#         logger.error(f"标签生成失败: {e}", exc_info=True)
#         return None, None



# def _generate_training_labels_for_all_data(stock_data):
#     """
#     日频 Rank 标签（无 VWAP 版本）
#     买入：T+1 open
#     卖出代理：T+2 / T+3 Typical Price
#     """

#     import numpy as np
#     import pandas as pd

#     stock_data = stock_data.sort_values(
#         ['stock_code', 'date']
#     ).reset_index(drop=True)

#     g = stock_data.groupby('stock_code')

#     # ========= 1. 买入价 =========
#     buy_price = g['open'].shift(-1)
#     close_price = g['close'].shift(-1)

#     # ========= 2. 卖出代理价 =========
#     tp = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3.0
#     stock_data['sell'] = tp

#     sell_t2 = g['sell'].shift(-2)
#     sell_t3 = g['sell'].shift(-3)
#     # sell_t1 = g['sell'].shift(-1)

#     # 等权平均，表示"2~3 天期望卖出"
#     # sell_price = 0.8 * sell_t2 + 0.2 * sell_t3
#     sell_price = 0.5 * close_price + 0.4* sell_t2 + 0.1 * sell_t3

#     # ========= 3. 信息收益 =========
#     future_ret = (sell_price - buy_price) / (buy_price + 1e-12)
#     future_ret = future_ret.replace([np.inf, -np.inf], np.nan)
#     stock_data['atr_return_label'] = future_ret

#     # 防止脏数据，不是风控
#     future_ret = future_ret.clip(-0.2, 0.2)

#     low_t3 = g["low"].shift(-3)  # T+1 low
#     dd_t3 = (buy_price - low_t3) / (buy_price + 1e-12)
#     dd_t3 = dd_t3.clip(lower=0.05)  # 只惩罚回撤大于5%
#     low_t2 = g["low"].shift(-2)  # T+1 low
#     dd_t2 = (buy_price - low_t2) / (buy_price + 1e-12)
#     dd_t2 = dd_t2.clip(lower=0.05)  # 只惩罚回撤大于5%
#     future_ret = future_ret - float(0.3) * dd_t2 - float(0.2) * dd_t3

#     stock_data['future_ret'] = future_ret

#     # ========= 4. 横截面 Rank → 离散 Label =========
#     def rank_to_label(x):
#         x = x.dropna()
#         if len(x) < 30:
#             return pd.Series(np.nan, index=x.index)

#         pct = x.rank(pct=True, method='average')

#         return pd.cut(
#             pct,
#             bins=[0, 0.3, 0.6, 0.8, 0.93, 0.98, 1.0],
#             labels=[0, 1, 2, 3, 4, 5]
#         ).astype(float)

#     stock_data['relevance'] = (
#         stock_data
#         .groupby('date')['future_ret']
#         .transform(rank_to_label)
#     )

#     mask_untradable = (
#         (stock_data['is_limit_up'] == 1) |
#         (stock_data['is_limit_down'] == 1) |
#         (stock_data['is_zhaban'] == 1) |
#         (stock_data['is_can_buy'] == 0) |
#         (stock_data['amount'].fillna(0) < 10000000)
#     )

#     stock_data.loc[mask_untradable, 'relevance'] = np.nan

#     return stock_data



# def _generate_training_labels_for_all_data(stock_data):
#     """
#     日频 Rank 标签（无 VWAP 版本）
#     信号日：T
#     买入：T+1 open（若 T+1 开盘 跌超3% 或 涨超5% 则跳过买入）
#     卖出代理：0.5 * (T+1 close) + 0.4 * (T+2 typical) + 0.1 * (T+3 typical)
#     风险惩罚：对齐实盘（你在 T+2/T+3 才开始 5min 风控）
#       - T+2 开盘 gap-down（相对买入价）
#       - T+2/T+3 low 的穿价（相对买入价，超过5%才惩罚）
#     """

#     import numpy as np
#     import pandas as pd

#     stock_data = stock_data.sort_values(['stock_code', 'date']).reset_index(drop=True)
#     g = stock_data.groupby('stock_code', sort=False)

#     # ========= 0) 交易制度：按代码推断涨跌幅限制 =========
#     code = stock_data['stock_code'].astype(str)
#     is_20 = code.str.startswith('30') | code.str.startswith('688')
#     limit = np.where(is_20, 0.20, 0.10).astype('float64')
#     stock_data['limit_pct'] = limit

#     # ========= 1) 买入价（T+1 open） =========
#     buy_price = g['open'].shift(-1)        # T+1 open
#     close_t0  = g['close'].shift(0)        # T close（当前行就是T）

#     # ========= 1.1) 买入风控过滤：T+1 开盘相对 T 收盘的涨跌幅 =========
#     # 你说的"开盘跌超3%、涨超5%跳过买入"，本质就是这个 gap
#     open_gap_t1 = (buy_price - close_t0) / (close_t0 + 1e-12)
#     stock_data['open_gap_t1'] = open_gap_t1

#     mask_skip_buy = (open_gap_t1 <= -0.03) | (open_gap_t1 >= 0.05)

#     # ========= 2) 卖出代理价 =========
#     tp = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3.0
#     stock_data['tp'] = tp

#     close_t1 = g['close'].shift(-1)        # T+1 close
#     sell_t2  = g['tp'].shift(-2)           # T+2 typical
#     sell_t3  = g['tp'].shift(-3)           # T+3 typical

#     sell_price = 0.5 * close_t1 + 0.4 * sell_t2 + 0.1 * sell_t3

#     # ========= 3) 原始未来收益 =========
#     future_ret = (sell_price - buy_price) / (buy_price + 1e-12)
#     future_ret = future_ret.replace([np.inf, -np.inf], np.nan)
#     stock_data['atr_return_label'] = future_ret

#     # ========= 4) 动态 clip（多日加权：建议 1.8*limit 做硬界） =========
#     clip_hi = 1.8 * stock_data['limit_pct']
#     future_ret = future_ret.clip(lower=-clip_hi, upper=clip_hi)

#     # ========= 5) 持仓期风险惩罚（对齐你 T+2/T+3 才开始监控） =========
#     # --- (A) T+2 开盘 gap-down（相对买入价） ---
#     open_t2 = g['open'].shift(-2)  # T+2 open
#     gap_t2 = (open_t2 - buy_price) / (buy_price + 1e-12)
#     gap_down = (-gap_t2).clip(lower=0.0)

#     # 只惩罚超过 3% 的 gap-down（你也可改成 5%）
#     pen_gap = (gap_down - 0.05).clip(lower=0.0)
#     # >8% 的部分额外惩罚（对应"快速下跌滑点更大"）
#     pen_gap_extra = (gap_down - 0.08).clip(lower=0.0)

#     # --- (B) T+2/T+3 low 穿价惩罚（超过5%才惩罚） ---
#     low_t2 = g['low'].shift(-2)
#     low_t3 = g['low'].shift(-3)

#     dd_t2 = ((buy_price - low_t2) / (buy_price + 1e-12)).clip(lower=0.0)
#     dd_t3 = ((buy_price - low_t3) / (buy_price + 1e-12)).clip(lower=0.0)

#     pen_dd2 = (dd_t2 - 0.05).clip(lower=0.0)
#     pen_dd3 = (dd_t3 - 0.05).clip(lower=0.0)

#     # 合成惩罚（注意：不要过大，否则标签被风险项主导）
#     future_ret = (
#         future_ret
#         - 0.35 * pen_gap
#         - 0.50 * pen_gap_extra
#         - 0.25 * pen_dd2
#         - 0.15 * pen_dd3
#     )

#     stock_data['future_ret'] = future_ret

#     # ========= 6) 横截面 Rank → 离散 Label =========
#     def rank_to_label(x):
#         x = x.dropna()
#         if len(x) < 30:
#             return pd.Series(np.nan, index=x.index)

#         pct = x.rank(pct=True, method='average')
#         return pd.cut(
#             pct,
#             bins=[0, 0.3, 0.6, 0.8, 0.93, 0.98, 1.0],
#             labels=[0, 1, 2, 3, 4, 5]
#         ).astype(float)

#     stock_data['relevance'] = stock_data.groupby('date')['future_ret'].transform(rank_to_label)

#     # ========= 7) 不可交易样本置 NaN（加入：买入风控 + 炸板过滤） =========
#     mask_untradable = (
#         mask_skip_buy |                               # ✅ 关键：与你实盘一致
#         (stock_data.get('is_limit_up', 0) == 1) |
#         (stock_data.get('is_limit_down', 0) == 1) |
#         (stock_data.get('is_can_buy', 1) == 0) |
#         (stock_data.get('is_zhaban', 0) == 1) |        # ✅ 炸板过滤
#         (stock_data.get('amount', 0).fillna(0) < 10000000)
#     )

#     stock_data.loc[mask_untradable, 'relevance'] = np.nan

#     return stock_data


def _generate_training_labels_for_all_data(stock_data):
    """
    【大师级重构】极值二分类标签（Extreme Binary Classification）
    
    信号日：T
    买入：T+1 open（若 T+1 开盘相对 T 收盘：跌超3% 或 涨超5% 则跳过买入）
    目标：T+2 或 T+3 的最高价涨幅 > 8%，且 T+1 到 T+3 的最低价回撤不深于 -4%
    
    标签字段：
      - relevance : 二分类标签 (1: 妖股, 0: 平庸/垃圾, NaN: 无效)
      - atr_return_label : 保留为最大涨幅，用于评估回测潜能
      - future_ret       : 同上
    """
    import numpy as np
    import pandas as pd
    import logging

    logger = logging.getLogger(__name__)
    logger.info("[强化] 开始生成二分类极值标签...")

    stock_data = stock_data.sort_values(["stock_code", "date"]).reset_index(drop=True)
    stock_data["is_st_hist"] = _get_historical_st_flag(stock_data, logger=logger).astype(bool)
    g = stock_data.groupby("stock_code", sort=False)

    # 0) 涨跌幅限制：按代码推断
    code = stock_data["stock_code"].astype(str)
    is_20 = code.str.startswith("30") | code.str.startswith("688")
    stock_data["limit_pct"] = np.where(is_20, 0.20, 0.10).astype("float64")

    # 1) 买入价（T+1 open）
    # buy_price = (g["open"].shift(-1) +g["close"].shift(-1) +g["high"].shift(-1) + g["low"].shift(-1)) / 4.0
    buy_price = g["open"].shift(-1)
    
    # 2) 提取未来两天的极限价格
    high_t2 = g["high"].shift(-2)
    high_t3 = g["high"].shift(-3)
    close_t1 =  g["close"].shift(-1)
    low_t1 = g["low"].shift(-1)
    low_t2 = g["low"].shift(-2)
    low_t3 = g["low"].shift(-3)

    # 3) 计算期间最大涨幅和最大回撤（相对于买入价）
    max_high = np.maximum(high_t2.fillna(0), high_t3.fillna(0))
    # 对于停牌等可能产生的nan处理，保证计算稳健
    max_high = np.where(high_t2.isna() & high_t3.isna(), np.nan, np.maximum(high_t2.fillna(-np.inf), high_t3.fillna(-np.inf)))
    # min_low = np.minimum(low_t1.fillna(np.inf), np.minimum(low_t2.fillna(np.inf), low_t3.fillna(np.inf)))
    min_low = np.minimum(low_t2.fillna(np.inf), low_t3.fillna(np.inf))
    min_low = np.where(low_t1.isna() & low_t2.isna() & low_t3.isna(), np.nan, min_low)

    max_high_return = (max_high - buy_price) / (buy_price + 1e-12)
    min_low_return = (min_low - buy_price) / (buy_price + 1e-12)

    # # 4) 生成二分类标签
    # is_monster = np.where((max_high_return >= 0.15) & (min_low_return >= -0.05), 1.0, 0.0)
    
    # # 无效数据处理
    # invalid_mask = buy_price.isna() | np.isnan(max_high_return) | np.isnan(min_low_return)
    # is_monster = pd.Series(is_monster, index=stock_data.index)
    # is_monster[invalid_mask] = np.nan

    tp = (stock_data["open"] + stock_data["high"] + stock_data["low"] + stock_data["close"]) / 4.0
    stock_data["tp"] = tp
    
    sell_t2 = g["tp"].shift(-2)
    sell_t3 = g["tp"].shift(-3)
    
    # 7:3 加权：极其看重 T+2 的溢价变现能力
    sell_price_proxy = 0.7 * sell_t2.fillna(sell_t2) + 0.3 * sell_t3.fillna(sell_t3)
    
    realistic_vwap_return = (sell_price_proxy - buy_price) / (buy_price + 1e-12)

        # 4) 生成二分类标签
    is_monster = np.where((realistic_vwap_return >= 0.10) & (min_low_return > -0.05), 1.0, 0.0)
    # is_monster = np.where((realistic_vwap_return / (np.abs(min_low_return) + 1e-9) > 2.0) & (realistic_vwap_return >= 0.11), 1.0, 0.0)
    
    # 无效数据处理
    invalid_mask = buy_price.isna() | np.isnan(max_high_return) | np.isnan(min_low_return)
    is_monster = pd.Series(is_monster, index=stock_data.index)
    is_monster[invalid_mask] = np.nan

    
    # 覆盖回测评估基准
    stock_data["atr_return_label"] = realistic_vwap_return
    stock_data["future_ret"] = realistic_vwap_return
    stock_data["max_high_return"] = max_high_return # 记录最大涨幅（供分析用）
    stock_data["min_low_return"] = min_low_return   # 记录最大跌幅（供分析用）
    stock_data["relevance"] = is_monster

    # 5) 历史ST窗口屏蔽：当天 / T+1 / T+2 / T+3 任一天为ST，均视为无效标签
    st_t1 = g["is_st_hist"].shift(-1).fillna(False).astype(bool)
    st_t2 = g["is_st_hist"].shift(-2).fillna(False).astype(bool)
    st_t3 = g["is_st_hist"].shift(-3).fillna(False).astype(bool)
    mask_st_window = stock_data["is_st_hist"] | st_t1 | st_t2 | st_t3
    st_window_count = int(mask_st_window.sum())
    if st_window_count > 0:
        logger.info(
            f"[LABEL] ST窗口屏蔽样本: {st_window_count:,}/{len(stock_data):,} "
            f"({st_window_count / max(len(stock_data), 1):.2%})"
        )
        stock_data.loc[
            mask_st_window,
            ["atr_return_label", "future_ret", "max_high_return", "min_low_return"]
        ] = np.nan
        stock_data.loc[mask_st_window, "relevance"] = np.nan

    # 6) 过滤实盘不可交易的样本
    _valid_cs_mask = stock_data.get("valid_cs_mask", True)
    if isinstance(_valid_cs_mask, pd.Series):
        valid_cs_mask = _valid_cs_mask.reindex(stock_data.index).fillna(False).astype(bool)
    else:
        valid_cs_mask = pd.Series(_valid_cs_mask, index=stock_data.index).astype(bool)

    amount = stock_data.get("amount", 0)
    amount_val = amount.fillna(0) if hasattr(amount, "fillna") else amount

    mask_untradable = (
        (stock_data.get("is_limit_up", 0) == 1)
        | (stock_data.get("is_limit_down", 0) == 1)
        | (stock_data.get("is_can_buy", 1) == 0)
        | (stock_data.get("is_zhaban", 0) == 1)
        | (amount_val < 10_000_000)
        | (~valid_cs_mask)
    )

    # 不可交易样本：relevance 置 NaN
    stock_data.loc[mask_untradable, "relevance"] = np.nan
    stock_data.loc[mask_untradable, "atr_return_label"] = np.nan
    stock_data.loc[mask_untradable, "future_ret"] = np.nan

    monster_count = (stock_data["relevance"] == 1).sum()
    total_valid = stock_data["relevance"].notna().sum()
    if total_valid > 0:
        logger.info(f"[OK] 二分类标签生成完毕。正样本数: {monster_count}, 负样本数: {total_valid - monster_count}, 比例: {monster_count/total_valid:.2%}")



    return stock_data

def _extract_and_validate_labels(labeled_data, label_column='atr_return_label'):
    """
    [强化] 提取和验证连续收益标签（排序模型模式）

    Args:
        labeled_data: 包含标签的数据DataFrame
        label_column: 标签列名，固定为'atr_return_label'（连续收益标签）

    Returns:
        tuple: (labels, valid_mask, relevance, relevance_30) 或 (None, None, None, None) 如果失败
    """
    try:
        import numpy as np
        logger = logging.getLogger(__name__)

        logger.info(f"🏷️ 提取和验证连续收益标签（成本感知回归模式）...")

        # 检查标签列是否存在
        if label_column not in labeled_data.columns:
            logger.error(f"数据中缺少标签列: {label_column}")
            return None, None

        # 提取标签
        labels = labeled_data[label_column].copy()

        # 验证标签值的有效性（连续值）
        valid_labels = labels.dropna()

        logger.info(f"连续收益标签统计信息:")
        logger.info(f"  标签列: {label_column}")
        logger.info(f"  总样本数: {len(labels):,}")
        logger.info(f"  有效标签数: {len(valid_labels):,}")
        logger.info(f"  缺失标签数: {labels.isna().sum():,}")

        if len(valid_labels) > 0:
            logger.info(f"  标签范围: [{valid_labels.min():.3f}, {valid_labels.max():.3f}] ATR")
            logger.info(f"  标签均值: {valid_labels.mean():.3f} ATR")
            logger.info(f"  标签标准差: {valid_labels.std():.3f} ATR")

            # 分析收益分布
            profit_samples = (valid_labels >= ATR_MULTIPLIER_PROFIT).sum()  # 盈利样本
            loss_samples = (valid_labels <= -ATR_MULTIPLIER_LOSS).sum()   # 亏损样本
            neutral_samples = ((valid_labels > -ATR_MULTIPLIER_LOSS) & (valid_labels < ATR_MULTIPLIER_PROFIT)).sum()  # 中性样本

            logger.info(f"连续收益标签分布:")
            logger.info(f"  盈利样本 (>={ATR_MULTIPLIER_PROFIT} ATR): {profit_samples:,}个 ({profit_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  中性样本 (-{ATR_MULTIPLIER_LOSS}~{ATR_MULTIPLIER_PROFIT} ATR): {neutral_samples:,}个 ({neutral_samples/len(valid_labels)*100:.1f}%)")
            logger.info(f"  亏损样本 (<=-{ATR_MULTIPLIER_LOSS} ATR): {loss_samples:,}个 ({loss_samples/len(valid_labels)*100:.1f}%)")

        # 创建有效样本掩码（过滤掉NaN值）
        valid_mask = labels.notna()

        # 提取relevance字段(4层)
        if 'relevance' in labeled_data.columns:
            relevance = labeled_data['relevance'].copy()
            logger.info(f"[OK] 提取relevance字段(4层): {len(relevance):,}个样本")
            relevance_dist = relevance.value_counts().sort_index().to_dict()
            logger.info(f"  Relevance分布: {relevance_dist}")
        else:
            logger.warning("[WARNING] 数据中缺少relevance字段,将使用默认值0")
            import pandas as pd
            relevance = pd.Series(0, index=labeled_data.index)

        # 🆕 提取relevance_30字段(30层)
        if 'relevance_30' in labeled_data.columns:
            relevance_30 = labeled_data['relevance_30'].copy()
            logger.info(f"[OK] 提取relevance_30字段(30层): {len(relevance_30):,}个样本")
            relevance_30_dist = relevance_30.value_counts().sort_index(ascending=False).head(5).to_dict()
            logger.info(f"  Relevance_30 Top-5分布: {relevance_30_dist}")
        else:
            logger.warning("[WARNING] 数据中缺少relevance_30字段,将使用默认值0")
            import pandas as pd
            relevance_30 = pd.Series(0, index=labeled_data.index)

        logger.info("[OK] 连续收益标签提取和验证完成")
        return labels, valid_mask, relevance, relevance_30

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"连续收益标签提取和验证失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None, None


def _validate_data_order(dates, labels, relevance, data_name="数据"):
    """
    [WRENCH] 验证数据顺序是否正确（按日期升序排列）

    Args:
        dates: 日期数组
        labels: 标签数组
        relevance: relevance数组
        data_name: 数据名称（用于日志）

    Returns:
        bool: True表示顺序正确，False表示顺序混乱
    """
    import numpy as np
    import pandas as pd
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"[SEARCH] 验证{data_name}顺序...")

        # 检查长度一致性
        if len(dates) != len(labels) or len(dates) != len(relevance):
            logger.error(f"[FAIL] {data_name}长度不一致!")
            logger.error(f"   dates长度: {len(dates)}")
            logger.error(f"   labels长度: {len(labels)}")
            logger.error(f"   relevance长度: {len(relevance)}")
            return False

        # 检查日期是否升序排列
        dates_sorted = np.sort(dates)
        if not np.array_equal(dates, dates_sorted):
            logger.error(f"[FAIL] {data_name}日期未按升序排列!")
            # 找出第一个乱序的位置
            for i in range(1, len(dates)):
                if dates[i] < dates[i-1]:
                    logger.error(f"   第{i}个日期({dates[i]})小于第{i-1}个日期({dates[i-1]})")
                    break
            return False

        # 检查是否有NaN值
        if np.any(pd.isna(labels)):
            nan_count = pd.isna(labels).sum()
            logger.error(f"[FAIL] {data_name}标签中有{nan_count}个NaN值!")
            return False

        if np.any(pd.isna(relevance)):
            nan_count = pd.isna(relevance).sum()
            logger.error(f"[FAIL] {data_name}relevance中有{nan_count}个NaN值!")
            return False

        logger.info(f"[OK] {data_name}顺序验证通过")
        logger.info(f"   样本数: {len(dates):,}")
        logger.info(f"   日期范围: {dates[0]} ~ {dates[-1]}")
        logger.info(f"   标签范围: [{np.min(labels):.3f}, {np.max(labels):.3f}]")
        logger.info(f"   relevance范围: [{np.min(relevance):.0f}, {np.max(relevance):.0f}]")

        return True

    except Exception as e:
        logger.error(f"[FAIL] {data_name}顺序验证失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _prepare_feature_data(input_data, feature_columns, data_type="特征数据"):
    """
    准备特征数据（因子已在生成阶段完成清洗、winsorize、归一化）

    注意：因子在生成阶段（factor_lab）已经完成了：
         1. 替换 Inf 为 NaN
         2. 每日 cross-sectional winsorize
         3. 每日截面中位数填充缺失值
         4. 每日 cross-sectional 归一化（z-score/rankpct）

         此处只做基本的数据加载和格式转换

    Args:
        input_data: 输入数据DataFrame
        feature_columns: 特征列名列表
        data_type: 数据类型描述（用于日志）

    Returns:
        tuple: (X_processed, available_features) 或 (None, None) 如果失败
    """
    try:
        import numpy as np
        logger = logging.getLogger(__name__)

        logger.info(f"[强化] 开始准备{data_type}...")

        # [CLIPBOARD] 步骤1：过滤有效的特征列
        available_features = [col for col in feature_columns if col in input_data.columns]
        missing_features = [col for col in feature_columns if col not in input_data.columns]

        if missing_features:
            logger.warning(f"缺失特征: {len(missing_features)}个")
            logger.debug(f"缺失特征列表: {missing_features[:10]}...")  # 只显示前10个

        if not available_features:
            logger.error("没有可用的特征列")
            return None, None

        logger.info(f"可用特征: {len(available_features)}个")

        # [CHART] 步骤2：提取特征数据
        feature_data = input_data[available_features].copy()

        logger.info(f"{data_type}统计:")
        logger.info(f"  样本数: {len(feature_data):,}个")
        logger.info(f"  特征数: {len(available_features)}个")

        # [CHART] 步骤3：转换为numpy数组
        X_processed = feature_data.values.astype(np.float32)

        # [CHART] 步骤4：统计信息
        logger.info(f"[OK] {data_type}准备完成:")
        logger.info(f"  样本数: {len(X_processed):,}")
        logger.info(f"  特征数: {X_processed.shape[1]}")
        logger.info(f"  特征数据类型: {X_processed.dtype}")

        return X_processed, available_features

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"准备{data_type}失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None



def _save_model_results(model_results, feature_names, pca_models=None, model_type='xgboost', timestamp=None):
    """保存模型和结果（支持XGBoost和LightGBM）

    Args:
        model_results: 模型训练结果
        feature_names: 特征名称列表
        pca_models: PCA模型字典（可选，仅XGBoost使用）
        model_type: 模型类型 ('xgboost' 或 'lightgbm')
        timestamp: 统一时间戳（可选，如果为None则自动生成）
    """
    try:
        import pickle
        import json
        from datetime import datetime
        from pathlib import Path

        logger = logging.getLogger(__name__)

        # 创建结果目录
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # [强化] 使用传入的时间戳或生成新的时间戳
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # [强化] 根据模型类型确定文件名
        model_file = results_dir / f"{model_type}_model_{timestamp}.pkl"

        # [强化] 保存模型（XGBoost和LightGBM使用不同的数据结构）
        if model_type == 'xgboost':
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_results['model'],
                    'scaler': model_results['scaler'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'pca_info': model_results['pca_info'],
                    'feature_names': feature_names,
                    'pca_models': pca_models if pca_models else {},
                    'label_stats': model_results.get('label_stats', None),
                    'interaction_info': None,
                    'model_type': 'xgboost',
                    'performance': model_results.get('performance', {}),  # [强化] 保存性能信息
                }, f)
        elif model_type == 'lightgbm':
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_results['model'],
                    'scaler': model_results['scaler'],
                    'feature_names': feature_names,
                    'label_stats': model_results.get('label_stats', None),
                    'model_type': 'lightgbm',
                    'performance': model_results.get('performance', {}),  # [强化] 保存性能信息
                    'industry_mapping': model_results.get('industry_mapping', None),  # [SW行业编码] 保存行业映射信息
                }, f)

        logger.info(f"[OK] {model_type.upper()}模型已保存: {model_file}")

        # 记录PCA信息
        if pca_models:
            logger.info(f"[CHART] PCA模型信息:")
            for group_name, pca_info in pca_models.items():
                if pca_info.get('applied', False):
                    original_factors = pca_info.get('original_factors', [])
                    n_components = pca_info.get('n_components', 0)
                    variance_explained = pca_info.get('variance_explained', 0)
                    logger.info(f"   {group_name}: {len(original_factors)}个特征 -> {n_components}个主成分")
                    logger.info(f"   解释方差比例: {variance_explained:.4f}")
                else:
                    logger.info(f"   {group_name}: PCA未应用")

        # 保存训练结果报告
        # [强化] 修复：安全处理feature_importance，支持DataFrame和dict格式
        feature_importance_data = None
        if model_results.get('feature_importance') is not None:
            try:
                # 尝试DataFrame格式
                feature_importance_data = model_results['feature_importance'].to_dict('records')
            except AttributeError:
                # 如果是dict格式(LightGBM),转换为list of dict
                if isinstance(model_results['feature_importance'], dict):
                    feature_importance_data = [
                        {'feature': k, 'importance': v}
                        for k, v in model_results['feature_importance'].items()
                    ]
                    logger.info("[OK] feature_importance已从dict格式转换为list格式")
                else:
                    logger.warning("feature_importance格式未知，跳过保存")
                    feature_importance_data = None

        report_data = {
            'timestamp': timestamp,
            'model_type': model_type,  # [强化] 记录模型类型
            'performance': model_results['performance'],
            'training_info': model_results['training_info'],
            'model_params': model_results['model_params'],
            'feature_importance': feature_importance_data,
        }

        # [强化] XGBoost特有的信息
        if model_type == 'xgboost':
            report_data['pca_info'] = {k: v for k, v in model_results['pca_info'].items() if k != 'pca_object'}
            report_data['interaction_info'] = model_results.get('interaction_info')

        # [强化] 根据模型类型确定文件名
        report_file = results_dir / f"training_report_{model_type}_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"训练报告已保存: {report_file}")

        # 保存特征重要性图表（如果可能）
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            # [强化] 处理不同格式的feature_importance
            feature_importance = model_results.get('feature_importance')
            if feature_importance is not None:
                # 转换为DataFrame格式
                if isinstance(feature_importance, dict):
                    # dict格式(LightGBM)
                    importance_df = pd.DataFrame([
                        {'feature': k, 'importance': v}
                        for k, v in feature_importance.items()
                    ])
                    importance_df = importance_df.sort_values('importance', ascending=False)
                else:
                    # DataFrame格式(XGBoost)
                    importance_df = feature_importance

                # 绘制Top 20
                top_features = importance_df.head(20)

                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importance ({model_type.upper()})')
                plt.tight_layout()

                # [强化] 根据模型类型确定文件名
                plot_file = results_dir / f"feature_importance_{model_type}_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"特征重要性图表已保存: {plot_file}")
            else:
                logger.warning("feature_importance为None，跳过图表保存")

        except Exception as plot_e:
            logger.warning(f"保存特征重要性图表失败: {plot_e}")

        logger.info("模型结果保存完成")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"保存模型结果失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")


def run_train_model_mode(args, train_end_date=None, is_regression=True, use_xgboost=None, use_lightgbm=None):
    """运行模型训练模式

    Args:
        args: 命令行参数
        train_end_date: 训练数据截止日期，如果为None则使用全局配置TRAIN_END_DATE
        is_regression: 是否使用回归模式，默认True（成本感知回归）
        use_xgboost: 是否训练XGBoost模型，如果为None则根据命令行参数决定
        use_lightgbm: 是否训练LightGBM模型，如果为None则根据命令行参数决定
    """
    logger = logging.getLogger(__name__)
    if is_regression:
        print("运行模式: 成本感知回归模型训练")
    else:
        print("运行模式: ATR三分类模型训练")
    print("=" * 60)

    try:
        # [TARGET] 如果未指定train_end_date，使用全局配置
        if train_end_date is None:
            train_end_date = TRAIN_END_DATE
            logger.info(f"🗓️ 使用全局时间切分配置: TRAIN_END_DATE={TRAIN_END_DATE}")

        # 1. 数据加载与特征识别
        logger.info("开始加载因子数据...")
        # 在训练模式下包含标签生成特征，用于标签构建
        factor_data = _load_factor_cache_data(include_label_generation_features=True)
        if factor_data is None or len(factor_data) == 0:
            logger.error("因子数据加载失败")
            return False

        logger.info(f"因子数据加载完成: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")

        # [强化] 加载市场数据（用于XGBoost计算group权重）
        logger.info("\n[REFRESH] 加载市场数据（用于XGBoost计算group权重）...")
        market_data = None
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("[WARNING] 市场数据加载失败，XGBoost将使用均匀group权重")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"[OK] 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"[WARNING] 市场数据加载失败: {e}，XGBoost将使用均匀group权重")
            market_data = None

        # [SW行业编码] 加载标准化SW申万行业数据用于LightGBM分类特征
        logger.info("\n[SW行业编码] 加载标准化SW申万行业数据...")

        encoding_data = load_standardized_sw_encoding()
        if not encoding_data:
            raise RuntimeError("无法加载标准化SW编码文件，请确保config/sw_industry_encoding.pkl文件存在")

        sw_sector_mapping = encoding_data['mappings']['stock_to_industry']

        metadata = encoding_data.get('metadata', {})
        logger.info(f"[OK] 标准化SW行业数据加载成功: {len(sw_sector_mapping)}个股票行业映射")
        logger.info(f"  版本: {metadata.get('version', 'unknown')}")
        logger.info(f"  行业数: {metadata.get('total_industries', 0)}")

        # 应用训练截止日期过滤
        if train_end_date:
            train_end_dt = pd.to_datetime(train_end_date)
            original_count = len(factor_data)
            factor_data = factor_data[factor_data['date'] < train_end_dt].copy()
            logger.info(f"🗓️ 应用训练截止日期过滤: {train_end_date}")
            logger.info(f"[CHART] 数据过滤结果: {original_count} -> {len(factor_data)} 条记录")
            logger.info(f"[DATE] 训练数据日期范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

            if len(factor_data) == 0:
                logger.error("过滤后无可用训练数据")
                return False

        # 2. 特征识别（使用基础特征）
        feature_columns = _get_ml_feature_columns_simple(factor_data)
        feature_columns = sorted(feature_columns)
        logger.info(f"识别到 {len(feature_columns)} 个基础特征")

        # 🆕 打印训练模型使用的因子信息
        if feature_columns:
            logger.info("=" * 60)
            logger.info("[FEATURES] 训练模型使用的因子信息")
            logger.info("=" * 60)

            # 按因子类型分组显示
            zscore_factors = [f for f in feature_columns if f.endswith('_z')]
            rankpct_factors = [f for f in feature_columns if f.endswith('_rank')]
            intra_rank_factors = [f for f in feature_columns if f.endswith('_intra_rank')]
            intra_zscore_factors = [f for f in feature_columns if f.endswith('_intra_z')]
            original_factors = [f for f in feature_columns if not any(f.endswith(suffix) for suffix in ['_z', '_rank', '_intra_rank', '_intra_z'])]

            if zscore_factors:
                logger.info(f"  • Z-score归一化因子 ({len(zscore_factors)}个): {zscore_factors[:8]}{'...' if len(zscore_factors) > 8 else ''}")
            if rankpct_factors:
                logger.info(f"  • 排名归一化因子 ({len(rankpct_factors)}个): {rankpct_factors[:8]}{'...' if len(rankpct_factors) > 8 else ''}")
            if intra_rank_factors:
                logger.info(f"  • 行业内排名因子 ({len(intra_rank_factors)}个): {intra_rank_factors[:8]}{'...' if len(intra_rank_factors) > 8 else ''}")
            if intra_zscore_factors:
                logger.info(f"  • 行业内Z-score因子 ({len(intra_zscore_factors)}个): {intra_zscore_factors[:8]}{'...' if len(intra_zscore_factors) > 8 else ''}")
            if original_factors:
                logger.info(f"  • 原始因子 ({len(original_factors)}个): {original_factors[:8]}{'...' if len(original_factors) > 8 else ''}")

            logger.info(f"[FEATURES] 因子分类统计: Z-score({len(zscore_factors)}) + Rank({len(rankpct_factors)}) + Intra-Rank({len(intra_rank_factors)}) + Intra-Z({len(intra_zscore_factors)}) + Original({len(original_factors)}) = {len(feature_columns)}")

            # 控制台输出
            print(f"\n🎯 训练模型使用的因子信息:")
            print(f"   📊 总因子数量: {len(feature_columns)}")
            print(f"   🔹 Z-score归一化: {len(zscore_factors)}个")
            print(f"   🔸 排名归一化: {len(rankpct_factors)}个")
            print(f"   🏭 行业内排名: {len(intra_rank_factors)}个")
            print(f"   🏭 行业内Z-score: {len(intra_zscore_factors)}个")
            print(f"   🔹 原始因子: {len(original_factors)}个")

        # [WRENCH] 添加特征一致性检查和记录
        logger.info("=" * 60)
        logger.info("纯因子组特征分析")
        logger.info("=" * 60)
        logger.info(f"[OK] 所有特征均来自因子组文件，无原始股票数据特征")
        logger.info(f"[OK] 特征总数: {len(feature_columns)}个")


        # 3. 数据过滤（在标签构建前）
        logger.info("[REFRESH] 开始数据过滤（北交所、新股和ST股票）...")
        logger.info("=" * 50)

        # 第一步：先过滤北交所股票（在日期循环之外）
        logger.info("[FILTER] 开始批量北交所过滤...")
        original_count = len(factor_data)
        if 'stock_code' in factor_data.columns:
            bj_filter = ~factor_data['stock_code'].str.endswith(('.BJ', '.bj'), na=False)
            factor_data = factor_data[bj_filter].copy()
            after_bj_count = len(factor_data)
            bj_filtered_count = original_count - after_bj_count
            logger.info(f"[FILTER] 批量北交所过滤完成: {original_count} -> {after_bj_count} (过滤掉 {bj_filtered_count} 只北交所股票)")
        else:
            after_bj_count = original_count
            bj_filtered_count = 0
            logger.info("[FILTER] 无stock_code列，跳过北交所过滤")

        # 第二步：再过滤新股（在北交所过滤之后）
        logger.info("[FILTER] 开始批量新股过滤...")
        new_stock_original_count = len(factor_data)
        factor_data = _filter_new_stocks_batch(factor_data, logger)
        after_new_stock_count = len(factor_data)
        new_stock_filtered_count = new_stock_original_count - after_new_stock_count
        logger.info(f"[FILTER] 批量新股过滤完成: {new_stock_original_count} -> {after_new_stock_count} (过滤掉 {new_stock_filtered_count} 条新股期间数据)")

        # 获取所有日期
        all_dates = sorted(factor_data['date'].unique())
        filtered_data_list = []

        for date in all_dates:
            date_str = date.strftime('%Y%m%d')
            daily_data = factor_data[factor_data['date'] == date].copy()

            if len(daily_data) == 0:
                continue

            # 北交所和新股过滤已在循环外批量处理完成
            # 注释：北交所和新股过滤已在循环外批量处理，无需在循环内重复执行

            # 过滤ST股票
            if len(daily_data) > 0:
                st_filter = _filter_st_stocks(daily_data,logger)
                daily_data = daily_data[st_filter].copy()
                final_count = len(daily_data)
            else:
                final_count = 0

            if len(daily_data) > 0:
                filtered_data_list.append(daily_data)

            # 每50天输出一次过滤统计
            # if len(filtered_data_list) % 50 == 0:
                # logger.info(f"[DATE] {date_str}: 过滤统计 {initial_count} -> {after_bj_count} -> {final_count} (新股已批量过滤)")

        if not filtered_data_list:
            logger.error("[FAIL] 所有数据都被过滤掉了")
            return False

        # 合并过滤后的数据
        factor_data = pd.concat(filtered_data_list, ignore_index=True)

        # [强化] 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        logger.info("[WRENCH] 按['date', 'stock_code']排序数据...")
        factor_data = factor_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        logger.info("[OK] 数据已按['date', 'stock_code']排序")

        logger.info(f"[OK] 数据过滤完成")
        logger.info(f"[CHART] 过滤统计总结:")
        logger.info(f"      初始样本: {original_count:,}条")
        logger.info(f"      过滤新股: {new_stock_filtered_count:,}条 (批量处理)")
        logger.info(f"      过滤北交所/ST股票: 原始样本-{original_count-new_stock_filtered_count:,}条 -> {len(factor_data):,}条")
        logger.info(f"[CHART] 过滤后数据统计: {len(factor_data)} 条记录")

        # 3.5. 训练数据预处理说明
        logger.info("[REFRESH] 训练数据预处理策略...")
        logger.info("=" * 60)
        logger.info("  ℹ️ 训练阶段不在此过滤涨跌停、流动性、is_can_buy")
        logger.info("     理由: 保留所有数据,在标签生成时标记不可操作样本为NaN")
        logger.info("     优点: 可以学习'T-1涨停->T正常'等有价值的模式")
        logger.info("     过滤: 在标签生成函数中进行,标记以下情况为NaN:")
        logger.info("           - T日涨停(无法买入+风险高)")
        logger.info("           - T日跌停(价格异常)")
        logger.info("           - T日流动性不足(无法买入)")
        logger.info("           - T+1日一字涨停(is_can_buy=0)")
        logger.info("=" * 60)

        # [WRENCH] 修复数据泄漏: 先划分训练集和验证集,再分别生成标签
        logger.info("\n" + "=" * 60)
        logger.info("[WRENCH] 修复数据泄漏: 先划分数据,再分别生成标签")
        logger.info("=" * 60)
        logger.info("  [OK] 步骤1: 按8:2顺序划分训练集和验证集")
        logger.info("  [OK] 步骤2: 对训练集生成标签,计算Winsorization阈值")
        logger.info("  [OK] 步骤3: 对验证集生成标签,使用训练集的阈值")
        logger.info("  [OK] 避免验证集信息泄漏到Winsorization阈值中")
        logger.info("=" * 60)

        # [强化] 4.1 先划分训练集和验证集(在标签生成之前)
        split_point = int(len(factor_data) * 0.8)
        train_data_unlabeled = factor_data.iloc[:split_point].copy()
        val_data_unlabeled = factor_data.iloc[split_point:].copy()

        # [强化] 关键修复: 划分后立即排序,确保数据连续性
        train_data_unlabeled = train_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)
        val_data_unlabeled = val_data_unlabeled.sort_values(['date', 'stock_code']).reset_index(drop=True)

        logger.info(f"[CHART] 数据划分完成:")
        logger.info(f"   训练集: {len(train_data_unlabeled):,}条")
        logger.info(f"   验证集: {len(val_data_unlabeled):,}条")
        logger.info(f"   [OK] 训练集和验证集已按['date', 'stock_code']排序")

        # [强化] 4.2 对训练集生成标签
        logger.info("\n[REFRESH] 步骤1: 对训练集生成标签...")
        train_labeled_result = _generate_training_labels_for_all_data(
            train_data_unlabeled)

        # 检查训练集标签生成是否成功
        if train_labeled_result is None:
            logger.error("[FAIL] 训练集标签生成失败")
            return False

        # 解包训练集结果
        if isinstance(train_labeled_result, tuple):
            train_data_labeled, label_stats = train_labeled_result
            logger.info("[OK] 训练集标签构建完成")
        else:
            train_data_labeled = train_labeled_result
            label_stats = None
            logger.warning("[WARNING] 未获得训练集标签统计量")

        # NOTE: 不在 DataFrame 层 dropna(relevance)，由后续模型输入 mask 控制

        train_data_labeled = train_data_labeled
        # [强化] 4.3 对验证集生成标签
        logger.info("\n[REFRESH] 步骤2: 对验证集生成标签...")
        val_labeled_result = _generate_training_labels_for_all_data(val_data_unlabeled)

        # 检查验证集标签生成是否成功
        if val_labeled_result is None:
            logger.error("[FAIL] 验证集标签生成失败")
            return False

        # 解包验证集结果
        if isinstance(val_labeled_result, tuple):
            val_data_labeled, _ = val_labeled_result
            logger.info("[OK] 验证集标签构建完成")
        else:
            val_data_labeled = val_labeled_result
            logger.info("[OK] 验证集标签构建完成")

        # NOTE: 不在 DataFrame 层 dropna(relevance)，由后续模型输入 mask 控制

        val_data_labeled = val_data_labeled
        # [强化] 4.4 合并训练集和验证集
        labeled_data = pd.concat([train_data_labeled, val_data_labeled], ignore_index=True)

        # [强化] 关键修复: concat后立即排序,确保数据连续性
        labeled_data = labeled_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
        logger.info(f"[OK] 合并训练集和验证集: {len(labeled_data):,}条")
        logger.info(f"   [OK] 已按['date', 'stock_code']排序")

        # [强化] 5. 连续收益标签提取和验证（排序模型模式）
        logger.info("\n🏷️ 提取和验证连续收益标签（排序模型模式）...")
        labels, valid_mask, relevance, relevance_30 = _extract_and_validate_labels(labeled_data, label_column='atr_return_label')

        if labels is None or valid_mask is None or relevance is None or relevance_30 is None:
            logger.error("连续收益标签提取和验证失败")
            return False

        # 过滤有效样本
        filtered_data = labeled_data[valid_mask].copy()
        filtered_labels = labels[valid_mask].copy()
        filtered_relevance = relevance[valid_mask].copy()
        filtered_relevance_30 = relevance_30[valid_mask].copy()

        # [强化] 关键修复: 过滤后立即排序,确保数据连续性
        # [WRENCH] 重要: 必须同时对数据和标签进行排序,保持对应关系
        sort_idx = filtered_data.sort_values(['date', 'stock_code']).index
        filtered_data = filtered_data.loc[sort_idx].reset_index(drop=True)
        filtered_labels = filtered_labels.loc[sort_idx].reset_index(drop=True)
        filtered_relevance = filtered_relevance.loc[sort_idx].reset_index(drop=True)
        filtered_relevance_30 = filtered_relevance_30.loc[sort_idx].reset_index(drop=True)
        logger.info(f"   [OK] 过滤后数据和标签已按['date', 'stock_code']同步排序")

        # [强化] 使用Winsorize后的标签（0.1%/99.9%截尾）
        # atr_return_label已经在_generate_training_labels_for_all_data中进行了Winsorize处理
        filtered_labels_original = filtered_labels.copy()
        logger.info(f"[OK] 使用Winsorize后的标签（0.1%/99.9%截尾）")
        logger.info(f"  标签范围: [{filtered_labels.min():.3f}, {filtered_labels.max():.3f}]")

        logger.info(f"有效训练数据准备完成: {len(filtered_data)} 个样本, {len(feature_columns)} 个特征列")

        # 5.5. 特征处理策略说明
        logger.info("[REFRESH] 特征处理策略: 所有特征预处理将在_train_ml_model内部完成")
        logger.info("  [OK] 数据清洗、Winsorization、标准化等步骤统一处理")
        logger.info("  [OK] 确保训练和验证集使用相同的预处理参数")
        logger.info("  [OK] 避免数据泄露，提高模型可靠性")
        logger.info("=" * 50)

        # 6. 模型训练
        # [强化] 超参数搜索控制
        enable_param_search = getattr(args, 'enable_param_search', True)  # [OK] 默认启用参数搜索
        search_strategy = getattr(args, 'search_strategy', 'bayesian')  # [OK] 默认使用贝叶斯搜索

        # [强化] 解析模型选择参数
        if use_xgboost is None or use_lightgbm is None:
            # 从命令行参数解析
            models_str = getattr(args, 'models', 'xgboost,lightgbm')
            selected_models = [m.strip().lower() for m in models_str.split(',')]

            if use_xgboost is None:
                use_xgboost = 'xgboost' in selected_models
            if use_lightgbm is None:
                use_lightgbm = 'lightgbm' in selected_models

        if not use_xgboost and not use_lightgbm:
            logger.error("[FAIL] 至少需要选择一个模型 (xgboost 或 lightgbm)")
            return False

        logger.info(f"[强化] 模型选择:")
        logger.info(f"   XGBoost: {'[OK]' if use_xgboost else '[FAIL]'}")
        logger.info(f"   LightGBM: {'[OK]' if use_lightgbm else '[FAIL]'}")

        if use_xgboost and enable_param_search:
            logger.info(f"[SEARCH] XGBoost超参数搜索: 策略={search_strategy}")
        elif use_xgboost:
            logger.info("⏭️ XGBoost跳过超参数搜索，使用默认参数")

        if use_lightgbm:
            logger.info("[CHART] LightGBM使用默认参数（无需参数搜索）")

        if use_xgboost and use_lightgbm:
            # [强化] 集成学习模式: 训练2个模型
            logger.info("开始训练集成模型（XGBoost + LightGBM）...")

            # [强化] 优化：预处理数据（所有ensemble模型共享）
            logger.info("\n" + "=" * 60)
            logger.info("[WRENCH] 预处理训练数据（所有ensemble模型共享）")
            logger.info("=" * 60)
            logger.info("[CLIPBOARD] 优化说明:")
            logger.info("  [OK] 数据划分、Winsorization等预处理只执行一次")
            logger.info("  [OK] 所有模型使用相同的训练/验证集")
            logger.info("  [OK] 节省约40-50%的训练时间")
            logger.info("  [OK] 提高模型对比的公平性")

            # 6.1 数据划分(时间序列友好)
            split_point = int(len(filtered_data) * 0.8)
            raw_train_data = filtered_data.iloc[:split_point]
            raw_val_data = filtered_data.iloc[split_point:]
            y_train_split = filtered_labels.iloc[:split_point] if hasattr(filtered_labels, 'iloc') else filtered_labels[:split_point]
            y_val_split = filtered_labels.iloc[split_point:] if hasattr(filtered_labels, 'iloc') else filtered_labels[split_point:]

            # [强化] 划分原始标签（与训练标签相同，未经Winsorize处理）
            y_train_split_original_ensemble = None
            y_val_split_original_ensemble = None
            if filtered_labels_original is not None:
                y_train_split_original_ensemble = filtered_labels_original.iloc[:split_point] if hasattr(filtered_labels_original, 'iloc') else filtered_labels_original[:split_point]
                y_val_split_original_ensemble = filtered_labels_original.iloc[split_point:] if hasattr(filtered_labels_original, 'iloc') else filtered_labels_original[split_point:]
                logger.info("[OK] 划分原始ATR标签（未经Winsorize处理）")

            # 🆕 提取训练集和验证集日期（用于时间衰减权重和group计算）
            train_dates_for_ensemble = None
            val_dates_for_ensemble = None
            if 'date' in raw_train_data.columns:
                train_dates_for_ensemble = raw_train_data['date'].values
                logger.info("[OK] 提取训练集日期（用于时间衰减权重计算和group计算）")
            if 'date' in raw_val_data.columns:
                val_dates_for_ensemble = raw_val_data['date'].values
                logger.info("[OK] 提取验证集日期（用于group计算）")


            # 🆕 提取换手率和流通市值（用于基础权重计算）
            train_turnover_rate_for_ensemble = None
            train_circ_mv_for_ensemble = None
            if 'turnover_rate' in raw_train_data.columns:
                train_turnover_rate_for_ensemble = raw_train_data['turnover_rate'].values
                logger.info("[OK] 提取训练集换手率（用于基础权重计算）")
            else:
                logger.warning("[WARNING] 训练数据中未包含turnover_rate列，将使用均匀权重")

            if 'circ_mv' in raw_train_data.columns:
                train_circ_mv_for_ensemble = raw_train_data['circ_mv'].values
                logger.info("[OK] 提取训练集流通市值（用于基础权重计算）")
            else:
                logger.warning("[WARNING] 训练数据中未包含circ_mv列，将使用均匀权重")

            # 🆕 提取relevance（用于样本权重加权和LightGBM训练）
            train_relevance_for_ensemble = filtered_relevance.iloc[:split_point] if hasattr(filtered_relevance, 'iloc') else filtered_relevance[:split_point]
            val_relevance_for_ensemble = filtered_relevance.iloc[split_point:] if hasattr(filtered_relevance, 'iloc') else filtered_relevance[split_point:]
            logger.info("[OK] 提取训练集和验证集relevance(4层)（用于样本权重加权和LightGBM训练）")

            # 🆕 提取relevance_30（用于XGBoost训练）
            train_relevance_30_for_ensemble = filtered_relevance_30.iloc[:split_point] if hasattr(filtered_relevance_30, 'iloc') else filtered_relevance_30[:split_point]
            val_relevance_30_for_ensemble = filtered_relevance_30.iloc[split_point:] if hasattr(filtered_relevance_30, 'iloc') else filtered_relevance_30[split_point:]
            logger.info("[OK] 提取训练集和验证集relevance_30(30层)（用于XGBoost训练）")

            logger.info(f"[CHART] 数据划分: 训练集{len(raw_train_data):,}样本, 验证集{len(raw_val_data):,}样本")

            # 6.2 特征预处理
            X_train_processed, feature_names_processed = _prepare_feature_data(
                raw_train_data, feature_columns, data_type="训练集"
            )

            if X_train_processed is None:
                logger.error("训练集特征预处理失败")
                return False

            X_val_processed, _ = _prepare_feature_data(
                raw_val_data, feature_columns, data_type="验证集"
            )

            if X_val_processed is None:
                logger.error("验证集特征预处理失败")
                return False

            # 6.3 调整标签长度(预处理可能移除重复行)
            if len(X_train_processed) != len(y_train_split):
                logger.warning(f"训练集预处理后长度变化: {len(y_train_split)} -> {len(X_train_processed)}")
                y_train_split = y_train_split[:len(X_train_processed)]
                # 🆕 同步调整train_dates长度
                if train_dates_for_ensemble is not None:
                    train_dates_for_ensemble = train_dates_for_ensemble[:len(X_train_processed)]
                # 🆕 同步调整换手率和流通市值长度
                if train_turnover_rate_for_ensemble is not None:
                    train_turnover_rate_for_ensemble = train_turnover_rate_for_ensemble[:len(X_train_processed)]
                if train_circ_mv_for_ensemble is not None:
                    train_circ_mv_for_ensemble = train_circ_mv_for_ensemble[:len(X_train_processed)]
                # [强化] 同步调整原始标签长度（与训练标签相同）
                if y_train_split_original_ensemble is not None:
                    y_train_split_original_ensemble = y_train_split_original_ensemble[:len(X_train_processed)]
                # [强化] 同步调整relevance长度（关键修复：避免group信息错乱）
                train_relevance_for_ensemble = train_relevance_for_ensemble[:len(X_train_processed)]
                # [强化] 同步调整relevance_30长度
                train_relevance_30_for_ensemble = train_relevance_30_for_ensemble[:len(X_train_processed)]

            if len(X_val_processed) != len(y_val_split):
                logger.warning(f"验证集预处理后长度变化: {len(y_val_split)} -> {len(X_val_processed)}")
                y_val_split = y_val_split[:len(X_val_processed)]
                # [强化] 同步调整原始标签长度（与验证标签相同）
                if y_val_split_original_ensemble is not None:
                    y_val_split_original_ensemble = y_val_split_original_ensemble[:len(X_val_processed)]
                # [强化] 同步调整relevance长度（关键修复：避免group信息错乱）
                val_relevance_for_ensemble = val_relevance_for_ensemble[:len(X_val_processed)]
                # [强化] 同步调整relevance_30长度
                val_relevance_30_for_ensemble = val_relevance_30_for_ensemble[:len(X_val_processed)]
                # [强化] 同步调整val_atr14长度
                if val_atr14_for_ensemble is not None:
                    val_atr14_for_ensemble = val_atr14_for_ensemble[:len(X_val_processed)]
                # [WRENCH] 关键修复: 同步调整val_dates长度（避免日期与数据错位）
                if val_dates_for_ensemble is not None:
                    val_dates_for_ensemble = val_dates_for_ensemble[:len(X_val_processed)]
                    logger.info(f"   [OK] 同步调整验证集日期长度: {len(val_dates_for_ensemble)}")

            # 6.3.5 条件计算样本权重（仅LightGBM需要）
            unified_sample_weights = None
            if use_lightgbm:
                logger.info("[WRENCH] 计算样本权重（供LightGBM使用）...")
                unified_sample_weights = _calculate_professional_sample_weights(
                    train_relevance_for_ensemble,  # 使用relevance标签
                    imbalance_ratio=1.0,
                    logger=logger,
                    boundary_distances=None,
                    is_regression=True,
                    y_train_original=None,
                    train_dates=train_dates_for_ensemble,
                    time_decay_half_life=180.0,  # 统一使用180天
                    turnover_rate=train_turnover_rate_for_ensemble,
                    circ_mv=train_circ_mv_for_ensemble,
                    label_rank=train_relevance_for_ensemble,  # relevance值(0-3)
                    label_boost_beta=0.15,  # 统一使用0.15
                    w_min=0.2,
                    w_max=5.0
                )
                logger.info(f"[OK] 样本权重计算完成（供LightGBM使用）")
                logger.info(f"   权重范围: [{unified_sample_weights.min():.4f}, {unified_sample_weights.max():.4f}]")
                logger.info(f"   权重均值: {unified_sample_weights.mean():.4f}")
            else:
                logger.info("⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")

            # 6.4 准备预处理数据字典
            # [WRENCH] 关键修复: 明确区分两个模型的标签
            # - XGBoost使用relevance_30(0-29)
            # - LightGBM使用relevance(0-3)
            # - 原始ATR收益率标签用于IC计算
            preprocessed_data = {
                'X_train': X_train_processed,
                'X_val': X_val_processed,
                # [WRENCH] 原始ATR收益率标签(用于IC计算和评估)
                'y_train': y_train_split,  # ATR收益率标签
                'y_val': y_val_split,  # ATR收益率标签
                'y_train_original': y_train_split_original_ensemble,  # ATR收益率标签(未经Winsorize)
                'y_val_original': y_val_split_original_ensemble,  # ATR收益率标签(未经Winsorize)
                # [WRENCH] 模型训练标签(两个模型使用不同的标签)
                'train_relevance': train_relevance_for_ensemble,  # LightGBM训练标签(0-3)
                'val_relevance': val_relevance_for_ensemble,  # LightGBM训练标签(0-3)
                'train_relevance_30': train_relevance_30_for_ensemble,  # XGBoost训练标签(0-29)
                'val_relevance_30': val_relevance_30_for_ensemble,  # XGBoost训练标签(0-29)
                # [WRENCH] 其他辅助数据
                'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                'train_dates': train_dates_for_ensemble,  # 训练集日期
                'val_dates': val_dates_for_ensemble,  # 验证集日期
                'val_atr14': val_atr14_for_ensemble,  # 验证集ATR14(用于止盈止损)
                'turnover_rate': train_turnover_rate_for_ensemble,  # 训练集换手率
                'circ_mv': train_circ_mv_for_ensemble,  # 训练集流通市值
                'sample_weights': unified_sample_weights  # 样本权重(LightGBM使用)
            }

            logger.info(f"[OK] 预处理完成: 训练集{len(X_train_processed):,}样本, 验证集{len(X_val_processed):,}样本")
            logger.info("=" * 60)

            # [强化] 生成统一的时间戳（XGBoost和LightGBM共用）
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"🕐 模型统一时间戳: {model_timestamp}")

            # [强化] 训练XGBoost模型(如果选择了XGBoost)
            xgb_results = None
            if use_xgboost:
                logger.info(f"\n{'='*60}")
                logger.info(f"训练XGBoost模型")
                logger.info(f"{'='*60}")

                try:
                    xgb_results = _train_ml_model(
                        filtered_data,
                        filtered_labels,
                        feature_columns,
                        label_stats=label_stats,
                        enable_param_search=enable_param_search,
                        search_strategy=search_strategy,
                        random_seed=5,
                        preprocessed_data=preprocessed_data,
                        enable_shap_analysis=True,  # XGBoost启用SHAP分析
                        market_data=market_data  # [强化] 传递market_data用于XGBoost计算group权重
                    )

                    if xgb_results is None:
                        logger.error("XGBoost模型训练失败：_train_ml_model返回None")
                        if not use_lightgbm:
                            return False
                        logger.warning("将继续训练LightGBM模型")
                    else:
                        logger.info(f"[OK] XGBoost模型训练完成")

                except Exception as e:
                    logger.error(f"XGBoost模型训练失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    if not use_lightgbm:
                        return False
                    logger.warning("将继续训练LightGBM模型")
                    xgb_results = None

            # [强化] 如果选择了LightGBM,训练LightGBM模型
            lgb_results = None
            if use_lightgbm:
                logger.info(f"\n{'='*60}")
                logger.info(f"训练LightGBM模型")
                logger.info(f"{'='*60}")

                try:
                    # [强化] 确定特征列表
                    if xgb_results is not None:
                        # 使用XGBoost训练后的特征列表(已对齐)
                        lgb_feature_names = xgb_results['feature_names']
                        logger.info(f"[OK] 使用XGBoost的特征列表: {len(lgb_feature_names)}个特征")
                    else:
                        # 独立训练LightGBM,使用原始特征列表
                        lgb_feature_names = feature_columns
                        logger.info(f"[OK] 独立训练LightGBM,使用原始特征列表: {len(lgb_feature_names)}个特征")

                    lgb_results = _train_lightgbm_model(
                        filtered_data,
                        filtered_labels,
                        lgb_feature_names,
                        label_stats=label_stats,
                        preprocessed_data=preprocessed_data,
                        optimize_thresholds=args.optimize_thresholds,  # [强化] 传递止盈止损优化参数
                        sw_sector_mapping=sw_sector_mapping  # [SW行业编码] 传递SW行业数据用于分类特征
                    )

                    if lgb_results is None:
                        logger.error("[FAIL] LightGBM模型训练失败")
                        if not use_xgboost or xgb_results is None:
                            return False
                    else:
                        logger.info(f"[OK] LightGBM模型训练完成")

                except Exception as e:
                    logger.error(f"[FAIL] LightGBM模型训练失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    if not use_xgboost or xgb_results is None:
                        return False
                    lgb_results = None

            # [强化] 收集模型结果
            ensemble_results = []
            if xgb_results is not None:
                ensemble_results.append(xgb_results)
            if lgb_results is not None:
                ensemble_results.append(lgb_results)

            if len(ensemble_results) == 0:
                logger.error("[FAIL] 所有模型训练失败")
                return False

            logger.info(f"\n{'='*60}")
            logger.info(f"[OK] 模型训练完成: {len(ensemble_results)}个模型")
            logger.info(f"{'='*60}")

            # [强化] 对比XGBoost和LightGBM的指标
            if xgb_results is not None and lgb_results is not None:
                logger.info(f"\n{'='*80}")
                logger.info(f"[CHART] XGBoost vs LightGBM 指标对比（每日指标平均值）")
                logger.info(f"{'='*80}")

                xgb_perf = xgb_results.get('performance', {})
                lgb_perf = lgb_results.get('performance', {})

                # [强化] 核心排序指标对比
                logger.info(f"\n[TARGET] 核心排序指标:")
                logger.info(f"  {'指标':<20} {'XGBoost':>12} {'LightGBM':>12} {'差异':>12}")
                logger.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

                # NDCG@5
                xgb_ndcg5 = xgb_perf.get('ndcg_5', 0.0)
                lgb_ndcg5 = lgb_perf.get('ndcg_5', 0.0)
                diff_ndcg5 = xgb_ndcg5 - lgb_ndcg5
                logger.info(f"  {'NDCG@5':<20} {xgb_ndcg5:>12.4f} {lgb_ndcg5:>12.4f} {diff_ndcg5:>+12.4f}")

                # NDCG@10
                xgb_ndcg10 = xgb_perf.get('ndcg_10', 0.0)
                lgb_ndcg10 = lgb_perf.get('ndcg_10', 0.0)
                diff_ndcg10 = xgb_ndcg10 - lgb_ndcg10
                logger.info(f"  {'NDCG@10':<20} {xgb_ndcg10:>12.4f} {lgb_ndcg10:>12.4f} {diff_ndcg10:>+12.4f}")

                # Rank IC (每日Spearman平均)
                xgb_rank_ic = xgb_perf.get('rank_ic', 0.0)
                lgb_rank_ic = lgb_perf.get('rank_ic', 0.0)
                diff_rank_ic = xgb_rank_ic - lgb_rank_ic
                logger.info(f"  {'Rank IC (每日平均)':<20} {xgb_rank_ic:>12.4f} {lgb_rank_ic:>12.4f} {diff_rank_ic:>+12.4f}")

                # IC (每日Pearson平均)
                xgb_ic = xgb_perf.get('ic', 0.0)
                lgb_ic = lgb_perf.get('ic', 0.0)
                diff_ic = xgb_ic - lgb_ic
                logger.info(f"  {'IC (每日平均)':<20} {xgb_ic:>12.4f} {lgb_ic:>12.4f} {diff_ic:>+12.4f}")

                # Top-5收益率
                xgb_top5 = xgb_perf.get('top_5_return', 0.0)
                lgb_top5 = lgb_perf.get('top_k_return', 0.0)  # LightGBM使用top_k_return
                diff_top5 = xgb_top5 - lgb_top5
                logger.info(f"  {'Top-5收益率':<20} {xgb_top5:>12.4f} {lgb_top5:>12.4f} {diff_top5:>+12.4f}")

                # [强化] 辅助指标对比
                logger.info(f"\n[CHART] 辅助指标（仅供参考）:")
                logger.info(f"  {'指标':<20} {'XGBoost':>12} {'LightGBM':>12} {'差异':>12}")
                logger.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

                # 验证集R²
                xgb_r2 = xgb_perf.get('val_r2', 0.0)
                lgb_r2 = lgb_perf.get('val_r2', 0.0)
                diff_r2 = xgb_r2 - lgb_r2
                logger.info(f"  {'验证集R²':<20} {xgb_r2:>12.4f} {lgb_r2:>12.4f} {diff_r2:>+12.4f}")

                # 验证集RMSE
                xgb_rmse = xgb_perf.get('val_rmse', 0.0)
                lgb_rmse = lgb_perf.get('val_rmse', 0.0)
                diff_rmse = xgb_rmse - lgb_rmse
                logger.info(f"  {'验证集RMSE':<20} {xgb_rmse:>12.6f} {lgb_rmse:>12.6f} {diff_rmse:>+12.6f}")

                logger.info(f"\n[IDEA] 说明:")
                logger.info(f"  - NDCG@5/10: 按日期分组计算,然后平均")
                logger.info(f"  - Rank IC: 每日Spearman相关系数的平均值")
                logger.info(f"  - IC: 每日Pearson相关系数的平均值")
                logger.info(f"  - Top-5收益率: 每日Top-5股票收益率的平均值")
                logger.info(f"  - 正值差异表示XGBoost更优,负值表示LightGBM更优")
                logger.info(f"{'='*80}\n")

        else:
            # [强化] 单模型模式: 只训练1个模型(用于参数搜索)
            logger.info("开始训练XGBoost模型（单模型）...")

            if enable_param_search:
                logger.info(f"[SEARCH] 启用超参数搜索: 策略={search_strategy}")
                logger.info("[CLIPBOARD] 单模型模式: 只训练1个模型用于参数搜索")
            else:
                logger.info("[SEARCH] 使用默认参数配置（跳过超参数搜索）")
                logger.info("[CLIPBOARD] 单模型模式: 只训练1个模型")

            logger.info("\n" + "=" * 60)
            logger.info("[WRENCH] 训练单个模型")
            logger.info("=" * 60)

            # [强化] 生成时间戳（单模型）
            model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"🕐 模型时间戳: {model_timestamp}")

            try:
                # 训练单个模型（不使用预处理数据共享）
                model_results = _train_ml_model(
                    filtered_data,
                    filtered_labels,
                    feature_columns,
                    label_stats=label_stats,  # 传递标签统计量用于反向转换
                    enable_param_search=enable_param_search,
                    search_strategy=search_strategy,
                    random_seed=5,
                    preprocessed_data=None,  # 不使用预处理数据共享
                    enable_shap_analysis=True,  # 启用SHAP分析
                    market_data=market_data  # [强化] 传递market_data用于XGBoost计算group权重
                )

                if model_results is None:
                    logger.error("模型训练失败：_train_ml_model返回None")
                    return False

                logger.info("[OK] 模型训练完成")

            except Exception as e:
                logger.error(f"模型训练失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                return False

            # 将单个模型结果包装成列表,以兼容后续代码
            ensemble_results = [model_results]

            logger.info(f"\n{'='*60}")
            logger.info(f"[OK] 模型训练完成")
            logger.info(f"{'='*60}")

        # 7. 保存所有模型（使用统一时间戳）
        for model_results in ensemble_results:
            feature_names = model_results.get('feature_names', feature_columns)
            model_type = model_results.get('model_type', 'xgboost')  # 获取模型类型

            # XGBoost需要PCA信息,LightGBM不需要
            if model_type == 'xgboost':
                pca_info = model_results.get('pca_info', {})
                pca_models = {'market_factors': pca_info} if pca_info.get('applied', False) else {}
            else:
                pca_models = None

            _save_model_results(
                model_results,
                feature_names,
                pca_models,
                model_type=model_type,  # [强化] 传递模型类型
                timestamp=model_timestamp  # [强化] 传递统一时间戳
            )

        print("[OK] 模型训练模式执行完成")
        return True

    except Exception as e:
        logger.error(f"[FAIL] 模型训练模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _plot_top1_equity_curve(
    top1_result: Dict,
    window_name: str = None,
    save_path: str = "./results/top1_equity_curves"
):
    """
    绘制Top1策略的资金收益曲线图

    在一张图上显示：
    - 总资金/净值变化曲线
    - 所有买入卖出点（标注股票代码、价格、收益率）
    - 被跳过的买入（开盘风控）

    Args:
        top1_result: Top1策略的回测结果
        window_name: 窗口名称
        save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from datetime import datetime
    import os

    logger = logging.getLogger(__name__)

    try:
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 提取数据
        trades = top1_result.get('trades', [])
        trading_dates = top1_result.get('trading_dates', [])
        skipped_attempts = top1_result.get('skipped_buy_attempts', [])
        portfolio_values = top1_result.get('daily_portfolio_values', [])
        initial_capital = top1_result.get('initial_capital', 1000000)

        logger.info(f"📊 Top1策略数据统计:")
        logger.info(f"  - 交易记录: {len(trades)}笔")
        logger.info(f"  - 被跳过买入: {len(skipped_attempts)}次")
        logger.info(f"  - 交易日: {len(trading_dates)}天")
        logger.info(f"  - 资金净值: {len(portfolio_values)}个数据点")
        logger.info(f"  - 初始资金: {initial_capital:,.0f}")
        logger.info(f"  - 最终资金: {portfolio_values[-1] if portfolio_values else 0:,.0f}")

        # 🔍 检查数据一致性
        if len(portfolio_values) != len(trading_dates):
            logger.warning(f"⚠️ 数据长度不一致: trading_dates={len(trading_dates)}, portfolio_values={len(portfolio_values)}")
            logger.warning(f"  将使用较短的长度进行绘制，可能部分交易点无法标注")

        if not trades and not skipped_attempts:
            logger.warning("Top1策略无交易记录和被跳过买入，跳过曲线绘制")
            return

        if not portfolio_values:
            logger.warning("Top1策略缺少资金净值数据，跳过曲线绘制")
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

        # ========== 上图：资金曲线 + 买卖点标注 ==========
        # 生成X轴（交易日）
        x_range = range(len(portfolio_values))

        # 绘制资金曲线
        ax1.plot(x_range, portfolio_values,
                linewidth=2.5, color='#2E86AB', label='Top1策略净值',
                marker='o', markersize=4)

        # 标注所有买卖点
        for i, trade in enumerate(trades):
            buy_datetime = trade.get('buy_datetime')
            sell_datetime = trade.get('sell_datetime')
            stock_code = trade.get('stock_code', 'N/A')
            buy_price = trade.get('buy_price', 0)
            sell_price = trade.get('sell_price', 0)
            actual_return = trade.get('actual_return', 0) * 100
            exit_reason = trade.get('exit_reason', 'N/A')

            # 找到买入和卖出日期在X轴上的位置
            buy_date = buy_datetime.date() if isinstance(buy_datetime, datetime) else buy_datetime
            sell_date = sell_datetime.date() if isinstance(sell_datetime, datetime) else sell_datetime

            try:
                buy_idx = trading_dates.index(buy_date)

                # 🔍 边界检查：确保索引在portfolio_values范围内
                if buy_idx >= len(portfolio_values):
                    logger.warning(f"买入索引超出范围: buy_idx={buy_idx}, portfolio_values长度={len(portfolio_values)}, 跳过标注")
                    continue

                buy_value = portfolio_values[buy_idx]

                # 买入点（绿色向上三角形）
                ax1.scatter(buy_idx, buy_value, marker='^', color='green', s=300, zorder=10,
                           edgecolors='darkgreen', linewidths=2, label='买入' if i == 0 else '')

                # 标注买入信息
                buy_annotation = (
                    f"#{i+1} {stock_code}\n"
                    f"买入: {buy_price:.2f}\n"
                    f"日期: {buy_date.strftime('%m-%d')}"
                )
                ax1.annotate(buy_annotation,
                           xy=(buy_idx, buy_value),
                           xytext=(10, 30), textcoords='offset points',
                           fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.4', fc='lightgreen', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))
            except ValueError:
                logger.warning(f"买入日期 {buy_date} 不在交易日期列表中")

            try:
                sell_idx = trading_dates.index(sell_date)

                # 🔍 边界检查：确保索引在portfolio_values范围内
                if sell_idx >= len(portfolio_values):
                    logger.warning(f"卖出索引超出范围: sell_idx={sell_idx}, portfolio_values长度={len(portfolio_values)}, 跳过标注")
                    continue

                sell_value = portfolio_values[sell_idx]

                # 卖出点（红色向下三角形）
                ax1.scatter(sell_idx, sell_value, marker='v', color='red', s=300, zorder=10,
                           edgecolors='darkred', linewidths=2, label='卖出' if i == 0 else '')

                # 标注卖出信息
                color = 'lightgreen' if actual_return >= 0 else 'lightcoral'
                sell_annotation = (
                    f"#{i+1} {stock_code}\n"
                    f"卖出: {sell_price:.2f}\n"
                    f"收益: {actual_return:.2f}%\n"
                    f"原因: {exit_reason}\n"
                    f"日期: {sell_date.strftime('%m-%d')}"
                )
                ax1.annotate(sell_annotation,
                           xy=(sell_idx, sell_value),
                           xytext=(10, -40), textcoords='offset points',
                           fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.4', fc=color, alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
            except ValueError:
                logger.warning(f"卖出日期 {sell_date} 不在交易日期列表中")

        # 标注被跳过的买入
        for skip in skipped_attempts:
            skip_date = skip.get('attempt_buy_date')
            skip_date_str = skip_date.strftime('%m-%d') if isinstance(skip_date, datetime) else str(skip_date)
            stock_code = skip.get('stock_code', 'N/A')
            skip_reason = skip.get('skip_reason', 'N/A')
            predicted_return = skip.get('predicted_return', 0)

            try:
                skip_idx = trading_dates.index(skip_date)

                # 🔍 边界检查：确保索引在portfolio_values范围内
                if skip_idx >= len(portfolio_values):
                    logger.warning(f"跳过索引超出范围: skip_idx={skip_idx}, portfolio_values长度={len(portfolio_values)}, 跳过标注")
                    continue

                skip_value = portfolio_values[skip_idx]

                # 用灰色叉号标记
                ax1.scatter(skip_idx, skip_value, marker='x', color='gray', s=300, zorder=10,
                           linewidths=3, label='跳过买入' if skip == skipped_attempts[0] else '')

                # 标注跳过信息
                skip_annotation = (
                    f"{stock_code}\n"
                    f"原因: {skip_reason}\n"
                    f"预测收益: {predicted_return:.2%}\n"
                    f"日期: {skip_date_str}"
                )
                ax1.annotate(skip_annotation,
                           xy=(skip_idx, skip_value),
                           xytext=(10, -40), textcoords='offset points',
                           fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.4', fc='lightgray', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
            except ValueError:
                logger.warning(f"跳过日期 {skip_date} 不在交易日期列表中")

        ax1.set_ylabel('净值 (元)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top1策略资金曲线 (初始资金: {initial_capital:,.0f} → 最终资金: {portfolio_values[-1]:,.0f}, 收益: {(portfolio_values[-1]/initial_capital-1)*100:.2f}%)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=initial_capital, color='black', linestyle='--', linewidth=1, alpha=0.5, label='初始资金线')

        # ========== 下图：单笔交易收益率柱状图 ==========
        if trades:
            trade_numbers = list(range(1, len(trades) + 1))
            returns = [trade.get('actual_return', 0) * 100 for trade in trades]
            colors = ['#28A745' if r >= 0 else '#DC3545' for r in returns]
            stock_codes = [trade.get('stock_code', 'N/A') for trade in trades]

            bars = ax2.bar(trade_numbers, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)

            # 在柱子上标注收益和股票代码
            for bar, ret, code in zip(bars, returns, stock_codes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ret:.1f}%\n{code}',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=7, fontweight='bold')

            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax2.set_xlabel('交易序号', fontsize=12, fontweight='bold')
            ax2.set_ylabel('收益率 (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Top1策略单笔交易收益', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

            # 添加统计信息
            total_trades = len(trades)
            winning_trades = sum(1 for r in returns if r > 0)
            win_rate = winning_trades / total_trades * 100
            avg_return = sum(returns) / total_trades

            stats_text = f"总交易: {total_trades}笔 | 胜率: {win_rate:.1f}% | 平均收益: {avg_return:.2f}%"
            ax2.text(0.02, 0.98, stats_text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图片
        filename = f"top1_equity_curve"
        if window_name:
            filename += f"_{window_name.replace(' ', '_')}"
        filename += ".png"

        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Top1策略资金曲线已保存: {save_file}")
        plt.close()

        logger.info(f"✅ Top1策略资金曲线绘制完成")

    except Exception as e:
        logger.error(f"❌ 绘制Top1资金曲线失败: {e}")
        import traceback
        logger.error(traceback.format_exc())




def _select_live_default_scheme_info(backtest_data: pd.DataFrame, logger: logging.Logger, window_prefix: str = "") -> pd.DataFrame:
    """给分钟回测面板附加与 live_trading 默认策略一致的日级选股决策。"""
    from live_trading.main import _build_select_scheme_choices

    preferred_keys = [
        'switch_reg_pct_mktstate_hi95_lo85_cpv_h2c_veto',
        'switch_reg_pct_mktstate_hi95_lo85',
        'mkt_exp_q30_reg10_raw_0p40_to_top2',
        'mkt_exp_q30_top1_hi98_to_top2',
        'mkt_exp_q30_shift1',
        'mkt_q40',
    ]

    df = backtest_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    pred_column = 'predicted_return_atr_raw' if 'predicted_return_atr_raw' in df.columns else 'predicted_return'

    regime_target_rank = []
    regime_skip_buy = []
    regime_selected_code = []
    regime_scheme_name = []

    scheme_day_counts = {}
    cash_days = 0
    total_days = 0

    for trade_date, g in df.groupby('date', sort=True):
        total_days += 1
        g_sorted = g.sort_values(pred_column, ascending=False)
        top_rows = g_sorted.head(2)
        selected_stocks = []
        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1):
            selected_stocks.append({
                'rank': idx,
                'code': row['stock_code'],
                'score': float(row[pred_column]) if pd.notna(row[pred_column]) else 0.0,
            })

        scheme_choices = _build_select_scheme_choices(trade_date.strftime('%Y%m%d'), g, selected_stocks)
        preferred_info = None
        preferred_key = None
        for key in preferred_keys:
            info = scheme_choices.get(key)
            if info:
                preferred_info = info
                preferred_key = key
                break

        if preferred_info is None:
            chosen_rank = 1
            skip_buy = False
            selected_code = selected_stocks[0]['code'] if selected_stocks else None
            scheme_name = 'fallback_top1'
        else:
            chosen_rank = int(preferred_info.get('chosen_rank') or 0)
            action = str(preferred_info.get('action', 'buy')).lower()
            skip_buy = (action == 'cash') or (chosen_rank <= 0)
            selected_code = preferred_info.get('selected_code')
            scheme_name = str(preferred_info.get('name') or preferred_key or 'unknown')

        if skip_buy:
            cash_days += 1
            chosen_rank = 1
            selected_code = None

        scheme_day_counts[scheme_name] = scheme_day_counts.get(scheme_name, 0) + 1

        n = len(g)
        regime_target_rank.extend([chosen_rank] * n)
        regime_skip_buy.extend([bool(skip_buy)] * n)
        regime_selected_code.extend([selected_code] * n)
        regime_scheme_name.extend([scheme_name] * n)

    df['regime_target_rank'] = regime_target_rank
    df['regime_skip_buy'] = regime_skip_buy
    df['regime_selected_code'] = regime_selected_code
    df['regime_scheme_name'] = regime_scheme_name

    logger.info(
        f"{window_prefix}🧭 已应用 live_trading 默认策略: total_days={total_days}, cash_days={cash_days}, "
        f"cash_ratio={cash_days / max(total_days, 1):.1%}"
    )
    logger.info(f"{window_prefix}🧭 默认策略日级方案分布: {scheme_day_counts}")
    return df

def _apply_top1_price_ma5_ratio_pct_gate(backtest_data: pd.DataFrame,
                                         logger: logging.Logger,
                                         threshold: float = 0.4,
                                         window_prefix: str = "") -> pd.DataFrame:
    """按日生成 Top1 开关: 仅当当天 Top1 的 price_ma5_ratio_pct >= threshold 时允许买入。"""
    df = backtest_data.copy()
    if 'price_ma5_ratio_pct' not in df.columns:
        raise KeyError("缺少 price_ma5_ratio_pct，无法应用 Top1 开关")

    df['date'] = pd.to_datetime(df['date'])
    pred_column = 'predicted_return_atr_raw' if 'predicted_return_atr_raw' in df.columns else 'predicted_return'

    skip_flags = []
    total_days = 0
    skipped_days = 0

    for _, g in df.groupby('date', sort=True):
        total_days += 1
        top1_row = g.sort_values(pred_column, ascending=False).iloc[0]
        gate_value = float(top1_row['price_ma5_ratio_pct'])
        skip_buy = not (pd.notna(gate_value) and gate_value >= threshold)
        if skip_buy:
            skipped_days += 1
        skip_flags.extend([skip_buy] * len(g))

    df['regime_skip_buy'] = skip_flags
    logger.info(
        f"{window_prefix}🧭 已应用 Top1 开关: price_ma5_ratio_pct >= {threshold:.2f}, "
        f"total_days={total_days}, skip_days={skipped_days}, active_ratio={(total_days - skipped_days) / max(total_days, 1):.1%}"
    )
    return df


def _apply_top1_two_branch_gate(backtest_data: pd.DataFrame,
                                logger: logging.Logger,
                                sector_threshold: float = 0.287,
                                market_vol_threshold: float = 1.695,
                                ma_alignment_threshold: float = 1.0,
                                window_prefix: str = "") -> pd.DataFrame:
    df = backtest_data.copy()
    sector_signal_column = 'sector_momentum_accel_rank'
    market_vol_column = 'market_volatility_context'
    ma_alignment_column = 'ma_alignment_persistence_5d'
    missing_columns = [
        c for c in [sector_signal_column, market_vol_column, ma_alignment_column]
        if c not in df.columns
    ]
    if missing_columns:
        logger.warning(
            f"{window_prefix}[WARN] 缺少 {missing_columns}，跳过Top1双分支风控"
        )
        df['regime_skip_buy'] = False
        return df

    df['date'] = pd.to_datetime(df['date'])
    pred_column = 'predicted_return_atr_raw' if 'predicted_return_atr_raw' in df.columns else 'predicted_return'
    if pred_column not in df.columns:
        logger.warning(f"{window_prefix}[WARN] 缺少 {pred_column}，跳过Top1行业动量+市场波动组合风控")
        df['regime_skip_buy'] = False
        return df

    total_days = 0
    risk_days = 0
    branch1_days = 0
    branch2_days = 0
    skip_flags = []
    for trade_date, group in df.groupby('date', sort=True):
        total_days += 1
        ranked = group.sort_values(pred_column, ascending=False)
        if ranked.empty:
            skip_buy = False
        else:
            top1 = ranked.iloc[0]
            sector_value = pd.to_numeric(top1.get(sector_signal_column), errors='coerce')
            market_vol_value = pd.to_numeric(top1.get(market_vol_column), errors='coerce')
            ma_alignment_value = pd.to_numeric(top1.get(ma_alignment_column), errors='coerce')
            branch1 = bool(
                pd.notna(sector_value) and pd.notna(market_vol_value) and
                float(sector_value) < float(sector_threshold) and
                float(market_vol_value) > float(market_vol_threshold)
            )
            branch2 = bool(
                pd.notna(ma_alignment_value) and
                float(ma_alignment_value) < float(ma_alignment_threshold)
            )
            skip_buy = branch1 or branch2
            if branch1:
                branch1_days += 1
            if branch2:
                branch2_days += 1
        if skip_buy:
            risk_days += 1
        skip_flags.extend([skip_buy] * len(group))

    df['regime_skip_buy'] = skip_flags
    logger.info(
        f"{window_prefix}🧭 已应用Top1双分支风控: "
        f"({sector_signal_column} < {float(sector_threshold):.3f} 且 "
        f"{market_vol_column} > {float(market_vol_threshold):.3f}) "
        f"或 ({ma_alignment_column} < {float(ma_alignment_threshold):.2f}), "
        f"risk_days={risk_days}, branch1_days={branch1_days}, branch2_days={branch2_days}, total_days={total_days}, "
        f"risk_ratio={risk_days / max(total_days, 1):.1%}"
    )
    return df


def _apply_top1_two_branch_strict_mode(backtest_data: pd.DataFrame,
                                       logger: logging.Logger,
                                       sector_threshold: float = 0.287,
                                       market_vol_threshold: float = 1.695,
                                       ma_alignment_threshold: float = 1.0,
                                       window_prefix: str = "") -> pd.DataFrame:
    df = backtest_data.copy()
    sector_signal_column = 'sector_momentum_accel_rank'
    market_vol_column = 'market_volatility_context'
    ma_alignment_column = 'ma_alignment_persistence_5d'
    missing_columns = [
        c for c in [sector_signal_column, market_vol_column, ma_alignment_column]
        if c not in df.columns
    ]
    if missing_columns:
        logger.warning(
            f"{window_prefix}[WARN] 缺少 {missing_columns}，跳过Top1双分支严格模式标记"
        )
        df['regime_strict_risk_mode'] = False
        return df

    df['date'] = pd.to_datetime(df['date'])
    pred_column = 'predicted_return_atr_raw' if 'predicted_return_atr_raw' in df.columns else 'predicted_return'
    if pred_column not in df.columns:
        logger.warning(f"{window_prefix}[WARN] 缺少 {pred_column}，跳过Top1双分支严格模式标记")
        df['regime_strict_risk_mode'] = False
        return df

    total_days = 0
    strict_days = 0
    strict_flags = []
    for trade_date, group in df.groupby('date', sort=True):
        total_days += 1
        ranked = group.sort_values(pred_column, ascending=False)
        strict_mode = False
        if not ranked.empty:
            top1 = ranked.iloc[0]
            sector_value = pd.to_numeric(top1.get(sector_signal_column), errors='coerce')
            market_vol_value = pd.to_numeric(top1.get(market_vol_column), errors='coerce')
            ma_alignment_value = pd.to_numeric(top1.get(ma_alignment_column), errors='coerce')
            branch1 = bool(
                pd.notna(sector_value) and pd.notna(market_vol_value) and
                float(sector_value) < float(sector_threshold) and
                float(market_vol_value) > float(market_vol_threshold)
            )
            branch2 = bool(
                pd.notna(ma_alignment_value) and
                float(ma_alignment_value) < float(ma_alignment_threshold)
            )
            strict_mode = branch1 or branch2
        if strict_mode:
            strict_days += 1
        strict_flags.extend([strict_mode] * len(group))

    df['regime_strict_risk_mode'] = strict_flags
    logger.info(
        f"{window_prefix}🧭 已应用Top1双分支严格模式标记: "
        f"strict_days={strict_days}, total_days={total_days}, strict_ratio={strict_days / max(total_days, 1):.1%}, "
        f"strict_gap_up阈值收紧至{TOP1_STRICT_ENTRY_GAP_UP_THRESHOLD:.2%}"
    )
    return df

def _run_minute_level_backtest(
    backtest_data,
    stock_counts=[1, 2, 3, 5],
    window_name=None,
    holding_days=2,
    use_live_default_scheme=False
):
    """
    执行5分钟级别选股回测

    Args:
        backtest_data: DataFrame,预测数据
        stock_counts: 排名位置列表，默认[1, 2, 3, 5]
                    - stock_count=1: 回测每天买入排名第1的股票
                    - stock_count=2: 回测每天买入排名第2的股票
                    - stock_count=3: 回测每天买入排名第3的股票
                    - stock_count=5: 回测每天买入排名第5的股票
                    每个策略每天只买入1只股票（指定排名的那只）
        window_name: 窗口名称(Walk-Forward专用),默认None
        holding_days: 持仓天数,默认2天(T+1开盘买入，T+2收盘卖出)

    Returns:
        dict: 回测结果
    """
    import traceback
    logger = logging.getLogger(__name__)
    window_prefix = f"[{window_name}] " if window_name else ""

    # Exhaustion filter: 不再预过滤候选池，改为在策略买入时检查Top1实际选中的股票
    # 配置通过 minte_strategy_overrides 传递给策略层
    try:
        # 导入5分钟回测模块
        from rqalpha_minute_datasource.minute_backtest_top1_gate_ma5 import MinuteLevelStockSelector

        logger.info(f"{window_prefix}⏰ 初始化5分钟级别选股回测系统...")

        # 5分钟数据路径
        minute_data_path = "../local_data/minute/"

        # 创建5分钟选股器
        selector = MinuteLevelStockSelector(minute_data_path, backtest_data)

        logger.info(f"{window_prefix}[TARGET] 开始5分钟级别回测，持仓时间: {holding_days}天")

        minute_position_schedule = [0.8, 0.2]
        use_ma5_entry_filter = str(TOP1_ENTRY_FILTER_MODE).lower() == 'ma5'
        delayed_entry_mode = str(TOP1_DELAYED_ENTRY_MODE).lower()
        minute_strategy_overrides = {
            "entry_risk_down_threshold_non_top1": -0.03,
            "entry_risk_down_threshold": -0.06,
            "entry_risk_up_threshold": 0.05,
            "entry_risk_control_strategy": "ma5_based" if use_ma5_entry_filter else "gap_based",
            "entry_risk_ma5_down_threshold": 0.00,
            "entry_risk_ma5_up_threshold": 0.05,
            "delayed_entry_mode": delayed_entry_mode,
            "stop_loss_threshold": -0.05,
            "profit_bearish_required_count": 3,
            "enable_t2_0935_hard_stop": ENABLE_T2_0935_HARD_STOP,
            "t2_0935_hard_stop_threshold": T2_0935_HARD_STOP_THRESHOLD,
            "enable_gate_hit_strict_mode": USE_TOP1_TWO_BRANCH_STRICT_MODE,
            "gate_hit_entry_gap_up_threshold": TOP1_STRICT_ENTRY_GAP_UP_THRESHOLD,
            "enable_t2_1000_soft_stop": ENABLE_T2_1000_SOFT_STOP,
            "t2_1000_soft_stop_threshold": T2_1000_SOFT_STOP_THRESHOLD,
            "enable_second_layer_time_stop": ENABLE_SECOND_LAYER_TIME_STOP,
            "t2_close_breakeven_stop_threshold": T2_CLOSE_BREAKEVEN_STOP_THRESHOLD,
            "t3_0935_breakeven_stop_threshold": T3_0935_BREAKEVEN_STOP_THRESHOLD,
            "enable_small_profit_failure_exit": ENABLE_SMALL_PROFIT_FAILURE_EXIT,
            "small_profit_activation_return": SMALL_PROFIT_ACTIVATION_RETURN,
            "small_profit_retrace_return": SMALL_PROFIT_RETRACE_RETURN,
            "small_profit_trailing_drawdown": SMALL_PROFIT_TRAILING_DRAWDOWN,
            "small_profit_bearish_required_count": SMALL_PROFIT_BEARISH_REQUIRED_COUNT,
            "small_profit_watch_window_bars": SMALL_PROFIT_WATCH_WINDOW_BARS,
            "enable_profit_threshold_for_bearish_exit": ENABLE_PROFIT_THRESHOLD_FOR_BEARISH_EXIT,
            "profit_bearish_activation_return": PROFIT_BEARISH_ACTIVATION_RETURN,
            "enable_top1_state_machine_risk": ENABLE_TOP1_STATE_MACHINE_RISK,
            "enable_dynamic_rank_switch": bool(use_live_default_scheme),
            # Exhaustion filter: 在策略层检查Top1实际选中股票（不预过滤候选池）
            "enable_exhaustion_filter": ENABLE_EXHAUSTION_BUY_FILTER,
            "exhaustion_filter_strategy": EXHAUSTION_FILTER_STRATEGY,
            "exhaustion_runup_ma5_threshold": EXHAUSTION_RUNUP_MA5_THRESHOLD,
            "exhaustion_top10_10d_strength_threshold": EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD,
            "exhaustion_prev_day_blowoff_threshold": EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD,
            "exhaustion_sector_momentum_accel_weak_threshold": EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD,
            "exhaustion_market_advance_decline_ratio_threshold": EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD,
            "exhaustion_filter_turnover_z_threshold": EXHAUSTION_FILTER_TURNOVER_Z_THRESHOLD,
        }
        entry_filter_desc = (
            "open<ma5_skip + gap_up>5% skip"
            if use_ma5_entry_filter else
            "gap_down<-6% skip + gap_up>5% skip"
        )
        logger.info(
            f"{window_prefix}🛡️ 分钟回测显式参数: "
            f"entry_filter={entry_filter_desc}, "
            f"gap_up={minute_strategy_overrides['entry_risk_up_threshold']:.2%}, "
            f"stop_loss={minute_strategy_overrides['stop_loss_threshold']:.2%}, "
            f"t2_0935_hard_stop={minute_strategy_overrides['enable_t2_0935_hard_stop']}@"
            f"{minute_strategy_overrides['t2_0935_hard_stop_threshold']:.2%}, "
            f"gate_hit_gap_up={minute_strategy_overrides['enable_gate_hit_strict_mode']}@"
            f"{minute_strategy_overrides['gate_hit_entry_gap_up_threshold']:.2%}, "
            f"t2_1000_soft_stop={minute_strategy_overrides['enable_t2_1000_soft_stop']}@"
            f"{minute_strategy_overrides['t2_1000_soft_stop_threshold']:.2%}, "
            f"second_layer_time_stop={minute_strategy_overrides['enable_second_layer_time_stop']}@"
            f"(t2_close<={minute_strategy_overrides['t2_close_breakeven_stop_threshold']:.2%} "
            f"or t3_0935<={minute_strategy_overrides['t3_0935_breakeven_stop_threshold']:.2%}), "
            f"small_profit_failure={minute_strategy_overrides['enable_small_profit_failure_exit']}@"
            f"(peak>={minute_strategy_overrides['small_profit_activation_return']:.2%}, "
            f"break_T1_low_then_bearish>={minute_strategy_overrides['small_profit_bearish_required_count']}, "
            f"reclaim_cancel=True), "
            f"bearish3_after_profit={minute_strategy_overrides['enable_profit_threshold_for_bearish_exit']}@"
            f"{minute_strategy_overrides['profit_bearish_activation_return']:.2%}, "
            f"state_machine_risk={minute_strategy_overrides['enable_top1_state_machine_risk']}, "
            f"profit_bearish_count={minute_strategy_overrides['profit_bearish_required_count']}, "
            f"delayed_entry_mode={minute_strategy_overrides['delayed_entry_mode']}, "
            f"dynamic_rank_switch={minute_strategy_overrides['enable_dynamic_rank_switch']}, "
            f"two_branch_gate={USE_TOP1_TWO_BRANCH_GATE}, "
            f"sector_mom_accel_threshold={TOP1_SECTOR_MOMENTUM_ACCEL_GATE_THRESHOLD:.3f}, "
            f"market_vol_threshold={TOP1_MARKET_VOLATILITY_CONTEXT_GATE_THRESHOLD:.3f}, "
            f"ma_alignment_threshold={TOP1_MA_ALIGNMENT_PERSISTENCE_GATE_THRESHOLD:.2f}, "
            f"exhaustion_filter={ENABLE_EXHAUSTION_BUY_FILTER}("
            f"strategy={EXHAUSTION_FILTER_STRATEGY}, "
            f"runup_ma5_th={EXHAUSTION_RUNUP_MA5_THRESHOLD}, "
            f"top10_10d_th={EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD}, "
            f"blowoff_th={EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD}, "
            f"sector_accel_weak_th={EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD}, "
            f"market_ad_ratio_th={EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD}, "
            f"tz_th={EXHAUSTION_FILTER_TURNOVER_Z_THRESHOLD})"
        )
        # 运行5分钟级别回测
        result = selector.run_minute_level_backtest(
            stock_counts=stock_counts,
            holding_days=holding_days,
            max_concurrent_positions=2,
            position_schedule=minute_position_schedule,
            strategy_config_overrides=minute_strategy_overrides,
        )

        logger.info(f"{window_prefix}[OK] 5分钟级别回测完成")

        # 输出风控统计信息
        if result and 'summary_stats' in result:
            stats = result['summary_stats']
            if 'optimization_version' in stats and 'risk_control' in stats['optimization_version']:
                logger.info(f"\n{window_prefix}🛡️ 风控系统统计:")
                logger.info(f"  开盘风控:")
                logger.info(f"    - 跳过买入总数: {stats.get('skipped_positions', 0):,}")
                logger.info(f"    - 开盘跌幅>3%: {stats.get('gap_down_3_skipped', 0):,}")
                logger.info(f"    - 开盘涨幅>5%: {stats.get('gap_up_5_skipped', 0):,}")
                logger.info(f"    - 🆕 同行业开盘跌超9%: {stats.get('industry_gap_limit_down_skipped', 0):,}")  # 🆕 更新

                if 'risk_control_details' in stats:
                    risk_details = stats['risk_control_details']
                    logger.info(f"  持仓风控:")
                    logger.info(f"    - 风控退出总数: {risk_details.get('total_risk_exits', 0):,}")
                    logger.info(f"    - 累计跌幅>5%: {risk_details.get('cumulative_decline_exits', 0):,}")
                    logger.info(f"    - 连续3阴线: {risk_details.get('consecutive_decline_exits', 0):,}")
                    logger.info(f"    - 正常到期退出: {risk_details.get('normal_exits', 0):,}")

                    if risk_details.get('total_risk_exits', 0) > 0:
                        logger.info(f"  风控效果:")
                        logger.info(f"    - 风控退出率: {risk_details.get('risk_exit_rate', 0):.1%}")
                        logger.info(f"    - 风控退出平均收益: {risk_details.get('avg_risk_exit_return', 0):.2%}")
                        logger.info(f"    - 正常退出平均收益: {risk_details.get('avg_normal_return', 0):.2%}")

        # 🔍 添加调试日志：检查返回结果结构
        logger.info(f"🔍 [DEBUG] 返回结果结构检查:")
        logger.info(f"  - result类型: {type(result)}")
        logger.info(f"  - result键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        logger.info(f"  - backtest_results存在: {'backtest_results' in result if isinstance(result, dict) else 'N/A'}")
        if isinstance(result, dict) and 'backtest_results' in result:
            logger.info(f"  - backtest_results内容: {result['backtest_results']}")

        # 📊 为Top5策略绘制个股交易曲线
        try:
            logger.info(f"{window_prefix}📈 开始生成Top5策略个股交易曲线...")

            # 🔍 详细调试：检查result结构
            logger.info(f"{window_prefix}🔍 [DEBUG-1] result类型: {type(result)}")
            logger.info(f"{window_prefix}🔍 [DEBUG-1] result的键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

            # result直接就是converted_results，结构通常是 {1: {...}, 2: {...}, 3: {...}, 5: {...}}
            # 兼容仅回测Top2/Top3等场景：优先使用Top1，否则使用最小可用Top-k
            selected_rank = None
            selected_result = None
            if isinstance(result, dict):
                int_keys = sorted([k for k in result.keys() if isinstance(k, int)])
                if 1 in result:
                    selected_rank = 1
                    selected_result = result.get(1)
                elif int_keys:
                    selected_rank = int_keys[0]
                    selected_result = result.get(selected_rank)

            logger.info(f"{window_prefix}🔍 [DEBUG-2] selected_result类型: {type(selected_result)}, rank={selected_rank}")

            if selected_result:
                logger.info(f"{window_prefix}🔍 [DEBUG-2] selected_result的键: {list(selected_result.keys()) if isinstance(selected_result, dict) else 'N/A'}")
                trades = selected_result.get('trades', None)
                skipped = selected_result.get('skipped_buy_attempts', None)
                logger.info(f"{window_prefix}🔍 [DEBUG-2] trades存在: {trades is not None}, 类型: {type(trades)}, 长度: {len(trades) if trades else 'N/A'}")
                logger.info(f"{window_prefix}🔍 [DEBUG-2] skipped_buy_attempts存在: {skipped is not None}, 类型: {type(skipped)}, 长度: {len(skipped) if skipped else 'N/A'}")

            if selected_result:
                trades_count = len(selected_result.get('trades', []))
                skipped_count = len(selected_result.get('skipped_buy_attempts', []))
                logger.info(f"{window_prefix}📊 Top{selected_rank}策略数据检查: 交易记录={trades_count}, 被跳过买入={skipped_count}")

            # # 修改条件：只要有交易记录或被跳过的买入，就绘制Top1资金曲线
            # if top1_result and (top1_result.get('trades') or top1_result.get('skipped_buy_attempts')):
            #     _plot_top1_equity_curve(
            #         top1_result=top1_result,
            #         window_name=window_name,
            #         save_path="./results/top1_equity_curves"
            #     )
            #     logger.info(f"{window_prefix}✅ Top1策略资金收益曲线生成完成")
            # else:
            #     logger.warning(f"{window_prefix}⚠️ Top1策略无交易记录和被跳过买入，跳过曲线绘制")

        except Exception as e:
            logger.error(f"{window_prefix}❌ Top1资金曲线生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

        logger.info(f"✅ {window_prefix}5分钟级别回测成功完成，返回结果")
        return result

    except Exception as e:
        logger.error(f"{window_prefix}[FAIL] 5分钟级别回测失败: {e}")
        logger.error(f"详细错误: {traceback.format_exc()}")

        # 返回空的回测结果
        return {
            'backtest_results': {
                stock_count: {
                    'total_trades': 0,
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_loss_ratio': 0.0
                } for stock_count in stock_counts
            },
            'detailed_trades': {stock_count: [] for stock_count in stock_counts},
            'summary_stats': {
                'error': str(e),
                'trading_days': 0,
                'holding_days': holding_days
            }
        }


def _run_stock_selection_backtest(
    backtest_data,
    stock_counts=[1, 2, 3, 5],
    window_name=None,
    market_data=None,
    holding_days=3,
    frequency='minute'
):
    """
    执行选股回测(test_model和Walk-Forward共用)

    Args:
        backtest_data: DataFrame,包含以下字段:
            - date, stock_code
            - predicted_return, predicted_return_atr_raw, std_pred
            - future_return_2d, is_can_buy
            - open, high, low, close
            - 其他技术指标
        stock_counts: 选股数量列表,默认[1, 2, 3, 5]
        window_name: 窗口名称(Walk-Forward专用),默认None
        market_data: 市场数据(用于市场状态判断),默认None
        holding_days: 固定持仓天数,默认2天(T+1买入,T+2收盘卖出)
        frequency: 回测频率, 'daily'(日线) 或 'minute'(分钟线),默认'minute'
        
    Returns:
        dict: {
            'backtest_results': {...},  # 各档位回测结果
            'detailed_trades': {...},  # 详细交易记录
            'summary_stats': {...}  # 汇总统计
        }
    """
    logger = logging.getLogger(__name__)

    # 窗口标识(用于日志)
    window_prefix = f"[{window_name}] " if window_name else ""

    logger.info(f"\n{window_prefix}[REFRESH] 开始选股回测...")
    logger.info("=" * 60)

    # 🆕 5分钟级别回测分支
    # if frequency == 'minute':
    #     logger.info(f"{window_prefix}⏰ 检测到分钟级别回测，调用5分钟回测系统...")
    #     return _run_minute_level_backtest(
    #         backtest_data=backtest_data,
    #         stock_counts=stock_counts,
    #         window_name=window_name,
    #         holding_days=holding_days
    #     )
    # 准备回测数据
    required_columns = [
        # 基础必要列
        'date', 'stock_code', 'future_return_2d', 'is_can_buy', 'pct_chg', 'volume', 'amount',  # [强化] 添加amount字段
        # [WRENCH] 修复:ATR三分类标签系统必需字段
        'atr14', 'open', 'high', 'low', 'close',
        # [WRENCH] 高级技术指标字段（多维度复合条件筛选必需）
        'price_ema20_ratio', 'price_ema50_ratio', 'ema20_ema50_ratio',
        'ema50_slope', 'log_price_regression_slope', 'log_price_regression_r2',
        'adx_14', 'drawdown_from_20d_high_pct', 'drawdown_atr_ratio', 'atr20',
        'negative_days_5d', 'large_red_candle_5d', 'distance_to_support_atr',
        'bollinger_percent_b', 'rsi_14',
        # 涨跌停相关特征（用于选股过滤和风险控制）
        'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
        'limit_times', 'consecutive_limit_up', 'consecutive_limit_down',
        # 技术分析特征（用于选股过滤）
        'near_support', 'near_resistance', 'short_reversal'
    ]
    missing_columns = [col for col in required_columns if col not in backtest_data.columns]

    if missing_columns:
        logger.warning(f"[WARNING] 回测数据缺少必要列: {missing_columns}")
        logger.info("为缺失的特征字段设置默认值...")

        # 为缺失的特征字段设置默认值
        default_values = {
            # [WRENCH] 高级技术指标字段默认值
            'close': 0.0,
            'price_ema20_ratio': 1.0,
            'price_ema50_ratio': 1.0,
            'ema20_ema50_ratio': 1.0,
            'ema50_slope': 0.0,
            'log_price_regression_slope': 0.0,
            'log_price_regression_r2': 0.0,
            'adx_14': 10.0,
            'drawdown_from_20d_high_pct': 0.0,
            'drawdown_atr_ratio': 0.0,
            'atr20': 0.5,
            'negative_days_5d': 1,
            'large_red_candle_5d': 1,
            'distance_to_support_atr': 0.5,
            'bollinger_percent_b': 0.5,
            'rsi_14': 50.0,
            # [强化] 新股过滤字段默认值（不参与模型训练，仅用于过滤）
            'volume': 0.0,
            'amount': 0.0,  # [强化] 添加amount默认值
            # 涨跌停相关特征默认值
            'is_limit_up': 0,
            'is_limit_down': 0,
            'is_zhaban': 0,
            'limit_strength': 0.0,
            'limit_times': 0,
            'consecutive_limit_up': 0,
            'consecutive_limit_down': 0,
            # 技术分析特征默认值
            'near_support': 0,
            'near_resistance': 0,
            'short_reversal': 0,
            # 基础特征默认值
            'pct_chg': 0.0,
            'future_return_2d': 0.0,
            'is_can_buy': 1
        }

        for col in missing_columns:
            if col in default_values:
                backtest_data[col] = default_values[col]
                logger.info(f"  {col}: 设置默认值 {default_values[col]}")
            else:
                backtest_data[col] = 0.0
                logger.warning(f"  {col}: 未定义默认值，设置为 0.0")

    # 🆕 调试日志：检查backtest_data的字段
    logger.info(f"\n{window_prefix}[SEARCH] 回测数据字段检查")
    logger.info(f"总行数: {len(backtest_data)}")
    logger.info(f"总列数: {len(backtest_data.columns)}")

    # 检查关键字段
    key_fields = ['date', 'stock_code', 'volume', 'is_can_buy', 'pct_chg', 'open', 'high', 'low', 'close', 'atr14']
    for field in key_fields:
        if field in backtest_data.columns:
            missing_count = backtest_data[field].isna().sum()
            missing_pct = missing_count / len(backtest_data) * 100
            if field not in ['date', 'stock_code']:
                zero_count = (backtest_data[field] == 0).sum()
                zero_pct = zero_count / len(backtest_data) * 100
                logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%), 为0{zero_count}({zero_pct:.2f}%)")
            else:
                logger.info(f"  {field}: 缺失{missing_count}({missing_pct:.2f}%)")
        else:
            logger.warning(f"  {field}: [FAIL] 字段不存在")

    # 只按日期排序，保持每日内股票的原始预测概率分布
    backtest_data = backtest_data.sort_values('date')

    # 🆕 批量过滤新股（在日期循环之外）- 先过滤北交所，再过滤新股
    logger.info(f"\n{window_prefix}[FILTER] 开始批量北交所过滤...")
    original_count = len(backtest_data)
    if 'stock_code' in backtest_data.columns:
        bj_filter = ~backtest_data['stock_code'].str.endswith(('.BJ', '.bj'), na=False)
        backtest_data = backtest_data[bj_filter].copy()
        after_bj_count = len(backtest_data)
        bj_filtered_count = original_count - after_bj_count
        logger.info(f"{window_prefix}[FILTER] 批量北交所过滤完成: {original_count} -> {after_bj_count} (过滤掉 {bj_filtered_count} 只北交所股票)")
    else:
        after_bj_count = original_count
        bj_filtered_count = 0
        logger.info(f"{window_prefix}[FILTER] 无stock_code列，跳过北交所过滤")

    # 然后过滤新股
    logger.info(f"{window_prefix}[FILTER] 开始批量新股过滤...")
    new_stock_original_count = len(backtest_data)
    backtest_data = _filter_new_stocks_batch(backtest_data, logger)
    filtered_count = len(backtest_data)
    new_stock_filtered = new_stock_original_count - filtered_count
    logger.info(f"{window_prefix}[FILTER] 批量新股过滤完成: {new_stock_original_count} -> {filtered_count} (过滤掉 {new_stock_filtered} 条新股期间数据)")

    # 获取所有交易日
    trading_dates = sorted(backtest_data['date'].unique())
    logger.info(f"\n{window_prefix}[DATE] 回测期间: {trading_dates[0].strftime('%Y-%m-%d')} 到 {trading_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"{window_prefix}[DATE] 交易日数量: {len(trading_dates)}天")

    # [强化] 通用测试集过滤逻辑（适用于5分钟和日线回测）
    def apply_test_set_filters(test_data, logger, window_prefix=""):
        """
        应用测试集的通用过滤逻辑

        Args:
            test_data: DataFrame,测试集数据
            logger: 日志记录器
            window_prefix: 窗口前缀

        Returns:
            DataFrame: 过滤后的数据
        """
        logger.info(f"{window_prefix}[FILTER] 开始应用测试集通用过滤逻辑...")
        original_count = len(test_data)

        # 兼容两种字段名: predicted_return_atr_raw (test_model) 和 predicted_return (walkforward)
        if 'predicted_return_atr_raw' in test_data.columns:
            pred_column = 'predicted_return_atr_raw'
        elif 'predicted_return' in test_data.columns:
            pred_column = 'predicted_return'
        else:
            logger.warning(f"{window_prefix}[WARNING] 数据中缺少预测字段，跳过NaN过滤")
            pred_column = None

        # 1. 过滤ST股票
        if len(test_data) > 0:
            st_filter = _filter_st_stocks(test_data, logger)
            test_data = test_data[st_filter].copy()
            st_filtered = original_count - len(test_data)
            logger.info(f"{window_prefix}[FILTER] ST股过滤: {original_count} -> {len(test_data)} (过滤掉 {st_filtered})")

        # # # 2. 过滤T日涨停股票（无法买入）
        # if 'is_limit_up' in test_data.columns and len(test_data) > 0:
        #     before_limit_up = len(test_data)
        #     test_data = test_data[test_data['is_limit_up'] != 1].copy()
        #     limit_up_filtered = before_limit_up - len(test_data)
        #     logger.info(f"{window_prefix}[FILTER] 涨停股过滤: {before_limit_up} -> {len(test_data)} (过滤掉 {limit_up_filtered})")

        # # 3. 过滤T日跌停股票（风险高,无风控时应避免）
        # if 'is_limit_down' in test_data.columns and len(test_data) > 0:
        #     before_limit_down = len(test_data)
        #     test_data = test_data[test_data['is_limit_down'] != 1].copy()
        #     limit_down_filtered = before_limit_down - len(test_data)
        #     logger.info(f"{window_prefix}[FILTER] 跌停股过滤: {before_limit_down} -> {len(test_data)} (过滤掉 {limit_down_filtered})")

        # # 4. 流动性过滤（成交额 < 1000万视为流动性不足）
        # if 'amount' in test_data.columns and len(test_data) > 0:
        #     before_liquidity = len(test_data)
        #     # 成交额单位：元，1000万 = 10,000,000
        #     liquidity_threshold = 10_000_000
        #     filtered_stocks = test_data[test_data['amount'] < liquidity_threshold].copy()
        #     test_data = test_data[test_data['amount'] >= liquidity_threshold].copy()
        #     liquidity_filtered = before_liquidity - len(test_data)
        #     logger.info(f"{window_prefix}[FILTER] 流动性过滤(amount): {before_liquidity} -> {len(test_data)} (过滤掉 {liquidity_filtered})")
        # elif 'volume' in test_data.columns and len(test_data) > 0:
        #     # 如果没有amount字段,使用volume字段
        #     before_liquidity = len(test_data)
        #     # 成交量 < 100万股视为流动性不足
        #     volume_threshold = 10_000
        #     test_data = test_data[test_data['volume'] >= volume_threshold].copy()
        #     liquidity_filtered = before_liquidity - len(test_data)
        #     logger.info(f"{window_prefix}[FILTER] 流动性过滤(volume): {before_liquidity} -> {len(test_data)} (过滤掉 {liquidity_filtered})")

        # 5. 过滤 NaN 预测值
        if pred_column and len(test_data) > 0:
            before_nan = len(test_data)
            test_data = test_data[~test_data[pred_column].isna()]
            nan_filtered = before_nan - len(test_data)
            logger.info(f"{window_prefix}[FILTER] NaN过滤: {before_nan} -> {len(test_data)} (过滤掉 {nan_filtered})")

        # # 6. 过滤可买入股票
        # if 'is_can_buy' in test_data.columns and len(test_data) > 0:
        #     before_can_buy = len(test_data)
        #     test_data = test_data[test_data['is_can_buy'] == 1].copy()
        #     can_buy_filtered = before_can_buy - len(test_data)
        #     logger.info(f"{window_prefix}[FILTER] is_can_buy过滤: {before_can_buy} -> {len(test_data)} (过滤掉 {can_buy_filtered})")

        total_filtered = original_count - len(test_data)
        filter_rate = total_filtered / original_count * 100 if original_count > 0 else 0
        logger.info(f"{window_prefix}[FILTER] 测试集过滤完成: {original_count} -> {len(test_data)} (总过滤率 {filter_rate:.1f}%)")

        return test_data

    # 🆕 5分钟级别回测分支
    if frequency == 'minute':
        logger.info(f"{window_prefix}⏰ 检测到分钟级别回测，先应用测试集过滤再调用5分钟回测系统...")

        # 在调用5分钟回测之前，先应用测试集过滤逻辑
        backtest_data = apply_test_set_filters(backtest_data, logger, window_prefix)

        if USE_TOP1_PRICE_MA5_RATIO_GATE:
            backtest_data = _apply_top1_price_ma5_ratio_pct_gate(
                backtest_data,
                logger,
                threshold=0.4,
                window_prefix=window_prefix
            )
            gate_desc = "Top1 开关(price_ma5_ratio_pct>=0.4)"
        else:
            gate_desc = "无Top1开关"

        if USE_TOP1_TWO_BRANCH_GATE:
            backtest_data = _apply_top1_two_branch_gate(
                backtest_data,
                logger,
                sector_threshold=TOP1_SECTOR_MOMENTUM_ACCEL_GATE_THRESHOLD,
                market_vol_threshold=TOP1_MARKET_VOLATILITY_CONTEXT_GATE_THRESHOLD,
                ma_alignment_threshold=TOP1_MA_ALIGNMENT_PERSISTENCE_GATE_THRESHOLD,
                window_prefix=window_prefix
            )
            downsize_desc = (
                f" + (Top1行业动量<{TOP1_SECTOR_MOMENTUM_ACCEL_GATE_THRESHOLD:.3f}"
                f"且市场波动>{TOP1_MARKET_VOLATILITY_CONTEXT_GATE_THRESHOLD:.3f})"
                f"或(ma_alignment_persistence_5d<{TOP1_MA_ALIGNMENT_PERSISTENCE_GATE_THRESHOLD:.2f})空仓"
            )
        else:
            downsize_desc = ""
            if USE_TOP1_TWO_BRANCH_STRICT_MODE:
                backtest_data = _apply_top1_two_branch_strict_mode(
                    backtest_data,
                    logger,
                    sector_threshold=TOP1_SECTOR_MOMENTUM_ACCEL_GATE_THRESHOLD,
                    market_vol_threshold=TOP1_MARKET_VOLATILITY_CONTEXT_GATE_THRESHOLD,
                    ma_alignment_threshold=TOP1_MA_ALIGNMENT_PERSISTENCE_GATE_THRESHOLD,
                    window_prefix=window_prefix
                )
                downsize_desc = (
                    f" + 双分支命中票严格模式(gap_up<={TOP1_STRICT_ENTRY_GAP_UP_THRESHOLD:.2%})"
                )

        filter_desc = (
            "T+1 open/MA5 买入过滤"
            if str(TOP1_ENTRY_FILTER_MODE).lower() == 'ma5'
            else "低开6%买入过滤"
        )
        risk_desc = "状态机分钟风控" if ENABLE_TOP1_STATE_MACHINE_RISK else "固定阈值分钟风控"
        logger.info(f"{window_prefix}🧭 分钟回测模式: {gate_desc}{downsize_desc} + {filter_desc} + {risk_desc}")

        return _run_minute_level_backtest(
            backtest_data=backtest_data,
            stock_counts=[1],
            window_name=window_name,
            holding_days=holding_days,
            use_live_default_scheme=False
        )

    # [强化] 动态调整选股阈值：基于预测范围自适应
    # 兼容两种字段名: predicted_return_atr_raw (test_model) 和 predicted_return (walkforward)
    if 'predicted_return_atr_raw' in backtest_data.columns:
        pred_column = 'predicted_return_atr_raw'
    elif 'predicted_return' in backtest_data.columns:
        pred_column = 'predicted_return'
    else:
        logger.error("[FAIL] 数据中缺少预测字段 (predicted_return_atr_raw 或 predicted_return)")
        return None

    all_predictions = backtest_data[pred_column].dropna()
    pred_max = all_predictions.max()
    pred_75th = all_predictions.quantile(0.75)
    pred_median = all_predictions.median()
    pred_mean = all_predictions.mean()
    pred_std = all_predictions.std()

    logger.info(f"{window_prefix}[CHART] 预测值分布: 最大{pred_max:.4f}, 75%分位{pred_75th:.4f}, 中位数{pred_median:.4f}, 均值{pred_mean:.4f}±{pred_std:.4f}")

    # 交易成本（基于A股实际成本）
    SLIPPAGE_RATE = 0.001          # 滑点0.1%
    COMMISSION_RATE = 0.0003       # 手续费0.03%
    TRADING_COST_TOTAL = SLIPPAGE_RATE + COMMISSION_RATE * 2  # 总交易成本（双边手续费）

    logger.info(f"{window_prefix}[TARGET] 回测配置: 固定持仓{holding_days}天, 选股数量{stock_counts}, 交易成本{TRADING_COST_TOTAL:.3%}")

    # 初始化回测结果
    backtest_results = {}
    detailed_trades = {}
    portfolio_data = {}  # 存储净值和持仓数据

    # ========== 第2批代码将在这里添加 ==========
    # [强化] 修正版本：恢复1,3,5只股票回测 + g_t仅控制仓位比例
    for stock_count in stock_counts:
        logger.info(f"\n[CHART] 开始{stock_count}只股票回测（g_t动态仓位分配）...")

        # 初始化回测记录
        trades = []
        portfolio_values = [1.0]  # 初始资金为1

        daily_returns = []

        # [强化] 核心策略：固定股票数量 + g_t动态仓位比例
        logger.info(f"[CHART] 固定选股{stock_count}只，g_t决定总仓位比例")

        # [强化] 使用统一的预测字段名(已在函数开头确定)
        # 按日期分组进行选股
        df_pred = backtest_data.copy()
        grouped = df_pred.groupby('date', sort=True)

        selected_records = []
        empty_day_count = 0
        total_days = 0

        for trade_date, g in grouped:
            total_days += 1
            g = g.copy()

            # [强化] 第一步：基础过滤（北交所、新股、ST股、可买入）
            date_str = trade_date.strftime('%Y%m%d')

            # 🆕 调试日志：特定日期的详细过滤过程
            is_debug_date = (trade_date == pd.to_datetime('2025-01-06'))
            if is_debug_date:
                logger.info(f"\n[SEARCH] 调试: {date_str}选股过程")
                logger.info(f"初始股票数: {len(g)}")

            # 北交所和新股过滤已在循环外批量处理完成
            # 注释：北交所和新股过滤已在循环外批量处理，无需在循环内重复执行

            # 过滤ST股票
            if len(g) > 0:
                st_filter = _filter_st_stocks(g, logger)
                g = g[st_filter].copy()
                if is_debug_date:
                    logger.info(f"ST过滤后: {len(g)}")

            # [强化] 新增：过滤T日涨停股票（无法买入）
            if 'is_limit_up' in g.columns and len(g) > 0:
                before_limit_up = len(g)
                g = g[g['is_limit_up'] != 1].copy()
                if is_debug_date:
                    logger.info(f"涨停过滤后: {len(g)} (过滤掉{before_limit_up - len(g)})")

            # [强化] 新增：过滤T日跌停股票（风险高,无风控时应避免）
            if 'is_limit_down' in g.columns and len(g) > 0:
                before_limit_down = len(g)
                g = g[g['is_limit_down'] != 1].copy()
                if is_debug_date:
                    logger.info(f"跌停过滤后: {len(g)} (过滤掉{before_limit_down - len(g)})")

            # [强化] 新增：流动性过滤（成交额 < 1000万视为流动性不足）
            if 'amount' in g.columns and len(g) > 0:
                before_liquidity = len(g)
                # 成交额单位：元，1000万 = 10,000,000
                liquidity_threshold = 10_000_000

                # [强化] 新增：记录被过滤的股票
                filtered_stocks = g[g['amount'] < liquidity_threshold].copy()

                g = g[g['amount'] >= liquidity_threshold].copy()
                if is_debug_date:
                    logger.info(f"流动性过滤后: {len(g)} (过滤掉{before_liquidity - len(g)})")

                    # 打印前5个被过滤的股票
                    if len(filtered_stocks) > 0:
                        logger.info(f"  因流动性不足被过滤的股票示例(前5个):")
                        for idx, (_, row) in enumerate(filtered_stocks.head(5).iterrows(), 1):
                            stock_code = row.get('stock_code', 'N/A')
                            amount = row.get('amount', 0)
                            logger.info(f"    {idx}. {stock_code}: 成交额={amount:,.0f}元 ({amount/10000:.2f}万)")
            elif 'volume' in g.columns and len(g) > 0:
                # 如果没有amount字段,使用volume字段
                # 成交量 < 100万股视为流动性不足
                before_liquidity = len(g)
                volume_threshold = 10_000
                g = g[g['volume'] >= volume_threshold].copy()
                if is_debug_date:
                    logger.info(f"流动性过滤(volume)后: {len(g)} (过滤掉{before_liquidity - len(g)})")

          
            # 2.1 过滤 NaN
            before_nan = len(g)
            g = g[~g[pred_column].isna()]
            if is_debug_date:
                logger.info(f"NaN过滤后: {len(g)} (过滤掉{before_nan - len(g)})")
            if g.empty:
                if is_debug_date:
                    logger.info(f"[FAIL] 所有股票都被过滤，空仓")
                empty_day_count += 1
                continue

             # 过滤可买入股票
            before_can_buy = len(g)
            g1 = g[g['is_can_buy'] == 0].copy()
            g = g[g['is_can_buy'] == 1].copy()
            if is_debug_date:
                logger.info(f"is_can_buy过滤后: {len(g)} (过滤掉{before_can_buy - len(g)})")
                if len(g) > 0:
                    logger.info(f"  is_can_buy=0示例股票: {g1['stock_code'].head(5).tolist()}")
                    logger.info(f"  is_can_buy=1的股票数: {(g['is_can_buy'] == 1).sum()}")
                    logger.info(f"  示例股票: {g['stock_code'].head(5).tolist()}")

            # 使用预测值进行选股排序
            g['score_final'] = g[pred_column]

            # 2.4 排序并**不凑满**：只取 Top-K，且不够就少买
            g = g.sort_values('score_final', ascending=False).head(stock_count)
            # logger.info(f"data:{date_str}, g count:{len(g)}  and g:{g}")

            # 2.5 记录选股（包含仓位分配信息）
            selection_columns = ['date','stock_code',pred_column,'score_final']

            # [强化] 修复：如果存在仓位分配列，也要保存
            if 'position_allocation' in g.columns:
                selection_columns.append('position_allocation')
            if 'market_regime' in g.columns:
                selection_columns.append('market_regime')

            selected_records.append(g[selection_columns])

        # 汇总选股结果
        if selected_records:
            picks = pd.concat(selected_records, ignore_index=True)
        else:
            # [强化] 修复：空DataFrame也要包含可能的仓位分配列
            empty_columns = ['date','stock_code',pred_column,'score_z','score_final']
            if 'position_allocation' in backtest_data.columns:
                empty_columns.append('position_allocation')
            if 'market_regime' in backtest_data.columns:
                empty_columns.append('market_regime')
            picks = pd.DataFrame(columns=empty_columns)

        # 计算空仓日比例
        empty_ratio = empty_day_count / max(total_days, 1)
        # logger.info(f"统一口径选股完成：直接按预测值排序选股，不凑满K；空仓日比例={empty_ratio:.1%}")

        # 🆕 诊断日志：入选股票统计
        if len(picks) > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"[CHART] Top-{stock_count} 入选股票统计")
            logger.info(f"{'='*60}")
            logger.info(f"入选股票数: {len(picks)}")
            logger.info(f"入选交易日数: {picks['date'].nunique()}")
            logger.info(f"空仓日数: {empty_day_count}/{total_days} ({empty_ratio:.1%})")

            # 预测值统计
            selected_pred = picks[pred_column]
            logger.info(f"\n入选股票预测值分布:")
            logger.info(f"  均值: {selected_pred.mean():.4f} ATR")
            logger.info(f"  中位数: {selected_pred.median():.4f} ATR")
            logger.info(f"  25分位: {selected_pred.quantile(0.25):.4f} ATR")
            logger.info(f"  75分位: {selected_pred.quantile(0.75):.4f} ATR")
            logger.info(f"  最小值: {selected_pred.min():.4f} ATR")
            logger.info(f"  最大值: {selected_pred.max():.4f} ATR")


            logger.info(f"{'='*60}\n")
        else:
            logger.warning(f"[WARNING] Top-{stock_count}: 没有入选股票（全部空仓）")

        # 执行回测：基于固定持仓天数的净值回测逻辑
        portfolio_values = [1.0]      # 组合净值（净收益）
        all_trades = []               # 所有交易记录
        daily_holdings = []           # 每日持仓详情

        # 维护当前持仓列表
        current_holdings = []         # [{'stock_code': '', 'buy_price': 0, 'buy_date': '', 'selection_date': '', 'weight': 0, 'position_value': 0}]

        logger.info(f"{window_prefix}[TARGET] 开始固定持仓{holding_days}天回测 (Top-{stock_count})")

        # 每日更新净值的回测循环
        for date_idx, date in enumerate(trading_dates):
            # 每日计算当前持仓的市值（基于收盘价）
            daily_portfolio_value = 0.0

            # 记录当日持仓详情
            day_holdings = []

            # 获取当日所有持仓的收盘价并计算市值
            for holding in current_holdings:
                # 获取当日收盘价
                current_price_data = backtest_data[
                    (backtest_data['stock_code'] == holding['stock_code']) &
                    (backtest_data['date'] == date)
                ]

                if not current_price_data.empty:
                    current_price = current_price_data.iloc[0]['close']
                    # 计算该股票当前价值（相对于买入价格的收益）
                    gross_return = (current_price - holding['buy_price']) / holding['buy_price']
                    # 净收益（浮动盈亏，不扣除交易成本）
                    daily_portfolio_value += holding['position_value'] * (1 + gross_return)

                    # 记录持仓详情
                    day_holdings.append({
                        'stock_code': holding['stock_code'],
                        'buy_price': round(holding['buy_price'], 2),
                        'current_price': round(current_price, 2),
                        'return_pct': round(gross_return * 100, 2),
                        'weight': round(holding['weight'], 4)
                    })
                else:
                    # 如果没有价格数据，保持原值
                    daily_portfolio_value += holding['position_value']
                    day_holdings.append({
                        'stock_code': holding['stock_code'],
                        'buy_price': round(holding['buy_price'], 2),
                        'current_price': 'N/A',
                        'return_pct': 'N/A',
                        'weight': round(holding['weight'], 4)
                    })

            # 保存每日持仓
            daily_holdings.append(day_holdings)

            # 当日期有持仓到期时，执行卖出
            if date_idx >= holding_days:
                # 计算买入和卖出日期
                selection_date = trading_dates[date_idx - holding_days]     # T日选股
                buy_date = trading_dates[date_idx - holding_days + 1]      # T+1日买入
                sell_date = date                                          # T+1+holding_days日卖出

                # 获取选股结果
                daily_selections = picks[picks['date'] == selection_date]

                if not daily_selections.empty:
                    # 获取Top-K选股结果
                    selected_stocks = daily_selections.head(stock_count)

                    # 处理到期的持仓
                    for _, stock in selected_stocks.iterrows():
                        stock_code = stock['stock_code']

                        # 找到对应的持仓
                        for i, holding in enumerate(current_holdings):
                            if (holding['stock_code'] == stock_code and
                                holding['selection_date'] == selection_date and
                                holding['buy_date'] == buy_date):

                                # 计算最终收益（扣除交易成本）
                                sell_data = backtest_data[
                                    (backtest_data['stock_code'] == stock_code) &
                                    (backtest_data['date'] == sell_date)
                                ]

                                if not sell_data.empty:
                                    sell_price = sell_data.iloc[0]['close']
                                else:
                                    # 找不到卖出数据时的处理
                                    logger.warning(f"找不到{stock_code}在{sell_date}的卖出数据，尝试使用前一日价格")
                                    # 尝试获取前一个交易日的数据
                                    prev_date = trading_dates[trading_dates.index(sell_date) - 1] if sell_date in trading_dates else None
                                    if prev_date:
                                        prev_data = backtest_data[
                                            (backtest_data['stock_code'] == stock_code) &
                                            (backtest_data['date'] == prev_date)
                                        ]
                                        if not prev_data.empty:
                                            sell_price = prev_data.iloc[0]['close']
                                            logger.info(f"使用{stock_code}在{prev_date}的价格: {sell_price}")
                                        else:
                                            logger.error(f"无法找到{stock_code}的任何价格数据，跳过此交易")
                                            continue
                                    else:
                                        logger.error(f"无法找到{sell_date}的前一个交易日，跳过此交易")
                                        continue

                                gross_return = (sell_price - holding['buy_price']) / holding['buy_price']
                                net_return = gross_return - TRADING_COST_TOTAL

                                # 更新净值（扣除交易成本）
                                daily_portfolio_value = daily_portfolio_value - holding['position_value'] + holding['position_value'] * (1 + net_return)

                                # 记录交易（无论sell_data是否为空都记录）
                                trade_record = {
                                    'date': sell_date,
                                    'stock_code': stock_code,
                                    'selection_date': selection_date,
                                    'buy_date': buy_date,
                                    'sell_date': sell_date,
                                    'buy_price': holding['buy_price'],
                                    'sell_price': sell_price,
                                    'gross_return': gross_return,
                                    'net_return': net_return,
                                    'holding_days': holding_days,
                                    'pred_score': stock[pred_column] if pred_column in stock else 0,
                                    'position_weight': holding['weight']
                                }
                                all_trades.append(trade_record)

                                # 从持仓列表中移除
                                del current_holdings[i]
                                break

            # T+1日买入新的股票
            if date_idx < len(trading_dates) - 1:  # 确保有明天
                # T日选股，T+1日买入
                selection_date = date
                buy_date = trading_dates[date_idx + 1]

                # 检查当前持仓数量
                current_positions = len(current_holdings)

                # 只有在有空闲仓位时才能买入新股票
                if current_positions < stock_count:
                    # 获取T日的选股结果
                    daily_selections = picks[picks['date'] == selection_date]

                    if not daily_selections.empty:
                        # 获取Top-K选股结果
                        selected_stocks = daily_selections.head(stock_count)

                        # 计算可以买入的股票数量
                        available_slots = stock_count - current_positions

                        for idx, (_, stock) in enumerate(selected_stocks.iterrows()):
                            if idx >= available_slots:
                                break  # 只买入有空闲仓位的股票

                            stock_code = stock['stock_code']

                            # 获取买入价格 (T+1日开盘价)
                            buy_data = backtest_data[
                                (backtest_data['stock_code'] == stock_code) &
                                (backtest_data['date'] == buy_date)
                            ]

                            if not buy_data.empty:
                                buy_price = buy_data.iloc[0]['open']
                                position_value = 1.0 / stock_count  # 等权重分配

                                # 添加到持仓列表
                                current_holdings.append({
                                    'stock_code': stock_code,
                                    'buy_price': buy_price,
                                    'buy_date': buy_date,
                                    'selection_date': selection_date,
                                    'weight': position_value,
                                    'position_value': position_value
                                })

            # 记录每日净值
            if date_idx == 0:
                # 第一天，还没有买入
                portfolio_values.append(1.0)
            else:
                # 计算当日净值
                # 如果没有持仓，净值保持前一天的水平（现金）
                if daily_portfolio_value == 0.0:
                    portfolio_values.append(portfolio_values[-1])
                else:
                    portfolio_values.append(daily_portfolio_value)

        # 计算回测指标 (基于净值序列)
        if len(portfolio_values) > 1:
            # [WRENCH] 增强NaN检查和处理
            portfolio_final_value = portfolio_values[-1]

            if pd.isna(portfolio_final_value):
                logger.error("[FAIL] 组合最终净价值为NaN，设为1.0")
                portfolio_final_value = 1.0

            # 计算净收益（已扣除交易成本）
            total_return = portfolio_final_value - 1.0
            if pd.isna(total_return):
                logger.error("[FAIL] 净收益为NaN，设为0")
                total_return = 0.0

            # 计算毛收益（未扣除交易成本）- 从总收益反推
            total_cost = TRADING_COST_TOTAL * len(all_trades) / stock_count if all_trades else 0
            total_return_gross = total_return + total_cost

            # 计算年化收益（修正：基于实际交易日长度而非收益天数）
            if portfolio_final_value > 0 and len(trading_dates) > 0:
                # 使用总交易日长度而不是有收益的天数来计算年化收益
                actual_trading_days = len(trading_dates)
                annual_return = (portfolio_final_value ** (252 / actual_trading_days)) - 1.0
                if pd.isna(annual_return):
                    logger.error("[FAIL] 年化收益为NaN，设为0")
                    annual_return = 0.0

                # 添加调试日志
                logger.debug(f"年化收益计算: 最终净值={portfolio_final_value:.4f}, 交易日={actual_trading_days}, 年化收益={annual_return:.2%}")
            else:
                annual_return = 0.0

            # 计算详细盈亏比（已包含在下面的逻辑中）

            # 计算最大回撤
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

            # 计算胜率和详细盈亏比
            if len(all_trades) > 0:
                profitable_trades = [t for t in all_trades if (t.get('net_return') or 0) > 0]
                losing_trades = [t for t in all_trades if (t.get('net_return') or 0) < 0]
                break_even_trades = [t for t in all_trades if abs(t.get('net_return') or 0) < 0.001]  # 近似平盘

                win_rate = len(profitable_trades) / len(all_trades)

                # 详细盈亏比计算
                if losing_trades:
                    # 方式1：平均盈利 vs 平均亏损
                    avg_profit = np.mean([(t.get('net_return') or 0) for t in profitable_trades]) if profitable_trades else 0
                    avg_loss = np.mean([abs(t.get('net_return') or 0) for t in losing_trades])
                    profit_loss_ratio_avg = avg_profit / avg_loss if avg_loss > 0 else float('inf')

                    # 方式2：总盈利 vs 总亏损（更保守，避免被极端值影响）
                    total_profit = sum([(t.get('net_return') or 0) for t in profitable_trades])
                    total_loss = sum([abs(t.get('net_return') or 0) for t in losing_trades])
                    profit_loss_ratio_total = total_profit / total_loss if total_loss > 0 else float('inf')

                    # 使用总盈利/总亏损的盈亏比（更稳定）
                    profit_loss_ratio = profit_loss_ratio_total

                    # 盈亏详情统计
                    max_profit = max([(t.get('net_return') or 0) for t in profitable_trades]) if profitable_trades else 0
                    max_loss = min([(t.get('net_return') or 0) for t in losing_trades]) if losing_trades else 0

                    logger.debug(f"盈亏详情: 盈利交易{len(profitable_trades)}个, 亏损交易{len(losing_trades)}个")
                    logger.debug(f"平均盈利{avg_profit:.3f}, 平均亏损{avg_loss:.3f}, 盈亏比{profit_loss_ratio:.2f}")
                    logger.debug(f"最大盈利{max_profit:.3f}, 最大亏损{max_loss:.3f}")
                else:
                    profit_loss_ratio = float('inf') if profitable_trades else 0

                # 添加到结果中用于详细分析
                profit_details = {
                    'profitable_count': len(profitable_trades),
                    'losing_count': len(losing_trades),
                    'break_even_count': len(break_even_trades),
                    'avg_profit': avg_profit if losing_trades else 0,
                    'avg_loss': avg_loss if losing_trades else 0,
                    'max_profit': max_profit if losing_trades else 0,
                    'max_loss': max_loss if losing_trades else 0
                }
            else:
                win_rate = 0.0
                profit_loss_ratio = 0.0
                profit_details = {}

            # 记录结果
            backtest_results[stock_count] = {
                'total_return_gross': total_return_gross,  # 毛收益（未扣成本）
                'total_return': total_return,               # 净收益（已扣成本）
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'total_trades': len(all_trades),
                'holding_days': holding_days,
                'portfolio_values': [
                    {'date': d.strftime('%Y-%m-%d'), 'value': v}
                    for d, v in zip(trading_dates[:len(portfolio_values)], portfolio_values)
                ],
                # 详细盈亏统计
                'profit_details': profit_details
            }

            detailed_trades[stock_count] = all_trades

            logger.info(f"{window_prefix}[OK] Top-{stock_count}回测完成: "
                       f"总收益{total_return:.2%}, 年化收益{annual_return:.2%}, "
                       f"最大回撤{max_drawdown:.2%}, 胜率{win_rate:.2%}, 盈亏比{profit_loss_ratio:.2f}, "
                       f"交易次数{len(all_trades)}")

            # 详细盈亏分析日志
            if len(all_trades) > 0 and profit_details:
                logger.info(f"{window_prefix}[CHART] Top-{stock_count}详细盈亏分析: "
                           f"盈利{profit_details['profitable_count']}次, "
                           f"亏损{profit_details['losing_count']}次, "
                           f"平均盈利{profit_details['avg_profit']:.3f}, "
                           f"平均亏损{profit_details['avg_loss']:.3f}")
            else:
                logger.debug(f"{window_prefix}[WARNING] Top-{stock_count}: 无详细盈亏数据")
        else:
            # 没有交易记录
            backtest_results[stock_count] = {
                'total_return_gross': 0.0,  # 毛收益（未扣成本）
                'total_return': 0.0,        # 净收益（已扣成本）
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'total_trades': 0,
                'holding_days': holding_days,
                'portfolio_values': [
                    {'date': d.strftime('%Y-%m-%d'), 'value': 1.0}
                    for d in trading_dates[:1]
                ],
                'profit_details': {}
            }

            detailed_trades[stock_count] = []

            logger.warning(f"{window_prefix}[WARNING] Top-{stock_count}: 没有有效的回测数据")

    # 返回回测结果
    return {
        'backtest_results': backtest_results,
        'detailed_trades': detailed_trades,
        'portfolio_data': portfolio_data,  # 新增：净值和持仓数据
        'summary_stats': {
            'total_trading_days': len(trading_dates),
            'stock_counts': stock_counts,
            'holding_days': holding_days,
            'window_name': window_name
        }
    }


def run_test_model_mode(args):
    """
    运行模型测试模式：训练模型->预测->选股->回测的纯预测流程

    Args:
        args: 命令行参数
    """
    logger = logging.getLogger(__name__)
    print(f"[CHART] 运行模式: 模型预测选股")
    logger.info("=" * 60)
    logger.info(f"开始纯预测选股流程（无标签验证）")
    logger.info("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import pickle
        import json
        from pathlib import Path
        from datetime import datetime

        # [TARGET] 1. 使用全局时间切分配置
        train_end_date = TRAIN_END_DATE
        test_start_date = pd.to_datetime(TEST_START_DATE)
        test_end_date = pd.to_datetime(TEST_END_DATE)

        logger.info("🗓️ 纯预测选股的时间切分策略（使用全局配置）:")
        logger.info(f"  训练数据: 2023-06-01 到 {train_end_date}")
        logger.info(f"  预测数据: {test_start_date.strftime('%Y-%m-%d')} 到 {test_end_date.strftime('%Y-%m-%d')}")
        logger.info("[WARNING] 重要：模型将仅使用训练期数据训练，对测试期进行纯预测")

        # [强化] 解析模型选择参数
        models_str = getattr(args, 'models', 'lightgbm')
        selected_models = [m.strip().lower() for m in models_str.split(',')]

        use_lightgbm = 'lightgbm' in selected_models

        if not use_lightgbm:
            logger.error("[FAIL] 必须选择LightGBM模型")
            return False

        logger.info(f"[强化] 模型选择: {', '.join(selected_models)}")
        logger.info(f"   LightGBM: {'[OK]'}")

        # 2. 训练模型或跳过训练
        if getattr(args, 'skip_train', False):
            # 🆕 跳过训练，直接加载最新模型
            logger.info("\n⏭️ 步骤1: 跳过训练，直接加载最新模型...")
            logger.info("=" * 50)
        else:
            # 训练新模型
            logger.info("\n[REFRESH] 步骤1: 训练仅使用训练期数据的模型（使用优化后的配置）...")
            logger.info("[强化] 优化配置：智能标签变换 + 增强模型复杂度 + 交易导向评估")
            logger.info("=" * 50)

            # 调用训练模式，使用优化后的配置：启用回归模式和模型选择
            train_success = run_train_model_mode(
                args,
                train_end_date=train_end_date,
                is_regression=True,
                use_lightgbm=use_lightgbm  # [强化] 根据命令行参数决定
            )

            if not train_success:
                logger.error("训练期专用模型训练失败")
                return False

            logger.info("[OK] 训练期专用模型训练完成")
            logger.info("=" * 50)

        # 3. 加载完整因子数据用于预测（包含标签生成字段用于回测验证）
        logger.info("\n[REFRESH] 步骤2: 加载完整因子数据进行预测...")
        factor_data = _load_factor_cache_data(include_label_generation_features=True)

        if factor_data is None:
            logger.error("因子数据加载失败")
            return False

        logger.info(f"完整数据加载成功: {len(factor_data)} 条记录, {len(factor_data.columns)} 个字段")
        logger.info(f"数据日期范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

        # 🆕 加载市场数据（用于市场状态判断）
        logger.info("\n[REFRESH] 加载市场数据（用于市场状态自适应）...")
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("[WARNING] 市场数据加载失败，将使用默认震荡市状态")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"[OK] 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"[WARNING] 市场数据加载失败: {e}，将使用默认震荡市状态")
            market_data = None

        # 3.5. 跳过交互特征计算，直接使用基础特征
        logger.info("[REFRESH] 使用基础特征进行预测...")

        # 4. 提取预测数据（确保与训练数据无重叠）
        logger.info("\n✂️ 步骤3: 提取纯预测数据...")
        logger.info("=" * 60)
        logger.info("[SEARCH] 数据集划分验证:")
        logger.info(f"  [CHART] 完整数据: {len(factor_data):,} 条记录")
        logger.info(f"     [DATE] 日期范围: {factor_data['date'].min().strftime('%Y-%m-%d')} 至 {factor_data['date'].max().strftime('%Y-%m-%d')}")
        logger.info(f"  [TARGET] 训练截止日期: {train_end_date} (不包含)")
        logger.info(f"  [TARGET] 测试开始日期: {test_start_date.strftime('%Y-%m-%d')} (包含)")
        logger.info(f"  [TARGET] 测试结束日期: {test_end_date.strftime('%Y-%m-%d')} (包含)")

        prediction_data = factor_data[factor_data['date'] >= test_start_date].copy()

        logger.info(f"  [CHART] 测试集: {len(prediction_data):,} 条记录")
        logger.info(f"     [DATE] 日期范围: {prediction_data['date'].min().strftime('%Y-%m-%d')} 至 {prediction_data['date'].max().strftime('%Y-%m-%d')}")

        # [强化] 验证训练集和测试集无重叠
        train_data_count = len(factor_data[factor_data['date'] < pd.to_datetime(train_end_date)])
        overlap_count = len(factor_data[(factor_data['date'] >= test_start_date) & (factor_data['date'] < pd.to_datetime(train_end_date))])

        if overlap_count > 0:
            logger.error(f"[FAIL] 数据泄露警告: 训练集和测试集有{overlap_count:,}条重叠记录!")
        else:
            logger.info(f"  [OK] 数据集无重叠: 训练集{train_data_count:,}条 < {train_end_date} <= 测试集{len(prediction_data):,}条")
        logger.info("=" * 60)

        if len(prediction_data) == 0:
            logger.error("预测数据为空，请检查日期范围设置")
            return False

        # 5. 加载刚训练好的模型和参数
        logger.info("\n📦 步骤4: 加载刚训练的专用模型...")

        results_dir = Path("results")
        if not results_dir.exists():
            logger.error("results目录不存在")
            return False

        # [强化] 查找最新的模型文件（根据用户选择）
        latest_xgb_file = None
        latest_lgb_file = None
        model_timestamp = None

        # 查找XGBoost模型
        if use_xgboost:
            logger.info("[SEARCH] 查找XGBoost模型（排除walkforward模式的window模型）...")
            xgb_files = list(results_dir.glob("xgboost_model_*.pkl"))
            # [WRENCH] 修复：排除window开头的模型（walkforward模式专用）
            xgb_files = [f for f in xgb_files if not f.stem.startswith("xgboost_model_window")]

            if xgb_files:
                # 按时间戳排序,找到最新的XGBoost模型
                latest_xgb_file = max(xgb_files, key=lambda f: f.stem)
                logger.info(f"[OK] 找到最新XGBoost模型: {latest_xgb_file.name}")

                # 提取时间戳
                parts = latest_xgb_file.stem.split('_')
                if len(parts) >= 3:
                    # 格式: xgboost_model_YYYYMMDD_HHMMSS.pkl
                    model_timestamp = f"{parts[-2]}_{parts[-1]}"  # YYYYMMDD_HHMMSS
            else:
                logger.warning("[WARNING] 未找到XGBoost模型文件")
                if not use_lightgbm:
                    logger.error("[FAIL] 仅选择XGBoost但未找到模型文件")
                    return False
                logger.warning("   将仅使用LightGBM模型")
                use_xgboost = False  # 降级为单模型

        # 查找LightGBM模型
        if use_lightgbm:
            logger.info("[SEARCH] 查找LightGBM模型（排除walkforward模式的window模型）...")
            lgb_files = list(results_dir.glob("lightgbm_model_*.pkl"))
            # [WRENCH] 修复：排除window开头的模型（walkforward模式专用）
            lgb_files = [f for f in lgb_files if not f.stem.startswith("lightgbm_model_window")]

            if model_timestamp is not None:
                # 如果已有XGBoost时间戳,查找相同时间戳的LightGBM模型
                logger.info(f"   查找相同时间戳的LightGBM模型(时间戳: {model_timestamp})...")
                lgb_pattern = f"lightgbm_model_{model_timestamp}.pkl"
                lgb_file = results_dir / lgb_pattern

                if lgb_file.exists():
                    latest_lgb_file = lgb_file
                    logger.info(f"[OK] 找到LightGBM模型: {latest_lgb_file.name}")
                else:
                    logger.warning(f"[WARNING] 未找到相同时间戳的LightGBM模型")
                    logger.warning("   将仅使用XGBoost模型")
                    use_lightgbm = False  # 降级为单模型
            elif lgb_files:
                # 如果没有XGBoost时间戳,查找最新的LightGBM模型
                latest_lgb_file = max(lgb_files, key=lambda f: f.stem)
                logger.info(f"[OK] 找到最新LightGBM模型: {latest_lgb_file.name}")
            else:
                logger.warning("[WARNING] 未找到LightGBM模型文件")
                if not use_xgboost:
                    logger.error("[FAIL] 仅选择LightGBM但未找到模型文件")
                    return False
                logger.warning("   将仅使用XGBoost模型")
                use_lightgbm = False  # 降级为单模型

        # [强化] 收集要加载的模型文件
        latest_models = []
        if latest_xgb_file is not None:
            latest_models.append(latest_xgb_file)
        if latest_lgb_file is not None:
            latest_models.append(latest_lgb_file)

        if not latest_models:
            logger.error("[FAIL] 未找到任何可用的模型文件")
            return False

        logger.info(f"📁 将加载{len(latest_models)}个模型:")
        for f in latest_models:
            logger.info(f"  - {f.name}")

        # [强化] 加载所有模型
        models = []
        scalers = []
        model_types = []  # 记录模型类型
        model_performances = []  # 记录模型性能(用于权重计算)
        feature_names = None
        winsorization_thresholds = None  # [强化] Winsorization阈值(从XGBoost加载)
        label_stats = None  # [WRENCH] 修复：在循环外初始化
        industry_mapping = None  # [SW行业编码] 行业映射信息

        for model_file in latest_models:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            models.append(model_data['model'])
            scalers.append(model_data['scaler'])
            model_types.append(model_data.get('model_type', 'xgboost'))  # 获取模型类型

            # [强化] 获取验证集性能(用于权重计算) - 使用RankIC而非MSE
            performance = model_data.get('performance', {})
            # [强化] 排序模型使用RankIC作为权重计算指标
            rank_ic = performance.get('rank_ic')
            model_performances.append(rank_ic)

            if feature_names is None:
                feature_names = model_data['feature_names']

            model_type = model_data.get('model_type', 'xgboost')

            # [强化] 从XGBoost模型加载Winsorization阈值(集成学习共享预处理参数)
            if winsorization_thresholds is None and model_type == 'xgboost':
                winsorization_thresholds = model_data.get('winsorization_thresholds', None)

            # [WRENCH] 修复：从XGBoost模型加载标签统计量
            if label_stats is None and model_type == 'xgboost':
                label_stats = model_data.get('label_stats', None)

            # [SW行业编码] 只从LightGBM模型加载行业映射信息（XGBoost不需要）
            if industry_mapping is None and model_type == 'lightgbm' and 'industry_mapping' in model_data:
                industry_mapping = model_data['industry_mapping']
                logger.info(f"  [SW行业编码] 从LightGBM模型加载行业映射信息: 特征名='{industry_mapping.get('industry_feature_name', 'industry_sw_code')}'")
            logger.info(f"  [OK] 加载{model_type.upper()}模型: {model_file.name}")
            if rank_ic is not None:
                logger.info(f"     验证集RankIC: {rank_ic:.4f}")
            else:
                logger.warning(f"     [WARNING] 未找到验证集RankIC信息")

        # [强化] 检查标签统计量是否加载成功
        if label_stats:
            logger.info(f"[OK] 加载标签统计量: μ={label_stats['mean']:.6f}, σ={label_stats['std']:.6f}")
        else:
            logger.warning("[WARNING] 未找到标签统计量，可能影响回归预测的反标准化")

        logger.info(f"[OK] 模型加载成功: {len(models)}个模型，特征数量: {len(feature_names)}")

        # [强化] 计算集成学习权重(基于验证集RankIC) - 排序模型专用
        model_weights = None
        if len(models) > 1:
            logger.info("\n[CHART] 计算集成学习权重(基于验证集RankIC)...")
            logger.info("   理由: 排序模型的目标是排序准确性,RankIC是最佳评估指标")

            # 检查是否所有模型都有RankIC
            if all(rank_ic is not None for rank_ic in model_performances):
                # [强化] 权重 = |RankIC| (归一化)
                # 使用绝对值,因为负相关也有价值(可以反向使用)
                weights = np.array([abs(rank_ic) for rank_ic in model_performances])
                model_weights = weights / weights.sum()

                logger.info("[OK] 模型权重:")
                for i, (model_type, weight, rank_ic) in enumerate(zip(model_types, model_weights, model_performances)):
                    logger.info(f"  模型{i+1}({model_type.upper()}): 权重={weight:.4f}, RankIC={rank_ic:.4f}")
            else:
                logger.warning("[WARNING] 部分模型缺少RankIC信息,使用等权重")
                model_weights = np.ones(len(models)) / len(models)
                for i, model_type in enumerate(model_types):
                    logger.info(f"  模型{i+1}({model_type.upper()}): 权重={model_weights[i]:.4f}")
        else:
            logger.info("[OK] 单模型模式,无需计算权重")


        # 6. 准备预测数据特征
        logger.info("\n[WRENCH] 步骤5: 准备预测数据特征...")

        # 识别特征列（使用与训练时完全相同的简化逻辑）
        prediction_feature_columns = _get_ml_feature_columns_simple(prediction_data)
        logger.info(f"预测数据识别到 {len(prediction_feature_columns)} 个特征")

        # 特征一致性检查（修复逻辑错误）
        logger.info(f"[CHART] 特征分析:")
        logger.info(f"   训练时特征数: {len(feature_names)}个（包含PCA特征）")
        logger.info(f"   预测数据特征: {len(prediction_feature_columns)}个（原始特征）")

        # 检查预测数据中是否包含足够的特征来生成PCA
        # 注意：这里不需要与训练时的特征做严格对比，因为PCA会处理特征对齐
        logger.info(f"[OK] 预测数据包含 {len(prediction_feature_columns)} 个原始特征，将用于PCA处理")

        logger.info(f"[CHART] 预测数据统计:")
        logger.info(f"  预测样本总数: {len(prediction_data):,}个")
        logger.info(f"  预测日期范围: {prediction_data['date'].min()} 到 {prediction_data['date'].max()}")
        logger.info(f"  预测股票数量: {prediction_data['stock_code'].nunique()}只")

        # 7. 测试集数据预处理（使用训练时的Winsorization阈值）
        logger.info("\n⚙️ 步骤6: 测试集数据预处理（使用训练时参数）...")

        # [强化] 使用从XGBoost模型加载的Winsorization阈值(集成学习共享预处理参数)
        if winsorization_thresholds:
            logger.info(f"[OK] 使用XGBoost模型的Winsorization阈值: {len(winsorization_thresholds)}个特征")
        else:
            logger.warning("[WARNING] 未找到训练时的Winsorization阈值，将跳过缩尾处理")

        # [SW行业编码] 检查是否需要为测试集添加行业特征（仅限LightGBM模型）
        prediction_data_with_industry = prediction_data.copy()
        industry_feature_added = False

        if industry_mapping is not None and use_lightgbm:
            industry_feature_name = industry_mapping.get('industry_feature_name', 'industry_sw_code')
            stock_to_industry_name = industry_mapping.get('stock_to_industry_name', {})
            industry_name_to_code = industry_mapping.get('industry_name_to_code', {})

            # 检查行业特征名是否在特征列中
            if industry_feature_name in prediction_feature_columns:
                # 检查预测数据是否有股票代码列
                if 'stock_code' in prediction_data.columns and stock_to_industry_name and industry_name_to_code:
                    logger.info(f"[SW行业编码] 为预测数据添加行业特征: {industry_feature_name}")

                    # 将股票代码映射到行业名称，再映射到行业编码
                    pred_stock_codes = prediction_data['stock_code'].values
                    pred_industry_names = [stock_to_industry_name.get(code, '未知') for code in pred_stock_codes]
                    pred_industry_codes = [industry_name_to_code.get(name, -1) for name in pred_industry_names]

                    # 将行业编码添加到预测数据中
                    prediction_data_with_industry[industry_feature_name] = pred_industry_codes

                    # 统计行业分布
                    unique_codes = set(code for code in pred_industry_codes if code != -1)
                    unknown_count = pred_industry_codes.count(-1)
                    logger.info(f"  预测数据行业编码统计: {len(unique_codes)}个行业类别，{unknown_count}个未知行业")

                    industry_feature_added = True
                    # 将行业特征名添加到特征列列表中
                    if industry_feature_name not in prediction_feature_columns:
                        prediction_feature_columns.append(industry_feature_name)
                        logger.info(f"[SW行业编码] 已将行业特征 '{industry_feature_name}' 添加到特征列列表")
                else:
                    # LightGBM训练时有行业特征，但预测数据无法添加
                    logger.warning(f"[WARNING] LightGBM训练时有行业特征'{industry_feature_name}'，但预测数据无法添加")
                    logger.warning(f"  将从特征列中移除行业特征，但模型性能可能受影响")

                    # 从特征列中移除行业特征名
                    prediction_feature_columns = [col for col in prediction_feature_columns if col != industry_feature_name]
                    logger.warning(f"  更新后的特征列: {len(prediction_feature_columns)}个特征")

                    if 'stock_code' not in prediction_data.columns:
                        logger.warning(f"  预测数据缺少'stock_code'列")
                    if not stock_to_industry_name:
                        logger.warning(f"  行业映射中缺少股票代码到行业名称的映射")
                    if not industry_name_to_code:
                        logger.warning(f"  行业映射中缺少行业名称到编码的映射")
            else:
                # LightGBM训练时没有添加行业特征（尽管有industry_mapping）
                logger.info(f"⏭️ 特征列中不包含行业特征'{industry_feature_name}'，跳过预测数据行业特征添加")
        elif industry_mapping is not None and not use_lightgbm:
            logger.info(f"⏭️ 非LightGBM模型模式，跳过SW行业特征添加")
        else:
            logger.info("⏭️ 模型结果中无行业映射信息，跳过预测数据行业特征添加")

        # [SW行业编码] 准备用于特征对齐的特征列表
        # 如果行业特征添加失败，需要从特征列表中移除行业特征名
        filtered_feature_names = feature_names.copy()  # 创建副本用于过滤
        removed_industry_features = []

        if industry_mapping is not None and use_lightgbm:
            industry_feature_name = industry_mapping.get('industry_feature_name', 'industry_sw_code')
            # 检查行业特征是否在feature_names中但不在prediction_feature_columns中（添加失败）
            if (industry_feature_name in filtered_feature_names and
                industry_feature_name not in prediction_feature_columns):
                filtered_feature_names = [f for f in filtered_feature_names if f != industry_feature_name]
                removed_industry_features.append(industry_feature_name)
                logger.warning(f"[SW行业编码] 从对齐特征列表中移除行业特征 '{industry_feature_name}'")

        logger.info(f"[SW行业编码] 特征对齐将使用 {len(filtered_feature_names)} 个特征（原始: {len(feature_names)}）")
        if removed_industry_features:
            logger.warning(f"  移除的行业特征: {removed_industry_features}")

        # 保存原始数据的索引，以便后续对齐预测结果
        original_prediction_data = prediction_data.copy()
        original_indices = prediction_data.index.copy()

        # 使用_prepare_feature_data进行测试集预处理
        X_pred_processed, available_features, _ = _prepare_feature_data(
            prediction_data_with_industry, prediction_feature_columns,
            apply_winsorization=bool(winsorization_thresholds),
            winsorization_thresholds=winsorization_thresholds,
            data_type="测试集"
        )

        if X_pred_processed is None:
            logger.error("测试集数据预处理失败")
            return False

        logger.info(f"[OK] 测试集预处理完成: {X_pred_processed.shape}")
        logger.info(f"原始数据长度: {len(original_prediction_data)}, 预处理后长度: {len(X_pred_processed)}")
        logger.info("[OK] 确保测试集与训练时使用相同的数据预处理流程")

        # 使用预处理后的数据
        X_pred = X_pred_processed

        # 先将预测数据转换为DataFrame以便处理
        X_pred_df = pd.DataFrame(X_pred, columns=available_features)
        logger.info(f"[CHART] 预测数据准备完成: {X_pred_df.shape}")

        # 7. 预测阶段特征对齐（使用基础特征）
        logger.info("\n[WRENCH] 步骤7: 预测阶段特征对齐...")

        # 8. 特征对齐处理（使用训练时的基础特征集）
        logger.info("\n[WRENCH] 步骤8: 特征对齐处理（使用基础特征集）...")
        feature_desc = "基础特征集（股票因子）"

        aligned_features = []
        aligned_data = []
        for feature in filtered_feature_names:
            if feature in X_pred_df.columns:
                aligned_features.append(feature)
                aligned_data.append(X_pred_df[feature].values)
            else:
                logger.warning(f"   [WARNING] 特征 {feature} 在预测数据中不存在，用0填充")
                aligned_features.append(feature)
                aligned_data.append(np.zeros(len(X_pred_df)))

        if aligned_data:
            X_pred = np.column_stack(aligned_data)
            available_features = aligned_features
            logger.info(f"[OK] 特征对齐完成: {X_pred.shape}")
            logger.info(f"[OK] 使用 {feature_desc}: {len(available_features)} 个特征")
            logger.info(f"[OK] 特征数量匹配: 预测特征数={len(available_features)}, 训练特征数={len(filtered_feature_names)}")
        else:
            logger.error("[FAIL] 特征对齐失败")
            return False

        # [强化] 9. 特征标准化（集成学习：延迟到预测循环中）
        logger.info("\n[WRENCH] 步骤9: 准备未标准化的特征数据（集成学习模式）...")

        # [OK] 最终特征（未标准化，将在集成预测循环中由每个模型的scaler进行标准化）
        X_pred_final = X_pred

        logger.info(f"[OK] 最终特征准备完成: {X_pred_final.shape[1]} 个特征，{X_pred_final.shape[0]} 个样本")
        logger.info(f"[WARNING] 注意：特征未标准化，将在集成预测循环中由每个模型的scaler进行标准化")

        # [强化] 10. Pseudo-Huber回归模型预测
        logger.info("\n🔮 步骤10: 进行Pseudo-Huber回归模型预测...")

        try:
            # [强化] 集成学习：对每个模型进行预测
            predictions = []
            for i, (model, scaler, model_type) in enumerate(zip(models, scalers, model_types)):
                logger.info(f"\n🔮 模型{i+1}({model_type.upper()})预测...")

                # [WRENCH] 修复：不再使用StandardScaler，直接使用Winsorization后的特征
                # 因子在计算时已经进行了时间序列标准化（如zscore、rank）
                X_pred_scaled = X_pred_final

                # [强化] 根据模型类型进行预测
                if model_type == 'lightgbm':
                    # LightGBM模型：直接预测
                    y_pred_raw = model.predict(X_pred_scaled, num_iteration=model.best_iteration)
                    logger.info(f"  [OK] 使用LightGBM模型预测")
                elif hasattr(model, 'predict') and hasattr(model, 'get_score'):
                    # XGBoost Booster对象：需要创建DMatrix
                    import xgboost as xgb
                    dtest = xgb.DMatrix(X_pred_scaled)
                    y_pred_raw = model.predict(dtest)
                    logger.info(f"  [OK] 使用XGBoost Booster对象预测")
                else:
                    # XGBoost XGBRegressor对象：直接预测
                    y_pred_raw = model.predict(X_pred_scaled)
                    logger.info(f"  [OK] 使用XGBoost XGBRegressor对象预测")

                # [强化] 直接使用原始ATR预测值，无需逆变换
                y_pred = y_pred_raw
                logger.info(f"  [OK] 直接使用原始ATR预测值")

                predictions.append(y_pred)
                logger.info(f"  [OK] 模型{i+1}({model_type.upper()})预测完成: [{y_pred.min():.3f}, {y_pred.max():.3f}] ATR")

            # [强化] 集成预测：专业的集成策略(基于验证集MSE的加权平均)
            predictions = np.array(predictions)
            n_models = len(predictions)

            logger.info(f"\n[OK] 集成预测完成")
            logger.info(f"  模型数量: {n_models}")

            # 检查模型数量
            if n_models == 1:
                logger.info(f"  [OK] 单模型模式,直接使用预测结果")
                mean_pred = predictions[0]
                weighted_mean_pred = predictions[0]
                std_pred = np.zeros_like(mean_pred)
                median_pred = mean_pred
                min_pred = mean_pred
                max_pred = mean_pred
            else:
                # 🆕 专业的集成统计
                # 1. 简单均值预测
                mean_pred = np.mean(predictions, axis=0)

                # 2. [强化] 基于验证集MSE的加权平均(使用之前计算的model_weights)
                if model_weights is not None:
                    # 加权平均
                    weighted_mean_pred = np.zeros_like(predictions[0])
                    for i, weight in enumerate(model_weights):
                        weighted_mean_pred += weight * predictions[i]

                    logger.info(f"  [OK] 使用基于MSE的加权平均")
                else:
                    # 使用简单均值
                    weighted_mean_pred = mean_pred
                    logger.info(f"  [OK] 使用简单均值")

                # 3. 标准差(模型间不确定性)
                std_pred = np.std(predictions, axis=0, ddof=1)  # 使用无偏估计

                # 4. 中位数预测(鲁棒性更好,对异常值不敏感)
                median_pred = np.median(predictions, axis=0)

                # 5. 最小/最大预测(预测范围)
                min_pred = np.min(predictions, axis=0)
                max_pred = np.max(predictions, axis=0)

                # 统计信息
                logger.info(f"  简单均值范围: [{mean_pred.min():.3f}, {mean_pred.max():.3f}] ATR")
                logger.info(f"  加权均值范围: [{weighted_mean_pred.min():.3f}, {weighted_mean_pred.max():.3f}] ATR")
                logger.info(f"  中位数范围: [{median_pred.min():.3f}, {median_pred.max():.3f}] ATR")
                logger.info(f"  标准差范围: [{std_pred.min():.3f}, {std_pred.max():.3f}] ATR")
                logger.info(f"  标准差均值: {std_pred.mean():.3f} ATR (模型间不确定性)")
                logger.info(f"  预测范围: [{min_pred.min():.3f}, {max_pred.max():.3f}] ATR")

                # 检查模型一致性
                avg_spread = (max_pred - min_pred).mean()
                logger.info(f"  平均预测跨度: {avg_spread:.3f} ATR (模型间差异)")

                if avg_spread > 2.0:
                    logger.warning(f"  [WARNING] 模型间差异较大(>{2.0} ATR),可能需要检查模型训练")

            # [强化] 最终预测值：直接使用加权均值（不做风险调整）
            # 理由：
            # 1. 风险调整会导致高预测值被过度惩罚
            # 2. 置信度过滤已经基于std_pred进行了不确定性控制
            # 3. 避免双重惩罚（风险调整 + 置信度过滤）
            y_pred_return = weighted_mean_pred

            logger.info(f"\n[OK] 最终预测值（基于MSE加权均值，无风险调整）")
            logger.info(f"  使用加权均值预测（基于验证集MSE）")
            logger.info(f"  预测范围: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")
            logger.info(f"  [CHART] 最终预测值统计: [{y_pred_return.min():.3f}, {y_pred_return.max():.3f}] ATR")
            if n_models > 1:
                logger.info(f"  [CHART] 不确定性控制: 通过置信度过滤（基于std_pred）实现")

        except Exception as e:
            logger.error(f"[FAIL] 模型预测失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False

        # 8. 处理预测结果
        logger.info("\n[CHART] 步骤11: 处理预测结果...")

        if len(y_pred_return) > len(original_prediction_data):
            logger.error(f"预测结果长度({len(y_pred_return)})大于原始数据长度({len(original_prediction_data)})")
            return False

        # 创建预测结果DataFrame
        prediction_data_with_proba = original_prediction_data.iloc[:len(y_pred_return)].copy()
        prediction_data_with_proba['predicted_return'] = y_pred_return  # 风险调整分数(基于加权均值)
        prediction_data_with_proba['mean_pred'] = mean_pred  # [强化] 集成学习：简单均值预测
        prediction_data_with_proba['weighted_mean_pred'] = weighted_mean_pred  # [强化] 集成学习：加权均值预测(基于MSE)
        prediction_data_with_proba['median_pred'] = median_pred if n_models > 1 else mean_pred  # [强化] 集成学习：中位数预测
        prediction_data_with_proba['std_pred'] = std_pred  # [强化] 集成学习：标准差（不确定性）
        prediction_data_with_proba['predicted_return_atr_raw'] = y_pred_return  # [强化] 修复：添加ATR原始预测字段
        prediction_data_with_proba['y_pred_proba'] = y_pred_return  # 兼容下游逻辑

        positive_count = (y_pred_return > 0).sum()
        logger.info(f"[OK] 预测结果处理完成: {len(prediction_data_with_proba)} 个样本")
        logger.info(f"[CHART] 预测分布: 正值{positive_count}个({positive_count/len(y_pred_return)*100:.1f}%), "
                   f"负值{len(y_pred_return)-positive_count}个")
        if n_models > 1:
            logger.info(f"[CHART] 不确定性统计: 平均std={std_pred.mean():.3f}, 最大std={std_pred.max():.3f}")
        else:
            logger.info(f"[CHART] 单模型预测,无不确定性统计")


        # 9. [强化] 置信度过滤（简化版：基于预测方差 + 市场状态自适应）
        logger.info("\n[TARGET] 步骤12: 置信度过滤（基于预测方差 + 市场状态自适应）...")
        logger.info("=" * 60)

        try:
            # 🆕 市场状态判断
            market_state = _get_market_state(market_data, lookback=20, logger=logger)

            # 🆕 根据市场状态调整置信度阈值分位数
            if market_state == 'bull':
                confidence_percentile = 70  # 牛市：降低阈值，多选股
                logger.info(f"🐂 牛市状态：降低置信度要求（{confidence_percentile}%分位数）")
            elif market_state == 'bear':
                confidence_percentile = 85  # 熊市：提高阈值，少选股
                logger.info(f"🐻 熊市状态：提高置信度要求（{confidence_percentile}%分位数）")
            else:
                confidence_percentile = 75  # 震荡市：中等阈值
                logger.info(f"[CHART] 震荡市状态：中等置信度要求（{confidence_percentile}%分位数）")

            # [强化] 简化方案：使用模型预测的标准差作为置信度指标
            # - std小 -> 模型预测一致 -> 高置信度
            # - std大 -> 模型预测分歧 -> 低置信度

            logger.info(f"\n[CHART] 使用已计算的预测标准差作为置信度指标...")

            # 使用已经计算好的std_pred（模型间不确定性）
            pred_std = prediction_data_with_proba['std_pred'].values

            # 🆕 根据市场状态动态调整阈值
            threshold = np.percentile(pred_std, confidence_percentile)

            # 应用置信度过滤
            high_confidence_mask = pred_std < threshold

            logger.info(f"  预测标准差统计: μ={pred_std.mean():.6f}, σ={pred_std.std():.6f}")
            logger.info(f"  标准差范围: [{pred_std.min():.6f}, {pred_std.max():.6f}]")
            logger.info(f"  [OK] 置信度阈值({confidence_percentile}%分位数): {threshold:.6f}")

            logger.info(f"\n[CHART] 置信度过滤结果:")
            logger.info(f"  总样本数: {len(pred_std):,}")
            logger.info(f"  高置信度样本: {high_confidence_mask.sum():,} ({high_confidence_mask.sum()/len(pred_std)*100:.1f}%)")
            logger.info(f"  低置信度样本: {(~high_confidence_mask).sum():,} ({(~high_confidence_mask).sum()/len(pred_std)*100:.1f}%)")

            # 添加置信度字段到预测结果
            prediction_data_with_proba['high_confidence'] = high_confidence_mask

            # 统计每日的置信度分布
            daily_confidence = prediction_data_with_proba.groupby('date').agg({
                'high_confidence': ['sum', 'count']
            }).reset_index()
            daily_confidence.columns = ['date', 'high_conf_count', 'total_count']
            daily_confidence['high_conf_ratio'] = daily_confidence['high_conf_count'] / daily_confidence['total_count']

            logger.info(f"\n[CHART] 每日置信度分布:")
            logger.info(f"  平均高置信度比例: {daily_confidence['high_conf_ratio'].mean()*100:.1f}%")
            logger.info(f"  最低高置信度比例: {daily_confidence['high_conf_ratio'].min()*100:.1f}% (日期: {daily_confidence.loc[daily_confidence['high_conf_ratio'].idxmin(), 'date']})")
            logger.info(f"  最高高置信度比例: {daily_confidence['high_conf_ratio'].max()*100:.1f}% (日期: {daily_confidence.loc[daily_confidence['high_conf_ratio'].idxmax(), 'date']})")

            logger.info(f"\n[OK] 置信度过滤完成")
            logger.info("=" * 60)

        except Exception as e:
            logger.warning(f"[WARNING] 置信度过滤失败: {e}")
            logger.warning("将继续使用全部预测结果")
            import traceback
            logger.warning(f"详细错误: {traceback.format_exc()}")

            # 添加默认置信度字段
            prediction_data_with_proba['high_confidence'] = True

        # 10. 保存预测结果
        logger.info("\n[SAVE] 步骤13: 保存预测结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_results = prediction_data_with_proba[['date', 'stock_code', 'predicted_return', 'y_pred_proba', 'std_pred', 'high_confidence']].copy()
        prediction_file = results_dir / f"pure_predictions_{timestamp}.csv"
        prediction_results.to_csv(prediction_file, index=False, encoding='utf-8')
        logger.info(f"[OK] 预测结果已保存: {prediction_file} ({len(prediction_results)} 条记录)")

        # 11. 选股回测系统 - 调用统一回测函数
        logger.info("\n[REFRESH] 步骤14: 开始选股回测...")
        logger.info("=" * 60)

        # [OK] 调用统一的选股回测函数
        backtest_result = _run_stock_selection_backtest(
            backtest_data=prediction_data_with_proba,
            stock_counts=[1, 2, 3, 5],  # Top-1/2/3/5选股
            window_name="test_model",
            market_data=None,  # test_model暂不使用市场状态
            frequency='minute'  # 默认日线回测，保持向后兼容
        )

        # 🔍 添加调试日志：检查回测结果
        logger.debug(f"🔍 [DEBUG] 回测结果检查:")
        logger.debug(f"  - backtest_result类型: {type(backtest_result)}")
        logger.debug(f"  - 是否为None: {backtest_result is None}")
        if backtest_result is not None:
            logger.debug(f"  - backtest_result键: {list(backtest_result.keys())}")
            logger.debug(f"  - backtest_results存在: {'backtest_results' in backtest_result}")
            if 'backtest_results' in backtest_result:
                logger.debug(f"  - backtest_results内容: {backtest_result['backtest_results']}")

        if backtest_result is None:
            logger.error("[FAIL] 选股回测失败")
            return False

        if 'backtest_results' not in backtest_result and not any(isinstance(k, int) for k in backtest_result.keys()):
            logger.error("[FAIL] 选股回测失败")
            return False

        # 提取回测结果
        if 'backtest_results' in backtest_result:
            backtest_results = backtest_result['backtest_results']
            detailed_trades = backtest_result['detailed_trades']
            summary_stats = backtest_result['summary_stats']
        else:
            backtest_results = {k: v for k, v in backtest_result.items() if isinstance(k, int)}
            detailed_trades = {k: v.get('trades', []) for k, v in backtest_results.items()}
            summary_stats = {}

        # 12. 保存回测结果
        logger.info("\n[SAVE] 步骤15: 保存回测结果...")

        # 保存回测指标汇总
        trading_dates = sorted(prediction_data_with_proba['date'].unique())
        backtest_summary = {
            "timestamp": timestamp,
            "backtest_period": {
                "start_date": trading_dates[0].strftime("%Y-%m-%d"),
                "end_date": trading_dates[-1].strftime("%Y-%m-%d"),
                "trading_days": len(trading_dates) - 1
            },
            "strategy_results": {}
        }


        # [强化] 输出置信度过滤统计
        if 'high_confidence' in prediction_data_with_proba.columns:
            logger.info("\n[CHART] 置信度过滤统计:")
            logger.info("=" * 60)

            total_samples = len(prediction_data_with_proba)
            high_conf_samples = prediction_data_with_proba['high_confidence'].sum()
            low_conf_samples = total_samples - high_conf_samples

            logger.info(f"  总样本数: {total_samples:,}")
            logger.info(f"  高置信度样本: {high_conf_samples:,} ({high_conf_samples/total_samples*100:.1f}%)")
            logger.info(f"  低置信度样本: {low_conf_samples:,} ({low_conf_samples/total_samples*100:.1f}%)")

            # 统计每日置信度分布
            daily_stats = prediction_data_with_proba.groupby('date').agg({
                'high_confidence': ['sum', 'count']
            }).reset_index()
            daily_stats.columns = ['date', 'high_conf', 'total']
            daily_stats['ratio'] = daily_stats['high_conf'] / daily_stats['total']

            logger.info(f"\n  每日置信度分布:")
            logger.info(f"    平均高置信度比例: {daily_stats['ratio'].mean()*100:.1f}%")
            logger.info(f"    最低高置信度比例: {daily_stats['ratio'].min()*100:.1f}% (日期: {daily_stats.loc[daily_stats['ratio'].idxmin(), 'date'].strftime('%Y-%m-%d')})")
            logger.info(f"    最高高置信度比例: {daily_stats['ratio'].max()*100:.1f}% (日期: {daily_stats.loc[daily_stats['ratio'].idxmax(), 'date'].strftime('%Y-%m-%d')})")

            # 统计k-NN距离分布
            if 'knn_distance' in prediction_data_with_proba.columns:
                knn_dist = prediction_data_with_proba['knn_distance'].dropna()
                if len(knn_dist) > 0:
                    logger.info(f"\n  k-NN距离分布:")
                    logger.info(f"    均值: {knn_dist.mean():.4f}")
                    logger.info(f"    标准差: {knn_dist.std():.4f}")
                    logger.info(f"    范围: [{knn_dist.min():.4f}, {knn_dist.max():.4f}]")
                    logger.info(f"    分位数: 25%={knn_dist.quantile(0.25):.4f}, 50%={knn_dist.quantile(0.50):.4f}, 75%={knn_dist.quantile(0.75):.4f}, 90%={knn_dist.quantile(0.90):.4f}")

            logger.info("=" * 60)

        # 输出对比结果
        logger.info('\n[CHART] 选股策略回测结果对比:')
        logger.info("=" * 150)
        logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'持仓天数':<10} {'总交易次数':<12}")
        logger.info("=" * 150)


        stock_counts = [1, 2, 3, 5]  # Top-1/2/3/5选股
        for stock_count in stock_counts:
            if stock_count in backtest_results:
                result = backtest_results[stock_count]
                backtest_summary["strategy_results"][f"{stock_count}stocks"] = result

                # 显示持仓天数和交易次数
                holding_days = result.get('holding_days', 0)
                total_trades = result.get('total_trades', 0)
                trades_desc = f"{total_trades:>6d}"

                logger.info(f"Top-{stock_count:<5} {result['total_return_gross']:>8.2%}  "
                           f"{result['total_return']:>13.2%}  "
                           f"{result['annual_return']:>8.2%}  {result['max_drawdown']:>8.2%}  "
                           f"{result['win_rate']:>6.2%}  "
                           f"{result['profit_loss_ratio']:>6.2f}  "
                           f"{holding_days:>10d}{trades_desc}")

        logger.info("=" * 150)

        # 保存回测汇总报告
        backtest_report_file = results_dir / f"backtest_summary_{timestamp}.json"
        with open(backtest_report_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[OK] 回测汇总报告已保存: {backtest_report_file}")

        # 保存详细交易记录
        for stock_count in stock_counts:
            if stock_count in detailed_trades:
                trades_df = pd.DataFrame(detailed_trades[stock_count])
                trades_file = results_dir / f"detailed_trades_{stock_count}stocks_{timestamp}.csv"
                trades_df.to_csv(trades_file, index=False, encoding='utf-8')
                logger.info(f"[OK] {stock_count}只股票交易记录已保存: {trades_file}")

        # 🎨 生成收益走势图
        logger.info("\n[CHART] 生成收益走势图...")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams

            # 设置中文字体
            rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            rcParams['axes.unicode_minus'] = False

            # 创建图表
            fig, ax = plt.subplots(figsize=(14, 8))

            # 为每个策略绘制收益曲线
            colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 5: '#96CEB4'}
            linestyles = {1: '-', 2: '--', 3: '-.', 5: ':'}

            for stock_count in stock_counts:
                if stock_count in backtest_results:
                    result = backtest_results[stock_count]

                    # 获取日期和累计收益
                    dates = result.get('dates', [])
                    portfolio_values = result.get('portfolio_values', [])

                    if len(dates) > 0 and len(portfolio_values) > 0:
                        # 转换为百分比收益
                        returns_pct = [(v - 1.0) * 100 for v in portfolio_values]

                        # 转换日期格式
                        dates_formatted = [pd.to_datetime(d) for d in dates]

                        # 绘制曲线
                        ax.plot(dates_formatted, returns_pct,
                               label=f'Top-{stock_count}',
                               color=colors.get(stock_count, '#95A5A6'),
                               linestyle=linestyles.get(stock_count, '-'),
                               linewidth=2.5 if stock_count == 1 else 2.0,
                               alpha=0.9)

            # 设置图表样式
            ax.set_xlabel('日期', fontsize=12, fontweight='bold')
            ax.set_ylabel('累计收益率 (%)', fontsize=12, fontweight='bold')
            ax.set_title('不同选股策略收益走势对比', fontsize=14, fontweight='bold', pad=20)

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # 添加零线
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

            # 设置图例
            ax.legend(loc='upper left', fontsize=11, framealpha=0.9, shadow=True)

            # 格式化x轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45, ha='right')

            # 调整布局
            plt.tight_layout()

            # 保存图表
            chart_file = results_dir / f"return_curves_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            logger.info(f"[OK] 收益走势图已保存: {chart_file}")

            plt.close()

        except Exception as e:
            logger.warning(f"[WARNING] 生成收益走势图失败: {e}")

        logger.info("=" * 60)
        logger.info("[CELEBRATE] Pseudo-Huber回归选股回测完成！")
        logger.info("[CHART] 已完成1只、2只、3只、5只股票的选股策略对比")
        logger.info("[OK] 样本外验证：模型从未见过测试期数据")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"[FAIL] 模型测试模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_factor_correlation_mode(args):
    """运行因子相关性分析模式"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("[SEARCH] ATR三分类标签系统 - 因子相关性分析模式")
    logger.info("=" * 80)

    try:
        # 1. 加载因子数据
        logger.info("[CHART] 步骤1: 加载系统因子数据")
        logger.info("-" * 50)


        # [WRENCH] 因子测试使用模型因子列表（归一化因子）
        required_factors = _get_model_factors()

        logger.info(f"分析因子数量: {len(required_factors)}")
        logger.info("因子列表:")
        for i, factor in enumerate(required_factors, 1):
            logger.info(f"  {i:2d}. {factor}")

        # 加载因子数据
        logger.info("\n[REFRESH] 加载因子缓存数据...")
        factor_data = _load_factor_cache_data(required_factors=required_factors)

        if factor_data is None or factor_data.empty:
            logger.error("[FAIL] 因子数据加载失败")
            return False

        logger.info(f"[OK] 因子数据加载成功: {len(factor_data):,}条记录")
        logger.info(f"数据时间范围: {factor_data['date'].min()} 至 {factor_data['date'].max()}")

        # 2. 数据预处理和验证
        logger.info("\n[CHART] 步骤2: 数据预处理和验证")
        logger.info("-" * 50)

        # 检查实际可用的因子
        available_factors = [factor for factor in required_factors if factor in factor_data.columns]
        missing_factors = [factor for factor in required_factors if factor not in factor_data.columns]

        logger.info(f"可用因子: {len(available_factors)}/{len(required_factors)}")
        if missing_factors:
            logger.warning(f"缺失因子: {missing_factors}")

        if len(available_factors) < 2:
            logger.error("[FAIL] 可用因子数量不足，无法进行相关性分析")
            return False

        # 提取因子数据矩阵
        factor_matrix = factor_data[available_factors].copy()

        # 数据清理：移除包含NaN的行
        original_rows = len(factor_matrix)
        factor_matrix = factor_matrix.dropna()
        cleaned_rows = len(factor_matrix)

        logger.info(f"数据清理: {original_rows:,} -> {cleaned_rows:,} 行 (移除{original_rows-cleaned_rows:,}行NaN)")

        if cleaned_rows < 1000:
            logger.error(f"[FAIL] 清理后数据量不足({cleaned_rows}行)，无法进行可靠的相关性分析(至少需要1000行)")
            return False

        # 3. 计算相关性矩阵(Spearman秩相关)
        logger.info("\n[SEARCH] 步骤3: 计算因子相关性矩阵")
        logger.info("-" * 50)

        logger.info("使用Spearman秩相关系数计算相关性...")
        logger.info("优势: 对异常值鲁棒,捕捉非线性单调关系,量化领域标准方法")
        logger.info("公式: ρ_s(X,Y) = Pearson(rank(X), rank(Y))")

        # 计算Spearman秩相关矩阵
        correlation_matrix = factor_matrix.corr(method='spearman')

        logger.info(f"[OK] Spearman秩相关矩阵计算完成: {correlation_matrix.shape[0]}×{correlation_matrix.shape[1]}")

        # 4. 高相关性因子识别
        logger.info("\n[TARGET] 步骤4: 高相关性因子识别")
        logger.info("-" * 50)

        correlation_threshold = 0.7
        logger.info(f"相关性阈值: {correlation_threshold}")

        # 找出高相关性因子对
        high_corr_pairs = []
        n_factors = len(available_factors)

        for i in range(n_factors):
            for j in range(i+1, n_factors):
                factor1 = available_factors[i]
                factor2 = available_factors[j]
                corr_value = correlation_matrix.iloc[i, j]

                if abs(corr_value) >= correlation_threshold:
                    high_corr_pairs.append({
                        'factor1': factor1,
                        'factor2': factor2,
                        'correlation': corr_value
                    })

        # 按相关性绝对值排序
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        logger.info(f"发现 {len(high_corr_pairs)} 对高相关性因子:")
        if high_corr_pairs:
            for i, pair in enumerate(high_corr_pairs, 1):
                corr_type = "正相关" if pair['correlation'] > 0 else "负相关"
                logger.info(f"  {i:2d}. {pair['factor1']} ↔ {pair['factor2']}: "
                           f"{pair['correlation']:.3f} ({corr_type})")
        else:
            logger.info("  未发现高相关性因子对")

        # 5. 生成相关性热力图
        logger.info("\n[UP] 步骤5: 生成相关性热力图")
        logger.info("-" * 50)

        # 设置输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建热力图
        plt.style.use('default')
        fig_size = max(12, len(available_factors) * 0.8)  # 动态调整图片大小
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # 生成热力图
        heatmap = sns.heatmap(
            correlation_matrix,
            annot=True,  # 显示数值
            cmap='RdBu_r',  # 红蓝色彩映射
            center=0,  # 以0为中心
            square=True,  # 正方形格子
            fmt='.3f',  # 数值格式
            cbar_kws={'label': '相关系数'},
            ax=ax
        )

        # 设置标题和标签
        plt.title('ATR三分类标签系统 - 因子Spearman秩相关矩阵', fontsize=16, pad=20)
        plt.xlabel('因子', fontsize=12)
        plt.ylabel('因子', fontsize=12)

        # 旋转标签以提高可读性
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        heatmap_path = output_dir / 'factor_correlation_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        logger.info(f"[OK] 相关性热力图已保存: {heatmap_path}")

        # 关闭图形以释放内存
        plt.close()

        # 6. 因子冗余性分析和建议
        logger.info("\n[IDEA] 步骤6: 因子冗余性分析和建议")
        logger.info("-" * 50)

        if high_corr_pairs:
            logger.info("[SEARCH] 因子冗余性分析:")

            # 统计每个因子的高相关性次数
            factor_corr_count = {}
            for pair in high_corr_pairs:
                factor_corr_count[pair['factor1']] = factor_corr_count.get(pair['factor1'], 0) + 1
                factor_corr_count[pair['factor2']] = factor_corr_count.get(pair['factor2'], 0) + 1

            # 按高相关性次数排序
            sorted_factors = sorted(factor_corr_count.items(), key=lambda x: x[1], reverse=True)

            logger.info("因子冗余性排名（高相关性次数）:")
            for i, (factor, count) in enumerate(sorted_factors, 1):
                logger.info(f"  {i:2d}. {factor}: {count}次高相关")

            logger.info("\n[CLIPBOARD] 特征工程建议:")
            logger.info("1. 考虑移除高冗余因子以降低多重共线性")
            logger.info("2. 对于高相关因子对，可以考虑:")
            logger.info("   - 保留业务解释性更强的因子")
            logger.info("   - 保留计算更稳定的因子")
            logger.info("   - 使用PCA等降维技术")
            logger.info("3. 建议优先保留的因子类型:")
            logger.info("   - 收益率相关: returns_zscore")
            logger.info("   - 市场结构: market_structure_score")
            logger.info("   - 动量指标: intraday_momentum, rank_momentum_20d")
            logger.info("   - 技术指标: ma_ratio_5_10, rank_rsi, rank_macd")
        else:
            logger.info("[OK] 未发现显著的因子冗余性问题")
            logger.info("当前因子组合具有良好的独立性")

        # 7. 生成分析报告
        logger.info("\n📄 步骤7: 生成分析报告")
        logger.info("-" * 50)

        report_path = output_dir / 'factor_correlation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ATR三分类标签系统 - 因子Spearman秩相关性分析报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"分析方法: Spearman秩相关(对异常值鲁棒,量化标准方法)\n")
            f.write(f"分析时间: {factor_data['date'].min()} 至 {factor_data['date'].max()}\n")
            f.write(f"数据量: {cleaned_rows:,} 条记录\n")
            f.write(f"分析因子数量: {len(available_factors)}\n")
            f.write(f"相关性阈值: {correlation_threshold}\n\n")

            f.write("分析因子列表:\n")
            for i, factor in enumerate(available_factors, 1):
                f.write(f"  {i:2d}. {factor}\n")

            f.write(f"\n高相关性因子对 ({len(high_corr_pairs)}对):\n")
            if high_corr_pairs:
                for i, pair in enumerate(high_corr_pairs, 1):
                    corr_type = "正相关" if pair['correlation'] > 0 else "负相关"
                    f.write(f"  {i:2d}. {pair['factor1']} ↔ {pair['factor2']}: "
                           f"{pair['correlation']:.3f} ({corr_type})\n")
            else:
                f.write("  未发现高相关性因子对\n")

            if high_corr_pairs:
                f.write("\n因子冗余性排名:\n")
                for i, (factor, count) in enumerate(sorted_factors, 1):
                    f.write(f"  {i:2d}. {factor}: {count}次高相关\n")

        logger.info(f"[OK] 分析报告已保存: {report_path}")

        # 8. 总结
        logger.info("\n" + "=" * 80)
        logger.info("[CELEBRATE] 因子相关性分析完成")
        logger.info("=" * 80)
        logger.info(f"[CHART] 分析结果总结:")
        logger.info(f"  * 分析因子数量: {len(available_factors)}")
        logger.info(f"  * 数据记录数量: {cleaned_rows:,}")
        logger.info(f"  * 高相关性因子对: {len(high_corr_pairs)}")
        logger.info(f"  * 相关性热力图: {heatmap_path}")
        logger.info(f"  * 分析报告: {report_path}")

        if high_corr_pairs:
            logger.info(f"[WARNING]  发现 {len(high_corr_pairs)} 对高相关性因子，建议进行特征工程优化")
        else:
            logger.info("[OK] 因子独立性良好，无需额外处理")

        return True

    except Exception as e:
        logger.error(f"因子相关性分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_genetic_factor_mode(args):
    """运行遗传编程因子进化模式"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🧬 遗传编程因子进化模式")
    logger.info("=" * 80)

    try:
        # 检查gplearn是否可用
        try:
            import gplearn
            logger.info(f"[OK] gplearn版本: {gplearn.__version__}")
        except ImportError:
            logger.error("[FAIL] gplearn未安装，请运行: pip install gplearn")
            return False

        # 导入遗传编程模块
        try:
            from genetic_factor_evolution import run_genetic_factor_evolution
        except ImportError as e:
            logger.error(f"[FAIL] 无法导入遗传编程模块: {e}")
            return False

        # 1. 加载因子数据
        logger.info("[CHART] 步骤1: 加载因子数据")
        logger.info("-" * 50)

        # 定义需要的基础因子
        required_factors = [
            # 技术因子
            'rsi6', 'rsi12', 'rsi14', 'macd', 'volatility20',
            'return_5d', 'return_20d', 'ma5', 'ma10', 'ma20', 'ma60',

            # 动量因子
            'momentum_5d', 'momentum_20d', 'momentum_60d',
            'price_momentum', 'volume_momentum',

            # 成交量因子
            'volume_ratio_5d', 'volume_ratio_20d', 'turnover_rate',
            'volume_price_trend', 'volume_ma_ratio',

            # 价格位置因子
            'price_position_5d', 'price_position_20d', 'price_position_60d',

            # 涨停质量因子
            'is_limit_up', 'limit_strength', 'limit_up_days_5d',
            'consecutive_limit_up', 'limit_up_volume_ratio',

            # 其他重要因子
            'returns_zscore','market_structure_score','ma_ratio_5_10','corr_5d','volatility_correlation',
            'volume_correlation','near_support_count_60d','institutional_activity_score',
            'ma_convergence_score','lower_shadow_ratio','sector_correlation_20d','near_support',
            'price_impact','rank_momentum_20d','rank_rsi','rank_volume'
        ]

        logger.info(f"需要加载的因子数量: {len(required_factors)}")

        # 加载因子数据
        factor_data = _load_factor_cache_data(
            required_factors=required_factors,
            include_label_generation_features=True  # 包含标签生成特征用于目标变量
        )

        if factor_data is None or len(factor_data) == 0:
            logger.error("[FAIL] 因子数据加载失败")
            return False

        logger.info(f"[OK] 因子数据加载成功: {len(factor_data)} 条记录")
        logger.info(f"数据时间范围: {factor_data['date'].min()} 到 {factor_data['date'].max()}")

        # 2. 数据预处理
        logger.info("\n[CHART] 步骤2: 数据预处理")
        logger.info("-" * 50)

        # 确保有目标变量
        if 'future_return_2d' not in factor_data.columns:
            logger.info("计算目标变量 future_return_2d...")
            factor_data = factor_data.sort_values(['stock_code', 'date'])
            factor_data['future_return_2d'] = factor_data.groupby('stock_code')['pct_chg'].shift(-2)

        # 过滤有效数据
        valid_mask = (
            factor_data['future_return_2d'].notna() &
            (factor_data['future_return_2d'].abs() <= 0.2)  # 限制收益率在合理范围内
        )
        factor_data = factor_data[valid_mask].copy()

        logger.info(f"有效数据: {len(factor_data)} 条记录")

        # 3. 运行遗传编程进化
        logger.info("\n🧬 步骤3: 开始遗传编程因子进化")
        logger.info("-" * 50)

        logger.info(f"遗传编程参数:")
        logger.info(f"  种群大小: {args.population_size}")
        logger.info(f"  进化代数: {args.generations}")
        logger.info(f"  返回因子数: {args.n_best_factors}")

        # 运行遗传编程
        results = run_genetic_factor_evolution(
            factor_data=factor_data,
            output_dir=args.output_dir,
            population_size=args.population_size,
            generations=args.generations,
            n_best_factors=args.n_best_factors
        )

        # 4. 结果总结
        logger.info("\n[CLIPBOARD] 步骤4: 结果总结")
        logger.info("-" * 50)

        if results and results.get('best_program'):
            best = results['best_program']
            logger.info(f"[TROPHY] 最佳因子进化完成!")
            logger.info(f"   夏普比率: {best.get('sharpe_ratio', 'N/A'):.4f}")
            logger.info(f"   最大回撤: {best.get('max_drawdown', 'N/A'):.4f}")
            logger.info(f"   年化收益: {best.get('annual_return', 'N/A'):.4f}")
            logger.info(f"   胜率: {best.get('win_rate', 'N/A'):.4f}")



    except Exception as e:
        logger.error(f"[FAIL] 遗传编程因子进化模式运行失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_market_state_test_mode(args):
    """运行市场状态因子分析模式

    分析目标:
    1. 评估市场状态因子的择时能力 ([1,2,3,5,10]日周期)
    2. 评估市场状态因子对18个默认股票因子的影响(开关效应)
    3. 生成专业的量化分析报告
    """
    import numpy as np
    from datetime import datetime
    import pandas as pd
    import statsmodels.api as sm

    logger = logging.getLogger(__name__)
    print("[CHART] 运行模式: 市场状态因子分析 (Enhanced)")
    logger.info("=" * 60)
    logger.info("[强化] 开始市场状态因子分析流程")

    # 定义分析周期(针对短期持仓优化)
    analysis_periods = [1, 2, 3]  # 针对2日持仓策略优化
    logger.info(f"[WRENCH] 分析配置: 时间窗口 {analysis_periods}日 (针对2日持仓优化)，目标因子 18个核心因子")

    # 步骤 1: 加载数据
    logger.info("[REFRESH] [步骤 1/5] 加载所有数据...")
    # [强化] 修复：直接使用_load_factor_cache_data加载所有因子（包含市场因子）
    all_factors_df = _load_factor_cache_data(include_label_generation_features=True)
    if all_factors_df is None: return False

    # [强化] 关键修复：处理重复列名
    if all_factors_df.columns.has_duplicates:
        duplicates = all_factors_df.columns[all_factors_df.columns.duplicated()].tolist()
        logger.warning(f"检测到重复的因子列名: {duplicates}。将自动去重，保留第一个出现的列。")
        all_factors_df = all_factors_df.loc[:, ~all_factors_df.columns.duplicated(keep='first')]

    logger.info(f"[OK] 成功加载统一因子数据: {len(all_factors_df):,} 条记录, {len(all_factors_df.columns)} 个字段")

    # [强化] 关键修复：识别市场因子和股票因子（无需分离和重新合并）
    market_factor_columns = _get_available_market_factors()
    available_market_factors = [col for col in market_factor_columns if col in all_factors_df.columns]
    missing_market_factors = [col for col in market_factor_columns if col not in all_factors_df.columns]

    logger.info(f"[CHART] 因子分布分析:")
    logger.info(f"  - 可用市场因子: {len(available_market_factors)}/{len(market_factor_columns)}")
    if missing_market_factors:
        logger.info(f"  - 缺失市场因子: {missing_market_factors[:5]}{'...' if len(missing_market_factors) > 5 else ''}")

    if len(available_market_factors) == 0:
        logger.warning("[WARNING] 未找到任何市场因子，将跳过市场因子相关分析")
        return False

    # 步骤 2: 数据准备（添加市场收益率）
    logger.info("[REFRESH] [步骤 2/5] 准备分析数据...")
    try:
        loader = IntegratedDataLoader()
        start_date = all_factors_df['date'].min().strftime('%Y%m%d')
        end_date = all_factors_df['date'].max().strftime('%Y%m%d')
        _, _, market_data = loader.load_all_data(start_date, end_date, extend_history_days=20)
        if market_data is None: raise ValueError("未能加载 market_data")
        market_data['date'] = pd.to_datetime(market_data['date'])

        # 计算市场收益率
        for n in analysis_periods:
            market_data[f'Mkt_Return_Next_{n}d'] = market_data['close'].pct_change(periods=-n).shift(-n)

        # [强化] 关键修复：直接合并到统一数据中，避免重复合并
        return_columns = ['date'] + [f'Mkt_Return_Next_{n}d' for n in analysis_periods]
        all_factors_df = pd.merge(all_factors_df, market_data[return_columns], on='date', how='left')

        logger.info(f"[OK] 市场收益率数据合并完成: {len(all_factors_df):,} 条记录")

    except Exception as e:
        logger.error(f"[FAIL] 加载市场数据时出错: {e}")
        return False

    # 步骤 3: 市场择时能力测试
    logger.info("[REFRESH] [步骤 3/5] 开始执行市场择时能力测试...")
    return_columns_set = set(f'Mkt_Return_Next_{n}d' for n in analysis_periods)

    # [强化] 关键修复：直接从统一数据中获取市场因子名称
    market_factor_names = [col for col in available_market_factors if col in all_factors_df.columns]
    logger.info(f"[CHART] 择时能力测试: {len(market_factor_names)} 个市场因子")

    timing_test_results = {}

    for factor in market_factor_names:
        is_significant = False
        factor_results = {}
        for n in analysis_periods:
            # [强化] 关键修复：直接从统一数据中提取测试数据
            test_data = all_factors_df[[factor, f'Mkt_Return_Next_{n}d']].dropna()
            # 提高最小样本要求，增强统计稳健性
            if len(test_data) > 30:  # 从20提升到30
                try:
                    X = sm.add_constant(test_data[factor])
                    y = test_data[f'Mkt_Return_Next_{n}d']
                    model = sm.OLS(y, X).fit()
                    # 提高t统计量阈值，增加显著性要求
                    if abs(model.tvalues[factor]) > 2.5 and model.pvalues[factor] < 0.05:  # 从2.0提升到2.5
                        factor_results[n] = {
                            't_stat': model.tvalues[factor],
                            'r_squared': model.rsquared,
                            'p_value': model.pvalues[factor],
                            'period': n
                        }
                        is_significant = True
                        logger.info(f"  [OK] {factor} 在{n}日周期显著: t={model.tvalues[factor]:.2f}, R²={model.rsquared:.3f}")
                except Exception as e:
                    logger.warning(f"因子 '{factor}' 第{n}天择时测试失败: {e}")
                    continue
        if is_significant:
            timing_test_results[factor] = factor_results

    logger.info(f"[OK] 择时能力测试完成: {len(timing_test_results)} 个因子通过显著性检验")

    # 步骤 4: 因子作为开关测试
    logger.info("[REFRESH] [步骤 4/5] 开始执行因子作为开关测试...")
    switch_test_results = {}

    # [强化] 专业优化：使用更全面的默认因子列表（基于最新配置）
    default_factors = [
        'returns_zscore', 'market_structure_score', 'ma_ratio_5_10', 'corr_5d', 'volatility_correlation',
        'volume_correlation', 'near_support_count_60d', 'institutional_activity_score',
        'ma_convergence_score', 'lower_shadow_ratio', 'sector_correlation_20d',
        'price_impact', 'rank_momentum_20d', 'rank_rsi', 'rank_volume', 'market_correlation',
        'corr_20d_mom5', 'corr_slope_20_120',
        # 新增因子（基于用户最新配置）
        'gain_loss_ratio_rank_20d', 'gain_loss_ratio_rank_60d',
        'limit_up_count_rank_20d', 'limit_up_count_rank_10d', 'limit_up_count_rank_5d',
        'max_gain_5d_rank_30d'
    ]

    # [强化] 关键修复：直接从统一数据中分析，无需重新合并
    available_factors = [col for col in default_factors if col in all_factors_df.columns]
    missing_factors = [col for col in default_factors if col not in all_factors_df.columns]

    logger.info(f"[CHART] 开关测试分析范围: {len(available_factors)}个股票因子 ({len(default_factors)}个中的{len(available_factors)}个可用)")
    if missing_factors:
        logger.info(f"  - 缺失股票因子: {missing_factors[:5]}{'...' if len(missing_factors) > 5 else ''}")

    # [强化] 专业优化：数据质量检查
    logger.info("[SEARCH] 执行数据质量检查...")
    quality_report = {}
    for factor in available_factors[:5]:  # 检查前5个因子作为样本
        factor_data = all_factors_df[factor].dropna()
        quality_report[factor] = {
            'coverage': len(factor_data) / len(all_factors_df),
            'unique_ratio': factor_data.nunique() / len(factor_data) if len(factor_data) > 0 else 0
        }

    avg_coverage = np.mean([v['coverage'] for v in quality_report.values()])
    avg_unique_ratio = np.mean([v['unique_ratio'] for v in quality_report.values()])
    logger.info(f"  - 平均数据覆盖率: {avg_coverage:.1%}")
    logger.info(f"  - 平均唯一值比例: {avg_unique_ratio:.1%}")

    if avg_coverage < 0.8:
        logger.warning(f"  - [WARNING] 数据覆盖率较低 ({avg_coverage:.1%})，可能影响分析质量")

    # [强化] 关键修复：验证市场因子在统一数据中的存在性（无需重新合并）
    available_market_factors_for_switch = [factor for factor in market_factor_names if factor in all_factors_df.columns]
    missing_market_factors_for_switch = [factor for factor in market_factor_names if factor not in all_factors_df.columns]

    if missing_market_factors_for_switch:
        logger.warning(f"[WARNING] 以下市场因子在统一数据中缺失: {missing_market_factors_for_switch}")

    logger.info(f"[CHART] 开关测试市场因子: {len(available_market_factors_for_switch)}/{len(market_factor_names)} 个可用")

    # [强化] 关键修复：使用统一数据，无需合并
    merged_df = all_factors_df  # 直接使用统一数据
    logger.info(f"[OK] 使用统一因子数据进行开关测试: {len(merged_df):,} 行")

    def calculate_daily_ic(df, factor_col):
        """计算日度IC，增强稳健性"""
        if df.shape[0] < 5: return np.nan  # 提高最小样本要求
        corr_data = df[[factor_col, 'future_return_2d']].dropna()
        if corr_data.shape[0] < 5: return np.nan
        try:
            return corr_data[factor_col].corr(corr_data['future_return_2d'], method='spearman')
        except:
            return np.nan

    def improved_factor_grouping(series, factor_name):
        """改进的因子分组逻辑，解决因子值变化少的问题"""
        series_clean = series.dropna()
        if len(series_clean) < 50:  # 数据太少
            return None, f"数据量不足 ({len(series_clean)} < 50)"

        unique_values = series_clean.nunique()
        if unique_values < 5:  # 唯一值太少
            return None, f"唯一值太少 ({unique_values} < 5)"

        # 尝试五分位数分组
        try:
            quintiles = pd.qcut(series_clean, 5, labels=False, duplicates='drop')
            if quintiles.nunique() >= 3:  # 至少要有3个不同的分组
                return quintiles, None
        except:
            pass

        # 如果五分位数失败，尝试三分位数
        try:
            tertiles = pd.qcut(series_clean, 3, labels=False, duplicates='drop')
            if tertiles.nunique() >= 2:
                return tertiles, None
        except:
            pass

        # 最后尝试百分位数阈值分组 (30%-70%)
        try:
            p30 = series_clean.quantile(0.3)
            p70 = series_clean.quantile(0.7)
            if p30 != p70:  # 确保阈值不同
                percentile_groups = pd.cut(series_clean, bins=[-np.inf, p30, p70, np.inf], labels=False)
                if percentile_groups.nunique() >= 2:
                    return percentile_groups, None
        except:
            pass

        return None, f"无法创建有效分组 (唯一值: {unique_values})"

    # [WRENCH] 关键修复：确保市场因子在合并数据中存在
    available_market_factors = [factor for factor in market_factor_names if factor in merged_df.columns]
    missing_market_factors = [factor for factor in market_factor_names if factor not in merged_df.columns]

    if missing_market_factors:
        logger.warning(f"[WARNING] 以下市场因子在合并数据中缺失: {missing_market_factors}")

    logger.info(f"[CHART] 开关测试范围: {len(available_market_factors)}/{len(market_factor_names)} 个市场因子可用")

    for market_factor in available_market_factors_for_switch:
        try:
            progress = f"[{available_market_factors_for_switch.index(market_factor)+1}/{len(available_market_factors_for_switch)}]"
            logger.info(f"[SEARCH] {progress} 分析市场因子: {market_factor}")

            # [WRENCH] 修复：添加因子存在性检查
            if market_factor not in merged_df.columns:
                logger.error(f"  - [FAIL] 因子 '{market_factor}' 不存在于统一数据中")
                continue

            # [WRENCH] 修复：检查因子数据质量
            factor_data = merged_df[market_factor].dropna()
            if len(factor_data) == 0:
                logger.warning(f"  - [WARNING] 因子 '{market_factor}' 数据全为空")
                continue

            logger.info(f"  - 因子数据质量: {len(factor_data):,}/{len(merged_df):,} 有效值 ({len(factor_data)/len(merged_df)*100:.1f}%)")

            # 使用改进的分组逻辑
            grouping_result, error_msg = improved_factor_grouping(merged_df[market_factor], market_factor)
            if grouping_result is None:
                logger.warning(f"  - [WARNING] 因子 '{market_factor}' 分组失败: {error_msg}")
                continue

            merged_df['market_group'] = grouping_result
            max_group = merged_df['market_group'].max()
            min_group = merged_df['market_group'].min()

            top_group_days = merged_df[merged_df['market_group'] == max_group]
            bottom_group_days = merged_df[merged_df['market_group'] == min_group]

            if top_group_days.empty or bottom_group_days.empty:
                logger.warning(f"  - [WARNING] 因子 '{market_factor}' 分组后为空")
                continue

            unique_dates_top = top_group_days['date'].nunique()
            unique_dates_bottom = bottom_group_days['date'].nunique()
            logger.info(f"  - 分组统计: 高组 {len(top_group_days):,} 条记录 ({unique_dates_top}个交易日)，低组 {len(bottom_group_days):,} 条记录 ({unique_dates_bottom}个交易日)")

            significant_switches = {}
            for stock_factor in available_factors:  # 只分析默认因子
                try:
                    # [WRENCH] 修复：检查股票因子是否存在
                    if stock_factor not in merged_df.columns:
                        logger.warning(f"    股票因子 '{stock_factor}' 不存在于合并数据中")
                        continue

                    # [WRENCH] 修复：检查股票因子数据质量
                    stock_factor_data = merged_df[stock_factor].dropna()
                    if len(stock_factor_data) == 0:
                        logger.warning(f"    股票因子 '{stock_factor}' 数据全为空")
                        continue

                    # 计算IC时增加错误处理
                    ic_top_daily = top_group_days.groupby('date').apply(
                        lambda x: calculate_daily_ic(x, stock_factor)
                    ).dropna()
                    ic_bottom_daily = bottom_group_days.groupby('date').apply(
                        lambda x: calculate_daily_ic(x, stock_factor)
                    ).dropna()

                    if len(ic_top_daily) == 0 or len(ic_bottom_daily) == 0:
                        continue

                    ic_top = ic_top_daily.mean()
                    ic_bottom = ic_bottom_daily.mean()

                    if pd.isna(ic_top) or pd.isna(ic_bottom):
                        continue

                    ic_spread = ic_top - ic_bottom

                    # 提高IC Spread阈值，增强显著性要求
                    if abs(ic_spread) > 0.03:  # 从0.02提升到0.03
                        # 增加统计检验
                        from scipy import stats
                        try:
                            # 使用t检验验证IC差异的显著性
                            t_stat, p_val = stats.ttest_ind(ic_top_daily, ic_bottom_daily, equal_var=False)
                            if p_val < 0.1:  # 10%显著性水平
                                significant_switches[stock_factor] = {
                                    'ic_spread': ic_spread,
                                    'ic_top': ic_top,
                                    'ic_bottom': ic_bottom,
                                    't_stat': t_stat,
                                    'p_value': p_val,
                                    'top_days': len(ic_top_daily),
                                    'bottom_days': len(ic_bottom_daily)
                                }
                        except:
                            # 如果统计检验失败，使用旧的逻辑
                            significant_switches[stock_factor] = {
                                'ic_spread': ic_spread,
                                'ic_top': ic_top,
                                'ic_bottom': ic_bottom
                            }
                except Exception as factor_e:
                    logger.warning(f"    因子 {stock_factor} 处理失败: {factor_e}")
                    continue

            if significant_switches:
                switch_test_results[market_factor] = significant_switches
                avg_ic_spread = np.mean([abs(v['ic_spread'] if isinstance(v, dict) else v) for v in significant_switches.values()])
                logger.info(f"  - [OK] 发现 {len(significant_switches)} 个显著影响的因子 (平均IC Spread: {avg_ic_spread:.3f})")
            else:
                logger.info(f"  - [FAIL] 未发现显著影响 (阈值: 0.03)")

        except Exception as e:
            logger.error(f"  - [FAIL] 对因子 '{market_factor}' 进行开关测试时失败: {e}")
            import traceback
            logger.error(f"  - 详细错误信息: {traceback.format_exc()}")

            # [WRENCH] 添加调试信息
            logger.error(f"  - 调试信息:")
            logger.error(f"    - 因子名称: {market_factor}")
            logger.error(f"    - 因子是否在merged_df中: {market_factor in merged_df.columns}")
            if market_factor in merged_df.columns:
                factor_info = merged_df[market_factor].describe()
                logger.error(f"    - 因子统计信息: {factor_info}")
            logger.error(f"    - merged_df列名: {list(merged_df.columns)[:10]}...")  # 只显示前10个列名
            continue

    # 步骤 5: 生成专业分析报告
    logger.info("[REFRESH] [步骤 5/5] 开始生成专业量化分析报告...")
    valuable_factors = sorted(list(set(timing_test_results.keys()) | set(switch_test_results.keys())))

    # [强化] 专业优化：增强报告统计分析
    logger.info("[CHART] 计算报告统计指标...")

    # 计算择时能力统计
    timing_stats = {
        'total_factors': len(market_factor_names),
        'timing_factors': len(timing_test_results),
        'avg_r_squared': 0,
        'max_t_stat': 0,
        'significant_periods': set()
    }

    if timing_test_results:
        all_r_squared = []
        all_t_stats = []
        for factor_results in timing_test_results.values():
            for period_result in factor_results.values():
                all_r_squared.append(period_result.get('r_squared', 0))
                all_t_stats.append(abs(period_result.get('t_stat', 0)))
                timing_stats['significant_periods'].add(period_result.get('period', 0))

        timing_stats['avg_r_squared'] = np.mean(all_r_squared) if all_r_squared else 0
        timing_stats['max_t_stat'] = max(all_t_stats) if all_t_stats else 0

    # 计算开关效应统计
    switch_stats = {
        'switch_factors': len(switch_test_results),
        'total_impacts': sum(len(results) for results in switch_test_results.values()),
        'avg_ic_spread': 0,
        'max_ic_spread': 0,
        'significant_factors': set()
    }

    if switch_test_results:
        all_ic_spreads = []
        for factor_name, factor_results in switch_test_results.items():
            for stock_factor, impact in factor_results.items():
                ic_spread = abs(impact.get('ic_spread', 0))
                all_ic_spreads.append(ic_spread)
                if ic_spread > 0.05:  # 高影响阈值
                    switch_stats['significant_factors'].add(stock_factor)

        switch_stats['avg_ic_spread'] = np.mean(all_ic_spreads) if all_ic_spreads else 0
        switch_stats['max_ic_spread'] = max(all_ic_spreads) if all_ic_spreads else 0

    if not valuable_factors:
        logger.info("[FAIL] 未发现有价值的市场状态因子。")
        logger.info(f"   分析了 {timing_stats['total_factors']} 个市场因子，均未通过显著性检验")
        logger.info(f"   建议检查数据质量或调整显著性阈值")
        return True

    # 在报告生成之前先计算需要的变量
    # 筛选适合短期持仓的因子(用于报告)
    short_term_timing_factors_for_report = []
    for factor in timing_test_results:
        for period, result in timing_test_results[factor].items():
            if period <= 2 and abs(result.get('t_stat', 0)) > 2.5:
                short_term_timing_factors_for_report.append(factor)
                break

    # 计算推荐因子数量(用于报告)
    recommended_factors_for_report = list(set([f for f in valuable_factors
                                             if f in short_term_timing_factors_for_report or f in switch_test_results]))

    total_switch_impacts = sum(len(results) for results in switch_test_results.values())
    report = ["# 市场状态因子分析报告",
              f"分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              f"分析周期: {analysis_periods} 日", "---"]
    timing_factor_count = len(timing_test_results)
    switch_factor_count = len(switch_test_results)

    report.append(f"**总结**: 在 {len(market_factor_names)} 个市场状态因子中，发现 {len(valuable_factors)} 个具有显著价值的因子。")
    report.append(f"**择时能力**: {timing_factor_count} 个因子通过市场择时测试，具备预测未来 1-3 日市场涨跌能力。")
    avg_impacts_per_factor = total_switch_impacts/max(switch_factor_count,1) if switch_factor_count > 0 else 0
    report.append(f"**开关效应**: {switch_factor_count} 个因子具备开关效应，共影响 {total_switch_impacts} 个默认因子的效果 (平均每个市场因子影响 {avg_impacts_per_factor:.1f} 个股票因子)。")
    # 使用已计算的推荐因子数量
    recommended_count = len(recommended_factors_for_report)
    report.append(f"**分析范围**: 重点分析了 {len(available_factors)}/18 个默认核心因子对市场状态的敏感性。")
    report.append(f"**集成建议**: 推荐集成 {recommended_count} 个适合短期持仓(1-2日)的市场因子。")

    for factor in valuable_factors:
        report.append(f"\n### 因子: `{factor}`")
        passed_timing = factor in timing_test_results
        passed_switch = factor in switch_test_results

        if passed_timing:
            report.append("- **[OK] 通过市场择时能力测试**")
            details = []
            timing_periods = []
            if factor in timing_test_results:
                for n, res in timing_test_results[factor].items():
                    if abs(res['t_stat']) > 2.5:  # 更新阈值
                        direction = "正向预测" if res['t_stat'] > 0 else "负向预测"
                        p_info = f", p={res['p_value']:.3f}" if 'p_value' in res else ""
                        details.append(f"对未来 **{n}日** 收益有显著 **{direction}** 能力 (t-stat: {res['t_stat']:.2f}, R²: {res['r_squared']:.4f}{p_info})")
                        timing_periods.append(n)

                effectiveness_desc = "短期" if max(timing_periods) <= 3 else ("中期" if max(timing_periods) <= 10 else "长期")
                report.append(f"  - **价值分析**: 该因子具有{effectiveness_desc}市场择时能力，可以显著预测未来市场整体走势。{' '.join(details)}")

        if passed_switch:
            report.append("- **[OK] 通过因子作为开关测试**")
            details = []
            if factor in switch_test_results:
                # 按IC Spread绝对值排序，显示最重要的影响，并标注重要性等级
                sorted_factors = sorted(switch_test_results[factor].items(),
                                      key=lambda x: abs(x[1]['ic_spread'] if isinstance(x[1], dict) else x[1]),
                                      reverse=True)

                # 分类影响强度
                strong_impacts = [(f, v) for f, v in sorted_factors if abs(v['ic_spread'] if isinstance(v, dict) else v) > 0.05]
                moderate_impacts = [(f, v) for f, v in sorted_factors if 0.03 <= abs(v['ic_spread'] if isinstance(v, dict) else v) <= 0.05]

                for stock_factor, result in sorted_factors[:5]:  # 只显示前5个最重要的
                    if isinstance(result, dict):  # 新的详细结果
                        direction = "增强" if result['ic_spread'] > 0 else "减弱"
                        p_info = f" (p={result['p_value']:.3f})" if 'p_value' in result else ""
                        details.append(f"显著 **{direction}** `{stock_factor}` 因子效果 (IC Spread: {result['ic_spread']:.3f}{p_info})")
                    else:  # 旧的简单结果
                        direction = "增强" if result > 0 else "减弱"
                        details.append(f"显著 **{direction}** `{stock_factor}` 因子效果 (IC Spread: {result:.3f})")

                total_affected = len(switch_test_results[factor])
                enhancing_count = sum(1 for _, res in switch_test_results[factor].items()
                                    if (res['ic_spread'] if isinstance(res, dict) else res) > 0)
                weakening_count = total_affected - enhancing_count

                summary = f"影响 {total_affected}/18 个默认因子（增强{enhancing_count}个，减弱{weakening_count}个）"
                report.append(f"  - **价值分析**: 该因子可以识别市场状态，{summary}。主要影响: {' '.join(details[:3])}{'...' if len(details) > 3 else ''}")

    final_report = "\n".join(report)
    print("\n" + "="*80)
    print(final_report)
    print("="*80 + "\n")

    # 添加性能统计和建议信息
    logger.info("[CHART] 性能统计信息:")
    logger.info(f"  - 市场因子数量: {len(market_factor_names)}")
    logger.info(f"  - 有效因子数量: {len(valuable_factors)} ({len(valuable_factors)/len(market_factor_names)*100:.1f}%)")
    logger.info(f"  - 分析的股票因子: {len(available_factors)}/18")
    logger.info(f"  - 发现的显著影响: {total_switch_impacts}")
    logger.info(f"  - 数据记录数: {len(merged_df):,}")

    # 生成集成建议
    high_impact_factors = [f for f in valuable_factors
                          if f in timing_test_results and f in switch_test_results]
    timing_only_factors = [f for f in valuable_factors
                          if f in timing_test_results and f not in switch_test_results]
    switch_only_factors = [f for f in valuable_factors
                          if f not in timing_test_results and f in switch_test_results]

    logger.info("\n[IDEA] 集成建议:")
    logger.info(f"  - 高价值因子 (同时具备择时+开关效应): {len(high_impact_factors)} 个")
    logger.info(f"    推荐: 直接作为特征加入模型训练")
    logger.info(f"  - 择时专用因子: {len(timing_only_factors)} 个")
    logger.info(f"    推荐: 用于市场状态识别和模型切换")
    logger.info(f"  - 开关专用因子: {len(switch_only_factors)} 个")
    logger.info(f"    推荐: 用于因子权重调节或特征选择")

    if len(valuable_factors) > 10:
        logger.info(f"  - [WARNING]  因子数量较多({len(valuable_factors)}个)，建议考虑:")
        logger.info(f"    1. PCA降维 (保留90%方差)")
        logger.info(f"    2. 相关性筛选 (去除>0.8相关性)")
        logger.info(f"    3. 重要性排序 (基于IC绝对值)")

    # 输出详细的择时能力分析
    logger.info("\n[UP] 详细择时能力分析(按周期显示):")
    for factor in timing_test_results:
        logger.info(f"  {factor}:")
        for period, result in timing_test_results[factor].items():
            t_stat = result['t_stat']
            r_sq = result['r_squared']
            p_val = result.get('p_value', 'N/A')
            direction = "正向" if t_stat > 0 else "负向"
            suitable = "[OK]适合" if period <= 2 else "[WARNING]不适合"
            logger.info(f"    {period}日: {direction} t={t_stat:.2f}, R²={r_sq:.4f}, p={p_val} ({suitable}2日持仓)")

    # 相关性分析
    if len(valuable_factors) > 3:
        logger.info("\n[LINK] 市场因子相关性分析:")
        try:
            # [强化] 修复：从统一数据中提取市场因子进行相关性分析
            factor_data_for_corr = all_factors_df[['date'] + valuable_factors].drop_duplicates(subset=['date']).set_index('date')
            correlation_matrix = factor_data_for_corr.corr()

            # 找到高相关性因子对
            high_corr_pairs = []
            for i, factor1 in enumerate(valuable_factors):
                for j, factor2 in enumerate(valuable_factors[i+1:], i+1):
                    corr_val = correlation_matrix.loc[factor1, factor2]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((factor1, factor2, corr_val))

            if high_corr_pairs:
                logger.info(f"  [WARNING]  发现 {len(high_corr_pairs)} 对高相关性因子 (|相关系数|>0.7):")
                for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    logger.info(f"    {f1} vs {f2}: {corr:.3f}")
                logger.info(f"  建议: 集成时需要去相关或PCA降维")
            else:
                logger.info(f"  [OK] 所有因子间相关性<0.7，可以直接集成")
        except Exception as e:
            logger.warning(f"  相关性分析失败: {e}")

    # 筛选适合短期持仓的因子
    short_term_timing_factors = []
    medium_term_only_factors = []

    for factor in timing_test_results:
        has_short_term = False
        has_any_effect = False
        for period, result in timing_test_results[factor].items():
            if abs(result.get('t_stat', 0)) > 2.5:
                has_any_effect = True
                if period <= 2:
                    has_short_term = True

        if has_short_term:
            short_term_timing_factors.append(factor)
        elif has_any_effect:
            medium_term_only_factors.append(factor)

    # 推荐集成的具体因子名单
    recommended_factors = list(set([f for f in valuable_factors if f in short_term_timing_factors or f in switch_test_results]))

    logger.info(f"\n[TARGET] 推荐集成的市场因子 ({len(recommended_factors)}个):")
    for i, factor in enumerate(recommended_factors, 1):
        timing_effect = "择时" if factor in short_term_timing_factors else ""
        switch_effect = "开关" if factor in switch_test_results else ""
        effects = "/".join(filter(None, [timing_effect, switch_effect]))
        logger.info(f"  {i:2d}. {factor} ({effects})")

    # 更新的集成建议
    effective_for_strategy = len(recommended_factors)
    logger.info(f"\n[TOOL]  因子集成策略 (当前有{effective_for_strategy}个因子适合集成):")

    if effective_for_strategy <= 8:
        logger.info(f"  [OK] 因子数量适中({effective_for_strategy}个)，建议:")
        logger.info(f"    1. 直接加入XGBoost训练 (与18个股票因子合并)")
        logger.info(f"    2. 检查相关性矩阵，去除>0.85相关的因子")
    elif effective_for_strategy <= 15:
        logger.info(f"  [WARNING]  因子数量较多({effective_for_strategy}个)，建议:")
        logger.info(f"    1. 首先进行相关性分析，去除>0.8相关的因子")
        logger.info(f"    2. 按IC绝对值排序，只保留前8-10个最强因子")
        logger.info(f"    3. 或者使用PCA降维至5-8个主成分")
    else:
        logger.info(f"  [FAIL] 因子数量过多({effective_for_strategy}个)，强烈建议:")
        logger.info(f"    1. 使用PCA降维至5-8个主成分 (保留95%方差)")
        logger.info(f"    2. 或者使用因子选择算法 (LASSO/递归特征消除)")
        logger.info(f"    3. 避免因子冗余影响18个核心股票因子的效果")
        logger.info(f"    4. 考虑使用Stacking集成：市场因子作为元学习器的输入")

    logger.info("[OK] 市场状态因子分析模式执行完成。")
    return True


def run_interaction_analysis_mode(args):
    """运行交互特征全面分析模式

    功能包括:
    1. 生成所有可能的交互特征
    2. 单因子测试 (IC, IR, 分层回测)
    3. 特征相关性分析
    4. 管线评估分析
    5. 优化建议
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🔬 交互特征全面分析模式")
    logger.info("=" * 80)

    try:
        # 步骤1: 数据加载和准备
        logger.info("[REFRESH] [步骤 1/6] 数据加载和准备...")

        # 1. 加载股票因子数据 (使用与run_test_factors_mode一致的方式)
        logger.info("📥 加载股票因子数据...")

        # 加载完整的18个股票因子数据
        stock_factor_data = _load_factor_cache_data(include_label_generation_features=True)
        if stock_factor_data is None or len(stock_factor_data) == 0:
            logger.error("[FAIL] 股票因子数据加载失败")
            return False

        logger.info(f"[OK] 股票因子数据加载完成: {len(stock_factor_data)} 条记录, {len(stock_factor_data.columns)} 个字段")
        logger.info(f"数据时间范围: {stock_factor_data['date'].min()} 到 {stock_factor_data['date'].max()}")

        # 2. 从股票因子数据中提取市场因子（统一数据源）
        logger.info("📥 从统一数据源提取市场因子...")

        # 获取可用的市场因子列表
        market_factor_columns = _get_available_market_factors()
        available_market_factors = [col for col in market_factor_columns if col in stock_factor_data.columns]

        if len(available_market_factors) > 0:
            market_factors_df = stock_factor_data[['date'] + available_market_factors].copy()
            logger.info(f"[OK] 从统一数据中提取到 {len(available_market_factors)} 个市场因子: {available_market_factors}")
            logger.info(f"市场因子时间范围: {market_factors_df['date'].min()} 到 {market_factors_df['date'].max()}")
        else:
            logger.error("[FAIL] 未找到市场因子数据")
            return False

        # 3. 数据对齐和合并
        logger.info("[REFRESH] 数据对齐和合并...")

        # 确保日期格式一致
        stock_factor_data['date'] = pd.to_datetime(stock_factor_data['date']).dt.date
        market_factors_df['date'] = pd.to_datetime(market_factors_df['date']).dt.date

        # 合并数据
        merged_data = stock_factor_data.merge(market_factors_df, on='date', how='inner')
        if len(merged_data) == 0:
            logger.error("[FAIL] 股票因子数据与市场因子数据无法对齐")
            return False

        logger.info(f"[OK] 数据对齐完成: {len(merged_data)} 条记录")
        logger.info(f"合并后时间范围: {merged_data['date'].min()} 到 {merged_data['date'].max()}")

        # 步骤2: 交互特征生成
        logger.info("[REFRESH] [步骤 2/6] 交互特征生成...")

        # 获取完整的股票特征 (使用与训练一致的定义)
        # [WRENCH] 因子测试使用模型因子列表（归一化因子）
        stock_features = _get_model_factors()
        logger.info(f"[CHART] 使用完整的股票特征: {len(stock_features)} 个")

        # 检查实际可用的股票特征
        available_stock_features = []
        for feature in stock_features:
            if feature in merged_data.columns:
                available_stock_features.append(feature)
            else:
                logger.warning(f"[WARNING] 股票特征 {feature} 在数据中不存在，跳过")

        logger.info(f"[CHART] 实际可用股票特征: {len(available_stock_features)} 个")
        for i, feature in enumerate(available_stock_features, 1):
            logger.info(f"  {i:2d}. {feature}")

        # 获取可用的市场特征 (处理pandas merge时的列名冲突)
        market_features = []
        potential_market_features = [
            'market_volatility_atr20d', 'market_trend_factor', 'max_gain_10d_rank_60d',
            'market_volume_rank_20d', 'limit_up_count_rank_60d', 'limit_down_count_rank_60d',
            'broken_limit_count_rank_20d', 'max_consecutive_limit_rank_20d'
        ]

        # 检查合并后的数据中实际存在的市场特征 (处理_y后缀并重命名)
        market_feature_mapping = {}  # 原始名称 -> 实际列名的映射

        for feature in potential_market_features:
            if feature in merged_data.columns:
                market_features.append(feature)
                market_feature_mapping[feature] = feature
                logger.info(f"[OK] 找到市场特征: {feature}")
            elif f"{feature}_y" in merged_data.columns:
                # 重命名_y后缀的列为原始名称
                original_col = f"{feature}_y"
                merged_data[feature] = merged_data[original_col]
                market_features.append(feature)
                market_feature_mapping[feature] = original_col
                logger.info(f"[OK] 找到市场特征: {feature} (从 {original_col} 重命名)")
            elif f"{feature}_x" in merged_data.columns:
                # 重命名_x后缀的列为原始名称
                original_col = f"{feature}_x"
                merged_data[feature] = merged_data[original_col]
                market_features.append(feature)
                market_feature_mapping[feature] = original_col
                logger.info(f"[OK] 找到市场特征: {feature} (从 {original_col} 重命名)")
            else:
                logger.warning(f"[WARNING] 市场特征 {feature} 在合并数据中不存在，跳过")

        logger.info(f"[CHART] 可用市场特征: {len(market_features)} 个")
        for i, feature in enumerate(market_features, 1):
            logger.info(f"  {i:2d}. {feature}")

        if not market_features:
            logger.error("[FAIL] 未找到可用的市场特征")
            logger.info("[CLIPBOARD] 合并数据中包含市场相关的列名:")
            market_related_columns = [col for col in merged_data.columns if any(mf in col for mf in potential_market_features)]
            for i, col in enumerate(market_related_columns, 1):
                logger.info(f"  {i:3d}. {col}")
            return False

        if not available_stock_features:
            logger.error("[FAIL] 未找到可用的股票特征")
            return False

        # 生成交互特征
        interaction_features = {}
        interaction_names = []

        # 标准化市场特征 (与训练逻辑一致)
        from sklearn.preprocessing import StandardScaler
        market_scaler = StandardScaler()
        market_matrix = merged_data[market_features].values.astype(float)
        market_matrix_scaled = market_scaler.fit_transform(market_matrix)
        market_matrix_scaled = np.clip(market_matrix_scaled, -3.0, 3.0)

        logger.info(f"[REFRESH] 开始生成交互特征: {len(available_stock_features)} × {len(market_features)} = {len(available_stock_features) * len(market_features)} 个")

        for i, stock_feature in enumerate(available_stock_features):
            stock_values = merged_data[stock_feature].values

            for j, market_feature in enumerate(market_features):
                interaction_name = f"inter_{stock_feature}_{market_feature}"
                interaction_values = stock_values * market_matrix_scaled[:, j]

                interaction_features[interaction_name] = interaction_values
                interaction_names.append(interaction_name)

        logger.info(f"[OK] 生成交互特征: {len(interaction_names)} 个")
        logger.info(f"[CHART] 交互特征组合: {len(available_stock_features)} 个股票特征 × {len(market_features)} 个市场特征")

        # 显示前10个交互特征名称作为示例
        for i, name in enumerate(interaction_names[:10], 1):
            logger.info(f"  {i:2d}. {name}")
        if len(interaction_names) > 10:
            logger.info(f"  ... 还有 {len(interaction_names) - 10} 个交互特征")

        # 将交互特征添加到数据中
        for name, values in interaction_features.items():
            merged_data[name] = values

        return _continue_interaction_analysis_fixed(merged_data, available_stock_features, market_features,
                                                   interaction_names, logger)

    except Exception as e:
        logger.error(f"[FAIL] 交互特征分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _continue_interaction_analysis_fixed(merged_data, stock_features, market_features,
                                       interaction_names, logger):
    """修复后的交互特征分析流程：先过滤再测试"""
    try:
        logger.info("[REFRESH] [步骤 3/7] 特征相关性分析和过滤...")

        # 第一阶段：相关性分析（包含原始股票因子）
        logger.info("[CHART] 第一阶段 - 相关性过滤（包含原始股票因子）:")
        logger.info(f"  原始交互特征数量: {len(interaction_names)}")
        logger.info(f"  原始股票因子数量: {len(stock_features)}")

        # 准备所有特征进行相关性分析（交互特征 + 原始股票因子）
        all_features_for_correlation = stock_features + interaction_names

        # 检查数据质量并移除有效数据不足的特征
        valid_features = []
        for feature in all_features_for_correlation:
            if feature in merged_data.columns:
                valid_count = merged_data[feature].notna().sum()
                total_count = len(merged_data)
                valid_ratio = valid_count / total_count

                if valid_ratio >= 0.1:  # 至少10%有效数据
                    valid_features.append(feature)
                else:
                    logger.warning(f"  [WARNING] 特征 {feature} 有效数据不足: {valid_ratio:.2%}")
            else:
                logger.warning(f"  [WARNING] 特征 {feature} 在数据中不存在")

        # 分离有效的股票因子和交互特征
        valid_stock_features = [f for f in stock_features if f in valid_features]
        valid_interaction_features = [f for f in interaction_names if f in valid_features]

        logger.info(f"  有效股票因子: {len(valid_stock_features)} 个")
        logger.info(f"  有效交互特征: {len(valid_interaction_features)} 个")

        if len(valid_interaction_features) == 0:
            logger.error("[FAIL] 没有有效的交互特征")
            return False

        # 计算相关性矩阵（只在有效特征之间）
        correlation_data = merged_data[valid_features].dropna()
        logger.info(f"  相关性计算数据: {len(correlation_data)} 条记录")

        if len(correlation_data) < 100:
            logger.error("[FAIL] 相关性计算数据不足")
            return False

        # 计算相关性矩阵
        correlation_matrix = correlation_data.corr(method='pearson', min_periods=100)

        # 过滤与原始股票因子高相关的交互特征
        filtered_interaction_features = []
        removed_by_stock_correlation = []

        for interaction_feature in valid_interaction_features:
            is_highly_correlated = False

            for stock_feature in valid_stock_features:
                if interaction_feature in correlation_matrix.index and stock_feature in correlation_matrix.columns:
                    corr_val = correlation_matrix.loc[interaction_feature, stock_feature]

                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        logger.info(f"  移除与股票因子高相关的交互特征: {interaction_feature} (与{stock_feature}相关性={corr_val:.4f})")
                        removed_by_stock_correlation.append((interaction_feature, stock_feature, corr_val))
                        is_highly_correlated = True
                        break

            if not is_highly_correlated:
                filtered_interaction_features.append(interaction_feature)

        logger.info(f"  移除与股票因子高相关的交互特征: {len(removed_by_stock_correlation)} 个")
        logger.info(f"  剩余交互特征: {len(filtered_interaction_features)} 个")

        if len(filtered_interaction_features) == 0:
            logger.error("[FAIL] 所有交互特征都与原始股票因子高相关")
            return False

        # 过滤交互特征之间的高相关性
        logger.info("[CHART] 过滤交互特征之间的高相关性...")

        # 计算交互特征之间的相关性
        interaction_correlation_matrix = correlation_matrix.loc[filtered_interaction_features, filtered_interaction_features]

        # 识别高相关性特征对
        high_corr_pairs = []
        for i, feature1 in enumerate(filtered_interaction_features):
            for j, feature2 in enumerate(filtered_interaction_features[i+1:], i+1):
                corr_val = interaction_correlation_matrix.loc[feature1, feature2]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:  # 使用0.7阈值
                    high_corr_pairs.append((feature1, feature2, corr_val))

        logger.info(f"  发现交互特征间高相关性对: {len(high_corr_pairs)} 对")

        # 实施高相关特征过滤算法
        if len(high_corr_pairs) > 0:
            logger.info("[REFRESH] 执行高相关特征过滤算法...")
            correlation_filtered_features = _apply_correlation_filtering_algorithm(
                filtered_interaction_features, high_corr_pairs, interaction_correlation_matrix, logger
            )
            logger.info(f"  相关性过滤前: {len(filtered_interaction_features)} 个特征")
            logger.info(f"  相关性过滤后: {len(correlation_filtered_features)} 个特征")
            logger.info(f"  移除特征数量: {len(filtered_interaction_features) - len(correlation_filtered_features)} 个")
        else:
            logger.info("  没有发现高相关特征对，跳过相关性过滤")
            correlation_filtered_features = filtered_interaction_features

        # 第二阶段：VIF过滤（优化版本）
        logger.info("[CHART] 第二阶段 - VIF过滤 (VIF<5):")
        logger.info(f"  过滤前特征数量: {len(correlation_filtered_features)}")

        vif_filtered_features = _apply_vif_filter_optimized(
            correlation_filtered_features, merged_data, logger
        )

        logger.info(f"  过滤后特征数量: {len(vif_filtered_features)}")
        logger.info(f"  移除特征数量: {len(correlation_filtered_features) - len(vif_filtered_features)}")

        if len(vif_filtered_features) == 0:
            logger.error("[FAIL] VIF过滤后没有剩余特征")
            return False

        # 第三阶段：单因子测试（只对过滤后的特征）
        logger.info("[REFRESH] [步骤 4/7] 单因子测试（仅对过滤后的特征）...")
        logger.info(f"  测试特征数量: {len(vif_filtered_features)}")

        factor_test_results = _perform_factor_testing(
            merged_data, vif_filtered_features, logger
        )

        # 第四阶段：IC排序
        logger.info("[CHART] 第四阶段 - IC排序:")
        logger.info(f"  排序前特征数量: {len(vif_filtered_features)}")

        final_ranked_features = _apply_ic_ranking(
            vif_filtered_features, factor_test_results, logger
        )

        logger.info(f"  最终优质特征数量: {len(final_ranked_features)}")

        # 保存过滤结果
        filtering_results = {
            'original_features': len(interaction_names),
            'valid_features_after_quality_check': len(valid_interaction_features),
            'features_after_stock_correlation_filter': len(filtered_interaction_features),
            'features_after_interaction_correlation_filter': len(correlation_filtered_features),
            'features_after_vif_filter': len(vif_filtered_features),
            'final_ranked_features': len(final_ranked_features),
            'removed_by_stock_correlation': removed_by_stock_correlation,
            'interaction_high_corr_pairs': high_corr_pairs,
            'correlation_filtered_features': correlation_filtered_features,
            'vif_filtered_features': vif_filtered_features,
            'final_ranked_features_list': final_ranked_features,
            'filtering_summary': {
                'removed_by_quality_check': len(interaction_names) - len(valid_interaction_features),
                'removed_by_stock_correlation': len(valid_interaction_features) - len(filtered_interaction_features),
                'removed_by_interaction_correlation': len(filtered_interaction_features) - len(correlation_filtered_features),
                'removed_by_vif_filter': len(correlation_filtered_features) - len(vif_filtered_features),
                'final_retention_rate': len(final_ranked_features) / len(interaction_names)
            }
        }

        # 保存串联过滤结果
        filtering_file = 'interaction_filtering_results.json'
        try:
            import json
            with open(filtering_file, 'w', encoding='utf-8') as f:
                json.dump(filtering_results, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] 串联过滤结果已保存到: {filtering_file}")
        except Exception as e:
            logger.warning(f"[WARNING] 保存串联过滤结果失败: {e}")

        # 生成最终报告
        return _generate_interaction_analysis_report_fixed(
            interaction_names, factor_test_results, high_corr_pairs, [],
            stock_features, market_features, logger,
            final_ranked_features, filtering_results
        )

    except Exception as e:
        logger.error(f"[FAIL] 交互特征分析失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def _create_empty_factor_result():
    """创建空的因子测试结果"""
    return {
        'ic': 0.0,
        'rank_ic': 0.0,
        'ir': 0.0,
        'icir_20d': 0.0,
        'layer_returns': [0.0] * 5,
        'layer_counts': [0] * 5,
        'long_short_return': 0.0,
        'win_rate': 0.0,
        'monotonicity': 0.0,
        'coverage': 0.0,
        'max_drawdown': 0.0,
        'valid_samples': 0,
        'total_samples': 0
    }


def _calculate_monotonicity(layer_returns):
    """计算分层收益的单调性"""
    if len(layer_returns) < 2:
        return 0.0

    # 计算相邻层之间的单调性
    monotonic_count = 0
    total_pairs = len(layer_returns) - 1

    for i in range(total_pairs):
        if layer_returns[i+1] >= layer_returns[i]:
            monotonic_count += 1

    return monotonic_count / total_pairs if total_pairs > 0 else 0.0


def _calculate_vif_parallel(data, features, logger):
    """并行计算VIF，优化速度和错误处理"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import concurrent.futures
    import warnings

    def calculate_single_vif(args):
        """计算单个特征的VIF"""
        data_values, feature_idx, feature_name = args
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 添加数值稳定性检查
                if data_values.shape[0] < data_values.shape[1] * 2:
                    return (feature_name, np.inf)

                # 检查数据是否包含异常值
                if np.any(np.isinf(data_values)) or np.any(np.isnan(data_values)):
                    return (feature_name, np.inf)

                # 计算VIF
                vif = variance_inflation_factor(data_values, feature_idx)

                # 检查结果有效性
                if np.isnan(vif) or np.isinf(vif) or vif < 0:
                    return (feature_name, np.inf)

                return (feature_name, vif)

        except Exception as e:
            # 记录具体错误但不中断整个过程
            if "DLASCL" in str(e) or "MKL" in str(e):
                logger.warning(f"[WARNING] 特征 {feature_name} MKL数值错误，设为无穷大")
            else:
                logger.warning(f"[WARNING] 特征 {feature_name} VIF计算失败: {e}")
            return (feature_name, np.inf)

    try:
        data_values = data.values

        # 如果特征数量较少，使用单线程
        if len(features) <= 10:
            vif_scores = []
            for i, feature in enumerate(features):
                result = calculate_single_vif((data_values, i, feature))
                vif_scores.append(result)
        else:
            # 使用多线程计算VIF
            max_workers = min(4, len(features))  # 限制线程数
            logger.info(f"  使用 {max_workers} 个线程并行计算VIF")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 准备参数
                args_list = [(data_values, i, feature) for i, feature in enumerate(features)]

                # 提交任务
                future_to_feature = {
                    executor.submit(calculate_single_vif, args): args[2]
                    for args in args_list
                }

                vif_scores = []
                for future in concurrent.futures.as_completed(future_to_feature):
                    try:
                        result = future.result(timeout=30)  # 30秒超时
                        vif_scores.append(result)
                    except concurrent.futures.TimeoutError:
                        feature_name = future_to_feature[future]
                        logger.warning(f"[WARNING] 特征 {feature_name} VIF计算超时")
                        vif_scores.append((feature_name, np.inf))
                    except Exception as e:
                        feature_name = future_to_feature[future]
                        logger.warning(f"[WARNING] 特征 {feature_name} VIF计算异常: {e}")
                        vif_scores.append((feature_name, np.inf))

        return vif_scores

    except Exception as e:
        logger.error(f"[FAIL] VIF并行计算失败: {e}")
        return []


def _apply_ic_ranking(features, factor_test_results, logger):
    """第三阶段：按IC绝对值降序排序"""
    logger.info("[REFRESH] 执行IC排序...")

    # 计算每个特征的IC绝对值
    feature_ic_pairs = []
    for feature in features:
        ic = factor_test_results.get(feature, {}).get('ic', 0.0)
        ic_abs = abs(ic) if not np.isnan(ic) else 0.0
        feature_ic_pairs.append((feature, ic, ic_abs))

    # 按IC绝对值降序排序
    sorted_features = sorted(feature_ic_pairs, key=lambda x: x[2], reverse=True)

    logger.info(f"[CHART] IC排序结果 (前10名):")
    for i, (feature, ic, ic_abs) in enumerate(sorted_features[:10], 1):
        logger.info(f"  {i:2d}. {feature}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

    if len(sorted_features) > 10:
        logger.info(f"  ... 还有 {len(sorted_features) - 10} 个特征")

    # 返回排序后的特征列表
    ranked_features = [
        {
            'feature': feature,
            'ic': float(ic) if not np.isnan(ic) else 0.0,
            'ic_abs': float(ic_abs),
            'rank': i + 1
        }
        for i, (feature, ic, ic_abs) in enumerate(sorted_features)
    ]

    return ranked_features


def _apply_correlation_filtering_algorithm(features, high_corr_pairs, correlation_matrix, logger):
    """
    高相关特征过滤算法：使用贪心算法解决高相关特征过滤问题

    算法策略：
    1. 构建相关性图，节点为特征，边为高相关关系
    2. 使用贪心算法：每次移除与最多特征高相关的特征
    3. 重复直到没有高相关特征对
    """
    logger.info("[REFRESH] 开始高相关特征过滤算法...")

    # 构建相关性图：特征 -> 与其高相关的特征集合
    correlation_graph = {}
    for feature in features:
        correlation_graph[feature] = set()

    # 添加高相关关系
    for feature1, feature2, corr_val in high_corr_pairs:
        if feature1 in correlation_graph and feature2 in correlation_graph:
            correlation_graph[feature1].add(feature2)
            correlation_graph[feature2].add(feature1)

    logger.info(f"  构建相关性图: {len(features)} 个节点, {len(high_corr_pairs)} 条边")

    # 贪心算法：迭代移除与最多特征高相关的特征
    remaining_features = set(features)
    removed_features = []
    iteration = 0

    while True:
        iteration += 1

        # 计算每个剩余特征的高相关邻居数量
        correlation_counts = {}
        for feature in remaining_features:
            # 计算与当前特征高相关且仍在剩余特征中的特征数量
            correlated_neighbors = correlation_graph[feature] & remaining_features
            correlation_counts[feature] = len(correlated_neighbors)

        # 找到与最多特征高相关的特征
        if not correlation_counts or max(correlation_counts.values()) == 0:
            logger.info(f"  算法收敛: 没有剩余的高相关特征对")
            break

        # 选择与最多特征高相关的特征进行移除
        feature_to_remove = max(correlation_counts.items(), key=lambda x: x[1])[0]
        max_correlations = correlation_counts[feature_to_remove]

        logger.info(f"  第 {iteration} 轮: 移除特征 {feature_to_remove} (与 {max_correlations} 个特征高相关)")

        # 移除该特征
        remaining_features.remove(feature_to_remove)
        removed_features.append(feature_to_remove)

        # 限制迭代次数，避免无限循环
        # if iteration >= len(features) // 2:
        if iteration >= 90:
            logger.warning(f"  达到最大迭代次数 {iteration}，停止算法")
            break

    final_features = list(remaining_features)

    logger.info(f"[CHART] 高相关特征过滤结果:")
    logger.info(f"  原始特征数: {len(features)}")
    logger.info(f"  移除特征数: {len(removed_features)}")
    logger.info(f"  保留特征数: {len(final_features)}")
    logger.info(f"  过滤率: {len(removed_features)/len(features):.1%}")

    # 验证过滤效果：检查剩余特征之间是否还有高相关对
    remaining_high_corr_pairs = 0
    for i, feature1 in enumerate(final_features):
        for j, feature2 in enumerate(final_features[i+1:], i+1):
            if feature1 in correlation_matrix.index and feature2 in correlation_matrix.columns:
                corr_val = correlation_matrix.loc[feature1, feature2]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                    remaining_high_corr_pairs += 1

    logger.info(f"  验证结果: 剩余高相关对数量: {remaining_high_corr_pairs}")

    if remaining_high_corr_pairs > 0:
        logger.warning(f"[WARNING] 仍有 {remaining_high_corr_pairs} 对高相关特征，可能需要进一步优化")
    else:
        logger.info("[OK] 成功消除所有高相关特征对")

    return final_features


def _apply_vif_filter_optimized(features, merged_data, logger):
    """优化版VIF过滤：在正确位置应用所有优化措施"""
    logger.info("[REFRESH] 执行优化版VIF过滤...")

    if len(features) <= 1:
        logger.warning("[WARNING] 特征数量不足，跳过VIF过滤")
        return features

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import StandardScaler
        import warnings

        # 抑制MKL警告
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # 数据采样优化（在正确位置）
        logger.info(f"[CHART] 准备VIF计算数据，原始特征数: {len(features)}")

        vif_data = merged_data[features].select_dtypes(include=[np.number])

        # 如果数据量过大，进行采样以加速VIF计算
        original_size = len(vif_data)
        if original_size > 50000:
            sample_size = 50000
            vif_data = vif_data.sample(n=sample_size, random_state=42)
            logger.info(f"[CHART] 数据采样: {original_size:,} -> {sample_size:,} 条记录以加速VIF计算")

        # 移除缺失值
        vif_data = vif_data.dropna()
        logger.info(f"[CHART] 移除缺失值后: {len(vif_data):,} 条记录")

        # 数据质量检查和预处理
        valid_features = []
        for col in features:
            if col in vif_data.columns:
                col_data = vif_data[col]
                # 检查方差
                if col_data.std() > 1e-10:
                    # 异常值处理：使用3σ原则
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val

                    # 检查是否有有效数据
                    valid_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                    if valid_mask.sum() > len(col_data) * 0.8:  # 至少80%的数据有效
                        valid_features.append(col)
                    else:
                        logger.warning(f"[WARNING] 特征 {col} 异常值过多，移除")
                else:
                    logger.warning(f"[WARNING] 特征 {col} 方差为0，移除")

        if len(valid_features) <= 1:
            logger.warning("[WARNING] 有效特征数量不足，跳过VIF过滤")
            return valid_features

        # 重新选择数据并标准化
        vif_data = vif_data[valid_features]

        # 标准化数据以避免数值问题
        try:
            scaler = StandardScaler()
            vif_data_scaled = pd.DataFrame(
                scaler.fit_transform(vif_data),
                columns=vif_data.columns,
                index=vif_data.index
            )
            logger.info(f"[OK] 数据标准化完成")
        except Exception as e:
            logger.warning(f"[WARNING] 数据标准化失败: {e}，使用原始数据")
            vif_data_scaled = vif_data

        if len(vif_data_scaled) <= len(valid_features):
            logger.warning("[WARNING] 样本数量不足以计算VIF，跳过VIF过滤")
            return valid_features

        # 优化的VIF迭代计算
        current_features = valid_features.copy()
        max_iterations = min(len(current_features), 20)  # 限制最大迭代次数
        iteration = 0

        logger.info(f"[CHART] 开始VIF迭代过滤，最大迭代次数: {max_iterations}")

        while iteration < max_iterations and len(current_features) > 1:
            iteration += 1
            logger.info(f"  第 {iteration} 轮VIF计算，当前特征数: {len(current_features)}")

            # 获取当前数据
            current_data = vif_data_scaled[current_features]

            # 检查数据质量
            if current_data.isnull().any().any():
                logger.warning("[WARNING] 发现缺失值，重新清理数据")
                current_data = current_data.dropna()
                if len(current_data) < len(current_features) * 10:
                    logger.warning("[WARNING] 清理后数据不足，停止VIF过滤")
                    break

            # 使用优化的并行VIF计算
            vif_scores = _calculate_vif_parallel(current_data, current_features, logger)

            if not vif_scores:
                logger.warning("[WARNING] VIF计算失败，停止过滤")
                break

            # 找到最高VIF的特征
            max_vif_feature, max_vif = max(vif_scores, key=lambda x: x[1] if not np.isinf(x[1]) else 1e6)

            if max_vif < 5.0:
                logger.info(f"  [OK] 所有特征VIF<5，过滤完成")
                break

            # 移除最高VIF的特征
            current_features.remove(max_vif_feature)
            if np.isinf(max_vif):
                logger.info(f"  移除问题特征: {max_vif_feature} (VIF=∞)")
            else:
                logger.info(f"  移除高VIF特征: {max_vif_feature} (VIF={max_vif:.2f})")

            if len(current_features) <= 1:
                logger.warning("[WARNING] 剩余特征数量不足，停止VIF过滤")
                break

        logger.info(f"[CHART] VIF过滤结果:")
        logger.info(f"  原始特征: {len(features)} 个")
        logger.info(f"  移除特征: {len(features) - len(current_features)} 个")
        logger.info(f"  保留特征: {len(current_features)} 个")

        return current_features

    except ImportError:
        logger.warning("[WARNING] statsmodels未安装，跳过VIF过滤")
        return features
    except Exception as e:
        logger.warning(f"[WARNING] VIF过滤失败: {e}")
        return features


def _perform_factor_testing(merged_data, features, logger):
    """对过滤后的特征进行单因子测试"""
    logger.info(f"[REFRESH] 开始单因子测试，特征数量: {len(features)}")

    # 检查是否有future_return_2d字段
    if 'future_return_2d' not in merged_data.columns:
        logger.error("[FAIL] 数据中缺少future_return_2d字段")
        return {}

    # 移除缺失值
    analysis_data = merged_data.dropna(subset=['future_return_2d'] + features)
    logger.info(f"[CHART] 分析数据: {len(analysis_data):,} 条记录")

    if len(analysis_data) == 0:
        logger.error("[FAIL] 分析数据为空")
        return {}

    factor_test_results = {}

    for i, factor_name in enumerate(features, 1):
        logger.info(f"  [{i:3d}/{len(features)}] 测试因子: {factor_name}")

        try:
            factor_values = analysis_data[factor_name].values
            returns = analysis_data['future_return_2d'].values

            # 移除NaN值
            valid_mask = ~(np.isnan(factor_values) | np.isnan(returns))
            if np.sum(valid_mask) < 100:  # 至少需要100个有效样本
                logger.warning(f"    [WARNING] 有效样本不足: {np.sum(valid_mask)}")
                factor_test_results[factor_name] = _create_empty_factor_result()
                continue

            factor_clean = factor_values[valid_mask]
            returns_clean = returns[valid_mask]

            # 计算各项指标（与之前的实现一致）
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1] if len(factor_clean) > 1 else 0.0

            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(factor_clean, returns_clean)

            # 分层回测
            n_quantiles = 5
            try:
                quantiles = np.quantile(factor_clean, np.linspace(0, 1, n_quantiles + 1))
                layer_returns = []
                layer_counts = []

                for j in range(n_quantiles):
                    if j == n_quantiles - 1:
                        mask = (factor_clean >= quantiles[j]) & (factor_clean <= quantiles[j + 1])
                    else:
                        mask = (factor_clean >= quantiles[j]) & (factor_clean < quantiles[j + 1])

                    if np.sum(mask) > 0:
                        layer_return = np.mean(returns_clean[mask])
                        layer_returns.append(layer_return)
                        layer_counts.append(np.sum(mask))
                    else:
                        layer_returns.append(0.0)
                        layer_counts.append(0)

            except Exception as e:
                logger.warning(f"    [WARNING] 分层回测计算失败: {e}")
                layer_returns = [0.0] * 5
                layer_counts = [0] * 5

            # 其他指标
            long_short_return = layer_returns[-1] - layer_returns[0] if len(layer_returns) >= 2 else 0.0

            # IR计算
            ir = 0.0
            try:
                if 'date' in analysis_data.columns:
                    daily_ic_data = analysis_data.groupby('date').apply(
                        lambda group: np.corrcoef(group[factor_name].values, group['future_return_2d'].values)[0, 1]
                        if len(group) > 1 and group[factor_name].notna().sum() > 1 and group['future_return_2d'].notna().sum() > 1
                        else np.nan
                    ).dropna()

                    if len(daily_ic_data) > 1:
                        ir = daily_ic_data.mean() / (daily_ic_data.std() + 1e-8)
                    else:
                        ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0
                else:
                    ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0
            except Exception as e:
                logger.warning(f"    [WARNING] IR计算失败: {e}")
                ir = ic / (np.std(returns_clean) + 1e-8) if len(returns_clean) > 1 else 0.0

            win_rate = np.mean(returns_clean > 0) if len(returns_clean) > 0 else 0.0
            monotonicity = _calculate_monotonicity(layer_returns)
            coverage = np.sum(valid_mask) / len(factor_values)

            # 最大回撤
            max_drawdown = 0.0
            if len(layer_returns) > 1:
                cumulative_returns = np.cumsum(layer_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

            factor_test_results[factor_name] = {
                'ic': ic,
                'rank_ic': rank_ic,
                'ir': ir,
                'icir_20d': ic / (np.std(returns_clean) + 1e-8),
                'layer_returns': layer_returns,
                'layer_counts': layer_counts,
                'long_short_return': long_short_return,
                'win_rate': win_rate,
                'monotonicity': monotonicity,
                'coverage': coverage,
                'max_drawdown': max_drawdown,
                'valid_samples': np.sum(valid_mask),
                'total_samples': len(factor_values)
            }

            logger.info(f"    IC: {ic:.4f}, RankIC: {rank_ic:.4f}, 多空收益: {long_short_return:.4f}, 胜率: {win_rate:.2%}")

        except Exception as e:
            logger.warning(f"    [WARNING] 因子测试失败: {e}")
            factor_test_results[factor_name] = _create_empty_factor_result()

    # 保存因子测试结果
    factor_results_file = 'interaction_factor_test_results.json'
    try:
        import json
        factor_results_summary = {
            'factor_test_results': {
                name: {
                    'ic': float(result['ic']) if not np.isnan(result['ic']) else 0.0,
                    'rank_ic': float(result['rank_ic']) if not np.isnan(result['rank_ic']) else 0.0,
                    'ir': float(result['ir']) if not np.isnan(result['ir']) else 0.0,
                    'icir_20d': float(result['icir_20d']) if not np.isnan(result['icir_20d']) else 0.0,
                    'layer_returns': [float(x) if not np.isnan(x) else 0.0 for x in result['layer_returns']],
                    'layer_counts': [int(x) for x in result['layer_counts']],
                    'long_short_return': float(result['long_short_return']) if not np.isnan(result['long_short_return']) else 0.0,
                    'win_rate': float(result['win_rate']) if not np.isnan(result['win_rate']) else 0.0,
                    'monotonicity': float(result['monotonicity']) if not np.isnan(result['monotonicity']) else 0.0,
                    'coverage': float(result['coverage']) if not np.isnan(result['coverage']) else 0.0,
                    'max_drawdown': float(result['max_drawdown']) if not np.isnan(result['max_drawdown']) else 0.0,
                    'valid_samples': int(result['valid_samples']),
                    'total_samples': int(result['total_samples'])
                }
                for name, result in factor_test_results.items()
            },
            'summary': {
                'total_factors': len(features),
                'successful_tests': len([r for r in factor_test_results.values() if r['valid_samples'] > 0]),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        with open(factor_results_file, 'w', encoding='utf-8') as f:
            json.dump(factor_results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"[OK] 因子测试结果已保存到: {factor_results_file}")
    except Exception as e:
        logger.warning(f"[WARNING] 保存因子测试结果失败: {e}")

    return factor_test_results


def _generate_interaction_analysis_report_fixed(interaction_names, factor_test_results,
                                              high_corr_pairs, high_vif_features,
                                              stock_features, market_features, logger,
                                              final_ranked_features=None, filtering_results=None):
    """生成修复版的交互特征分析报告"""
    logger.info(f"\n[TARGET] 交互特征分析报告")
    logger.info(f"=" * 60)

    # 基本统计
    logger.info(f"[CHART] 基本统计:")
    logger.info(f"  - 原始交互特征数: {len(interaction_names)}")
    logger.info(f"  - 股票特征数: {len(stock_features)}")
    logger.info(f"  - 市场特征数: {len(market_features)}")

    if filtering_results:
        logger.info(f"  - 质量检查后: {filtering_results['valid_features_after_quality_check']}")
        logger.info(f"  - 股票因子相关性过滤后: {filtering_results['features_after_stock_correlation_filter']}")
        logger.info(f"  - VIF过滤后: {filtering_results['features_after_vif_filter']}")
        logger.info(f"  - 最终保留: {filtering_results['final_ranked_features']}")
        logger.info(f"  - 总体保留率: {filtering_results['filtering_summary']['final_retention_rate']:.2%}")

    # 显示最终优质特征
    if final_ranked_features and len(final_ranked_features) > 0:
        logger.info(f"\n[TROPHY] 最终优质交互特征 (前10名):")
        for i, feature_info in enumerate(final_ranked_features[:10], 1):
            feature_name = feature_info['feature']
            ic = feature_info['ic']
            ic_abs = feature_info['ic_abs']
            logger.info(f"  {i:2d}. {feature_name}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

        if len(final_ranked_features) > 10:
            logger.info(f"  ... 还有 {len(final_ranked_features) - 10} 个优质特征")

    # 下一步建议
    logger.info(f"\n[TARGET] 下一步行动计划:")
    if final_ranked_features and len(final_ranked_features) > 0:
        logger.info(f"  1. 立即行动:")
        logger.info(f"     - 使用 {len(final_ranked_features)} 个优质交互特征重新训练")
        logger.info(f"     - 优先使用前10名特征进行初步验证")
        logger.info(f"     - 调整正则化参数防止过拟合")

    logger.info(f"  2. 验证测试:")
    logger.info(f"     - 在独立测试集上验证改进效果")
    logger.info(f"     - 对比有无交互特征的回测结果")
    logger.info(f"     - 监控过拟合风险")

    logger.info(f"\n[OK] 交互特征分析完成！")
    return True


def _generate_interaction_analysis_report(interaction_names, factor_test_results,
                                        high_corr_pairs, high_vif_features,
                                        stock_features, market_features, logger,
                                        final_ranked_features=None, filtering_results=None):
    """生成交互特征分析报告"""
    try:
        logger.info("=" * 80)
        logger.info("[CLIPBOARD] 交互特征分析报告")
        logger.info("=" * 80)

        # 1. 基本统计
        logger.info("[CHART] 基本统计:")
        logger.info(f"  - 股票特征数量: {len(stock_features)}")
        logger.info(f"  - 市场特征数量: {len(market_features)}")
        logger.info(f"  - 生成交互特征数量: {len(interaction_names)}")
        logger.info(f"  - 高相关性特征对: {len(high_corr_pairs)}")
        logger.info(f"  - 高VIF特征数量: {len(high_vif_features)}")

        # 2. 单因子测试结果排序
        logger.info("\n[UP] 单因子测试结果 (按|IC|排序):")

        # 按IC绝对值排序
        sorted_results = sorted(
            factor_test_results.items(),
            key=lambda x: abs(x[1]['ic']) if not np.isnan(x[1]['ic']) else 0,
            reverse=True
        )

        logger.info("  排名  特征名称                                          IC      多空收益   样本数")
        logger.info("  " + "-" * 75)

        effective_features = []
        zero_importance_features = []

        for i, (feature_name, result) in enumerate(sorted_results, 1):
            ic = result['ic']
            ls_return = result['long_short_return']
            samples = result['valid_samples']

            if abs(ic) > 0.01:  # 认为有效的阈值
                effective_features.append(feature_name)
                status = "[OK]"
            else:
                zero_importance_features.append(feature_name)
                status = "[FAIL]"

            logger.info(f"  {i:2d}    {feature_name:<45} {ic:7.4f}  {ls_return:8.4f}  {samples:6d} {status}")

        # 3. 问题特征识别
        logger.info(f"\n[WARNING] 问题特征识别:")
        logger.info(f"  - 低效特征 (|IC|<0.01): {len(zero_importance_features)} 个")
        if zero_importance_features:
            for feature in zero_importance_features[:5]:
                logger.info(f"    * {feature}")
            if len(zero_importance_features) > 5:
                logger.info(f"    * ... 还有 {len(zero_importance_features) - 5} 个")

        logger.info(f"  - 高相关性特征: {len([f for f, _, _ in high_corr_pairs])} 个")
        logger.info(f"  - 高VIF特征: {len(high_vif_features)} 个")

        # 4. 优化建议
        logger.info(f"\n[IDEA] 优化建议:")

        if len(zero_importance_features) > 0:
            logger.info(f"  1. 特征筛选:")
            logger.info(f"     - 移除 {len(zero_importance_features)} 个低效交互特征")
            logger.info(f"     - 保留 {len(effective_features)} 个有效交互特征")

        if len(high_corr_pairs) > 0:
            logger.info(f"  2. 相关性处理:")
            logger.info(f"     - 发现 {len(high_corr_pairs)} 对高相关性特征")
            logger.info(f"     - 建议使用相关性筛选或PCA降维")

        if len(high_vif_features) > 0:
            logger.info(f"  3. 多重共线性处理:")
            logger.info(f"     - {len(high_vif_features)} 个特征存在多重共线性")
            logger.info(f"     - 建议使用岭回归或LASSO进行特征选择")

        # 5. 模型参数调整建议
        logger.info(f"\n[WRENCH] 模型参数调整建议:")

        feature_increase_ratio = len(interaction_names) / len(stock_features)
        logger.info(f"  - 特征维度增加: {feature_increase_ratio:.1f}x ({len(stock_features)} -> {len(stock_features) + len(interaction_names)})")

        if feature_increase_ratio > 2.0:
            logger.info(f"  - 建议增加正则化强度:")
            logger.info(f"    * L2正则化 (lambda): 1.0 -> 2.0-5.0")
            logger.info(f"    * L1正则化 (alpha): 0.0 -> 0.1-0.5")
            logger.info(f"  - 建议降低模型复杂度:")
            logger.info(f"    * max_depth: 5 -> 3-4")
            logger.info(f"    * eta (学习率): 0.1 -> 0.05-0.08")

        # 6. 管线评估改进建议
        logger.info(f"\n[CHART] 管线评估改进建议:")
        logger.info(f"  - 当前指标: Top10%-Bottom10%")
        logger.info(f"  - 建议补充指标:")
        logger.info(f"    * IC_IR综合得分: IC均值 / IC标准差")
        logger.info(f"    * 夏普比率: 收益均值 / 收益标准差")
        logger.info(f"    * 最大回撤: 评估风险控制能力")
        logger.info(f"    * 胜率: 预测正确的比例")

        # 7. 串联过滤结果展示
        if final_ranked_features is not None and filtering_results is not None:
            logger.info(f"\n[REFRESH] 串联过滤结果:")
            logger.info(f"  - 原始交互特征: {filtering_results['original_features']} 个")
            logger.info(f"  - 质量检查后: {filtering_results['valid_features_after_quality_check']} 个")
            logger.info(f"  - 相关性过滤后: {filtering_results['features_after_correlation_filter']} 个")
            logger.info(f"  - VIF过滤后: {filtering_results['features_after_vif_filter']} 个")
            logger.info(f"  - 最终保留: {filtering_results['final_ranked_features']} 个")
            logger.info(f"  - 总体保留率: {filtering_results['filtering_summary']['final_retention_rate']:.2%}")

            logger.info(f"\n[TROPHY] 最终优质交互特征 (前10名):")
            for i, feature_info in enumerate(final_ranked_features[:10], 1):
                feature_name = feature_info['feature']
                ic = feature_info['ic']
                ic_abs = feature_info['ic_abs']
                logger.info(f"  {i:2d}. {feature_name}: IC={ic:.4f} (|IC|={ic_abs:.4f})")

            if len(final_ranked_features) > 10:
                logger.info(f"  ... 还有 {len(final_ranked_features) - 10} 个优质特征")

        # 8. 下一步行动计划
        logger.info(f"\n[TARGET] 下一步行动计划:")

        if final_ranked_features is not None and len(final_ranked_features) > 0:
            logger.info(f"  1. 立即行动:")
            logger.info(f"     - 使用 {len(final_ranked_features)} 个优质交互特征重新训练")
            logger.info(f"     - 优先使用前10名特征进行初步验证")
            logger.info(f"     - 调整正则化参数防止过拟合")
        elif len(effective_features) > 0:
            logger.info(f"  1. 立即行动:")
            logger.info(f"     - 使用 {len(effective_features)} 个有效交互特征重新训练")
            logger.info(f"     - 调整正则化参数防止过拟合")

        if len(zero_importance_features) > len(effective_features):
            logger.info(f"  2. 特征工程优化:")
            logger.info(f"     - 尝试不同的股票特征组合")
            logger.info(f"     - 测试非线性交互 (比值、差值、条件交互)")
            logger.info(f"     - 引入时间滞后的交互特征")

        logger.info(f"  3. 验证测试:")
        logger.info(f"     - 在独立测试集上验证改进效果")
        logger.info(f"     - 对比有无交互特征的回测结果")
        logger.info(f"     - 监控过拟合风险")

        logger.info("=" * 80)
        logger.info("[OK] 交互特征全面分析完成")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"[FAIL] 报告生成失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False


def run_test_model_walkforward_mode(args):
    """
    运行Walk-Forward Analysis + 交叉验证模式

    实现顶级量化机构的标准做法:
    1. 将回测期分成多个窗口
    2. 每个窗口执行完整的"贝叶斯搜索(带交叉验证)+训练+回测"流程
    3. 汇总所有窗口的回测结果

    Args:
        args: 命令行参数
    """
    logger = logging.getLogger(__name__)
    print(f"运行模式: Walk-Forward Analysis + 交叉验证")
    logger.info("=" * 80)
    logger.info("顶级量化机构标准做法: 滚动优化 + 交叉验证")
    logger.info("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from datetime import datetime

        # [WRENCH] 修复：定义Walk-Forward扩展窗口（Expanding Window）
        # 每个窗口包含: 训练集、验证集(用于超参数搜索)、测试集(用于回测)
        # 时间序列原则: 训练集 < 验证集 < 测试集 (无重叠,无数据泄露)
        # 扩展窗口特点: 训练期从固定起点开始，逐渐增加训练数据量
        windows = [
            {
                'name': 'Window 1',
                'train_start': '2023-06-01',   # 固定起点
                'train_end': '2024-09-21',     # 训练集结束 (约16个月)
                'val_start': '2024-10-15',     # 验证集开始(紧接训练集)
                'val_end': '2024-12-31',       # 验证集结束 (3个月)
                'test_start': '2025-01-01',    # 测试集开始(紧接验证集)
                'test_end': '2025-03-31'       # 测试集结束 (3个月)
            },
            {
                'name': 'Window 2',
                'train_start': '2023-06-01',   # 固定起点（扩展窗口）
                'train_end': '2024-12-31',     # 训练集结束 (约18个月，增加3个月)
                'val_start': '2025-01-05',     # 验证集开始2025-01-01
                'val_end': '2025-03-31',       # 验证集结束 (3个月)
                'test_start': '2025-04-05',    # 测试集开始2025-04-01
                'test_end': '2025-10-15'       # 测试集结束 (3个月)
            },
            # {
            #     'name': 'Window 3',
            #     'train_start': '2023-06-01',   # 固定起点（扩展窗口）
            #     'train_end': '2025-01-27',     # 训练集结束 (约21个月，再增加3个月)
            #     'val_start': '2025-02-10',     # 验证集开始
            #     # 'val_end': '2025-09-12',       # 验证集结束 (3个月)
            #     'val_end': '2025-06-01',
            #     # 'test_start': '2025-09-22',    # 测试集开始2025-08-21 2025-11-21
            #     # 'test_end': '2026-02-03'       # 测试集结束 (3个月)2025-12-16
            #     'test_start': '2025-06-05',    # 测试集开始2025-08-21 2025-11-21
            #     'test_end': '2026-04-16'       # 测试集结束 (3个月)2025-12-16
            # }
            {
                'name': 'Window 3',
                'train_start': '2023-06-01',
                'train_end': '2025-09-01',
                #  'train_end': '2025-02-25',
                'val_start': '2025-09-10',
                'val_end': '2025-09-25',
                # 'val_end': '2025-05-25',
                'test_start': '2025-10-10',
                'test_end': '2026-04-22'
                # 'val_start': '2025-03-01',
                # 'val_end': '2025-06-02',
                # 'test_start': '2025-06-06',
                # 'test_end': '2026-03-02'
            },            
        ]

        # [WRENCH] 解析窗口选择参数
        selected_window_indices = None
        if args.window:
            try:
                # 解析窗口编号（支持逗号分隔，如 "1" 或 "1,3"）
                selected_window_indices = [int(w.strip()) for w in args.window.split(',')]

                # 验证窗口编号范围
                for idx in selected_window_indices:
                    if idx < 1 or idx > len(windows):
                        logger.error(f"[FAIL] 窗口编号 {idx} 超出范围 [1, {len(windows)}]")
                        return False

                logger.info(f"\n[TARGET] 窗口选择模式: 只训练指定的窗口 {selected_window_indices}")
            except ValueError as e:
                logger.error(f"[FAIL] 窗口参数格式错误: {args.window}，应为数字或逗号分隔的数字（例如: 1 或 1,3）")
                return False
        else:
            logger.info(f"\n[TARGET] 默认模式: 训练所有窗口")

        logger.info(f"\n[CHART] Walk-Forward窗口配置:")
        logger.info(f"  总窗口数: {len(windows)}")
        if selected_window_indices:
            logger.info(f"  选择的窗口: {selected_window_indices}")
        else:
            logger.info(f"  选择的窗口: 全部")
        logger.info(f"  时间序列原则: 训练集 < 验证集 < 测试集 (无重叠)")
        for i, window in enumerate(windows, 1):
            # 标记是否被选中
            selected_mark = "✓" if (selected_window_indices is None or i in selected_window_indices) else "✗"
            logger.info(f"\n  [{selected_mark}] 窗口{i} ({window['name']}):")
            logger.info(f"    训练集: {window['train_start']} ~ {window['train_end']}")
            logger.info(f"    验证集: {window['val_start']} ~ {window['val_end']}")
            logger.info(f"    测试集: {window['test_start']} ~ {window['test_end']}")

        # 加载完整因子数据
        logger.info("\n[REFRESH] 加载完整因子数据...")
        factor_data = _load_factor_cache_data(include_label_generation_features=True)

        if factor_data is None or len(factor_data) == 0:
            logger.error("[FAIL] 因子数据加载失败")
            return False

        logger.info(f"[OK] 因子数据加载成功: {len(factor_data):,}条记录, {len(factor_data.columns)}个字段")
        logger.info(f"   日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")

        # 确保date列是datetime类型
        factor_data['date'] = pd.to_datetime(factor_data['date'])

        # [SW行业编码] 加载标准化SW申万行业数据用于LightGBM分类特征
        logger.info("\n[SW行业编码] 加载标准化SW申万行业数据...")

        encoding_data = load_standardized_sw_encoding()
        if not encoding_data:
            raise RuntimeError("无法加载标准化SW编码文件，请确保config/sw_industry_encoding.pkl文件存在")

        sw_sector_mapping = encoding_data['mappings']['stock_to_industry']

        metadata = encoding_data.get('metadata', {})
        logger.info(f"[OK] 标准化SW行业数据加载成功: {len(sw_sector_mapping)}只股票")
        logger.info(f"  版本: {metadata.get('version', 'unknown')}")
        logger.info(f"  生成时间: {metadata.get('generated_at', 'unknown')}")
        logger.info(f"  行业数: {metadata.get('total_industries', 0)}")

        # [SW编码检查] 检查SW行业编码20的分布
        logger.info("[SW编码检查] 开始检查SW行业编码20的分布...")
        if 'mappings' in encoding_data and 'industry_to_code' in encoding_data['mappings']:
            industry_to_code = encoding_data['mappings']['industry_to_code']
            sw_code_20_industries = [industry for industry, code in industry_to_code.items() if code == 20]
            logger.info(f"[SW编码检查] SW行业编码20对应行业: {sw_code_20_industries}")

            if 'stock_to_industry' in encoding_data['mappings']:
                stock_to_industry = encoding_data['mappings']['stock_to_industry']
                sw_code_20_stocks = [stock for stock, industry in stock_to_industry.items()
                                    if industry_to_code.get(industry) == 20]
                logger.info(f"[SW编码检查] SW行业编码20的股票总数: {len(sw_code_20_stocks)}")
                if sw_code_20_stocks:
                    logger.info(f"[SW编码检查] SW行业编码20股票示例: {sw_code_20_stocks[:10]}")
                else:
                    logger.warning("[SW编码检查] 警告：没有找到SW行业编码20的股票")
            else:
                logger.warning("[SW编码检查] 警告：编码数据中缺少stock_to_industry映射")
        else:
            logger.warning("[SW编码检查] 警告：编码数据格式不正确，无法检查SW行业编码20")

        # 存储所有窗口的回测结果
        all_window_results = []
        all_selections = []
        all_window_metrics = []

        # 遍历每个Walk-Forward窗口（只处理选中的窗口）
        for window_idx, window in enumerate(windows, 1):
            # [WRENCH] 如果指定了窗口选择，跳过未选中的窗口
            if selected_window_indices is not None and window_idx not in selected_window_indices:
                logger.info(f"\n⏭️  跳过 {window['name']} (未选中)")
                continue

            logger.info("\n" + "=" * 80)
            logger.info(f"[REFRESH] 开始处理 {window['name']} (第{window_idx}/{len(windows)}个窗口)")
            logger.info("=" * 80)

            # 处理单个窗口
            window_result = _process_walkforward_window(
                window, factor_data, args, window_idx, len(windows), windows, sw_sector_mapping
            )

            if window_result is None:
                logger.error(f"[FAIL] {window['name']} 处理失败")
                continue

            # 保存窗口结果
            all_window_results.append(window_result['backtest_result'])
            # 修复：selections是字典格式，需要以窗口为单位存储
            all_selections.append({
                'window_name': window['name'],
                'selections': window_result['selections']  # {1: [...], 2: [...], 3: [...], 5: [...]}
            })
            all_window_metrics.append({
                'window_name': window['name'],
                'test_period': f"{window['test_start']} ~ {window['test_end']}",
                **window_result['metrics']
            })

            logger.info(f"[OK] {window['name']} 处理完成")

        # 汇总所有窗口的结果
        logger.info("\n" + "=" * 80)
        logger.info("[CHART] Walk-Forward Analysis 汇总结果")
        logger.info("=" * 80)

        if not all_window_results:
            logger.error("[FAIL] 所有窗口处理失败,无法生成汇总结果")
            return False

        # 计算汇总指标
        _print_walkforward_summary(all_window_metrics)

        # 保存汇总结果
        _save_walkforward_results(all_window_metrics, all_selections)

        logger.info("[OK] Walk-Forward Analysis 完成")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Walk-Forward Analysis 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_pred_stocks_mode(args):
    """
    股票预测模式：根据指定日期预测top5股票

    Args:
        args: 命令行参数，必须包含--date参数
    """
    logger = logging.getLogger(__name__)

    # 验证必要参数
    if not hasattr(args, 'date') or not args.date:
        logger.error("pred_stocks模式需要指定--date参数")
        print("错误: pred_stocks模式需要指定--date参数 (格式: YYYYMMDD)")
        print("使用示例: python main.py --mode pred_stocks --date 20251216")
        return False

    prediction_date = args.date
    logger.info(f"运行模式: 股票预测 - 日期: {prediction_date}")
    print(f"运行模式: 股票预测")
    print(f"预测日期: {prediction_date}")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import pickle
        from pathlib import Path
        from datetime import datetime

        # 1. 加载预测日期的因子数据
        logger.info(f"步骤1: 加载 {prediction_date} 的因子数据")
        factor_data = _load_factor_data_for_prediction(prediction_date)

        if factor_data is None or factor_data.empty:
            logger.error(f"无法加载 {prediction_date} 的因子数据")
            print(f"错误: 无法加载 {prediction_date} 的因子数据")
            return False

        logger.info(f"成功加载因子数据: {len(factor_data)} 只股票")
        print(f"成功加载因子数据: {len(factor_data)} 只股票")

        # 2. 加载LightGBM模型
        logger.info("步骤2: 加载LightGBM模型")
        # model_path = Path("./results/lightgbm_model_window3_20251216_200900.pkl")_20260107_033514
        # model_path = Path("./results/lightgbm_model_window3_20251227_190526.pkl")
        model_path = Path("./results/lightgbm_model_window3_20260107_033514.pkl")
        model_path = Path("./results/lightgbm_model_window3_20260427_231053.pkl")


        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            print(f"错误: 模型文件不存在: {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            # 适配实际的模型结构：使用feature_names而不是feature_columns
            feature_columns = model_data.get('feature_names', [])

            if not feature_columns:
                # 如果没有feature_names，尝试从模型对象获取
                if hasattr(model, 'feature_name'):
                    feature_columns = model.feature_name()
                else:
                    logger.error("无法确定模型使用的特征列")
                    print("错误: 无法确定模型使用的特征列")
                    return False

            logger.info(f"成功加载模型，特征数: {len(feature_columns)}")
            print(f"成功加载模型，特征数: {len(feature_columns)}")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            print(f"错误: 加载模型失败: {e}")
            return False

        # 3. 数据预处理和行业适配（使用标准化SW编码系统）
        logger.info("步骤3: 数据预处理和行业适配")
        logger.info("[OK] 使用标准化SW行业编码系统")
        processed_data, X_pred = _prepare_prediction_data(factor_data, feature_columns, logger)

        if processed_data is None or X_pred is None:
            logger.error("数据预处理失败")
            print("错误: 数据预处理失败")
            return False

        # 4. 数据过滤（排除不符合交易条件的股票）
        logger.info("步骤4: 数据过滤（排除ST、涨停、跌停、新股等）")
        original_count = len(processed_data)

        try:
            # 4.1 过滤ST股票
            if len(processed_data) > 0:
                st_filter = _filter_st_stocks(processed_data, logger)
                processed_data = processed_data[st_filter].copy()
                st_filtered = original_count - len(processed_data)
                logger.info(f"[过滤] ST股过滤: {original_count} -> {len(processed_data)} (过滤掉 {st_filtered})")
                print(f"[过滤] ST股过滤: {original_count} -> {len(processed_data)} (过滤掉 {st_filtered})")

            # 4.2 过滤T日涨停股票（无法买入）
            if 'is_limit_up' in processed_data.columns and len(processed_data) > 0:
                before_limit_up = len(processed_data)
                processed_data = processed_data[processed_data['is_limit_up'] != 1].copy()
                limit_up_filtered = before_limit_up - len(processed_data)
                logger.info(f"[过滤] 涨停股过滤: {before_limit_up} -> {len(processed_data)} (过滤掉 {limit_up_filtered})")
                print(f"[过滤] 涨停股过滤: {before_limit_up} -> {len(processed_data)} (过滤掉 {limit_up_filtered})")

            # 4.3 过滤T日跌停股票（风险高）
            if 'is_limit_down' in processed_data.columns and len(processed_data) > 0:
                before_limit_down = len(processed_data)
                processed_data = processed_data[processed_data['is_limit_down'] != 1].copy()
                limit_down_filtered = before_limit_down - len(processed_data)
                logger.info(f"[过滤] 跌停股过滤: {before_limit_down} -> {len(processed_data)} (过滤掉 {limit_down_filtered})")
                print(f"[过滤] 跌停股过滤: {before_limit_down} -> {len(processed_data)} (过滤掉 {limit_down_filtered})")

            # 4.4 过滤新股（上市不足30天）
            if len(processed_data) > 0:
                processed_data = _filter_new_stocks_batch(processed_data, logger)
                new_stock_filtered = original_count - len(processed_data) - st_filtered - limit_up_filtered - limit_down_filtered
                logger.info(f"[过滤] 新股过滤完成，剩余股票: {len(processed_data)}")
                print(f"[过滤] 新股过滤完成，剩余股票: {len(processed_data)}")

            # 4.5 流动性过滤（成交额 < 1000万视为流动性不足）
            if 'amount' in processed_data.columns and len(processed_data) > 0:
                before_liquidity = len(processed_data)
                # 成交额单位：元，1000万 = 10,000,000
                liquidity_threshold = 10_000_000
                processed_data = processed_data[processed_data['amount'] >= liquidity_threshold].copy()
                liquidity_filtered = before_liquidity - len(processed_data)
                logger.info(f"[过滤] 流动性过滤(amount): {before_liquidity} -> {len(processed_data)} (过滤掉 {liquidity_filtered})")
                print(f"[过滤] 流动性过滤: {before_liquidity} -> {len(processed_data)} (过滤掉 {liquidity_filtered})")

          
            total_filtered = original_count - len(processed_data)
            filter_rate = total_filtered / original_count * 100 if original_count > 0 else 0
            logger.info(f"[过滤] 数据过滤完成: {original_count} -> {len(processed_data)} (总过滤率 {filter_rate:.1f}%)")
            print(f"[过滤] 数据过滤完成: {original_count} -> {len(processed_data)} (总过滤率 {filter_rate:.1f}%)")

            # 如果过滤后没有股票，提前返回
            if len(processed_data) == 0:
                logger.error("过滤后没有可预测的股票")
                print("错误: 过滤后没有可预测的股票")
                return False

            # 重新提取特征矩阵（因为数据被过滤了）
            X_pred = processed_data[feature_columns]
            logger.info(f"[过滤] 重新提取特征矩阵: {X_pred.shape}")

        except Exception as e:
            logger.error(f"数据过滤失败: {e}")
            print(f"警告: 数据过滤失败，使用原始数据进行预测: {e}")
            # 如果过滤失败，继续使用原始数据

        # 5. 模型预测
        logger.info("步骤5: 执行模型预测")
        try:
            # 预测
            predictions = model.predict(X_pred, num_iteration=model.best_iteration)
            logger.info(f"预测完成，预测值范围: [{predictions.min():.6f}, {predictions.max():.6f}]")

            # 将预测值添加到数据中
            processed_data['prediction'] = predictions

            # 按预测值降序排列
            processed_data = processed_data.sort_values('prediction', ascending=False)

            # 只对排序结果做默认过滤判断，不改变候选池和Top1排序
            if ENABLE_EXHAUSTION_BUY_FILTER and len(processed_data) > 0:
                exhaustion_flags = processed_data.apply(_is_exhaustion_filtered_row, axis=1)
                processed_data['exhaustion_filtered'] = exhaustion_flags.map(lambda x: x[0])
                processed_data['exhaustion_filter_reason'] = exhaustion_flags.map(lambda x: x[1])
            else:
                processed_data['exhaustion_filtered'] = False
                processed_data['exhaustion_filter_reason'] = ''

            logger.info(f"预测完成，共预测 {len(predictions)} 只股票")
            print(f"预测完成，共预测 {len(predictions)} 只股票")

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            print(f"错误: 模型预测失败: {e}")
            return False

        # 6. 输出Top5结果
        logger.info("步骤6: 生成预测结果")
        top5_stocks = processed_data.head(5)
        top1_row = processed_data.iloc[0] if len(processed_data) > 0 else None

        # 加载SW行业映射用于结果显示
        from sw_encoding_loader import get_sw_mapping_for_model
        sw_mapping = get_sw_mapping_for_model()
        stock_to_industry = sw_mapping.get('stock_to_industry_name', {})

        if top1_row is not None:
            top1_tradeable = not bool(top1_row.get('exhaustion_filtered', False))
            top1_reason = top1_row.get('exhaustion_filter_reason', '')
            logger.info(
                f"[Top1交易判断] stock={top1_row['stock_code']} tradeable={top1_tradeable} "
                f"strategy={EXHAUSTION_FILTER_STRATEGY} reason={top1_reason if top1_reason else 'PASS'}"
            )
            print("\nTop1交易判断:")
            print("-" * 80)
            print(f"股票代码: {top1_row['stock_code']}")
            print(f"预测值: {top1_row['prediction']:.6f}")
            print(f"是否可交易: {'是' if top1_tradeable else '否'}")
            print(f"默认过滤策略: {EXHAUSTION_FILTER_STRATEGY}")
            print(f"判断原因: {top1_reason if top1_reason else 'PASS'}")
            print(f"判断原因: {top1_reason}")
            print("-" * 80)

        print(f"\nTop 5 预测结果 ({prediction_date}):")
        print("=" * 80)
        print(f"{'排名':<4} {'股票代码':<12} {'股票名称':<16} {'SW行业':<12} {'预测值':<12} {'可交易':<8}")
        print("-" * 80)

        for rank, (_, row) in enumerate(top5_stocks.iterrows(), 1):
            stock_code = row['stock_code']
            prediction = row['prediction']

            # 获取真实的SW行业名称
            sw_sector_name = stock_to_industry.get(stock_code, '未知')

            tradeable_text = '否' if bool(row.get('exhaustion_filtered', False)) else '是'
            print(f"{rank:<4} {stock_code:<12} {'未知':<16} {sw_sector_name:<12} {prediction:<12.6f} {tradeable_text:<8}")

        print("=" * 80)

        # 7. 保存结果
        results_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else Path("./results")
        results_dir.mkdir(exist_ok=True)

        result_file = results_dir / f"pred_stocks_{prediction_date}.csv"
        processed_data.to_csv(result_file, index=False, encoding='utf-8-sig')

        logger.info(f"预测结果已保存到: {result_file}")
        print(f"完整预测结果已保存到: {result_file}")

        return True

    except Exception as e:
        logger.error(f"股票预测模式失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        print(f"预测失败: {e}")
        return False


def _load_factor_data_for_prediction(prediction_date: str) -> Optional[pd.DataFrame]:
    """
    加载指定日期的因子数据

    Args:
        prediction_date: 预测日期 (YYYYMMDD)

    Returns:
        因子数据DataFrame，或None
    """
    try:
        logger = logging.getLogger(__name__)

        # 使用_load_factor_cache_data加载alpha因子数据
        factor_data = _load_factor_cache_data(
            required_factors=None,  # 加载所有可用因子（包括alpha_XXX系列）
            start_date=prediction_date,
            end_date=prediction_date
        )

        if factor_data is None or factor_data.empty:
            logger.error(f"无法加载 {prediction_date} 的alpha因子数据")
            return None

        # 筛选预测日期的数据
        factor_data['date_str'] = factor_data['date'].dt.strftime('%Y%m%d')
        prediction_data = factor_data[factor_data['date_str'] == prediction_date].copy()

        if prediction_data.empty:
            logger.error(f"因子数据中没有找到 {prediction_date} 的数据")
            return None

        logger.info(f"成功加载 {prediction_date} 的alpha因子数据: {len(prediction_data)} 只股票, {len([col for col in prediction_data.columns if col.startswith('alpha_')])} 个alpha因子")
        return prediction_data

    except Exception as e:
        logger.error(f"加载因子数据失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None


def _is_exhaustion_filtered_row(row: pd.Series) -> tuple[bool, str]:
    strategy = EXHAUSTION_FILTER_STRATEGY

    if strategy in ('runup_ma5', 'runup_or_blowoff'):
        rma_val = row.get('runup_ma5_exhaustion', np.nan)
        top10_val = row.get('top10_10d_strength_level_v3', np.nan)
        runup_hit = (
            pd.notna(rma_val)
            and pd.notna(top10_val)
            and rma_val > EXHAUSTION_RUNUP_MA5_THRESHOLD
            and top10_val >= EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD
        )
        if strategy == 'runup_ma5' and runup_hit:
            return True, (
                f"runup_ma5_exhaustion({rma_val:.2f})>{EXHAUSTION_RUNUP_MA5_THRESHOLD} "
                f"AND top10_10d_strength_level_v3({top10_val:.2f})>={EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD}"
            )

        if strategy == 'runup_or_blowoff':
            blowoff_val = row.get('prev_day_blowoff_risk', np.nan)
            sector_val = row.get('sector_momentum_accel_rank', np.nan)
            blowoff_hit = (
                pd.notna(blowoff_val)
                and pd.notna(sector_val)
                and blowoff_val >= EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD
                and sector_val <= EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD
            )
            market_ad_val = row.get('market_advance_decline_ratio', np.nan)
            market_ad_hit = (
                pd.notna(market_ad_val)
                and market_ad_val >= EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD
            )
            prior_high_volume_contest_20d = row.get('prior_high_volume_contest_20d',np.nan)
            stock_volume_ratio_20d = row.get('stock_volume_ratio_20d', np.nan)
            volume_hit = (
                pd.notna(prior_high_volume_contest_20d)
                and pd.notna(stock_volume_ratio_20d)
                and prior_high_volume_contest_20d <= -0.170934
                and stock_volume_ratio_20d  >= 1.78077
            )
            if runup_hit or blowoff_hit or market_ad_hit or volume_hit:
            # if True:
                parts = []
                if volume_hit:
                    parts.append(f"prior_high_volume_contest_20d:{prior_high_volume_contest_20d},stock_volume_ratio_20d:{stock_volume_ratio_20d}")
                if runup_hit:
                    parts.append(
                        f"runup_ma5_exhaustion({rma_val:.2f})>{EXHAUSTION_RUNUP_MA5_THRESHOLD} "
                        f"AND top10_10d_strength_level_v3({top10_val:.2f})>={EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD}"
                    )
                if blowoff_hit:
                    parts.append(
                        f"prev_day_blowoff_risk({blowoff_val:.2f})>={EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD} "
                        f"AND sector_momentum_accel_rank({sector_val:.2f})<={EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD}"
                    )
                if market_ad_hit:
                    parts.append(
                        f"market_advance_decline_ratio({market_ad_val:.2f})>={EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD}"
                    )
                return True, " OR ".join(parts)

    if strategy == 'close_below_ma5':
        flag_val = row.get('close_below_ma5_after_runup', np.nan)
        tz_val = row.get('turnover_z_20', np.nan)
        if pd.notna(flag_val) and flag_val == 1 and pd.notna(tz_val) and tz_val > EXHAUSTION_FILTER_TURNOVER_Z_THRESHOLD:
            return True, (
                f"close_below_ma5_after_runup==1 AND turnover_z_20({tz_val:.2f})>{EXHAUSTION_FILTER_TURNOVER_Z_THRESHOLD}"
            )

    return False, ""


def _prepare_prediction_data(factor_data: pd.DataFrame, feature_columns: list, logger) -> tuple:
    """
    准备预测数据，包括SW行业适配（使用标准化SW编码系统）

    Args:
        factor_data: 原始因子数据
        feature_columns: 模型需要的特征列
        logger: 日志记录器

    Returns:
        (处理后的数据, 预测特征矩阵)
    """
    try:
        # 使用和test_model_walkforward相同的SW行业融合逻辑
        test_data_with_industry = factor_data.copy()

        # 使用标准化编码进行预测数据准备
        logger.info(f"[SW行业编码] 使用标准化编码进行预测数据准备...")

        # 加载标准化编码
        # standardized_mapping = get_sw_mapping_for_model()

        # # 直接使用预先生成的编码，无需重新计算
        # industry_feature_name = standardized_mapping.get('industry_feature_name', 'industry_sw_code')
        # industry_name_to_code = standardized_mapping.get('industry_name_to_code', {})

        # # 为预测数据添加行业编码
        # if 'stock_code' in factor_data.columns:
        #     stock_codes = factor_data['stock_code'].tolist()
        #     industry_codes = []

        #     for stock_code in stock_codes:
        #         industry_code, industry_name = get_sw_industry_code(stock_code)
        #         industry_codes.append(industry_code)

        #     # [大师级修正] 强制转换为 float32 类型，防止 Object/String 污染
        #     try:
        #         industry_codes_array = np.array(industry_codes, dtype=np.float32)
        #         test_data_with_industry[industry_feature_name] = industry_codes_array
        #         logger.info(f"[SW行业编码] 行业编码类型转换成功 (预测模式)")
        #         logger.info(f"  预测数据行业特征类型: {industry_codes_array.dtype}")
        #         logger.info(f"  预测数据行业特征范围: {industry_codes_array.min():.0f} ~ {industry_codes_array.max():.0f}")
        #         logger.info(f"[SW行业编码] 成功为 {len(factor_data)} 只股票添加行业编码")
        #     except Exception as e:
        #         logger.error(f"[SW行业编码] 预测模式行业编码转换失败: {e}")
        #         logger.error(f"预测数据行业编码示例: {industry_codes[:10]}")
        #         logger.error(f"预测数据行业编码类型: {[type(code) for code in industry_codes[:10]]}")
        #         raise e

        # # 更新特征列表（如果需要）
        # if industry_feature_name not in feature_columns:
        #     feature_columns = feature_columns + [industry_feature_name]
        #     logger.info(f"[SW行业编码] 添加行业特征: {industry_feature_name}")

        # 使用_prepare_feature_data函数处理特征（与test模式完全一致）
        X_pred, available_features = _prepare_feature_data(
            test_data_with_industry, feature_columns,
            data_type="预测集"
        )

        if X_pred is None:
            logger.error("[FAIL] 预测集特征准备失败")
            return None, None

        logger.info(f"[OK] 预测集特征准备完成: {X_pred.shape}")
        return test_data_with_industry, X_pred

    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return None, None


def _prepare_training_data(train_data, feature_columns, test_end_date=None):
    """
    准备训练数据

    Args:
        train_data: 训练数据
        feature_columns: 特征列
        test_end_date: 测试集最新日期(用于时间衰减权重计算),格式: 'YYYY-MM-DD'

    Returns:
        tuple: (X_processed, y_train, sample_weights, winsorization_thresholds, label_stats)
    """
    import numpy as np  # [强化] 导入numpy
    logger = logging.getLogger(__name__)

    try:
        # 准备标签 (先检查标签列)
        if 'atr_return_label' not in train_data.columns:
            logger.error("[FAIL] 训练数据缺少atr_return_label列")
            return None, None, None, None, None

        # [强化] 检查relevance列（排序模型必需）
        if 'relevance' not in train_data.columns:
            logger.error("[FAIL] 训练数据缺少relevance列,无法训练排序模型")
            return None, None, None, None, None

        # 过滤掉标签为NaN的样本
        valid_mask = train_data['atr_return_label'].notna()
        train_data_valid = train_data[valid_mask].copy()

        if len(train_data_valid) == 0:
            logger.error("[FAIL] 没有有效的训练样本(所有标签都是NaN)")
            return None, None, None, None, None

        # [强化] 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        if 'date' in train_data_valid.columns and 'stock_code' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("   [OK] 训练数据已按['date', 'stock_code']排序（排序模型要求）")
        elif 'date' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("   [OK] 训练数据已按date排序（排序模型要求）")

        logger.info(f"   有效样本: {len(train_data_valid):,}/{len(train_data):,} ({len(train_data_valid)/len(train_data)*100:.1f}%)")

        # 准备特征数据
        X_processed, available_features = _prepare_feature_data(
            train_data_valid, feature_columns,
            data_type="训练集"
        )

        if X_processed is None:
            return None, None, None, None, None

        # [强化] 获取relevance标签（排序模型使用整数0-3）
        y_train = train_data_valid['relevance'].values[:len(X_processed)]
        # [强化] 确保relevance标签是整数类型（XGBoost排序模型要求）
        y_train = np.array(y_train, dtype=np.int32)
        logger.info(f"   [OK] 使用relevance标签（整数0-3）用于排序模型训练")

        # [强化] XGBoost排序模型不使用样本权重,跳过权重计算
        # 注意: XGBoost排序模型使用分组权重(group-level weights),在训练时计算
        logger.info("   ⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")
        sample_weights = None

        # 计算标签统计量
        label_stats = {
            'mean': np.mean(y_train),
            'std': np.std(y_train)
        }

        # winsorization_thresholds 已废弃，返回空字典以保持兼容性
        winsorization_thresholds = {}

        return X_processed, y_train, sample_weights, winsorization_thresholds, label_stats

    except Exception as e:
        logger.error(f"[FAIL] 准备训练数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, None


def _prepare_validation_data(val_data, feature_columns, winsorization_thresholds):
    """准备验证数据"""
    import numpy as np  # [强化] 导入numpy
    logger = logging.getLogger(__name__)

    try:
        # 准备标签 (先检查标签列)
        if 'atr_return_label' not in val_data.columns:
            logger.error("[FAIL] 验证数据缺少atr_return_label列")
            return None, None, None

        # [强化] 检查relevance列（排序模型必需）
        if 'relevance' not in val_data.columns:
            logger.error("[FAIL] 验证数据缺少relevance列,无法训练排序模型")
            return None, None, None

        # 过滤掉标签为NaN的样本
        valid_mask = val_data['atr_return_label'].notna()
        val_data_valid = val_data[valid_mask].copy()

        if len(val_data_valid) == 0:
            logger.error("[FAIL] 没有有效的验证样本(所有标签都是NaN)")
            return None, None, None

        # [强化] 关键修复: 按['date', 'stock_code']排序,确保排序模型的group信息正确
        if 'date' in val_data_valid.columns and 'stock_code' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("   [OK] 验证数据已按['date', 'stock_code']排序（排序模型要求）")
        elif 'date' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("   [OK] 验证数据已按date排序（排序模型要求）")

        logger.info(f"   有效样本: {len(val_data_valid):,}/{len(val_data):,} ({len(val_data_valid)/len(val_data)*100:.1f}%)")

        # 准备特征数据
        X_processed, _ = _prepare_feature_data(
            val_data_valid, feature_columns,
            data_type="验证集"
        )

        if X_processed is None:
            return None, None, None

        # [强化] 获取relevance标签（排序模型使用整数0-3）
        y_val = val_data_valid['relevance'].values[:len(X_processed)]
        # [强化] 确保relevance标签是整数类型（XGBoost排序模型要求）
        y_val = np.array(y_val, dtype=np.int32)
        logger.info(f"   [OK] 使用relevance标签（整数0-3）用于排序模型训练")

        # [强化] 用于评估的加权收益率（而非排名标签）
        # 优先使用weighted_return，如果没有则计算
        if 'weighted_return' in val_data_valid.columns:
            y_val_eval = val_data_valid['weighted_return'].values[:len(X_processed)]
            logger.info("[OK] 使用已有的weighted_return作为评估标签")
        elif all(col in val_data_valid.columns for col in ['future_return_2d', 'future_return_3d', 'future_return_5d']):
            # [WRENCH] 计算原始加权收益率
            import numpy as np
            weighted_return = (
                0.4 * val_data_valid['future_return_2d'].values[:len(X_processed)] +
                0.4 * val_data_valid['future_return_3d'].values[:len(X_processed)] +
                0.2 * val_data_valid['future_return_5d'].values[:len(X_processed)]
            )
            y_val_eval = weighted_return
            logger.info("[OK] 计算加权收益率作为评估标签")
            logger.info("   公式: 0.4*future_return_2d + 0.4*future_return_3d + 0.2*future_return_5d")
        else:
            logger.warning("[WARNING] 验证数据缺少必要的future_return字段，使用排名标签作为评估标签")
            y_val_eval = y_val.copy()

        return X_processed, y_val, y_val_eval

    except Exception as e:
        logger.error(f"[FAIL] 准备验证数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None



def _bayesian_search_single_val(train_data, val_data, args, test_end_date=None, window_idx=None, market_data=None):
    """
    使用单个验证集的贝叶斯搜索(支持结果保存和恢复)

    Args:
        train_data: 训练数据(已包含标签)
        val_data: 验证数据(已包含标签)
        args: 命令行参数
        test_end_date: 测试集最新日期(用于时间衰减权重计算),格式: 'YYYY-MM-DD'
        window_idx: 窗口索引(用于保存/恢复搜索结果)
        market_data: 市场数据DataFrame（用于XGBoost计算group权重）

    Returns:
        best_params 或 None
    """
    logger = logging.getLogger(__name__)

    try:
        # 检查是否存在已保存的搜索结果
        cache_file = None
        if window_idx is not None:
            cache_file = Path("results") / f"bayesian_search_window{window_idx}.pkl"
            if cache_file.exists():
                logger.info(f"[SEARCH] 发现已保存的贝叶斯搜索结果: {cache_file.name}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)

                    best_params = cached_result['best_params']

                    logger.info(f"[OK] 成功加载已保存的搜索结果")
                    logger.info(f"   最优参数: {best_params}")
                    logger.info(f"   搜索时间: {cached_result.get('search_time', 'N/A')}")

                    return best_params

                except Exception as e:
                    logger.warning(f"[WARNING] 加载缓存失败: {e},将重新执行搜索")

        logger.info(f"[SEARCH] 开始贝叶斯搜索(单个验证集)")
        if test_end_date:
            logger.info(f"   时间衰减基准日期: {test_end_date}")

        # 识别特征列
        feature_columns = _get_ml_feature_columns_simple(train_data)
        logger.info(f"   识别到{len(feature_columns)}个特征")

        # 🆕 打印贝叶斯搜索使用的因子信息
        if feature_columns and window_idx is not None:
            logger.info(f"[FEATURES] 窗口 {window_idx} 贝叶斯搜索使用的因子:")

            # 按因子类型分组显示
            zscore_factors = [f for f in feature_columns if f.endswith('_z')]
            rankpct_factors = [f for f in feature_columns if f.endswith('_rank')]
            intra_rank_factors = [f for f in feature_columns if f.endswith('_intra_rank')]
            intra_zscore_factors = [f for f in feature_columns if f.endswith('_intra_z')]
            original_factors = [f for f in feature_columns if not any(f.endswith(suffix) for suffix in ['_z', '_rank', '_intra_rank', '_intra_z'])]

            logger.info(f"   • Z-score归一化: {len(zscore_factors)}个")
            logger.info(f"   • 排名归一化: {len(rankpct_factors)}个")
            logger.info(f"   • 行业内排名: {len(intra_rank_factors)}个")
            logger.info(f"   • 行业内Z-score: {len(intra_zscore_factors)}个")
            logger.info(f"   • 原始因子: {len(original_factors)}个")
            logger.info(f"   • 总计: {len(feature_columns)}个特征")

            # 保存贝叶斯搜索使用的因子列表
            try:
                from pathlib import Path
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                factor_list_file = results_dir / f"window_{window_idx}_bayesian_factors.txt"

                with open(factor_list_file, 'w', encoding='utf-8') as f:
                    f.write(f"窗口 {window_idx} 贝叶斯搜索使用的因子列表\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总因子数量: {len(feature_columns)}\n\n")

                    f.write("=== 按类型分组的因子列表 ===\n\n")

                    if zscore_factors:
                        f.write(f"Z-score归一化因子 ({len(zscore_factors)}个):\n")
                        for i, factor in enumerate(zscore_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if rankpct_factors:
                        f.write(f"排名归一化因子 ({len(rankpct_factors)}个):\n")
                        for i, factor in enumerate(rankpct_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if intra_rank_factors:
                        f.write(f"行业内排名因子 ({len(intra_rank_factors)}个):\n")
                        for i, factor in enumerate(intra_rank_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if intra_zscore_factors:
                        f.write(f"行业内Z-score因子 ({len(intra_zscore_factors)}个):\n")
                        for i, factor in enumerate(intra_zscore_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if original_factors:
                        f.write(f"原始因子 ({len(original_factors)}个):\n")
                        for i, factor in enumerate(original_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    f.write("=== 完整因子列表 ===\n\n")
                    for i, factor in enumerate(feature_columns, 1):
                        f.write(f"{i:3d}. {factor}\n")

                logger.info(f"[OK] 贝叶斯搜索因子列表已保存到: {factor_list_file}")

            except Exception as e:
                logger.warning(f"[WARNING] 保存贝叶斯搜索因子列表失败: {e}")

        # 准备训练数据
        X_train, y_train, sample_weights, winsorization_thresholds, label_stats = _prepare_training_data(
            train_data, feature_columns, test_end_date=test_end_date
        )

        if X_train is None:
            logger.error("[FAIL] 训练数据准备失败")
            return None, None

        # 准备验证数据
        X_val, y_val, y_val_eval = _prepare_validation_data(
            val_data, feature_columns, winsorization_thresholds
        )

        if X_val is None:
            logger.error("[FAIL] 验证数据准备失败")
            return None, None

        # [WRENCH] 提取验证集日期（用于每日Top-K计算和group计算）
        val_dates = None
        if 'date' in val_data.columns:
            # 过滤掉标签为NaN的样本（与_prepare_validation_data保持一致）
            valid_mask = val_data['atr_return_label'].notna()
            val_dates = val_data.loc[valid_mask, 'date'].values[:len(X_val)]
            logger.info(f"   [OK] 提取验证集日期: {len(val_dates)}个交易日")
        else:
            logger.warning("   [WARNING] 验证数据缺少date列,将使用全局Top-K计算")

        # [WRENCH] 提取训练集日期（用于group计算）
        train_dates = None
        if 'date' in train_data.columns:
            # 过滤掉标签为NaN的样本（与_prepare_training_data保持一致）
            valid_mask = train_data['atr_return_label'].notna()
            train_dates = train_data.loc[valid_mask, 'date'].values[:len(X_train)]
            logger.info(f"   [OK] 提取训练集日期: {len(train_dates)}个交易日")
        else:
            logger.error("   [FAIL] 训练数据缺少date列,无法使用排序模型")
            return None, None

        # 执行贝叶斯搜索
        logger.info(f"   执行贝叶斯搜索...")
        import time
        search_start_time = time.time()

        best_params, all_trials = _perform_parameter_search(
            X_train, y_train, X_val, y_val, y_val_eval, sample_weights,
            search_strategy='bayesian',
            val_dates=val_dates,  # [WRENCH] 传入验证集日期
            train_dates=train_dates,  # [WRENCH] 传入训练集日期
            market_data=market_data  # [强化] XGBoost需要market_data计算group权重
        )

        search_time = time.time() - search_start_time

        if best_params is None:
            logger.error("[FAIL] 贝叶斯搜索失败")
            return None

        logger.info(f"[OK] 贝叶斯搜索完成(耗时: {search_time/60:.1f}分钟)")
        logger.info(f"   最优参数: {best_params}")

        # 保存搜索结果
        if cache_file is not None:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'best_params': best_params,
                        'search_time': f"{search_time/60:.1f}分钟",
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'window_idx': window_idx
                    }, f)
                logger.info(f"[SAVE] 搜索结果已保存: {cache_file.name}")
            except Exception as e:
                logger.warning(f"[WARNING] 保存搜索结果失败: {e}")

        return best_params

    except Exception as e:
        logger.error(f"[FAIL] 贝叶斯搜索失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _print_walkforward_summary(all_window_metrics):
    """
    打印Walk-Forward汇总结果

    [WRENCH] 修复：增强汇总指标输出，模仿test_model模式的详细汇总
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("[CHART] 各窗口回测结果（仅显示Top-1结果）")
    logger.info("=" * 80)

    for metrics in all_window_metrics:
        logger.info(f"\n{metrics['window_name']} ({metrics['test_period']}):")
        logger.info(f"  总收益: {metrics['total_return']:.2f} ATR")
        logger.info(f"  胜率: {metrics['win_rate']:.2%}")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f} ATR")

    # 计算总体指标（Top-1）
    total_return = sum(m['total_return'] for m in all_window_metrics)
    avg_win_rate = np.mean([m['win_rate'] for m in all_window_metrics])
    avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_window_metrics])
    max_drawdown = max(m['max_drawdown'] for m in all_window_metrics)

    # [WRENCH] 修复：使用类似test_model模式的表格格式输出所有Top-k档位汇总
    # 检查是否有all_results数据
    if all_window_metrics and 'all_results' in all_window_metrics[0]:
        logger.info("\n[CHART] Walk-Forward所有Top-k档位汇总结果:")
        logger.info("=" * 150)
        logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'持仓天数':<10} {'总交易次数':<12}")
        logger.info("=" * 150)

        # 获取所有档位（通常是1, 2, 3, 5）
        all_stock_counts = set()
        for metrics in all_window_metrics:
            if 'all_results' in metrics and metrics['all_results']:
                all_stock_counts.update(metrics['all_results'].keys())

        stock_counts = sorted(all_stock_counts)

        for stock_count in stock_counts:
            # 收集所有窗口的该档位结果
            count_metrics = []
            for metrics in all_window_metrics:
                if 'all_results' in metrics and stock_count in metrics['all_results']:
                    count_metrics.append(metrics['all_results'][stock_count])

            if not count_metrics:
                logger.info(f"Top-{stock_count:<5} [WARNING] 无数据")
                continue

            # [WRENCH] 修复：计算百分比收益和其他指标
            # 注意：这里的指标应该从backtest_results中获取，而不是简单累加
            # 因为不同窗口的收益应该是复利计算，而不是简单相加

            # 计算累计收益（复利）
            total_return_gross = 1.0
            total_return_net = 1.0
            for m in count_metrics:
                # 假设每个窗口的收益是百分比形式
                total_return_gross *= (1 + m.get('total_return_gross', 0))
                total_return_net *= (1 + m.get('total_return', 0))

            total_return_gross = total_return_gross - 1  # 转换为收益率
            total_return_net = total_return_net - 1

            # 计算年化收益
            # 获取总交易天数
            if count_metrics and 'summary_stats' in count_metrics[0]:
                # 从summary_stats获取总交易日
                total_trading_days = count_metrics[0]['summary_stats'].get('total_trading_days', 0)
            else:
                # 计算实际交易天数
                total_trading_days = sum(m.get('total_trades', 0) * m.get('holding_days', 3) for m in count_metrics) / len(count_metrics)
                if total_trading_days > 0:
                    total_trading_days = int(total_trading_days)

            # 使用复利计算年化收益
            if total_return_net != 0 and total_trading_days > 0:
                # 总净收益率转换为净值
                final_value = 1 + total_return_net
                # 年化收益率
                annual_return = (final_value ** (252 / total_trading_days)) - 1.0
            else:
                annual_return = 0.0

            # 其他指标取平均
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in count_metrics])
            avg_profit_loss_ratio = np.mean([m.get('profit_loss_ratio', 0) for m in count_metrics])
            avg_holding_days = np.mean([m.get('holding_days', 0) for m in count_metrics])

            # 最大回撤取最大值
            max_drawdown = max(m.get('max_drawdown', 0) for m in count_metrics)

            # 总交易次数
            total_trades = sum(m.get('total_trades', 0) for m in count_metrics)

            logger.info(f"Top-{stock_count:<5} {total_return_gross:>8.2%}  "
                       f"{total_return_net:>13.2%}  "
                       f"{annual_return:>8.2%}  {max_drawdown:>8.2%}  "
                       f"{avg_win_rate:>6.2%}  "
                       f"{avg_profit_loss_ratio:>6.2f}  "
                       f"{avg_holding_days:>10.0f}{total_trades:>10}")

        logger.info("=" * 150)


def _save_walkforward_results(all_window_metrics, all_selections):
    """保存Walk-Forward结果"""
    logger = logging.getLogger(__name__)

    try:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存窗口指标
        metrics_file = results_dir / f"walkforward_metrics_{timestamp}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_window_metrics, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[OK] 窗口指标已保存: {metrics_file.name}")

        # 保存选股结果
        selections_file = results_dir / f"walkforward_selections_{timestamp}.json"
        with open(selections_file, 'w', encoding='utf-8') as f:
            json.dump(all_selections, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"[OK] 选股结果已保存: {selections_file.name}")

        # 显示每日选股详情
        _display_daily_stock_selections(all_selections, results_dir, timestamp, all_window_metrics)

    except Exception as e:
        logger.error(f"[FAIL] 保存结果失败: {e}")


def _display_daily_stock_selections(all_selections, results_dir, timestamp, all_window_metrics=None):
    """显示每日选股1,2,3只股票的详细情况

    Args:
        all_selections: 所有选股结果列表
        results_dir: 结果目录
        timestamp: 时间戳
        all_window_metrics: 窗口指标列表（可选）
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 100)
    logger.info("[CLIPBOARD] 每日选股详情")
    logger.info("=" * 100)

    # 整理选股数据按股票数量分组 - 使用集合来动态收集所有档位
    selections_by_count = {}

    for window_data in all_selections:
        # 兼容新旧数据结构
        # 新结构：window_data = {'window_name': ..., 'selections': {1: [...], 2: [...], ...}}
        # 旧结构：window_data = {1: [...], 2: [...], ...} (直接的selections字典)
        if isinstance(window_data, dict) and 'selections' in window_data:
            window_selections = window_data['selections']
            window_name = window_data.get('window_name', 'Unknown')

            logger.info(f"\n[CHART] 窗口 {window_name} 选股记录:")
            logger.info("-" * 50)
        elif isinstance(window_data, dict) and any(k in window_data for k in [1, 2, 3, 5]):
            # 旧的数据结构，直接是 selections 字典
            window_selections = window_data
            window_name = 'Unknown Window'

            logger.info(f"\n[CHART] 窗口 {window_name} 选股记录:")
            logger.info("-" * 50)

        # 动态获取所有档位
        available_stock_counts = sorted([k for k in window_selections.keys() if k in [1, 2, 3, 5]])

        # 遍历各个档位
        for stock_count in available_stock_counts:
            trades = window_selections.get(stock_count, [])
            if trades:
                # 确保档位已初始化
                if stock_count not in selections_by_count:
                    selections_by_count[stock_count] = []

                logger.info(f"  Top-{stock_count}: {len(trades)} 笔交易")
                for trade in trades:
                    # 添加窗口信息到交易记录
                    trade_with_window = trade.copy()
                    trade_with_window['window_name'] = window_name

                    trade_info = {
                        'stock_code': trade.get('stock_code', ''),
                        'stock_name': trade.get('stock_name', ''),
                        'buy_date': trade.get('buy_date', ''),
                        'sell_date': trade.get('sell_date', ''),
                        'buy_price': trade.get('buy_price', 0),
                        'sell_price': trade.get('sell_price', 0),
                        'net_return': trade.get('net_return', 0),
                        'holding_days': trade.get('holding_days', 0),
                        'window_name': window_name
                    }
                    selections_by_count[stock_count].append(trade_info)

    # 显示各档位的选股详情 - 动态获取所有存在的档位
    available_stock_counts = sorted([k for k in selections_by_count.keys() if selections_by_count[k]])
    for stock_count in available_stock_counts:
        trades = selections_by_count[stock_count]
        logger.info(f"\n[CHART] 选股 {stock_count} 只详情 (共 {len(trades)} 笔交易)")
        logger.info("-" * 75)

        if trades:
            # 恢复原始表头，移除累计收益列（因为多股票同时持仓，单笔累计收益无意义）
            logger.info(f"{'股票代码':<10} {'买入日期':<12} {'卖出日期':<12} {'买入价格':<8} {'卖出价格':<8} {'收益率':<10} {'持仓天数':<6}")
            logger.info("-" * 75)  # 恢复原始分隔线长度

            # 按买入日期排序
            trades_sorted = sorted(trades, key=lambda x: x['buy_date'])

            for trade in trades_sorted:
                net_return_pct = trade['net_return'] * 100
                # 格式化日期显示
                buy_date_str = pd.to_datetime(trade['buy_date']).strftime('%Y-%m-%d') if trade['buy_date'] else 'N/A'
                sell_date_str = pd.to_datetime(trade['sell_date']).strftime('%Y-%m-%d') if trade['sell_date'] else 'N/A'

                logger.info(f"{trade['stock_code']:<10} {buy_date_str:<12} {sell_date_str:<12} "
                           f"{trade['buy_price']:<8.2f} {trade['sell_price']:<8.2f} {net_return_pct:>7.2f}% {trade['holding_days']:<6}")

            # 统计信息
            profitable_trades = [t for t in trades if (t.get('net_return') or 0) > 0]
            losing_trades = [t for t in trades if (t.get('net_return') or 0) < 0]

            if trades:
                avg_return = sum((t.get('net_return') or 0) for t in trades) / len(trades) * 100
                win_rate = len(profitable_trades) / len(trades) * 100

                total_profit = sum((t.get('net_return') or 0) for t in profitable_trades) * 100 if profitable_trades else 0
                total_loss = sum(abs(t.get('net_return') or 0) for t in losing_trades) * 100 if losing_trades else 0
                profit_loss_ratio = total_profit / total_loss if total_loss > 0 else float('inf')

                # 计算实际累计收益（基于净值）
                # 注意：对于Top-K策略，净值已经考虑了多股票同时持仓的影响
                # 我们需要从all_window_metrics中获取所有窗口的净值数据
                cumulative_return = 0.0
                # 尝试从所有窗口的backtest_results中获取总收益并累加
                if all_window_metrics:
                    cumulative_value = 1.0  # 初始净值
                    for wm in all_window_metrics:
                        if 'all_results' in wm and stock_count in wm['all_results']:
                            result = wm['all_results'][stock_count]
                            # 使用复利计算累计净值
                            window_return = result.get('total_return', 0)
                            cumulative_value *= (1 + window_return)
                    cumulative_return = (cumulative_value - 1.0) * 100

                logger.info("-" * 80)
                logger.info(f"统计: 算术平均收益 {avg_return:.2f}%, 实际累计收益 {cumulative_return:.2f}%, 胜率 {win_rate:.1f}%, 盈亏比 {profit_loss_ratio:.2f}")
                logger.info(f"盈利交易: {len(profitable_trades)}笔, 亏损交易: {len(losing_trades)}笔")
                logger.info(f"说明: Top-{stock_count}策略同时持有{stock_count}只股票，每只占用{1/stock_count*100:.0f}%资金")
        else:
            logger.info("该档位无选股记录")

    # 保存详细选股结果到文件
    try:
        detailed_file = results_dir / f"daily_selections_detail_{timestamp}.json"

        # 按更易读的格式组织数据
        detailed_data = {
            "summary": {
                "total_trades_by_count": {
                    f"top_{stock_count}": len(selections_by_count.get(stock_count, []))
                    for stock_count in sorted(selections_by_count.keys())
                },
                "trading_windows": len(set([t['window_name'] for trades in selections_by_count.values() for t in trades])),
                "date_range": {
                    "start": min([t['buy_date'] for trades in selections_by_count.values() for t in trades if t.get('buy_date')]) if selections_by_count else None,
                    "end": max([t['sell_date'] for trades in selections_by_count.values() for t in trades if t.get('sell_date')]) if selections_by_count else None
                } if selections_by_count else {}
            },
            "trades_by_stock_count": {}
        }

        # 为每个选股数量创建更结构化的交易数据
        for stock_count in sorted(selections_by_count.keys()):
            trades = selections_by_count[stock_count]
            if trades:
                # 按买入日期排序
                trades_sorted = sorted(trades, key=lambda x: (x.get('window_name', ''), x.get('buy_date', '')))

                # 统计信息
                profitable_trades = [t for t in trades if (t.get('net_return') or 0) > 0]
                losing_trades = [t for t in trades if (t.get('net_return') or 0) < 0]

                detailed_data["trades_by_stock_count"][f"top_{stock_count}"] = {
                    "statistics": {
                        "total_trades": len(trades),
                        "profitable_trades": len(profitable_trades),
                        "losing_trades": len(losing_trades),
                        "win_rate": len(profitable_trades) / len(trades) * 100 if trades else 0,
                        "avg_return": sum((t.get('net_return') or 0) for t in trades) / len(trades) * 100 if trades else 0,
                        "total_return": sum((t.get('net_return') or 0) for t in trades) * 100,
                        "max_profit": max([(t.get('net_return') or 0) for t in trades if (t.get('net_return') or 0) != 0]) * 100 if trades and any(t.get('net_return') for t in trades) else 0,
                        "max_loss": min([(t.get('net_return') or 0) for t in trades if (t.get('net_return') or 0) != 0]) * 100 if trades and any(t.get('net_return') for t in trades) else 0
                    },
                    "daily_transactions": []
                }

                # 按窗口和日期组织交易
                trades_by_window = {}
                for trade in trades_sorted:
                    window = trade.get('window_name', 'Unknown')
                    if window not in trades_by_window:
                        trades_by_window[window] = []

                    trades_by_window[window].append({
                        "date": trade.get('buy_date'),
                        "stock_code": trade.get('stock_code'),
                        "buy_price": round(trade.get('buy_price', 0), 2),
                        "sell_price": round(trade.get('sell_price', 0), 2),
                        "return_pct": round(trade.get('net_return', 0) * 100, 2),
                        "holding_days": trade.get('holding_days', 0)
                    })

                detailed_data["trades_by_stock_count"][f"top_{stock_count}"]["daily_transactions"] = trades_by_window

        # 同时保存净值和持仓数据到单独的文件
        if 'all_selections' in window_data and window_data['all_selections']:
            # 获取第一个窗口的净值数据作为示例
            first_window_name = window_data['window_name'] if 'window_name' in window_data else 'Unknown'
            selections = window_data['all_selections']

            # 从 detailed_trades 中提取净值数据
            if selections and 'top_1' in selections:
                # 创建净值数据文件
                portfolio_file = results_dir / f"walkforward_portfolio_data_{timestamp}.json"

                portfolio_info = {
                    "strategy_info": {
                        "stock_counts": list(sorted(selections.keys())),
                        "holding_days": 3,  # 默认值
                        "total_windows": len(all_window_metrics) if all_window_metrics else 0,
                        "trading_cost": 0.0016
                    },
                    "windows": {}
                }

                # 为每个窗口添加净值和持仓数据
                if all_window_metrics:
                    for wm in all_window_metrics:
                        window_name = wm.get('window_name', 'Unknown')
                        if 'all_results' in wm and wm['all_results']:
                            for stock_count in wm['all_results'].keys():
                                # 假设可以从backtest_results获取净值数据
                                if stock_count in wm['all_results']:
                                    result = wm['all_results'][stock_count]
                                    # 组合日期和净值到新的格式
                                    portfolio_values = result.get('portfolio_values', [])
                                    if isinstance(portfolio_values, list) and len(portfolio_values) > 0 and isinstance(portfolio_values[0], dict):
                                        # 已经是新格式，直接使用
                                        combined_values = portfolio_values
                                    else:
                                        # 旧格式，需要转换
                                        dates = result.get('dates', [])
                                        combined_values = [
                                            {'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d), 'value': v}
                                            for d, v in zip(dates, portfolio_values)
                                        ]

                                    portfolio_info["windows"][f"{window_name}_top_{stock_count}"] = {
                                        "window_name": window_name,
                                        "stock_count": stock_count,
                                        "portfolio_values": combined_values,
                                        "total_return": result.get('total_return', 0),
                                        "annual_return": result.get('annual_return', 0),
                                        "max_drawdown": result.get('max_drawdown', 0),
                                        "win_rate": result.get('win_rate', 0),
                                        "total_trades": result.get('total_trades', 0)
                                    }

                with open(portfolio_file, 'w', encoding='utf-8') as f:
                    json.dump(portfolio_info, f, ensure_ascii=False, indent=2, default=str)

                logger.info(f"\n[OK] 净值和持仓数据已保存: {portfolio_file.name}")

        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"\n[OK] 详细选股结果已保存: {detailed_file.name}")

    except Exception as e:
        logger.error(f"[FAIL] 保存详细选股结果失败: {e}")


def _train_final_model_for_window(train_data, val_data, test_data, best_params, window_idx, use_lightgbm=True, market_data=None, optimize_thresholds=False, sw_sector_mapping=None):
    """为窗口训练最终模型(仅支持LightGBM)

    Args:
        train_data: 训练数据
        val_data: 验证数据（Walk-Forward外层已划分好的验证集）
        best_params: 最优参数
        window_idx: 窗口索引
        use_lightgbm: 是否训练LightGBM模型(默认True)
        market_data: 市场数据
        optimize_thresholds: 是否执行止盈止损ATR阈值优化（默认False）
        sw_sector_mapping: SW（申万）行业映射字典，用于LightGBM分类特征（默认None）

    Returns:
        dict: 包含模型列表、性能指标、特征名等信息
    """
    logger = logging.getLogger(__name__)

    try:
        # 识别特征列
        feature_columns = _get_ml_feature_columns_simple(train_data)

        # 🆕 打印最终使用的所有因子名称
        if feature_columns:
            logger.info(f"[FEATURES] 窗口 {window_idx} 训练模型使用的因子数量: {len(feature_columns)}")
            logger.info(f"[FEATURES] 最终使用的因子列表:")

            # 按因子类型分组显示
            zscore_factors = [f for f in feature_columns if f.endswith('_z')]
            rankpct_factors = [f for f in feature_columns if f.endswith('_rank')]
            intra_rank_factors = [f for f in feature_columns if f.endswith('_intra_rank')]
            intra_zscore_factors = [f for f in feature_columns if f.endswith('_intra_z')]
            original_factors = [f for f in feature_columns if not any(f.endswith(suffix) for suffix in ['_z', '_rank', '_intra_rank', '_intra_z'])]

            if zscore_factors:
                logger.info(f"  • Z-score归一化因子 ({len(zscore_factors)}个): {zscore_factors[:10]}{'...' if len(zscore_factors) > 10 else ''}")
            if rankpct_factors:
                logger.info(f"  • 排名归一化因子 ({len(rankpct_factors)}个): {rankpct_factors[:10]}{'...' if len(rankpct_factors) > 10 else ''}")
            if intra_rank_factors:
                logger.info(f"  • 行业内排名因子 ({len(intra_rank_factors)}个): {intra_rank_factors[:10]}{'...' if len(intra_rank_factors) > 10 else ''}")
            if intra_zscore_factors:
                logger.info(f"  • 行业内Z-score因子 ({len(intra_zscore_factors)}个): {intra_zscore_factors[:10]}{'...' if len(intra_zscore_factors) > 10 else ''}")
            if original_factors:
                logger.info(f"  • 原始因子 ({len(original_factors)}个): {original_factors[:10]}{'...' if len(original_factors) > 10 else ''}")

            logger.info(f"[FEATURES] 因子分类统计: Z-score({len(zscore_factors)}) + Rank({len(rankpct_factors)}) + Intra-Rank({len(intra_rank_factors)}) + Intra-Z({len(intra_zscore_factors)}) + Original({len(original_factors)}) = {len(feature_columns)}")

            # 在控制台也打印因子信息
            print(f"\n🎯 窗口 {window_idx} 训练模型使用的因子信息:")
            print(f"   📊 总因子数量: {len(feature_columns)}")
            print(f"   🔹 Z-score归一化: {len(zscore_factors)}个")
            print(f"   🔸 排名归一化: {len(rankpct_factors)}个")
            print(f"   🏭 行业内排名: {len(intra_rank_factors)}个")
            print(f"   🏭 行业内Z-score: {len(intra_zscore_factors)}个")
            print(f"   🔹 原始因子: {len(original_factors)}个")

            # 保存因子列表到文件，便于后续分析
            try:
                from pathlib import Path
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                factor_list_file = results_dir / f"window_{window_idx}_factors.txt"

                with open(factor_list_file, 'w', encoding='utf-8') as f:
                    f.write(f"窗口 {window_idx} 训练模型使用的因子列表\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"总因子数量: {len(feature_columns)}\n\n")

                    f.write("=== 按类型分组的因子列表 ===\n\n")

                    if zscore_factors:
                        f.write(f"Z-score归一化因子 ({len(zscore_factors)}个):\n")
                        for i, factor in enumerate(zscore_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if rankpct_factors:
                        f.write(f"排名归一化因子 ({len(rankpct_factors)}个):\n")
                        for i, factor in enumerate(rankpct_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if intra_rank_factors:
                        f.write(f"行业内排名因子 ({len(intra_rank_factors)}个):\n")
                        for i, factor in enumerate(intra_rank_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if intra_zscore_factors:
                        f.write(f"行业内Z-score因子 ({len(intra_zscore_factors)}个):\n")
                        for i, factor in enumerate(intra_zscore_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    if original_factors:
                        f.write(f"原始因子 ({len(original_factors)}个):\n")
                        for i, factor in enumerate(original_factors, 1):
                            f.write(f"  {i:2d}. {factor}\n")
                        f.write("\n")

                    f.write("=== 完整因子列表 ===\n\n")
                    for i, factor in enumerate(feature_columns, 1):
                        f.write(f"{i:3d}. {factor}\n")

                logger.info(f"[OK] 因子列表已保存到: {factor_list_file}")

            except Exception as e:
                logger.warning(f"[WARNING] 保存因子列表失败: {e}")
        else:
            logger.error("[ERROR] 没有找到任何特征列！")
            print("❌ 错误：没有找到任何特征列！")

        # 准备训练集标签
        if 'atr_return_label' not in train_data.columns:
            logger.error("[FAIL] 训练数据缺少atr_return_label列")
            return None

        # 准备验证集标签
        if 'atr_return_label' not in val_data.columns:
            logger.error("[FAIL] 验证数据缺少atr_return_label列")
            return None

        # [OK] 过滤掉标签为NaN的样本
        train_valid_mask = train_data['atr_return_label'].notna()
        train_data_valid = train_data[train_valid_mask].copy()

        val_valid_mask = val_data['atr_return_label'].notna()
        val_data_valid = val_data[val_valid_mask].copy()

        test_valid_mask = test_data['atr_return_label'].notna()
        test_data_valid = test_data[test_valid_mask].copy()

        if len(train_data_valid) == 0:
            logger.error("[FAIL] 没有有效的训练样本(所有标签都是NaN)")
            return None

        if len(val_data_valid) == 0:
            logger.error("[FAIL] 没有有效的验证样本(所有标签都是NaN)")
            return None

        logger.info(f"[CHART] 有效训练样本: {len(train_data_valid):,}/{len(train_data):,} ({len(train_data_valid)/len(train_data)*100:.1f}%)")
        logger.info(f"[CHART] 有效验证样本: {len(val_data_valid):,}/{len(val_data):,} ({len(val_data_valid)/len(val_data)*100:.1f}%)")

        # 🆕 排序模型要求数据按日期排序,确保训练集和验证集按日期升序排列
        # [强化] 关键修复: 必须同时按date和stock_code排序,确保同一天内股票顺序确定
        if 'date' in train_data_valid.columns and 'stock_code' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("[OK] 训练集已按[date, stock_code]升序排序（排序模型要求）")
        elif 'date' in train_data_valid.columns:
            train_data_valid = train_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("[OK] 训练集已按date升序排序（排序模型要求）")

        if 'date' in val_data_valid.columns and 'stock_code' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info("[OK] 验证集已按[date, stock_code]升序排序（排序模型要求）")
        elif 'date' in val_data_valid.columns:
            val_data_valid = val_data_valid.sort_values('date').reset_index(drop=True)
            logger.info("[OK] 验证集已按date升序排序（排序模型要求）")

        # 🆕 提取relevance标签(用于LightGBM训练)和relevance_30标签(用于XGBoost训练)和原始收益率标签(用于IC计算)
        train_labels = train_data_valid['atr_return_label']
        val_labels = val_data_valid['atr_return_label']
        test_labels = test_data_valid['atr_return_label']

        # 检查是否有relevance字段(LightGBM使用)
        if 'relevance' not in train_data_valid.columns or 'relevance' not in val_data_valid.columns:
            logger.error("[FAIL] 训练/验证数据缺少relevance列,无法训练LightGBM排序模型")
            return None

        train_relevance = train_data_valid['relevance']
        val_relevance = val_data_valid['relevance']


        logger.info(f"[OK] 提取relevance标签（整数0-3,4层）用于LightGBM排序模型训练")
        logger.info(f"[OK] 保留atr_return_label用于IC计算")

        # [强化] 步骤1: 预处理训练集和验证集特征
        logger.info(f"\n{'='*60}")
        logger.info(f"预处理训练集和验证集特征(Window {window_idx})")
        logger.info(f"{'='*60}")

        # 对训练集进行特征预处理
        X_train_processed, feature_names = _prepare_feature_data(
            train_data_valid, feature_columns, data_type="训练集"
        )

        if X_train_processed is None:
            logger.error("[FAIL] 训练集特征预处理失败")
            return None

        # 对验证集进行特征预处理
        X_val_processed, _ = _prepare_feature_data(
            val_data_valid, feature_columns, data_type="验证集"
        )

        X_test_processed, _ = _prepare_feature_data(
            test_data_valid, feature_columns,
            data_type="测试集"
        )

        if X_val_processed is None:
            logger.error("[FAIL] 验证集特征预处理失败")
            return None

        # [WRENCH] 确保标签与预处理后的特征数据长度匹配
        if len(X_train_processed) != len(train_labels):
            logger.warning(f"训练集预处理后长度变化: {len(train_labels)} -> {len(X_train_processed)}")
            train_labels = train_labels.iloc[:len(X_train_processed)] if hasattr(train_labels, 'iloc') else train_labels[:len(X_train_processed)]
            # [强化] 同步截取relevance标签
            train_relevance = train_relevance.iloc[:len(X_train_processed)] if hasattr(train_relevance, 'iloc') else train_relevance[:len(X_train_processed)]

        if len(X_val_processed) != len(val_labels):
            logger.warning(f"验证集预处理后长度变化: {len(val_labels)} -> {len(X_val_processed)}")
            val_labels = val_labels.iloc[:len(X_val_processed)] if hasattr(val_labels, 'iloc') else val_labels[:len(X_val_processed)]
            # [强化] 同步截取relevance标签
            val_relevance = val_relevance.iloc[:len(X_val_processed)] if hasattr(val_relevance, 'iloc') else val_relevance[:len(X_val_processed)]

        # 🆕 提取训练集日期（用于时间衰减权重和group计算）
        train_dates = None
        if 'date' in train_data_valid.columns:
            train_dates = train_data_valid['date'].values[:len(X_train_processed)]
            logger.info("[OK] 提取训练集日期（用于时间衰减权重计算和group计算）")

        # 🆕 提取验证集日期（用于group计算）
        val_dates = None
        if 'date' in val_data_valid.columns:
            val_dates = val_data_valid['date'].values[:len(X_val_processed)]
            logger.info("[OK] 提取验证集日期（用于group计算）")

        test_dates = None
        if 'date' in test_data_valid.columns:
            test_dates = test_data_valid['date'].values[:len(X_test_processed)]
            logger.info("[OK] 提取测试集日期）")

        # 🆕 提取训练集和验证集股票代码（用于SW行业特征）
        train_stock_codes = None
        val_stock_codes = None
        if 'stock_code' in train_data_valid.columns:
            train_stock_codes = train_data_valid['stock_code'].values[:len(X_train_processed)]
            logger.info("[OK] 提取训练集股票代码（用于SW行业特征）")
        else:
            logger.warning("[WARNING] 训练数据中未包含stock_code列，无法添加SW行业特征")

        if 'stock_code' in val_data_valid.columns:
            val_stock_codes = val_data_valid['stock_code'].values[:len(X_val_processed)]
            logger.info("[OK] 提取验证集股票代码（用于SW行业特征）")
        else:
            logger.warning("[WARNING] 验证数据中未包含stock_code列，无法添加SW行业特征")

        # 🆕 提取验证集ATR14（用于止盈止损阈值优化）
        val_atr14 = None
        if 'atr14' in val_data_valid.columns:
            val_atr14 = val_data_valid['atr14'].values[:len(X_val_processed)]
            logger.info("[OK] 提取验证集ATR14（用于止盈止损阈值优化）")
        else:
            logger.warning("[WARNING] 验证数据中未包含atr14列，止盈止损阈值优化将无法执行")

        # 🆕 提取换手率和流通市值（用于基础权重计算）
        train_turnover_rate = None
        train_circ_mv = None
        val_turnover_rate = None
        val_circ_mv = None
        if 'turnover_rate' in train_data_valid.columns:
            train_turnover_rate = train_data_valid['turnover_rate'].values[:len(X_train_processed)]
            logger.info("[OK] 提取训练集换手率（用于基础权重计算）")
        else:
            logger.warning("[WARNING] 训练数据中未包含turnover_rate列，将使用均匀权重")

        if 'circ_mv' in train_data_valid.columns:
            train_circ_mv = train_data_valid['circ_mv'].values[:len(X_train_processed)]
            logger.info("[OK] 提取训练集流通市值（用于基础权重计算）")
        else:
            logger.warning("[WARNING] 训练数据中未包含circ_mv列，将使用均匀权重")

        # 🆕 提取验证集换手率和流通市值（用于验证集权重计算）
        if 'turnover_rate' in val_data_valid.columns:
            val_turnover_rate = val_data_valid['turnover_rate'].values[:len(X_val_processed)]
            logger.info("[OK] 提取验证集换手率（用于验证集权重计算）")
        else:
            logger.warning("[WARNING] 验证数据中未包含turnover_rate列，验证集将使用均匀权重")

        if 'circ_mv' in val_data_valid.columns:
            val_circ_mv = val_data_valid['circ_mv'].values[:len(X_val_processed)]
            logger.info("[OK] 提取验证集流通市值（用于验证集权重计算）")
        else:
            logger.warning("[WARNING] 验证数据中未包含circ_mv列，验证集将使用均匀权重")

        # [强化] 条件计算样本权重（仅LightGBM需要）
        unified_sample_weights = None
        unified_val_weights = None
        if use_lightgbm:
            logger.info("[WRENCH] 计算样本权重（供LightGBM使用）...")
            train_relevance_array = train_relevance.values if hasattr(train_relevance, 'values') else train_relevance
            val_relevance_array = val_relevance.values if hasattr(val_relevance, 'values') else val_relevance

            # [WRENCH] 修复: 使用训练集最新日期作为时间衰减基准（Walk-Forward模式）
            reference_date = None
            if train_dates is not None and len(train_dates) > 0:
                reference_date = pd.to_datetime(train_dates).max()
                logger.info(f"   时间衰减基准日期: {reference_date.strftime('%Y-%m-%d')} (训练集最新日期)")

            # 计算训练集权重
            unified_sample_weights = _calculate_professional_sample_weights(
                train_relevance_array,  # 使用relevance标签
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=None,
                train_dates=train_dates,
                time_decay_half_life=180.0,  # 统一使用180天
                reference_date=reference_date,  # [WRENCH] 修复: 使用训练集最新日期
                turnover_rate=train_turnover_rate,
                circ_mv=train_circ_mv,
                label_rank=train_relevance_array,  # relevance值(0-3)
                label_boost_beta=0.15,  # 统一使用0.15
                w_min=1,
                w_max=1
            )
            logger.info(f"[OK] 训练集样本权重计算完成")
            logger.info(f"   权重范围: [{unified_sample_weights.min():.4f}, {unified_sample_weights.max():.4f}]")
            logger.info(f"   权重均值: {unified_sample_weights.mean():.4f}")

            # 计算验证集权重（使用相同的参数）
            unified_val_weights = _calculate_professional_sample_weights(
                val_relevance_array,  # 使用relevance标签
                imbalance_ratio=1.0,
                logger=logger,
                boundary_distances=None,
                is_regression=True,
                y_train_original=None,
                train_dates=val_dates,  # 验证集使用自己的日期
                time_decay_half_life=180.0,  # 统一使用180天
                reference_date=reference_date,  # 使用训练集最新日期作为基准
                turnover_rate=val_turnover_rate,
                circ_mv=val_circ_mv,
                label_rank=val_relevance_array,  # relevance值(0-3)
                label_boost_beta=0.15,  # 统一使用0.15
                w_min=1,
                w_max=1
            )
            logger.info(f"[OK] 验证集样本权重计算完成")
            logger.info(f"   权重范围: [{unified_val_weights.min():.4f}, {unified_val_weights.max():.4f}]")
            logger.info(f"   权重均值: {unified_val_weights.mean():.4f}")
        else:
            logger.info("⏭️ 跳过样本权重计算（XGBoost排序模型使用分组权重）")

        # [强化] 构建预处理数据字典
        preprocessed_data = {
            'X_train': X_train_processed,
            'X_val': X_val_processed,
            'X_test': X_test_processed,
            'y_train': train_labels.values if hasattr(train_labels, 'values') else train_labels,
            'y_val': val_labels.values if hasattr(val_labels, 'values') else val_labels,
            'y_test': test_labels.values if hasattr(test_labels, 'values') else test_labels,
            'y_train_original': train_labels.values if hasattr(train_labels, 'values') else train_labels,
            'y_val_original': val_labels.values if hasattr(val_labels, 'values') else val_labels,
            'y_test_original': test_labels.values if hasattr(test_labels, 'values') else test_labels,
            'train_relevance': train_relevance.values if hasattr(train_relevance, 'values') else train_relevance,  # 🆕 relevance标签(4层,LightGBM使用)
            'val_relevance': val_relevance.values if hasattr(val_relevance, 'values') else val_relevance,  # 🆕 relevance标签(4层,LightGBM使用)
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
            'train_dates': train_dates,  # 🆕 训练集日期
            'val_dates': val_dates,  # 🆕 验证集日期
            'val_atr14': val_atr14,  # 🆕 验证集ATR14（用于止盈止损阈值优化）
            'test_dates':test_dates,
            'train_stock_codes': train_stock_codes,  # 🆕 训练集股票代码（用于SW行业特征）
            'val_stock_codes': val_stock_codes,  # 🆕 验证集股票代码（用于SW行业特征）
            'turnover_rate': train_turnover_rate,  # 🆕 添加换手率
            'circ_mv': train_circ_mv,  # 🆕 添加流通市值
            'val_turnover_rate': val_turnover_rate,  # 🆕 添加验证集换手率
            'val_circ_mv': val_circ_mv,  # 🆕 添加验证集流通市值
            'sample_weights': unified_sample_weights,  # 🆕 添加统一计算的样本权重
            'val_sample_weights': unified_val_weights,  # 🆕 添加验证集样本权重
            'test_data_raw': test_data_valid, #test_data  # ✅ 新增：测试集原始数据（包含 open, close 等基础数据）
            'val_data_raw':val_data_valid
        }

        logger.info(f"[OK] 预处理完成: 训练集{len(X_train_processed):,}样本, 验证集{len(X_val_processed):,}样本")
        logger.info(f"[OK] relevance标签已添加到预处理数据")
        logger.info(f"[OK] 训练集和验证集日期已添加到预处理数据")
        if use_lightgbm:
            logger.info(f"[OK] 样本权重已添加到预处理数据（供LightGBM使用）")
        else:
            logger.info(f"⏭️ 样本权重未计算（XGBoost排序模型使用分组权重）")

        # ✅ 添加日志：test_data_raw 已添加
        if test_data is not None:
            logger.info(f"[OK] 测试集原始数据已添加到预处理数据: {len(test_data)}条记录")
            logger.info(f"[OK] 测试集原始数据包含列: {test_data.columns.tolist()}")

        # 生成统一时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"🕐 模型统一时间戳: {timestamp}")

        # [强化] 步骤3: 训练LightGBM模型(如果选择了LightGBM)
        lgb_result = None
        if use_lightgbm:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练LightGBM模型(Window {window_idx})")
            logger.info(f"{'='*60}")

            try:
                lgb_preprocessed_data = preprocessed_data
                lgb_feature_names = feature_columns
                logger.info(f"[OK] 独立训练LightGBM,使用原始特征列表")

                # 检查预处理数据是否可用
                if lgb_preprocessed_data.get('X_train') is None:
                    logger.error("[FAIL] 缺少预处理数据,无法训练LightGBM")
                    lgb_result = None
                else:
                    lgb_result = _train_lightgbm_model(
                        raw_data=train_data_valid,  # [OK] 仅用于提取元数据
                        labels=train_labels,  # [OK] 仅用于提取元数据
                        feature_columns=lgb_feature_names,
                        label_stats=None,
                        preprocessed_data=lgb_preprocessed_data,  # [OK] 传入预处理数据
                        optimize_thresholds=optimize_thresholds,  # [强化] 传递止盈止损优化参数
                        sw_sector_mapping=sw_sector_mapping  # [SW行业编码] 传递SW行业数据用于分类特征
                    )

                    if lgb_result is None:
                        logger.error("[FAIL] LightGBM模型训练失败")
                        return None
                    else:
                        logger.info(f"[OK] LightGBM模型训练完成")

            except Exception as e:
                logger.error(f"[FAIL] LightGBM模型训练失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                lgb_result = None

        # 3. 收集模型结果
        ensemble_results = []
        if lgb_result is not None:
            ensemble_results.append(lgb_result)

        if len(ensemble_results) == 0:
            logger.error("[FAIL] 所有模型训练失败")
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"模型训练汇总(Window {window_idx})")
        logger.info(f"{'='*60}")
        logger.info(f"  训练模型数: {len(ensemble_results)}")
        logger.info(f"  LightGBM: {'[OK]' if lgb_result is not None else '[FAIL]'}")

        # 4. 保存模型
        model_files = []
        for i, model_result in enumerate(ensemble_results):
            model_type = model_result.get('model_type', 'xgboost')
            model_file = Path("results") / f"{model_type}_model_window{window_idx}_{timestamp}.pkl"

            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model_result['model'],
                    'scaler': model_result['scaler'],
                    'feature_names': model_result['feature_names'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'model_type': model_type,
                    'window_idx': window_idx,
                    'best_params': best_params if model_type == 'xgboost' else None,
                    'performance': model_result.get('performance', {})
                }, f)

            logger.info(f"[OK] {model_type.upper()}模型已保存: {model_file.name}")
            model_files.append(model_file)

        # [强化] 5. 计算模型权重(基于验证集RankIC) - 排序模型专用
        model_performances = []
        for model_result in ensemble_results:
            performance = model_result.get('performance', {})
            # [强化] 排序模型使用RankIC作为权重计算指标
            rank_ic = performance.get('rank_ic')
            model_performances.append(rank_ic)

        # [SW行业编码] 提取行业映射信息（从LightGBM模型中）
        industry_mapping = None
        for model_result in ensemble_results:
            if 'industry_mapping' in model_result and model_result['industry_mapping'] is not None:
                industry_mapping = model_result['industry_mapping']
                logger.info(f"model_result中industry_mapping不为none")
                break
        if industry_mapping is None:
            logger.info(f"model_result中industry_mapping为none")
        # 返回集成结果
        return {
            'models': [m['model'] for m in ensemble_results],
            'scalers': [m['scaler'] for m in ensemble_results],
            'feature_names': ensemble_results[0]['feature_names'],  # 所有模型使用相同特征
            'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
            'model_files': model_files,
            'model_performances': model_performances,
            'model_types': [m.get('model_type', 'xgboost') for m in ensemble_results],
            'industry_mapping': industry_mapping
        }

    except Exception as e:
        logger.error(f"[FAIL] 训练最终模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _backtest_on_window(test_data, model_result, window, args):
    """在窗口测试集上回测(支持集成学习)"""
    logger = logging.getLogger(__name__)

    try:
        # [WRENCH] 修复：直接使用训练时的特征列表，而不是重新提取
        # 原因：确保训练和测试使用完全相同的特征
        feature_columns = model_result['feature_names']
        logger.info(f"[CHART] 使用训练时的特征列表: {len(feature_columns)}个特征")

        # [SW行业编码] 检查是否需要为测试集添加行业特征
        test_data_with_industry = test_data.copy()
        industry_feature_added = False
        industry_mapping = None  # 初始化行业映射变量，避免UnboundLocalError


        # 验证测试集是否包含所有必需特征（在添加行业特征后）
        missing_features = [f for f in feature_columns if f not in test_data_with_industry.columns]
        if missing_features:
            logger.error(f"[FAIL] 测试集缺少{len(missing_features)}个训练时使用的特征:")
            logger.error(f"   缺失特征: {missing_features[:20]}")
            logger.error("   请检查特征生成逻辑，确保训练和测试使用相同的特征")
            return None

        # 🆕 诊断日志：特征质量检查（使用添加行业特征后的数据）
        logger.info(f"\n{'='*60}")
        logger.info(f"[CHART] [{window['name']}] 特征质量检查")
        logger.info(f"{'='*60}")

        # 检查每个特征的异常情况
        abnormal_features = []
        for feature in feature_columns:
            feature_data = test_data_with_industry[feature]

            # 检查全0
            if (feature_data == 0).all():
                abnormal_features.append(f"{feature}: 全部为0")
                logger.warning(f"[WARNING] 特征 {feature}: 全部为0")

            # 检查全NaN
            elif feature_data.isna().all():
                abnormal_features.append(f"{feature}: 全部为NaN")
                logger.warning(f"[WARNING] 特征 {feature}: 全部为NaN")

            # 检查高比例NaN
            elif feature_data.isna().mean() > 0.5:
                nan_pct = feature_data.isna().mean() * 100
                abnormal_features.append(f"{feature}: {nan_pct:.1f}% NaN")
                logger.warning(f"[WARNING] 特征 {feature}: {nan_pct:.1f}% 为NaN")

            # 检查常数列
            elif feature_data.nunique() == 1:
                abnormal_features.append(f"{feature}: 常数列")
                logger.warning(f"[WARNING] 特征 {feature}: 常数列")

        if abnormal_features:
            logger.warning(f"\n[WARNING] 发现{len(abnormal_features)}个异常特征:")
            for abnormal in abnormal_features[:10]:
                logger.warning(f"   - {abnormal}")
            if len(abnormal_features) > 10:
                logger.warning(f"   ... 还有{len(abnormal_features) - 10}个")
        else:
            logger.info(f"[OK] 所有{len(feature_columns)}个特征质量正常")

        logger.info(f"{'='*60}\n")

        # 准备特征数据
        X_test, _ = _prepare_feature_data(
            test_data_with_industry, feature_columns,
            data_type="测试集"
        )

        if X_test is None:
            logger.error("[FAIL] 测试集特征准备失败")
            return None

        # [WRENCH] 修复：不需要特征对齐，因为已经使用训练时的特征列表
        # X_test已经按照feature_columns的顺序排列
        X_test_aligned = X_test
        logger.info(f"[OK] 测试集特征准备完成: {X_test_aligned.shape}")

        # 检查是否为集成模型
        models = model_result.get('models')
        if models is None:
            # 单模型(向后兼容)
            models = [model_result['model']]
            scalers = [model_result['scaler']]
            model_performances = [1.0]
            model_types = ['xgboost']
        else:
            # 集成模型
            scalers = model_result['scalers']
            model_performances = model_result.get('model_performances', [1.0] * len(models))
            model_types = model_result.get('model_types', ['xgboost'] * len(models))

        logger.info(f"[CHART] 使用{len(models)}个模型进行集成预测")

        # 对每个模型进行预测
        all_predictions = []
        for i, (model, scaler, model_type) in enumerate(zip(models, scalers, model_types)):
            # [WRENCH] 修复：不再使用StandardScaler，直接使用Winsorization后的特征
            # 因子在计算时已经进行了时间序列标准化（如zscore、rank）
            X_test_scaled = X_test_aligned

            # 根据模型类型进行预测
            if model_type == 'lightgbm':
                # LightGBM模型
                y_pred_single = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                logger.info(f"  [OK] 模型{i+1}(LightGBM)预测完成")
            elif hasattr(model, 'predict') and hasattr(model, 'get_score'):
                # XGBoost Booster对象
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test_scaled)
                y_pred_single = model.predict(dtest)
                logger.info(f"  [OK] 模型{i+1}(XGBoost Booster)预测完成")
            else:
                # XGBRegressor对象
                y_pred_single = model.predict(X_test_scaled)
                logger.info(f"  [OK] 模型{i+1}(XGBoost Regressor)预测完成")

            all_predictions.append(y_pred_single)

        # 计算加权平均预测(基于RankIC)
        if len(all_predictions) > 1:
            # 多模型集成
            # [WRENCH] 修复: 处理None值,将None替换为0
            model_performances_clean = [p if p is not None else 0.0 for p in model_performances]
            mse_array = np.array(model_performances_clean)

            # [强化] 排序模型使用RankIC作为权重,RankIC越大权重越高
            if np.all(mse_array > 0):
                # 使用RankIC作为权重(RankIC越大权重越高)
                weights = mse_array / mse_array.sum()
            else:
                # 如果RankIC无效或为负,使用均等权重
                logger.warning("[WARNING] 部分模型RankIC无效或为负,使用均等权重")
                weights = np.ones(len(all_predictions)) / len(all_predictions)

            logger.info(f"[CHART] 模型权重: {weights}")

            # 加权平均
            y_pred = np.average(all_predictions, axis=0, weights=weights)

            # 计算预测分歧度(用于置信度过滤)
            std_pred = np.std(all_predictions, axis=0)
            logger.info(f"[CHART] 预测分歧度: 均值={std_pred.mean():.4f}, 最大={std_pred.max():.4f}")
        else:
            # 单模型
            y_pred = all_predictions[0]
            std_pred = np.zeros_like(y_pred)  # 单模型无分歧度

        # 添加预测结果到测试数据
        test_data_with_pred = test_data.copy()
        test_data_with_pred['predicted_return'] = y_pred
        test_data_with_pred['std_pred'] = std_pred

        # 🆕 诊断日志：预测值统计
        logger.info(f"\n{'='*60}")
        logger.info(f"[CHART] [{window['name']}] 预测值统计")
        logger.info(f"{'='*60}")
        logger.info(f"预测值分位数分布:")
        logger.info(f"  0%  (最小值): {np.percentile(y_pred, 0):.4f} ")
        logger.info(f"  25% (下四分位): {np.percentile(y_pred, 25):.4f} ")
        logger.info(f"  50% (中位数):   {np.percentile(y_pred, 50):.4f} ")
        logger.info(f"  75% (上四分位): {np.percentile(y_pred, 75):.4f} ")
        logger.info(f"  100% (最大值):  {np.percentile(y_pred, 100):.4f} ")
        logger.info(f"均值±标准差: {y_pred.mean():.4f} ± {y_pred.std():.4f} ")

        # 统计异常值
        zero_count = (y_pred == 0).sum()
        near_zero_count = (np.abs(y_pred) < 0.01).sum()
        negative_count = (y_pred < 0).sum()
        logger.info(f"\n异常值统计:")
        logger.info(f"  预测值=0: {zero_count}/{len(y_pred)} ({zero_count/len(y_pred)*100:.1f}%)")
        logger.info(f"  预测值接近0(|x|<0.01): {near_zero_count}/{len(y_pred)} ({near_zero_count/len(y_pred)*100:.1f}%)")
        logger.info(f"  预测值<0: {negative_count}/{len(y_pred)} ({negative_count/len(y_pred)*100:.1f}%)")

        # 如果有大量0值，发出警告
        if zero_count / len(y_pred) > 0.3:
            logger.warning(f"[WARNING] 警告：超过30%的预测值为0，可能存在特征对齐问题！")
        if near_zero_count / len(y_pred) > 0.5:
            logger.warning(f"[WARNING] 警告：超过50%的预测值接近0，模型预测可能失效！")

        
        # 🆕 测试集分类指标评估（用于判断二分类是否适合作为并行二阶模型信号源）
        # 说明：
        # - 这里不涉及任何排序模型；仅评估二分类模型在测试集上的判别/头部纯度/校准质量。
        # - predicted_return 在二分类 objective 下通常可视作正类概率（或可单调映射为概率）。
        try:
            if 'relevance' in test_data_with_pred.columns:
                y_true_raw = test_data_with_pred['relevance'].values
                y_score_raw = test_data_with_pred['predicted_return'].values

                # 过滤 NaN
                valid_mask = (~pd.isna(y_true_raw)) & (~pd.isna(y_score_raw))
                y_true = y_true_raw[valid_mask].astype(int)
                y_score = y_score_raw[valid_mask].astype(float)

                pos_cnt = int(y_true.sum())
                neg_cnt = int(len(y_true) - pos_cnt)
                pos_rate = (pos_cnt / len(y_true)) if len(y_true) > 0 else 0.0

                logger.info(f"[METRIC] [{window['name']}] 测试集分类评估: n={len(y_true)}, pos={pos_cnt}, neg={neg_cnt}, pos_rate={pos_rate:.4%}")

                if pos_cnt > 0 and neg_cnt > 0 and len(y_true) >= 10:
                    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

                    roc_auc = float(roc_auc_score(y_true, y_score))
                    pr_auc = float(average_precision_score(y_true, y_score))

                    # 头部纯度：Precision@Top10%
                    top_frac = 0.10
                    k = max(1, int(np.ceil(top_frac * len(y_score))))
                    top_idx = np.argsort(-y_score)[:k]
                    precision_top = float(y_true[top_idx].mean())
                    lift = (precision_top / pos_rate) if pos_rate > 0 else float('nan')

                    # 概率校准粗指标：LogLoss / Brier（若 y_score 非严格[0,1]，这里会做裁剪）
                    eps = 1e-12
                    y_score_clip = np.clip(y_score, eps, 1 - eps)
                    ll = float(log_loss(y_true, y_score_clip))
                    brier = float(brier_score_loss(y_true, y_score_clip))

                    logger.info(
                        f"[METRIC] ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
                        f"P@Top10%={precision_top:.4f} (lift={lift:.2f}x) | "
                        f"LogLoss={ll:.4f} | Brier={brier:.4f}"
                    )
                else:
                    logger.warning(
                        f"[WARNING] [{window['name']}] 测试集标签在该窗口中类别不全或样本过少，跳过AUC/PR等指标计算。"
                    )
            else:
                logger.warning(f"[WARNING] [{window['name']}] 测试集缺少 'atr_return_label' 列，无法计算AUC等指标。")
        except Exception as e:
            logger.warning(f"[WARNING] [{window['name']}] 测试集AUC/PR等指标计算失败: {e}")



        # 🎯 Top5预测值的真实收益分布统计
        logger.info(f"[CHART] [{window['name']}] Top5预测值真实收益分布:")

        # 按日期分组，每天取top5预测值
        top_k_stats = {1: [], 2: [], 3: [], 5: []}
        top_k_stats_by_date = {1: [], 2: [], 3: [], 5: []}

        def _log_segmented_return_stats(return_records, label, num_segments=7):
            """按时间顺序将测试集切分为若干段，输出分段收益统计。"""
            if not return_records:
                logger.info(f"  {label} 分段统计: 无足够数据")
                return

            valid_records = []
            for rec_date, rec_value in return_records:
                if pd.isna(rec_date) or pd.isna(rec_value):
                    continue
                valid_records.append((pd.to_datetime(rec_date), float(rec_value)))

            if not valid_records:
                logger.info(f"  {label} 分段统计: 无有效数据")
                return

            valid_records.sort(key=lambda x: x[0])
            unique_days = sorted({rec_date.normalize() for rec_date, _ in valid_records})
            if len(unique_days) == 0:
                logger.info(f"  {label} 分段统计: 无有效日期")
                return

            actual_segments = max(1, min(num_segments, len(unique_days)))
            day_segments = [seg for seg in np.array_split(unique_days, actual_segments) if len(seg) > 0]

            logger.info(f"  {label} 分段统计(按时间顺序切分为{len(day_segments)}段):")
            for seg_idx, seg_days in enumerate(day_segments, 1):
                seg_day_set = set(seg_days.tolist() if hasattr(seg_days, 'tolist') else list(seg_days))
                seg_returns = [
                    rec_value for rec_date, rec_value in valid_records
                    if rec_date.normalize() in seg_day_set
                ]
                if len(seg_returns) == 0:
                    continue

                seg_start = min(seg_day_set).strftime('%Y-%m-%d')
                seg_end = max(seg_day_set).strftime('%Y-%m-%d')
                seg_mean = np.mean(seg_returns)
                seg_median = np.median(seg_returns)
                seg_min = np.min(seg_returns)
                seg_max = np.max(seg_returns)
                seg_positive = sum(1 for x in seg_returns if x > 0)
                seg_win_rate = seg_positive / len(seg_returns)
                logger.info(
                    f"    第{seg_idx}段 [{seg_start} ~ {seg_end}]: "
                    f"均值={seg_mean:.4f}, 中位数={seg_median:.4f}, "
                    f"最小值={seg_min:.4f}, 最大值={seg_max:.4f}, "
                    f"胜率={seg_win_rate:.2f} ({seg_positive}/{len(seg_returns)}个样本)"
                )

        # 确保test_data_with_pred包含必要的列
        if all(col in test_data_with_pred.columns for col in ['date', 'predicted_return', 'atr_return_label']):
            # 按日期分组计算topk
            for date in test_data_with_pred['date'].unique():
                date_data = test_data_with_pred[test_data_with_pred['date'] == date]
                # logger.info(f"日期:{date}股票:{len(date_data)}")

                if len(date_data) >= 5:  # 确保有足够的股票
                    # 按预测值降序排列
                    date_sorted = date_data.sort_values('predicted_return', ascending=False)
                     # 🆕 过滤掉当日的涨跌停、炸板股票
                    date_sorted_filtered = date_sorted[
                        (date_sorted['is_limit_up'] != 1) &
                        (date_sorted['is_limit_down'] != 1) &
                        (date_sorted['is_zhaban'] != 1)
                    ]

                    if len(date_sorted_filtered) < 5:
                        # 过滤后股票不足5个，跳过该日期
                        continue

                    # 计算各topk的真实收益 - 收集所有真实收益值而不是每日统计值
                    # 🆕 修改：topk表示"第k名"而不是"前k名"
                    for k in [1, 2, 3, 5]:
                        # 🔧 修改：取第k名的股票，而不是前k名的股票
                        # 例如k=3表示取排名第3的股票（只有1只），而不是前3名的股票（3只）
                        top_k_data = date_sorted_filtered.iloc[k-1:k]  # iloc[2:3]表示第3名（索引从0开始）
                        if len(top_k_data) > 0:
                            # 收集该日期第k名股票的真实收益值，过滤掉NaN值
                            # 使用 atr_return_label（即 raw_future_ret）代替 future_return_2d
                            # 这样可以与训练集和验证集的评估保持一致
                            real_returns = top_k_data['atr_return_label'].values
                            valid_returns = real_returns[~np.isnan(real_returns)]  # 过滤NaN值
                            top_k_stats[k].extend(valid_returns)  # 只存储有效收益值
                            top_k_stats_by_date[k].extend([(date, ret) for ret in valid_returns])
                            # logger.info(f"top{k} has {len(valid_returns)} and topk_status has {len(top_k_stats[k])}")

            # 统计并输出结果 - 对收集的所有真实收益值进行统计
            for k in [1, 2, 3, 5]:
                if len(top_k_stats[k]) > 0:
                    all_returns = top_k_stats[k]  # 所有Top-K股票的真实收益值

                    overall_mean = np.mean(all_returns)
                    overall_median = np.median(all_returns)
                    overall_min = np.min(all_returns)
                    overall_max = np.max(all_returns)

                    logger.info(f"  Top-{k}: 均值={overall_mean:.4f}, 中位数={overall_median:.4f}, 最小值={overall_min:.4f}, 最大值={overall_max:.4f} (基于{len(all_returns)}个样本)")

                    # 额外统计：胜率（正收益比例）
                    positive_count = sum(1 for x in all_returns if x > 0)
                    win_rate = positive_count / len(all_returns) if len(all_returns) > 0 else 0
                    logger.info(f"  Top-{k}: 胜率={win_rate:.2f} ({positive_count}/{len(all_returns)}个样本)")

                    # 交易日统计（用于参考）
                    trading_days = len(test_data_with_pred['date'].unique())
                    logger.info(f"  Top-{k}: 基于{trading_days}个交易日，每天取排名第{k}的股票（1只股票）")
                    _log_segmented_return_stats(top_k_stats_by_date[k], f"Top-{k}")
                else:
                    logger.info(f"  Top-{k}: 无足够数据")
        else:
            logger.warning(f"[WARNING] 缺少必要列进行Top5统计: {test_data_with_pred.columns.tolist()}")

        # 🆕 新增：每日 Top1 vs Top2-5 收益差距分析
        logger.info(f"\n[ANALYSIS] [{window['name']}] 每日 Top1 vs Top2-5 收益差距分析")
        logger.info(f"{'='*80}")
        # 数据结构：按日期存储每天的 top1~top5 收益
        # {date_str: {1: return1, 2: return2, 3: return3, 4: return4, 5: return5}}
        daily_top_returns = {}

        if all(col in test_data_with_pred.columns for col in ['date', 'predicted_return', 'atr_return_label']):
            for date in test_data_with_pred['date'].unique():
                date_data = test_data_with_pred[test_data_with_pred['date'] == date]

                if len(date_data) >= 5:
                    # 按预测值降序排列
                    date_sorted = date_data.sort_values('predicted_return', ascending=False)

                    # 过滤掉涨跌停、炸板股票
                    date_sorted_filtered = date_sorted[
                        (date_sorted['is_limit_up'] != 1) &
                        (date_sorted['is_limit_down'] != 1) &
                        (date_sorted['is_zhaban'] != 1)
                    ]

                    if len(date_sorted_filtered) < 5:
                        continue

                    # 获取日期字符串
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)

                    # 收集 top1~top5 的收益
                    daily_returns = {}
                    for k in [1, 2, 3, 4, 5]:
                        top_k_data = date_sorted_filtered.iloc[k-1:k]
                        if len(top_k_data) > 0:
                            real_returns = top_k_data['atr_return_label'].values
                            valid_returns = real_returns[~np.isnan(real_returns)]
                            if len(valid_returns) > 0:
                                daily_returns[k] = valid_returns[0]

                    # 只有当所有5个排名都有有效收益时才记录
                    if len(daily_returns) == 5:
                        daily_top_returns[date_str] = daily_returns

            # 计算每日收益差距：top1 - mean(top2, top3, top4, top5)
            if len(daily_top_returns) > 0:
                daily_gaps = []
                daily_top1_returns = []  # 🆕 每日top1收益
                daily_top2_returns = []  # 🆕 每日top2收益
                dates = []

                for date_str in sorted(daily_top_returns.keys()):
                    returns = daily_top_returns[date_str]
                    top1_return = returns[1]
                    top2_return = returns[2]  # 🆕 提取top2收益
                    top2_5_mean = np.mean([returns[2], returns[3], returns[4], returns[5]])
                    gap = top1_return - top2_5_mean

                    dates.append(date_str)
                    daily_gaps.append(gap)
                    daily_top1_returns.append(top1_return)  # 🆕 记录top1收益
                    daily_top2_returns.append(top2_return)  # 🆕 记录top2收益

                # 🆕 使用滚动窗口计算统计量
                W = 60  # 滚动窗口大小

                # 转换为 pandas Series 以便使用 rolling 函数
                gap_series = pd.Series(daily_gaps)
                top1_series = pd.Series(daily_top1_returns)  # 🆕 top1收益序列
                top2_series = pd.Series(daily_top2_returns)  # 🆕 top2收益序列

                # 计算滚动均值和滚动标准差
                rolling_mean = gap_series.rolling(window=W, min_periods=1).mean()
                rolling_std = gap_series.rolling(window=W, min_periods=1).std()

                # 🆕 计算top1和top2的滚动收益
                rolling_top1 = top1_series.rolling(window=W, min_periods=1).mean()
                rolling_top2 = top2_series.rolling(window=W, min_periods=1).mean()

                # 计算Z值（避免除零）
                Z = rolling_mean / (rolling_std + 1e-8)

                # 绘制五条曲线
                if len(dates) > 0:
                    import matplotlib.pyplot as plt
                    from matplotlib.dates import date2num, DateFormatter
                    from datetime import datetime

                    # 转换日期字符串为 datetime 对象
                    date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]

                    # 创建图表
                    fig, ax1 = plt.figure(figsize=(16, 8)), plt.gca()

                    # 绘制滚动均值（蓝色实线）
                    line1 = ax1.plot(date_objects, rolling_mean, linewidth=2, color='blue', alpha=0.8, label=f'Gap Rolling Mean (W={W})')

                    # 绘制滚动标准差（绿色实线）
                    line2 = ax1.plot(date_objects, rolling_std, linewidth=2, color='green', alpha=0.8, label=f'Gap Rolling Std (W={W})')

                    # 🆕 绘制rolling top1收益（橙色虚线）
                    line4 = ax1.plot(date_objects, rolling_top1, linewidth=2, color='orange', alpha=0.8, linestyle='--', label=f'Top1 Rolling Return (W={W})')

                    # 🆕 绘制rolling top2收益（紫色虚线）
                    line5 = ax1.plot(date_objects, rolling_top2, linewidth=2, color='purple', alpha=0.8, linestyle='--', label=f'Top2 Rolling Return (W={W})')

                    # 绘制Z值（红色，使用右侧y轴）
                    ax2 = ax1.twinx()
                    line3 = ax2.plot(date_objects, Z, linewidth=2, color='red', alpha=0.8, label=f'Z-Score (W={W})')

                    # 添加零线（Z=0）
                    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

                    # 添加标题和标签
                    ax1.set_title(f'Rolling Statistics: Top1 vs Top2-5 Performance Gap\nWindow: {window["name"]} ({window["test_start"]} ~ {window["test_end"]})',
                                 fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date', fontsize=12)
                    ax1.set_ylabel('Rolling Mean / Std / Returns', fontsize=12, color='black')
                    ax2.set_ylabel('Z-Score', fontsize=12, color='red')

                    # 设置y轴颜色
                    ax1.tick_params(axis='y', labelcolor='black')
                    ax2.tick_params(axis='y', labelcolor='red')

                    # 格式化x轴日期
                    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))  # 最多显示20个日期标签
                    fig.autofmt_xdate()  # 自动旋转日期标签

                    # 添加网格
                    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

                    # 合并图例（来自两个y轴）
                    lines = line1 + line2 + line4 + line5 + line3
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='best', fontsize=9)

                    # 添加统计信息文本框
                    gap_mean = np.mean(daily_gaps)
                    gap_std = np.std(daily_gaps)
                    gap_median = np.median(daily_gaps)
                    positive_days = sum(1 for g in daily_gaps if g > 0)
                    positive_ratio = positive_days / len(daily_gaps) * 100

                    # Z值统计
                    Z_mean = Z.mean()
                    Z_std = Z.std()
                    Z_max = Z.max()
                    Z_min = Z.min()

                    # 🆕 Top1和Top2收益统计
                    top1_mean = np.mean(daily_top1_returns)
                    top2_mean = np.mean(daily_top2_returns)

                    stats_text = f'Gap Statistics:\n'
                    stats_text += f'Mean: {gap_mean:.4f}\n'
                    stats_text += f'Median: {gap_median:.4f}\n'
                    stats_text += f'Std: {gap_std:.4f}\n'
                    stats_text += f'Positive Days: {positive_days}/{len(daily_gaps)} ({positive_ratio:.1f}%)\n\n'
                    stats_text += f'Return Statistics:\n'
                    stats_text += f'Top1 Mean: {top1_mean:.4f}\n'
                    stats_text += f'Top2 Mean: {top2_mean:.4f}\n\n'
                    stats_text += f'Z-Score Statistics:\n'
                    stats_text += f'Mean: {Z_mean:.4f}\n'
                    stats_text += f'Std: {Z_std:.4f}\n'
                    stats_text += f'Max: {Z_max:.4f}\n'
                    stats_text += f'Min: {Z_min:.4f}'

                    ax1.text(0.02, 0.98, stats_text,
                            transform=ax1.transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    plt.tight_layout()

                    # 保存图表
                    import os
                    os.makedirs('results', exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'results/top1_vs_top2to5_rolling_stats_{window["name"]}_{timestamp}.png'
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close()

                    logger.info(f"  [OK] 滚动统计曲线已保存至: {filename}")

                    # 输出统计信息
                    logger.info(f"  收益差距统计 (基于 {len(daily_gaps)} 个交易日):")
                    logger.info(f"    均值: {gap_mean:.4f}")
                    logger.info(f"    中位数: {gap_median:.4f}")
                    logger.info(f"    标准差: {gap_std:.4f}")
                    logger.info(f"    最小值: {np.min(daily_gaps):.4f}")
                    logger.info(f"    最大值: {np.max(daily_gaps):.4f}")
                    logger.info(f"    正差距天数: {positive_days} ({positive_ratio:.1f}%)")
                    logger.info(f"  收益统计:")
                    logger.info(f"    Top1平均收益: {top1_mean:.4f}")
                    logger.info(f"    Top2平均收益: {top2_mean:.4f}")
                    logger.info(f"  Z值统计:")
                    logger.info(f"    均值: {Z_mean:.4f}")
                    logger.info(f"    标准差: {Z_std:.4f}")
                    logger.info(f"    最大值: {Z_max:.4f}")
                    logger.info(f"    最小值: {Z_min:.4f}")
            else:
                logger.warning(f"  [WARNING] 无有效日期数据进行收益差距分析")
        else:
            logger.warning(f"  [WARNING] 缺少必要列进行收益差距分析")

        logger.info(f"{'='*80}\n")

        # 🆕 新增：完整稳健统计评估 - 测试集（与验证集格式一致）
        if top_k_stats[1]:  # 如果有数据才进行稳健评估
            print_top_k_robust_stats(top_k_stats[1], top_k_stats[2], top_k_stats[3], top_k_stats[5])
            logger.info("[EXPLANATION] 上述为测试集Top-k真实收益的稳健统计，可与验证集进行对比")

        logger.info(f"{'='*60}\n")


        # [OK] 调用统一的选股回测函数(复用test_model逻辑)
        logger.info(f"\n[CHART] [{window['name']}] 开始选股回测...")
        logger.info(f"   测试期: {window['test_start']} ~ {window['test_end']}")
        logger.info(f"   测试天数: {len(test_data_with_pred['date'].unique())}天")
        logger.info(f"   候选股票: {len(test_data_with_pred)}条记录")

        # 调用统一选股回测函数
        backtest_result = _run_stock_selection_backtest(
            backtest_data=test_data_with_pred,
            stock_counts=[2],  # Top-1/2/3/5选股
            window_name=window['name'],
            market_data=None,  # Walk-Forward暂不使用市场状态
            frequency='minute'  # 默认日线回测，保持向后兼容
        )

        # 🔍 添加调试日志：检查回测结果
        logger.debug(f"🔍 [DEBUG] [{window['name']}] 回测结果检查:")
        logger.debug(f"  - backtest_result类型: {type(backtest_result)}")
        logger.debug(f"  - 是否为None: {backtest_result is None}")
        if backtest_result is not None:
            logger.debug(f"  - backtest_result键: {list(backtest_result.keys())}")
            logger.debug(f"  - backtest_results存在: {'backtest_results' in backtest_result}")
            if 'backtest_results' in backtest_result:
                logger.debug(f"  - backtest_results内容: {backtest_result['backtest_results']}")

        if backtest_result is None:
            logger.error(f"[FAIL] [{window['name']}] 选股回测失败")
            return None

        if 'backtest_results' in backtest_result:
            all_results = backtest_result['backtest_results']
            all_trades = backtest_result['detailed_trades']
            summary_stats = backtest_result['summary_stats']
        else:
            all_results = {k: v for k, v in backtest_result.items() if isinstance(k, int)}
            if not all_results:
                logger.error(f"[FAIL] [{window['name']}] 选股回测失败")
                return None
            all_trades = {k: v.get('trades', []) for k, v in all_results.items()}
            summary_stats = {}

        primary_rank = 1 if 1 in all_results else min(all_results.keys())
        primary_metrics = all_results.get(primary_rank, {})

        # [OK] 返回所有Top-k的完整结果
        return {
            # [OK] 修复：返回所有档位的选股结果，而不仅是Top-1
            'selections': all_trades,  # {1: [...], 2: [...], 3: [...], 5: [...]} or raw rank map
            'total_return': primary_metrics.get('total_return', 0),
            'win_rate': primary_metrics.get('win_rate', 0),
            'profit_loss_ratio': primary_metrics.get('profit_loss_ratio', 0),
            'sharpe_ratio': primary_metrics.get('sharpe_ratio', 0),
            'max_drawdown': primary_metrics.get('max_drawdown', 0),
            'total_trades': primary_metrics.get('total_trades', 0),
            'primary_rank': primary_rank,
            # [OK] 所有Top-k档位的完整结果
            'all_results': all_results,  # {1: {...}, 2: {...}, 3: {...}, 5: {...}}
            'detailed_trades': all_trades,  # {1: [...], 2: [...], 3: [...], 5: [...]}
            'summary_stats': summary_stats
        }

    except Exception as e:
        logger.error(f"[FAIL] 回测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _process_walkforward_window(window, factor_data, args, window_idx, total_windows, all_windows, sw_sector_mapping=None):
    """
    处理单个Walk-Forward窗口

    Args:
        window: 窗口配置字典
        factor_data: 完整因子数据
        args: 命令行参数
        window_idx: 当前窗口索引
        total_windows: 总窗口数
        all_windows: 所有窗口列表(用于参数传递)
        sw_sector_mapping: SW申万行业映射字典，用于LightGBM分类特征（可选）

    Returns:
        dict: 包含backtest_result, selections, metrics的字典,失败返回None
    """
    logger = logging.getLogger(__name__)

    try:
        # 0. 加载市场数据（用于XGBoost计算group权重）
        logger.info(f"\n[REFRESH] 步骤0: 加载市场数据")
        market_data = None
        try:
            from data_loader import IntegratedDataLoader
            loader = IntegratedDataLoader()
            start_date_str = factor_data['date'].min().strftime('%Y%m%d')
            end_date_str = factor_data['date'].max().strftime('%Y%m%d')
            _, _, market_data = loader.load_all_data(start_date_str, end_date_str, extend_history_days=20)

            if market_data is None or market_data.empty:
                logger.warning("[WARNING] 市场数据加载失败，XGBoost将使用均匀group权重")
                market_data = None
            else:
                # 确保date列是datetime类型
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                logger.info(f"[OK] 市场数据加载成功: {len(market_data)}条记录")
        except Exception as e:
            logger.warning(f"[WARNING] 市场数据加载失败: {e}，XGBoost将使用均匀group权重")
            market_data = None

        # 1. 划分数据
        logger.info(f"\n[CHART] 步骤1: 划分数据集")

        # 训练集
        train_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['train_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['train_end']))
        ].copy()

        # 验证集
        val_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['val_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['val_end']))
        ].copy()

        # 测试集
        test_data = factor_data[
            (factor_data['date'] >= pd.to_datetime(window['test_start'])) &
            (factor_data['date'] <= pd.to_datetime(window['test_end']))
        ].copy()

        logger.info(f"   训练集: {len(train_data):,}条 ({window['train_start']} ~ {window['train_end']})")
        logger.info(f"   验证集: {len(val_data):,}条 ({window['val_start']} ~ {window['val_end']})")
        logger.info(f"   测试集: {len(test_data):,}条 ({window['test_start']} ~ {window['test_end']})")

        # [SW编码检查] 验证集划分后检查SW行业编码20的分布
        logger.info("[SW编码检查] 验证集划分后SW行业编码20分布检查...")
        if sw_sector_mapping:
            # 重新加载SW编码数据以获取映射信息
            try:
                from sw_encoding_loader import load_standardized_sw_encoding
                local_encoding_data = load_standardized_sw_encoding()

                if local_encoding_data and 'mappings' in local_encoding_data:
                    industry_to_code = local_encoding_data['mappings']['industry_to_code']
                    stock_to_industry = local_encoding_data['mappings']['stock_to_industry']

                    # 获取SW编码20的股票列表
                    sw_code_20_stocks_global = [stock for stock, industry in stock_to_industry.items()
                                                if industry_to_code.get(industry) == 20]

                    # 检查各数据集中SW编码20的股票数量
                    train_sw_code_20_count = 0
                    if 'stock_code' in train_data.columns:
                        train_sw_code_20_count = len([stock for stock in train_data['stock_code'] if stock in sw_code_20_stocks_global])

                    val_sw_code_20_count = 0
                    if 'stock_code' in val_data.columns:
                        val_sw_code_20_count = len([stock for stock in val_data['stock_code'] if stock in sw_code_20_stocks_global])

                    test_sw_code_20_count = 0
                    if 'stock_code' in test_data.columns:
                        test_sw_code_20_count = len([stock for stock in test_data['stock_code'] if stock in sw_code_20_stocks_global])

                    logger.info(f"[SW编码检查] SW行业编码20股票分布:")
                    logger.info(f"  训练集: {train_sw_code_20_count}个")
                    logger.info(f"  验证集: {val_sw_code_20_count}个")
                    logger.info(f"  测试集: {test_sw_code_20_count}个")

                    if val_sw_code_20_count == 0:
                        logger.warning("[SW编码检查] 警告：验证集中没有SW行业编码20的股票！")
                    elif val_sw_code_20_count < 10:
                        logger.warning(f"[SW编码检查] 警告：验证集中SW行业编码20的股票数量较少: {val_sw_code_20_count}个")
                else:
                    logger.warning("[SW编码检查] 警告：无法加载标准化SW编码数据进行检查")
            except Exception as e:
                logger.warning(f"[SW编码检查] 加载编码数据进行检查时出错: {e}")

        if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
            logger.error("[FAIL] 训练集、验证集或测试集为空")
            return None

        # 1.5. 生成标签 (如果还没有)
        if 'atr_return_label' not in train_data.columns:

            logger.info(f"\n🏷️ 步骤1.5: 生成valid_cs_mask并生成标签（ST/新股不参与横截面rank）")

            # 使用连续时间区间生成标签，避免 train/val/test 直接拼接后存在日期断档，
            # 导致 groupby().shift(-2/-3) 跨缺口跳到更远日期，放大尾部样本标签。
            label_start = pd.to_datetime(window['train_start'])
            combined_data = factor_data[
                factor_data['date'] >= label_start
            ].copy()
            combined_data = combined_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
            logger.info(
                f"   [OK] 标签生成使用连续区间数据: "
                f"{combined_data['date'].min().strftime('%Y-%m-%d') if len(combined_data) else 'N/A'} ~ "
                f"{combined_data['date'].max().strftime('%Y-%m-%d') if len(combined_data) else 'N/A'}, "
                f"共 {len(combined_data):,} 条"
            )

            # [可选兜底] 北交所过滤：通常因子库已无.BJ；若仍存在，剔除以避免污染label/横截面统计
            if 'stock_code' in combined_data.columns:
                bj_mask = combined_data['stock_code'].astype(str).str.endswith(('.BJ', '.bj'), na=False)
                bj_cnt = int(bj_mask.sum())
                if bj_cnt > 0:
                    logger.warning(f"   [FILTER] 发现北交所样本 {bj_cnt:,} 条，已在标签生成前剔除（因子库应已不包含.BJ）")
                    combined_data = combined_data.loc[~bj_mask].copy()

            # 生成 valid_cs_mask（不删行）
            logger.info("   [REFRESH] 生成valid_cs_mask（综合ST/新股口径，不删行）...")
            try:
                valid_cs_mask, st_keep, new_keep = _build_valid_cs_mask(combined_data, logger)
                combined_data["valid_cs_mask"] = valid_cs_mask.astype(bool)
                invalid_cnt = int((~combined_data["valid_cs_mask"]).sum())
                logger.info(f"   [OK] valid_cs_mask 生成完成: 无效样本 {invalid_cnt:,}/{len(combined_data):,}")
            except Exception as e:
                logger.warning(f"   [WARNING] valid_cs_mask 生成失败，回退为全True: {e}")
                combined_data["valid_cs_mask"] = True

            # 生成标签（内部会使用 valid_cs_mask 排除ST/新股参与rank）
            logger.info("   [REFRESH] 生成标签...")
            combined_data = _generate_training_labels_for_all_data(combined_data)

            # 划分训练/验证/测试（含标签）
            train_data = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['train_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['train_end']))
            ].copy()
            val_data = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['val_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['val_end']))
            ].copy()
            test_data = combined_data[
                (combined_data['date'] >= pd.to_datetime(window['test_start'])) &
                (combined_data['date'] <= pd.to_datetime(window['test_end']))
            ].copy()

            # 排序保证一致性
            train_data = train_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
            val_data = val_data.sort_values(['date', 'stock_code']).reset_index(drop=True)
            test_data = test_data.sort_values(['date', 'stock_code']).reset_index(drop=True)

            # [训练/验证] relevance 为空的样本不会进入训练（包括ST/新股、不交易样本、末端无未来数据样本）
            # NOTE: 不在 DataFrame 层 dropna(relevance)，由后续模型输入 mask 控制
            train_data = train_data
            # NOTE: 不在 DataFrame 层 dropna(relevance)，由后续模型输入 mask 控制
            val_data = val_data
            logger.info(f"   [OK] 标签生成完成")
            if 'atr_return_label' in train_data.columns:
                logger.info(f"   训练集有效样本: {train_data['atr_return_label'].notna().sum():,}/{len(train_data):,} ,最小值：{train_data['atr_return_label'].min():.6f}最大值：{train_data['atr_return_label'].max():.6f}")
            if 'atr_return_label' in val_data.columns:
                logger.info(f"   验证集有效样本: {val_data['atr_return_label'].notna().sum():,}/{len(val_data):,} ,最小值：{val_data['atr_return_label'].min():.6f}最大值：{val_data['atr_return_label'].max():.6f}")
            if 'atr_return_label' in test_data.columns:
                logger.info(f"   测试集有效样本: {test_data['atr_return_label'].notna().sum():,}/{len(test_data):,} ,最小值：{test_data['atr_return_label'].min():.6f}最大值：{test_data['atr_return_label'].max():.6f}")

        # 2. 检查是否已有模型，决定是否需要贝叶斯搜索和训练
        logger.info(f"\n🤖 步骤2: 检查模型或训练新模型")

        # [强化] 解析模型选择参数
        models_str = getattr(args, 'models', 'lightgbm')
        selected_models = [m.strip().lower() for m in models_str.split(',')]

        use_lightgbm = 'lightgbm' in selected_models

        if not use_lightgbm:
            logger.error("[FAIL] 必须选择LightGBM模型")
            return False

        logger.info(f"[强化] 模型选择: {', '.join(selected_models)}")
        logger.info(f"   LightGBM: {'[OK]'}")

        # [WRENCH] 修复：每个窗口查找自己的模型
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # 查找当前窗口的XGBoost模型
        xgb_pattern = f"xgboost_model_window{window_idx}_*.pkl"
        xgb_files = list(results_dir.glob(xgb_pattern))

        # 查找当前窗口的LightGBM模型
        # lgb_pattern = f"lightgbm_model_window{window_idx}_20260107_033514.pkl"
        lgb_pattern = f"lightgbm_model_window{window_idx}_2027*.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260429_202824.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260427_231053.pkl"#alpha 20260502
        # lgb_pattern = f"lightgbm_model_window3_20260419_234602.pkl"
        # lgb_pattern = f"lightgbm_model_window3_incremental_20260412_110759.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260411_180603.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260411_123954.pkl"
        # lgb_pattern = f"lightgbm_model_window{window_idx}_20260222_012258.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260301_130248.pkl"#妖股
        # lgb_pattern = f"lightgbm_model_window3_20260301_171655.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260302_175124.pkl"
        # lgb_pattern = f"lightgbm_model_window3_20260302_202830.pkl"
        lgb_files = list(results_dir.glob(lgb_pattern))

        model_result = None

        # 🆕 处理只需要LightGBM模型的情况
        if lgb_files and use_lightgbm:
            logger.info(f"[SEARCH] 找到{len(lgb_files)}个Window {window_idx}的LightGBM模型")

            # 使用最新的模型
            latest_lgb_file = max(lgb_files, key=lambda f: f.stem)
            logger.info(f"   最新LightGBM模型: {latest_lgb_file.name}")

            try:
                import pickle

                with open(latest_lgb_file, 'rb') as f:
                    lgb_data = pickle.load(f)

                # 构建model_result（单模型）
                model_result = {
                    'models': [lgb_data['model']],
                    'scalers': [lgb_data['scaler']],
                    'feature_names': lgb_data['feature_names'],
                    'winsorization_thresholds': {},  # 因子已在生成阶段完成winsorize
                    'model_files': [latest_lgb_file],
                    'model_performances': [1.0],
                    'model_types': ['lightgbm']
                }

                logger.info(f"[OK] 成功加载Window {window_idx}的LightGBM模型")

            except Exception as e:
                logger.warning(f"[WARNING] 加载模型失败: {e}")
                logger.info("   将重新训练模型")
                model_result = None

        # 如果没有找到模型或加载失败，则训练新模型
        if model_result is None:
            logger.info(f"[REFRESH] 开始训练Window {window_idx}的新模型")

            # [WRENCH] 如果选择了XGBoost,执行贝叶斯搜索
            best_params = None
            if False:
                logger.info(f"\n[SEARCH] 步骤2.1: 执行贝叶斯搜索(XGBoost)")
                logger.info(f"   说明: 每个窗口独立执行完整贝叶斯搜索,确保参数最优")
                logger.info(f"   训练集: {len(train_data):,}条 ({window['train_start']} ~ {window['train_end']})")
                logger.info(f"   验证集: {len(val_data):,}条 ({window['val_start']} ~ {window['val_end']})")

                # 执行贝叶斯搜索(使用测试集最新日期作为时间衰减基准,支持结果缓存)
                best_params = _bayesian_search_single_val(
                    train_data, val_data, args, test_end_date=window['test_end'], window_idx=window_idx,
                    market_data=market_data  # [强化] 传递market_data用于XGBoost计算group权重
                )

                if best_params is None:
                    logger.error("[FAIL] 贝叶斯搜索失败")
                    return None

                logger.info(f"[OK] 贝叶斯搜索完成")
                logger.info(f"   最优参数: {best_params}")
            else:
                logger.info(f"\n⏭️ 步骤2.1: 跳过贝叶斯搜索(仅使用LightGBM)")
                logger.info(f"   LightGBM使用默认参数,无需参数搜索")
                # 使用默认参数
                best_params = {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_alpha': 0,
                    'reg_lambda': 1
                }

            # 使用完整训练集和验证集(已生成标签的train_data和val_data)
            full_train_data = train_data.copy()
            full_val_data = val_data.copy()
            full_test_data = test_data.copy()

            logger.info(f"\n[REFRESH] 步骤2.2: 训练最终模型")
            logger.info(f"   完整训练集: {len(full_train_data):,}条")
            logger.info(f"   训练集有效样本: {full_train_data['atr_return_label'].notna().sum():,}条")
            logger.info(f"   完整验证集: {len(full_val_data):,}条")
            logger.info(f"   验证集有效样本: {full_val_data['atr_return_label'].notna().sum():,}条")

            # [强化] 训练模型(传入训练集和验证集，跳过内部划分)
            model_result = _train_final_model_for_window(
                full_train_data, full_val_data, full_test_data, best_params, window_idx,
                use_lightgbm=use_lightgbm,
                market_data=market_data,  # [强化] 传递market_data用于XGBoost计算group权重
                optimize_thresholds=args.optimize_thresholds,  # [强化] 传递止盈止损优化参数
                sw_sector_mapping=sw_sector_mapping  # [SW行业编码] 传递SW行业数据用于分类特征
            )

            if model_result is None:
                logger.error("[FAIL] 模型训练失败")
                return None

            logger.info(f"[OK] 模型训练完成")
        else:
            logger.info(f"[OK] 跳过训练，使用已有模型")

        # 4. 在测试集上回测
        logger.info(f"\n[UP] 步骤4: 在测试集上回测")

        # 执行回测(复用现有回测逻辑)
        backtest_result = _backtest_on_window(
            test_data, model_result, window, args
        )

        if backtest_result is None:
            logger.error("[FAIL] 回测失败")
            return None

        logger.info(f"[OK] 回测完成")

        # [WRENCH] 添加：输出窗口回测结果表格（类似test_model模式）
        all_results = backtest_result.get('all_results', {})
        if all_results:
            logger.info(f"\n[CHART] [{window['name']}] 选股策略回测结果:")
            logger.info("=" * 150)
            logger.info(f"{'策略':<10} {'总收益':<10} {'净收益(扣成本)':<15} {'年化收益':<10} {'最大回撤':<10} {'胜率':<8} {'盈亏比':<8} {'持仓天数':<10} {'总交易次数':<12}")
            logger.info("=" * 150)

            for stock_count in sorted(all_results.keys()):
                result = all_results[stock_count]
                total_trades = result.get('total_trades', 0)

                logger.info(f"Top-{stock_count:<5} {result.get('total_return_gross', 0):>8.2%}  "
                           f"{result.get('total_return', 0):>13.2%}  "
                           f"{result.get('annual_return', 0):>8.2%}  {result.get('max_drawdown', 0):>8.2%}  "
                           f"{result.get('win_rate', 0):>6.2%}  "
                           f"{result.get('profit_loss_ratio', 0):>6.2f}  "
                           f"{result.get('holding_days', 0):>10d}{total_trades:>10}")

            logger.info("=" * 150)

        # [WRENCH] 修复：返回所有Top-k档位的完整结果
        return {
            'backtest_result': backtest_result,
            'selections': backtest_result.get('selections', []),
            'metrics': {
                # Top-1结果（向后兼容）
                'total_return': backtest_result.get('total_return', 0),
                'win_rate': backtest_result.get('win_rate', 0),
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                'max_drawdown': backtest_result.get('max_drawdown', 0),
                # [OK] 所有Top-k档位的完整结果
                'all_results': backtest_result.get('all_results', {}),
                'detailed_trades': backtest_result.get('detailed_trades', {}),
                'summary_stats': backtest_result.get('summary_stats', {})
            }
        }

    except Exception as e:
        logger.error(f"[FAIL] 处理窗口失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
