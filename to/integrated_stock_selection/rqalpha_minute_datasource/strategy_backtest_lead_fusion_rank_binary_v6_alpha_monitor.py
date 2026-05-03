#!/usr/bin/env python3
"""
优化版分钟级回测系统 - 基于策略的架构
解决之前犯过的错误：
1. 时间循环vs TopK循环的架构问题
2. 状态管理混乱问题
3. 数据重复加载问题
4. 变量作用域错误问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import logging
import os
import pyarrow.parquet as pq
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """策略类型枚举"""
    TOP_1 = 1
    TOP_2 = 2
    TOP_3 = 3
    TOP_5 = 5


class RiskControlStrategy:
    """风控策略类型"""
    CONSECUTIVE_BEARISH = "consecutive_bearish"  # 连续阴线策略
    DAILY_MA5_BREAK = "daily_ma5_break"  # 日线MA5跌破策略
    T1_LOW_BREAK_WITH_HIGH_DRAWDOWN = "t1_low_break_with_high_drawdown"  # 🆕 T+1最低价跌破+高位回撤策略


class RiskControlConfig:
    """风控配置"""
    # 风控策略选择（默认使用日线MA5跌破策略）
    STRATEGY = RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN

    # 日线MA5跌破策略参数
    MA5_CONSECUTIVE_BREAKS = 2  # 连续跌破MA5的次数（连续N个5分钟收盘价低于MA5）

    # 连续阴线策略参数（保持原有逻辑）
    CONSECUTIVE_BEARISH_COUNT = 3  # 连续阴线数量

    # 🆕 T+1最低价跌破策略参数
    T1_LOW_CONSECUTIVE_BREAKS = 2  # 连续跌破T+1日最低价的次数（亏损状态下使用）


class EntryRiskControlStrategy:
    """开盘风控策略类型"""
    GAP_BASED = "gap_based"  # 基于涨跌幅的风控（原有逻辑）
    MA5_BASED = "ma5_based"  # 基于MA5的风控（新增逻辑）


class EntryRiskControlConfig:
    """开盘风控配置"""
    # 开盘风控策略选择（默认使用MA5风控）
    STRATEGY = EntryRiskControlStrategy.GAP_BASED

    # MA5风控参数
    MA5_MAX_UP_THRESHOLD = 0.05  # 高开超MA5 3%禁止买入
    MA5_MAX_DOWN_THRESHOLD = -0.03  # 低开超MA5 3%禁止买入

    # 涨跌幅风控参数（保持原有逻辑）
    GAP_MAX_DOWN_THRESHOLD = -0.03  # 跌空超过3%禁止买入
    GAP_MAX_UP_THRESHOLD = 0.05  # 高开超过5%禁止买入

@dataclass
class StrategyConfig:
    """策略配置类"""
    strategy_type: StrategyType
    stock_count: int
    holding_days: int = 3
    initial_capital: float = 1000000.0
    max_position_ratio: float = 0.25  # 单只股票最大仓位比例（用于多股票策略的资金分配）

    # 🆕 二阶段结构：Stage-1 分类门控TopN（0=禁用）
    cls_gate_topn: int = 0

    # 🆕 Regime / Alpha 监控配置（基于“每日成熟诊断”而非成交笔数）
    regime_enabled: bool = True
    regime_window_trades: int = 10          # 短窗：candidate / rerank alpha
    regime_long_window: int = 20            # 长窗：score margin / decay 诊断
    regime_good_topn: int = 5               # 已由离线分析确定：Top5 -> rerank top1 最优
    # 基于 20250605~20260112 开发段的分位数阈值（由 JSON 统计得到）
    regime_candidate_bad_threshold: float = -0.008781929721793272
    regime_rerank_bad_threshold: float = -0.005819903394815669
    regime_score_margin_bad_threshold: float = 0.012012661325866935
    regime_candidate_good_threshold: float = 0.016996457032999
    regime_rerank_good_threshold: float = 0.00949520104495771
    # 是否真的在 BAD_* 状态停机。默认 False：先做监控不做交易层硬切换，避免再次过度过滤。
    regime_enforce_stop: bool = False
    regime_retrain_persistence: int = 20
    regime_bad_action: str = "skip"

    # 持仓卖出止损阈值（用于T+2、T+3及以后的持仓风控）
    # 说明：
    #   - T+1日为买入日，不存在卖出风控
    #   - 从T+2日开始，如果累计收益低于此阈值，触发止损卖出
    #   - 示例：
    #       - stop_loss_threshold = -0.05  : 跌幅超过5%才卖出（原有逻辑，默认值）
    #       - stop_loss_threshold = 0.00   : 收益为负（0%）就卖出（激进风控）
    #       - stop_loss_threshold = 0.01   : 收益低于1%就卖出（要求正收益）
    #       - stop_loss_threshold = -0.10  : 跌幅超过10%才卖出（宽松风控）
    stop_loss_threshold: float = -0.08  # 默认-5%，保持向后兼容

    # 🆕 风控策略配置
    risk_control_strategy: str = RiskControlConfig.STRATEGY  # 风控策略选择
    risk_control_ma5_consecutive_breaks: int = RiskControlConfig.MA5_CONSECUTIVE_BREAKS  # MA5连续跌破次数
    risk_control_t1_low_consecutive_breaks: int = RiskControlConfig.T1_LOW_CONSECUTIVE_BREAKS  # 🆕 T+1最低价连续跌破次数

    # 🆕 开盘风控策略配置
    entry_risk_control_strategy: str = EntryRiskControlConfig.STRATEGY  # 开盘风控策略选择
    entry_risk_ma5_up_threshold: float = EntryRiskControlConfig.MA5_MAX_UP_THRESHOLD  # MA5高开阈值
    entry_risk_ma5_down_threshold: float = EntryRiskControlConfig.MA5_MAX_DOWN_THRESHOLD  # MA5低开阈值
    entry_risk_up_threshold: float = EntryRiskControlConfig.GAP_MAX_UP_THRESHOLD  # MA5高开阈值
    entry_risk_down_threshold: float = EntryRiskControlConfig.GAP_MAX_DOWN_THRESHOLD  # MA5低开阈值

    # 🆕 高位回撤风控参数（用于MA5跌破策略）
    high_return_threshold: float = 0.04  # 最高浮盈阈值（4%），触发后开始监控回撤
    high_drawdown_threshold: float = -0.03  # 从最高点回撤阈值（-3%），超过则卖出

@dataclass
class Position:
    """持仓记录类 - 解决之前字典结构不一致的问题"""
    stock_code: str
    buy_date: datetime.date
    buy_datetime: datetime
    buy_price: float
    quantity: int
    predicted_return: float
    actual_return: float = 0.0
    sell_price: float = 0.0
    sell_datetime: Optional[datetime] = None
    holding_end_date: Optional[datetime.date] = None
    entry_reason: str = "normal"
    exit_reason: str = ""

    # 🆕 跟踪买入后的最高价和最高浮盈（用于高位回撤风控）
    highest_price_since_buy: float = 0.0  # 买入后的最高价格
    highest_return_since_buy: float = -999.0  # 买入后的最高浮盈（相对买入价）
    t1_day_lowest_price: float = 0.0  # 🆕 T+1日（买入日）的最低价（从日线数据获取）

    # 🆕 Regime/decision metadata（在买入时记录，卖出时转化为已完成诊断）
    prediction_date: Optional[datetime.date] = None
    regime_state: str = "UNKNOWN"
    regime_selector: str = "legacy_rank"
    regime_base_conditional_gap: Optional[float] = None
    regime_base_score_margin: Optional[float] = None
    regime_rolling_return: Optional[float] = None
    regime_rolling_gap: Optional[float] = None
    regime_rolling_score_margin: Optional[float] = None

class BacktestStrategy:
    """单个策略的回测类 - 完全独立，避免状态干扰"""

    def __init__(self, config: StrategyConfig, prediction_data: pd.DataFrame, minute_data_path: str):
        self.config = config
        # lead_z 历史（用于鲁棒 z-score），只使用过去数据，避免 look-ahead
        self._lead_history: list[float] = []
        self.minute_data_path = minute_data_path
        self.cls_gate_topn = int(getattr(config, 'cls_gate_topn', 0) or 0)

        # 标准化预测数据的日期格式 - 统一为datetime.date
        prediction_data = prediction_data.copy()
        if 'date' in prediction_data.columns:
            # 将pandas Timestamp转换为datetime.date
            prediction_data['date'] = pd.to_datetime(prediction_data['date']).dt.date
            logger.info(f"📅 标准化日期格式: pandas Timestamp -> datetime.date")

        self.prediction_data = prediction_data

        # 策略状态 - 独立管理
        self.positions: List[Position] = []
        self.cash: float = config.initial_capital
        self.current_holdings: Dict[str, Position] = {}

        # 添加买入周期状态跟踪
        self.last_buy_date: Optional[datetime.date] = None
        self.current_buy_cycle_date: Optional[datetime.date] = None
        self.last_position_end_date: Optional[datetime.date] = None  # 上一轮持仓到期日（下一轮的T日）
        self.is_holding: bool = False
        self.current_holding_end_date: Optional[datetime.date] = None

        # 正确的T+1逻辑：当前轮次的T日状态
        self.current_t_date: Optional[datetime.date] = None  # 当前轮次的T日

        # 回测结果
        self.trades: List[Dict] = []
        self.daily_portfolio_values: List[float] = [config.initial_capital]
        self.daily_returns: List[float] = [0.0]
        self.trading_dates_list: List[datetime.date] = []  # 保存交易日期列表，用于绘图
        self.skipped_buy_attempts: List[Dict] = []  # 保存被跳过的买入尝试（开盘风控等），用于绘图

        # 仓位控制状态机（NORMAL: 正常模式 limit=0.0, DEFENSIVE: 防御模式 limit=0.01）
        self.position_control_state = "NORMAL"  # 初始状态为正常模式

        # 数据缓存
        self.minute_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.required_stocks_by_date: Dict[str, List[str]] = {}  # 按日期保存每日Top-K股票，按预测值排序
        self.filtered_candidates_by_date: Dict[str, pd.DataFrame] = {}  # 按日期保存通过基础过滤的候选全集（供动态状态机使用）

        # 🆕 日度Regime诊断（按每个交易日/预测日预计算，不依赖实际是否成交，避免持仓期缺失日期）
        self.daily_regime_diagnostics: List[Dict[str, Any]] = []
        self.regime_decision_log: List[Dict[str, Any]] = []
        self.latest_regime_state: Dict[str, Any] = {}
        self.trade_history = self.trades

        # 预测日索引（用于 maturity_lag，避免未来函数）
        self.sorted_prediction_dates: List[datetime.date] = sorted(prediction_data['date'].dropna().unique().tolist()) if 'date' in prediction_data.columns else []
        self.prediction_date_to_idx: Dict[datetime.date, int] = {d: i for i, d in enumerate(self.sorted_prediction_dates)}

        # 🆕 加载申万行业映射（用于行业跌停风控）
        self.stock_to_sw_industry = {}
        try:
            import sys
            import os
            # 添加父目录到路径以便导入sw_encoding_loader
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from sw_encoding_loader import load_sw_sector_mapping
            self.stock_to_sw_industry = load_sw_sector_mapping()
            logger.info(f"✅ 成功加载申万行业映射: {len(self.stock_to_sw_industry)}只股票")
        except Exception as e:
            logger.warning(f"⚠️ 无法加载申万行业映射: {e}，将跳过行业跌停风控")
            self.stock_to_sw_industry = {}

        logger.info(f"🚀 初始化策略: Top-{config.stock_count}, 初始资金: {config.initial_capital:,.0f}")


    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        """稳定logit，避免p=0/1导致无穷大。"""
        try:
            x = float(p)
        except Exception:
            return 0.0
        if np.isnan(x):
            return 0.0
        x = min(max(x, eps), 1.0 - eps)
        return float(np.log(x / (1.0 - x)))



    def _select_target_stock_for_top1_fusion(self, selected_stocks: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
        """
        Top-1策略：在rank Top1/Top2之间，用融合分数二选一。
        - 需要列：predicted_return, p_cls, stock_code
        - 若缺少p_cls或候选不足2只，则退回rank Top1。
        """
        if selected_stocks is None or len(selected_stocks) == 0:
            return None
        if len(selected_stocks) < 2:
            return selected_stocks.iloc[0]
        if 'p_cls' not in selected_stocks.columns:
            return selected_stocks.iloc[0]

        r1 = selected_stocks.iloc[0]
        r2 = selected_stocks.iloc[1]
        s1 = float(r1.get('predicted_return', 0.0))
        s2 = float(r2.get('predicted_return', 0.0))
        p1 = r1.get('p_cls', np.nan)
        p2 = r2.get('p_cls', np.nan)
        z1 = self._safe_logit(p1)
        z2 = self._safe_logit(p2)
        score1 = s1 + float(alpha) * z1
        score2 = s2 + float(alpha) * z2

        logger.info(f"🧠 [FUSION] Top1/Top2融合打分: "
                    f"Top1({r1.get('stock_code')}): s={s1:.6f}, p={float(p1) if p1==p1 else np.nan:.4f}, logit={z1:.4f}, score={score1:.6f} | "
                    f"Top2({r2.get('stock_code')}): s={s2:.6f}, p={float(p2) if p2==p2 else np.nan:.4f}, logit={z2:.4f}, score={score2:.6f}")

        if score2 > score1:
            logger.info(f"✅ [FUSION] 选择Top2替代Top1: {r2.get('stock_code')} (score2>score1)")
            return r2
        else:
            logger.info(f"✅ [FUSION] 保持Top1: {r1.get('stock_code')} (score1>=score2)")
            return r1


    def _safe_float(self, x: Any) -> Optional[float]:
        try:
            v = float(x)
            if np.isnan(v):
                return None
            return v
        except Exception:
            return None

    def _label_return_col(self) -> Optional[str]:
        for c in ["atr_return_label", "real_return", "realistic_vwap_return", "future_return_2d"]:
            if c in self.prediction_data.columns:
                return c
        return None

    def _compute_base_candidate_metrics(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """
        基于“每个交易日的完整过滤候选集”计算基础诊断：
        - cls_top1_return: 全候选按 p_cls 选 Top1 的标签收益
        - good_return: 全候选按 p_cls 取 Top5，再按 rank_score 取 Top1 的标签收益
        - base_conditional_gap: good_return - cls_top1_return
        - base_score_margin: Top5 内 rank Top1 vs Top2 的分数差
        这些值按交易日预计算，并通过 maturity lag 纳入状态机历史。
        """
        out = {
            'cls_top1_return': None,
            'good_return': None,
            'base_conditional_gap': None,
            'base_score_margin': None,
            'rank_top1_code': None,
            'cls_top1_code': None,
            'top5_rank1_code': None,
        }
        if filtered_df is None or len(filtered_df) == 0:
            return out

        df = filtered_df.copy()
        label_col = self._label_return_col()

        rank_sorted_all = df.sort_values('predicted_return', ascending=False).reset_index(drop=True)
        out['rank_top1_code'] = str(rank_sorted_all.iloc[0]['stock_code']) if len(rank_sorted_all) >= 1 else None

        if 'p_cls' not in df.columns:
            if label_col and len(rank_sorted_all) >= 1:
                r1 = self._safe_float(rank_sorted_all.iloc[0].get(label_col))
                out['cls_top1_return'] = r1
                out['good_return'] = r1
            return out

        cls_sorted = df.sort_values('p_cls', ascending=False).reset_index(drop=True)
        cls_top1 = cls_sorted.iloc[0]
        out['cls_top1_code'] = str(cls_top1['stock_code']) if len(cls_sorted) >= 1 else None
        out['cls_top1_return'] = self._safe_float(cls_top1.get(label_col)) if label_col else None

        good_n = min(int(getattr(self.config, 'regime_good_topn', 5) or 5), len(cls_sorted))
        top5 = cls_sorted.head(good_n).copy()
        top5_rank_sorted = top5.sort_values('predicted_return', ascending=False).reset_index(drop=True)
        if len(top5_rank_sorted) >= 1:
            out['top5_rank1_code'] = str(top5_rank_sorted.iloc[0]['stock_code'])
            out['good_return'] = self._safe_float(top5_rank_sorted.iloc[0].get(label_col)) if label_col else None
        if len(top5_rank_sorted) >= 2:
            s1 = self._safe_float(top5_rank_sorted.iloc[0].get('predicted_return'))
            s2 = self._safe_float(top5_rank_sorted.iloc[1].get('predicted_return'))
            if s1 is not None and s2 is not None:
                out['base_score_margin'] = float(s1 - s2)

        # 候选域整体 alpha（分类 TopN 均值）与 rerank alpha（Top1 相对候选均值的增量）
        if label_col and len(top5_rank_sorted) >= 1:
            cand_vals = [self._safe_float(x) for x in top5.get(label_col).tolist()] if len(top5) > 0 else []
            cand_vals = [v for v in cand_vals if v is not None]
            if cand_vals:
                out['candidate_alpha'] = float(np.mean(cand_vals))
                out['candidate_topk_mean_return'] = out['candidate_alpha']
            if out.get('good_return') is not None and out.get('candidate_alpha') is not None:
                out['rerank_alpha'] = float(out['good_return'] - out['candidate_alpha'])

        if out['good_return'] is not None and out['cls_top1_return'] is not None:
            out['base_conditional_gap'] = float(out['good_return'] - out['cls_top1_return'])
        return out

    def _compute_regime_state(self, prediction_date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        基于“每日成熟诊断”的三层 Alpha 监控：
        1) candidate_alpha：分类 Top5 候选域均值
        2) rerank_alpha：Top5 内 rerank 增量
        3) score_margin：Top5 内排序分离度

        只使用 maturity_lag 个交易日前已经成熟的日度诊断，避免未来函数；
        同时不会因为持仓期没有成交而丢失交易日。
        """
        history = self.daily_regime_diagnostics or []
        maturity_lag = int(getattr(self.config, 'holding_days', 3) or 3)

        if prediction_date is not None and prediction_date in self.prediction_date_to_idx:
            cur_idx = self.prediction_date_to_idx[prediction_date]
            history = [x for x in history if x.get('prediction_idx') is not None and x.get('prediction_idx') <= cur_idx - maturity_lag]

        short_n = int(getattr(self.config, 'regime_window_trades', 10) or 10)
        long_n = int(getattr(self.config, 'regime_long_window', 20) or 20)

        recent_short = history[-short_n:] if len(history) >= 1 else []
        recent_long = history[-long_n:] if len(history) >= 1 else []

        def _avg(items, key):
            vals = [self._safe_float(x.get(key)) for x in items]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        # 慢变量（rolling）
        roll_candidate = _avg(recent_short, 'candidate_alpha')
        roll_rerank = _avg(recent_short, 'rerank_alpha')
        roll_margin = _avg(recent_long, 'score_margin')

        # 快变量（最近一个成熟交易日）
        latest = history[-1] if len(history) >= 1 else {}
        latest_candidate = self._safe_float(latest.get('candidate_alpha'))
        latest_rerank = self._safe_float(latest.get('rerank_alpha'))
        latest_margin = self._safe_float(latest.get('score_margin'))

        cand_bad_th = float(getattr(self.config, 'regime_candidate_bad_threshold'))
        rerank_bad_th = float(getattr(self.config, 'regime_rerank_bad_threshold'))
        margin_bad_th = float(getattr(self.config, 'regime_score_margin_bad_threshold'))
        cand_good_th = float(getattr(self.config, 'regime_candidate_good_threshold'))
        rerank_good_th = float(getattr(self.config, 'regime_rerank_good_threshold'))

        bad_market = False
        model_decay = False
        weak_alpha = False
        fast_recovery = False

        state = 'WARMUP'
        if len(recent_short) >= short_n and len(recent_long) >= min(long_n, max(5, len(recent_long))):
            # 市场坏：候选域显著转弱，rerank 没有明显增量，但分离度仍正常 -> 更像市场不给钱
            bad_market = (roll_candidate is not None and roll_candidate <= cand_bad_th) and                          (roll_rerank is not None and roll_rerank <= 0.0) and                          (roll_margin is not None and roll_margin > margin_bad_th)

            # 模型坏：rerank alpha 明显转负，且 score margin 也掉到低分位
            model_decay = (roll_rerank is not None and roll_rerank <= rerank_bad_th) and                           (roll_margin is not None and roll_margin <= margin_bad_th)

            # alpha 变弱但未到坏状态
            weak_alpha = not bad_market and not model_decay and (
                (roll_candidate is not None and roll_candidate < cand_good_th) or
                (roll_rerank is not None and roll_rerank < rerank_good_th)
            )

            # 快速恢复触发器：最近一个成熟日重新转正，避免 rolling 滞后导致恢复过慢
            fast_recovery = (
                latest_candidate is not None and latest_candidate > 0 and
                latest_rerank is not None and latest_rerank > 0 and
                latest_margin is not None and latest_margin > margin_bad_th
            )

            if bad_market and model_decay:
                state = 'BAD_MARKET+MODEL_DECAY'
            elif model_decay:
                state = 'MODEL_DECAY'
            elif bad_market:
                state = 'BAD_MARKET'
            elif weak_alpha:
                state = 'WEAK_ALPHA'
            else:
                state = 'GOOD'

            if fast_recovery and state in ('BAD_MARKET', 'MODEL_DECAY', 'BAD_MARKET+MODEL_DECAY'):
                state = 'WEAK_ALPHA_RECOVERY'

        retrain_flag = False
        if len(recent_long) >= int(getattr(self.config, 'regime_retrain_persistence', 20) or 20):
            retrain_flag = bool(model_decay)

        return {
            'state': state,
            'history_count': len(history),
            'window_n': short_n,
            'long_window_n': long_n,
            'maturity_lag_days': maturity_lag,
            # 与旧字段兼容
            'rolling_return': roll_candidate,
            'rolling_gap': roll_rerank,
            'rolling_score_margin': roll_margin,
            # 新字段（语义更清楚）
            'rolling_candidate_alpha': roll_candidate,
            'rolling_rerank_alpha': roll_rerank,
            'rolling_score_margin_long': roll_margin,
            'latest_candidate_alpha': latest_candidate,
            'latest_rerank_alpha': latest_rerank,
            'latest_score_margin': latest_margin,
            'candidate_bad_threshold': cand_bad_th,
            'rerank_bad_threshold': rerank_bad_th,
            'score_margin_bad_threshold': margin_bad_th,
            'candidate_good_threshold': cand_good_th,
            'rerank_good_threshold': rerank_good_th,
            'bad_env': bool(bad_market),
            'model_decay': bool(model_decay),
            'weak_alpha': bool(weak_alpha),
            'fast_recovery': bool(fast_recovery),
            'enforce_stop': bool(getattr(self.config, 'regime_enforce_stop', False)),
            'retrain_flag': retrain_flag,
        }

    def _select_top1_with_regime(self, filtered_df: pd.DataFrame, prediction_date: datetime.date) -> Dict[str, Any]:
        """
        最终方案（基于离线分析定型）：
        - 主策略始终是：分类 Top5 -> 排序选1
        - Regime 默认只做“监控与日志”，不做交易层硬切换
        - 只有当 regime_enforce_stop=True 且 state 属于 BAD_MARKET/MODEL_DECAY 时才停机
        """
        if filtered_df is None or len(filtered_df) == 0:
            return {'skip': True, 'reason': 'filtered_df empty'}

        regime = self._compute_regime_state(prediction_date)
        base = self._compute_base_candidate_metrics(filtered_df)

        df = filtered_df.copy()
        if 'p_cls' not in df.columns:
            rank_sorted = df.sort_values('predicted_return', ascending=False).reset_index(drop=True)
            target = rank_sorted.iloc[0]
            return {
                'skip': False, 'target_stock': target, 'selector': 'legacy_rank_top1',
                'regime_state': regime['state'], 'regime_metrics': regime, **base
            }

        cls_sorted = df.sort_values('p_cls', ascending=False).reset_index(drop=True)
        state = regime['state']

        good_n = int(min(max(int(getattr(self.config, 'regime_good_topn', 5) or 5), 1), len(cls_sorted)))
        topk = cls_sorted.head(good_n).copy()
        target = topk.sort_values('predicted_return', ascending=False).reset_index(drop=True).iloc[0]

        stop_states = {'BAD_MARKET', 'MODEL_DECAY', 'BAD_MARKET+MODEL_DECAY'}
        enforce_stop = bool(regime.get('enforce_stop', False))

        if enforce_stop and state in stop_states:
            return {
                'skip': True,
                'reason': f'{state} skip (r10_cand={regime.get("rolling_candidate_alpha")}, r10_rerank={regime.get("rolling_rerank_alpha")}, r20_margin={regime.get("rolling_score_margin")})',
                'selector': f'{state}_skip',
                'regime_state': state,
                'regime_metrics': regime,
                **base
            }

        selector = f'{state}_clsTop{good_n}_rankTop1'
        if state in stop_states and not enforce_stop:
            selector = f'{state}_monitorOnly_clsTop{good_n}_rankTop1'

        return {
            'skip': False,
            'target_stock': target,
            'selector': selector,
            'regime_state': state,
            'regime_metrics': regime,
            **base
        }

    def pre_analyze_required_stocks(self, max_k: int = 5) -> None:
        """
        预分析需要加载的股票 - 解决96.5%数据被错误过滤的问题
        """
        logger.info("🔍 开始智能分析候选股票...")

        self.required_stocks_by_date = {}  # 按日期保存每日Top-K股票
        trading_dates = sorted(self.prediction_data['date'].unique())

        for date in trading_dates:
            daily_predictions = self.prediction_data[self.prediction_data['date'] == date]
            logger.info(f"📅 {date}: 日线数据 {len(daily_predictions)} 条记录")
            filtered_stocks = []
            filtering_stats = {
                'total': len(daily_predictions),
                'is_can_buy_filtered': 0,
                'exchange_filtered': 0,
                'st_filtered': 0,
                'price_filtered': 0,
                'liquidity_filtered': 0,
                'limit_filtered': 0,
                'zhaban_filtered': 0,
                'passed': 0
            }

            for _, stock in daily_predictions.iterrows():
                stock_code = stock['stock_code']

                # 修复流动性过滤逻辑
                if stock['is_can_buy'] != 1:
                    filtering_stats['is_can_buy_filtered'] += 1
                    continue
                if stock_code.startswith(('8', '4')):  # 北交所
                    filtering_stats['exchange_filtered'] += 1
                    continue
                if any(x in stock_code for x in ['ST', 'st', '*ST', '*st']):
                    filtering_stats['st_filtered'] += 1
                    continue
                if stock['close'] <= 1.0 or stock['close'] >= 10000:
                    filtering_stats['price_filtered'] += 1
                    continue
                # 使用成交额而不是成交量
                if 'amount' in stock and stock['amount'] < 10000000:  # 1000万
                    filtering_stats['liquidity_filtered'] += 1
                    continue
                elif 'amount' not in stock and stock['volume'] < 1000000:  # 回退到成交量
                    filtering_stats['liquidity_filtered'] += 1
                    continue
                if stock['is_limit_up'] == 1 or stock['is_limit_down'] == 1:
                    filtering_stats['limit_filtered'] += 1
                    # 🔍 添加调试日志：记录被过滤的涨停跌停股票
                    # logger.info(f"🚫 涨跌停过滤: {stock['stock_code']} is_limit_up={stock.get('is_limit_up', 'N/A')}, is_limit_down={stock.get('is_limit_down', 'N/A')}, pct_chg={stock.get('pct_chg', 'N/A')}")
                    continue
                if stock['is_zhaban'] == 1:
                    filtering_stats['zhaban_filtered'] += 1
                    continue

                filtered_stocks.append(stock)
                filtering_stats['passed'] += 1
                # 🔍 添加调试日志：记录通过过滤的股票关键信息
                if stock['stock_code'] == '603778.SH':  # 专门跟踪问题股票
                    logger.info(f"✅ 603778.SH通过过滤: is_limit_up={stock.get('is_limit_up', 'N/A')}, is_limit_down={stock.get('is_limit_down', 'N/A')}, is_zhaban={stock.get('is_zhaban', 'N/A')}, pct_chg={stock.get('pct_chg', 'N/A')}, predicted_return={stock.get('predicted_return', 'N/A')}")

            # 记录过滤统计信息
            logger.info(f"🔍 {date} 过滤统计:")
            logger.info(f"   总候选: {filtering_stats['total']}只")
            logger.info(f"   可买入过滤: {filtering_stats['is_can_buy_filtered']}只")
            logger.info(f"   交易所过滤: {filtering_stats['exchange_filtered']}只")
            logger.info(f"   ST股过滤: {filtering_stats['st_filtered']}只")
            logger.info(f"   价格过滤: {filtering_stats['price_filtered']}只")
            logger.info(f"   流动性过滤: {filtering_stats['liquidity_filtered']}只")
            logger.info(f"   涨跌停过滤: {filtering_stats['limit_filtered']}只")
            logger.info(f"   炸板过滤: {filtering_stats['zhaban_filtered']}只")
            logger.info(f"   通过过滤: {filtering_stats['passed']}只")

            if filtered_stocks:
                filtered_df = pd.DataFrame(filtered_stocks).copy()
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                self.filtered_candidates_by_date[date_str] = filtered_df.copy()

                # 🧪 每个交易日预计算Regime诊断（不依赖是否成交；后续通过maturity_lag接入状态机）
                try:
                    base_metrics = self._compute_base_candidate_metrics(filtered_df)
                    diag = {
                        'prediction_date': date,
                        'prediction_idx': self.prediction_date_to_idx.get(date, None),
                        'serial_return': base_metrics.get('good_return'),  # 兼容旧字段名
                        'candidate_alpha': base_metrics.get('candidate_alpha', base_metrics.get('candidate_topk_mean_return')),
                        'rerank_alpha': base_metrics.get('rerank_alpha', base_metrics.get('base_conditional_gap')),
                        'conditional_gap': base_metrics.get('base_conditional_gap'),  # 兼容旧字段名
                        'score_margin': base_metrics.get('base_score_margin'),
                        'cls_top1_return': base_metrics.get('cls_top1_return'),
                        'rank_top1_code': base_metrics.get('rank_top1_code'),
                        'cls_top1_code': base_metrics.get('cls_top1_code'),
                        'top5_rank1_code': base_metrics.get('top5_rank1_code'),
                    }
                    self.daily_regime_diagnostics.append(diag)
                    logger.info(
                        f"🧪 {date_str} 日度Regime诊断: cand_alpha={diag.get('candidate_alpha')} rerank_alpha={diag.get('rerank_alpha')} "
                        f"good_ret={diag['serial_return']} cls_ret={diag['cls_top1_return']} gap={diag['conditional_gap']} margin={diag['score_margin']}"
                    )
                except Exception as _e_daily_diag:
                    logger.warning(f"⚠️ {date_str} 日度Regime诊断预计算失败: {_e_daily_diag}")

                # 兼容旧逻辑：仍然保存一个按rank排过的Top-K列表，用于非Top1策略
                if getattr(self.config, 'cls_gate_topn', 0) and 'p_cls' in filtered_df.columns:
                    gate_n = int(self.config.cls_gate_topn)
                    if gate_n > 0 and len(filtered_df) > gate_n:
                        gated_df = filtered_df.sort_values('p_cls', ascending=False).head(gate_n)
                        logger.info(f"🧱 {date} Stage-1门控: p_cls Top{gate_n} / {len(filtered_df)}")
                    else:
                        gated_df = filtered_df
                    filtered_sorted = gated_df.sort_values('predicted_return', ascending=False)
                else:
                    gated_df = filtered_df
                    filtered_sorted = filtered_df.sort_values('predicted_return', ascending=False)

                top_candidates = filtered_sorted.head(max_k)

                # 额外日志：保存候选全集中分类Top1 / GOOD模式Top5->rank1 / rankTop1
                try:
                    cls_top1 = gated_df.sort_values('p_cls', ascending=False).head(1) if 'p_cls' in gated_df.columns else pd.DataFrame()
                    rank_top1 = filtered_sorted.head(1)
                    good_n = min(int(getattr(self.config, 'regime_good_topn', 5) or 5), len(gated_df)) if 'p_cls' in gated_df.columns else 0
                    good_top1 = gated_df.sort_values('p_cls', ascending=False).head(good_n).sort_values('predicted_return', ascending=False).head(1) if good_n > 0 else pd.DataFrame()
                    if len(cls_top1) > 0:
                        logger.info(f"🧭 {date_str} 预分析/分类Top1: {cls_top1.iloc[0]['stock_code']} p_cls={float(cls_top1.iloc[0].get('p_cls', np.nan)):.4f} rank={float(cls_top1.iloc[0].get('predicted_return', np.nan)):.4f}")
                    if len(good_top1) > 0:
                        logger.info(f"🧭 {date_str} 预分析/GOOD候选Top{good_n}->rank1: {good_top1.iloc[0]['stock_code']} p_cls={float(good_top1.iloc[0].get('p_cls', np.nan)):.4f} rank={float(good_top1.iloc[0].get('predicted_return', np.nan)):.4f}")
                    if len(rank_top1) > 0:
                        logger.info(f"🧭 {date_str} 预分析/rankTop1: {rank_top1.iloc[0]['stock_code']} rank={float(rank_top1.iloc[0].get('predicted_return', np.nan)):.4f}")
                except Exception:
                    pass

                logger.info(f"🏆 {date} Top-{max_k}候选股票:")
                for i, (_, candidate) in enumerate(top_candidates.iterrows(), 1):
                    logger.info(f"   {i}. {candidate['stock_code']}: 预测收益={candidate['predicted_return']:.4f}, 收盘价={candidate['close']:.2f}")

                self.required_stocks_by_date[date_str] = top_candidates['stock_code'].tolist()
                logger.info(f"💾 {date_str} 保存Top-{max_k}股票: {len(self.required_stocks_by_date[date_str])}只；保存过滤候选全集: {len(filtered_df)}只")
            else:
                logger.warning(f"⚠️ {date}: 无通过过滤的候选股票")
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                self.required_stocks_by_date[date_str] = []

        total_stocks = len(self.prediction_data['stock_code'].unique())
        total_top_stocks = sum(len(stocks) for stocks in self.required_stocks_by_date.values())
        total_dates = len(self.required_stocks_by_date)

        logger.info(f"🎯 智能分析完成:")
        logger.info(f"   总股票数: {total_stocks}")
        logger.info(f"   分析日期数: {total_dates}天")
        logger.info(f"   Top-K候选总数: {total_top_stocks}只")

        # 检查关键日期和股票
        for check_date, check_code in [('2025-07-11', '002809.SZ')]:
            if check_date in self.required_stocks_by_date:
                if check_code in self.required_stocks_by_date[check_date]:
                    logger.info(f"✅ {check_code} 已包含在{check_date}的Top-K候选股票中")
                else:
                    logger.warning(f"❌ {check_code} 未包含在{check_date}的Top-K候选股票中")
                    logger.info(f"   {check_date}候选股票: {self.required_stocks_by_date[check_date]}")
            else:
                logger.warning(f"❌ 未找到{check_date}的候选股票数据")

    def load_minute_data_on_demand(self, date_str: str, target_stocks: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        优化的按需加载5分钟数据 - 支持指定股票列表的精准加载
        """
        # 统一缓存Key策略：只按日期缓存，不按股票列表缓存
        cache_key = date_str
        if cache_key in self.minute_data_cache:
            cached_data = self.minute_data_cache[cache_key]
            # 如果指定了目标股票，检查缓存中是否包含这些股票
            if target_stocks:
                # 检查所有目标股票是否都在缓存中
                found_stocks = [stock for stock in target_stocks if stock in cached_data and not cached_data[stock].empty]
                missing_stocks = [stock for stock in target_stocks if stock not in found_stocks]

                if missing_stocks:
                    logger.info(f"🔍 {date_str}: 缓存中缺少{len(missing_stocks)}只目标股票，重新加载: {missing_stocks[:3]}...")
                    logger.debug(f"   缓存中已有: {list(cached_data.keys())[:5]}")
                    # 重新加载所有目标股票（避免复杂的数据合并逻辑）
                    return self._reload_target_stocks(date_str, target_stocks)
                else:
                    # 所有目标股票都在缓存中
                    filtered_data = {code: cached_data[code] for code in target_stocks if code in cached_data}
                    logger.info(f"📋 {date_str}: 使用缓存数据，共{len(cached_data)}只股票，返回{len(filtered_data)}只目标股票")
                    return filtered_data
            else:
                logger.info(f"📋 {date_str}: 使用缓存数据，{len(cached_data)}只股票")
                return cached_data

        try:
            logger.info(f"📂 开始加载{date_str}的5分钟数据...")
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            year_month = target_date.strftime('%Y%m')
            month_dir = os.path.join(self.minute_data_path, year_month)
            minute_file = os.path.join(month_dir, "minute_data.parquet")

            normalized_path = minute_file.replace(os.sep, '/')
            if not os.path.exists(minute_file):
                logger.warning(f"❌ 未找到{date_str}的数据文件: {normalized_path}")
                return {}

            logger.info(f"📁 数据文件路径: {normalized_path}")
            logger.info(f"✅ 文件存在，大小: {os.path.getsize(minute_file)/(1024*1024):.1f}MB")

            from datetime import datetime as dt
            start_time = dt.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')
            end_time = dt.strptime(f'{date_str} 23:59:59', '%Y-%m-%d %H:%M:%S')

            # 策略选择：如果指定了目标股票且数量合理，使用精准加载
            if target_stocks and len(target_stocks) <= 20:
                return self._load_specific_stocks(date_str, minute_file, start_time, end_time, target_stocks)
            else:
                return self._load_all_stocks_optimized(date_str, minute_file, start_time, end_time)

        except Exception as e:
            logger.error(f"加载{date_str}分钟数据失败: {e}")
            return {}

    def _load_specific_stocks(self, date_str: str, minute_file: str, start_time: datetime,
                             end_time: datetime, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        精准加载指定的股票数据 - 避免加载不必要的数据
        """
        logger.info(f"🎯 精准加载{len(target_stocks)}只目标股票的5分钟数据: {target_stocks[:5]}...")

        try:
            # 组合过滤器：日期 + 指定股票列表
            filters = [
                ('datetime', '>=', start_time),
                ('datetime', '<=', end_time),
                ('stock_code', 'in', target_stocks)
            ]

            # 只读取必要的列，减少内存占用
            columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

            logger.info(f"🔍 使用精准过滤器加载指定股票数据...")
            table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
            df = table.to_pandas()

            logger.info(f"✅ 精准数据加载成功，实际数据{len(df)}条")

            # 按股票分组
            stock_mapping = {}
            if not df.empty and 'stock_code' in df.columns:
                for stock_code in target_stocks:
                    stock_data = df[df['stock_code'] == stock_code].sort_values('datetime')
                    if not stock_data.empty:
                        stock_mapping[stock_code] = stock_data
                        logger.debug(f"   ✓ {stock_code}: {len(stock_data)}条5分钟数据")
                    else:
                        logger.warning(f"   ⚠️ {stock_code}: 无数据")

            # 缓存结果 - 统一使用日期作为Key，覆盖式缓存避免复杂合并
            cache_key = date_str
            self.minute_data_cache[cache_key] = stock_mapping.copy()
            logger.info(f"📊 精准加载完成并缓存: {len(stock_mapping)}只股票 (缓存Key: {cache_key})")
            return stock_mapping

        except Exception as e:
            logger.error(f"精准加载{date_str}数据失败: {e}")
            # 降级到逐只股票加载
            return self._load_stocks_one_by_one(date_str, minute_file, start_time, end_time, target_stocks)

    def _load_stocks_one_by_one(self, date_str: str, minute_file: str, start_time: datetime,
                               end_time: datetime, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        降级方案：逐只股票加载，避免内存问题
        """
        logger.info(f"🔄 使用降级方案：逐只加载{len(target_stocks)}只股票...")

        stock_mapping = {}
        columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

        for stock_code in target_stocks:
            try:
                # 单只股票过滤器
                filters = [
                    ('datetime', '>=', start_time),
                    ('datetime', '<=', end_time),
                    ('stock_code', '=', stock_code)  # 精确匹配单只股票
                ]

                table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
                df = table.to_pandas()

                if not df.empty:
                    stock_mapping[stock_code] = df.sort_values('datetime')
                    logger.debug(f"   ✓ {stock_code}: {len(df)}条数据")
                else:
                    logger.warning(f"   ⚠️ {stock_code}: 无数据")

            except Exception as e:
                logger.error(f"   ❌ {stock_code}: 加载失败 - {e}")
                continue

        logger.info(f"📊 降级加载完成: {len(stock_mapping)}/{len(target_stocks)}只股票")

        # 缓存结果 - 统一使用日期作为Key，覆盖式缓存避免复杂合并
        if stock_mapping:  # 只有当有数据时才缓存
            cache_key = date_str
            self.minute_data_cache[cache_key] = stock_mapping.copy()
            logger.info(f"📊 降级加载完成并缓存: {len(stock_mapping)}只股票 (缓存Key: {cache_key})")

        return stock_mapping

    def _load_all_stocks_optimized(self, date_str: str, minute_file: str,
                                  start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
        """
        优化的全量股票加载 - 使用内存映射和列过滤
        """
        logger.info(f"📊 加载{date_str}所有股票的5分钟数据（优化模式）...")

        try:
            # 只使用日期过滤器，但限制列读取
            filters = [
                ('datetime', '>=', start_time),
                ('datetime', '<=', end_time)
            ]

            # 只读取必要的列
            columns = ['datetime', 'stock_code', 'open', 'close', 'high', 'low', 'volume']

            logger.info(f"🔍 使用内存映射和列过滤优化...")
            table = pq.read_table(minute_file, filters=filters, columns=columns, memory_map=True)
            df = table.to_pandas()

            logger.info(f"✅ 数据加载成功，原始数据{len(df)}条，内存使用优化")

            # 按股票分组
            stock_mapping = {}
            if not df.empty and 'stock_code' in df.columns:
                logger.info(f"🔄 按股票分组处理...")
                for stock_code, stock_data in df.groupby('stock_code'):
                    stock_data_sorted = stock_data.sort_values('datetime')
                    stock_mapping[stock_code] = stock_data_sorted
                logger.info(f"✅ 成功分组{len(stock_mapping)}只股票")

            # 缓存结果
            self.minute_data_cache[date_str] = stock_mapping
            logger.info(f"📊 {date_str}数据加载完成: {len(stock_mapping)}只股票")
            return stock_mapping

        except Exception as e:
            logger.error(f"优化加载{date_str}数据失败: {e}")
            return {}

    def _preload_stocks_data(self, dates: List[datetime.date], target_stocks: List[str]):
        """
        批量预加载指定日期和股票的5分钟数据 - 提高效率，避免重复IO
        """
        logger.info(f"🚀 开始批量预加载: {len(dates)}个日期 x {len(target_stocks)}只股票")

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                # 使用精准批量加载
                stock_mapping = self.load_minute_data_on_demand(date_str, target_stocks)
                logger.info(f"📦 {date_str}: 预加载完成 {len(stock_mapping)}只股票数据")

            except Exception as e:
                logger.error(f"❌ {date_str}: 预加载失败 - {e}")
                continue

        logger.info(f"✅ 批量预加载完成")

    def _reload_target_stocks(self, date_str: str, target_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """
        重新加载指定日期的目标股票数据，覆盖现有缓存
        """
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            year_month = target_date.strftime('%Y%m')
            month_dir = os.path.join(self.minute_data_path, year_month)
            minute_file = os.path.join(month_dir, "minute_data.parquet")

            if not os.path.exists(minute_file):
                logger.warning(f"❌ 未找到{date_str}的数据文件")
                return {stock: pd.DataFrame() for stock in target_stocks}

            start_time = datetime.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(f'{date_str} 23:59:59', '%Y-%m-%d %H:%M:%S')

            # 策略选择：如果目标股票较少，使用精准加载
            if len(target_stocks) <= 20:
                return self._load_specific_stocks(date_str, minute_file, start_time, end_time, target_stocks)
            else:
                # 加载所有股票然后过滤
                all_stocks_data = self._load_all_stocks_optimized(date_str, minute_file, start_time, end_time)
                # 更新缓存
                self.minute_data_cache[date_str] = all_stocks_data
                # 返回目标股票的数据
                return {stock: all_stocks_data.get(stock, pd.DataFrame()) for stock in target_stocks}

        except Exception as e:
            logger.error(f"重新加载{date_str}目标股票失败: {e}")
            return {stock: pd.DataFrame() for stock in target_stocks}

    def get_stock_data(self, stock_code: str, date: datetime.date) -> pd.DataFrame:
        """获取指定股票在指定日期的5分钟数据 - 使用精准查询优化"""
        date_str = date.strftime('%Y-%m-%d')

        try:
            # 使用精准加载：只加载指定的股票数据
            stock_mapping = self.load_minute_data_on_demand(date_str, [stock_code])
            return stock_mapping.get(stock_code, pd.DataFrame())

        except Exception as e:
            logger.error(f"获取{stock_code}在{date_str}的5分钟数据失败: {e}")
            return pd.DataFrame()

    def _calculate_daily_ma5_at_open(self, stock_code: str, buy_date: datetime.date,
                                     buy_open: float) -> Optional[float]:
        """
        计算买入日开盘时的日线MA5（前4天日线close + 买入日开盘价）

        Args:
            stock_code: 股票代码
            buy_date: 买入日期
            buy_open: 买入日开盘价

        Returns:
            MA5值，如果前4天数据不足返回None

        Note:
            MA5 = (前4天日线close之和 + 买入日开盘价) / 5

        Example:
            买入日: 2026-01-10, 开盘价: 10.5
            前4天日线close: [10.2, 10.3, 10.1, 10.4]
            MA5 = (10.2 + 10.3 + 10.1 + 10.4 + 10.5) / 5 = 10.3
        """
        # 获取前4个交易日的日线close
        daily_closes = self._get_daily_closes_before(stock_code, buy_date, 4)

        # 检查数据是否充足
        if len(daily_closes) < 4:
            logger.info(f"⚠️ {stock_code} 在{buy_date}前不足4个交易日数据")
            # return None

        # 计算MA5：前4天日线 + 买入日开盘
        ma5 = (sum(daily_closes) + buy_open) / 5.0

        logger.info(f"📊 {stock_code} 买入日开盘MA5计算:")
        logger.info(f"   前4天日线close: {daily_closes}")
        logger.info(f"   买入日开盘价: {buy_open}")
        logger.info(f"   MA5: {ma5:.4f}")

        return ma5

    def check_entry_signal(self, stock_code: str, buy_date: datetime.date,
                          prev_close: Optional[float], buy_open: float) -> Dict[str, Any]:
        """
        开盘风控检查 - 支持配置化策略选择

        支持两种策略：
        1. GAP_BASED: 基于涨跌幅的风控（原有逻辑）
        2. MA5_BASED: 基于MA5的风控（新增逻辑）
        """
        logger.info(f"🔎 {stock_code} 开盘风控检查: buy_date={buy_date}, prev_close={prev_close}, buy_open={buy_open}")
        logger.info(f"📋 开盘风控策略: {self.config.entry_risk_control_strategy}")

        # ===== 根据配置选择策略 =====
        if self.config.entry_risk_control_strategy == EntryRiskControlStrategy.GAP_BASED:
            # ===== 原有策略: 基于涨跌幅的风控 =====
            if prev_close is None:
                logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
                return {'can_buy': True, 'reason': '无前日数据'}

            gap_ratio = (buy_open - prev_close) / prev_close
            logger.info(f"📊 {stock_code}: 跳空计算 (buy_open={buy_open} - prev_close={prev_close}) / prev_close = {gap_ratio:.4f} ({gap_ratio:.2%})")

            # 风控条件1: 跌空超过3%
            if gap_ratio < self.config.entry_risk_down_threshold:  # 使用-0.03
                logger.info(f"🚫 {stock_code}: 跌空超过3% ({gap_ratio:.2%})，禁止买入")
                return {
                    'can_buy': False,
                    'reason': f'开盘跳空{-gap_ratio:.2%}',
                    'gap_ratio': gap_ratio,
                    'skip_category': '开盘风控'
                }

            # 风控条件2: 高开超过5%
            elif gap_ratio > self.config.entry_risk_up_threshold:  # 使用0.05
                logger.info(f"🚫 {stock_code}: 高开超过5% ({gap_ratio:.2%})，禁止买入")
                return {
                    'can_buy': False,
                    'reason': f'开盘跳空{gap_ratio:.2%}',
                    'gap_ratio': gap_ratio,
                    'skip_category': '开盘风控'
                }

            logger.info(f"✅ {stock_code}: 通过所有风控检查，跳空{gap_ratio:.2%}在范围内")
            return {'can_buy': True, 'reason': '通过风控', 'gap_ratio': gap_ratio}

        elif self.config.entry_risk_control_strategy == EntryRiskControlStrategy.MA5_BASED:
            # ===== 🆕 新策略: 基于MA5的风控 =====
            # 计算MA5（前4天日线close + 买入日开盘价）
            ma5 = self._calculate_daily_ma5_at_open(stock_code, buy_date, buy_open)

            if ma5 is None:
                # 无法计算MA5，使用涨跌幅风控作为后备
                logger.warning(f"⚠️ {stock_code}: 无法计算MA5，使用涨跌幅风控作为后备")
                if prev_close is None:
                    logger.info(f"✅ {stock_code}: 无前日数据，允许买入")
                    return {'can_buy': True, 'reason': '无前日数据'}

                gap_ratio = (buy_open - prev_close) / prev_close
                # 简单涨跌幅风控
                if abs(gap_ratio) > 0.10:  # 跌幅或涨幅超10%才跳过
                    logger.info(f"🚫 {stock_code}: 跳空超10% ({gap_ratio:.2%})，禁止买入")
                    return {
                        'can_buy': False,
                        'reason': f'跳空超10%',
                        'gap_ratio': gap_ratio,
                        'skip_category': '开盘风控'
                    }

                return {'can_buy': True, 'reason': '后备涨跌幅风控通过', 'gap_ratio': gap_ratio}

            # 计算开盘价相对于MA5的涨跌幅
            ma5_gap_ratio = (buy_open - ma5) / ma5
            gap_ratio = (buy_open - prev_close) / prev_close
            logger.info(f"📊 {stock_code}: MA5跳空计算 (buy_open={buy_open} - MA5={ma5}) / MA5 = {ma5_gap_ratio:.4f} ({ma5_gap_ratio:.2%})")

            # 获取阈值
            down_threshold = self.config.entry_risk_ma5_down_threshold  # 默认-0.03
            up_threshold = self.config.entry_risk_ma5_up_threshold  # 默认0.03

            # 风控条件1: 低开超MA5 3%
            if ma5_gap_ratio < down_threshold:
                logger.info(f"🚫 {stock_code}: 低开超MA5 {abs(down_threshold):.0%} ({ma5_gap_ratio:.2%})，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}, 低开: {ma5_gap_ratio:.2%}")
                return {
                    'can_buy': False,
                    'reason': f'低开超MA5{abs(down_threshold):.0%}',
                    'gap_ratio': ma5_gap_ratio,
                    'skip_category': '开盘MA5风控',
                    'ma5_value': ma5  # 记录MA5值
                }

            # 风控条件2: 高开超MA5 3%
            elif gap_ratio > up_threshold:#ma5_gap_ratio > up_threshold:
                logger.info(f"🚫 {stock_code}: 高开超MA5 {up_threshold:.0%} ({ma5_gap_ratio:.2%})，禁止买入")
                logger.info(f"   MA5: {ma5:.4f}, 开盘价: {buy_open:.4f}, 高开: {ma5_gap_ratio:.2%}")
                return {
                    'can_buy': False,
                    'reason': f'高开超MA5{up_threshold:.0%}',
                    'gap_ratio': ma5_gap_ratio,
                    'skip_category': '开盘MA5风控',
                    'ma5_value': ma5  # 记录MA5值
                }

            logger.info(f"✅ {stock_code}: 通过所有风控检查，MA5跳空{ma5_gap_ratio:.2%}在范围内")
            return {
                'can_buy': True,
                'reason': '通过MA5风控',
                'gap_ratio': ma5_gap_ratio,
                'ma5_value': ma5  # 记录MA5值
            }

    def _get_daily_closes_before(self, stock_code: str, date: datetime.date, n: int) -> List[float]:
        """
        获取指定日期前N个交易日的日线close价格

        Args:
            stock_code: 股票代码
            date: 基准日期
            n: 需要获取的天数

        Returns:
            前N个交易日的close价格列表（按时间正序排列）
            如果数据不足，返回尽可能多的数据

        Example:
            如果 date=2026-01-10, n=4
            返回: [10.2, 10.3, 10.1, 10.4]  # 2026-01-06, 01-07, 01-08, 01-09的close
        """
        # 获取所有交易日期（已排序）
        all_dates = sorted(self.prediction_data['date'].unique())

        # 找到基准日期在交易日列表中的位置
        try:
            date_index = all_dates.index(date)
        except ValueError:
            # 如果基准日期不在交易日列表中，找最近的前一个交易日
            earlier_dates = [d for d in all_dates if d < date]
            if not earlier_dates:
                return []
            date = earlier_dates[-1]
            date_index = all_dates.index(date)

        # 获取前N个交易日
        start_index = max(0, date_index - n)
        target_dates = all_dates[start_index:date_index]

        # 获取这些日期的close价格
        closes = []
        for target_date in target_dates:
            daily_data = self.prediction_data[
                (self.prediction_data['date'] == target_date) &
                (self.prediction_data['stock_code'] == stock_code)
            ]
            if not daily_data.empty:
                closes.append(daily_data.iloc[0]['close'])
            else:
                logger.warning(f"⚠️ {stock_code} 在 {target_date} 无日线数据")

        logger.info(f"📊 {stock_code} 在{date}前{n}天的日线close: {closes}")
        return closes

    def _get_daily_low(self, stock_code: str, date: datetime.date) -> Optional[float]:
        """
        获取指定日期的日线最低价（用于T+1最低价跌破策略）

        Args:
            stock_code: 股票代码
            date: 指定日期（买入日/T+1日）

        Returns:
            指定日期的日线最低价，如果数据不存在返回None

        Example:
            如果 date=2026-01-10
            返回: 10.1  # 2026-01-10当天的最低价
        """
        try:
            # 从prediction_data中获取指定日期的日线数据
            daily_data = self.prediction_data[
                (self.prediction_data['date'] == date) &
                (self.prediction_data['stock_code'] == stock_code)
            ]

            if daily_data.empty:
                logger.warning(f"⚠️ {stock_code} 在 {date} 无日线数据")
                return None

            daily_low = daily_data.iloc[0]['close']
            logger.info(f"📊 {stock_code} 在{date}的日线最低价: {daily_low:.4f}")
            return daily_low

        except Exception as e:
            logger.error(f"❌ 获取{stock_code}在{date}的日线最低价失败: {e}")
            return None

    def _calculate_daily_ma5_realtime(self, stock_code: str, current_date: datetime.date,
                                      current_5min_close: float) -> Optional[float]:
        """
        实时计算日线MA5（前4天日线close + 当前5分钟close）

        Args:
            stock_code: 股票代码
            current_date: 当前日期
            current_5min_close: 当前5分钟K线的close价格

        Returns:
            MA5值，如果前4天数据不足返回None

        Note:
            MA5 = (前4天日线close之和 + 当前5分钟close) / 5

        Example:
            当前日期: 2026-01-10, 5分钟close: 10.5
            前4天日线close: [10.2, 10.3, 10.1, 10.4]
            MA5 = (10.2 + 10.3 + 10.1 + 10.4 + 10.5) / 5 = 10.3
        """
        # 获取前4个交易日的日线close
        daily_closes = self._get_daily_closes_before(stock_code, current_date, 4)

        # 检查数据是否充足
        if len(daily_closes) < 4:
            logger.info(f"⚠️ {stock_code} 在{current_date}前不足4个交易日数据")
            # return None

        # 计算MA5：前4天日线 + 当前5分钟
        ma5 = (sum(daily_closes) + current_5min_close) / 5.0

        logger.info(f"📊 {stock_code} 日线MA5计算:")
        logger.info(f"   前4天日线close: {daily_closes}")
        logger.info(f"   当前5分钟close: {current_5min_close}")
        logger.info(f"   MA5: {ma5:.4f}")

        return ma5

    def check_position_risk(self, position: Position, current_date: datetime.date, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """持仓风控检查 - 使用交易日期列表，避免非交易日"""
        logger.info(f"🛡️ 开始风控检查: {position.stock_code}, 买入日: {position.buy_date}, 当前日: {current_date}")

        # 确保trading_dates按时间顺序排列
        trading_dates = sorted(trading_dates)

        try:
            # 找到买入日在交易日期列表中的索引
            buy_date_index = trading_dates.index(position.buy_date)
            current_date_index = trading_dates.index(current_date)
            logger.debug(f"   买入日索引: {buy_date_index}, 当前日索引: {current_date_index}")
        except ValueError as e:
            logger.error(f"❌ 日期索引查找失败: {e}")
            return {'should_exit': False, 'reason': '日期索引错误'}

        # 检查是否到了T+2（买入日的下1个交易日）
        if current_date_index <= buy_date_index:
            days_to_wait = (buy_date_index + 1) - current_date_index
            logger.info(f"⏸️ {position.stock_code}: 未到T+2，还需等待{days_to_wait}个交易日")
            return {'should_exit': False, 'reason': f'未到T+2(还需{days_to_wait}日)'}

        # 确定风控检查范围：T+2交易日（买入日的下1个交易日）~ min(T+holding_days交易日, 当前日)
        start_index = buy_date_index + 1  # T+2交易日（买入日的下一个交易日）
        end_index = min(
            buy_date_index + self.config.holding_days,  # T+holding_days交易日
            current_date_index + 1  # 当前日（包含）
        )

        # 获取风控检查的实际日期范围
        # ⚠️ 关键修复：本函数每天都会被调用一次，若每次都把 T+2~当前日 的分钟数据重新拼接并从头扫描，
        # 会导致“历史分钟K线被重复检查”，而 position.highest_price_since_buy 等状态又是跨天累计的，
        # 从而在新的一天把上一交易日的第一根K线当成“第1根”重新计算，出现误触发。
        # 因此：每次只检查【当前交易日】的 5 分钟序列，状态（最高价/最高浮盈等）由 position 跨天携带。
        risk_check_dates = [current_date]
        logger.info(f"📊 {position.stock_code} 风控检查范围: {current_date} ~ {current_date} (当日{len(risk_check_dates)}个交易日)")

        # 获取所有风控检查日内的5分钟数据
        minute_data_list = []
        logger.info(f"📊 收集{position.stock_code}在{len(risk_check_dates)}个交易日内的5分钟数据进行风控检查...")

        for check_date in risk_check_dates:
            day_data = self.get_stock_data(position.stock_code, check_date)
            if not day_data.empty:
                logger.debug(f"   {check_date}: 获取到{len(day_data)}条5分钟数据")
                for _, row in day_data.iterrows():
                    minute_data_list.append(row.to_dict())
            else:
                logger.warning(f"   ⚠️ {check_date}: 无5分钟数据")

        logger.info(f"📈 {position.stock_code}: 风控检查范围内总共收集到{len(minute_data_list)}条5分钟数据")

        if len(minute_data_list) < 3:
            logger.info(f"⚠️ {position.stock_code}: 数据不足({len(minute_data_list)}条)，跳过阴线检查")
            return {'should_exit': False, 'reason': '数据不足'}

        # 统一时间序列风控检查 - 实现"先满足先触发"原则
        logger.debug(f"🛡️ 开始统一风控检查（使用止损阈值: {self.config.stop_loss_threshold:.2%}）...")
        logger.info(f"📋 风控策略: {self.config.risk_control_strategy}")

        # 🆕 日线MA5策略状态变量
        consecutive_below_ma5 = 0  # 连续跌破MA5的次数
        last_check_date = None  # 上次检查的日期（用于跨天重置）


        # 初始化状态变量  对应T1_LOW_BREAK_WITH_HIGH_DRAWDOWN策略
        consecutive_below_t1_low = 0  # 连续跌破T+1日最低价的次数
        consecutive_bearish_in_profit = 0  # 🆕 盈利状态下连续阴线次数

        # 按时间顺序逐个K线检查，实现先满足先触发
        for i, minute_data in enumerate(minute_data_list):
            close_price = minute_data['close']
            high_price = minute_data['high']
            current_datetime = minute_data['datetime']

            # 提取当前K线的日期
            if isinstance(current_datetime, str):
                check_date = datetime.fromisoformat(current_datetime).date()
            elif hasattr(current_datetime, 'to_pydatetime'):
                check_date = current_datetime.to_pydatetime().date()
            elif isinstance(current_datetime, datetime):
                check_date = current_datetime.date()
            else:
                check_date = current_date

            # 🆕 跨天处理：如果日期变化，重置连续跌破计数
            if last_check_date is not None and check_date != last_check_date:
                if consecutive_below_ma5 > 0:
                    logger.info(f"📅 {position.stock_code}: 跨天检测，重置MA5跌破计数（从{consecutive_below_ma5}重置为0）")
                consecutive_below_ma5 = 0
            last_check_date = check_date

            # 1. 检查累计下跌/收益低于阈值（基于买入价，使用收盘价）
            decline_ratio = (close_price - position.buy_price) / position.buy_price
            if decline_ratio <= self.config.stop_loss_threshold:  # 使用配置的止损阈值
                # 处理不同的datetime格式
                if isinstance(current_datetime, str):
                    exit_time = datetime.fromisoformat(current_datetime)
                elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                    exit_time = current_datetime.to_pydatetime()
                elif isinstance(current_datetime, datetime):
                    exit_time = current_datetime
                else:
                    exit_time = datetime.now()

                # 动态生成阈值描述
                threshold_desc = f"{self.config.stop_loss_threshold:.2%}"
                if self.config.stop_loss_threshold == 0.0:
                    threshold_desc = "0% (盈亏平衡)"
                elif self.config.stop_loss_threshold > 0:
                    threshold_desc = f"{self.config.stop_loss_threshold:.2%} (要求正收益)"

                logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发止损!")
                logger.debug(f"   止损K线: 时间={current_datetime}, 开盘={minute_data['open']:.4f}, 收盘={close_price:.4f}")
                logger.debug(f"   买入价: {position.buy_price:.4f}, 收盘价: {close_price:.4f}, 收益: {decline_ratio:.2%}")
                logger.debug(f"   止损阈值: {threshold_desc}")

                return {
                    'should_exit': True,
                    'reason': f'累计收益低于{threshold_desc}',
                    'exit_price': close_price,  # 使用收盘价成交
                    'exit_time': exit_time,
                    'decline_ratio': decline_ratio,
                    'stop_loss_threshold': self.config.stop_loss_threshold,  # 记录使用的阈值
                }

            # ===== 风控条件2: 根据配置选择策略 =====
            if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
                # ===== 原有策略: 连续阴线风控 =====
                if i >= 2:  # 至少有3根K线可以检查连续性
                    # 获取最近的3根K线：第i-2, i-1, i根
                    # candle_0 = minute_data_list[i -3]
                    candle_1 = minute_data_list[i - 2]
                    candle_2 = minute_data_list[i - 1]
                    candle_3 = minute_data

                    # 检查这3根K线是否都是阴线
                    # is_red_0 = candle_0['close'] < candle_0['open']
                    is_red_1 = candle_1['close'] < candle_1['open']
                    is_red_2 = candle_2['close'] < candle_2['open']
                    is_red_3 = candle_3['close'] < candle_3['open']

                    if is_red_1 and is_red_2 and is_red_3: #is_red_0 and 
                        # 发现连续3根阴线，触发风控
                        logger.info(f"🚨 {position.stock_code}: 在第{i-1}-{i+1}根K线发现连续3根阴线!")
                        logger.debug(f"   阴线1: 时间={candle_1['datetime']}, 开盘={candle_1['open']:.4f}, 收盘={candle_1['close']:.4f}")
                        logger.debug(f"   阴线2: 时间={candle_2['datetime']}, 开盘={candle_2['open']:.4f}, 收盘={candle_2['close']:.4f}")
                        logger.debug(f"   阴线3: 时间={candle_3['datetime']}, 开盘={candle_3['open']:.4f}, 收盘={candle_3['close']:.4f}")

                        # 处理不同的datetime格式
                        if isinstance(current_datetime, str):
                            exit_time = datetime.fromisoformat(current_datetime)
                        elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                            exit_time = current_datetime.to_pydatetime()
                        elif isinstance(current_datetime, datetime):
                            exit_time = current_datetime
                        else:
                            exit_time = datetime.now()

                        return {
                            'should_exit': True,
                            'reason': '连续3根阴线',
                            'exit_price': candle_3['close'],
                            'exit_time': exit_time
                        }

            elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
                # ===== 🆕 新风控：MA5跌破 + 高位回撤双条件策略 =====

                # 初始化退出标记和结果字典
                should_exit_flag = False
                exit_result = {}

                # 1️⃣ 计算当前收益和更新最高价
                current_return = (close_price - position.buy_price) / position.buy_price

                # 更新买入后的最高价和最高浮盈
                if high_price > position.highest_price_since_buy:
                    position.highest_price_since_buy = high_price
                    position.highest_return_since_buy = (high_price - position.buy_price) / position.buy_price

                # 2️⃣ 计算动态MA5（前4天日线 + 当前5分钟）
                ma5 = self._calculate_daily_ma5_realtime(
                    position.stock_code,
                    check_date,
                    close_price
                )

                if ma5 is None:
                    logger.info(f"⏸️ {position.stock_code}: 无法计算MA5，跳过检查")
                    # ✅ 修改：不直接return，而是跳过本次检查继续循环
                    continue

                # 3️⃣ 条件1：MA5跌破 + 当前收益≤0
                is_below_ma5 = close_price < ma5
                current_return_negative = current_return <= 0

                if is_below_ma5:
                    consecutive_below_ma5 += 1
                    logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                    logger.info(f"   close({close_price:.4f}) < MA5({ma5:.4f}), 收益:{current_return*100:.2f}%")
                    logger.info(f"   连续跌破MA5: {consecutive_below_ma5}次")
                else:
                    if consecutive_below_ma5 > 0:
                        logger.info(f"📈 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}) >= MA5({ma5:.4f}), 收益:{current_return*100:.2f}%")
                        logger.info(f"   重置跌破计数（从{consecutive_below_ma5}重置为0）")
                    consecutive_below_ma5 = 0

                # 检查条件1是否满足（亏损或持平状态 + 连续跌破）
                condition1_trigger = (current_return_negative and
                                  consecutive_below_ma5 >= self.config.risk_control_ma5_consecutive_breaks)

                # 4️⃣ 条件2：高位回撤
                # 判断是否曾经达到最高浮盈阈值（4%）
                highest_return_above_threshold = position.highest_return_since_buy >= self.config.high_return_threshold

                condition2_trigger = False
                if highest_return_above_threshold:
                    # 从最高点回撤超过3%（-0.03）
                    drawdown_from_highest = (close_price - position.highest_price_since_buy) / position.highest_price_since_buy
                    condition2_trigger = (drawdown_from_highest <= self.config.high_drawdown_threshold)

                    if condition2_trigger:
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}), 最高:{position.highest_price_since_buy:.4f}")
                        logger.info(f"   最高浮盈:{position.highest_return_since_buy*100:.2f}%")
                        logger.info(f"   回撤:{abs(drawdown_from_highest)*100:.2f}%")

                # 5️⃣ 判断是否触发卖出（两个条件OR关系）
                if condition1_trigger or condition2_trigger:
                    # ✅ 修改：不直接return，而是记录退出状态
                    should_exit_flag = True

                    exit_reason_parts = []

                    if condition1_trigger:
                        exit_reason_parts.append(
                            f"亏损状态下连续{consecutive_below_ma5}次跌破MA5"
                        )

                    if condition2_trigger:
                        drawdown_from_highest = (close_price - position.highest_price_since_buy) / position.highest_price_since_buy
                        exit_reason_parts.append(
                            f"最高浮盈{position.highest_return_since_buy*100:.2f}%后回撤{abs(drawdown_from_highest)*100:.2f}%"
                        )

                    exit_reason = " | ".join(exit_reason_parts)

                    # 记录退出信息（稍后return）
                    # 标准化退出时间
                    # 处理不同的datetime格式
                    if isinstance(current_datetime, str):
                        exit_time_standardized = datetime.fromisoformat(current_datetime)
                    elif hasattr(current_datetime, 'to_pydatetime'):  # pandas Timestamp
                        exit_time_standardized = current_datetime.to_pydatetime()
                    elif isinstance(current_datetime, datetime):
                        exit_time_standardized = current_datetime
                    else:
                        exit_time_standardized = datetime.now()

                    exit_result = {
                        'should_exit': True,
                        'reason': exit_reason,
                        'exit_price': close_price,
                        'exit_time': exit_time_standardized,  # 标准化时间
                        'ma5_value': ma5,
                        'current_return_pct': current_return * 100,
                        'highest_return_pct': position.highest_return_since_buy * 100,
                        'drawdown_from_highest_pct': drawdown_from_highest * 100 if condition2_trigger else None
                    }
                    return exit_result

                    # ✅ 修改：触发后立即退出循环，不再检查后续K线
                continue

            elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
                # ===== 🆕 新风控：T+1最低价跌破 + 高位回撤双状态策略 =====

                close_price = minute_data['close']
                high_price = minute_data['high']
                current_datetime = minute_data['datetime']

                # 提取当前K线的日期
                if isinstance(current_datetime, str):
                    check_date = datetime.fromisoformat(current_datetime).date()
                elif hasattr(current_datetime, 'to_pydatetime'):
                    check_date = current_datetime.to_pydatetime().date()
                elif isinstance(current_datetime, datetime):
                    check_date = current_datetime.date()
                else:
                    check_date = current_date

                # 1. 计算当前收益和更新最高价
                current_return = (close_price - position.buy_price) / position.buy_price

                # 更新买入后的最高价和最高浮盈
                if high_price > position.highest_price_since_buy:
                    position.highest_price_since_buy = high_price
                    position.highest_return_since_buy = (high_price - position.buy_price) / position.buy_price

                # ===== 状态判断：亏损/持平 vs 盈利 =====
                if current_return <= 0:
                    # ===== 状态A：亏损/持平状态 - T+1最低价跌破策略 =====
                    continue

                    # 检查是否跌破T+1日最低价
                    is_below_t1_low = close_price < position.t1_day_lowest_price

                    if is_below_t1_low:
                        consecutive_below_t1_low += 1
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线:")
                        logger.info(f"   close({close_price:.4f}) < T+1最低价({position.t1_day_lowest_price:.4f})")
                        logger.info(f"   收益:{current_return*100:.2f}%")
                        logger.info(f"   连续跌破T+1最低价: {consecutive_below_t1_low}次")
                    else:
                        if consecutive_below_t1_low > 0:
                            logger.info(f"📈 {position.stock_code} 第{i+1}根K线:")
                            logger.info(f"   close({close_price:.4f}) >= T+1最低价({position.t1_day_lowest_price:.4f})")
                            logger.info(f"   重置跌破计数（从{consecutive_below_t1_low}重置为0）")
                        consecutive_below_t1_low = 0

                    # 检查是否触发卖出：连续跌破T+1日最低价
                    if consecutive_below_t1_low >= self.config.risk_control_t1_low_consecutive_breaks:
                        logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发T+1最低价跌破风控!")
                        logger.debug(f"   触发K线: 时间={current_datetime}, close={close_price:.4f}")
                        logger.debug(f"   T+1最低价: {position.t1_day_lowest_price:.4f}")
                        logger.debug(f"   连续跌破次数: {consecutive_below_t1_low}")
                        logger.debug(f"   当前收益: {current_return*100:.2f}%")

                        # 处理不同的datetime格式
                        if isinstance(current_datetime, str):
                            exit_time = datetime.fromisoformat(current_datetime)
                        elif hasattr(current_datetime, 'to_pydatetime'):
                            exit_time = current_datetime.to_pydatetime()
                        elif isinstance(current_datetime, datetime):
                            exit_time = current_datetime
                        else:
                            exit_time = datetime.now()

                        return {
                            'should_exit': True,
                            'reason': f'亏损状态下连续{consecutive_below_t1_low}次跌破T+1最低价',
                            'exit_price': close_price,
                            'exit_time': exit_time,
                            't1_lowest_price': position.t1_day_lowest_price,
                            'current_return_pct': current_return * 100,
                            'consecutive_breaks': consecutive_below_t1_low
                        }
                else:
                    # ===== 状态B：盈利状态 - 连续阴线策略 =====

                    # 检查当前K线是否为阴线
                    is_bearish = close_price < minute_data['open']

                    if is_bearish:
                        consecutive_bearish_in_profit += 1
                        logger.info(f"📉 {position.stock_code} 第{i+1}根K线(盈利状态):")
                        logger.info(f"   open={minute_data['open']:.4f}, close={close_price:.4f} (阴线)")
                        logger.info(f"   当前收益:{current_return*100:.2f}%")
                        logger.info(f"   连续阴线: {consecutive_bearish_in_profit}次")
                    else:
                        # 阳线，重置计数
                        if consecutive_bearish_in_profit > 0:
                            logger.info(f"📈 {position.stock_code} 第{i+1}根K线(盈利状态):")
                            logger.info(f"   open={minute_data['open']:.4f}, close={close_price:.4f} (阳线)")
                            logger.info(f"   重置阴线计数（从{consecutive_bearish_in_profit}重置为0）")
                        consecutive_bearish_in_profit = 0

                    # 检查是否触发卖出：连续3根阴线
                    if consecutive_bearish_in_profit >= 3:
                        logger.info(f"🚨 {position.stock_code}: 在第{i+1}根K线触发连续阴线风控(盈利状态)!")
                        logger.info(f"   触发K线: 时间={current_datetime}")
                        logger.info(f"   连续阴线次数: {consecutive_bearish_in_profit}")

                        # 处理不同的datetime格式
                        if isinstance(current_datetime, str):
                            exit_time = datetime.fromisoformat(current_datetime)
                        elif hasattr(current_datetime, 'to_pydatetime'):
                            exit_time = current_datetime.to_pydatetime()
                        elif isinstance(current_datetime, datetime):
                            exit_time = current_datetime
                        else:
                            exit_time = datetime.now()

                        return {
                            'should_exit': True,
                            'reason': f'盈利状态下连续{consecutive_bearish_in_profit}根阴线',
                            'exit_price': close_price,
                            'exit_time': exit_time,
                            'current_return_pct': current_return * 100,
                            'consecutive_bearish': consecutive_bearish_in_profit
                        }

                # end for - T+1最低价跌破策略循环结束



        # 根据策略显示不同的日志信息
        if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 连续3根阴线）")
        elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 日线MA5跌破）")
        elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%} 或 T+1最低价跌破/盈利状态连续阴线）")
        else:
            logger.info(f"✅ {position.stock_code}: 未发现风控信号（止损阈值: {self.config.stop_loss_threshold:.2%}）")
        return {'should_exit': False, 'reason': '无风控信号'}



    def select_top_k_stocks(self, filtered_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """选择Top-K股票"""
        if len(filtered_df) < k:
            return filtered_df
        return filtered_df.nlargest(k, 'predicted_return')

    def add_trading_days(self, start_date: datetime.date, n: int,
                        trading_dates: List[datetime.date]) -> Optional[datetime.date]:
        """
        从start_date开始，计算n个交易日后的日期

        Args:
            start_date: 起始日期
            n: 要增加的交易日数量
            trading_dates: 完整的交易日期列表

        Returns:
            n个交易日后的日期，如果start_date不在trading_dates中则返回None
        """
        try:
            start_index = trading_dates.index(start_date)
            end_index = start_index + n

            # 边界检查：确保索引不超过列表长度
            if end_index < len(trading_dates):
                return trading_dates[end_index]
            else:
                # 如果超出范围，返回最后一个交易日
                logger.warning(f"⚠️ 到期日超出交易日期范围，使用最后一个交易日: {trading_dates[-1]}")
                return trading_dates[-1]
        except ValueError:
            # 如果买入日期不在trading_dates中，返回None
            logger.error(f"❌ 买入日期 {start_date} 不在交易日期列表中")
            return None

    def _is_last_trade_stop_loss(self) -> bool:
        """
        检查上一次交易是否触发5%止损

        Returns:
            True: 上次交易止损（≤ -5%）
            False: 上次交易未止损，或无历史交易
        """
        if not self.trades:
            return False

        last_trade = self.trades[-1]
        actual_return = last_trade.get('actual_return', 0.0)

        # 方式1: 检查收益率是否 ≤ -5%（使用配置的止损阈值）
        if actual_return <= self.config.stop_loss_threshold:
            logger.info(f"🛡️ [止损检测] 上次交易触发{self.config.stop_loss_threshold*100:.1f}%止损: "
                       f"{last_trade['stock_code']}, return={actual_return:.4f}")
            return True

        # 方式2: 检查退出原因是否包含"止损"字样
        exit_reason = last_trade.get('exit_reason', '')
        if '止损' in exit_reason or 'stop_loss' in exit_reason.lower():
            logger.info(f"🛡️ [止损检测] 上次交易触发止损(原因): {last_trade['stock_code']}, reason={exit_reason}")
            return True

        return False


    def _has_consecutive_losses(self, n: int) -> bool:
        """
        检查是否连续N次交易都是负收益

        Args:
            n: 连续次数

        Returns:
            True: 最近连续N次都是亏损
            False: 不足N次，或有盈利
        """
        if len(self.trades) < n:
            return False

        # 获取最近N次交易
        recent_trades = self.trades[-n:]

        # 检查是否都是负收益
        all_losses = all((trade.get('actual_return', 0.0) < 0 and trade.get('actual_return') > self.config.stop_loss_threshold) for trade in recent_trades)

        if all_losses:
            returns = [trade.get('actual_return', 0.0) for trade in recent_trades]
            logger.info(f"📉 [连续亏损] 检测到连续{n}次亏损: {[f'{r*100:.2f}%' for r in returns]}")

        return all_losses


    def _has_consecutive_profits(self, n: int) -> bool:
        """
        检查是否连续N次交易都盈利

        Args:
            n: 连续次数

        Returns:
            True: 最近连续N次都是盈利
            False: 不足N次，或有亏损
        """
        if len(self.trades) < n:
            return False

        # 获取最近N次交易
        recent_trades = self.trades[-n:]

        # 检查是否都盈利（actual_return > 0）
        all_profits = all(trade.get('actual_return', 0.0) > 0 for trade in recent_trades)

        if all_profits:
            returns = [trade.get('actual_return', 0.0) for trade in recent_trades]
            logger.info(f"📈 [连续盈利] 检测到连续{n}次盈利: {[f'{r*100:.2f}%' for r in returns]}")

        return all_profits


    def _get_position_limit_by_trade_history(self) -> float:
        """根据最近交易结果动态调整 lead_z 的过滤阈值。

        状态机（用来做简易的“风控/降频”）：
          - NORMAL: 默认阈值较宽（更容易开仓）
          - DEFENSIVE1: 小幅收紧
          - DEFENSIVE: 大幅收紧（连续亏损后）

        触发逻辑：
          - 若上一笔交易触发 stop_loss -> 进入 DEFENSIVE1
          - 若连续亏损 >= 3 -> 进入 DEFENSIVE
          - 若在 DEFENSIVE/DEFENSIVE1 且连续盈利 >= 2 -> 回到 NORMAL

        返回值：
          - NORMAL     -> -0.4
          - DEFENSIVE1 ->  0.0
          - DEFENSIVE  ->  0.3
        """
        # 默认值
        base_normal = -0.4
        base_defensive1 = 0.0
        base_defensive = 0.3

        try:
            trade_history = getattr(self, "trade_history", []) or []
            if len(trade_history) == 0:
                return base_normal

            last_trade = trade_history[-1]
            last_reason = str(last_trade.get("exit_reason", "")).lower()

            # 连亏/连盈统计（从末尾往前数）
            consecutive_losses = 0
            consecutive_profits = 0
            for t in reversed(trade_history):
                ret = float(t.get("return_pct", 0.0))
                if ret <= 0:
                    consecutive_losses += 1
                    consecutive_profits = 0
                else:
                    consecutive_profits += 1
                    consecutive_losses = 0
                if consecutive_losses >= 3 or consecutive_profits >= 2:
                    break

            current_state = getattr(self, "position_control_state", "NORMAL")

            # 状态迁移
            if "stop_loss" in last_reason:
                current_state = "DEFENSIVE1"
            if consecutive_losses >= 3:
                current_state = "DEFENSIVE"
            if current_state in ("DEFENSIVE", "DEFENSIVE1") and consecutive_profits >= 2:
                current_state = "NORMAL"

            self.position_control_state = current_state

            # 阈值输出
            if current_state == "DEFENSIVE":
                limit = base_defensive
            elif current_state == "DEFENSIVE1":
                limit = base_defensive1
            else:
                limit = base_normal

            logger.info(
                f"📊 [仓位控制/lead_z] 状态={current_state}, 连亏={consecutive_losses}, 连赢={consecutive_profits}, z_limit={limit:.2f}"
            )
            return float(limit)
        except Exception as e:
            logger.warning(f"⚠️ 仓位控制状态机异常，使用默认阈值 {base_normal:.2f}: {e}")
            return float(base_normal)


    def _position_pct_by_lead(self, lead_z: float, trade_date: Optional[datetime.date] = None) -> float:
        """根据 lead_z（topK 内相对优势的鲁棒 z-score）给出仓位比例。

        lead_z 计算（使用过去 rolling_window 的历史 lead_raw，不含当天，避免 look-ahead）：
            lead_raw = top1_score - mean(topK_scores)
            med = median(hist)
            mad = median(|hist - med|)
            lead_z = (lead_raw - med) / (1.4826 * mad + eps)

        仓位/过滤规则（与历史交易结果联动）：
            - 通过历史交易结果得到动态阈值 z_limit（默认 -0.4；防御模式更严格为 0.0）
            - 若 lead_z < z_limit：跳过（0%）
            - 否则：全仓（100%）

        说明：
            你之前用 lead_raw 的固定阈值容易受模型尺度漂移影响；lead_z 通过滚动 robust 标准化减敏。
        """
        try:
            z = float(lead_z)
        except Exception:
            return 0.0

        z_limit = float(self._get_position_limit_by_trade_history())

        # if z < z_limit:
        #     return 0.0
        return 1.0

    def _robust_mad(self, values: list[float]) -> float:
        """MAD = median(|x - median(x)|). values 需为非空。"""
        if not values:
            return 0.0
        import numpy as np
        arr = np.asarray(values, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return mad

    def _calc_lead_z(self, lead_raw: float, rolling_window: int = 60, eps: float = 1e-9) -> float:
        """用历史 lead_raw（不含当前 lead_raw）计算鲁棒 z-score。"""
        try:
            x = float(lead_raw)
        except Exception:
            return 0.0

        hist = getattr(self, "_lead_history", [])
        if not hist:
            return 0.0

        # 只取最近 rolling_window
        h = [float(v) for v in hist[-int(rolling_window):] if v is not None]
        if len(h) < 10:
            return 0.0

        import numpy as np
        arr = np.asarray(h, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        denom = 1.4826 * mad + float(eps)
        return (x - med) / denom

    def _update_lead_history(self, lead_raw: float, rolling_window: int = 60) -> None:
        """更新 lead_raw 历史（滚动窗口）。"""
        try:
            x = float(lead_raw)
        except Exception:
            return
        if not hasattr(self, "_lead_history"):
            self._lead_history = []
        self._lead_history.append(x)
        # trim
        if len(self._lead_history) > int(rolling_window) * 2:
            # 保留更长一点，避免频繁 slice
            self._lead_history = self._lead_history[-int(rolling_window) * 2:]

    def execute_buy(self, stock_code: str, buy_date: datetime.date,
                    predicted_return: float, trading_dates: List[datetime.date], position_pct: float) -> Optional[Position]:
        """执行买入操作"""
        logger.info(f"🛒 准备买入 {stock_code}, 预测收益: {predicted_return:.4f}")

        stock_data = self.get_stock_data(stock_code, buy_date)
        if stock_data.empty:
            logger.warning(f"⚠️ {stock_code} 买入日无分钟数据，跳过买入")
            return None

        logger.info(f"📊 {stock_code} 获取到{len(stock_data)}条5分钟数据")

        # 使用开盘价买入
        buy_price = stock_data.iloc[0]['open']
        raw_datetime = stock_data.iloc[0]['datetime']

        # 处理不同的datetime格式
        if isinstance(raw_datetime, str):
            buy_datetime = datetime.fromisoformat(raw_datetime)
        elif hasattr(raw_datetime, 'to_pydatetime'):  # pandas Timestamp
            buy_datetime = raw_datetime.to_pydatetime()
        elif isinstance(raw_datetime, datetime):
            buy_datetime = raw_datetime
        else:
            logger.warning(f"⚠️ 未知的datetime格式: {type(raw_datetime)}, 使用当前时间")
            buy_datetime = datetime.now()

        logger.info(f"💰 {stock_code} 开盘价: {buy_price:.4f}, 开盘时间: {buy_datetime}, 当前现金: {self.cash:.2f}")

        # ✅ 修复: 基于当前可用现金计算买入数量
        # 🆕 新逻辑：所有策略都每天只买入1只股票（指定排名的那只），应该全仓买入

        # 检查现金是否足够购买至少1手
        min_cost = buy_price * 100  # 至少买1手的成本
        if self.cash < min_cost:
            logger.warning(f"⚠️ 现金不足，无法买入{stock_code}: 可用现金={self.cash:.2f}, 最少需要={min_cost:.2f}")
            return None

        # 🆕 关键修复：所有策略都全仓买入
        # stock_count=1: 买入排名第1的股票，全仓
        # stock_count=2: 买入排名第2的股票，全仓
        # stock_count=3: 买入排名第3的股票，全仓
        # stock_count=5: 买入排名第5的股票，全仓
        # 每个策略每天只买入1只股票，所以应该用全部可用现金
        position_pct = float(position_pct)
        position_pct = max(0.0, min(1.0, position_pct))

        asset_cash = self.cash  # 或你当前现金变量
        use_cash = asset_cash * position_pct
        position_amount = use_cash
        # position_amount = self.cash
        logger.info(f"📊 Top-{self.config.stock_count}策略: 全仓买入排名第{self.config.stock_count}的股票，使用全部现金 {self.cash:.2f}")
        quantity = int(position_amount / buy_price / 100) * 100  # 整手
        actual_cost = buy_price * quantity

        # 再次确保不会超支
        if actual_cost > self.cash:
            # 如果计算出的成本超过可用现金，减少买入数量
            quantity = int(self.cash / buy_price / 100) * 100
            actual_cost = buy_price * quantity
            if actual_cost > self.cash:  # 如果还是超支，调整到能买起的数量
                quantity = max(0, int(self.cash / buy_price / 100) * 100 - 100)
                actual_cost = buy_price * quantity

        # 最终验证
        if actual_cost > self.cash:
            logger.error(f"❌ 买入成本超支: 成本={actual_cost:.2f}, 可用现金={self.cash:.2f}")
            return None

        logger.info(f"💵 资金使用情况: 分配金额={position_amount:.2f}, 买入数量={quantity}股, 实际花费={actual_cost:.2f}")
        logger.info(f"💰 买入后现金: {self.cash - actual_cost:.2f} (买入前: {self.cash:.2f})")

        # 计算持仓到期日 - 使用交易日而非自然日
        holding_end_date = self.add_trading_days(buy_date, self.config.holding_days - 1, trading_dates)

        # 如果计算失败（买入日期不在交易列表中），回退到自然日计算
        if holding_end_date is None:
            logger.warning(f"⚠️ 无法通过交易日计算到期日，使用自然日计算")
            holding_end_date = buy_date + timedelta(days=self.config.holding_days - 1)

        # 计算实际持有的交易日数量（用于日志）
        try:
            buy_index = trading_dates.index(buy_date)
            end_index = trading_dates.index(holding_end_date)
            actual_trading_days = end_index - buy_index + 1
            logger.info(f"📅 持仓计划: 买入日={buy_date} (交易日索引#{buy_index}), "
                       f"到期日={holding_end_date} (交易日索引#{end_index}), "
                       f"持有{actual_trading_days}个交易日 (配置: {self.config.holding_days}天)")
        except (ValueError, Exception):
            logger.info(f"📅 持仓计划: 买入日={buy_date}, 到期日={holding_end_date} (配置: {self.config.holding_days}天)")

        # 🆕 获取T+1日（买入日）的最低价（用于T+1最低价跌破策略）
        t1_day_lowest_price = 0.0
        if self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            t1_day_lowest_price = self._get_daily_low(stock_code, buy_date)
            if t1_day_lowest_price is None:
                logger.warning(f"⚠️ 无法获取{stock_code}在买入日{buy_date}的最低价，使用0.0")
                t1_day_lowest_price = 0.0

        # 创建持仓记录
        position = Position(
            stock_code=stock_code,
            buy_date=buy_date,
            buy_datetime=buy_datetime,
            buy_price=buy_price,
            highest_price_since_buy=buy_price,
            quantity=quantity,
            predicted_return=predicted_return,
            holding_end_date=holding_end_date,
            t1_day_lowest_price=t1_day_lowest_price  # 🆕 记录T+1日最低价
        )

        self.current_holdings[stock_code] = position
        self.cash -= buy_price * quantity

        # 更新买入周期状态 (关键修复)
        self.last_buy_date = buy_date
        self.current_buy_cycle_date = buy_date
        self.is_holding = True
        self.current_holding_end_date = position.holding_end_date

        # 🆕 设置本轮已尝试买入的标志（用于判断是否需要重置）
        self.has_attempted_buy_this_cycle = True

        # 修复T+1买入bug：买入成功后清除动态T日，使用正式的last_position_end_date
        self.dynamic_t_date = None  # 清除动态T日，买入后使用正式逻辑

        logger.info(f"✅ 买入成功: {stock_code}, 价格: {buy_price:.4f}, 数量: {quantity}, 时间: {buy_datetime}, 剩余现金: {self.cash:.2f}")
        logger.info(f"📅 买入周期状态: 最后买入日={self.last_buy_date}, 预计卖出日={self.current_holding_end_date}")

        return position

    def _should_reset_t_date(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> bool:
        """
        判断是否应该重置T日（空仓优化）

        重置条件：
        1. 当前没有持仓（is_holding = False）
        2. 距离上次T日更新已经过了一段时间（避免频繁重置）
        3. 不是在卖出当天（避免重复重置）

        Args:
            trade_date: 当前交易日
            trading_dates: 完整的交易日列表

        Returns:
            bool: 是否应该重置T日
        """
        # 如果没有current_t_date，需要初始化
        if not hasattr(self, 'current_t_date') or self.current_t_date is None:
            return True

        # 🆕 关键修复：如果T日就是当前交易日，说明刚在卖出时重置过，不需要再次重置
        if self.current_t_date == trade_date:
            logger.debug(f"📅 [重置判断] T日={trade_date}，刚在卖出时重置过 → 不重置")
            return False

        # 获取当前交易日的索引
        try:
            current_index = trading_dates.index(trade_date)
        except ValueError:
            return False

        # 获取T日的索引
        try:
            t_index = trading_dates.index(self.current_t_date)
        except ValueError:
            # T日不在交易日列表中，需要重置
            return True

        days_since_t = current_index - t_index

        # 🆕 核心判断逻辑：
        # 如果距离T日已经超过1天（已过T+1买入窗口），且没有持仓
        # 说明上一轮的买入机会已经完全错过，应该立即重置
        if days_since_t > 1 and not self.is_holding:
            logger.info(f"📅 [重置判断] 距T日{days_since_t}天，已过T+1窗口，且无持仓 → 应该重置")
            return True

        # 如果正好是T+1日，但在空仓状态（可能之前全部跳过）
        # 不需要重置，应该尝试买入
        if days_since_t == 1 and not self.is_holding:
            logger.info(f"📅 [重置判断] T+1日空仓，应该尝试买入 → 不重置")
            # 🆕 关键修复：不在判断时设置标志，而是在真正尝试买入时设置
            # 这样可以避免"判断为T+1日但实际未买入"的误判
            return False

        return False

    def _get_previous_trading_day(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> Optional[datetime.date]:
        """获取前一个交易日"""
        try:
            current_index = trading_dates.index(trade_date)
            if current_index > 0:
                return trading_dates[current_index - 1]
        except ValueError:
            pass
        return None

    def _should_attempt_buy(self, trade_date: datetime.date, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """检查当前交易日是否应该尝试买入 - 实现正确的T+1逻辑"""
        try:
            logger.debug(f"🤔 判断是否买入: trade_date={trade_date}, is_holding={self.is_holding}, "
                        f"last_buy_date={self.last_buy_date}, current_t_date={getattr(self, 'current_t_date', None)}")

            # 如果当前有持仓，不能买入
            if self.is_holding:
                logger.debug(f"🚫 当前有持仓，不允许买入")
                return {'should_buy': False, 'reason': f'持仓期间 (预计{self.current_holding_end_date}到期)', 'prediction_date': None}

            # 首次运行：设置current_t_date为交易日的前一日，允许买入
            # 📌 优先级最高：首次运行逻辑必须在空仓重置逻辑之前
            if not self.last_buy_date:
                # 为了找到T日，我们需要找当前交易日之前的最近一个预测数据日期
                available_dates = sorted(self.prediction_data['date'].unique())
                earlier_dates = [d for d in available_dates if d < trade_date]

                if not earlier_dates:
                    # 第一个交易日，无法设置T日，但需要初始化current_t_date
                    # 将前一个交易日作为T日（如果没有则使用当前日）
                    prev_day = self._get_previous_trading_day(trade_date, trading_dates)
                    self.current_t_date = prev_day if prev_day else trade_date
                    logger.info(f"🆕 首个交易日: 设置T日={self.current_t_date}")
                    return {'should_buy': False, 'reason': f'{trade_date} 是第一个交易日，无前一日预测数据', 'prediction_date': None}

                # 设置T日为最近的有预测数据的日期
                self.current_t_date = max(earlier_dates)
                logger.info(f"🆕 首次运行: 设置T日={self.current_t_date}, 允许在{trade_date}买入")

                # 检查是否是T+1日
                current_index = trading_dates.index(trade_date)
                t_index = trading_dates.index(self.current_t_date)
                days_since_t = current_index - t_index

                if days_since_t == 1:
                    # 正好是T+1日，可以买入
                    return {'should_buy': True, 'reason': '首次买入', 'prediction_date': self.current_t_date}
                else:
                    # 还不是T+1日
                    logger.info(f"⏰ 首次运行，还未到T+1买入日 (还需{1-days_since_t}日)")
                    return {'should_buy': False, 'reason': f'首次运行，未到T+1日 (还需{1-days_since_t}日)', 'prediction_date': None}

            # 🆕 空仓时立即重置T日，避免日期浪费（只在非首次运行时执行）
            if not self.is_holding and self.last_buy_date:  # ⭐ 关键：只在last_buy_date不为空时才重置
                if self._should_reset_t_date(trade_date, trading_dates):
                    # 将前一日设为新的T日，当日为T+1日
                    prev_day = self._get_previous_trading_day(trade_date, trading_dates)
                    if prev_day:
                        logger.info(f"🔄 [空仓重置] T日重置: {self.current_t_date} → {prev_day}")
                        logger.info(f"📅 [空仓重置] 当前日{trade_date}作为T+1日，允许立即买入")

                        self.current_t_date = prev_day
                        self.last_position_end_date = prev_day  # 记录重置日期

                        # ✅ 修复bug：不要直接返回False
                        # 此时 days_since_t = trade_date - prev_day = 1，正好是T+1日
                        # 让代码继续执行到line 969-981的T+1判断逻辑，允许立即买入
                        logger.info(f"✅ [空仓重置] T日已重置为{prev_day}，当前日{trade_date}是T+1日，继续执行买入判断")
                        # 不返回，继续执行后续的T+1判断逻辑

            # 正常运行：按照用户指定的正确逻辑
            # "当前轮次的T+holding_days日，在T+holding_days日结束时，把该日当作下一轮的T日"

            try:
                # 获取当前交易日在交易日列表中的索引
                current_index = trading_dates.index(trade_date)

                if not hasattr(self, 'current_t_date') or self.current_t_date is None:
                    # 如果没有current_t_date，说明需要重新设置
                    # T日应该是当前交易日的上一交易日
                    if current_index > 0:
                        self.current_t_date = trading_dates[current_index - 1]
                        logger.info(f"🔄 重新设置T日: {self.current_t_date}")
                    else:
                        return {'should_buy': False, 'reason': '无法确定T日', 'prediction_date': None}

                # 检查当前交易日是否是T+1日
                t_index = trading_dates.index(self.current_t_date)
                days_since_t = current_index - t_index

                logger.debug(f"📅 T日逻辑: T日={self.current_t_date}, 当前={trade_date}, 距离T日={days_since_t}日")

                if days_since_t == 1:
                    # 正好是T+1日，可以买入
                    # 使用T日的预测数据
                    available_dates = sorted(self.prediction_data['date'].unique())
                    prediction_dates = [d for d in available_dates if d <= self.current_t_date]

                    if prediction_dates:
                        prediction_date = max(prediction_dates)
                    else:
                        prediction_date = self.current_t_date

                    logger.info(f"✅ T+1买入: T日={self.current_t_date}, T+1日={trade_date}, 使用预测日={prediction_date}")
                    return {'should_buy': True, 'reason': 'T+1买入', 'prediction_date': prediction_date}

                elif days_since_t < 1:
                    # 还没到T+1日
                    logger.debug(f"⏰ 还未到T+1买入日: {trade_date}（T日：{self.current_t_date}，还需{1-days_since_t}日）")
                    return {'should_buy': False, 'reason': f'未到T+1日 (T日={self.current_t_date}, 当前={trade_date}, 还需{1-days_since_t}日)', 'prediction_date': None}

                else:
                    # days_since_t > 1，已经过了T+1买入窗口
                    # 根据用户逻辑，此时应该等待T+holding_days日结束，然后将该日设为下一轮的T日
                    days_past_t1 = days_since_t - 1  # 已经过了T+1日的天数
                    days_in_holding_period = days_past_t1 + 1  # T+1日到当前日的总天数

                    logger.info(f"📅 已过T+1买入窗口: T日={self.current_t_date}, 当前={trade_date}, 已过T+1({days_past_t1}日)")

                    if days_in_holding_period >= self.config.holding_days:
                        # 已过持仓周期，按照用户逻辑：将当前日设为下一轮的T日
                        logger.info(f"🔄 持仓周期已过，更新T日: {self.current_t_date} → {trade_date}")
                        self.current_t_date = trade_date
                        return {'should_buy': False, 'reason': f'持仓周期已过，更新T日为{trade_date}', 'prediction_date': None}
                    else:
                        # 还在持仓周期内，不能买入
                        logger.debug(f"⏳ 持仓期间: 已过{days_in_holding_period}/{self.config.holding_days}日")
                        return {'should_buy': False, 'reason': f'已过T+1买入窗口，持仓中({days_in_holding_period}/{self.config.holding_days}日)', 'prediction_date': None}

            except ValueError as e:
                logger.warning(f"⚠️ 交易日索引错误: {e}")
                return {'should_buy': False, 'reason': '日期索引错误', 'prediction_date': None}

        except Exception as e:
            logger.error(f"❌ 买入判断逻辑异常: {e}")
            # 异常情况下默认不允许买入，避免错误状态
            return {'should_buy': False, 'reason': f'逻辑异常: {e}', 'prediction_date': None}

    def execute_sell(self, position: Position, sell_date: datetime.date,
                     exit_reason: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        """执行卖出操作"""
        if exit_price is None:
            stock_data = self.get_stock_data(position.stock_code, sell_date)
            if stock_data.empty:
                # 使用买入价止损卖出
                exit_price = position.buy_price * 0.98  # 假设2%止损
                sell_datetime = datetime.combine(sell_date, datetime.min.time())
                logger.warning(f"⚠️ {position.stock_code} 无卖出数据，使用止损价: {exit_price:.4f}")
            else:
                exit_price = stock_data.iloc[-1]['close']
                raw_sell_datetime = stock_data.iloc[-1]['datetime']

                # 处理不同的datetime格式
                if isinstance(raw_sell_datetime, str):
                    sell_datetime = datetime.fromisoformat(raw_sell_datetime)
                elif hasattr(raw_sell_datetime, 'to_pydatetime'):  # pandas Timestamp
                    sell_datetime = raw_sell_datetime.to_pydatetime()
                elif isinstance(raw_sell_datetime, datetime):
                    sell_datetime = raw_sell_datetime
                else:
                    logger.warning(f"⚠️ 卖出时未知的datetime格式: {type(raw_sell_datetime)}, 使用当前时间")
                    sell_datetime = datetime.now()
        else:
            sell_datetime = datetime.combine(sell_date, datetime.min.time())

        # 计算实际收益
        actual_return = (exit_price - position.buy_price) / position.buy_price
        position.actual_return = actual_return
        position.sell_price = exit_price
        position.sell_datetime = sell_datetime
        position.exit_reason = exit_reason

        # 更新现金
        sell_value = exit_price * position.quantity
        self.cash += sell_value

        # 创建交易记录
        trade_record = {
            'stock_code': position.stock_code,
            'buy_datetime': position.buy_datetime,
            'sell_datetime': sell_datetime,
            'buy_price': position.buy_price,
            'sell_price': exit_price,
            'quantity': position.quantity,
            'predicted_return': position.predicted_return,
            'actual_return': actual_return,
            'profit_loss': (exit_price - position.buy_price) * position.quantity,
            'profit_loss_pct': actual_return * 100,
            'holding_days': (sell_date - position.buy_date).days,
            'entry_reason': position.entry_reason,
            'exit_reason': exit_reason,
            'strategy_type': f"Top-{self.config.stock_count}"
        }

        self.trades.append(trade_record)
        self.trade_history = self.trades

        # 🧪 卖出时仅打印成交结果；Regime状态机不再依赖成交笔数，而依赖每日预计算诊断
        self.latest_regime_state = self._compute_regime_state(None)
        logger.info(
            f"🧪 卖出完成: completion={sell_date} selector={getattr(position, 'regime_selector', 'legacy_rank_top1')} "
            f"state={getattr(position, 'regime_state', 'UNKNOWN')} actual_ret={actual_return:.4f} "
            f"latest_state={self.latest_regime_state.get('state')} latest_roll_ret={self.latest_regime_state.get('rolling_return')} "
            f"latest_roll_gap={self.latest_regime_state.get('rolling_gap')} latest_roll_margin={self.latest_regime_state.get('rolling_score_margin')}"
        )

        del self.current_holdings[position.stock_code]

        # 检查是否所有持仓都已卖出，如果是则重置买入周期状态
        if not self.current_holdings:
            logger.info("🔄 所有持仓已清空，重置买入周期状态")
            self.is_holding = False
            self.current_holding_end_date = None

            # 🆕 关键修复：卖出后立即重置T日，避免日期浪费
            # 将卖出日本身作为新的T日，下一个交易日就是新的T+1日
            # 这样次日可以立即尝试买入，不需要等待
            self.current_t_date = sell_date
            logger.info(f"📅 [卖出重置] T日重置: {sell_date} (卖出日本身)")
            logger.info(f"📅 [卖出重置] 下个交易日将是T+1日，可以立即买入")

            # 记录持仓结束日期
            self.last_position_end_date = sell_date
            # 注意：保留last_buy_date用于下次买入周期判断

        logger.info(f"💰 卖出: {position.stock_code}, 价格: {exit_price:.4f}, "
                   f"收益: {actual_return:.4f} ({actual_return*100:.2f}%), "
                   f"原因: {exit_reason}")
        if not self.current_holdings:
            logger.info(f"📊 当前状态: 无持仓, 现金: {self.cash:.2f}, 最后买入日: {self.last_buy_date}")

        return trade_record

    def run_backtest(self, trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """运行完整的回测"""
        logger.info(f"🚀 开始回测 Top-{self.config.stock_count} 策略")
        logger.info(f"   时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
        logger.info(f"   总交易日: {len(trading_dates)}")

        # 确保trading_dates按时间顺序排列
        trading_dates = sorted(trading_dates)
        logger.info(f"✅ 交易日期已按时间排序")

        # 预分析候选股票
        self.pre_analyze_required_stocks()

        for i, trade_date in enumerate(trading_dates):
            # 保存交易日期，用于后续绘图
            self.trading_dates_list.append(trade_date)

            # 🆕 重置尝试标志（每个交易日开始时重置）
            if not self.is_holding:
                self.has_attempted_buy_this_cycle = False

            logger.info(f"📅 交易日 {i+1}/{len(trading_dates)}: {trade_date}")
            logger.info(f"💼 当前持仓: {len(self.current_holdings)}只, 现金: {self.cash:.2f}")

            # 风控检查（优先执行 - 在到期卖出之前检查）
            risk_exited_positions = []
            positions_to_remove = []

            for stock_code, position in list(self.current_holdings.items()):
                # 使用交易日期列表进行精确的风控检查
                logger.debug(f"🔍 检查{position.stock_code}在{trade_date}的风控状态 (买入日: {position.buy_date})")
                risk_check = self.check_position_risk(position, trade_date, trading_dates)
                if risk_check['should_exit']:
                    logger.info(f"🛡️ 风控触发: {position.stock_code}, 原因: {risk_check['reason']}")
                    trade = self.execute_sell(
                        position, trade_date,
                        f"风控: {risk_check['reason']}",
                        risk_check.get('exit_price')
                    )
                    if trade:
                        risk_exited_positions.append(trade)
                        positions_to_remove.append(stock_code)

            # 检查持仓到期（在风控检查之后执行）
            expired_positions = []
            for stock_code, position in self.current_holdings.items():
                if stock_code not in positions_to_remove and position.holding_end_date and trade_date >= position.holding_end_date:
                    expired_positions.append(position)

            # 到期卖出
            if expired_positions:
                logger.info(f"⏰ {len(expired_positions)}只持仓到期")
                for position in expired_positions:
                    self.execute_sell(position, trade_date, "到期")

            # 空仓选股 - 检查买入时机
            if not self.current_holdings and i < len(trading_dates) - 1:
                # 检查是否是正确的买入时机
                should_buy = self._should_attempt_buy(trade_date, trading_dates)

                if not should_buy['should_buy']:
                    logger.info(f"⏸️ {should_buy['reason']}, 跳过买入")
                    continue

                # 🆕 关键修复：只在真正进入买入流程时设置标志
                # 避免在判断时设置，导致"判断为可以买但实际未买"的误判
                self.has_attempted_buy_this_cycle = True

                # T+1交易逻辑：使用T日预测数据在T+1日买入
                prediction_date = should_buy['prediction_date']
                logger.info(f"🔍 T+1交易逻辑: T+1日={trade_date}, T日={prediction_date}，开始选股")

                # 获取预测数据 - 对Top-1策略使用“按日期保存的过滤候选全集”，以便根据已完成交易动态切换策略状态
                prediction_date_str = prediction_date.strftime('%Y-%m-%d')
                filtered_candidates = self.filtered_candidates_by_date.get(prediction_date_str, None)

                if filtered_candidates is None or len(filtered_candidates) == 0:
                    top_k_stocks = self.required_stocks_by_date.get(prediction_date_str, [])
                    if not top_k_stocks:
                        logger.warning(f"⚠️ {prediction_date} 无预分析候选股票，跳过买入")
                        continue
                    daily_predictions = self.prediction_data[(self.prediction_data['date'] == prediction_date) & (self.prediction_data['stock_code'].isin(top_k_stocks))].copy()
                else:
                    daily_predictions = filtered_candidates.copy()

                if daily_predictions.empty:
                    logger.warning(f"⚠️ {prediction_date} 候选股票无预测数据，跳过")
                    continue

                logger.info(f"📊 {prediction_date} 过滤候选全集: {len(daily_predictions)}只股票")
                selected_stocks = daily_predictions.sort_values('predicted_return', ascending=False).reset_index(drop=True)

                logger.info(f"🎆 使用过滤候选全集进行买入检查(展示前5只rank):")
                for _, stock in selected_stocks.head(5).iterrows():
                    logger.info(f"   📈 {stock['stock_code']}: rank={stock['predicted_return']:.4f}, p_cls={float(stock.get('p_cls', np.nan)) if 'p_cls' in stock.index else np.nan:.4f}")

                logger.info(f"🔄 开始对Top-{self.config.stock_count}策略执行买入逻辑...")

                target_rank = self.config.stock_count  # 目标排名
                decision_meta = None

                if target_rank == 1 and getattr(self.config, 'regime_enabled', True):
                    decision_meta = self._select_top1_with_regime(selected_stocks, prediction_date)
                    regime_metrics = decision_meta.get('regime_metrics', {}) if isinstance(decision_meta, dict) else {}
                    logger.info(
                        f"🧭 Regime状态[{prediction_date}]: state={decision_meta.get('regime_state')} history={regime_metrics.get('history_count')} "
                        f"r10_cand={regime_metrics.get('rolling_candidate_alpha')} r10_rerank={regime_metrics.get('rolling_rerank_alpha')} "
                        f"r20_margin={regime_metrics.get('rolling_score_margin')} fast_recovery={regime_metrics.get('fast_recovery')} retrain={regime_metrics.get('retrain_flag')} stop_enforced={regime_metrics.get('enforce_stop')}"
                    )
                    if decision_meta.get('skip', False):
                        logger.warning(f"⛔ {prediction_date} 状态机决策为跳过买入: {decision_meta.get('reason')}")
                        continue
                    target_stock = decision_meta.get('target_stock')
                    if target_stock is None:
                        logger.warning(f"⚠️ 状态机未返回目标股票，跳过买入")
                        continue
                    target_stocks_for_today = [target_stock['stock_code']]
                    logger.info(
                        f"🎯 Top1状态机选股: selector={decision_meta.get('selector')} state={decision_meta.get('regime_state')} "
                        f"stock={target_stock['stock_code']} rank={float(target_stock.get('predicted_return', np.nan)):.4f} "
                        f"p_cls={float(target_stock.get('p_cls', np.nan)) if 'p_cls' in target_stock.index else np.nan:.4f} "
                        f"base_gap={decision_meta.get('base_conditional_gap')} base_margin={decision_meta.get('base_score_margin')}"
                    )
                else:
                    if len(selected_stocks) < target_rank:
                        logger.warning(f"⚠️ 候选股票数{len(selected_stocks)}不足目标排名{target_rank}，跳过买入")
                        continue
                    target_stock = selected_stocks.iloc[target_rank - 1]
                    target_stocks_for_today = [target_stock['stock_code']]

                logger.info(f"📦 预加载{trade_date}和{prediction_date}的目标股票5分钟数据: 排名第{target_rank}的{target_stock['stock_code']}")

                # 批量预加载T日和T+1日的股票数据
                self._preload_stocks_data([prediction_date, trade_date], target_stocks_for_today)

                # 🆕 修改：只买入目标排名的那一只股票
                logger.info(f"🎯 处理Top-{self.config.stock_count}策略: 买入排名第{target_rank}的股票")
                stock = target_stock
                stock_code = stock['stock_code']

                # 正确的T+1风控检查：使用T日收盘价 vs T+1日开盘价
                logger.info(f"📅 {stock_code} T+1买入逻辑: T日={prediction_date}(预测日), T+1日={trade_date}(执行日)")

                # 获取T日（预测日）的收盘价用于风控（现在使用缓存数据）
                logger.info(f"🔍 获取{stock_code} T日({prediction_date})数据用于风控检查...")
                prediction_day_data = self.get_stock_data(stock_code, prediction_date)

                if not prediction_day_data.empty:
                    prev_close = prediction_day_data.iloc[-1]['close']
                    logger.info(f"📊 {stock_code} T日({prediction_date})收盘价: {prev_close:.2f}")
                    logger.info(f"📊 {stock_code} T日数据条数: {len(prediction_day_data)}")
                    logger.info(f"📊 {stock_code} T日最后一条数据时间: {prediction_day_data.iloc[-1]['datetime']}")
                else:
                    prev_close = None
                    logger.warning(f"⚠️ {stock_code} T日({prediction_date})无数据")

                # 获取T+1日（执行日）的开盘价用于买入
                logger.info(f"🔍 获取{stock_code} T+1日({trade_date})数据用于买入...")
                execution_day_data = self.get_stock_data(stock_code, trade_date)

                # 初始化buy_data变量
                buy_data = execution_day_data

                if not execution_day_data.empty:
                    buy_open = execution_day_data.iloc[0]['open']
                    logger.info(f"📊 {stock_code} T+1日({trade_date})开盘价: {buy_open:.2f}")
                    logger.info(f"📊 {stock_code} T+1日数据条数: {len(execution_day_data)}")
                    logger.info(f"📊 {stock_code} T+1日第一条数据时间: {execution_day_data.iloc[0]['datetime']}")
                else:
                    buy_open = None

                if not buy_data.empty:
                    # 重新获取buy_open（确保使用正确的值）
                    buy_open = buy_data.iloc[0]['open']
                    prev_close_str = f"{prev_close:.4f}" if prev_close is not None else "N/A"
                    logger.info(f"🔍 {stock_code} 买入检查: 开盘价={buy_open:.4f}, 前收盘={prev_close_str}")

                    entry_check = self.check_entry_signal(
                        stock_code, trade_date, prev_close, buy_open
                    )

                    logger.info(f"🚦 {stock_code} 买入信号: {entry_check['can_buy']}, 原因: {entry_check['reason']}")

                    if entry_check['can_buy']:
                        scores = pd.to_numeric(selected_stocks["predicted_return"], errors="coerce").to_numpy(dtype=float)
                        scores = scores[np.isfinite(scores)]  # 去掉 NaN/inf

                        if scores.size == 0:
                            topk_mean = float(stock["predicted_return"])
                        else:
                            topk_mean = float(scores.mean())
                        lead_raw = float(stock['predicted_return']) - float(topk_mean)
                        lead_z = self._calc_lead_z(lead_raw, rolling_window=60, eps=1e-9)
                        # 更新历史（在做完当日决策后再纳入分布；这里更新不影响 lead_z（已用过去数据）
                        self._update_lead_history(lead_raw, rolling_window=60)
                        position_pct = self._position_pct_by_lead(lead_z, prediction_date)

                        if position_pct <= 0.0:
                            logger.info(f"🚫 {stock_code} 触发预测强度过滤: lead_z={lead_z:.2f} (limit={self._get_position_limit_by_trade_history():.2f})，跳过买入")
                            continue
                        else:
                            logger.info(f"✅ {stock_code} 预测强度通过: lead_z={lead_z:.2f} -> 仓位 {position_pct:.0%}")

                        position = self.execute_buy(
                            stock_code, trade_date, stock['predicted_return'], trading_dates, position_pct=position_pct  # trade_date 是执行日
                        )
                        if position:
                            # 🆕 把决策时的状态机/诊断元数据挂到持仓上，待平仓后写入成熟历史
                            if decision_meta:
                                position.prediction_date = prediction_date
                                position.regime_state = str(decision_meta.get('regime_state', 'UNKNOWN'))
                                position.regime_selector = str(decision_meta.get('selector', 'legacy_rank_top1'))
                                position.regime_base_conditional_gap = decision_meta.get('base_conditional_gap')
                                position.regime_base_score_margin = decision_meta.get('base_score_margin')
                                rm = decision_meta.get('regime_metrics', {}) or {}
                                position.regime_rolling_return = rm.get('rolling_return')
                                position.regime_rolling_gap = rm.get('rolling_gap')
                                position.regime_rolling_score_margin = rm.get('rolling_score_margin')
                                self.regime_decision_log.append({
                                    'prediction_date': prediction_date,
                                    'trade_date': trade_date,
                                    'stock_code': stock_code,
                                    'selector': position.regime_selector,
                                    'regime_state': position.regime_state,
                                    'base_conditional_gap': position.regime_base_conditional_gap,
                                    'base_score_margin': position.regime_base_score_margin,
                                    'rolling_return': position.regime_rolling_return,
                                    'rolling_gap': position.regime_rolling_gap,
                                    'rolling_score_margin': position.regime_rolling_score_margin,
                                })
                            self.positions.append(position)
                            logger.info(f"✅ {stock_code} 买入成功，添加到持仓列表")
                        else:
                            logger.warning(f"❌ {stock_code} 买入失败，execute_buy返回None")
                    else:
                        logger.info(f"🚫 {stock_code} 买入被拒绝: {entry_check['reason']}")
                        # 保存被跳过的买入尝试，用于后续分析
                        skipped_record = {
                            'stock_code': stock_code,
                            'prediction_date': prediction_date,  # T日（预测日）
                            'attempt_buy_date': trade_date,  # T+1日（尝试买入日）
                            'buy_open': buy_open,
                            'prev_close': prev_close,
                            'gap_ratio': entry_check.get('gap_ratio', 0),
                            'skip_reason': entry_check['reason'],
                            'predicted_return': stock['predicted_return'],
                            'skip_category': entry_check.get('skip_category', '其他风控')  # 🆕 从entry_check获取分类
                        }
                        self.skipped_buy_attempts.append(skipped_record)
                else:
                    logger.warning(f"⚠️ {stock_code} {trade_date}无5分钟数据，跳过买入")

                # 🆕 关键修复：买入流程结束后，如果仍然空仓，立即重置T日
                # 避免等到下一个交易日才重置，导致日期浪费
                if not self.current_holdings:
                    logger.info(f"🔄 [买入失败重置] 所有股票买入失败（风控跳过或无数据），立即重置T日")
                    self.current_t_date = trade_date  # 将当前日设为新的T日
                    logger.info(f"📅 [买入失败重置] T日重置: {trade_date} (买入日)")
                    logger.info(f"📅 [买入失败重置] 下个交易日将是T+1日，可以立即买入")

            # 计算当日收益
            daily_pnl = sum([trade['profit_loss'] for trade in self.trades
                             if trade['sell_datetime'].date() == trade_date])
            current_value = self.cash + sum([
                pos.quantity * self.get_stock_data(
                    pos.stock_code, trade_date
                ).iloc[-1]['close']
                if not self.get_stock_data(pos.stock_code, trade_date).empty
                else pos.buy_price
                for pos in self.current_holdings.values()
            ])

            portfolio_value = current_value
            daily_return = (portfolio_value / self.daily_portfolio_values[-1] - 1) if self.daily_portfolio_values else 0

            self.daily_portfolio_values.append(portfolio_value)
            self.daily_returns.append(daily_return)

            # 定期输出进度
            if (i + 1) % 10 == 0 or i == len(trading_dates) - 1:
                logger.info(f"📊 [{i+1}/{len(trading_dates)}] {trade_date}: "
                           f"净值 {portfolio_value:.0f}, "
                           f"当日收益 {daily_return:.4f} "
                           f"({daily_return*100:.2f}%)")

        # 最终统计
        final_value = self.daily_portfolio_values[-1]
        total_return = (final_value / self.config.initial_capital - 1) * 100

        # 交易统计
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['actual_return'] > 0])
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        avg_return = np.mean([t['actual_return'] for t in self.trades]) if self.trades else 0

        # 🆕 生成风控统计信息
        summary_stats = self._generate_risk_control_stats()

        result = {
            'strategy_config': self.config,
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': self._calculate_annual_return(
                trading_dates[0], trading_dates[-1], total_return
            ),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(trading_dates),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'trades': self.trades,
            'daily_portfolio_values': self.daily_portfolio_values,
            'daily_returns': self.daily_returns,
            'trading_dates': self.trading_dates_list,  # 添加交易日期列表
            'skipped_buy_attempts': self.skipped_buy_attempts,  # 添加被跳过的买入尝试
            'final_positions': list(self.current_holdings.values()),
            'summary_stats': summary_stats,  # 🆕 添加风控统计
            'regime_latest_state': self.latest_regime_state,
            'regime_decision_log': self.regime_decision_log,
            'daily_regime_diagnostics': self.daily_regime_diagnostics,
        }

        logger.info(f"🎯 Top-{self.config.stock_count} 策略回测完成:")
        logger.info(f"   总收益: {total_return:.2f}%")
        logger.info(f"   年化收益: {result['annual_return'] * 100:.2f}%")
        logger.info(f"   最大回撤: {result['max_drawdown']:.2f}%")
        logger.info(f"   夏普比率: {result['sharpe_ratio']:.3f}")
        logger.info(f"   交易次数: {total_trades}")
        logger.info(f"   胜率: {win_rate:.1f}%")

        # 添加风控统计输出
        if summary_stats:
            logger.info(f"   📊 风控统计:")

            # 买入风控统计
            gap_down = summary_stats.get('gap_down_3_skipped', 0)
            gap_up = summary_stats.get('gap_up_5_skipped', 0)

            if gap_down > 0 or gap_up > 0:
                logger.info(f"      买入风控 (开盘): {gap_down + gap_up}次")
                if gap_down > 0:
                    logger.info(f"        - 跌超3%: {gap_down}次")
                if gap_up > 0:
                    logger.info(f"        - 涨超5%: {gap_up}次")
            else:
                logger.info(f"      买入风控 (开盘): 无触发")

            # 🆕 MA5开盘风控统计
            ma5_gap_down = summary_stats.get('ma5_gap_down_skipped', 0)
            ma5_gap_up = summary_stats.get('ma5_gap_up_skipped', 0)

            if ma5_gap_down > 0 or ma5_gap_up > 0:
                logger.info(f"      买入风控 (开盘-MA5): {ma5_gap_down + ma5_gap_up}次")
                if ma5_gap_down > 0:
                    logger.info(f"        - 低开超MA5 3%: {ma5_gap_down}次")
                if ma5_gap_up > 0:
                    logger.info(f"        - 高开超MA5 3%: {ma5_gap_up}次")

            # 卖出风控统计
            risk_details = summary_stats.get('risk_control_details', {})
            risk_exits = risk_details.get('total_risk_exits', 0)

            if risk_exits > 0:
                threshold_desc = risk_details.get('stop_loss_threshold_desc', '-5.00%')
                logger.info(f"      卖出风控 (持仓，止损阈值: {threshold_desc}): {risk_exits}次 / {total_trades}笔交易")

                cumulative = risk_details.get('cumulative_decline_exits', 0)
                consecutive = risk_details.get('consecutive_decline_exits', 0)
                ma5_break = risk_details.get('ma5_break_exits', 0)  # 🆕 获取MA5跌破次数
                t1_low_break = risk_details.get('t1_low_break_exits', 0)  # 🆕 获取T+1最低价跌破次数
                profit_bearish = risk_details.get('profit_bearish_exits', 0)  # 🆕 获取盈利状态连续阴线次数

                if cumulative > 0:
                    logger.info(f"        - 累计收益低于阈值: {cumulative}次")
                if consecutive > 0:
                    logger.info(f"        - 连续3根阴线: {consecutive}次")
                if ma5_break > 0:  # 🆕 显示MA5跌破统计
                    logger.info(f"        - 日线MA5跌破: {ma5_break}次")
                if t1_low_break > 0:  # 🆕 显示T+1最低价跌破统计
                    logger.info(f"        - T+1最低价跌破: {t1_low_break}次")
                if profit_bearish > 0:  # 🆕 显示盈利状态连续阴线统计
                    logger.info(f"        - 盈利状态连续阴线: {profit_bearish}次")

                risk_rate = risk_details.get('risk_exit_rate', 0)
                logger.info(f"        风控退出率: {risk_rate:.1%}")
            else:
                threshold_desc = risk_details.get('stop_loss_threshold_desc', '-5.00%')
                logger.info(f"      卖出风控 (持仓，止损阈值: {threshold_desc}): 无触发")

            # 总计风控保护
            total_skipped = summary_stats.get('skipped_positions', 0)
            if total_skipped > 0:
                logger.info(f"   🛡️ 总计风控保护: {total_skipped}次买入被跳过")

        if self.latest_regime_state:
            logger.info(f"   🧭 Regime最新状态: {self.latest_regime_state.get('state')} | history={self.latest_regime_state.get('history_count')} | "
                        f"r10_cand={self.latest_regime_state.get('rolling_candidate_alpha')} | r10_rerank={self.latest_regime_state.get('rolling_rerank_alpha')} | "
                        f"r20_margin={self.latest_regime_state.get('rolling_score_margin')} | fast_recovery={self.latest_regime_state.get('fast_recovery')} | retrain={self.latest_regime_state.get('retrain_flag')} | stop_enforced={self.latest_regime_state.get('enforce_stop')}")
            selector_counts = {}
            for x in self.regime_decision_log:
                selector_counts[x.get('selector','UNKNOWN')] = selector_counts.get(x.get('selector','UNKNOWN'),0)+1
            if selector_counts:
                logger.info(f"   🧭 状态机选择统计: {selector_counts}")

        return result

    def _calculate_annual_return(self, start_date: datetime.date,
                                 end_date: datetime.date, total_return: float) -> float:
        """计算年化收益率"""
        days = (end_date - start_date).days
        if days == 0:
            return 0
        return (1 + total_return / 100) ** (365.25 / days) - 1

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_portfolio_values:
            return 0

        peak = self.daily_portfolio_values[0]
        max_drawdown = 0

        for value in self.daily_portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, trading_dates: List[datetime.date]) -> float:
        """计算夏普比率"""
        if len(self.daily_returns) < 2:
            return 0

        returns_array = np.array(self.daily_returns[1:])  # 跳过第一天的0收益
        days = (trading_dates[-1] - trading_dates[0]).days

        if days == 0:
            return 0

        annual_return = np.mean(returns_array) * 252  # 年化
        annual_volatility = np.std(returns_array) * np.sqrt(252)

        if annual_volatility == 0:
            return 0

        return annual_return / annual_volatility

    def _generate_risk_control_stats(self) -> Dict[str, Any]:
        """
        🆕 生成风控统计信息

        统计各类风控跳过买入的次数

        Returns:
            Dict: 风控统计字典
        """
        # 统计各类风控跳过次数
        stats = {
            'skipped_positions': len(self.skipped_buy_attempts),  # 总跳过次数
            'gap_down_3_skipped': 0,  # 跌空超过3%
            'gap_up_5_skipped': 0,  # 高开超过5%
            # 🆕 添加MA5风控统计
            'ma5_gap_down_skipped': 0,  # 低开超MA5 3%
            'ma5_gap_up_skipped': 0,  # 高开超MA5 3%
            'entry_risk_strategy': self.config.entry_risk_control_strategy,  # 🆕 记录使用的策略
            'optimization_version': 'v3.0_ma5_entry_control',  # 🆕 版本号更新
            'risk_control': {
                'gap_control': True,
                'ma5_control': True  # 🆕 MA5风控启用
            }
        }

        # 按skip_category分类统计
        for skip_record in self.skipped_buy_attempts:
            skip_category = skip_record.get('skip_category', '其他风控')

            if skip_category == '开盘风控':
                gap_ratio = skip_record.get('gap_ratio', 0)
                if gap_ratio < self.config.entry_risk_down_threshold:
                    stats['gap_down_3_skipped'] += 1
                elif gap_ratio > self.config.entry_risk_up_threshold:
                    stats['gap_up_5_skipped'] += 1

            # 🆕 MA5风控统计
            elif skip_category == '开盘MA5风控':
                ma5_gap_ratio = skip_record.get('gap_ratio', 0)
                if ma5_gap_ratio < 0:
                    stats['ma5_gap_down_skipped'] += 1
                elif ma5_gap_ratio > 0:
                    stats['ma5_gap_up_skipped'] += 1

        # 持仓风控详情统计
        risk_exits = [t for t in self.trades if '风控' in t.get('exit_reason', '')]
        normal_exits = [t for t in self.trades if '到期' in t.get('exit_reason', '')]

        # 🔍 调试：打印所有风控退出的exit_reason
        if risk_exits:
            logger.info(f"🔍 [风控统计调试] 所有风控退出的exit_reason:")
            for t in risk_exits:
                logger.info(f"   - {t.get('exit_reason', 'N/A')}")

        # 动态生成阈值描述，用于匹配
        threshold_desc = f"{self.config.stop_loss_threshold:.2%}"
        if self.config.stop_loss_threshold == 0.0:
            threshold_desc = "0% (盈亏平衡)"
        elif self.config.stop_loss_threshold > 0:
            threshold_desc = f"{self.config.stop_loss_threshold:.2%} (要求正收益)"

        # 统计各类风控退出次数（使用动态阈值描述）
        cumulative_decline_exits = len([t for t in risk_exits if f'累计收益低于{threshold_desc}' in t.get('exit_reason', '')])
        consecutive_decline_exits = len([t for t in risk_exits if '连续3根阴线' in t.get('exit_reason', '')])
        # 🆕 统计MA5跌破退出次数
        ma5_break_exits = len([t for t in risk_exits if '日线MA5' in t.get('exit_reason', '')])
        # 🆕 统计T+1最低价跌破退出次数
        t1_low_break_exits = len([t for t in risk_exits if 'T+1最低价' in t.get('exit_reason', '')])
        # 🆕 统计盈利状态连续阴线退出次数
        profit_bearish_exits = len([t for t in risk_exits if '盈利状态下连续' in t.get('exit_reason', '') and '阴线' in t.get('exit_reason', '')])

        # 🔍 调试：打印统计结果
        logger.info(f"🔍 [风控统计调试] 使用阈值: {threshold_desc}")
        logger.info(f"🔍 [风控统计调试] 风控策略: {self.config.risk_control_strategy}")
        logger.info(f"🔍 [风控统计调试] 累计收益低于阈值: {cumulative_decline_exits}次")
        if self.config.risk_control_strategy == RiskControlStrategy.CONSECUTIVE_BEARISH:
            logger.info(f"🔍 [风控统计调试] 连续3根阴线: {consecutive_decline_exits}次")
        elif self.config.risk_control_strategy == RiskControlStrategy.DAILY_MA5_BREAK:
            logger.info(f"🔍 [风控统计调试] 日线MA5跌破: {ma5_break_exits}次")
        elif self.config.risk_control_strategy == RiskControlStrategy.T1_LOW_BREAK_WITH_HIGH_DRAWDOWN:
            logger.info(f"🔍 [风控统计调试] T+1最低价跌破: {t1_low_break_exits}次")
            logger.info(f"🔍 [风控统计调试] 盈利状态连续阴线: {profit_bearish_exits}次")

        stats['risk_control_details'] = {
            'total_risk_exits': len(risk_exits),
            'stop_loss_threshold': self.config.stop_loss_threshold,  # 🆕 记录使用的止损阈值
            'stop_loss_threshold_desc': threshold_desc,  # 🆕 阈值描述
            'cumulative_decline_exits': cumulative_decline_exits,
            'consecutive_decline_exits': consecutive_decline_exits,
            'ma5_break_exits': ma5_break_exits,  # 🆕 MA5跌破退出次数
            't1_low_break_exits': t1_low_break_exits,  # 🆕 T+1最低价跌破退出次数
            'profit_bearish_exits': profit_bearish_exits,  # 🆕 盈利状态连续阴线退出次数
            'normal_exits': len(normal_exits),
            'risk_exit_rate': len(risk_exits) / len(self.trades) if self.trades else 0,
            'avg_risk_exit_return': np.mean([t['actual_return'] for t in risk_exits]) if risk_exits else 0,
            'avg_normal_return': np.mean([t['actual_return'] for t in normal_exits]) if normal_exits else 0
        }

        return stats


class OptimizedBacktester:
    """优化版回测器 - 管理多个策略"""

    def __init__(self, prediction_data: pd.DataFrame, minute_data_path: str, cls_gate_topn: int = 0):
        self.prediction_data = prediction_data
        self.minute_data_path = minute_data_path
        self.cls_gate_topn = int(cls_gate_topn) if cls_gate_topn else 0
        self.strategies: List[BacktestStrategy] = []

    # =====================================================
    # 🆕 rank + binary 融合：Top1/Top2 二选一（只在Top-1策略生效）
    #   score = s_rank + alpha * logit(p_cls)
    # =====================================================
    @staticmethod
    def _safe_logit(p: float, eps: float = 1e-6) -> float:
        """稳定logit，避免p=0/1导致无穷大。"""
        try:
            x = float(p)
        except Exception:
            return 0.0
        if np.isnan(x):
            return 0.0
        x = min(max(x, eps), 1.0 - eps)
        return float(np.log(x / (1.0 - x)))

    def _select_target_stock_for_top1_fusion(self, selected_stocks: pd.DataFrame, alpha: float = 0.1) -> pd.Series:
        """
        Top-1策略：在rank Top1/Top2之间，用融合分数二选一。
        - 需要列：predicted_return, p_cls, stock_code
        - 若缺少p_cls或候选不足2只，则退回rank Top1。
        """
        if selected_stocks is None or len(selected_stocks) == 0:
            return None
        if len(selected_stocks) < 2:
            return selected_stocks.iloc[0]
        if 'p_cls' not in selected_stocks.columns:
            return selected_stocks.iloc[0]

        r1 = selected_stocks.iloc[0]
        r2 = selected_stocks.iloc[1]
        s1 = float(r1.get('predicted_return', 0.0))
        s2 = float(r2.get('predicted_return', 0.0))
        p1 = r1.get('p_cls', np.nan)
        p2 = r2.get('p_cls', np.nan)
        z1 = self._safe_logit(p1)
        z2 = self._safe_logit(p2)
        score1 = s1 + float(alpha) * z1
        score2 = s2 + float(alpha) * z2

        logger.info(f"🧠 [FUSION] Top1/Top2融合打分: "
                    f"Top1({r1.get('stock_code')}): s={s1:.6f}, p={float(p1) if p1==p1 else np.nan:.4f}, logit={z1:.4f}, score={score1:.6f} | "
                    f"Top2({r2.get('stock_code')}): s={s2:.6f}, p={float(p2) if p2==p2 else np.nan:.4f}, logit={z2:.4f}, score={score2:.6f}")

        if score2 > score1:
            logger.info(f"✅ [FUSION] 选择Top2替代Top1: {r2.get('stock_code')} (score2>score1)")
            return r2
        else:
            logger.info(f"✅ [FUSION] 保持Top1: {r1.get('stock_code')} (score1>=score2)")
            return r1

    def add_strategy(self, config: StrategyConfig) -> None:
        """添加策略"""
        strategy = BacktestStrategy(config, self.prediction_data, self.minute_data_path)
        self.strategies.append(strategy)

    def run_strategies(self, stock_counts: List[int], trading_dates: List[datetime.date]) -> Dict[str, Any]:
        """运行所有策略"""
        logger.info(f"🚀 开始运行 {len(stock_counts)} 个策略")

        results = {}

        for stock_count in stock_counts:
            config = StrategyConfig(
                strategy_type=StrategyType(stock_count),
                stock_count=stock_count,
                cls_gate_topn=self.cls_gate_topn
            )

            self.add_strategy(config)

        # 每个策略独立运行完整的回测周期
        for strategy in self.strategies:
            result = strategy.run_backtest(trading_dates)
            results[f"Top-{strategy.config.stock_count}"] = result

        return results


# 向后兼容的接口函数
def run_optimized_backtest(prediction_data: pd.DataFrame,
                          stock_counts: List[int],
                          minute_data_path: str,
                          holding_days: int = 2,
                          cls_gate_topn: int = 0) -> Dict[str, Any]:
    """
    向后兼容的回测接口
    """
    logger.info("🔄 启动优化版回测系统")

    backtester = OptimizedBacktester(prediction_data, minute_data_path, cls_gate_topn=cls_gate_topn)
    trading_dates = sorted(prediction_data['date'].unique())

    # 过滤有效的交易日期
    valid_trading_dates = []
    for date in trading_dates:
        if isinstance(date, pd.Timestamp):
            date = date.date()
        valid_trading_dates.append(date)

    logger.info(f"📅 交易日期范围: {valid_trading_dates[0]} ~ {valid_trading_dates[-1]}")

    results = backtester.run_strategies(stock_counts, valid_trading_dates)

    return results