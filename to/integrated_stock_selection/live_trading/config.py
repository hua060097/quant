"""
实盘交易系统配置
"""

import os
from pathlib import Path

# ============================================================
# 迅投API配置
# ============================================================
class XtQuantConfig:
    """迅投API配置"""
    # MiniQMT客户端userdata_mini路径(用户需要修改)
    USERDATA_MINI_PATH = "D:\\国金证券QMT交易端\\userdata_mini"  # 例如: "D:\\迅投极速交易终端\\userdata_mini"

    # 会话ID基础值(用于生成唯一ID)
    SESSION_ID_BASE = 123456  # 基础ID，实际使用时会自动添加时间偏移

    # 资金账号(用户需要修改)
    ACCOUNT_ID = "8886921123"  # 例如: "1000000365"

    # 账号类型
    ACCOUNT_TYPE = "STOCK"  # STOCK-股票, CREDIT-信用, FUTURE-期货

    @staticmethod
    def generate_session_id() -> int:
        """
        生成唯一的session_id

        迅投要求：不同的Python策略必须使用不同的session_id，必须保证唯一性。
        本方法通过基础ID + 时间戳偏移的方式，确保每次运行都有唯一的session_id。

        生成规则：
        - 使用基础ID + 时间戳的最后3位（000-999）
        - 确保不同程序实例的session_id不重复
        - 支持同时运行多个策略实例

        Returns:
            int: 唯一的session_id（6位整数）

        Example:
            >>> session_id = XtQuantConfig.generate_session_id()
            >>> print(session_id)
            123568  # 基础ID 123456 + 偏移 112
        """
        import time

        # 使用时间戳的最后3位作为偏移（000-999）
        # 即使在同一秒内启动多个实例，偏移值也足够大
        time_offset = int(time.time()) % 1000

        # 组合生成：基础ID + 偏移
        session_id = XtQuantConfig.SESSION_ID_BASE + time_offset

        return session_id


# ============================================================
# 交易参数配置
# ============================================================
class TradingConfig:
    """交易参数配置"""
    # 选股阈值(ATR倍数)
    MIN_POS_THRESHOLD_ATR = 1.0

    # 风险厌恶系数(用于集成预测)
    RISK_AVERSION = 0.5

    # 交易成本(ATR倍数)
    TRADING_COST_ATR = 0.03

    # 最小交易单位(股)
    MIN_TRADE_UNIT = 100

    # 报价类型
    PRICE_TYPE_BUY = "LATEST_PRICE"   # 买入使用最新价
    PRICE_TYPE_SELL = "LATEST_PRICE"  # 卖出使用最新价

    # 委托类型
    ORDER_TYPE_BUY = "STOCK_BUY"
    ORDER_TYPE_SELL = "STOCK_SELL"


class StrategyConfig:
    """策略层配置"""
    # 当前 copy.py 默认 Top1 策略不启用 live_default q40 Top1/Top2 动态切换
    ENABLE_DYNAMIC_RANK_SWITCH = False

    # 策略级可用总资金上限（独立于账户总资产）
    STRATEGY_CAPITAL = 78_000.0

    # 并行槽位和固定槽位仓位
    MAX_CONCURRENT_POSITIONS = 3
    POSITION_SCHEDULE = [0.8, 0.2, 0.4]

    # pred/select 输出保留的候选数
    PRED_TOPN = 5

    # 默认单模型（与当前主策略保持一致）
    DEFAULT_LIGHTGBM_MODEL_FILE = "lightgbm_model_window3_20260427_231053.pkl"

    # 默认高位衰竭/承接失败过滤（先排序，再判断候选是否可交易）
    ENABLE_EXHAUSTION_BUY_FILTER = True
    EXHAUSTION_FILTER_STRATEGY = "three_rule_v3"
    # v3: three_rule_v3 (基于50+162双日志鲁棒性分析)
    EXHAUSTION_V3_CPV_THRESHOLD = -0.4              # cpv_factor <= 此值（配合runup>0.8）
    EXHAUSTION_V3_BLOWOFF_THRESHOLD = 2.0           # prev_day_blowoff_risk >= 此值
    EXHAUSTION_V3_SECTOR_ACCEL_THRESHOLD = 0.36     # sector_momentum_accel_rank <= 此值
    EXHAUSTION_V3_SECTOR_LEADERSHIP_THRESHOLD = -0.18  # sector_leadership_persistence_v3 <= 此值
    EXHAUSTION_V3_RETURN_20D_INTRA_Z_THRESHOLD = 0.40  # return_20d_intra_z <= 此值
    EXHAUSTION_V3_MARKET_LIQ_THRESHOLD = 0.94  # market_liquidity_change < 此值
    EXHAUSTION_V3_ULCER_THRESHOLD = -2.25       # ulcer_index < 此值 (路径A)
    EXHAUSTION_V3_MOM_PEAK_THRESHOLD = 3.20     # momentum_peak_decay_3d > 此值 (路径B)
    EXHAUSTION_V3_INTRA_SECTOR_RANK_THRESHOLD = 0.67  # intra_sector_momentum_rank < 此值 (路径C)
    EXHAUSTION_V3_MKT_DISP_THRESHOLD = 0.98    # mkt_dispersion_shock_v2 < 此值
    EXHAUSTION_V3_MKT_ADR_THRESHOLD = 0.38     # market_advance_decline_ratio < 此值
    ENABLE_EXHAUSTION_LIMIT_DOWN_3D = True      # Rule6: 近3日跌停过滤
    EXHAUSTION_V3_TOP10_STRENGTH_THRESHOLD = 0.80   # top10_10d_strength_level_v3 >= 此值 (Rule7)
    EXHAUSTION_V3_BUY_SM_AMOUNT_PCT_THRESHOLD = 0.50  # buy_sm_amount_rate_pct <= 此值 (Rule7当前口径)
    EXHAUSTION_V3_BUY_SM_AMOUNT_THRESHOLD = -1.50   # 备用Rule7: buy_sm_amount_rate <= 此值；pct版本不稳定时切回
    EXHAUSTION_V3_RULE7_REG_SEVERE_EXEMPT_PCT_THRESHOLD = 0.999  # Rule7豁免: 极强趋势环境
    EXHAUSTION_V3_RULE7_TOP10_3D_EXEMPT_THRESHOLD = 0.85          # Rule7豁免: 近3日头部强势延续
    EXHAUSTION_V3_NON_FIRST_LIMIT_UP_PERF_PCT20_THRESHOLD = 0.80  # Rule8: 连板情绪近20日高分位
    EXHAUSTION_V3_KDJ_J_Z_THRESHOLD = 0.75  # Rule8: kdj_j_9为截面robust z-score后的J值
    EXHAUSTION_V3_SCORE_Z_IN_DAY_THRESHOLD = 1.50  # Rule9: 当日Top1相对候选池预测强度不足
    EXHAUSTION_V3_REG_SEVERE_UP_10D_PCT_THRESHOLD = 0.98  # Rule10: 个股严重上涨占比当日最高2%
    EXHAUSTION_V3_REG_HOT_ADR_THRESHOLD = 4.50  # Rule10: 市场涨跌比极热
    # Rule11: MA5乖离率 >= 9% + 均线收敛度 >= 0.20 → 高位衰竭
    # 拦截4笔增量大亏，0误杀大涨/小赚
    EXHAUSTION_V3_MA5_RAW_THRESHOLD = 9.0
    EXHAUSTION_V3_MA_CONVERGENCE_THRESHOLD = 0.20
    # Rule12: MA5回撤深度极深(<=-0.05) + 短期加速上涨(>=6) → 高位回撤加速衰竭
    EXHAUSTION_V3_MA5_PULLBACK_DEPTH_THRESHOLD = -0.05  # ma5_pullback_depth_10d <= 此值(Rule12)
    EXHAUSTION_V3_RET_3D_VS_10D_ACCEL_THRESHOLD = 6.0   # ret_3d_vs_10d_accel >= 此值(Rule12)
    # v2: runup_ma5 / runup_or_blowoff (旧规则，保留)
    EXHAUSTION_RUNUP_MA5_THRESHOLD = 0.8
    EXHAUSTION_TOP10_10D_STRENGTH_THRESHOLD = 0.5
    EXHAUSTION_PREV_DAY_BLOWOFF_THRESHOLD = 4.15
    EXHAUSTION_SECTOR_MOMENTUM_ACCEL_WEAK_THRESHOLD = 0.41
    EXHAUSTION_MARKET_ADVANCE_DECLINE_RATIO_THRESHOLD = 5.39559
    EXHAUSTION_TURNOVER_Z20_THRESHOLD = 1.0

    # q40 训练窗
    Q40_TRAIN_START = "2023-06-01"
    Q40_TRAIN_END = "2025-03-01"

    # 开盘跳空风控（与当前回测 gap6 / gap_up5 对齐）
    GAP_DOWN_THRESHOLD_TOP1 = -0.06
    GAP_DOWN_THRESHOLD_STRONG_TOP2 = -0.06
    GAP_DOWN_THRESHOLD_NON_TOP1 = -0.06
    GAP_DOWN_THRESHOLD_WEAK_TOP2 = -0.06
    GAP_UP_THRESHOLD = 0.05


# ============================================================
# 止盈止损配置
# ============================================================
class StopLossConfig:
    """止盈止损配置"""
    # 止盈倍数(ATR)
    ATR_MULTIPLIER_PROFIT = 1.8

    # 止损倍数(ATR)
    ATR_MULTIPLIER_LOSS = 0.9

    # 全局硬止损兜底
    MAX_STOP_LOSS_PCT = -0.05

    # 与分钟回测一致的状态机参数
    ENABLE_TOP1_STATE_MACHINE_RISK = True
    T2_0935_HARD_STOP_THRESHOLD = -0.04
    ENABLE_T2_1000_SOFT_STOP = True
    T2_1000_SOFT_STOP_THRESHOLD = -0.02
    ENABLE_SECOND_LAYER_TIME_STOP = True
    T2_CLOSE_BREAKEVEN_STOP_THRESHOLD = 0.00
    T3_0935_BREAKEVEN_STOP_THRESHOLD = 0.00

    ENABLE_SMALL_PROFIT_FAILURE_EXIT = True
    SMALL_PROFIT_ACTIVATION_RETURN = 0.04
    SMALL_PROFIT_BEARISH_REQUIRED_COUNT = 2
    SMALL_PROFIT_WATCH_WINDOW_BARS = 3

    ENABLE_PROFIT_THRESHOLD_FOR_BEARISH_EXIT = True
    PROFIT_BEARISH_ACTIVATION_RETURN = 0.05
    PROFIT_BEARISH_MIN_PEAK_RETURN = 0.09
    PROFIT_BEARISH_TRAILING_DRAWDOWN = -0.04
    PROFIT_BEARISH_MIN_CURRENT_RETURN = 0.06
    PROFIT_BEARISH_REQUIRED_COUNT = 3

    # 最大持仓天数
    MAX_HOLDING_DAYS = 3

    # 监控检查间隔(秒)
    CHECK_INTERVAL = 3


# ============================================================
# A股交易时间配置
# ============================================================
class MarketTimeConfig:
    """A股交易时间配置"""
    # 开盘时间
    MORNING_OPEN = "09:30:00"
    MORNING_CLOSE = "11:30:00"
    
    AFTERNOON_OPEN = "13:00:00"
    AFTERNOON_CLOSE = "15:00:00"
    
    # 集合竞价时间
    CALL_AUCTION_START = "09:15:00"
    CALL_AUCTION_END = "09:25:00"


# ============================================================
# 文件路径配置
# ============================================================
class PathConfig:
    """文件路径配置"""
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 选股结果文件
    STOCK_SELECTIONS_FILE = PROJECT_ROOT / "live_trading" / "stock_selections.json"
    
    # 交易记录文件
    TRADE_RECORDS_FILE = PROJECT_ROOT / "live_trading" / "trade_records.json"
    
    # 模型目录
    MODEL_DIR = PROJECT_ROOT / "results"
    
    # 日志目录
    LOG_DIR = PROJECT_ROOT / "live_trading" / "logs"
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.STOCK_SELECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 过滤条件配置
# ============================================================
class FilterConfig:
    """过滤条件配置"""
    # 新股上市天数阈值
    NEW_STOCK_DAYS = 60
    
    # ST股票过滤
    FILTER_ST = True
    
    # 涨停股过滤
    FILTER_LIMIT_UP = True
    
    # 涨停阈值(%)
    LIMIT_UP_THRESHOLD = 9.9  # 考虑浮点误差


# 初始化目录
PathConfig.ensure_dirs()
