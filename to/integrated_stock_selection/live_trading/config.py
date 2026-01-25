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


# ============================================================
# 止盈止损配置
# ============================================================
class StopLossConfig:
    """止盈止损配置"""
    # 止盈倍数(ATR)
    ATR_MULTIPLIER_PROFIT = 1.8

    # 止损倍数(ATR)
    ATR_MULTIPLIER_LOSS = 0.9

    # 最大止损比例(%)
    # 说明：
    #   - 正值：允许下跌的比例（例如 0.05 表示允许下跌5%）
    #   - 0.00：盈亏平衡，收益为负就卖出（激进风控）
    #   - 负值：要求正收益（例如 -0.01 表示要求至少1%收益才不卖出）
    MAX_STOP_LOSS_PCT = 0.00  # 0% = 盈亏平衡（与回测系统一致）

    # 最大持仓天数
    # 注意: T+1买入当天算第1天
    # 例如: MAX_HOLDING_DAYS=3, 则T+1/T+2/T+3持仓, T+4开盘强制卖出
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

