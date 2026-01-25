#!/usr/bin/env python3
"""
一体化选股系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

@dataclass
class SystemConfig:
    """系统配置"""
    
    # 数据配置
    start_date: str = "20230601"
    end_date: str = "20250328"
    train_end_date: str = "20240601"  # 训练集结束日期
    test_start_date: str = "20240101"  # 测试集开始日期
    
    # 数据路径
    project_root: Path = Path(__file__).parent.parent
    stock_features_dir: Path = project_root / "local_data/daily"  # 修正路径
    sector_features_dir: Path = project_root / "local_data/sector_features"
    market_status_dir: Path = project_root / "local_data/market_status"
    board_limit_dir: Path = project_root / "local_data/sector_features_board"
    cache_dir: Path = project_root / "integrated_stock_selection/cache"
    results_dir: Path = project_root / "integrated_stock_selection/results"
    
    # 选股配置
    daily_stock_count: int = 2  # 每日选股数量
    sector_filter_count: int = 5  # 板块筛选数量（减少到5个）
    min_market_cap: float = 10e8  # 最小市值（10亿，降低阈值）
    min_volume: float = 1000e4  # 最小成交额（1000万，降低阈值）
    max_volatility: float = 0.20  # 最大波动率（放宽到0.20）

    # 连板配置
    max_lookback_days: int = 20  # 连板历史回看天数
    limit_up_penalty_factor: float = 0.8  # 连板惩罚因子
    
    # 多因子权重配置（提高板块权重）
    factor_weights: Dict[str, float] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.factor_weights is None:
            self.factor_weights = {
                'technical': 0.15,    # 技术面（提高）
                'fundamental': 0.15,  # 基本面（提高）
                'sector': 0.55,       # 板块面（提高）
                'capital_flow': 0.15  # 资金面（提高）
            }
        
        # 创建必要目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class SectorConfig:
    """板块配置"""
    
    # 板块评分权重（来自之前的最优配置）
    sector_weights: Dict[str, float] = None
    
    # 市场环境调整因子
    env_adjustments: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.sector_weights is None:
            # 使用技术主导的最优配置
            self.sector_weights = {
                'technical': 0.30,
                'capital': 0.15,
                'sentiment': 0.20,
                'relative': 0.15,
                'momentum': 0.10,
                'volatility': 0.05,
                'structure': 0.03,
                'quality': 0.02
            }
        
        if self.env_adjustments is None:
            self.env_adjustments = {
                'strong_market': {
                    'technical': 1.2, 'capital': 1.0, 'sentiment': 1.4, 'relative': 0.9,
                    'momentum': 1.3, 'volatility': 0.8, 'structure': 1.1, 'quality': 1.0
                },
                'weak_market': {
                    'technical': 0.9, 'capital': 1.2, 'sentiment': 0.8, 'relative': 1.3,
                    'momentum': 0.8, 'volatility': 1.2, 'structure': 1.2, 'quality': 1.3
                },
                'volatile_market': {
                    'technical': 1.0, 'capital': 1.1, 'sentiment': 1.2, 'relative': 1.2,
                    'momentum': 1.1, 'volatility': 1.0, 'structure': 1.0, 'quality': 1.1
                },
                'normal_market': {
                    'technical': 1.0, 'capital': 1.0, 'sentiment': 1.0, 'relative': 1.0,
                    'momentum': 1.0, 'volatility': 1.0, 'structure': 1.0, 'quality': 1.0
                }
            }

@dataclass
class BacktestConfig:
    """回测配置"""
    
    # 交易成本
    commission_rate: float = 0.001  # 手续费率 0.1%
    slippage_rate: float = 0.0005   # 滑点率 0.05%
    
    # 风险控制
    max_position_size: float = 0.5  # 单只股票最大仓位
    stop_loss_pct: float = -0.08    # 止损比例 -8%
    take_profit_pct: float = 0.15   # 止盈比例 15%
    
    # 持仓配置
    holding_days: int = 1  # 持仓天数（T+1）
    rebalance_freq: str = "daily"  # 调仓频率
    
    # 评价指标
    benchmark: str = "000300.SH"  # 基准指数（沪深300）
    risk_free_rate: float = 0.03  # 无风险利率

# 全局配置实例
system_config = SystemConfig()
sector_config = SectorConfig()
backtest_config = BacktestConfig()

# 特征配置
STOCK_FEATURE_COLUMNS = [
    # 基础信息（排除future_return_2d）
    'stock_code', 'date',
    
    # 价格特征
    'open', 'high', 'low', 'close', 'pct_chg',
    'return_5d', 'return_20d',
    
    # 成交量特征
    'volume', 'amount', 'turnover_rate', 'volatility20',
    
    # 技术指标
    'ma5', 'ma10', 'ma20', 'ma60',
    'rsi6', 'rsi12', 'rsi14',
    'macd', 'macd_signal', 'macd_hist',
    
    # 涨跌停特征
    'is_limit_up', 'is_limit_down', 'limit_times',
    'limit_strength', 'limit_quality_score',
    
    # 基本面特征
    'total_mv', 'circ_mv', 'float_ratio',
    
    # 高级特征
    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'atr14', 'trend_strength', 'trend_consistency',
    'relative_strength_20d', 'momentum_acceleration',
    'technical_score', 'momentum_score', 'composite_score',
    
    # 目标变量
    'future_return_1d'
]

# 排除的特征列
EXCLUDED_FEATURES = [
    'future_return_2d',  # 排除2日收益率
    'date_str',          # 重复的日期字符串
]

# 板块特征列（关键特征）
SECTOR_FEATURE_COLUMNS = [
    'sector_code', 'date', 'change_percent', 'volume', 'amount',
    'limit_up_count', 'limit_down_count', 'board_counts_1板', 'board_counts_2板',
    'up_count', 'down_count', 'stock_count'
]
