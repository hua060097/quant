"""因子定义模块，导入后立即完成注册。"""
from .moving_average import (
    ma_convergence_score,
    ma60_slope,
    price_ma60_ratio,
    ma_alignment_score,
    price_ma5_ratio
)  # noqa: F401
from .candlestick import (
    body_efficiency,
    shadow_dominance,
    gap_power,
    range_compression,
)  # noqa: F401
from .market_corr import (
    relative_strength_20d,
    beta_60d,
)  # noqa: F401
from .microstructure import price_impact  # noqa: F401
from .sector import (
    intra_sector_momentum_rank,
    sector_momentum_dispersion_rank,
    sector_dispersion,
    sector_relative_volatility,
    sector_momentum_consistency,
    sector_correlation_20d,
    sector_volatility_5d,
    sector_momentum_interaction,
    sw_circ_mv_rank_intra_sector
)  # noqa: F401
from .kdj import (
    kdj_golden_cross_strength,
    kdj_diff_9,
    kdj_k_9,
    kdj_j_9,
) # noqa: F401
from .technical_indicators import (
    ppo_strength,
    mfi_strength,
    vortex_trend,
    mass_index_reversal,
)  # noqa: F401
from .statistical_moments import (
    skewness_20d,
    kurtosis_20d,
    coskewness,
    downside_beta
)  # noqa: F401
from .momentum_reversal import (
    pct_chg_intra_z,
    pct_chg_sector_mean,
    pct_chg_1d,
    pct_chg_2d,
    return_3d_intra_z,
    return_3d_sector_mean,
    return_5d_intra_z,
    return_5d_sector_mean,
    return_10d_intra_z,
    return_10d_sector_mean,
    return_20d_intra_z,
    return_20d_sector_mean,
    intraday_clv,
    momentum_decay,
    trend_acceleration,
    trend_consistency,
    trend_persistence_10d,
    trend_persistence_20d,
    cpv_factor,
    stable_momentum,
    high_volatility_pullback_alpha,
    volume_confirmed_pullback_alpha,
    trend_pullback_composite_alpha,
    regime_aware_reversal_alpha,
    volume_structure_alpha,
    volatility_structure_alpha
)  # noqa: F401
from .liquidity import (
    turnover_stability,
    abnormal_turnover,
    money_flow_pressure,
    amihud_illiquidity,
)  # noqa: F401
from .risk_control import (
    efficiency_ratio,
    ulcer_index,
    max_drawdown_20d,
    max_rally_20d
)  # noqa: F401
from .behavioral import (
    distance_from_52w_high,
    distance_from_52w_low,
    price_acceleration,
)  # noqa: F401
from .intraday import (
    overnight_return,
    intraday_reversal,
    open_to_high_ratio,
)  # noqa: F401
from .worldquant import (
    alpha_001,
    alpha_017,
    alpha_002, 
)  # noqa: F401
from .style_rotation import (
    sector_concentration,
    sector_breadth,
    size_rotation_strength,
    value_growth_rotation,
    sector_rank_change_5d,
    sector_momentum_accel_rank,
    sector_turnover_change_5d_20d,
    sector_hhi_change,
    sector_relative_strength_vs_market,
)  # noqa: F401
from .sector_interaction import (
    sector_vols_divergence_alpha,
    sector_turnover_surge_pv_alpha,
    sector_dispersion_leader_alpha,
    sector_trend_gated_momentum_intra_z,
    sector_rotation_breakout_intra_z,
    sector_leader_capture_rank,
    sector_strong_leader_alpha,
    sector_liquidity_gated_turnover_z,
    sector_weak_reversal_alpha,
)  # noqa: F401
from .cross_level_interaction import (
    sector_excess_intra_rank,
    market_vols_adaptive_iv,
    market_trend_adaptive_beta,
)
from .market_structure_factors import (
    mkt_sector_dispersion,
    sector_mkt_corr_20d,
    intra_sector_breadth,
    stock_sector_pos_align,
)
from .support_resistance import (
    support_distance,
    resistance_distance,
)  # noqa: F401
from .market_context import (
    market_trend_strength,
    market_sentiment_ad,
    market_volatility_context,
    market_liquidity_change,
)


