"""因子定义模块，导入后立即完成注册。"""
from .behavioral_optimized_noswitch_validmask_v2 import (
    distance_from_52w_high,
    distance_from_52w_low,
    price_acceleration,
)  # noqa: F401
from .candlestick_optimized_noswitch_validmask_v2 import (
    body_efficiency,
    shadow_dominance,
    gap_power,
    range_compression,
)  # noqa: F401
from .cross_level_interaction_optimized_noswitch_validmask_fixed_v2 import (
    sector_excess_intra_rank,
    market_vols_adaptive_iv,
    market_trend_adaptive_beta,
)

from .cycle_spectral_optimized_noswitch_slim_validmask import (
    cycle_strength_20d,
    spectral_entropy_20d,
    wavelet_highfreq_energy_20d,
    volume_cycle_strength_20d,
)  # noqa: F401
from .intraday_microstructure_slim_optimized_noswitch_validmask import (
    overnight_return,
    intraday_reversal,
    price_impact,
)  
from .limit_interaction_optimized_noswitch_validmask_v2 import (
    sector_limit_overflow_alpha,
    sector_broken_resilience_alpha,
    limit_sustain_alpha,
    market_emotion_resonance_alpha,
)
from .liquidity_optimized_noswitch_validmask import (
    turnover_stability,
    abnormal_turnover,
    money_flow_pressure,
    amihud_illiquidity,
)  # noqa: F401
from .market_corr_optimized_noswitch_validmask import (
    relative_strength_20d,
    beta_60d,
)  # noqa: F401
from .market_regime_interaction_optimized_noswitch_validmask import (
    mrk_emotion_x_pullback_quality,
    mrk_limit_x_breakout20_quality,
    mrk_breadth_x_relstrength_5d,
    mrk_volreg_x_lowvol_trend_20d,
    mrk_liqreg_x_volume_surge,
    mrk_emotion_x_intraday_failure,
    mrk_breadth_x_gap_surprise,
    mrk_limit_x_reversal_risk,
)
from .market_structure_factors_optimized_noswitch_validmask_v4 import (
    mkt_sector_dispersion,
    sector_mkt_corr_20d,
    intra_sector_breadth,
    stock_sector_pos_align,
)
from .momentum_reversal_optimized_noswitch_slim_validmask import (
    high_to_close_drawdown,
    intraday_clv,
    return_10d,
    return_20d,
    momentum_decay,
    # trend_acceleration,
    trend_consistency,
    cpv_factor,
    stable_momentum,
    trend_pullback_composite_alpha,
    volume_confirmed_pullback_alpha,
)  # noqa: F401

from .moving_average_optimized_noswitch_validmask import (
    ma_convergence_score,
    ma60_slope,
    price_ma60_ratio,
    price_ma60_ratio_z,
    ma_alignment_score,
    price_ma5_ratio,
    price_ma10_ratio,
    price_ma10_ratio_z,
    price_ma5_ratio_z,
)  # noqa: F401

from .risk_control_optimized_noswitch_validmask import (
    efficiency_ratio,
    ulcer_index,
    max_drawdown_20d,
    max_rally_20d
)  # noqa: F401
from .sector_interaction_optimized_noswitch_validmask_v3 import (
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

from .sector_optimized_noswitch_v2_validmask_fixed_v4 import (
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
from .sector_stage_factors_optimized_noswitch_validmask_fixed_v2 import (
    sector_emotion_heat_level_20d,
    sector_emotion_heat_accel_3d,
    sector_divergence_risk,
    sector_breadth_level_20d,
    sector_concentration_level_20d,
    sector_streak_strength_10d,
    sector_drawdown_from_peak_10d,
    breakout_z,
)
from .sector_wavelet_interaction_optimized_noswitch_validmask_fixed3_v5 import (
    sw_wv_chop_gate_intra_reversal,
    sw_wv_trend_gate_intra_momentum,
    sw_wv_burst_gate_intra_leader,
    sw_wv_energy_gate_intra_lowvol,
    sw_wv_energy_delta_gate_intra_momentum,
)
from .statistical_moments_optimized_noswitch_validmask_fixed_v2 import (
    skewness_20d,
    kurtosis_20d,
    coskewness,
    downside_beta
)  # noqa: F401

from .style_rotation_optimized_noswitch_validmask_fixed_v3 import (
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

from .tech_sr_worldquant_slim_optimized_noswitch_validmask_fixed2_v2 import(
    support_distance,
    resistance_distance,
    ppo_strength,
    mfi_strength,
    alpha_002,
    alpha_017,
)

from .wavelet_interaction_factors_optimized_noswitch_validmask_fixed_v5 import(
    wv_chop_gate_reversal,
    wv_trend_gate_momentum,
    wv_burst_gate_liquidity,
    wv_burst_gate_lowvol_defensive,
    wv_energy_gate_lowvol,
    wv_energy_delta_gate_momentum,
    wv_trend_gate_beta,
)


