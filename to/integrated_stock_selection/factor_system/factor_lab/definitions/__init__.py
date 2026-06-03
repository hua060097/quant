"""因子定义模块，导入后立即完成注册。"""
from .moving_average import (
    ma_ribbon_expansion_3d,
    ma_alignment_persistence_5d,
    ma_convergence_score,
    log_price_slope_5d,
    log_price_slope_10d,
    log_price_slope_20d,
    log_price_slope_10d_x_log_circ_mv,
    ma60_slope,
    price_ma60_ratio,
    price_ma60_ratio_z,
    ma_alignment_score,
    price_ma5_ratio,
    price_ma5_ratio_raw,
    price_ma10_ratio,
    price_ma10_ratio_z,
    price_ma5_ratio_z,
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
    beta_20d,
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
    sw_circ_mv_rank_intra_sector,
    sector_ma5_followthrough_v3,
    limit_up_sw2_concentration,
    sw2_high_low_position_return_spread,
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
    high_to_close_drawdown,
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
    cpv_factor,
    stable_momentum,
    high_volatility_pullback_alpha,
    volume_confirmed_pullback_alpha,
    trend_pullback_composite_alpha,
    regime_aware_reversal_alpha,
    volume_structure_alpha,
    volume_price_exhaustion,
    failed_breakout_pressure,
    ma5_ma10_flattening_after_runup,
    prev_day_blowoff_risk,
    carryover_overheat_risk,
    distribution_after_strength_risk,
    volatility_structure_alpha
)  # noqa: F401
from .liquidity import (
    turnover_stability,
    abnormal_turnover,
    money_flow_pressure,
    amihud_illiquidity,
)  # noqa: F401
# from .moment_interaction import (
#     leader_acceleration_score,
#     vol_price_explosion_score,
#     dragon_resonance_score,
#     ma5_ratio_acceleration,
#     sector_rank_stability_3d,
#     ma5_ratio_risk_penalty,
# )
from .risk_control import (
    efficiency_ratio,
    ulcer_index,
    max_drawdown_20d,
    max_rally_20d,
    max_rally_10d,
    max_rally_20d_raw,
    max_rally_10d_raw,
    limit_down_3d,
)  # noqa: F401
from .behavioral import (
    distance_from_52w_high,
    distance_from_52w_low,
    price_acceleration,
    post_accumulation_supply_tightness,
    true_turnover_10d,
    tightness_3d,
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
    prior_high_distance_20d,
    prior_high_proximity_20d,
    prior_high_volume_contest_20d,
)  # noqa: F401
from .market_context import (
    market_trend_strength,
    market_sentiment_ad,
    market_volatility_context,
    market_liquidity_change,
    market_advance_decline_ratio,
    market_advance_decline_ratio_percentile_20d,
    market_prev_60d_new_high_avg_return,
    market_60d_new_high_ratio,
    market_60d_new_high_ratio_train_percentile,
    limit_up_count_percentile_20d,
    zhaban_count_percentile_20d,
    limit_down_count_percentile_20d,
)
from .regulatory_risk import (
    abnormal_deviation_3d_ratio,
    abnormal_event_count_10d,
    severe_abnormal_pressure_10d,
    reg_severe_up_10d_ratio,
    severe_abnormal_pressure_30d,
)
from .market_regime_context_v2 import (
    mkt_regime_trend_pressure_v2,
    mkt_breadth_thrust_v2,
    mkt_emotion_stability_v2,
    mkt_dispersion_shock_v2,
    mkt_liquidity_impulse_v2,
)
from .market_regime_state_v3 import (
    mkt_ma5_followthrough_v3,
    mkt_ma5_followthrough_v3_expanding_percentile,
    sector_leadership_persistence_v3,
    sector_hotspot_concentration_v3,
    ma5_strength_breadth_v3,
    strong_stock_volume_confirmation_v3,
    emotion_quality_balance_v3,
    top10_10d_recent3_strength_v3,
    top10_5d_recent3_strength_v3,
    top10_3d_recent3_strength_v3,
    top10_10d_strength_level_v3,
    top10_10d_today_return_level_v3,
)
from .limit_interaction import (
    sector_limit_overflow_alpha,
    sector_broken_resilience_alpha,
    limit_sustain_alpha,
    market_emotion_resonance_alpha,
    prev_first_limit_up_performance,
    prev_first_limit_up_performance_percentile_20d,
    prev_non_first_limit_up_performance,
    prev_non_first_limit_up_performance_percentile_20d,
)
from .moneyflow_features import (
    buy_lg_amount_rate,
    buy_sm_amount_rate,
    lg_pos_sm_neg_10d_days,
    net_d5_amount_to_turnover,
    main_outflow_retail_takeover,
)
# from .cycle_spectral import(
#     cycle_dominant_period_20d,
#     cycle_strength_20d,
#     spectral_entropy_20d,
#     cycle_phase_sin_20d,
#     cycle_phase_cos_20d,
#     wavelet_highfreq_energy_20d,
#     wavelet_dominant_level_20d,
#     wavelet_cycle_period_20d,
#     volume_cycle_strength_20d,

# )
# from .market_breadth_limit_cycle_light_no_loop_v3 import(
#     mkt_adv_dec,
#     mkt_limit_balance,
#     mkt_limit_up_ratio,
#     mkt_limit_down_ratio,
#     mkt_zhaban_ratio,
#     mkt_emotion_heat,
#     mkt_emotion_heat_dom_period_32d,
#     mkt_emotion_heat_dom_strength_32d,
#     mkt_emotion_heat_spec_entropy_32d,
#     mkt_emotion_heat_hf_energy_ratio_32d,
#     mkt_adv_dec_ma3,
#     mkt_limit_balance_ma3,
#     mkt_emotion_heat_ma3,
#     mkt_adv_dec_delta3,
#     mkt_limit_balance_delta3,
#     mkt_emotion_heat_delta3,
#     mkt_adv_dec_z10,
#     mkt_limit_balance_z10,
#     mkt_emotion_heat_z10,
# )
from .high_position_exhaustion import (
    close_position_20d,
    close_position_60d,
    upper_shadow_ratio_3d_mean,
    upper_shadow_pressure_5d,
    lower_shadow_absorption_5d,
    close_to_high_3d_mean,
    turnover_z_20,
    stock_volume_ratio_20d,
    high_position_volume_surge,
    high_volume_no_gain_3d,
    pullback_depth_from_10d_high,
    high_pos_upper_shadow,
    high_pos_volume_no_gain,
    high_pos_reversal_volume,
    shadow_reversal_volume_pressure,
    ret_3d_vs_10d_accel,
    ma5_pullback_depth_10d,
    ret_5d_vs_20d_accel,
    momentum_peak_decay_3d,
    consecutive_up_days_5d,
    large_up_days_10d,
    gap_up_count_5d,
    close_below_ma5_after_runup,
    low_break_prev_low_after_runup,
    runup_ma5_exhaustion,
)
