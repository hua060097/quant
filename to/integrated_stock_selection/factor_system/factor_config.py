#!/usr/bin/env python3
"""
因子配置管理器

管理因子组的配置、版本控制和依赖关系
"""

import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

class FactorConfig:
    """因子配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = Path(config_file) if config_file else None
        self._config = self._load_default_config()
        
        if self.config_file and self.config_file.exists():
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            'factor_groups': {
                'technical_factors': {
                    'enabled': True,
                    'version': 'v2.1',
                    'features': [
                        'rsi6', 'rsi12', 'rsi14', 'macd', 'volatility20',
                        'return_5d', 'return_20d', 'ma5', 'ma10', 'ma20', 'ma60'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,  # 缓存7天
                    'priority': 1
                },
                'fundamental_factors': {
                    'enabled': True,
                    'version': 'v1.5',
                    'features': [
                        'pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa',
                        'debt_to_equity', 'current_ratio', 'quick_ratio'
                    ],
                    'dependencies': [],
                    'cache_ttl': 30,  # 缓存30天
                    'priority': 2
                },
                'kline_pattern_factors': {
                    'enabled': True,
                    'version': 'v3.0',
                    'features': [
                        'pattern_hammer', 'pattern_doji', 'pattern_engulfing_bull',
                        'pattern_engulfing_bear', 'pattern_morning_star', 'pattern_evening_star',
                        'w_bottom_similarity', 'head_shoulders_similarity', 'cup_handle_similarity'
                    ],
                    'dependencies': [],  # K线形态因子不依赖其他因子组
                    'cache_ttl': 7,
                    'priority': 3
                },
                'microstructure_factors': {
                    'enabled': True,
                    'version': 'v2.3',
                    'features': [
                        'bid_ask_spread', 'order_flow_imbalance', 'price_impact',
                        'volume_weighted_price', 'tick_direction', 'trade_intensity'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致
                    'cache_ttl': 1,  # 缓存1天
                    'priority': 4
                },
                'risk_factors': {
                    'enabled': True,
                    'version': 'v1.8',
                    'features': [
                        'risk_adjusted_return', 'returns_zscore', 'vol_term_structure',
                        'price_vol_change_ratio', 'high_volatility', 'recent_limit_down'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致
                    'cache_ttl': 7,
                    'priority': 5
                },

                'momentum_factors': {
                    'enabled': True,
                    'version': 'v2.5',
                    'features': [
                        'momentum_quality_score', 'cumulative_return_3d', 'cumulative_return_5d',
                        'cumulative_return_10d', 'price_momentum', 'earnings_momentum'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致
                    'cache_ttl': 3,
                    'priority': 7
                },
                'volume_factors': {
                    'enabled': True,
                    'version': 'v1.9',
                    'features': [
                        'volume_ratio', 'turnover_rate', 'volume_price_trend',
                        'volume_weighted_return', 'volume_momentum'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致
                    'cache_ttl': 7,
                    'priority': 8
                },
                'cross_sectional_factors': {
                    'enabled': True,
                    'version': 'v1.6',
                    'features': [
                        'rank_pct_chg', 'rank_volume', 'rank_turnover', 'rank_amplitude',
                        'rank_momentum_5d', 'rank_momentum_20d', 'rank_rsi', 'rank_macd'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致
                    'cache_ttl': 1,
                    'priority': 9
                },

                'limit_up_quality_factors': {  # 重构为直接加载原始特征模式（v2.0）
                    'enabled': True,
                    'version': 'v2.0',
                    'features': [
                        'is_limit_up', 'is_limit_down', 'is_zhaban', 'limit_strength',
                        'limit_up_days_5d', 'limit_down_days_5d', 'limit_up_days_20d',
                        'consecutive_limit_up', 'consecutive_limit_down', 'limit_up_volume_ratio',
                        'limit_up_turnover_ratio', 'amihud_illiquidity', 'near_support'
                    ],
                    'dependencies': [],  # 直接加载模式，无依赖
                    'cache_ttl': 7,
                    'priority': 11
                },
                'enhanced_correlation_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        'corr_5d', 'corr_10d', 'corr_20d', 'corr_60d', 'corr_120d',
                        'beta_5d', 'beta_10d', 'beta_20d', 'beta_60d', 'beta_120d',
                        'corr_slope_5_20', 'corr_slope_10_60', 'corr_slope_20_120',
                        'beta_slope_5_20', 'beta_slope_10_60', 'beta_slope_20_120',
                        'corr_10d_mom5', 'corr_20d_mom5', 'corr_60d_mom5',
                        'beta_10d_mom5', 'beta_20d_mom5', 'beta_60d_mom5',
                        'corr_20d_zscore250', 'corr_60d_zscore250',
                        'beta_20d_zscore250', 'beta_60d_zscore250'
                    ],
                    'dependencies': [],  # 修复：与代码实现保持一致，不依赖其他因子组
                    'cache_ttl': 7,
                    'priority': 12
                },
                'comprehensive_ma_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        'ma_convergence', 'price_ma5_distance', 'price_ma10_distance',
                        'price_ma20_distance', 'price_ma60_distance', 'ma_support_strength',
                        'ma5_above_ma10_strength', 'ma5_below_ma10_strength',
                        'ma_alignment_score', 'ma_convergence_score',
                        'ma5_ma20_distance_pct', 'ma20_ma60_distance_pct',
                        'macd_line', 'macd_signal', 'macd_histogram',
                        'ema12_ema26_cross', 'ma_deviation_rate_5d', 'ma_deviation_rate_20d',
                        'ma_trend_strength', 'ma_volatility_ratio',
                        'ma_ratio_5_20', 'ma_ratio_20_60', 'ma_ratio_10_30', 'ma_ratio_5_60',
                        'price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20',
                        'price_ma_ratio_60', 'price_ma_ratio_120'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,
                    'priority': 13
                },
                'market_relative_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        # 基础相对因子（1个）
                        'volatility_ratio_to_market_20d',

                        # 个股涨幅相关因子（6个）
                        'pct_chg_corr_limit_up_rank_5d',
                        'pct_chg_corr_limit_up_rank_10d',
                        'pct_chg_corr_limit_up_rank_20d',
                        'pct_chg_corr_volume_rank_5d',
                        'pct_chg_corr_volume_rank_10d',
                        'pct_chg_corr_volume_rank_20d',

                        # 个股动量相关因子（6个）
                        'momentum_corr_limit_up_rank_5d',
                        'momentum_corr_limit_up_rank_10d',
                        'momentum_corr_limit_up_rank_20d',
                        'momentum_corr_volume_rank_5d',
                        'momentum_corr_volume_rank_10d',
                        'momentum_corr_volume_rank_20d',
                    ],
                    'dependencies': [],  # 不依赖其他因子组，直接使用原始数据
                    'cache_ttl': 7,
                    'priority': 14
                },
                'price_position_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        'price_position_in_range_5d', 'price_position_in_range_10d',
                        'price_position_in_range_20d', 'price_position_in_range_60d',
                        'price_vs_ma5_ratio', 'price_vs_ma10_ratio',
                        'price_vs_ma20_ratio', 'price_vs_ma60_ratio'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,
                    'priority': 14
                },
                'advanced_technical_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        # 趋势与质量指标
                        'price_ema20_ratio', 'price_ema50_ratio', 'ema20_ema50_ratio', 'ema50_ema200_ratio',
                        'ema20_slope', 'ema50_slope', 'log_price_regression_slope', 'log_price_regression_r2',
                        'adx_14', 'di_difference',

                        # 回调刻画指标
                        'drawdown_from_20d_high_pct', 'drawdown_atr_ratio', 'atr20', 'atr20_percentile_60d',
                        'rsi_14', 'bollinger_percent_b', 'price_ema20_zscore', 'negative_days_5d', 'large_red_candle_5d',

                        # 支撑与强度指标
                        'distance_to_support_atr', 'support_touch_count_120d', 'support_rebound_avg_return', 'support_rebound_win_rate',

                        # 流动性与环境指标
                        'turnover_percentile_20d', 'amount_percentile_20d', 'prev_limit_up', 'prev_limit_down'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,
                    'priority': 15
                },
                'sector_relative_factors': {
                    'enabled': True,
                    'version': 'v2.0',
                    'features': [
                        # 基础相对收益因子
                        'sector_relative_return_1d', 'sector_relative_return_5d',
                        'sector_relative_return_20d', 'sector_relative_strength',
                        'sector_outperform_ratio_20d', 'sector_correlation_20d',

                        # 板块成交量相对排名因子
                        'sector_volume_rank_3d', 'sector_volume_rank_5d',
                        'sector_volume_rank_10d', 'sector_volume_rank_20d',

                        # 板块涨幅相对排名因子
                        'sector_return_rank_3d', 'sector_return_rank_5d',
                        'sector_return_rank_10d', 'sector_return_rank_20d',

                        # 板块动量强度因子
                        'sector_momentum_strength', 'sector_momentum_consistency',
                        'sector_momentum_acceleration',

                        # 板块流动性因子
                        'sector_liquidity_score', 'sector_liquidity_stability',
                        'sector_relative_liquidity'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,
                    'priority': 16
                },
                'elite_support_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        # 历史支撑位识别与强度
                        'historical_support_strength', 'support_zone_proximity',
                        'support_test_frequency', 'support_hold_success_rate',
                        'volume_at_support_ratio', 'support_formation_quality',

                        # 技术支撑分析
                        'moving_average_support_score', 'fibonacci_retracement_level',
                        'previous_resistance_support', 'round_number_support_effect',
                        'gap_fill_support_strength',

                        # 支撑质量验证
                        'support_volume_confirmation', 'support_time_consistency',
                        'support_price_action_quality', 'support_market_structure_alignment',

                        # 风险与交易信号
                        'support_breakdown_probability', 'support_bounce_probability',
                        'support_entry_timing_score', 'support_risk_reward_ratio',
                        'support_position_sizing_score',

                        # 市场环境与相对强度
                        'market_volatility_regime', 'sector_relative_strength',
                        'support_vs_market_correlation', 'institutional_activity_score',
                        'breakout_momentum_score',

                        # 高频交易特征
                        'intraday_support_stability', 'support_order_flow_imbalance',
                        'support_microstructure_quality', 'support_liquidity_depth',

                        # 资金流与情绪特征
                        'smart_money_flow_signal', 'retail_vs_institutional_ratio',
                        'market_sentiment_support', 'fear_greed_support_impact',
                        'market_trend_alignment', 'sector_rotation_impact',
                        'market_liquidity_condition'
                    ],
                    'dependencies': [],
                    'cache_ttl': 7,
                    'priority': 17
                },
                'market_corr_factors': {
                    'enabled': True,
                    'version': 'v1.0',
                    'features': [
                        'volume_correlation',      # 股票成交量与市场成交量的相关性
                        'corr_5d',                # 5日短期价格相关性
                        'volatility_correlation'   # 股票波动率与市场波动率的相关性
                    ],
                    'dependencies': [],  # 不依赖其他因子组，直接使用原始数据计算相关性特征
                    'cache_ttl': 7,
                    'priority': 18
                }
            },
            'global_settings': {
                'max_parallel_groups': 4,
                'enable_caching': True,
                'cache_compression': True,
                'validation_enabled': True,
                'auto_dependency_resolution': True,
                # 内存优化配置
                'enable_memory_optimization': True,
                'enable_batch_save': True,
                'memory_monitor_interval': 10,
                'enable_chunked_processing': True,
                'chunk_size': 50000
            }
        }
    
    def _load_config_file(self):
        """从文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 合并配置
            self._merge_config(file_config)
            self.logger.info(f"成功加载配置文件: {self.config_file}")
            
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}，使用默认配置")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """合并配置"""
        for key, value in file_config.items():
            if key in self._config:
                if isinstance(value, dict) and isinstance(self._config[key], dict):
                    self._config[key].update(value)
                else:
                    self._config[key] = value
            else:
                self._config[key] = value
    
    def get_factor_groups(self) -> Dict[str, Dict[str, Any]]:
        """获取所有因子组配置"""
        return self._config['factor_groups']
    
    def get_enabled_factor_groups(self) -> Dict[str, Dict[str, Any]]:
        """获取启用的因子组配置"""
        return {
            name: config for name, config in self._config['factor_groups'].items()
            if config.get('enabled', True)
        }
    
    def get_factor_group_config(self, group_name: str) -> Optional[Dict[str, Any]]:
        """获取指定因子组的配置"""
        return self._config['factor_groups'].get(group_name)
    
    def get_factor_group_features(self, group_name: str) -> List[str]:
        """获取指定因子组的特征列表"""
        config = self.get_factor_group_config(group_name)
        return config.get('features', []) if config else []
    
    def get_factor_group_dependencies(self, group_name: str) -> List[str]:
        """获取指定因子组的依赖关系"""
        config = self.get_factor_group_config(group_name)
        return config.get('dependencies', []) if config else []
    
    def get_factor_group_version(self, group_name: str) -> str:
        """获取指定因子组的版本号"""
        config = self.get_factor_group_config(group_name)
        return config.get('version', 'v1.0') if config else 'v1.0'
    
    def get_factor_group_cache_ttl(self, group_name: str) -> int:
        """获取指定因子组的缓存TTL（天数）"""
        config = self.get_factor_group_config(group_name)
        return config.get('cache_ttl', 7) if config else 7
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """获取全局设置"""
        return self._config.get('global_settings', {}).get(key, default)
    
    def resolve_dependencies(self, target_groups: List[str]) -> List[str]:
        """解析依赖关系，返回正确的计算顺序"""
        if not self.get_global_setting('auto_dependency_resolution', True):
            return target_groups
        
        resolved = []
        visited = set()
        temp_visited = set()
        
        def dfs(group_name: str):
            if group_name in temp_visited:
                raise ValueError(f"检测到循环依赖: {group_name}")
            if group_name in visited:
                return
            
            temp_visited.add(group_name)
            dependencies = self.get_factor_group_dependencies(group_name)
            
            for dep in dependencies:
                if dep in self._config['factor_groups']:
                    dfs(dep)
            
            temp_visited.remove(group_name)
            visited.add(group_name)
            if group_name not in resolved:
                resolved.append(group_name)
        
        for group in target_groups:
            if group in self._config['factor_groups']:
                dfs(group)
        
        return resolved
    
    def get_factor_group_hash(self, group_name: str) -> str:
        """获取因子组的哈希值（用于版本控制）"""
        config = self.get_factor_group_config(group_name)
        if not config:
            return ""
        
        # 使用特征列表和版本生成哈希
        features = sorted(config.get('features', []))
        version = config.get('version', 'v1.0')
        hash_string = f"{group_name}|{version}|{'|'.join(features)}"
        
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def save_config(self, output_file: Optional[str] = None):
        """保存配置到文件"""
        output_path = Path(output_file) if output_file else self.config_file
        if not output_path:
            raise ValueError("未指定输出文件路径")
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            raise
