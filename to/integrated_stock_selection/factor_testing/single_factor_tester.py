#!/usr/bin/env python3
"""
单因子测试器

提供单因子测试的核心功能，包括IC分析、分层回测等
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SingleFactorTester:
    """单因子测试器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def test_single_factor(self, data: pd.DataFrame, factor_name: str, 
                          return_col: str = 'next_day_return', 
                          quantiles: int = 5) -> Dict[str, Any]:
        """
        测试单个因子
        
        Args:
            data: 包含因子和收益率的数据
            factor_name: 因子名称
            return_col: 收益率列名
            quantiles: 分位数数量
            
        Returns:
            测试结果字典
        """
        try:
            self.logger.info(f"开始测试因子: {factor_name}")
            
            # 验证输入数据
            if not self._validate_input_data(data, factor_name, return_col):
                return self._create_error_result(factor_name, "输入数据验证失败")
            
            # 准备测试数据
            test_data = self._prepare_test_data(data, factor_name, return_col)
            
            if test_data.empty:
                return self._create_error_result(factor_name, "测试数据为空")
            
            # 执行各项测试
            results = {
                'factor_name': factor_name,
                'test_date': datetime.now().isoformat(),
                'data_summary': self._get_data_summary(test_data, factor_name, return_col),
                'ic_analysis': self._calculate_ic_analysis(test_data, factor_name, return_col),
                'quantile_analysis': self._calculate_quantile_analysis(test_data, factor_name, return_col, quantiles),
                'distribution_analysis': self._analyze_factor_distribution(test_data, factor_name),
                'stability_analysis': self._analyze_factor_stability(test_data, factor_name, return_col),
                'overall_score': 0.0,
                'recommendation': 'UNKNOWN'
            }
            
            # 计算综合评分
            results['overall_score'] = self._calculate_overall_score(results)
            results['recommendation'] = self._get_recommendation(results['overall_score'])
            
            self.logger.info(f"因子 {factor_name} 测试完成，综合评分: {results['overall_score']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"测试因子 {factor_name} 失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return self._create_error_result(factor_name, str(e))
    
    def _validate_input_data(self, data: pd.DataFrame, factor_name: str, return_col: str) -> bool:
        """验证输入数据"""
        try:
            if data.empty:
                self.logger.error("输入数据为空")
                return False
            
            required_columns = ['stock_code', 'date', factor_name, return_col]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"缺少必需列: {missing_columns}")
                return False
            
            # 检查因子数据质量
            factor_data = data[factor_name].dropna()
            if len(factor_data) == 0:
                self.logger.error(f"因子 {factor_name} 没有有效数据")
                return False
            
            if len(factor_data) < 100:
                self.logger.warning(f"因子 {factor_name} 有效数据量较少: {len(factor_data)}")
            
            # 检查收益率数据
            return_data = data[return_col].dropna()
            if len(return_data) == 0:
                self.logger.error(f"收益率列 {return_col} 没有有效数据")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证输入数据失败: {e}")
            return False
    
    def _prepare_test_data(self, data: pd.DataFrame, factor_name: str, return_col: str) -> pd.DataFrame:
        """准备测试数据"""
        try:
            # 只保留有效的因子和收益率数据
            test_data = data[[
                'stock_code', 'date', factor_name, return_col
            ]].copy()
            
            # 移除缺失值
            test_data = test_data.dropna(subset=[factor_name, return_col])
            
            # 移除无穷大值
            test_data = test_data[
                np.isfinite(test_data[factor_name]) & 
                np.isfinite(test_data[return_col])
            ]
            
            # 确保日期格式正确
            test_data['date'] = pd.to_datetime(test_data['date'])
            
            # 按日期排序
            test_data = test_data.sort_values(['date', 'stock_code'])
            
            self.logger.debug(f"测试数据准备完成: {len(test_data)} 条记录")
            return test_data
            
        except Exception as e:
            self.logger.error(f"准备测试数据失败: {e}")
            return pd.DataFrame()
    
    def _get_data_summary(self, data: pd.DataFrame, factor_name: str, return_col: str) -> Dict[str, Any]:
        """获取数据摘要"""
        try:
            summary = {
                'total_records': len(data),
                'unique_stocks': data['stock_code'].nunique(),
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d'),
                    'trading_days': data['date'].nunique()
                },
                'factor_stats': {
                    'mean': float(data[factor_name].mean()),
                    'std': float(data[factor_name].std()),
                    'min': float(data[factor_name].min()),
                    'max': float(data[factor_name].max()),
                    'median': float(data[factor_name].median()),
                    'skewness': float(data[factor_name].skew()),
                    'kurtosis': float(data[factor_name].kurtosis())
                },
                'return_stats': {
                    'mean': float(data[return_col].mean()),
                    'std': float(data[return_col].std()),
                    'min': float(data[return_col].min()),
                    'max': float(data[return_col].max()),
                    'median': float(data[return_col].median())
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取数据摘要失败: {e}")
            return {}
    
    def _calculate_ic_analysis(self, data: pd.DataFrame, factor_name: str, return_col: str) -> Dict[str, Any]:
        """计算IC分析"""
        try:
            ic_results = {
                'overall_ic': 0.0,
                'ic_std': 0.0,
                'ic_ir': 0.0,
                'positive_ic_ratio': 0.0,
                'monthly_ic': {},
                'rolling_ic': []
            }
            
            # 计算整体IC
            overall_corr = data[factor_name].corr(data[return_col])
            if not np.isnan(overall_corr):
                ic_results['overall_ic'] = float(overall_corr)
            
            # 按日期计算IC
            daily_ic = []
            for date in data['date'].unique():
                date_data = data[data['date'] == date]
                if len(date_data) >= 10:  # 至少10只股票
                    daily_corr = date_data[factor_name].corr(date_data[return_col])
                    if not np.isnan(daily_corr):
                        daily_ic.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'ic': float(daily_corr),
                            'stock_count': len(date_data)
                        })
            
            if daily_ic:
                ic_values = [item['ic'] for item in daily_ic]
                ic_results['ic_std'] = float(np.std(ic_values))
                ic_results['ic_ir'] = float(np.mean(ic_values) / np.std(ic_values)) if np.std(ic_values) > 0 else 0.0
                ic_results['positive_ic_ratio'] = float(np.mean([ic > 0 for ic in ic_values]))
                ic_results['rolling_ic'] = daily_ic
                
                # 按月统计IC
                ic_df = pd.DataFrame(daily_ic)
                ic_df['date'] = pd.to_datetime(ic_df['date'])
                ic_df['month'] = ic_df['date'].dt.to_period('M')
                
                monthly_ic = ic_df.groupby('month')['ic'].agg(['mean', 'std', 'count']).to_dict('index')
                ic_results['monthly_ic'] = {
                    str(month): {
                        'mean_ic': float(stats['mean']),
                        'std_ic': float(stats['std']) if not np.isnan(stats['std']) else 0.0,
                        'trading_days': int(stats['count'])
                    }
                    for month, stats in monthly_ic.items()
                }
            
            return ic_results
            
        except Exception as e:
            self.logger.error(f"计算IC分析失败: {e}")
            return {'overall_ic': 0.0, 'error': str(e)}
    
    def _calculate_quantile_analysis(self, data: pd.DataFrame, factor_name: str, 
                                   return_col: str, quantiles: int) -> Dict[str, Any]:
        """计算分位数分析"""
        try:
            quantile_results = {
                'quantile_returns': {},
                'long_short_return': 0.0,
                'monotonicity': 0.0,
                'top_bottom_ratio': 0.0
            }
            
            # 按日期分组计算分位数
            all_quantile_returns = []
            
            for date in data['date'].unique():
                date_data = data[data['date'] == date].copy()
                
                if len(date_data) >= quantiles * 2:  # 确保每个分位数至少有2只股票
                    # 计算分位数标签
                    date_data['quantile'] = pd.qcut(
                        date_data[factor_name], 
                        q=quantiles, 
                        labels=False, 
                        duplicates='drop'
                    )
                    
                    # 计算每个分位数的平均收益率
                    quantile_returns = date_data.groupby('quantile')[return_col].mean()
                    
                    for q in range(quantiles):
                        if q in quantile_returns.index:
                            all_quantile_returns.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'quantile': q,
                                'return': float(quantile_returns[q]),
                                'stock_count': int((date_data['quantile'] == q).sum())
                            })
            
            if all_quantile_returns:
                # 计算平均分位数收益率
                quantile_df = pd.DataFrame(all_quantile_returns)
                avg_returns = quantile_df.groupby('quantile')['return'].mean()
                
                quantile_results['quantile_returns'] = {
                    f'Q{q+1}': float(avg_returns[q]) if q in avg_returns.index else 0.0
                    for q in range(quantiles)
                }
                
                # 计算多空收益率（最高分位数 - 最低分位数）
                if len(avg_returns) >= 2:
                    top_return = avg_returns.iloc[-1]
                    bottom_return = avg_returns.iloc[0]
                    quantile_results['long_short_return'] = float(top_return - bottom_return)
                    
                    # 计算单调性（相邻分位数收益率差的符号一致性）
                    return_diffs = avg_returns.diff().dropna()
                    if len(return_diffs) > 0:
                        monotonicity = np.mean(np.sign(return_diffs) == np.sign(return_diffs.iloc[0]))
                        quantile_results['monotonicity'] = float(monotonicity)
                    
                    # 计算头尾比率
                    if bottom_return != 0:
                        quantile_results['top_bottom_ratio'] = float(top_return / bottom_return)
            
            return quantile_results
            
        except Exception as e:
            self.logger.error(f"计算分位数分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_factor_distribution(self, data: pd.DataFrame, factor_name: str) -> Dict[str, Any]:
        """分析因子分布"""
        try:
            factor_data = data[factor_name]
            
            distribution_analysis = {
                'percentiles': {
                    'p1': float(factor_data.quantile(0.01)),
                    'p5': float(factor_data.quantile(0.05)),
                    'p10': float(factor_data.quantile(0.10)),
                    'p25': float(factor_data.quantile(0.25)),
                    'p50': float(factor_data.quantile(0.50)),
                    'p75': float(factor_data.quantile(0.75)),
                    'p90': float(factor_data.quantile(0.90)),
                    'p95': float(factor_data.quantile(0.95)),
                    'p99': float(factor_data.quantile(0.99))
                },
                'outlier_analysis': self._detect_outliers(factor_data),
                'unique_values': int(factor_data.nunique()),
                'zero_ratio': float((factor_data == 0).mean()),
                'extreme_value_ratio': 0.0
            }
            
            # 计算极端值比例（超过3个标准差）
            mean_val = factor_data.mean()
            std_val = factor_data.std()
            if std_val > 0:
                extreme_mask = np.abs(factor_data - mean_val) > 3 * std_val
                distribution_analysis['extreme_value_ratio'] = float(extreme_mask.mean())
            
            return distribution_analysis
            
        except Exception as e:
            self.logger.error(f"分析因子分布失败: {e}")
            return {'error': str(e)}
    
    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """检测异常值"""
        try:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (data < lower_bound) | (data > upper_bound)
            
            return {
                'outlier_count': int(outliers.sum()),
                'outlier_ratio': float(outliers.mean()),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
        except Exception:
            return {'outlier_count': 0, 'outlier_ratio': 0.0}
    
    def _analyze_factor_stability(self, data: pd.DataFrame, factor_name: str, return_col: str) -> Dict[str, Any]:
        """分析因子稳定性"""
        try:
            stability_analysis = {
                'ic_stability': 0.0,
                'return_stability': 0.0,
                'factor_autocorr': 0.0,
                'decay_analysis': {}
            }
            
            # 按月分析IC稳定性
            data['month'] = data['date'].dt.to_period('M')
            monthly_ic = []
            
            for month in data['month'].unique():
                month_data = data[data['month'] == month]
                if len(month_data) >= 30:  # 至少30个观测值
                    monthly_corr = month_data[factor_name].corr(month_data[return_col])
                    if not np.isnan(monthly_corr):
                        monthly_ic.append(monthly_corr)
            
            if len(monthly_ic) >= 3:
                stability_analysis['ic_stability'] = float(1.0 - np.std(monthly_ic) / (np.abs(np.mean(monthly_ic)) + 0.01))
            
            return stability_analysis
            
        except Exception as e:
            self.logger.error(f"分析因子稳定性失败: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算综合评分"""
        try:
            score = 0.0
            
            # IC分析权重 40%
            ic_analysis = results.get('ic_analysis', {})
            overall_ic = abs(ic_analysis.get('overall_ic', 0))
            ic_ir = abs(ic_analysis.get('ic_ir', 0))
            positive_ic_ratio = ic_analysis.get('positive_ic_ratio', 0.5)
            
            ic_score = (overall_ic * 0.5 + ic_ir * 0.3 + (positive_ic_ratio - 0.5) * 0.2) * 0.4
            
            # 分位数分析权重 30%
            quantile_analysis = results.get('quantile_analysis', {})
            long_short_return = abs(quantile_analysis.get('long_short_return', 0))
            monotonicity = quantile_analysis.get('monotonicity', 0)
            
            quantile_score = (long_short_return * 100 * 0.6 + monotonicity * 0.4) * 0.3
            
            # 稳定性分析权重 20%
            stability_analysis = results.get('stability_analysis', {})
            ic_stability = stability_analysis.get('ic_stability', 0)
            
            stability_score = ic_stability * 0.2
            
            # 分布分析权重 10%
            distribution_analysis = results.get('distribution_analysis', {})
            outlier_ratio = distribution_analysis.get('outlier_analysis', {}).get('outlier_ratio', 0)
            
            distribution_score = (1.0 - min(outlier_ratio, 0.2) / 0.2) * 0.1
            
            score = ic_score + quantile_score + stability_score + distribution_score
            return min(max(score, 0.0), 1.0)  # 限制在0-1之间
            
        except Exception as e:
            self.logger.error(f"计算综合评分失败: {e}")
            return 0.0
    
    def _get_recommendation(self, score: float) -> str:
        """获取推荐等级"""
        if score >= 0.8:
            return 'EXCELLENT'
        elif score >= 0.6:
            return 'GOOD'
        elif score >= 0.4:
            return 'FAIR'
        elif score >= 0.2:
            return 'POOR'
        else:
            return 'VERY_POOR'
    
    def _create_error_result(self, factor_name: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'factor_name': factor_name,
            'test_date': datetime.now().isoformat(),
            'error': error_msg,
            'overall_score': 0.0,
            'recommendation': 'ERROR'
        }
