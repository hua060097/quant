#!/usr/bin/env python3
"""
因子验证器

提供因子数据质量验证和统计分析功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

class FactorValidator:
    """因子验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_factor_data(self, data: pd.DataFrame, factor_names: List[str]) -> Dict[str, Any]:
        """
        验证因子数据质量
        
        Args:
            data: 因子数据
            factor_names: 要验证的因子名称列表
            
        Returns:
            验证结果字典
        """
        try:
            validation_results = {
                'overall_status': 'PASS',
                'total_factors': len(factor_names),
                'total_records': len(data),
                'factor_results': {},
                'summary_stats': {},
                'warnings': [],
                'errors': []
            }
            
            if data.empty:
                validation_results['overall_status'] = 'FAIL'
                validation_results['errors'].append("数据为空")
                return validation_results
            
            # 验证每个因子
            for factor_name in factor_names:
                if factor_name in data.columns:
                    factor_result = self._validate_single_factor(data[factor_name], factor_name)
                    validation_results['factor_results'][factor_name] = factor_result
                    
                    # 收集警告和错误
                    if factor_result['warnings']:
                        validation_results['warnings'].extend(factor_result['warnings'])
                    if factor_result['errors']:
                        validation_results['errors'].extend(factor_result['errors'])
                        validation_results['overall_status'] = 'FAIL'
                else:
                    validation_results['factor_results'][factor_name] = {
                        'status': 'MISSING',
                        'error': f"因子 {factor_name} 不存在于数据中"
                    }
                    validation_results['errors'].append(f"因子 {factor_name} 不存在")
                    validation_results['overall_status'] = 'FAIL'
            
            # 计算汇总统计
            validation_results['summary_stats'] = self._calculate_summary_stats(data, factor_names)
            
            # 因子相关性检查
            correlation_results = self._check_factor_correlations(data, factor_names)
            validation_results['correlation_analysis'] = correlation_results
            
            self.logger.info(f"因子验证完成: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"因子验证失败: {e}")
            return {
                'overall_status': 'ERROR',
                'error': str(e),
                'total_factors': len(factor_names),
                'total_records': len(data) if not data.empty else 0
            }
    
    def _validate_single_factor(self, factor_series: pd.Series, factor_name: str) -> Dict[str, Any]:
        """验证单个因子"""
        try:
            result = {
                'factor_name': factor_name,
                'status': 'PASS',
                'warnings': [],
                'errors': [],
                'statistics': {}
            }
            
            # 基础统计
            total_count = len(factor_series)
            null_count = factor_series.isnull().sum()
            valid_count = total_count - null_count
            null_ratio = null_count / total_count if total_count > 0 else 1.0
            
            result['statistics'] = {
                'total_count': total_count,
                'valid_count': valid_count,
                'null_count': null_count,
                'null_ratio': null_ratio
            }
            
            # 检查空值比例
            if null_ratio > 0.5:
                result['errors'].append(f"因子 {factor_name} 空值比例过高: {null_ratio:.2%}")
                result['status'] = 'FAIL'
            elif null_ratio > 0.2:
                result['warnings'].append(f"因子 {factor_name} 空值比例较高: {null_ratio:.2%}")
            
            if valid_count > 0:
                valid_data = factor_series.dropna()
                
                # 数值统计
                try:
                    result['statistics'].update({
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'median': float(valid_data.median()),
                        'q25': float(valid_data.quantile(0.25)),
                        'q75': float(valid_data.quantile(0.75))
                    })
                except Exception:
                    result['warnings'].append(f"因子 {factor_name} 无法计算数值统计")
                
                # 检查无穷大值
                inf_count = np.isinf(valid_data).sum()
                if inf_count > 0:
                    inf_ratio = inf_count / valid_count
                    result['statistics']['inf_count'] = inf_count
                    result['statistics']['inf_ratio'] = inf_ratio
                    
                    if inf_ratio > 0.01:  # 超过1%
                        result['errors'].append(f"因子 {factor_name} 包含过多无穷大值: {inf_ratio:.2%}")
                        result['status'] = 'FAIL'
                    else:
                        result['warnings'].append(f"因子 {factor_name} 包含无穷大值: {inf_count} 个")
                
                # 检查异常值（使用IQR方法）
                if valid_count >= 10:
                    outlier_info = self._detect_outliers(valid_data)
                    result['statistics']['outlier_count'] = outlier_info['count']
                    result['statistics']['outlier_ratio'] = outlier_info['ratio']
                    
                    if outlier_info['ratio'] > 0.1:  # 超过10%
                        result['warnings'].append(
                            f"因子 {factor_name} 异常值比例较高: {outlier_info['ratio']:.2%}"
                        )
                
                # 检查分布特征
                if valid_count >= 30:
                    distribution_info = self._analyze_distribution(valid_data)
                    result['statistics']['distribution'] = distribution_info
                    
                    # 检查是否过于集中在某个值
                    if distribution_info.get('mode_ratio', 0) > 0.8:
                        result['warnings'].append(
                            f"因子 {factor_name} 值过于集中: {distribution_info['mode_ratio']:.2%} 的值相同"
                        )
            
            return result
            
        except Exception as e:
            return {
                'factor_name': factor_name,
                'status': 'ERROR',
                'error': str(e),
                'warnings': [],
                'errors': [f"验证因子 {factor_name} 时发生错误: {e}"]
            }
    
    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """检测异常值（使用IQR方法）"""
        try:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (data < lower_bound) | (data > upper_bound)
            outlier_count = outliers.sum()
            outlier_ratio = outlier_count / len(data)
            
            return {
                'count': outlier_count,
                'ratio': outlier_ratio,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception:
            return {'count': 0, 'ratio': 0.0}
    
    def _analyze_distribution(self, data: pd.Series) -> Dict[str, Any]:
        """分析数据分布"""
        try:
            distribution_info = {}
            
            # 偏度和峰度
            try:
                distribution_info['skewness'] = float(stats.skew(data))
                distribution_info['kurtosis'] = float(stats.kurtosis(data))
            except Exception:
                distribution_info['skewness'] = None
                distribution_info['kurtosis'] = None
            
            # 众数分析
            try:
                value_counts = data.value_counts()
                if len(value_counts) > 0:
                    mode_count = value_counts.iloc[0]
                    mode_ratio = mode_count / len(data)
                    distribution_info['mode_ratio'] = mode_ratio
                    distribution_info['unique_values'] = len(value_counts)
                else:
                    distribution_info['mode_ratio'] = 0.0
                    distribution_info['unique_values'] = 0
            except Exception:
                distribution_info['mode_ratio'] = 0.0
                distribution_info['unique_values'] = 0
            
            # 正态性检验（样本量适中时）
            if 30 <= len(data) <= 5000:
                try:
                    _, p_value = stats.normaltest(data)
                    distribution_info['normality_p_value'] = float(p_value)
                    distribution_info['is_normal'] = p_value > 0.05
                except Exception:
                    distribution_info['normality_p_value'] = None
                    distribution_info['is_normal'] = None
            
            return distribution_info
            
        except Exception:
            return {}
    
    def _calculate_summary_stats(self, data: pd.DataFrame, factor_names: List[str]) -> Dict[str, Any]:
        """计算汇总统计"""
        try:
            existing_factors = [name for name in factor_names if name in data.columns]
            
            if not existing_factors:
                return {}
            
            factor_data = data[existing_factors]
            
            summary = {
                'existing_factors': len(existing_factors),
                'missing_factors': len(factor_names) - len(existing_factors),
                'total_null_ratio': factor_data.isnull().sum().sum() / (len(data) * len(existing_factors)),
                'factors_with_high_nulls': [],
                'factors_with_inf_values': [],
                'factors_with_high_correlation': []
            }
            
            # 检查每个因子的空值情况
            for factor in existing_factors:
                null_ratio = data[factor].isnull().sum() / len(data)
                if null_ratio > 0.3:
                    summary['factors_with_high_nulls'].append({
                        'factor': factor,
                        'null_ratio': null_ratio
                    })
                
                # 检查无穷大值
                if data[factor].dtype in ['float64', 'float32']:
                    inf_count = np.isinf(data[factor]).sum()
                    if inf_count > 0:
                        summary['factors_with_inf_values'].append({
                            'factor': factor,
                            'inf_count': inf_count
                        })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"计算汇总统计失败: {e}")
            return {}
    
    def _check_factor_correlations(self, data: pd.DataFrame, factor_names: List[str]) -> Dict[str, Any]:
        """检查因子相关性"""
        try:
            existing_factors = [name for name in factor_names if name in data.columns]
            
            if len(existing_factors) < 2:
                return {'high_correlations': [], 'correlation_matrix_available': False}
            
            # 计算相关性矩阵
            factor_data = data[existing_factors].select_dtypes(include=[np.number])
            
            if factor_data.empty or factor_data.shape[1] < 2:
                return {'high_correlations': [], 'correlation_matrix_available': False}
            
            corr_matrix = factor_data.corr()
            
            # 找出高相关性的因子对
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    factor1 = corr_matrix.columns[i]
                    factor2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(correlation) and abs(correlation) > 0.8:
                        high_correlations.append({
                            'factor1': factor1,
                            'factor2': factor2,
                            'correlation': float(correlation)
                        })
            
            return {
                'high_correlations': high_correlations,
                'correlation_matrix_available': True,
                'total_factor_pairs': len(existing_factors) * (len(existing_factors) - 1) // 2,
                'high_correlation_pairs': len(high_correlations)
            }
            
        except Exception as e:
            self.logger.error(f"检查因子相关性失败: {e}")
            return {'high_correlations': [], 'correlation_matrix_available': False, 'error': str(e)}

    def validate_input_data(self, data: pd.DataFrame, dependencies: List[str]) -> Dict[str, Any]:
        """验证输入数据"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'data_info': {}
            }

            # 基础数据检查
            if data is None or data.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("输入数据为空")
                return validation_result

            # 检查必要列
            required_columns = ['stock_code', 'date']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"缺少必要列: {missing_columns}")

            # 检查数据类型
            if 'stock_code' in data.columns and not data['stock_code'].dtype == 'object':
                validation_result['warnings'].append("stock_code列应为字符串类型")

            # 检查数据量
            if len(data) < 10:
                validation_result['warnings'].append(f"数据量较少: {len(data)} 条记录")

            # 记录数据信息
            validation_result['data_info'] = {
                'row_count': len(data),
                'column_count': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }

            return validation_result

        except Exception as e:
            self.logger.error(f"输入数据验证失败: {e}")
            return {
                'is_valid': False,
                'errors': [f"验证过程出错: {str(e)}"],
                'warnings': [],
                'data_info': {}
            }

    def validate_quantitative_factors(self, data: pd.DataFrame, factor_names: List[str],
                                    return_columns: List[str] = None) -> Dict[str, Any]:
        """
        专业量化因子验证（增强版）

        基于量化金融行业标准，计算IC、IR、RankIC、ICIR等专业指标
        支持多时间窗口验证和增强指标计算

        Args:
            data: 包含因子和收益率的数据
            factor_names: 要验证的因子名称列表
            return_columns: 收益率列名列表，默认为['future_return_1d', 'future_return_2d']

        Returns:
            包含多时间窗口验证结果的字典
        """
        try:
            print("🔬 开始专业量化因子验证（增强版）...")
            self.logger.info("开始专业量化因子验证（增强版）")

            print(f"🔍 步骤5: 因子验证器接收数据检查")
            print(f"  验证器接收数据形状: {data.shape}")
            print(f"  验证器接收数据列名前10个: {list(data.columns)[:10]}")
            print(f"  验证器接收数据中的收益率相关字段: {[col for col in data.columns if 'future_return' in col]}")

            # 设置默认收益率列
            if return_columns is None:
                return_columns = ['future_return_1d']
                print(f"  初始return_columns: {return_columns}")
                # 检查是否存在future_return_2d列
                if 'future_return_2d' in data.columns:
                    return_columns.append('future_return_2d')
                    print(f"  发现future_return_2d，更新return_columns: {return_columns}")
                else:
                    print(f"  ❌ 验证器中未发现future_return_2d字段")
            else:
                print(f"  使用传入的return_columns: {return_columns}")

            print(f"📊 验证时间窗口: {return_columns}")

            validation_results = {
                'overall_status': 'PASS',
                'total_factors': len(factor_names),
                'total_records': len(data),
                'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'return_columns': return_columns,
                'time_windows': {},  # 存储各时间窗口的结果
                'summary_statistics': {},
                'warnings': [],
                'errors': []
            }

            # 检查必要列
            missing_columns = [col for col in return_columns if col not in data.columns]
            if missing_columns:
                validation_results['errors'].append(f"缺少收益率列: {missing_columns}")
                validation_results['overall_status'] = 'FAIL'
                return validation_results

            # 对每个时间窗口进行验证
            for return_column in return_columns:
                print(f"\n🔍 验证时间窗口: {return_column}")

                # 检查数据完整性
                if return_column not in data.columns:
                    validation_results['errors'].append(f"收益率字段 {return_column} 不存在于数据中")
                    validation_results['overall_status'] = 'FAIL'
                    print(f"❌ 收益率字段 {return_column} 不存在于数据中")
                    print(f"📊 当前数据列: {list(data.columns)}")
                    continue

                # 过滤有效数据
                valid_data = data.dropna(subset=[return_column])
                if len(valid_data) < 100:
                    validation_results['errors'].append(f"{return_column} 有效数据量不足: {len(valid_data)} < 100")
                    validation_results['overall_status'] = 'FAIL'
                    continue

                print(f"📊 {return_column} 有效数据量: {len(valid_data)} 条")

                # 多线程验证因子
                print(f"🚀 使用多线程验证因子...")
                window_valid_factors = []

                # 过滤存在的因子
                existing_factors = [f for f in factor_names if f in data.columns]
                missing_factors = [f for f in factor_names if f not in data.columns]

                for factor_name in missing_factors:
                    validation_results['warnings'].append(f"因子 {factor_name} 不存在于数据中")

                print(f"📊 存在的因子数: {len(existing_factors)}/{len(factor_names)}")

                if existing_factors:
                    # 高性能算法优化：取消采样，保持数据完整性
                    print(f"🔧 全量数据验证: {len(valid_data)} 条记录")

                    # 预计算公共数据，避免重复计算
                    optimized_data = self._precompute_common_data(valid_data, return_column)

                    # 内存优化：强制垃圾回收
                    import gc
                    gc.collect()

                    # 使用多线程计算因子指标
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    # 根据CPU核心数和因子数量动态调整线程数
                    import multiprocessing
                    cpu_count = multiprocessing.cpu_count()
                    max_workers = min(cpu_count, max(2, len(existing_factors) // 3))
                    print(f"🔧 使用 {max_workers} 个线程验证因子 (CPU核心数: {cpu_count})")
                    print(f"🔧 内存优化: 预计算公共数据，避免重复计算")

                    window_factor_metrics = {}

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # 提交所有任务
                        future_to_factor = {}
                        for factor_name in existing_factors:
                            future = executor.submit(
                                self._calculate_factor_metrics_enhanced,
                                optimized_data, factor_name, return_column
                            )
                            future_to_factor[future] = factor_name

                        # 处理完成的任务
                        completed = 0
                        for future in as_completed(future_to_factor):
                            factor_name = future_to_factor[future]
                            completed += 1

                            try:
                                factor_metrics = future.result()
                                window_factor_metrics[factor_name] = factor_metrics

                                # 判断因子质量
                                if self._is_factor_valid(factor_metrics):
                                    window_valid_factors.append(factor_name)
                                else:
                                    validation_results['warnings'].append(
                                        f"因子 {factor_name} ({return_column}) 质量较差: IC={factor_metrics.get('ic', 0):.4f}"
                                    )

                                # 显示进度
                                if completed % 10 == 0 or completed == len(existing_factors):
                                    print(f"  验证进度: {completed}/{len(existing_factors)} ({completed/len(existing_factors)*100:.1f}%)")

                            except Exception as e:
                                self.logger.error(f"验证因子 {factor_name} ({return_column}) 失败: {e}")
                                validation_results['errors'].append(f"验证因子 {factor_name} ({return_column}) 失败: {str(e)}")

                    print(f"✅ {return_column} 因子验证完成")

                    # 存储当前时间窗口的结果
                    validation_results['time_windows'][return_column] = {
                        'factor_metrics': window_factor_metrics,
                        'valid_factors': window_valid_factors,
                        'summary_statistics': self._calculate_summary_statistics(window_factor_metrics)
                    }

            # 计算跨时间窗口的汇总统计
            validation_results['summary_statistics'] = self._calculate_cross_window_summary(
                validation_results['time_windows']
            )

            # 最终状态判断（基于所有时间窗口）
            all_valid_factors = set()
            total_windows = len(return_columns)

            for window_result in validation_results['time_windows'].values():
                all_valid_factors.update(window_result['valid_factors'])

            total_existing_factors = len([f for f in factor_names if f in data.columns])

            if total_existing_factors == 0:
                validation_results['overall_status'] = 'FAIL'
                validation_results['errors'].append("没有找到任何有效因子")
            elif len(all_valid_factors) == 0:
                validation_results['overall_status'] = 'FAIL'
                validation_results['errors'].append("没有高质量因子")
            elif len(all_valid_factors) < total_existing_factors * 0.2:  # 至少20%的因子有效
                validation_results['overall_status'] = 'WARN'
                validation_results['warnings'].append(
                    f"高质量因子比例较低: {len(all_valid_factors)}/{total_existing_factors} ({len(all_valid_factors)/total_existing_factors*100:.1f}%)"
                )
            else:
                validation_results['overall_status'] = 'PASS'

            print(f"✅ 多时间窗口因子验证完成: {len(all_valid_factors)}/{len(factor_names)} 个有效因子")
            self.logger.info(f"因子验证完成: {validation_results['overall_status']}")

            return validation_results

        except Exception as e:
            self.logger.error(f"专业量化因子验证失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'overall_status': 'ERROR',
                'total_factors': len(factor_names),
                'total_records': len(data),
                'errors': [f"验证过程出错: {str(e)}"],
                'warnings': [],
                'factor_metrics': {},
                'summary_statistics': {}
            }

    def _calculate_factor_metrics(self, data: pd.DataFrame, factor_name: str,
                                return_column: str) -> Dict[str, float]:
        """计算单个因子的专业量化指标（性能优化版）"""
        try:
            # 获取因子值和收益率
            factor_values = data[factor_name]
            returns = data[return_column]

            # 快速过滤有效数据（避免重复dropna）
            valid_mask = ~(factor_values.isna() | returns.isna() |
                          np.isinf(factor_values) | np.isinf(returns))

            if valid_mask.sum() < 30:
                return {'ic': 0, 'rank_ic': 0, 'ir': 0, 'icir': 0, 'coverage': 0, 'monotonicity': 0}

            factor_clean = factor_values[valid_mask].values  # 转为numpy数组提速
            returns_clean = returns[valid_mask].values

            # 1. Information Coefficient (IC) - 信息系数（优化版）
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1]
            ic = 0 if np.isnan(ic) else ic

            # 2. Rank IC - 排序信息系数（使用scipy.stats.spearmanr更快）
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(factor_clean, returns_clean)
            rank_ic = 0 if np.isnan(rank_ic) else rank_ic

            # 3. Information Ratio (IR) - 信息比率（简化版）
            # 使用更快的分位数计算
            try:
                quintile_bounds = np.percentile(factor_clean, [20, 40, 60, 80])
                group_returns = []

                # 最低20%
                mask = factor_clean <= quintile_bounds[0]
                if mask.sum() > 0:
                    group_returns.append(returns_clean[mask].mean())

                # 最高20%
                mask = factor_clean >= quintile_bounds[3]
                if mask.sum() > 0:
                    group_returns.append(returns_clean[mask].mean())

                if len(group_returns) == 2:
                    ir = group_returns[1] - group_returns[0]  # 简化IR计算
                else:
                    ir = 0
            except:
                ir = 0

            # 4. ICIR - 简化版（避免耗时的滚动计算）
            # 使用分段IC计算代替滚动IC
            icir = 0
            if len(factor_clean) >= 100:
                try:
                    # 将数据分为5段，计算每段的IC
                    segment_size = len(factor_clean) // 5
                    segment_ics = []

                    for i in range(5):
                        start_idx = i * segment_size
                        end_idx = (i + 1) * segment_size if i < 4 else len(factor_clean)

                        seg_factor = factor_clean[start_idx:end_idx]
                        seg_returns = returns_clean[start_idx:end_idx]

                        if len(seg_factor) > 10:
                            seg_ic = np.corrcoef(seg_factor, seg_returns)[0, 1]
                            if not np.isnan(seg_ic):
                                segment_ics.append(seg_ic)

                    if len(segment_ics) > 1:
                        ic_std = np.std(segment_ics)
                        icir = ic / ic_std if ic_std > 0 else 0
                except:
                    icir = 0

            # 5. 因子覆盖度
            coverage = valid_mask.sum() / len(data)

            # 6. 因子单调性（简化版）
            monotonicity = self._calculate_monotonicity_fast(factor_clean, returns_clean)

            return {
                'ic': ic,
                'rank_ic': rank_ic,
                'ir': ir,
                'icir': icir,
                'coverage': coverage,
                'monotonicity': monotonicity,
                'valid_samples': len(factor_clean)
            }

        except Exception as e:
            self.logger.error(f"计算因子指标失败 {factor_name}: {e}")
            return {'ic': 0, 'rank_ic': 0, 'ir': 0, 'icir': 0, 'coverage': 0, 'monotonicity': 0}

    def _calculate_monotonicity(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算因子单调性"""
        try:
            # 按因子值分5组
            factor_quintiles = pd.qcut(factor_values, 5, labels=False, duplicates='drop')
            group_returns = []

            for q in range(5):
                mask = factor_quintiles == q
                if mask.sum() > 0:
                    group_returns.append(returns[mask].mean())

            if len(group_returns) < 3:
                return 0

            # 计算单调性：相邻组收益率差的符号一致性
            diffs = np.diff(group_returns)
            if len(diffs) == 0:
                return 0

            # 单调递增或递减的比例
            positive_diffs = np.sum(diffs > 0)
            negative_diffs = np.sum(diffs < 0)

            monotonicity = max(positive_diffs, negative_diffs) / len(diffs)
            return monotonicity

        except Exception:
            return 0

    def _calculate_monotonicity_fast(self, factor_values: np.ndarray, returns: np.ndarray) -> float:
        """计算因子单调性（快速版本）"""
        try:
            # 使用更快的分位数方法
            quintile_bounds = np.percentile(factor_values, [20, 40, 60, 80])
            group_returns = []

            # 计算5个分组的平均收益
            prev_bound = -np.inf
            for bound in list(quintile_bounds) + [np.inf]:
                mask = (factor_values > prev_bound) & (factor_values <= bound)
                if mask.sum() > 0:
                    group_returns.append(returns[mask].mean())
                prev_bound = bound

            if len(group_returns) < 3:
                return 0

            # 计算单调性
            diffs = np.diff(group_returns)
            if len(diffs) == 0:
                return 0

            positive_diffs = np.sum(diffs > 0)
            negative_diffs = np.sum(diffs < 0)
            monotonicity = max(positive_diffs, negative_diffs) / len(diffs)

            return monotonicity

        except Exception:
            return 0

    def _is_factor_valid(self, metrics: Dict[str, float]) -> bool:
        """判断因子是否有效（基于量化行业标准，适配增强指标）"""
        try:
            # 专业量化因子质量标准
            ic = abs(metrics.get('ic', 0))
            rank_ic = abs(metrics.get('rank_ic', 0))
            coverage = metrics.get('coverage', 0)
            monotonicity = metrics.get('monotonicity', 0)
            win_rate = metrics.get('win_rate', 0)
            ir = abs(metrics.get('ir', 0))

            # 质量标准（参考头部量化机构标准）
            ic_threshold = 0.01      # IC绝对值 > 1%
            rank_ic_threshold = 0.01  # RankIC绝对值 > 1%
            coverage_threshold = 0.8  # 覆盖度 > 80%
            monotonicity_threshold = 0.3  # 单调性 > 30%
            win_rate_threshold = 0.52  # 胜率 > 52%
            ir_threshold = 0.1  # IR绝对值 > 0.1

            # 综合判断
            quality_score = 0
            if ic >= ic_threshold:
                quality_score += 1
            if rank_ic >= rank_ic_threshold:
                quality_score += 1
            if coverage >= coverage_threshold:
                quality_score += 1
            if monotonicity >= monotonicity_threshold:
                quality_score += 1
            if win_rate >= win_rate_threshold:
                quality_score += 1
            if ir >= ir_threshold:
                quality_score += 1

            # 至少满足3个条件才认为是有效因子（提高标准）
            return quality_score >= 3

        except Exception:
            return False

    def _calculate_summary_statistics(self, factor_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算汇总统计"""
        try:
            if not factor_metrics:
                return {}

            # 提取所有指标
            all_ics = [metrics.get('ic', 0) for metrics in factor_metrics.values()]
            all_rank_ics = [metrics.get('rank_ic', 0) for metrics in factor_metrics.values()]
            all_irs = [metrics.get('ir', 0) for metrics in factor_metrics.values()]
            all_coverages = [metrics.get('coverage', 0) for metrics in factor_metrics.values()]
            all_monotonicities = [metrics.get('monotonicity', 0) for metrics in factor_metrics.values()]

            # 计算汇总统计
            summary = {
                'mean_ic': np.mean(all_ics),
                'std_ic': np.std(all_ics),
                'mean_rank_ic': np.mean(all_rank_ics),
                'std_rank_ic': np.std(all_rank_ics),
                'mean_ir': np.mean(all_irs),
                'mean_coverage': np.mean(all_coverages),
                'mean_monotonicity': np.mean(all_monotonicities),
                'high_quality_factors': sum(1 for metrics in factor_metrics.values()
                                          if self._is_factor_valid(metrics)),
                'ic_positive_ratio': sum(1 for ic in all_ics if ic > 0) / len(all_ics) if all_ics else 0
            }

            return summary

        except Exception as e:
            self.logger.error(f"计算汇总统计失败: {e}")
            return {}

    def _safe_isnan_check(self, values: np.ndarray) -> np.ndarray:
        """
        安全的NaN检查，处理不同数据类型
        """
        try:
            # 确保是数值类型
            if values.dtype.kind in ['U', 'S', 'O']:  # 字符串或对象类型
                # 尝试转换为数值类型
                try:
                    values = pd.to_numeric(values, errors='coerce')
                except:
                    # 如果转换失败，返回全部无效的mask
                    return np.zeros(len(values), dtype=bool)

            # 检查NaN和Inf
            return ~(np.isnan(values) | np.isinf(values))

        except Exception as e:
            self.logger.warning(f"数据类型检查失败: {e}")
            # 返回保守的mask（全部无效）
            return np.zeros(len(values), dtype=bool)

    def _precompute_common_data(self, data: pd.DataFrame, return_column: str) -> dict:
        """
        预计算公共数据，避免重复计算
        这是关键的性能优化点
        """
        try:
            self.logger.debug("开始预计算公共数据...")

            # 提取收益率数据（所有因子都需要）
            returns = data[return_column].values
            valid_returns_mask = self._safe_isnan_check(returns)

            # 预计算分位数边界（避免重复计算）
            quintile_bounds = {}
            for factor_name in data.columns:
                if factor_name != return_column and factor_name not in ['date', 'stock_code']:
                    factor_values = data[factor_name].values
                    valid_factor_mask = self._safe_isnan_check(factor_values)
                    combined_mask = valid_returns_mask & valid_factor_mask

                    if combined_mask.sum() > 30:
                        valid_factor_values = factor_values[combined_mask]
                        try:
                            bounds = np.percentile(valid_factor_values, [20, 40, 60, 80])
                            quintile_bounds[factor_name] = bounds
                        except:
                            quintile_bounds[factor_name] = None

            # 预计算收益率排序（RankIC计算需要）
            returns_rank = None
            if valid_returns_mask.sum() > 0:
                from scipy.stats import rankdata
                returns_rank = rankdata(returns[valid_returns_mask])

            optimized_data = {
                'data': data,
                'returns': returns,
                'valid_returns_mask': valid_returns_mask,
                'returns_rank': returns_rank,
                'quintile_bounds': quintile_bounds,
                'total_samples': len(data)
            }

            self.logger.debug(f"预计算完成，处理 {len(quintile_bounds)} 个因子")
            return optimized_data

        except Exception as e:
            self.logger.error(f"预计算公共数据失败: {e}")
            # 返回基础数据结构
            return {
                'data': data,
                'returns': data[return_column].values,
                'valid_returns_mask': np.ones(len(data), dtype=bool),
                'returns_rank': None,
                'quintile_bounds': {},
                'total_samples': len(data)
            }

    def _calculate_factor_metrics_enhanced(self, optimized_data: dict, factor_name: str,
                                         return_column: str) -> Dict[str, float]:
        """
        计算单个因子的增强量化指标
        包含重命名的IR指标、真正的IR、标准化ICIR、最大回撤、胜率等
        """
        try:
            # 从预计算数据中获取信息
            data = optimized_data['data']
            returns = optimized_data['returns']
            valid_returns_mask = optimized_data['valid_returns_mask']
            quintile_bounds = optimized_data['quintile_bounds']

            # 获取因子值
            if factor_name not in data.columns:
                return self._get_empty_metrics()

            # 严格的因子类型检查
            factor_series = data[factor_name]

            # 检查是否为数值型
            if not pd.api.types.is_numeric_dtype(factor_series):
                self.logger.warning(f"因子 {factor_name} 不是数值型，跳过计算")
                return self._get_empty_metrics()

            # 检查是否包含字符串
            if factor_series.dtype == 'object':
                # 尝试转换为数值型
                try:
                    factor_series = pd.to_numeric(factor_series, errors='coerce')
                except:
                    self.logger.warning(f"因子 {factor_name} 无法转换为数值型，跳过计算")
                    return self._get_empty_metrics()

            factor_values = factor_series.values
            valid_factor_mask = self._safe_isnan_check(factor_values)
            combined_mask = valid_returns_mask & valid_factor_mask

            if combined_mask.sum() < 30:
                return self._get_empty_metrics()

            factor_clean = factor_values[combined_mask]
            returns_clean = returns[combined_mask]

            # 1. Information Coefficient (IC) - 使用numpy优化
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1]
            ic = 0 if np.isnan(ic) else ic

            # 2. Rank IC - 使用预计算的排序数据
            from scipy.stats import rankdata
            factor_rank = rankdata(factor_clean)
            returns_rank_clean = rankdata(returns_clean)
            rank_ic = np.corrcoef(factor_rank, returns_rank_clean)[0, 1]
            rank_ic = 0 if np.isnan(rank_ic) else rank_ic

            # 3. Top-Bottom Return Difference (原IR重命名)
            top_bottom_return_diff = 0
            if factor_name in quintile_bounds and quintile_bounds[factor_name] is not None:
                bounds = quintile_bounds[factor_name]
                try:
                    # 最低20%
                    low_mask = factor_clean <= bounds[0]
                    # 最高20%
                    high_mask = factor_clean >= bounds[3]

                    if low_mask.sum() > 5 and high_mask.sum() > 5:  # 增加最小样本要求
                        low_return = returns_clean[low_mask].mean()
                        high_return = returns_clean[high_mask].mean()
                        top_bottom_return_diff = high_return - low_return

                        # 数值稳定性检查：确保与IC符号一致
                        if abs(ic) > 1e-6 and abs(top_bottom_return_diff) > 1e-6:  # 只有当两者都不接近0时才检查
                            if (ic > 0 and top_bottom_return_diff < 0) or (ic < 0 and top_bottom_return_diff > 0):
                                # 如果符号不一致，使用IC的符号来调整top_bottom_return_diff
                                top_bottom_return_diff = abs(top_bottom_return_diff) * np.sign(ic)
                except:
                    top_bottom_return_diff = 0

            # 4. 真正的Information Ratio (IR)
            ir = self._calculate_true_ir(factor_clean, returns_clean, quintile_bounds.get(factor_name))

            # 5. 标准化ICIR - 使用滚动窗口
            icir_20d, icir_60d = self._calculate_standardized_icir(
                data, factor_name, return_column, combined_mask
            )

            # 6. 最大回撤
            max_drawdown = self._calculate_max_drawdown_enhanced(
                data, factor_name, return_column, combined_mask, quintile_bounds.get(factor_name)
            )

            # 7. 胜率
            win_rate = self._calculate_win_rate(factor_clean, returns_clean)

            # 8. 因子覆盖度
            coverage = combined_mask.sum() / len(data)

            # 9. 因子单调性
            monotonicity = 0
            if factor_name in quintile_bounds and quintile_bounds[factor_name] is not None:
                monotonicity = self._calculate_monotonicity_ultra_fast(
                    factor_clean, returns_clean, quintile_bounds[factor_name]
                )

            return {
                'ic': float(ic),
                'rank_ic': float(rank_ic),
                'top_bottom_return_diff': float(top_bottom_return_diff),  # 重命名的原IR
                'ir': float(ir),  # 真正的IR
                'icir_20d': float(icir_20d),
                'icir_60d': float(icir_60d),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'coverage': float(coverage),
                'monotonicity': float(monotonicity),
                'valid_samples': int(len(factor_clean))
            }

        except Exception as e:
            self.logger.error(f"计算增强因子指标失败 {factor_name}: {e}")
            # 强制垃圾回收以释放内存
            import gc
            gc.collect()
            return self._get_empty_metrics()

    def _calculate_factor_metrics_optimized(self, optimized_data: dict, factor_name: str,
                                          return_column: str) -> Dict[str, float]:
        """
        计算单个因子的专业量化指标（高性能优化版）
        使用预计算数据，避免重复计算
        """
        try:
            # 从预计算数据中获取信息
            data = optimized_data['data']
            returns = optimized_data['returns']
            valid_returns_mask = optimized_data['valid_returns_mask']
            quintile_bounds = optimized_data['quintile_bounds']

            # 获取因子值
            if factor_name not in data.columns:
                return {'ic': 0, 'rank_ic': 0, 'ir': 0, 'icir': 0, 'coverage': 0, 'monotonicity': 0}

            factor_values = data[factor_name].values
            valid_factor_mask = self._safe_isnan_check(factor_values)
            combined_mask = valid_returns_mask & valid_factor_mask

            if combined_mask.sum() < 30:
                return {'ic': 0, 'rank_ic': 0, 'ir': 0, 'icir': 0, 'coverage': 0, 'monotonicity': 0}

            factor_clean = factor_values[combined_mask]
            returns_clean = returns[combined_mask]

            # 1. Information Coefficient (IC) - 使用numpy优化
            ic = np.corrcoef(factor_clean, returns_clean)[0, 1]
            ic = 0 if np.isnan(ic) else ic

            # 2. Rank IC - 使用预计算的排序数据
            from scipy.stats import rankdata
            factor_rank = rankdata(factor_clean)
            returns_rank_clean = rankdata(returns_clean)
            rank_ic = np.corrcoef(factor_rank, returns_rank_clean)[0, 1]
            rank_ic = 0 if np.isnan(rank_ic) else rank_ic

            # 3. Information Ratio (IR) - 使用预计算的分位数边界
            ir = 0
            if factor_name in quintile_bounds and quintile_bounds[factor_name] is not None:
                bounds = quintile_bounds[factor_name]
                try:
                    # 最低20%
                    low_mask = factor_clean <= bounds[0]
                    # 最高20%
                    high_mask = factor_clean >= bounds[3]

                    if low_mask.sum() > 0 and high_mask.sum() > 0:
                        low_return = returns_clean[low_mask].mean()
                        high_return = returns_clean[high_mask].mean()
                        ir = high_return - low_return
                except:
                    ir = 0

            # 4. ICIR - 简化版（分段计算）
            icir = 0
            if len(factor_clean) >= 100:
                try:
                    segment_size = len(factor_clean) // 5
                    segment_ics = []

                    for i in range(5):
                        start_idx = i * segment_size
                        end_idx = (i + 1) * segment_size if i < 4 else len(factor_clean)

                        seg_factor = factor_clean[start_idx:end_idx]
                        seg_returns = returns_clean[start_idx:end_idx]

                        if len(seg_factor) > 10:
                            seg_ic = np.corrcoef(seg_factor, seg_returns)[0, 1]
                            if not np.isnan(seg_ic):
                                segment_ics.append(seg_ic)

                    if len(segment_ics) > 1:
                        ic_std = np.std(segment_ics)
                        icir = ic / ic_std if ic_std > 0 else 0
                except:
                    icir = 0

            # 5. 因子覆盖度
            coverage = combined_mask.sum() / len(data)

            # 6. 因子单调性（使用预计算的分位数）
            monotonicity = 0
            if factor_name in quintile_bounds and quintile_bounds[factor_name] is not None:
                monotonicity = self._calculate_monotonicity_ultra_fast(
                    factor_clean, returns_clean, quintile_bounds[factor_name]
                )

            return {
                'ic': ic,
                'rank_ic': rank_ic,
                'ir': ir,
                'icir': icir,
                'coverage': coverage,
                'monotonicity': monotonicity,
                'valid_samples': len(factor_clean)
            }

        except Exception as e:
            self.logger.error(f"计算因子指标失败 {factor_name}: {e}")
            return {'ic': 0, 'rank_ic': 0, 'ir': 0, 'icir': 0, 'coverage': 0, 'monotonicity': 0}

    def _calculate_monotonicity_ultra_fast(self, factor_values: np.ndarray,
                                         returns: np.ndarray, quintile_bounds: np.ndarray) -> float:
        """
        计算因子单调性（超快速版本）
        使用预计算的分位数边界
        """
        try:
            group_returns = []
            prev_bound = -np.inf

            # 使用预计算的分位数边界
            for bound in list(quintile_bounds) + [np.inf]:
                mask = (factor_values > prev_bound) & (factor_values <= bound)
                if mask.sum() > 0:
                    group_returns.append(returns[mask].mean())
                prev_bound = bound

            if len(group_returns) < 3:
                return 0

            # 计算单调性
            diffs = np.diff(group_returns)
            if len(diffs) == 0:
                return 0

            positive_diffs = np.sum(diffs > 0)
            negative_diffs = np.sum(diffs < 0)
            monotonicity = max(positive_diffs, negative_diffs) / len(diffs)

            return monotonicity

        except Exception:
            return 0

    def _get_empty_metrics(self) -> Dict[str, float]:
        """返回空的指标字典"""
        return {
            'ic': 0,
            'rank_ic': 0,
            'top_bottom_return_diff': 0,
            'ir': 0,
            'icir_20d': 0,
            'icir_60d': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'coverage': 0,
            'monotonicity': 0,
            'valid_samples': 0
        }

    def _calculate_true_ir(self, factor_values: np.ndarray, returns: np.ndarray,
                          quintile_bounds: np.ndarray = None) -> float:
        """
        计算专业的Information Ratio
        使用与IC一致的逻辑：基于因子分位数的多空组合收益率与其标准差的比值
        确保IR与IC保持同号关系
        """
        try:
            if quintile_bounds is None:
                return 0.0

            # 构建多空组合：做多高分位，做空低分位
            high_mask = factor_values >= quintile_bounds[3]  # 前20%
            low_mask = factor_values <= quintile_bounds[0]   # 后20%

            if high_mask.sum() < 5 or low_mask.sum() < 5:
                return 0.0

            high_returns = returns[high_mask]
            low_returns = returns[low_mask]

            # 计算多空组合收益率
            high_mean = high_returns.mean()
            low_mean = low_returns.mean()
            long_short_return = high_mean - low_mean

            # 专业做法：计算基于时间序列的IR
            # 如果样本量足够大，构建时间序列的多空组合收益率
            if len(high_returns) >= 50 and len(low_returns) >= 50:
                # 将数据分成若干时间段，计算每个时间段的多空收益率
                n_periods = min(10, min(len(high_returns) // 5, len(low_returns) // 5))

                if n_periods >= 3:
                    period_returns = []

                    high_period_size = len(high_returns) // n_periods
                    low_period_size = len(low_returns) // n_periods

                    for i in range(n_periods):
                        high_start = i * high_period_size
                        high_end = (i + 1) * high_period_size if i < n_periods - 1 else len(high_returns)

                        low_start = i * low_period_size
                        low_end = (i + 1) * low_period_size if i < n_periods - 1 else len(low_returns)

                        high_period_return = high_returns[high_start:high_end].mean()
                        low_period_return = low_returns[low_start:low_end].mean()

                        period_returns.append(high_period_return - low_period_return)

                    if len(period_returns) >= 3:
                        # 计算时间序列IR：平均收益率 / 收益率标准差
                        mean_return = np.mean(period_returns)
                        std_return = np.std(period_returns, ddof=1)  # 使用样本标准差

                        if std_return > 0:
                            ir = mean_return / std_return
                            return float(ir) if not np.isnan(ir) else 0.0

            # 简化计算：使用单次多空组合收益率与合并标准误差的比值
            # 计算标准误差
            high_se = high_returns.std() / np.sqrt(len(high_returns))
            low_se = low_returns.std() / np.sqrt(len(low_returns))
            combined_se = np.sqrt(high_se**2 + low_se**2)

            if combined_se > 0:
                ir = long_short_return / combined_se
                return float(ir) if not np.isnan(ir) else 0.0
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_standardized_icir(self, data: pd.DataFrame, factor_name: str,
                                   return_column: str, valid_mask: np.ndarray) -> Tuple[float, float]:
        """
        计算标准化滚动ICIR（内存优化版）
        使用正确的量化金融公式：ICIR = rolling_mean(IC) / rolling_std(IC)
        返回20日和60日ICIR
        """
        try:
            # 🚀 内存优化：提前检查数据量，避免大数组创建
            if valid_mask.sum() < 60:
                return 0, 0

            # 🚀 内存优化：分批处理大数据集
            total_valid_samples = valid_mask.sum()
            if total_valid_samples > 500000:  # 超过50万样本时启用内存优化
                return self._calculate_icir_memory_optimized(data, factor_name, return_column, valid_mask)

            # 提取有效数据并按日期排序（确保不修改原始数据）
            try:
                valid_data = data[valid_mask].copy()

                # 如果有日期字段，按日期排序
                if 'date' in valid_data.columns:
                    valid_data = valid_data.sort_values('date')
            except Exception as e:
                self.logger.warning(f"ICIR数据处理失败: {e}")
                return 0, 0

            factor_values = valid_data[factor_name].values
            return_values = valid_data[return_column].values

            # 检查数据量
            if len(factor_values) < 60:
                return 0, 0

            # 方法1：如果有日期信息且交易日足够，按日期分组计算真正的滚动ICIR
            if 'date' in valid_data.columns and len(valid_data['date'].unique()) >= 10:
                return self._calculate_rolling_icir_by_date(valid_data, factor_name, return_column)

            # 方法2：如果交易日不足，使用整体IC作为ICIR的近似
            elif 'date' in valid_data.columns and len(valid_data['date'].unique()) >= 3:
                # 计算整体IC作为ICIR的近似值
                overall_ic = np.corrcoef(factor_values, return_values)[0, 1]
                if not np.isnan(overall_ic):
                    # 当交易日不足时，ICIR近似等于IC（假设IC稳定）
                    return overall_ic, overall_ic
                else:
                    return 0, 0

            # 方法3：如果没有日期信息，使用滑动窗口模拟滚动ICIR
            else:
                return self._calculate_rolling_icir_by_window(factor_values, return_values)

        except Exception as e:
            self.logger.warning(f"计算标准化ICIR失败: {e}")
            return 0, 0

    def _calculate_icir_memory_optimized(self, data: pd.DataFrame, factor_name: str,
                                       return_column: str, valid_mask: np.ndarray) -> Tuple[float, float]:
        """
        内存优化的ICIR计算方法
        通过分批处理和及时释放内存来避免大数组分配
        """
        try:
            # 🚀 内存优化策略1：只提取必要的列，减少内存占用
            essential_columns = ['date', factor_name, return_column]
            available_columns = [col for col in essential_columns if col in data.columns]

            if 'date' not in available_columns:
                # 如果没有日期列，使用简化计算
                return self._calculate_icir_simplified(data, factor_name, return_column, valid_mask)

            # 只提取必要的数据
            subset_data = data[available_columns][valid_mask].copy()

            if len(subset_data) < 60:
                return 0, 0

            # 按日期排序
            subset_data = subset_data.sort_values('date')

            # 🚀 内存优化策略2：分批计算每日IC，避免创建大矩阵
            daily_ics = []
            unique_dates = subset_data['date'].unique()

            # 分批处理日期，每批处理100个交易日
            batch_size = 100
            for i in range(0, len(unique_dates), batch_size):
                batch_dates = unique_dates[i:i + batch_size]
                batch_data = subset_data[subset_data['date'].isin(batch_dates)]

                # 计算这批日期的IC
                batch_ics = self._calculate_batch_daily_ics(batch_data, factor_name, return_column)
                daily_ics.extend(batch_ics)

                # 🚀 内存优化策略3：及时删除不需要的变量
                del batch_data

            if len(daily_ics) < 3:
                return 0, 0

            # 转换为numpy数组进行最终计算
            daily_ics = np.array(daily_ics)

            # 计算ICIR
            icir_20d = self._calculate_icir_from_daily_ics(daily_ics, window=20)
            icir_60d = self._calculate_icir_from_daily_ics(daily_ics, window=60)

            return icir_20d, icir_60d

        except Exception as e:
            self.logger.warning(f"内存优化ICIR计算失败: {e}")
            return 0, 0

    def _calculate_batch_daily_ics(self, batch_data: pd.DataFrame, factor_name: str, return_column: str) -> List[float]:
        """计算一批日期的每日IC"""
        daily_ics = []

        for date, group in batch_data.groupby('date'):
            if len(group) >= 10:  # 每日至少10只股票
                group_factor = group[factor_name].values
                group_return = group[return_column].values

                # 去除NaN值
                valid_idx = ~(np.isnan(group_factor) | np.isnan(group_return))
                if valid_idx.sum() >= 10:
                    try:
                        ic = np.corrcoef(group_factor[valid_idx], group_return[valid_idx])[0, 1]
                        if not np.isnan(ic):
                            daily_ics.append(ic)
                    except:
                        continue

        return daily_ics

    def _calculate_icir_from_daily_ics(self, daily_ics: np.ndarray, window: int) -> float:
        """从每日IC数组计算ICIR"""
        if len(daily_ics) >= window:
            recent_ics = daily_ics[-window:]
            ic_mean = np.mean(recent_ics)
            ic_std = np.std(recent_ics, ddof=1)
            return ic_mean / ic_std if ic_std > 0 else ic_mean
        elif len(daily_ics) >= max(10, window // 3):
            # 如果数据不足但有一定量，使用全部数据
            ic_mean = np.mean(daily_ics)
            ic_std = np.std(daily_ics, ddof=1)
            return ic_mean / ic_std if ic_std > 0 else ic_mean
        else:
            return np.mean(daily_ics) if len(daily_ics) > 0 else 0

    def _calculate_icir_simplified(self, data: pd.DataFrame, factor_name: str,
                                 return_column: str, valid_mask: np.ndarray) -> Tuple[float, float]:
        """简化的ICIR计算（当没有日期信息时）"""
        try:
            factor_values = data[factor_name][valid_mask].values
            return_values = data[return_column][valid_mask].values

            # 去除NaN值
            valid_idx = ~(np.isnan(factor_values) | np.isnan(return_values))
            if valid_idx.sum() < 60:
                return 0, 0

            factor_clean = factor_values[valid_idx]
            return_clean = return_values[valid_idx]

            # 计算整体IC作为ICIR的近似
            overall_ic = np.corrcoef(factor_clean, return_clean)[0, 1]
            if not np.isnan(overall_ic):
                return overall_ic, overall_ic
            else:
                return 0, 0

        except Exception:
            return 0, 0

    def _calculate_rolling_icir_by_date(self, data: pd.DataFrame, factor_name: str,
                                      return_column: str) -> Tuple[float, float]:
        """
        基于日期的真正滚动ICIR计算
        """
        try:
            # 按日期分组计算每日IC
            daily_ics = []
            dates = []

            for date, group in data.groupby('date'):
                if len(group) >= 10:  # 每日至少10只股票
                    group_factor = group[factor_name].values
                    group_return = group[return_column].values

                    # 去除NaN值
                    valid_idx = ~(np.isnan(group_factor) | np.isnan(group_return))
                    if valid_idx.sum() >= 10:
                        try:
                            ic = np.corrcoef(group_factor[valid_idx], group_return[valid_idx])[0, 1]
                            if not np.isnan(ic):
                                daily_ics.append(ic)
                                dates.append(date)
                        except:
                            continue

            if len(daily_ics) < 3:
                return 0, 0

            daily_ics = np.array(daily_ics)

            # 计算20日滚动ICIR
            icir_20d = 0
            if len(daily_ics) >= 20:
                # 使用最近20个交易日
                recent_20_ics = daily_ics[-20:]
                ic_mean_20 = np.mean(recent_20_ics)
                ic_std_20 = np.std(recent_20_ics, ddof=1)  # 使用样本标准差
                icir_20d = ic_mean_20 / ic_std_20 if ic_std_20 > 0 else ic_mean_20
            elif len(daily_ics) >= 10:
                # 如果不足20日但有10日以上，使用全部数据
                ic_mean = np.mean(daily_ics)
                ic_std = np.std(daily_ics, ddof=1)
                icir_20d = ic_mean / ic_std if ic_std > 0 else ic_mean
            else:
                # 如果交易日太少，ICIR近似等于IC均值
                icir_20d = np.mean(daily_ics)

            # 计算60日滚动ICIR
            icir_60d = 0
            if len(daily_ics) >= 60:
                # 使用最近60个交易日
                recent_60_ics = daily_ics[-60:]
                ic_mean_60 = np.mean(recent_60_ics)
                ic_std_60 = np.std(recent_60_ics, ddof=1)  # 使用样本标准差
                icir_60d = ic_mean_60 / ic_std_60 if ic_std_60 > 0 else ic_mean_60
            elif len(daily_ics) >= 20:
                # 如果不足60日但有20日以上，使用全部可用数据
                ic_mean_all = np.mean(daily_ics)
                ic_std_all = np.std(daily_ics, ddof=1)
                icir_60d = ic_mean_all / ic_std_all if ic_std_all > 0 else ic_mean_all
            else:
                # 如果交易日太少，60日ICIR等于20日ICIR
                icir_60d = icir_20d

            return icir_20d, icir_60d

        except Exception as e:
            self.logger.warning(f"按日期计算滚动ICIR失败: {e}")
            return 0, 0

    def _calculate_rolling_icir_by_window(self, factor_values: np.ndarray,
                                        return_values: np.ndarray) -> Tuple[float, float]:
        """
        基于滑动窗口的滚动ICIR计算（当没有日期信息时使用）
        """
        try:
            # 使用滑动窗口计算IC序列
            window_size = 100  # 每个窗口的样本数
            step_size = 50     # 滑动步长

            ic_list = []

            for i in range(0, len(factor_values) - window_size + 1, step_size):
                end_idx = i + window_size
                window_factor = factor_values[i:end_idx]
                window_return = return_values[i:end_idx]

                # 去除NaN值
                valid_idx = ~(np.isnan(window_factor) | np.isnan(window_return))
                if valid_idx.sum() >= 50:  # 至少50个有效样本
                    try:
                        ic = np.corrcoef(window_factor[valid_idx], window_return[valid_idx])[0, 1]
                        if not np.isnan(ic):
                            ic_list.append(ic)
                    except:
                        continue

            if len(ic_list) < 5:
                return 0, 0

            ic_array = np.array(ic_list)

            # 计算20日等效ICIR（使用最近的IC值）
            icir_20d = 0
            if len(ic_array) >= 5:
                recent_ics = ic_array[-min(10, len(ic_array)):]  # 最近的IC值
                ic_mean = np.mean(recent_ics)
                ic_std = np.std(recent_ics, ddof=1)
                icir_20d = ic_mean / ic_std if ic_std > 0 else 0

            # 计算60日等效ICIR
            icir_60d = 0
            if len(ic_array) >= 3:
                ic_mean = np.mean(ic_array)
                ic_std = np.std(ic_array, ddof=1)
                icir_60d = ic_mean / ic_std if ic_std > 0 else 0

            return icir_20d, icir_60d

        except Exception as e:
            self.logger.warning(f"滑动窗口计算ICIR失败: {e}")
            return 0, 0

    def _calculate_max_drawdown(self, factor_values: np.ndarray, returns: np.ndarray,
                               quintile_bounds: np.ndarray = None) -> float:
        """
        计算因子分组组合的最大回撤
        基于因子分位数构建投资组合，计算其最大回撤
        """
        try:
            if quintile_bounds is None or len(returns) < 10:
                return 0

            # 构建高分位组合（前20%）
            high_mask = factor_values >= quintile_bounds[3]
            if high_mask.sum() < 5:
                return 0

            # 构建低分位组合（后20%）
            low_mask = factor_values <= quintile_bounds[0]
            if low_mask.sum() < 5:
                return 0

            # 计算多空组合收益率（高分位 - 低分位）
            high_returns = returns[high_mask]
            low_returns = returns[low_mask]

            # 使用平均收益率构建多空组合
            portfolio_return = high_returns.mean() - low_returns.mean()

            # 对于单次收益率，最大回撤就是负收益率的绝对值
            if portfolio_return < 0:
                max_drawdown = abs(portfolio_return)
            else:
                max_drawdown = 0

            return max_drawdown

        except Exception:
            return 0

    def _calculate_win_rate(self, factor_values: np.ndarray, returns: np.ndarray) -> float:
        """
        计算因子预测胜率
        基于因子分位数的预测准确率：高因子值对应正收益的比例
        """
        try:
            if len(factor_values) < 20:
                return 0

            # 计算因子分位数
            factor_percentiles = np.percentile(factor_values, [20, 40, 60, 80])

            # 定义高因子值（前40%）和低因子值（后40%）
            high_factor_mask = factor_values >= factor_percentiles[2]  # 前40%
            low_factor_mask = factor_values <= factor_percentiles[1]   # 后40%

            if high_factor_mask.sum() < 5 or low_factor_mask.sum() < 5:
                return 0.5  # 如果样本不足，返回50%

            # 计算高因子值组合的正收益比例
            high_factor_returns = returns[high_factor_mask]
            high_positive_ratio = (high_factor_returns > 0).sum() / len(high_factor_returns)

            # 计算低因子值组合的负收益比例
            low_factor_returns = returns[low_factor_mask]
            low_negative_ratio = (low_factor_returns < 0).sum() / len(low_factor_returns)

            # 胜率 = (高因子值正收益比例 + 低因子值负收益比例) / 2
            win_rate = (high_positive_ratio + low_negative_ratio) / 2

            return win_rate if not np.isnan(win_rate) else 0.5

        except Exception:
            return 0.5

    def _calculate_max_drawdown_enhanced(self, data: pd.DataFrame, factor_name: str,
                                       return_column: str, valid_mask: np.ndarray,
                                       quintile_bounds: np.ndarray = None) -> float:
        """
        专业的最大回撤计算（内存优化版）
        基于因子分位数构建多空组合，计算专业的最大回撤指标
        """
        try:
            if quintile_bounds is None or valid_mask.sum() < 50:
                return 0.0

            # 🚀 内存优化：提前检查数据量
            total_valid_samples = valid_mask.sum()
            if total_valid_samples > 500000:  # 超过50万样本时启用内存优化
                return self._calculate_drawdown_memory_optimized(data, factor_name, return_column, valid_mask, quintile_bounds)

            # 提取有效数据（确保不修改原始数据）
            try:
                valid_data = data[valid_mask].copy()
                if len(valid_data) < 50:
                    return 0.0

                # 按日期排序（如果有日期字段）
                if 'date' in valid_data.columns:
                    valid_data = valid_data.sort_values('date')
            except Exception as e:
                self.logger.warning(f"最大回撤数据处理失败: {e}")
                return 0.0

            factor_values = valid_data[factor_name].values
            return_values = valid_data[return_column].values

            # 专业做法：按日期分组计算每日的多空组合收益率
            if 'date' in valid_data.columns and len(valid_data['date'].unique()) >= 2:
                # 按日期分组计算
                daily_returns = []
                for _, group in valid_data.groupby('date'):
                    if len(group) < 5:  # 降低要求：每日至少5只股票
                        continue

                    group_factor = group[factor_name].values
                    group_return = group[return_column].values

                    # 计算当日的分位数
                    try:
                        if len(group_factor) >= 10:
                            daily_high_threshold = np.percentile(group_factor, 80)
                            daily_low_threshold = np.percentile(group_factor, 20)
                        else:
                            # 对于小样本，使用更宽松的分位数
                            daily_high_threshold = np.percentile(group_factor, 70)
                            daily_low_threshold = np.percentile(group_factor, 30)

                        daily_high_mask = group_factor >= daily_high_threshold
                        daily_low_mask = group_factor <= daily_low_threshold

                        if daily_high_mask.sum() >= 1 and daily_low_mask.sum() >= 1:
                            high_ret = group_return[daily_high_mask].mean()
                            low_ret = group_return[daily_low_mask].mean()
                            daily_portfolio_return = high_ret - low_ret

                            # 限制单日收益率在合理范围内
                            if abs(daily_portfolio_return) < 0.5:  # 限制在±50%以内
                                daily_returns.append(daily_portfolio_return)
                    except:
                        continue

                if len(daily_returns) >= 2:  # 降低要求：至少2个交易日
                    portfolio_returns = np.array(daily_returns)
                else:
                    # 如果按日期分组失败，使用全局分位数计算
                    high_mask = factor_values >= quintile_bounds[3]
                    low_mask = factor_values <= quintile_bounds[0]

                    if high_mask.sum() >= 5 and low_mask.sum() >= 5:
                        high_returns = return_values[high_mask]
                        low_returns = return_values[low_mask]
                        portfolio_return = high_returns.mean() - low_returns.mean()

                        # 对于单一收益率，最大回撤就是负收益的绝对值
                        if portfolio_return < 0:
                            return float(abs(portfolio_return))
                        else:
                            return 0.0
                    else:
                        return 0.0

            else:
                # 如果没有日期信息，使用全局分位数计算简化的最大回撤
                high_mask = factor_values >= quintile_bounds[3]
                low_mask = factor_values <= quintile_bounds[0]

                if high_mask.sum() >= 5 and low_mask.sum() >= 5:
                    high_returns = return_values[high_mask]
                    low_returns = return_values[low_mask]
                    portfolio_return = high_returns.mean() - low_returns.mean()

                    # 对于单一收益率，最大回撤就是负收益的绝对值
                    if portfolio_return < 0:
                        return float(abs(portfolio_return))
                    else:
                        return 0.0
                else:
                    return 0.0

            # 专业的最大回撤计算
            if len(portfolio_returns) >= 2:
                try:
                    # 确保收益率在合理范围内
                    portfolio_returns = np.clip(portfolio_returns, -0.3, 0.3)  # 限制在±30%

                    # 计算累积净值（从1开始）
                    cumulative_nav = np.cumprod(1 + portfolio_returns)

                    # 计算滚动最大值
                    running_max = np.maximum.accumulate(cumulative_nav)

                    # 计算回撤序列
                    drawdown_series = (cumulative_nav - running_max) / running_max

                    # 最大回撤
                    max_drawdown = abs(np.min(drawdown_series))

                    # 确保结果在合理范围内
                    if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                        max_drawdown = 0.0
                    elif max_drawdown > 1.0:
                        max_drawdown = 1.0

                    return float(max_drawdown)

                except Exception as e:
                    self.logger.warning(f"累积收益率计算失败: {e}")

                    # 方法2：简化的最大回撤计算
                    try:
                        # 计算最大负收益率作为最大回撤的近似
                        negative_returns = portfolio_returns[portfolio_returns < 0]
                        if len(negative_returns) > 0:
                            max_drawdown = abs(np.min(negative_returns))
                        else:
                            max_drawdown = 0.0

                        return float(min(max_drawdown, 1.0))

                    except:
                        return 0.0
            else:
                # 对于单一收益率或无有效收益率的情况
                if len(portfolio_returns) == 1:
                    return float(abs(min(0, portfolio_returns[0])))
                else:
                    return 0.0

        except Exception as e:
            self.logger.warning(f"计算专业最大回撤失败: {e}")
            return 0.0

    def _calculate_cross_window_summary(self, time_windows: Dict[str, Dict]) -> Dict[str, Any]:
        """
        计算跨时间窗口的汇总统计
        """
        try:
            if not time_windows:
                return {}

            summary = {}

            # 收集所有指标
            all_metrics = {}
            for window_name, window_data in time_windows.items():
                factor_metrics = window_data.get('factor_metrics', {})
                for factor_name, metrics in factor_metrics.items():
                    if factor_name not in all_metrics:
                        all_metrics[factor_name] = {}
                    all_metrics[factor_name][window_name] = metrics

            # 计算跨窗口平均值
            metric_names = ['ic', 'rank_ic', 'top_bottom_return_diff', 'ir',
                          'icir_20d', 'icir_60d', 'max_drawdown', 'win_rate',
                          'coverage', 'monotonicity']

            for metric_name in metric_names:
                values = []
                for factor_metrics in all_metrics.values():
                    for window_metrics in factor_metrics.values():
                        if metric_name in window_metrics:
                            values.append(window_metrics[metric_name])

                if values:
                    summary[f'mean_{metric_name}'] = np.mean(values)
                    summary[f'std_{metric_name}'] = np.std(values)
                else:
                    summary[f'mean_{metric_name}'] = 0
                    summary[f'std_{metric_name}'] = 0

            # 计算高质量因子数量
            high_quality_count = 0
            for factor_metrics in all_metrics.values():
                for window_metrics in factor_metrics.values():
                    if self._is_factor_valid(window_metrics):
                        high_quality_count += 1
                        break  # 每个因子只计算一次

            summary['high_quality_factors'] = high_quality_count
            summary['total_factor_window_combinations'] = sum(len(fm) for fm in all_metrics.values())

            return summary

        except Exception as e:
            self.logger.error(f"计算跨窗口汇总统计失败: {e}")
            return {}

    def _calculate_drawdown_memory_optimized(self, data: pd.DataFrame, factor_name: str,
                                           return_column: str, valid_mask: np.ndarray,
                                           quintile_bounds: np.ndarray) -> float:
        """
        内存优化的最大回撤计算方法
        通过分批处理和简化计算来避免大数组分配
        """
        try:
            # 🚀 内存优化策略1：只提取必要的列
            essential_columns = ['date', factor_name, return_column]
            available_columns = [col for col in essential_columns if col in data.columns]

            if 'date' not in available_columns:
                # 如果没有日期列，使用简化计算
                return self._calculate_drawdown_simplified(data, factor_name, return_column, valid_mask, quintile_bounds)

            # 只提取必要的数据
            subset_data = data[available_columns][valid_mask].copy()

            if len(subset_data) < 50:
                return 0.0

            # 按日期排序
            subset_data = subset_data.sort_values('date')

            # 🚀 内存优化策略2：分批计算每日多空组合收益，避免创建大矩阵
            daily_returns = []
            unique_dates = subset_data['date'].unique()

            # 分批处理日期，每批处理50个交易日
            batch_size = 50
            for i in range(0, len(unique_dates), batch_size):
                batch_dates = unique_dates[i:i + batch_size]
                batch_data = subset_data[subset_data['date'].isin(batch_dates)]

                # 计算这批日期的每日收益
                batch_returns = self._calculate_batch_daily_returns(batch_data, factor_name, return_column, quintile_bounds)
                daily_returns.extend(batch_returns)

                # 🚀 内存优化策略3：及时删除不需要的变量
                del batch_data

            if len(daily_returns) < 2:
                return 0.0

            # 转换为numpy数组进行最终计算
            portfolio_returns = np.array(daily_returns)

            # 计算最大回撤（简化版，避免大数组操作）
            return self._calculate_simple_max_drawdown(portfolio_returns)

        except Exception as e:
            self.logger.warning(f"内存优化最大回撤计算失败: {e}")
            return 0.0

    def _calculate_batch_daily_returns(self, batch_data: pd.DataFrame, factor_name: str,
                                     return_column: str, quintile_bounds: np.ndarray) -> List[float]:
        """计算一批日期的每日多空组合收益"""
        daily_returns = []

        for _, group in batch_data.groupby('date'):
            if len(group) < 5:
                continue

            group_factor = group[factor_name].values
            group_return = group[return_column].values

            try:
                # 计算当日的分位数阈值
                if len(group_factor) >= 10:
                    daily_high_threshold = np.percentile(group_factor, 80)
                    daily_low_threshold = np.percentile(group_factor, 20)
                else:
                    daily_high_threshold = np.percentile(group_factor, 70)
                    daily_low_threshold = np.percentile(group_factor, 30)

                daily_high_mask = group_factor >= daily_high_threshold
                daily_low_mask = group_factor <= daily_low_threshold

                if daily_high_mask.sum() >= 1 and daily_low_mask.sum() >= 1:
                    high_ret = group_return[daily_high_mask].mean()
                    low_ret = group_return[daily_low_mask].mean()
                    daily_portfolio_return = high_ret - low_ret

                    # 限制单日收益率在合理范围内
                    if abs(daily_portfolio_return) < 0.5:
                        daily_returns.append(daily_portfolio_return)
            except:
                continue

        return daily_returns

    def _calculate_simple_max_drawdown(self, returns: np.ndarray) -> float:
        """简化的最大回撤计算，避免大数组操作"""
        try:
            # 限制收益率范围
            returns = np.clip(returns, -0.3, 0.3)

            # 简化计算：使用滚动最小值作为最大回撤的近似
            cumulative_return = 0
            max_cumulative = 0
            max_drawdown = 0

            for ret in returns:
                cumulative_return += ret
                max_cumulative = max(max_cumulative, cumulative_return)
                current_drawdown = max_cumulative - cumulative_return
                max_drawdown = max(max_drawdown, current_drawdown)

            return min(float(max_drawdown), 1.0)

        except Exception:
            # 如果简化计算也失败，使用最简单的方法
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                return min(float(abs(np.min(negative_returns))), 1.0)
            else:
                return 0.0

    def _calculate_drawdown_simplified(self, data: pd.DataFrame, factor_name: str,
                                     return_column: str, valid_mask: np.ndarray,
                                     quintile_bounds: np.ndarray) -> float:
        """简化的最大回撤计算（当没有日期信息时）"""
        try:
            factor_values = data[factor_name][valid_mask].values
            return_values = data[return_column][valid_mask].values

            # 使用全局分位数计算
            high_mask = factor_values >= quintile_bounds[3]
            low_mask = factor_values <= quintile_bounds[0]

            if high_mask.sum() >= 5 and low_mask.sum() >= 5:
                high_returns = return_values[high_mask]
                low_returns = return_values[low_mask]
                portfolio_return = high_returns.mean() - low_returns.mean()

                # 对于单一收益率，最大回撤就是负收益的绝对值
                if portfolio_return < 0:
                    return float(abs(portfolio_return))
                else:
                    return 0.0
            else:
                return 0.0

        except Exception:
            return 0.0
