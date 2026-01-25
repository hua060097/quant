#!/usr/bin/env python3
"""
因子测试套件

提供完整的因子测试流程，包括单因子测试、批量测试和报告生成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import time

from .single_factor_tester import SingleFactorTester
from .factor_ic_analyzer import FactorICAnalyzer
from .factor_turnover_analyzer import FactorTurnoverAnalyzer
from .factor_correlation_analyzer import FactorCorrelationAnalyzer
from .factor_performance_analyzer import FactorPerformanceAnalyzer
from .factor_report_generator import FactorReportGenerator

class FactorTestSuite:
    """因子测试套件"""
    
    def __init__(self, output_dir: str = "./factor_test_results"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个分析器
        self.single_factor_tester = SingleFactorTester()
        self.ic_analyzer = FactorICAnalyzer()
        self.turnover_analyzer = FactorTurnoverAnalyzer()
        self.correlation_analyzer = FactorCorrelationAnalyzer()
        self.performance_analyzer = FactorPerformanceAnalyzer()
        self.report_generator = FactorReportGenerator()
        
        self.logger.info(f"因子测试套件初始化完成，输出目录: {self.output_dir}")
    
    def test_single_factor(self, data: pd.DataFrame, factor_name: str,
                          return_col: str = 'next_day_return',
                          save_results: bool = True) -> Dict[str, Any]:
        """
        测试单个因子
        
        Args:
            data: 包含因子和收益率的数据
            factor_name: 因子名称
            return_col: 收益率列名
            save_results: 是否保存结果
            
        Returns:
            测试结果字典
        """
        try:
            self.logger.info(f"开始单因子测试: {factor_name}")
            start_time = time.time()
            
            # 基础单因子测试
            basic_results = self.single_factor_tester.test_single_factor(
                data, factor_name, return_col
            )
            
            # 详细IC分析
            ic_results = self.ic_analyzer.analyze_factor_ic(
                data, factor_name, return_col
            )
            
            # 换手率分析
            turnover_results = self.turnover_analyzer.analyze_factor_turnover(
                data, factor_name
            )
            
            # 表现分析
            performance_results = self.performance_analyzer.analyze_factor_performance(
                data, factor_name, return_col
            )
            
            # 合并结果
            comprehensive_results = {
                'factor_name': factor_name,
                'test_timestamp': datetime.now().isoformat(),
                'test_duration_seconds': time.time() - start_time,
                'basic_analysis': basic_results,
                'ic_analysis': ic_results,
                'turnover_analysis': turnover_results,
                'performance_analysis': performance_results,
                'summary': self._create_factor_summary(
                    basic_results, ic_results, turnover_results, performance_results
                )
            }
            
            # 保存结果
            if save_results:
                self._save_single_factor_results(comprehensive_results)
            
            self.logger.info(f"因子 {factor_name} 测试完成，耗时 {time.time() - start_time:.2f}s")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"测试因子 {factor_name} 失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {
                'factor_name': factor_name,
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    def test_multiple_factors(self, data: pd.DataFrame, factor_names: List[str],
                            return_col: str = 'next_day_return',
                            save_results: bool = True,
                            generate_report: bool = True) -> Dict[str, Any]:
        """
        批量测试多个因子
        
        Args:
            data: 包含因子和收益率的数据
            factor_names: 因子名称列表
            return_col: 收益率列名
            save_results: 是否保存结果
            generate_report: 是否生成报告
            
        Returns:
            批量测试结果
        """
        try:
            self.logger.info(f"开始批量因子测试，共 {len(factor_names)} 个因子")
            start_time = time.time()
            
            # 验证因子是否存在
            existing_factors = [name for name in factor_names if name in data.columns]
            missing_factors = [name for name in factor_names if name not in data.columns]
            
            if missing_factors:
                self.logger.warning(f"以下因子不存在于数据中: {missing_factors}")
            
            # 逐个测试因子
            factor_results = {}
            for i, factor_name in enumerate(existing_factors, 1):
                self.logger.info(f"测试进度: {i}/{len(existing_factors)} - {factor_name}")
                
                try:
                    factor_result = self.test_single_factor(
                        data, factor_name, return_col, save_results=False
                    )
                    factor_results[factor_name] = factor_result
                except Exception as e:
                    self.logger.error(f"测试因子 {factor_name} 失败: {e}")
                    factor_results[factor_name] = {
                        'factor_name': factor_name,
                        'error': str(e)
                    }
            
            # 因子相关性分析
            correlation_results = self.correlation_analyzer.analyze_factor_correlations(
                data, existing_factors
            )
            
            # 创建批量测试结果
            batch_results = {
                'test_timestamp': datetime.now().isoformat(),
                'test_duration_seconds': time.time() - start_time,
                'total_factors_requested': len(factor_names),
                'total_factors_tested': len(existing_factors),
                'missing_factors': missing_factors,
                'factor_results': factor_results,
                'correlation_analysis': correlation_results,
                'ranking': self._rank_factors(factor_results),
                'summary_statistics': self._calculate_batch_summary(factor_results)
            }
            
            # 保存结果
            if save_results:
                self._save_batch_results(batch_results)
            
            # 生成报告
            if generate_report:
                report_path = self._generate_batch_report(batch_results)
                batch_results['report_path'] = str(report_path)
            
            self.logger.info(f"批量因子测试完成，耗时 {time.time() - start_time:.2f}s")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"批量因子测试失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return {
                'error': str(e),
                'test_timestamp': datetime.now().isoformat()
            }
    
    def compare_factors(self, data: pd.DataFrame, factor_names: List[str],
                       return_col: str = 'next_day_return') -> Dict[str, Any]:
        """
        比较多个因子的表现
        
        Args:
            data: 包含因子和收益率的数据
            factor_names: 要比较的因子名称列表
            return_col: 收益率列名
            
        Returns:
            因子比较结果
        """
        try:
            self.logger.info(f"开始因子比较分析，共 {len(factor_names)} 个因子")
            
            # 获取所有因子的基础测试结果
            factor_results = {}
            for factor_name in factor_names:
                if factor_name in data.columns:
                    result = self.single_factor_tester.test_single_factor(
                        data, factor_name, return_col
                    )
                    factor_results[factor_name] = result
            
            # 创建比较分析
            comparison_results = {
                'comparison_timestamp': datetime.now().isoformat(),
                'factors_compared': list(factor_results.keys()),
                'comparison_metrics': self._create_comparison_metrics(factor_results),
                'ranking_by_metric': self._rank_by_different_metrics(factor_results),
                'correlation_matrix': self.correlation_analyzer.calculate_correlation_matrix(
                    data, list(factor_results.keys())
                ),
                'performance_comparison': self._compare_performance(factor_results),
                'recommendations': self._generate_comparison_recommendations(factor_results)
            }
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"因子比较分析失败: {e}")
            return {'error': str(e)}
    
    def _create_factor_summary(self, basic_results: Dict, ic_results: Dict,
                             turnover_results: Dict, performance_results: Dict) -> Dict[str, Any]:
        """创建因子摘要"""
        try:
            summary = {
                'overall_score': basic_results.get('overall_score', 0.0),
                'recommendation': basic_results.get('recommendation', 'UNKNOWN'),
                'key_metrics': {
                    'ic': basic_results.get('ic_analysis', {}).get('overall_ic', 0.0),
                    'ic_ir': basic_results.get('ic_analysis', {}).get('ic_ir', 0.0),
                    'long_short_return': basic_results.get('quantile_analysis', {}).get('long_short_return', 0.0),
                    'monotonicity': basic_results.get('quantile_analysis', {}).get('monotonicity', 0.0)
                },
                'strengths': [],
                'weaknesses': [],
                'usage_suggestions': []
            }
            
            # 分析优势和劣势
            ic = summary['key_metrics']['ic']
            ic_ir = summary['key_metrics']['ic_ir']
            long_short_return = summary['key_metrics']['long_short_return']
            monotonicity = summary['key_metrics']['monotonicity']
            
            if abs(ic) > 0.05:
                summary['strengths'].append(f"IC值较高 ({ic:.3f})")
            if abs(ic_ir) > 1.0:
                summary['strengths'].append(f"IC信息比率较好 ({ic_ir:.3f})")
            if abs(long_short_return) > 0.02:
                summary['strengths'].append(f"多空收益率较高 ({long_short_return:.3f})")
            if monotonicity > 0.7:
                summary['strengths'].append(f"单调性较好 ({monotonicity:.3f})")
            
            if abs(ic) < 0.02:
                summary['weaknesses'].append(f"IC值较低 ({ic:.3f})")
            if abs(ic_ir) < 0.5:
                summary['weaknesses'].append(f"IC信息比率较低 ({ic_ir:.3f})")
            if abs(long_short_return) < 0.01:
                summary['weaknesses'].append(f"多空收益率较低 ({long_short_return:.3f})")
            if monotonicity < 0.5:
                summary['weaknesses'].append(f"单调性较差 ({monotonicity:.3f})")
            
            # 使用建议
            if summary['overall_score'] >= 0.6:
                summary['usage_suggestions'].append("可以作为主要选股因子使用")
            elif summary['overall_score'] >= 0.4:
                summary['usage_suggestions'].append("可以作为辅助因子与其他因子组合使用")
            else:
                summary['usage_suggestions'].append("建议进一步优化或谨慎使用")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"创建因子摘要失败: {e}")
            return {}
    
    def _rank_factors(self, factor_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """对因子进行排名"""
        try:
            rankings = []
            
            for factor_name, result in factor_results.items():
                if 'error' not in result:
                    basic_analysis = result.get('basic_analysis', {})
                    score = basic_analysis.get('overall_score', 0.0)
                    ic = basic_analysis.get('ic_analysis', {}).get('overall_ic', 0.0)
                    
                    rankings.append({
                        'factor_name': factor_name,
                        'overall_score': score,
                        'ic': ic,
                        'recommendation': basic_analysis.get('recommendation', 'UNKNOWN')
                    })
            
            # 按综合评分排序
            rankings.sort(key=lambda x: x['overall_score'], reverse=True)
            
            # 添加排名
            for i, item in enumerate(rankings, 1):
                item['rank'] = i
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"因子排名失败: {e}")
            return []
    
    def _calculate_batch_summary(self, factor_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算批量测试摘要统计"""
        try:
            valid_results = [
                result for result in factor_results.values() 
                if 'error' not in result and 'basic_analysis' in result
            ]
            
            if not valid_results:
                return {}
            
            scores = [
                result['basic_analysis'].get('overall_score', 0.0) 
                for result in valid_results
            ]
            
            ics = [
                result['basic_analysis'].get('ic_analysis', {}).get('overall_ic', 0.0)
                for result in valid_results
            ]
            
            summary = {
                'total_valid_factors': len(valid_results),
                'score_statistics': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                },
                'ic_statistics': {
                    'mean': float(np.mean(ics)),
                    'std': float(np.std(ics)),
                    'min': float(np.min(ics)),
                    'max': float(np.max(ics)),
                    'median': float(np.median(ics))
                },
                'recommendation_distribution': {}
            }
            
            # 统计推荐等级分布
            recommendations = [
                result['basic_analysis'].get('recommendation', 'UNKNOWN')
                for result in valid_results
            ]
            
            for rec in set(recommendations):
                summary['recommendation_distribution'][rec] = recommendations.count(rec)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"计算批量摘要失败: {e}")
            return {}
    
    def _save_single_factor_results(self, results: Dict[str, Any]):
        """保存单因子测试结果"""
        try:
            factor_name = results['factor_name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"single_factor_{factor_name}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"单因子测试结果已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存单因子测试结果失败: {e}")
    
    def _save_batch_results(self, results: Dict[str, Any]):
        """保存批量测试结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_factor_test_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"批量测试结果已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存批量测试结果失败: {e}")
    
    def _generate_batch_report(self, results: Dict[str, Any]) -> Path:
        """生成批量测试报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"factor_test_report_{timestamp}.html"
            report_path = self.output_dir / report_filename
            
            # 使用报告生成器创建HTML报告
            html_content = self.report_generator.generate_batch_report(results)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"批量测试报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成批量测试报告失败: {e}")
            return self.output_dir / "report_generation_failed.txt"
