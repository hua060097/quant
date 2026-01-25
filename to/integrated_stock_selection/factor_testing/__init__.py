#!/usr/bin/env python3
"""
因子测试模块

提供专业的单因子测试和分析功能
"""

from .single_factor_tester import SingleFactorTester
from .factor_ic_analyzer import FactorICAnalyzer
from .factor_turnover_analyzer import FactorTurnoverAnalyzer
from .factor_correlation_analyzer import FactorCorrelationAnalyzer
from .factor_performance_analyzer import FactorPerformanceAnalyzer
from .factor_report_generator import FactorReportGenerator
from .factor_test_suite import FactorTestSuite

__all__ = [
    'SingleFactorTester',
    'FactorICAnalyzer',
    'FactorTurnoverAnalyzer',
    'FactorCorrelationAnalyzer',
    'FactorPerformanceAnalyzer',
    'FactorReportGenerator',
    'FactorTestSuite'
]

__version__ = '1.0.0'
__author__ = 'Quantitative Research Team'
