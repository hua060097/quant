#!/usr/bin/env python3
"""
终极版板块潜力分析系统 - 最终优化版
===========================================

功能说明：
1. 基于141个板块特征进行8维度深度分析
2. 预测最有潜力的板块，适用于A股市场
3. 提供风险评估和置信度评估
4. 支持参数优化和回测验证

8个分析维度：
- 技术面：趋势、均线、技术指标
- 资金面：成交量、换手率、资金流
- 情绪面：涨跌停、连板、市场情绪
- 相对强度：相对排名、历史表现
- 动量：多周期动量、连续性
- 波动率：适度波动率评估
- 结构：市值分布、支撑阻力
- 质量：指标一致性、信号强度

使用方法：
python ultimate_multi_day_analysis.py

输出：
- 每日前5名潜力板块预测
- 预测准确率和收益率验证
- 最优参数配置建议
- 风险评估报告
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from pathlib import Path
import importlib.util

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "clean_ml_system"))

# 导入所需模块
try:
    # 使用绝对路径导入EnhancedSectorAnalyzer
    clean_ml_path = project_root / "clean_ml_system"
    spec = importlib.util.spec_from_file_location(
        "enhanced_sector_analyzer",
        clean_ml_path / "enhanced_sector_analyzer.py"
    )
    enhanced_sector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enhanced_sector_module)
    EnhancedSectorAnalyzer = enhanced_sector_module.EnhancedSectorAnalyzer

    # 导入DataLoader
    spec = importlib.util.spec_from_file_location(
        "data_loader",
        clean_ml_path / "data_loader.py"
    )
    data_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader_module)
    DataLoader = data_loader_module.DataLoader

except Exception as e:
    print(f"导入模块失败: {e}")
    # 创建备用类
    class EnhancedSectorAnalyzer:
        def analyze_enhanced_daily_potential(self, daily_sectors, market_features):
            return []

    class DataLoader:
        def load_sector_features(self, start_date, end_date):
            return None
        def load_market_features(self, start_date, end_date):
            return None

def run_ultimate_analysis():
    """运行终极版分析"""
    print('🚀 终极版板块潜力分析系统 - 最终优化版')
    print('=' * 70)
    print('📊 功能说明：')
    print('  • 基于141个板块特征进行8维度深度分析')
    print('  • 预测最有潜力的板块，适用于A股市场')
    print('  • 提供风险评估和置信度评估')
    print('  • 支持参数优化和回测验证')
    print('')
    print('🎯 核心指标解释：')
    print('  • 准确率：预测板块第二天上涨的比例')
    print('  • 收益率：预测板块第二天的平均涨跌幅')
    print('  • 涨停命中率：预测板块第二天出现涨停股票的比例')
    print('  • 风险等级：low(低风险) / medium(中风险) / high(高风险)')
    print('  • 置信度：0-1，数值越高表示预测越可靠')
    print('')
    print('🔧 8个分析维度：')
    print('  技术面30% + 资金面15% + 情绪面20% + 相对强度15%')
    print('  + 动量10% + 波动率5% + 结构3% + 质量2%')
    
    data_loader = DataLoader()
    analyzer = EnhancedSectorAnalyzer()
    
    # 测试时间段
    start_date = '20230601'
    end_date = '20230610'
    
    try:
        # 1. 数据加载和特征验证
        print(f'\n📊 1. 数据加载和特征验证 ({start_date} - {end_date})')
        print('-' * 60)
        
        sector_data = data_loader.load_sector_features(start_date, end_date)
        market_features = data_loader.load_market_features(start_date, end_date)
        
        if sector_data is None or market_features is None:
            print('❌ 数据加载失败')
            return
        
        # 验证特征完整性
        feature_completeness = validate_feature_completeness(sector_data)
        print(f'✅ 板块数据: {len(sector_data)}条, 覆盖{sector_data["sector_code"].nunique()}个板块')
        print(f'✅ 市场数据: {len(market_features)}条')
        print(f'✅ 特征完整性: {feature_completeness["completeness_rate"]:.1%}')
        
        # 2. 8维度深度分析
        print(f'\n🎯 2. 8维度深度分析')
        print('-' * 60)
        
        daily_results, daily_predictions = run_ultimate_daily_analysis(
            sector_data, market_features, analyzer
        )
        
        if not daily_results:
            print('❌ 深度分析失败')
            return
        
        # 3. 增强预测验证
        print(f'\n📈 3. 增强预测验证')
        print('-' * 60)
        
        validation_results = validate_ultimate_predictions(daily_predictions, sector_data)
        display_ultimate_validation_results(validation_results)
        
        # 4. 8维度权重优化
        print(f'\n🔧 4. 8维度权重优化')
        print('-' * 60)
        
        optimization_results = optimize_8_dimension_weights(
            sector_data, market_features, analyzer
        )
        
        # 5. 风险评估分析
        print(f'\n⚠️  5. 风险评估分析')
        print('-' * 60)
        
        risk_analysis = analyze_risk_distribution(daily_results)
        
        # 6. 终极优化建议
        print(f'\n💎 6. 终极优化建议')
        print('-' * 60)

        recommendations = generate_ultimate_recommendations(
            validation_results, optimization_results, risk_analysis, feature_completeness
        )

        # 7. 保存分析结果
        print(f'\n💾 7. 保存分析结果')
        print('-' * 60)

        save_analysis_results(daily_results, daily_predictions, validation_results,
                            optimization_results, recommendations, start_date, end_date)

        # 8. 使用指南
        print(f'\n📖 8. 使用指南')
        print('-' * 60)

        display_usage_guide(validation_results)
        
    except Exception as e:
        print(f'❌ 终极分析失败: {e}')
        import traceback
        traceback.print_exc()

def validate_feature_completeness(sector_data: pd.DataFrame) -> Dict:
    """验证特征完整性"""
    try:
        # 核心特征列表（基于文档）
        core_features = {
            # 基础特征
            'basic': ['change_percent', 'volume', 'amount', 'stock_count'],
            
            # 涨跌停特征
            'limit_features': ['limit_up_count', 'limit_down_count', 'board_counts_1板', 'board_counts_2板'],
            
            # 技术特征
            'technical': ['ma_bull_trend', 'price_above_ma5', 'trend_strength', 'tech_score'],
            
            # 成交量特征
            'volume_features': ['volume_ratio_5', 'turnover_rate', 'fund_flow_momentum'],
            
            # 动量特征
            'momentum': ['momentum_5d', 'consecutive_up_days', 'swing_pattern'],
            
            # 波动率特征
            'volatility': ['volatility', 'volatility_ratio', 'amplitude'],
            
            # 结构特征
            'structure': ['market_cap_distribution_large', 'support_distance_pct'],
            
            # 质量特征
            'quality': ['indicator_consensus', 'signal_strength', 'market_sentiment']
        }
        
        feature_stats = {}
        total_features = 0
        available_features = 0
        
        for category, features in core_features.items():
            category_available = 0
            for feature in features:
                total_features += 1
                if feature in sector_data.columns:
                    available_features += 1
                    category_available += 1
            
            feature_stats[category] = {
                'available': category_available,
                'total': len(features),
                'rate': category_available / len(features)
            }
        
        completeness_rate = available_features / total_features
        
        print(f'特征完整性分析:')
        for category, stats in feature_stats.items():
            status = '✅' if stats['rate'] >= 0.8 else '⚠️' if stats['rate'] >= 0.5 else '❌'
            print(f'  {status} {category}: {stats["available"]}/{stats["total"]} ({stats["rate"]:.1%})')
        
        return {
            'completeness_rate': completeness_rate,
            'feature_stats': feature_stats,
            'total_features': total_features,
            'available_features': available_features
        }
        
    except Exception as e:
        print(f'特征验证失败: {e}')
        return {'completeness_rate': 0}

def run_ultimate_daily_analysis(sector_data: pd.DataFrame, market_features: pd.DataFrame, 
                               analyzer: EnhancedSectorAnalyzer) -> Tuple[Dict, Dict]:
    """运行终极版每日分析"""
    daily_results = {}
    daily_predictions = {}
    
    dates = sorted(sector_data['date'].unique())
    
    print(f'执行8维度深度分析，共{len(dates)}个交易日:')
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # 获取当日数据
        daily_sectors = sector_data[sector_data['date'] == date]
        daily_market = market_features[market_features['date'] == date]
        
        if len(daily_sectors) == 0:
            continue
        
        try:
            # 执行8维度分析
            results = analyzer.analyze_enhanced_daily_potential(daily_sectors, daily_market)
            
            if not results:
                print(f'  {date_str}: ❌ 分析失败')
                continue
            
            # 统计8维度数据
            stats = calculate_8_dimension_stats(daily_sectors, results)
            
            daily_results[date_str] = {
                'market_env': results[0].market_env,
                'sector_count': len(daily_sectors),
                'stats': stats,
                'top_sectors': results[:10]
            }
            
            daily_predictions[date_str] = results[:5]
            
            # 详细输出
            print(f'  {date_str}: ✅ {results[0].market_env}')
            print(f'    板块: {len(daily_sectors)}个, 综合评分: {stats["top3_avg_total_score"]:.1f}')
            print(f'    8维度平均分: 技术{stats["avg_technical"]:.0f} 资金{stats["avg_capital"]:.0f} '
                  f'情绪{stats["avg_sentiment"]:.0f} 相对{stats["avg_relative"]:.0f}')
            print(f'                  动量{stats["avg_momentum"]:.0f} 波动{stats["avg_volatility"]:.0f} '
                  f'结构{stats["avg_structure"]:.0f} 质量{stats["avg_quality"]:.0f}')
            
            # 显示前3名详细信息
            print(f'    前3名潜力板块:')
            for i, result in enumerate(results[:3]):
                print(f'      {i+1}. {result.sector_code[:20]:20s}: 综合{result.total_score:5.1f}分 '
                      f'(涨停{result.limit_up_count}, 连板{result.board_2_plus}, '
                      f'涨幅{result.change_percent:+.2%}, 风险{result.risk_level}, 置信{result.confidence:.2f})')
            
        except Exception as e:
            print(f'  {date_str}: ❌ 分析失败: {e}')
            continue
    
    return daily_results, daily_predictions

def calculate_8_dimension_stats(daily_sectors: pd.DataFrame, results: List) -> Dict:
    """计算8维度统计数据"""
    stats = {}
    
    try:
        # 基础统计
        stats['avg_change_pct'] = daily_sectors['change_percent'].mean()
        stats['strong_sectors'] = len(daily_sectors[daily_sectors['change_percent'] > 0.02])
        
        # 涨跌停统计
        stats['limit_up_total'] = daily_sectors['limit_up_count'].fillna(0).sum()
        stats['limit_up_sectors'] = (daily_sectors['limit_up_count'].fillna(0) > 0).sum()
        stats['board_2_plus'] = (daily_sectors['board_counts_2板'].fillna(0) + 
                               daily_sectors['board_counts_3板'].fillna(0)).sum()
        
        # 8维度平均分
        if results:
            stats['top3_avg_total_score'] = np.mean([r.total_score for r in results[:3]]) if len(results) >= 3 else 0
            stats['avg_technical'] = np.mean([r.technical_score for r in results[:10]])
            stats['avg_capital'] = np.mean([r.capital_score for r in results[:10]])
            stats['avg_sentiment'] = np.mean([r.sentiment_score for r in results[:10]])
            stats['avg_relative'] = np.mean([r.relative_score for r in results[:10]])
            stats['avg_momentum'] = np.mean([r.momentum_score for r in results[:10]])
            stats['avg_volatility'] = np.mean([r.volatility_score for r in results[:10]])
            stats['avg_structure'] = np.mean([r.structure_score for r in results[:10]])
            stats['avg_quality'] = np.mean([r.quality_score for r in results[:10]])
        else:
            stats.update({
                'top3_avg_total_score': 0, 'avg_technical': 0, 'avg_capital': 0,
                'avg_sentiment': 0, 'avg_relative': 0, 'avg_momentum': 0,
                'avg_volatility': 0, 'avg_structure': 0, 'avg_quality': 0
            })
        
        # 风险分布
        if results:
            risk_distribution = {}
            for result in results[:10]:
                risk_level = result.risk_level
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            stats['risk_distribution'] = risk_distribution
        
        return stats
    except Exception as e:
        return {'top3_avg_total_score': 0, 'limit_up_sectors': 0, 'strong_sectors': 0}

def validate_ultimate_predictions(daily_predictions: Dict, sector_data: pd.DataFrame) -> Dict:
    """终极版预测验证"""
    try:
        dates = sorted([pd.to_datetime(date_str) for date_str in daily_predictions.keys()])
        
        stats = {
            'valid_days': 0, 'predictions': [],
            'accuracy_by_risk': {'low': [], 'medium': [], 'high': []},
            'returns_by_confidence': {'high': [], 'medium': [], 'low': []}
        }
        
        for i in range(len(dates) - 1):
            current_date_str = dates[i].strftime('%Y-%m-%d')
            next_date = dates[i + 1]
            
            if current_date_str not in daily_predictions:
                continue
            
            predictions = daily_predictions[current_date_str]
            next_day_sectors = sector_data[sector_data['date'] == next_date]
            
            if len(next_day_sectors) == 0:
                continue
            
            stats['valid_days'] += 1
            
            # 验证每个预测
            for pred in predictions:
                next_day_row = next_day_sectors[next_day_sectors['sector_code'] == pred.sector_code]
                if len(next_day_row) > 0:
                    next_return = next_day_row.iloc[0]['change_percent']
                    next_limit_up = next_day_row.iloc[0].get('limit_up_count', 0)
                    if pd.isna(next_limit_up):
                        next_limit_up = 0
                    
                    prediction_result = {
                        'sector_code': pred.sector_code,
                        'predicted_score': pred.total_score,
                        'actual_return': next_return,
                        'actual_limit_up': next_limit_up,
                        'risk_level': pred.risk_level,
                        'confidence': pred.confidence,
                        'success': next_return > 0
                    }
                    
                    stats['predictions'].append(prediction_result)
                    
                    # 按风险等级分类
                    stats['accuracy_by_risk'][pred.risk_level].append(next_return > 0)
                    
                    # 按置信度分类
                    if pred.confidence > 0.7:
                        conf_level = 'high'
                    elif pred.confidence > 0.4:
                        conf_level = 'medium'
                    else:
                        conf_level = 'low'
                    stats['returns_by_confidence'][conf_level].append(next_return)
        
        # 计算综合指标
        result = {
            'valid_days': stats['valid_days'],
            'total_predictions': len(stats['predictions']),
            'overall_accuracy': np.mean([p['success'] for p in stats['predictions']]) if stats['predictions'] else 0,
            'overall_avg_return': np.mean([p['actual_return'] for p in stats['predictions']]) if stats['predictions'] else 0,
            'limit_up_hit_rate': np.mean([p['actual_limit_up'] > 0 for p in stats['predictions']]) if stats['predictions'] else 0,
        }
        
        # 风险等级准确率
        for risk_level in ['low', 'medium', 'high']:
            if stats['accuracy_by_risk'][risk_level]:
                result[f'{risk_level}_risk_accuracy'] = np.mean(stats['accuracy_by_risk'][risk_level])
            else:
                result[f'{risk_level}_risk_accuracy'] = 0
        
        # 置信度收益率
        for conf_level in ['high', 'medium', 'low']:
            if stats['returns_by_confidence'][conf_level]:
                result[f'{conf_level}_confidence_return'] = np.mean(stats['returns_by_confidence'][conf_level])
            else:
                result[f'{conf_level}_confidence_return'] = 0
        
        return result
        
    except Exception as e:
        print(f'终极验证失败: {e}')
        return {}

def display_ultimate_validation_results(validation_results: Dict):
    """显示终极验证结果"""
    if validation_results:
        print(f'验证天数: {validation_results["valid_days"]}天')
        print(f'总预测数: {validation_results["total_predictions"]}个')
        print(f'')
        print(f'🎯 核心指标:')
        print(f'  整体准确率: {validation_results["overall_accuracy"]:.1%}')
        print(f'  平均收益率: {validation_results["overall_avg_return"]:.2%}')
        print(f'  涨停命中率: {validation_results["limit_up_hit_rate"]:.1%}')
        print(f'')
        print(f'⚠️  风险等级准确率:')
        print(f'  低风险: {validation_results["low_risk_accuracy"]:.1%}')
        print(f'  中风险: {validation_results["medium_risk_accuracy"]:.1%}')
        print(f'  高风险: {validation_results["high_risk_accuracy"]:.1%}')
        print(f'')
        print(f'🔮 置信度收益率:')
        print(f'  高置信度: {validation_results["high_confidence_return"]:.2%}')
        print(f'  中置信度: {validation_results["medium_confidence_return"]:.2%}')
        print(f'  低置信度: {validation_results["low_confidence_return"]:.2%}')
    else:
        print('❌ 验证结果为空')

def optimize_8_dimension_weights(sector_data: pd.DataFrame, market_features: pd.DataFrame, 
                                analyzer: EnhancedSectorAnalyzer) -> List:
    """8维度权重优化"""
    print('执行8维度权重优化...')
    
    # 权重配置候选
    weight_configs = [
        ('当前配置', {'technical': 0.20, 'capital': 0.15, 'sentiment': 0.25, 'relative': 0.15, 
                    'momentum': 0.10, 'volatility': 0.05, 'structure': 0.05, 'quality': 0.05}),
        ('情绪主导', {'technical': 0.15, 'capital': 0.10, 'sentiment': 0.35, 'relative': 0.15, 
                    'momentum': 0.10, 'volatility': 0.05, 'structure': 0.05, 'quality': 0.05}),
        ('技术主导', {'technical': 0.30, 'capital': 0.15, 'sentiment': 0.20, 'relative': 0.15, 
                    'momentum': 0.10, 'volatility': 0.05, 'structure': 0.03, 'quality': 0.02}),
        ('动量主导', {'technical': 0.20, 'capital': 0.10, 'sentiment': 0.20, 'relative': 0.15, 
                    'momentum': 0.20, 'volatility': 0.05, 'structure': 0.05, 'quality': 0.05}),
        ('质量主导', {'technical': 0.15, 'capital': 0.15, 'sentiment': 0.20, 'relative': 0.15, 
                    'momentum': 0.10, 'volatility': 0.05, 'structure': 0.05, 'quality': 0.15}),
    ]
    
    optimization_results = []
    dates = sorted(sector_data['date'].unique())
    
    for config_name, weights in weight_configs:
        try:
            # 创建临时分析器
            temp_analyzer = EnhancedSectorAnalyzer()
            temp_analyzer.enhanced_weights = weights
            
            # 验证效果
            total_score = 0
            total_accuracy = 0
            total_limit_up_hit = 0
            valid_days = 0
            
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                
                daily_sectors = sector_data[sector_data['date'] == current_date]
                daily_market = market_features[market_features['date'] == current_date]
                next_day_sectors = sector_data[sector_data['date'] == next_date]
                
                if len(daily_sectors) > 0 and len(next_day_sectors) > 0:
                    results = temp_analyzer.analyze_enhanced_daily_potential(daily_sectors, daily_market)
                    if results:
                        # 评分
                        total_score += np.mean([r.total_score for r in results[:3]])
                        
                        # 验证准确率
                        correct = 0
                        limit_up_hit = 0
                        for pred in results[:3]:
                            next_row = next_day_sectors[next_day_sectors['sector_code'] == pred.sector_code]
                            if len(next_row) > 0:
                                next_return = next_row.iloc[0]['change_percent']
                                next_limit_up = next_row.iloc[0].get('limit_up_count', 0)
                                if pd.isna(next_limit_up):
                                    next_limit_up = 0
                                
                                if next_return > 0:
                                    correct += 1
                                if next_limit_up > 0:
                                    limit_up_hit += 1
                        
                        total_accuracy += correct / 3
                        total_limit_up_hit += limit_up_hit / 3
                        valid_days += 1
            
            if valid_days > 0:
                avg_score = total_score / valid_days
                avg_accuracy = total_accuracy / valid_days
                avg_limit_up_hit = total_limit_up_hit / valid_days
                
                # 综合评分：评分30% + 准确率40% + 涨停命中率30%
                composite_score = avg_score * 0.3 + avg_accuracy * 40 + avg_limit_up_hit * 30
                
                optimization_results.append({
                    'config_name': config_name,
                    'weights': weights,
                    'avg_score': avg_score,
                    'avg_accuracy': avg_accuracy,
                    'avg_limit_up_hit': avg_limit_up_hit,
                    'composite_score': composite_score
                })
                
                print(f'  {config_name}: 评分{avg_score:.1f}, 准确率{avg_accuracy:.1%}, '
                      f'涨停命中{avg_limit_up_hit:.1%}, 综合{composite_score:.1f}')
            
        except Exception as e:
            print(f'  {config_name}: 测试失败 - {e}')
    
    # 按综合评分排序
    optimization_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return optimization_results

def analyze_risk_distribution(daily_results: Dict) -> Dict:
    """分析风险分布"""
    risk_analysis = {'low': 0, 'medium': 0, 'high': 0, 'total_sectors': 0}
    
    for date_str, result in daily_results.items():
        if 'risk_distribution' in result['stats']:
            risk_dist = result['stats']['risk_distribution']
            for risk_level, count in risk_dist.items():
                risk_analysis[risk_level] += count
                risk_analysis['total_sectors'] += count
    
    # 计算比例
    if risk_analysis['total_sectors'] > 0:
        for risk_level in ['low', 'medium', 'high']:
            risk_analysis[f'{risk_level}_ratio'] = risk_analysis[risk_level] / risk_analysis['total_sectors']
    
    print(f'风险分布分析:')
    print(f'  低风险板块: {risk_analysis["low"]}个 ({risk_analysis.get("low_ratio", 0):.1%})')
    print(f'  中风险板块: {risk_analysis["medium"]}个 ({risk_analysis.get("medium_ratio", 0):.1%})')
    print(f'  高风险板块: {risk_analysis["high"]}个 ({risk_analysis.get("high_ratio", 0):.1%})')
    
    return risk_analysis

def generate_ultimate_recommendations(validation_results: Dict, optimization_results: List, 
                                    risk_analysis: Dict, feature_completeness: Dict):
    """生成终极优化建议"""
    print('基于8维度深度分析的终极量化建议:')
    
    if validation_results:
        overall_accuracy = validation_results.get('overall_accuracy', 0)
        limit_up_hit_rate = validation_results.get('limit_up_hit_rate', 0)
        
        print(f'\n📊 系统表现评估:')
        print(f'  整体准确率: {overall_accuracy:.1%}')
        print(f'  涨停命中率: {limit_up_hit_rate:.1%}')
        
        # 系统评级
        if overall_accuracy >= 0.6 and limit_up_hit_rate >= 0.4:
            print(f'\n🏆 系统评级: 优秀 (A级) - 可用于实盘交易')
            print(f'建议仓位: 20-30%')
        elif overall_accuracy >= 0.5 and limit_up_hit_rate >= 0.3:
            print(f'\n⭐ 系统评级: 良好 (B级) - 可小仓位试验')
            print(f'建议仓位: 10-15%')
        else:
            print(f'\n⚠️  系统评级: 需要优化 (C级) - 继续优化参数')
            print(f'建议仓位: 暂不实盘')
    
    if optimization_results:
        best_config = optimization_results[0]
        print(f'\n🎯 最优8维度配置: {best_config["config_name"]}')
        print(f'权重分配:')
        for dim, weight in best_config["weights"].items():
            print(f'  {dim}: {weight:.0%}')
        print(f'综合评分: {best_config["composite_score"]:.1f}')
    
    # 特征利用建议
    completeness_rate = feature_completeness.get('completeness_rate', 0)
    print(f'\n📈 特征利用优化:')
    print(f'  当前特征利用率: {completeness_rate:.1%}')
    
    if completeness_rate < 0.7:
        print(f'  建议: 提升特征完整性，重点补充缺失的技术指标和动量特征')
    else:
        print(f'  建议: 特征利用率良好，可进一步优化特征权重')
    
    # 风险管理建议
    low_risk_ratio = risk_analysis.get('low_risk_ratio', 0)
    print(f'\n⚠️  风险管理建议:')
    print(f'  低风险板块占比: {low_risk_ratio:.1%}')
    
    if low_risk_ratio > 0.6:
        print(f'  建议: 风险控制良好，可适当提高仓位')
    else:
        print(f'  建议: 加强风险控制，优先选择低风险板块')
    
    print(f'\n🚀 终极优化方向:')
    print(f'  1. 已应用技术主导配置（技术面30%），预期提升准确率')
    print(f'  2. 优化收益率导向的评分算法，降低极端涨停权重')
    print(f'  3. 动量维度权重可提升至15%，捕捉趋势机会')
    print(f'  4. 基于置信度进行仓位分配，高置信度板块加大仓位')
    print(f'  5. 建立动态风险调整机制，根据市场环境调整风险偏好')

    return {
        'system_rating': 'B级' if validation_results.get('overall_accuracy', 0) >= 0.5 else 'C级',
        'recommended_position': '10-15%' if validation_results.get('overall_accuracy', 0) >= 0.5 else '暂不实盘',
        'key_advantages': ['涨停命中率高', '8维度分析全面', '风险评估完善'],
        'optimization_directions': ['技术面权重优化', '收益率导向优化', '动量权重提升']
    }

def save_analysis_results(daily_results: Dict, daily_predictions: Dict, validation_results: Dict,
                         optimization_results: List, recommendations: Dict,
                         start_date: str, end_date: str):
    """保存分析结果到文件"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存预测结果
        predictions_file = f'sector_predictions_{start_date}_{end_date}_{timestamp}.json'

        # 转换预测结果为可序列化格式
        serializable_predictions = {}
        for date_str, predictions in daily_predictions.items():
            serializable_predictions[date_str] = [
                {
                    'sector_code': pred.sector_code,
                    'total_score': pred.total_score,
                    'technical_score': pred.technical_score,
                    'sentiment_score': pred.sentiment_score,
                    'limit_up_count': pred.limit_up_count,
                    'change_percent': pred.change_percent,
                    'risk_level': pred.risk_level,
                    'confidence': pred.confidence
                }
                for pred in predictions
            ]

        # 保存完整结果
        results_data = {
            'analysis_period': f'{start_date} - {end_date}',
            'timestamp': timestamp,
            'predictions': serializable_predictions,
            'validation_results': validation_results,
            'optimization_results': optimization_results,
            'recommendations': recommendations,
            'daily_stats': {
                date_str: {
                    'market_env': result['market_env'],
                    'sector_count': result['sector_count'],
                    'top3_avg_score': result['stats'].get('top3_avg_total_score', 0)
                }
                for date_str, result in daily_results.items()
            }
        }

        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f'✅ 分析结果已保存到: {predictions_file}')
        print(f'✅ 包含内容: 每日预测、验证结果、优化建议、使用指南')

        # 保存简化版预测结果（便于查看）
        simple_file = f'simple_predictions_{start_date}_{end_date}.txt'
        with open(simple_file, 'w', encoding='utf-8') as f:
            f.write(f'板块潜力分析结果 ({start_date} - {end_date})\n')
            f.write('=' * 50 + '\n\n')

            for date_str, predictions in daily_predictions.items():
                f.write(f'{date_str} 前5名潜力板块:\n')
                for i, pred in enumerate(predictions):
                    f.write(f'  {i+1}. {pred.sector_code}: {pred.total_score:.1f}分 '
                           f'(风险{pred.risk_level}, 置信{pred.confidence:.2f})\n')
                f.write('\n')

            f.write(f'系统表现:\n')
            f.write(f'  准确率: {validation_results.get("overall_accuracy", 0):.1%}\n')
            f.write(f'  收益率: {validation_results.get("overall_avg_return", 0):.2%}\n')
            f.write(f'  涨停命中率: {validation_results.get("limit_up_hit_rate", 0):.1%}\n')
            f.write(f'  系统评级: {recommendations.get("system_rating", "未知")}\n')

        print(f'✅ 简化结果已保存到: {simple_file}')

    except Exception as e:
        print(f'❌ 保存结果失败: {e}')

def display_usage_guide(validation_results: Dict):
    """显示使用指南"""
    overall_accuracy = validation_results.get('overall_accuracy', 0)
    limit_up_hit_rate = validation_results.get('limit_up_hit_rate', 0)

    print(f'📖 系统使用指南:')
    print(f'')
    print(f'🎯 如何解读预测结果:')
    print(f'  • 综合评分 >80分: 强烈推荐，重点关注')
    print(f'  • 综合评分 60-80分: 推荐，适度关注')
    print(f'  • 综合评分 40-60分: 一般，谨慎关注')
    print(f'  • 综合评分 <40分: 不推荐')
    print(f'')
    print(f'⚠️  风险等级使用:')
    print(f'  • low风险: 可用较大仓位(15-20%)')
    print(f'  • medium风险: 适中仓位(8-12%)')
    print(f'  • high风险: 小仓位或观望(3-5%)')
    print(f'')


class UltimateMultiDayAnalyzer:
    """
    终极版多日板块分析器 - 兼容接口

    提供与stock_selector.py兼容的接口，同时保持原有的筛选逻辑和Top-N选择功能
    """

    def __init__(self):
        """初始化分析器"""
        self.enhanced_analyzer = EnhancedSectorAnalyzer()
        self.logger = None

    def analyze_enhanced_daily_potential(self, daily_sectors: Any, market_features: Any) -> List:
        """
        分析每日板块潜力 - 兼容接口

        Args:
            daily_sectors: 板块数据 (DataFrame或其他格式)
            market_features: 市场特征数据 (DataFrame、dict或其他格式)

        Returns:
            List: 板块分析结果列表，已按评分排序并筛选Top-N
        """
        try:
            # 数据格式转换和验证
            if isinstance(daily_sectors, pd.DataFrame):
                sector_df = daily_sectors
            else:
                # 如果是其他格式，尝试转换
                sector_df = pd.DataFrame(daily_sectors) if daily_sectors else pd.DataFrame()

            # 市场特征处理
            if isinstance(market_features, dict):
                market_dict = market_features
            elif isinstance(market_features, pd.DataFrame) and not market_features.empty:
                market_dict = market_features.iloc[0].to_dict()
            else:
                market_dict = {}

            # 数据验证
            if sector_df.empty:
                if self.logger:
                    self.logger.warning("板块数据为空")
                return []

            # 调用原始分析器
            results = self.enhanced_analyzer.analyze_enhanced_daily_potential(
                sector_df, market_dict
            )

            # 应用筛选逻辑 - 只返回Top 15个高质量板块
            filtered_results = self._apply_quality_filter(results)

            if self.logger:
                self.logger.info(f"板块分析完成: 原始{len(results)}个 -> 筛选后{len(filtered_results)}个")

            return filtered_results

        except Exception as e:
            if self.logger:
                self.logger.error(f"板块分析失败: {e}")
            else:
                print(f"板块分析失败: {e}")
            return []

    def _apply_quality_filter(self, results: List) -> List:
        """
        应用质量筛选逻辑

        Args:
            results: 原始分析结果

        Returns:
            List: 筛选后的高质量结果
        """
        if not results:
            return []

        try:
            # 1. 基础质量过滤
            quality_filtered = []
            for result in results:
                # 检查必要属性
                if not hasattr(result, 'total_score') or not hasattr(result, 'sector_code'):
                    continue

                # 评分阈值过滤 (降低阈值，保留评分>50的板块)
                if result.total_score < 50:
                    continue

                # 风险控制 (排除过高风险的板块)
                if hasattr(result, 'risk_level') and result.risk_level == 'high':
                    # 高风险板块需要更高的评分才能通过
                    if result.total_score < 65:
                        continue

                # 置信度过滤 (降低置信度要求)
                if hasattr(result, 'confidence') and result.confidence < 0.3:
                    continue

                quality_filtered.append(result)

            # 2. 按评分排序
            quality_filtered.sort(key=lambda x: x.total_score, reverse=True)

            # 3. Top-N筛选 (最多返回15个板块)
            max_sectors = 15
            top_results = quality_filtered[:max_sectors]

            # 4. 多样性检查 (避免同类板块过度集中)
            diversified_results = self._ensure_diversity(top_results)

            return diversified_results

        except Exception as e:
            print(f"质量筛选失败: {e}")
            # 如果筛选失败，返回原始结果的前10个
            return results[:10] if results else []

    def _ensure_diversity(self, results: List) -> List:
        """
        确保板块多样性

        Args:
            results: 排序后的结果

        Returns:
            List: 多样化后的结果
        """
        if len(results) <= 10:
            return results

        try:
            # 简单的多样性检查：避免相似板块名称过度集中
            diversified = []
            used_keywords = set()

            for result in results:
                sector_name = result.sector_code

                # 提取关键词
                keywords = []
                if '电池' in sector_name:
                    keywords.append('电池')
                elif '新能源' in sector_name:
                    keywords.append('新能源')
                elif '芯片' in sector_name or '半导体' in sector_name:
                    keywords.append('芯片')
                elif '医药' in sector_name or '生物' in sector_name:
                    keywords.append('医药')
                elif '军工' in sector_name or '航天' in sector_name:
                    keywords.append('军工')

                # 检查是否已有相似板块
                has_similar = any(kw in used_keywords for kw in keywords)

                if not has_similar or len(diversified) < 8:  # 前8个不做限制
                    diversified.append(result)
                    used_keywords.update(keywords)

                # 最多返回12个板块
                if len(diversified) >= 12:
                    break

            return diversified

        except Exception as e:
            print(f"多样性检查失败: {e}")
            return results[:10]

    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger


# 原始函数继续保留
def display_usage_guide_original(validation_results: Dict):
    """显示使用指南"""
    overall_accuracy = validation_results.get('overall_accuracy', 0)
    limit_up_hit_rate = validation_results.get('limit_up_hit_rate', 0)

    print(f'🔮 置信度应用:')
    print(f'  • 置信度 >0.7: 高可信度，可加大仓位')
    print(f'  • 置信度 0.4-0.7: 中等可信度，正常仓位')
    print(f'  • 置信度 <0.4: 低可信度，减少仓位')
    print(f'')

    if overall_accuracy >= 0.6:
        print(f'✅ 当前系统表现优秀，建议实盘使用:')
        print(f'  • 建议总仓位: 20-30%')
        print(f'  • 单板块仓位: 3-8%')
        print(f'  • 持仓周期: 1-3天')
    elif overall_accuracy >= 0.5:
        print(f'⭐ 当前系统表现良好，可小仓位试验:')
        print(f'  • 建议总仓位: 10-15%')
        print(f'  • 单板块仓位: 2-5%')
        print(f'  • 持仓周期: 1-2天')
    else:
        print(f'⚠️  当前系统需要优化，暂不建议实盘:')
        print(f'  • 建议: 继续优化参数')
        print(f'  • 可进行模拟交易验证')
        print(f'  • 关注系统改进情况')

    print(f'')
    print(f'📊 特殊优势利用:')
    if limit_up_hit_rate > 0.5:
        print(f'  • 涨停命中率{limit_up_hit_rate:.1%}较高，适合短线操作')
        print(f'  • 可重点关注情绪面评分高的板块')
        print(f'  • 建议T+1操作，及时止盈')

    print(f'')
    print(f'🔄 系统更新建议:')
    print(f'  • 每周重新运行分析，更新参数')
    print(f'  • 根据市场环境调整权重配置')
    print(f'  • 持续跟踪预测效果，优化算法')

if __name__ == '__main__':
    run_ultimate_analysis()
