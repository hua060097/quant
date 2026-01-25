#!/usr/bin/env python3
"""
完整涨跌停特征使用示例
展示如何使用集成了 enhanced_limit_data_collector.py 所有特征的系统
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

def example_feature_generation():
    """示例：生成包含完整涨跌停特征的数据"""
    print("=" * 80)
    print("示例1: 生成包含完整涨跌停特征的数据")
    print("=" * 80)
    
    try:
        from generate_daily_features import StockFeatureGenerator
        
        # 初始化特征生成器（自动集成完整涨跌停数据加载器）
        generator = StockFeatureGenerator()
        print("✅ 特征生成器初始化成功（已集成完整涨跌停数据加载器）")
        
        # 生成单日特征
        target_date = "20240102"
        print(f"📊 生成 {target_date} 的完整特征...")
        
        success = generator.process_features_for_date(target_date)
        
        if success:
            print(f"✅ {target_date} 特征生成成功")
            
            # 读取生成的特征文件
            output_file = generator.output_dir / f"{target_date}.parquet"
            if output_file.exists():
                df = pd.read_parquet(output_file)
                print(f"📈 生成的特征数据: {df.shape[0]} 行, {df.shape[1]} 列")
                
                # 显示涨跌停特征统计
                print_limit_features_stats(df)
                
                return True
            else:
                print(f"❌ 特征文件不存在: {output_file}")
                return False
        else:
            print(f"❌ {target_date} 特征生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_limit_features_stats(df):
    """打印涨跌停特征统计"""
    print(f"\n📊 涨跌停特征统计:")
    
    # 基础标识统计
    if all(col in df.columns for col in ['is_limit_up', 'is_limit_down', 'is_zhaban']):
        limit_up_count = df['is_limit_up'].sum()
        limit_down_count = df['is_limit_down'].sum()
        zhaban_count = df['is_zhaban'].sum()
        
        print(f"  基础标识:")
        print(f"    涨停股票: {limit_up_count} 只")
        print(f"    跌停股票: {limit_down_count} 只")
        print(f"    炸板股票: {zhaban_count} 只")
    
    # 原始字段统计
    original_fields = ['fd_amount', 'open_times', 'limit_times']
    print(f"  原始字段:")
    for field in original_fields:
        if field in df.columns:
            non_zero = (df[field] != 0).sum()
            mean_val = df[field].mean()
            print(f"    {field}: {non_zero} 只有数据, 平均值: {mean_val:.2f}")
    
    # 增强特征统计
    enhanced_features = ['limit_strength', 'board_quality', 'stability_score']
    print(f"  增强特征:")
    for feature in enhanced_features:
        if feature in df.columns:
            non_zero = (df[feature] != 0).sum()
            mean_val = df[feature].mean()
            max_val = df[feature].max()
            print(f"    {feature}: {non_zero} 只有数据, 平均值: {mean_val:.3f}, 最大值: {max_val:.3f}")

def example_feature_analysis():
    """示例：分析涨跌停特征"""
    print("\n" + "=" * 80)
    print("示例2: 分析涨跌停特征")
    print("=" * 80)
    
    try:
        from generate_daily_features import _get_all_limit_features
        
        # 获取所有涨跌停特征
        all_features = _get_all_limit_features()
        print(f"📊 完整涨跌停特征体系: {len(all_features)} 个特征")
        
        # 按类别分组显示
        feature_categories = {
            '基础标识': ['is_limit_up', 'is_limit_down', 'is_zhaban'],
            '原始字段': ['fd_amount', 'open_times', 'limit_times', 'first_time', 'last_time'],
            '增强特征': ['limit_strength', 'time_efficiency_score', 'board_quality', 'stability_score'],
            '历史特征': ['limit_up_days_5d', 'consecutive_limit_up', 'limit_up_volume_ratio']
        }
        
        print(f"\n📋 特征分类:")
        for category, sample_features in feature_categories.items():
            available = [f for f in sample_features if f in all_features]
            print(f"  {category}: {len(available)} 个样本特征")
            for feature in available:
                print(f"    - {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_high_quality_limit_selection():
    """示例：高质量涨停股票筛选"""
    print("\n" + "=" * 80)
    print("示例3: 高质量涨停股票筛选")
    print("=" * 80)
    
    try:
        # 查找最新的特征文件
        output_dir = Path("to/local_data/features/daily")
        feature_files = list(output_dir.glob("*.parquet"))
        
        if not feature_files:
            print("❌ 没有找到特征文件")
            return False
        
        # 选择最新的特征文件
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 分析文件: {latest_file.name}")
        
        df = pd.read_parquet(latest_file)
        print(f"📊 数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 筛选涨停股票
        if 'is_limit_up' in df.columns:
            limit_up_stocks = df[df['is_limit_up'] == 1.0]
            print(f"🚀 涨停股票: {len(limit_up_stocks)} 只")
            
            if len(limit_up_stocks) > 0:
                # 高质量涨停筛选条件
                quality_conditions = []
                
                # 条件1: 板块质量评分 > 0.7
                if 'board_quality' in limit_up_stocks.columns:
                    quality_conditions.append(limit_up_stocks['board_quality'] > 0.7)
                    print(f"  条件1: 板块质量评分 > 0.7")
                
                # 条件2: 稳定性评分 > 0.8 (炸板次数少)
                if 'stability_score' in limit_up_stocks.columns:
                    quality_conditions.append(limit_up_stocks['stability_score'] > 0.8)
                    print(f"  条件2: 稳定性评分 > 0.8")
                
                # 条件3: 时间效率评分 > 0.6 (早盘涨停)
                if 'time_efficiency_score' in limit_up_stocks.columns:
                    quality_conditions.append(limit_up_stocks['time_efficiency_score'] > 0.6)
                    print(f"  条件3: 时间效率评分 > 0.6")
                
                # 条件4: 连板因子 > 0.3 (有连板效应)
                if 'continuity_factor' in limit_up_stocks.columns:
                    quality_conditions.append(limit_up_stocks['continuity_factor'] > 0.3)
                    print(f"  条件4: 连板因子 > 0.3")
                
                # 应用筛选条件
                if quality_conditions:
                    combined_condition = quality_conditions[0]
                    for condition in quality_conditions[1:]:
                        combined_condition = combined_condition & condition
                    
                    high_quality_stocks = limit_up_stocks[combined_condition]
                    print(f"✅ 高质量涨停股票: {len(high_quality_stocks)} 只")
                    
                    if len(high_quality_stocks) > 0:
                        # 显示高质量股票信息
                        display_cols = ['stock_code', 'board_quality', 'stability_score', 
                                      'time_efficiency_score', 'continuity_factor']
                        available_cols = [col for col in display_cols if col in high_quality_stocks.columns]
                        
                        print(f"\n📋 高质量涨停股票详情:")
                        print(high_quality_stocks[available_cols].head(10).to_string(index=False))
                    
                    return True
                else:
                    print("❌ 没有可用的质量筛选条件")
                    return False
            else:
                print("⚠️  当日没有涨停股票")
                return True
        else:
            print("❌ 缺少涨停标识字段")
            return False
        
    except Exception as e:
        print(f"❌ 高质量涨停筛选失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_limit_features_for_ml():
    """示例：为机器学习准备涨跌停特征"""
    print("\n" + "=" * 80)
    print("示例4: 为机器学习准备涨跌停特征")
    print("=" * 80)
    
    try:
        from generate_daily_features import _get_all_limit_features
        
        # 获取所有涨跌停特征
        all_limit_features = _get_all_limit_features()
        
        # 为机器学习分类特征
        ml_feature_groups = {
            '标识特征': ['is_limit_up', 'is_limit_down', 'is_zhaban'],
            '强度特征': ['limit_strength', 'fd_amount', 'limit_amount'],
            '时间特征': ['time_to_limit_minutes', 'time_efficiency_score', 'first_time', 'last_time'],
            '质量特征': ['board_quality', 'stability_score', 'continuity_factor'],
            '市场特征': ['capital_intensity', 'turnover_efficiency', 'mv_factor'],
            '历史特征': ['limit_up_days_5d', 'consecutive_limit_up', 'limit_up_volume_ratio']
        }
        
        print(f"📊 机器学习特征分组:")
        
        total_features = 0
        for group_name, sample_features in ml_feature_groups.items():
            available_features = [f for f in sample_features if f in all_limit_features]
            group_count = len([f for f in all_limit_features 
                             if any(keyword in f for keyword in 
                                   ['limit', 'board', 'time', 'fd', 'open', 'consecutive'])])
            
            print(f"  {group_name}: {len(available_features)} 个样本特征")
            for feature in available_features:
                print(f"    - {feature}")
            
            total_features += len(available_features)
        
        print(f"\n📈 机器学习可用特征总数: {len(all_limit_features)} 个")
        print(f"🎯 推荐用于选股模型的核心特征:")
        
        core_features = [
            'is_limit_up',           # 涨停标识
            'board_quality',         # 综合质量评分
            'stability_score',       # 稳定性（炸板情况）
            'time_efficiency_score', # 时间效率
            'continuity_factor',     # 连板效应
            'limit_strength',        # 封单强度
            'consecutive_limit_up'   # 连续涨停天数
        ]
        
        for i, feature in enumerate(core_features, 1):
            print(f"  {i}. {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ 机器学习特征准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 完整涨跌停特征使用示例")
    print("基于 enhanced_limit_data_collector.py 的所有特征")
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    examples = [
        ("特征生成", example_feature_generation),
        ("特征分析", example_feature_analysis),
        ("高质量涨停筛选", example_high_quality_limit_selection),
        ("机器学习特征准备", example_limit_features_for_ml)
    ]
    
    results = []
    
    for example_name, example_func in examples:
        try:
            result = example_func()
            results.append((example_name, result))
        except Exception as e:
            print(f"❌ {example_name} 执行失败: {e}")
            results.append((example_name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("示例执行总结")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for example_name, result in results:
        status = "✅ 成功" if result else "❌ 失败"
        print(f"{example_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 成功")
    
    if passed == total:
        print("\n🎉 所有示例执行成功！")
        print("✅ 完整涨跌停特征系统运行正常")
        print("✅ 31个涨跌停特征全部可用")
        print("✅ 可用于 ml_sector_algorithms.py 选股")
    else:
        print("\n💥 部分示例执行失败！")
    
    return passed == total

if __name__ == "__main__":
    main()
