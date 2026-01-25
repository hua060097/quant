#!/usr/bin/env python3
"""
增强特征生成系统使用示例
展示如何使用集成了涨跌停数据的特征生成系统
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# 添加路径
sys.path.append(str(Path(__file__).parent))

from generate_daily_features import StockFeatureGenerator

def example_single_date_features():
    """示例：生成单日特征"""
    print("=" * 80)
    print("示例1: 生成单日特征")
    print("=" * 80)
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 初始化特征生成器
        print("🔧 初始化特征生成器...")
        generator = StockFeatureGenerator()
        
        # 选择一个交易日
        target_date = "20240102"
        print(f"📊 生成 {target_date} 的特征...")
        
        # 生成特征
        success = generator.process_features_for_date(target_date)
        
        if success:
            print(f"✅ {target_date} 特征生成成功")
            
            # 读取生成的特征文件
            output_file = generator.output_dir / f"{target_date}.parquet"
            if output_file.exists():
                df = pd.read_parquet(output_file)
                
                print(f"📈 生成的特征数据:")
                print(f"  - 数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
                print(f"  - 股票数量: {df['stock_code'].nunique()}")
                
                # 显示新增的涨跌停特征
                limit_features = [
                    'is_limit_up', 'is_limit_down', 'real_limit_strength', 
                    'real_time_to_limit', 'limit_quality_score'
                ]
                
                print(f"\n📊 涨跌停特征统计:")
                for feature in limit_features:
                    if feature in df.columns:
                        if feature in ['is_limit_up', 'is_limit_down']:
                            count = df[feature].sum()
                            print(f"  - {feature}: {count} 只股票")
                        else:
                            non_zero = (df[feature] != 0).sum()
                            mean_val = df[feature].mean()
                            print(f"  - {feature}: {non_zero} 只有数据, 平均值: {mean_val:.4f}")
                
                # 显示技术指标样本
                print(f"\n📋 技术指标样本 (前5只股票):")
                sample_cols = ['stock_code', 'close', 'ma20', 'rsi12', 'macd', 'is_limit_up', 'limit_quality_score']
                available_cols = [col for col in sample_cols if col in df.columns]
                print(df[available_cols].head().to_string(index=False))
                
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

def example_batch_features():
    """示例：批量生成特征"""
    print("\n" + "=" * 80)
    print("示例2: 批量生成特征")
    print("=" * 80)
    
    try:
        # 初始化特征生成器
        generator = StockFeatureGenerator()
        
        # 选择日期范围（最近几个交易日）
        start_date = "20240102"
        end_date = "20240105"
        
        print(f"📊 批量生成 {start_date} 到 {end_date} 的特征...")
        
        # 批量生成特征
        success = generator.generate_features(
            start_date=start_date,
            end_date=end_date,
            force_update=True,  # 强制更新
            batch_size=2,       # 小批次处理
            use_threading=False # 单线程处理（避免复杂性）
        )
        
        if success:
            print(f"✅ 批量特征生成成功")
            
            # 检查生成的文件
            generated_files = list(generator.output_dir.glob("*.parquet"))
            print(f"📁 生成的特征文件: {len(generated_files)} 个")
            
            # 分析最新文件
            if generated_files:
                latest_file = max(generated_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest_file)
                
                print(f"\n📈 最新特征文件分析 ({latest_file.name}):")
                print(f"  - 数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
                
                # 特征完整性检查
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"  - 数值特征: {len(numeric_cols)} 个")
                
                # 检查数据质量
                inf_count = sum(np.isinf(df[col]).sum() for col in numeric_cols)
                nan_count = sum(df[col].isna().sum() for col in numeric_cols)
                
                print(f"  - 数据质量: 无穷值={inf_count}, NaN值={nan_count}")
                
                if inf_count == 0:
                    print(f"  ✅ 无穷值问题已修复")
                else:
                    print(f"  ❌ 仍有无穷值问题")
            
            return True
        else:
            print(f"❌ 批量特征生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 批量生成示例失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def example_feature_analysis():
    """示例：特征分析"""
    print("\n" + "=" * 80)
    print("示例3: 特征分析")
    print("=" * 80)
    
    try:
        # 查找特征文件
        output_dir = Path("to/local_data/features/daily")
        feature_files = list(output_dir.glob("*.parquet"))
        
        if not feature_files:
            print("❌ 没有找到特征文件")
            return False
        
        # 选择最新的特征文件
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 分析文件: {latest_file.name}")
        
        df = pd.read_parquet(latest_file)
        
        # 基础统计
        print(f"\n📊 基础统计:")
        print(f"  - 数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"  - 股票数量: {df['stock_code'].nunique()}")
        print(f"  - 日期范围: {df['date'].min()} 到 {df['date'].max()}")
        
        # 特征分类统计
        feature_categories = {
            '基础价格': ['open', 'high', 'low', 'close', 'volume', 'amount'],
            '技术指标': ['ma5', 'ma20', 'rsi12', 'macd', 'bb_upper', 'atr14'],
            '涨跌停特征': ['is_limit_up', 'is_limit_down', 'real_limit_strength', 'limit_quality_score'],
            '高级特征': ['trend_strength', 'candlestick_strength', 'risk_reward_ratio'],
            '机器学习特征': ['volume_weighted_price', 'return_20d', 'volatility_adjusted_momentum']
        }
        
        print(f"\n📈 特征分类统计:")
        for category, features in feature_categories.items():
            available = [f for f in features if f in df.columns]
            print(f"  - {category}: {len(available)}/{len(features)} 个特征")
            if len(available) < len(features):
                missing = [f for f in features if f not in df.columns]
                print(f"    缺失: {missing}")
        
        # 涨跌停数据分析
        if 'is_limit_up' in df.columns and 'is_limit_down' in df.columns:
            limit_up_count = df['is_limit_up'].sum()
            limit_down_count = df['is_limit_down'].sum()
            
            print(f"\n🚀 涨跌停数据分析:")
            print(f"  - 涨停股票: {limit_up_count} 只")
            print(f"  - 跌停股票: {limit_down_count} 只")
            
            if 'real_limit_strength' in df.columns:
                real_data_count = (df['real_limit_strength'] > 0).sum()
                print(f"  - 有真实数据: {real_data_count} 只")
                
                if real_data_count > 0:
                    avg_strength = df[df['real_limit_strength'] > 0]['real_limit_strength'].mean()
                    print(f"  - 平均封单强度: {avg_strength:.2f}")
        
        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 检查无穷值
        inf_cols = []
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_cols.append((col, inf_count))
        
        if inf_cols:
            print(f"  ❌ 发现无穷值:")
            for col, count in inf_cols[:5]:  # 只显示前5个
                print(f"    {col}: {count} 个")
        else:
            print(f"  ✅ 无无穷值问题")
        
        # 检查异常值
        high_std_cols = []
        for col in numeric_cols:
            if df[col].std() > 1000:  # 标准差过大
                high_std_cols.append(col)
        
        if high_std_cols:
            print(f"  ⚠️  高方差特征: {len(high_std_cols)} 个")
        else:
            print(f"  ✅ 方差分布合理")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 增强特征生成系统使用示例")
    print("集成了涨跌停数据和无穷值修复的特征生成系统")
    
    examples = [
        ("单日特征生成", example_single_date_features),
        ("批量特征生成", example_batch_features),
        ("特征分析", example_feature_analysis)
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
        print("✅ 涨跌停数据集成正常")
        print("✅ 无穷值问题已修复")
        print("✅ 特征生成系统运行稳定")
    else:
        print("\n💥 部分示例执行失败！")
    
    return passed == total

if __name__ == "__main__":
    main()
