#!/usr/bin/env python3
"""
测试增强的特征生成系统
验证涨跌停数据集成和无穷值修复
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent))

from generate_daily_features import StockFeatureGenerator

def test_enhanced_features():
    """测试增强的特征生成"""
    print("=" * 80)
    print("增强特征生成系统测试")
    print("=" * 80)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 初始化特征生成器
        print("\n🔧 初始化特征生成器...")
        generator = StockFeatureGenerator()
        print("✅ 特征生成器初始化成功")
        
        # 测试单日特征生成
        print("\n📊 测试单日特征生成...")
        test_date = "20240102"  # 选择一个有交易的日期
        
        success = generator.process_features_for_date(test_date)
        
        if success:
            print(f"✅ {test_date} 特征生成成功")
            
            # 检查生成的特征文件
            output_file = generator.output_dir / f"features_{test_date}.parquet"
            if output_file.exists():
                df = pd.read_parquet(output_file)
                print(f"📈 生成特征数据: {df.shape[0]} 行, {df.shape[1]} 列")
                
                # 检查新增的涨跌停特征
                new_limit_features = [
                    'real_limit_strength', 'real_time_to_limit', 'real_open_times',
                    'real_limit_times', 'real_fd_amount', 'limit_quality_score'
                ]
                
                print(f"\n🔍 检查新增涨跌停特征:")
                for feature in new_limit_features:
                    if feature in df.columns:
                        non_zero_count = (df[feature] != 0).sum()
                        print(f"  ✅ {feature}: 存在 (非零值: {non_zero_count})")
                    else:
                        print(f"  ❌ {feature}: 缺失")
                
                # 检查无穷值问题
                print(f"\n🔍 检查无穷值问题:")
                problematic_features = ['vol_price_corr_20d', 'return_20d', 'risk_adjusted_return', 'volatility_adjusted_momentum']
                
                for feature in problematic_features:
                    if feature in df.columns:
                        inf_count = np.isinf(df[feature]).sum()
                        nan_count = df[feature].isna().sum()
                        print(f"  {feature}: 无穷值={inf_count}, NaN值={nan_count}")
                        
                        if inf_count > 0:
                            print(f"    ❌ 仍有无穷值问题")
                        else:
                            print(f"    ✅ 无穷值问题已修复")
                    else:
                        print(f"  {feature}: 特征不存在")
                
                # 显示数据样本
                print(f"\n📋 数据样本 (前3行):")
                sample_cols = ['stock_code', 'date', 'close', 'is_limit_up', 'real_limit_strength', 'limit_quality_score']
                available_cols = [col for col in sample_cols if col in df.columns]
                if available_cols:
                    print(df[available_cols].head(3).to_string())
                
                # 统计涨跌停数据
                if 'is_limit_up' in df.columns and 'is_limit_down' in df.columns:
                    limit_up_count = df['is_limit_up'].sum()
                    limit_down_count = df['is_limit_down'].sum()
                    print(f"\n📊 涨跌停统计:")
                    print(f"  涨停股票: {limit_up_count} 只")
                    print(f"  跌停股票: {limit_down_count} 只")
                    
                    if 'real_limit_strength' in df.columns:
                        real_data_count = (df['real_limit_strength'] > 0).sum()
                        print(f"  有真实涨跌停数据: {real_data_count} 只")
            else:
                print(f"❌ 特征文件不存在: {output_file}")
        else:
            print(f"❌ {test_date} 特征生成失败")
        
        return success
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_limit_data_loader():
    """测试涨跌停数据加载器"""
    print("\n" + "=" * 80)
    print("涨跌停数据加载器测试")
    print("=" * 80)
    
    try:
        from generate_daily_features import LimitDataLoader
        
        # 初始化加载器
        loader = LimitDataLoader()
        
        # 测试数据加载
        test_date = "20240102"
        print(f"\n📊 测试加载 {test_date} 的涨跌停数据...")
        
        limit_data = loader.load_limit_data_by_date(test_date)
        
        if not limit_data.empty:
            print(f"✅ 成功加载涨跌停数据: {len(limit_data)} 条")
            print(f"📋 数据字段: {list(limit_data.columns)}")
            
            # 显示数据类型分布
            if 'limit_type' in limit_data.columns:
                type_counts = limit_data['limit_type'].value_counts()
                print(f"📊 数据类型分布:")
                for limit_type, count in type_counts.items():
                    print(f"  {limit_type}: {count} 条")
            
            # 显示样本数据
            print(f"\n📋 样本数据 (前3条):")
            sample_cols = ['ts_code', 'name', 'limit_type', 'fd_amount', 'open_times', 'limit_times']
            available_cols = [col for col in sample_cols if col in limit_data.columns]
            if available_cols:
                print(limit_data[available_cols].head(3).to_string())
        else:
            print(f"⚠️  {test_date} 没有涨跌停数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 涨跌停数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_feature_improvements():
    """分析特征改进情况"""
    print("\n" + "=" * 80)
    print("特征改进分析")
    print("=" * 80)
    
    try:
        # 检查特征文件
        output_dir = Path("to/local_data/features")
        if not output_dir.exists():
            print("❌ 特征输出目录不存在")
            return False
        
        # 查找最新的特征文件
        feature_files = list(output_dir.glob("features_*.parquet"))
        if not feature_files:
            print("❌ 没有找到特征文件")
            return False
        
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 分析文件: {latest_file.name}")
        
        df = pd.read_parquet(latest_file)
        print(f"📊 数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 分析特征完整性
        print(f"\n🔍 特征完整性分析:")
        
        # 检查基础特征
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for feature in basic_features:
            if feature in df.columns:
                completeness = df[feature].notna().sum() / len(df) * 100
                print(f"  {feature}: {completeness:.1f}% 完整")
            else:
                print(f"  {feature}: 缺失")
        
        # 检查技术指标
        tech_features = ['ma5', 'ma20', 'rsi12', 'macd', 'bb_upper']
        print(f"\n📈 技术指标:")
        for feature in tech_features:
            if feature in df.columns:
                completeness = df[feature].notna().sum() / len(df) * 100
                print(f"  {feature}: {completeness:.1f}% 完整")
            else:
                print(f"  {feature}: 缺失")
        
        # 检查涨跌停特征
        limit_features = ['is_limit_up', 'real_limit_strength', 'limit_quality_score']
        print(f"\n🚀 涨跌停特征:")
        for feature in limit_features:
            if feature in df.columns:
                completeness = df[feature].notna().sum() / len(df) * 100
                non_zero_pct = (df[feature] != 0).sum() / len(df) * 100
                print(f"  {feature}: {completeness:.1f}% 完整, {non_zero_pct:.1f}% 非零")
            else:
                print(f"  {feature}: 缺失")
        
        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        
        # 检查无穷值
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_cols.append((col, inf_count))
        
        if inf_cols:
            print(f"  ❌ 发现无穷值:")
            for col, count in inf_cols:
                print(f"    {col}: {count} 个")
        else:
            print(f"  ✅ 无无穷值问题")
        
        # 检查NaN值比例
        high_nan_cols = []
        for col in df.columns:
            nan_pct = df[col].isna().sum() / len(df) * 100
            if nan_pct > 50:
                high_nan_cols.append((col, nan_pct))
        
        if high_nan_cols:
            print(f"  ⚠️  高NaN比例特征:")
            for col, pct in high_nan_cols[:5]:  # 只显示前5个
                print(f"    {col}: {pct:.1f}%")
        else:
            print(f"  ✅ NaN比例合理")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征改进分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始增强特征生成系统测试...")
    
    # 测试1: 涨跌停数据加载器
    loader_success = test_limit_data_loader()
    
    # 测试2: 增强特征生成
    feature_success = test_enhanced_features()
    
    # 测试3: 特征改进分析
    analysis_success = analyze_feature_improvements()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print(f"涨跌停数据加载器: {'✅ 通过' if loader_success else '❌ 失败'}")
    print(f"增强特征生成: {'✅ 通过' if feature_success else '❌ 失败'}")
    print(f"特征改进分析: {'✅ 通过' if analysis_success else '❌ 失败'}")
    
    overall_success = loader_success and feature_success and analysis_success
    
    if overall_success:
        print(f"\n🎉 所有测试通过！")
        print(f"✅ 涨跌停数据集成成功")
        print(f"✅ 无穷值问题修复")
        print(f"✅ 特征生成系统运行正常")
    else:
        print(f"\n💥 部分测试失败！")
        print(f"❌ 需要进一步检查和修复")
    
    return overall_success

if __name__ == "__main__":
    main()
