#!/usr/bin/env python3
"""
详细分析时间字段问题
重点检查enhanced_limit_data_collector.py的时间处理逻辑和实际数据的差异
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_time_processing_logic():
    """分析时间处理逻辑的问题"""
    print("=" * 80)
    print("分析enhanced_limit_data_collector.py的时间处理逻辑")
    print("=" * 80)
    
    # 模拟 _format_time 函数的逻辑
    def _format_time(time_str):
        """复制enhanced_limit_data_collector.py中的_format_time逻辑"""
        if pd.isna(time_str) or time_str is None:
            return None
        
        try:
            time_str = str(int(time_str))  # 这里是问题所在！
            if len(time_str) == 6:
                return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            elif len(time_str) == 5:
                return f"0{time_str[0]}:{time_str[1:3]}:{time_str[3:5]}"
            else:
                return time_str
        except:
            return None
    
    # 测试不同类型的输入
    test_cases = [
        ('93021', 'str'),           # 正常字符串
        (93021, 'int'),             # 整数
        (93021.0, 'float'),         # 浮点数
        ('93021.0', 'str_float'),   # 字符串形式的浮点数
        (np.nan, 'nan'),            # NaN
        (None, 'none'),             # None
        ('', 'empty_str'),          # 空字符串
        ('abc', 'invalid_str'),     # 无效字符串
    ]
    
    print("🔍 测试_format_time函数对不同输入的处理:")
    
    for test_input, input_type in test_cases:
        try:
            result = _format_time(test_input)
            print(f"  输入: {test_input} ({input_type}) -> 输出: {result}")
        except Exception as e:
            print(f"  输入: {test_input} ({input_type}) -> 错误: {e}")
    
    print(f"\n🔍 关键发现:")
    print(f"  1. _format_time函数使用 str(int(time_str)) 进行转换")
    print(f"  2. 这会将浮点数转换为整数，可能丢失精度")
    print(f"  3. 对于字符串形式的浮点数（如'93021.0'），会先转为int再转为str")
    print(f"  4. 这个过程可能在某些情况下产生意外的结果")

def check_actual_data_types():
    """检查实际数据中的类型分布"""
    print("\n" + "=" * 80)
    print("检查实际涨跌停数据中的first_time字段类型分布")
    print("=" * 80)
    
    # 数据文件路径
    limit_data_file = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data/all_limit_data.parquet")
    
    if not limit_data_file.exists():
        print(f"❌ 数据文件不存在: {limit_data_file}")
        return None
    
    try:
        # 读取数据
        df = pd.read_parquet(limit_data_file)
        print(f"📊 数据加载成功: {df.shape}")
        
        # 检查first_time字段的详细类型分布
        if 'first_time' in df.columns:
            print(f"\n🔍 first_time字段详细分析:")
            
            # 统计不同类型的数量
            type_stats = {}
            sample_values = {}
            
            for i, val in enumerate(df['first_time']):
                val_type = type(val).__name__
                type_stats[val_type] = type_stats.get(val_type, 0) + 1
                
                # 收集样本值
                if val_type not in sample_values:
                    sample_values[val_type] = []
                if len(sample_values[val_type]) < 5:
                    sample_values[val_type].append(val)
            
            print(f"  类型分布:")
            for val_type, count in sorted(type_stats.items()):
                percentage = count / len(df) * 100
                print(f"    {val_type}: {count} 条 ({percentage:.2f}%)")
                print(f"      样本值: {sample_values[val_type]}")
            
            # 检查是否有浮点数类型
            float_values = df[df['first_time'].apply(lambda x: isinstance(x, float) and not pd.isna(x))]
            if not float_values.empty:
                print(f"\n⚠️  发现 {len(float_values)} 个浮点数类型的first_time值:")
                print(f"  样本: {float_values['first_time'].head().tolist()}")
                
                # 检查这些浮点数的日期分布
                if 'trade_date' in float_values.columns:
                    date_dist = float_values['trade_date'].value_counts().head(10)
                    print(f"  日期分布（前10）:")
                    for date, count in date_dist.items():
                        print(f"    {date}: {count} 条")
            
            return df
        else:
            print(f"❌ 数据中没有first_time字段")
            return None
            
    except Exception as e:
        print(f"❌ 数据分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def simulate_processing_pipeline():
    """模拟数据处理管道，找出问题环节"""
    print("\n" + "=" * 80)
    print("模拟数据处理管道")
    print("=" * 80)
    
    # 模拟tushare返回的原始数据
    print("🔍 步骤1: 模拟tushare原始数据")
    raw_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'first_time': ['93021', '94530'],  # tushare返回的是字符串
        'last_time': ['145030', '150000']
    })
    
    print(f"  原始数据类型:")
    for col in ['first_time', 'last_time']:
        print(f"    {col}: {raw_data[col].dtype}")
        for i, val in enumerate(raw_data[col]):
            print(f"      [{i}]: {val} (type: {type(val).__name__})")
    
    # 模拟enhanced_limit_data_collector的处理
    print(f"\n🔍 步骤2: 模拟enhanced_limit_data_collector处理")
    
    def _format_time(time_str):
        if pd.isna(time_str) or time_str is None:
            return None
        try:
            time_str = str(int(time_str))
            if len(time_str) == 6:
                return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            elif len(time_str) == 5:
                return f"0{time_str[0]}:{time_str[1:3]}:{time_str[3:5]}"
            else:
                return time_str
        except:
            return None
    
    processed_data = raw_data.copy()
    processed_data['first_time_formatted'] = processed_data['first_time'].apply(_format_time)
    processed_data['last_time_formatted'] = processed_data['last_time'].apply(_format_time)
    
    print(f"  处理后数据:")
    for col in ['first_time', 'last_time', 'first_time_formatted', 'last_time_formatted']:
        if col in processed_data.columns:
            print(f"    {col}: {processed_data[col].dtype}")
            for i, val in enumerate(processed_data[col]):
                print(f"      [{i}]: {val} (type: {type(val).__name__})")
    
    # 模拟保存和重新加载过程
    print(f"\n🔍 步骤3: 模拟保存和重新加载")
    
    # 保存为parquet
    test_file = Path("test_time_processing.parquet")
    try:
        processed_data.to_parquet(test_file, index=False)
        print(f"  ✅ 保存成功")
        
        # 重新加载
        reloaded_data = pd.read_parquet(test_file)
        print(f"  ✅ 重新加载成功")
        
        print(f"  重新加载后数据类型:")
        for col in ['first_time', 'last_time', 'first_time_formatted', 'last_time_formatted']:
            if col in reloaded_data.columns:
                print(f"    {col}: {reloaded_data[col].dtype}")
                for i, val in enumerate(reloaded_data[col]):
                    print(f"      [{i}]: {val} (type: {type(val).__name__})")
        
        # 清理测试文件
        test_file.unlink()
        
        return reloaded_data
        
    except Exception as e:
        print(f"  ❌ 保存/加载失败: {e}")
        return None

def identify_root_cause():
    """识别根本原因"""
    print("\n" + "=" * 80)
    print("根本原因分析")
    print("=" * 80)
    
    print("🔍 可能的问题来源:")
    print("1. tushare API返回数据类型变化")
    print("   - 早期返回字符串，后期返回数值类型")
    print("   - 或者某些字段在某些情况下返回NaN/None")
    
    print("\n2. enhanced_limit_data_collector.py处理逻辑")
    print("   - _format_time函数使用str(int(time_str))可能有问题")
    print("   - 对于浮点数输入，int()转换可能丢失信息")
    
    print("\n3. pandas数据类型推断")
    print("   - 保存/加载parquet时的类型推断")
    print("   - 混合类型列的处理")
    
    print("\n4. generate_daily_features.py中的类型转换")
    print("   - 将数值类型强制转换为float")
    print("   - 后续又要求转换为字符串")

def main():
    """主函数"""
    print("🚀 开始详细时间字段问题分析...")
    
    # 1. 分析时间处理逻辑
    analyze_time_processing_logic()
    
    # 2. 检查实际数据类型
    actual_data = check_actual_data_types()
    
    # 3. 模拟处理管道
    pipeline_result = simulate_processing_pipeline()
    
    # 4. 识别根本原因
    identify_root_cause()
    
    # 5. 总结和建议
    print("\n" + "=" * 80)
    print("总结和建议")
    print("=" * 80)
    
    print("📋 分析结果:")
    if actual_data is not None:
        print("✅ 实际数据分析完成")
    else:
        print("❌ 实际数据分析失败")
    
    if pipeline_result is not None:
        print("✅ 处理管道模拟完成")
    else:
        print("❌ 处理管道模拟失败")
    
    print("\n📝 建议措施:")
    print("1. 检查enhanced_limit_data_collector.py中的_format_time函数")
    print("2. 确保时间字段在整个处理链中保持一致的数据类型")
    print("3. 在generate_daily_features.py中增强字符串转换逻辑")
    print("4. 考虑重新生成涨跌停数据以确保数据一致性")

if __name__ == "__main__":
    main()
