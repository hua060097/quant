#!/usr/bin/env python3
"""
分析涨跌停数据中first_time字段的问题
检查20230519前后的数据差异，确定是数据源问题还是代码逻辑问题
"""

import pandas as pd
import numpy as np
import tushare as ts
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_limit_data_file():
    """分析涨跌停数据文件中first_time字段的问题"""
    print("=" * 80)
    print("分析涨跌停数据文件中的first_time字段")
    print("=" * 80)
    
    # 数据文件路径
    limit_data_file = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data/all_limit_data.parquet")
    
    if not limit_data_file.exists():
        print(f"❌ 涨跌停数据文件不存在: {limit_data_file}")
        return None
    
    try:
        # 读取涨跌停数据
        print(f"📊 读取涨跌停数据文件: {limit_data_file}")
        df = pd.read_parquet(limit_data_file)
        print(f"✅ 数据加载成功，形状: {df.shape}")
        
        # 检查列名
        print(f"\n📋 数据列名: {list(df.columns)}")
        
        # 检查是否有first_time字段
        if 'first_time' not in df.columns:
            print("❌ 数据中没有first_time字段")
            return None
        
        # 转换日期格式
        if 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        elif 'date' in df.columns:
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date'])
        else:
            print("❌ 数据中没有日期字段")
            return None
        
        # 定义关键日期
        key_date = datetime(2023, 5, 19)
        before_start = key_date - timedelta(days=10)  # 5月9日
        after_end = key_date + timedelta(days=10)     # 5月29日
        
        print(f"\n🔍 分析时间范围:")
        print(f"  关键日期: {key_date.strftime('%Y-%m-%d')}")
        print(f"  分析范围: {before_start.strftime('%Y-%m-%d')} 到 {after_end.strftime('%Y-%m-%d')}")
        
        # 筛选数据
        analysis_data = df[(df['date'] >= before_start) & (df['date'] <= after_end)].copy()
        print(f"  分析数据量: {len(analysis_data)} 条")
        
        if analysis_data.empty:
            print("❌ 指定时间范围内没有数据")
            return None
        
        # 按日期分组分析first_time字段
        print(f"\n📊 按日期分析first_time字段类型:")
        
        date_analysis = []
        for date, group in analysis_data.groupby(analysis_data['date'].dt.date):
            first_time_values = group['first_time'].dropna()
            
            if len(first_time_values) == 0:
                continue
            
            # 分析数据类型
            types_count = {}
            sample_values = []
            
            for val in first_time_values.head(10):  # 取前10个样本
                val_type = type(val).__name__
                types_count[val_type] = types_count.get(val_type, 0) + 1
                sample_values.append(f"{val} ({val_type})")
            
            date_analysis.append({
                'date': date,
                'total_records': len(group),
                'first_time_records': len(first_time_values),
                'types': types_count,
                'samples': sample_values[:5]  # 只保留前5个样本
            })
        
        # 显示分析结果
        print(f"\n📈 详细分析结果:")
        for item in sorted(date_analysis, key=lambda x: x['date']):
            date_str = item['date'].strftime('%Y-%m-%d')
            is_key_date = item['date'] == key_date.date()
            marker = " ⭐ [关键日期]" if is_key_date else ""
            
            print(f"\n  {date_str}{marker}")
            print(f"    总记录数: {item['total_records']}")
            print(f"    first_time记录数: {item['first_time_records']}")
            print(f"    数据类型分布: {item['types']}")
            print(f"    样本值: {item['samples']}")
        
        # 检查关键日期前后的类型变化
        print(f"\n🔍 关键发现:")
        
        before_dates = [item for item in date_analysis if item['date'] < key_date.date()]
        after_dates = [item for item in date_analysis if item['date'] > key_date.date()]
        
        if before_dates and after_dates:
            # 分析类型变化
            before_types = set()
            after_types = set()
            
            for item in before_dates:
                before_types.update(item['types'].keys())
            
            for item in after_dates:
                after_types.update(item['types'].keys())
            
            print(f"  关键日期前的数据类型: {before_types}")
            print(f"  关键日期后的数据类型: {after_types}")
            
            if before_types != after_types:
                print(f"  ⚠️  发现类型变化！")
                print(f"    新增类型: {after_types - before_types}")
                print(f"    消失类型: {before_types - after_types}")
                return True  # 发现问题
            else:
                print(f"  ✅ 数据类型保持一致")
                return False  # 没有发现问题
        
        return analysis_data
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_tushare_data_for_comparison():
    """从tushare获取原始数据进行对比"""
    print("\n" + "=" * 80)
    print("从tushare获取原始涨跌停数据进行对比")
    print("=" * 80)
    
    try:
        # 初始化tushare
        # 注意：需要设置tushare token
        # ts.set_token('your_token_here')
        pro = ts.pro_api()
        
        # 定义测试日期
        test_dates = [
            '20230515',  # 关键日期前
            '20230516',
            '20230517',
            '20230518',
            '20230519',  # 关键日期
            '20230522',  # 关键日期后
            '20230523',
            '20230524',
            '20230525',
            '20230526'
        ]
        
        print(f"📊 获取测试日期的tushare原始数据:")
        
        all_tushare_data = []
        
        for date in test_dates:
            try:
                print(f"\n  获取 {date} 的数据...")
                
                # 获取涨停数据
                limit_up_data = pro.limit_list_d(
                    trade_date=date,
                    limit_type='U'  # 涨停
                )
                
                if not limit_up_data.empty:
                    limit_up_data['limit_type'] = 'U'
                    limit_up_data['date'] = date
                    all_tushare_data.append(limit_up_data)
                    
                    # 分析first_time字段
                    if 'first_time' in limit_up_data.columns:
                        first_time_sample = limit_up_data['first_time'].dropna().head(5)
                        print(f"    涨停数据 first_time 样本:")
                        for i, val in enumerate(first_time_sample):
                            print(f"      [{i}]: {val} (type: {type(val).__name__})")
                    else:
                        print(f"    ❌ 涨停数据中没有first_time字段")
                
                # 获取跌停数据
                limit_down_data = pro.limit_list_d(
                    trade_date=date,
                    limit_type='D'  # 跌停
                )
                
                if not limit_down_data.empty:
                    limit_down_data['limit_type'] = 'D'
                    limit_down_data['date'] = date
                    all_tushare_data.append(limit_down_data)
                
                print(f"    ✅ {date}: 涨停 {len(limit_up_data)} 条, 跌停 {len(limit_down_data)} 条")
                
            except Exception as e:
                print(f"    ❌ 获取 {date} 数据失败: {e}")
                continue
        
        if all_tushare_data:
            # 合并所有数据
            combined_data = pd.concat(all_tushare_data, ignore_index=True)
            print(f"\n✅ tushare数据获取完成，总计 {len(combined_data)} 条记录")
            
            # 分析tushare数据中的first_time字段
            print(f"\n📊 tushare数据中first_time字段分析:")
            
            if 'first_time' in combined_data.columns:
                first_time_data = combined_data['first_time'].dropna()
                
                # 统计数据类型
                type_counts = {}
                for val in first_time_data:
                    val_type = type(val).__name__
                    type_counts[val_type] = type_counts.get(val_type, 0) + 1
                
                print(f"  数据类型分布: {type_counts}")
                
                # 显示样本
                print(f"  样本值:")
                for i, val in enumerate(first_time_data.head(10)):
                    print(f"    [{i}]: {val} (type: {type(val).__name__})")
                
                return combined_data
            else:
                print(f"  ❌ tushare数据中没有first_time字段")
                return None
        else:
            print(f"❌ 没有获取到任何tushare数据")
            return None
            
    except Exception as e:
        print(f"❌ tushare数据获取失败: {e}")
        print("请确保已正确设置tushare token")
        import traceback
        traceback.print_exc()
        return None

def check_enhanced_collector_logic():
    """检查enhanced_limit_data_collector.py的逻辑"""
    print("\n" + "=" * 80)
    print("检查enhanced_limit_data_collector.py的处理逻辑")
    print("=" * 80)
    
    try:
        # 检查enhanced_limit_data_collector.py文件
        collector_file = Path("to/main_sector_stock/enhanced_limit_data_collector.py")
        
        if not collector_file.exists():
            print(f"❌ 文件不存在: {collector_file}")
            return None
        
        print(f"📄 分析文件: {collector_file}")
        
        # 读取文件内容
        with open(collector_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找first_time相关的处理逻辑
        print(f"\n🔍 查找first_time相关的处理逻辑:")
        
        lines = content.split('\n')
        first_time_lines = []
        
        for i, line in enumerate(lines):
            if 'first_time' in line.lower():
                # 获取上下文
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = lines[start:end]
                
                first_time_lines.append({
                    'line_num': i + 1,
                    'line': line.strip(),
                    'context': context
                })
        
        if first_time_lines:
            print(f"  找到 {len(first_time_lines)} 处first_time相关代码:")
            
            for item in first_time_lines:
                print(f"\n    行 {item['line_num']}: {item['line']}")
                print(f"    上下文:")
                for j, ctx_line in enumerate(item['context']):
                    marker = "  >>> " if j == 2 else "      "
                    print(f"{marker}{ctx_line}")
        else:
            print(f"  ❌ 没有找到first_time相关的处理逻辑")
        
        # 查找数据类型转换相关的逻辑
        print(f"\n🔍 查找数据类型转换相关的逻辑:")
        
        conversion_keywords = ['astype', 'to_numeric', 'float', 'int', 'str']
        conversion_lines = []
        
        for i, line in enumerate(lines):
            for keyword in conversion_keywords:
                if keyword in line and ('first_time' in line or 'last_time' in line):
                    conversion_lines.append({
                        'line_num': i + 1,
                        'line': line.strip(),
                        'keyword': keyword
                    })
        
        if conversion_lines:
            print(f"  找到 {len(conversion_lines)} 处类型转换相关代码:")
            for item in conversion_lines:
                print(f"    行 {item['line_num']}: {item['line']} (关键词: {item['keyword']})")
        else:
            print(f"  ✅ 没有找到可疑的类型转换逻辑")
        
        return {
            'first_time_lines': first_time_lines,
            'conversion_lines': conversion_lines,
            'total_lines': len(lines)
        }
        
    except Exception as e:
        print(f"❌ 检查代码逻辑时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("🚀 开始分析涨跌停数据中first_time字段的问题...")
    
    # 1. 分析现有数据文件
    print("\n" + "🔍 步骤1: 分析现有涨跌停数据文件")
    file_analysis_result = analyze_limit_data_file()
    
    # 2. 检查代码逻辑
    print("\n" + "🔍 步骤2: 检查enhanced_limit_data_collector.py逻辑")
    code_analysis_result = check_enhanced_collector_logic()
    
    # 3. 获取tushare原始数据对比（可选）
    print("\n" + "🔍 步骤3: 获取tushare原始数据对比")
    print("注意: 需要配置tushare token才能执行此步骤")
    
    try:
        tushare_result = fetch_tushare_data_for_comparison()
    except Exception as e:
        print(f"⚠️  tushare数据获取跳过: {e}")
        tushare_result = None
    
    # 4. 总结分析结果
    print("\n" + "=" * 80)
    print("📋 分析总结")
    print("=" * 80)
    
    if file_analysis_result is True:
        print("🔴 发现问题: 涨跌停数据文件中first_time字段在20230519前后存在类型变化")
        print("   建议: 需要进一步调查数据源或代码逻辑")
    elif file_analysis_result is False:
        print("🟢 数据文件检查: first_time字段类型在20230519前后保持一致")
        print("   可能原因: 问题出现在数据处理的其他环节")
    elif file_analysis_result is None:
        print("🟡 数据文件检查: 无法完成分析（文件不存在或格式问题）")
    
    if code_analysis_result:
        if code_analysis_result['conversion_lines']:
            print("🔴 代码逻辑检查: 发现可疑的类型转换逻辑")
            print("   建议: 检查enhanced_limit_data_collector.py中的数据处理逻辑")
        else:
            print("🟢 代码逻辑检查: 没有发现明显的类型转换问题")
    
    if tushare_result is not None:
        print("🟢 tushare数据获取: 成功获取原始数据用于对比")
    else:
        print("🟡 tushare数据获取: 跳过或失败")
    
    # 5. 给出建议
    print("\n📝 建议措施:")
    print("1. 如果发现数据类型变化，检查enhanced_limit_data_collector.py的处理逻辑")
    print("2. 如果tushare原始数据正常，问题可能在数据收集或处理环节")
    print("3. 考虑重新生成20230519之后的涨跌停数据")
    print("4. 验证generate_daily_features.py中的字符串转换逻辑是否足够健壮")

if __name__ == "__main__":
    main()
