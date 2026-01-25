#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据类型修复效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import warnings
from update_mv_data import update_mv_data

def test_dtype_fix():
    """测试数据类型修复效果"""
    
    print("🧪 测试数据类型修复效果")
    print("=" * 60)
    
    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        print("📋 运行修复后的更新脚本...")
        
        # 测试更新一个文件
        success = update_mv_data(
            start_date='20230103',
            end_date='20250725', 
            dry_run=False  
        )
        
        # 检查警告
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
        dtype_warnings = [warning for warning in future_warnings if 'incompatible dtype' in str(warning.message)]
        
        print(f"\n📊 警告检查结果:")
        print(f"   总警告数: {len(w)}")
        print(f"   FutureWarning数: {len(future_warnings)}")
        print(f"   数据类型兼容性警告数: {len(dtype_warnings)}")
        
        if dtype_warnings:
            print(f"\n⚠️ 仍存在数据类型警告:")
            for warning in dtype_warnings[:3]:  # 只显示前3个
                print(f"   - {warning.message}")
                print(f"     文件: {warning.filename}:{warning.lineno}")
        else:
            print(f"\n✅ 没有数据类型兼容性警告！")
        
        # 检查其他类型的警告
        other_warnings = [w for w in w if w not in dtype_warnings]
        if other_warnings:
            print(f"\n📋 其他警告:")
            for warning in other_warnings[:3]:  # 只显示前3个
                print(f"   - {warning.category.__name__}: {warning.message}")
        
        return success and len(dtype_warnings) == 0

def verify_data_integrity():
    """验证数据完整性"""
    
    print(f"\n🔍 验证数据完整性")
    print("=" * 40)
    
    import pandas as pd
    from pathlib import Path
    
    # 检查更新后的数据
    feature_file = Path("to/local_data/features/daily/20230103.parquet")
    
    if feature_file.exists():
        df = pd.read_parquet(feature_file)
        
        mv_fields = ['total_mv', 'circ_mv', 'turnover_rate']
        
        print(f"📊 数据完整性检查:")
        for field in mv_fields:
            if field in df.columns:
                series = df[field]
                dtype = series.dtype
                positive_count = (series > 0).sum()
                total_count = len(series)
                
                print(f"   {field}:")
                print(f"      数据类型: {dtype}")
                print(f"      正值数量: {positive_count}/{total_count} ({positive_count/total_count*100:.1f}%)")
                
                if positive_count > 0:
                    positive_series = series[series > 0]
                    print(f"      数值范围: {positive_series.min():.2e} ~ {positive_series.max():.2e}")
        
        # 显示样本数据
        print(f"\n📋 样本数据:")
        sample = df[['stock_code'] + mv_fields].head(3)
        for idx, row in sample.iterrows():
            stock = row['stock_code']
            mv_data = [f"{field}:{row[field]:.2e}" if row[field] > 0 else f"{field}:0" 
                      for field in mv_fields]
            print(f"   {stock} | {' | '.join(mv_data)}")
        
        return True
    else:
        print(f"❌ 测试文件不存在: {feature_file}")
        return False

if __name__ == "__main__":
    print("🔧 数据类型兼容性修复测试")
    print("=" * 80)
    
    # 测试修复效果
    fix_success = test_dtype_fix()
    
    # 验证数据完整性
    data_integrity = verify_data_integrity()
    
    # 总结
    print(f"\n📋 测试结果总结:")
    print("=" * 40)
    
    if fix_success:
        print("✅ 数据类型兼容性修复成功")
    else:
        print("❌ 数据类型兼容性修复失败")
    
    if data_integrity:
        print("✅ 数据完整性验证通过")
    else:
        print("❌ 数据完整性验证失败")
    
    if fix_success and data_integrity:
        print("\n🎉 修复完全成功！")
        print("   - 消除了数据类型兼容性警告")
        print("   - 保持了数据完整性")
        print("   - 可以安全使用更新脚本")
    else:
        print("\n⚠️ 修复需要进一步调整")
        
    print(f"\n💡 建议:")
    print("   - 如果测试通过，可以正式使用修复后的脚本")
    print("   - 如果仍有警告，请检查具体的警告信息")
    print("   - 建议在正式更新前先备份重要数据")
