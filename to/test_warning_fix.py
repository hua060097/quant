"""
测试 SettingWithCopyWarning 修复
"""
import warnings
import pandas as pd
from enhanced_limit_data_collector import CompleteLimitDataCollector

def test_warning_fix():
    """测试警告修复是否有效"""
    print("🧪 测试 SettingWithCopyWarning 修复...")
    print("="*60)
    
    # 捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        collector = CompleteLimitDataCollector()
        
        # 获取少量数据进行测试
        print("\n📥 获取测试数据...")
        limit_up_data, limit_down_data, limit_zhaban_data = collector.get_complete_limit_data_by_date_range(
            start_date='20250601',
            end_date='20250605'
        )
        
        print(f"涨停数据: {len(limit_up_data)}条")
        print(f"跌停数据: {len(limit_down_data)}条")
        print(f"炸板数据: {len(limit_zhaban_data)}条")
        
        # 处理和保存数据
        print("\n💾 处理和保存数据（增量更新）...")
        all_data = collector.process_and_save_complete_data(
            limit_up_data, limit_down_data, limit_zhaban_data
        )
        
        print(f"\n✅ 数据保存完成: {len(all_data)}条")
        
        # 检查是否有 SettingWithCopyWarning
        setting_warnings = [warning for warning in w if 'SettingWithCopyWarning' in str(warning.category)]
        
        print("\n" + "="*60)
        print("🔍 警告检查结果:")
        print("="*60)
        
        if setting_warnings:
            print(f"❌ 发现 {len(setting_warnings)} 个 SettingWithCopyWarning:")
            for warning in setting_warnings:
                print(f"\n警告信息:")
                print(f"  文件: {warning.filename}")
                print(f"  行号: {warning.lineno}")
                print(f"  消息: {warning.message}")
        else:
            print("✅ 未发现 SettingWithCopyWarning")
            print("✅ 警告修复成功！")
        
        # 检查其他警告
        other_warnings = [warning for warning in w if 'SettingWithCopyWarning' not in str(warning.category)]
        if other_warnings:
            print(f"\n⚠️  发现 {len(other_warnings)} 个其他警告:")
            for warning in other_warnings[:3]:  # 只显示前3个
                print(f"  - {warning.category.__name__}: {warning.message}")
        
        # 验证数据完整性
        print("\n" + "="*60)
        print("🔍 数据完整性验证:")
        print("="*60)
        
        # 检查日期格式
        if 'trade_date' in all_data.columns:
            print(f"✅ trade_date 字段存在")
            print(f"   数据类型: {all_data['trade_date'].dtype}")
            print(f"   日期范围: {all_data['trade_date'].min()} 到 {all_data['trade_date'].max()}")
            print(f"   唯一日期数: {all_data['trade_date'].nunique()}个")
        
        # 检查数据是否按日期排序
        if len(all_data) > 1:
            is_sorted = all_data['trade_date'].is_monotonic_increasing
            if is_sorted:
                print(f"✅ 数据已按日期排序")
            else:
                print(f"⚠️  数据未按日期排序")
        
        # 检查是否有重复记录
        duplicates = all_data.duplicated(subset=['trade_date', 'ts_code'], keep=False)
        if duplicates.any():
            print(f"⚠️  发现 {duplicates.sum()} 条重复记录")
        else:
            print(f"✅ 无重复记录")
        
        print("\n🎉 测试完成！")
        
        return len(setting_warnings) == 0

if __name__ == "__main__":
    success = test_warning_fix()
    if success:
        print("\n" + "="*60)
        print("✅ 所有测试通过！SettingWithCopyWarning 已修复！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 测试失败！仍存在 SettingWithCopyWarning")
        print("="*60)

