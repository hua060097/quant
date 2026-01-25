"""
测试增量更新功能
"""
import pandas as pd
from pathlib import Path
from enhanced_limit_data_collector import CompleteLimitDataCollector

def test_incremental_update():
    """测试增量更新功能"""
    print("🧪 开始测试增量更新功能...")
    
    collector = CompleteLimitDataCollector()
    
    # 测试场景1：首次保存数据（文件不存在）
    print("\n" + "="*60)
    print("📝 测试场景1: 首次保存数据")
    print("="*60)
    
    # 获取少量数据进行测试
    limit_up_data, limit_down_data, limit_zhaban_data = collector.get_complete_limit_data_by_date_range(
        start_date='20250101',
        end_date='20250110'
    )
    
    print(f"获取到涨停数据: {len(limit_up_data)}条")
    print(f"获取到跌停数据: {len(limit_down_data)}条")
    print(f"获取到炸板数据: {len(limit_zhaban_data)}条")
    
    # 首次保存
    all_data_1 = collector.process_and_save_complete_data(
        limit_up_data, limit_down_data, limit_zhaban_data
    )
    
    print(f"\n✅ 首次保存完成，总数据量: {len(all_data_1)}条")
    
    # 测试场景2：增量添加新数据（日期不重复）
    print("\n" + "="*60)
    print("📝 测试场景2: 增量添加新数据（日期不重复）")
    print("="*60)
    
    limit_up_data_2, limit_down_data_2, limit_zhaban_data_2 = collector.get_complete_limit_data_by_date_range(
        start_date='20250111',
        end_date='20250120'
    )
    
    print(f"获取到新涨停数据: {len(limit_up_data_2)}条")
    print(f"获取到新跌停数据: {len(limit_down_data_2)}条")
    print(f"获取到新炸板数据: {len(limit_zhaban_data_2)}条")
    
    # 增量保存
    all_data_2 = collector.process_and_save_complete_data(
        limit_up_data_2, limit_down_data_2, limit_zhaban_data_2
    )
    
    print(f"\n✅ 增量保存完成，总数据量: {len(all_data_2)}条")
    print(f"📊 数据增长: {len(all_data_2) - len(all_data_1)}条")
    
    # 测试场景3：覆盖已有数据（日期重复）
    print("\n" + "="*60)
    print("📝 测试场景3: 覆盖已有数据（日期重复）")
    print("="*60)
    
    # 重新获取部分已有日期的数据
    limit_up_data_3, limit_down_data_3, limit_zhaban_data_3 = collector.get_complete_limit_data_by_date_range(
        start_date='20250105',
        end_date='20250115'
    )
    
    print(f"获取到涨停数据（含重复日期）: {len(limit_up_data_3)}条")
    print(f"获取到跌停数据（含重复日期）: {len(limit_down_data_3)}条")
    print(f"获取到炸板数据（含重复日期）: {len(limit_zhaban_data_3)}条")
    
    # 覆盖保存
    all_data_3 = collector.process_and_save_complete_data(
        limit_up_data_3, limit_down_data_3, limit_zhaban_data_3
    )
    
    print(f"\n✅ 覆盖保存完成，总数据量: {len(all_data_3)}条")
    
    # 验证数据完整性
    print("\n" + "="*60)
    print("🔍 验证数据完整性")
    print("="*60)
    
    # 读取最终数据
    all_file = collector.data_dir / "all_limit_data.parquet"
    final_data = pd.read_parquet(all_file)
    
    print(f"最终数据总量: {len(final_data)}条")
    print(f"日期范围: {final_data['trade_date'].min()} 到 {final_data['trade_date'].max()}")
    print(f"唯一日期数: {final_data['trade_date'].nunique()}个")
    
    # 检查是否有重复日期+股票代码的记录
    duplicates = final_data.duplicated(subset=['trade_date', 'ts_code'], keep=False)
    if duplicates.any():
        print(f"⚠️  发现重复记录: {duplicates.sum()}条")
        print(final_data[duplicates][['trade_date', 'ts_code', 'name', 'limit_type']].head(10))
    else:
        print("✅ 无重复记录")
    
    # 按类型统计
    print("\n📊 按类型统计:")
    type_counts = final_data['limit_type'].value_counts()
    for limit_type, count in type_counts.items():
        print(f"  {limit_type}: {count}条")
    
    print("\n🎉 增量更新功能测试完成！")

if __name__ == "__main__":
    test_incremental_update()

