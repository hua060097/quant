#!/usr/bin/env python3
"""
同花顺概念板块数据下载工具使用示例
"""

from ths_data_downloader import THSSectorDataDownloader
import pandas as pd
import json
from datetime import datetime, timedelta

def example_basic_usage():
    """基本使用示例"""
    print("🚀 同花顺概念板块数据下载工具使用示例")
    print("="*60)
    
    # 1. 初始化下载器
    print("\n📋 步骤1: 初始化下载器")
    downloader = THSSectorDataDownloader()
    print("✅ 下载器初始化完成")
    
    # 2. 下载概念板块基础信息
    print("\n📋 步骤2: 下载概念板块基础信息")
    try:
        index_df = downloader.download_ths_index(type_filter='N', exchange='A')
        print(f"✅ 成功下载 {len(index_df)} 个概念板块")
        
        # 显示前几个概念板块
        print("\n🔍 前10个概念板块:")
        for i, row in index_df.head(10).iterrows():
            print(f"   {i+1:2d}. {row['ts_code']} - {row['name']}")
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return
    
    # 3. 下载部分概念板块的成分股（为了演示，只下载前5个）
    print(f"\n📊 步骤3: 下载前5个概念板块的成分股")
    try:
        sample_codes = index_df['ts_code'].head(5).tolist()
        print(f"   🎯 选择的概念板块: {sample_codes}")
        
        members_df = downloader.download_ths_members(concept_codes=sample_codes)
        print(f"✅ 成功下载 {len(members_df)} 条成分股记录")
        
        # 显示各概念板块的成分股数量
        concept_counts = members_df.groupby('ts_code').size()
        print("\n🔍 各概念板块成分股数量:")
        for ts_code, count in concept_counts.items():
            concept_name = index_df[index_df['ts_code'] == ts_code]['name'].iloc[0]
            print(f"   {ts_code} ({concept_name}): {count} 只股票")
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return
    
    # 4. 下载最近30天的行情数据
    print(f"\n📈 步骤4: 下载最近30天的行情数据")
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        print(f"   📅 日期范围: {start_date} - {end_date}")
        
        daily_df = downloader.download_ths_daily(
            start_date=start_date,
            end_date=end_date,
            concept_codes=sample_codes
        )
        print(f"✅ 成功下载 {len(daily_df)} 条行情记录")
        
        # 显示数据统计
        print(f"\n🔍 行情数据统计:")
        print(f"   📊 涉及概念: {daily_df['ts_code'].nunique()}")
        print(f"   📅 日期范围: {daily_df['trade_date'].min()} - {daily_df['trade_date'].max()}")
        print(f"   📈 平均涨跌幅: {daily_df['pct_change'].mean():.2f}%")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return
    
    print("\n🎉 示例完成！")

def example_data_analysis():
    """数据分析示例"""
    print("\n📊 数据分析示例")
    print("="*60)
    
    downloader = THSSectorDataDownloader()
    
    # 读取已下载的数据
    try:
        # 读取概念板块基础信息
        concept_file = downloader.base_path / "ths_concept_index_N_A.json"
        if concept_file.exists():
            with open(concept_file, 'r', encoding='utf-8') as f:
                concept_data = json.load(f)
            print(f"📋 概念板块总数: {concept_data['count']}")
        
        # 读取成分股信息
        members_file = downloader.base_path / "ths_concept_members.csv"
        if members_file.exists():
            members_df = pd.read_csv(members_file)
            print(f"📊 成分股记录数: {len(members_df)}")
            print(f"   🏷️  涉及概念: {members_df['ts_code'].nunique()}")
            print(f"   📈 涉及股票: {members_df['con_code'].nunique()}")
            
            # 分析概念板块规模分布
            concept_sizes = members_df.groupby('ts_code').size()
            print(f"\n🔍 概念板块规模分析:")
            print(f"   📊 平均成分股数: {concept_sizes.mean():.1f}")
            print(f"   📊 最大成分股数: {concept_sizes.max()}")
            print(f"   📊 最小成分股数: {concept_sizes.min()}")
            
            # 显示规模最大的5个概念板块
            top_concepts = concept_sizes.nlargest(5)
            print(f"\n🏆 规模最大的5个概念板块:")
            for i, (ts_code, size) in enumerate(top_concepts.items(), 1):
                concept_name = next((item['name'] for item in concept_data['data'] 
                                   if item['ts_code'] == ts_code), 'Unknown')
                print(f"   {i}. {concept_name} ({ts_code}): {size} 只股票")
        
        # 读取行情数据
        daily_files = list(downloader.base_path.glob("ths_concept_daily_*.parquet"))
        if daily_files:
            latest_file = max(daily_files, key=lambda x: x.stat().st_mtime)
            daily_df = pd.read_parquet(latest_file)
            
            print(f"\n📈 行情数据分析:")
            print(f"   📊 记录总数: {len(daily_df)}")
            print(f"   📅 日期范围: {daily_df['trade_date'].min()} - {daily_df['trade_date'].max()}")
            
            # 计算各概念板块的平均表现
            concept_performance = daily_df.groupby('ts_code')['pct_change'].agg(['mean', 'std', 'count'])
            concept_performance = concept_performance.sort_values('mean', ascending=False)
            
            print(f"\n🏆 表现最好的5个概念板块 (按平均涨跌幅):")
            for i, (ts_code, row) in enumerate(concept_performance.head(5).iterrows(), 1):
                concept_name = next((item['name'] for item in concept_data['data'] 
                                   if item['ts_code'] == ts_code), 'Unknown')
                print(f"   {i}. {concept_name} ({ts_code}): {row['mean']:.2f}% (±{row['std']:.2f}%)")
        
    except Exception as e:
        print(f"❌ 数据分析失败: {e}")

def example_incremental_update():
    """增量更新示例"""
    print("\n🔄 增量更新示例")
    print("="*60)
    
    downloader = THSSectorDataDownloader()
    
    try:
        # 执行增量更新
        print("🔄 执行增量更新 (最近7天)...")
        update_df = downloader.update_daily_data(days_back=7)
        
        if not update_df.empty:
            print(f"✅ 增量更新成功")
            print(f"   📊 更新记录数: {len(update_df)}")
            print(f"   📅 日期范围: {update_df['trade_date'].min()} - {update_df['trade_date'].max()}")
            print(f"   🏷️  概念数量: {update_df['ts_code'].nunique()}")
        else:
            print("✅ 数据已是最新，无需更新")
            
    except Exception as e:
        print(f"❌ 增量更新失败: {e}")

def show_usage_summary():
    """显示使用总结"""
    print("\n📚 使用总结")
    print("="*60)
    
    print("🎯 主要功能:")
    print("   1. 下载同花顺概念板块基础信息")
    print("   2. 下载概念板块成分股信息")
    print("   3. 下载概念板块历史行情数据")
    print("   4. 支持增量更新")
    print("   5. 提供数据摘要和分析")
    
    print("\n💡 使用建议:")
    print("   1. 首次使用建议先下载概念板块基础信息")
    print("   2. 根据需要选择性下载成分股和行情数据")
    print("   3. 定期使用增量更新功能保持数据最新")
    print("   4. 注意API调用频率限制，避免过于频繁的请求")
    
    print("\n📁 数据文件:")
    print("   • ths_concept_index_N_A.json - 概念板块基础信息")
    print("   • ths_concept_members.csv/json - 成分股信息")
    print("   • ths_concept_daily_*.parquet - 历史行情数据")
    
    print("\n🔧 命令行使用:")
    print("   python main.py --action download_all")
    print("   python main.py --action update")
    print("   python main.py --action summary")

if __name__ == "__main__":
    # 运行基本使用示例
    example_basic_usage()
    
    # 运行数据分析示例
    example_data_analysis()
    
    # 运行增量更新示例
    # example_incremental_update()
    
    # 显示使用总结
    show_usage_summary()
