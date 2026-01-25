#!/usr/bin/env python3
"""
同花顺概念板块数据下载主程序
"""

import sys
import argparse
from pathlib import Path
import json
from ths_data_downloader import THSSectorDataDownloader

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='同花顺概念板块数据下载工具')
    parser.add_argument('--action', choices=['download_all', 'download_index', 'download_members', 
                                           'download_daily', 'update', 'update_sector', 'summary', 'down_hot_sector'], 
                       default='download_all', help='执行的操作')
    parser.add_argument('--start_date', default='20230101', help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end_date', help='结束日期 (YYYYMMDD)，默认为今天')
    parser.add_argument('--days_back', type=int, default=30, help='增量更新向前天数')
    
    args = parser.parse_args()
    
    # 初始化下载器
    downloader = THSSectorDataDownloader()
    
    try:
        if args.action == 'download_all':
            print("🚀 开始下载所有同花顺概念板块数据...")
            
            # 1. 下载概念板块基础信息
            print("\n📋 步骤1: 下载概念板块基础信息")
            index_df = downloader.download_ths_index(type_filter='N', exchange='A')
            print(f"✅ 成功下载 {len(index_df)} 个概念板块")
            
            # 2. 下载成分股信息
            print("\n📊 步骤2: 下载概念板块成分股信息")
            members_df = downloader.download_ths_members()
            print(f"✅ 成功下载 {len(members_df)} 条成分股记录")
            
            # 3. 下载历史行情数据
            print(f"\n📈 步骤3: 下载概念板块历史行情数据 ({args.start_date} - {args.end_date or '今天'})")
            daily_df = downloader.download_ths_daily(
                start_date=args.start_date, 
                end_date=args.end_date
            )
            print(f"✅ 成功下载 {len(daily_df)} 条行情记录")
            
            print("\n🎉 所有数据下载完成！")
        elif args.action == 'down_hot_sector':
            print("🔥 下载每日最强板块数据...")
            if args.start_date and args.end_date:
                df = downloader.download_hot_sectors(
                    start_date=args.start_date,
                    end_date=args.end_date
                )
                print(f"✅ 成功下载 {len(df)} 条最强板块记录")
            else:
                print("❌ 请提供开始日期和结束日期参数")
                print("示例: python main.py down_hot_sector --start_date 20230601 --end_date 20250724")
        elif args.action == 'download_index':
            print("📋 下载概念板块基础信息...")
            df = downloader.download_ths_index(type_filter='N', exchange='A')
            print(f"✅ 成功下载 {len(df)} 个概念板块")
            
        elif args.action == 'download_members':
            print("📊 下载概念板块成分股信息...")
            df = downloader.download_ths_members()
            print(f"✅ 成功下载 {len(df)} 条成分股记录")
            
        elif args.action == 'download_daily':
            print(f"📈 下载概念板块历史行情数据 ({args.start_date} - {args.end_date or '今天'})...")
            df = downloader.download_ths_daily(
                start_date=args.start_date, 
                end_date=args.end_date
            )
            print(f"✅ 成功下载 {len(df)} 条行情记录")
            
        elif args.action == 'update':
            print(f"🔄 增量更新行情数据 (向前{args.days_back}天)...")
            df = downloader.update_daily_data(days_back=args.days_back)
            if not df.empty:
                print(f"✅ 成功更新 {len(df)} 条行情记录")
            else:
                print("✅ 数据已是最新，无需更新")
        elif args.action == 'update_sector':
            # 同时更新最强板块数据
            print(f"🔥 增量更新最强板块数据 (向前{args.days_back}天)...")
            hot_df = downloader.update_hot_sectors(days_back=args.days_back)
            if not hot_df.empty:
                print(f"✅ 成功更新 {len(hot_df)} 条最强板块记录")
            else:
                print("✅ 最强板块数据已是最新，无需更新")
                
        elif args.action == 'summary':
            print("📊 获取数据摘要...")
            summary = downloader.get_concept_summary()
            print("\n" + "="*60)
            print("📋 同花顺概念板块数据摘要")
            print("="*60)
            
            # 概念板块基础信息
            concept_info = summary['concept_index']
            print(f"\n🏷️  概念板块基础信息:")
            if concept_info['file_exists']:
                if 'error' in concept_info:
                    print(f"   ❌ 读取错误: {concept_info['error']}")
                else:
                    print(f"   ✅ 文件存在")
                    print(f"   📅 更新时间: {concept_info.get('update_time', 'N/A')}")
                    print(f"   📊 概念数量: {concept_info.get('count', 'N/A')}")
                    print(f"   🏢 市场类型: {concept_info.get('exchange', 'N/A')}")
            else:
                print(f"   ❌ 文件不存在")
            
            # 成分股信息
            members_info = summary['concept_members']
            print(f"\n📊 成分股信息:")
            if members_info['file_exists']:
                if 'error' in members_info:
                    print(f"   ❌ 读取错误: {members_info['error']}")
                else:
                    print(f"   ✅ 文件存在")
                    print(f"   📅 更新时间: {members_info.get('update_time', 'N/A')}")
                    print(f"   📊 总记录数: {members_info.get('total_records', 'N/A')}")
                    print(f"   🏷️  概念数量: {members_info.get('unique_concepts', 'N/A')}")
                    print(f"   📈 股票数量: {members_info.get('unique_stocks', 'N/A')}")
            else:
                print(f"   ❌ 文件不存在")
            
            # 行情数据
            daily_info = summary['daily_data']
            print(f"\n📈 行情数据:")
            if daily_info['file_exists']:
                if 'error' in daily_info:
                    print(f"   ❌ 读取错误: {daily_info['error']}")
                else:
                    print(f"   ✅ 文件存在")
                    print(f"   📁 最新文件: {daily_info.get('latest_file', 'N/A')}")
                    print(f"   📊 总记录数: {daily_info.get('total_records', 'N/A')}")
                    print(f"   🏷️  概念数量: {daily_info.get('unique_concepts', 'N/A')}")
                    date_range = daily_info.get('date_range', {})
                    print(f"   📅 日期范围: {date_range.get('min_date', 'N/A')} - {date_range.get('max_date', 'N/A')}")
            else:
                print(f"   ❌ 文件不存在")
            
            print("\n" + "="*60)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
