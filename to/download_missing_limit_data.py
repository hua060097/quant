#!/usr/bin/env python3
"""
下载缺失的涨跌停历史数据
补充20240920-20250619期间的涨跌停数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

def get_trading_days_simple(start_date: str, end_date: str) -> list:
    """获取指定范围内的交易日列表（简化版本，排除周末）"""
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    trading_days = []
    current = start
    while current <= end:
        # 排除周末
        if current.weekday() < 5:  # 0-4是周一到周五
            trading_days.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return trading_days

def download_historical_limit_data():
    """下载历史涨跌停数据"""
    print("📥 下载缺失的历史涨跌停数据")
    print("=" * 60)
    
    # 1. 确定需要下载的日期范围
    target_start = "20240920"
    target_end = "20250619"
    
    print(f"🎯 目标范围: {target_start} - {target_end}")
    
    # 2. 获取交易日列表
    required_dates = get_trading_days_simple(target_start, target_end)
    print(f"📅 需要下载的交易日数: {len(required_dates)}")
    print(f"📅 日期范围: {required_dates[0]} - {required_dates[-1]}")
    
    # 3. 检查现有数据
    limit_data_file = Path("local_data/limit_data/all_limit_data.parquet")
    existing_dates = set()
    
    if limit_data_file.exists():
        try:
            existing_data = pd.read_parquet(limit_data_file)
            if 'trade_date' in existing_data.columns:
                existing_dates = set(existing_data['trade_date'].dt.strftime('%Y%m%d').unique())
            print(f"📊 现有数据日期数: {len(existing_dates)}")
        except Exception as e:
            print(f"⚠️  读取现有数据失败: {e}")
    
    # 4. 确定需要下载的日期
    missing_dates = [date for date in required_dates if date not in existing_dates]
    print(f"📥 需要下载的日期数: {len(missing_dates)}")
    
    if not missing_dates:
        print("✅ 所有数据都已存在，无需下载")
        return
    
    # 5. 分批下载数据
    print(f"\n🚀 开始下载历史数据...")
    
    # 按月份分组下载
    dates_by_month = {}
    for date in missing_dates:
        month = date[:6]  # YYYYMM
        if month not in dates_by_month:
            dates_by_month[month] = []
        dates_by_month[month].append(date)
    
    print(f"📦 分为 {len(dates_by_month)} 个月份批次下载")
    
    # 6. 调用enhanced_limit_data_collector下载数据
    try:
        # 导入数据收集器
        sys.path.append("main_sector_stock")
        from enhanced_limit_data_collector import EnhancedLimitDataCollector
        
        # 初始化收集器
        collector = EnhancedLimitDataCollector()
        
        # 按月份下载
        for month in sorted(dates_by_month.keys()):
            dates = dates_by_month[month]
            print(f"\n📅 下载 {month} 月数据: {len(dates)} 个交易日")
            
            # 计算月份的开始和结束日期
            month_start = month + "01"
            if month == "202412":
                month_end = "20241231"
            elif month in ["202501", "202503", "202505", "202507", "202508", "202510", "202512"]:
                month_end = month + "31"
            elif month in ["202504", "202506", "202509", "202511"]:
                month_end = month + "30"
            elif month == "202502":
                month_end = "20250228"  # 2025年不是闰年
            else:
                # 其他月份
                if month.endswith("02"):
                    month_end = month + "28"
                elif month.endswith(("04", "06", "09", "11")):
                    month_end = month + "30"
                else:
                    month_end = month + "31"
            
            try:
                # 下载该月份的数据
                print(f"  下载范围: {month_start} - {month_end}")
                collector.collect_and_save_data(month_start, month_end)
                print(f"  ✅ {month} 月数据下载完成")
                
            except Exception as e:
                print(f"  ❌ {month} 月数据下载失败: {e}")
                continue
        
        print(f"\n🎉 历史数据下载完成！")
        
        # 7. 验证下载结果
        print(f"\n🔍 验证下载结果...")
        
        if limit_data_file.exists():
            updated_data = pd.read_parquet(limit_data_file)
            if 'trade_date' in updated_data.columns:
                updated_dates = set(updated_data['trade_date'].dt.strftime('%Y%m%d').unique())
                
                print(f"📊 更新后数据统计:")
                print(f"  总记录数: {len(updated_data)}")
                print(f"  日期数: {len(updated_dates)}")
                print(f"  日期范围: {min(updated_dates)} - {max(updated_dates)}")
                
                # 检查完整性
                still_missing = [date for date in required_dates if date not in updated_dates]
                print(f"  仍缺失日期数: {len(still_missing)}")
                print(f"  完整性: {(len(required_dates) - len(still_missing)) / len(required_dates) * 100:.1f}%")
                
                if still_missing:
                    print(f"  ⚠️  仍缺失的日期: {still_missing[:10]}{'...' if len(still_missing) > 10 else ''}")
                else:
                    print(f"  ✅ 所有需要的数据都已下载完成！")
        
    except ImportError as e:
        print(f"❌ 无法导入enhanced_limit_data_collector: {e}")
        print(f"💡 请确保enhanced_limit_data_collector.py在main_sector_stock目录中")
        print(f"💡 或者手动运行涨跌停数据更新程序")
        
        # 提供手动下载的建议
        print(f"\n📝 手动下载建议:")
        print(f"1. 进入main_sector_stock目录")
        print(f"2. 运行: python enhanced_limit_data_collector.py")
        print(f"3. 设置日期范围: {target_start} - {target_end}")
        
    except Exception as e:
        print(f"❌ 下载过程中出现错误: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """主函数"""
    print("🔧 涨跌停历史数据下载工具")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 检查必要的目录和文件
    required_dirs = [
        "local_data/limit_data",
        "main_sector_stock"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ 缺少目录: {dir_path}")
            return
        else:
            print(f"✅ 目录存在: {dir_path}")
    
    # 开始下载
    download_historical_limit_data()

if __name__ == "__main__":
    main()
