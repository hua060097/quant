#!/usr/bin/env python3
"""
测试增量缓存功能
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """运行命令并记录时间"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    print(f"⏱️ 执行时间: {elapsed_time:.2f}秒")
    print(f"返回码: {result.returncode}")
    
    if result.stdout:
        print("标准输出:")
        print(result.stdout[-2000:])  # 只显示最后2000字符
    
    if result.stderr:
        print("错误输出:")
        print(result.stderr[-1000:])  # 只显示最后1000字符
    
    return result.returncode == 0, elapsed_time

def check_cache_structure():
    """检查缓存目录结构"""
    print(f"\n{'='*60}")
    print(f"📁 检查缓存目录结构")
    print(f"{'='*60}")
    
    cache_dir = Path("factor_cache")
    if cache_dir.exists():
        print(f"缓存根目录: {cache_dir}")
        
        # 检查因子组目录
        for group_dir in cache_dir.iterdir():
            if group_dir.is_dir():
                print(f"  因子组: {group_dir.name}")
                cache_files = list(group_dir.glob("*.pkl"))
                print(f"    缓存文件数: {len(cache_files)}")
                for cache_file in sorted(cache_files)[:5]:  # 只显示前5个
                    size_mb = cache_file.stat().st_size / 1024 / 1024
                    print(f"      {cache_file.name} ({size_mb:.2f}MB)")
                if len(cache_files) > 5:
                    print(f"      ... 还有 {len(cache_files) - 5} 个文件")
    else:
        print("❌ 缓存目录不存在")

def main():
    """主测试流程"""
    print("🧪 增量缓存功能测试")
    
    # 测试1: 第一次运行 (20230801-20230810)
    success1, time1 = run_command(
        "python main.py --mode generate_factors --factor_groups kline_pattern_factors --start_date 20230801 --end_date 20230810",
        "第一次运行: 20230801-20230810 (应该全量计算)"
    )
    
    check_cache_structure()
    
    if not success1:
        print("❌ 第一次运行失败")
        return
    
    # 测试2: 相同日期范围再次运行 (应该完全缓存命中)
    success2, time2 = run_command(
        "python main.py --mode generate_factors --factor_groups kline_pattern_factors --start_date 20230801 --end_date 20230810",
        "第二次运行: 20230801-20230810 (应该完全缓存命中)"
    )
    
    if not success2:
        print("❌ 第二次运行失败")
        return
    
    # 测试3: 扩展日期范围 (20230801-20230815) (应该增量计算)
    success3, time3 = run_command(
        "python main.py --mode generate_factors --factor_groups kline_pattern_factors --start_date 20230801 --end_date 20230815",
        "第三次运行: 20230801-20230815 (应该增量计算)"
    )
    
    check_cache_structure()
    
    if not success3:
        print("❌ 第三次运行失败")
        return
    
    # 测试4: 强制重新计算
    success4, time4 = run_command(
        "python main.py --mode generate_factors --factor_groups kline_pattern_factors --start_date 20230801 --end_date 20230810 --force",
        "第四次运行: 20230801-20230810 --force (应该强制重新计算)"
    )
    
    if not success4:
        print("❌ 第四次运行失败")
        return
    
    # 结果分析
    print(f"\n{'='*60}")
    print(f"📊 测试结果分析")
    print(f"{'='*60}")
    print(f"第一次运行 (全量计算): {time1:.2f}秒")
    print(f"第二次运行 (完全缓存): {time2:.2f}秒")
    print(f"第三次运行 (增量计算): {time3:.2f}秒")
    print(f"第四次运行 (强制重算): {time4:.2f}秒")
    
    # 性能分析
    if time2 < time1 * 0.1:
        print("✅ 缓存命中性能优秀")
    else:
        print("⚠️ 缓存命中性能可能有问题")
    
    if time3 < time1 * 0.8:
        print("✅ 增量更新性能优秀")
    else:
        print("⚠️ 增量更新性能可能有问题")

if __name__ == "__main__":
    main()
