#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数据生成流水线脚本
按顺序执行所有数据生成脚本，确保每个脚本执行完成后再执行下一个

使用方法:
    python run_data_pipeline.py --start 20250101 --end 20250131
    python run_data_pipeline.py -s 20250101 -e 20250131

作者: Claude
日期: 2025-01-14
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_script(script_path, description, args=None):
    """
    运行单个脚本并等待其完成

    Args:
        script_path (str): 脚本路径
        description (str): 脚本描述
        args (list): 传递给脚本的参数列表

    Returns:
        bool: 成功返回True，失败返回False
    """
    print("\n" + "="*80)
    print(f"[开始执行] {description}")
    print(f"[脚本] {script_path}")
    if args:
        print(f"[参数] {' '.join(args)}")
    print("="*80 + "\n")

    # 构建命令
    cmd = [sys.executable, script_path] + (args or [])

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行脚本并等待完成
        # 注意：不捕获stdout/stderr，让子脚本直接输出到控制台（避免编码问题）
        result = subprocess.run(
            cmd,
            check=True,
            # stdout=subprocess.PIPE,  # 注释掉，让子脚本直接输出
            # stderr=subprocess.STDOUT,  # 注释掉
            # text=True,
            # encoding='utf-8',
            # errors='replace'
        )

        # 计算耗时
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*80)
        print(f"[成功] {description} 执行成功!")
        print(f"[耗时] {minutes}分{seconds}秒")
        print("="*80 + "\n")

        return True

    except subprocess.CalledProcessError as e:
        # 计算耗时
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*80)
        print(f"[失败] {description} 执行失败!")
        print(f"[耗时] {minutes}分{seconds}秒")
        print(f"[返回码] {e.returncode}")
        print(f"[提示] 请查看该脚本的日志文件获取详细错误信息")
        print("="*80 + "\n")

        return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"[异常] {description} 执行时发生异常!")
        print(f"[耗时] {elapsed_time:.2f}秒")
        print(f"[异常信息] {str(e)}")
        print("="*80 + "\n")
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='统一数据生成流水线 - 按顺序执行所有数据生成脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_data_pipeline.py --start 20250101 --end 20250131
  python run_data_pipeline.py -s 20250101 -e 20250131
  python run_data_pipeline.py --start 20250101 --end 20250131 --stop-on-error
        """
    )

    parser.add_argument(
        '--start', '-s',
        type=str,
        required=True,
        help='开始日期 (YYYYMMDD格式)'
    )

    parser.add_argument(
        '--end', '-e',
        type=str,
        required=True,
        help='结束日期 (YYYYMMDD格式)'
    )

    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='遇到错误时停止执行 (默认会继续执行后续脚本)'
    )

    args = parser.parse_args()

    # 验证日期格式
    if len(args.start) != 8 or not args.start.isdigit():
        print("[错误] 开始日期格式不正确，应为YYYYMMDD格式 (例如: 20250101)")
        sys.exit(1)

    if len(args.end) != 8 or not args.end.isdigit():
        print("[错误] 结束日期格式不正确，应为YYYYMMDD格式 (例如: 20250131)")
        sys.exit(1)

    # 项目根目录
    project_root = Path(__file__).parent

    # 定义要执行的脚本列表
    # 注意: 顺序很重要，必须按照数据依赖关系执行
    scripts = [
        {
            'path': project_root / 'download.py',
            'desc': '1. 下载A股Tick数据和股票列表数据',
            'args': ['--start', args.start, '--end', args.end]
        },
        {
            'path': project_root / 'download_daily.py',
            'desc': '2. 下载日线数据',
            'args': ['--start', args.start, '--end', args.end]
        },
        {
            'path': project_root / 'fetch_st_stocks.py',
            'desc': '3. 下载st数据',
            'args': ['--s', args.start, '--e', args.end]
        },
        {
            'path': project_root / 'integrated_stock_selection' /  'scripts' / 'download_regulatory_indices.py',
            'desc': '4. 下载指数数据',
            'args': ['--start', args.start, '--end', args.end]
        },
        {
            'path': project_root / 'advanced_quant_data_fixed.py',
            'desc': '5. 高级量化数据下载和存储',
            'args': ['--start_date', args.start, '--end_date', args.end]
        },
        {
            'path': project_root / 'enhanced_limit_data_collector.py',
            'desc': '6. 涨跌停数据收集器',
            'args': ['--start_date', args.start, '--end_date', args.end]
        },
        {
            'path': project_root / 'generate_daily_features_simple.py',
            'desc': '7. 生成日线特征',
            'args': ['--start_date', args.start, '--end_date', args.end]
        },
        {
            'path': project_root / 'download_moneyflow_ths.py',
            'desc': '8. 下载个股资金流',
            'args': ['--start', args.start, '--end', args.end]
        },
        {
            'path': project_root / 'market_status' / 'market_status_processor.py',
            'desc': '9. 市场状态处理',
            'args': ['--start_date', args.start, '--end_date', args.end]
        },
        {
            'path': project_root / 'integrated_stock_selection' / 'main_alpha原始_overlay_origin.py',
            'desc': '10. 生成因子数据',
            'args': [
                '--mode', 'generate_factors',
                '--start_date', '20230601',  #args.start,
                '--end_date', args.end,
                # '--incremental_save'
            ]
        },
    ]

    # 打印配置信息
    print("\n" + "="*80)
    print("[统一数据生成流水线]")
    print("="*80)
    print(f"[日期范围] {args.start} - {args.end}")
    print(f"[脚本数量] {len(scripts)}")
    print(f"[错误处理] {'遇到错误时停止' if args.stop_on_error else '遇到错误时继续'}")
    print("="*80 + "\n")

    # 执行所有脚本
    total_start_time = time.time()
    success_count = 0
    failure_count = 0
    failed_scripts = []

    for idx, script_info in enumerate(scripts, 1):
        script_path = script_info['path']
        description = script_info['desc']
        script_args = script_info['args']

        # 检查脚本是否存在
        if not script_path.exists():
            print(f"\n[警告] 脚本不存在，跳过: {script_path}")
            failure_count += 1
            failed_scripts.append((description, "脚本不存在"))
            if args.stop_on_error:
                print("[错误] 由于设置了--stop-on-error，停止执行")
                break
            continue

        # 执行脚本
        success = run_script(str(script_path), description, script_args)

        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_scripts.append((description, "执行失败"))
            if args.stop_on_error:
                print("\n[错误] 由于设置了--stop-on-error，停止执行")
                break

    # 计算总耗时
    total_elapsed_time = time.time() - total_start_time
    total_minutes = int(total_elapsed_time // 60)
    total_seconds = int(total_elapsed_time % 60)

    # 打印最终统计
    print("\n" + "="*80)
    print("[执行完成统计]")
    print("="*80)
    print(f"[成功] {success_count}/{len(scripts)}")
    print(f"[失败] {failure_count}/{len(scripts)}")
    print(f"[总耗时] {total_minutes}分{total_seconds}秒")

    if failed_scripts:
        print("\n[失败的脚本]:")
        for desc, reason in failed_scripts:
            print(f"  - {desc}: {reason}")

    print("="*80 + "\n")

    # 根据执行结果返回退出码
    sys.exit(0 if failure_count == 0 else 1)


if __name__ == "__main__":
    main()
