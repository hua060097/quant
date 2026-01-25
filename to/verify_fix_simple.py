"""
简单验证：测试修复后的 download_daily.py

下载小范围数据（10只股票 × 2天）验证数据正确性
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

def main():
    print("=" * 80)
    print("验证修复后的 download_daily.py")
    print("=" * 80)

    # 测试参数：只下载2天，10只股票
    test_start = "20260108"
    test_end = "20260109"

    print(f"\n[测试参数]")
    print(f"  日期范围: {test_start} ~ {test_end}")
    print(f"  预期天数: 2 天")

    print(f"\n[步骤1] 运行 download_daily.py...")

    # 运行下载脚本
    cmd = [
        sys.executable,
        "download_daily.py",
        "--start", test_start,
        "--end", test_end
    ]

    print(f"  命令: {' '.join(cmd)}")
    print(f"\n开始下载...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800  # 30分钟超时
    )

    print(f"\n下载完成")
    print(f"  返回码: {result.returncode}")

    # 显示日志（最后20行）
    print(f"\n[日志输出] (最后20行)")
    log_lines = result.stdout.split('\n')
    for line in log_lines[-20:]:
        if line.strip():
            print(f"  {line}")

    # 验证数据
    print(f"\n[步骤2] 验证数据质量...")

    daily_file = Path('local_data/daily/202601/daily_data.parquet')

    if not daily_file.exists():
        print(f"  [FAIL] 文件不存在: {daily_file}")
        return 1

    df = pd.read_parquet(daily_file)

    # 检查测试日期的数据
    df_test = df[df['date_str'].isin([test_start, test_end])].copy()

    print(f"\n[数据统计]")
    print(f"  总行数: {len(df_test):,}")
    print(f"  股票数: {df_test['stock_code'].nunique():,}")
    print(f"  日期数: {df_test['date_str'].nunique()}")

    # 检查重复
    dup = df_test.duplicated(subset=['stock_code', 'date_str'], keep=False).sum()

    print(f"\n[重复检查]")
    print(f"  重复行数: {dup}")

    if dup == 0:
        print(f"  [OK] 无重复数据")
    else:
        print(f"  [FAIL] 存在 {dup} 条重复数据")
        return 1

    # 计算预期行数
    expected_rows = df_test['stock_code'].nunique() * df_test['date_str'].nunique()

    print(f"\n[数据量检查]")
    print(f"  实际行数: {len(df_test):,}")
    print(f"  预期行数: {expected_rows:,}")

    if len(df_test) == expected_rows:
        print(f"  [OK] 数据量正常")
    else:
        print(f"  [FAIL] 数据量异常（比值: {len(df_test)/expected_rows:.2f}x）")
        return 1

    # 抽样检查
    print(f"\n[抽样检查] 随机选择3只股票")
    sample_stocks = df_test['stock_code'].unique()[:3]

    for stock in sample_stocks:
        stock_df = df_test[df_test['stock_code'] == stock]
        print(f"\n  {stock}: {len(stock_df)} 条记录")
        print(stock_df[['date_str', 'open', 'close']].head(5))

    # 最终结论
    print(f"\n" + "=" * 80)
    print(f"[验证结论]")
    print(f"=" * 80)

    if dup == 0 and len(df_test) == expected_rows:
        print(f"[SUCCESS] 验证通过！")
        print(f"  - 无重复数据")
        print(f"  - 数据量正常")
        print(f"  - 修复成功！")
        return 0
    else:
        print(f"[FAIL] 验证失败")
        print(f"  - 重复数据: {dup}")
        print(f"  - 数据量: {len(df_test)} (预期 {expected_rows})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
