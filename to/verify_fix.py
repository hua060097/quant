"""
验证20260109数据清理结果
"""

import pandas as pd
from pathlib import Path

def main():
    daily_file = Path('local_data/daily/202601/daily_data.parquet')

    if not daily_file.exists():
        print(f"文件不存在: {daily_file}")
        return

    # 读取数据
    df = pd.read_parquet(daily_file)
    df_0609 = df[df['date_str'] == '20260109'].copy()

    print("=" * 80)
    print("20260109数据验证")
    print("=" * 80)

    print(f"\n[基本信息]")
    print(f"  总行数: {len(df_0609):,}")
    print(f"  股票数: {df_0609['stock_code'].nunique()}")

    # 检查重复
    dup = df_0609.duplicated(subset=['stock_code', 'date_str'], keep=False).sum()
    print(f"\n[重复检查]")
    print(f"  重复行数: {dup}")

    # 验证结果
    print(f"\n[验证结果]")
    if len(df_0609) == df_0609['stock_code'].nunique() and dup == 0:
        print(f"  状态: 通过 [OK]")
        print(f"  详情: 每只股票只有一条记录，无重复数据")
        print(f"\n[结论]")
        print(f"  日线数据清理成功！")
        print(f"  预期行数: 5,001")
        print(f"  实际行数: {len(df_0609):,}")
        if abs(len(df_0609) - 5001) < 100:
            print(f"  数据质量: 优秀")
    else:
        print(f"  状态: 失败 [FAIL]")
        print(f"  详情: 仍有重复数据或记录数异常")

    # 显示文件大小
    file_size_mb = daily_file.stat().st_size / (1024 * 1024)
    print(f"\n[文件信息]")
    print(f"  文件大小: {file_size_mb:.2f} MB")

    # 对比20260108
    df_0608 = df[df['date_str'] == '20260108'].copy()
    print(f"\n[对比20260108]")
    print(f"  20260108行数: {len(df_0608):,}")
    print(f"  20260109行数: {len(df_0609):,}")
    ratio = len(df_0609) / len(df_0608) if len(df_0608) > 0 else 0
    print(f"  比值: {ratio:.2f}x")

    if 0.95 <= ratio <= 1.05:
        print(f"  评价: 数据量一致，修复成功！")
    else:
        print(f"  评价: 数据量差异较大，需要检查")

if __name__ == "__main__":
    main()
