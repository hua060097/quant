"""
检查日线数据下载是否成功

成功标准：
1. 股票数量 > 4500
2. 没有大量空值/NaN值
"""

import pandas as pd
from pathlib import Path
import sys

def check_date_data(df, target_date_str):
    """
    检查指定日期的数据质量

    Args:
        df: 完整的DataFrame
        target_date_str: 目标日期字符串，如 '20260108'

    Returns:
        dict: 检查结果
    """
    print(f"\n{'='*80}")
    print(f"检查日期: {target_date_str}")
    print(f"{'='*80}")

    # 1. 过滤出指定日期的数据
    date_df = df[df['date_str'] == target_date_str]

    if len(date_df) == 0:
        print(f"[FAIL] {target_date_str}: 没有找到任何数据")
        return {
            'date': target_date_str,
            'success': False,
            'reason': '无数据',
            'stock_count': 0,
            'null_count': 0
        }

    # 2. 统计股票数量
    stock_count = date_df['stock_code'].nunique()
    print(f"[INFO] 股票数量: {stock_count}")

    # 3. 检查空值情况
    null_summary = {}

    for col in date_df.columns:
        null_count = date_df[col].isna().sum()
        null_pct = (null_count / len(date_df)) * 100

        if null_count > 0:
            null_summary[col] = {
                'count': null_count,
                'percentage': null_pct
            }

    # 4. 检查关键字段的空值
    critical_fields = ['open', 'high', 'low', 'close', 'volume']
    critical_null = 0

    print(f"\n[CHECK] 关键字段空值检查:")
    for field in critical_fields:
        if field in date_df.columns:
            null_count = date_df[field].isna().sum()
            null_pct = (null_count / len(date_df)) * 100
            print(f"  {field}: {null_count} ({null_pct:.2f}%)")

            if null_count > 0:
                critical_null += null_count

    # 5. 显示有数据的列信息
    print(f"\n[SUMMARY] 数据概览:")
    print(f"  总行数: {len(date_df)}")
    print(f"  股票数量: {stock_count}")
    print(f"  字段数量: {len(date_df.columns)}")
    print(f"  字段列表: {list(date_df.columns)}")

    # 6. 检查是否有大片NaN（超过50%的数据行有空值）
    rows_with_null = date_df.isnull().any(axis=1).sum()
    rows_with_null_pct = (rows_with_null / len(date_df)) * 100

    print(f"\n[WARNING] 空值行统计:")
    print(f"  含有空值的行数: {rows_with_null} ({rows_with_null_pct:.2f}%)")

    # 7. 显示前几行数据样本
    print(f"\n[SAMPLE] 数据样本（前5行）:")
    print(date_df.head(5).to_string())

    # 8. 判断是否成功
    success = True
    reasons = []

    if stock_count <= 4500:
        success = False
        reasons.append(f"股票数量不足（{stock_count} <= 4500）")

    if critical_null > stock_count * 0.1:  # 超过10%的股票有关键字段空值
        success = False
        reasons.append(f"关键字段空值过多（{critical_null}个）")

    if rows_with_null_pct > 50:  # 超过50%的行有空值
        success = False
        reasons.append(f"空值行过多（{rows_with_null_pct:.1f}%）")

    return {
        'date': target_date_str,
        'success': success,
        'reason': '; '.join(reasons) if reasons else '正常',
        'stock_count': stock_count,
        'total_rows': len(date_df),
        'critical_null': critical_null,
        'rows_with_null_pct': rows_with_null_pct,
        'null_summary': null_summary
    }


def main():
    # 文件路径
    file_path = Path('local_data/daily/202601/daily_data.parquet')

    print(f"{'='*80}")
    print(f"日线数据质量检查")
    print(f"{'='*80}")
    print(f"文件路径: {file_path}")

    # 检查文件是否存在
    if not file_path.exists():
        print(f"\n[ERROR] 文件不存在")
        print(f"请确认路径是否正确: {file_path}")
        return

    # 读取数据
    print(f"\n[READ] 读取数据...")
    try:
        df = pd.read_parquet(file_path)
        print(f"[OK] 文件读取成功")
        print(f"  总行数: {len(df):,}")
        print(f"  字段数量: {len(df.columns)}")
        print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  股票数量: {df['stock_code'].nunique()}")
        print(f"  日期数量: {df['date'].nunique()}")
    except Exception as e:
        print(f"[ERROR] 文件读取失败: {e}")
        return

    # 检查指定日期
    target_dates = ['20260108', '20260109']
    results = []

    for target_date in target_dates:
        result = check_date_data(df, target_date)
        results.append(result)

    # 总结报告
    print(f"\n{'='*80}")
    print(f"[REPORT] 检查总结报告")
    print(f"{'='*80}")

    for result in results:
        status = "[OK]" if result['success'] else "[FAIL]"
        print(f"\n日期: {result['date']}")
        print(f"状态: {status}")
        print(f"股票数量: {result['stock_count']}")
        print(f"总行数: {result['total_rows']}")

        if not result['success']:
            print(f"失败原因: {result['reason']}")

        if 'critical_null' in result:
            print(f"关键字段空值: {result['critical_null']}")

        if 'rows_with_null_pct' in result:
            print(f"空值行占比: {result['rows_with_null_pct']:.2f}%")

    # 最终判断
    all_success = all(r['success'] for r in results)

    print(f"\n{'='*80}")
    if all_success:
        print(f"[SUCCESS] 所有日期数据检查通过！")
        print(f"{'='*80}")
        return 0
    else:
        print(f"[FAIL] 部分日期数据检查失败！")
        print(f"{'='*80}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
