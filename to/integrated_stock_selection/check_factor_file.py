#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查因子文件状态的工具
"""

import sys
import pandas as pd
from pathlib import Path

def check_factor_file():
    factor_file = Path("factor_lab_data/factors1.parquet")

    if not factor_file.exists():
        print("因子文件不存在")
        return False

    print(f"检查因子文件: {factor_file}")
    print(f"文件大小: {factor_file.stat().st_size / (1024*1024):.2f} MB")

    try:
        df = pd.read_parquet(factor_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 检查索引列
        index_cols = ['date', 'stock_code']
        missing_index = [col for col in index_cols if col not in df.columns]
        if missing_index:
            print(f"缺少索引列: {missing_index}")

        # 检查因子列
        factor_cols = [col for col in df.columns if col not in index_cols]
        print(f"因子列数: {len(factor_cols)}")

        if factor_cols:
            print(f"因子列表: {factor_cols[:10]}{'...' if len(factor_cols) > 10 else ''}")

            # 检查所有因子列的数据质量
            print("\n=== 因子数据质量报告 ===")
            factor_quality = {}

            for col in factor_cols:
                nan_count = df[col].isna().sum()
                total_count = len(df)
                nan_ratio = nan_count / total_count if total_count > 0 else 0
                factor_quality[col] = {
                    'nan_count': nan_count,
                    'nan_ratio': nan_ratio,
                    'valid_count': total_count - nan_count
                }

            # 按NaN比例排序并显示
            sorted_factors = sorted(factor_quality.items(), key=lambda x: x[1]['nan_ratio'])

            print(f"\n所有因子的NaN值统计 (共{len(factor_cols)}个因子):")
            print("因子名称".ljust(30) + "NaN数量".rjust(12) + "NaN比例".rjust(12) + "有效数据".rjust(12))
            print("-" * 70)

            for col, quality in sorted_factors:
                print(f"{col[:28]:<30} {quality['nan_count']:>10,} {quality['nan_ratio']:>10.2%} {quality['valid_count']:>10,}")

            # 统计质量等级
            excellent = sum(1 for q in factor_quality.values() if q['nan_ratio'] == 0)
            good = sum(1 for q in factor_quality.values() if 0 < q['nan_ratio'] <= 0.05)
            fair = sum(1 for q in factor_quality.values() if 0.05 < q['nan_ratio'] <= 0.20)
            poor = sum(1 for q in factor_quality.values() if q['nan_ratio'] > 0.20)

            print(f"\n=== 数据质量等级统计 ===")
            print(f"优秀 (NaN=0%): {excellent} 个因子 ({excellent/len(factor_cols)*100:.1f}%)")
            print(f"良好 (NaN≤5%): {good} 个因子 ({good/len(factor_cols)*100:.1f}%)")
            print(f"一般 (5%<NaN≤20%): {fair} 个因子 ({fair/len(factor_cols)*100:.1f}%)")
            print(f"较差 (NaN>20%): {poor} 个因子 ({poor/len(factor_cols)*100:.1f}%)")

            # 找出质量最差的因子
            worst_factors = sorted(factor_quality.items(), key=lambda x: x[1]['nan_ratio'], reverse=True)[:5]
            print(f"\n质量最差的5个因子:")
            for i, (col, quality) in enumerate(worst_factors, 1):
                print(f"{i}. {col}: NaN率 {quality['nan_ratio']:.2%}")

        else:
            print("没有发现任何因子列！")
            return False

        # 检查日期范围
        if 'date' in df.columns:
            date_min = df['date'].min()
            date_max = df['date'].max()
            print(f"\n日期范围: {date_min} ~ {date_max}")
            print(f"总天数: {(date_max - date_min).days + 1}天")

        # 检查股票数量
        if 'stock_code' in df.columns:
            stock_count = df['stock_code'].nunique()
            print(f"股票数量: {stock_count}")

        # 检查数据完整性
        total_records = len(df)
        expected_records = None

        if 'date' in df.columns and 'stock_code' in df.columns:
            unique_dates = df['date'].nunique()
            unique_stocks = df['stock_code'].nunique()
            expected_records = unique_dates * unique_stocks

            print(f"\n=== 数据完整性检查 ===")
            print(f"实际记录数: {total_records:,}")
            print(f"预期记录数: {expected_records:,}")
            print(f"数据完整性: {total_records/expected_records*100:.2f}%" if expected_records > 0 else "无法计算完整性")

        print("因子文件检查完成")
        return True

    except Exception as e:
        print(f"读取因子文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_factor_file()
    sys.exit(0 if success else 1)