"""
深度诊断 sector_broken_resilience_alpha 和 market_trend_adaptive_beta 的异常模式
"""
import pandas as pd
import numpy as np
from pathlib import Path

def deep_diagnose_factors(file1_path, file2_path):
    """深度诊断因子差异"""

    print("=" * 100)
    print("深度异常模式诊断")
    print("=" * 100)

    # 读取文件
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    print(f"\n文件1 (副本): {df1.shape}")
    print(f"文件2 (当前): {df2.shape}")

    # 对齐索引
    common_cols = list(set(df1.columns) & set(df2.columns))

    # 使用stock_code和date作为对齐键（如果存在）
    if 'stock_code' in common_cols and 'date' in common_cols:
        print("\n[使用 stock_code 和 date 对齐数据]")
        df1_temp = df1[common_cols].copy()
        df2_temp = df2[common_cols].copy()

        # 创建复合键
        df1_temp['_key'] = df1_temp['stock_code'].astype(str) + '_' + df1_temp['date'].astype(str)
        df2_temp['_key'] = df2_temp['stock_code'].astype(str) + '_' + df2_temp['date'].astype(str)

        # Merge对齐
        merged = df1_temp.merge(
            df2_temp,
            on='_key',
            suffixes=('_file1', '_file2'),
            how='inner',
            indicator=True
        )

        print(f"  对齐后行数: {len(merged)}")
        print(f"  仅在文件1: {(merged['_merge'] == 'left_only').sum()}")
        print(f"  仅在文件2: {(merged['_merge'] == 'right_only').sum()}")
        print(f"  共同行: {(merged['_merge'] == 'both').sum()}")

        # 保留共同行
        merged = merged[merged['_merge'] == 'both'].copy()

        # 分析 market_trend_adaptive_beta
        if 'market_trend_adaptive_beta_file1' in merged.columns:
            print("\n" + "=" * 100)
            print("因子: market_trend_adaptive_beta")
            print("=" * 100)

            col1 = merged['market_trend_adaptive_beta_file1']
            col2 = merged['market_trend_adaptive_beta_file2']

            # 检查异常值模式
            print("\n[异常值模式分析]")

            # 文件2中的异常值
            abnormal_threshold = 50.0
            abnormal_mask_file2 = abs(col2) > abnormal_threshold
            abnormal_count_file2 = abnormal_mask_file2.sum()

            print(f"  文件2中绝对值>{abnormal_threshold}的行数: {abnormal_count_file2} ({abnormal_count_file2/len(col2)*100:.2f}%)")

            if abnormal_count_file2 > 0:
                # 获取异常值
                abnormal_values = col2[abnormal_mask_file2].value_counts()
                print(f"  文件2异常值的分布:")
                for val, count in abnormal_values.head(10).items():
                    pct = count / abnormal_count_file2 * 100
                    print(f"    值={val:.6f}: {count}次 ({pct:.2f}%)")

                # 检查这些异常行在文件1中的值
                print(f"\n  异常行在文件1中的对应值:")
                corresponding_values = col1[abnormal_mask_file2]
                print(f"    均值: {corresponding_values.mean():.6f}")
                print(f"    标准差: {corresponding_values.std():.6f}")
                print(f"    最小值: {corresponding_values.min():.6f}")
                print(f"    最大值: {corresponding_values.max():.6f}")
                print(f"    值分布:")
                for val, count in corresponding_values.value_counts().head(10).items():
                    pct = count / len(corresponding_values) * 100
                    print(f"      值={val:.6f}: {count}次 ({pct:.2f}%)")

                # 检查这些异常行的其他特征
                print(f"\n  异常行的股票分布:")
                stock_codes = merged.loc[abnormal_mask_file2, 'stock_code_file1']
                for stock, count in stock_codes.value_counts().head(10).items():
                    pct = count / abnormal_count_file2 * 100
                    print(f"    {stock}: {count}次 ({pct:.2f}%)")

                print(f"\n  异常行的日期分布:")
                dates = merged.loc[abnormal_mask_file2, 'date_file1']
                for date, count in dates.value_counts().head(10).items():
                    pct = count / abnormal_count_file2 * 100
                    print(f"    {date}: {count}次 ({pct:.2f}%)")

                # 检查是否与某个条件相关
                print(f"\n  检查异常行是否与某些条件相关:")

                # 检查其他列在这异常行的情况
                check_cols = [
                    'pct_chg_1d', 'beta_60d', 'sector_mkt_corr_20d',
                    'market_trend_strength', 'market_volatility_context'
                ]

                for check_col in check_cols:
                    if f'{check_col}_file1' in merged.columns:
                        col_data = merged.loc[abnormal_mask_file2, f'{check_col}_file1']
                        print(f"    {check_col}:")
                        print(f"      均值: {col_data.mean():.6f}, 标准差: {col_data.std():.6f}")

            # 文件1中的异常值
            abnormal_mask_file1 = abs(col1) > abnormal_threshold
            abnormal_count_file1 = abnormal_mask_file1.sum()

            print(f"\n  文件1中绝对值>{abnormal_threshold}的行数: {abnormal_count_file1} ({abnormal_count_file1/len(col1)*100:.2f}%)")

            if abnormal_count_file1 > 0:
                abnormal_values = col1[abnormal_mask_file1].value_counts()
                print(f"  文件1异常值的分布:")
                for val, count in abnormal_values.head(10).items():
                    pct = count / abnormal_count_file1 * 100
                    print(f"    值={val:.6f}: {count}次 ({pct:.2f}%)")

        # 分析 sector_broken_resilience_alpha
        if 'sector_broken_resilience_alpha_file1' in merged.columns:
            print("\n" + "=" * 100)
            print("因子: sector_broken_resilience_alpha")
            print("=" * 100)

            col1 = merged['sector_broken_resilience_alpha_file1']
            col2 = merged['sector_broken_resilience_alpha_file2']

            # 基本统计
            print(f"\n[基本统计]")
            print(f"  文件1: 均值={col1.mean():.6f}, 标准差={col1.std():.6f}, 范围=[{col1.min():.6f}, {col1.max():.6f}]")
            print(f"  文件2: 均值={col2.mean():.6f}, 标准差={col2.std():.6f}, 范围=[{col2.min():.6f}, {col2.max():.6f}]")

            # 差异分析
            diff = col2 - col1
            abs_diff = np.abs(diff)

            large_diff_mask = abs_diff > 1.0
            large_diff_count = large_diff_mask.sum()

            print(f"\n[大差异分析 (差异>1.0)]")
            print(f"  大差异行数: {large_diff_count} ({large_diff_count/len(diff)*100:.2f}%)")

            if large_diff_count > 0:
                print(f"  最大差异: {abs_diff.max():.6f}")
                print(f"  平均差异: {abs_diff.mean():.6f}")

                # 分析大差异的模式
                print(f"\n  大差异行的值分布:")

                # 检查文件1的值分布
                val1_large_diff = col1[large_diff_mask]
                print(f"    文件1的值:")
                print(f"      均值: {val1_large_diff.mean():.6f}, 标准差: {val1_large_diff.std():.6f}")
                val1_counts = val1_large_diff.value_counts()
                for val, count in val1_counts.head(10).items():
                    pct = count / large_diff_count * 100
                    print(f"        值={val:.6f}: {count}次 ({pct:.2f}%)")

                # 检查文件2的值分布
                val2_large_diff = col2[large_diff_mask]
                print(f"    文件2的值:")
                print(f"      均值: {val2_large_diff.mean():.6f}, 标准差: {val2_large_diff.std():.6f}")
                val2_counts = val2_large_diff.value_counts()
                for val, count in val2_counts.head(10).items():
                    pct = count / large_diff_count * 100
                    print(f"        值={val:.6f}: {count}次 ({pct:.2f}%)")

                # 差异分布
                print(f"    差异值分布:")
                diff_counts = diff[large_diff_mask].value_counts()
                for val, count in diff_counts.head(10).items():
                    pct = count / large_diff_count * 100
                    print(f"      差异={val:.6f}: {count}次 ({pct:.2f}%)")

                # 检查是否与某些股票或日期相关
                print(f"\n  大差异行的股票分布:")
                stock_codes = merged.loc[large_diff_mask, 'stock_code_file1']
                for stock, count in stock_codes.value_counts().head(10).items():
                    pct = count / large_diff_count * 100
                    print(f"    {stock}: {count}次 ({pct:.2f}%)")

                print(f"\n  大差异行的日期分布:")
                dates = merged.loc[large_diff_mask, 'date_file1']
                for date, count in dates.value_counts().head(10).items():
                    pct = count / large_diff_count * 100
                    print(f"    {date}: {count}次 ({pct:.2f}%)")

            # 检查是否有固定模式
            print(f"\n[模式检查]")

            # 检查是否是简单的缩放或平移
            valid_both = ~(col1.isna() | col2.isna())
            if valid_both.sum() > 100:
                # 计算相关性
                correlation = col1[valid_both].corr(col2[valid_both])
                print(f"  相关系数: {correlation:.6f}")

                # 尝试线性关系: col2 = a * col1 + b
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    col1[valid_both].dropna(),
                    col2[valid_both].dropna()
                )
                print(f"  线性拟合: col2 = {slope:.6f} * col1 + {intercept:.6f}")
                print(f"  R^2: {r_value**2:.6f}")

                # 计算拟合误差
                predicted = slope * col1[valid_both] + intercept
                residual = col2[valid_both] - predicted
                print(f"  拟合误差: 均值={residual.mean():.6f}, 标准差={residual.std():.6f}")

        # 检查是否有系统性偏移
        print("\n" + "=" * 100)
        print("[系统性差异检查]")
        print("=" * 100)

        # 检查所有数值列的差异
        numeric_cols = [col for col in common_cols if col not in ['stock_code', 'date']]

        col_diff_summary = []
        for col in numeric_cols:
            if f'{col}_file1' in merged.columns and f'{col}_file2' in merged.columns:
                data1 = pd.to_numeric(merged[f'{col}_file1'], errors='coerce')
                data2 = pd.to_numeric(merged[f'{col}_file2'], errors='coerce')

                valid = ~(data1.isna() | data2.isna())
                if valid.sum() > 0:
                    diff = np.abs(data2 - data1)
                    max_diff = diff[valid].max()
                    mean_diff = diff[valid].mean()
                    pct_diff = (diff[valid] > 0.01).sum() / valid.sum() * 100

                    col_diff_summary.append({
                        'column': col,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'pct_diff': pct_diff
                    })

        # 找出差异最大的列
        col_diff_summary.sort(key=lambda x: x['pct_diff'], reverse=True)

        print(f"\n差异>5%的列 (按差异行数比例排序):")
        for item in col_diff_summary[:20]:
            if item['pct_diff'] > 5.0:
                print(f"  {item['column']:<50} 最大差异={item['max_diff']:>10.6f}, 平均差异={item['mean_diff']:>10.6f}, 差异>0.01占比={item['pct_diff']:>6.2f}%")

    print("\n[分析完成]")

if __name__ == "__main__":
    # 文件路径
    base_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\factor_lab_data")
    file1 = base_path / "factors - 副本.parquet"
    file2 = base_path / "factors.parquet"

    deep_diagnose_factors(file1, file2)
