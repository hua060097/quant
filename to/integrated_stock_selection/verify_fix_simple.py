#!/usr/bin/env python3
"""
验证修复效果：比较不同end_date生成的因子文件
验证成交量过滤修复是否解决了因子不一致的问题
"""

import pandas as pd
import numpy as np

def compare_factors_between_files():
    """比较factors.parquet和factors1.parquet中的三个特定因子"""

    print("=" * 80)
    print("验证修复效果：比较不同end_date生成的因子文件")
    print("=" * 80)

    # 要比较的三个因子
    target_factors = ['support_distance', 'idiosyncratic_volatility_sector', 'sector_correlation_20d']

    # 读取两个文件
    try:
        df0 = pd.read_parquet('factor_lab_data/factors.parquet')
        df1 = pd.read_parquet('factor_lab_data/factors1.parquet')
        print(f"成功读取文件:")
        print(f"   factors.parquet: {len(df0):,}条记录")
        print(f"   factors1.parquet: {len(df1):,}条记录")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return False

    # 转换日期列
    df0['date'] = pd.to_datetime(df0['date'])
    df1['date'] = pd.to_datetime(df1['date'])

    # 检查时间范围
    print(f"\n时间范围:")
    print(f"   factors.parquet: {df0['date'].min()} ~ {df0['date'].max()}")
    print(f"   factors1.parquet: {df1['date'].min()} ~ {df1['date'].max()}")

    # 获取共同的时间范围（应该是2023-06-01到2023-09-30）
    start_date = pd.to_datetime('2023-06-01')
    end_date = pd.to_datetime('2023-09-30')

    df0_common = df0[(df0['date'] >= start_date) & (df0['date'] <= end_date)].copy()
    df1_common = df1[(df1['date'] >= start_date) & (df1['date'] <= end_date)].copy()

    print(f"\n共同时间范围 ({start_date.date()} ~ {end_date.date()}):")
    print(f"   factors.parquet: {len(df0_common):,}条记录")
    print(f"   factors1.parquet: {len(df1_common):,}条记录")

    if len(df0_common) != len(df1_common):
        print(f"警告：两个文件在共同时间范围内的记录数不一致")
        print(f"   可能存在问题，继续分析...")

    # 检查是否包含所有目标因子
    missing_factors_0 = [f for f in target_factors if f not in df0_common.columns]
    missing_factors_1 = [f for f in target_factors if f not in df1_common.columns]

    if missing_factors_0:
        print(f"factors.parquet缺少因子: {missing_factors_0}")
        return False

    if missing_factors_1:
        print(f"factors1.parquet缺少因子: {missing_factors_1}")
        return False

    print(f"所有目标因子都存在: {target_factors}")

    # 按日期和股票对齐数据
    df0_sorted = df0_common.sort_values(['date', 'stock_code']).reset_index(drop=True)
    df1_sorted = df1_common.sort_values(['date', 'stock_code']).reset_index(drop=True)

    # 确保数据完全对齐
    min_len = min(len(df0_sorted), len(df1_sorted))
    df0_aligned = df0_sorted.head(min_len)
    df1_aligned = df1_sorted.head(min_len)

    # 检查日期和股票代码是否完全对齐
    date_match = (df0_aligned['date'] == df1_aligned['date']).all()
    stock_match = (df0_aligned['stock_code'] == df1_aligned['stock_code']).all()

    if not date_match or not stock_match:
        print(f"数据对齐失败:")
        print(f"   日期对齐: {date_match}")
        print(f"   股票对齐: {stock_match}")
        return False

    print(f"数据完全对齐: {len(df0_aligned):,}条记录")

    # 逐个比较因子
    all_factors_identical = True

    print(f"\n开始比较因子值:")
    print("-" * 60)

    for factor_name in target_factors:
        print(f"\n比较因子: {factor_name}")

        # 提取因子值
        values0 = df0_aligned[factor_name].values
        values1 = df1_aligned[factor_name].values

        # 计算差异
        abs_diff = np.abs(values0 - values1)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        median_diff = np.median(abs_diff)

        # 统计完全相同的数量
        exact_match = np.sum(abs_diff < 1e-12)

        # 统计不同等级的差异
        tiny_diff = np.sum((abs_diff >= 1e-12) & (abs_diff < 1e-6))
        small_diff = np.sum((abs_diff >= 1e-6) & (abs_diff < 1e-3))
        medium_diff = np.sum((abs_diff >= 1e-3) & (abs_diff < 1e-1))
        large_diff = np.sum(abs_diff >= 1e-1)

        print(f"   最大差异: {max_diff:.12f}")
        print(f"   平均差异: {mean_diff:.12f}")
        print(f"   中位数差异: {median_diff:.12f}")
        print(f"   完全相同: {exact_match:,}/{len(values0):,} ({exact_match/len(values0)*100:.6f}%)")

        if max_diff < 1e-12:
            print(f"   SUCCESS: {factor_name}: 完全一致!")
        elif max_diff < 1e-6:
            print(f"   WARNING: {factor_name}: 基本一致(仅有浮点精度差异)")
            all_factors_identical = False
        else:
            print(f"   ERROR: {factor_name}: 存在显著差异!")
            all_factors_identical = False

            # 显示差异最大的几个例子
            if large_diff > 0 or medium_diff > 0:
                significant_diff_indices = np.where(abs_diff > 1e-3)[0]
                if len(significant_diff_indices) > 0:
                    print(f"   差异最大的前3个例子:")
                    for idx in significant_diff_indices[:3]:
                        date = df0_aligned.iloc[idx]['date']
                        stock = df0_aligned.iloc[idx]['stock_code']
                        val0 = values0[idx]
                        val1 = values1[idx]
                        print(f"     {date.strftime('%Y-%m-%d')} {stock}: {val0:.6f} vs {val1:.6f} (差异: {abs(val0-val1):.6f})")

    # 总体结论
    print(f"\n" + "=" * 80)
    print("修复效果验证结果")
    print("=" * 80)

    if all_factors_identical:
        print("恭喜! 修复成功!")
        print("SUCCESS: 所有三个因子在两个文件中完全一致")
        print("SUCCESS: 成交量过滤修复已解决end_date导致的因子不一致问题")
        print()
        print("验证详情:")
        print(f"   - 比较时间范围: {start_date.date()} ~ {end_date.date()}")
        print(f"   - 比较记录数: {len(df0_aligned):,}条")
        print(f"   - 比较因子: {len(target_factors)}个")
        print(f"   - 最大差异: < 1e-12 (完全一致)")
        print()
        print("修复内容:")
        print("   - 在_filter_new_stocks_batch中添加了成交量过滤")
        print("   - 移除了所有volume=0的无效数据记录")
        print("   - 确保不同end_date设置下股票池保持一致")
        return True
    else:
        print("修复未完全成功")
        print("WARNING: 仍有部分因子存在差异")
        print("可能的原因:")
        print("   - 修复不完整，还有其他影响因子计算的因素")
        print("   - 数据缓存问题，需要清理缓存重新生成")
        print("   - 其他代码路径仍然存在问题")
        return False

def check_cache_consistency():
    """检查缓存一致性"""
    print(f"\n检查缓存一致性...")

    cache_files = [
        'factor_lab_data/.cache',
        'factor_lab_data/cache'
    ]

    for cache_dir in cache_files:
        try:
            from pathlib import Path
            cache_path = Path(cache_dir)
            if cache_path.exists():
                file_count = len(list(cache_path.rglob('*')))
                print(f"   发现缓存目录: {cache_dir} ({file_count}个文件)")
            else:
                print(f"   缓存目录不存在: {cache_dir}")
        except Exception as e:
            print(f"   检查缓存失败: {e}")

def main():
    """主函数"""
    print("开始验证修复效果...")

    # 检查缓存
    check_cache_consistency()

    # 比较因子
    success = compare_factors_between_files()

    if success:
        print(f"\n建议:")
        print(f"   修复成功！可以继续使用当前的因子生成流程。")
        print(f"   建议定期验证因子一致性以确保数据质量。")
    else:
        print(f"\n建议:")
        print(f"   1. 清理缓存文件后重新运行因子生成")
        print(f"   2. 检查是否还有其他影响因子一致性的代码路径")
        print(f"   3. 继续调试以找到并修复剩余问题")

if __name__ == "__main__":
    main()