#!/usr/bin/env python3
"""
测试增量更新逻辑

重现incremental_save没有生效的问题
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_incremental_merge_logic():
    """测试增量合并逻辑"""
    print("=" * 80)
    print("测试增量更新合并逻辑")
    print("=" * 80)

    # 模拟第一次运行的数据（2023-06-01 到 2023-06-19）
    print("\n1. 模拟第一次运行数据:")
    first_run_data = pd.DataFrame({
        'date': pd.date_range('2023-06-01', '2023-06-19', freq='D'),
        'stock_code': '000001.SZ',
        'factor1': np.random.randn(14),
        'factor2': np.random.randn(14)
    })
    # 移除周末
    first_run_data = first_run_data[first_run_data['date'].dt.weekday < 5]
    print(f"   第一次运行: {len(first_run_data)}条记录")
    print(f"   日期范围: {first_run_data['date'].min()} ~ {first_run_data['date'].max()}")

    # 模拟第二次运行的数据（2023-06-20 到 2023-06-30）
    print("\n2. 模拟第二次运行数据:")
    second_run_data = pd.DataFrame({
        'date': pd.date_range('2023-06-20', '2023-06-30', freq='D'),
        'stock_code': '000001.SZ',
        'factor1': np.random.randn(11),
        'factor2': np.random.randn(11)
    })
    # 移除周末和端午节
    second_run_data = second_run_data[second_run_data['date'].dt.weekday < 5]
    second_run_data = second_run_data[~second_run_data['date'].isin(['2023-06-22', '2023-06-23'])]
    print(f"   第二次运行: {len(second_run_data)}条记录")
    print(f"   日期范围: {second_run_data['date'].min()} ~ {second_run_data['date'].max()}")

    # 测试智能合并逻辑
    print("\n3. 测试智能合并逻辑:")

    # 导入智能合并函数
    from factor_system.factor_lab.store import _smart_merge_dataframes

    # 执行合并
    try:
        merged_data = _smart_merge_dataframes(
            existing_data=first_run_data,
            new_data=second_run_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        print(f"   合并成功!")
        print(f"   合并后数据: {len(merged_data)}条记录")
        print(f"   日期范围: {merged_data['date'].min()} ~ {merged_data['date'].max()}")

        # 检查日期连续性
        expected_dates = set(pd.date_range('2023-06-01', '2023-06-30', freq='D'))
        expected_dates = {d for d in expected_dates if d.weekday() < 5 and d.strftime('%Y-%m-%d') not in ['2023-06-22', '2023-06-23']}
        actual_dates = set(merged_data['date'])

        missing_dates = expected_dates - actual_dates
        extra_dates = actual_dates - expected_dates

        print(f"\n4. 日期连续性检查:")
        print(f"   预期日期数: {len(expected_dates)}")
        print(f"   实际日期数: {len(actual_dates)}")

        if missing_dates:
            print(f"   缺失日期: {sorted(missing_dates)}")
        if extra_dates:
            print(f"   多余日期: {sorted(extra_dates)}")

        if not missing_dates and not extra_dates:
            print("   ✅ 日期完全连续")
        else:
            print("   ❌ 日期不连续")

    except Exception as e:
        print(f"   ❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()

def test_actual_file_analysis():
    """分析实际的factors.parquet文件"""
    print("\n" + "=" * 80)
    print("分析实际factors.parquet文件")
    print("=" * 80)

    factors_file = Path("factor_lab_data/factors.parquet")
    if not factors_file.exists():
        print("❌ factors.parquet文件不存在")
        return

    try:
        df = pd.read_parquet(factors_file)
        df['date'] = pd.to_datetime(df['date'])

        print(f"\n文件基本信息:")
        print(f"   总记录数: {len(df):,}")
        print(f"   股票数: {df['stock_code'].nunique()}")
        print(f"   日期数: {df['date'].nunique()}")
        print(f"   日期范围: {df['date'].min()} ~ {df['date'].max()}")

        # 按日期统计
        print(f"\n按日期统计记录数:")
        date_counts = df.groupby('date').size().sort_index()

        for date, count in date_counts.items():
            date_str = date.strftime('%Y-%m-%d')
            weekday = date.strftime('%A')
            print(f"   {date_str} ({weekday[:3]}): {count:4d}条记录")

        # 检查缺失的日期
        print(f"\n检查日期连续性:")
        all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        trading_days = [d for d in all_dates if d.weekday() < 5]

        # 移除已知节假日
        holidays = ['2023-06-22', '2023-06-23']  # 端午节
        trading_days = [d for d in trading_days if d.strftime('%Y-%m-%d') not in holidays]

        expected_dates = set(trading_days)
        actual_dates = set(df['date'].unique())

        missing_dates = expected_dates - actual_dates

        print(f"   预期交易日: {len(expected_dates)}天")
        print(f"   实际有数据: {len(actual_dates)}天")

        if missing_dates:
            print(f"   缺失日期:")
            for date in sorted(missing_dates):
                date_str = date.strftime('%Y-%m-%d')
                weekday = date.strftime('%A')
                print(f"     {date_str} ({weekday[:3]}) - 应该有数据但没有")
        else:
            print("   ✅ 日期连续性良好")

        # 分析问题
        print(f"\n=== 问题分析 ===")

        # 检查是否有6月1-19日的数据
        early_dates = df[df['date'] <= '2023-06-19']
        if len(early_dates) == 0:
            print("❌ 关键问题: 没有发现2023-06-01到2023-06-19的数据!")
            print("   这说明第一次运行的数据在增量更新时丢失了")
        else:
            print(f"✅ 发现早期数据: {len(early_dates)}条记录")
            print(f"   早期日期范围: {early_dates['date'].min()} ~ {early_dates['date'].max()}")

        # 检查6月20日之后的数据
        late_dates = df[df['date'] >= '2023-06-20']
        if len(late_dates) > 0:
            print(f"✅ 发现新数据: {len(late_dates)}条记录")
            print(f"   新日期范围: {late_dates['date'].min()} ~ {late_dates['date'].max()}")
        else:
            print("❌ 没有发现2023-06-20之后的数据")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def simulate_incremental_save_flow():
    """模拟incremental_save的完整流程"""
    print("\n" + "=" * 80)
    print("模拟incremental_save完整流程")
    print("=" * 80)

    print("\n=== 第一次运行 (2023-06-01 到 2023-06-19) ===")

    # 模拟第一次运行生成并保存数据
    first_data = pd.DataFrame({
        'date': pd.date_range('2023-06-01', '2023-06-19', freq='D'),
        'stock_code': 'TEST001',
        'pct_chg_1d': np.random.randn(14) * 0.02,  # 正常涨跌幅
        'factor1': np.random.randn(14)
    })
    first_data = first_data[first_data['date'].dt.weekday < 5]  # 移除周末

    print(f"生成数据: {len(first_data)}条记录")
    print(f"日期范围: {first_data['date'].min()} ~ {first_data['date'].max()}")

    # 检查涨跌幅是否正常
    abnormal_pct = first_data[abs(first_data['pct_chg_1d']) > 0.2]
    print(f"异常涨跌幅(>20%): {len(abnormal_pct)}条")

    print("\n=== 第二次运行 (2023-06-20 到 2023-06-30) ===")

    # 模拟第二次运行，incremental_save=True
    second_data = pd.DataFrame({
        'date': pd.date_range('2023-06-20', '2023-06-30', freq='D'),
        'stock_code': 'TEST001',
        'pct_chg_1d': np.random.randn(11) * 5.0,  # 异常涨跌幅
        'factor1': np.random.randn(11)
    })
    second_data = second_data[second_data['date'].dt.weekday < 5]
    second_data = second_data[~second_data['date'].isin(['2023-06-22', '2023-06-23'])]

    print(f"生成数据: {len(second_data)}条记录")
    print(f"日期范围: {second_data['date'].min()} ~ {second_data['date'].max()}")

    # 检查涨跌幅是否正常
    abnormal_pct = second_data[abs(second_data['pct_chg_1d']) > 0.2]
    print(f"异常涨跌幅(>20%): {len(abnormal_pct)}条")
    print(f"最严重的涨跌幅: {second_data['pct_chg_1d'].abs().max():.2%}")

    print("\n=== 模拟增量合并 ===")

    try:
        from factor_system.factor_lab.store import _smart_merge_dataframes

        # 模拟增量合并
        merged = _smart_merge_dataframes(
            existing_data=first_data,
            new_data=second_data,
            index_columns=['date', 'stock_code'],
            overlap_mode='merge'
        )

        print(f"合并成功: {len(merged)}条记录")
        print(f"最终日期范围: {merged['date'].min()} ~ {merged['date'].max()}")

        # 检查数据完整性
        early_count = len(merged[merged['date'] <= '2023-06-19'])
        late_count = len(merged[merged['date'] >= '2023-06-20'])

        print(f"早期数据保留: {early_count}条")
        print(f"新增数据: {late_count}条")

        if early_count == 0:
            print("❌ 问题重现: 早期数据丢失!")
            return False
        else:
            print("✅ 增量合并正常")
            return True

    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return False

if __name__ == "__main__":
    try:
        # 测试合并逻辑
        test_incremental_merge_logic()

        # 分析实际文件
        test_actual_file_analysis()

        # 模拟完整流程
        simulate_incremental_save_flow()

        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()