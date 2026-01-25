#!/usr/bin/env python3
"""
使用正确路径测试2023-06-01日MA120计算可行性
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def test_ma120_with_correct_path():
    """使用正确路径测试MA120计算可行性"""
    print("=" * 80)
    print("测试2023-06-01日MA120计算可行性（使用正确路径）")
    print("=" * 80)

    try:
        # 使用正确路径
        correct_base_path = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily")
        print(f"检查路径: {correct_base_path}")

        if not correct_base_path.exists():
            print(f"[错误] 路径不存在: {correct_base_path}")
            return False

        print(f"[成功] 找到数据目录: {correct_base_path}")

        # 查找2023年6月的数据文件（扁平结构）
        june_files = []
        for parquet_file in correct_base_path.glob("*.parquet"):
            if "202306" in parquet_file.name:
                june_files.append(parquet_file)

        print(f"找到2023年6月数据文件: {len(june_files)} 个")

        if not june_files:
            print("[错误] 没有找到2023年6月的数据文件")
            return False

        # 显示找到的文件
        for jf in sorted(june_files)[:10]:  # 显示前10个
            print(f"  - {jf.name}")

        # 查找2023-06-01的数据文件
        target_file = None
        for jf in june_files:
            if "20230601" in jf.name:
                target_file = jf
                break

        if target_file:
            print(f"\n[成功] 找到2023-06-01数据文件: {target_file.name}")
        else:
            # 如果没有找到确切的日期，找最近的文件
            target_file = sorted(june_files)[0]
            print(f"\n[信息] 没有找到2023-06-01，使用最近的文件: {target_file.name}")

        # 加载足够的历史数据
        try:
            # 加载多个文件以获得足够的历史数据
            df = load_multiple_files_for_ma120(correct_base_path)
            if df is None:
                return False

            print(f"列名: {list(df.columns)[:15]}...")  # 显示前15个列名

            # 检查日期信息
            if 'date' in df.columns:
                unique_dates = df['date'].unique()
                print(f"数据日期: {len(unique_dates)} 个不同日期")
                if len(unique_dates) <= 20:
                    print(f"具体日期: {sorted(unique_dates)}")

                # 检查股票数量
                if 'stock_code' in df.columns:
                    unique_stocks = df['stock_code'].unique()
                    print(f"股票数量: {len(unique_stocks)} 只")

                    return analyze_ma120_availability(df, target_file)
            else:
                print("[警告] 数据中没有date列")
                return False

        except Exception as e:
            print(f"[错误] 加载数据失败: {e}")
            return False

    except Exception as e:
        print(f"[错误] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_multiple_files_for_ma120(correct_base_path, target_date_str="20230601"):
    """加载多个文件以获得足够的历史数据来计算MA120"""
    print(f"\n加载2023-06-01前120天的历史数据...")

    target_date = pd.to_datetime(target_date_str)
    start_date = target_date - pd.DateOffset(days=150)  # 多加载一些确保有足够数据
    start_date_str = start_date.strftime('%Y%m%d')

    all_files = []
    for parquet_file in correct_base_path.glob("*.parquet"):
        file_date_str = parquet_file.stem
        if start_date_str <= file_date_str <= target_date_str:
            all_files.append(parquet_file)

    all_files.sort()
    print(f"找到历史数据文件: {len(all_files)} 个")

    if not all_files:
        print(f"[错误] 没有找到 {start_date_str} 到 {target_date_str} 期间的数据文件")
        return None

    # 读取并合并所有数据
    all_data = []
    for i, file_path in enumerate(all_files):
        if i % 20 == 0:
            print(f"读取进度: {i+1}/{len(all_files)}")
        try:
            df = pd.read_parquet(file_path)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"[警告] 读取文件 {file_path.name} 失败: {e}")

    if not all_data:
        print("[错误] 所有文件读取失败")
        return None

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"[成功] 合并数据: {combined_data.shape}")
    return combined_data


def analyze_ma120_availability(df, file_path):
    """分析MA120可用性"""
    print(f"\n" + "=" * 60)
    print("分析MA120计算可用性")
    print("=" * 60)

    try:
        # 检查是否有收盘价数据
        if 'close' not in df.columns:
            print("[错误] 数据中没有close列")
            return False

        print(f"[成功] 数据包含close列，可用于计算MA")

        # 分析日期分布
        if 'date' in df.columns and 'stock_code' in df.columns:
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            target_date = pd.to_datetime("2023-06-01")

            # 检查目标日期的数据
            target_data = df[df['date'] == target_date]
            if target_data.empty:
                print(f"[错误] 2023-06-01没有数据")
                return False

            print(f"[成功] 目标日期 {target_date.date()} 有 {len(target_data)} 只股票数据")
            print(f"数据时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

            # 分析历史数据可用性
            print(f"\n分析各股票的历史数据...")
            results = []

            for stock_code in target_data['stock_code'].unique()[:10]:  # 分析前10只股票
                stock_all_data = df[df['stock_code'] == stock_code].sort_values('date')
                stock_historical = stock_all_data[stock_all_data['date'] <= target_date]

                data_days = len(stock_historical)
                can_calculate_ma120 = data_days >= 120

                results.append({
                    'stock_code': stock_code,
                    'data_days': data_days,
                    'can_calculate_ma120': can_calculate_ma120
                })

            if results:
                results_df = pd.DataFrame(results)
                print(f"\n分析结果 ({len(results_df)} 只股票):")
                print(f"能计算MA120的股票: {results_df['can_calculate_ma120'].sum()} 只")
                print(f"不能计算MA120的股票: {(~results_df['can_calculate_ma120']).sum()} 只")

                if results_df['can_calculate_ma120'].sum() > 0:
                    print(f"\n[成功] 有股票可以计算MA120!")
                    can_calculate = results_df[results_df['can_calculate_ma120']].sort_values('data_days', ascending=False).head()
                    print("历史数据最多的股票:")
                    for _, row in can_calculate.iterrows():
                        print(f"  {row['stock_code']}: {row['data_days']} 天历史数据")

                    # 实际计算MA120示例
                    print(f"\n实际计算MA120示例:")
                    sample_stock = can_calculate.iloc[0]['stock_code']
                    stock_historical = df[(df['stock_code'] == sample_stock) & (df['date'] <= target_date)].sort_values('date')

                    if len(stock_historical) >= 120:
                        close_prices = stock_historical['close'].values
                        ma120_value = close_prices[-120:].mean()
                        target_close = stock_historical[stock_historical['date'] == target_date]['close'].iloc[0]
                        ratio = target_close / ma120_value if ma120_value > 0 else 0

                        print(f"  股票: {sample_stock}")
                        print(f"  历史数据: {len(stock_historical)} 天")
                        print(f"  MA120值: {ma120_value:.4f}")
                        print(f"  当日收盘价: {target_close:.4f}")
                        print(f"  价格/MA120比例: {ratio:.4f}")

                        print(f"\n[成功] MA120计算验证通过!")
                        return True
                    else:
                        print(f"[错误] 选中股票历史数据不足")
                        return False
                else:
                    print(f"\n[失败] 没有股票有足够的历史数据计算MA120")
                    print("历史数据最多的股票:")
                    cannot_calculate = results_df.sort_values('data_days', ascending=False).head(5)
                    for _, row in cannot_calculate.iterrows():
                        print(f"  {row['stock_code']}: {row['data_days']} 天 (需要120天)")
                    return False

        return False

    except Exception as e:
        print(f"[错误] 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试2023-06-01日MA120计算可行性（使用正确路径）...\n")

    success = test_ma120_with_correct_path()

    print("\n" + "=" * 80)
    print("测试结果总结:")
    print("=" * 80)

    if success:
        print("[成功] 2023-06-01日的MA120可以正常计算")
        print("[成功] 有足够的历史数据支持")
        print("[成功] ma_convergence_score因子应该能正常工作")
        print("\n结论: ma_convergence_score_z常数列问题可能不是由于MA120数据不足造成的")
        print("建议检查其他可能原因，如:")
        print("1. 数据质量问题")
        print("2. 计算逻辑问题")
        print("3. 市场环境因素")
    else:
        print("[失败] 2023-06-01日的MA120无法计算")
        print("[失败] 历史数据不足")
        print("[失败] 这可能是ma_convergence_score常数列问题的主要原因")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)