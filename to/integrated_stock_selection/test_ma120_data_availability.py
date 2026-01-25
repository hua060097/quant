#!/usr/bin/env python3
"""
测试2023-06-01日的MA120计算可行性
检查是否有足够的历史数据来计算120日移动平均
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_ma120_availability():
    """测试MA120数据的可用性"""
    print("=" * 80)
    print("测试2023-06-01日MA120计算可行性")
    print("=" * 80)

    try:
        # 尝试不同的数据加载方式
        data_loaded = False
        stock_data = None

        # 方法1: 尝试使用data_loaders
        try:
            from data_loaders import FeatureDataLoader
            loader = FeatureDataLoader()

            print("方法1: 使用FeatureDataLoader加载数据...")
            # 计算2023-06-01往前推130天，确保有足够的历史数据
            test_date = datetime(2023, 6, 1)
            start_date = (test_date - timedelta(days=130)).strftime('%Y-%m-%d')
            end_date = test_date.strftime('%Y-%m-%d')

            print(f"时间范围: {start_date} ~ {end_date}")

            stock_data = loader.load_stock_data(
                start_date=start_date,
                end_date=end_date,
                columns=['stock_code', 'date', 'close']
            )

            if not stock_data.empty:
                print(f"✅ 成功加载数据: {stock_data.shape}")
                data_loaded = True
            else:
                print("❌ 数据为空")

        except ImportError as e:
            print(f"[错误] 无法导入data_loaders: {e}")
        except Exception as e:
            print(f"[错误] 加载数据失败: {e}")

        # 方法2: 尝试直接读取parquet文件
        if not data_loaded:
            try:
                print("\n方法2: 直接读取parquet文件...")

                # 查找数据文件
                local_data_path = Path("local_data")
                if local_data_path.exists():
                    print(f"找到local_data目录: {local_data_path}")

                    # 查找daily数据
                    daily_path = local_data_path / "features" / "daily"
                    if daily_path.exists():
                        print(f"找到daily目录: {daily_path}")

                        # 列出所有年份的parquet文件
                        parquet_files = []
                        for year_dir in daily_path.iterdir():
                            if year_dir.is_dir():
                                for parquet_file in year_dir.glob("*.parquet"):
                                    parquet_files.append(parquet_file)

                        if parquet_files:
                            print(f"找到 {len(parquet_files)} 个parquet文件")
                            # 按时间排序
                            parquet_files.sort()

                            # 读取2023-06-01前后几个月的数据
                            all_data = []
                            target_date = pd.to_datetime("2023-06-01")
                            start_date = target_date - pd.DateOffset(months=6)
                            end_date = target_date + pd.DateOffset(days=10)

                            for parquet_file in parquet_files:
                                # 解析文件名中的日期
                                try:
                                    file_date_str = parquet_file.stem
                                    file_date = pd.to_datetime(file_date_str)
                                    if start_date <= file_date <= end_date:
                                        print(f"读取文件: {parquet_file.name}")
                                        df = pd.read_parquet(parquet_file)
                                        if not df.empty:
                                            all_data.append(df)
                                except:
                                    continue

                            if all_data:
                                stock_data = pd.concat(all_data, ignore_index=True)
                                print(f"[成功] 合并数据: {stock_data.shape}")
                                data_loaded = True
                            else:
                                print("[错误] 没有找到目标时间段的数据")
                        else:
                            print("[错误] 没有找到parquet文件")
                    else:
                        print("[错误] 没有找到daily目录")
                else:
                    print("[错误] 没有找到local_data目录")

            except Exception as e:
                print(f"[错误] 读取parquet文件失败: {e}")

        if not data_loaded or stock_data is None:
            print("\n[错误] 无法获取数据进行测试")
            return False

        # 分析数据可用性
        print("\n" + "=" * 60)
        print("分析数据可用性")
        print("=" * 60)

        # 检查数据时间范围
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        min_date = stock_data['date'].min()
        max_date = stock_data['date'].max()
        total_days = (max_date - min_date).days

        print(f"数据时间范围: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        print(f"总天数: {total_days}")

        # 检查2023-06-01是否在数据范围内
        target_date = pd.to_datetime("2023-06-01")
        is_in_range = min_date <= target_date <= max_date
        print(f"2023-06-01在数据范围内: {'是' if is_in_range else '否'}")

        if not is_in_range:
            print(f"[错误] 数据不包含2023-06-01，无法测试MA120")
            return False

        # 计算每只股票在2023-06-01的历史数据天数
        print(f"\n分析各股票在2023-06-01的历史数据可用性...")

        # 计算每只股票在2023-06-01之前的数据天数
        results = []
        for stock_code in stock_data['stock_code'].unique():
            stock_data_subset = stock_data[stock_data['stock_code'] == stock_code].sort_values('date')

            # 找到2023-06-01的索引
            target_idx = stock_data_subset[stock_data_subset['date'] == target_date].index
            if len(target_idx) == 0:
                continue

            target_row_idx = target_idx[0]
            # 计算这只股票在2023-06-01及之前的数据行数
            historical_data = stock_data_subset[stock_data_subset.index <= target_row_idx]
            data_days = len(historical_data)

            results.append({
                'stock_code': stock_code,
                'data_days': data_days,
                'can_calculate_ma120': data_days >= 120
            })

        if results:
            results_df = pd.DataFrame(results)
            print(f"\n统计结果 (共 {len(results_df)} 只股票):")
            print(f"能计算MA120的股票: {results_df['can_calculate_ma120'].sum()} 只")
            print(f"不能计算MA120的股票: {(~results_df['can_calculate_ma120']).sum()} 只")

            if results_df['can_calculate_ma120'].sum() > 0:
                print(f"\n[成功] 有股票可以计算MA120!")
                # 显示前5只可以计算MA120的股票
                can_calculate = results_df[results_df['can_calculate_ma120']].sort_values('data_days', ascending=False).head()
                print("前5只历史数据最多的股票:")
                for _, row in can_calculate.iterrows():
                    print(f"  {row['stock_code']}: {row['data_days']} 天历史数据")
            else:
                print(f"\n[失败] 没有股票有足够的历史数据计算MA120")
                # 显示历史数据最多的几只股票
                cannot_calculate = results_df.sort_values('data_days', ascending=False).head(5)
                print("历史数据最多的5只股票:")
                for _, row in cannot_calculate.iterrows():
                    print(f"  {row['stock_code']}: {row['data_days']} 天历史数据 (需要120天)")

            # 测试实际的MA120计算
            print(f"\n测试实际MA120计算...")
            return test_actual_ma120_calculation(stock_data, target_date)

        else:
            print("[错误] 没有找到在2023-06-01有数据的股票")
            return False

    except Exception as e:
        print(f"[错误] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actual_ma120_calculation(stock_data, target_date):
    """测试实际的MA120计算"""
    try:
        # 选择几只有足够数据的股票进行测试
        target_date_str = target_date.strftime('%Y-%m-%d')
        print(f"测试日期: {target_date_str}")

        # 获取在目标日期有数据的股票
        target_date_data = stock_data[stock_data['date'] == target_date].copy()
        if target_date_data.empty:
            print(f"[错误] {target_date_str} 没有股票数据")
            return False

        # 获取目标日期之前的历史数据
        historical_data = stock_data[stock_data['date'] <= target_date].copy()

        # 按股票分组计算MA120
        print(f"\n计算MA120...")
        results = []

        for stock_code in target_date_data['stock_code'].unique()[:5]:  # 测试前5只股票
            stock_historical = historical_data[historical_data['stock_code'] == stock_code].sort_values('date')

            if len(stock_historical) < 120:
                print(f"[失败] {stock_code}: 只有 {len(stock_historical)} 天数据，不足120天")
                continue

            # 计算MA120
            close_prices = stock_historical['close'].values
            ma120_value = close_prices[-120:].mean()  # 最后120天的均值

            # 获取目标日期的收盘价
            target_row = target_date_data[target_date_data['stock_code'] == stock_code]
            if target_row.empty:
                continue

            target_close = target_row['close'].iloc[0]

            results.append({
                'stock_code': stock_code,
                'historical_days': len(stock_historical),
                'ma120_value': round(ma120_value, 4),
                'target_close': round(target_close, 4),
                'ma_price_ratio': round(target_close / ma120_value, 4) if ma120_value > 0 else None
            })

        if results:
            print(f"\n[成功] MA120计算结果:")
            for result in results:
                print(f"  {result['stock_code']}:")
                print(f"    历史数据: {result['historical_days']} 天")
                print(f"    MA120值: {result['ma120_value']}")
                print(f"    目标日收盘价: {result['target_close']}")
                print(f"    价格/MA120比例: {result['ma_price_ratio']}")

            print(f"\n[成功] MA120可以正常计算!")
            return True
        else:
            print(f"[失败] 没有股票有足够数据计算MA120")
            return False

    except Exception as e:
        print(f"[错误] MA120计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试2023-06-01日MA120计算可行性...\n")

    success = test_ma120_availability()

    print("\n" + "=" * 80)
    print("测试结果总结:")
    print("=" * 80)

    if success:
        print("[成功] 2023-06-01日的MA120可以正常计算")
        print("[成功] 有足够的历史数据支持")
        print("[成功] ma_convergence_score因子应该能正常工作")
    else:
        print("[失败] 2023-06-01日的MA120无法计算")
        print("[失败] 历史数据不足")
        print("[失败] 这可能导致ma_convergence_score因子在某些时期变为常数列")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)