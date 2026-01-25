#!/usr/bin/env python3
"""
检查本地数据文件中的股票复权信息

从本地保存的parquet文件中读取数据，检查不同复权类型的差异
"""

import pandas as pd
from pathlib import Path


def check_local_dividend_data(
    stock_list=None,
    dates=None
):
    """
    从本地数据文件检查复权信息

    Args:
        stock_list: 股票代码列表
        dates: 日期列表 ['20250605', '20250606']
    """
    if stock_list is None:
        stock_list = ['300619.SZ', '301150.SZ']

    if dates is None:
        dates = ['20250605', '20250606']

    # 构建路径（参考 check_is_can_buy.py 的路径逻辑）
    current_file = Path(__file__).resolve()
    integrated_stock_selection_dir = current_file.parent
    to_dir = integrated_stock_selection_dir.parent

    print("=" * 100)
    print(f"检查本地数据中的复权信息")
    print(f"股票列表：{stock_list}")
    print(f"日期：{dates}")
    print("=" * 100)

    # 尝试从日线数据中读取
    for date_str in dates:
        year_month = date_str[:4] + date_str[4:6]

        # 尝试两个可能的路径
        possible_paths = [
            to_dir / "local_data" / "daily" / year_month / "daily_data.parquet",
            to_dir / "integrated_stock_selection" / "local_data" / "daily" / year_month / "daily_data.parquet"
        ]

        data_file = None
        for path in possible_paths:
            if path.exists():
                data_file = path
                break

        if data_file is None:
            print(f"\n[WARNING] 未找到 {date_str} 的数据文件")
            continue

        print(f"\n{'=' * 100}")
        print(f"读取文件：{data_file}")
        print(f"{'=' * 100}")

        try:
            # 读取数据
            df = pd.read_parquet(data_file)
            print(f"成功读取数据：{len(df):,} 行")

            # 检查列名
            print(f"\n可用列：{list(df.columns)[:20]}")

            # 筛选目标股票
            if 'stock_code' in df.columns:
                df_filtered = df[df['stock_code'].isin(stock_list)]

                if len(df_filtered) > 0:
                    print(f"\n找到 {len(df_filtered)} 条记录")

                    # 筛选日期
                    if 'date' in df_filtered.columns:
                        df_filtered['date_str'] = pd.to_datetime(df_filtered['date']).dt.strftime('%Y%m%d')
                        df_date = df_filtered[df_filtered['date_str'] == date_str]

                        if len(df_date) > 0:
                            print(f"\n{date_str} 的数据：")
                            print("-" * 100)

                            # 显示相关列
                            display_cols = ['stock_code', 'date', 'open', 'close']
                            related_cols = [col for col in df_date.columns if any(keyword in col.lower() for keyword in ['adjust', 'factor', 'ratio', 'dividend'])]
                            display_cols.extend(related_cols[:5])

                            available_cols = [col for col in display_cols if col in df_date.columns]

                            print(df_date[available_cols].to_string(index=False))

                            # 检查是否有复权因子列
                            factor_cols = [col for col in df_date.columns if 'factor' in col.lower() or 'adj' in col.lower()]
                            if factor_cols:
                                print(f"\n发现复权因子列：{factor_cols}")
                else:
                    print(f"[WARNING] 未找到股票 {stock_list} 的数据")
            else:
                print(f"[WARNING] 数据中没有 'stock_code' 列")

        except Exception as e:
            print(f"[ERROR] 读取文件失败: {e}")
            import traceback
            traceback.print_exc()

    # 尝试从特征数据中读取
    print(f"\n{'=' * 100}")
    print("尝试从特征数据中读取")
    print(f"{'=' * 100}")

    for date_str in dates:
        # 特征数据路径
        feature_file = to_dir / "local_data" / "features" / "daily" / f"{date_str}.parquet"

        if not feature_file.exists():
            alternative = to_dir / "integrated_stock_selection" / "local_data" / "features" / "daily" / f"{date_str}.parquet"
            if alternative.exists():
                feature_file = alternative
            else:
                continue

        print(f"\n读取特征文件：{feature_file}")

        try:
            df = pd.read_parquet(feature_file)

            if 'stock_code' in df.columns:
                df_filtered = df[df['stock_code'].isin(stock_list)]
                df_date = df_filtered[df_filtered['date'] == date_str]

                if len(df_date) > 0:
                    print(f"\n{date_str} 的特征数据：")
                    print("-" * 100)

                    # 显示开收盘价
                    display_cols = ['stock_code', 'date', 'open', 'close']
                    available_cols = [col for col in display_cols if col in df_date.columns]

                    if available_cols:
                        print(df_date[available_cols].to_string(index=False))
                    else:
                        print(f"可用列：{list(df_date.columns)[:10]}")

        except Exception as e:
            print(f"[ERROR] 读取特征文件失败: {e}")


def main():
    """主函数"""
    # 配置参数
    stock_list = ['300619.SZ', '301150.SZ']
    dates = ['20250605', '20250606']

    # 检查本地数据
    check_local_dividend_data(
        stock_list=stock_list,
        dates=dates
    )

    print(f"\n{'=' * 100}")
    print("[OK] 检查完成")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
