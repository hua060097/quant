#!/usr/bin/env python3
"""
检查1月数据和跨年数据的连续性
"""

import pandas as pd
from pathlib import Path

def check_january_data():
    """检查1月数据"""
    stock_code = '000002.SZ'

    print("=" * 100)
    print(f"检查股票: {stock_code}")
    print(f"重点查找: 2025-12-29, 2026-01-01等日期的开盘价是否为17.12")
    print("=" * 100)

    base_dir = Path(__file__).parent.parent / "local_data" / "daily"

    # 检查1月数据文件
    january_file = base_dir / "202601" / "daily_data.parquet"

    if january_file.exists():
        print(f"\n[*] 找到1月数据文件: {january_file}")

        try:
            df = pd.read_parquet(january_file)
            print(f"[OK] 成功读取: {len(df):,} 行")

            # 筛选目标股票
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df['date'] = pd.to_datetime(stock_df['date'])

            print(f"\n[OK] 找到 {stock_code} 在1月的数据: {len(stock_df)} 条记录")

            # 显示所有1月数据
            display_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'pct_chg']
            available_cols = [col for col in display_cols if col in stock_df.columns]

            print(f"\n1月数据:")
            print(stock_df[available_cols].to_string(index=False))

            # 查找17.12这个价格
            print(f"\n查找17.12这个价格:")
            for idx, row in stock_df.iterrows():
                if abs(row['open'] - 17.12) < 0.01:
                    print(f"  ✓ 找到! {row['date'].strftime('%Y-%m-%d')} 开盘价: {row['open']:.2f}")
                if abs(row['close'] - 17.12) < 0.01:
                    print(f"  ✓ 找到! {row['date'].strftime('%Y-%m-%d')} 收盘价: {row['close']:.2f}")

        except Exception as e:
            print(f"[ERROR] 读取1月数据失败: {e}")
    else:
        print(f"\n[WARNING] 未找到1月数据文件: {january_file}")

    # 也检查特征文件
    print(f"\n{'=' * 100}")
    print(f"检查特征文件:")
    print(f"{'=' * 100}")

    # 检查12-26的特征文件
    feature_file_1226 = Path(__file__).parent.parent / "local_data" / "features" / "daily" / "20251226.parquet"

    if feature_file_1226.exists():
        print(f"\n[*] 找到特征文件: {feature_file_1226}")
        try:
            df_feat = pd.read_parquet(feature_file_1226)
            print(f"[OK] 读取特征文件: {len(df_feat)} 行")

            stock_feat = df_feat[df_feat['stock_code'] == stock_code]

            if len(stock_feat) > 0:
                print(f"\n{stock_code} 在2025-12-26的特征数据:")
                display_cols = ['stock_code', 'date', 'open', 'close', 'future_return_1d', 'is_can_buy']
                available_cols = [col for col in display_cols if col in stock_feat.columns]
                print(stock_feat[available_cols].to_string(index=False))
            else:
                print(f"[WARNING] 特征文件中未找到该股票")

        except Exception as e:
            print(f"[ERROR] 读取特征文件失败: {e}")
    else:
        print(f"[WARNING] 未找到特征文件: {feature_file_1226}")


if __name__ == "__main__":
    check_january_data()
