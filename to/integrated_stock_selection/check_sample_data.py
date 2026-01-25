#!/usr/bin/env python3
"""
检查样本数据的具体结构，找出NaN的真正原因
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_sample_data():
    """检查样本数据的详细结构"""

    data_dir = Path("../local_data/features/daily")

    # 检查几个具体日期的文件
    test_dates = ["20251021", "20251031", "20251115", "20251201", "20251216"]

    for date_str in test_dates:
        file_path = data_dir / f"{date_str}.parquet"

        if not file_path.exists():
            print(f"{date_str}: File not found")
            continue

        try:
            df = pd.read_parquet(file_path)
            print(f"\n{date_str}:")
            print(f"  Total records: {len(df)}")
            print(f"  Columns: {list(df.columns)}")

            if 'volume' in df.columns:
                volume_stats = {
                    'total': len(df),
                    'non_null': df['volume'].notna().sum(),
                    'is_zero': (df['volume'] == 0).sum(),
                    'min': df['volume'].min(),
                    'max': df['volume'].max(),
                    'mean': df['volume'].mean(),
                    'std': df['volume'].std()
                }
                print(f"  Volume stats: {volume_stats}")

                # 检查前几行数据
                print(f"  First 5 volume values: {df['volume'].head().tolist()}")

                # 检查是否有异常值
                volume_unique = df['volume'].dropna().nunique()
                print(f"  Unique volume values: {volume_unique}")

                # 检查是否都是同一个值
                if volume_unique == 1:
                    print(f"  WARNING: All volume values are the same!")

            # 检查其他可能相关的字段
            for col in ['close', 'open', 'high', 'low', 'pct_chg']:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        print(f"  {col} has {null_count} null values")

            # 检查数据类型
            print(f"  Data types: {dict(df.dtypes)}")

        except Exception as e:
            print(f"  Error reading file: {e}")

if __name__ == "__main__":
    check_sample_data()