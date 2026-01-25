#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Factor Lab的完整列名
"""

import pandas as pd
from pathlib import Path

def check_columns():
    """检查所有列名"""
    factor_file = Path("factor_lab_data/factors.parquet")
    df = pd.read_parquet(factor_file)

    print(f"总列数: {len(df.columns)}")
    print(f"\n所有列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")

if __name__ == "__main__":
    check_columns()
