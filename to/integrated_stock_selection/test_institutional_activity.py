#!/usr/bin/env python3
"""
直接测试 institutional_activity_score 函数
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_institutional_activity():
    """测试institutional_activity_score函数的计算逻辑"""

    data_dir = Path("../local_data/features/daily")

    # 选择一个测试日期
    test_date = "20251021"
    file_path = data_dir / f"{test_date}.parquet"

    if not file_path.exists():
        print(f"File not found: {test_date}")
        return

    df = pd.read_parquet(file_path)
    print(f"{test_date}: Loaded {len(df)} records")

    # 模拟 institutional_activity_score 的计算逻辑
    def simulate_institutional_activity_score(volume, close, current_price):
        """模拟计算逻辑"""
        print(f"Input volume length: {len(volume)}")
        print(f"Input close length: {len(close)}")
        print(f"Current price: {current_price}")

        if len(volume) < 30:
            print("Volume length < 30, returning 50.0")
            return 50.0

        # 转换为float32
        volume = volume.astype(np.float32, copy=False)
        close = close.astype(np.float32, copy=False)

        print(f"After conversion - volume type: {volume.dtype}, close type: {close.dtype}")

        # 检查原始数据中的NaN
        volume_nan_count = np.isnan(volume).sum()
        close_nan_count = np.isnan(close).sum()
        print(f"NaN count in volume: {volume_nan_count}")
        print(f"NaN count in close: {close_nan_count}")

        # 数据切片
        recent_volume = volume[-10:]
        hist_volume = volume[-30:-10]

        print(f"Recent volume shape: {recent_volume.shape}")
        print(f"Recent volume NaN count: {np.isnan(recent_volume).sum()}")
        print(f"Recent volume sample: {recent_volume[:5]}")

        print(f"Hist volume shape: {hist_volume.shape}")
        print(f"Hist volume NaN count: {np.isnan(hist_volume).sum()}")
        print(f"Hist volume sample: {hist_volume[:5]}")

        # 使用向量化操作计算均值
        recent_mean = np.mean(recent_volume)
        hist_mean = np.mean(hist_volume)

        print(f"Recent mean: {recent_mean}")
        print(f"Hist mean: {hist_mean}")

        # 检查除法
        volume_amplification = recent_mean / hist_mean if hist_mean > 0 else 1.0
        print(f"Volume amplification: {volume_amplification}")

        # 检查price_volatility计算
        recent_close = close[-10:]
        close_mean = np.mean(recent_close)
        print(f"Close mean: {close_mean}")
        print(f"Recent close NaN count: {np.isnan(recent_close).sum()}")

        if close_mean > 0:
            price_volatility = np.std(recent_close) / close_mean
        else:
            price_volatility = 0.1

        print(f"Price volatility: {price_volatility}")

        # 最终计算
        activity_score = (
            min(volume_amplification / 2, 1.0) * 0.3 +
            (np.clip(1 - price_volatility * 15, 0, 1)) * 0.25 +
            0.5 * 0.25 +  # 简化的volume_consistency
            0.5 * 0.2     # 简化的efficiency_score
        ) * 100

        final_score = float(min(100.0, max(0.0, activity_score)))
        print(f"Final score: {final_score}")
        print(f"Is final score NaN? {np.isnan(final_score)}")

        return final_score

    # 选择几只股票进行测试
    test_stock_codes = df['stock_code'].unique()[:5]

    print(f"\nTesting {len(test_stock_codes)} stocks:")

    for stock_code in test_stock_codes:
        stock_data = df[df['stock_code'] == stock_code]
        if len(stock_data) == 0:
            continue

        current_price = float(stock_data['close'].iloc[-1]) if not pd.isna(stock_data['close'].iloc[-1]) else 100.0

        # 获取历史数据（这里假设历史数据在同一个文件中）
        # 实际情况下可能需要从多个文件获取
        volume_data = stock_data['volume'].values
        close_data = stock_data['close'].values

        print(f"\n=== Testing {stock_code} ===")
        print(f"Stock data length: {len(stock_data)}")
        print(f"Volume data length: {len(volume_data)}")
        print(f"Close data length: {len(close_data)}")
        print(f"Volume sample: {volume_data[:10]}")
        print(f"Close sample: {close_data[:10]}")

        try:
            result = simulate_institutional_activity_score(volume_data, close_data, current_price)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_institutional_activity()