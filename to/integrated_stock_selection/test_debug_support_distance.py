#!/usr/bin/env python3
"""
Test script for debugging support_distance calculation for 300699.SZ on 2023-06-01
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_debug_support_distance():
    """Run support_distance calculation with debug logging for specific stock and date"""

    print("=" * 80)
    print("TESTING DEBUG SUPPORT_DISTANCE CALCULATION")
    print("=" * 80)

    # Import the debug function
    try:
        from factor_system.factor_lab.definitions.support_resistance_debug import _support_resistance_distances_debug
        print("[SUCCESS] Successfully imported debug function")
    except ImportError as e:
        print(f"[ERROR] Failed to import debug function: {e}")
        return False

    # Load test data - we need to load the raw data that feeds into the factor calculation
    print("\n📊 Loading test data...")

    # We'll need to reconstruct the data format expected by the factor calculation
    # This should be a DataFrame with columns: stock_code, date, close, high, low, volume
    # For now, let's check if we can find existing daily data
    try:
        # Check if we have cached features data
        cache_file = "cache/stock_features_20230601_20240110.parquet"
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            print(f"[SUCCESS] Loaded cached features from {cache_file}")
            print(f"   Data shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")

            # Filter for our target stock and date range
            target_stock = "300699.SZ"
            target_date = "2023-06-01"

            # Convert date column if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Filter for specific stock and wider date range to include history
            stock_data = df[df['stock_code'] == target_stock].copy()

            if len(stock_data) == 0:
                print(f"[ERROR] No data found for {target_stock}")
                return False

            # Get data around the target date to include lookback period
            target_dt = pd.to_datetime(target_date)
            lookback_start = target_dt - pd.Timedelta(days=120)  # Extra days for lookback
            lookback_end = target_dt + pd.Timedelta(days=10)

            stock_data = stock_data[
                (stock_data['date'] >= lookback_start) &
                (stock_data['date'] <= lookback_end)
            ].sort_values('date').reset_index(drop=True)

            print(f"📈 Filtered data for {target_stock}:")
            print(f"   Date range: {stock_data['date'].min()} ~ {stock_data['date'].max()}")
            print(f"   Records: {len(stock_data)}")
            print(f"   Target date {target_date} present: {target_dt in stock_data['date'].values}")

            # Check if required columns exist
            required_cols = {'stock_code', 'date', 'close', 'high', 'low', 'volume'}
            missing_cols = required_cols - set(stock_data.columns)
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"   Available columns: {list(stock_data.columns)}")
                return False

            # Select only required columns
            test_data = stock_data[list(required_cols)].copy()
            print(f"✅ Test data prepared with shape: {test_data.shape}")

            # Run the debug calculation
            print(f"\n🔍 Running debug support_distance calculation...")
            print(f"   Target: {target_stock} on {target_date}")

            try:
                support_dist, resistance_dist = _support_resistance_distances_debug(test_data)
                print(f"✅ Calculation completed")
                print(f"   Support distance shape: {support_dist.shape}")
                print(f"   Resistance distance shape: {resistance_dist.shape}")

                # Find the result for our target date
                target_row = test_data[test_data['date'] == target_dt]
                if len(target_row) > 0:
                    target_idx = target_row.index[0]
                    if target_idx in support_dist.index:
                        target_support = support_dist.loc[target_idx]
                        target_resistance = resistance_dist.loc[target_idx]

                        print(f"\n📋 RESULTS for {target_stock} on {target_date}:")
                        print(f"   Support distance: {target_support:.15f}")
                        print(f"   Resistance distance: {target_resistance:.15f}")

                        # Also show the input values
                        input_row = test_data.iloc[test_data.index.get_loc(target_idx)]
                        print(f"\n📊 INPUT VALUES:")
                        print(f"   Close: {input_row['close']:.10f}")
                        print(f"   High: {input_row['high']:.10f}")
                        print(f"   Low: {input_row['low']:.10f}")
                        print(f"   Volume: {input_row['volume']}")

                        return True
                    else:
                        print(f"❌ Target date index {target_idx} not found in results")
                        return False
                else:
                    print(f"❌ Target date {target_date} not found in filtered data")
                    return False

            except Exception as e:
                print(f"❌ Error during calculation: {e}")
                import traceback
                traceback.print_exc()
                return False

        else:
            print(f"❌ Cache file {cache_file} not found")
            return False

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def cleanup_debug_files():
    """Clean up debug files"""
    debug_file = "support_distance_debug.log"
    if os.path.exists(debug_file):
        os.remove(debug_file)
        print(f"🧹 Cleaned up {debug_file}")

if __name__ == "__main__":
    # Clean up any existing debug file
    cleanup_debug_files()

    print("Starting debug test for support_distance calculation...")

    # Run the test
    success = test_debug_support_distance()

    if success:
        print("\n✅ Debug test completed successfully!")
        print(f"📄 Detailed logs saved to: support_distance_debug.log")
        print("\n🔍 To analyze the differences:")
        print("   1. Run this script multiple times with the same input data")
        print("   2. Compare the detailed logs to identify any randomness")
        print("   3. Focus on the numeric values in each calculation step")
        print("   4. Pay special attention to:")
        print("      - Peak detection results")
        print("      - Histogram calculations")
        print("      - Weight computations")
        print("      - Final distance calculations")
    else:
        print("\n❌ Debug test failed!")
        print("Please check the error messages above.")

    print("\n📝 Note: After debugging, restore the original support_resistance.py file")