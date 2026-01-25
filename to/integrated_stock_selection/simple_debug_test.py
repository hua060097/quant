#!/usr/bin/env python3
"""
Simple test script for debugging support_distance calculation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_debug_test():
    """Run a simple debug test for support_distance"""

    print("=" * 80)
    print("DEBUG SUPPORT_DISTANCE CALCULATION TEST")
    print("=" * 80)

    # Try to import and test the debug function directly
    try:
        from factor_system.factor_lab.definitions.support_resistance_debug import _support_resistance_distances_debug
        print("Successfully imported debug function")
    except ImportError as e:
        print(f"Failed to import debug function: {e}")
        return False

    # Create minimal test data
    print("Creating test data...")

    # Check if we have any cache files with suitable data
    cache_files = [
        "cache/stock_features_20230601_20240110.parquet",
        "cache/stock_features_20221201_20240110.parquet"
    ]

    cache_file = None
    for cf in cache_files:
        if os.path.exists(cf):
            cache_file = cf
            break

    if not cache_file:
        print("No suitable cache file found")
        return False

    try:
        df = pd.read_parquet(cache_file)
        print(f"Loaded data from {cache_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Check required columns
        required_cols = {'stock_code', 'date', 'close', 'high', 'low', 'volume'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        # Filter for target stock
        target_stock = "300699.SZ"
        stock_data = df[df['stock_code'] == target_stock].copy()

        if len(stock_data) == 0:
            print(f"No data found for {target_stock}")
            return False

        # Sort by date
        stock_data = stock_data.sort_values('date').reset_index(drop=True)

        print(f"Found {len(stock_data)} records for {target_stock}")
        print(f"Date range: {stock_data['date'].min()} ~ {stock_data['date'].max()}")

        # Check for target date
        target_date = pd.to_datetime("2023-06-01")
        target_record = stock_data[stock_data['date'] == target_date]

        if len(target_record) == 0:
            print(f"Target date {target_date.date()} not found")
            # Show available dates
            print(f"Available dates around target:")
            nearby_dates = stock_data[
                (stock_data['date'] >= target_date - pd.Timedelta(days=5)) &
                (stock_data['date'] <= target_date + pd.Timedelta(days=5))
            ]['date'].dt.date.unique()
            for date in nearby_dates:
                print(f"  {date}")
            return False

        print(f"Found target record for {target_date.date()}")
        print(f"Target record values:")
        for col in ['close', 'high', 'low', 'volume']:
            val = target_record[col].iloc[0]
            print(f"  {col}: {val}")

        # Prepare data for calculation (just use the target stock's data)
        test_data = stock_data[list(required_cols)].copy()

        print(f"Prepared test data shape: {test_data.shape}")
        print("Running support_distance calculation...")

        # Run the calculation
        support_dist, resistance_dist = _support_resistance_distances_debug(test_data)

        print(f"Calculation completed")
        print(f"Support distance shape: {support_dist.shape}")
        print(f"Resistance distance shape: {resistance_dist.shape}")

        # Find the result for target date
        if target_date in test_data['date'].values:
            # Find the index
            target_idx = test_data[test_data['date'] == target_date].index[0]
            if target_idx in support_dist.index:
                target_support = support_dist.loc[target_idx]
                target_resistance = resistance_dist.loc[target_idx]

                print(f"\nRESULTS for {target_stock} on {target_date.date()}:")
                print(f"  Support distance: {target_support:.15f}")
                print(f"  Resistance distance: {target_resistance:.15f}")

                print(f"\nINPUT VALUES for {target_date.date()}:")
                target_row = test_data.iloc[test_data.index.get_loc(target_idx)]
                for col in ['close', 'high', 'low', 'volume']:
                    val = target_row[col]
                    print(f"  {col}: {val}")

                print(f"\nCheck the log file for detailed calculation steps:")
                print(f"  support_distance_debug.log")

                return True
            else:
                print(f"Target date index {target_idx} not found in results")
                return False
        else:
            print(f"Target date {target_date.date()} not in test data")
            return False

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting simple debug test...")

    # Clean up any existing debug file
    debug_file = "support_distance_debug.log"
    if os.path.exists(debug_file):
        os.remove(debug_file)
        print(f"Cleaned up {debug_file}")

    # Run the test
    success = simple_debug_test()

    if success:
        print("\nTest completed successfully!")
        print(f"Detailed logs saved to: {debug_file}")
        print("\nNext steps:")
        print("1. Run this script multiple times with the same input data")
        print("2. Compare the detailed logs to identify randomness")
        print("3. Restore the original support_resistance.py when done")
    else:
        print("\nTest failed!")
        print("Check the error messages above.")