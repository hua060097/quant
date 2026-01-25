#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Data Quality Analysis Report
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_factor_quality():
    """Analyze factor data quality and generate detailed report"""

    factor_file = Path("factor_lab_data/factors1.parquet")

    if not factor_file.exists():
        print("Factor file not found!")
        return

    print("=" * 80)
    print("FACTOR DATA QUALITY ANALYSIS REPORT")
    print("=" * 80)

    # Load data
    try:
        df = pd.read_parquet(factor_file)
        print(f"File: {factor_file}")
        print(f"File size: {factor_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"Data shape: {df.shape}")
        print()

        # Identify index and factor columns
        index_cols = ['date', 'stock_code']
        factor_cols = [col for col in df.columns if col not in index_cols]

        print(f"Index columns: {index_cols}")
        print(f"Factor columns: {len(factor_cols)}")
        print()

        # Analyze each factor
        factor_analysis = {}

        for col in factor_cols:
            nan_count = df[col].isna().sum()
            total_count = len(df)
            valid_count = total_count - nan_count
            nan_ratio = nan_count / total_count if total_count > 0 else 0

            # Calculate basic statistics for non-NaN values
            if valid_count > 0:
                valid_values = df[col].dropna()
                mean_val = valid_values.mean()
                std_val = valid_values.std()
                min_val = valid_values.min()
                max_val = valid_values.max()
            else:
                mean_val = std_val = min_val = max_val = np.nan

            factor_analysis[col] = {
                'nan_count': nan_count,
                'nan_ratio': nan_ratio,
                'valid_count': valid_count,
                'valid_ratio': valid_count / total_count,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }

        # Sort by NaN ratio (best quality first)
        sorted_factors = sorted(factor_analysis.items(), key=lambda x: x[1]['nan_ratio'])

        print("FACTOR QUALITY RANKING (best to worst):")
        print("-" * 100)
        print(f"{'Factor Name':<35} {'NaN Count':>12} {'NaN Ratio':>12} {'Valid Ratio':>12} {'Mean':>12} {'Std':>12}")
        print("-" * 100)

        for col, stats in sorted_factors:
            print(f"{col[:33]:<35} {stats['nan_count']:>10,} {stats['nan_ratio']:>10.2%} {stats['valid_ratio']:>10.2%} {stats['mean']:>10.3f} {stats['std']:>10.3f}")

        # Quality classification
        excellent = [col for col, stats in factor_analysis.items() if stats['nan_ratio'] == 0]
        good = [col for col, stats in factor_analysis.items() if 0 < stats['nan_ratio'] <= 0.05]
        fair = [col for col, stats in factor_analysis.items() if 0.05 < stats['nan_ratio'] <= 0.20]
        poor = [col for col, stats in factor_analysis.items() if stats['nan_ratio'] > 0.20]

        print("\n" + "=" * 80)
        print("QUALITY CLASSIFICATION SUMMARY")
        print("=" * 80)
        print(f"Excellent (NaN = 0%):     {len(excellent):3d} factors ({len(excellent)/len(factor_cols)*100:5.1f}%)")
        print(f"Good (0% < NaN ≤ 5%):     {len(good):3d} factors ({len(good)/len(factor_cols)*100:5.1f}%)")
        print(f"Fair (5% < NaN ≤ 20%):    {len(fair):3d} factors ({len(fair)/len(factor_cols)*100:5.1f}%)")
        print(f"Poor (NaN > 20%):         {len(poor):3d} factors ({len(poor)/len(factor_cols)*100:5.1f}%)")

        # Show worst factors
        worst_factors = sorted(factor_analysis.items(), key=lambda x: x[1]['nan_ratio'], reverse=True)[:10]

        print("\n" + "=" * 80)
        print("TOP 10 WORST QUALITY FACTORS")
        print("=" * 80)
        for i, (col, stats) in enumerate(worst_factors, 1):
            print(f"{i:2d}. {col:<35} NaN: {stats['nan_ratio']:>6.2%} ({stats['nan_count']:,}/{len(df):,})")

        # Show best factors
        best_factors = sorted(factor_analysis.items(), key=lambda x: x[1]['nan_ratio'])[:10]

        print("\n" + "=" * 80)
        print("TOP 10 BEST QUALITY FACTORS")
        print("=" * 80)
        for i, (col, stats) in enumerate(best_factors, 1):
            print(f"{i:2d}. {col:<35} NaN: {stats['nan_ratio']:>6.2%} ({stats['nan_count']:,}/{len(df):,})")

        # Data completeness analysis
        print("\n" + "=" * 80)
        print("DATA COMPLETENESS ANALYSIS")
        print("=" * 80)

        if 'date' in df.columns and 'stock_code' in df.columns:
            unique_dates = df['date'].nunique()
            unique_stocks = df['stock_code'].nunique()
            expected_records = unique_dates * unique_stocks
            actual_records = len(df)
            completeness = actual_records / expected_records * 100 if expected_records > 0 else 0

            print(f"Unique dates: {unique_dates:,}")
            print(f"Unique stocks: {unique_stocks:,}")
            print(f"Expected records: {expected_records:,}")
            print(f"Actual records: {actual_records:,}")
            print(f"Data completeness: {completeness:.2f}%")

            # Date range
            date_min = df['date'].min()
            date_max = df['date'].max()
            date_range_days = (date_max - date_min).days + 1
            print(f"Date range: {date_min} to {date_max} ({date_range_days} days)")

            # Check for missing dates
            if date_range_days != unique_dates:
                missing_dates = date_range_days - unique_dates
                print(f"Missing trading days: {missing_dates}")

        # Memory usage analysis
        print("\n" + "=" * 80)
        print("MEMORY USAGE ANALYSIS")
        print("=" * 80)

        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / (1024**2)  # MB
        index_memory = memory_usage[index_cols].sum() / (1024**2)  # MB
        factor_memory = memory_usage[factor_cols].sum() / (1024**2)  # MB

        print(f"Total memory usage: {total_memory:.2f} MB")
        print(f"Index columns memory: {index_memory:.2f} MB ({index_memory/total_memory*100:.1f}%)")
        print(f"Factor columns memory: {factor_memory:.2f} MB ({factor_memory/total_memory*100:.1f}%)")
        print(f"Average memory per factor: {factor_memory/len(factor_cols):.2f} MB")

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if len(poor) > 0:
            print(f"1. URGENT: {len(poor)} factors have >20% NaN values and should be investigated:")
            for col in poor[:5]:
                stats = factor_analysis[col]
                print(f"   - {col}: {stats['nan_ratio']:.1%} NaN")
            if len(poor) > 5:
                print(f"   ... and {len(poor)-5} more")

        if len(excellent) == 0:
            print("2. WARNING: No factor has 100% valid data")
        else:
            print(f"2. GOOD: {len(excellent)} factors have perfect data quality")

        if len(good) > 0:
            print(f"3. ACCEPTABLE: {len(good)} factors have acceptable quality (≤5% NaN)")

        print("4. Consider data imputation strategies for high-NaN factors")
        print("5. Review factor calculation logic for poor quality factors")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"Error analyzing factor file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_factor_quality()