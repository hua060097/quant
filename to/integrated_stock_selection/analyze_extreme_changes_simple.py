#!/usr/bin/env python3
"""
Analyze extreme price changes > 20%

Distinguish between normal high volatility and truly abnormal data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_extreme_changes():
    """Analyze extreme data with >20% price changes"""
    print("=" * 80)
    print("Extreme Price Changes Analysis (>20%)")
    print("=" * 80)

    # Read factors.parquet file
    factors_file = Path("factor_lab_data/factors.parquet")
    if not factors_file.exists():
        print("ERROR: factors.parquet file not found")
        return

    try:
        df = pd.read_parquet(factors_file)
        print(f"OK: Successfully read data: {len(df):,} records")

        # Ensure pct_chg_1d column exists
        if 'pct_chg_1d' not in df.columns:
            print("ERROR: pct_chg_1d column not found")
            return

        # Find records with >20% price changes
        extreme_mask = df['pct_chg_1d'].abs() > 0.2
        extreme_data = df[extreme_mask].copy()

        print(f"\n=== Extreme Data Statistics ===")
        print(f"Records with >20% price changes: {len(extreme_data):,}")
        print(f"Percentage of total records: {len(extreme_data)/len(df)*100:.3f}%")

        if len(extreme_data) == 0:
            print("OK: No records with >20% price changes found")
            return

        # Convert date format
        extreme_data['date'] = pd.to_datetime(extreme_data['date'])

        # Categorize by price change magnitude
        extreme_data['abs_pct_chg'] = extreme_data['pct_chg_1d'].abs()
        extreme_data['change_type'] = np.where(extreme_data['pct_chg_1d'] > 0, 'Up', 'Down')

        # Categorize by extreme level
        def categorize_extremeness(abs_change):
            if 0.2 <= abs_change < 0.3:
                return 'Mild (20-30%)'
            elif 0.3 <= abs_change < 0.5:
                return 'Moderate (30-50%)'
            elif 0.5 <= abs_change < 1.0:
                return 'Severe (50-100%)'
            else:
                return 'Extreme (>100%)'

        extreme_data['extreme_level'] = extreme_data['abs_pct_chg'].apply(categorize_extremeness)

        # Statistics by extreme level
        print(f"\n=== Extreme Level Distribution ===")
        level_stats = extreme_data['extreme_level'].value_counts()
        for level, count in level_stats.items():
            print(f"   {level}: {count:,} records")

        # Statistics by direction
        print(f"\n=== Direction Distribution ===")
        direction_stats = extreme_data['change_type'].value_counts()
        for direction, count in direction_stats.items():
            print(f"   {direction}: {count:,} records")

        # Statistics by date
        print(f"\n=== Daily Extreme Data Statistics ===")
        date_stats = extreme_data.groupby(extreme_data['date'].dt.strftime('%Y-%m-%d')).agg({
            'stock_code': 'count',
            'pct_chg_1d': ['max', 'min', 'mean']
        }).round(6)

        date_stats.columns = ['Count', 'Max_Gain', 'Max_Loss', 'Avg_Change']
        date_stats = date_stats.sort_values('Count', ascending=False)

        print("Top 10 dates with most extreme data:")
        for date, row in date_stats.head(10).iterrows():
            print(f"   {date}: {row['Count']:3d} records, Max gain:{row['Max_Gain']:+.2%}, Max loss:{row['Max_Loss']:+.2%}")

        # Most severe extreme data
        print(f"\n=== Top 20 Most Severe Extreme Data ===")
        extreme_sorted = extreme_data.sort_values('abs_pct_chg', ascending=False)

        for i, (_, row) in enumerate(extreme_sorted.head(20).iterrows(), 1):
            date_str = row['date'].strftime('%Y-%m-%d')
            stock_code = row['stock_code']
            pct_chg = row['pct_chg_1d']
            level = row['extreme_level']
            change_type = row['change_type']

            print(f"   {i:2d}. {date_str} {stock_code} {change_type} {pct_chg:+.2%} ({level})")

        # Stocks with most extreme data
        print(f"\n=== Top 20 Stocks with Most Extreme Data ===")
        stock_stats = extreme_data.groupby('stock_code').agg({
            'pct_chg_1d': 'count',
            'abs_pct_chg': 'max'
        }).round(6)
        stock_stats.columns = ['Extreme_Count', 'Max_Change']
        stock_stats = stock_stats.sort_values('Extreme_Count', ascending=False)

        for i, (stock_code, row) in enumerate(stock_stats.head(20).iterrows(), 1):
            print(f"   {i:2d}. {stock_code}: {row['Extreme_Count']:2d} times, Max change:{row['Max_Change']:+.2%}")

        # Normal vs abnormal criteria
        print(f"\n=== Normal vs Abnormal Criteria ===")

        # Distribution by ranges
        ranges = [
            (0.2, 0.25, "20-25%: Possibly normal high volatility"),
            (0.25, 0.3, "25-30%: Borderline abnormal, needs attention"),
            (0.3, 0.4, "30-40%: Clearly abnormal"),
            (0.4, 0.5, "40-50%: Severely abnormal"),
            (0.5, 1.0, "50-100%: Extremely abnormal"),
            (1.0, float('inf'), "100%+: Data error or major event")
        ]

        for min_val, max_val, description in ranges:
            count = len(extreme_data[(extreme_data['abs_pct_chg'] >= min_val) &
                                   (extreme_data['abs_pct_chg'] < max_val)])
            print(f"   {description}: {count:,} records")

        # Recommendations
        print(f"\n=== Analysis Recommendations ===")
        print(f"   1. 20-25% changes: Relatively normal, may be ST stocks or new stocks")
        print(f"   2. 25-30% changes: Need close attention, check for special reasons")
        print(f"   3. >30% changes: Strongly recommend checking data quality and major events")
        print(f"   4. >50% changes: Very likely data errors, need manual verification")
        print(f"   5. >100% changes: Almost certainly data quality issues")

        # Save detailed report
        report_file = "extreme_changes_report.csv"
        extreme_sorted.to_csv(report_file, index=False, encoding='utf-8')
        print(f"\nDetailed report saved to: {report_file}")

        return extreme_sorted

    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_data_quality():
    """Check data quality"""
    print("\n" + "=" * 80)
    print("Data Quality Check")
    print("=" * 80)

    try:
        df = pd.read_parquet("factor_lab_data/factors.parquet")

        # Basic statistics for pct_chg_1d
        print(f"\n=== Basic Statistics for pct_chg_1d ===")
        stats = df['pct_chg_1d'].describe()
        print(f"   Total records: {stats['count']:,.0f}")
        print(f"   Mean: {stats['mean']:+.6f}")
        print(f"   Std: {stats['std']:.6f}")
        print(f"   Min: {stats['min']:+.6f}")
        print(f"   Max: {stats['max']:+.6f}")
        print(f"   Median: {df['pct_chg_1d'].median():+.6f}")

        # Check for obvious errors
        print(f"\n=== Obvious Data Error Check ===")

        # Changes over ±50% (usually unreasonable except special cases)
        very_extreme = df[df['pct_chg_1d'].abs() > 0.5]
        print(f"   Changes over ±50%: {len(very_extreme):,} records")

        if len(very_extreme) > 0:
            print(f"   Examples:")
            for _, row in very_extreme.head(5).iterrows():
                print(f"     {row['date']} {row['stock_code']} {row['pct_chg_1d']:+.2%}")

        # Check exactly ±100% changes (usually data errors)
        exactly_100 = df[df['pct_chg_1d'].abs() == 1.0]
        print(f"   Exactly ±100% changes: {len(exactly_100):,} records (likely data errors)")

        # Check missing values
        missing_count = df['pct_chg_1d'].isna().sum()
        print(f"   Missing values: {missing_count:,} records")

        # Check infinite values
        inf_count = np.isinf(df['pct_chg_1d']).sum()
        print(f"   Infinite values: {inf_count:,} records")

    except Exception as e:
        print(f"ERROR in data quality check: {e}")

if __name__ == "__main__":
    try:
        # Execute extreme data analysis
        extreme_data = analyze_extreme_changes()

        # Execute data quality check
        check_data_quality()

        print("\n" + "=" * 80)
        print("Analysis Complete")
        print("=" * 80)

        if extreme_data is not None:
            print("OK: Extreme data analysis completed")
            print("Recommendation: Focus on data with >30% changes, as these may contain data quality issues")
        else:
            print("ERROR: Extreme data analysis failed")

    except Exception as e:
        print(f"ERROR: Program execution failed: {e}")
        import traceback
        traceback.print_exc()