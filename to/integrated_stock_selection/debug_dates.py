"""
Simple date check debug - just check if T+1/T+2/T+3 dates exist in test set
"""
import pandas as pd
from pathlib import Path
from datetime import timedelta
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("Date Check Debug")
print("=" * 80)

# Load cached data
cache_file = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\cache\stock_features_20230601_20260212.parquet")
df = pd.read_parquet(cache_file)

# Filter test set
test_start = pd.to_datetime('20250605')
prediction_data = df[df['date'] >= test_start].copy()

print(f"\nTest set info:")
print(f"  Total records: {len(prediction_data)}")
print(f"  Date range: {prediction_data['date'].min()} to {prediction_data['date'].max()}")
print(f"  Trading days: {len(prediction_data['date'].unique())}")

# Get unique dates and sort
unique_dates = sorted(prediction_data['date'].unique())
print(f"\nLast 10 trading days:")
for i, d in enumerate(unique_dates[-10:], 1):
    print(f"  {i}. {d}")

# Simulate what happens for the last few trading days
print("\n\nSimulate T+1/T+2/T+3 check for last 5 trading days:")
print("=" * 80)

# Get last 5 unique dates
last_5_dates = unique_dates[-5:]

for buy_date in last_5_dates:
    print(f"\nBuy date: {buy_date}")

    for day_offset in range(1, 4):  # T+1, T+2, T+3
        target_date = buy_date + timedelta(days=day_offset)
        target_date_ts = pd.Timestamp(target_date)

        exists = target_date_ts in prediction_data['date'].values
        exists_str = "[OK] Exists" if exists else "[X] Not exists"

        print(f"  T+{day_offset}: {target_date_ts.strftime('%Y-%m-%d')} - {exists_str}")

        if exists:
            # Count how many stocks on this date
            target_stocks = len(prediction_data[prediction_data['date'] == target_date_ts])
            print(f"        Stocks on this date: {target_stocks}")

print("\n" + "=" * 80)
print("Analysis Conclusion:")
print("=" * 80)

# Check if we have any valid sequences
valid_sequences = 0
for buy_date in unique_dates:
    has_t1 = (buy_date + timedelta(days=1)) in prediction_data['date'].values
    has_t2 = (buy_date + timedelta(days=2)) in prediction_data['date'].values
    has_t3 = (buy_date + timedelta(days=3)) in prediction_data['date'].values

    if has_t1 or has_t2 or has_t3:
        valid_sequences += 1

print(f"Total trading days: {len(unique_dates)}")
print(f"Trading days with at least one T+1/T+2/T+3: {valid_sequences}")
print(f"Percentage: {valid_sequences/len(unique_dates)*100:.1f}%")

if valid_sequences == 0:
    print("\n[WARNING] Problem found: All buy dates have no T+1/T+2/T+3 in test set!")
    print("  This explains why no valid return records were found.")
else:
    print(f"\n[OK] Should have {valid_sequences} buy dates that can calculate returns")
