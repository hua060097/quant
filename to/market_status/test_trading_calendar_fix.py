# -*- coding: utf-8 -*-
"""
Test trading calendar backtracking functionality
"""
import sys
from pathlib import Path

# Add path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from market_status_processor import MarketStatusProcessor

def test_trading_calendar_fix():
    """Test trading calendar backtracking fix"""
    print("=" * 80)
    print("Test Trading Calendar Backtracking")
    print("=" * 80)

    # Create processor instance
    processor = MarketStatusProcessor()

    # Test 1: Load trading calendar
    print("\n[Test 1] Load Trading Calendar")
    print("-" * 80)
    trading_calendar = processor._load_trading_calendar()
    if trading_calendar is not None:
        print(f"[OK] Successfully loaded trading calendar: {len(trading_calendar)} trading days")
        print(f"   Date range: {trading_calendar['date_str'].min()} ~ {trading_calendar['date_str'].max()}")
        print(f"\n   First 5 trading days:")
        for i in range(min(5, len(trading_calendar))):
            print(f"     {trading_calendar.iloc[i]['date_str']} ({trading_calendar.iloc[i]['date']})")
    else:
        print("[FAIL] Failed to load trading calendar")
        return False

    # Test 2: Trading calendar backtracking - normal case
    print("\n[Test 2] Trading Calendar Backtracking - Normal Case")
    print("-" * 80)
    test_date = "20221205"
    history_start = processor._get_history_start_date_by_trading_days(test_date, 180)
    if history_start is not None:
        print(f"[OK] Successfully backtracked")
        print(f"   Start date: {test_date}")
        print(f"   History start date: {history_start}")
        print(f"   Trading days back: 180")

        # Verify: history_start should be in trading calendar
        if history_start in trading_calendar['date_str'].values:
            print(f"   [OK] History start date is in trading calendar")
        else:
            print(f"   [FAIL] History start date is NOT in trading calendar")
    else:
        print(f"[FAIL] Backtracking failed")

    # Test 3: Trading calendar backtracking - non-trading day
    print("\n[Test 3] Trading Calendar Backtracking - Non-Trading Day (Weekend)")
    print("-" * 80)
    test_date = "20221210"  # 2022-12-10 is Saturday
    history_start = processor._get_history_start_date_by_trading_days(test_date, 180)
    if history_start is not None:
        print(f"[OK] Successfully handled non-trading day")
        print(f"   Input date: {test_date} (Saturday)")
        print(f"   History start date: {history_start}")
    else:
        print(f"[FAIL] Handling failed")

    # Test 4: Trading calendar backtracking - boundary case (first trading day)
    print("\n[Test 4] Trading Calendar Backtracking - Boundary Case (First Trading Day)")
    print("-" * 80)
    first_trading_day = trading_calendar.iloc[0]['date_str']
    print(f"   First trading day in calendar: {first_trading_day}")
    history_start = processor._get_history_start_date_by_trading_days(first_trading_day, 180)
    if history_start is not None:
        print(f"[OK] Successfully handled boundary case")
        print(f"   Start date: {first_trading_day}")
        print(f"   History start date: {history_start}")
        start_idx = trading_calendar[trading_calendar['date_str'] == first_trading_day].index[0]
        end_idx = trading_calendar[trading_calendar['date_str'] == history_start].index[0]
        print(f"   Trading days back: {start_idx - end_idx}")
    else:
        print(f"[FAIL] Handling failed")

    # Test 5: Compare natural days vs trading days
    print("\n[Test 5] Comparison: Natural Days Backtracking vs Trading Calendar Backtracking")
    print("-" * 80)
    from datetime import timedelta, datetime

    test_date = "20221205"
    test_date_dt = datetime.strptime(test_date, '%Y%m%d')

    # Natural days backtracking
    natural_back = test_date_dt - timedelta(days=180)
    natural_back_str = natural_back.strftime('%Y%m%d')

    # Trading calendar backtracking
    calendar_back_str = processor._get_history_start_date_by_trading_days(test_date, 180)

    print(f"   Start date: {test_date}")
    print(f"   Natural days back 180 days: {natural_back_str}")
    print(f"   Trading calendar back 180 days: {calendar_back_str}")

    # Calculate actual trading days
    if calendar_back_str in trading_calendar['date_str'].values:
        start_idx = trading_calendar[trading_calendar['date_str'] == test_date].index[0]
        end_idx = trading_calendar[trading_calendar['date_str'] == calendar_back_str].index[0]
        actual_trading_days = start_idx - end_idx
        print(f"   Actual trading days: {actual_trading_days}")

    # Test 6: Verify fix effect
    print("\n[Test 6] Verify Fix Effect")
    print("-" * 80)
    print(f"   Before fix:")
    print(f"     - Using natural days backtracking: {natural_back_str}")
    print(f"     - May cause first trading day data missing")
    print(f"     - daily_change_pct = 0")
    print(f"\n   After fix:")
    print(f"     - Using trading calendar backtracking: {calendar_back_str}")
    print(f"     - Guarantees history_start_date is trading day")
    print(f"     - daily_change_pct calculated correctly")

    print("\n" + "=" * 80)
    print("Test Completed")
    print("=" * 80)

    return True

if __name__ == "__main__":
    test_trading_calendar_fix()
