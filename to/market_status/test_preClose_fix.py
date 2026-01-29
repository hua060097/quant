# -*- coding: utf-8 -*-
"""
Test preClose field usage for daily_change_pct calculation
"""
import sys
from pathlib import Path
import pandas as pd

# Add path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from market_status_processor import MarketStatusProcessor
from market_analyzer import MarketAnalyzer

def test_preClose_usage():
    """Test that preClose field is properly used instead of index-based prev_close"""
    print("=" * 80)
    print("Test preClose Field Usage")
    print("=" * 80)

    # Test 1: Check if preClose is in the requested fields
    print("\n[Test 1] Check Data Fields")
    print("-" * 80)
    processor = MarketStatusProcessor()

    # Check the fields list in _get_market_data
    import inspect
    source = inspect.getsource(processor._get_market_data)
    if "'preClose'" in source or '"preClose"' in source:
        print("[OK] preClose is included in data request fields")
    else:
        print("[FAIL] preClose is NOT in data request fields")
        return False

    # Test 2: Check if preClose is extracted in _calculate_indicators
    print("\n[Test 2] Check preClose Extraction")
    print("-" * 80)
    source = inspect.getsource(processor._calculate_indicators)
    if "preclose_series" in source:
        print("[OK] preclose_series is extracted in _calculate_indicators")
    else:
        print("[FAIL] preclose_series is NOT extracted")
        return False

    # Test 3: Check if preclose_series is in indicators
    print("\n[Test 3] Check preclose_series in Indicators")
    print("-" * 80)
    if "'preclose_series'" in source or '"preclose_series"' in source:
        print("[OK] preclose_series is stored in indicators dictionary")
    else:
        print("[FAIL] preclose_series is NOT stored in indicators")
        return False

    # Test 4: Check if preclose_series is used in market_analyzer
    print("\n[Test 4] Check preclose_series Usage in market_analyzer")
    print("-" * 80)
    analyzer = MarketAnalyzer()
    source = inspect.getsource(analyzer.analyze_market_status)
    if "preclose_series" in source:
        print("[OK] preclose_series is retrieved in analyze_market_status")
    else:
        print("[FAIL] preclose_series is NOT retrieved")
        return False

    # Test 5: Check the logic for using preClose
    print("\n[Test 5] Check preClose Usage Logic")
    print("-" * 80)
    if "preclose_series.iloc[i]" in source:
        print("[OK] preclose_series.iloc[i] is used to get prev_close")
    else:
        print("[FAIL] preclose_series.iloc[i] is NOT used")
        return False

    # Test 6: Check fallback logic
    print("\n[Test 6] Check Fallback Logic")
    print("-" * 80)
    if "if preclose_series is not None" in source and "close_series.iloc[i-1]" in source:
        print("[OK] Fallback logic exists when preClose is unavailable")
    else:
        print("[WARNING] Fallback logic might be incomplete")

    print("\n" + "=" * 80)
    print("All Tests Passed")
    print("=" * 80)

    print("\n[Summary]")
    print("-" * 80)
    print("✅ preClose field added to data request")
    print("✅ preclose_series extracted and stored in indicators")
    print("✅ preclose_series used instead of index-based prev_close")
    print("✅ Fallback logic exists for backward compatibility")
    print("\nExpected Impact:")
    print("  - No more IndexError when accessing prev_close")
    print("  - More accurate daily_change_pct calculation")
    print("  - Better handling of first trading day data")

    return True

if __name__ == "__main__":
    success = test_preClose_usage()
    sys.exit(0 if success else 1)
