#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline Feature Factory: Pre-calculates and stores features for all stocks.

This script loads raw daily market data, calculates a comprehensive set of
technical indicators and features using optimized methods (vectorized/Numba),
and saves the results partitioned by date in Parquet format for fast loading
during backtesting.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import traceback
import gc # Garbage Collection

# --- Add project root to sys.path ---
# Assuming this script is in the 'to' directory
current_dir = Path(__file__).resolve().parent # Use resolve() for robustness
project_root = current_dir # Project root is 'to' directory
# If the script is one level down from 'to', use current_dir.parent
# project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Ensure the parent of 'to' is also in path if necessary for imports like advanced_quant_data
parent_of_to = project_root.parent
if str(parent_of_to) not in sys.path:
     sys.path.insert(0, str(parent_of_to))
# --- End Add project root ---

# --- Imports from project modules ---
try:
    # Assumes advanced_quant_data.py is directly under the parent of 'to' or accessible
    from advanced_quant_data import ParquetDataStorage
    # We might not need MarketStatusProcessor directly if we reimplement MA logic
    # from market_status.market_status_processor import MarketStatusProcessor
    # Assuming vectorized/numba logic is accessible or replicated here
    # If they are in separate files, import them:
    # from vectorized_features import calculate_vectorized_features
    # from numba_accelerator import calculate_numba_features
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure necessary modules (like advanced_quant_data.py) are in the correct path relative to generate_features.py.")
    sys.exit(1)
# --- End Imports ---

# --- Configuration ---
# Adjust FEATURE_OUTPUT_DIR relative to the script location or use absolute path
FEATURE_OUTPUT_DIR = project_root / "local_data" / "features" / "daily"
LOG_DIR = project_root / "log"
MAX_LOOKBACK = 200  # Max lookback needed for any feature (e.g., MA120 + buffer)
# Consider adding a configuration for required fields for robustness
REQUIRED_RAW_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount']

# --- Setup Logging ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Ensure only one FileHandler and StreamHandler are added
log_file_path = LOG_DIR / "generate_features.log"
logger = logging.getLogger("FeatureFactory")
if not logger.handlers: # Prevent adding handlers multiple times if script is re-run in same session
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
# --- End Logging Setup ---


# --- Feature Calculation Functions ---
# Using optimized pandas operations. Replace with numba/vectorized calls if available.

def calculate_ma(series, n):
    """Calculates Simple Moving Average, handling NaNs at the start."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty or len(series) < n:
        return pd.Series(np.nan, index=series.index)
    # Ensure input is float to prevent integer MA issues
    ma = series.astype(float).rolling(window=n, min_periods=n).mean()
    return ma

def calculate_ema(series, n):
    """Calculates Exponential Moving Average."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty or len(series) < n:
        return pd.Series(np.nan, index=series.index)
    # Ensure input is float
    return series.astype(float).ewm(span=n, adjust=False, min_periods=n).mean()

def calculate_rsi(series, n=14):
    """Calculates Relative Strength Index (RSI)."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty or len(series) < n + 1:
         return pd.Series(50.0, index=series.index) # Return neutral RSI if not enough data

    delta = series.astype(float).diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    # Calculate initial average using SMA
    # avg_gain = gain.rolling(window=n, min_periods=n).mean()
    # avg_loss = loss.rolling(window=n, min_periods=n).mean()
    # Using EWM directly is more common and efficient in pandas
    ema_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ema_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs = ema_gain / ema_loss.replace(0, 1e-10) # Avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # rsi.iloc[:n] = np.nan # Mark initial periods as NaN (ewm min_periods handles this mostly)
    rsi.fillna(50.0, inplace=True) # Fill any remaining NaNs (e.g., from division by zero if loss is persistently 0)
    rsi = rsi.clip(0, 100) # Ensure RSI is within 0-100
    return rsi

def calculate_macd(series, n_fast=12, n_slow=26, n_signal=9):
    """Calculates MACD, MACD Signal, and MACD Histogram."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty or len(series) < n_slow:
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series

    ema_fast = calculate_ema(series, n_fast)
    ema_slow = calculate_ema(series, n_slow)
    macd_line = ema_fast - ema_slow
    # Calculate signal line only on non-NaN MACD values for stability
    macd_signal = calculate_ema(macd_line.dropna(), n_signal)
    macd_hist = macd_line - macd_signal
    # Reindex signal and hist to match original macd_line index
    macd_signal = macd_signal.reindex(macd_line.index)
    macd_hist = macd_hist.reindex(macd_line.index)
    return macd_line, macd_signal, macd_hist

def calculate_bollinger_bands(series, n=20, k=2):
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if series.empty or len(series) < n:
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series

    middle_band = calculate_ma(series, n)
    std_dev = series.astype(float).rolling(window=n, min_periods=n).std()
    upper_band = middle_band + k * std_dev
    lower_band = middle_band - k * std_dev
    return upper_band, middle_band, lower_band

def calculate_atr(high, low, close, n=14):
    """Calculates Average True Range (ATR)."""
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        raise TypeError("Inputs must be pandas Series")
    if high.empty or len(high) < n or low.empty or len(low) < n or close.empty or len(close) < n:
         return pd.Series(np.nan, index=close.index)

    # Ensure float type
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    high_low = high - low
    high_prev_close = abs(high - close.shift(1))
    low_prev_close = abs(low - close.shift(1))

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1, skipna=False)
    # Calculate EMA on TR, starting calculation after the first NaN which occurs due to shift(1)
    atr = calculate_ema(tr.iloc[1:], n)
    # Reindex to original index, filling the first value with NaN
    atr = atr.reindex(close.index)
    return atr

def calculate_support_resistance(low, high, n=14):
    """Calculates simple rolling support and resistance."""
    if not isinstance(low, pd.Series) or not isinstance(high, pd.Series):
        raise TypeError("Inputs must be pandas Series")
    if low.empty or len(low) < n or high.empty or len(high) < n:
        nan_series = pd.Series(np.nan, index=low.index)
        return nan_series, nan_series

    support = low.astype(float).rolling(window=n, min_periods=n).min()
    resistance = high.astype(float).rolling(window=n, min_periods=n).max()
    return support, resistance

def calculate_all_features_for_stock(stock_df):
    """
    Calculates all defined features for a single stock's DataFrame.
    Input DataFrame must have 'date' as index and columns like
    'open', 'high', 'low', 'close', 'volume', 'amount'.
    """
    stock_code = stock_df['stock_code'].iloc[0] if not stock_df.empty else "Unknown"
    logger.debug(f"Calculating features for {stock_code} - Input shape: {stock_df.shape}")
    try:
        if stock_df.empty:
            logger.warning(f"Input DataFrame for {stock_code} is empty.")
            return pd.DataFrame()

        # Ensure index is datetime and sorted
        if not isinstance(stock_df.index, pd.DatetimeIndex):
             logger.error(f"Index for {stock_code} is not DatetimeIndex. Type: {type(stock_df.index)}")
             try:
                 stock_df.index = pd.to_datetime(stock_df.index)
             except Exception as conv_err:
                  logger.error(f"Failed to convert index to DatetimeIndex for {stock_code}: {conv_err}")
                  return pd.DataFrame()
        if not stock_df.index.is_monotonic_increasing:
            logger.warning(f"Index for {stock_code} is not sorted. Sorting...")
            stock_df = stock_df.sort_index()


        # Ensure required columns exist
        if not all(col in stock_df.columns for col in REQUIRED_RAW_FIELDS):
             missing = [col for col in REQUIRED_RAW_FIELDS if col not in stock_df.columns]
             logger.error(f"Missing required columns for {stock_code}: {missing}")
             return pd.DataFrame()

        # Convert data to numeric, coercing errors - DO THIS EARLY
        for col in REQUIRED_RAW_FIELDS:
            if stock_df[col].dtype == 'object': # Only convert if necessary
                stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')

        # Check for excessive NaNs in 'close' price AFTER conversion
        if stock_df['close'].isnull().all() or stock_df['close'].isnull().sum() / len(stock_df) > 0.8:
            logger.warning(f"Stock {stock_code} has excessive (>80% or all) NaN close prices, skipping.")
            return pd.DataFrame()

        # Optional: Impute limited NaNs (e.g., forward fill) - Use with caution
        # stock_df[REQUIRED_RAW_FIELDS] = stock_df[REQUIRED_RAW_FIELDS].fillna(method='ffill', limit=2) # Limit ffill to avoid propagating errors too far

        # --- Feature Calculation ---
        features = pd.DataFrame(index=stock_df.index)

        # Keep original data as features too
        features['open'] = stock_df['open']
        features['high'] = stock_df['high']
        features['low'] = stock_df['low']
        features['close'] = stock_df['close']
        features['volume'] = stock_df['volume']
        features['amount'] = stock_df['amount']

        # Use already cleaned/converted series
        close_prices = stock_df['close']
        high_prices = stock_df['high']
        low_prices = stock_df['low']
        volume = stock_df['volume']

        # Moving Averages
        features['ma5'] = calculate_ma(close_prices, 5)
        features['ma10'] = calculate_ma(close_prices, 10)
        features['ma20'] = calculate_ma(close_prices, 20)
        features['ma60'] = calculate_ma(close_prices, 60)
        features['ma120'] = calculate_ma(close_prices, 120)

        # RSI
        features['rsi6'] = calculate_rsi(close_prices, 6)
        features['rsi12'] = calculate_rsi(close_prices, 12)
        features['rsi24'] = calculate_rsi(close_prices, 24)

        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = calculate_macd(close_prices)

        # Bollinger Bands
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = calculate_bollinger_bands(close_prices)
        # Calculate BB Width and Position (handle division by zero)
        bb_width_denom = features['bb_middle'].replace(0, np.nan)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_width_denom
        bb_range = (features['bb_upper'] - features['bb_lower']).replace(0, np.nan)
        features['bb_position'] = (features['close'] - features['bb_lower']) / bb_range

        # ATR
        features['atr14'] = calculate_atr(high_prices, low_prices, close_prices, 14)

        # Support/Resistance
        features['support14'], features['resistance14'] = calculate_support_resistance(low_prices, high_prices, 14)

        # Volume MA
        features['volume_ma20'] = calculate_ma(volume, 20)

        # --- Add more features as needed ---

        # Price change %
        features['change_pct'] = close_prices.pct_change() * 100
        # Volatility (Annualized Rolling Std Dev of pct change)
        pct_change = close_prices.pct_change()
        features['volatility20'] = pct_change.rolling(window=20, min_periods=15).std() * np.sqrt(252) # Use 15 min periods
        features['volatility60'] = pct_change.rolling(window=60, min_periods=45).std() * np.sqrt(252) # Use 45 min periods

        # Add stock code back for easy merging later
        features['stock_code'] = stock_code

        # --- Final Cleanup ---
        # Drop rows where essential calculations result in NaNs at the beginning due to lookback
        # Use 'close' as a reference - if it's NaN, the row is likely invalid original data
        # Use a feature with a long lookback like ma120 to determine the valid start
        # features = features.dropna(subset=['close', 'ma120']) # Drop rows if close OR ma120 is NaN

        # Drop initial rows based on the longest lookback period used
        # This ensures all features have had enough data to be calculated
        first_valid_index = features['ma120'].first_valid_index() # Assuming MA120 is longest lookback
        if first_valid_index is not None:
             features = features.loc[first_valid_index:]
        else:
             # If MA120 is all NaN (e.g., less than 120 data points total)
             # Fallback to a shorter lookback or return empty
             first_valid_idx_short = features['ma20'].first_valid_index()
             if first_valid_idx_short is not None:
                  features = features.loc[first_valid_idx_short:]
             else:
                   logger.warning(f"Could not determine a valid starting index for features of {stock_code}. Insufficient data?")
                   return pd.DataFrame()


        if features.empty:
             logger.warning(f"No valid feature rows remaining for {stock_code} after dropping initial NaNs.")
             return pd.DataFrame()

        logger.debug(f"Finished calculating features for {stock_code}. Shape after cleanup: {features.shape}")
        return features.reset_index() # Reset index to have 'date' as a column

    except Exception as e:
        logger.error(f"Unhandled error calculating features for stock {stock_code}: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame() # Return empty DataFrame on error

# --- Main Generation Logic ---

def generate_features_for_stock_wrapper(args):
    """ Wrapper function for parallel processing """
    stock_code, stock_data_df = args
    if stock_data_df is None or stock_data_df.empty:
        logger.warning(f"Received empty data for stock {stock_code} in wrapper.")
        return pd.DataFrame()

    logger.debug(f"Processing stock via wrapper: {stock_code}")
    try:
        # Ensure data is sorted by date BEFORE setting index
        stock_data_df = stock_data_df.sort_values('date').copy()
        # Set 'date' as index for calculations
        stock_data_df = stock_data_df.set_index('date')
        # Add stock_code column for the calculation function
        stock_data_df['stock_code'] = stock_code
        result_df = calculate_all_features_for_stock(stock_data_df) # result_df will have 'date' as a column now
        if not result_df.empty:
             logger.debug(f"Finished wrapper for stock: {stock_code}, Features calculated: {len(result_df)}")
             # Ensure 'date' column exists before returning
             if 'date' not in result_df.columns:
                  logger.error(f"Wrapper result for {stock_code} missing 'date' column.")
                  return pd.DataFrame()
             return result_df
        else:
             logger.warning(f"Stock {stock_code} resulted in empty features DataFrame.")
             return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in wrapper for stock {stock_code}: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

def generate_all_features(start_date_str, end_date_str, stock_list=None, max_workers=None):
    """
    Generates and saves features for the specified date range and stocks.
    """
    logger.info(f"Starting feature generation from {start_date_str} to {end_date_str}")

    # --- 1. Initialization ---

    start_total_time = time.time()
    storage = ParquetDataStorage()
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2) # Use half the cores by default
        logger.info(f"Using default max_workers: {max_workers}")
    elif max_workers <= 0:
         max_workers = 1
         logger.info(f"Adjusted max_workers to: {max_workers}")

    if not stock_list:
        stock_list = storage.metadata.get('stock_list')
        if not stock_list:
            logger.info("Stock list not found in metadata, downloading...")
            stock_list = storage.download_stock_list()
            if not stock_list:
                logger.error("Failed to get stock list. Exiting.")
                return
    logger.info(f"Processing {len(stock_list)} stocks.")

    # --- 2. Determine Date Range & Load Raw Data ---

    try:
        start_date_dt = datetime.strptime(start_date_str, '%Y%m%d')
        end_date_dt = datetime.strptime(end_date_str, '%Y%m%d')
        # Add buffer for lookback calculation
        # Calculate buffer needed based on MAX_LOOKBACK and potential non-trading days
        buffer_days = int(MAX_LOOKBACK * 1.7) # Estimate ~70% trading days + buffer
        required_start_date = start_date_dt - timedelta(days=buffer_days)
        required_start_date_str = required_start_date.strftime('%Y%m%d')
        logger.info(f"Required raw data range: {required_start_date_str} to {end_date_str} (includes buffer for lookback)")
    except ValueError:
        logger.error(f"Invalid date format. Please use YYYYMMDD.")
        return

    logger.info("Loading raw daily data...")
    load_start_time = time.time()

    raw_daily_data = storage.get_daily_data(
        stock_list=stock_list,
        start_date=required_start_date_str,
        end_date=end_date_str,
        fields=REQUIRED_RAW_FIELDS
    )
    load_end_time = time.time()

    if raw_daily_data is None or raw_daily_data.empty:
        logger.error(f"未能加载到 {required_start_date_str} 至 {end_date_str} 的原始日线数据。")
        logger.error(f"请检查 'to/local_data/daily/' 目录下是否包含对应日期分区的数据。")
        logger.error(f"如果数据缺失，请先使用 'advanced_quant_data.py' 或相关功能下载所需日期的原始数据。")
        # Optional: Attempt download if configured
        # logger.info("Attempting to download missing daily data...")
        # success = storage.download_daily_data(stock_list, required_start_date_str, end_date_str)
        # if success: raw_daily_data = storage.get_daily_data(...) etc.
        return

    logger.info(f"Raw daily data loaded ({len(raw_daily_data):,} rows). Time: {load_end_time - load_start_time:.2f}s")

    # --- Data Preparation ---
    try:
        # Convert 'date_str' to datetime objects for filtering and indexing
        raw_daily_data['date'] = pd.to_datetime(raw_daily_data['date_str'], errors='coerce')
        raw_daily_data = raw_daily_data.dropna(subset=['date']) # Drop rows where date conversion failed
        # Sort data globally by stock and date - IMPORTANT for rolling calculations
        raw_daily_data = raw_daily_data.sort_values(by=['stock_code', 'date'])
        logger.info("Raw data date conversion and sorting completed.")
    except Exception as e:
        logger.error(f"Error preparing raw data (date conversion/sorting): {e}")
        return

    # --- 3. Parallel Feature Calculation per Stock ---

    logger.info(f"Calculating features in parallel using {max_workers} workers...")
    calculation_start_time = time.time()

    all_feature_dfs = []

    # Group data *after* sorting
    grouped_data = raw_daily_data.groupby('stock_code')
    # Filter groups with insufficient data BEFORE submitting to pool
    tasks = []
    min_data_points = MAX_LOOKBACK # Need at least MAX_LOOKBACK points for longest feature
    for stock_code, group in grouped_data:
         if len(group) >= min_data_points:
              tasks.append((stock_code, group.copy())) # Pass copy to avoid modifying original df in parallel
         else:
              logger.warning(f"Stock {stock_code} has insufficient data ({len(group)} < {min_data_points}), skipping.")

    del raw_daily_data # Free memory
    gc.collect()

    processed_count = 0
    total_valid_stocks = len(tasks)
    logger.info(f"Submitting {total_valid_stocks} stocks with sufficient data for parallel processing.")
    futures_dict = {}

    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit initial batch of tasks
        for i in range(min(total_valid_stocks, max_workers * 5)): # Submit a larger initial batch
            task = tasks[i]
            stock_code = task[0]
            futures_dict[executor.submit(generate_features_for_stock_wrapper, task)] = stock_code

        tasks_submitted = len(futures_dict)

        while futures_dict:
            # Wait for next future to complete
            try:
                # Use a timeout to prevent hanging indefinitely
                completed_futures = as_completed(futures_dict, timeout=600) # 10 min timeout per future?
                future = next(completed_futures)
                stock_code = futures_dict.pop(future)
            except TimeoutError:
                logger.warning(f"Timeout waiting for a future to complete. Processed: {processed_count}/{total_valid_stocks}")
                # Potentially log which futures are still running or attempt to cancel them
                continue # Continue waiting for others
            except StopIteration: # Handle empty futures_dict if somehow reached
                 logger.info("StopIteration received from as_completed, likely all tasks done.")
                 break
            except Exception as wait_err:
                 logger.error(f"Error waiting for future completion: {wait_err}")
                 # It's hard to know which future caused the error here.
                 # We might need to break or implement more robust tracking.
                 break

            try:
                stock_feature_df = future.result() # Get the result
                if stock_feature_df is not None and not stock_feature_df.empty:
                    all_feature_dfs.append(stock_feature_df)
                    logger.debug(f"Successfully processed features for {stock_code}")
                # else: # Already logged in wrapper if empty
                #     pass

            except Exception as e:
                # Log error from the future's execution
                logger.error(f"Error retrieving result for stock {stock_code}: {e}")
                logger.debug(traceback.format_exc())
            finally:
                processed_count += 1
                if processed_count % 100 == 0 or processed_count == total_valid_stocks:
                     logger.info(f"Feature calculation progress: {processed_count}/{total_valid_stocks} stocks processed.")

                # Submit next task if available
                if tasks_submitted < total_valid_stocks:
                    next_task = tasks[tasks_submitted]
                    next_stock_code = next_task[0]
                    try:
                         futures_dict[executor.submit(generate_features_for_stock_wrapper, next_task)] = next_stock_code
                         tasks_submitted += 1
                    except Exception as submit_err:
                         logger.error(f"Failed to submit task for {next_stock_code}: {submit_err}")
                         # How to handle? Skip this stock or retry? For now, just log.

    calculation_end_time = time.time()
    logger.info(f"Feature calculation finished. Time: {calculation_end_time - calculation_start_time:.2f}s")

    if not all_feature_dfs:
        logger.error("No features were calculated successfully across all stocks. Exiting.")
        return

    # --- 4. Combine and Save Features by Date ---

    logger.info(f"Combining features from {len(all_feature_dfs)} stock DataFrames...")
    combine_start_time = time.time()
    try:
        # Filter out any potential None or empty DataFrames again just to be safe
        valid_feature_dfs = [df for df in all_feature_dfs if isinstance(df, pd.DataFrame) and not df.empty]
        if not valid_feature_dfs:
             logger.error("No valid feature DataFrames remaining after filtering.")
             return

        combined_features = pd.concat(valid_feature_dfs, ignore_index=True)
        logger.info(f"Combined feature DataFrame shape: {combined_features.shape}")

        # Ensure 'date' column is datetime type before filtering
        if 'date' not in combined_features.columns:
             logger.error("Combined features DataFrame is missing the crucial 'date' column!")
             return
        combined_features['date'] = pd.to_datetime(combined_features['date'], errors='coerce')
        combined_features = combined_features.dropna(subset=['date'])

    except Exception as e:
        logger.error(f"Error combining feature DataFrames: {e}")
        logger.debug(traceback.format_exc())
        return
    finally:
         del all_feature_dfs # Free memory
         del valid_feature_dfs
         gc.collect()

    # Filter for the target date range (inclusive) AFTER combining
    combined_features = combined_features[
         (combined_features['date'] >= start_date_dt) &
         (combined_features['date'] <= end_date_dt)
    ].copy() # Use .copy()

    if combined_features.empty:
        logger.error(f"No feature data found within the target date range {start_date_str} - {end_date_str}. Exiting.")
        return

    # Create 'date_str' for partitioning/saving
    combined_features['date_str'] = combined_features['date'].dt.strftime('%Y%m%d')

    combine_end_time = time.time()
    logger.info(f"Features combined and filtered for target range. Time: {combine_end_time - combine_start_time:.2f}s")

    logger.info("Saving features by date...")
    save_start_time = time.time()

    # Get unique dates within the final features to save
    dates_to_save = sorted(combined_features['date_str'].unique())
    logger.info(f"Found {len(dates_to_save)} unique dates with features to save.")

    saved_count = 0
    error_count = 0
    # Make sure output dir exists
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by date and save each group
    for date_str, group_df in combined_features.groupby('date_str'):
        try:
            output_file = FEATURE_OUTPUT_DIR / f"{date_str}.parquet"
            df_to_save = group_df.copy()

            # --- Type Optimization for Parquet ---
            for col in df_to_save.columns:
                # Skip non-numeric like stock_code, date_str
                if col in ['stock_code', 'date_str', 'date']:
                     continue

                # Handle potential 'object' dtype after calculations/joins
                if df_to_save[col].dtype == 'object':
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce') # Coerce errors to NaN

                # Convert pandas Boolean to numpy bool
                if str(df_to_save[col].dtype) == 'boolean':
                    df_to_save[col] = df_to_save[col].astype('bool')

                # Downcast floats
                if df_to_save[col].dtype == 'float64':
                    df_to_save[col] = df_to_save[col].astype('float32')
                # Downcast integers
                elif df_to_save[col].dtype == 'int64':
                    if not df_to_save[col].isnull().any(): # Cannot convert int with NaNs easily
                         min_val, max_val = df_to_save[col].min(), df_to_save[col].max()
                         if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                              df_to_save[col] = df_to_save[col].astype('int32')
                         elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                              df_to_save[col] = df_to_save[col].astype('int16')
            # --- End Type Optimization ---

            # Drop the original datetime column if desired, keep date_str
            if 'date' in df_to_save.columns:
                 df_to_save = df_to_save.drop(columns=['date'])

            # Use pyarrow engine for better type handling and compression
            table = pa.Table.from_pandas(df_to_save, preserve_index=False)
            pq.write_table(table, output_file, compression='snappy')

            saved_count += 1
            if saved_count % 50 == 0 or saved_count == len(dates_to_save):
                logger.info(f"Saved {saved_count}/{len(dates_to_save)} date files...")
                gc.collect() # Collect garbage periodically during long save loop

        except Exception as e:
            logger.error(f"Error saving features for date {date_str}: {e}")
            logger.debug(traceback.format_exc())
            error_count += 1
        finally:
             del group_df # Free memory
             if 'df_to_save' in locals(): del df_to_save
             # gc.collect() # Can add GC here too if memory is tight

    save_end_time = time.time()
    logger.info(f"Saving finished. Time: {save_end_time - save_start_time:.2f}s")
    logger.info(f"Successfully saved {saved_count} daily feature files. Errors: {error_count}")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    logger.info(f"Total feature generation time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    logger.info(f"Features saved to: {FEATURE_OUTPUT_DIR}")

# --- Command Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Feature Factory")
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date for feature generation (YYYYMMDD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="End date for feature generation (YYYYMMDD)"
    )
    parser.add_argument(
        "--stocks",
        type=str,
        default=None,
        help="Comma-separated list of stock codes (e.g., 000001.SZ,600000.SH). Default: All stocks from metadata."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for calculation. Default: Half CPU cores"
    )

    args = parser.parse_args()

    stock_list_arg = None
    if args.stocks:
        stock_list_arg = [s.strip() for s in args.stocks.split(',') if s.strip()]
        logger.info(f"Processing specific stocks: {stock_list_arg}")

    # --- Execute Main Function ---
    generate_all_features(
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        stock_list=stock_list_arg,
        max_workers=args.workers
    )
    logger.info("Feature generation script finished.")
