"""
从Tushare下载日线数据并按月存储为Parquet格式 (多线程并发优化修复版)

修复说明:
1. ✅ 修复 AttributeError: PartitionCache 类增加了 clear() 方法。
2. ✅ 增强日志：下载结束后打印所有下载失败的股票代码。
3. ✅ 保持多线程并发与智能重试机制。

"""

import xtquant.xtdata as xtdata
from datetime import datetime, timedelta
import time
import os
import pandas as pd
import numpy as np
import threading
from data_loaders import get_stock_list
import tushare as ts
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# 尝试导入 tqdm 用于进度条，如果没有则使用普通日志
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("to/log/download_tushare_daily.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("download_tushare_daily")

# 全局配置
# pro_bar(adj='qfq') 内部会额外访问 adj_factor。并发过高时很容易触发 adj_factor 500次/分钟限制。
MAX_WORKERS = 4
API_RETRY_COUNT = 20
# pro_bar(qfq) 通常会触发 daily + adj_factor 等内部请求。
# 0.18s 约等于 333次pro_bar/分钟，仍低于 adj_factor 500次/分钟上限；触发限流后会自动降速。
API_MIN_INTERVAL_SECONDS = 0.18
API_MIN_INTERVAL_FLOOR_SECONDS = 0.16
API_MIN_INTERVAL_CEILING_SECONDS = 0.45
API_RATE_LIMIT_COOLDOWN_SECONDS = 50


class TushareRateLimiter:
    """进程内全局自适应限速，避免多个下载线程同时打到 pro_bar/adj_factor 限流。"""
    def __init__(
        self,
        min_interval_seconds: float,
        floor_seconds: float = API_MIN_INTERVAL_FLOOR_SECONDS,
        ceiling_seconds: float = API_MIN_INTERVAL_CEILING_SECONDS,
    ):
        self.floor_seconds = max(0.01, float(floor_seconds))
        self.ceiling_seconds = max(self.floor_seconds, float(ceiling_seconds))
        self.min_interval_seconds = min(max(float(min_interval_seconds), self.floor_seconds), self.ceiling_seconds)
        self._lock = threading.Lock()
        self._last_call_ts = 0.0
        self._cooldown_until_ts = 0.0
        self._success_since_adjust = 0

    def wait(self):
        with self._lock:
            now = time.time()
            interval_wait = self.min_interval_seconds - (now - self._last_call_ts)
            cooldown_wait = self._cooldown_until_ts - now
            wait_seconds = max(interval_wait, cooldown_wait)
            if wait_seconds > 0:
                time.sleep(wait_seconds + random.random() * 0.05)
            self._last_call_ts = time.time()

    def cooldown(self, seconds: float):
        with self._lock:
            self._cooldown_until_ts = max(self._cooldown_until_ts, time.time() + float(seconds))

    def record_success(self):
        with self._lock:
            self._success_since_adjust += 1
            if self._success_since_adjust >= 120 and self.min_interval_seconds > self.floor_seconds:
                old_interval = self.min_interval_seconds
                self.min_interval_seconds = max(self.floor_seconds, self.min_interval_seconds * 0.96)
                self._success_since_adjust = 0
                if old_interval - self.min_interval_seconds >= 0.005:
                    logger.info(
                        f"[限流] 连续成功，API间隔自动下调: "
                        f"{old_interval:.3f}s -> {self.min_interval_seconds:.3f}s"
                    )

    def record_rate_limit(self):
        with self._lock:
            old_interval = self.min_interval_seconds
            self.min_interval_seconds = min(self.ceiling_seconds, max(self.min_interval_seconds * 1.25, self.min_interval_seconds + 0.03))
            self._success_since_adjust = 0
            logger.warning(
                f"[限流] 检测到频率限制，API间隔自动上调: "
                f"{old_interval:.3f}s -> {self.min_interval_seconds:.3f}s"
            )

    def current_interval(self) -> float:
        with self._lock:
            return float(self.min_interval_seconds)


TUSHARE_RATE_LIMITER = TushareRateLimiter(API_MIN_INTERVAL_SECONDS)


def _is_tushare_rate_limit_error(error_msg: str) -> bool:
    rate_limit_keywords = [
        '频率超限',
        '每分钟',
        '每小时',
        '500次/分钟',
        '访问接口',
        'frequency',
        'rate limit',
        'limit',
    ]
    lower_msg = str(error_msg).lower()
    return any(keyword.lower() in lower_msg for keyword in rate_limit_keywords)

def get_stock_list_from_sw_sector():
    """获取所有股票列表，排除北交所"""
    # 路径根据您的实际环境可能需要微调
    json_path = Path('local_data/sw_sector2/sw_sector.json')

    if not json_path.exists():
        # 如果找不到文件，尝试从相对路径查找
        json_path = Path('../local_data/sw_sector2/sw_sector.json')
        if not json_path.exists():
             logger.warning(f"申万行业JSON文件未找到，尝试直接获取全市场股票")
             return []

    with open(json_path, 'r', encoding='utf-8') as f:
        sector_data = json.load(f)

    all_stocks = list(sector_data.keys())
    # filtered_stocks = [stock for stock in all_stocks if not stock.endswith('.BJ')]

    logger.info(f"原始股票数量: {len(all_stocks)}")
    # logger.info(f"过滤后股票数量（排除北交所）: {len(filtered_stocks)}")
    # return filtered_stocks
    return all_stocks

def retry_tushare_call(func):
    """
    [核心优化] Tushare 接口调用的智能重试装饰器
    处理网络超时和接口限流问题
    """
    def wrapper(*args, **kwargs):
        backoff = 1
        last_exception = None
        for i in range(API_RETRY_COUNT):
            try:
                TUSHARE_RATE_LIMITER.wait()
                result = func(*args, **kwargs)
                TUSHARE_RATE_LIMITER.record_success()
                return result
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                # 检查是否是频率限制错误
                if _is_tushare_rate_limit_error(error_msg):
                    TUSHARE_RATE_LIMITER.record_rate_limit()
                    sleep_time = API_RATE_LIMIT_COOLDOWN_SECONDS + random.random() * 12
                    TUSHARE_RATE_LIMITER.cooldown(sleep_time)
                    logger.warning(
                        f"触发Tushare流控: {error_msg}；休眠 {sleep_time:.1f} 秒后重试 "
                        f"(第 {i+1}/{API_RETRY_COUNT} 次)..."
                    )
                    time.sleep(sleep_time)
                    backoff = 1 # 重置退避
                else:
                    # 其他网络错误，指数退避
                    sleep_time = backoff + random.random()
                    logger.warning(f"API调用异常: {e}，{sleep_time:.1f} 秒后重试 (第 {i+1}/{API_RETRY_COUNT} 次)...")
                    time.sleep(sleep_time)
                    backoff = min(backoff * 2, 30) # 最大等待30秒
        
        # 重试耗尽，抛出最后一次异常
        logger.error(f"重试 {API_RETRY_COUNT} 次后依然失败")
        raise RuntimeError(f"API Call Failed after {API_RETRY_COUNT} retries: {last_exception}") from last_exception
    return wrapper

@retry_tushare_call
def fetch_single_stock(stock_code, adj, start_date, end_date):
    """单独下载一只股票的数据，包裹在重试逻辑中"""
    return ts.pro_bar(
        ts_code=stock_code,
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )

def download_daily_from_tushare(
    stock_list: List[str],
    start_date: str,
    end_date: str,
    adj: str = 'qfq',
    batch_size: int = 500, # 批次可以设大一点，因为内部是并发的
    rows_per_save: int = 50000,
    resume: bool = True,
    progress_file: str = 'local_data/download_progress_tushare.pkl',
    max_workers: Optional[int] = None,
    api_min_interval_seconds: Optional[float] = None,
) -> bool:
    """
    多线程下载主函数
    """
    global TUSHARE_RATE_LIMITER
    effective_max_workers = max(1, int(max_workers or MAX_WORKERS))
    effective_api_interval = float(api_min_interval_seconds or API_MIN_INTERVAL_SECONDS)
    TUSHARE_RATE_LIMITER = TushareRateLimiter(effective_api_interval)

    logger.info("=" * 100)
    logger.info("开始从Tushare下载日线数据 (极速多线程版)")
    logger.info("=" * 100)
    logger.info(f"并发线程数: {effective_max_workers}")
    logger.info(
        f"API最小调用间隔: {effective_api_interval:.2f} 秒 "
        f"(自适应范围 {API_MIN_INTERVAL_FLOOR_SECONDS:.2f}~{API_MIN_INTERVAL_CEILING_SECONDS:.2f} 秒)"
    )
    logger.info(f"API最大重试次数: {API_RETRY_COUNT}")
    logger.info(f"股票数量: {len(stock_list)}")
    
    # 初始化tushare API (在主线程初始化)
    # ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq') 
    ts.set_token('gg99b01a244311f7742512463d9e1588954d174b86d695f8a9f1cc4b')
    pro = ts.pro_api()
    # df_st = pro.stock_st(trade_date='20250813')

    # if df_st is not None and len(df_st) > 0:
    #     st_list = df_st['ts_code'].tolist()
    #     logger.info(f"ST股票数量: {len(st_list)}")
    #     # 过滤ST股票
    #     stock_list = [stock for stock in stock_list if stock not in st_list]
    #     logger.info(f"过滤后股票数量: {len(stock_list)}")
    # else:
    #     logger.warning("未获取到ST股票数据，跳过ST股票过滤")

    # if '600721.SH' in stock_list:
    #     logger.info(f"股票600721需要下载")

    # 加载进度
    progress = _load_progress(progress_file) if resume else None
    if progress:
        logger.info(f"断点续传: 已有 {len(progress.get('completed_stocks', {}))} 个分区记录")
        progress['failed_stocks'] = []
    else:
        progress = {
            'completed_stocks': {}, 
            'failed_stocks': [],
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    capital_df = _load_capital_data()
    partition_cache = PartitionCache()

    accumulated_data = []
    total_rows = 0
    failed_stocks = []
    expected_dates_by_stock = {}
    empty_stocks = set()
    
    for i in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(stock_list) + batch_size - 1) // batch_size

        logger.info(f"\n[大批次 {batch_num}/{total_batches}] 启动并发下载 {len(batch_stocks)} 只股票...")
        
        batch_data_list = []
        
        # === 核心优化：线程池并发下载 ===
        with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            # 提交任务
            future_to_stock = {
                executor.submit(fetch_single_stock, stock, adj, start_date, end_date): stock 
                for stock in batch_stocks
            }
            
            # 处理结果（使用 tqdm 显示进度条）
            iterator = as_completed(future_to_stock)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(batch_stocks), desc=f"批次进度", unit="股")
            
            for future in iterator:
                stock_code = future_to_stock[future]
                try:
                    df_raw = future.result()
                    if df_raw is not None and not df_raw.empty:
                        # 转换数据格式
                        df_trans = _transform_tushare_data(df_raw, stock_code)
                        expected_dates_by_stock[stock_code] = set(df_trans['date_str'].astype(str).tolist())

                        # 🐛 调试日志：600721.SH 下载详情
                        if stock_code == '600721.SH':
                            logger.info("=" * 80)
                            logger.info(f"🔍 [600721.SH] 下载成功详情")
                            logger.info("=" * 80)
                            logger.info(f"  日期范围: {df_trans['date'].min()} 至 {df_trans['date'].max()}")
                            logger.info(f"  数据行数: {len(df_trans)}行")
                            logger.info(f"  日期明细:")
                            for _, row in df_trans.iterrows():
                                logger.info(f"    {row['date_str']}: close={row['close']:.2f}, "
                                          f"open={row['open']:.2f}, high={row['high']:.2f}, low={row['low']:.2f}, "
                                          f"volume={row['volume']}")
                            logger.info("=" * 80)

                        batch_data_list.append(df_trans)
                    else:
                        # 空数据不一定是失败，可能是停牌或未上市
                        empty_stocks.add(stock_code)
                        if stock_code == '600721.SH':
                            logger.warning(f"⚠️ [600721.SH] 下载成功但数据为空（可能停牌或未上市）")
                        pass
                except Exception as e:
                    logger.error(f"股票 {stock_code} 下载最终失败: {e}")
                    if stock_code == '600721.SH':
                        logger.error(f"❌ [600721.SH] 下载失败: {e}")
                    failed_stocks.append(stock_code)
                    for pending_future in future_to_stock:
                        if pending_future is not future:
                            pending_future.cancel()
                    raise RuntimeError(f"股票 {stock_code} 下载失败，程序终止") from e

        # 合并当前批次数据
        if batch_data_list:
            batch_df = pd.concat(batch_data_list, ignore_index=True)
            # 去重
            batch_df = batch_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
            
            accumulated_data.append(batch_df)
            total_rows += len(batch_df)
            
            logger.info(f"批次完成，获取有效数据 {len(batch_df)} 行")

        # 检查是否需要保存 (内存阈值)
        if total_rows >= rows_per_save or (i + batch_size) >= len(stock_list):
            logger.info(f"正在写入磁盘 (累积 {total_rows} 行)...")
            if accumulated_data:
                all_data_df = pd.concat(accumulated_data, ignore_index=True)
                
                # 全局去重
                all_data_df = all_data_df.drop_duplicates(subset=['stock_code', 'date_str'], keep='first')
                
                # 计算衍生字段
                all_data_df = _calculate_derived_fields(all_data_df, capital_df)
                
                # 写入缓存
                _save_by_partition_with_cache(all_data_df, partition_cache, progress['completed_stocks'])
                
                # 清理
                del all_data_df
                accumulated_data = []
                total_rows = 0
        
        # 实时保存进度
        progress['failed_stocks'] = list(set(progress.get('failed_stocks', []) + failed_stocks))
        _save_progress(progress, progress_file)

    # 刷写缓存到磁盘
    logger.info(f"\n所有下载完成，正在最终写入磁盘...")
    partition_cache.flush_all()
    partition_cache.clear() # [修复] 现在PartitionCache类中有clear方法了

    _validate_saved_download_data(
        expected_dates_by_stock=expected_dates_by_stock,
        empty_stocks=empty_stocks,
        start_date=start_date,
        end_date=end_date,
        base_dir=Path('local_data/daily')
    )

    progress['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _save_progress(progress, progress_file)

    # [增强] 打印失败的股票列表
    logger.info("=" * 100)
    fail_count = len(progress['failed_stocks'])
    logger.info(f"全部任务完成。失败股票数: {fail_count}")
    
    if fail_count > 0:
        logger.error("以下股票下载失败，请检查或稍后重试:")
        # 每行打印10个股票代码
        fail_list = progress['failed_stocks']
        for i in range(0, len(fail_list), 10):
            logger.error(f"  {', '.join(fail_list[i:i+10])}")
            
    logger.info("=" * 100)
    
    return len(progress['failed_stocks']) == 0

# ==============================================================================
# 辅助类和函数
# ==============================================================================

def _load_progress(progress_file: str) -> Optional[Dict]:
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
    except Exception:
        return None

def _save_progress(progress: Dict, progress_file: str):
    try:
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'wb') as f:
            pickle.dump(progress, f)
    except Exception:
        pass

def _load_capital_data() -> Optional[pd.DataFrame]:
    capital_file = Path('local_data/stock_capital_data.parquet')
    if capital_file.exists():
        return pd.read_parquet(capital_file).rename(columns={'stock': 'stock_code'})
    return None

def _transform_tushare_data(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    df = df.rename(columns={'ts_code': 'stock_code', 'trade_date': 'date'})
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    if 'vol' in df.columns:
        df['volume'] = df['vol'] * 100
        df = df.drop(columns=['vol'])
    if 'amount' in df.columns:
        df['amount'] = df['amount'] * 1000
    df['date_str'] = df['date'].dt.strftime('%Y%m%d')
    return df

def _calculate_derived_fields(df: pd.DataFrame, capital_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if capital_df is not None and 'stock_code' in capital_df.columns:
        df = df.merge(capital_df[['stock_code', 'total_capital', 'circulating_capital']], on='stock_code', how='left')
        df['total_mv'] = df['close'] * df['total_capital']
        df['circ_mv'] = df['close'] * df['circulating_capital']
        mask = df['circulating_capital'] > 0
        df.loc[mask, 'turnover_rate'] = (df.loc[mask, 'volume'] / df.loc[mask, 'circulating_capital']) * 100
        df = df.drop(columns=['total_capital', 'circulating_capital'])
    else:
        df['total_mv'] = None
        df['circ_mv'] = None
        df['turnover_rate'] = None
    return df


def _iter_month_partitions(start_date: str, end_date: str) -> List[str]:
    start_ts = pd.to_datetime(start_date, format='%Y%m%d')
    end_ts = pd.to_datetime(end_date, format='%Y%m%d')
    months = pd.period_range(start_ts, end_ts, freq='M')
    return [str(month).replace('-', '') for month in months]


def _validate_saved_download_data(
    expected_dates_by_stock: Dict[str, set],
    empty_stocks: set,
    start_date: str,
    end_date: str,
    base_dir: Path,
):
    """校验本次接口成功返回的数据是否全部写入本地分区文件。"""
    expected_pair_count = sum(len(dates) for dates in expected_dates_by_stock.values())
    logger.info("=" * 100)
    logger.info("[校验] 开始校验本次下载数据写盘结果")
    logger.info(f"[校验] 接口成功返回股票数: {len(expected_dates_by_stock)}")
    logger.info(f"[校验] 接口返回空数据股票数: {len(empty_stocks)}")
    logger.info(f"[校验] 预期股票-日期记录数: {expected_pair_count:,}")

    if expected_pair_count == 0:
        logger.warning("[校验] 本次没有任何非空下载结果，跳过写盘完整性校验")
        return

    required_columns = {'stock_code', 'date_str', 'date', 'open', 'high', 'low', 'close', 'volume'}
    saved_pairs = set()
    loaded_rows = 0
    missing_partition_files = []

    for partition_key in _iter_month_partitions(start_date, end_date):
        partition_file = base_dir / partition_key / 'daily_data.parquet'
        if not partition_file.exists():
            missing_partition_files.append(str(partition_file))
            continue

        df = pd.read_parquet(partition_file)
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise RuntimeError(f"[校验失败] 分区文件 {partition_file} 缺少必要列: {sorted(missing_columns)}")

        df['date_str'] = df['date_str'].astype(str)
        mask = (df['date_str'] >= start_date) & (df['date_str'] <= end_date)
        df = df.loc[mask, ['stock_code', 'date_str']]
        loaded_rows += len(df)
        saved_pairs.update(zip(df['stock_code'].astype(str), df['date_str'].astype(str)))

    missing_pairs = []
    for stock_code, expected_dates in expected_dates_by_stock.items():
        for date_str in expected_dates:
            if (str(stock_code), str(date_str)) not in saved_pairs:
                missing_pairs.append((stock_code, date_str))

    if missing_pairs:
        logger.error("[校验失败] 接口已成功返回但本地parquet缺失以下股票-日期记录:")
        for stock_code, date_str in missing_pairs[:100]:
            logger.error(f"  {stock_code} {date_str}")
        if len(missing_pairs) > 100:
            logger.error(f"  ... 还有 {len(missing_pairs) - 100} 条缺失记录")
        if missing_partition_files:
            logger.error(f"[校验] 缺失分区文件: {missing_partition_files}")
        raise RuntimeError(f"[校验失败] 本地写盘缺失 {len(missing_pairs)} 条成功下载记录，程序终止")

    if missing_partition_files:
        logger.warning(f"[校验] 以下月份无分区文件，但本次成功返回数据不涉及这些月份: {missing_partition_files}")

    logger.info(f"[校验通过] 已加载本地范围内记录数: {loaded_rows:,}")
    logger.info(f"[校验通过] 本次成功返回的 {expected_pair_count:,} 条股票-日期记录均已写入")
    logger.info("=" * 100)


class PartitionCache:
    """分区数据缓存，管理内存中的Parquet数据"""
    def __init__(self):
        self.cache = {}
        self.base_dir = Path('local_data/daily')

    def load_partition(self, partition_key: str) -> pd.DataFrame:
        if partition_key not in self.cache:
            partition_file = self.base_dir / partition_key / 'daily_data.parquet'
            if partition_file.exists():
                try:
                    self.cache[partition_key] = pd.read_parquet(partition_file)
                except:
                    self.cache[partition_key] = pd.DataFrame()
            else:
                self.cache[partition_key] = pd.DataFrame()
        return self.cache[partition_key]

    def append_data(self, partition_key: str, new_data: pd.DataFrame):
        self.load_partition(partition_key)
        existing_df = self.cache[partition_key]
        
        if len(existing_df) == 0:
            self.cache[partition_key] = new_data
            return len(new_data)
        else:
            combined = pd.concat([existing_df, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['stock_code', 'date_str'], keep='last') 
            self.cache[partition_key] = combined
            return len(combined) - len(existing_df)

    def flush_all(self):
        """将缓存写入磁盘"""
        for partition_key, df in self.cache.items():
            if len(df) == 0: continue
            partition_dir = self.base_dir / partition_key
            partition_dir.mkdir(parents=True, exist_ok=True)
            partition_file = partition_dir / 'daily_data.parquet'
            df.sort_values(['stock_code', 'date']).to_parquet(partition_file, index=False, compression='snappy')
        self.cache = {} # 写入后清空引用

    def clear(self):
        """[新增] 清空缓存方法"""
        self.cache = {}

def _save_by_partition_with_cache(df: pd.DataFrame, partition_cache: PartitionCache, completed_stocks: Dict) -> int:
    df['partition_key'] = df['date'].dt.strftime('%Y%m')
    saved_total = 0
    for partition_key, partition_df in df.groupby('partition_key'):
        saved_total += partition_cache.append_data(partition_key, partition_df)
    return saved_total

# ==============================================================================
# 程序入口
# ==============================================================================
if __name__ == "__main__":
    import argparse

    # ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq') 
    # pro = ts.pro_api()

    # df=pro.stk_auction_o(trade_date='20260417')
    # logger.info(f"{df}")
    
    os.makedirs('local_data/log', exist_ok=True)
    os.makedirs('local_data/daily', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=str, default=None)
    parser.add_argument('--end', '-e', type=str, default=None)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help=f'并发下载线程数，默认 {MAX_WORKERS}')
    parser.add_argument(
        '--api_interval',
        type=float,
        default=API_MIN_INTERVAL_SECONDS,
        help=f'Tushare pro_bar调用最小间隔秒数，默认 {API_MIN_INTERVAL_SECONDS}'
    )
    args = parser.parse_args()

    if args.start is None:
        args.start = datetime.now().strftime('%Y%m%d')
    if args.end is None:
        args.end = datetime.now().strftime('%Y%m%d')

    try:
        stock_list = get_stock_list_from_sw_sector()
    except Exception as e:
        logger.warning(f"获取股票列表失败: {e}，将使用测试列表")
        stock_list = ['000001.SZ', '600000.SH'] 

    download_daily_from_tushare(
        stock_list=stock_list,
        start_date=args.start,
        end_date=args.end,
        adj='qfq',
        batch_size=500,
        max_workers=args.workers,
        api_min_interval_seconds=args.api_interval
    )
