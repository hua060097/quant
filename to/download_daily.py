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
MAX_WORKERS = 5  # 推荐 4-8，10000积分用户建议设为 5-6
API_RETRY_COUNT = 7 # 最大重试次数

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
    filtered_stocks = [stock for stock in all_stocks if not stock.endswith('.BJ')]

    logger.info(f"原始股票数量: {len(all_stocks)}")
    logger.info(f"过滤后股票数量（排除北交所）: {len(filtered_stocks)}")
    return filtered_stocks

def retry_tushare_call(func):
    """
    [核心优化] Tushare 接口调用的智能重试装饰器
    处理网络超时和接口限流问题
    """
    def wrapper(*args, **kwargs):
        backoff = 1
        for i in range(API_RETRY_COUNT):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                # 检查是否是频率限制错误
                if "每分钟最多访问" in error_msg or "frequency" in error_msg:
                    sleep_time = 60 + random.random() * 5 # 遇到分钟限制，强制休眠1分钟以上
                    logger.warning(f"触发Tushare流控，休眠 {sleep_time:.1f} 秒后重试 (第 {i+1} 次)...")
                    time.sleep(sleep_time)
                    backoff = 1 # 重置退避
                elif "每小时" in error_msg:
                    logger.error("触发小时级流控，程序将长时间休眠...")
                    time.sleep(3600)
                else:
                    # 其他网络错误，指数退避
                    sleep_time = backoff + random.random()
                    logger.warning(f"API调用异常: {e}，{sleep_time:.1f} 秒后重试...")
                    time.sleep(sleep_time)
                    backoff = min(backoff * 2, 30) # 最大等待30秒
        
        # 重试耗尽，抛出最后一次异常
        logger.error(f"重试 {API_RETRY_COUNT} 次后依然失败")
        raise Exception(f"API Call Failed after retries")
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
    progress_file: str = 'local_data/download_progress_tushare.pkl'
) -> bool:
    """
    多线程下载主函数
    """
    logger.info("=" * 100)
    logger.info("开始从Tushare下载日线数据 (极速多线程版)")
    logger.info("=" * 100)
    logger.info(f"并发线程数: {MAX_WORKERS}")
    logger.info(f"股票数量: {len(stock_list)}")
    
    # 初始化tushare API (在主线程初始化)
    ts.set_token('022fe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c10000') 
    pro = ts.pro_api()
    df_st = pro.stock_st(trade_date='20250813')

    if df_st is not None and len(df_st) > 0:
        st_list = df_st['ts_code'].tolist()
        logger.info(f"ST股票数量: {len(st_list)}")
        # 过滤ST股票
        stock_list = [stock for stock in stock_list if stock not in st_list]
        logger.info(f"过滤后股票数量: {len(stock_list)}")
    else:
        logger.warning("未获取到ST股票数据，跳过ST股票过滤")

    if '600721.SH' in stock_list:
        logger.info(f"股票600721需要下载")

    # 加载进度
    progress = _load_progress(progress_file) if resume else None
    if progress:
        logger.info(f"断点续传: 已有 {len(progress.get('completed_stocks', {}))} 个分区记录")
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
    
    for i in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(stock_list) + batch_size - 1) // batch_size

        logger.info(f"\n[大批次 {batch_num}/{total_batches}] 启动并发下载 {len(batch_stocks)} 只股票...")
        
        batch_data_list = []
        
        # === 核心优化：线程池并发下载 ===
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
                        if stock_code == '600721.SH':
                            logger.warning(f"⚠️ [600721.SH] 下载成功但数据为空（可能停牌或未上市）")
                        pass
                except Exception as e:
                    logger.error(f"股票 {stock_code} 下载最终失败: {e}")
                    if stock_code == '600721.SH':
                        logger.error(f"❌ [600721.SH] 下载失败: {e}")
                    failed_stocks.append(stock_code)

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
    
    os.makedirs('local_data/log', exist_ok=True)
    os.makedirs('local_data/daily', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=str, default=None)
    parser.add_argument('--end', '-e', type=str, default=None)
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
        batch_size=500 
    )