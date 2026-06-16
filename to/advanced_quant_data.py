#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级量化数据处理模块，包含:
1. 使用Parquet格式存储和加载本地数据
2. 将日线/分钟线数据处理为Parquet文件格式
3. 优化数据结构和列式存储
4. 支持快速读取和查询
"""

import json
import logging
import os
import random
import re
import threading
import time
import warnings
import gc  # 添加gc模块导入
import shutil  # 确保shutil也被导入
from datetime import datetime, timedelta
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from xtquant import xtdata
import traceback
from typing import List

from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 添加项目根目录

# 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('log/local_data.log', encoding='utf-8')
#     ]
# )
logger = logging.getLogger(__name__)

# 忽略pandas的SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# 定义数据目录 - 使用绝对路径，避免因调用目录不同导致的路径问题
_SCRIPT_DIR = Path(__file__).resolve().parent  # 获取脚本所在目录 (advanced_quant_data.py所在目录)
# PROJECT_ROOT = _SCRIPT_DIR.parent  # 获取项目根目录 (to/目录)
PROJECT_ROOT = _SCRIPT_DIR

DATA_ROOT = PROJECT_ROOT / "local_data"  # ✅ 绝对路径，不依赖CWD
DAILY_DIR = DATA_ROOT / "daily"
MINUTE_DIR = DATA_ROOT / "minute"
CALENDAR_DIR = DATA_ROOT / "calendar"
METADATA_DIR = DATA_ROOT / "metadata"

# 获取路径工具函数
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

def get_data_path(relative_path=None):
    """获取数据目录路径"""
    # 使用全局的DATA_ROOT，确保路径一致性
    print(f"DATA_ROOT (绝对路径): {DATA_ROOT.absolute()}")
    if relative_path:
        return DATA_ROOT / relative_path
    return DATA_ROOT

# 确保目录存在
for dir_path in [DAILY_DIR, MINUTE_DIR, CALENDAR_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 添加一个用于进程池的适配器函数（放在类外部以支持多进程序列化）
def process_batch_for_multiprocessing(args):
    """
    进程池工作函数适配器 - 用于将参数传递给download_minute_data_process_worker
    
    参数:
        args: 元组，包含(batch_stocks, start_date, end_date, fields, batch_idx, result_queue)
    
    返回:
        处理结果
    """
    # 解包参数并调用已有的worker函数
    batch_stocks, start_date, end_date, fields, batch_idx, _ = args
    # 不使用result_queue，直接返回结果
    return download_minute_data_process_worker(batch_stocks, start_date, end_date, fields, batch_idx, None)

# 定义MultiProcess worker函数，在ParquetDataStorage类之前
def download_minute_data_process_worker(batch_stocks, start_date, end_date, fields, process_id, result_queue):
    """多进程工作函数，用于下载一批股票的5分钟数据
    
    参数:
        batch_stocks: 要下载的股票代码列表
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        fields: 要下载的字段列表
        process_id: 进程ID，用于日志标识
        result_queue: 结果队列，用于返回数据
    """
    import baostock as bs
    import pandas as pd
    import time
    import random
    import os
    import traceback
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(f'download_worker_{process_id}')
    
    process_start = time.time()
    pid = os.getpid()  # 获取真实进程ID
    
    # 设置重试参数
    max_retries = 5  # 最大重试次数
    retry_delay = 2  # 基础重试延迟(秒)
    
    # 进程开始日志
    logger.info(f"进程 {process_id} (PID: {pid}) 开始下载 {len(batch_stocks)} 只股票数据")
    
    # 每个进程独立登录BaoStock
    retry_count = 0
    while retry_count < max_retries:
        try:
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"进程 {process_id} (PID: {pid}) 登录失败: {lg.error_msg}, 重试 {retry_count+1}/{max_retries}")
                retry_count += 1
                time.sleep(retry_delay * (1 + random.random()))  # 指数退避策略
                continue
            logger.info(f"进程 {process_id} (PID: {pid}) 登录成功")
            break
        except Exception as e:
            logger.error(f"进程 {process_id} (PID: {pid}) 登录异常: {str(e)}, 重试 {retry_count+1}/{max_retries}")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"进程 {process_id} (PID: {pid}) 登录失败次数过多，放弃下载")
                if result_queue is not None:
                    result_queue.put((process_id, {}))
                    return
                else:
                    return (process_id, {})
            time.sleep(retry_delay * (1 + random.random()))
    
    # 转换日期格式
    start_date_str = start_date
    end_date_str = end_date
    if '-' in start_date:
        # 如果日期包含'-'，转换为YYYYMMDD格式
        start_date_str = start_date.replace('-', '')
    if '-' in end_date:
        # 如果日期包含'-'，转换为YYYYMMDD格式
        end_date_str = end_date.replace('-', '')
    
    # 转换为BaoStock日期格式 (YYYY-MM-DD)
    bs_start_date = f"{start_date_str[:4]}-{start_date_str[4:6]}-{start_date_str[6:8]}"
    bs_end_date = f"{end_date_str[:4]}-{end_date_str[4:6]}-{end_date_str[6:8]}"
    print(f"bs_start_date: {bs_start_date}, bs_end_date: {bs_end_date}")
    
    # 转换字段列表为BaoStock需要的格式
    bs_fields = ['date', 'time', 'code']
    field_mapping = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'amount': 'amount'
    }
    for field in fields:
        if field in field_mapping:
            bs_fields.append(field_mapping[field])
    bs_fields_str = ",".join(bs_fields)
    
    # 处理批次中的每只股票
    all_results = {}
    for i, stock_code in enumerate(batch_stocks):
        stock_start = time.time()
       
        
        # 转换股票代码格式
        if '.' in stock_code:
            market, code = stock_code.split('.')
            bs_stock_code = f"{market.lower()}.{code}"
        else:
            # 如果已经是BaoStock格式，直接使用
            bs_stock_code = stock_code
        
        logger.info(f"进程 {process_id} 开始下载 {bs_stock_code} ({i+1}/{len(batch_stocks)}) 数据...")
        # 设置重试计数器
        retry_count = 0
        
        # 使用重试机制查询数据
        while retry_count < max_retries:
            try:
                # 查询5分钟线数据
                rs = bs.query_history_k_data_plus(
                    bs_stock_code,
                    bs_fields_str,
                    start_date=bs_start_date,
                    end_date=bs_end_date,
                    frequency="5",  # 5分钟线
                    adjustflag="2"  # 不复权
                )
                
                # if rs.error_code != '0':
                #     logger.error(f"进程 {process_id} 查询 {stock_code} 失败: {rs.error_msg}, 重试 {retry_count+1}/{max_retries}")
                #     retry_count += 1
                #     if retry_count >= max_retries:
                #         break
                #     time.sleep(retry_delay * (1 + random.random()))
                #     continue
                
                # 获取数据
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    logger.warning(f"进程 {process_id} 获取 {stock_code} 数据: 无记录，耗时: {time.time() - stock_start:.2f}秒")
                    break
                
                # 转换为DataFrame
                stock_df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 转换日期时间格式
                stock_df['date_str'] = stock_df['date'].str.replace('-', '')
                stock_df['time_str'] = stock_df['time']
                
                # 正确解析时间格式(BaoStock返回格式为YYYYMMDDHHMMSSsss)
                try:
                    # 时间格式为YYYYMMDDHHMMSSsss，需要提取时分秒
                    if 'time' in stock_df.columns and not stock_df.empty:
                        # 检查时间格式
                        sample_time = stock_df['time'].iloc[0]
                        if len(sample_time) >= 8:  # 至少包含HHMMSS
                            if len(sample_time) > 8:  # 如果格式为YYYYMMDDHHMMSSsss
                                # 提取HHMMSS部分
                                stock_df['time_format'] = stock_df['time'].apply(
                                    lambda x: f"{x[8:10]}:{x[10:12]}:{x[12:14]}" if len(x) >= 14 else "00:00:00"
                                )
                            else:  # 如果格式为HHMMSS
                                stock_df['time_format'] = stock_df['time'].apply(
                                    lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) >= 6 else "00:00:00"
                                )
                        else:
                            # 使用空时间
                            stock_df['time_format'] = "00:00:00"
                            logger.warning(f"进程 {process_id} 时间格式异常: {sample_time}，使用默认时间00:00:00")
                        
                        # 合并日期和时间
                        stock_df['datetime'] = pd.to_datetime(
                            stock_df['date'] + ' ' + stock_df['time_format'],
                            format='%Y-%m-%d %H:%M:%S',
                            errors='coerce'
                        )
                    else:
                        # 只使用日期
                        stock_df['datetime'] = pd.to_datetime(stock_df['date'], errors='coerce')
                        logger.warning(f"进程 {process_id} 未找到时间列，只使用日期")
                
                except Exception as e:
                    logger.error(f"进程 {process_id} 处理时间格式时出错: {e}")
                    logger.error(traceback.format_exc())
                    # 回退到只使用日期
                    stock_df['datetime'] = pd.to_datetime(stock_df['date'], errors='coerce')
                
                # 转换数据类型
                for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    if field in stock_df.columns:
                        stock_df[field] = pd.to_numeric(stock_df[field], errors='coerce')
                
                # 添加股票代码(使用标准格式)
                stock_df['stock_code'] = stock_code
                
                all_results[stock_code] = stock_df
                logger.info(f"进程 {process_id} 成功获取 {stock_code} 数据: {len(stock_df)} 条记录，耗时: {time.time() - stock_start:.2f}秒")
                break  # 成功获取数据，跳出重试循环
                
            except Exception as e:
                logger.error(f"进程 {process_id} 查询 {stock_code} 异常: {str(e)}")
                logger.error(traceback.format_exc())
                retry_count += 1
                if retry_count >= max_retries:
                    break
                time.sleep(retry_delay * (1 + random.random()))
    
    # 登出系统
    try:
        bs.logout()
        logger.info(f"进程 {process_id} (PID: {pid}) 成功登出BaoStock")
    except Exception as e:
        logger.error(f"进程 {process_id} (PID: {pid}) 登出异常: {str(e)}")
    
    process_end = time.time()
    logger.info(f"进程 {process_id} (PID: {pid}) 处理完成，共处理 {len(batch_stocks)} 只股票，获取 {len(all_results)} 只有效数据，总耗时: {process_end - process_start:.2f}秒")
    
    # 根据result_queue是否为None决定返回方式
    if result_queue is not None:
        # 如果有队列，则放入队列
        result_queue.put((process_id, all_results))
        return process_id
    else:
        # 如果没有队列，则直接返回结果
        return (process_id, all_results)

class ParquetDataStorage:
    """使用Parquet格式的本地数据存储和读取类"""
    
    def __init__(self):
        """初始化"""
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """加载元数据"""
        metadata_file = METADATA_DIR / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载元数据失败: {e}")
        
        # 创建新的元数据
        return {
            'last_update': {
                'daily': None,
                'minute': None,
                'calendar': None
            },
            'stock_list': [],
            'partitions': {
                'daily': [],  # 格式: ["202301", "202302", ...]
                'minute': []  # 格式: ["202301", "202302", ...]
            },
            'calendar_range': {
                'start': None,
                'end': None
            }
        }
        
    def _save_metadata(self):
        """保存元数据"""
        metadata_file = METADATA_DIR / "metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def download_stock_list(self):
        """下载股票列表"""
        logger.info("开始获取股票列表...")
        try:
            # 确保API连接
            if not xtdata.get_client().is_connected():
                xtdata.get_client().connect()
            from data_loaders import get_bj_stock_list
            bj_stock_list = get_bj_stock_list()
            # 获取沪深A股股票列表
            all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
            all_stocks = all_stocks + bj_stock_list
            
            logger.info(f"成功获取股票列表，共 {len(all_stocks)} 只股票")
            self.metadata['stock_list'] = all_stocks
            self._save_metadata()
            
            return all_stocks
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_stock_list(self):
        """
        获取股票列表，从元数据中读取
        
        Returns:
            list: 股票代码列表，如果元数据中没有则尝试下载
        """
        if not self.metadata['stock_list']:
            logger.info("元数据中没有股票列表，尝试下载...")
            return self.download_stock_list()
        
        logger.info(f"从元数据获取股票列表，共 {len(self.metadata['stock_list'])} 只股票")
        return self.metadata['stock_list']
    
    def download_trading_calendar(self, start_date=None, end_date=None):
        """下载交易日历数据
        
        参数:
            start_date: 开始日期，格式 'YYYYMMDD'，默认为 '20230101'
            end_date: 结束日期，格式 'YYYYMMDD'，默认为当前日期
        """
        logger.info("开始下载交易日历...")
        
        # 设置默认日期
        if start_date is None:
            start_date = '20230101'
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        # 确保日期格式正确
        try:
            # 如果是datetime对象，转换为YYYYMMDD格式
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y%m%d')
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y%m%d')
                
            # 确保是字符串格式
            start_date = str(start_date)
            end_date = str(end_date)
            
            # 验证日期格式
            if not (len(start_date) == 8 and len(end_date) == 8):
                logger.error(f"日期格式错误: start_date={start_date}, end_date={end_date}")
                return False
                
            logger.info(f"下载交易日历: {start_date} 至 {end_date}")
        except Exception as e:
            logger.error(f"日期格式转换失败: {e}")
            return False
        
        try:
            # 使用Tushare下载交易日历
            import tushare as ts

            # 初始化tushare
            token = '697175c83be329ca8b9c2fd96a58d5fd9bc9377acfe5f70f9c006d49'
            pro = ts.pro_api(token)

            # 下载交易日历
            cal_data = pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'  # 只获取交易日
            )

            if cal_data is None or cal_data.empty:
                logger.warning("Tushare下载交易日历返回空数据")
                return False

            logger.info(f"成功获取Tushare交易日历，共 {len(cal_data)} 个交易日")

            # 提取日期字符串
            calendar = cal_data['cal_date'].tolist()[::-1]
            logger.info(f"日期范围: {calendar[0]} 至 {calendar[-1]}")
            print(f"日期范围: {calendar[0]} 至 {calendar[-1]}")
            
            # 创建DataFrame并保存为Parquet文件
            calendar_df = pd.DataFrame(calendar, columns=['date_str'])
            
            # 确保日期格式正确
            try:
                # 转换为日期类型
                calendar_df['date'] = pd.to_datetime(calendar_df['date_str'], format='%Y%m%d')
                calendar_df['market'] = 'SH'  # 默认为上证市场
            except Exception as e:
                print(f"日期格式转换错误: {e}")
                return False
            
            # 合并现有日历（如果存在）
            calendar_file = CALENDAR_DIR / "trading_calendar.parquet"
            if calendar_file.exists():
                try:
                    existing_calendar = pd.read_parquet(calendar_file)
                    if not existing_calendar.empty:
                        # 合并并去重
                        calendar_df = pd.concat([existing_calendar, calendar_df]).drop_duplicates('date_str').sort_values('date')
                        print(f"合并现有交易日历，更新后共 {len(calendar_df)} 个交易日")
                except Exception as e:
                    print(f"读取现有交易日历失败: {e}")
            
            # 保存为Parquet文件
            calendar_df.to_parquet(calendar_file, index=False)
            
            # 更新元数据
            self.metadata['calendar_range']['start'] = calendar_df['date_str'].min()
            self.metadata['calendar_range']['end'] = calendar_df['date_str'].max()
            self.metadata['last_update']['calendar'] = datetime.now().strftime('%Y%m%d')
            self._save_metadata()
            
            print(f"交易日历保存成功，共 {len(calendar_df)} 个交易日")
            return True
        except Exception as e:
            logger.error(f"下载交易日历失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_partition_key(self, date_str):
        """生成分区键（按月分区）"""
        return date_str[:6]  # 取YYYYMM部分

    def _download_with_tushare(self, start_date: str, end_date: str, stock_list: List[str], trading_dates: List[str]) -> bool:
        """使用Tushare下载数据"""
        try:
            import tushare as ts

            # 初始化tushare
            token = 'e75c6baca18c97b4337dd2f1946da4aa1a6a875804948000017a7525'
            pro = ts.pro_api(token)

            logger.info("开始使用Tushare下载数据...")

            # 转换股票代码格式（从xtdata格式转为tushare格式）
            tushare_stocks = []
            for stock in stock_list:
                if '.SZ' in stock:
                    tushare_stocks.append(stock.replace('.SZ', '.SZ'))
                elif '.SH' in stock:
                    tushare_stocks.append(stock.replace('.SH', '.SH'))
                else:
                    tushare_stocks.append(stock)

            # 分批下载数据（避免超过调用限制）
            batch_size = 100  # 每批100只股票
            all_data = []

            for i in range(0, len(tushare_stocks), batch_size):
                batch_stocks = tushare_stocks[i:i+batch_size]
                logger.info(f"下载批次 {i//batch_size + 1}/{(len(tushare_stocks)-1)//batch_size + 1}: {len(batch_stocks)} 只股票")

                # 下载基础行情数据
                try:
                    daily_data = pro.daily(
                        start_date=start_date,
                        end_date=end_date,
                        fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                    )

                    if daily_data is not None and not daily_data.empty:
                        # 筛选当前批次的股票
                        batch_data = daily_data[daily_data['ts_code'].isin(batch_stocks)]
                        if not batch_data.empty:
                            all_data.append(batch_data)
                            logger.info(f"批次 {i//batch_size + 1} 下载成功: {len(batch_data)} 条记录")
                        else:
                            logger.warning(f"批次 {i//batch_size + 1} 无数据")
                    else:
                        logger.warning(f"批次 {i//batch_size + 1} 下载失败")

                except Exception as e:
                    logger.error(f"批次 {i//batch_size + 1} 下载失败: {e}")
                    continue

                # 添加延迟避免超过调用限制
                time.sleep(0.2)

            if not all_data:
                logger.error("所有批次下载都失败")
                return False

            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"合并数据完成: {len(combined_data)} 条记录")

            # 下载市值数据
            logger.info("下载市值数据...")
            mv_data = self._download_market_value_data(pro, start_date, end_date)

            # 处理和保存数据
            success = self._process_and_save_tushare_data(combined_data, mv_data, trading_dates)

            return success

        except Exception as e:
            logger.error(f"Tushare下载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def download_daily_data_tushare(self, start_date: str, end_date: str, stock_list: List[str] = None) -> bool:
        """
        使用Tushare下载日线数据的主入口方法

        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            stock_list: 股票列表，如果为None则下载所有股票

        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info(f"开始使用Tushare智能下载日线数据: {start_date} - {end_date}")

            # 1. 获取交易日历
            trading_dates = self._get_trading_dates_for_tushare(start_date, end_date)
            if not trading_dates:
                logger.error("无法获取交易日历")
                return False

            logger.info(f"交易日数量: {len(trading_dates)}")

            # 2. 获取股票列表
            if stock_list is None:
                stock_list = self.download_stock_list()

            if not stock_list:
                logger.error("无法获取股票列表")
                return False

            logger.info(f"股票数量: {len(stock_list)}")

            # 3. 精确检测缺失数据
            missing_data_map = self._detect_missing_data_precise(trading_dates, stock_list)

            if not missing_data_map:
                logger.info("所有数据都已存在，无需下载")
                return True

            # 4. 智能下载缺失数据
            success = self._download_missing_data_smart(missing_data_map, stock_list)

            if success:
                logger.info("Tushare数据下载完成")

                # 5. 验证下载结果，用xtdata补充仍然缺失的数据
                remaining_missing = self._detect_missing_data_precise(trading_dates, stock_list)
                if remaining_missing:
                    logger.info(f"Tushare下载后仍有缺失数据，使用xtdata补充: {len(remaining_missing)} 个分区")
                    xtdata_success = self._supplement_with_xtdata(remaining_missing, stock_list)
                    if xtdata_success:
                        logger.info("xtdata补充完成")
                    else:
                        logger.warning("xtdata补充失败，但Tushare主要数据已下载")

                return True
            else:
                logger.error("Tushare数据下载失败")
                return False

        except Exception as e:
            logger.error(f"Tushare下载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_trading_dates_for_tushare(self, start_date: str, end_date: str) -> List[str]:
        """获取Tushare交易日历"""
        try:
            import tushare as ts

            # 初始化tushare
            token = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'
            pro = ts.pro_api(token)

            # 获取交易日历
            cal_data = pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'  # 只获取交易日
            )

            if cal_data is None or cal_data.empty:
                logger.error("获取Tushare交易日历失败")
                return []

            trading_dates = cal_data['cal_date'].tolist()[::-1]
            logger.info(f"获取到 {len(trading_dates)} 个交易日")
            return trading_dates

        except Exception as e:
            logger.error(f"获取Tushare交易日历失败: {e}")
            return []

    def _detect_missing_data_precise(self, trading_dates: List[str], stock_list: List[str]) -> dict:
        """精确检测缺失数据（智能分析现有分区）"""
        try:
            logger.info("开始精确检测缺失数据...")

            # 1. 首先检查现有分区
            daily_data_dir = get_data_path("daily")
            existing_partitions = []
            if daily_data_dir.exists():
                existing_partitions = [p.name for p in daily_data_dir.iterdir() if p.is_dir()]
                existing_partitions.sort()

            logger.info(f"现有分区: {existing_partitions}")

            # 2. 按月份分组交易日
            date_partitions = {}
            for date in trading_dates:
                partition_key = self._get_partition_key(date)
                if partition_key not in date_partitions:
                    date_partitions[partition_key] = []
                date_partitions[partition_key].append(date)

            all_partitions = sorted(date_partitions.keys())
            logger.info(f"需要的分区: {all_partitions}")

            missing_data_map = {}  # {partition_key: [missing_dates]}

            # 3. 分别处理现有分区和新分区
            for partition_key, partition_dates in date_partitions.items():
                logger.info(f"检查分区 {partition_key}...")

                partition_dir = get_data_path("daily") / partition_key
                partition_file = partition_dir / "daily_data.parquet"

                if partition_key in existing_partitions and partition_file.exists():
                    # 现有分区：检查具体缺失日期
                    try:
                        existing_data = pd.read_parquet(partition_file)
                        if 'date_str' in existing_data.columns:
                            existing_dates = set(existing_data['date_str'].unique())
                            expected_dates = set(partition_dates)
                            missing_dates = expected_dates - existing_dates

                            if missing_dates:
                                missing_data_map[partition_key] = sorted(list(missing_dates))
                                logger.info(f"现有分区 {partition_key} 部分缺失，需要下载 {len(missing_dates)} 个交易日: {sorted(missing_dates)}")
                            else:
                                logger.info(f"现有分区 {partition_key} 数据完整，包含 {len(existing_dates)} 个交易日")
                        else:
                            # 数据格式异常，重新下载整个分区
                            missing_data_map[partition_key] = partition_dates
                            logger.warning(f"现有分区 {partition_key} 数据格式异常，需要重新下载")
                    except Exception as e:
                        # 文件损坏，重新下载整个分区
                        missing_data_map[partition_key] = partition_dates
                        logger.warning(f"现有分区 {partition_key} 文件损坏，需要重新下载: {e}")
                else:
                    # 新分区：完全缺失
                    missing_data_map[partition_key] = partition_dates
                    logger.info(f"新分区 {partition_key} 完全缺失，需要下载 {len(partition_dates)} 个交易日")

            # 4. 汇总统计
            total_missing_dates = sum(len(dates) for dates in missing_data_map.values())
            existing_complete_partitions = len([p for p in all_partitions if p not in missing_data_map])

            logger.info(f"检测完成:")
            logger.info(f"  - 完整分区: {existing_complete_partitions} 个")
            logger.info(f"  - 需要更新的分区: {len(missing_data_map)} 个")
            logger.info(f"  - 需要下载的交易日: {total_missing_dates} 个")

            if missing_data_map:
                logger.info(f"详细缺失情况:")
                for partition_key, missing_dates in missing_data_map.items():
                    logger.info(f"  - {partition_key}: {len(missing_dates)} 个交易日")

            return missing_data_map

        except Exception as e:
            logger.error(f"检测缺失数据失败: {e}")
            return {}

    def _download_missing_data_smart(self, missing_data_map: dict, stock_list: List[str]) -> bool:
        """智能下载缺失数据"""
        try:
            logger.info("开始智能下载缺失数据...")

            import tushare as ts

            # 初始化tushare
            # token = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'
            token = 'e75c6baca18c97b4337dd2f1946da4aa1a6a875804948000017a7525'
            pro = ts.pro_api(token)

            # 转换股票代码格式
            tushare_stocks = [self._convert_stock_code_to_tushare(stock) for stock in stock_list]

            total_success = True

            for partition_key, missing_dates in missing_data_map.items():
                logger.info(f"处理分区 {partition_key}，缺失日期: {len(missing_dates)} 个")

                # 按日期分批下载（优化API调用）
                success = self._download_partition_data_by_dates(
                    pro, partition_key, missing_dates, tushare_stocks, stock_list
                )

                if not success:
                    logger.error(f"分区 {partition_key} 下载失败")
                    total_success = False
                else:
                    logger.info(f"分区 {partition_key} 下载成功")

            return total_success

        except Exception as e:
            logger.error(f"智能下载失败: {e}")
            return False

    def _convert_stock_code_to_tushare(self, stock_code: str) -> str:
        """将xtdata股票代码转换为tushare格式"""
        if '.SZ' in stock_code:
            return stock_code.replace('.SZ', '.SZ')
        elif '.SH' in stock_code:
            return stock_code.replace('.SH', '.SH')
        else:
            # 根据代码判断交易所
            code = stock_code.split('.')[0]
            if code.startswith('0') or code.startswith('3'):
                return f"{code}.SZ"
            elif code.startswith('6'):
                return f"{code}.SH"
            else:
                return stock_code

    def _download_partition_data_by_dates(self, pro, partition_key: str, missing_dates: List[str],
                                        tushare_stocks: List[str], original_stocks: List[str]) -> bool:
        """按日期分批下载分区数据"""
        try:
            logger.info(f"开始下载分区 {partition_key} 的 {len(missing_dates)} 个日期")

            # 按日期分批（优化：每批3个交易日，充分利用6000条数据限制）
            batch_size = 3
            all_downloaded_data = []

            for i in range(0, len(missing_dates), batch_size):
                batch_dates = missing_dates[i:i+batch_size]
                # 确保日期范围正确：start_date <= end_date
                batch_start = min(batch_dates)
                batch_end = max(batch_dates)

                logger.info(f"下载日期批次: {batch_start} - {batch_end} ({len(batch_dates)} 个日期)")
                logger.info(f"批次日期列表: {batch_dates}")

                # 下载基础行情数据
                daily_data = self._download_daily_basic_with_retry(
                    pro, batch_start, batch_end, batch_dates
                )

                if daily_data is not None and not daily_data.empty:
                    # 下载市值数据
                    mv_data = self._download_market_value_with_retry(
                        pro, batch_start, batch_end, batch_dates
                    )

                    # 处理数据格式
                    processed_data = self._process_tushare_batch_data(
                        daily_data, mv_data, batch_dates, original_stocks
                    )

                    if processed_data is not None and not processed_data.empty:
                        all_downloaded_data.append(processed_data)
                        logger.info(f"批次 {batch_start}-{batch_end} 处理成功: {len(processed_data)} 条记录")
                    else:
                        logger.warning(f"批次 {batch_start}-{batch_end} 处理后数据为空")
                else:
                    logger.warning(f"批次 {batch_start}-{batch_end} 下载失败")

                # API调用间隔（保守策略：每分钟最多200次调用，约0.3秒间隔）
                time.sleep(0.35)

            if not all_downloaded_data:
                logger.error(f"分区 {partition_key} 所有批次都下载失败")
                return False

            # 合并所有下载的数据
            new_data = pd.concat(all_downloaded_data, ignore_index=True)
            logger.info(f"分区 {partition_key} 新下载数据: {len(new_data)} 条记录")

            # 与现有数据合并
            success = self._merge_and_save_partition_data(partition_key, new_data)

            return success

        except Exception as e:
            logger.error(f"下载分区 {partition_key} 数据失败: {e}")
            return False

    def _download_daily_basic_with_retry(self, pro, start_date: str, end_date: str, target_dates: List[str]) -> pd.DataFrame:
        """带重试的基础行情数据下载（无限重试版）"""
        import time  # 确保time模块在方法开始时就可用
        retry_delay = 180  # 增加到3分钟，确保API限制重置

        attempt = 0
        while True:  # 无限重试
            attempt += 1
            try:
                logger.info(f"下载基础行情数据 {start_date}-{end_date} (尝试 {attempt})")
                logger.info(f"目标日期: {target_dates}")

                # 改为单日查询，避免6000条限制导致的数据截断
                if len(target_dates) == 1:
                    # 单日查询
                    data = pro.daily(
                        trade_date=target_dates[0],
                        fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                    )
                else:
                    # 多日期时，逐个查询并合并
                    all_data = []
                    for date in target_dates:
                        single_data = pro.daily(
                            trade_date=date,
                            fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                        )
                        if single_data is not None and not single_data.empty:
                            all_data.append(single_data)
                        # API调用间隔（保守策略）
                        import time
                        time.sleep(0.15)

                    # 合并所有数据
                    if all_data:
                        data = pd.concat(all_data, ignore_index=True)
                    else:
                        data = pd.DataFrame()

                if data is not None and not data.empty:
                    # 筛选目标日期的数据
                    filtered_data = data[data['trade_date'].isin(target_dates)]
                    logger.info(f"基础行情数据下载成功: 原始{len(data)}条 -> 筛选后{len(filtered_data)}条")

                    if not filtered_data.empty:
                        actual_dates = sorted(filtered_data['trade_date'].unique())
                        logger.info(f"实际获取日期: {actual_dates}")
                        return filtered_data
                    else:
                        logger.warning(f"筛选后数据为空，原始数据日期: {sorted(data['trade_date'].unique())}")
                else:
                    logger.warning(f"基础行情数据为空 (尝试 {attempt + 1})")

            except Exception as e:
                logger.warning(f"基础行情数据下载失败 (尝试 {attempt}): {e}")

                if "每分钟最多访问" in str(e) or "频率限制" in str(e) or "抱歉" in str(e):
                    # 频率限制错误，使用指数退避策略
                    wait_time = min(retry_delay * (1.2 ** (attempt - 1)), 600)  # 最大等待10分钟
                    logger.info(f"遇到频率限制，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt})")
                    time.sleep(wait_time)
                    continue
                elif "网络" in str(e) or "连接" in str(e) or "timeout" in str(e).lower():
                    # 网络错误，短时间重试
                    logger.info(f"遇到网络错误，等待30秒后重试... (尝试 {attempt})")
                    time.sleep(30)
                    continue
                else:
                    # 其他错误，等待较短时间后重试
                    logger.info(f"遇到其他错误，等待60秒后重试... (尝试 {attempt}): {e}")
                    time.sleep(60)
                    continue

    def _download_market_value_with_retry(self, pro, start_date: str, end_date: str, target_dates: List[str]) -> pd.DataFrame:
        """带重试的市值数据下载（无限重试版）"""
        import time  # 确保time模块在方法开始时就可用
        retry_delay = 180  # 增加到3分钟

        attempt = 0
        while True:  # 无限重试
            attempt += 1
            try:
                logger.info(f"下载市值数据 {start_date}-{end_date} (尝试 {attempt})")
                logger.info(f"目标日期: {target_dates}")

                # 如果只有一个日期，直接使用该日期
                if len(target_dates) == 1:
                    actual_start = actual_end = target_dates[0]
                else:
                    actual_start = start_date
                    actual_end = end_date

                data = pro.daily_basic(
                    start_date=actual_start,
                    end_date=actual_end,
                    fields='ts_code,trade_date,total_mv,circ_mv,turnover_rate'
                )

                if data is not None and not data.empty:
                    # 筛选目标日期的数据
                    filtered_data = data[data['trade_date'].isin(target_dates)]
                    logger.info(f"市值数据下载成功: 原始{len(data)}条 -> 筛选后{len(filtered_data)}条")

                    if not filtered_data.empty:
                        actual_dates = sorted(filtered_data['trade_date'].unique())
                        logger.info(f"实际获取日期: {actual_dates}")
                        return filtered_data
                    else:
                        logger.warning(f"筛选后数据为空，原始数据日期: {sorted(data['trade_date'].unique())}")
                else:
                    logger.warning(f"市值数据为空 (尝试 {attempt + 1})")

            except Exception as e:
                logger.warning(f"市值数据下载失败 (尝试 {attempt}): {e}")

                if "每分钟最多访问" in str(e) or "频率限制" in str(e) or "抱歉" in str(e):
                    # 频率限制错误，使用指数退避策略
                    wait_time = min(retry_delay * (1.2 ** (attempt - 1)), 600)  # 最大等待10分钟
                    logger.info(f"遇到频率限制，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt})")
                    time.sleep(wait_time)
                    continue
                elif "网络" in str(e) or "连接" in str(e) or "timeout" in str(e).lower():
                    # 网络错误，短时间重试
                    logger.info(f"遇到网络错误，等待30秒后重试... (尝试 {attempt})")
                    time.sleep(30)
                    continue
                else:
                    # 其他错误，等待较短时间后重试
                    logger.info(f"遇到其他错误，等待60秒后重试... (尝试 {attempt}): {e}")
                    time.sleep(60)
                    continue

    def _process_tushare_batch_data(self, daily_data: pd.DataFrame, mv_data: pd.DataFrame,
                                  target_dates: List[str], original_stocks: List[str]) -> pd.DataFrame:
        """处理Tushare批次数据"""
        try:
            if daily_data.empty:
                return pd.DataFrame()

            # 字段名映射
            field_mapping = {
                'ts_code': 'stock_code',
                'trade_date': 'date_str',
                'pre_close': 'preClose',
                'vol': 'volume',
            }

            # 重命名字段
            processed_data = daily_data.rename(columns=field_mapping)

            # 转换股票代码格式
            processed_data['stock_code'] = processed_data['stock_code'].apply(self._convert_stock_code_to_xtdata)

            # 转换数据类型和单位
            processed_data['volume'] = processed_data['volume'] * 100  # 手转股
            processed_data['amount'] = processed_data['amount'] * 1000  # 千元转元

            # 处理市值数据
            if not mv_data.empty:
                mv_data = mv_data.rename(columns={'ts_code': 'stock_code', 'trade_date': 'date_str'})
                mv_data['stock_code'] = mv_data['stock_code'].apply(self._convert_stock_code_to_xtdata)

                # 合并市值数据
                processed_data = pd.merge(processed_data, mv_data, on=['stock_code', 'date_str'], how='left')

            # 筛选目标日期和股票
            processed_data = processed_data[
                (processed_data['date_str'].isin(target_dates)) &
                (processed_data['stock_code'].isin(original_stocks))
            ]

            logger.info(f"数据处理完成: {len(processed_data)} 条记录")
            return processed_data

        except Exception as e:
            logger.error(f"处理Tushare数据失败: {e}")
            return pd.DataFrame()

    def _merge_and_save_partition_data(self, partition_key: str, new_data: pd.DataFrame) -> bool:
        """合并并保存分区数据"""
        try:
            partition_dir = get_data_path("daily") / partition_key
            partition_dir.mkdir(exist_ok=True, parents=True)
            partition_file = partition_dir / "daily_data.parquet"

            if partition_file.exists():
                # 读取现有数据
                existing_data = pd.read_parquet(partition_file)
                logger.info(f"读取现有数据: {len(existing_data)} 条记录")

                # 合并数据（新数据覆盖旧数据）
                # 先删除新数据中已存在的日期和股票组合
                if not existing_data.empty and 'date_str' in existing_data.columns and 'stock_code' in existing_data.columns:
                    new_keys = set(zip(new_data['date_str'], new_data['stock_code']))
                    existing_mask = ~existing_data.apply(
                        lambda row: (row['date_str'], row['stock_code']) in new_keys, axis=1
                    )
                    filtered_existing = existing_data[existing_mask]

                    # 合并数据
                    combined_data = pd.concat([filtered_existing, new_data], ignore_index=True)
                else:
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

                logger.info(f"合并后数据: {len(combined_data)} 条记录")
            else:
                # 新分区，直接保存
                combined_data = new_data
                logger.info(f"新分区数据: {len(combined_data)} 条记录")

            # 排序并保存
            if not combined_data.empty:
                combined_data = combined_data.sort_values(['date_str', 'stock_code'])
                combined_data.to_parquet(partition_file, index=False)
                logger.info(f"分区 {partition_key} 保存成功")

                # 更新元数据
                if partition_key not in self.metadata['partitions']['daily']:
                    self.metadata['partitions']['daily'].append(partition_key)
                    self.metadata['partitions']['daily'].sort()
                    self._save_metadata()

                return True
            else:
                logger.warning(f"分区 {partition_key} 合并后数据为空")
                return False

        except Exception as e:
            logger.error(f"合并保存分区 {partition_key} 失败: {e}")
            return False

    def _download_market_value_data(self, pro, start_date: str, end_date: str) -> pd.DataFrame:
        """下载市值数据"""
        try:
            logger.info("开始下载市值数据...")

            # 分批下载市值数据（daily_basic接口）
            mv_data_list = []

            # 按日期分批下载（避免数据量过大）
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            current_dt = start_dt
            while current_dt <= end_dt:
                # 每次下载30天的数据
                batch_end_dt = min(current_dt + timedelta(days=30), end_dt)
                batch_start = current_dt.strftime('%Y%m%d')
                batch_end = batch_end_dt.strftime('%Y%m%d')

                try:
                    batch_mv_data = pro.daily_basic(
                        start_date=batch_start,
                        end_date=batch_end,
                        fields='ts_code,trade_date,total_mv,circ_mv,turnover_rate'
                    )

                    if batch_mv_data is not None and not batch_mv_data.empty:
                        mv_data_list.append(batch_mv_data)
                        logger.info(f"市值数据批次 {batch_start}-{batch_end}: {len(batch_mv_data)} 条记录")

                except Exception as e:
                    logger.warning(f"市值数据批次 {batch_start}-{batch_end} 下载失败: {e}")

                current_dt = batch_end_dt + timedelta(days=1)
                time.sleep(0.2)  # 避免调用限制

            if mv_data_list:
                mv_data = pd.concat(mv_data_list, ignore_index=True)
                logger.info(f"市值数据下载完成: {len(mv_data)} 条记录")
                return mv_data
            else:
                logger.warning("市值数据下载失败，返回空DataFrame")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"下载市值数据失败: {e}")
            return pd.DataFrame()

    def _process_and_save_tushare_data(self, daily_data: pd.DataFrame, mv_data: pd.DataFrame, trading_dates: List[str]) -> bool:
        """处理和保存Tushare数据"""
        try:
            logger.info("开始处理Tushare数据...")

            # 字段名映射
            field_mapping = {
                'ts_code': 'stock_code',
                'trade_date': 'date_str',
                'pre_close': 'preClose',
                'vol': 'volume',  # tushare的vol是手，需要转换
                # amount保持不变，但tushare是千元
            }

            # 重命名字段
            daily_data = daily_data.rename(columns=field_mapping)

            # 转换股票代码格式（从tushare格式转为xtdata格式）
            daily_data['stock_code'] = daily_data['stock_code'].apply(self._convert_stock_code_to_xtdata)

            # 转换数据类型
            daily_data['volume'] = daily_data['volume'] * 100  # 手转为股
            daily_data['amount'] = daily_data['amount'] * 1000  # 千元转为元
            daily_data['date'] = daily_data['date_str']

            # 处理市值数据
            if not mv_data.empty:
                mv_data = mv_data.rename(columns={'ts_code': 'stock_code', 'trade_date': 'date_str'})
                mv_data['stock_code'] = mv_data['stock_code'].apply(self._convert_stock_code_to_xtdata)

                # 合并市值数据
                daily_data = pd.merge(daily_data, mv_data, on=['stock_code', 'date_str'], how='left')

            # 按月份分区保存
            date_partitions = {}
            for date in trading_dates:
                partition_key = self._get_partition_key(date)
                if partition_key not in date_partitions:
                    date_partitions[partition_key] = []
                date_partitions[partition_key].append(date)

            logger.info(f"数据将分为 {len(date_partitions)} 个月份分区")

            # 按分区保存数据
            for partition_key, partition_dates in date_partitions.items():
                # 使用统一的路径获取方法
                partition_dir = get_data_path("daily") / partition_key
                partition_dir.mkdir(exist_ok=True, parents=True)

                # 筛选当前分区的数据
                partition_data = daily_data[daily_data['date_str'].isin(partition_dates)]

                if not partition_data.empty:
                    # 保存数据
                    partition_file = partition_dir / "daily_data.parquet"
                    partition_data.to_parquet(partition_file, index=False)
                    logger.info(f"分区 {partition_key} 保存成功，共 {len(partition_data):,} 行数据")
                else:
                    logger.warning(f"分区 {partition_key} 无数据")

            return True

        except Exception as e:
            logger.error(f"处理Tushare数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _supplement_with_xtdata(self, missing_data_map: dict, stock_list: List[str]) -> bool:
        """使用xtdata补充Tushare缺失的数据"""
        try:
            logger.info("开始使用xtdata补充缺失数据...")

            from xtquant import xtdata

            # 确保xtdata连接
            if not xtdata.get_client().is_connected():
                xtdata.get_client().connect()
                time.sleep(1)

            total_success = True

            for partition_key, missing_dates in missing_data_map.items():
                logger.info(f"使用xtdata补充分区 {partition_key}，缺失日期: {missing_dates}")

                # 为每个缺失日期单独下载
                for date in missing_dates:
                    success = self._download_single_date_with_xtdata(date, stock_list, partition_key)
                    if not success:
                        logger.warning(f"xtdata补充日期 {date} 失败")
                        total_success = False
                    else:
                        logger.info(f"xtdata补充日期 {date} 成功")

            return total_success

        except Exception as e:
            logger.error(f"xtdata补充失败: {e}")
            return False

    def _download_single_date_with_xtdata(self, date: str, stock_list: List[str], partition_key: str) -> bool:
        """使用xtdata下载单个日期的数据"""
        try:
            from xtquant import xtdata

            logger.info(f"使用xtdata下载 {date} 的数据...")

            # 字段列表
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']

            # 下载数据
            data = xtdata.get_market_data(
                stock_list=stock_list,
                period='1d',
                start_time=date,
                end_time=date,
                field_list=fields,
                dividend_type='front',
                fill_data=True
            )

            if not data or not any(field in data for field in fields):
                logger.warning(f"xtdata下载 {date} 失败: 无数据")
                return False

            # 转换数据格式
            converted_data = self._convert_xtdata_to_dataframe_single_date(data, stock_list, date)

            if converted_data.empty:
                logger.warning(f"xtdata数据转换失败: {date}")
                return False

            # 合并到现有分区
            success = self._merge_and_save_partition_data(partition_key, converted_data)

            if success:
                logger.info(f"xtdata补充 {date} 成功: {len(converted_data)} 条记录")
                return True
            else:
                logger.error(f"xtdata补充 {date} 保存失败")
                return False

        except Exception as e:
            logger.error(f"xtdata下载单日数据失败 {date}: {e}")
            return False

    def _convert_xtdata_to_dataframe_single_date(self, data: dict, stock_list: List[str], date: str) -> pd.DataFrame:
        """转换xtdata单日数据为DataFrame格式"""
        try:
            result_data = []

            # 获取股本数据用于计算市值
            capital_file = get_data_path("stock_capital_data.parquet")
            if capital_file.exists():
                capital_df = pd.read_parquet(capital_file)
                capital_df = capital_df.set_index('stock_code')
            else:
                capital_df = pd.DataFrame()

            for stock in stock_list:
                row = {
                    'stock_code': stock,
                    'date_str': date
                }

                # 添加基础字段
                for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']:
                    if field in data:
                        field_df = data[field]
                        if isinstance(field_df, pd.DataFrame) and stock in field_df.index:
                            # 查找对应日期的数据
                            date_cols = [col for col in field_df.columns if str(col)[:10].replace('-', '') == date]
                            if date_cols:
                                row[field] = field_df.loc[stock, date_cols[0]]
                            else:
                                row[field] = None
                        else:
                            row[field] = None
                    else:
                        row[field] = None

                # 计算涨跌幅
                if row.get('close') is not None and row.get('preClose') is not None and row['preClose'] != 0:
                    row['pct_chg'] = (row['close'] - row['preClose']) / row['preClose'] * 100
                    row['change'] = row['close'] - row['preClose']
                else:
                    row['pct_chg'] = None
                    row['change'] = None

                # 计算市值
                if row.get('close') is not None and not capital_df.empty and stock in capital_df.index:
                    total_capital = capital_df.loc[stock, 'total_capital']
                    circulating_capital = capital_df.loc[stock, 'circulating_capital']

                    if total_capital is not None:
                        row['total_mv'] = row['close'] * total_capital
                    else:
                        row['total_mv'] = None

                    if circulating_capital is not None:
                        row['circ_mv'] = row['close'] * circulating_capital
                        # 计算换手率
                        if row.get('volume') is not None and circulating_capital > 0:
                            row['turnover_rate'] = row['volume'] / circulating_capital * 100
                        else:
                            row['turnover_rate'] = None
                    else:
                        row['circ_mv'] = None
                        row['turnover_rate'] = None
                else:
                    row['total_mv'] = None
                    row['circ_mv'] = None
                    row['turnover_rate'] = None

                result_data.append(row)

            df = pd.DataFrame(result_data)

            # 过滤掉没有有效数据的记录
            df = df.dropna(subset=['close'])

            logger.info(f"xtdata单日数据转换完成: {len(df)} 条有效记录")
            return df

        except Exception as e:
            logger.error(f"xtdata单日数据转换失败: {e}")
            return pd.DataFrame()

    def _convert_stock_code_to_xtdata(self, ts_code: str) -> str:
        """将tushare股票代码转换为xtdata格式"""
        if '.SZ' in ts_code or '.SH' in ts_code:
            return ts_code
        else:
            # 如果没有后缀，根据代码判断
            code = ts_code.split('.')[0]
            if code.startswith('0') or code.startswith('3'):
                return f"{code}.SZ"
            elif code.startswith('6'):
                return f"{code}.SH"
            else:
                return ts_code
    
    def download_daily_data(self, stock_list=None, start_date=None, end_date=None, batch_size=None, max_workers=None, use_tushare=False):
        """下载日线数据并以Parquet格式保存

        参数:
            stock_list: 股票列表，默认为元数据中的股票列表
            start_date: 开始日期，格式 'YYYYMMDD'，默认为 '20230101'
            end_date: 结束日期，格式 'YYYYMMDD'，默认为当前日期
            batch_size: 已废弃参数，保留用于兼容性
            max_workers: 已废弃参数，保留用于兼容性
            use_tushare: 是否使用tushare下载，默认False使用xtdata
        """
        logger.info("开始下载日线数据...")
        
        # 设置默认参数
        if stock_list is None:
            stock_list = self.metadata['stock_list']
            if not stock_list:
                stock_list = self.download_stock_list()
        # stock_list = self.download_stock_list()
        if start_date is None:
            start_date = '20230101'
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 字段列表
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']
        
        # 获取交易日历
        calendar_file = CALENDAR_DIR / "trading_calendar.parquet"
        if not calendar_file.exists():
            success = self.download_trading_calendar(start_date, end_date)
            if not success:
                logger.error("无法获取交易日历，日线数据下载失败")
                return False
        
        try:
            calendar_df = pd.read_parquet(calendar_file)
            trading_dates = calendar_df[(calendar_df['date_str'] >= start_date) & 
                                    (calendar_df['date_str'] <= end_date)]['date_str'].tolist()
            
            if not trading_dates:
                logger.warning(f"指定日期范围内没有交易日: {start_date} 至 {end_date}")
                return False
                
            logger.info(f"找到 {len(trading_dates)} 个交易日")
            
        except Exception as e:
            logger.error(f"读取交易日历失败: {e}")
            return False
        
        # 读取股本数据
        try:
            capital_file = get_data_path("stock_capital_data.parquet")
            # capital_file = 'to/local_data/stock_capital_data.parquet'
            logger.info(f"读取股本数据: {capital_file}")
            capital_df = pd.read_parquet(capital_file)
            
            # 将股票代码设为索引以加快查询
            capital_df.set_index('stock', inplace=True)
            logger.info(f"成功读取股本数据: {len(capital_df)}条记录")
        except Exception as e:
            logger.error(f"读取股本数据失败: {e}")
            return False
        
        start_time = time.time()
        
        try:
            if use_tushare:
                # 使用Tushare下载数据
                logger.info(f"开始使用Tushare下载日线数据: {len(stock_list)} 只股票，日期范围 {start_date} 至 {end_date}")

                success = self._download_with_tushare(start_date, end_date, stock_list, trading_dates)

                if not success:
                    logger.error("Tushare下载失败")
                    return False

                logger.info(f"Tushare数据下载完成，耗时: {time.time() - start_time:.2f}秒")
                return True
            else:
                # 使用xtdata下载数据（默认方式）
                logger.info(f"开始使用xtdata下载日线数据: {len(stock_list)} 只股票，日期范围 {start_date} 至 {end_date}")

                success = self._download_with_xtdata(start_date, end_date, stock_list, trading_dates, fields)

                if not success:
                    logger.error("xtdata下载失败")
                    return False

                print(f"xtdata数据下载完成，耗时: {time.time() - start_time:.2f}秒")
                return True
            
        except KeyboardInterrupt:
            logger.warning("用户中断下载过程")
            return False
        except Exception as e:
            logger.error(f"下载日线数据时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False




    def _download_with_xtdata(self, start_date: str, end_date: str, stock_list: List[str], trading_dates: List[str], fields: List[str]) -> bool:
        """使用xtdata下载日线数据（默认方式）

        参数:
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表
            trading_dates: 交易日列表
            fields: 字段列表
        """
        try:
            from xtquant import xtdata

            # 确保xtdata连接
            if not xtdata.get_client().is_connected():
                logger.info("连接xtdata API...")
                xtdata.get_client().connect()
                time.sleep(1)

            if not xtdata.get_client().is_connected():
                logger.error("无法连接到xtdata API")
                return False

            logger.info("xtdata API连接成功")

            # 按月份分组下载
            monthly_groups = {}
            for date in trading_dates:
                month_key = date[:6]  # YYYYMM
                if month_key not in monthly_groups:
                    monthly_groups[month_key] = []
                monthly_groups[month_key].append(date)

            logger.info(f"按月份分组下载: {list(monthly_groups.keys())}")

            total_success = True

            for month_key, month_dates in monthly_groups.items():
                logger.info(f"下载月份 {month_key}，包含 {len(month_dates)} 个交易日")

                # 下载整个月份的数据
                month_start = month_dates[0]
                month_end = month_dates[-1]

                success = self._download_month_with_xtdata(month_key, month_start, month_end, stock_list, fields)

                if not success:
                    logger.error(f"月份 {month_key} 下载失败")
                    total_success = False
                else:
                    logger.info(f"月份 {month_key} 下载成功")

            return total_success

        except Exception as e:
            logger.error(f"xtdata下载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _download_month_with_xtdata(self, month_key: str, start_date: str, end_date: str, stock_list: List[str], fields: List[str]) -> bool:
        """使用xtdata下载单个月份的数据

        参数:
            month_key: 月份键，格式YYYYMM
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表
            fields: 字段列表
        """
        try:
            from xtquant import xtdata

            logger.info(f"使用xtdata下载月份 {month_key} 的数据: {start_date} - {end_date}")

            # 下载基础行情数据
            data = xtdata.get_market_data(
                stock_list=stock_list,
                period='1d',
                start_time=start_date,
                end_time=end_date,
                field_list=fields,
                dividend_type='front',
                fill_data=True
            )

            if not data or not any(field in data for field in fields):
                logger.warning(f"xtdata下载月份 {month_key} 失败: 无数据")
                return False

            # 转换数据格式，确保与202503格式完全一致
            converted_data = self._convert_xtdata_to_dataframe_monthly(data, stock_list, start_date, end_date)

            if converted_data.empty:
                logger.warning(f"xtdata数据转换失败: 月份 {month_key}")
                return False

            # 保存到对应的月份分区
            partition_key = month_key
            success = self._save_monthly_partition_data(partition_key, converted_data)

            if success:
                logger.info(f"xtdata月份 {month_key} 下载成功: {len(converted_data)} 条记录")
                return True
            else:
                logger.error(f"xtdata月份 {month_key} 保存失败")
                return False

        except Exception as e:
            logger.error(f"xtdata下载月份数据失败 {month_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _convert_xtdata_to_dataframe_monthly(self, data: dict, stock_list: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """转换xtdata月度数据为DataFrame格式，确保与202503格式完全一致

        202503格式标准:
        - 字段: ['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preClose', 'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate', 'date_str']
        - volume单位: 股 (不是手)
        - amount单位: 元
        - 市值单位: 万元

        xtdata数据结构:
        - 每个字段是一个DataFrame，行是股票代码，列是日期
        """
        try:
            logger.info(f"转换xtdata月度数据为DataFrame格式: {start_date} - {end_date}")

            result_data = []

            # 🔧 获取股本数据用于计算市值（修复市值数据缺失问题）
            capital_file = get_data_path("stock_capital_data.parquet")
            if capital_file.exists():
                capital_df = pd.read_parquet(capital_file)
                logger.info(f"原始股本数据: {len(capital_df)} 条记录")

                # 🔧 修复set_index用法：inplace=True会返回None，应该分开操作
                capital_df.set_index('stock', inplace=True)
                logger.info(f"成功加载股本数据: {len(capital_df)} 只股票")
            else:
                capital_df = pd.DataFrame()
                logger.warning(f"股本数据文件不存在: {capital_file}")

            # 获取所有可用的日期（从任一字段中获取）
            available_dates = set()
            for field_name, field_data in data.items():
                if isinstance(field_data, pd.DataFrame):
                    available_dates.update(field_data.columns)

            if not available_dates:
                logger.warning("xtdata数据中没有找到日期")
                return pd.DataFrame()

            logger.info(f"找到 {len(available_dates)} 个日期: {sorted(available_dates)}")

            # 遍历每只股票和每个日期
            for stock_code in stock_list:
                for date_str in sorted(available_dates):
                    # 转换日期格式
                    try:
                        date_dt = pd.to_datetime(date_str, format='%Y%m%d')
                    except:
                        continue

                    # 构建记录
                    record = {
                        'stock_code': stock_code,
                        'date': date_dt,
                        'date_str': date_str
                    }

                    # 提取基础行情数据
                    has_data = False
                    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']:
                        if field in data:
                            field_data = data[field]
                            if isinstance(field_data, pd.DataFrame) and stock_code in field_data.index and date_str in field_data.columns:
                                value = field_data.loc[stock_code, date_str]
                                if pd.notna(value):
                                    record[field] = float(value)
                                    has_data = True
                                else:
                                    record[field] = None
                            else:
                                record[field] = None
                        else:
                            record[field] = None

                    # 只有当至少有一个字段有数据时才添加记录
                    if not has_data:
                        continue

                    # 计算衍生字段
                    if record.get('close') and record.get('preClose'):
                        # 计算涨跌幅 (%)
                        record['pct_chg'] = ((record['close'] - record['preClose']) / record['preClose']) * 100
                    else:
                        record['pct_chg'] = None

                    # 🔧 修复市值计算逻辑（参考_convert_xtdata_to_dataframe_single_date实现）
                    if record.get('close') is not None and not capital_df.empty and stock_code in capital_df.index:
                        try:
                            total_capital = capital_df.loc[stock_code, 'total_capital']
                            circulating_capital = capital_df.loc[stock_code, 'circulating_capital']

                            # 计算总市值（与单日处理方法保持一致的单位）
                            if pd.notna(total_capital) and total_capital > 0:
                                record['total_mv'] = record['close'] * total_capital
                            else:
                                record['total_mv'] = None

                            # 计算流通市值和换手率
                            if pd.notna(circulating_capital) and circulating_capital > 0:
                                record['circ_mv'] = record['close'] * circulating_capital

                                # 计算换手率 (%)
                                if record.get('volume') is not None and record['volume'] > 0:
                                    record['turnover_rate'] = (record['volume'] / circulating_capital) * 100
                                else:
                                    record['turnover_rate'] = None
                            else:
                                record['circ_mv'] = None
                                record['turnover_rate'] = None

                        except Exception as e:
                            logger.debug(f"计算股票 {stock_code} 市值时出错: {e}")
                            record['total_mv'] = None
                            record['circ_mv'] = None
                            record['turnover_rate'] = None
                    else:
                        # 没有收盘价或股本数据时设置为None
                        record['total_mv'] = None
                        record['circ_mv'] = None
                        record['turnover_rate'] = None

                    result_data.append(record)

            if not result_data:
                logger.warning("没有有效的xtdata数据")
                return pd.DataFrame()

            # 创建DataFrame
            df = pd.DataFrame(result_data)

            # 确保数据类型与202503一致
            df['stock_code'] = df['stock_code'].astype('object')
            df['date'] = pd.to_datetime(df['date'])
            df['date_str'] = df['date_str'].astype('object')

            # 数值字段
            numeric_fields = ['open', 'high', 'low', 'close', 'preClose', 'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')

            # volume字段特殊处理：确保是整数类型，单位为股
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('int64')

            # amount字段：确保是浮点类型，单位为元
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

            # 过滤掉没有有效数据的记录
            df = df.dropna(subset=['close'])

            # 确保字段顺序与202503一致
            expected_columns = ['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preClose', 'pct_chg', 'total_mv', 'circ_mv', 'turnover_rate', 'date_str']

            # 重新排列列顺序
            df = df.reindex(columns=expected_columns)

            logger.info(f"xtdata月度数据转换完成: {len(df)} 条有效记录")
            return df

        except Exception as e:
            logger.error(f"xtdata月度数据转换失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _save_monthly_partition_data(self, partition_key: str, data: pd.DataFrame) -> bool:
        """保存月度分区数据

        参数:
            partition_key: 分区键，格式YYYYMM
            data: 要保存的数据
        """
        try:
            if data.empty:
                logger.warning(f"分区 {partition_key} 数据为空，跳过保存")
                return True

            # 创建分区目录
            daily_data_dir = get_data_path("daily")
            partition_dir = daily_data_dir / partition_key
            partition_dir.mkdir(parents=True, exist_ok=True)

            # 保存为parquet文件
            output_file = partition_dir / "daily_data.parquet"

            logger.info(f"保存分区 {partition_key} 数据到: {output_file}")
            logger.info(f"数据规模: {len(data)} 行, {len(data.columns)} 列")

            # 确保数据格式正确
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])

            # 保存数据
            data.to_parquet(output_file, index=False, engine='pyarrow')

            logger.info(f"分区 {partition_key} 保存成功")
            return True

        except Exception as e:
            logger.error(f"保存分区 {partition_key} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _verify_download_completeness(self, start_date: str, end_date: str, expected_dates: list) -> bool:
        """验证下载数据的完整性"""
        try:
            logger.info(f"验证下载数据完整性: {start_date} - {end_date}")

            # 按月份分组期望的日期
            expected_partitions = {}
            for date in expected_dates:
                partition_key = self._get_partition_key(date)
                if partition_key not in expected_partitions:
                    expected_partitions[partition_key] = []
                expected_partitions[partition_key].append(date)

            logger.info(f"期望的分区: {list(expected_partitions.keys())}")

            # 检查每个分区的文件是否存在且数据完整
            missing_partitions = []
            incomplete_partitions = []

            for partition_key, partition_dates in expected_partitions.items():
                partition_dir = get_data_path("daily") / partition_key
                partition_file = partition_dir / "daily_data.parquet"

                if not partition_file.exists():
                    missing_partitions.append(partition_key)
                    logger.warning(f"分区文件不存在: {partition_file}")
                    continue

                try:
                    # 读取分区数据验证
                    partition_data = pd.read_parquet(partition_file)
                    if partition_data.empty:
                        incomplete_partitions.append(partition_key)
                        logger.warning(f"分区数据为空: {partition_key}")
                        continue

                    # 检查日期覆盖（允许部分缺失，特别是未来日期）
                    if 'date_str' in partition_data.columns:
                        actual_dates = set(partition_data['date_str'].unique())
                        expected_dates_set = set(partition_dates)
                        missing_dates = expected_dates_set - actual_dates

                        if missing_dates:
                            # 计算缺失比例
                            missing_ratio = len(missing_dates) / len(expected_dates_set)
                            logger.info(f"分区 {partition_key} 缺少 {len(missing_dates)}/{len(expected_dates_set)} 个日期 ({missing_ratio:.1%})")
                            logger.info(f"缺失日期: {sorted(missing_dates)}")

                            # 如果缺失超过50%，认为不完整
                            if missing_ratio > 0.5:
                                incomplete_partitions.append(partition_key)
                                logger.warning(f"分区 {partition_key} 缺失日期过多: {missing_ratio:.1%}")
                                continue
                            else:
                                logger.info(f"分区 {partition_key} 缺失日期在可接受范围内: {missing_ratio:.1%}")
                        else:
                            logger.info(f"分区 {partition_key} 日期完整")

                    # 检查股票数量
                    stock_count = partition_data['stock_code'].nunique() if 'stock_code' in partition_data.columns else 0
                    if stock_count < 1000:  # 假设至少应该有1000只股票
                        incomplete_partitions.append(partition_key)
                        logger.warning(f"分区 {partition_key} 股票数量不足: {stock_count}")
                        continue

                    logger.info(f"分区 {partition_key} 验证通过: {len(partition_data)} 条记录, {stock_count} 只股票")

                except Exception as e:
                    incomplete_partitions.append(partition_key)
                    logger.error(f"读取分区 {partition_key} 失败: {e}")

            # 汇总验证结果
            if missing_partitions:
                logger.error(f"缺失的分区: {missing_partitions}")

            if incomplete_partitions:
                logger.error(f"不完整的分区: {incomplete_partitions}")

            total_issues = len(missing_partitions) + len(incomplete_partitions)
            if total_issues > 0:
                logger.error(f"数据验证失败: {total_issues} 个分区有问题")
                return False

            logger.info("所有分区验证通过")
            return True

        except Exception as e:
            logger.error(f"验证下载完整性失败: {e}")
            return False

    def _download_minute_batch(self, batch_stocks, start_date, end_date, fields):
        """下载一批股票的5分钟数据
        
        参数:
            batch_stocks: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        返回:
            dict: 股票代码到数据的映射
        """
        import baostock as bs
        import time
        import random
        
        # 设置重试参数
        max_retries = 3  # 最大重试次数
        retry_delay = 2  # 基础重试延迟(秒)
        
        # 登录baostock
        retry_count = 0
        while retry_count < max_retries:
            try:
                lg = bs.login()
                if lg.error_code != '0':
                    logger.error(f"登录baostock失败: {lg.error_msg}")
                    retry_count += 1
                    time.sleep(retry_delay * (1 + random.random()))  # 指数退避策略
                    continue
                break
            except Exception as e:
                logger.error(f"登录异常: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error("登录失败次数过多，放弃下载")
                    return {}
                time.sleep(retry_delay * (1 + random.random()))
        
        # 转换日期格式
        start_date_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_date_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # 转换字段
        bs_fields = ['date', 'time', 'code']
        field_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'amount': 'amount'
        }
        
        for field in fields:
            if field in field_mapping:
                bs_fields.append(field_mapping[field])
        
        bs_fields_str = ",".join(bs_fields)
        
        all_stocks_data = {}
        
        # 对每个股票进行处理
        for stock_idx, stock_code in enumerate(batch_stocks):
            logger.info(f"下载股票 {stock_code} 5分钟数据 ({stock_idx+1}/{len(batch_stocks)})")
            
            # 转换股票代码格式（从000001.SZ到sz.000001）
            market, code = stock_code.split('.')
            bs_stock_code = f"{market.lower()}.{code}"
            
            stock_data = []
            retry_count = 0
            success = False
            
            # 重试循环
            while retry_count < max_retries and not success:
                try:
                    
                    # 查询5分钟线数据
                    rs = bs.query_history_k_data_plus(
                                bs_stock_code,
                                bs_fields_str,
                                start_date=start_date_str,
                                end_date=end_date_str,
                                frequency="5",
                                adjustflag="3"  # 不复权
                            )      
        
                    while (rs.error_code == '0') & rs.next():
                        # 获取一条记录，将记录合并在一起
                        stock_data.append(rs.get_row_data())
                   
                    
                    # 如果至少获取到一条数据，则认为成功
                    if len(stock_data) > 0:
                        success = True
                        logger.info(f"获取股票 {stock_code} 5分钟数据成功")
                    else:
                        logger.warning(f"获取股票 {stock_code} 5分钟数据成功，但数据为空")
                        success = True  # 也标记为成功，因为API调用没有错误
                
                except UnicodeDecodeError as e:
                    logger.warning(f"获取股票 {stock_code} 5分钟数据时发生编码错误: {str(e)}")
                    retry_count += 1
        
            
            # 如果获取到了数据
            if success and stock_data:
                try:
                    # 尝试创建DataFrame
                    stock_df = pd.DataFrame(stock_data, columns=rs.fields)
                    
                    # 处理日期时间
                    stock_df['date_str'] = stock_df['date'].str.replace('-', '')
                    
                    # 处理时间格式 - 修复BaoStock返回的非标准时间格式
                    # 时间格式为 YYYYMMDDHHMMSSsss，需要提取出HHMMSS部分
                    import re
                    try:
                        # 先尝试从time字段直接提取时分秒部分（HHMMSS）
                        # 格式一般是20230103093500000，需要提取093500部分
                        stock_df['time_fixed'] = stock_df['time'].apply(
                            lambda x: x[8:14] if isinstance(x, str) and len(x) >= 14 else None
                        )
                        
                        # 如果上面的方法提取失败，尝试使用正则表达式
                        if stock_df['time_fixed'].isna().any():
                            logger.info("使用正则表达式提取时间部分")
                            stock_df['time_fixed'] = stock_df['time'].apply(
                                lambda x: re.search(r'^\d{8}(\d{6})', x).group(1) 
                                if isinstance(x, str) and re.search(r'^\d{8}(\d{6})', x) 
                                else None
                            )
                            
                        # 记录日志检查提取的时间格式
                        if not stock_df['time_fixed'].isna().all():
                            sample_times = stock_df['time_fixed'].dropna().head(3).tolist()
                            logger.info(f"提取的时间样本: {sample_times}")
                        else:
                            logger.warning("时间提取失败，所有值都是NA")
                            # 最后尝试直接从原始time字段获取任何可能的6位数字
                            stock_df['time_fixed'] = stock_df['time'].apply(
                                lambda x: ''.join(re.findall(r'\d{6}', x)[:1])
                                if isinstance(x, str)
                                else None
                            )
                    except Exception as e:
                        logger.error(f"时间处理出错: {str(e)}")
                        # 简单回退策略：任何6位连续数字
                        stock_df['time_fixed'] = stock_df['time'].apply(
                            lambda x: re.search(r'(\d{6})', str(x)).group(1)
                            if isinstance(x, str) and re.search(r'(\d{6})', str(x))
                            else None
                        )
                    
                    # 使用提取的时间部分
                    stock_df['time_str'] = stock_df['time_fixed']
                    
                    # 创建正确格式的datetime对象 - 多层错误处理
                    try:
                        stock_df['datetime'] = pd.to_datetime(
                            stock_df['date'] + ' ' + stock_df['time_str'],
                            format='%Y-%m-%d %H%M%S'
                        )
                    except Exception as e:
                        logger.warning(f"日期时间转换失败，尝试备用方式1: {e}")
                        try:
                            # 备用方式1：从date_str直接提取日期部分，再加上time_str
                            stock_df['datetime'] = pd.to_datetime(
                                stock_df['date_str'] + stock_df['time_str'], 
                                format='%Y%m%d%H%M%S'
                            )
                        except Exception as e2:
                            logger.warning(f"日期时间转换失败，尝试备用方式2: {e2}")
                            try:
                                # 备用方式2：直接从time字段提取日期和时间
                                stock_df['datetime'] = pd.to_datetime(
                                    stock_df['time'].apply(
                                        lambda x: f"{x[:8]} {x[8:14]}" if isinstance(x, str) and len(x) >= 14 else None
                                    ),
                                    format='%Y%m%d %H%M%S',
                                    errors='coerce'
                                )
                            except Exception as e3:
                                logger.error(f"所有日期时间转换方法都失败: {e3}")
                                # 最后尝试，使用dateutil解析器（更灵活但更慢）
                                stock_df['datetime'] = pd.to_datetime(
                                    stock_df['date_str'] + stock_df['time_str'], 
                                    errors='coerce'
                                )
                    
                    # 检查是否有任何有效的datetime
                    if stock_df['datetime'].isna().all():
                        logger.error("所有日期时间转换后都是NA，可能需要检查原始数据格式")
                        # 记录一些原始数据样本，以便调试
                        sample_data = {
                            'date': stock_df['date'].head(3).tolist(),
                            'time': stock_df['time'].head(3).tolist(),
                            'time_fixed': stock_df['time_fixed'].head(3).tolist()
                        }
                        logger.error(f"样本数据: {sample_data}")
                    
                    # 转换数据类型
                    for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        if field in stock_df.columns:
                            stock_df[field] = pd.to_numeric(stock_df[field], errors='coerce')
                    
                    # 添加股票代码(使用标准格式)
                    stock_df['stock_code'] = stock_code
                    
                    # 保存数据
                    all_stocks_data[stock_code] = stock_df
                    logger.info(f"股票 {stock_code} 5分钟数据下载完成，获取 {len(stock_df)} 条记录")
                    
                except Exception as e:
                    logger.error(f"处理股票 {stock_code} 数据时出错: {str(e)}")
            else:
                logger.warning(f"无法获取股票 {stock_code} 的5分钟数据，跳过")
            
            # 每5个股票添加一次随机延迟，避免频繁请求
            if (stock_idx + 1) % 5 == 0 and stock_idx < len(batch_stocks) - 1:
                delay = random.uniform(1.0, 3.0)
                logger.debug(f"已处理 {stock_idx+1} 只股票，休息 {delay:.2f} 秒...")
                time.sleep(delay)
        
        # 登出系统
        try:
            bs.logout()
        except Exception as e:
            logger.warning(f"登出baostock时出错: {str(e)}")
        
        return all_stocks_data
    
    def _process_and_reshape_batch(self, batch_data, fields):
        """处理和重塑批次数据为长格式DataFrame
        
        参数:
            batch_data: 批次数据，格式为baostock返回的DataFrame
            fields: 字段列表
            
        返回:
            pd.DataFrame: 长格式数据
        """
        if batch_data is None:
            logger.warning("批次数据为空")
            return None
        
        try:
            # 检查是否已经是DataFrame格式(baostock下载的数据)
            if isinstance(batch_data, pd.DataFrame):
                logger.info(f"处理baostock下载的数据，形状: {batch_data.shape}")
                
                # 确保数据包含所需的列
                required_cols = ['stock_code', 'date_str', 'datetime']
                missing_cols = [col for col in required_cols if col not in batch_data.columns]
                if missing_cols:
                    logger.error(f"批次数据缺少必要的列: {missing_cols}")
                    return None
                
                # 确保datetime列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(batch_data['datetime']):
                    logger.info("将datetime列转换为datetime类型")
                    batch_data['datetime'] = pd.to_datetime(batch_data['datetime'], errors='coerce')
                    batch_data = batch_data.dropna(subset=['datetime'])
                
                # 确保数值列是数值类型
                numeric_cols = [col for col in fields if col in batch_data.columns]
                for col in numeric_cols:
                    if not pd.api.types.is_numeric_dtype(batch_data[col]):
                        logger.info(f"将{col}列转换为数值类型")
                        batch_data[col] = pd.to_numeric(batch_data[col], errors='coerce')
                
                # 返回处理后的数据
                logger.info(f"baostock数据处理完成，返回 {len(batch_data)} 行数据")
                return batch_data
            
            # 原始xtdata数据处理逻辑 (保留以兼容)
            logger.info("处理xtdata格式数据")
            
            # 检查数据有效性
            if not isinstance(batch_data, dict):
                logger.error(f"不支持的数据类型: {type(batch_data)}")
                return None
                
            if 'close' not in batch_data or batch_data['close'].empty:
                logger.warning("批次数据中缺少收盘价数据")
                return None
            
            # 获取索引(股票代码列表)和列(datetime)
            stock_codes = batch_data['close'].index.tolist()
            datetimes = batch_data['close'].columns.tolist()
            
            if not stock_codes or not datetimes:
                logger.warning("批次数据中股票代码或日期时间为空")
                return None
                
            # 创建多索引的空DataFrame
            rows = []
            
            # 对每个股票和每个时间点
            for stock in stock_codes:
                for dt in datetimes:
                    # 创建一行数据
                    row = {'stock_code': stock, 'datetime': dt}
            
            # 添加每个字段的值
            for field in fields:
                if field in batch_data and not batch_data[field].empty:
                    try:
                        row[field] = batch_data[field].loc[stock, dt]
                    except:
                        row[field] = np.nan                 
                    
                    rows.append(row)
            
            # 创建DataFrame
            result_df = pd.DataFrame(rows)
            
            # 处理日期和时间
            if not result_df.empty and 'datetime' in result_df.columns:
                # 确保datetime列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(result_df['datetime']):
                    result_df['datetime'] = pd.to_datetime(result_df['datetime'], errors='coerce')
                
                # 提取日期字符串
                result_df['date_str'] = result_df['datetime'].dt.strftime('%Y%m%d')
                
                # 提取时间字符串
                result_df['time_str'] = result_df['datetime'].dt.strftime('%H%M%S')
            
            logger.info(f"xtdata数据处理完成，返回 {len(result_df)} 行数据")
            return result_df
            
        except Exception as e:
            logger.error(f"处理和重塑批次数据时出错: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _process_and_save_partition(self, partition_key, partition_dfs):
        """
        Processes a list of DataFrames for a single partition:
        Concatenates, sorts, adds time_str, optimizes types, and saves to Parquet.
        """
        if not partition_dfs:
            logger.warning(f"No data received for partition {partition_key}")
            return (partition_key, 0)

        pid = os.getpid()
        logger.info(f"[PID {pid}] Processing partition: {partition_key}, combining {len(partition_dfs)} DataFrames.")
        combine_start = time.time()
        try:
            # Concatenate all DataFrames for this partition
            partition_df = pd.concat(partition_dfs, ignore_index=True)
            logger.info(f"[PID {pid}] Partition {partition_key} combined shape: {partition_df.shape}. Time: {time.time()-combine_start:.2f}s")

            # --- Essential Post-processing ---
            # 1. Sort by stock_code and datetime (CRITICAL, was the memory bottleneck before)
            sort_start = time.time()
            partition_df = partition_df.sort_values(by=['stock_code', 'datetime']).reset_index(drop=True)
            logger.info(f"[PID {pid}] Partition {partition_key} sorted. Time: {time.time()-sort_start:.2f}s")

            # 2. Add time_str column
            partition_df['time_str'] = partition_df['datetime'].dt.strftime('%H%M%S')

            # 3. Type Optimization
            optimize_start = time.time()
            for col in partition_df.columns:
                if col in ['stock_code', 'date_str', 'time_str', 'datetime']:
                    continue
                if partition_df[col].dtype == 'object':
                     partition_df[col] = pd.to_numeric(partition_df[col], errors='coerce')
                if str(partition_df[col].dtype) == 'boolean':
                     partition_df[col] = partition_df[col].astype('bool')
                if partition_df[col].dtype == 'float64':
                     partition_df[col] = partition_df[col].astype('float32')
                elif partition_df[col].dtype == 'int64':
                     if not partition_df[col].isnull().any():
                          min_val, max_val = partition_df[col].min(), partition_df[col].max()
                          if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                               partition_df[col] = partition_df[col].astype('int32')
                          elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                               partition_df[col] = partition_df[col].astype('int16')
            logger.info(f"[PID {pid}] Partition {partition_key} types optimized. Time: {time.time()-optimize_start:.2f}s")


            # --- Saving ---
            save_start = time.time()
            partition_dir = MINUTE_DIR / partition_key
            partition_dir.mkdir(exist_ok=True, parents=True)
            partition_file = partition_dir / "minute_data.parquet"
            
            # Drop the datetime column before saving if needed
            if 'datetime' in partition_df.columns:
                 partition_df = partition_df.drop(columns=['datetime'])

            table = pa.Table.from_pandas(partition_df, preserve_index=False)
            pq.write_table(table, partition_file, compression='snappy')

            row_count = len(partition_df)
            logger.info(f"[PID {pid}] Partition {partition_key} saved ({row_count:,} rows). Time: {time.time()-save_start:.2f}s")
            return (partition_key, row_count)

        except MemoryError as me:
             logger.error(f"[PID {pid}] MemoryError processing partition {partition_key}. Shape before error: {partition_df.shape if 'partition_df' in locals() else 'N/A'}. Error: {me}")
             logger.error(traceback.format_exc())
             return (partition_key, -1) # Indicate memory error
        except Exception as e:
            logger.error(f"[PID {pid}] Error processing/saving partition {partition_key}: {e}")
            logger.error(traceback.format_exc())
            return (partition_key, 0) # Indicate general error
        finally:
            # Explicitly delete large objects to help GC
            if 'partition_df' in locals(): del partition_df
            if 'table' in locals(): del table
            gc.collect()

    def download_minute_data(self, stock_list=None, start_date=None, end_date=None, batch_size=50, max_workers=5, use_multithreading=False, use_multiprocessing=True):
        """下载5分钟数据

        参数:
            stock_list: 股票列表，None表示所有股票
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            batch_size: 每批处理的股票数
            max_workers: 并行下载的线程/进程数
            use_multithreading: 是否使用多线程，默认为False(单线程模式更稳定)
            use_multiprocessing: 是否使用多进程，默认为True(多进程模式可绕过BaoStock单线程限制)
        """
        if start_date is None or end_date is None:
            raise ValueError("必须指定开始日期和结束日期")

        # 冲突检查：多线程与多进程不能同时使用
        if use_multithreading and use_multiprocessing:
            logger.warning("多线程和多进程不能同时使用，优先使用多进程")
            use_multithreading = False
        
        # 限制最大工作者数量
        max_workers = min(max_workers, 5)  # 不超过5个工作者
        
        # 多进程模式下，更少的工作者通常更稳定
        if use_multiprocessing:
            max_workers = min(max_workers, 4)  # 多进程模式下最多3个进程
        
        # 获取要下载的股票列表
        if stock_list is None:
            stock_list = self.metadata.get('stock_list', [])
            if not stock_list:
                logger.warning("未找到股票列表，请先下载股票列表")
                return
        else:
            # 确保stock_list是列表类型
            if isinstance(stock_list, str):
                stock_list = [stock_list]
        
        # 确保日期格式正确
        start_date = start_date.replace('-', '')
        end_date = end_date.replace('-', '')
        
        # 获取相关日期范围的分区
        start_partition = self._get_partition_key(start_date)
        end_partition = self._get_partition_key(end_date)
        
        # 确保分区目录存在
        current_partition = start_partition
        while current_partition <= end_partition:
            partition_dir = MINUTE_DIR / current_partition
            if not partition_dir.exists():
                partition_dir.mkdir(parents=True, exist_ok=True)
            
            # 计算下一个分区
            year = int(current_partition[:4])
            month = int(current_partition[4:6])
            
            month += 1
            if month > 12:
                year += 1
                month = 1
            
            current_partition = f"{year:04d}{month:02d}"
        
        # 添加分区到元数据
        partitions = set(self.metadata['partitions']['minute'])
        current_partition = start_partition
        while current_partition <= end_partition:
            partitions.add(current_partition)
            
            # 计算下一个分区
            year = int(current_partition[:4])
            month = int(current_partition[4:6])
            
            month += 1
            if month > 12:
                year += 1
                month = 1
            
            current_partition = f"{year:04d}{month:02d}"
        
        self.metadata['partitions']['minute'] = sorted(list(partitions))
        self._save_metadata()
        
        # 确定要下载的字段
        fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        # 批量下载并处理数据
        total_stocks = len(stock_list)
        logger.info(f"开始下载 {total_stocks} 支股票的5分钟数据，时间范围: {start_date} 至 {end_date}")
        
        # 调整batch_size，确保不会过小
        batch_size = max(batch_size, 5)
        
        # 将股票列表分成多个批次
        batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]
        logger.info(f"共分为 {len(batches)} 个批次，每批次最多 {batch_size} 支股票")
        logger.info(f"使用{'多进程' if use_multiprocessing else '单线程'}模式下载数据")
        
        # 存储所有批次数据
        all_batch_data = {}
        
        start_time = time.time()
        
        # 根据模式选择处理方式
        if use_multiprocessing:
            # 多进程模式 - 使用进程池
            import multiprocessing
            from multiprocessing import Pool
            import random
            import os
            
            # 配置日志
            max_processes = min(max_workers, multiprocessing.cpu_count())
            logger.info(f"启动多进程池下载模式，CPU核心数: {multiprocessing.cpu_count()}, 使用进程数: {max_processes}")
            
            # 准备进程池的参数
            # 将所有批次分组，每个进程处理多个批次
            process_batches = []
            for i in range(max_processes):
                # 计算每个进程应该处理的批次
                process_batches.append([])
            
            # 将批次分配给进程（轮询方式）
            for i, batch in enumerate(batches):
                process_idx = i % max_processes
                process_batches[process_idx].append(batch)
            
            # 准备进程池的参数，每个进程处理多个批次
            process_args = []
            for i in range(max_processes):
                if process_batches[i]:  # 确保这个进程有批次要处理
                    # 合并该进程的所有批次
                    all_stocks_for_process = []
                    for batch in process_batches[i]:
                        all_stocks_for_process.extend(batch)
                    
                    # 只有在有股票要处理时才添加任务
                    if all_stocks_for_process:
                        process_args.append((all_stocks_for_process, start_date, end_date, fields, i, None))
            
            logger.info(f"将{len(batches)}个批次分配给{len(process_args)}个进程，每个进程平均处理约{len(batches)/max(len(process_args), 1):.1f}个批次")
            
            try:
                # 使用进程池处理所有批次
                with Pool(processes=max_processes) as pool:
                    # 使用全局函数处理
                    results = pool.map(process_batch_for_multiprocessing, process_args)
                    
                    # 关闭进程池并等待所有任务完成
                    pool.close()
                    pool.join()
                
                # 收集所有进程的结果
                for result in results:
                    if result and len(result) == 2:
                        process_id, batch_data = result
                        if batch_data:
                            all_batch_data.update(batch_data)
                            logger.info(f"从进程 {process_id} 收集数据: {len(batch_data)} 只股票")
                
                logger.info(f"所有进程池任务完成，共处理 {len(all_batch_data)} 只股票的数据")
                
            except Exception as e:
                logger.error(f"进程池执行过程中发生错误: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            # 单线程模式
            for i, batch in enumerate(batches):
                logger.info(f"处理批次 {i+1}/{len(batches)} ({(i+1)/len(batches)*100:.1f}%)")
                try:
                    batch_data = self._download_minute_batch(batch, start_date, end_date, fields)
                    if batch_data:
                        all_batch_data.update(batch_data)
                        logger.info(f"批次 {i+1} 获取了 {len(batch_data)} 只股票的数据")
                    else:
                        logger.warning(f"批次 {i+1} 未获取到数据")
                except Exception as e:
                    logger.error(f"批次 {i+1} 下载失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
        
        total_time = time.time() - start_time
        logger.info(f"下载耗时: {total_time:.2f}秒，平均每只股票 {total_time/total_stocks:.2f}秒")
        
        # 按日期分区处理数据
        if all_batch_data:
            # 转换为DataFrame列表
            all_dfs = list(all_batch_data.values())
            
            # 按日期分组，形成分区数据
            partition_dfs = {}
            
            for df in all_dfs:
                if df is None or df.empty:
                    continue
                
                # 确保日期字段存在
                if 'date_str' not in df.columns and 'datetime' in df.columns:
                    df['date_str'] = df['datetime'].dt.strftime('%Y%m%d')
                
                # 按日期分组
                for date, date_df in df.groupby('date_str'):
                    # 获取分区键
                    partition_key = self._get_partition_key(date)
                    
                    # 添加到相应分区
                    if partition_key not in partition_dfs:
                        partition_dfs[partition_key] = []
                    
                    partition_dfs[partition_key].append(date_df)
            
            # 处理并保存每个分区
            for partition_key, dfs in partition_dfs.items():
                self._process_and_save_partition(partition_key, dfs)
            
            logger.info(f"成功下载并保存 {len(all_batch_data)} 支股票的5分钟数据，共 {sum(len(df) for df in all_dfs if df is not None)} 条记录")
        else:
            logger.warning("未成功下载任何5分钟数据")
    
    def get_trading_calendar(self, start_date=None, end_date=None):
        """获取交易日历
        
        参数:
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            
        返回:
            pd.DataFrame: 交易日历数据
        """
        calendar_file = CALENDAR_DIR / "trading_calendar.parquet"
        if not calendar_file.exists():
            logger.warning("交易日历文件不存在，尝试下载")
            self.download_trading_calendar()
        print(f"address:{calendar_file}")
        try:
            calendar_df = pd.read_parquet(calendar_file)
            print(f"范围：{calendar_df}")
            
            # 过滤掉特殊值'date_str'
            if 'date_str' in calendar_df.columns:
                invalid_mask = calendar_df['date_str'] == 'date_str'
                if invalid_mask.any():
                    logger.warning(f"从交易日历中移除了 {invalid_mask.sum()} 个特殊值'date_str'")
                    calendar_df = calendar_df[~invalid_mask]
            # 如果指定了日期范围，则筛选
            if start_date or end_date:
                if start_date:
                    calendar_df = calendar_df[calendar_df['date_str'] >= start_date]
                if end_date:
                    calendar_df = calendar_df[calendar_df['date_str'] <= end_date]
            
            return calendar_df
        except Exception as e:
            logger.error(f"读取交易日历失败: {e}")
            return pd.DataFrame()
    
    def get_daily_data(self, stock_list=None, start_date=None, end_date=None, fields=None):
        """获取日线数据 (使用 pandas 直接读取 parquet 文件)
        
        参数:
            stock_list: 股票列表，None表示所有股票
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            fields: 字段列表，None表示读取 Parquet 文件中的所有可用字段
            
        返回:
            pd.DataFrame: 日线数据
        """
        if start_date is None or end_date is None:
            logger.error("必须指定开始日期和结束日期")
            return pd.DataFrame()
        
        try:
            # 尝试解析日期，验证格式
            datetime.strptime(start_date, '%Y%m%d')
            datetime.strptime(end_date, '%Y%m%d')
        except ValueError:
            logger.error(f"开始或结束日期格式无效: {start_date}, {end_date}。请使用 YYYYMMDD 格式。")
            return pd.DataFrame()

        # 定义数据目录路径
        daily_dir_path = get_data_path("daily")
        if not daily_dir_path.exists():
            logger.error(f"日线数据基础目录不存在: {daily_dir_path}")
            # 尝试下载数据
            logger.info("尝试下载日线数据...")
            success = self.download_daily_data(stock_list, start_date, end_date)
            if not success:
                logger.error("下载日线数据失败")
                return pd.DataFrame()
            # 如果下载成功，继续执行
        
        logger.info(f"尝试从 {daily_dir_path} 读取数据，日期范围: {start_date} 到 {end_date}")

        try:
            # 生成日期范围内需要的月份分区
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')

            # 生成需要读取的月份列表
            months_to_read = set()
            current_dt = start_dt.replace(day=1)  # 从月初开始
            while current_dt <= end_dt:
                month_str = current_dt.strftime('%Y%m')
                months_to_read.add(month_str)
                # 移动到下个月
                if current_dt.month == 12:
                    current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
                else:
                    current_dt = current_dt.replace(month=current_dt.month + 1)

            logger.info(f"需要读取的月份分区: {sorted(months_to_read)}")

            all_data = []
            date_field = None
            stock_field = None

            # 遍历每个月份分区
            for month_str in sorted(months_to_read):
                partition_dir = daily_dir_path / month_str
                partition_file = partition_dir / "daily_data.parquet"

                if partition_file.exists():
                    try:
                        logger.info(f"读取月份分区文件: {partition_file}")

                        # 直接使用pandas读取parquet文件，避免PyArrow过滤器问题
                        partition_df = pd.read_parquet(partition_file)

                        if partition_df.empty:
                            logger.warning(f"月份分区 {month_str} 数据为空")
                            continue

                        # 每个分区都重新检测字段名（修复列名不一致问题）
                        available_fields = partition_df.columns.tolist()
                        logger.info(f"分区 {month_str} 可用字段: {available_fields}")

                        # 检测当前分区的日期字段名
                        current_date_field = None
                        for field_name in ['date_str', 'date', 'trade_date', 'trading_date']:
                            if field_name in available_fields:
                                current_date_field = field_name
                                break

                        # 检测当前分区的股票代码字段名
                        current_stock_field = None
                        for field_name in ['stock_code', 'code', 'symbol', 'ts_code']:
                            if field_name in available_fields:
                                current_stock_field = field_name
                                break

                        if current_date_field is None:
                            logger.error(f"分区 {month_str} 未找到日期字段，可用字段: {available_fields}")
                            continue  # 跳过这个分区而不是返回空DataFrame

                        if current_stock_field is None:
                            logger.error(f"分区 {month_str} 未找到股票代码字段，可用字段: {available_fields}")
                            continue  # 跳过这个分区而不是返回空DataFrame

                        logger.info(f"分区 {month_str} 使用字段映射: 日期字段='{current_date_field}', 股票字段='{current_stock_field}'")

                        # 设置全局字段名（用于第一次检测）
                        if date_field is None:
                            date_field = current_date_field
                        if stock_field is None:
                            stock_field = current_stock_field

                        # 确保日期字段格式一致（使用当前分区检测到的字段名）
                        if current_date_field in partition_df.columns:
                            # 如果日期字段是datetime类型，转换为字符串
                            if pd.api.types.is_datetime64_any_dtype(partition_df[current_date_field]):
                                partition_df['date_str'] = partition_df[current_date_field].dt.strftime('%Y%m%d')
                            else:
                                # 否则直接使用原始字段
                                partition_df['date_str'] = partition_df[current_date_field].astype(str)

                            # 如果当前字段不是date_str，确保date_str列存在
                            if current_date_field != 'date_str' and 'date_str' not in partition_df.columns:
                                if pd.api.types.is_datetime64_any_dtype(partition_df[current_date_field]):
                                    partition_df['date_str'] = partition_df[current_date_field].dt.strftime('%Y%m%d')
                                else:
                                    partition_df['date_str'] = partition_df[current_date_field].astype(str)
                                logger.info(f"分区 {month_str} 从 '{current_date_field}' 创建了 'date_str' 列")
                        else:
                            # 如果没有找到日期字段，尝试从其他列创建
                            logger.warning(f"分区 {month_str} 中缺少日期字段 '{current_date_field}'，尝试从其他列创建")
                            for alt_field in ['date', 'trade_date', 'trading_date', 'time']:
                                if alt_field in partition_df.columns:
                                    if pd.api.types.is_datetime64_any_dtype(partition_df[alt_field]):
                                        partition_df['date_str'] = partition_df[alt_field].dt.strftime('%Y%m%d')
                                    else:
                                        partition_df['date_str'] = partition_df[alt_field].astype(str)
                                    logger.info(f"使用 '{alt_field}' 创建了 'date_str' 列")
                                    break
                            else:
                                # 如果所有尝试都失败，使用分区月份作为日期的一部分
                                logger.warning(f"无法从现有列创建日期字段，使用分区月份 {month_str} 作为日期前缀")
                                partition_df['date_str'] = month_str + '01'  # 默认使用月份的第一天

                        # 按日期范围过滤
                        if 'date_str' in partition_df.columns:
                            logger.info(f"分区 {month_str} 日期过滤前: {len(partition_df)} 行")
                            logger.info(f"分区 {month_str} 日期范围: {partition_df['date_str'].min()} - {partition_df['date_str'].max()}")
                            logger.info(f"请求日期范围: {start_date} - {end_date}")

                            date_mask = (partition_df['date_str'] >= start_date) & (partition_df['date_str'] <= end_date)
                            partition_df = partition_df[date_mask]

                            logger.info(f"分区 {month_str} 日期过滤后: {len(partition_df)} 行")
                            if len(partition_df) == 0:
                                logger.warning(f"分区 {month_str} 日期过滤后无数据，可能请求的日期不在数据范围内")
                                available_dates = sorted(pd.read_parquet(partition_file)['date_str'].unique())
                                logger.info(f"分区 {month_str} 可用日期: {available_dates[:10]}...（显示前10个）")
                        else:
                            logger.warning(f"分区 {month_str} 中缺少 'date_str' 列，无法按日期过滤")

                        # 按股票代码过滤（使用当前分区检测到的字段名）
                        if stock_list and current_stock_field in partition_df.columns:
                            logger.info(f"分区 {month_str} 股票过滤前: {len(partition_df)} 行")
                            logger.info(f"请求的股票数量: {len(stock_list)}")
                            logger.info(f"请求的股票样本: {stock_list[:5]}")

                            stock_list_str = [str(s) for s in stock_list]
                            available_stocks = set(partition_df[current_stock_field].unique())
                            requested_stocks = set(stock_list_str)
                            matched_stocks = requested_stocks & available_stocks

                            logger.info(f"分区 {month_str} 可用股票数量: {len(available_stocks)}")
                            logger.info(f"分区 {month_str} 匹配的股票数量: {len(matched_stocks)}")

                            partition_df = partition_df[partition_df[current_stock_field].isin(stock_list_str)]
                            logger.info(f"分区 {month_str} 股票过滤后: {len(partition_df)} 行")

                            if len(partition_df) == 0:
                                logger.warning(f"分区 {month_str} 股票过滤后无数据")
                                missing_stocks = requested_stocks - available_stocks
                                if missing_stocks:
                                    logger.warning(f"分区 {month_str} 缺失的股票: {list(missing_stocks)[:10]}")
                        elif stock_list:
                            logger.warning(f"分区 {month_str} 缺少股票字段 '{current_stock_field}'，跳过股票过滤")

                        # 按字段过滤（使用当前分区检测到的字段名）
                        if fields and not partition_df.empty:
                            # 保证基础列总是被读取
                            base_cols = [current_stock_field, current_date_field, 'date_str']
                            if 'date_str' not in base_cols:
                                base_cols.append('date_str')
                            fields_str = [str(f) for f in fields]
                            read_columns = list(set(base_cols + fields_str))

                            # 检查请求的列是否存在
                            valid_read_columns = [col for col in read_columns if col in partition_df.columns]
                            if len(valid_read_columns) < len(read_columns):
                                missing_cols = set(read_columns) - set(valid_read_columns)
                                logger.debug(f"月份分区 {month_str} 中缺少字段: {missing_cols}")

                            partition_df = partition_df[valid_read_columns]

                        if not partition_df.empty:
                            all_data.append(partition_df)
                            logger.info(f"月份分区 {month_str} 读取成功，{len(partition_df)} 行数据")

                    except Exception as e:
                        logger.warning(f"读取月份分区 {month_str} 失败: {e}")
                        logger.warning(traceback.format_exc())
                else:
                    logger.warning(f"月份分区文件不存在: {partition_file}")
            
            # 如果没有找到数据，不自动下载（临时禁用以防止数据被覆盖）
            if not all_data:
                logger.warning(f"在日期范围 {start_date}-{end_date} 内未找到任何数据")
                logger.error("数据缺失，已禁用自动下载。请检查:")
                logger.error("1. 请求的日期是否为交易日")
                logger.error("2. 请求的股票代码是否正确")
                logger.error("3. 数据文件是否完整")
                return pd.DataFrame()

                # 原自动下载代码（已禁用）
                # logger.info("尝试下载日线数据...")
                # success = self.download_daily_data(stock_list, start_date, end_date)
                # if success:
                #     # 重新尝试读取数据
                #     logger.info("下载成功，重新尝试读取数据...")
                #     return self.get_daily_data(stock_list, start_date, end_date, fields)
                # else:
                #     logger.error("下载日线数据失败")
                #     return pd.DataFrame()

            if not all_data:
                logger.warning("没有找到符合条件的数据")
                return pd.DataFrame()

            # --- 在合并前确保所有分区都有一致的date字段 ---
            logger.info("正在标准化分区的date字段...")

            standardized_data = []
            for i, partition_df in enumerate(all_data):
                # 为每个分区创建副本以避免修改原始数据
                df_copy = partition_df.copy()

                # 优先使用date_str字段重建date字段，确保数据质量
                if 'date_str' in df_copy.columns:
                    logger.debug(f"分区 {i} 从date_str重建date字段")
                    df_copy['date'] = pd.to_datetime(df_copy['date_str'], format='%Y%m%d', errors='coerce')

                    # 检查转换失败的情况
                    failed_count = df_copy['date'].isna().sum()
                    if failed_count > 0:
                        logger.warning(f"分区 {i} 有 {failed_count} 行date转换失败，将被移除")
                        df_copy = df_copy.dropna(subset=['date'])
                elif 'date' in df_copy.columns:
                    # 如果只有date字段，确保其为datetime类型
                    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                        logger.debug(f"分区 {i} 转换date字段为datetime类型")
                        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
                        df_copy = df_copy.dropna(subset=['date'])
                    else:
                        # 即使是datetime类型，也要检查NaT值
                        nat_count = df_copy['date'].isna().sum()
                        if nat_count > 0:
                            logger.warning(f"分区 {i} 的date字段有 {nat_count} 个NaT值，将被移除")
                            df_copy = df_copy.dropna(subset=['date'])
                else:
                    logger.warning(f"分区 {i} 既没有date字段也没有date_str字段，跳过")
                    continue

                standardized_data.append(df_copy)

            if not standardized_data:
                logger.error("所有分区的date字段标准化都失败了")
                return pd.DataFrame()

            # 现在合并标准化后的数据
            df = pd.concat(standardized_data, ignore_index=True)
            logger.info(f"成功合并 {len(standardized_data)} 个标准化分区的数据，总计 {len(df)} 行")

            # --- 后续处理 ---
            # 标准化字段名：确保有统一的date_str和stock_code字段
            if stock_field != 'stock_code' and stock_field in df.columns:
                df['stock_code'] = df[stock_field]

            # 确保 'date_str' 列存在
            if 'date_str' not in df.columns:
                logger.warning("合并后的数据中缺少 'date_str' 列，尝试创建")
                if date_field in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[date_field]):
                        df['date_str'] = df[date_field].dt.strftime('%Y%m%d')
                    else:
                        df['date_str'] = df[date_field].astype(str)
                else:
                    logger.error("无法创建 'date_str' 列，因为缺少日期字段")
                    # 创建一个默认的日期字符串列，使用开始日期
                    df['date_str'] = start_date
                    logger.warning(f"创建了默认的 'date_str' 列，值为 {start_date}")

            # 最终验证date字段（应该已经在合并前处理过了）
            if 'date' not in df.columns:
                logger.error("合并后仍然没有date字段，这不应该发生")
                return pd.DataFrame()

            # 检查是否有NaT值
            nat_count = df['date'].isna().sum()
            if nat_count > 0:
                logger.warning(f"合并后发现 {nat_count} 个NaT值，将被移除")
                df = df.dropna(subset=['date'])

            # 确保date字段是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                logger.warning("date字段不是datetime类型，尝试转换...")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])

            # 按日期和股票代码排序
            sort_cols = []
            if 'date' in df.columns: sort_cols.append('date')
            if 'stock_code' in df.columns: sort_cols.append('stock_code')
            if sort_cols:
                try:
                    df = df.sort_values(by=sort_cols).reset_index(drop=True)
                    logger.debug("数据已按日期和股票代码排序")
                except Exception as sort_err:
                    logger.warning(f"排序数据时出错: {sort_err}")
            else:
                logger.warning("缺少 'date' 或 'stock_code' 列，无法排序")

            logger.info(f"成功读取并处理日线数据，最终返回 {len(df)} 行")
            return df

        except Exception as e:
            logger.error(f"读取日线数据时发生错误: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_minute_data(self, stock_list=None, start_date=None, end_date=None, fields=None):
        """获取5分钟数据
        
        参数:
            stock_list: 股票列表，None表示所有股票
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            fields: 字段列表，None表示所有字段
            
        返回:
            pd.DataFrame: 5分钟数据
        """
        if start_date is None or end_date is None:
            logger.error("必须指定开始日期和结束日期")
            return pd.DataFrame()
        
        # 解析日期和时间部分
        start_date_part, start_time_part = self._parse_datetime_str(start_date)
        end_date_part, end_time_part = self._parse_datetime_str(end_date)
        
        if start_date_part is None or end_date_part is None:
            logger.error(f"日期格式无效: {start_date}, {end_date}")
            return pd.DataFrame()
            
        logger.info(f"获取5分钟数据: 日期范围 {start_date_part} 至 {end_date_part}" + 
                    (f", 时间范围 {start_time_part} 至 {end_time_part}" if start_time_part and end_time_part else ""))
        
        # 确定需要的分区
        start_partition = self._get_partition_key(start_date_part)
        end_partition = self._get_partition_key(end_date_part)
        
        # 获取所有涉及的分区
        available_partitions = self.metadata['partitions']['minute']
        relevant_partitions = [p for p in available_partitions 
                               if p >= start_partition and p <= end_partition]
        
        logger.info(f"找到 {len(relevant_partitions)} 个相关分区: {relevant_partitions}")
        
        # 记录成功获取的分区
        successful_dates = set()
        all_data = []
        
        # 读取每个分区的数据
        for partition in relevant_partitions:
            partition_dir = MINUTE_DIR / partition
            partition_file = partition_dir / "minute_data.parquet"
            
            if not partition_file.exists():
                logger.warning(f"分区 {partition} 的数据文件不存在: {partition_file}")
                continue
            
            try:
                # 读取分区数据
                logger.info(f"正在读取分区 {partition} 的数据文件: {partition_file}")
                partition_data = pd.read_parquet(partition_file)
                logger.info(f"分区 {partition} 读取成功，原始数据包含 {len(partition_data)} 行")
                
                if partition_data.empty:
                    logger.warning(f"分区 {partition} 的数据为空")
                    continue
                
                # 检查并输出分区数据的列
                logger.info(f"分区 {partition} 数据列: {partition_data.columns.tolist()}")
                
                # 重建datetime列
                if 'datetime' not in partition_data.columns:
                    logger.info(f"分区 {partition} 数据中缺少datetime列，尝试重建")
                    
                    # 优先使用date_str和time_str (如果存在)，因为保存时只保留了这些列
                    if 'date_str' in partition_data.columns and 'time_str' in partition_data.columns:
                        try:
                            logger.info(f"使用'date_str'和'time_str'列重建datetime")
                            # 将YYYYMMDD格式转换为YYYY-MM-DD
                            date_formatted = partition_data['date_str'].str[:4] + '-' + \
                                           partition_data['date_str'].str[4:6] + '-' + \
                                           partition_data['date_str'].str[6:8]
                            
                            # 检查时间格式并标准化
                            sample_time = partition_data['time_str'].iloc[0] if not partition_data.empty else ""
                            logger.info(f"时间字符串样例: '{sample_time}'")
                            
                            # 创建一个函数来处理时间格式问题
                            def format_time(x):
                                return self._format_time_str(x, "00:00:00")
                                
                            # 应用格式化
                            partition_data['time_format'] = partition_data['time_str'].apply(format_time)
                            
                            # 合并日期和时间，使用更明确的格式
                            partition_data['datetime'] = pd.to_datetime(
                                date_formatted + ' ' + partition_data['time_format'],
                                format='%Y-%m-%d %H:%M:%S',
                                errors='coerce'
                            )
                            
                            # 检查结果
                            valid_count = partition_data['datetime'].notna().sum()
                            logger.info(f"成功重建datetime列，有效值数量: {valid_count}")
                            
                            # 如果重建失败，尝试直接使用日期
                            if valid_count == 0:
                                logger.warning("使用date_str和time_format重建datetime失败，尝试仅使用日期")
                                try:
                                    partition_data['datetime'] = pd.to_datetime(date_formatted, errors='coerce')
                                    valid_count = partition_data['datetime'].notna().sum()
                                    logger.info(f"使用日期重建，有效值数量: {valid_count}")
                                except Exception as e:
                                    logger.error(f"使用日期重建datetime失败: {e}")
                        except Exception as e:
                            logger.error(f"使用date_str和time_str重建datetime失败: {e}")
                            logger.error(traceback.format_exc())
                    
                    # 然后才尝试使用date和time_str (如果存在)
                    elif 'date' in partition_data.columns and 'time_str' in partition_data.columns:
                        try:
                            logger.info(f"使用'date'和'time_str'列重建datetime")
                            # 将date转为字符串，确保格式一致
                            date_str = partition_data['date'].astype(str)
                            
                            # 检查日期格式并进行标准化
                            sample_date = date_str.iloc[0] if not date_str.empty else ""
                            logger.info(f"日期字符串样例: '{sample_date}'")
                            
                            # 尝试将日期格式化为YYYY-MM-DD
                            try:
                                # 如果是类似 "2023-01-03" 这样的格式，直接使用
                                if '-' in sample_date and len(sample_date.split('-')) == 3:
                                    formatted_date = date_str
                                # 如果是类似 "20230103" 这样的格式，转换为 "2023-01-03"
                                elif len(sample_date) == 8 and sample_date.isdigit():
                                    formatted_date = date_str.str[:4] + '-' + date_str.str[4:6] + '-' + date_str.str[6:8]
                                else:
                                    # 尝试使用pd.to_datetime来解析并重新格式化
                                    temp_date = pd.to_datetime(date_str, errors='coerce')
                                    formatted_date = temp_date.dt.strftime('%Y-%m-%d')
                                    
                                logger.info(f"格式化后的日期样例: '{formatted_date.iloc[0]}'")
                            except Exception as e:
                                logger.error(f"日期格式化失败: {e}")
                                formatted_date = date_str  # 使用原始日期字符串
                            
                            # 检查时间格式并标准化
                            sample_time = partition_data['time_str'].iloc[0] if not partition_data.empty else ""
                            logger.info(f"时间字符串样例: '{sample_time}'")
                            
                            # 创建一个函数来处理时间格式问题
                            def format_time(x):
                                return self._format_time_str(x, "00:00:00")
                                
                            # 应用格式化
                            partition_data['time_format'] = partition_data['time_str'].apply(format_time)
                            
                            # 合并日期和时间，尝试多种方式创建datetime
                            try:
                                logger.info("尝试方式1: 使用formatted_date + time_format")
                                partition_data['datetime'] = pd.to_datetime(
                                    formatted_date + ' ' + partition_data['time_format'],
                                    format='%Y-%m-%d %H:%M:%S',
                                    errors='coerce'
                                )
                                
                                # 检查结果
                                valid_count = partition_data['datetime'].notna().sum()
                                logger.info(f"方式1结果: 有效值数量: {valid_count}")
                                
                                # 如果第一种方式效果不好，尝试第二种
                                if valid_count < len(partition_data) * 0.1:  # 如果有效值小于10%
                                    logger.info("尝试方式2: 直接使用pd.to_datetime解析")
                                    # 直接使用pandas的智能解析
                                    partition_data['datetime'] = pd.to_datetime(
                                        date_str + ' ' + partition_data['time_format'],
                                        errors='coerce'
                                    )
                                    valid_count = partition_data['datetime'].notna().sum()
                                    logger.info(f"方式2结果: 有效值数量: {valid_count}")
                            except Exception as e:
                                logger.error(f"创建datetime失败: {e}")
                                # 尝试最基本的方法
                                try:
                                    partition_data['datetime'] = pd.to_datetime(date_str, errors='coerce')
                                except:
                                    # 如果所有方法都失败，创建一个空列
                                    partition_data['datetime'] = pd.NaT
                            
                            # 最终检查结果
                            valid_count = partition_data['datetime'].notna().sum()
                            logger.info(f"成功重建datetime列，有效值数量: {valid_count}")
                        except Exception as e:
                            logger.error(f"使用date和time_str重建datetime失败: {e}")
                            logger.error(traceback.format_exc())
                
                # 根据时间和股票筛选数据
                # 在这里添加按股票和日期范围筛选的代码
                
                # 如果有数据，则添加到结果列表
                if not partition_data.empty:
                    all_data.append(partition_data)
                
            except Exception as e:
                logger.error(f"处理分区 {partition} 数据失败: {e}")
                logger.error(traceback.format_exc())
        
        # 合并所有分区数据
        if not all_data:
            logger.warning("未找到任何符合条件的数据")
            return pd.DataFrame()

        try:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"合并后的数据大小: {len(result)} 行")
            return result
        except Exception as e:
            logger.error(f"合并分区数据失败: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _parse_datetime_str(self, datetime_str):
        """解析日期时间字符串，支持YYYYMMDD或YYYYMMDDHHMMSS格式
        
        参数:
            datetime_str: 日期时间字符串
            
        返回:
            tuple: (日期部分, 时间部分)，如果格式无效则返回(None, None)
        """
        if datetime_str is None:
            return None, None
        
        datetime_str = str(datetime_str).strip()
        
        try:
            if len(datetime_str) == 8:  # YYYYMMDD
                # 验证日期格式
                datetime.strptime(datetime_str, '%Y%m%d')
                return datetime_str, None
            elif len(datetime_str) == 14:  # YYYYMMDDHHMMSS
                # 验证日期时间格式
                datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                return datetime_str[:8], datetime_str[8:]
            else:
                logger.warning(f"无法识别的日期时间格式: {datetime_str}")
                return None, None
        except ValueError as e:
            logger.error(f"日期时间格式无效: {datetime_str}, 错误: {e}")
            return None, None
    
    def update_stock_list(self):
        """更新股票列表"""
        return self.download_stock_list()
    
    def update_trading_calendar(self):
        """增量更新交易日历"""
        last_update = self.metadata['last_update']['calendar']
        if last_update:
            # 从上次更新的下一天开始更新
            last_date = datetime.strptime(last_update, '%Y%m%d')
            start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            print(f"last_date:{start_date} and end_date:{end_date}")
            
            if start_date <= end_date:
                return self.download_trading_calendar(start_date, end_date)
            else:
                logger.info("交易日历已是最新")
                return True
        else:
            # 如果从未更新过，则下载所有数据
            return self.download_trading_calendar()
    
    def update_daily_data(self, stock_list=None):
        """增量更新日线数据"""
        if stock_list is None:
            stock_list = self.metadata['stock_list']
            if not stock_list:
                stock_list = self.download_stock_list()
        
        last_update = self.metadata['last_update']['daily']
        if last_update:
            # 从上次更新的下一天开始更新
            last_date = datetime.strptime(last_update, '%Y%m%d')
            start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            logger.info(f"update start:{start_date}, end_date:{end_date}")
            if start_date <= end_date:
                return self.download_daily_data(stock_list, start_date, end_date)
            else:
                logger.info("日线数据已是最新")
                return True
        else:
            # 如果从未更新过，则下载所有数据
            return self.download_daily_data(stock_list)
    
    def update_minute_data(self, stock_list=None):
        """增量更新5分钟数据"""
        if stock_list is None:
            stock_list = self.metadata['stock_list']
            if not stock_list:
                stock_list = self.download_stock_list()
        
        last_update = self.metadata['last_update']['minute']
        if last_update:
            # 从上次更新的下一天开始更新
            last_date = datetime.strptime(last_update, '%Y%m%d')
            start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
            end_date = datetime.now().strftime('%Y%m%d')
            
            if start_date <= end_date:
                return self.download_minute_data(stock_list, start_date, end_date)
            else:
                logger.info("5分钟数据已是最新")
                return True
        else:
            # 如果从未更新过，则下载所有数据
            return self.download_minute_data(stock_list)

    def test_trading_calendar(self):
        """测试获取交易日历数据"""
        logger.info("测试获取交易日历数据...")
        
        # 读取交易日历
        calendar_df = self.get_trading_calendar()
        
        if calendar_df.empty:
            logger.warning("交易日历为空，请先下载交易日历数据")
            return
        
        # 打印交易日历统计信息
        logger.info(f"交易日历数据统计:")
        logger.info(f"  总交易日数量: {len(calendar_df)}")
        logger.info(f"  起始日期: {calendar_df['date_str'].min()}")
        logger.info(f"  结束日期: {calendar_df['date_str'].max()}")
        
        # 打印样例数据
        sample_size = min(5, len(calendar_df))
        logger.info(f"交易日历样例数据(前{sample_size}行):")
        print(calendar_df.head(sample_size))
        
        # 测试日期筛选
        start_date = '20230101'
        end_date = '20230331'
        filtered_df = self.get_trading_calendar(start_date, end_date)
        logger.info(f"2023年第一季度交易日数量: {len(filtered_df)}")
        
        return calendar_df
    
    def test_daily_data(self, start_date='20230101', end_date='20230131', sample_stocks=None):
        """测试获取日线数据
        
        参数:
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            sample_stocks: 测试用的股票列表，默认为前5只股票
        """
        logger.info(f"测试获取日线数据: {start_date} 至 {end_date}...")
        
        # 获取股票列表
        if sample_stocks is None:
            all_stocks = self.metadata['stock_list']
            if not all_stocks:
                logger.warning("股票列表为空，请先下载股票列表")
                return
            sample_stocks = all_stocks[:5]  # 取前5只股票作为样本
        
        logger.info(f"选取样本股票: {sample_stocks}")
        
        # 获取数据
        fields = ['open', 'high', 'low', 'close', 'volume']
        daily_df = self.get_daily_data(sample_stocks, start_date, end_date, fields)
        
        if daily_df.empty:
            logger.warning("日线数据为空，请检查数据是否已下载或日期范围是否合适")
            return
        
        # 打印统计信息
        logger.info(f"日线数据统计:")
        logger.info(f"  数据行数: {len(daily_df)}")
        logger.info(f"  数据列数: {len(daily_df.columns)}")
        logger.info(f"  日期范围: {daily_df['date_str'].min()} - {daily_df['date_str'].max()}")
        
        # 每只股票的数据点数
        for stock in sample_stocks:
            stock_data = daily_df[daily_df['stock_code'] == stock]
            logger.info(f"  {stock} 数据点数: {len(stock_data)}")
        
        # 打印样例数据
        logger.info(f"日线数据样例(前5行):")
        # print(daily_df.head(5))
        print(daily_df)
        
        # 计算每只股票的简单统计量
        logger.info("样本股票收盘价统计:")
        for stock in sample_stocks:
            stock_data = daily_df[daily_df['stock_code'] == stock]
            if not stock_data.empty and 'close' in stock_data.columns:
                close_prices = stock_data['close']
                stats = {
                    'mean': close_prices.mean(),
                    'std': close_prices.std(),
                    'min': close_prices.min(),
                    'max': close_prices.max()
                }
                logger.info(f"  {stock} 收盘价统计: {stats}")
        
        return daily_df
    
    def test_minute_data(self, start_date='20230101', end_date='20230103', sample_stocks=None):
        """测试5分钟数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            sample_stocks: 测试用的股票列表
        """
        logger.info(f"测试5分钟数据: {start_date} 至 {end_date}...")
        
        # 获取股票列表 - 使用确定在数据中存在的股票
        # if sample_stocks is None:
        #     # 如果日期在2024年之前，使用000008.SZ，否则使用000001.SZ
        #     if start_date < '20240101':
        #         sample_stocks = ['000008.SZ']  # 确定存在于早期分区的股票
        #         logger.info("使用000008.SZ来测试2023年数据")
        #     else:
        #         sample_stocks = ['000001.SZ']  # 适用于2024年及以后分区
        #         logger.info("使用000001.SZ来测试2024年及以后数据")
        
        logger.info(f"选取样本股票: {sample_stocks}")
        
        
        # 获取5分钟数据
        minute_df = self.get_minute_data(sample_stocks, start_date, end_date)
        
        # 如果获取数据失败，则尝试在线获取
        if minute_df is None:
            minute_df = pd.DataFrame()  # 确保minute_df不是None
        
        if minute_df.empty:
            logger.warning("未能获取到本地5分钟数据，尝试在线获取...")
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
            minute_df = self._get_online_minute_data(sample_stocks, start_date, end_date, fields)
            
            if minute_df is None or minute_df.empty:
                logger.warning("未能获取到在线5分钟数据")
            return
        
        logger.info(f"成功获取5分钟数据，共 {len(minute_df)} 行")
        
        # 检查数据正确性
        if 'close' in minute_df.columns and 'datetime' in minute_df.columns:
            # 统计每个股票的数据量
            stock_counts = minute_df.groupby('stock_code').size()
            logger.info(f"每支股票的数据量: {stock_counts}")
            
            # 统计总交易日天数
            date_counts = minute_df['date_str'].nunique()
            logger.info(f"总交易日天数: {date_counts}")
            
            # 计算每天的平均数据量
            per_day_avg = len(minute_df) / date_counts if date_counts > 0 else 0
            logger.info(f"每天平均数据量: {per_day_avg:.1f}")
        
        return minute_df
    
    def test_data_query(self, date='20230120', stock_code=None):
        """测试数据查询和过滤功能
        
        参数:
            date: 查询日期，格式 'YYYYMMDD'
            stock_code: 测试用的股票代码，默认为元数据中的第一只股票
        """
        logger.info(f"测试特定日期({date})的数据查询功能...")
        
        # 获取股票代码
        if stock_code is None:
            all_stocks = self.metadata['stock_list']
            if not all_stocks:
                logger.warning("股票列表为空，请先下载股票列表")
                return
            stock_code = all_stocks[0]
        
        logger.info(f"选取股票: {stock_code}")
        
        # 获取日线数据
        daily_df = self.get_daily_data([stock_code], date, date)
        
        if daily_df.empty:
            logger.warning(f"日期 {date} 的日线数据为空，可能是非交易日或数据未下载")
        else:
            logger.info(f"日线数据查询结果:")
            print(daily_df)
        
        # 获取5分钟数据
        minute_df = self.get_minute_data([stock_code], date, date)
        
        if minute_df.empty:
            logger.warning(f"日期 {date} 的5分钟数据为空，可能是非交易日或数据未下载")
        else:
            logger.info(f"5分钟数据查询结果数量: {len(minute_df)} 行")
            logger.info("5分钟数据样例:")
            print(minute_df.head())
            
            # 计算分钟数据的VWAP
            if 'close' in minute_df.columns and 'volume' in minute_df.columns:
                vwap = (minute_df['close'] * minute_df['volume']).sum() / minute_df['volume'].sum()
                logger.info(f"{stock_code} 在 {date} 的成交量加权平均价(VWAP): {vwap:.4f}")
        
        # 获取交易日历
        calendar_df = self.get_trading_calendar(date, date)
        
        if calendar_df.empty:
            logger.info(f"{date} 可能不是交易日")
        else:
            logger.info(f"{date} 是交易日")
        
        return {
            'daily': daily_df,
            'minute': minute_df,
            'calendar': calendar_df
        }

    def get_market_data(self, field_list=None, stock_list=None, period='1d', 
                       start_time=None, end_time=None, 
                       dividend_type='none', fill_data=True, subscribe=True):
        """获取市场数据，仿照xtdata.get_market_data的接口格式
        
        参数:
            field_list (list): 需要的字段列表，如['open', 'high', 'low', 'close', 'volume']
            stock_list (list): 股票代码列表
            period (str): 周期，支持 '1d'(日线), '5m'(5分钟线)
            start_time (str): 开始时间，格式 'YYYYMMDD'(日线)或'YYYYMMDDHHMMSS'(分钟线)
            end_time (str): 结束时间，格式 'YYYYMMDD'(日线)或'YYYYMMDDHHMMSS'(分钟线)
            dividend_type (str): 复权类型，不影响本地数据
            fill_data (bool): 是否填充数据，不影响本地数据
            subscribe (bool): 是否订阅数据，不影响本地数据
            
        返回:
            dict: 市场数据，格式为 {field1: df1, field2: df2, ...}，
                 其中df是DataFrame，行索引为股票代码，列为日期
        """
        logger.info(f"获取市场数据: period={period}, {len(stock_list) if stock_list else 0}支股票, "
                   f"{start_time}至{end_time}, {len(field_list) if field_list else 0}个字段")
        
        # 默认字段列表
        if field_list is None or len(field_list) == 0:
            field_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        # 默认股票列表 - 如果为空，使用所有股票
        if stock_list is None or len(stock_list) == 0:
            stock_list = self.metadata['stock_list']
            if not stock_list:
                logger.warning("股票列表为空，请先下载股票列表")
                return {}
        
        # 默认日期范围 - 如未指定，使用最近的交易日
        if start_time is None or end_time is None:
            # 读取交易日历获取最近的交易日
            calendar_df = self.get_trading_calendar()
            if calendar_df.empty:
                logger.warning("交易日历为空，请先下载交易日历")
                return {}
            
            # 默认使用最近10个交易日
            if end_time is None:
                end_time = calendar_df['date_str'].max()
            
            if start_time is None:
                # 获取结束日期之前的10个交易日
                end_idx = calendar_df[calendar_df['date_str'] <= end_time].index.max()
                if end_idx is not None and end_idx >= 10:
                    start_idx = end_idx - 9  # 总共10天
                    start_time = calendar_df.iloc[start_idx]['date_str']
                else:
                    # 如果交易日少于10天，使用第一个交易日
                    start_time = calendar_df['date_str'].min()
        
        start_time = str(start_time)
        end_time = str(end_time)
        
        # 根据period参数选择数据源
        raw_data = None
        
        if period == '1d':
            # 获取日线数据
            raw_data = self.get_daily_data(stock_list, start_time, end_time, field_list)
        elif period == '5m':
            # 获取5分钟数据
            raw_data = self.get_minute_data(stock_list, start_time, end_time, field_list)
        else:
            logger.error(f"不支持的周期类型: {period}，目前只支持 '1d' 和 '5m'")
            return {}
        
        if raw_data.empty:
            logger.warning(f"未获取到 {period} 周期的数据")
            return {}
        
        # 转换数据格式，与xtdata.get_market_data返回格式保持一致
        result = {}
        
        # 确保datetime列存在
        if 'datetime' not in raw_data.columns and 'date' in raw_data.columns:
            raw_data['datetime'] = raw_data['date']
            
            # 对每个字段创建一个DataFrame
            for field in field_list:
                if field in raw_data.columns:
                    # 创建透视表：行索引为股票代码，列索引为日期
                    if period == '1d':
                        # 日线数据使用date作为列索引
                        pivot_df = raw_data.pivot(index='stock_code', columns='date', values=field)
                    else:
                        # 分钟数据使用datetime作为列索引
                        pivot_df = raw_data.pivot(index='stock_code', columns='datetime', values=field)
                    
                    result[field] = pivot_df
                else:
                    logger.warning(f"请求的字段 '{field}' 在数据中不存在")
        
        # 添加time字段，保持与xtdata.get_market_data返回格式一致
        if period == '1d':
            # 日线数据的time字段是日期
            result['time'] = raw_data.pivot(index='stock_code', columns='date', values='date')
        else:
            # 分钟数据的time字段是datetime
            result['time'] = raw_data.pivot(index='stock_code', columns='datetime', values='datetime')
        
        logger.info(f"成功转换 {period} 周期数据为市场数据格式，共 {len(result) - 1} 个字段")
        return result
    
    def test_market_data_interface(self, start_date=None, end_date=None, sample_stocks=None):
        """测试市场数据接口
        
        参数:
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            sample_stocks: 测试用的股票列表，默认为前5只股票
        """
        logger.info("测试市场数据接口...")
        
        # 获取股票列表
        if sample_stocks is None:
            all_stocks = self.metadata['stock_list']
            if not all_stocks:
                logger.warning("股票列表为空，请先下载股票列表")
                return
            sample_stocks = all_stocks[:5]  # 取前5只股票作为样本
        
        logger.info(f"选取样本股票: {sample_stocks}")
        
        # 测试日线数据
        logger.info("\n测试日线数据接口...")
        fields = ['open', 'high', 'low', 'close', 'volume']
        daily_data = self.get_market_data(
            field_list=fields,
            stock_list=sample_stocks,
            period='1d',
            start_time=start_date,
            end_time=end_date
        )
        
        if not daily_data:
            logger.warning("获取日线数据失败")
        else:
            # 打印示例数据
            logger.info(f"成功获取日线数据，共{len(daily_data)}个字段")
            for field, df in daily_data.items():
                logger.info(f"字段 {field} 数据形状: {df.shape}")
                if not df.empty:
                    logger.info(f"字段 {field} 数据示例:")
                    print(df.iloc[:3, :3])  # 只显示前3行3列
            
            # 测试数据访问方式
            logger.info("\n测试数据访问方式:")
            if 'close' in daily_data and not daily_data['close'].empty:
                # 访问第一只股票的收盘价
                first_stock = sample_stocks[0]
                if first_stock in daily_data['close'].index:
                    stock_prices = daily_data['close'].loc[first_stock]
                    logger.info(f"股票 {first_stock} 的收盘价数据: {stock_prices.head()}")
                    
                    # 计算收盘价均值
                    mean_price = stock_prices.mean()
                    logger.info(f"股票 {first_stock} 收盘价均值: {mean_price:.4f}")
        
        # 测试分钟数据
        logger.info("\n测试分钟数据接口...")
        minute_data = self.get_market_data(
            field_list=fields,
            stock_list=sample_stocks[:2],  # 只测试前2只股票，减少数据量
            period='5m',
            start_time=start_date,
            end_time=end_date
        )
        
        if not minute_data:
            logger.warning("获取分钟数据失败")
        else:
            # 打印示例数据
            logger.info(f"成功获取分钟数据，共{len(minute_data)}个字段")
            for field, df in minute_data.items():
                logger.info(f"字段 {field} 数据形状: {df.shape}")
                if not df.empty:
                    logger.info(f"字段 {field} 数据示例:")
                    print(df.iloc[:2, :3])  # 只显示前2行3列
        
        return {
            'daily': daily_data,
            'minute': minute_data
        }

    def _get_online_minute_data(self, stock_codes, start_date, end_date, fields_list):
        """从在线获取5分钟数据
        
        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期，格式 'YYYYMMDD'
            fields_list: 字段列表
            
        返回:
            pd.DataFrame: 5分钟数据
        """
        import baostock as bs
        import traceback
        
        try:
            # 登录系统
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"登录失败: {lg.error_msg}")
                return None
            
            all_data = []
            field_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount'
            }
            
            # 转换字段名称
            baostock_fields = ['date', 'time', 'code']
            for field in fields_list:
                if field in field_mapping:
                    baostock_fields.append(field_mapping[field])
            
            baostock_fields_str = ",".join(baostock_fields)
            
            # 处理日期格式
            start_date_formatted = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_formatted = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            
            for stock_code in stock_codes:
                try:
                    # 转换股票代码格式从 000001.SZ 到 sz.000001
                    market, code = stock_code.split('.')
                    baostock_code = f"{market.lower()}.{code}"
                    
                    # 查询5分钟线数据
                    rs = bs.query_history_k_data_plus(
                        baostock_code,
                        baostock_fields_str,
                        start_date=start_date_formatted,
                        end_date=end_date_formatted,
                        frequency="5",  # 5分钟线
                        adjustflag="3"  # 不复权
                    )
                        
                    if rs.error_code != '0':
                        logger.warning(f"获取股票 {stock_code} 数据失败: {rs.error_msg}")
                        continue
                    
                    # 获取数据
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if not data_list:
                        logger.warning(f"股票 {stock_code} 没有获取到数据")
                        continue
                    
                    # 转换为DataFrame
                    stock_df = pd.DataFrame(data_list, columns=rs.fields)
                    
                    # 转换日期时间格式
                    stock_df['date_str'] = stock_df['date'].str.replace('-', '')
                    stock_df['time_str'] = stock_df['time']
                    stock_df['datetime'] = pd.to_datetime(stock_df['date'] + ' ' + stock_df['time'])
                    
                    # 转换数据类型
                    for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                        if field in stock_df.columns:
                            stock_df[field] = pd.to_numeric(stock_df[field], errors='coerce')
                    
                    # 添加股票代码(使用标准格式)
                    stock_df['stock_code'] = stock_code
                    
                    all_data.append(stock_df)
                except Exception as e:
                    logger.warning(f"处理股票 {stock_code} 数据时出错: {e}")
                    continue
            
            # 登出系统
            bs.logout()
            
            # 合并所有数据
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                return result_df
            else:
                return None
            
        except Exception as e:
            logger.error(f"在线获取5分钟数据时出错: {e}")
            logger.debug(traceback.format_exc())
            try:
                bs.logout()  # 确保异常情况下也能登出
            except:
                pass
            return None

    def _format_time_str(self, time_str, default_format=None):
        """格式化时间字符串为标准格式 HH:MM:SS
        
        参数:
            time_str: 原始时间字符串，如'093000'或'0930'
            default_format: 如果无法解析，返回的默认值
            
        返回:
            str: 格式化后的时间字符串，如'09:30:00'
        """
        if not time_str:
            return default_format
            
        try:
            # 如果是纯数字
            if time_str.isdigit():
                # 根据长度处理不同情况
                length = len(time_str)
                if length == 6:  # HHMMSS
                    return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                elif length == 4:  # HHMM
                    return f"{time_str[:2]}:{time_str[2:4]}:00"
                elif length == 5:  # 0HHMM
                    return f"0{time_str[:1]}:{time_str[1:3]}:{time_str[3:5]}"
                else:
                    # 尝试根据长度填充
                    padded = time_str.zfill(6)
                    return f"{padded[:2]}:{padded[2:4]}:{padded[4:6]}"
            
            # 如果已经有冒号分隔
            elif ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return time_str
                elif len(parts) == 2:  # HH:MM
                    return f"{parts[0]}:{parts[1]}:00"
            
            # 其他情况，返回默认值
            return default_format
        except Exception as e:
            logger.error(f"格式化时间字符串出错: {time_str}, 错误: {e}")
            return default_format

    def check_missing_stocks_in_minute_data(self, partition_key='202301'):
        """检查指定分区中哪些股票没有5分钟数据
        
        参数:
            partition_key: 分区键，格式为YYYYMM，默认为202301（2023年1月）
            
        返回:
            dict: 包含以下键值对:
                - 'missing_stocks': 在分区中没有数据的股票列表
                - 'available_stocks': 在分区中有数据的股票列表
                - 'stock_data_counts': 每只股票的数据点数量
        """
        import pyarrow.parquet as pq
        
        # 确保股票列表已经加载
        all_stocks = self.metadata['stock_list']
        if not all_stocks:
            logger.warning("股票列表为空，请先下载股票列表")
            return {'missing_stocks': [], 'available_stocks': [], 'stock_data_counts': {}}
        
        logger.info(f"检查分区 {partition_key} 中的股票数据...")
        
        # 获取分区文件路径
        partition_dir = MINUTE_DIR / partition_key
        partition_file = partition_dir / "minute_data.parquet"
        
        if not partition_file.exists():
            logger.error(f"分区文件不存在: {partition_file}")
            return {'missing_stocks': all_stocks, 'available_stocks': [], 'stock_data_counts': {}}
        
        try:
            # 读取整个分区文件
            logger.info(f"读取分区文件: {partition_file}")
            table = pq.read_table(partition_file)
            df = table.to_pandas()
            
            if df.empty:
                logger.warning(f"分区 {partition_key} 的数据为空")
                return {'missing_stocks': all_stocks, 'available_stocks': [], 'stock_data_counts': {}}
            
            # 获取分区中的股票列表
            available_stocks = df['stock_code'].unique().tolist()
            
            # 计算每只股票的数据点数量
            stock_data_counts = df['stock_code'].value_counts().to_dict()
            
            # 找出没有数据的股票
            missing_stocks = [stock for stock in all_stocks if stock not in available_stocks]
            
            # 输出统计信息
            total_stocks = len(all_stocks)
            missing_count = len(missing_stocks)
            available_count = len(available_stocks)
            logger.info(f"missing_stocks: {missing_stocks}")
            
            logger.info(f"总股票数: {total_stocks}")
            logger.info(f"有数据的股票数: {available_count} ({available_count/total_stocks*100:.2f}%)")
            logger.info(f"无数据的股票数: {missing_count} ({missing_count/total_stocks*100:.2f}%)")
            
            # 打印部分无数据的股票作为示例
            if missing_stocks:
                sample_size = min(10, len(missing_stocks))
                logger.info(f"部分无数据的股票 (前{sample_size}只): {missing_stocks[:sample_size]}")
            
            # 打印部分有数据的股票作为示例
            if available_stocks:
                sample_size = min(10, len(available_stocks))
                logger.info(f"部分有数据的股票 (前{sample_size}只): {available_stocks[:sample_size]}")
                
                # 打印每只股票的数据量（前10只）
                logger.info("部分股票的数据点数量:")
                for stock in available_stocks[:sample_size]:
                    logger.info(f"  {stock}: {stock_data_counts.get(stock, 0)} 条记录")
            
            return {
                'missing_stocks': missing_stocks,
                'available_stocks': available_stocks,
                'stock_data_counts': stock_data_counts
            }
        
        except Exception as e:
            logger.error(f"检查分区 {partition_key} 中的股票数据时出错: {e}")
            logger.error(traceback.format_exc())
            return {'missing_stocks': [], 'available_stocks': [], 'stock_data_counts': {}}

    def download_missing_stocks_data(self, partition_key='202301', start_date=None, end_date=None, batch_size=50, max_workers=5, use_multithreading=False, use_multiprocessing=True):
        """补充下载分区中缺失的股票数据
        
        参数:
            partition_key: 分区键，格式为YYYYMM，默认为202301
            start_date: 开始日期，默认为分区对应月份的第一天
            end_date: 结束日期，默认为分区对应月份的最后一天
            batch_size: 每批处理的股票数，默认50
            max_workers: 并行下载的线程/进程数，默认5
            use_multithreading: 是否使用多线程，默认False
            use_multiprocessing: 是否使用多进程，默认True
            
        返回:
            bool: 操作是否成功
        """
        global MINUTE_DIR
        import pyarrow as pa
        import pyarrow.parquet as pq
        import shutil
        from datetime import datetime, timedelta
        
        # 检查哪些股票数据缺失
        logger.info(f"检查分区 {partition_key} 中缺失的股票数据")
        result = self.check_missing_stocks_in_minute_data(partition_key)
        missing_stocks = result.get('missing_stocks', [])
        
        if not missing_stocks:
            logger.info(f"分区 {partition_key} 中没有缺失的股票数据")
            return True
        
        # 设置默认日期范围（如果未指定）
        if start_date is None:
            start_date = f"{partition_key}01"  # 分区月份的第一天
        
        if end_date is None:
            year = int(partition_key[:4])
            month = int(partition_key[4:6])
            # 计算月末日期
            if month == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month + 1
                next_year = year
            next_month_start = f"{next_year:04d}{next_month:02d}01"
            # 前一天即为当月最后一天
            end_date = (datetime.strptime(next_month_start, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        
        logger.info(f"开始补充下载分区 {partition_key} 中缺失的 {len(missing_stocks)} 只股票数据")
        logger.info(f"日期范围: {start_date} 至 {end_date}")
        
        # 备份原始数据
        partition_dir = MINUTE_DIR / partition_key
        partition_file = partition_dir / "minute_data.parquet"
        backup_file = partition_dir / f"minute_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet"
        
        if partition_file.exists():
            try:
                shutil.copy(str(partition_file), str(backup_file))
                logger.info(f"已备份原始数据到 {backup_file}")
            except Exception as e:
                logger.error(f"备份原始数据失败: {e}")
                return False
        
        # 读取原始数据（如果存在）
        original_df = None
        if partition_file.exists():
            try:
                original_df = pd.read_parquet(partition_file)
                logger.info(f"成功读取原始数据，共 {len(original_df)} 行")
            except Exception as e:
                logger.error(f"读取原始数据失败: {e}")
                return False
        
        # 下载缺失股票数据
        batch_size = min(batch_size, len(missing_stocks))  # 分批下载，避免一次处理太多
        
        # 临时修改分区的存储路径，避免覆盖原始数据
        temp_dir = MINUTE_DIR / f"{partition_key}_temp"
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始路径
        original_minute_dir_value = MINUTE_DIR
        
        try:
            # 临时修改存储路径
            MINUTE_DIR = temp_dir.parent
            
            # 下载缺失股票数据
            result = self.download_minute_data(
                stock_list=missing_stocks,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                max_workers=max_workers,
                use_multithreading=use_multithreading,
                use_multiprocessing=use_multiprocessing
            )
            
            # 恢复原始路径
            MINUTE_DIR = original_minute_dir_value
            
            # 直接检查新数据文件是否已创建，不再依赖download_minute_data的返回值
            new_data_file = temp_dir / partition_key / "minute_data.parquet"
            if not new_data_file.exists():
                logger.warning(f"未找到新下载的数据文件: {new_data_file}")
                if temp_dir.exists():
                    shutil.rmtree(str(temp_dir))
                return False
            
            try:
                new_df = pd.read_parquet(new_data_file)
                logger.info(f"成功读取新下载的数据，共 {len(new_df)} 行")
            except Exception as e:
                logger.error(f"读取新下载的数据失败: {e}")
                return False
            
            # 合并数据
            if original_df is not None:
                # 保留原始数据的所有列和所有已有股票数据
                merged_df = original_df.copy()
                logger.info(f"原始数据包含 {len(merged_df)} 行")
                
                # 创建一个标识列来确定哪些股票是新的（不在原始数据中的）
                original_stocks = set(original_df['code'].unique())
                logger.info(f"原始数据包含 {len(original_stocks)} 只股票")
                
                # 只添加新的股票数据，不修改已有股票的数据
                new_stocks_mask = ~new_df['code'].isin(original_stocks)
                new_stocks_df = new_df[new_stocks_mask]
                
                # 确保新股票数据有原始数据的所有列
                for col in merged_df.columns:
                    if col not in new_stocks_df.columns:
                        new_stocks_df[col] = None
                
                # 只添加新股票数据，不进行全局去重
                merged_df = pd.concat([merged_df, new_stocks_df[merged_df.columns]], ignore_index=True)
                logger.info(f"合并后共 {len(merged_df)} 行，新增了 {len(new_stocks_df)} 行数据")
                
                # 释放内存
                del original_df
                del new_df
                gc.collect()
            else:
                merged_df = new_df
                logger.info(f"没有原始数据，使用新下载的数据 {len(merged_df)} 行")
                del new_df
                gc.collect()
            
            # 确保分区目录存在
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存合并后的数据并释放内存
            try:
                # 确保使用与原始数据相同的参数保存
                table = pa.Table.from_pandas(merged_df, preserve_index=False)
                
                # 先写入临时文件，确认成功后再替换原文件
                temp_partition_file = partition_dir / f"minute_data.parquet.temp"
                pq.write_table(table, str(temp_partition_file), compression='snappy')
                
                # 验证临时文件的完整性
                test_read = pq.read_table(temp_partition_file)
                if len(test_read) == len(merged_df):
                    # 临时文件完好，替换原文件
                    if temp_partition_file.exists():
                        if partition_file.exists():
                            partition_file.unlink()  # 删除原文件
                        temp_partition_file.rename(partition_file)  # 重命名临时文件
                    logger.info(f"成功保存合并后的数据到 {partition_file}")
                else:
                    raise Exception(f"临时文件验证失败: 预期 {len(merged_df)} 行，实际 {len(test_read)} 行")
                
                # 释放内存
                del merged_df
                del table
                gc.collect()
            except Exception as e:
                logger.error(f"保存合并数据失败: {e}")
                # 如果保存失败且有备份，恢复备份
                if backup_file.exists():
                    shutil.copy(str(backup_file), str(partition_file))
                    logger.info("已恢复原始数据")
                return False
            
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir))
                logger.info(f"已清理临时目录 {temp_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"补充下载过程中发生错误: {e}")
            # 恢复原始路径
            MINUTE_DIR = original_minute_dir_value
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(str(temp_dir))
            
            # 手动触发垃圾回收
            gc.collect()
            return False

def download_all_data(storage, start_date='20230101', end_date=None, batch_size=None, workers=None):
    """下载所有数据
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'，默认为当前日期
        batch_size: 已废弃参数，保留用于兼容性
        workers: 已废弃参数，保留用于兼容性
    """
    logger.info("开始下载所有数据...")
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # 下载股票列表
    logger.info("1. 下载股票列表")
    # download_stock_list(storage)
    
    # 下载交易日历
    logger.info("2. 下载交易日历")
    download_calendar(storage, start_date, end_date)
    
    # 下载日线数据
    logger.info("3. 下载日线数据")
    download_daily_data(storage, start_date, end_date)
    
    # # 下载5分钟数据
    # logger.info("4. 下载5分钟数据")
    # download_minute_data(storage, start_date, end_date)
    
    logger.info("所有数据下载完成")
    return True

def download_stock_list(storage):
    """仅下载股票列表
    
    参数:
        storage: ParquetDataStorage实例
    """
    logger.info("下载股票列表...")
    stocks = storage.download_stock_list()
    logger.info(f"股票列表下载完成，共 {len(stocks)} 只股票")
    return stocks

def download_calendar(storage, start_date='20230101', end_date=None):
    """仅下载交易日历
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'，默认为当前日期
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    logger.info(f"下载交易日历: {start_date} 至 {end_date}...")
    print(f"下载交易日历: {start_date} 至 {end_date}...")
    success = storage.download_trading_calendar(start_date, end_date)
    if success:
        logger.info("交易日历下载完成")
        return True
    else:
        logger.error("交易日历下载失败")
        return False

def download_daily_data(storage, start_date='20230101', end_date=None, batch_size=100, workers=5):
    """仅下载日线数据
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'，默认为当前日期
        batch_size: 批处理大小
        workers: 线程数
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    # 获取股票列表
    stocks = storage.metadata['stock_list']
    if not stocks:
        logger.info("股票列表为空，先下载股票列表...")
        stocks = storage.download_stock_list()
    
    # 获取交易日历以验证日期范围
    calendar_df = storage.get_trading_calendar()
    if calendar_df.empty:
        logger.error("交易日历为空，请先下载交易日历")
        return False
        
    # 筛选指定日期范围内的交易日
    valid_dates = calendar_df[
        (calendar_df['date_str'] >= start_date) & 
        (calendar_df['date_str'] <= end_date)
    ]['date_str'].tolist()
    
    if not valid_dates:
        logger.error(f"指定日期范围 {start_date} 至 {end_date} 内没有有效的交易日")
        return False
        
    logger.info(f"下载日线数据: {start_date} 至 {end_date}，共 {len(stocks)} 只股票...")
    logger.info(f"该日期范围内有 {len(valid_dates)} 个交易日")
    
    # 确保API连接
    if not xtdata.get_client().is_connected():
        xtdata.get_client().connect()
        time.sleep(0.5)
    
    success = storage.download_daily_data(stocks, start_date, end_date, batch_size, workers)
    if success:
        logger.info("日线数据下载完成")
        return True
    else:
        logger.error("日线数据下载失败")
        return False

def download_minute_data(storage, start_date='20230101', end_date=None, batch_size=50, workers=5, use_multithreading=False, use_multiprocessing=True):
    """下载5分钟数据
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        batch_size: 每批处理的股票数
        workers: 并行下载的线程/进程数
        use_multithreading: 是否使用多线程
        use_multiprocessing: 是否使用多进程
    
    返回:
        bool: 是否下载成功
    """
    # 如果未指定结束日期，则使用当前日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
        
    
    try:
        logger.info(f"开始下载5分钟数据，时间范围: {start_date} - {end_date}")
        storage.download_minute_data(
            stock_list=None,  # 下载所有股票
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            max_workers=workers,
            use_multithreading=use_multithreading
        )
        logger.info("5分钟数据下载完成")
        return True
    except Exception as e:
        logger.error(f"下载5分钟数据时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_all_data(storage):
    """增量更新所有数据
    
    参数:
        storage: ParquetDataStorage实例
    """
    logger.info("增量更新所有数据...")
    
    # 获取股票列表
    stocks = storage.metadata['stock_list']
    if not stocks:
        logger.info("股票列表为空，先下载股票列表...")
        stocks = storage.download_stock_list()
    
    # 更新交易日历
    logger.info("更新交易日历...")
    storage.update_trading_calendar()
    
    # 更新日线数据
    logger.info("更新日线数据...")
    storage.update_daily_data(stocks)
    
    # 更新5分钟数据
    logger.info("更新5分钟数据...")
    storage.update_minute_data(stocks)
    
    logger.info("所有数据更新完成")

def run_all_tests(storage, test_date='20230201', end_test_date=None, test_stock=None):
    """运行所有测试函数
    
    参数:
        storage: ParquetDataStorage实例
        test_date: 测试日期，格式 'YYYYMMDD'
        end_test_date: 结束测试日期，格式 'YYYYMMDD'，默认与test_date相同
        test_stock: 测试股票代码
    """
    logger.info("运行所有测试函数...")
    
    if end_test_date is None:
        end_test_date = test_date
    
    test_stocks = [test_stock] if test_stock else None
    
    # 测试交易日历数据
    storage.test_trading_calendar()
    
    # 测试日线数据
    storage.test_daily_data(test_date, end_test_date, test_stocks)
    
    # 测试5分钟数据
    storage.test_minute_data(test_date, end_test_date, test_stocks)
    
    # 测试数据查询功能
    storage.test_data_query(test_date, test_stock)
    
    # 测试市场数据接口
    storage.test_market_data_interface(test_date, end_test_date, test_stocks)
    
    logger.info("所有测试完成")

def test_calendar(storage):
    """测试交易日历数据
    
    参数:
        storage: ParquetDataStorage实例
    """
    logger.info("测试交易日历数据...")
    storage.test_trading_calendar()

def test_daily_data(storage, start_date='20230201', end_date=None, test_stock=None):
    """测试日线数据
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 测试开始日期，格式 'YYYYMMDD'
        end_date: 测试结束日期，格式 'YYYYMMDD'，默认与start_date相同
        test_stock: 测试股票代码
    """
    if end_date is None:
        end_date = start_date
        
    test_stocks = [test_stock] if test_stock else None
    
    logger.info(f"测试日线数据: {start_date} 至 {end_date}...")
    storage.test_daily_data(start_date, end_date, test_stocks)

def test_minute_data(storage, start_date='20230201', end_date=None, test_stock=None):
    """测试5分钟数据
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 测试开始日期，格式 'YYYYMMDD'
        end_date: 测试结束日期，格式 'YYYYMMDD'，默认与start_date相同
        test_stock: 测试股票代码
    """
    if end_date is None:
        end_date = start_date
        
    test_stocks = [test_stock] if test_stock else None
    
    logger.info(f"测试5分钟数据: {start_date} 至 {end_date}...")
    storage.test_minute_data(start_date, end_date, test_stocks)

def test_query(storage, test_date='20230201', test_stock=None):
    """测试数据查询功能
    
    参数:
        storage: ParquetDataStorage实例
        test_date: 测试日期，格式 'YYYYMMDD'
        test_stock: 测试股票代码
    """
    logger.info(f"测试数据查询功能: 日期={test_date}...")
    storage.test_data_query(test_date, test_stock)

def test_market_interface(storage, start_date='20230201', end_date=None, test_stock=None):
    """测试市场数据接口
    
    参数:
        storage: ParquetDataStorage实例
        start_date: 测试开始日期，格式 'YYYYMMDD'
        end_date: 测试结束日期，格式 'YYYYMMDD'，默认与start_date相同
        test_stock: 测试股票代码
    """
    if end_date is None:
        end_date = start_date
        
    test_stocks = [test_stock] if test_stock else None
    
    logger.info(f"测试市场数据接口: {start_date} 至 {end_date}...")
    storage.test_market_data_interface(start_date, end_date, test_stocks)

def test_bulk_data_performance(storage, stock_count=100, days_range=30, test_type='both'):
    """测试批量获取多个股票多个日期数据的性能
    
    参数:
        storage: ParquetDataStorage实例
        stock_count: 测试用的股票数量，默认为100
        days_range: 测试的日期范围(天数)，默认为30天
        test_type: 测试类型，可选值为'daily', 'minute', 'both'
    
    返回:
        dict: 测试结果，包含耗时、数据量等信息
    """
    logger.info(f"开始批量数据获取性能测试: {stock_count}只股票, {days_range}天数据...")
    
    results = {}
    
    # 测试日线数据
    if test_type in ['daily', 'both']:
        logger.info("执行日线数据性能测试")
        daily_results = test_bulk_daily_data_performance(storage, stock_count, days_range)
        results['daily'] = daily_results
    
    # 测试5分钟数据
    if test_type in ['minute', 'both']:
        logger.info("执行5分钟数据性能测试")
        minute_results = test_bulk_minute_data_performance(storage, stock_count, days_range)
        results['minute'] = minute_results
    
    logger.info("批量数据获取性能测试完成")
    return results

def test_bulk_data_efficiency(storage, stock_counts=[50, 100, 200, 500], day_ranges=[10, 30, 60]):
    """测试不同规模的数据获取效率
    
    参数:
        storage: ParquetDataStorage实例
        stock_counts: 测试的股票数量列表
        day_ranges: 测试的日期范围列表
    """
    logger.info("开始测试不同规模的数据获取效率...")
    
    all_results = {}
    
    for stock_count in stock_counts:
        for days_range in day_ranges:
            test_key = f"stocks={stock_count}_days={days_range}"
            logger.info(f"测试配置: {test_key}")
            
            try:
                results = test_bulk_data_performance(storage, stock_count, days_range)
                all_results[test_key] = results
                
                # 输出摘要
                daily_stats = results.get('daily', {})
                minute_stats = results.get('minute', {})
                
                logger.info(f"配置 {test_key} 摘要:")
                if daily_stats.get('time'):
                    logger.info(f"  日线: {daily_stats['rows']:,}行, "
                              f"{daily_stats['time']:.2f}秒, "
                              f"{daily_stats['rows_per_sec']:,.2f}行/秒")
                
                if minute_stats.get('time'):
                    logger.info(f"  分钟: {minute_stats['rows']:,}行, "
                              f"{minute_stats['time']:.2f}秒, "
                              f"{minute_stats['rows_per_sec']:,.2f}行/秒")
                
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"测试配置 {test_key} 时出错: {e}")
                logger.error(traceback.format_exc())
    
    # 输出比较结果
    logger.info("\n数据获取效率测试总结:")
    
    if all_results:
        # 找出日线数据效率最高的配置
        best_daily_config = max(all_results.items(), 
                             key=lambda x: x[1].get('daily', {}).get('rows_per_sec', 0) 
                                   if x[1].get('daily') else 0)
        
        # 找出分钟数据效率最高的配置
        best_minute_config = max(all_results.items(), 
                              key=lambda x: x[1].get('minute', {}).get('rows_per_sec', 0)
                                    if x[1].get('minute') else 0)
        
        logger.info(f"最高效率配置:")
        
        daily_stats = best_daily_config[1].get('daily', {})
        if daily_stats.get('time'):
            logger.info(f"  日线: {best_daily_config[0]}, "
                      f"{daily_stats['rows_per_sec']:,.2f}行/秒, "
                      f"{daily_stats['mb_per_sec']:.2f}MB/秒")
        
        minute_stats = best_minute_config[1].get('minute', {})
        if minute_stats.get('time'):
            logger.info(f"  分钟: {best_minute_config[0]}, "
                      f"{minute_stats['rows_per_sec']:,.2f}行/秒, "
                      f"{minute_stats['mb_per_sec']:.2f}MB/秒")
    
    return all_results

def test_bulk_daily_data_performance(storage, stock_count=100, days_range=30):
    """测试批量获取多个股票多个日期的日线数据性能
    
    参数:
        storage: ParquetDataStorage实例
        stock_count: 测试用的股票数量，默认为100
        days_range: 测试的日期范围(天数)，默认为30天
    
    返回:
        dict: 测试结果，包含耗时、数据量等信息
    """
    logger.info(f"开始批量日线数据获取性能测试: {stock_count}只股票, {days_range}天数据...")
    
    # 获取股票列表
    all_stocks = storage.metadata['stock_list']
    if not all_stocks or len(all_stocks) < stock_count:
        logger.warning(f"股票列表不足{stock_count}只，尝试下载股票列表")
        all_stocks = storage.download_stock_list()
    
    # 选择测试用的股票
    test_stocks = all_stocks[:stock_count] if len(all_stocks) >= stock_count else all_stocks
    logger.info(f"选取了{len(test_stocks)}只股票进行测试")
    
    # 获取交易日历
    calendar_df = storage.get_trading_calendar()
    if calendar_df.empty:
        logger.error("交易日历为空，无法进行测试")
        return {}
    
    # 选择最近的交易日期范围
    recent_dates = sorted(calendar_df['date_str'].tolist(), reverse=True)
    if len(recent_dates) < days_range:
        logger.warning(f"交易日历中的日期数量({len(recent_dates)})少于请求的天数({days_range})")
        test_dates = recent_dates
    else:
        test_dates = recent_dates[:days_range]
    
    # 设置日期范围
    end_date = test_dates[0]
    start_date = test_dates[-1]
    logger.info(f"测试日期范围: {start_date} 至 {end_date}, 共{len(test_dates)}个交易日")
    
    # 检查分区可用性
    start_partition = storage._get_partition_key(start_date)
    end_partition = storage._get_partition_key(end_date)
    
    # 测试结果字典
    results = {
        'time': None, 
        'rows': 0, 
        'size_mb': 0, 
        'rows_per_sec': 0, 
        'mb_per_sec': 0
    }
    
    # 测试字段
    fields = ['open', 'high', 'low', 'close', 'volume']
    
    # 测试日线数据
    daily_partitions = storage.metadata['partitions']['daily']
    daily_relevant = [p for p in daily_partitions if p >= start_partition and p <= end_partition]
    
    if not daily_relevant:
        logger.warning(f"未找到日线数据分区: {start_partition} - {end_partition}")
        return results
    
    logger.info(f"开始测试日线数据获取性能...")
    start_time = time.time()
    
    daily_df = storage.get_daily_data(test_stocks, start_date, end_date, fields)
    
    elapsed = time.time() - start_time
    
    if daily_df.empty:
        logger.warning("获取的日线数据为空")
        return results
    
    # 计算数据大小（估算）
    size_mb = daily_df.memory_usage(deep=True).sum() / (1024 * 1024)
    rows = len(daily_df)
    rows_per_sec = rows / elapsed if elapsed > 0 else 0
    mb_per_sec = size_mb / elapsed if elapsed > 0 else 0
    
    logger.info(f"日线数据获取性能:")
    logger.info(f"  耗时: {elapsed:.2f}秒")
    logger.info(f"  数据行数: {rows:,}")
    logger.info(f"  数据大小: {size_mb:.2f} MB")
    logger.info(f"  每秒行数: {rows_per_sec:,.2f}")
    logger.info(f"  每秒数据量: {mb_per_sec:.2f} MB/s")
    
    # 打印数据样例
    logger.info(f"日线数据样例 (前5行):")
    if not daily_df.empty:
        print(daily_df.head())
        
        # 打印股票统计信息
        stock_counts = daily_df.groupby('stock_code').size()
        logger.info(f"每只股票的数据点数量 (前5只):")
        print(stock_counts.head())
        
        # 打印日期范围
        date_range = daily_df['date_str'].nunique()
        logger.info(f"获取到的日期总数: {date_range}")
        logger.info(f"首日: {daily_df['date_str'].min()}, 末日: {daily_df['date_str'].max()}")
    
    # 测试单一字段查询性能
    logger.info(f"测试列式存储下的字段筛选性能...")
    
    # 测试仅读取close字段
    start_time = time.time()
    daily_close_df = storage.get_daily_data(test_stocks, start_date, end_date, ['close'])
    elapsed_daily_single = time.time() - start_time
    
    logger.info(f"单字段(close)查询性能:")
    logger.info(f"  耗时: {elapsed_daily_single:.2f}秒")
    
    if elapsed > 0 and elapsed_daily_single > 0:
        daily_ratio = elapsed / elapsed_daily_single
        logger.info(f"  与多字段查询相比，效率提升: {daily_ratio:.2f}倍")
    
    results = {
        'time': elapsed,
        'rows': rows,
        'size_mb': size_mb,
        'rows_per_sec': rows_per_sec,
        'mb_per_sec': mb_per_sec,
        'single_field_time': elapsed_daily_single,
        'speedup_ratio': daily_ratio if 'daily_ratio' in locals() else None
    }
    
    logger.info("日线数据批量获取性能测试完成")
    return results

def test_bulk_minute_data_performance(storage, stock_count=50, days_range=10):
    """测试批量获取多个股票多个日期的5分钟数据性能
    
    参数:
        storage: ParquetDataStorage实例
        stock_count: 测试用的股票数量，默认为50
        days_range: 测试的日期范围(天数)，默认为10天
    
    返回:
        dict: 测试结果，包含耗时、数据量等信息
    """
    logger.info(f"开始批量5分钟数据获取性能测试: {stock_count}只股票, {days_range}天数据...")
    
    # 获取股票列表
    all_stocks = storage.metadata['stock_list']
    if not all_stocks or len(all_stocks) < stock_count:
        logger.warning(f"股票列表不足{stock_count}只，尝试下载股票列表")
        all_stocks = storage.download_stock_list()
    
    # 选择测试用的股票
    test_stocks = all_stocks[:stock_count] if len(all_stocks) >= stock_count else all_stocks
    logger.info(f"选取了{len(test_stocks)}只股票进行测试")
    
    # 获取交易日历
    calendar_df = storage.get_trading_calendar()
    if calendar_df.empty:
        logger.error("交易日历为空，无法进行测试")
        return {}
    
    # 选择最近的交易日期范围
    recent_dates = sorted(calendar_df['date_str'].tolist(), reverse=True)
    if len(recent_dates) < days_range:
        logger.warning(f"交易日历中的日期数量({len(recent_dates)})少于请求的天数({days_range})")
        test_dates = recent_dates
    else:
        test_dates = recent_dates[:days_range]
    
    # 设置日期范围
    end_date = test_dates[0]
    start_date = test_dates[-1]
    logger.info(f"测试日期范围: {start_date} 至 {end_date}, 共{len(test_dates)}个交易日")
    
    # 检查分区可用性
    start_partition = storage._get_partition_key(start_date)
    end_partition = storage._get_partition_key(end_date)
    
    # 测试结果字典
    results = {
        'time': None, 
        'rows': 0, 
        'size_mb': 0, 
        'rows_per_sec': 0, 
        'mb_per_sec': 0
    }
    
    # 测试字段
    fields = ['open', 'high', 'low', 'close', 'volume']
    
    # 测试5分钟数据
    minute_partitions = storage.metadata['partitions']['minute']
    minute_relevant = [p for p in minute_partitions if p >= start_partition and p <= end_partition]
    
    if not minute_relevant:
        logger.warning(f"未找到5分钟数据分区: {start_partition} - {end_partition}")
        return results
    
    logger.info(f"开始测试5分钟数据获取性能...")
    start_time = time.time()
    
    minute_df = storage.get_minute_data(test_stocks, start_date, end_date, fields)
    
    elapsed = time.time() - start_time
    
    if minute_df.empty:
        logger.warning("获取的5分钟数据为空")
        return results
    
    # 计算数据大小（估算）
    size_mb = minute_df.memory_usage(deep=True).sum() / (1024 * 1024)
    rows = len(minute_df)
    rows_per_sec = rows / elapsed if elapsed > 0 else 0
    mb_per_sec = size_mb / elapsed if elapsed > 0 else 0
    
    logger.info(f"5分钟数据获取性能:")
    logger.info(f"  耗时: {elapsed:.2f}秒")
    logger.info(f"  数据行数: {rows:,}")
    logger.info(f"  数据大小: {size_mb:.2f} MB")
    logger.info(f"  每秒行数: {rows_per_sec:,.2f}")
    logger.info(f"  每秒数据量: {mb_per_sec:.2f} MB/s")
    
    # 打印数据样例
    logger.info(f"5分钟数据样例 (前5行):")
    if not minute_df.empty:
        print(minute_df.head())
        
        # 统计每只股票的记录数量
        stock_counts = minute_df.groupby('stock_code').size()
        logger.info(f"每只股票的数据点数量 (前5只):")
        print(stock_counts.head())
        
        # 统计时间段分布
        if 'time_str' in minute_df.columns:
            time_counts = minute_df.groupby('time_str').size().sort_values(ascending=False)
            logger.info(f"交易时段分布 (前10个):")
            print(time_counts.head(10))
        
        # 打印日期范围
        date_range = minute_df['date_str'].nunique()
        unique_times = minute_df['time_str'].nunique() if 'time_str' in minute_df.columns else 0
        logger.info(f"获取到的日期总数: {date_range}, 交易时段数: {unique_times}")
        logger.info(f"首日: {minute_df['date_str'].min()}, 末日: {minute_df['date_str'].max()}")
    
    # 测试单一字段查询性能
    logger.info(f"测试列式存储下的字段筛选性能...")
    
    # 测试仅读取close字段
    start_time = time.time()
    minute_close_df = storage.get_minute_data(test_stocks, start_date, end_date, ['close'])
    elapsed_minute_single = time.time() - start_time
    
    logger.info(f"单字段(close)查询性能:")
    logger.info(f"  耗时: {elapsed_minute_single:.2f}秒")
    
    if elapsed > 0 and elapsed_minute_single > 0:
        minute_ratio = elapsed / elapsed_minute_single
        logger.info(f"  与多字段查询相比，效率提升: {minute_ratio:.2f}倍")
    
    results = {
        'time': elapsed,
        'rows': rows,
        'size_mb': size_mb,
        'rows_per_sec': rows_per_sec,
        'mb_per_sec': mb_per_sec,
        'single_field_time': elapsed_minute_single,
        'speedup_ratio': minute_ratio if 'minute_ratio' in locals() else None
    }
    
    logger.info("5分钟数据批量获取性能测试完成")
    return results

def main():
    """主函数，通过注释/取消注释下面的功能来选择要执行的操作"""
    # 确保API连接
    try:
        if not xtdata.get_client().is_connected():
            logger.info("尝试连接到xtdata API...")
            xtdata.get_client().connect()
            time.sleep(1)  # 等待连接稳定
            
        if not xtdata.get_client().is_connected():
            logger.error("无法连接到xtdata API，请检查API配置")
            return
        else:
            logger.info("成功连接到xtdata API")
    except Exception as e:
        logger.error(f"连接xtdata API时出错: {e}")
        logger.error(traceback.format_exc())
        return
    
    # 创建数据存储对象
    storage = ParquetDataStorage()
    
    # 设置时间区间参数
    start_date = '20240701'  # 开始日期，使用YYYYMMDD格式字符串
    # end_date = datetime.now().strftime('%Y%m%d')  # 结束日期，默认为当前日期
    end_date = '20251022'
    batch_size = 500  # 批处理大小
    workers = 5  # 线程数
    
    # 设置测试参数
    test_date = '20230301'  # 测试日期
    end_test_date = '20230405'  # 测试结束日期
    test_stock = '000001.SZ'  # 测试股票代码，设为None则使用默认股票
    
    # =================== 请选择要执行的功能，取消注释即可 ===================
    
    # === 1. 数据下载功能 ===
    # 下载股票列表 (如果未下载，这个通常应该首先执行)
    # download_stock_list(storage)
    # print(f"下载交易日历: {test_date} 至 {end_test_date}")
    # 下载交易日历
    download_calendar(storage, test_date, end_test_date)

    # 更新交易日历
    # storage.update_trading_calendar()

    
    # 下载日线数据
    download_daily_data(storage, start_date, end_date, batch_size, workers)
    
    # 下载5分钟数据
    # download_minute_data(storage, start_date, end_date, batch_size//2, workers)
    
    # 下载所有数据（包括股票列表、交易日历、日线和5分钟数据）
    # download_all_data(storage, start_date, end_date, batch_size, workers)
    
    # === 2. 数据更新功能 ===
    # 更新股票列表
    # storage.update_stock_list()
    

    
    # 更新日线数据
    # storage.update_daily_data()
    
    # 更新5分钟数据
    # storage.update_minute_data()
    
    # 更新所有数据
    # update_all_data(storage)
    
    # === 3. 数据获取和测试功能 ===
    # 测试交易日历
    # test_calendar(storage)
    
    # 测试日线数据
    # test_daily_data(storage, test_date, end_test_date, test_stock)
    
    # 测试5分钟数据
    # test_minute_data(storage, test_date, end_test_date, test_stock)

    # 使用多进程下载缺失数据
    # storage.download_missing_stocks_data(
    #     partition_key='202302',
    #     max_workers=8,
    #     use_multiprocessing=True
    # )
    
    # 检查2023年1月哪些股票没有5分钟数据
    # storage.check_missing_stocks_in_minute_data('202301')
    
    # 测试数据查询
    # test_query(storage, test_date, test_stock)
    
    # 测试市场数据接口
    # test_market_interface(storage, test_date, end_test_date, test_stock)
    
    # 测试批量数据获取性能
    # test_bulk_data_performance(storage, stock_count=100, days_range=30)
    
    # 只测试日线数据批量获取性能
    # test_bulk_daily_data_performance(storage, stock_count=100, days_range=30)
    
    # 只测试5分钟数据批量获取性能
    # test_bulk_minute_data_performance(storage, stock_count=50, days_range=10)
    
    # 测试不同规模的数据获取效率
    # test_bulk_data_efficiency(storage, stock_counts=[50, 200], day_ranges=[10, 30])
    
    # 运行所有测试
    # run_all_tests(storage, test_date, end_test_date, test_stock)
    
    # === 4. 数据诊断功能 ===
    # 检查不同分区的数据可用性
    # logger.info("运行数据诊断，检查必要的目录和文件是否存在...")
    
    # # 检查是否已有股票列表
    # if not storage.metadata['stock_list']:
    #     logger.info("股票列表为空，需要先下载股票列表")
    #     # 取消下面的注释以自动下载股票列表
    #     # download_stock_list(storage)
    # else:
    #     logger.info(f"已有股票列表，共 {len(storage.metadata['stock_list'])} 只股票")
    
    # # 检查交易日历
    # calendar_file_exists = (CALENDAR_DIR / "trading_calendar.parquet").exists()
    # if not calendar_file_exists:
    #     logger.warning("交易日历文件不存在，需要先下载交易日历")
    #     # 取消下面的注释以自动下载交易日历
    #     # download_calendar(storage, start_date, end_date)
    # else:
    #     # 验证交易日历数据
    #     calendar_df = storage.get_trading_calendar()
    #     if calendar_df.empty:
    #         logger.error("交易日历文件存在但数据为空，需要重新下载")
    #         # 取消下面的注释以自动重新下载交易日历
    #         # download_calendar(storage, start_date, end_date)
    #     else:
    #         logger.info(f"交易日历数据范围: {calendar_df['date_str'].min()} 至 {calendar_df['date_str'].max()}")
    
    # # 检查当前数据分区
    # daily_partitions = storage.metadata['partitions']['daily']
    # minute_partitions = storage.metadata['partitions']['minute']
    # logger.info(f"当前日线数据分区: {daily_partitions}")
    # logger.info(f"当前分钟数据分区: {minute_partitions}")
    
    # # 检查测试日期所在分区是否已下载
    # test_partition = storage._get_partition_key(test_date)
    # if test_partition not in storage.metadata['partitions']['daily']:
    #     logger.warning(f"测试日期 {test_date} 所在分区 {test_partition} 的日线数据尚未下载")
    #     # 取消下面的注释以自动下载测试所需的日线数据
    #     # download_daily_data(storage, test_date, end_test_date, batch_size, workers)
    # else:
    #     logger.info(f"测试日期 {test_date} 所在分区 {test_partition} 的日线数据已存在")
    
    # # 检查测试日期所在分区的5分钟数据
    # if test_partition not in storage.metadata['partitions']['minute']:
    #     logger.warning(f"测试日期 {test_date} 所在分区 {test_partition} 的5分钟数据尚未下载")
    #     # 取消下面的注释以自动下载测试所需的5分钟数据
    #     # download_minute_data(storage, test_date, end_test_date, batch_size//2, workers)
    # else:
    #     logger.info(f"测试日期 {test_date} 所在分区 {test_partition} 的5分钟数据已存在")
    
    logger.info("数据诊断完成，请根据上述信息选择适当的功能执行")
    
    # === 自动化数据准备示例 ===
    # 以下是一个自动化数据准备的示例，可以取消注释以执行
    """
    # 1. 确保有股票列表
    if not storage.metadata['stock_list']:
        logger.info("下载股票列表...")
        download_stock_list(storage)
    
    # 2. 确保有交易日历
    if not (CALENDAR_DIR / "trading_calendar.parquet").exists():
        logger.info("下载交易日历...")
        download_calendar(storage, start_date, end_date)
    
    # 3. 下载测试所需的日线数据
    test_partition = storage._get_partition_key(test_date)
    if test_partition not in storage.metadata['partitions']['daily']:
        logger.info("下载测试所需的日线数据...")
        download_daily_data(storage, test_date, end_test_date, batch_size, workers)
    
    # 4. 下载测试所需的5分钟数据
    if test_partition not in storage.metadata['partitions']['minute']:
        logger.info("下载测试所需的5分钟数据...")
        download_minute_data(storage, test_date, end_test_date, batch_size//2, workers)
    
    # 5. 运行测试
    logger.info("运行测试...")
    test_daily_data(storage, test_date, end_test_date, test_stock)
    test_minute_data(storage, test_date, end_test_date, test_stock)
    """
    
    logger.info("任务完成")

if __name__ == "__main__":
    main()