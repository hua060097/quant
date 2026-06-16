#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[大师级修复版] 股票日线特征生成脚本
核心修复：
1. 修正停牌日处理：如果 T 日停牌(Volume=0)，强制 is_can_buy=0 且 future_return=NaN。
   解决“停牌日显示可买且有大幅负收益”的Bug。
2. 保持向量化逻辑：继续使用 groupby+shift 处理 T+1 逻辑。

保留字段(共24个):
1. 基础数据-后复权: open, high, low, close, volume, amount, pct_chg, total_mv, circ_mv, turnover_rate
2. 基础数据-前复权: open_front, high_front, low_front, close_front
3. 涨跌停: is_limit_up, is_limit_down, is_zhaban, is_can_buy
4. ST标记: is_st
5. 未来收益: future_return_1d, future_return_2d, future_return_3d, future_return_4d, future_return_5d
6. 元数据: stock_code, date, date_str
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import traceback
import gc

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 动态导入ParquetDataStorage
try:
    from to.advanced_quant_data import ParquetDataStorage
except ImportError:
    try:
        from advanced_quant_data import ParquetDataStorage
    except ImportError:
        print("错误: 无法导入ParquetDataStorage,请确保advanced_quant_data.py存在")
        sys.exit(1)

# 配置日志
log_file = project_root / 'generate_daily_features_simple.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleLimitDataLoader:
    """简化版涨跌停数据加载器"""

    def __init__(self):
        # 请根据实际路径修改
        self.data_dir = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/limit_data")
        self.logger = logging.getLogger(__name__)
        self._all_data = None
        
        self._preload_all_data()

    def _preload_all_data(self):
        try:
            all_data_file = self.data_dir / "all_limit_data.parquet"
            if all_data_file.exists():
                self.logger.info(f"开始预加载涨跌停数据: {all_data_file}")
                self._all_data = pd.read_parquet(all_data_file)

                if 'trade_date' in self._all_data.columns:
                    self._all_data['date'] = pd.to_datetime(self._all_data['trade_date'], format='%Y%m%d', errors='coerce')

                if 'ts_code' in self._all_data.columns:
                    self._all_data['stock_code'] = self._all_data['ts_code']

                if 'limit' in self._all_data.columns:
                    self._all_data['is_limit_up'] = (self._all_data['limit'] == 'U').astype(int)
                    self._all_data['is_limit_down'] = (self._all_data['limit'] == 'D').astype(int)
                    self._all_data['is_zhaban'] = (self._all_data['limit'] == 'Z').astype(int)

                self.logger.info(f"涨跌停数据预加载完成: {len(self._all_data)} 条记录")
            else:
                self.logger.warning(f"涨跌停数据文件不存在: {all_data_file}")
                self._all_data = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"预加载涨跌停数据失败: {e}")
            self._all_data = pd.DataFrame()

    def get_limit_data(self, start_date, end_date):
        if self._all_data is None or self._all_data.empty:
            return pd.DataFrame()

        try:
            if 'trade_date' in self._all_data.columns:
                mask = (self._all_data['trade_date'] >= start_date) & (self._all_data['trade_date'] <= end_date)
                range_data = self._all_data[mask].copy()
                return range_data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"加载日期范围 {start_date}-{end_date} 的涨跌停数据失败: {e}")
            return pd.DataFrame()


class SimpleSTDataLoader:
    """ST 股票数据加载器 (来源: fetch_st_stocks.py 生成的 st.parquet)"""

    def __init__(self):
        self.data_file = Path("C:/Users/86151/.trae-cn/pythons/path/to/local_data/st.parquet")
        self.logger = logging.getLogger(__name__)
        self._all_data = None

        self._preload_all_data()

    def _preload_all_data(self):
        try:
            if self.data_file.exists():
                self.logger.info(f"开始预加载ST数据: {self.data_file}")
                df = pd.read_parquet(self.data_file)

                # 字段标准化
                if 'ts_code' in df.columns:
                    df['stock_code'] = df['ts_code']

                if 'trade_date' in df.columns:
                    # 保持 trade_date 为 YYYYMMDD 字符串, 同时构造 date 列 (datetime)
                    df['trade_date'] = df['trade_date'].astype(str)
                    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')

                # 仅保留需要的列
                keep_cols = [c for c in ['date', 'trade_date', 'stock_code'] if c in df.columns]
                self._all_data = df[keep_cols].drop_duplicates(subset=['date', 'stock_code']).copy()
                self._all_data['is_st'] = 1

                self.logger.info(f"ST数据预加载完成: {len(self._all_data)} 条记录")
            else:
                self.logger.warning(f"ST数据文件不存在: {self.data_file}")
                self._all_data = pd.DataFrame()
        except Exception as e:
            self.logger.error(f"预加载ST数据失败: {e}")
            self._all_data = pd.DataFrame()

    def get_st_data(self, start_date, end_date):
        """返回区间内的 ST 标记 DataFrame, 列: date, stock_code, is_st"""
        if self._all_data is None or self._all_data.empty:
            return pd.DataFrame()

        try:
            if 'trade_date' in self._all_data.columns:
                mask = (self._all_data['trade_date'] >= start_date) & (self._all_data['trade_date'] <= end_date)
                return self._all_data.loc[mask, ['date', 'stock_code', 'is_st']].copy()
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"加载日期范围 {start_date}-{end_date} 的ST数据失败: {e}")
            return pd.DataFrame()


class SimpleFeatureGenerator:
    """简化版特征生成器 - 修复停牌逻辑版"""
    
    def __init__(self, output_dir=None, max_workers=None):
        if output_dir is None:
            self.output_dir = project_root / "local_data" / "features" / "daily"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage = ParquetDataStorage()
        self.limit_loader = SimpleLimitDataLoader()
        self.st_loader = SimpleSTDataLoader()
        logger.info(f"特征生成器初始化完成, 输出目录: {self.output_dir}")
    
    def get_trading_days(self, start_date, end_date):
        try:
            calendar_df = self.storage.get_trading_calendar(start_date, end_date)
            if calendar_df is None or calendar_df.empty:
                return []
            
            if 'date_str' in calendar_df.columns:
                return calendar_df['date_str'].tolist()
            elif 'date' in calendar_df.columns:
                if pd.api.types.is_datetime64_any_dtype(calendar_df['date']):
                    return calendar_df['date'].dt.strftime('%Y%m%d').tolist()
                return calendar_df['date'].tolist()
            return []
        except Exception as e:
            logger.error(f"获取交易日列表失败: {e}")
            return []

    def get_stock_list(self):
        try:
            return self.storage.get_stock_list() or []
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
            
    def calculate_technical_indicators(self, df):
        """保留接口，只做排序"""
        if df is None or df.empty:
            return df
        return df.sort_values(['stock_code', 'date']).reset_index(drop=True)

    def calculate_future_returns_vectorized(self, df):
        """
        [核心修复] 全向量化计算
        """
        if df is None or df.empty:
            return df
            
        logger.info("开始向量化计算未来收益...")
        
        # 1. 强制排序
        df = df.sort_values(['stock_code', 'date']).reset_index(drop=True)
        grouped = df.groupby('stock_code')
        
        # 2. 识别【当前】(T日) 是否停牌
        # 迅投数据中，停牌日 Volume=0。如果是上市前数据则根本不存在行，不用处理。
        # 使用 1e-4 避免浮点误差
        if 'volume' in df.columns:
            is_suspended_today = (df['volume'] < 1e-4) | df['volume'].isna()
        else:
            is_suspended_today = pd.Series(False, index=df.index)

        # 3. 获取 T+1 数据
        next_open = grouped['open'].shift(-1)
        next_volume = grouped['volume'].shift(-1)
        
        # 4. 判断 is_can_buy
        df['is_can_buy'] = 1
        
        # 4.1 如果【当前】停牌，不可买 (且未来收益应为NaN)
        df.loc[is_suspended_today, 'is_can_buy'] = 0
        
        # 4.2 如果【T+1】停牌，不可买
        is_suspended_next = (next_volume < 1e-4) | next_volume.isna()
        df.loc[is_suspended_next, 'is_can_buy'] = 0
        
        # 4.3 如果【T+1】数据缺失 (到了最后一天)，不可买
        df.loc[next_open.isna(), 'is_can_buy'] = 0
        
        # 4.4 检查 T+1 是否一字涨停
        if 'is_limit_up' in df.columns:
            next_limit_up = grouped['is_limit_up'].shift(-1).fillna(0)
            next_high = grouped['high'].shift(-1)
            
            # T+1 一字涨停逻辑: 涨停 且 (开盘价 == 最高价)
            # 这是一个强过滤
            is_one_bar_limit = (next_limit_up == 1) & (next_open == next_high)
            df.loc[is_one_bar_limit, 'is_can_buy'] = 0

        # 5. 计算未来 N 日收益率
        # 逻辑：(Close_T+N - Open_T+1) / Open_T+1
        safe_next_open = next_open.replace(0, np.nan)
        
        for n in [1, 2, 3, 4, 5]:
            future_close = grouped['close'].shift(-n)
            col_name = f'future_return_{n}d'
            df[col_name] = (future_close - safe_next_open) / safe_next_open
            
            # [关键修复]：如果 T 日是停牌日，强制设为 NaN
            # 因为停牌日的信号是失效的，计算出的收益率也是无意义的噪音（比如复牌补跌导致的 -35%）
            df.loc[is_suspended_today, col_name] = np.nan
            
        return df

    def generate_features(self, start_date, end_date, stock_list=None, force_update=False, batch_size=None, use_threading=None):
        # 1. 获取交易日
        trading_days = self.get_trading_days(start_date, end_date)
        if not trading_days:
            logger.error(f"无法获取交易日: {start_date}-{end_date}")
            return False
            
        dates_to_process = []
        for date in trading_days:
            output_file = self.output_dir / f"{date}.parquet"
            if force_update or not output_file.exists():
                dates_to_process.append(date)
                
        if not dates_to_process:
            logger.info("所有特征已存在，跳过")
            return True
            
        logger.info(f"需要处理 {len(dates_to_process)} 个交易日")
        
        # 2. 批量加载数据
        # 预留 30 天 Buffer 计算 T+5
        buffer_end_obj = datetime.strptime(dates_to_process[-1], '%Y%m%d') + timedelta(days=30)
        load_end_date = buffer_end_obj.strftime('%Y%m%d')
        load_start_date = dates_to_process[0]
        
        if stock_list is None:
            stock_list = self.get_stock_list()

        logger.info(f"批量加载数据: {load_start_date} 到 {load_end_date}, 股票数: {len(stock_list)}")
        
        df_all = self.storage.get_daily_data(stock_list, load_start_date, load_end_date)
        if df_all is None or df_all.empty:
            logger.error("未加载到基础数据")
            return False
            
        # 3. 加载并合并涨跌停数据
        limit_data = self.limit_loader.get_limit_data(load_start_date, load_end_date)
        if limit_data is not None and not limit_data.empty:
            if 'date' in limit_data.columns and not pd.api.types.is_datetime64_any_dtype(limit_data['date']):
                 limit_data['date'] = pd.to_datetime(limit_data['date'], format='%Y%m%d', errors='coerce')
            
            if 'limit' in limit_data.columns:
                limit_data['is_limit_up'] = (limit_data['limit'] == 'U').astype(int)
                limit_data['is_limit_down'] = (limit_data['limit'] == 'D').astype(int)
                limit_data['is_zhaban'] = (limit_data['limit'] == 'Z').astype(int)
            
            cols_to_use = ['date', 'stock_code', 'is_limit_up', 'is_limit_down', 'is_zhaban', 'first_time_formatted']
            cols_to_use = [c for c in cols_to_use if c in limit_data.columns]
            
            df_all = df_all.merge(limit_data[cols_to_use], on=['date', 'stock_code'], how='left')
            
        for col in ['is_limit_up', 'is_limit_down', 'is_zhaban']:
            if col not in df_all.columns:
                df_all[col] = 0
            else:
                df_all[col] = df_all[col].fillna(0).astype(int)

        # 3.5 加载并合并 ST 数据
        st_data = self.st_loader.get_st_data(load_start_date, load_end_date)
        if st_data is not None and not st_data.empty:
            if 'date' in st_data.columns and not pd.api.types.is_datetime64_any_dtype(st_data['date']):
                st_data['date'] = pd.to_datetime(st_data['date'], format='%Y%m%d', errors='coerce')
            df_all = df_all.merge(st_data[['date', 'stock_code', 'is_st']], on=['date', 'stock_code'], how='left')

        if 'is_st' not in df_all.columns:
            df_all['is_st'] = 0
        else:
            df_all['is_st'] = df_all['is_st'].fillna(0).astype(int)

        # 4. 执行向量化计算 (包含修复逻辑)
        df_all = self.calculate_future_returns_vectorized(df_all)
        
        # 5. 切分保存
        logger.info("开始保存特征文件...")
        success_count = 0
        
        if pd.api.types.is_datetime64_any_dtype(df_all['date']):
            df_all['date_str'] = df_all['date'].dt.strftime('%Y%m%d')
        else:
            df_all['date_str'] = df_all['date'].astype(str)
            
        target_dates = set(dates_to_process)
        
        cols_to_save = [
            'stock_code', 'date', 'date_str',
            'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg',
            'total_mv', 'circ_mv', 'turnover_rate',
            'open_front', 'high_front', 'low_front', 'close_front',
            'is_limit_up', 'is_limit_down', 'is_zhaban', 'is_can_buy',
            'is_st',
            'future_return_1d', 'future_return_2d', 'future_return_3d', 'future_return_4d', 'future_return_5d'
        ]
        
        # 使用 groupby date_str 保存
        for date_str, group in df_all.groupby('date_str'):
            if date_str not in target_dates:
                continue
            try:
                valid_cols = [c for c in cols_to_save if c in group.columns]
                save_df = group[valid_cols].copy()
                output_path = self.output_dir / f"{date_str}.parquet"
                save_df.to_parquet(output_path, index=False)
                success_count += 1
            except Exception as e:
                logger.error(f"保存日期 {date_str} 失败: {e}")
        
        logger.info(f"处理完成: 成功 {success_count}/{len(dates_to_process)}")
        return success_count == len(dates_to_process)

def main():
    parser = argparse.ArgumentParser(description="[大师版] 生成股票日线特征")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    # 兼容参数
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--no_threading", action="store_true")
    
    args = parser.parse_args()

    generator = SimpleFeatureGenerator(args.output_dir)
    
    start_time = time.time()
    generator.generate_features(args.start_date, args.end_date, force_update=args.force)
    logger.info(f"总耗时: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()