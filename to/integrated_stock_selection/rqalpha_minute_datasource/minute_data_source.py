"""
RQAlpha 5分钟K线数据源模块

基于现有的5分钟parquet数据文件，实现RQAlpha兼容的数据源接口。
支持1m, 5m, 15m, 30m, 60m等多种时间周期的K线数据。

作者: Claude Code Assistant
日期: 2025-11-22
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from rqalpha.data.base_data_source import BaseDataSource

# 设置日志
logger = logging.getLogger(__name__)


class MinuteKDataSource:
    """
    5分钟K线数据源

    基于本地parquet格式的分钟级数据，提供类似RQAlpha的数据接口
    """

    def __init__(self, data_path, cache_size=100):
        """
        初始化数据源

        Args:
            data_path: 5分钟数据存储路径
            cache_size: 缓存股票数量
        """
        self.data_path = data_path
        self.cache_size = cache_size
        self.data_cache = {}  # 股票数据缓存
        self.stock_list = []  # 股票列表
        self.date_range = None  # 数据时间范围
        self._trading_calendar = None  # 交易日历缓存

        logger.info(f"初始化5分钟数据源，数据路径: {data_path}")
        self._load_stock_list()
        self._get_data_range()
        logger.info(f"数据源初始化完成，共{len(self.stock_list)}只股票，时间范围: {self.date_range[0]} ~ {self.date_range[1]}")

    def _load_stock_list(self):
        """加载股票列表"""
        try:
            # 从第一个数据文件中获取股票列表
            month_dirs = glob.glob(os.path.join(self.data_path, "*/"))
            if not month_dirs:
                raise FileNotFoundError(f"在路径 {self.data_path} 中未找到数据目录")

            sample_file = os.path.join(month_dirs[0], "minute_data.parquet")
            if not os.path.exists(sample_file):
                raise FileNotFoundError(f"未找到示例数据文件: {sample_file}")

            # 使用pyarrow读取parquet文件以避免numpy兼容性问题
            import pyarrow.parquet as pq
            table = pq.read_table(sample_file, columns=['stock_code'])
            df = table.to_pandas()

            self.stock_list = df['stock_code'].unique().tolist()
            logger.info(f"加载股票列表完成，共{len(self.stock_list)}只股票")

        except Exception as e:
            logger.error(f"加载股票列表失败: {e}")
            self.stock_list = []

    def _get_data_range(self):
        """获取数据时间范围"""
        try:
            month_dirs = glob.glob(os.path.join(self.data_path, "*/"))
            min_date = None
            max_date = None

            for month_dir in month_dirs:
                minute_file = os.path.join(month_dir, "minute_data.parquet")
                if os.path.exists(minute_file):
                    # 使用pyarrow读取
                    import pyarrow.parquet as pq
                    table = pq.read_table(minute_file, columns=['datetime'])
                    df = table.to_pandas()

                    if not df.empty:
                        current_min = df['datetime'].min()
                        current_max = df['datetime'].max()

                        if min_date is None or current_min < min_date:
                            min_date = current_min
                        if max_date is None or current_max > max_date:
                            max_date = current_max

            if min_date and max_date:
                self.date_range = (min_date.date(), max_date.date())
            else:
                self.date_range = (date(2024, 1, 1), date.today())

        except Exception as e:
            logger.error(f"获取数据时间范围失败: {e}")
            self.date_range = (date(2024, 1, 1), date.today())

    def _load_stock_data(self, stock_code, start_date=None, end_date=None):
        """
        加载指定股票的数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 股票数据
        """
        cache_key = f"{stock_code}_{start_date}_{end_date}" if start_date and end_date else stock_code

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            all_data = []

            # 按月份加载数据
            month_dirs = glob.glob(os.path.join(self.data_path, "*/"))

            for month_dir in sorted(month_dirs):
                minute_file = os.path.join(month_dir, "minute_data.parquet")
                if os.path.exists(minute_file):
                    # 使用pyarrow读取，只读取指定股票的数据
                    import pyarrow.parquet as pq

                    # 应用过滤器减少内存使用
                    filters = [('stock_code', '=', stock_code)]
                    table = pq.read_table(minute_file, filters=filters)
                    df = table.to_pandas()

                    if not df.empty:
                        all_data.append(df)

            if all_data:
                stock_data = pd.concat(all_data, ignore_index=True).sort_values('datetime')

                # 应用日期过滤
                if start_date and end_date:
                    mask = (stock_data['datetime'] >= start_date) & (stock_data['datetime'] <= end_date)
                    stock_data = stock_data[mask]

                # 缓存数据（控制缓存大小）
                if len(self.data_cache) < self.cache_size:
                    self.data_cache[cache_key] = stock_data
                elif self.data_cache:
                    # 删除最旧的缓存项
                    oldest_key = next(iter(self.data_cache))
                    del self.data_cache[oldest_key]
                    self.data_cache[cache_key] = stock_data

                return stock_data
            else:
                logger.warning(f"未找到股票 {stock_code} 的数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"加载股票 {stock_code} 数据失败: {e}")
            return pd.DataFrame()

    def _aggregate_by_frequency(self, df, frequency):
        """
        根据时间频率聚合数据

        Args:
            df: 原始数据
            frequency: 时间频率 ('1m', '5m', '15m', '30m', '60m')

        Returns:
            DataFrame: 聚合后的数据
        """
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '1H'
        }

        if frequency not in freq_map:
            raise ValueError(f"不支持的时间频率: {frequency}")

        # 复制数据避免修改原始数据
        df_work = df.copy()
        df_work.set_index('datetime', inplace=True)

        # 按频率聚合
        df_agg = df_work.resample(freq_map[frequency]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'  # amount字段（成交金额）
        }).dropna()

        df_agg.reset_index(inplace=True)
        return df_agg

    def _adjust_datetime_to_frequency(self, dt, frequency):
        """
        将时间调整到对应frequency的标准时间点

        Args:
            dt: 原始时间
            frequency: 时间频率

        Returns:
            datetime: 调整后的时间
        """
        freq_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '60m': 60
        }

        minutes = freq_minutes.get(frequency, 5)
        dt_rounded = dt.replace(
            minute=(dt.minute // minutes) * minutes,
            second=0,
            microsecond=0
        )
        return dt_rounded

    def get_trading_calendar(self):
        """获取交易日历"""
        if self._trading_calendar is None:
            try:
                all_dates = set()
                month_dirs = glob.glob(os.path.join(self.data_path, "*/"))

                for month_dir in month_dirs:
                    minute_file = os.path.join(month_dir, "minute_data.parquet")
                    if os.path.exists(minute_file):
                        # 使用pyarrow读取，只读取datetime列
                        import pyarrow.parquet as pq
                        table = pq.read_table(minute_file, columns=['datetime'])
                        df = table.to_pandas()
                        all_dates.update(df['datetime'].dt.date.unique())

                self._trading_calendar = sorted(all_dates)

            except Exception as e:
                logger.error(f"获取交易日历失败: {e}")
                # 生成默认交易日历（工作日）
                start_date = self.date_range[0]
                end_date = self.date_range[1]
                self._trading_calendar = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # 周一到周五
                        self._trading_calendar.append(current_date)
                    current_date += timedelta(days=1)

        return pd.DatetimeIndex(self._trading_calendar)

    def get_bar(self, instrument, dt, frequency):
        """
        获取单个K线数据

        Args:
            instrument: 交易标的对象
            dt: 时间点
            frequency: 数据频率

        Returns:
            dict: K线数据
        """
        try:
            # 只处理分钟级数据
            if frequency not in ['1m', '5m', '15m', '30m', '60m']:
                return super(MinuteKDataSource, self).get_bar(instrument, dt, frequency)

            stock_code = instrument.order_book_id

            # 加载股票数据
            start_date = dt.date() - timedelta(days=7)  # 多加载几天确保数据完整
            end_date = dt.date() + timedelta(days=1)

            df = self._load_stock_data(stock_code, start_date, end_date)
            if df.empty:
                logger.warning(f"未找到股票 {stock_code} 的数据")
                return None

            # 根据frequency聚合数据
            df_agg = self._aggregate_by_frequency(df, frequency)

            # 调整目标时间到标准时间点
            target_dt = self._adjust_datetime_to_frequency(dt, frequency)

            # 找到对应时间点的数据
            bar_data = df_agg[df_agg['datetime'] == target_dt]

            if bar_data.empty:
                # 如果没找到精确匹配，尝试找到最近的前一个时间点
                previous_data = df_agg[df_agg['datetime'] < target_dt]
                if not previous_data.empty:
                    bar_data = previous_data.iloc[[-1]]  # 取最近的一个
                else:
                    return None

            bar = bar_data.iloc[0]

            return {
                'datetime': bar['datetime'],
                'open': float(bar['open']),
                'high': float(bar['high']),
                'low': float(bar['low']),
                'close': float(bar['close']),
                'volume': int(bar['volume']),
                'total_turnover': float(bar['amount'])  # amount对应total_turnover（成交金额）
            }

        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return super(MinuteKDataSource, self).get_bar(instrument, dt, frequency)

    def history_bars(self, instrument, bar_count, frequency, fields, dt, skip_suspended=True):
        """
        获取历史K线数据

        Args:
            instrument: 交易标的对象
            bar_count: K线数量
            frequency: 数据频率
            fields: 字段列表
            dt: 截止时间
            skip_suspended: 是否跳过停牌

        Returns:
            np.array: 历史K线数据
        """
        try:
            # 只处理分钟级数据
            if frequency not in ['1m', '5m', '15m', '30m', '60m']:
                return super(MinuteKDataSource, self).history_bars(
                    instrument, bar_count, frequency, fields, dt, skip_suspended
                )

            stock_code = instrument.order_book_id

            # 计算需要加载的数据范围
            freq_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '60m': 60}
            minutes_per_bar = freq_minutes.get(frequency, 5)
            total_minutes = bar_count * minutes_per_bar * 2  # 多加载一些确保数据充足

            start_date = dt - timedelta(minutes=total_minutes)
            end_date = dt + timedelta(days=1)  # 多加载一天确保包含截止时间

            # 加载股票数据
            df = self._load_stock_data(stock_code, start_date.date(), end_date.date())
            if df.empty:
                logger.warning(f"未找到股票 {stock_code} 的历史数据")
                return None

            # 根据frequency聚合数据
            df_agg = self._aggregate_by_frequency(df, frequency)

            # 调整截止时间到标准时间点
            end_dt = self._adjust_datetime_to_frequency(dt, frequency)

            # 获取截止时间前的数据
            df_filtered = df_agg[df_agg['datetime'] <= end_dt].tail(bar_count)

            if df_filtered.empty or len(df_filtered) < bar_count:
                logger.warning(f"股票 {stock_code} 历史数据不足，需要{bar_count}根，实际{len(df_filtered)}根")
                return None

            # 处理字段
            if isinstance(fields, str):
                fields = [fields]

            # 字段映射 - amount映射为total_turnover
            field_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'total_turnover': 'amount'  # amount对应total_turnover（成交金额）
            }

            available_fields = []
            mapped_fields = []
            for field in fields:
                mapped_field = field_mapping.get(field, field)
                if mapped_field in df_filtered.columns:
                    available_fields.append(field)
                    mapped_fields.append(mapped_field)

            if not mapped_fields:
                logger.warning(f"没有可用的字段: {fields}")
                return None

            # 返回数据
            result_data = df_filtered[mapped_fields].values

            # 确保返回正确的形状
            if len(fields) == 1:
                return result_data.reshape(-1)
            else:
                return result_data

        except Exception as e:
            logger.error(f"获取历史K线数据失败: {e}")
            return super(MinuteKDataSource, self).history_bars(
                instrument, bar_count, frequency, fields, dt, skip_suspended
            )

    def current_snapshot(self, instrument, frequency, dt):
        """获取当前快照数据"""
        # 对于分钟级数据，快照就是当前的K线数据
        return self.get_bar(instrument, dt, frequency)

    def available_data_range(self, frequency):
        """
        返回数据可用时间范围

        Args:
            frequency: 数据频率

        Returns:
            tuple: (开始日期, 结束日期)
        """
        if frequency in ['1m', '5m', '15m', '30m', '60m']:
            return self.date_range
        else:
            # 对于其他频率，调用父类方法
            return super(MinuteKDataSource, self).available_data_range(frequency)

    def get_share_instruments(self):
        """获取股票标的列表"""
        # 这里需要根据RQAlpha的Instruments格式返回
        # 暂时返回空列表，需要根据实际需求实现
        return []

    def get_future_contracts(self, underlying_symbol):
        """获取期货合约列表"""
        return []

    def get_option_contracts(self, underlying_symbol):
        """获取期权合约列表"""
        return []