# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import xtquant.xtdata as xtdata
from tqdm import tqdm


class QMTBacktester:
    def __init__(self, start_date, end_date, capital=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = capital
        self.commission_rate = 0.0003
        self.slippage = 0.001
        
        self.positions = {}
        self.cash = capital
        self.equity = []
        
        self.trade_dates = self._get_trade_dates()
        self.current_date = None
        self.data_cache = {}

    def _get_trade_dates(self):
        dates = xtdata.get_trading_dates(
            'SH',
            self.start_date.strftime('%Y%m%d'),
            self.end_date.strftime('%Y%m%d')
        )
        return [datetime.fromtimestamp(timestamp / 1000) for timestamp in dates]

    def get_minute_data(self, code, date):
        date_str = date.strftime('%Y%m%d')
        cache_key = (code, date_str)
        
        if cache_key not in self.data_cache:
            try:
                # 日期有效性检查
                if date > datetime.now():
                    print(f"🔍 {code} {date_str} 未来日期跳过")
                    self.data_cache[cache_key] = None
                    return None
                    
                # 获取原始数据
                raw_data = xtdata.get_market_data(
                    field_list=['open', 'high', 'low', 'close', 'volume'],
                    stock_list=[code],
                    period='1d',
                    start_time=date_str,
                    end_time=date_str
                )

                #print(f"close: {raw_data.get('close').values[0][0]}")
                
                # 数据结构验证
                if not raw_data or not isinstance(raw_data, dict):
                    #print(f"🔍 {code} {date_str} 无效数据结构")
                    self.data_cache[cache_key] = None
                    return None
                    
                # 数据处理
                processed_data = {}
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    df = raw_data.get(field)
                    #print(f"🔍 {code} {date_str} {field} 数据: {df.values[0]}   类型: {type(df)}")
                    if isinstance(df, pd.DataFrame) and code in df.columns:
                        if not df.empty:
                            print(f"🔍$$$$$$$$$ {code} {date_str} {field} df[code].iloc[0]")
                            processed_data[field] = round(df[code].values[0][0], 2)
                
                # 数据完整性检查
                if not processed_data or 'close' not in processed_data:
                    #print(f"🔍 {code} {date_str} 数据不完整")
                    self.data_cache[cache_key] = None
                    return None
                    
                # 数值有效性检查
                close_value = processed_data['close']
                if pd.isna(close_value) or close_value <= 0:
                    #print(f"🔍 {code} {date_str} close值无效: {close_value}")
                    self.data_cache[cache_key] = None
                    return None
                    
                # 缓存有效数据
                self.data_cache[cache_key] = processed_data
                
            except Exception as e:
                print(f"🔴 {code} {date_str} 处理异常: {str(e)}")
                self.data_cache[cache_key] = None
                
        return self.data_cache[cache_key]

    def get_current_price(self, code):
        data = self.get_minute_data(code, self.current_date)
        if not data or 'close' not in data:
            return 0
        # 处理单个浮点数值和数组的情况
        return data['close'] if isinstance(data['close'], float) else data['close'][-1]

    def record_daily_equity(self):
        position_value = sum(
            self.get_current_price(code) * shares
            for code, shares in self.positions.items()
        )
        total_value = self.cash + position_value
        self.equity.append({
            'date': self.current_date,
            'total': total_value,
            'cash': self.cash,
            'positions': dict(self.positions)
        })

    def run_backtest(self):
        for date in tqdm(self.trade_dates, desc='回测进度'):
            self.current_date = date
            if not self._precheck(date):
                continue

            # 日线策略调整为盘后执行
            self.execute_strategy(date)
            self.record_daily_equity()
        self.generate_report()

    def analyze_market(self, date):
        hs300 = self.get_minute_data('399006.SZ', date)  # 修正指数代码
        if not hs300:
            return {'trend': 'neutral', 'volatility': 0}

        # 获取近20个交易日的收盘价
        history_dates = [d for d in self.trade_dates if d <= date][-20:]
        closes = []
        for d in history_dates:
            data = self.get_minute_data('399006.SZ', d)
            if data and 'close' in data:
                closes.append(data['close'])
        
        if len(closes) < 5:
            return {'trend': 'neutral', 'volatility': 0}

        close_array = np.array(closes)
        ma5 = talib.MA(close_array, timeperiod=5)[-1]
        ma20 = talib.MA(close_array, timeperiod=20)[-1] if len(closes) >= 20 else ma5
        
        # 改用日线波动率计算（最高/最低价）
        high = max(closes[-5:])
        low = min(closes[-5:])
        volatility = (high - low) / low

        return {
            'trend': 'up' if ma5 > ma20 else 'down',
            'volatility': volatility
        }

    def execute_strategy(self, date):  # 移除重复调用
        market_status = self.analyze_market(date)
        candidates = self.select_stocks(date)
        position_ratio = self.calculate_position(market_status['volatility'])
        self.rebalance_portfolio(candidates, position_ratio)
        candidates = self.select_stocks(date)
        position_ratio = self.calculate_position(market_status['volatility'])
        self.rebalance_portfolio(candidates, position_ratio)

    def select_stocks(self, date):
        # 获取当日涨停股票
        limit_up_pool = self._get_limit_up_stocks(date)
        
        # 获取热门板块（当日涨幅前5的板块）
        hot_sectors = xtdata.get_sector_list('板块', date.strftime('%Y%m%d'))[:5]
        
        candidates = []
        for sector in hot_sectors:
            # 获取板块成分股
            sector_stocks = xtdata.get_stock_list_in_sector(sector)
            
            for code in sector_stocks:
                try:
                    # 基础数据获取
                    data = self.get_minute_data(code, date)
                    hist_data = self._get_history_data(code, date, days=20)
                    
                    if not data or not hist_data:
                        continue
                        
                    # 量价筛选条件
                    condition_volume = data['volume'] > hist_data['avg_volume'] * 2  # 放量
                    condition_price = data['close'] > hist_data['ma5']  # 站上5日均线
                    condition_trend = hist_data['ma5'] > hist_data['ma10'] > hist_data['ma20']  # 多头排列
                    condition_macd = hist_data['macd'] > hist_data['signal']  # MACD金叉
                    condition_volatility = (hist_data['high'][-1] - hist_data['low'][-1]) / hist_data['low'][-1] > 0.05  # 波动率>5%
                    
                    # 风险回报比条件
                    stop_loss = hist_data['ma10']  # 止损位
                    risk_reward_ratio = (data['close'] - stop_loss) / (hist_data['high'][-5:].max() - data['close'])
                    
                    if all([condition_volume, condition_price, condition_trend,
                            condition_macd, condition_volatility, risk_reward_ratio >= 3]): 
                        # 涨停股优先
                        priority = 2 if code in limit_up_pool else 1
                        candidates.append((code, priority))
                        
                except Exception as e:
                    print(f"选股异常 {code}: {str(e)}")
                    continue
                    
        # 按优先级排序，选取前10
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [code for code, _ in candidates[:10]]

    def _get_history_data(self, code, date, days=20):
        """获取历史技术指标"""
        hist_dates = [d for d in self.trade_dates if d <= date][-days:]
        closes, highs, lows, volumes = [], [], [], []
        
        for d in hist_dates:
            data = self.get_minute_data(code, d)
            if data and 'close' in data:
                closes.append(data['close'])
                highs.append(data.get('high', data['close']))
                lows.append(data.get('low', data['close']))
                volumes.append(data.get('volume', 0))
                
        if len(closes) < 5:
            return None
            
        # 计算技术指标
        closes_array = np.array(closes)
        return {
            'ma5': talib.MA(closes_array, timeperiod=5)[-1],
            'ma10': talib.MA(closes_array, timeperiod=10)[-1],
            'ma20': talib.MA(closes_array, timeperiod=20)[-1],
            'macd': talib.MACD(closes_array)[0][-1],
            'signal': talib.MACD(closes_array)[1][-1],
            'high': np.array(highs),
            'low': np.array(lows),
            'avg_volume': np.mean(volumes[-5:])
        }

    def _get_limit_up_stocks(self, date):
        """获取当日涨停股票"""
        date_str = date.strftime('%Y%m%d')
        all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
        limit_up = []
        
        for code in all_stocks:
            try:
                pre_close = xtdata.get_market_data(
                    field_list='pre_close',
                    stock_list=[code],
                    period='1d',
                    start_time=date_str,
                    end_time=date_str
                )
                current_data = self.get_minute_data(code, date)
                
                if pre_close and current_data and 'close' in current_data:
                    pct_change = (current_data['close'] - pre_close[code].iloc[0]) / pre_close[code].iloc[0]
                    if pct_change >= 0.095:  # 涨停阈值
                        limit_up.append(code)
            except:
                continue
                
        return limit_up

    def calculate_position(self, volatility):
        if volatility < 0.01: return 0.8
        elif volatility < 0.03: return 0.6
        else: return 0.3

    def rebalance_portfolio(self, candidates, target_ratio):
        current_value = sum(
            self.get_current_price(code) * shares
            for code, shares in self.positions.items()
        )
        target_value = (current_value + self.cash) * target_ratio

        # 平仓不在候选列表的股票
        for code in list(self.positions.keys()):
            if code not in candidates:
                self.place_order(code, 0)

        # 等权重分配
        if len(candidates) == 0: return
        weight = 1.0 / len(candidates)
        for code in candidates:
            price = self.get_current_price(code)
            if price == 0: continue
            target_shares = (target_value * weight) / price
            self.place_order(code, target_shares)

    def place_order(self, code, target_shares):
        current_shares = self.positions.get(code, 0)
        delta = target_shares - current_shares
        price = self.get_current_price(code)

        if delta > 0:  # 买入
            cost = delta * price * (1 + self.commission_rate + self.slippage)
            if cost > self.cash: return
            self.cash -= cost
            self.positions[code] = current_shares + delta
        elif delta < 0:  # 卖出
            shares_to_sell = abs(delta)
            if current_shares < shares_to_sell: return
            proceeds = shares_to_sell * price * (1 - self.commission_rate - self.slippage)
            self.cash += proceeds
            self.positions[code] -= shares_to_sell
            if self.positions[code] == 0:
                del self.positions[code]



    def generate_report(self):
        if not self.equity:
            print("没有可用的回测数据")
            return
            
        df = pd.DataFrame(self.equity)
        if 'total' not in df.columns:
            print("数据记录错误，缺少'total'字段")
            return
            
        df['return'] = df['total'].pct_change()

        print("\n================ 回测结果 ================")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终净值: {df['total'].iloc[-1]:,.2f}")
        print(f"总收益率: {df['total'].iloc[-1] / self.initial_capital - 1:.2%}")
        print(f"年化收益率: {self._calc_annual_return(df):.2%}")
        print(f"最大回撤: {self._calc_max_drawdown(df):.2%}")
        print(f"夏普比率: {self._calc_sharpe(df):.2f}")

    def _calc_annual_return(self, df):
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        return (df['total'].iloc[-1] / self.initial_capital) ** (365 / days) - 1

    def _calc_max_drawdown(self, df):
        peak = df['total'].cummax()
        return (df['total'] / peak - 1).min()

    def _calc_sharpe(self, df):
        return df['return'].mean() / df['return'].std() * np.sqrt(252)

    def _precheck(self, date):
        """简化版市场检查"""
        # 基础日期检查
        if date.weekday() >= 5 or date > datetime.now():
            return False
            
        # 获取上证指数数据
        market_data = self.get_minute_data('000001.SZ', date)
        return bool(market_data and market_data.get('close', 0) > 0)

if __name__ == "__main__":
    backtester = QMTBacktester(
        start_date=datetime(2024, 4, 9),
        end_date=datetime(2024, 12, 31),
        capital=1000000
    )
    backtester.run_backtest()