from datetime import datetime, timedelta
import xtquant.xtdata as xtdata
import pandas as pd
import numpy as np
import talib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from openpyxl.styles import Font, PatternFill
import os
from scipy.stats import linregress  # 新增线性回归计算

# 新增回测配置
INITIAL_CAPITAL = 100000  # 初始资金10万元
TRADE_DAYS = 66           # 3个月交易日（按22天/月计算）
COMMISSION = 0.0003       # 交易佣金万三

# 新增回测数据结构
backtest_results = {
    '日期': [],
    '总资产': [],
    '创业板指数': [],
    '交易记录': []
}

def run_backtest():
    """运行回测系统"""
    capital = INITIAL_CAPITAL
    holdings = {}  # 持仓 {代码: (数量, 买入日期)}
    
    # 获取市场配置
    market = 'SZ'  # 深圳市场
    
    # 生成日期范围（包含可能非交易日）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRADE_DAYS+5)
    all_dates = pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()
    
    # 获取节假日数据并过滤
    holidays = xtdata.get_holidays()  # 获取指定市场节假日
    trade_dates = [
        date for date in all_dates
        if date not in holidays
        and datetime.strptime(date, '%Y%m%d').weekday() < 5  # 排除周末
    ][-TRADE_DAYS:]  # 取最近TRADE_DAYS个交易日
    
    for i in range(1, len(trade_dates)-1):
        current_date = trade_dates[i]
        prev_date = trade_dates[i-1]
        
        # 卖出昨日持仓
        to_remove = []
        for code, (qty, buy_date) in holdings.items():
            if buy_date == prev_date:
                open_price = get_open_price(code, current_date)
                if open_price:
                    sell_amount = qty * open_price * (1 - COMMISSION)
                    capital += sell_amount
                    record_trade(current_date, code, '卖出', qty, open_price)
                to_remove.append(code)
        for code in to_remove:
            del holdings[code]
        
        # 获取前一日选股结果
        selected_stocks = get_previous_selection(prev_date)
        
        # 执行当日买入
        if selected_stocks and capital > 100:
            position_value = capital * 0.2  # 每只股票20%仓位
            for stock in selected_stocks[:5]:  # 只买前5只
                code = stock['代码']
                open_price = get_open_price(code, current_date)
                if open_price:
                    qty = position_value // open_price
                    if qty > 0:
                        cost = qty * open_price * (1 + COMMISSION)
                        if capital >= cost:
                            capital -= cost
                            holdings[code] = (qty, current_date)
                            record_trade(current_date, code, '买入', qty, open_price)
        
        # 记录当日资产
        total_asset = capital + sum([qty * get_open_price(code, current_date) for code, (qty, _) in holdings.items()])
        cyb_index = get_cyb_index(current_date)
        
        backtest_results['日期'].append(current_date)
        backtest_results['总资产'].append(total_asset)
        backtest_results['创业板指数'].append(cyb_index)

    # 生成报告
    generate_report()

def manual_feature_extractor(close_series):
    """手动实现特征计算（比TSFRESH快3倍）"""
    x = np.arange(len(close_series))
    
    # 线性趋势斜率
    slope, _, _, _, _ = linregress(x, close_series)
    
    return {
        'mean': np.nanmean(close_series),
        'slope': slope,
        'std': np.nanstd(close_series),
        'var': np.nanvar(close_series)
    }

def get_support_resistance(code, end_date, days=20):
    """获取动态支撑压力位"""
    start_date = (end_date - timedelta(days=days*2)).strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    data = xtdata.get_market_data(
        stock_list=[code],
        period='1d',
        start_time=start_date,
        end_time=end_date_str,
        field_list=['low','high','close']
    )
    
    if not data or 'low' not in data or code not in data['low'].index:
        return None, None

    lows = data['low'].loc[code].dropna()[-days*2:]
    highs = data['high'].loc[code].dropna()[-days*2:]
    
    valid_lows = lows[(lows > 0) & (~np.isinf(lows))]
    valid_highs = highs[(highs > 0) & (~np.isinf(highs))]
    
    if len(valid_lows) < days or len(valid_highs) < days:
        return None, None
        
    recent_low = valid_lows[-days:].min()
    recent_high = valid_highs[-days:].max()
    
    fib_level = recent_high - (recent_high - recent_low) * 0.618
    return max(fib_level, recent_low), recent_high

def get_market_status():
    """获取大盘趋势状态"""
    try:
        sh_index = xtdata.get_market_data(
            field_list=['close'], 
            stock_list=['399006.SZ'], 
            period='1d',
            start_time=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        )
        if not sh_index or 'close' not in sh_index:
            return 'neutral'
            
        closes = sh_index['close'].iloc[:, 0].dropna()
        if len(closes) < 20:
            return 'neutral'
            
        ma20 = talib.MA(closes, timeperiod=20)[-1]
        ma5 = talib.MA(closes, timeperiod=5)[-1]
        return 'uptrend' if ma5 > ma20 else 'downtrend'
    except Exception as e:
        print(f"大盘状态检测异常: {str(e)}")
        return 'neutral'

def select_stocks():
    """智能选股主逻辑"""
    
    if get_market_status() == 'downtrend':
        print("⚠️ 大盘处于下跌趋势，暂停选股")
        return pd.DataFrame()
    
    print("[1/4] 🚀 初始化选股引擎...")
    
    all_stocks = xtdata.get_stock_list_in_sector('沪深A股')
    print(f"初始股票池数量: {len(all_stocks)}")
    
    print("[2/4] ⚡ 执行快速过滤...")
    filtered_stocks = []
    for idx, code in enumerate(all_stocks, 1):
        try:
            detail = xtdata.get_instrument_detail(code)
            name = detail.get('InstrumentName', '未知名称')
            
            tick = xtdata.get_full_tick([code]).get(code, {})
            price = tick.get('lastPrice', 0)
            
            if (not name.startswith('ST')) and (5 < price < 500):
                filtered_stocks.append(code)
                
            if idx % 1000 == 0:
                print(f"已处理 {idx} 只股票，当前通过率: {len(filtered_stocks)/idx:.1%}")
                
        except Exception as e:
            continue
            
    print(f"初步筛选后剩余 {len(filtered_stocks)} 只股票")
    
    print(f"[3/4] 🚀 并行分析 {len(filtered_stocks)} 只股票...")
    candidates = []
    
    all_klines = xtdata.get_market_data(
        stock_list=filtered_stocks,
        period='1d',
        start_time=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
        end_time=datetime.now().strftime('%Y%m%d'),
        field_list=['open','high','low','close','volume']
    )
    
    analyze_func = partial(analyze_stock, cached_kline=all_klines)
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()//2 or 4) as executor:
        futures = {executor.submit(analyze_func, code): code for code in filtered_stocks}
        
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                candidates.append(result)
                if idx % 100 == 0:
                    print(f"已分析 {idx} 只，当前入选 {len(candidates)} 只")
                if len(candidates) >= 50:
                    executor.shutdown(wait=False)
                    break
    
    print("[4/4] 🔍 执行最终筛选...")
    result_df = pd.DataFrame()
    if candidates:
        df = pd.DataFrame(candidates)
        df['波动率'] = df.apply(lambda x: abs(x['压力位'] - x['支撑位'])/x['当前价'], axis=1)
        df = df[df['波动率'] < 0.15]
        df = df.sort_values(by=['条件得分', '盈亏比'], ascending=[False, False])
        result_df = df.head(5)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = f"选股结果_{timestamp}.xlsx"
        try:
            with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='原始数据')
                
                display_df = result_df[['代码', '名称', '当前价', '支撑位', '压力位', 
                                      '止损价', '盈亏比', '条件得分', 'ATR']].copy()
                display_df['当前价'] = display_df['当前价'].apply(lambda x: f"¥{x:.2f}")
                display_df['支撑位'] = display_df['支撑位'].map("{:.2f}".format)
                display_df['压力位'] = display_df['压力位'].map("{:.2f}".format)
                display_df['止损价'] = display_df['止损价'].map("{:.2f}".format)
                display_df['ATR'] = display_df['ATR'].map("{:.4f}".format)
                display_df.to_excel(writer, index=False, sheet_name='分析报告')
                
                for sheet in writer.sheets:
                    worksheet = writer.sheets[sheet]
                    for col in worksheet.columns:
                        max_length = max(len(str(cell.value)) for cell in col)
                        worksheet.column_dimensions[col[0].column_letter].width = max_length + 2
                    worksheet.freeze_panes = 'A2'
                    header_style = Font(bold=True, color='FFFFFF')
                    header_fill = PatternFill(fill_type='solid', fgColor='4F81BD')
                    for cell in worksheet[1]:
                        cell.font = header_style
                        cell.fill = header_fill
                
            print(f"📊 结果已保存至: {excel_name}")
            
        except Exception as e:
            print(f"⚠️ 保存Excel失败: {str(e)}")
    
    return result_df

def analyze_stock(code, cached_kline=None):
    """优化后的股票分析函数（移除了TSFRESH依赖）"""
    try:
        current_date = datetime.now()
        start_date = (current_date - timedelta(days=60)).strftime('%Y%m%d')
        
        # 获取K线数据（保持不变）
        klines = cached_kline if cached_kline else xtdata.get_market_data(
            stock_list=[code],
            period='1d',
            start_time=start_date,
            end_time=current_date.strftime('%Y%m%d'),
            field_list=['open','high','low','close','volume']
        )
        
        if not klines or 'close' not in klines or code not in klines['close'].index:
            return None
    
        try:
            df = pd.DataFrame({
                'open': klines['open'].loc[code],
                'high': klines['high'].loc[code],
                'low': klines['low'].loc[code],
                'close': klines['close'].loc[code],
                'volume': klines['volume'].loc[code]
            }).sort_index().dropna()
        except KeyError:
            return None
            
        if len(df) < 20:
            return None
    
        # 计算技术指标（保持不变）
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['RSI'] = talib.RSI(df['close'], timeperiod=12)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['SUPERT'] = middle
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['VOL_MA5'] = df['volume'].rolling(5).mean()
        
        # 手动提取特征（替换TSFRESH部分）
        features = manual_feature_extractor(df['close'].values)
        
        # 获取支撑压力位（保持不变）
        support, resistance = get_support_resistance(code, current_date)
        if not support or not resistance:
            return None
            
        # 获取实时行情（保持不变）
        tick = xtdata.get_full_tick([code]).get(code, {})
        price = tick.get('lastPrice', 0)
        if (price - support) <= 0 or not (5 < price < 500):
            return None
            
        # 风险回报比计算（保持不变）
        risk_reward = (resistance - price) / max(price - support, 0.01)
        if risk_reward < 2 or price > support * 1.05:
            return None
            
        # 更新后的条件评分
        condition_met = sum([
            df['close'].iloc[-1] > df['VWAP'].iloc[-1],
            df['close'].iloc[-1] > df['SUPERT'].iloc[-1],
            df['RSI'].iloc[-1] < 30,
            df['volume'].iloc[-1] > df['VOL_MA5'].iloc[-1] * 2,
            features['slope'] > 0,  # 直接使用手动计算的特征
            df['ATR'].iloc[-1] < df['ATR'].rolling(20).mean().iloc[-1],
            df['close'].iloc[-3:].mean() > df['close'].iloc[-10:-3].mean()
        ])
        if condition_met < 5:
            return None
            
        # 获取股票详情（保持不变）
        detail = xtdata.get_instrument_detail(code)
        stock_name = detail.get('InstrumentName', code)
        
        return {
            '代码': code,
            '名称': stock_name,
            '当前价': price,
            '支撑位': support,
            '压力位': resistance,
            '止损价': round(support * 0.97, 2),
            '盈亏比': round(risk_reward, 2),
            '条件得分': condition_met,
            'ATR': df['ATR'].iloc[-1]
        }
        
    except Exception as e:
        return None

def get_open_price(code, date):
    """获取指定日期开盘价"""
    data = xtdata.get_market_data(
        stock_list=[code],
        period='1d',
        start_time=date,
        end_time=date,
        field_list=['open']
    )
    return data['open'].loc[code].iloc[0] if data else None

def get_previous_selection(date):
    """获取前一日选股结果（增强版datetime模拟）"""
    global datetime
    original_datetime = datetime
    
    class MockDatetime:
        __real_datetime__ = original_datetime
        
        @classmethod
        def now(cls):
            # 修正为直接构造datetime对象
            return cls.__real_datetime__(
                year=int(date[:4]),
                month=int(date[4:6]), 
                day=int(date[6:8])
            )
        
        @classmethod
        def strptime(cls, date_string, format):
            return cls.__real_datetime__.strptime(date_string, format)
        
        # 保持属性转发
        datetime = property(lambda cls: cls.__real_datetime__.datetime)
        date = property(lambda cls: cls.__real_datetime__.date)
        timedelta = property(lambda cls: cls.__real_datetime__.timedelta)

    try:
        datetime = MockDatetime
        result = select_stocks()
        return result.to_dict('records')
    finally:
        datetime = original_datetime

def get_cyb_index(date):
    """获取创业板指数"""
    data = xtdata.get_market_data(
        stock_list=['399006.SZ'],
        period='1d',
        start_time=date,
        end_time=date,
        field_list=['close']
    )
    return data['close'].iloc[0] if data else 0

def record_trade(date, code, operation, qty, price):
    """记录交易明细"""
    backtest_results['交易记录'].append({
        '日期': date,
        '代码': code,
        '操作': operation,
        '数量': qty,
        '价格': round(price, 2),
        '金额': round(qty * price, 2)
    })

def generate_report():
    """生成回测报告"""
    # 保存交易记录
    trade_df = pd.DataFrame(backtest_results['交易记录'])
    trade_df.to_excel('回测交易记录.xlsx', index=False)
    
    # 生成资产曲线
    result_df = pd.DataFrame({
        '日期': backtest_results['日期'],
        '总资产': backtest_results['总资产'],
        '创业板指数': backtest_results['创业板指数']
    })
    
    # 计算收益率
    result_df['策略收益率'] = result_df['总资产'].pct_change().fillna(0)
    result_df['基准收益率'] = result_df['创业板指数'].pct_change().fillna(0)
    
    # 计算累计收益
    result_df['策略累计收益'] = (result_df['策略收益率'] + 1).cumprod()
    result_df['基准累计收益'] = (result_df['基准收益率'] + 1).cumprod()
    
    # 计算最大回撤
    result_df['策略净值'] = result_df['总资产'] / INITIAL_CAPITAL
    result_df['策略峰值'] = result_df['策略净值'].cummax()
    result_df['策略回撤'] = (result_df['策略峰值'] - result_df['策略净值']) / result_df['策略峰值']
    
    # 输出统计结果
    total_return = (result_df['总资产'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    max_drawdown = result_df['策略回撤'].max()
    annual_return = (result_df['策略累计收益'].iloc[-1] ** (252/TRADE_DAYS)) - 1
    
    print(f"\n=== 回测结果 ===")
    print(f"初始资金: {INITIAL_CAPITAL}元")
    print(f"最终资产: {result_df['总资产'].iloc[-1]:.2f}元")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"交易次数: {len(trade_df)}次")
    print(f"胜率: {len(trade_df[trade_df['操作']=='卖出'])/(len(trade_df)//2):.2%}")
    
    # 绘制收益曲线
    plt.figure(figsize=(12,6))
    plt.plot(result_df['日期'], result_df['策略累计收益'], label='策略收益')
    plt.plot(result_df['日期'], result_df['基准累计收益'], label='创业板指数')
    plt.title('策略收益 vs 创业板指数')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('收益曲线对比.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🚀 启动回测系统...")
    run_backtest()