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

if __name__ == '__main__':
    print("\n🔍 开始执行智能选股...")
    start_time = datetime.now()
    
    selected_stocks = select_stocks()
    
    print("\n🎉 最终精选股票：")
    if not selected_stocks.empty:
        print(selected_stocks[['代码', '名称', '当前价', '盈亏比', '条件得分']].to_string(index=False))
    else:
        print("⚠️ 未找到符合条件股票")
        
    print(f"\n⏱️ 总耗时: {datetime.now()-start_time}")