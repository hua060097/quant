"""
评估风控修改方案的影响：
1. 还原每笔交易的风控触发详情
2. 模拟修改T+2止损规则后的效果
3. 评估对大亏交易和正常交易的影响
"""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
MINUTE_DIR = BASE_DIR.parent / "local_data" / "minute"
DAILY_DIR = BASE_DIR.parent / "local_data" / "daily"
FACTOR_FILE = BASE_DIR / "factor_lab_data" / "factors.parquet"
LOG_FILE = BASE_DIR / "stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log"

# ============================================================
# Step 1: 从日志提取所有117笔交易的完整信息
# ============================================================
print("=" * 80)
print("Step 1: 从日志提取所有交易及风控详情")
print("=" * 80)

lines = open(LOG_FILE, 'r', encoding='utf-8').readlines()

# 提取买入记录
buys = []
current_stock = None
current_buy_date = None
current_buy_price = None

for line in lines:
    m_stock = re.search(r'分层并行仓位生效: (\S+) 使用', line)
    if m_stock:
        current_stock = m_stock.group(1)
    m_tday = re.search(r'并行持仓买入:.*?T日=(\d{4}-\d{2}-\d{2})', line)
    if m_tday:
        current_buy_date = m_tday.group(1)
    m_buyp = re.search(r'实际买入价[：:]?\s*([\d.]+)', line)
    if m_buyp:
        current_buy_price = float(m_buyp.group(1))
    m_confirm = re.search(r'买入后持仓状态', line)
    if m_confirm and current_stock:
        buys.append({
            'stock_code': current_stock,
            'buy_date': current_buy_date,
            'buy_price': current_buy_price,
        })
        current_stock = None
        current_buy_date = None
        current_buy_price = None

# 提取卖出记录
sells = []
for line in lines:
    m = re.search(r'卖出: (\S+), 价格: ([\d.]+), 收益: -?[\d.]+ \((-?[\d.]+)%\), 原因: (.+)', line)
    if m:
        sells.append({
            'stock_code': m.group(1),
            'sell_price': float(m.group(2)),
            'return_pct': float(m.group(3)),
            'exit_reason': m.group(4).strip()
        })

# 配对
buy_used = [False] * len(buys)
trades = []
for sell in sells:
    for j, buy in enumerate(buys):
        if not buy_used[j] and buy['stock_code'] == sell['stock_code']:
            trades.append({
                'stock_code': sell['stock_code'],
                'buy_date': buy['buy_date'],
                'buy_price': buy['buy_price'],
                'sell_price': sell['sell_price'],
                'return_pct': sell['return_pct'],
                'exit_reason': sell['exit_reason'],
            })
            buy_used[j] = True
            break

df_trades = pd.DataFrame(trades)
print(f"总交易: {len(df_trades)}")

# 分类退出原因
def classify_exit(reason):
    if pd.isna(reason):
        return 'unknown'
    if 'T+2 09:35失败止损' in reason:
        return 'T2_0935_hard'
    if 'T+2 10:00继续走坏止损' in reason:
        return 'T2_1000_soft'
    if 'T+2收盘仍未转正' in reason:
        return 'T2_close_breakeven'
    if 'T+3 09:35仍未转正' in reason:
        return 'T3_0935_breakeven'
    if '累计收益低于' in reason:
        return 'cumulative_stop'
    if '利润保护' in reason:
        return 'profit_protect'
    if '小盈利后延续失败' in reason:
        return 'small_profit_fail'
    if '连续' in reason and '阴线' in reason:
        return 'consecutive_bearish'
    if '到期' in reason:
        return 'expiry'
    return 'other'

df_trades['exit_type'] = df_trades['exit_reason'].apply(classify_exit)
print("退出类型分布:")
print(df_trades['exit_type'].value_counts().to_string())

# ============================================================
# Step 2: 加载交易日历 + 日线数据 + 分钟数据
# ============================================================
print("\n" + "=" * 80)
print("Step 2: 加载数据")
print("=" * 80)

df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# 收集所有需要的月份
needed_months = set()
for _, t in df_trades.iterrows():
    bd = pd.Timestamp(str(t['buy_date'])[:10])
    if bd in date_to_idx:
        idx = date_to_idx[bd]
        for offset in range(5):
            if idx + offset < len(all_dates):
                ts = pd.Timestamp(all_dates[idx + offset])
                needed_months.add(f"{ts.year}{ts.month:02d}")

# 加载日线
daily_frames = []
for month_dir in sorted(DAILY_DIR.iterdir()):
    if month_dir.is_dir() and month_dir.name in needed_months:
        for f in month_dir.glob("*.parquet"):
            try:
                df_d = pd.read_parquet(f, columns=['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
                daily_frames.append(df_d)
            except:
                pass
daily_data = pd.concat(daily_frames, ignore_index=True)
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['lookup'] = daily_data['stock_code'] + '_' + daily_data['date'].dt.strftime('%Y-%m-%d')
daily_dict = daily_data.set_index('lookup').to_dict('index')
print(f"日线数据: {daily_data.shape}")

# ============================================================
# Step 3: 对每笔交易，还原完整持仓期间的日线和关键分钟数据
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 还原每笔交易的风控详情")
print("=" * 80)

def get_trading_date(buy_date_str, offset):
    bd = pd.Timestamp(buy_date_str[:10])
    if bd not in date_to_idx:
        return None
    idx = date_to_idx[bd]
    target_idx = idx + offset
    if 0 <= target_idx < len(all_dates):
        return all_dates[target_idx]
    return None

def load_minute_data(stock_code, date_str):
    ts = pd.Timestamp(date_str)
    ym = f"{ts.year}{ts.month:02d}"
    f = MINUTE_DIR / ym / "minute_data.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f, columns=['stock_code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        mask = (df['stock_code'] == stock_code) & (df['datetime'].dt.date == ts.date())
        return df[mask].sort_values('datetime')
    except:
        return None

trade_details = []

for _, trade in df_trades.iterrows():
    stock = trade['stock_code']
    buy_date = str(trade['buy_date'])[:10]
    buy_price = trade['buy_price']
    exit_type = trade['exit_type']

    detail = {
        'stock_code': stock,
        'buy_date': buy_date,
        'buy_price': buy_price,
        'actual_return_pct': trade['return_pct'],
        'exit_type': exit_type,
        'exit_reason': trade['exit_reason'],
    }

    # T+1 日线
    t1_date = get_trading_date(buy_date, 1)
    t1_key = f"{stock}_{pd.Timestamp(t1_date).strftime('%Y-%m-%d')}" if t1_date else None
    t1_daily = daily_dict.get(t1_key, {}) if t1_key else {}

    if t1_daily:
        detail['T1_close'] = t1_daily.get('close')
        detail['T1_low'] = t1_daily.get('low')
        detail['T1_high'] = t1_daily.get('high')
        detail['T1_open'] = t1_daily.get('open')
        detail['T1_return_pct'] = (t1_daily['close'] / buy_price - 1) * 100
        detail['T1_low_pct'] = (t1_daily['low'] / buy_price - 1) * 100
        detail['T1_volume'] = t1_daily.get('volume', 0)

    # T+2 日线
    t2_date = get_trading_date(buy_date, 2)
    t2_key = f"{stock}_{pd.Timestamp(t2_date).strftime('%Y-%m-%d')}" if t2_date else None
    t2_daily = daily_dict.get(t2_key, {}) if t2_key else {}

    if t2_daily:
        detail['T2_close'] = t2_daily.get('close')
        detail['T2_low'] = t2_daily.get('low')
        detail['T2_high'] = t2_daily.get('high')
        detail['T2_open'] = t2_daily.get('open')
        detail['T2_return_pct'] = (t2_daily['close'] / buy_price - 1) * 100
        detail['T2_low_pct'] = (t2_daily['low'] / buy_price - 1) * 100

    # T+3 日线
    t3_date = get_trading_date(buy_date, 3)
    t3_key = f"{stock}_{pd.Timestamp(t3_date).strftime('%Y-%m-%d')}" if t3_date else None
    t3_daily = daily_dict.get(t3_key, {}) if t3_key else {}

    if t3_daily:
        detail['T3_close'] = t3_daily.get('close')
        detail['T3_return_pct'] = (t3_daily['close'] / buy_price - 1) * 100

    # T+2 分钟级别特征
    if t2_date:
        t2_min = load_minute_data(stock, str(t2_date)[:10])
        if t2_min is not None and len(t2_min) > 0:
            first_bar = t2_min.iloc[0]  # 09:35
            first_30min = t2_min.head(6)  # 09:35-10:00

            # 09:35收益
            detail['T2_0935_return_pct'] = (first_bar['close'] / buy_price - 1) * 100
            detail['T2_0935_vs_T1close_pct'] = (first_bar['open'] / (detail.get('T1_close', first_bar['open'])) - 1) * 100 if detail.get('T1_close') else None

            # T+1低点相关
            if detail.get('T1_low'):
                detail['T2_0935_vs_T1low_pct'] = (first_bar['low'] / detail['T1_low'] - 1) * 100
                detail['T2_0935_broke_T1low'] = first_bar['low'] < detail['T1_low']

            if len(first_30min) > 0:
                detail['T2_1000_return_pct'] = (first_30min.iloc[-1]['close'] / buy_price - 1) * 100
                if detail.get('T1_close'):
                    detail['T2_1000_vs_T1close_pct'] = (first_30min.iloc[-1]['close'] / detail['T1_close'] - 1) * 100
                if detail.get('T1_low'):
                    detail['T2_1000_vs_T1low_pct'] = (first_30min['low'].min() / detail['T1_low'] - 1) * 100
                    detail['T2_1000_recovered_T1low'] = first_30min.iloc[-1]['close'] > detail['T1_low']

                # VWAP
                total_amount = first_30min['amount'].sum()
                total_volume = first_30min['volume'].sum()
                vwap = total_amount / total_volume / 100 if total_volume > 0 else None
                detail['T2_1000_close_vs_vwap_pct'] = ((first_30min.iloc[-1]['close'] / vwap) - 1) * 100 if vwap else None

                # 早盘反弹
                min_low_idx = first_30min['low'].idxmin()
                min_low_bar = first_30min.loc[min_low_idx]
                bars_after = first_30min[first_30min['datetime'] > min_low_bar['datetime']]
                detail['T2_1000_bounce_from_low_pct'] = (bars_after.iloc[-1]['close'] / min_low_bar['low'] - 1) * 100 if len(bars_after) > 0 else 0

                # 跌涨量比
                down_bars = first_30min[first_30min['close'] < first_30min['open']]
                up_bars = first_30min[first_30min['close'] > first_30min['open']]
                down_vol = down_bars['volume'].sum()
                up_vol = up_bars['volume'].sum()
                detail['T2_1000_down_up_volume_ratio'] = down_vol / up_vol if up_vol > 0 else (999 if down_vol > 0 else 1)

    trade_details.append(detail)

df_detail = pd.DataFrame(trade_details)
print(f"还原交易详情: {len(df_detail)} 笔")

# ============================================================
# Step 4: 定义风控修改方案并模拟
# ============================================================
print("\n" + "=" * 80)
print("Step 4: 风控修改方案模拟")
print("=" * 80)

# 现有风控规则回顾:
# T+2 09:35: 收益<=-4% → 硬止损 (enable_t2_0935_hard_stop=True)
# T+2 10:00: 收益<=-2% → 软止损 (enable_t2_1000_soft_stop=False, 默认关闭!)
# T+2 14:55: 收益<=0% → 保本出
# T+3 09:35: 收益<=0% → 保本出
# 累计收益<=-5% → 硬止损

# 分析: 当前实际日志中的止损触发了哪些规则?
# 从exit_type看:
#   T2_0935_hard: 风控:T+2 09:35失败止损(-4.00%)
#   T2_1000_soft: 风控:T+2 10:00继续走坏止损(-2.00%)
#   cumulative_stop: 累计收益低于-5.00%
#   T2_close_breakeven: T+2收盘仍未转正
#   T3_0935_breakeven: T+3 09:35仍未转正
#   profit_protect: 利润保护
#   small_profit_fail: 小盈利后延续失败
#   expiry: 到期

# 修改方案A: T+2 09:35 条件止损（加入早盘修复判断）
# 原规则: 收益<=-4% → 无条件止损
# 新规则: 收益<=-4% 时，检查以下"修复信号"：
#   如果10:00收盘价 > T+1收盘价 且 10:00跌涨量比<2.0 且 早盘反弹>2% → 延迟到10:00判断
#   否则保留硬止损

# 修改方案B: T+2 10:00 条件止损
# 原规则（已关闭）: 收益<=-2% → 软止损
# 新规则: 收益<=-4%时延迟到10:00判断
#   10:00收益<=-5% → 止损（真恶化）
#   10:00收益>-5% 且 收复T+1低点 → 继续持有
#   10:00收益>-5% 但未收复T+1低点 → 止损

# 修改方案C: 综合方案
# 09:35收益<=-4%时，不立即止损，延迟到10:00判断：
#   10:00收益<=-5% → 止损
#   10:00跌涨量比>2.5 → 止损（放量下杀）
#   10:00未收复T+1低点 且 10:00 vs T+1收盘<-4% → 止损
#   否则 → 继续持有到T+2 14:55保本出

def simulate_plan_C(row):
    """
    模拟方案C: T+2 09:35 硬止损改为条件止损
    返回: {'should_delay_0935': bool, 'should_stop_1000': bool, 'simulated_return_pct': float}
    """
    result = {
        'original_return_pct': row['actual_return_pct'],
        'original_exit_type': row['exit_type'],
        'simulated_return_pct': row['actual_return_pct'],  # 默认不变
        'simulated_exit_type': row['exit_type'],
        'changed': False,
        'change_reason': '',
    }

    # 只影响T+2 09:35硬止损的交易
    if row['exit_type'] != 'T2_0935_hard':
        return result

    # 检查是否有10:00数据
    if pd.isna(row.get('T2_1000_return_pct', np.nan)):
        return result

    t2_0935_ret = row.get('T2_0935_return_pct', None)
    t2_1000_ret = row.get('T2_1000_return_pct', None)
    t2_1000_vs_t1close = row.get('T2_1000_vs_T1close_pct', None)
    t2_1000_vs_t1low = row.get('T2_1000_vs_T1low_pct', None)
    vol_ratio = row.get('T2_1000_down_up_volume_ratio', None)
    recovered_t1low = row.get('T2_1000_recovered_T1low', None)
    bounce = row.get('T2_1000_bounce_from_low_pct', None)

    # 方案C: 延迟到10:00判断
    # 10:00止损条件（任一满足则止损）:
    stop_at_1000 = False
    stop_reason = ''

    # 条件1: 10:00收益<=-5%
    if pd.notna(t2_1000_ret) and t2_1000_ret <= -5:
        stop_at_1000 = True
        stop_reason = '10:00收益<=-5%'

    # 条件2: 放量下杀（跌涨量比>2.5）
    if pd.notna(vol_ratio) and vol_ratio > 2.5:
        stop_at_1000 = True
        stop_reason = f'放量下杀(vol_ratio={vol_ratio:.1f})'

    # 条件3: 未收复T+1低点 且 低于T+1收盘4%以上
    if pd.notna(t2_1000_vs_t1close) and t2_1000_vs_t1close < -4:
        if recovered_t1low == False or (pd.notna(t2_1000_vs_t1low) and t2_1000_vs_t1low < -2):
            stop_at_1000 = True
            stop_reason = f'未收复T+1低(vs_T1close={t2_1000_vs_t1close:.1f}%)'

    if stop_at_1000:
        # 止损但用10:00的收益（比09:35可能更低或更高）
        result['simulated_return_pct'] = t2_1000_ret
        result['simulated_exit_type'] = 'T2_1000_conditional_stop'
        result['changed'] = True
        result['change_reason'] = f'09:35延迟到10:00止损: {stop_reason}'
    else:
        # 继续持有到T+2 14:55保本出
        if pd.notna(row.get('T2_return_pct', np.nan)):
            result['simulated_return_pct'] = row['T2_return_pct']
        elif pd.notna(row.get('T3_return_pct', np.nan)):
            result['simulated_return_pct'] = row['T3_return_pct']
        result['simulated_exit_type'] = 'T2_close_breakeven_sim'
        result['changed'] = True
        result['change_reason'] = '09:35延迟到10:00有修复信号, 持有到T+2收盘'

    return result

# 模拟方案C
sim_results = []
for _, row in df_detail.iterrows():
    sim = simulate_plan_C(row)
    sim_results.append(sim)

df_sim = pd.DataFrame(sim_results)
df_detail['sim_return_pct'] = df_sim['simulated_return_pct']
df_detail['sim_exit_type'] = df_sim['simulated_exit_type']
df_detail['sim_changed'] = df_sim['changed']
df_detail['sim_change_reason'] = df_sim['change_reason']

# ============================================================
# Step 5: 分析修改影响
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 风控修改影响分析")
print("=" * 80)

# 5.1 受影响的交易
affected = df_detail[df_detail['sim_changed'] == True].copy()
not_affected = df_detail[df_detail['sim_changed'] == False].copy()
print(f"受影响交易: {len(affected)} 笔 / {len(df_detail)} 总")
print(f"不受影响交易: {len(not_affected)} 笔")

if len(affected) > 0:
    print(f"\n--- 受影响交易详情 ---")
    for _, r in affected.iterrows():
        orig = r['actual_return_pct']
        sim = r['sim_return_pct']
        diff = sim - orig
        print(f"  {r['stock_code']:10s} | buy={r['buy_date']} | "
              f"orig={orig:+.1f}% -> sim={sim:+.1f}% (diff={diff:+.1f}%) | "
              f"原因: {r['sim_change_reason']}")

    # 5.2 对大亏交易的影响
    print(f"\n--- 对大亏交易(return<-5%)的影响 ---")
    big_loss_trades = df_detail[df_detail['actual_return_pct'] < -5]
    big_loss_affected = big_loss_trades[big_loss_trades['sim_changed'] == True]

    if len(big_loss_affected) > 0:
        orig_avg = big_loss_trades['actual_return_pct'].mean()
        # 模拟后的平均收益（对受影响的用模拟值，未受影响的用原值）
        sim_returns = df_detail.apply(
            lambda r: r['sim_return_pct'] if r['sim_changed'] else r['actual_return_pct'], axis=1
        )
        big_loss_mask = df_detail['actual_return_pct'] < -5
        sim_avg = sim_returns[big_loss_mask].mean()

        orig_total = big_loss_trades['actual_return_pct'].sum()
        sim_total = sim_returns[big_loss_mask].sum()

        print(f"  大亏交易数: {len(big_loss_trades)}")
        print(f"  受影响的大亏交易: {len(big_loss_affected)}")
        print(f"  原平均收益: {orig_avg:.2f}%")
        print(f"  模拟平均收益: {sim_avg:.2f}%")
        print(f"  原总收益: {orig_total:.1f}%")
        print(f"  模拟总收益: {sim_total:.1f}%")
        print(f"  改善: {sim_total - orig_total:.1f}% (总) / {sim_avg - orig_avg:.2f}% (平均)")
    else:
        print("  无大亏交易受影响")

    # 5.3 对所有交易的影响
    print(f"\n--- 对所有交易的影响 ---")
    orig_all_avg = df_detail['actual_return_pct'].mean()
    sim_all_avg = sim_returns.mean()
    orig_all_total = df_detail['actual_return_pct'].sum()
    sim_all_total = sim_returns.sum()

    print(f"  原平均收益: {orig_all_avg:.2f}%")
    print(f"  模拟平均收益: {sim_all_avg:.2f}%")
    print(f"  改善: {sim_all_avg - orig_all_avg:.2f}% (平均) / {sim_all_total - orig_all_total:.1f}% (总)")

    # 5.4 风险分析：延迟止损是否导致更大的单笔亏损
    print(f"\n--- 风险分析: 延迟止损后是否有更大的单笔亏损 ---")
    for _, r in affected.iterrows():
        if r['sim_return_pct'] < r['actual_return_pct']:
            print(f"  WARNING: {r['stock_code']:10s} | orig={r['actual_return_pct']:+.1f}% -> sim={r['sim_return_pct']:+.1f}% "
                  f"(恶化{r['sim_return_pct'] - r['actual_return_pct']:+.1f}%) | {r['sim_change_reason']}")

    worsening = affected[affected['sim_return_pct'] < affected['actual_return_pct']]
    improving = affected[affected['sim_return_pct'] > affected['actual_return_pct']]
    print(f"\n  改善的交易: {len(improving)} 笔, 平均改善: {improving['sim_return_pct'].sub(improving['actual_return_pct']).mean():.2f}%")
    if len(worsening) > 0:
        print(f"  恶化的交易: {len(worsening)} 笔, 平均恶化: {worst_diff := worsening['sim_return_pct'].sub(worsening['actual_return_pct']).mean():.2f}%")
        print(f"  最大恶化: {worsening['sim_return_pct'].sub(worsening['actual_return_pct']).min():.2f}%")
    else:
        print(f"  恶化的交易: 0 笔")

    # 5.5 对盈利交易的影响
    print(f"\n--- 对盈利交易(return>0%)的影响 ---")
    profit_trades = df_detail[df_detail['actual_return_pct'] > 0]
    profit_affected = profit_trades[profit_trades['sim_changed'] == True]
    print(f"  盈利交易数: {len(profit_trades)}")
    print(f"  受影响的盈利交易: {len(profit_affected)}")
    if len(profit_affected) > 0:
        for _, r in profit_affected.iterrows():
            print(f"    {r['stock_code']:10s} | orig={r['actual_return_pct']:+.1f}% -> sim={r['sim_return_pct']:+.1f}% | {r['sim_change_reason']}")

# ============================================================
# Step 6: 对比现有规则与新规则
# ============================================================
print("\n" + "=" * 80)
print("Step 6: 现有规则 vs 方案C 规则对比")
print("=" * 80)

print("""
现有规则:
  T+2 09:35 收益<=-4% → 无条件硬止损
  T+2 10:00 收益<=-2% → 软止损(默认关闭)
  T+2 14:55 收益<=0% → 保本出(第二层)
  累计收益<=-5% → 硬止损

方案C:
  T+2 09:35 收益<=-4% → 不立即止损, 延迟到10:00判断:
    10:00收益<=-5% → 止损(比原-4%多容忍1%)
    10:00跌涨量比>2.5 → 止损(放量下杀)
    10:00未收复T+1低 且 vs T+1收盘<-4% → 止损(真恶化)
    否则 → 继续持有, 到T+2 14:55保本出
  T+2 14:55 收益<=0% → 保本出(不变)
  累计收益<=-5% → 硬止损(不变)

冲突分析:
  - 与T+2 09:35硬止损冲突: 直接替换, 09:35不再无条件止损
  - 与T+2 14:55保本出: 不冲突, 延迟后仍受保本出保护
  - 与累计收益-5%硬止损: 不冲突, 10:00如果<=-5%仍止损
  - 与利润保护/小盈利失败退出: 不冲突, 这些是更高层规则
  - 关键风险: 09:35到10:00之间(25分钟), 股价可能继续下跌
""")

# 检查09:35到10:00之间的额外风险
print("--- 09:35到10:00额外风险分析 ---")
t2_0935_stops = df_detail[df_detail['exit_type'] == 'T2_0935_hard']
for _, r in t2_0935_stops.iterrows():
    ret_0935 = r.get('T2_0935_return_pct', None)
    ret_1000 = r.get('T2_1000_return_pct', None)
    if pd.notna(ret_0935) and pd.notna(ret_1000):
        extra_loss = ret_1000 - ret_0935
        print(f"  {r['stock_code']:10s} | 09:35={ret_0935:+.1f}% -> 10:00={ret_1000:+.1f}% | "
              f"额外亏损={extra_loss:+.1f}% | T3={r.get('T3_return_pct', 'N/A')}")

# ============================================================
# Step 7: 保存
# ============================================================
print("\n" + "=" * 80)
print("Step 7: 保存结果")
print("=" * 80)

output_dir = BASE_DIR / "results"
df_detail.to_csv(output_dir / "risk_control_modification_simulation.csv", index=False, encoding='utf-8-sig')
print("已保存: risk_control_modification_simulation.csv")

print("\n分析完成!")
