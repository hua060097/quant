"""
分析T+2 14:55保本出和T+3 09:35保本出交易的日内走势特征
目标：找出能区分"继续持有能回本" vs "持有会亏更多"的盘中信号
"""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
MINUTE_DIR = BASE_DIR.parent / "local_data" / "minute"
DAILY_DIR = BASE_DIR.parent / "local_data" / "daily"
FACTOR_FILE = BASE_DIR / "factor_lab_data" / "factors.parquet"
LOG_FILE = BASE_DIR / "stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log"

# ============================================================
# Step 0: 提取交易 + 加载交易日历 + 日线数据
# ============================================================
print("加载数据...")
lines = open(LOG_FILE, 'r', encoding='utf-8').readlines()

buys = []
current_stock = current_buy_date = current_buy_price = None
for line in lines:
    m_stock = re.search(r'分层并行仓位生效: (\S+) 使用', line)
    if m_stock: current_stock = m_stock.group(1)
    m_tday = re.search(r'并行持仓买入:.*?T日=(\d{4}-\d{2}-\d{2})', line)
    if m_tday: current_buy_date = m_tday.group(1)
    m_buyp = re.search(r'实际买入价[：:]?\s*([\d.]+)', line)
    if m_buyp: current_buy_price = float(m_buyp.group(1))
    m_confirm = re.search(r'买入后持仓状态', line)
    if m_confirm and current_stock:
        buys.append({'stock_code': current_stock, 'buy_date': current_buy_date, 'buy_price': current_buy_price})
        current_stock = current_buy_date = current_buy_price = None

sells = []
for line in lines:
    m = re.search(r'卖出: (\S+), 价格: ([\d.]+), 收益: -?[\d.]+ \((-?[\d.]+)%\), 原因: (.+)', line)
    if m:
        sells.append({'stock_code': m.group(1), 'sell_price': float(m.group(2)),
                       'return_pct': float(m.group(3)), 'exit_reason': m.group(4).strip()})

buy_used = [False] * len(buys)
trades = []
for sell in sells:
    for j, buy in enumerate(buys):
        if not buy_used[j] and buy['stock_code'] == sell['stock_code']:
            trades.append({**buy, **sell})
            buy_used[j] = True
            break

df = pd.DataFrame(trades)

def classify_exit(reason):
    if pd.isna(reason): return 'unknown'
    if 'T+2 09:35失败止损' in reason: return 'T2_0935_hard'
    if 'T+2 10:00继续走坏' in reason: return 'T2_1000_soft'
    if 'T+2收盘仍未转正' in reason: return 'T2_close_breakeven'
    if 'T+3 09:35仍未转正' in reason: return 'T3_0935_breakeven'
    if '累计收益低于' in reason: return 'cumulative_stop'
    if '利润保护' in reason: return 'profit_protect'
    if '小盈利后延续失败' in reason: return 'small_profit_fail'
    if '到期' in reason: return 'expiry'
    return 'other'

df['exit_type'] = df['exit_reason'].apply(classify_exit)

# 交易日历
df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# 日线数据
needed_months = set()
for _, t in df.iterrows():
    bd = pd.Timestamp(str(t['buy_date'])[:10])
    if bd in date_to_idx:
        idx = date_to_idx[bd]
        for offset in range(5):
            if idx + offset < len(all_dates):
                ts = pd.Timestamp(all_dates[idx + offset])
                needed_months.add(f'{ts.year}{ts.month:02d}')

daily_frames = []
for month_dir in sorted(DAILY_DIR.iterdir()):
    if month_dir.is_dir() and month_dir.name in needed_months:
        for f in month_dir.glob('*.parquet'):
            try:
                df_d = pd.read_parquet(f, columns=['stock_code', 'date', 'open', 'high', 'low', 'close'])
                daily_frames.append(df_d)
            except:
                pass
daily_data = pd.concat(daily_frames, ignore_index=True)
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['lookup'] = daily_data['stock_code'] + '_' + daily_data['date'].dt.strftime('%Y-%m-%d')
daily_dict = daily_data.set_index('lookup').to_dict('index')

# 给每笔交易加T+2/T+3日线收盘价
def get_trading_date(buy_date_str, offset):
    bd = pd.Timestamp(buy_date_str[:10])
    if bd not in date_to_idx: return None
    idx = date_to_idx[bd]
    return all_dates[idx + offset] if idx + offset < len(all_dates) else None

for i, row in df.iterrows():
    stock = row['stock_code']
    bd = str(row['buy_date'])[:10]
    for offset, label in [(1, 'T1'), (2, 'T2'), (3, 'T3')]:
        td = get_trading_date(bd, offset)
        if td:
            key = f"{stock}_{pd.Timestamp(td).strftime('%Y-%m-%d')}"
            dd = daily_dict.get(key, {})
            if dd:
                df.at[i, f'{label}_open'] = dd.get('open')
                df.at[i, f'{label}_high'] = dd.get('high')
                df.at[i, f'{label}_low'] = dd.get('low')
                df.at[i, f'{label}_close'] = dd.get('close')
                df.at[i, f'{label}_return_pct'] = (dd['close'] / row['buy_price'] - 1) * 100 if row['buy_price'] else None

# ============================================================
# Step 1: 分析T+2 14:55保本出的8笔交易
# ============================================================
print("\n" + "=" * 80)
print("Step 1: T+2 14:55保本出交易 - 日内走势分析")
print("=" * 80)

t2_be = df[df['exit_type'] == 'T2_close_breakeven'].copy()
# 添加T+3收益标签
t2_be['hold_to_t3'] = t2_be.apply(
    lambda r: 'recovered' if pd.notna(r.get('T3_return_pct')) and r['T3_return_pct'] > r['return_pct']
    else ('worsened' if pd.notna(r.get('T3_return_pct')) and r['T3_return_pct'] < r['return_pct'] else 'same'), axis=1)

print(f"T+2 14:55保本出交易: {len(t2_be)} 笔")
for _, r in t2_be.iterrows():
    t3_ret = r.get('T3_return_pct', None)
    t3_str = f"{t3_ret:+.1f}%" if pd.notna(t3_ret) else "N/A"
    print(f"  {r['stock_code']:10s} | actual={r['return_pct']:+.1f}% | T+3收盘={t3_str} | {r['hold_to_t3']}")

# 加载T+2分钟数据，分析日内走势
def load_minute(stock, date_str):
    ts = pd.Timestamp(date_str)
    ym = f"{ts.year}{ts.month:02d}"
    f = MINUTE_DIR / ym / "minute_data.parquet"
    if not f.exists(): return None
    try:
        df_m = pd.read_parquet(f, columns=['stock_code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df_m['datetime'] = pd.to_datetime(df_m['datetime'])
        mask = (df_m['stock_code'] == stock) & (df_m['datetime'].dt.date == ts.date())
        return df_m[mask].sort_values('datetime').reset_index(drop=True)
    except:
        return None

print("\n--- T+2日内走势特征 ---")
print(f"{'Stock':10s} | {'14:55ret':>8s} | {'T3_ret':>7s} | {'AM_low':>7s} | {'PM_low':>7s} | {'PM_vs_AM':>7s} | {'14:00ret':>8s} | {'14:55ret':>8s} | {'bounce':>7s} | {'AM_ret':>7s} | {'PM_ret':>7s} | {'VWAP':>7s} | {'vs_VWAP':>7s} | Verdict")
print("-" * 160)

for _, r in t2_be.iterrows():
    stock = r['stock_code']
    bd = str(r['buy_date'])[:10]
    bp = r['buy_price']
    t2_date = get_trading_date(bd, 2)
    if not t2_date: continue

    minute = load_minute(stock, str(t2_date)[:10])
    if minute is None or len(minute) == 0:
        print(f"  {stock:10s} | 分钟数据缺失")
        continue

    # 上午(09:35-11:30) vs 下午(13:00-14:55)
    am = minute[minute['datetime'].dt.hour < 12]
    pm = minute[minute['datetime'].dt.hour >= 13]

    am_low = am['low'].min() if len(am) > 0 else None
    pm_low = pm['low'].min() if len(pm) > 0 else None
    am_high = am['high'].max() if len(am) > 0 else None
    pm_high = pm['high'].max() if len(pm) > 0 else None

    # AM/PM相对买入价收益
    am_close_ret = (am.iloc[-1]['close'] / bp - 1) * 100 if len(am) > 0 else None
    pm_open_ret = (pm.iloc[0]['open'] / bp - 1) * 100 if len(pm) > 0 else None
    pm_close_ret = (pm.iloc[-1]['close'] / bp - 1) * 100 if len(pm) > 0 else None

    # 下午是否创新低（相对上午）
    pm_vs_am = "PM新低" if pm_low and am_low and pm_low < am_low else "PM>AM低"

    # 下午走势：14:00 vs 14:55
    pm_14 = minute[(minute['datetime'].dt.hour == 14) & (minute['datetime'].dt.minute == 0)]
    ret_14 = (pm_14.iloc[0]['close'] / bp - 1) * 100 if len(pm_14) > 0 else None
    ret_1455 = (pm.iloc[-1]['close'] / bp - 1) * 100 if len(pm) > 0 else None

    # 日内最低点后反弹
    min_idx = minute['low'].idxmin()
    min_bar = minute.loc[min_idx]
    after_low = minute[minute.index > min_idx]
    bounce = (after_low.iloc[-1]['close'] / min_bar['low'] - 1) * 100 if len(after_low) > 0 else 0

    # 全天VWAP
    total_amt = minute['amount'].sum()
    total_vol = minute['volume'].sum()
    vwap = total_amt / total_vol / 100 if total_vol > 0 else None
    close_vs_vwap = (minute.iloc[-1]['close'] / vwap - 1) * 100 if vwap else None

    t3_ret = r.get('T3_return_pct', None)

    # 判断：下午走势是否走弱
    pm_trend = "PM走弱" if pm_close_ret and pm_open_ret and pm_close_ret < pm_open_ret else "PM企稳"

    def fmt(val, suffix="%"):
        return f"{val:+.1f}{suffix}" if pd.notna(val) else "N/A"

    print(f"  {stock:10s} | {fmt(r['return_pct'])} | {fmt(t3_ret)} | "
          f"{fmt((am_low/bp-1)*100) if am_low and bp else 'N/A':>7s} | "
          f"{fmt((pm_low/bp-1)*100) if pm_low and bp else 'N/A':>7s} | "
          f"{pm_vs_am:>7s} | "
          f"{fmt(ret_14)} | {fmt(ret_1455)} | "
          f"{fmt(bounce)} | {fmt(am_close_ret)} | {fmt(pm_close_ret)} | "
          f"{fmt(vwap, '') if vwap else 'N/A':>7s} | {fmt(close_vs_vwap)} | "
          f"{pm_trend}")

# ============================================================
# Step 2: 也分析T+3 09:35保本出的4笔
# ============================================================
print("\n" + "=" * 80)
print("Step 2: T+3 09:35保本出交易 - 日内走势分析")
print("=" * 80)

t3_be = df[df['exit_type'] == 'T3_0935_breakeven'].copy()
t3_be['hold_to_t3_close'] = t3_be.apply(
    lambda r: 'better_wait' if pd.notna(r.get('T3_return_pct')) and r['T3_return_pct'] > r['return_pct']
    else 'better_sell', axis=1)

print(f"T+3 09:35保本出交易: {len(t3_be)} 笔")
for _, r in t3_be.iterrows():
    t3_ret = r.get('T3_return_pct', None)
    t3_str = f"{t3_ret:+.1f}%" if pd.notna(t3_ret) else "N/A"
    print(f"  {r['stock_code']:10s} | actual={r['return_pct']:+.1f}% | T+3收盘={t3_str} | {r['hold_to_t3_close']}")

# T+2和T+3的分钟数据
print("\n--- T+2/T+3日内走势特征 ---")
for _, r in t3_be.iterrows():
    stock = r['stock_code']
    bd = str(r['buy_date'])[:10]
    bp = r['buy_price']

    for day_offset in [2, 3]:
        label = f"T+{day_offset}"
        td = get_trading_date(bd, day_offset)
        if not td: continue

        minute = load_minute(stock, str(td)[:10])
        if minute is None or len(minute) == 0:
            print(f"  {stock} {label}: 分钟数据缺失")
            continue

        am = minute[minute['datetime'].dt.hour < 12]
        pm = minute[minute['datetime'].dt.hour >= 13]

        # 整体走势
        open_ret = (minute.iloc[0]['open'] / bp - 1) * 100
        close_ret = (minute.iloc[-1]['close'] / bp - 1) * 100
        low_ret = (minute['low'].min() / bp - 1) * 100

        # 上午/下午分时
        am_close_ret = (am.iloc[-1]['close'] / bp - 1) * 100 if len(am) > 0 else None
        pm_open_ret = (pm.iloc[0]['open'] / bp - 1) * 100 if len(pm) > 0 else None
        pm_close_ret = (pm.iloc[-1]['close'] / bp - 1) * 100 if len(pm) > 0 else None

        # 09:35第一根
        first_ret = (minute.iloc[0]['close'] / bp - 1) * 100

        # 趋势
        if len(am) > 0 and len(pm) > 0:
            trend = "AM强PM弱" if am_close_ret and pm_close_ret and am_close_ret > pm_close_ret else "AM弱PM强" if am_close_ret and pm_close_ret else "N/A"
        else:
            trend = "N/A"

        def fmt(val):
            return f"{val:+.1f}%" if pd.notna(val) else "N/A"

        print(f"  {stock:10s} {label}: open={fmt(open_ret)} 09:35={fmt(first_ret)} close={fmt(close_ret)} low={fmt(low_ret)} | AM_close={fmt(am_close_ret)} PM_open={fmt(pm_open_ret)} PM_close={fmt(pm_close_ret)} | {trend}")

# ============================================================
# Step 3: 对比分析 - "如果等T+3收盘"的结果
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 如果T+2/T+3保本出改为条件放行，结果会怎样？")
print("=" * 80)

# 合并两组
breakeven_trades = pd.concat([t2_be, t3_be], ignore_index=True)

print(f"\n{'Stock':10s} | {'Exit':15s} | {'Actual':>7s} | {'T3_close':>8s} | {'Diff':>6s} | 如果放行到T+3收盘")
print("-" * 80)
for _, r in breakeven_trades.iterrows():
    t3_ret = r.get('T3_return_pct', None)
    actual = r['return_pct']
    diff = (t3_ret - actual) if pd.notna(t3_ret) else None

    if pd.notna(diff):
        verdict = "赚" if diff > 0.5 else "亏" if diff < -0.5 else "持平"
    else:
        verdict = "N/A"

    t3_str = f"{t3_ret:+.1f}%" if pd.notna(t3_ret) else "N/A"
    diff_str = f"{diff:+.1f}%" if pd.notna(diff) else "N/A"

    print(f"  {r['stock_code']:10s} | {r['exit_type']:15s} | {actual:+.1f}% | {t3_str:>8s} | {diff_str:>6s} | {verdict}")

# ============================================================
# Step 4: T+2 14:55放行条件模拟
# ============================================================
print("\n" + "=" * 80)
print("Step 4: T+2 14:55保本出改为条件放行的模拟")
print("=" * 80)

# 方案：T+2 14:55如果收益在-2%~0%之间，且下午没有继续创新低，则放行到T+3 14:55
# 否则维持T+2卖出

print("\n模拟条件: T+2 14:55 收益在[-2%, 0%] 且 PM低>=AM低 → 放行到T+3收盘")
print("          否则 → 维持T+2卖出")

total_diff = 0
for _, r in t2_be.iterrows():
    stock = r['stock_code']
    bd = str(r['buy_date'])[:10]
    bp = r['buy_price']
    actual = r['return_pct']
    t3_ret = r.get('T3_return_pct', None)

    t2_date = get_trading_date(bd, 2)
    if not t2_date: continue

    minute = load_minute(stock, str(t2_date)[:10])
    if minute is None or len(minute) == 0: continue

    am = minute[minute['datetime'].dt.hour < 12]
    pm = minute[minute['datetime'].dt.hour >= 13]

    # 条件1: 14:55收益在[-2%, 0%]
    ret_1455 = actual / 100  # 已经是百分比
    cond1 = -2 <= actual <= 0

    # 条件2: PM低点 >= AM低点
    am_low = am['low'].min() if len(am) > 0 else float('inf')
    pm_low = pm['low'].min() if len(pm) > 0 else float('inf')
    cond2 = pm_low >= am_low

    should_release = cond1 and cond2

    if should_release and pd.notna(t3_ret):
        sim_ret = t3_ret
        diff = t3_ret - actual
    else:
        sim_ret = actual
        diff = 0

    total_diff += diff

    cond_str = f"cond1={'Y' if cond1 else 'N'} cond2={'Y' if cond2 else 'N'}"
    result_str = f"放行→T+3={sim_ret:+.1f}%" if should_release else f"维持T+2卖出"
    diff_str = f"(diff={diff:+.1f}%)" if diff != 0 else ""

    print(f"  {stock:10s} | actual={actual:+.1f}% | {cond_str} | {'PASS' if should_release else 'BLOCK'} | {result_str} {diff_str}")

print(f"\n总效果: {total_diff:+.1f}% (正=改善, 负=恶化)")

# ============================================================
# Step 5: 更宽松的放行条件模拟
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 更宽松的放行条件模拟")
print("=" * 80)

print("\n方案A: 收益在[-2%, 0%] 且 PM低>=AM低 → 放行")
print("方案B: 收益在[-2%, 0%] 且 14:00之后收益>=14:00收益(下午没继续跌) → 放行")
print("方案C: 收益在[-2%, 0%] 且 收盘价>VWAP → 放行")

for plan_name, plan_desc in [("A", "PM低>=AM低"), ("B", "下午未继续跌"), ("C", "收盘价>VWAP")]:
    total_diff = 0
    n_released = 0
    details = []

    for _, r in t2_be.iterrows():
        stock = r['stock_code']
        bd = str(r['buy_date'])[:10]
        bp = r['buy_price']
        actual = r['return_pct']
        t3_ret = r.get('T3_return_pct', None)

        t2_date = get_trading_date(bd, 2)
        if not t2_date: continue

        minute = load_minute(stock, str(t2_date)[:10])
        if minute is None or len(minute) == 0: continue

        am = minute[minute['datetime'].dt.hour < 12]
        pm = minute[minute['datetime'].dt.hour >= 13]

        # 公共条件: 收益在[-2%, 0%]
        cond_common = -2 <= actual <= 0

        if plan_name == "A":
            am_low = am['low'].min() if len(am) > 0 else float('inf')
            pm_low = pm['low'].min() if len(pm) > 0 else float('inf')
            extra_cond = pm_low >= am_low
        elif plan_name == "B":
            # 14:00收益 vs 14:55收益
            pm_14 = minute[(minute['datetime'].dt.hour == 14) & (minute['datetime'].dt.minute == 0)]
            ret_1400 = (pm_14.iloc[0]['close'] / bp - 1) * 100 if len(pm_14) > 0 else None
            extra_cond = ret_1400 is not None and actual >= ret_1400  # 14:55收益>=14:00收益
        elif plan_name == "C":
            total_amt = minute['amount'].sum()
            total_vol = minute['volume'].sum()
            vwap = total_amt / total_vol / 100 if total_vol > 0 else None
            extra_cond = vwap is not None and minute.iloc[-1]['close'] > vwap
        else:
            extra_cond = False

        should_release = cond_common and extra_cond

        if should_release and pd.notna(t3_ret):
            diff = t3_ret - actual
            n_released += 1
        else:
            diff = 0

        total_diff += diff
        details.append((stock, actual, t3_ret, should_release, diff))

    print(f"\n方案{plan_name} ({plan_desc}): 释放{n_released}笔, 总效果={total_diff:+.1f}%")
    for stock, actual, t3_ret, released, diff in details:
        mark = "RELEASE" if released else "BLOCK"
        t3_str = f"{t3_ret:+.1f}%" if pd.notna(t3_ret) else "N/A"
        diff_str = f" diff={diff:+.1f}%" if diff != 0 else ""
        print(f"    {stock:10s} | actual={actual:+.1f}% T3={t3_str} | {mark}{diff_str}")

# ============================================================
# Step 6: 对T+3 09:35保本出也做类似模拟
# ============================================================
print("\n" + "=" * 80)
print("Step 6: T+3 09:35保本出改为条件放行")
print("=" * 80)

print("\n方案: T+3 09:35 收益在[-2%, 0%] → 放行到T+3收盘, 否则维持卖出")

total_diff_t3 = 0
for _, r in t3_be.iterrows():
    stock = r['stock_code']
    actual = r['return_pct']
    t3_ret = r.get('T3_return_pct', None)

    cond = -2 <= actual <= 0

    if cond and pd.notna(t3_ret):
        diff = t3_ret - actual
    else:
        diff = 0

    total_diff_t3 += diff
    mark = "RELEASE" if cond else "BLOCK"
    t3_str = f"{t3_ret:+.1f}%" if pd.notna(t3_ret) else "N/A"
    diff_str = f" diff={diff:+.1f}%" if diff != 0 else ""
    print(f"  {stock:10s} | actual={actual:+.1f}% T3_close={t3_str} | {mark}{diff_str}")

print(f"\nT+3保本出总效果: {total_diff_t3:+.1f}%")

# ============================================================
# Step 7: 总结
# ============================================================
print("\n" + "=" * 80)
print("Step 7: 总结与建议")
print("=" * 80)

print("""
当前保本出规则的问题:
  T+2 14:55: 收益<=0% → 无条件卖出
  T+3 09:35: 收益<=0% → 无条件卖出

问题: 阈值0%太绝对, 微亏(如-0.5%)就卖, 但有些微亏票T+3能大幅回本

修改建议:
  1. T+2 14:55保本出: 收益在[-2%, 0%]时, 不立即卖出,
     而是检查下午走势是否企稳(PM低>=AM低 或 收盘>VWAP)
     如果企稳 → 放行到T+3收盘
     如果仍在走弱 → 维持T+2卖出
  2. T+3 09:35保本出: 收益在[-2%, 0%]时, 放行到T+3收盘
     收益<-2% → 维持卖出

风险:
  - 放行后T+3可能继续下跌(如300955.SZ从-0.5%跌到-6.7%)
  - 需要更精细的信号来区分"即将修复"和"继续恶化"
  - 样本量小(8+4=12笔), 结论需要更多验证
""")
