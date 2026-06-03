#!/usr/bin/env python3
"""Apply recommended rules to 50.log actual trades."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"

# --- Parse 50.log trades ---
with open(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子50.log", 'r', encoding='utf-8') as f:
    lines = f.readlines()

trades = []
buy_dates = {}; cd = None
for line in lines:
    dm = re.search(r'T日=(\d{4}-\d{2}-\d{2})', line)
    if dm: cd = dm.group(1)
    sm = re.search(r'📅 ([\d.A-Z]+) T\+1买入逻辑', line)
    if sm: buy_dates[sm.group(1)] = cd
    sm = re.search(r'💰 卖出:\s([\d.A-Z]+),\s价格:\s([-\d.]+),\s收益:\s([-\d.]+)\s\(([-\d.]+)%\),\s原因:\s(.+)', line)
    if sm:
        trades.append({'code':sm.group(1),'ret':float(sm.group(4)),'buy_date':buy_dates.get(sm.group(1),'?')})

trades_df = pd.DataFrame(trades)
trades_df = trades_df[trades_df['buy_date'] != '?']
print('50.log trades with dates: {}'.format(len(trades_df)))

# Merge factor values
factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
factors['date_str'] = factors['date'].dt.strftime('%Y-%m-%d')
needed = ['runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
          'sector_momentum_accel_rank','sector_leadership_persistence_v3',
          'top10_10d_strength_level_v3']
trades_df = trades_df.merge(factors[['date_str','stock_code']+needed],
    left_on=['buy_date','code'], right_on=['date_str','stock_code'], how='left')

def classify(r):
    if r >= 5: return '大涨'
    if r >= 0: return '小涨'
    if r >= -5: return '小跌'
    return '大跌'
trades_df['cls'] = trades_df['ret'].apply(classify)
n_all = len(trades_df)
n_bg = int((trades_df.cls=='大涨').sum())
n_sg = int((trades_df.cls=='小涨').sum())
n_sl = int((trades_df.cls=='小跌').sum())
n_bl = int((trades_df.cls=='大跌').sum())
print('50.log trades: {} (大涨={}, 小涨={}, 小跌={}, 大跌={})'.format(n_all, n_bg, n_sg, n_sl, n_bl))

def analyze(name, cond):
    blocked = trades_df[cond.fillna(False)]
    n = len(blocked)
    if n == 0: return None
    d = blocked['cls'].value_counts()
    return {
        'name':name, 'hits':n,
        'bg':d.get('大涨',0), 'sg':d.get('小涨',0), 'sl':d.get('小跌',0), 'bl':d.get('大跌',0),
        'bmean':blocked['ret'].mean(), 'bmed':blocked['ret'].median(),
        'pmean':trades_df[~cond.fillna(False)]['ret'].mean(),
        'bg_pct':d.get('大涨',0)/n_bg*100 if n_bg else 0,
        'bl_pct':d.get('大跌',0)/n_bl*100 if n_bl else 0,
    }

# ================================================================
# R1: runup>0.8 & cpv <= X
# ================================================================
print('\n' + '#' * 100)
print('# R1: runup_ma5_exhaustion > 0.8 AND cpv_factor <= X')
print('# Base (runup>0.8): {} trades in 50.log'.format((trades_df['runup_ma5_exhaustion']>0.8).sum()))
print('#' * 100)
base1 = trades_df['runup_ma5_exhaustion'] > 0.8
print('{:>12s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s}'.format(
    'cpv<=X','Hits','大涨','小涨','小跌','大跌','BlkAvg%','BlkMed%','PasAvg%','大涨误杀%','大跌拦截%'))
for t in np.arange(-2.0, 2.1, 0.2):
    r = analyze('cpv<={:.1f}'.format(t), base1 & (trades_df['cpv_factor'] <= t))
    if r and r['hits'] > 0:
        note = ''
        if abs(t-(-0.4))<0.01: note = ' <-- 推荐'
        if abs(t-0.0)<0.01: note = ' <-- 当前'
        print('{:12s} {:5d} {:5d} {:5d} {:5d} {:5d} {:8.1f}% {:8.1f}% {:8.1f}% {:9.1f}% {:9.1f}%{}'.format(
            'cpv<={:.1f}'.format(t), r['hits'], r['bg'], r['sg'], r['sl'], r['bl'],
            r['bmean'], r['bmed'], r['pmean'], r['bg_pct'], r['bl_pct'], note))

# Old rule
r_old = analyze('OLD(top10>=0.5)', base1 & (trades_df['top10_10d_strength_level_v3']>=0.5))
if r_old:
    print('{:12s} {:5d} {:5d} {:5d} {:5d} {:5d} {:8.1f}% {:8.1f}% {:8.1f}% {:9.1f}% {:9.1f}% {}'.format(
        'OLD top10', r_old['hits'], r_old['bg'], r_old['sg'], r_old['sl'], r_old['bl'],
        r_old['bmean'], r_old['bmed'], r_old['pmean'], r_old['bg_pct'], r_old['bl_pct'], '<-- 旧规则'))

# ================================================================
# R2: blowoff>=X & sector<=Y
# ================================================================
print('\n' + '#' * 100)
print('# R2: blowoff >= X AND sector_accel <= Y')
print('#' * 100)
print('{:>20s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>9s} {:>9s} {:>9s}'.format(
    'Rule','Hits','大涨','小涨','小跌','大跌','BlkAvg%','BlkMed%','PasAvg%'))
for bf in [2.0, 2.5, 3.0, 3.5]:
    for sa in [0.20, 0.25, 0.30, 0.35, 0.41]:
        cond = (trades_df['prev_day_blowoff_risk'] >= bf) & (trades_df['sector_momentum_accel_rank'] <= sa)
        r = analyze('b>={:.0f}&s<={:.2f}'.format(bf,sa), cond)
        if r and r['hits'] > 0:
            note = ' <-- 当前' if abs(bf-3.5)<0.01 and abs(sa-0.30)<0.01 else ''
            print('{:20s} {:5d} {:5d} {:5d} {:5d} {:5d} {:8.1f}% {:8.1f}% {:8.1f}%{}'.format(
                'b>={:.0f}&s<={:.2f}'.format(bf,sa), r['hits'], r['bg'], r['sg'], r['sl'], r['bl'],
                r['bmean'], r['bmed'], r['pmean'], note))

# ================================================================
# R3: sector_leadership <= X
# ================================================================
print('\n' + '#' * 100)
print('# R3: sector_leadership_persistence_v3 <= X')
print('#    sl on 50.log: mean={:.4f}, std={:.4f}, P05={:.4f}, P10={:.4f}, P20={:.4f}'.format(
    trades_df['sector_leadership_persistence_v3'].mean(),
    trades_df['sector_leadership_persistence_v3'].std(),
    trades_df['sector_leadership_persistence_v3'].quantile(0.05),
    trades_df['sector_leadership_persistence_v3'].quantile(0.10),
    trades_df['sector_leadership_persistence_v3'].quantile(0.20)))
print('#' * 100)
print('{:>8s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s} {:>12s}'.format(
    'sl<=X','Hits','大涨','小涨','小跌','大跌','BlkAvg%','BlkMed%','PasAvg%','大涨误杀%','大跌拦截%','分位'))
for t in np.arange(-0.35, 0.05, 0.02):
    r = analyze('sl<={:.2f}'.format(t), trades_df['sector_leadership_persistence_v3'] <= t)
    if r and r['hits'] > 0:
        pct_all = (trades_df['sector_leadership_persistence_v3'] <= t).mean() * 100
        note = ''
        if abs(t-(-0.23))<0.01: note = ' P05(推荐)'
        if abs(t-(-0.18))<0.01: note = ' P10(当前)'
        if abs(t-(-0.20))<0.01: note = ' P08'
        print('{:8.2f} {:5d} {:5d} {:5d} {:5d} {:5d} {:8.1f}% {:8.1f}% {:8.1f}% {:9.1f}% {:9.1f}% {:>12s}'.format(
            t, r['hits'], r['bg'], r['sg'], r['sl'], r['bl'],
            r['bmean'], r['bmed'], r['pmean'], r['bg_pct'], r['bl_pct'],
            'P{:02.0f}'.format(pct_all) + note))

# ================================================================
# UNION comparison
# ================================================================
print('\n' + '#' * 100)
print('# UNION: 当前 vs 推荐 on 50.log')
print('#' * 100)

r1_now = (trades_df['runup_ma5_exhaustion']>0.8) & (trades_df['cpv_factor']<=0.0)
r2 = (trades_df['prev_day_blowoff_risk']>=3.5) & (trades_df['sector_momentum_accel_rank']<=0.30)
r3_now = trades_df['sector_leadership_persistence_v3'] <= -0.18
r1_new = (trades_df['runup_ma5_exhaustion']>0.8) & (trades_df['cpv_factor']<=-0.4)
r3_new = trades_df['sector_leadership_persistence_v3'] <= -0.23

for name, union, r1, r3 in [
    ('当前(cpv<=0.0, sl<=-0.18)', r1_now|r2|r3_now, r1_now, r3_now),
    ('推荐(cpv<=-0.4, sl<=-0.23)', r1_new|r2|r3_new, r1_new, r3_new),
]:
    r = analyze(name, union.fillna(False))
    if r:
        print('')
        print('  {}: hits={}, bg={}, sg={}, sl={}, bl={}, blk_avg={:+.1f}%, pas_avg={:+.1f}%'.format(
            name, r['hits'], r['bg'], r['sg'], r['sl'], r['bl'], r['bmean'], r['pmean']))
        print('    各规则: R1={}, R2={}, R3={}'.format(r1.sum(), r2.sum(), r3.sum()))
        blocked = trades_df[union.fillna(False)]
        if r['bg'] > 0:
            print('    误杀大涨:')
            for _, x in blocked[blocked['cls']=='大涨'].iterrows():
                print('      {} {}: {:+.1f}%'.format(x['code'], x['buy_date'], x['ret']))
        if r['bl'] > 0:
            print('    拦截大跌:')
            for _, x in blocked[blocked['cls']=='大跌'].iterrows():
                print('      {} {}: {:+.1f}%'.format(x['code'], x['buy_date'], x['ret']))
