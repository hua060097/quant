#!/usr/bin/env python3
"""Full R1 scan: vary both runup AND cpv thresholds on both 50 and 162 logs."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"
factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
factors['date_str'] = factors['date'].dt.strftime('%Y-%m-%d')
needed = ['runup_ma5_exhaustion', 'cpv_factor']

def parse_log(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    trades = []; buy_dates = {}; cd = None
    for line in lines:
        dm = re.search(r'T日=(\d{4}-\d{2}-\d{2})', line)
        if dm: cd = dm.group(1)
        sm = re.search(r'📅 ([\d.A-Z]+) T\+1买入逻辑', line)
        if sm: buy_dates[sm.group(1)] = cd
        sm = re.search(r'💰 卖出:\s([\d.A-Z]+),\s价格:\s([-\d.]+),\s收益:\s([-\d.]+)\s\(([-\d.]+)%\),\s原因:\s(.+)', line)
        if sm:
            trades.append({'code':sm.group(1),'ret':float(sm.group(4)),'buy_date':buy_dates.get(sm.group(1),'?')})
    df = pd.DataFrame(trades)
    df = df[df['buy_date'] != '?']
    df = df.merge(factors[['date_str','stock_code']+needed],
        left_on=['buy_date','code'], right_on=['date_str','stock_code'], how='left')
    def cls(r):
        if r>=5: return '大涨'
        if r>=0: return '小涨'
        if r>=-5: return '小跌'
        return '大跌'
    df['cls'] = df['ret'].apply(cls)
    return df

d50 = parse_log(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子50.log")
d162 = parse_log(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log")

# runup distribution
print('=== runup_ma5_exhaustion distribution ===')
for label, df in [('50.log',d50), ('162',d162)]:
    s = df['runup_ma5_exhaustion']
    print('{}: mean={:.3f}, P25={:.3f}, P50={:.3f}, P75={:.3f}, P90={:.3f}, P95={:.3f}, max={:.3f}'.format(
        label, s.mean(), s.quantile(0.25), s.quantile(0.50), s.quantile(0.75),
        s.quantile(0.90), s.quantile(0.95), s.max()))

# Count trades above various runup thresholds
print('\n=== Trades above runup thresholds ===')
for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5]:
    n50 = int((d50['runup_ma5_exhaustion']>t).sum())
    n162 = int((d162['runup_ma5_exhaustion']>t).sum())
    d50b = d50[d50['runup_ma5_exhaustion']>t]
    d162b = d162[d162['runup_ma5_exhaustion']>t]
    bg50 = int((d50b['cls']=='大涨').sum()); bl50 = int((d50b['cls']=='大跌').sum())
    bg162 = int((d162b['cls']=='大涨').sum()); bl162 = int((d162b['cls']=='大跌').sum())
    print('runup>{:.1f}: 50={:2d}(bg={},bl={})  162={:2d}(bg={},bl={})'.format(t, n50, bg50, bl50, n162, bg162, bl162))

# Scan: runup > X AND cpv <= Y
print('\n' + '='*110)
print('R1 SCAN: runup > X AND cpv <= Y  --  showing combinations with 0 BG on BOTH logs')
print('='*110)
print('{:>8s} {:>8s}  {:>20s}  {:>20s}'.format('runup>', 'cpv<=', '50.log', '162'))
print('{:>8s} {:>8s}  {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}  {:>4s} {:>4s} {:>4s} {:>8s} {:>8s} {:>12s}'.format(
    '','','H','BG','BL','BlkAvg','PasAvg','H','BG','BL','BlkAvg','PasAvg','Score'))

best = []
for runup_t in np.arange(0.0, 3.5, 0.1):
    for cpv_t in np.arange(-2.0, 2.1, 0.1):
        cond50 = (d50['runup_ma5_exhaustion'] > runup_t) & (d50['cpv_factor'] <= cpv_t)
        cond162 = (d162['runup_ma5_exhaustion'] > runup_t) & (d162['cpv_factor'] <= cpv_t)
        b50 = d50[cond50.fillna(False)]; b162 = d162[cond162.fillna(False)]
        n50 = len(b50); n162 = len(b162)
        if n50 == 0 and n162 == 0: continue

        bg50 = int((b50['cls']=='大涨').sum()); bl50 = int((b50['cls']=='大跌').sum())
        bg162 = int((b162['cls']=='大涨').sum()); bl162 = int((b162['cls']=='大跌').sum())

        if bg50 > 0 or bg162 > 0: continue  # only 0 BG
        if bl50 == 0 and bl162 == 0: continue  # need to catch some BL

        # Score: reward bl catches, reward higher hits
        score = bl50/20*0.4 + bl162/28*0.4 + (n50+n162)/(85+127)*0.2
        bmean50 = b50['ret'].mean(); bmean162 = b162['ret'].mean()
        pmean50 = d50[~cond50.fillna(False)]['ret'].mean()
        pmean162 = d162[~cond162.fillna(False)]['ret'].mean()

        best.append({
            'runup_t': runup_t, 'cpv_t': cpv_t,
            'n50': n50, 'bg50': bg50, 'bl50': bl50, 'bmean50': bmean50, 'pmean50': pmean50,
            'n162': n162, 'bg162': bg162, 'bl162': bl162, 'bmean162': bmean162, 'pmean162': pmean162,
            'score': score,
        })

best_df = pd.DataFrame(best).sort_values('score', ascending=False)
print('Found {} combinations with 0 BG on both logs'.format(len(best_df)))

for _, r in best_df.head(25).iterrows():
    note = ''
    if abs(r['runup_t']-0.8)<0.01 and abs(r['cpv_t']-0.0)<0.01: note = ' <-- 当前'
    if abs(r['runup_t']-0.8)<0.01 and abs(r['cpv_t']-(-0.4))<0.01: note = ' <-- 之前推荐'
    print('{:8.1f} {:8.1f}  {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%  {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}% {:11.3f}{}'.format(
        r['runup_t'], r['cpv_t'],
        int(r['n50']), int(r['bg50']), int(r['bl50']), r['bmean50'], r['pmean50'],
        int(r['n162']), int(r['bg162']), int(r['bl162']), r['bmean162'], r['pmean162'],
        r['score'], note))

# Show the runup>0.8 results as well for comparison
print('\n' + '='*110)
print('For reference: runup>0.8 fixed, varying cpv (including those with BG)')
print('='*110)
for cpv_t in np.arange(-2.0, 2.1, 0.2):
    cond50 = (d50['runup_ma5_exhaustion'] > 0.8) & (d50['cpv_factor'] <= cpv_t)
    cond162 = (d162['runup_ma5_exhaustion'] > 0.8) & (d162['cpv_factor'] <= cpv_t)
    b50 = d50[cond50.fillna(False)]; b162 = d162[cond162.fillna(False)]
    bg50 = int((b50['cls']=='大涨').sum()); bl50 = int((b50['cls']=='大跌').sum())
    bg162 = int((b162['cls']=='大涨').sum()); bl162 = int((b162['cls']=='大跌').sum())
    note = ''
    if abs(cpv_t-0.0)<0.01: note = ' <-- 当前'
    if abs(cpv_t-(-0.4))<0.01: note = ' <-- 之前推荐'
    if bg50>0 or bg162>0: note += ' KILLS_BG'
    print('runup>0.8 cpv<={:5.1f}: 50={:2d}(bg={},bl={},avg={:+.1f}%)  162={:2d}(bg={},bl={},avg={:+.1f}%){}'.format(
        cpv_t, len(b50), bg50, bl50, b50['ret'].mean(), len(b162), bg162, bl162, b162['ret'].mean(), note))
