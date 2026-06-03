#!/usr/bin/env python3
"""Cross-log analysis: 50.log vs 162.log for all 3 rules."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"
factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
factors['date_str'] = factors['date'].dt.strftime('%Y-%m-%d')
needed = ['runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
          'sector_momentum_accel_rank','sector_leadership_persistence_v3']

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

print('50.log: {} trades (bg={}, sg={}, sl={}, bl={})'.format(
    len(d50), int((d50.cls=='大涨').sum()), int((d50.cls=='小涨').sum()),
    int((d50.cls=='小跌').sum()), int((d50.cls=='大跌').sum())))
print('162:    {} trades (bg={}, sg={}, sl={}, bl={})'.format(
    len(d162), int((d162.cls=='大涨').sum()), int((d162.cls=='小涨').sum()),
    int((d162.cls=='小跌').sum()), int((d162.cls=='大跌').sum())))

def analyze(df, cond):
    blocked = df[cond.fillna(False)]; n=len(blocked)
    if n==0: return None
    d = blocked['cls'].value_counts()
    return {'hits':n, 'bg':d.get('大涨',0), 'sg':d.get('小涨',0),
            'sl':d.get('小跌',0), 'bl':d.get('大跌',0),
            'bmean':blocked['ret'].mean(), 'pmean':df[~cond.fillna(False)]['ret'].mean()}

# ============================================
# R1: runup>0.8 & cpv <= X
# ============================================
print('\n' + '='*110)
print('R1: runup>0.8 & cpv <= X')
print('='*110)
print('{:>10s}  {:>25s}  {:>25s}'.format('cpv<=X', '50.log', '162'))
print('{:>10s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}'.format(
    '','H','BG','SG','BL','BlkAvg','PasAvg','H','BG','SG','BL','BlkAvg','PasAvg'))
for t in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.5]:
    r50 = analyze(d50, (d50['runup_ma5_exhaustion']>0.8) & (d50['cpv_factor']<=t))
    r162 = analyze(d162, (d162['runup_ma5_exhaustion']>0.8) & (d162['cpv_factor']<=t))
    s50 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r50['hits'],r50['bg'],r50['sg'],r50['bl'],r50['bmean'],r50['pmean']) if r50 else '  -'
    s162 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r162['hits'],r162['bg'],r162['sg'],r162['bl'],r162['bmean'],r162['pmean']) if r162 else '  -'
    note = ''
    if abs(t-(-0.4))<0.01: note = ' <-- 推荐'
    if abs(t-0.0)<0.01: note = ' <-- 当前'
    print('{:10.1f}  {:>25s}  {:>25s}{}'.format(t, s50, s162, note))

# ============================================
# R2: blowoff >= X & sector <= Y
# ============================================
print('\n' + '='*110)
print('R2: blowoff >= X & sector <= Y')
print('='*110)
print('{:>20s}  {:>25s}  {:>25s}'.format('Rule', '50.log', '162'))
print('{:>20s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}'.format(
    '','H','BG','SG','BL','BlkAvg','PasAvg','H','BG','SG','BL','BlkAvg','PasAvg'))
for bf in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for sa in [0.20, 0.25, 0.30, 0.35, 0.41]:
        cond = lambda df: (df['prev_day_blowoff_risk']>=bf) & (df['sector_momentum_accel_rank']<=sa)
        r50 = analyze(d50, cond(d50)); r162 = analyze(d162, cond(d162))
        if (r50 and r50['hits']>0) or (r162 and r162['hits']>0):
            s50 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r50['hits'],r50['bg'],r50['sg'],r50['bl'],r50['bmean'],r50['pmean']) if r50 else '  -    -    -    -       -       -'
            s162 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r162['hits'],r162['bg'],r162['sg'],r162['bl'],r162['bmean'],r162['pmean']) if r162 else '  -    -    -    -       -       -'
            note = ''
            if abs(bf-3.5)<0.01 and abs(sa-0.30)<0.01: note = ' <-- 当前'
            print('{:20s}  {:>25s}  {:>25s}{}'.format('b>={:.0f}&s<={:.2f}'.format(bf,sa), s50, s162, note))

# ============================================
# R3: sector_leadership <= X
# ============================================
print('\n' + '='*110)
print('R3: sector_leadership <= X')
print('='*110)
print('{:>8s}  {:>25s}  {:>25s}'.format('sl<=X', '50.log', '162'))
print('{:>8s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}  {:>4s} {:>4s} {:>4s} {:>4s} {:>8s} {:>8s}'.format(
    '','H','BG','SG','BL','BlkAvg','PasAvg','H','BG','SG','BL','BlkAvg','PasAvg'))
for t in [-0.31, -0.29, -0.27, -0.25, -0.23, -0.21, -0.19, -0.18, -0.17, -0.15, -0.13, -0.11, -0.09]:
    r50 = analyze(d50, d50['sector_leadership_persistence_v3']<=t)
    r162 = analyze(d162, d162['sector_leadership_persistence_v3']<=t)
    s50 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r50['hits'],r50['bg'],r50['sg'],r50['bl'],r50['bmean'],r50['pmean']) if r50 else '  -'
    s162 = '{:4d} {:4d} {:4d} {:4d} {:7.1f}% {:7.1f}%'.format(r162['hits'],r162['bg'],r162['sg'],r162['bl'],r162['bmean'],r162['pmean']) if r162 else '  -'
    note = ''
    p50 = (d50['sector_leadership_persistence_v3']<=t).mean()*100
    p162 = (d162['sector_leadership_persistence_v3']<=t).mean()*100
    if abs(t-(-0.18))<0.01: note = ' <-- 当前(P50={:.0f}%,P162={:.0f}%)'.format(p50,p162)
    if abs(t-(-0.23))<0.01: note = ' <-- 推荐(P50={:.0f}%,P162={:.0f}%)'.format(p50,p162)
    print('{:8.2f}  {:>25s}  {:>25s}{}'.format(t, s50, s162, note))

# ============================================
# UNION: best combos
# ============================================
print('\n' + '='*110)
print('FINAL UNION COMPARISON')
print('='*110)

configs = [
    ('当前', 0.0, 3.5, 0.30, -0.18),
    ('推荐R1+R3', -0.4, 3.5, 0.30, -0.23),
    ('全部推荐', -0.4, 2.0, 0.20, -0.23),
    ('推荐R1+R2+R3', -0.4, 2.0, 0.25, -0.23),
]

for label, cpv_t, blow_t, sect_t, sl_t in configs:
    for dname, df in [('50.log',d50), ('162',d162)]:
        r1 = (df['runup_ma5_exhaustion']>0.8) & (df['cpv_factor']<=cpv_t)
        r2 = (df['prev_day_blowoff_risk']>=blow_t) & (df['sector_momentum_accel_rank']<=sect_t)
        r3 = df['sector_leadership_persistence_v3'] <= sl_t
        union = (r1|r2|r3).fillna(False)
        r = analyze(df, union)
        if r:
            print('{:20s} {}: hits={:2d}, bg={}, sg={}, sl={}, bl={}, blkAvg={:+.1f}%, pasAvg={:+.1f}%  (R1={},R2={},R3={})'.format(
                label, dname, r['hits'], r['bg'], r['sg'], r['sl'], r['bl'], r['bmean'], r['pmean'],
                r1.sum(), r2.sum(), r3.sum()))
