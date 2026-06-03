#!/usr/bin/env python3
"""Final comparison: current vs proposed rules on 162 actual trades."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"

with open(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log", 'r', encoding='utf-8') as f:
    lines = f.readlines()

trades = []
buy_dates = {}
cd = None
for line in lines:
    dm = re.search(r'T日=(\d{4}-\d{2}-\d{2})', line)
    if dm:
        cd = dm.group(1)
    sm = re.search(r'📅 ([\d.A-Z]+) T\+1买入逻辑', line)
    if sm:
        buy_dates[sm.group(1)] = cd
    sm = re.search(r'💰 卖出:\s([\d.A-Z]+),\s价格:\s([-\d.]+),\s收益:\s([-\d.]+)\s\(([-\d.]+)%\),\s原因:\s(.+)', line)
    if sm:
        trades.append({
            'code': sm.group(1),
            'ret': float(sm.group(4)),
            'buy_date': buy_dates.get(sm.group(1), '?')
        })

trades_df = pd.DataFrame(trades)
trades_df = trades_df[trades_df['buy_date'] != '?']

factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
factors['date_str'] = factors['date'].dt.strftime('%Y-%m-%d')
needed = ['runup_ma5_exhaustion', 'cpv_factor', 'prev_day_blowoff_risk',
          'sector_momentum_accel_rank', 'sector_leadership_persistence_v3']
trades_df = trades_df.merge(
    factors[['date_str', 'stock_code'] + needed],
    left_on=['buy_date', 'code'],
    right_on=['date_str', 'stock_code'],
    how='left')


def classify(r):
    if r >= 5: return '大涨'
    if r >= 0: return '小涨'
    if r >= -5: return '小跌'
    return '大跌'


trades_df['cls'] = trades_df['ret'].apply(classify)
n_all = len(trades_df)

print('基准: 162(无规则) {}笔交易, 大涨={}, 小涨={}, 小跌={}, 大跌={}'.format(
    n_all,
    int((trades_df.cls == '大涨').sum()),
    int((trades_df.cls == '小涨').sum()),
    int((trades_df.cls == '小跌').sum()),
    int((trades_df.cls == '大跌').sum())))

# ----- rules -----
r1_now = (trades_df['runup_ma5_exhaustion'] > 0.8) & (trades_df['cpv_factor'] <= 0.0)
r2 = (trades_df['prev_day_blowoff_risk'] >= 3.5) & (trades_df['sector_momentum_accel_rank'] <= 0.30)
r3_now = trades_df['sector_leadership_persistence_v3'] <= -0.18

r1_new = (trades_df['runup_ma5_exhaustion'] > 0.8) & (trades_df['cpv_factor'] <= -0.4)
r3_new = trades_df['sector_leadership_persistence_v3'] <= -0.23

union_now = (r1_now | r2 | r3_now).fillna(False)
union_new = (r1_new | r2 | r3_new).fillna(False)


def show(name, union, r1, r3):
    blocked = trades_df[union]
    passed = trades_df[~union]
    d = blocked['cls'].value_counts()
    n_bg = len(trades_df[trades_df.cls == '大涨'])
    n_bl = len(trades_df[trades_df.cls == '大跌'])

    print('')
    print('=' * 90)
    print(name)
    print('=' * 90)
    print('拦截 {} 笔 (大涨误杀 {}%, 大跌拦截 {}%)'.format(
        len(blocked),
        '{:.0f}'.format(d.get('大涨', 0) / n_bg * 100) if n_bg > 0 else '0',
        '{:.0f}'.format(d.get('大跌', 0) / n_bl * 100) if n_bl > 0 else '0'))
    print('  大涨={}, 小涨={}, 小跌={}, 大跌={}'.format(
        d.get('大涨', 0), d.get('小涨', 0), d.get('小跌', 0), d.get('大跌', 0)))
    print('  拦截均收益={:+.1f}%, 中位数={:+.1f}% | 通过均收益={:+.1f}%, 中位数={:+.1f}%'.format(
        blocked['ret'].mean(), blocked['ret'].median(),
        passed['ret'].mean(), passed['ret'].median()))
    print('  各规则: R1={}, R2={}, R3={}'.format(r1.sum(), r2.sum(), r3.sum()))

    bg_killed = blocked[blocked['cls'] == '大涨']
    if len(bg_killed) > 0:
        print('  【误杀大涨】')
        for _, r in bg_killed.iterrows():
            print('    {} {}: {:+.1f}% (runup={:.1f}, cpv={:.2f}, sl={:.2f})'.format(
                r['code'], r['buy_date'], r['ret'],
                r['runup_ma5_exhaustion'], r['cpv_factor'],
                r['sector_leadership_persistence_v3']))

    bl_caught = blocked[blocked['cls'] == '大跌']
    if len(bl_caught) > 0:
        print('  【拦截大跌】')
        for _, r in bl_caught.iterrows():
            print('    {} {}: {:+.1f}%'.format(r['code'], r['buy_date'], r['ret']))


show('当前规则 (cpv<=0.0, sl<=-0.18)', union_now, r1_now, r3_now)
show('推荐规则 (cpv<=-0.4, sl<=-0.23)', union_new, r1_new, r3_new)

# Diff
only_now = trades_df[union_now & ~union_new]
only_new = trades_df[union_new & ~union_now]
print('')
print('=' * 90)
print('差异分析')
print('=' * 90)
print('当前拦但推荐不拦 (放回): {}笔, 均收益={:+.1f}%'.format(
    len(only_now), only_now['ret'].mean() if len(only_now) > 0 else 0))
for _, r in only_now.iterrows():
    print('  {} {}: {:+.1f}%'.format(r['code'], r['buy_date'], r['ret']))
print('推荐拦但当前不拦 (新增): {}笔'.format(len(only_new)))
for _, r in only_new.iterrows():
    print('  {} {}: {:+.1f}%'.format(r['code'], r['buy_date'], r['ret']))
