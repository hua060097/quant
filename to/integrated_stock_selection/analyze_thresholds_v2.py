#!/usr/bin/env python3
"""Threshold analysis using actual 162 (no filter) trade returns."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"

# --- 1. Extract all trades from 162 (no filter) with their buy dates ---
with open(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log", 'r', encoding='utf-8') as f:
    lines = f.readlines()

trades = []
buy_dates = {}  # stock_code -> last buy date (T日)
current_buy_date = None

for line in lines:
    # Track buy date from T+1 logic: 'T日=2025-08-05, T+1日=2025-08-06'
    dm = re.search(r'T日=(\d{4}-\d{2}-\d{2})', line)
    if dm:
        current_buy_date = dm.group(1)

    # Track which stock is being bought: '📅 300885.SZ T+1买入逻辑: T日=...'
    sm = re.search(r'📅 ([\d.A-Z]+) T\+1买入逻辑', line)
    if sm:
        buy_dates[sm.group(1)] = current_buy_date

    # Extract sell records
    sm = re.search(r'💰 卖出:\s([\d.A-Z]+),\s价格:\s([-\d.]+),\s收益:\s([-\d.]+)\s\(([-\d.]+)%\),\s原因:\s(.+)', line)
    if sm:
        code = sm.group(1)
        ret = float(sm.group(4))
        trades.append({
            'code': code,
            'ret': ret,
            'buy_date': buy_dates.get(code, 'unknown'),
            'reason': sm.group(5)[:50]
        })

print('Extracted {} trades from 162 (no filter)'.format(len(trades)))
trades_df = pd.DataFrame(trades)
trades_df = trades_df[trades_df['buy_date'] != 'unknown']
print('With buy dates: {}'.format(len(trades_df)))

# --- 2. Load factor values ---
factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
factors['date_str'] = factors['date'].dt.strftime('%Y-%m-%d')

needed_cols = ['runup_ma5_exhaustion', 'cpv_factor', 'cpv_factor_pct',
               'prev_day_blowoff_risk', 'sector_momentum_accel_rank',
               'sector_leadership_persistence_v3',
               'top10_10d_strength_level_v3']
trades_df = trades_df.merge(
    factors[['date_str', 'stock_code'] + needed_cols],
    left_on=['buy_date', 'code'], right_on=['date_str', 'stock_code'], how='left')

# --- 3. Helper function ---
def classify(r):
    if r >= 5: return '大涨'
    if r >= 0: return '小涨'
    if r >= -5: return '小跌'
    return '大跌'

trades_df['cls'] = trades_df['ret'].apply(classify)

def analyze_rule(name, cond_series, extra_info=''):
    cond = cond_series.fillna(False)
    blocked = trades_df[cond]
    passed = trades_df[~cond]
    n = len(blocked)
    if n == 0:
        return None
    cls_counts = blocked['cls'].value_counts()
    bg = cls_counts.get('大涨', 0)
    sg = cls_counts.get('小涨', 0)
    sl = cls_counts.get('小跌', 0)
    bl = cls_counts.get('大跌', 0)
    return {
        'name': name, 'hits': n, 'passed': len(passed),
        'bg': bg, 'sg': sg, 'sl': sl, 'bl': bl,
        'blocked_mean': blocked['ret'].mean(),
        'blocked_median': blocked['ret'].median(),
        'passed_mean': passed['ret'].mean(),
        'passed_median': passed['ret'].median(),
        'bg_pct': bg / len(trades_df[trades_df['cls'] == '大涨']) * 100 if len(trades_df[trades_df['cls'] == '大涨']) > 0 else 0,
        'bl_pct': bl / len(trades_df[trades_df['cls'] == '大跌']) * 100 if len(trades_df[trades_df['cls'] == '大跌']) > 0 else 0,
        'extra': extra_info,
    }

# Total stats
n_all = len(trades_df)
n_bg = len(trades_df[trades_df['cls'] == '大涨'])
n_sg = len(trades_df[trades_df['cls'] == '小涨'])
n_sl = len(trades_df[trades_df['cls'] == '小跌'])
n_bl = len(trades_df[trades_df['cls'] == '大跌'])
print('\nAll 162 trades: {} (bg={}, sg={}, sl={}, bl={})'.format(n_all, n_bg, n_sg, n_sl, n_bl))

def print_table(title, results):
    print('\n' + '=' * 120)
    print(title)
    print('=' * 120)
    print('{:40s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>12s} {:>15s}'.format(
        'Rule', 'Hits', '大涨', '小涨', '小跌', '大跌', 'BlkAvg', 'BlkMed', 'PasAvg', 'PasMed', '大涨误杀%', 'Extra'))
    for r in results:
        if r is None: continue
        print('{:40s} {:5d} {:5d} {:5d} {:5d} {:5d} {:8.1f}% {:8.1f}% {:8.1f}% {:8.1f}% {:11.1f}% {:>15s}'.format(
            r['name'][:40], r['hits'], r['bg'], r['sg'], r['sl'], r['bl'],
            r['blocked_mean'], r['blocked_median'],
            r['passed_mean'], r['passed_median'],
            r['bg_pct'], r['extra']))

# ========================================================================
# RULE 1: runup_ma5_exhaustion > 0.8 AND cpv_factor <= X
# ========================================================================
print('\n' + '#' * 120)
print('# R1: runup_ma5_exhaustion > 0.8 AND cpv_factor <= X')
print('# cpv: mean={:.3f}, std={:.3f}, P10={:.3f}, P25={:.3f}, P50={:.3f}, P75={:.3f}'.format(
    trades_df['cpv_factor'].mean(), trades_df['cpv_factor'].std(),
    trades_df['cpv_factor'].quantile(0.10), trades_df['cpv_factor'].quantile(0.25),
    trades_df['cpv_factor'].quantile(0.50), trades_df['cpv_factor'].quantile(0.75)))
print('# Base (runup>0.8): {} trades'.format((trades_df['runup_ma5_exhaustion'] > 0.8).sum()))

base1 = trades_df['runup_ma5_exhaustion'] > 0.8
r1_results = []
for t in np.arange(-2.0, 2.1, 0.2):
    cond = base1 & (trades_df['cpv_factor'] <= t)
    r = analyze_rule('cpv<={:.1f}'.format(t), cond, '')
    if r and r['hits'] > 0:
        r1_results.append(r)

# Also try cpv_pct
for t in np.arange(0.1, 1.0, 0.1):
    cond = base1 & (trades_df['cpv_factor_pct'] <= t)
    r = analyze_rule('cpv_pct<={:.1f}'.format(t), cond, '')
    if r and r['hits'] > 0:
        r1_results.append(r)

# Old rule for comparison
old_r1 = base1 & (trades_df['top10_10d_strength_level_v3'] >= 0.5)
r_old = analyze_rule('OLD: runup>0.8&top10>=0.5', old_r1, '')
if r_old: r1_results.append(r_old)

print_table('R1 Sensitivity', r1_results)

# ========================================================================
# RULE 2: prev_day_blowoff_risk >= X AND sector_momentum_accel_rank <= Y
# ========================================================================
print('\n' + '#' * 120)
print('# R2: blowoff >= X AND sector_accel <= Y')
print('# blowoff: mean={:.3f}, P75={:.3f}, P90={:.3f}, P95={:.3f}'.format(
    trades_df['prev_day_blowoff_risk'].mean(),
    trades_df['prev_day_blowoff_risk'].quantile(0.75),
    trades_df['prev_day_blowoff_risk'].quantile(0.90),
    trades_df['prev_day_blowoff_risk'].quantile(0.95)))
print('# sector_accel: P10={:.3f}, P25={:.3f}, P50={:.3f}'.format(
    trades_df['sector_momentum_accel_rank'].quantile(0.10),
    trades_df['sector_momentum_accel_rank'].quantile(0.25),
    trades_df['sector_momentum_accel_rank'].quantile(0.50)))

r2_results = []
for bf in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for sa in [0.20, 0.25, 0.30, 0.35, 0.41]:
        cond = (trades_df['prev_day_blowoff_risk'] >= bf) & (trades_df['sector_momentum_accel_rank'] <= sa)
        r = analyze_rule('blowoff>={:.1f}&sector<={:.2f}'.format(bf, sa), cond, '')
        if r and r['hits'] > 0:
            r2_results.append(r)

print_table('R2 Sensitivity', r2_results)

# ========================================================================
# RULE 3: sector_leadership_persistence_v3 <= X
# ========================================================================
print('\n' + '#' * 120)
print('# R3: sector_leadership_persistence_v3 <= X (broadcast factor)')
print('# sl: mean={:.4f}, std={:.4f}, P05={:.4f}, P10={:.4f}, P20={:.4f}'.format(
    trades_df['sector_leadership_persistence_v3'].mean(),
    trades_df['sector_leadership_persistence_v3'].std(),
    trades_df['sector_leadership_persistence_v3'].quantile(0.05),
    trades_df['sector_leadership_persistence_v3'].quantile(0.10),
    trades_df['sector_leadership_persistence_v3'].quantile(0.20)))
print('#   daily broadcast: same value for all stocks on same day => this is a MARKET TIMING rule')

# Note: since sl is broadcast, we need to look at daily values, not individual trades
# For consistency with the trade-level analysis, we still show trade counts

r3_results = []
for t in np.arange(-0.35, 0.05, 0.02):
    cond = trades_df['sector_leadership_persistence_v3'] <= t
    r = analyze_rule('sl<={:.2f}'.format(t), cond, '')
    if r and r['hits'] > 0:
        extra = ''
        if abs(t - (-0.23)) < 0.01: extra = '<-- P05'
        if abs(t - (-0.20)) < 0.01: extra = '<-- P08'
        if abs(t - (-0.18)) < 0.01: extra = '<-- P10(当前)'
        if abs(t - (-0.15)) < 0.01: extra = '<-- P14'
        r['extra'] = extra
        r3_results.append(r)

print_table('R3 Sensitivity', r3_results)

# ========================================================================
# FINAL: Test combinations
# ========================================================================
print('\n' + '#' * 120)
print('# FINAL: Best threshold combinations')
print('#' * 120)

configs = [
    ('当前', 0.0, 3.5, 0.30, -0.18),
    ('收紧R3', 0.0, 3.5, 0.30, -0.23),
    ('收紧R3+R1保守', -0.5, 3.5, 0.30, -0.23),
    ('仅R1+R2', 0.0, 3.5, 0.30, -0.99),  # -0.99 effectively disables R3
    ('放宽R2', 0.0, 2.5, 0.35, -0.23),
]

for label, cpv_t, blow_t, sect_t, sl_t in configs:
    r1 = (trades_df['runup_ma5_exhaustion'] > 0.8) & (trades_df['cpv_factor'] <= cpv_t)
    r2 = (trades_df['prev_day_blowoff_risk'] >= blow_t) & (trades_df['sector_momentum_accel_rank'] <= sect_t)
    if sl_t > -0.9:
        r3 = trades_df['sector_leadership_persistence_v3'] <= sl_t
    else:
        r3 = pd.Series(False, index=trades_df.index)
    union = (r1 | r2 | r3).fillna(False)
    r = analyze_rule(label, union, 'cpv={} blow={}/{} sl={}'.format(cpv_t, blow_t, sect_t, sl_t))
    if r:
        print('  {}: hits={}, bg={}, sg={}, sl={}, bl={}, blk_avg={:+.1f}%, pas_avg={:+.1f}%'.format(
            label, r['hits'], r['bg'], r['sg'], r['sl'], r['bl'], r['blocked_mean'], r['passed_mean']))
