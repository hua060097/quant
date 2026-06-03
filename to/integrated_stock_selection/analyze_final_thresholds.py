#!/usr/bin/env python3
"""Final threshold analysis for all 3 rules."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"

with open(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子50.log", 'r', encoding='utf-8') as f:
    content = f.read()
hp = re.compile(r'(\d{4}-\d{2}-\d{2}) Top-5候选股票')
cp = re.compile(r'1\.\s([\d.A-Z]+):\s预测收益=([-\d.eE+]+),\s收盘价=([-\d.]+)')
top1 = []
for m in hp.finditer(content):
    date = m.group(1)
    chunk = content[m.end():m.end()+150]
    cm = cp.search(chunk)
    if cm:
        top1.append({'date': date, 'stock_code': cm.group(1)})
top1_df = pd.DataFrame(top1)
top1_df['date_dt'] = pd.to_datetime(top1_df['date'])

prices = pd.read_parquet(BASE + r"\cache\stock_features_20230504_20260511.parquet",
                         columns=['date', 'stock_code', 'pct_chg'])
prices['date'] = pd.to_datetime(prices['date'])
prices = prices.sort_values(['stock_code', 'date'])
prices['next_pct_chg'] = prices.groupby('stock_code')['pct_chg'].shift(-1)
merged = top1_df.merge(prices, left_on=['date_dt', 'stock_code'],
                       right_on=['date', 'stock_code'], how='left')

factors = pd.read_parquet(BASE + r"\factor_lab_data\factors.parquet")
factors['date'] = pd.to_datetime(factors['date'])
needed = ['runup_ma5_exhaustion', 'cpv_factor', 'cpv_factor_pct',
          'prev_day_blowoff_risk', 'sector_momentum_accel_rank',
          'sector_leadership_persistence_v3']
merged = merged.merge(factors[['date', 'stock_code'] + needed],
                      left_on=['date_dt', 'stock_code'], right_on=['date', 'stock_code'], how='left')


def classify(r):
    if pd.isna(r): return 'unknown'
    if r >= 5: return '大涨'
    if r >= 0: return '小涨'
    if r >= -5: return '小跌'
    return '大跌'


merged['cls'] = merged['next_pct_chg'].apply(classify)
valid = merged[merged['cls'] != 'unknown']
n_all = len(valid)
print('Top-1 dataset: {} stocks, 大涨={}, 小涨={}, 小跌={}, 大跌={}'.format(
    n_all,
    len(valid[valid.cls == '大涨']),
    len(valid[valid.cls == '小涨']),
    len(valid[valid.cls == '小跌']),
    len(valid[valid.cls == '大跌'])))

def eval_rule(cond, desc):
    cond = cond.fillna(False)
    hits = int(cond.sum())
    if hits == 0:
        return None
    d = valid.loc[cond, 'cls'].value_counts()
    bg, sg, sl, bl = d.get('大涨', 0), d.get('小涨', 0), d.get('小跌', 0), d.get('大跌', 0)
    blocked_ret = valid.loc[cond, 'next_pct_chg']
    passed_ret = valid.loc[~cond, 'next_pct_chg']
    return {
        'desc': desc, 'hits': hits,
        '大涨': bg, '小涨': sg, '小跌': sl, '大跌': bl,
        'blocked_mean': blocked_ret.mean(),
        'blocked_median': blocked_ret.median(),
        'passed_mean': passed_ret.mean(),
        'passed_median': passed_ret.median(),
    }


# ================================================================
# RULE 1: runup>0.8 & cpv_factor <= X
# ================================================================
print('')
print('#' * 100)
print('# RULE 1: runup_ma5_exhaustion > 0.8  AND  cpv_factor <= X')
print('# cpv on Top-1: mean={:.3f}, std={:.3f}, P10={:.3f}, P25={:.3f}, P50={:.3f}, P75={:.3f}, P90={:.3f}'.format(
    valid['cpv_factor'].mean(), valid['cpv_factor'].std(),
    valid['cpv_factor'].quantile(0.10), valid['cpv_factor'].quantile(0.25),
    valid['cpv_factor'].quantile(0.50), valid['cpv_factor'].quantile(0.75),
    valid['cpv_factor'].quantile(0.90)))
print('# cpv on ALL stocks in test period:')
all_cpv = factors[(factors['date'] >= '2025-08-01') & (factors['date'] <= '2026-04-20')]['cpv_factor'].dropna()
print('#   mean={:.3f}, std={:.3f}, P10={:.3f}, P25={:.3f}, P50={:.3f}, P75={:.3f}, P90={:.3f}'.format(
    all_cpv.mean(), all_cpv.std(),
    all_cpv.quantile(0.10), all_cpv.quantile(0.25),
    all_cpv.quantile(0.50), all_cpv.quantile(0.75), all_cpv.quantile(0.90)))
print('#' * 100)

base1 = valid['runup_ma5_exhaustion'] > 0.8
n_base1 = int(base1.sum())
print('Base (runup>0.8): {} stocks, 大涨={}, 小涨={}, 小跌={}, 大跌={}'.format(
    n_base1,
    int((valid.loc[base1, 'cls'] == '大涨').sum()),
    int((valid.loc[base1, 'cls'] == '小涨').sum()),
    int((valid.loc[base1, 'cls'] == '小跌').sum()),
    int((valid.loc[base1, 'cls'] == '大跌').sum())))

print('')
print('{:>8s}  {:>4s}  {:>4s} {:>4s} {:>4s} {:>4s}  {:>8s} {:>8s} {:>8s} {:>8s}  {:>10s}'.format(
    'cpv<=X', 'Hits', '大涨', '小涨', '小跌', '大跌', 'BlkMean', 'BlkMed', 'PassMean', 'PassMed', 'Note'))
for t in np.arange(-2.0, 2.1, 0.1):
    cond = base1 & (valid['cpv_factor'] <= t)
    r = eval_rule(cond, '')
    if r is None: continue
    note = ''
    if abs(t - (-0.3)) < 0.01: note = '←之前推荐'
    if abs(t - 0.0) < 0.01: note = '←0线'
    if abs(t - 0.5) < 0.01: note = '←P70全市场'
    print('{:8.2f}  {:4d}  {:4d} {:4d} {:4d} {:4d}  {:7.1%}  {:7.1%}  {:7.1%}  {:7.1%}  {}'.format(
        t, r['hits'], r['大涨'], r['小涨'], r['小跌'], r['大跌'],
        r['blocked_mean']/100, r['blocked_median']/100, r['passed_mean']/100, r['passed_median']/100, note))

# cpv_pct version too
print('')
print('--- cpv_factor_pct version ---')
print('# cpv_pct on Top-1: mean={:.3f}, P10={:.3f}, P25={:.3f}, P50={:.3f}, P75={:.3f}, P90={:.3f}'.format(
    valid['cpv_factor_pct'].mean(),
    valid['cpv_factor_pct'].quantile(0.10), valid['cpv_factor_pct'].quantile(0.25),
    valid['cpv_factor_pct'].quantile(0.50), valid['cpv_factor_pct'].quantile(0.75),
    valid['cpv_factor_pct'].quantile(0.90)))
print('{:>8s}  {:>4s}  {:>4s} {:>4s} {:>4s} {:>4s}  {:>8s} {:>8s} {:>8s} {:>8s}'.format(
    'cpv_pct', 'Hits', '大涨', '小涨', '小跌', '大跌', 'BlkMean', 'BlkMed', 'PassMean', 'PassMed'))
for t in np.arange(0.05, 1.0, 0.05):
    cond = base1 & (valid['cpv_factor_pct'] <= t)
    r = eval_rule(cond, '')
    if r is None: continue
    if r['大涨'] > 0: continue  # only show safe thresholds
    print('{:8.2f}  {:4d}  {:4d} {:4d} {:4d} {:4d}  {:7.1%}  {:7.1%}  {:7.1%}  {:7.1%}'.format(
        t, r['hits'], r['大涨'], r['小涨'], r['小跌'], r['大跌'],
        r['blocked_mean']/100, r['blocked_median']/100, r['passed_mean']/100, r['passed_median']/100))


# ================================================================
# RULE 2: blowoff>=X & sector_accel<=Y
# ================================================================
print('')
print('#' * 100)
print('# RULE 2: prev_day_blowoff_risk >= X  AND  sector_momentum_accel_rank <= Y')
print('# blowoff on Top-1: mean={:.3f}, std={:.3f}, P50={:.3f}, P75={:.3f}, P90={:.3f}, P95={:.3f}'.format(
    valid['prev_day_blowoff_risk'].mean(), valid['prev_day_blowoff_risk'].std(),
    valid['prev_day_blowoff_risk'].quantile(0.50), valid['prev_day_blowoff_risk'].quantile(0.75),
    valid['prev_day_blowoff_risk'].quantile(0.90), valid['prev_day_blowoff_risk'].quantile(0.95)))
print('# sector_accel on Top-1: mean={:.3f}, std={:.3f}, P10={:.3f}, P25={:.3f}, P50={:.3f}'.format(
    valid['sector_momentum_accel_rank'].mean(), valid['sector_momentum_accel_rank'].std(),
    valid['sector_momentum_accel_rank'].quantile(0.10), valid['sector_momentum_accel_rank'].quantile(0.25),
    valid['sector_momentum_accel_rank'].quantile(0.50)))
print('#' * 100)

print('')
print('{:>10s} {:>10s}  {:>4s}  {:>4s} {:>4s} {:>4s} {:>4s}  {:>8s} {:>8s}  {:>12s}'.format(
    'blowoff>=', 'sector<=', 'Hits', '大涨', '小涨', '小跌', '大跌', 'BlkMean', 'BlkMed', 'Note'))

for bf in [2.0, 2.5, 3.0, 3.5, 4.0, 4.15, 4.5, 5.0]:
    for sa in [0.25, 0.30, 0.35, 0.41, 0.45, 0.50, 0.55]:
        cond = (valid['prev_day_blowoff_risk'] >= bf) & (valid['sector_momentum_accel_rank'] <= sa)
        r = eval_rule(cond, '')
        if r is None or r['hits'] < 2: continue
        if r['大涨'] > 0: continue  # must not kill bg
        note = ''
        if abs(bf - 3.5) < 0.01 and abs(sa - 0.35) < 0.01: note = '←之前推荐'
        if abs(bf - 4.15) < 0.01 and abs(sa - 0.41) < 0.01: note = '←原始'
        if abs(bf - 3.0) < 0.01 and abs(sa - 0.30) < 0.01: note = '←更保守'
        print('{:10.1f} {:10.2f}  {:4d}  {:4d} {:4d} {:4d} {:4d}  {:7.1%}  {:7.1%}  {}'.format(
            bf, sa, r['hits'], r['大涨'], r['小涨'], r['小跌'], r['大跌'],
            r['blocked_mean']/100, r['blocked_median']/100, note))

# ================================================================
# RULE 3: sector_leadership_persistence_v3 <= X  (already done, summarize)
# ================================================================
print('')
print('#' * 100)
print('# RULE 3: sector_leadership_persistence_v3 <= X  (summary from previous analysis)')
print('#' * 100)
for t in [-0.25, -0.23, -0.20, -0.18, -0.17, -0.16, -0.15]:
    cond = valid['sector_leadership_persistence_v3'] <= t
    r = eval_rule(cond, '')
    if r is None: continue
    note = ''
    if abs(t - (-0.18)) < 0.01: note = '←推荐(P10)'
    if abs(t - (-0.20)) < 0.01: note = '←保守'
    if abs(t - (-0.15)) < 0.01: note = '←临界'
    print('  sl<={:.2f}: hits={:3d}, 大涨={}, 小涨={}, 小跌={}, 大跌={}, blk_mean={:+.1f}%, blk_med={:+.1f}%, pas_mean={:+.1f}%  {}'.format(
        t, r['hits'], r['大涨'], r['小涨'], r['小跌'], r['大跌'],
        r['blocked_mean'], r['blocked_median'], r['passed_mean'], note))


# ================================================================
# FINAL: Best 3-rule union
# ================================================================
print('')
print('#' * 100)
print('# FINAL 3-RULE UNION')
print('#' * 100)

# Candidate thresholds
candidates = {
    'R1_cpv_m03': lambda: (valid['runup_ma5_exhaustion'] > 0.8) & (valid['cpv_factor'] <= -0.3),
    'R1_cpv_00': lambda: (valid['runup_ma5_exhaustion'] > 0.8) & (valid['cpv_factor'] <= 0.0),
    'R1_cpv_pct_05': lambda: (valid['runup_ma5_exhaustion'] > 0.8) & (valid['cpv_factor_pct'] <= 0.50),

    'R2_b35_s30': lambda: (valid['prev_day_blowoff_risk'] >= 3.5) & (valid['sector_momentum_accel_rank'] <= 0.30),
    'R2_b35_s35': lambda: (valid['prev_day_blowoff_risk'] >= 3.5) & (valid['sector_momentum_accel_rank'] <= 0.35),
    'R2_b30_s30': lambda: (valid['prev_day_blowoff_risk'] >= 3.0) & (valid['sector_momentum_accel_rank'] <= 0.30),

    'R3_sl_m18': lambda: valid['sector_leadership_persistence_v3'] <= -0.18,
    'R3_sl_m20': lambda: valid['sector_leadership_persistence_v3'] <= -0.20,
    'R3_sl_m17': lambda: valid['sector_leadership_persistence_v3'] <= -0.17,
}

# Test a few combinations
combos = [
    ('方案A(推荐)', ['R1_cpv_m03', 'R2_b35_s30', 'R3_sl_m18']),
    ('方案B(保守)', ['R1_cpv_m03', 'R2_b35_s30', 'R3_sl_m20']),
    ('方案C(偏松)', ['R1_cpv_00', 'R2_b35_s35', 'R3_sl_m17']),
    ('方案D(原始参)', ['R1_cpv_pct_05', 'R2_b30_s30', 'R3_sl_m18']),
]

for name, rule_keys in combos:
    union = pd.Series(False, index=valid.index)
    for rk in rule_keys:
        union = union | candidates[rk]()
    union = union.fillna(False)
    r = eval_rule(union, name)
    if r:
        # Count rule overlaps
        individual = {}
        for rk in rule_keys:
            individual[rk] = candidates[rk]().fillna(False)
        n_individual = sum(c.sum() for c in individual.values())
        union_hits = r['hits']
        print('{}: hits={}, 大涨={}, 小涨={}, 小跌={}, 大跌={}, 合计拦截大跌={}/{}'.format(
            name, union_hits, r['大涨'], r['小涨'], r['小跌'], r['大跌'],
            r['大跌'], len(valid[valid.cls == '大跌'])))
        print('  blocked_mean={:+.1f}%, passed_mean={:+.1f}%, 各规则独有={}'.format(
            r['blocked_mean'], r['passed_mean'],
            {rk: int(c.sum()) for rk, c in individual.items()}))
        # Overlap
        if len(rule_keys) >= 2:
            for i in range(len(rule_keys)):
                for j in range(i+1, len(rule_keys)):
                    overlap = (individual[rule_keys[i]] & individual[rule_keys[j]]).sum()
                    print('  {} & {} overlap={}'.format(rule_keys[i], rule_keys[j], overlap))
