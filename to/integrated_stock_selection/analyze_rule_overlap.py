#!/usr/bin/env python3
"""Compare old 3 vs new 3 filter rules with overlap analysis."""
import re, pandas as pd, numpy as np

BASE = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"

with open(BASE + r"\stock_selection_test_mode_walkforward_分类模型_原因子50.log", 'r', encoding='utf-8') as f:
    content = f.read()

header_pat = re.compile(r'(\d{4}-\d{2}-\d{2}) Top-5候选股票')
cand_pat = re.compile(r'1\.\s([\d.A-Z]+):\s预测收益=([-\d.eE+]+),\s收盘价=([-\d.]+)')
top1 = []
for m in header_pat.finditer(content):
    date = m.group(1)
    chunk = content[m.end():m.end()+150]
    cm = cand_pat.search(chunk)
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
all_factors = [c for c in factors.columns if c not in ['date', 'stock_code']]
merged = merged.merge(factors[['date', 'stock_code'] + all_factors],
                      left_on=['date_dt', 'stock_code'], right_on=['date', 'stock_code'], how='left')


def classify(r):
    if pd.isna(r):
        return 'unknown'
    if r >= 5:
        return 'bg'
    if r >= -5:
        return 'mid'
    return 'bl'


merged['cls'] = merged['next_pct_chg'].apply(classify)
valid = merged[merged['cls'] != 'unknown']
n_all, n_bg, n_bl = len(valid), len(valid[valid.cls == 'bg']), len(valid[valid.cls == 'bl'])

# ===== Define ALL rules =====
rules = {}

# OLD
rules['R1_old'] = {
    'desc': 'runup>0.8 & top10>=0.5',
    'cond': (valid['runup_ma5_exhaustion'] > 0.8) & (valid['top10_10d_strength_level_v3'] >= 0.5)
}
rules['R2_old'] = {
    'desc': 'blowoff>=4.15 & sector_accel<=0.41',
    'cond': (valid['prev_day_blowoff_risk'] >= 4.15) & (valid['sector_momentum_accel_rank'] <= 0.41)
}
rules['R3_old'] = {
    'desc': 'market_ad>=5.39559',
    'cond': valid['market_advance_decline_ratio'] >= 5.39559
}

# NEW
rules['R1_new'] = {
    'desc': 'runup>0.8 & cpv<=-0.33',
    'cond': (valid['runup_ma5_exhaustion'] > 0.8) & (valid['cpv_factor'] <= -0.33)
}
rules['R2_new'] = {
    'desc': 'sector_leadership<=-0.15',
    'cond': valid['sector_leadership_persistence_v3'] <= -0.15
}
bl_q60 = valid['buy_lg_amount_rate_pct'].quantile(0.60)
eq_q20 = valid['emotion_quality_balance_v3'].quantile(0.20)
rules['R3_new'] = {
    'desc': 'buy_lg_pct<=Q60 & emotion_q<=Q20',
    'cond': (valid['buy_lg_amount_rate_pct'] <= bl_q60) & (valid['emotion_quality_balance_v3'] <= eq_q20)
}

for k in rules:
    rules[k]['cond'] = rules[k]['cond'].fillna(False)

# ===== Stats =====
print('=' * 110)
print('RULE COMPARISON: Old 3 vs New 3 ({} Top-1, {} bg, {} bl)'.format(n_all, n_bg, n_bl))
print('=' * 110)
print('')
print('{:48s} {:>6s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}'.format(
    'Rule', 'Hits', 'BgBlk', 'BlBlk', 'MidBlk', 'Bg/All', 'Bl/All'))
for name, r in rules.items():
    cond = r['cond']
    bg_b = int((valid.loc[cond, 'cls'] == 'bg').sum())
    bl_b = int((valid.loc[cond, 'cls'] == 'bl').sum())
    mid_b = int((valid.loc[cond, 'cls'] == 'mid').sum())
    print('{:48s} {:6d} {:8d} {:8d} {:8d} {:9.1%} {:9.1%}'.format(
        r['desc'], int(cond.sum()), bg_b, bl_b, mid_b, bg_b / n_bg, bl_b / n_bl))

# ===== Pairwise Overlap =====
print('')
print('=' * 110)
print('PAIRWISE OVERLAP (all 6 rules)')
print('=' * 110)
print('{:45s} {:45s} {:>6s} {:>8s} {:>8s} {:>8s}'.format(
    'Rule A', 'Rule B', 'Overlap', 'Jaccard', 'sharedBg', 'sharedBl'))
rule_names = list(rules.keys())
for i in range(len(rule_names)):
    for j in range(i + 1, len(rule_names)):
        ri, rj = rules[rule_names[i]], rules[rule_names[j]]
        intersect = (ri['cond'] & rj['cond']).sum()
        union = (ri['cond'] | rj['cond']).sum()
        jaccard = intersect / union if union > 0 else 0
        shared = valid[ri['cond'] & rj['cond']]
        shared_bg = int((shared['cls'] == 'bg').sum())
        shared_bl = int((shared['cls'] == 'bl').sum())
        print('{:45s} {:45s} {:6d} {:8.3f} {:8d} {:8d}'.format(
            ri['desc'][:45], rj['desc'][:45], intersect, jaccard, shared_bg, shared_bl))

# ===== Union comparison =====
print('')
print('=' * 110)
print('UNION COMPARISON')
print('=' * 110)

old_union = rules['R1_old']['cond'] | rules['R2_old']['cond'] | rules['R3_old']['cond']
new_union = rules['R1_new']['cond'] | rules['R2_new']['cond'] | rules['R3_new']['cond']

for label, uni in [('OLD union (3 rules)', old_union), ('NEW union (3 rules)', new_union)]:
    bg_u = int((valid.loc[uni, 'cls'] == 'bg').sum())
    bl_u = int((valid.loc[uni, 'cls'] == 'bl').sum())
    mid_u = int((valid.loc[uni, 'cls'] == 'mid').sum())
    print('  {}: hits={}, bg={}/{}({:.1%}), bl={}/{}({:.1%}), mid={}'.format(
        label, int(uni.sum()), bg_u, n_bg, bg_u / n_bg, bl_u, n_bl, bl_u / n_bl, mid_u))

new_only = new_union & ~old_union
nobg = int((valid.loc[new_only, 'cls'] == 'bg').sum())
nobl = int((valid.loc[new_only, 'cls'] == 'bl').sum())
print('  NEW-only (not in OLD): hits={}, bg={}, bl={}'.format(int(new_only.sum()), nobg, nobl))

old_only = old_union & ~new_union
oobg = int((valid.loc[old_only, 'cls'] == 'bg').sum())
oobl = int((valid.loc[old_only, 'cls'] == 'bl').sum())
print('  OLD-only (not in NEW): hits={}, bg={}, bl={}'.format(int(old_only.sum()), oobg, oobl))

combined = old_union | new_union
cbg = int((valid.loc[combined, 'cls'] == 'bg').sum())
cbl = int((valid.loc[combined, 'cls'] == 'bl').sum())
print('  COMBINED (old | new): hits={}, bg={}/{}({:.1%}), bl={}/{}({:.1%})'.format(
    int(combined.sum()), cbg, n_bg, cbg / n_bg, cbl, n_bl, cbl / n_bl))

# ===== Which rule is redundant? =====
print('')
print('=' * 110)
print('REDUNDANCY ANALYSIS: if we drop one new rule at a time')
print('=' * 110)
for drop_name in ['R1_new', 'R2_new', 'R3_new']:
    reduced_union = pd.Series(False, index=valid.index)
    for name in rule_names:
        if name.startswith('R') and 'new' in name and name != drop_name:
            reduced_union = reduced_union | rules[name]['cond']
    bg_u = int((valid.loc[reduced_union, 'cls'] == 'bg').sum())
    bl_u = int((valid.loc[reduced_union, 'cls'] == 'bl').sum())
    dropped_rule = rules[drop_name]
    dropped_only = dropped_rule['cond'] & ~reduced_union
    do_bg = int((valid.loc[dropped_only, 'cls'] == 'bg').sum())
    do_bl = int((valid.loc[dropped_only, 'cls'] == 'bl').sum())
    print('  Drop {}: remaining hits={}, bg={}, bl={}'.format(drop_name, int(reduced_union.sum()), bg_u, bl_u))
    print('         dropped-rule-unique: hits={}, bg={}, bl={}'.format(int(dropped_only.sum()), do_bg, do_bl))

# ===== R1_old vs R1_new detail =====
print('')
print('=' * 110)
print('R1 DETAIL: old (top10) vs new (cpv)')
print('=' * 110)
r1o = rules['R1_old']['cond']
r1n = rules['R1_new']['cond']
print('  R1_old hits: {}, bg={}, bl={}'.format(
    int(r1o.sum()),
    int((valid.loc[r1o, 'cls'] == 'bg').sum()),
    int((valid.loc[r1o, 'cls'] == 'bl').sum())))
print('  R1_new hits: {}, bg={}, bl={}'.format(
    int(r1n.sum()),
    int((valid.loc[r1n, 'cls'] == 'bg').sum()),
    int((valid.loc[r1n, 'cls'] == 'bl').sum())))
overlap = (r1o & r1n).sum()
r1o_only = r1o & ~r1n
r1n_only = r1n & ~r1o
print('  Overlap: {} stocks'.format(overlap))
print('  R1_old-only: hits={}, bg={}, bl={}'.format(
    int(r1o_only.sum()),
    int((valid.loc[r1o_only, 'cls'] == 'bg').sum()),
    int((valid.loc[r1o_only, 'cls'] == 'bl').sum())))
print('  R1_new-only: hits={}, bg={}, bl={}'.format(
    int(r1n_only.sum()),
    int((valid.loc[r1n_only, 'cls'] == 'bg').sum()),
    int((valid.loc[r1n_only, 'cls'] == 'bl').sum())))

# ===== What does R2_new catch that others don't? =====
print('')
print('=' * 110)
print('R2_new (sector_leadership) uniqueness')
print('=' * 110)
others = rules['R1_old']['cond'] | rules['R2_old']['cond'] | rules['R3_old']['cond'] | rules['R1_new']['cond'] | rules['R3_new']['cond']
r2n_unique = rules['R2_new']['cond'] & ~others
print('  R2_new unique (not in any other rule): hits={}, bg={}, bl={}'.format(
    int(r2n_unique.sum()),
    int((valid.loc[r2n_unique, 'cls'] == 'bg').sum()),
    int((valid.loc[r2n_unique, 'cls'] == 'bl').sum())))
