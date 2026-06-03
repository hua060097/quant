#!/usr/bin/env python3
"""Comprehensive filter rule scan: standalone, relaxed runup, factor pairs."""
import re, pandas as pd, numpy as np, time, itertools, sys

BASE_DIR = r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection"
LOG = BASE_DIR + r"\stock_selection_test_mode_walkforward_分类模型_原因子50.log"
PRICE = BASE_DIR + r"\cache\stock_features_20230504_20260511.parquet"
FACTOR = BASE_DIR + r"\factor_lab_data\factors.parquet"

t0 = time.time()

# --- Load data ---
with open(LOG, 'r', encoding='utf-8') as f:
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

prices = pd.read_parquet(PRICE, columns=['date', 'stock_code', 'pct_chg'])
prices['date'] = pd.to_datetime(prices['date'])
prices = prices.sort_values(['stock_code', 'date'])
prices['next_pct_chg'] = prices.groupby('stock_code')['pct_chg'].shift(-1)
merged = top1_df.merge(prices, left_on=['date_dt', 'stock_code'], right_on=['date', 'stock_code'], how='left')

factors = pd.read_parquet(FACTOR)
factors['date'] = pd.to_datetime(factors['date'])
all_factors = [c for c in factors.columns if c not in ['date', 'stock_code']]
merged = merged.merge(factors[['date', 'stock_code'] + all_factors],
                      left_on=['date_dt', 'stock_code'], right_on=['date', 'stock_code'], how='left')

def classify(r):
    if pd.isna(r): return 'unknown'
    if r >= 5: return 'bg'
    if r >= -5: return 'mid'
    return 'bl'
merged['cls'] = merged['next_pct_chg'].apply(classify)
valid = merged[merged['cls'] != 'unknown']
n_all, n_bg, n_bl = len(valid), len(valid[valid.cls == 'bg']), len(valid[valid.cls == 'bl'])
print('Top-1 candidates: {} stocks, bg={}, bl={}'.format(n_all, n_bg, n_bl))

# ========================================================================
# PART 1: Standalone single-factor filters on ALL 167 Top-1
# ========================================================================
print('\n' + '=' * 90)
print('PART 1: Standalone single-factor filters (no runup base)')
print('=' * 90)
print('  Score = -bg_blocked_pct*10 + bl_blocked_pct*3 - hits_pct*0.3')
print('  (hits limited to 3-50% of all stocks)')

solo_results = []
for factor in all_factors:
    fs = valid[factor].dropna()
    if len(fs) < 10 or fs.nunique() < 3:
        continue
    q_low, q_high = fs.quantile(0.05), fs.quantile(0.95)
    if q_high <= q_low or np.isnan(q_low) or np.isnan(q_high):
        continue
    for thresh in np.linspace(q_low, q_high, 30):
        for d in ['>=', '<=']:
            blocked = (valid[factor] >= thresh) if d == '>=' else (valid[factor] <= thresh)
            blocked = blocked.fillna(False)
            hits = int(blocked.sum())
            if hits < 3 or hits > n_all * 0.5:
                continue
            bg_b = int((valid.loc[blocked, 'cls'] == 'bg').sum())
            bl_b = int((valid.loc[blocked, 'cls'] == 'bl').sum())
            score = -bg_b / n_bg * 10 + bl_b / n_bl * 3 - hits / n_all * 0.3
            solo_results.append({
                'factor': factor, 'thresh': thresh, 'dir': d,
                'score': score, 'hits': hits,
                'bg_blocked': bg_b, 'bl_blocked': bl_b,
            })

solo_df = pd.DataFrame(solo_results).sort_values('score', ascending=False)
print('\nTop 20 standalone single-factor filters:')
print('{:42s} {:>4s} {:>8s} {:>8s} {:>5s} {:>6s} {:>6s} {:>9s} {:>9s}'.format(
    'Factor', 'Dir', 'Thresh', 'Score', 'Hits', 'BgBlk', 'BlBlk', 'Bg/AllBg', 'Bl/AllBl'))
for _, r in solo_df.head(20).iterrows():
    print('{:42s} {:>4s} {:8.3f} {:8.3f} {:5d} {:6d} {:6d} {:8.1%} {:8.1%}'.format(
        r['factor'][:42], r['dir'], r['thresh'], r['score'],
        r['hits'], r['bg_blocked'], r['bl_blocked'],
        r['bg_blocked'] / n_bg, r['bl_blocked'] / n_bl))

# ========================================================================
# PART 2: Relaxed runup_ma5_exhaustion thresholds + best second factor
# ========================================================================
print('\n' + '=' * 90)
print('PART 2: Relaxed runup_ma5_exhaustion base + second factor')
print('=' * 90)

for runup_thresh in [0.5, 0.6, 0.7, 0.8]:  # include 0.8 for comparison
    base = valid['runup_ma5_exhaustion'] > runup_thresh
    n_base = int(base.sum())
    n_bg_base = int((valid.loc[base, 'cls'] == 'bg').sum())
    n_bl_base = int((valid.loc[base, 'cls'] == 'bl').sum())
    if n_base < 5:
        continue
    print('\n  runup_ma5_exhaustion > {}: {} stocks in base, bg={}, bl={}'.format(
        runup_thresh, n_base, n_bg_base, n_bl_base))

    combo_results = []
    for factor in all_factors:
        if factor == 'runup_ma5_exhaustion':
            continue
        fs = valid[factor].dropna()
        if len(fs) < 10 or fs.nunique() < 3:
            continue
        q_low, q_high = fs.quantile(0.05), fs.quantile(0.95)
        if q_high <= q_low or np.isnan(q_low) or np.isnan(q_high):
            continue
        for thresh in np.linspace(q_low, q_high, 25):
            for d in ['>=', '<=']:
                cond = (valid[factor] >= thresh) if d == '>=' else (valid[factor] <= thresh)
                blocked = base & cond.fillna(False)
                hits = int(blocked.sum())
                if hits < 2:
                    continue
                bg_b = int((valid.loc[blocked, 'cls'] == 'bg').sum())
                bl_b = int((valid.loc[blocked, 'cls'] == 'bl').sum())
                score = -bg_b / n_bg * 10 + bl_b / n_bl * 3 - hits / n_all * 0.2
                combo_results.append({
                    'factor': factor, 'thresh': thresh, 'dir': d,
                    'score': score, 'hits': hits,
                    'bg_blocked': bg_b, 'bl_blocked': bl_b,
                })
    combo_df = pd.DataFrame(combo_results).sort_values('score', ascending=False)
    print('  Top 6 second factors:')
    for _, r in combo_df.head(6).iterrows():
        print('    {:38s} {:>4s} {:7.3f}  score={:7.3f}  hits={:3d}  bg_blk={}/{}  bl_blk={}/{}'.format(
            r['factor'][:38], r['dir'], r['thresh'], r['score'],
            r['hits'], r['bg_blocked'], n_bg, r['bl_blocked'], n_bl))

# ========================================================================
# PART 3: Two-factor AND combinations (from top standalone factors)
# ========================================================================
print('\n' + '=' * 90)
print('PART 3: Best two-factor AND combinations')
print('=' * 90)

top_solo_factors = list(solo_df.head(30)['factor'].unique())
print('Pairing top {} standalone factors ({} combinations)...'.format(
    len(top_solo_factors), len(top_solo_factors) * (len(top_solo_factors) - 1) // 2 * 9 * 9 * 4))

pair_results = []
for f1, f2 in itertools.combinations(top_solo_factors, 2):
    if f1 not in valid.columns or f2 not in valid.columns:
        continue
    for t1_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for t2_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for d1, d2 in [('>=', '>='), ('>=', '<='), ('<=', '>='), ('<=', '<=')]:
                t1 = valid[f1].quantile(t1_p)
                t2 = valid[f2].quantile(t2_p)
                if pd.isna(t1) or pd.isna(t2) or t1 == t2:
                    continue
                c1 = (valid[f1] >= t1) if d1 == '>=' else (valid[f1] <= t1)
                c2 = (valid[f2] >= t2) if d2 == '>=' else (valid[f2] <= t2)
                blocked = c1.fillna(False) & c2.fillna(False)
                hits = int(blocked.sum())
                if hits < 3 or hits > n_all * 0.4:
                    continue
                bg_b = int((valid.loc[blocked, 'cls'] == 'bg').sum())
                bl_b = int((valid.loc[blocked, 'cls'] == 'bl').sum())
                score = -bg_b / n_bg * 10 + bl_b / n_bl * 3 - hits / n_all * 0.5
                pair_results.append({
                    'f1': f1, 'd1': d1, 't1_p': t1_p, 't1_val': t1,
                    'f2': f2, 'd2': d2, 't2_p': t2_p, 't2_val': t2,
                    'score': score, 'hits': hits,
                    'bg_blocked': bg_b, 'bl_blocked': bl_b,
                })

pair_df = pd.DataFrame(pair_results).sort_values('score', ascending=False)
print('Top 15 two-factor AND combinations:')
for _, r in pair_df.head(15).iterrows():
    print('  {:30s} {:>4s} Q{:3.0f}%({:7.3f})  AND  {:30s} {:>4s} Q{:3.0f}%({:7.3f})'.format(
        r['f1'][:30], r['d1'], r['t1_p'] * 100, r['t1_val'],
        r['f2'][:30], r['d2'], r['t2_p'] * 100, r['t2_val']))
    print('    score={:7.3f}  hits={:3d}  bg_blk={}/{}  bl_blk={}/{}'.format(
        r['score'], r['hits'], r['bg_blocked'], n_bg, r['bl_blocked'], n_bl))

# ========================================================================
# PART 4: Current rules reference
# ========================================================================
print('\n' + '=' * 90)
print('PART 4: Current filter rules for reference')
print('=' * 90)

rules = {
    'R1(runup>0.8 & top10>=0.5)': (valid['runup_ma5_exhaustion'] > 0.8) & (valid['top10_10d_strength_level_v3'] >= 0.5),
    'R2(blowoff>=4.15 & sector<=0.41)': (valid['prev_day_blowoff_risk'] >= 4.15) & (valid['sector_momentum_accel_rank'] <= 0.41),
    'R3(market_ad>=5.39559)': valid['market_advance_decline_ratio'] >= 5.39559,
    'R4(prior_vol_contest<=-0.17 & vol_ratio>=1.78)': (valid['prior_high_volume_contest_20d'] <= -0.170934) & (valid['stock_volume_ratio_20d'] >= 1.78077),
}

current_union = pd.Series(False, index=valid.index)
for name, rule in rules.items():
    blocked = rule.fillna(False)
    current_union = current_union | blocked
    bg_b = int((valid.loc[blocked, 'cls'] == 'bg').sum())
    bl_b = int((valid.loc[blocked, 'cls'] == 'bl').sum())
    mid_b = int((valid.loc[blocked, 'cls'] == 'mid').sum())
    print('  {}: hits={}, bg={}, bl={}, mid={}'.format(name, int(blocked.sum()), bg_b, bl_b, mid_b))

bg_union = int((valid.loc[current_union, 'cls'] == 'bg').sum())
bl_union = int((valid.loc[current_union, 'cls'] == 'bl').sum())
print('  UNION: hits={}, bg={}/{}({:.1%}), bl={}/{}({:.1%})'.format(
    int(current_union.sum()), bg_union, n_bg, bg_union / n_bg, bl_union, n_bl, bl_union / n_bl))

print('\nTotal time: {:.1f}s'.format(time.time() - t0))
