#!/usr/bin/env python3
"""Sensitivity analysis: test rule robustness by perturbing thresholds."""
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
needed = ['runup_ma5_exhaustion', 'top10_10d_strength_level_v3', 'cpv_factor',
          'sector_leadership_persistence_v3', 'prev_day_blowoff_risk',
          'sector_momentum_accel_rank', 'market_advance_decline_ratio',
          'buy_lg_amount_rate_pct', 'emotion_quality_balance_v3']
merged = merged.merge(factors[['date', 'stock_code'] + needed],
                      left_on=['date_dt', 'stock_code'], right_on=['date', 'stock_code'], how='left')


def classify(r):
    if pd.isna(r):
        return 'unknown'
    if r >= 5: return 'bg'
    if r >= -5: return 'mid'
    return 'bl'


merged['cls'] = merged['next_pct_chg'].apply(classify)
valid = merged[merged['cls'] != 'unknown']
n_all, n_bg, n_bl = len(valid), len(valid[valid.cls == 'bg']), len(valid[valid.cls == 'bl'])

def evaluate_rule(name, cond):
    cond = cond.fillna(False)
    hits = int(cond.sum())
    bg = int((valid.loc[cond, 'cls'] == 'bg').sum())
    bl = int((valid.loc[cond, 'cls'] == 'bl').sum())
    mid = int((valid.loc[cond, 'cls'] == 'mid').sum())
    # quality = bl/(bg+bl) - penalize killing bg, reward catching bl
    if bg > 0:
        quality = 'POOR(kills bg)'
    elif bl == 0:
        quality = 'USELESS(no bl)'
    else:
        quality = 'GOOD'
    return hits, bg, bl, mid, quality


# ===================================================
# Sensitivity analysis
# ===================================================
def analyze_sensitivity(rule_name, base_cond_func, param_name, param_values):
    """Test how rule performance changes when we vary a threshold."""
    print(f'\n{"="*90}')
    print(f'Threshold sensitivity: {rule_name}')
    print(f'{"="*90}')
    print(f'  {"Param":>12s} {"Hits":>6s} {"BgBlk":>6s} {"BlBlk":>6s} {"MidBlk":>6s} {"Verdict"}')
    for pv in param_values:
        cond = base_cond_func(pv)
        hits, bg, bl, mid, quality = evaluate_rule(rule_name, cond)
        marker = ' <-- BEST' if (bg == 0 and bl > 0) else ''
        print(f'  {pv:12.3f} {hits:6d} {bg:6d} {bl:6d} {mid:6d} {quality}{marker}')


# ===== R1_NEW: runup>0.8 & cpv_factor <= X =====
print('\n' + '#' * 90)
print('# R1_NEW: runup_ma5_exhaustion > 0.8 AND cpv_factor <= X')
print('# cpv_factor distribution on Top-1: mean={:.3f}, std={:.3f}, P5={:.3f}, P95={:.3f}'.format(
    valid['cpv_factor'].mean(), valid['cpv_factor'].std(),
    valid['cpv_factor'].quantile(0.05), valid['cpv_factor'].quantile(0.95)))
print('#' * 90)

base1 = valid['runup_ma5_exhaustion'] > 0.8
cpv_vals = np.arange(-0.8, 0.3, 0.1)
analyze_sensitivity('R1_new: runup>0.8 & cpv<=X',
                    lambda x: base1 & (valid['cpv_factor'] <= x),
                    'cpv<=X', cpv_vals)

# ===== R1_OLD: runup>0.8 & top10>=X =====
print('\n' + '#' * 90)
print('# R1_OLD: runup_ma5_exhaustion > 0.8 AND top10_10d_strength_level_v3 >= X')
print('#' * 90)
top10_vals = np.arange(0.0, 1.0, 0.1)
analyze_sensitivity('R1_old: runup>0.8 & top10>=X',
                    lambda x: base1 & (valid['top10_10d_strength_level_v3'] >= x),
                    'top10>=X', top10_vals)

# ===== R2_NEW: sector_leadership <= X =====
print('\n' + '#' * 90)
print('# R2_NEW: sector_leadership_persistence_v3 <= X (standalone)')
print('# distribution: mean={:.3f}, std={:.3f}, P5={:.3f}, P95={:.3f}'.format(
    valid['sector_leadership_persistence_v3'].mean(),
    valid['sector_leadership_persistence_v3'].std(),
    valid['sector_leadership_persistence_v3'].quantile(0.05),
    valid['sector_leadership_persistence_v3'].quantile(0.95)))
print('#' * 90)

sl_vals = np.arange(-0.5, 0.3, 0.05)
analyze_sensitivity('R2_new: sector_leadership<=X',
                    lambda x: valid['sector_leadership_persistence_v3'] <= x,
                    'sl<=X', sl_vals)

# ===== R2_OLD: blowoff>=X & sector_accel<=Y =====
print('\n' + '#' * 90)
print('# R2_OLD: prev_day_blowoff_risk >= X AND sector_momentum_accel_rank <= Y')
print('#' * 90)
for bf in [3.0, 3.5, 4.0, 4.15, 4.5, 5.0]:
    for sa in [0.3, 0.35, 0.41, 0.45, 0.5]:
        cond = (valid['prev_day_blowoff_risk'] >= bf) & (valid['sector_momentum_accel_rank'] <= sa)
        hits, bg, bl, mid, quality = evaluate_rule('', cond)
        marker = ' <-- ORIG' if (bf == 4.15 and sa == 0.41) else ''
        if hits > 1:
            print(f'  blowoff>={bf:.1f} & sector<={sa:.2f}: hits={hits}, bg={bg}, bl={bl}{marker}')

# ===== R3_OLD: market_ad >= X =====
print('\n' + '#' * 90)
print('# R3_OLD: market_advance_decline_ratio >= X')
print('#' * 90)
ad_vals = np.arange(3.0, 10.0, 0.5)
analyze_sensitivity('R3_old: market_ad>=X',
                    lambda x: valid['market_advance_decline_ratio'] >= x,
                    'ad>=X', ad_vals)

# ===== R3_NEW: buy_lg <= X & emotion <= Y =====
print('\n' + '#' * 90)
print('# R3_NEW: buy_lg_amount_rate_pct <= X AND emotion_quality_balance_v3 <= Y')
print('#' * 90)
for bl in [0.4, 0.5, 0.55, 0.598, 0.6, 0.65, 0.7]:
    for em in [0.002, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015]:
        cond = (valid['buy_lg_amount_rate_pct'] <= bl) & (valid['emotion_quality_balance_v3'] <= em)
        hits, bg, bl, mid, quality = evaluate_rule('', cond)
        if hits >= 2 and hits <= 50:
            marker = ' <-- FITTED' if (abs(bl - 0.598) < 0.01 and abs(em - 0.006) < 0.001) else ''
            print(f'  buy_lg<={bl:.2f} & em<={em:.3f}: hits={hits:3d}, bg={bg}, bl={bl}{marker}')

# ===== Summary =====
print('\n' + '#' * 90)
print('# ROBUSTNESS SUMMARY')
print('#' * 90)
print('A rule is ROBUST if nearby thresholds produce similar quality results.')
print('A rule is FRAGILE if small threshold changes cause large quality degradation.')
print('')
print('R1_old (runup+top10): top10>=0.5 -- check range 0.0~0.9')
print('R1_new (runup+cpv):   cpv<=-0.33    -- check range -0.8~+0.3')
print('R2_old (blowoff+sect): blowoff>=4.15, sector<=0.41')
print('R2_new (sector_lead): sl<=-0.15     -- check range -0.5~+0.3')
print('R3_old (market_ad):   ad>=5.39559   -- check range 3.0~10.0')
print('R3_new (buy_lg+em):   buy_lg<=0.598, em<=0.006')
