"""Soft penalty simulation on 162 baseline"""
import re, pandas as pd, numpy as np

def parse_log(fp):
    trades, cb = [], {}
    with open(fp, 'r', encoding='utf-8') as f:
        for l in f:
            bm = re.search(r'buy_date=(\d{4}-\d{2}-\d{2})', l)
            sm = re.search(r'(\d{6}\.(?:SH|SZ))', l)
            if bm and sm: cb[sm.group(1)] = bm.group(1)
            m2 = re.search(r'(\d{6}\.(?:SH|SZ)).*? (-?\d+\.\d+) \((-?\d+\.\d+)%\)', l)
            if m2 and any(kw in l for kw in ['T+2','T+3','利润','到期','累计','走坏','未转正']):
                s = m2.group(1); ret = float(m2.group(3)); bd = cb.pop(s, None)
                if bd: trades.append(dict(s=s, bd=bd, ret=ret, src='real'))
    return pd.DataFrame(trades)

d162 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log')
df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

F = ['runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
     'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z',
     'market_liquidity_change','money_flow_pressure_pct',
     'mkt_dispersion_shock_v2','market_advance_decline_ratio']

rows = []
for _,t in d162.iterrows():
    bd = pd.Timestamp(t['bd']); sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < bd]
    if len(before) == 0: continue
    pd_date = before['date'].max(); r = sd[sd['date']==pd_date].iloc[0]
    rec = dict(s=t['s'], bd=str(t['bd']), ret=t['ret'])
    for f in F: rec[f] = r.get(f, np.nan)
    rows.append(rec)

rdf = pd.DataFrame(rows)
print('Matched: %d trades' % len(rdf))

# ============================================================
# 1. Compute feature stats (P80-P50 for temperature, P50 for threshold check)
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def robust_temp(feature_vals):
    """Temperature = P80 - P50, robust to outliers"""
    vals = feature_vals.dropna()
    p80 = vals.quantile(0.80)
    p50 = vals.quantile(0.50)
    return max(p80 - p50, 0.001)

def compute_risk(stock, stats):
    risk = 0.0
    details = {}

    # R1: runup exhaustion (x > 0.8) AND cpv collapse (x <= -0.4)
    z1 = (stock['runup_ma5_exhaustion'] - 0.8) / stats['runup_temp']
    z2 = (-stock['cpv_factor'] - 0.4) / stats['cpv_temp']
    s1 = sigmoid(z1) * sigmoid(z2)
    risk += 1.0 * s1
    details['R1'] = s1

    # R2: blowoff (x >= 2.0) AND sector deceleration (x <= 0.30)
    z3 = (stock['prev_day_blowoff_risk'] - 2.0) / stats['blowoff_temp']
    z4 = (-stock['sector_momentum_accel_rank'] + 0.30) / stats['sector_accel_temp']
    s2 = sigmoid(z3) * sigmoid(z4)
    risk += 0.6 * s2
    details['R2'] = s2

    # R3: sector leadership collapse (x <= -0.18) AND weak rel strength (x <= 0.40)
    z5 = (-stock['sector_leadership_persistence_v3'] - 0.18) / stats['sl_temp']
    z6 = (-stock['return_20d_intra_z'] + 0.40) / stats['ret20z_temp']
    s3 = sigmoid(z5) * sigmoid(z6)
    risk += 0.6 * s3
    details['R3'] = s3

    # R4: liquidity stress (x < 0.95) AND weak money flow (x < 0.78)
    z7 = (0.95 - stock['market_liquidity_change']) / stats['liq_temp']
    z8 = (0.78 - stock['money_flow_pressure_pct']) / stats['mfp_temp']
    s4 = sigmoid(z7) * sigmoid(z8)
    risk += 0.6 * s4
    details['R4'] = s4

    return risk, details

# Compute temperatures from trade sample
stats = {}
temp_source = {
    'runup_temp': 'runup_ma5_exhaustion', 'cpv_temp': 'cpv_factor',
    'blowoff_temp': 'prev_day_blowoff_risk', 'sector_accel_temp': 'sector_momentum_accel_rank',
    'sl_temp': 'sector_leadership_persistence_v3', 'ret20z_temp': 'return_20d_intra_z',
    'liq_temp': 'market_liquidity_change', 'mfp_temp': 'money_flow_pressure_pct',
}
for key, col in temp_source.items():
    stats[key] = robust_temp(rdf[col])

print('\n=== Temperature parameters (P80-P50) ===')
for k, v in stats.items():
    print('  %s = %.4f' % (k, v))

# ============================================================
# 2. Compute risk scores for all trades
# ============================================================
risks = []
details_r1 = []; details_r2 = []; details_r3 = []; details_r4 = []
for _, r in rdf.iterrows():
    risk, detail = compute_risk(r, stats)
    risks.append(risk)
    details_r1.append(detail['R1']); details_r2.append(detail['R2'])
    details_r3.append(detail['R3']); details_r4.append(detail['R4'])

rdf['risk_score'] = risks
rdf['risk_R1'] = details_r1; rdf['risk_R2'] = details_r2
rdf['risk_R3'] = details_r3; rdf['risk_R4'] = details_r4

# Hard rules
rdf['R1_hard'] = (rdf['runup_ma5_exhaustion']>0.8)&(rdf['cpv_factor']<=-0.4)
rdf['R2_hard'] = (rdf['prev_day_blowoff_risk']>=2.0)&(rdf['sector_momentum_accel_rank']<=0.30)
rdf['R3_hard'] = (rdf['sector_leadership_persistence_v3']<=-0.18)&(rdf['return_20d_intra_z']<=0.40)
rdf['R4_hard'] = (rdf['market_liquidity_change']<0.95)&(rdf['money_flow_pressure_pct']<0.78)
rdf['R5_hard'] = (rdf['mkt_dispersion_shock_v2']<0.98)&(rdf['market_advance_decline_ratio']<0.38)
rdf['any_hard'] = rdf[['R1_hard','R2_hard','R3_hard','R4_hard','R5_hard']].any(axis=1)

# ============================================================
# 3. Risk score distribution: big loss vs big win
# ============================================================
bl = rdf[rdf.ret < -5]
bw = rdf[rdf.ret > 10]

print('\n=== Risk Score Distribution ===')
print('BigLoss:  mean=%.3f  median=%.3f  max=%.3f' % (bl['risk_score'].mean(), bl['risk_score'].median(), bl['risk_score'].max()))
print('BigWin:   mean=%.3f  median=%.3f  max=%.3f' % (bw['risk_score'].mean(), bw['risk_score'].median(), bw['risk_score'].max()))
print('All:      mean=%.3f  median=%.3f' % (rdf['risk_score'].mean(), rdf['risk_score'].median()))

# ============================================================
# 4. Near-miss analysis: hard rule misses, soft catches
# ============================================================
print('\n=== Near-Miss Analysis (hard=0, soft>0.1) ===')
near = rdf[(~rdf['any_hard']) & (rdf['risk_score'] > 0.1)].sort_values('risk_score', ascending=False)
for _, r in near.iterrows():
    tag = 'BL' if r['ret']<-5 else ('BW' if r['ret']>10 else ('W' if r['ret']>0 else 'L'))
    components = []
    if r['risk_R1'] > 0.05: components.append('R1=%.3f'%r['risk_R1'])
    if r['risk_R2'] > 0.05: components.append('R2=%.3f'%r['risk_R2'])
    if r['risk_R3'] > 0.05: components.append('R3=%.3f'%r['risk_R3'])
    if r['risk_R4'] > 0.05: components.append('R4=%.3f'%r['risk_R4'])
    print('  %s buy=%s ret=%+5.1f%% [%s] risk=%.3f  %s' % (
        r['s'], r['bd'], r['ret'], tag, r['risk_score'], ', '.join(components)))

# ============================================================
# 5. Lambda grid search: multiplicative decay
# ============================================================
print('\n=== Lambda Grid Search (alpha * exp(-lambda * risk)) ===')
print('假设: alpha 与 predicted_return 成正比, 只看风险惩罚如何改变排序')
print()
fmt = '%-8s %5s %5s %5s %5s %5s %5s %5s %5s'
print(fmt % ('lambda','<0.1','0.1+','0.2+','0.3+','0.5+','0.7+','1.0+','2.0+'))
print('-'*55)

for lam in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    decay = np.exp(-lam * rdf['risk_score'])
    counts = [(decay < 0.99).sum(), (decay < 0.90).sum(), (decay < 0.80).sum(),
              (decay < 0.70).sum(), (decay < 0.50).sum(), (decay < 0.40).sum(),
              (decay < 0.30).sum(), (decay < 0.20).sum()]
    # How many big losses would get meaningful penalty?
    bl_decay = np.exp(-lam * bl['risk_score'])
    bl_penalized = (bl_decay < 0.80).sum()
    # How many big wins would get penalty?
    bw_decay = np.exp(-lam * bw['risk_score'])
    bw_penalized = (bw_decay < 0.80).sum()
    print(fmt % (('%.1f'%lam), counts[0], counts[1], counts[2], counts[3], counts[4],
                  counts[5], counts[6], counts[7]))
    print('  → BL with decay<0.8: %d/%d, BW with decay<0.8: %d/%d' % (
        bl_penalized, len(bl), bw_penalized, len(bw)))

# ============================================================
# 6. Show top near-miss big losses with their risk components
# ============================================================
print('\n=== Top Near-Miss Big Losses (hard=0, sorted by risk_score) ===')
near_bl = rdf[(~rdf['any_hard']) & (rdf['ret']<-5)].sort_values('risk_score', ascending=False)
for _, r in near_bl.iterrows():
    print('  %s buy=%s ret=%+5.1f%% risk=%.3f R1=%.3f R2=%.3f R3=%.3f R4=%.3f' % (
        r['s'], r['bd'], r['ret'], r['risk_score'],
        r['risk_R1'], r['risk_R2'], r['risk_R3'], r['risk_R4']))

# ============================================================
# 7. Key case: 603618 at lambda=0.5
# ============================================================
print('\n=== Key Cases: Risk Penalty Effect at lambda=0.5 ===')
key_stocks = [
    ('603618.SH', '2026-04-07', 'cpv bare miss'),  # runup=2.29, cpv=0.01
    ('300834.SZ', '2026-04-08', 'mfp max miss'),    # mfp=1.00
    ('603738.SH', '2026-04-29', 'multi near-miss'),
    ('300589.SZ', '2025-12-03', 'R4 near-miss mfp=0.01'),
]
for s, bd, note in key_stocks:
    row = rdf[(rdf['s']==s) & (rdf['bd']==bd)]
    if len(row) == 0: continue
    r = row.iloc[0]
    decay = np.exp(-0.5 * r['risk_score'])
    hard = 'BLOCK' if r['any_hard'] else 'PASS'
    print('  %s %s (%s):' % (s, bd, note))
    print('    ret=%+.1f%% hard=%s risk=%.3f decay=%.3f R1=%.3f R2=%.3f R3=%.3f R4=%.3f' % (
        r['ret'], hard, r['risk_score'], decay, r['risk_R1'], r['risk_R2'], r['risk_R3'], r['risk_R4']))
    print('    runup=%.2f cpv=%.2f blow=%.1f s_acc=%.2f sl=%.2f r20z=%.2f liq=%.2f mfp=%.2f' % (
        r['runup_ma5_exhaustion'], r['cpv_factor'], r['prev_day_blowoff_risk'],
        r['sector_momentum_accel_rank'], r['sector_leadership_persistence_v3'],
        r['return_20d_intra_z'], r['market_liquidity_change'], r['money_flow_pressure_pct']))
