# -*- coding: utf-8 -*-
import re, pandas as pd, numpy as np

def parse_log(fp):
    trades, cb = [], {}
    with open(fp, 'r', encoding='utf-8') as f:
        for l in f:
            bm = re.search(r'buy_date=(\d{4}-\d{2}-\d{2})', l)
            stock_m = re.search(r'(\d{6}\.(?:SH|SZ))', l)
            if bm and stock_m:
                cb[stock_m.group(1)] = bm.group(1)
            # Only match sell lines with exact format
            sm = re.search(r'卖出: (\d{6}\.(?:SH|SZ)).*?收益: (-?\d+\.?\d*) \((-?\d+\.?\d*)%\)', l)
            if sm:
                s = sm.group(1)
                ret = float(sm.group(3))
                bd = cb.pop(s, None)
                if bd:
                    trades.append(dict(stock_code=s, buy_date=bd, return_pct=ret, trade_type='real'))
    return pd.DataFrame(trades)

d172 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子172_optuna.log')
print('Real:', len(d172))

shadow = pd.read_csv('./results/walkforward_shadow_skipped_opportunities_20260513_230111.csv')
st = []
for _, r in shadow.iterrows():
    st.append(dict(stock_code=r['stock_code'], buy_date=r['buy_datetime'][:10],
                   return_pct=r['actual_return'], trade_type='shadow',
                   exhaustion_blocked=r['exhaustion_blocked']))
sf = pd.DataFrame(st)
print('Shadow:', len(sf))

allt = pd.concat([d172, sf], ignore_index=True)
print('Total:', len(allt))

# Match factors
df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

F = ['market_trend_strength','mkt_liquidity_impulse_v2','market_60d_new_high_ratio',
     'market_liquidity_change','money_flow_pressure_pct',
     'runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
     'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z']

rows = []
for _, t in allt.iterrows():
    s = t['stock_code']
    bd = pd.Timestamp(t['buy_date']) if pd.notna(t['buy_date']) else None
    if bd is None: continue
    sd = df[df['stock_code']==s].sort_values('date')
    before = sd[sd['date'] < bd]
    if len(before) == 0: continue
    pd_date = before['date'].max()
    r = sd[sd['date'] == pd_date].iloc[0]
    rec = dict(stock_code=s, buy_date=str(t['buy_date']), return_pct=t['return_pct'], trade_type=t['trade_type'])
    if 'exhaustion_blocked' in t: rec['exhaustion_blocked'] = t['exhaustion_blocked']
    for f in F:
        rec[f] = r.get(f, np.nan)
    rows.append(rec)

rdf = pd.DataFrame(rows)
print('Matched:', len(rdf))

# Apply rules
rdf['V3_R1'] = (rdf['runup_ma5_exhaustion'] > 0.8) & (rdf['cpv_factor'] <= -0.4)
rdf['V3_R2'] = (rdf['prev_day_blowoff_risk'] >= 2.0) & (rdf['sector_momentum_accel_rank'] <= 0.30)
rdf['V3_R3'] = (rdf['sector_leadership_persistence_v3'] <= -0.18) & (rdf['return_20d_intra_z'] <= 0.40)
rdf['V3_any'] = rdf['V3_R1'] | rdf['V3_R2'] | rdf['V3_R3']
rdf['R4'] = (rdf['market_liquidity_change'] < 0.95) & (rdf['money_flow_pressure_pct'] < 0.30)
rdf['V3R4'] = rdf['V3_any'] | rdf['R4']

# R5 configs
R5 = [
    ('R5_cur',  0,    0.90, 0.030),
    ('R5_t1',   0,    0.85, 0.025),
    ('R5_t2',   0,    0.85, 0.030),
    ('R5_t3',   0,    0.90, 0.025),
    ('R5_t4',  -0.01, 0.90, 0.030),
]

for name, tr_th, liq_th, nh_th in R5:
    rdf[name] = (rdf['market_trend_strength'] < tr_th) & \
                 (rdf['mkt_liquidity_impulse_v2'] < liq_th) & \
                 (rdf['market_60d_new_high_ratio'] < nh_th)

bl = rdf[rdf['return_pct'] < -5]
bw = rdf[rdf['return_pct'] > 10]
aw = rdf[rdf['return_pct'] > 0]
print('\nBigLoss: %d  BigWin: %d  AllWin: %d' % (len(bl), len(bw), len(aw)))

# 1. Rule performance
print('\n=== Rule Performance ===')
fmt = '%-25s %4s/%-2s %4s/%-2s %4s/%-2s %5s'
print(fmt % ('Rule', 'Catch', 'BL', 'FPbw', 'BW', 'FPaw', 'AW', 'Total'))
print('-'*75)
rules = [
    ('V3 (R1+R2+R3)', rdf['V3_any']),
    ('R4', rdf['R4']),
    ('V3+R4', rdf['V3R4']),
] + [(n, rdf[n]) for n,_,_,_ in R5]

for n, mask in rules:
    c = (mask & (rdf['return_pct']<-5)).sum()
    fpb = (mask & (rdf['return_pct']>10)).sum()
    fpa = (mask & (rdf['return_pct']>0)).sum()
    t = mask.sum()
    print(fmt % (n, c, len(bl), fpb, len(bw), fpa, len(aw), t))

# 2. Overlap
print('\n=== Overlap with V3+R4 ===')
for name,_,_,_ in R5:
    mask = rdf[name]
    ov = (mask & rdf['V3R4']).sum()
    uniq = mask.sum() - ov
    c = (mask & (rdf['return_pct']<-5)).sum()
    fpb = (mask & (rdf['return_pct']>10)).sum()
    print('%s: total=%d overlap=%d unique=%d catch=%d fp=%d' % (name, mask.sum(), ov, uniq, c, fpb))

# 3. R5_cur caught trades
print('\n=== R5_cur Caught Trades ===')
caught = rdf[rdf['R5_cur']].sort_values('return_pct')
for _, r in caught.iterrows():
    tag = 'BL' if r['return_pct']<-5 else ('BW' if r['return_pct']>10 else ('W' if r['return_pct']>0 else 'L'))
    ex = ''
    if r['V3_any']: ex += 'V3'
    if r['R4']: ex += 'R4'
    ex = '[%s]' % ex if ex else '[NEW]'
    print('  %6s %s buy=%s ret=%+5.1f%% %s %s trend=%.3f liq=%.3f nh=%.4f' % (
        r['trade_type'], r['stock_code'], r['buy_date'], r['return_pct'], tag, ex,
        r['market_trend_strength'], r['mkt_liquidity_impulse_v2'], r['market_60d_new_high_ratio']))

# 4. DD streak
print('\n=== DD Streak (2026-03-31 to 2026-04-08) ===')
dd = rdf[(rdf['buy_date'] >= '2026-03-27') & (rdf['buy_date'] <= '2026-04-10')].sort_values('buy_date')
for _, r in dd.iterrows():
    hit = ''
    if r['R5_cur']: hit = 'R5!'
    if r['V3_any']: hit += 'V3'
    if r['R4']: hit += 'R4'
    print('  %4s %s buy=%s ret=%+5.1f%%' % (hit, r['stock_code'], r['buy_date'], r['return_pct']))

# 5. Incremental
print('\n=== Incremental: V3+R4 vs V3+R4+R5_cur ===')
for label, mask in [('V3+R4', rdf['V3R4']), ('V3+R4+R5_cur', rdf['V3R4'] | rdf['R5_cur'])]:
    c = (mask & (rdf['return_pct']<-5)).sum()
    fpb = (mask & (rdf['return_pct']>10)).sum()
    fpa = (mask & (rdf['return_pct']>0)).sum()
    t = mask.sum()
    print('%s: catch=%d/%d fp_bw=%d/%d fp_aw=%d/%d total=%d' % (label, c, len(bl), fpb, len(bw), fpa, len(aw), t))
