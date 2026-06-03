"""Consolidated R4 + C5 + C8: liq-based rules, 2-decimal robustness"""
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
                s=m2.group(1); ret=float(m2.group(3)); bd=cb.pop(s,None)
                if bd: trades.append(dict(s=s,bd=bd,ret=ret,src='real'))
    return pd.DataFrame(trades)

d162 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log')
shadow = pd.read_csv('./results/walkforward_shadow_skipped_opportunities_20260513_230111.csv')
all_t = []
for _,r in d162.iterrows(): all_t.append(dict(s=r['s'],bd=r['bd'],ret=r['ret'],src='real'))
for _,r in shadow.iterrows(): all_t.append(dict(s=r['stock_code'],bd=r['buy_datetime'][:10],ret=r['actual_return']*100,src='shadow'))
at = pd.DataFrame(all_t)

df = pd.read_parquet('./factor_lab_data/factors.parquet'); df['date']=pd.to_datetime(df['date'])

need = ['market_liquidity_change','money_flow_pressure_pct',
    'momentum_peak_decay_3d','ulcer_index',
    'runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
    'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z',
    'mkt_dispersion_shock_v2','market_advance_decline_ratio']

rows = []
for _,t in at.iterrows():
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < pd.Timestamp(t['bd'])]
    if len(before)==0: continue
    r = sd[sd['date']==before['date'].max()].iloc[0]
    rec = dict(s=t['s'],bd=t['bd'],ret=t['ret'],src=t['src'])
    for f in need: rec[f] = r.get(f,np.nan)
    rows.append(rec)
rdf = pd.DataFrame(rows)
BLN = len(rdf[rdf.ret<-5]); BWN = len(rdf[rdf.ret>10]); AWN = len(rdf[rdf.ret>0])
print('%d trades, BL=%d, BW=%d, AW=%d' % (len(rdf), BLN, BWN, AWN))

# Existing rules
rdf['HR1']=(rdf['runup_ma5_exhaustion']>0.8)&(rdf['cpv_factor']<=-0.4)
rdf['HR2']=(rdf['prev_day_blowoff_risk']>=2.0)&(rdf['sector_momentum_accel_rank']<=0.30)|\
           (rdf['prev_day_blowoff_risk']>5.0)&(rdf['sector_momentum_accel_rank']<=0.50)|\
           (rdf['prev_day_blowoff_risk']>1.0)&(rdf['sector_momentum_accel_rank']<=0.15)
rdf['HR3']=(rdf['sector_leadership_persistence_v3']<=-0.18)&(rdf['return_20d_intra_z']<=0.40)
rdf['HR4']=(rdf['market_liquidity_change']<0.95)&(rdf['money_flow_pressure_pct']<0.78)
rdf['HR5']=(rdf['mkt_dispersion_shock_v2']<0.98)&(rdf['market_advance_decline_ratio']<0.38)
rdf['EXISTING']=rdf[['HR1','HR2','HR3','HR4','HR5']].any(axis=1)

# === 1. Factor percentiles (trade sample & full market) ===
print()
print('=== 1. Factor Percentiles (trade sample vs full market) ===')
for fname, th_list in [
    ('market_liquidity_change', [0.94, 0.95, 0.97]),
    ('ulcer_index', [-1.00, -0.90, -0.80, -0.70, -0.50, 0.00]),
    ('momentum_peak_decay_3d', [-5.00, -4.00, -3.00, -2.00, -1.00, 0.00]),
]:
    tv = rdf[fname].dropna(); fv = df[fname].dropna()
    print('  %s:' % fname)
    for th in th_list:
        tp = (tv < th).mean()*100; fpct = (fv < th).mean()*100
        mark = '<<' if abs(tp-fpct) < 5 else ''
        print('    <%.2f: trade=%.1f%% full=%.1f%% %s' % (th, tp, fpct, mark))

# === 2. Consolidated liq scan: liq < [th] & (mfp<0.78 | ulcer<Z) ===
print()
print('=== 2. Consolidated Scan: liq<X & (mfp<0.78 | ulcer<Z) ===')
print('%-45s %4s %4s %4s %4s %4s %4s' % ('Rule','C','FB','FA','Tot','New','Shd'))
print('-'*70)

for liq_th in [0.94, 0.95, 0.97, 0.99]:
    for ul_th in [-1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, 0.00]:
        # R4-extended: mfp OR ulcer
        mask = (rdf['market_liquidity_change']<liq_th) & \
               ((rdf['money_flow_pressure_pct']<0.78) | (rdf['ulcer_index']<ul_th))
        c = (mask&(rdf.ret<-5)).sum(); fb = (mask&(rdf.ret>10)).sum()
        fa = (mask&(rdf.ret>0)).sum(); t = mask.sum()
        nw = (mask&~rdf['EXISTING']&(rdf.ret<-5)).sum()
        sh = mask[rdf.src=='shadow'].sum()
        if c >= 5 and fb == 0:
            desc = 'liq<%.2f & (mfp<0.78|ulcer<%.2f)'%(liq_th,ul_th)
            print('%-45s %2d/%d %2d/%d %2d/%d %4d %4d %4d' % (desc,c,BLN,fb,BWN,fa,AWN,t,nw,sh))

# Also try mom_peak as alternative to ulcer
print()
print('=== 3. Consolidation with mom_peak ===')
for liq_th in [0.94, 0.95, 0.97]:
    for mp_th in [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0]:
        mask = (rdf['market_liquidity_change']<liq_th) & \
               ((rdf['money_flow_pressure_pct']<0.78) | (rdf['momentum_peak_decay_3d']>mp_th))
        c = (mask&(rdf.ret<-5)).sum(); fb = (mask&(rdf.ret>10)).sum()
        fa = (mask&(rdf.ret>0)).sum(); t = mask.sum()
        nw = (mask&~rdf['EXISTING']&(rdf.ret<-5)).sum()
        sh = mask[rdf.src=='shadow'].sum()
        if c >= 7 and fb == 0:
            desc = 'liq<%.2f & (mfp<0.78|mom_peak>%+.1f)'%(liq_th,mp_th)
            print('%-45s %2d/%d %2d/%d %2d/%d %4d %4d %4d' % (desc,c,BLN,fb,BWN,fa,AWN,t,nw,sh))

# === 4. Best pick: detailed breakdown ===
print()
print('=== 4. Best: liq<0.95 & (mfp<0.78 | ulcer<-0.80) ===')
best = (rdf['market_liquidity_change']<0.95) & ((rdf['money_flow_pressure_pct']<0.78)|(rdf['ulcer_index']<-0.80))
print('All caught trades:')
for _, r in rdf[best].sort_values('ret').iterrows():
    t='BL' if r['ret']<-5 else ('BW' if r['ret']>10 else ('W' if r['ret']>0 else 'L'))
    ex = 'Y' if r['EXISTING'] else 'NEW'
    wp = []
    if r['money_flow_pressure_pct']<0.78: wp.append('mfp')
    if r['ulcer_index']<-0.80: wp.append('ulcer')
    print('  %-5s %s buy=%s ret=%+5.1f%% [%s] %s by=%s' % (
        r['src'], r['s'], r['bd'], r['ret'], t, ex, '+'.join(wp)))

# Compare: old R4 vs new R4-extended
print()
c_old = (rdf['HR4']&(rdf.ret<-5)).sum()
fb_old = (rdf['HR4']&(rdf.ret>10)).sum()
c_new = (best&(rdf.ret<-5)).sum()
fb_new = (best&(rdf.ret>10)).sum()
print('R4 old (mfp<0.78): c=%d fb=%d' % (c_old, fb_old))
print('R4 new (+ulcer<-0.80): c=%d fb=%d' % (c_new, fb_new))
print('Incremental BL: %d, Incremental FB: %d' % (c_new-c_old, fb_new-fb_old))
