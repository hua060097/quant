"""Relaxed rule scan: all 5 rules, shadow test, robustness"""
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

F = ['runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
     'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z',
     'market_liquidity_change','money_flow_pressure_pct',
     'mkt_dispersion_shock_v2','market_advance_decline_ratio']

rows = []
for _,t in at.iterrows():
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < pd.Timestamp(t['bd'])]
    if len(before)==0: continue
    r = sd[sd['date']==before['date'].max()].iloc[0]
    rec = dict(s=t['s'],bd=t['bd'],ret=t['ret'],src=t['src'])
    for f in F: rec[f] = r.get(f,np.nan)
    rows.append(rec)
rdf = pd.DataFrame(rows)
BLN = len(rdf[rdf.ret<-5]); BWN = len(rdf[rdf.ret>10])

# Hard rules
rdf['HR1']=(rdf['runup_ma5_exhaustion']>0.8)&(rdf['cpv_factor']<=-0.4)
rdf['HR2']=(rdf['prev_day_blowoff_risk']>=2.0)&(rdf['sector_momentum_accel_rank']<=0.30)
rdf['HR3']=(rdf['sector_leadership_persistence_v3']<=-0.18)&(rdf['return_20d_intra_z']<=0.40)
rdf['HR4']=(rdf['market_liquidity_change']<0.95)&(rdf['money_flow_pressure_pct']<0.78)
rdf['HR5']=(rdf['mkt_dispersion_shock_v2']<0.98)&(rdf['market_advance_decline_ratio']<0.38)
rdf['hard'] = rdf[['HR1','HR2','HR3','HR4','HR5']].any(axis=1)

# === 1. XR2a/b on shadow ===
rdf['XR2a'] = (rdf['prev_day_blowoff_risk']>5.0)&(rdf['sector_momentum_accel_rank']<=0.50)
rdf['XR2b'] = (rdf['prev_day_blowoff_risk']>1.0)&(rdf['sector_momentum_accel_rank']<=0.15)

print('=== 1. XR2a/b on Shadow ===')
for name, col in [('XR2a', 'XR2a'), ('XR2b', 'XR2b')]:
    hits = rdf[(rdf.src=='shadow') & rdf[col]]
    print('%s: %d shadow hits' % (name, len(hits)))
    for _,r in hits.sort_values('ret').iterrows():
        t='BL' if r['ret']<-5 else ('BW' if r['ret']>10 else ('W' if r['ret']>0 else 'L'))
        print('  %s buy=%s ret=%+5.1f%% [%s] blow=%.1f s_acc=%.2f' % (
            r['s'],r['bd'],r['ret'],t,r['prev_day_blowoff_risk'],r['sector_momentum_accel_rank']))

# === 2. Relaxed scan for ALL 5 rules ===
print()
print('=== 2. Relaxed Candidates for ALL 5 Rules (zero FP on BW) ===')

rules = [
    ('R1','runup_ma5_exhaustion','>',0.8,'cpv_factor','<=',-0.4),
    ('R2','prev_day_blowoff_risk','>=',2.0,'sector_momentum_accel_rank','<=',0.30),
    ('R3','sector_leadership_persistence_v3','<=',-0.18,'return_20d_intra_z','<=',0.40),
    ('R4','market_liquidity_change','<',0.95,'money_flow_pressure_pct','<',0.78),
    ('R5','mkt_dispersion_shock_v2','<',0.98,'market_advance_decline_ratio','<',0.38),
]

def apply_cond(series, op, thresh):
    if op == '>': return series > thresh
    if op == '>=': return series >= thresh
    if op == '<': return series < thresh
    if op == '<=': return series <= thresh

def scan_relaxed(rdf, f1, op1, f2, op2, rname):
    f1v = rdf[f1].dropna()
    f2v = rdf[f2].dropna()
    found = []
    for p_ext in [0.05, 0.10, 0.15, 0.20]:
        for p_rel in [0.30, 0.35, 0.40, 0.45, 0.50]:
            # Case A: f1 extreme, f2 relaxed
            if op1 in ['>','>=']: t1e = f1v.quantile(1-p_ext)
            else: t1e = f1v.quantile(p_ext)
            if op2 in ['>','>=']: t2r = f2v.quantile(1-p_rel)
            else: t2r = f2v.quantile(p_rel)

            mask = apply_cond(rdf[f1],op1,t1e) & apply_cond(rdf[f2],op2,t2r)
            c = (mask & ~rdf['hard'] & (rdf.ret<-5)).sum()
            fp = (mask & (rdf.ret>10)).sum()
            if c >= 1 and fp == 0:
                found.append(('A:f1-ext+f2-rel', p_ext, p_rel, c, mask.sum(),
                    t1e, op1, t2r, op2))

            # Case B: f1 relaxed, f2 extreme
            if op1 in ['>','>=']: t1r = f1v.quantile(1-p_rel)
            else: t1r = f1v.quantile(p_rel)
            if op2 in ['>','>=']: t2e = f2v.quantile(1-p_ext)
            else: t2e = f2v.quantile(p_ext)

            mask = apply_cond(rdf[f1],op1,t1r) & apply_cond(rdf[f2],op2,t2e)
            c = (mask & ~rdf['hard'] & (rdf.ret<-5)).sum()
            fp = (mask & (rdf.ret>10)).sum()
            if c >= 1 and fp == 0:
                found.append(('B:f1-rel+f2-ext', p_ext, p_rel, c, mask.sum(),
                    t1r, op1, t2e, op2))

    found.sort(key=lambda x: (x[3], -x[4]))
    if found:
        print()
        print('%s:' % rname)
        seen_bl = set()
        for case, pe, pr, c, tot, t1, o1, t2, o2 in found[:3]:
            desc = '%s pE=%d%% pR=%d%% c=%d fp=0 t=%d' % (case, pe*100, pr*100, c, tot)
            mask = apply_cond(rdf[f1],o1,t1) & apply_cond(rdf[f2],o2,t2)
            new = rdf[mask & ~rdf['hard'] & (rdf.ret<-5)]
            for _,rr in new.iterrows():
                key = (rr['s'],rr['bd'])
                if key not in seen_bl:
                    print('  NEW: %s buy=%s ret=%+5.1f%% via %s (%s%s%.2f %s%s%.2f)' % (
                        rr['s'],rr['bd'],rr['ret'],case,rname,o1,t1,rname,o2,t2))
                    seen_bl.add(key)

scan_relaxed(rdf, 'runup_ma5_exhaustion','>','cpv_factor','<=','R1')
scan_relaxed(rdf, 'prev_day_blowoff_risk','>=','sector_momentum_accel_rank','<=','R2')
scan_relaxed(rdf, 'sector_leadership_persistence_v3','<=','return_20d_intra_z','<=','R3')
scan_relaxed(rdf, 'market_liquidity_change','<','money_flow_pressure_pct','<','R4')
scan_relaxed(rdf, 'mkt_dispersion_shock_v2','<','market_advance_decline_ratio','<','R5')

# === 3. Combined effect + overlap ===
print()
print('=== 3. Combined: hard + best relaxed ===')
# Best relaxed from scan above
rdf['XR1a'] = (rdf['runup_ma5_exhaustion']>1.2)&(rdf['cpv_factor']<=0.0)
rdf['XR2a'] = (rdf['prev_day_blowoff_risk']>5.0)&(rdf['sector_momentum_accel_rank']<=0.50)
rdf['XR2b'] = (rdf['prev_day_blowoff_risk']>1.0)&(rdf['sector_momentum_accel_rank']<=0.15)
rdf['XR5a'] = (rdf['mkt_dispersion_shock_v2']<0.90)&(rdf['market_advance_decline_ratio']<0.50)

# Build combined
new_rules = ['XR1a','XR2a','XR2b','XR5a']
rdf['ext'] = rdf['hard']
for nr in new_rules:
    rdf['ext'] = rdf['ext'] | rdf[nr]

# Show overlap: which relaxed rule catches which new BL
print()
print('New BLs caught by relaxed rules (not in hard):')
all_new = rdf[rdf['ext'] & ~rdf['hard'] & (rdf.ret<-5)].sort_values('ret')
for _,r in all_new.iterrows():
    which = []
    if r['XR1a']: which.append('XR1a')
    if r['XR2a']: which.append('XR2a')
    if r['XR2b']: which.append('XR2b')
    if r['XR5a']: which.append('XR5a')
    print('  %s buy=%s ret=%+5.1f%% [%s]' % (r['s'],r['bd'],r['ret'],'+'.join(which)))

# Final summary
print()
c_hard = (rdf['hard'] & (rdf.ret<-5)).sum()
c_ext = (rdf['ext'] & (rdf.ret<-5)).sum()
fp_h = (rdf['hard'] & (rdf.ret>10)).sum()
fp_e = (rdf['ext'] & (rdf.ret>10)).sum()
print('Hard:  catch=%d/%d fp=%d/%d' % (c_hard,BLN,fp_h,BWN))
print('+Ext:  catch=%d/%d fp=%d/%d' % (c_ext,BLN,fp_e,BWN))
