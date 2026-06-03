import re, pandas as pd, numpy as np

trades, cb = [], {}
with open('stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log', 'r', encoding='utf-8') as f:
    for l in f:
        bm = re.search(r'buy_date=(\d{4}-\d{2}-\d{2})', l)
        sm = re.search(r'(\d{6}\.(?:SH|SZ))', l)
        if bm and sm: cb[sm.group(1)] = bm.group(1)
        # Match sell lines: must contain both 'SH/SZ' AND '(' number '%)' AND one of the exit reason keywords
        m2 = re.search(r'(\d{6}\.(?:SH|SZ)).*? (-?\d+\.\d+) \((-?\d+\.\d+)%\)', l)
        if m2 and any(kw in l for kw in ['T+2','T+3','利润','到期','累计','走坏','未转正']):
            s = m2.group(1); ret = float(m2.group(3)); bd = cb.pop(s, None)
            if bd: trades.append(dict(s=s, bd=bd, ret=ret, src='real'))
d162 = pd.DataFrame(trades)

print('Real trades: %d' % len(d162))

shadow = pd.read_csv('./results/walkforward_shadow_skipped_opportunities_20260513_230111.csv')
all_t = []
for _,r in d162.iterrows():
    all_t.append(dict(s=r['s'], bd=r['bd'], ret=r['ret'], src='real'))
for _,r in shadow.iterrows():
    all_t.append(dict(s=r['stock_code'], bd=r['buy_datetime'][:10],
                      ret=r['actual_return']*100, src='shadow'))
at = pd.DataFrame(all_t)

df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

F = ['market_liquidity_change','money_flow_pressure_pct',
     'mkt_dispersion_shock_v2','market_advance_decline_ratio']

rows = []
for _,t in at.iterrows():
    bd = pd.Timestamp(t['bd'])
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < bd]
    if len(before) == 0: continue
    pd_date = before['date'].max()
    r = sd[sd['date']==pd_date].iloc[0]
    rec = dict(s=t['s'], bd=str(t['bd']), ret=t['ret'], src=t['src'])
    for f in F: rec[f] = r.get(f, np.nan)
    rows.append(rec)

rdf = pd.DataFrame(rows)
bl = rdf[rdf.ret<-5]; bw = rdf[rdf.ret>10]; aw = rdf[rdf.ret>0]
NBL = len(bl); NBW = len(bw); NAW = len(aw)
print('Total: %d (real=%d shadow=%d) BL=%d BW=%d AW=%d' % (
    len(rdf), len(rdf[rdf.src=='real']), len(rdf[rdf.src=='shadow']), NBL, NBW, NAW))

# ============================================================
# 1. Percentile Analysis
# ============================================================
print('\n' + '='*80)
print('1. Factor Percentiles (Trade Sample vs Full Market)')
print('='*80)

for fname, thresh, label in [
    ('market_liquidity_change', 0.95, 'R4 liq'),
    ('money_flow_pressure_pct', 0.78, 'R4 mfp'),
    ('mkt_dispersion_shock_v2', 0.977, 'MKT disp'),
    ('market_advance_decline_ratio', 0.387, 'MKT adr'),
]:
    tv = rdf[fname].dropna()
    fv = df[fname].dropna()
    t_pct = (tv < thresh).mean() * 100
    f_pct = (fv < thresh).mean() * 100
    print('\n%s < %.3f' % (fname, thresh))
    print('  Trade: %.1f%%, Market: %.1f%%' % (t_pct, f_pct))
    qs = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
    print('  Trade q: ' + ' '.join(['p%d=%.4f'%(int(q*100),tv.quantile(q)) for q in qs]))
    print('  Mkt q:   ' + ' '.join(['p%d=%.4f'%(int(q*100),fv.quantile(q)) for q in qs]))

# ============================================================
# 2. R4 mfp_pct Scan
# ============================================================
print('\n' + '='*80)
print('2. R4: liq<0.95 & mfp_pct < [th]  Robustness')
print('='*80)

fmt = '%-8s %3s/%-2s %3s/%-2s %3s/%-2s %4s %4s %4s %s'
print(fmt % ('mfp_th','C','BL','FP','BW','FA','AW','Tot','sBL','sFP',''))
print('-'*65)

for mfp_th in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90, 0.95]:
    mask = (rdf['market_liquidity_change']<0.95) & (rdf['money_flow_pressure_pct']<mfp_th)
    c = (mask & (rdf.ret<-5)).sum()
    fp = (mask & (rdf.ret>10)).sum()
    fpa = (mask & (rdf.ret>0)).sum()
    t = mask.sum()
    sc = (mask & (rdf.src=='shadow') & (rdf.ret<-5)).sum()
    sfp = (mask & (rdf.src=='shadow') & (rdf.ret>10)).sum()
    mark = 'ZERO_FP' if fp==0 else ''
    print(fmt % (('%.2f'%mfp_th), c, NBL, fp, NBW, fpa, NAW, t, sc, sfp, mark))

# ============================================================
# 3. MKT disp shock scan
# ============================================================
print('\n' + '='*80)
print('3. MKT: disp_shock < [th] & adr < 0.387')
print('='*80)

for disp_th in [0.90, 0.93, 0.95, 0.96, 0.97, 0.977, 0.98, 0.99, 1.00, 1.02, 1.05]:
    mask = (rdf['mkt_dispersion_shock_v2']<disp_th) & (rdf['market_advance_decline_ratio']<0.387)
    c = (mask & (rdf.ret<-5)).sum()
    fp = (mask & (rdf.ret>10)).sum()
    fpa = (mask & (rdf.ret>0)).sum()
    t = mask.sum()
    mark = 'ZERO_FP' if fp==0 else ''
    print('  disp<%.3f: c=%d/%d fp=%d/%d fpA=%d/%d t=%d %s' % (disp_th,c,NBL,fp,NBW,fpa,NAW,t,mark))

# ============================================================
# 4. MKT adr scan
# ============================================================
print('\n' + '='*80)
print('4. MKT: disp_shock < 0.977 & adr < [th]')
print('='*80)

for adr_th in [0.20, 0.25, 0.30, 0.35, 0.387, 0.40, 0.45, 0.50, 0.60, 0.80]:
    mask = (rdf['mkt_dispersion_shock_v2']<0.977) & (rdf['market_advance_decline_ratio']<adr_th)
    c = (mask & (rdf.ret<-5)).sum()
    fp = (mask & (rdf.ret>10)).sum()
    fpa = (mask & (rdf.ret>0)).sum()
    t = mask.sum()
    mark = 'ZERO_FP' if fp==0 else ''
    print('  adr<%.3f: c=%d/%d fp=%d/%d fpA=%d/%d t=%d %s' % (adr_th,c,NBL,fp,NBW,fpa,NAW,t,mark))

# ============================================================
# 5. MKT Joint Grid (zero FP only, c>=5)
# ============================================================
print('\n' + '='*80)
print('5. MKT Joint Grid: disp x adr (zero FP, c>=5)')
print('='*80)

best = []
for disp_th in [0.90, 0.93, 0.95, 0.96, 0.97, 0.977, 0.98, 0.99, 1.00, 1.02]:
    for adr_th in [0.20, 0.25, 0.30, 0.35, 0.387, 0.40, 0.45, 0.50, 0.60, 0.80]:
        mask = (rdf['mkt_dispersion_shock_v2']<disp_th) & (rdf['market_advance_decline_ratio']<adr_th)
        c = (mask & (rdf.ret<-5)).sum()
        fp = (mask & (rdf.ret>10)).sum()
        t = mask.sum()
        if fp == 0 and c >= 5:
            best.append((c, t, 'disp<%.3f & adr<%.3f' % (disp_th, adr_th)))

best.sort(key=lambda x: (-x[0], x[1]))
seen = set()
print(fmt % ('Rule','C','BL','FP','BW','FA','AW','Tot','','',''))
print('-'*65)
for c, t, desc in best[:25]:
    if desc in seen: continue
    seen.add(desc)
    print(fmt % (desc, c, NBL, 0, NBW, '0', NAW, t, '', '', ''))

# ============================================================
# 6. R4 liq threshold scan
# ============================================================
print('\n' + '='*80)
print('6. R4: liq < [th] & mfp<0.78')
print('='*80)

for liq_th in [0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 1.00]:
    mask = (rdf['market_liquidity_change']<liq_th) & (rdf['money_flow_pressure_pct']<0.78)
    c = (mask & (rdf.ret<-5)).sum()
    fp = (mask & (rdf.ret>10)).sum()
    fpa = (mask & (rdf.ret>0)).sum()
    t = mask.sum()
    mark = 'ZERO_FP' if fp==0 else ''
    print('  liq<%.2f: c=%d/%d fp=%d/%d fpA=%d/%d t=%d %s' % (liq_th,c,NBL,fp,NBW,fpa,NAW,t,mark))

# ============================================================
# 7. MKT blocked trades detail
# ============================================================
print('\n' + '='*80)
print('7. MKT (disp<0.977 & adr<0.387): All Blocked Trades')
print('='*80)

mask = (rdf['mkt_dispersion_shock_v2']<0.977) & (rdf['market_advance_decline_ratio']<0.387)
caught = rdf[mask].sort_values(['bd','ret'])
for _, r in caught.iterrows():
    tag = 'BL' if r.ret<-5 else ('BW' if r.ret>10 else ('W' if r.ret>0 else 'L'))
    print('  %-5s %s buy=%s ret=%+5.1f%% [%s] disp=%.3f adr=%.3f' % (
        r['src'], r['s'], r['bd'], r['ret'], tag,
        r['mkt_dispersion_shock_v2'], r['market_advance_decline_ratio']))
