"""Remaining big loss analysis: market+market and market+stock combos"""
import re, pandas as pd, numpy as np

def parse_log(fp):
    trades, cb = [], {}
    with open(fp, 'r', encoding='utf-8') as f:
        for l in f:
            bm = re.search(r'buy_date=(\d{4}-\d{2}-\d{2})', l); sm = re.search(r'(\d{6}\.(?:SH|SZ))', l)
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

# Market factors (existing + additional)
mkt = [
    'market_liquidity_change','market_trend_strength','market_volatility_context',
    'mkt_liquidity_impulse_v2','mkt_dispersion_shock_v2','market_advance_decline_ratio',
    'mkt_breadth_thrust_v2','mkt_emotion_stability_v2','mkt_ma5_followthrough_v3',
    'market_sentiment_ad','mkt_regime_trend_pressure_v2','mkt_sector_dispersion',
    'market_60d_new_high_ratio','mkt_ma5_followthrough_v3_exp',
]

# Stock factors (existing + user-suggested)
stk = [
    'post_accumulation_supply_tightness','tightness_3d','true_turnover_10d',
    'close_position_60d','close_position_20d','money_flow_pressure_pct',
    'volume_price_exhaustion','pullback_depth_from_10d_high',
    'efficiency_ratio','max_drawdown_20d','max_rally_10d',
    'runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
    'sector_momentum_accel_rank','sector_leadership_persistence_v3',
    'return_20d_intra_z','distance_from_52w_high','distance_from_52w_low',
    'ulcer_index','severe_abnormal_pressure_10d','prior_high_proximity_20d',
    'failed_breakout_pressure','high_volume_no_gain_3d',
    'momentum_peak_decay_3d','carryover_overheat_risk',
    'high_pos_volume_no_gain','upper_shadow_pressure_5d',
    'stock_volume_ratio_20d','turnover_z_20',
    'buy_lg_amount_rate','buy_sm_amount_rate','net_d5_amount_to_turnover',
    'main_outflow_retail_takeover','regime_aware_reversal_alpha',
    'high_volatility_pullback_alpha','volatility_structure_alpha',
]

all_f = mkt + stk

rows = []
for _,t in at.iterrows():
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < pd.Timestamp(t['bd'])]
    if len(before)==0: continue
    r = sd[sd['date']==before['date'].max()].iloc[0]
    rec = dict(s=t['s'],bd=t['bd'],ret=t['ret'],src=t['src'])
    for f in all_f:
        if f in r.index: rec[f] = r.get(f, np.nan)
    rows.append(rec)

rdf = pd.DataFrame(rows)
BLN = len(rdf[rdf.ret<-5]); BWN = len(rdf[rdf.ret>10]); AWN = len(rdf[rdf.ret>0])
print('Trades: %d, BL=%d, BW=%d, AW=%d' % (len(rdf), BLN, BWN, AWN))

# Apply all current rules to identify remaining leaks
rdf['HR1']=(rdf['runup_ma5_exhaustion']>0.8)&(rdf['cpv_factor']<=-0.4)
rdf['HR2']=(rdf['prev_day_blowoff_risk']>=2.0)&(rdf['sector_momentum_accel_rank']<=0.30) | \
           (rdf['prev_day_blowoff_risk']>5.0)&(rdf['sector_momentum_accel_rank']<=0.50) | \
           (rdf['prev_day_blowoff_risk']>1.0)&(rdf['sector_momentum_accel_rank']<=0.15)
rdf['HR3']=(rdf['sector_leadership_persistence_v3']<=-0.18)&(rdf['return_20d_intra_z']<=0.40)
rdf['HR4']=(rdf['market_liquidity_change']<0.95)&(rdf['money_flow_pressure_pct']<0.78)
rdf['HR5']=(rdf['mkt_dispersion_shock_v2']<0.98)&(rdf['market_advance_decline_ratio']<0.38)
rdf['hard'] = rdf[['HR1','HR2','HR3','HR4','HR5']].any(axis=1)

leaks = rdf[(rdf.ret<-5) & ~rdf['hard']]
print('Remaining leaks (not caught by any rule): %d/%d' % (len(leaks), BLN))

# ============================================================
# 1. Market+Market combos (zero FP on BW)
# ============================================================
print('\n' + '='*80)
print('1. Best Market+Market Combos (zero FP on BW, c>=3)')
print('='*80)

best_mm = []
for i,m1 in enumerate(mkt):
    if m1 not in rdf.columns: continue
    v1 = rdf[m1].dropna()
    for p1 in [0.10,0.15,0.20,0.25,0.30,0.35,0.40]:
        t1_lo = v1.quantile(p1); t1_hi = v1.quantile(1-p1)
        for m2 in mkt[i+1:]:
            if m2 not in rdf.columns: continue
            v2 = rdf[m2].dropna()
            for p2 in [0.10,0.15,0.20,0.25,0.30,0.35,0.40]:
                t2_lo = v2.quantile(p2); t2_hi = v2.quantile(1-p2)
                # both low
                mask = (rdf[m1]<t1_lo) & (rdf[m2]<t2_lo)
                c = (mask & (rdf.ret<-5)).sum(); fp = (mask & (rdf.ret>10)).sum()
                new_val = int((mask & (rdf.ret<-5) & ~rdf['hard']).sum())
                if c >= 3 and fp == 0:
                    best_mm.append((c, mask.sum(), new_val, '%s<p%d & %s<p%d'%(m1[:20],int(p1*100),m2[:20],int(p2*100))))
                # m1 low + m2 high
                mask = (rdf[m1]<t1_lo) & (rdf[m2]>t2_hi)
                c = (mask & (rdf.ret<-5)).sum(); fp = (mask & (rdf.ret>10)).sum()
                new_val = int((mask & (rdf.ret<-5) & ~rdf['hard']).sum())
                if c >= 3 and fp == 0:
                    best_mm.append((c, mask.sum(), new_val, '%s<p%d & %s>p%d'%(m1[:20],int(p1*100),m2[:20],int((1-p2)*100))))

best_mm.sort(key=lambda x: (-int(x[0]), -int(x[2]), int(x[1])))
seen = set()
print('%-55s %3s %4s %4s' % ('Rule','C/BL','Tot','New'))
print('-'*70)
for c, t, n, desc in best_mm[:30]:
    if desc in seen: continue; seen.add(desc)
    print('%-55s %3s   %3d  %3d' % (desc, '%d/%d'%(c,BLN), t, n))

# ============================================================
# 2. Market+Stock combos (zero FP on BW)
# ============================================================
print('\n' + '='*80)
print('2. Best Market+Stock Combos (zero FP on BW, c>=4)')
print('='*80)

best_ms = []
# Use top market factors as base
market_picks = [('market_60d_new_high_ratio','<'), ('mkt_liquidity_impulse_v2','<'),
                ('market_liquidity_change','<'), ('mkt_dispersion_shock_v2','<'),
                ('mkt_breadth_thrust_v2','<'), ('market_sentiment_ad','<'),
                ('market_volatility_context','>')]

for mf, mf_dir in market_picks:
    if mf not in rdf.columns: continue
    mv = rdf[mf].dropna()
    for mp in [0.10,0.15,0.20,0.25,0.30,0.35,0.40]:
        if mf_dir == '<': mt = mv.quantile(mp)
        else: mt = mv.quantile(1-mp)
        for sf in stk:
            if sf not in rdf.columns: continue
            sv = rdf[sf].dropna()
            for sp in [0.10,0.15,0.20,0.25,0.30,0.35,0.40]:
                # stock low
                st_lo = sv.quantile(sp); st_hi = sv.quantile(1-sp)
                if mf_dir == '<': mask = (rdf[mf]<mt) & (rdf[sf]<st_lo)
                else: mask = (rdf[mf]>mt) & (rdf[sf]<st_lo)
                c = (mask & (rdf.ret<-5)).sum(); fp = (mask & (rdf.ret>10)).sum()
                new_val = int((mask & (rdf.ret<-5) & ~rdf['hard']).sum())
                if c >= 4 and fp == 0:
                    best_ms.append((c, mask.sum(), new_val, '%s<p%d & %s<p%d'%(mf[:25],int(mp*100),sf[:25],int(sp*100))))
                # stock high
                if mf_dir == '<': mask = (rdf[mf]<mt) & (rdf[sf]>st_hi)
                else: mask = (rdf[mf]>mt) & (rdf[sf]>st_hi)
                c = (mask & (rdf.ret<-5)).sum(); fp = (mask & (rdf.ret>10)).sum()
                new_val = int((mask & (rdf.ret<-5) & ~rdf['hard']).sum())
                if c >= 4 and fp == 0:
                    best_ms.append((c, mask.sum(), new_val, '%s<p%d & %s>p%d'%(mf[:25],int(mp*100),sf[:25],int((1-sp)*100))))

best_ms.sort(key=lambda x: (-int(x[0]), -int(x[2]), int(x[1])))
seen2 = set()
print('%-60s %3s %4s %4s' % ('Rule','C/BL','Tot','New'))
print('-'*75)
for c, t, n, desc in best_ms[:40]:
    if desc in seen2: continue; seen2.add(desc)
    print('%-60s %3s   %3d  %3d' % (desc, '%d/%d'%(c,BLN), t, n))

# ============================================================
# 3. User's suggestion: new_high_ratio + severe_abnormal
# ============================================================
print('\n' + '='*80)
print("3. User's Example: market_60d_new_high_ratio + severe_abnormal_pressure_10d")
print('='*80)
for nh in rdf['market_60d_new_high_ratio'].quantile([0.10,0.15,0.20,0.25,0.30,0.40,0.50]):
    for sa in rdf['severe_abnormal_pressure_10d'].quantile([0.30,0.35,0.40,0.45,0.50]):
        mask = (rdf['market_60d_new_high_ratio']<nh) & (rdf['severe_abnormal_pressure_10d']>sa)
        c = (mask & (rdf.ret<-5)).sum(); fp = (mask & (rdf.ret>10)).sum()
        new = (mask & (rdf.ret<-5) & ~rdf['hard']).sum()
        if c >= 3 and fp <= 1:
            print('  nh<%.4f(p%d) & sap>%.3f(p%.0f): c=%d fp=%d new=%d' % (
                nh, int((rdf['market_60d_new_high_ratio']<nh).mean()*100),
                sa, (rdf['severe_abnormal_pressure_10d']>sa).mean()*100,
                c, fp, new_val))

# ============================================================
# 4. Show details of top 3 market+stock candidates
# ============================================================
print('\n' + '='*80)
print('4. Top Candidate Details')
print('='*80)

# Re-evaluate top candidates
top_candidates = best_ms[:5]
seen3 = set()
for c, t, n, desc in top_candidates:
    if desc in seen3: continue; seen3.add(desc)
    # Parse desc back: "<mf><pX> & <sf><pY>"
    parts = desc.split(' & ')
    mf_part = parts[0].split('<p')
    mf_name = mf_part[0].strip()
    mf_pct = int(mf_part[1])/100
    sf_name = parts[1].split('<p')[0].strip()
    sf_pct_dir = 'lo' if '<p' in parts[1] else 'hi'
    if sf_pct_dir == 'lo':
        sf_pct = int(parts[1].split('<p')[1])/100
    else:
        sf_pct = int(parts[1].split('>p')[1])/100

    # Find matching cols
    mf_col = [x for x in rdf.columns if x.startswith(mf_name[:15])]
    sf_col = [x for x in rdf.columns if x.startswith(sf_name[:15])]
    if not mf_col or not sf_col: continue
    mf_col = mf_col[0]; sf_col = sf_col[0]

    mf_th = rdf[mf_col].quantile(mf_pct)
    sf_th = rdf[sf_col].quantile(sf_pct) if sf_pct_dir=='lo' else rdf[sf_col].quantile(1-sf_pct)

    if sf_pct_dir == 'lo':
        mask = (rdf[mf_col]<mf_th) & (rdf[sf_col]<sf_th)
    else:
        mask = (rdf[mf_col]<mf_th) & (rdf[sf_col]>sf_th)

    nleaks = len(rdf[(rdf.ret<-5) & ~rdf['hard']])
    print('\n-- %s: %s<%.3f & %s<%.3f (new_leaks=%d/%d)' % (desc, mf_col, mf_th, sf_col, sf_th, n, nleaks))
    # Show all caught trades
    caught = rdf[mask].sort_values('ret')
    for _, r in caught.iterrows():
        tag = 'BL' if r.ret<-5 else ('BW' if r.ret>10 else ('W' if r.ret>0 else 'L'))
        in_leak = 'LEAK' if not r['hard'] and r.ret<-5 else ''
        print('  %s buy=%s ret=%+5.1f%% [%s] %s %s=%.3f %s=%.3f' % (
            r['s'],r['bd'],r['ret'],tag,in_leak, mf_col,r[mf_col],sf_col,r[sf_col]))
