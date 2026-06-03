"""Comprehensive analysis: all 8 candidates - overlap, FP, robustness"""
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

need = ['market_liquidity_change','mkt_liquidity_impulse_v2','market_trend_strength',
    'market_60d_new_high_ratio','market_advance_decline_ratio','mkt_regime_trend_pressure_v2',
    'money_flow_pressure_pct','momentum_peak_decay_3d','ulcer_index','true_turnover_10d',
    'runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
    'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z',
    'mkt_dispersion_shock_v2']

rows = []
for _,t in at.iterrows():
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < pd.Timestamp(t['bd'])]
    if len(before)==0: continue
    r = sd[sd['date']==before['date'].max()].iloc[0]
    rec = dict(s=t['s'],bd=t['bd'],ret=t['ret'],src=t['src'])
    for f in need:
        if f in r.index: rec[f] = r.get(f, np.nan)
    rows.append(rec)
rdf = pd.DataFrame(rows)

BLN = len(rdf[rdf.ret<-5]); BWN = len(rdf[rdf.ret>10]); AWN = len(rdf[rdf.ret>0])

# Existing 5 rules
rdf['HR1']=(rdf['runup_ma5_exhaustion']>0.8)&(rdf['cpv_factor']<=-0.4)
rdf['HR2']=(rdf['prev_day_blowoff_risk']>=2.0)&(rdf['sector_momentum_accel_rank']<=0.30) | \
           (rdf['prev_day_blowoff_risk']>5.0)&(rdf['sector_momentum_accel_rank']<=0.50) | \
           (rdf['prev_day_blowoff_risk']>1.0)&(rdf['sector_momentum_accel_rank']<=0.15)
rdf['HR3']=(rdf['sector_leadership_persistence_v3']<=-0.18)&(rdf['return_20d_intra_z']<=0.40)
rdf['HR4']=(rdf['market_liquidity_change']<0.95)&(rdf['money_flow_pressure_pct']<0.78)
rdf['HR5']=(rdf['mkt_dispersion_shock_v2']<0.98)&(rdf['market_advance_decline_ratio']<0.38)
rdf['EXISTING'] = rdf[['HR1','HR2','HR3','HR4','HR5']].any(axis=1)

# 8 candidates
def p(factor, pct):
    return rdf[factor].dropna().quantile(pct)

candidates = [
    # Market+Market (4)
    ('C1','liq_imp<%.3f(p30) & adr<%.3f(p15)'%(p('mkt_liquidity_impulse_v2',0.30),p('market_advance_decline_ratio',0.15)),
     (rdf['mkt_liquidity_impulse_v2']<p('mkt_liquidity_impulse_v2',0.30))&(rdf['market_advance_decline_ratio']<p('market_advance_decline_ratio',0.15))),
    ('C2','liq_chg<%.3f(p35) & nh<%.3f(p10)'%(p('market_liquidity_change',0.35),p('market_60d_new_high_ratio',0.10)),
     (rdf['market_liquidity_change']<p('market_liquidity_change',0.35))&(rdf['market_60d_new_high_ratio']<p('market_60d_new_high_ratio',0.10))),
    ('C3','trend<%.3f(p35) & regime_press>%.3f(p60)'%(p('market_trend_strength',0.35),p('mkt_regime_trend_pressure_v2',0.40)),
     (rdf['market_trend_strength']<p('market_trend_strength',0.35))&(rdf['mkt_regime_trend_pressure_v2']>p('mkt_regime_trend_pressure_v2',0.40))),
    ('C4','liq_chg<%.3f(p30) & adr<%.3f(p15)'%(p('market_liquidity_change',0.30),p('market_advance_decline_ratio',0.15)),
     (rdf['market_liquidity_change']<p('market_liquidity_change',0.30))&(rdf['market_advance_decline_ratio']<p('market_advance_decline_ratio',0.15))),
    # Market+Stock (4)
    ('C5','liq_chg<%.3f(p40) & mom_peak>%.3f(p75)'%(p('market_liquidity_change',0.40),p('momentum_peak_decay_3d',0.25)),
     (rdf['market_liquidity_change']<p('market_liquidity_change',0.40))&(rdf['momentum_peak_decay_3d']>p('momentum_peak_decay_3d',0.75))),
    ('C6','nh<%.3f(p35) & mfp<%.3f(p35)'%(p('market_60d_new_high_ratio',0.35),p('money_flow_pressure_pct',0.35)),
     (rdf['market_60d_new_high_ratio']<p('market_60d_new_high_ratio',0.35))&(rdf['money_flow_pressure_pct']<p('money_flow_pressure_pct',0.35))),
    ('C7','liq_imp<%.3f(p35) & turnover>%.3f(p80)'%(p('mkt_liquidity_impulse_v2',0.35),p('true_turnover_10d',0.20)),
     (rdf['mkt_liquidity_impulse_v2']<p('mkt_liquidity_impulse_v2',0.35))&(rdf['true_turnover_10d']>p('true_turnover_10d',0.80))),
    ('C8','liq_chg<%.3f(p25) & ulcer<%.3f(p20)'%(p('market_liquidity_change',0.25),p('ulcer_index',0.20)),
     (rdf['market_liquidity_change']<p('market_liquidity_change',0.25))&(rdf['ulcer_index']<p('ulcer_index',0.20))),
]

# ============================================================
# 1. Overlap + FP Analysis
# ============================================================
print('='*100)
print('1. Full Analysis: Overlap with Existing Rules + False Positives')
print('='*100)
fmt = '%-6s %4s/%-2s %4s/%-2s %4s/%-2s %4s %3s %3s %3s %3s'
print(fmt % ('Rule','C','BL','FB','BW','FA','AW','Tot','Ov','Nw','sBL','sBW'))
print('-'*75)

for name, desc, mask in candidates:
    c = (mask & (rdf.ret<-5)).sum()
    fp_bw = (mask & (rdf.ret>10)).sum()
    fp_aw = (mask & (rdf.ret>0)).sum()
    t = mask.sum()
    ov = (mask & rdf['EXISTING'] & (rdf.ret<-5)).sum()  # overlap with existing (big losses)
    nw = (mask & ~rdf['EXISTING'] & (rdf.ret<-5)).sum()  # new big losses
    # shadow
    s_bl = (mask & (rdf.src=='shadow') & (rdf.ret<-5)).sum()
    s_bw = (mask & (rdf.src=='shadow') & (rdf.ret>10)).sum()
    print(fmt % (name, c, BLN, fp_bw, BWN, fp_aw, AWN, t, ov, nw, s_bl, s_bw))

# ============================================================
# 2. Detailed overlap: which trades are caught by each candidate
# ============================================================
print()
print('='*100)
print('2. New Big Losses Caught (NOT in existing 5 rules)')
print('='*100)

for name, desc, mask in candidates:
    new_bl = rdf[mask & ~rdf['EXISTING'] & (rdf.ret<-5)].sort_values('ret')
    new_bw = rdf[mask & ~rdf['EXISTING'] & (rdf.ret>10)].sort_values('ret', ascending=False)
    n_bl = len(new_bl); n_bw = len(new_bw)
    print()
    print('-- %s (%s) --' % (name, desc))
    print('  New BL caught: %d (overlap BL with existing: %d)' % (n_bl, (mask & rdf['EXISTING'] & (rdf.ret<-5)).sum()))
    for _,r in new_bl.iterrows():
        print('    BL  %s %s buy=%s ret=%+5.1f%%' % (r['src'], r['s'], r['bd'], r['ret']))
    print('  New BW killed: %d' % n_bw)
    for _,r in new_bw.iterrows():
        print('    BW  %s %s buy=%s ret=%+5.1f%%' % (r['src'], r['s'], r['bd'], r['ret']))
    # Also show new small wins killed
    new_aw = rdf[mask & ~rdf['EXISTING'] & (rdf.ret>0) & (rdf.ret<=10)].sort_values('ret', ascending=False)
    if len(new_aw) > 0 and len(new_aw) <= 5:
        print('  New small wins killed: %d' % len(new_aw))
        for _,r in new_aw.iterrows():
            print('    SW  %s %s buy=%s ret=%+5.1f%%' % (r['src'], r['s'], r['bd'], r['ret']))

# ============================================================
# 3. Robustness: grid search around optimal thresholds
# ============================================================
print()
print('='*100)
print('3. Robustness Grid Search (zero FP on BW only)')
print('='*100)

# For each candidate pair of factors, scan nearby thresholds
robustness_configs = [
    ('C1', 'mkt_liquidity_impulse_v2', '<', 'market_advance_decline_ratio', '<'),
    ('C2', 'market_liquidity_change', '<', 'market_60d_new_high_ratio', '<'),
    ('C3', 'market_trend_strength', '<', 'mkt_regime_trend_pressure_v2', '>'),
    ('C4', 'market_liquidity_change', '<', 'market_advance_decline_ratio', '<'),
    ('C5', 'market_liquidity_change', '<', 'momentum_peak_decay_3d', '>'),
    ('C6', 'market_60d_new_high_ratio', '<', 'money_flow_pressure_pct', '<'),
    ('C7', 'mkt_liquidity_impulse_v2', '<', 'true_turnover_10d', '>'),
    ('C8', 'market_liquidity_change', '<', 'ulcer_index', '<'),
]

for name, f1, d1, f2, d2 in robustness_configs:
    print()
    print('--- %s: %s x %s ---' % (name, f1, f2))
    f1v = rdf[f1].dropna(); f2v = rdf[f2].dropna()

    # Current best percentiles (from scan)
    best_p = {
        'C1': (0.30, 0.15), 'C2': (0.35, 0.10), 'C3': (0.35, 0.40),
        'C4': (0.30, 0.15), 'C5': (0.40, 0.25), 'C6': (0.35, 0.35),
        'C7': (0.35, 0.20), 'C8': (0.25, 0.20),
    }
    p1_best, p2_best = best_p[name]

    print('  Best: f1=p%d f2=p%d' % (int(p1_best*100), int(p2_best*100)))
    print('  f1:%-7s f2:%-7s %8s %8s %8s %6s %6s' % ('th','th','C/BL','FB/BW','FA/AW','Tot','New'))

    for p1 in [x/100 for x in range(max(5,int((p1_best-0.10)*100)), min(45,int((p1_best+0.15)*100))+1, 5)]:
        for p2 in [x/100 for x in range(max(5,int((p2_best-0.10)*100)), min(45,int((p2_best+0.15)*100))+1, 5)]:
            if d1 == '<': t1 = f1v.quantile(p1)
            else: t1 = f1v.quantile(1-p1)
            if d2 == '<': t2 = f2v.quantile(p2)
            else: t2 = f2v.quantile(1-p2)

            mask = (rdf[f1] < t1 if d1=='<' else rdf[f1] > t1) & \
                   (rdf[f2] < t2 if d2=='<' else rdf[f2] > t2)
            c = (mask & (rdf.ret<-5)).sum()
            fp = (mask & (rdf.ret>10)).sum()
            fpa = (mask & (rdf.ret>0)).sum()
            t = mask.sum()
            nw = (mask & ~rdf['EXISTING'] & (rdf.ret<-5)).sum()

            mark = ' << BEST' if (p1==p1_best and p2==p2_best) else ''
            mark2 = ' ZERO_FP' if fp==0 else ''
            if c >= 3 and fp <= 2:
                print('  %.3f %.3f   %2d/%d   %2d/%d   %2d/%d   %4d  %3d%s' % (
                    t1, t2, c, BLN, fp, BWN, fpa, AWN, t, nw, mark+mark2))
