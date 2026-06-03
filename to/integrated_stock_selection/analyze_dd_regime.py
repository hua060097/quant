"""Compare factor distributions: DD streak periods vs normal periods (including shadow)"""
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

d185 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子185_optuna.log')
d185 = d185.sort_values('bd').reset_index(drop=True)

# Identify DD streaks (buy_date ranges with consecutive losses)
streak_dates = set()
for i, (_,r) in enumerate(d185.iterrows()):
    if r['ret'] < 0:
        # Find consecutive streak this belongs to
        j = i
        while j >= 0 and d185.iloc[j]['ret'] < 0: j -= 1
        streak_start = j + 1
        j = i
        while j < len(d185) and d185.iloc[j]['ret'] < 0: j += 1
        streak_end = j - 1
        if streak_end - streak_start >= 1:  # at least 2 consecutive losses
            streak_dates.add(d185.iloc[i]['bd'])

# Load shadow
shadow = pd.read_csv('./results/walkforward_shadow_skipped_opportunities_20260520_190404.csv')
shadow_trades = []
for _,r in shadow.iterrows():
    shadow_trades.append(dict(s=r['stock_code'],bd=r['buy_datetime'][:10],ret=r['actual_return']*100,src='shadow'))
sf = pd.DataFrame(shadow_trades)

# Combine all trades
all_t = pd.concat([d185, sf], ignore_index=True)

# Load factors
df = pd.read_parquet('./factor_lab_data/factors.parquet'); df['date']=pd.to_datetime(df['date'])

market_f = [
    'market_liquidity_change','market_trend_strength','market_volatility_context',
    'mkt_liquidity_impulse_v2','mkt_dispersion_shock_v2','market_advance_decline_ratio',
    'mkt_breadth_thrust_v2','mkt_emotion_stability_v2','mkt_ma5_followthrough_v3',
    'market_sentiment_ad','mkt_regime_trend_pressure_v2','mkt_sector_dispersion',
    'market_60d_new_high_ratio',
]

stock_f = [
    'runup_ma5_exhaustion','cpv_factor','prev_day_blowoff_risk',
    'sector_momentum_accel_rank','sector_leadership_persistence_v3','return_20d_intra_z',
    'money_flow_pressure_pct','ulcer_index','momentum_peak_decay_3d',
    'intra_sector_momentum_rank','close_position_60d','efficiency_ratio',
    'turnover_z_20','top10_10d_strength_level_v3','buy_sm_amount_rate',
    'severe_abnormal_pressure_10d','high_volume_no_gain_3d','max_drawdown_20d',
]
all_f = market_f + stock_f

rows = []
for _,t in all_t.iterrows():
    sd = df[df['stock_code']==t['s']].sort_values('date')
    before = sd[sd['date'] < pd.Timestamp(t['bd'])]
    if len(before)==0: continue
    r = sd[sd['date']==before['date'].max()].iloc[0]
    rec = dict(s=t['s'],bd=t['bd'],ret=t['ret'],src=t['src'])
    # Mark DD vs normal based on buy_date
    rec['is_dd'] = t['bd'] in streak_dates
    for f in all_f: rec[f] = r.get(f,np.nan)
    rows.append(rec)

rdf = pd.DataFrame(rows)
dd = rdf[rdf['is_dd']]; norm = rdf[~rdf['is_dd']]
print('DD period trades: %d, Normal: %d' % (len(dd), len(norm)))

# Compare factor distributions: DD vs Normal
print()
print('=== Market Factors: DD vs Normal ===')
print('%-35s %10s %10s %10s %8s %7s' % ('Factor','DD_mean','Norm_mean','Diff','DD_pct','N_pct'))
print('-'*80)
for f in market_f:
    dvm = dd[f].mean(); nvm = norm[f].mean()
    dd_pct = (norm[f] < dvm).mean()*100  # where does DD mean sit in normal distribution?
    print('%-35s %+10.3f %+10.3f %+10.3f %8.1f%%' % (f, dvm, nvm, dvm-nvm, dd_pct))

print()
print('=== Stock Factors: DD vs Normal ===')
print('%-35s %10s %10s %10s %8s' % ('Factor','DD_mean','Norm_mean','Diff','DD_pct'))
print('-'*80)
for f in stock_f:
    dvm = dd[f].mean(); nvm = norm[f].mean()
    dd_pct = (norm[f] < dvm).mean()*100
    print('%-35s %+10.3f %+10.3f %+10.3f %8.1f%%' % (f, dvm, nvm, dvm-nvm, dd_pct))

# Most extreme factors
print()
print('=== Top Discriminators (DD vs Normal) ===')
sep_scores = []
for f in all_f:
    dvm = dd[f].mean(); nvm = norm[f].mean()
    pooled = np.sqrt((dd[f].std()**2 + norm[f].std()**2)/2)
    sep = abs(dvm - nvm) / max(pooled, 0.001)
    sep_scores.append((sep, f, dvm, nvm))
sep_scores.sort(key=lambda x: -x[0])
for sep, f, dvm, nvm in sep_scores[:15]:
    dd_pct = (norm[f] < dvm).mean()*100
    print('  %-35s sep=%.2f DD=%.3f Norm=%.3f pct=%.0f%%' % (f, sep, dvm, nvm, dd_pct))
