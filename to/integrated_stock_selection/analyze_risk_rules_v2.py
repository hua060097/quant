"""分析大盘因子+个股因子组合风控规则"""
import re, pandas as pd, numpy as np

def parse_log(filepath):
    trades, current_buy = [], {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.search(r'🔎 (\d+\.\w+) 开盘风控检查: buy_date=(\d{4}-\d{2}-\d{2})', line)
            if m: current_buy[m.group(1)] = m.group(2)
            m = re.search(r'卖出: (\d+\.\w+),.*?收益: (-?\d+\.?\d*) \((-?\d+\.?\d*)%\)', line)
            if m:
                s, r = m.group(1), float(m.group(3))
                bd = current_buy.pop(s, None)
                trades.append({'stock_code': s, 'buy_date': bd, 'return_pct': r})
    return pd.DataFrame(trades)

df162 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子162_optuna.log')
df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

test_factors = [
    'post_accumulation_supply_tightness', 'true_turnover_10d', 'tightness_3d',
    'runup_ma5_exhaustion', 'cpv_factor', 'prev_day_blowoff_risk',
    'sector_momentum_accel_rank', 'sector_leadership_persistence_v3', 'return_20d_intra_z',
    'market_trend_strength', 'market_sentiment_ad', 'market_volatility_context',
    'market_liquidity_change', 'market_advance_decline_ratio',
    'mkt_ma5_followthrough_v3', 'mkt_regime_trend_pressure_v2',
    'mkt_breadth_thrust_v2', 'mkt_emotion_stability_v2',
    'mkt_dispersion_shock_v2', 'mkt_liquidity_impulse_v2',
    'close_position_20d', 'close_position_60d', 'upper_shadow_pressure_5d',
    'high_volume_no_gain_3d', 'pullback_depth_from_10d_high',
    'close_to_high_3d_mean', 'momentum_peak_decay_3d', 'volume_price_exhaustion',
    'money_flow_pressure', 'efficiency_ratio',
    'prior_high_proximity_20d', 'distance_from_52w_high',
    'max_drawdown_20d', 'ulcer_index', 'max_rally_10d', 'max_rally_20d',
    'failed_breakout_pressure', 'regime_aware_reversal_alpha',
    'stock_volume_ratio_20d', 'turnover_z_20',
    'net_d5_amount_to_turnover', 'buy_lg_amount_rate', 'buy_sm_amount_rate',
]

results = []
for _, trade in df162.iterrows():
    stock = trade['stock_code']
    buy_dt = pd.Timestamp(trade['buy_date']) if trade['buy_date'] else None
    if buy_dt is None: continue
    stock_data = df[df['stock_code'] == stock].sort_values('date')
    before = stock_data[stock_data['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    factor_row = stock_data[stock_data['date'] == pred_date]
    if len(factor_row) == 0: continue
    r = factor_row.iloc[0]
    rec = {'stock_code': stock, 'buy_date': trade['buy_date'], 'return_pct': trade['return_pct']}
    for nf in test_factors:
        if nf in df.columns: rec[nf] = r.get(nf, np.nan)
    results.append(rec)

rdf = pd.DataFrame(results)
big_loss = rdf[rdf['return_pct'] < -5]
big_win = rdf[rdf['return_pct'] > 10]

# === 1. Market factor values: big loss vs big win ===
print('=== 1. Market Factor: Big Loss vs Big Win ===')
market_factors = [
    'market_trend_strength', 'market_sentiment_ad', 'market_volatility_context',
    'market_liquidity_change', 'market_advance_decline_ratio',
    'mkt_ma5_followthrough_v3', 'mkt_regime_trend_pressure_v2',
    'mkt_breadth_thrust_v2', 'mkt_emotion_stability_v2',
    'mkt_dispersion_shock_v2', 'mkt_liquidity_impulse_v2',
]

for nf in market_factors:
    if nf not in rdf.columns: continue
    lv = big_loss[nf].dropna()
    wv = big_win[nf].dropna()
    if len(lv) < 3: continue
    # Also get percentile
    all_vals = rdf[nf].dropna()
    loss_pct = (all_vals < lv.mean()).mean() * 100
    win_pct = (all_vals < wv.mean()).mean() * 100
    print(f'  {nf:<40} loss={lv.mean():.3f}(p{loss_pct:.0f}) win={wv.mean():.3f}(p{win_pct:.0f})')

# === 2. Market+Stock combination scan ===
# Key insight: market weak AND stock weak = catch
# For each market factor, test "low" (weak) threshold
# For each stock factor, test various thresholds

stock_candidates = [
    ('post_accumulation_supply_tightness', 'lt', -0.65),
    ('post_accumulation_supply_tightness', 'lt', 0),
    ('tightness_3d', 'lt', -0.5),
    ('tightness_3d', 'lt', 0),
    ('close_position_60d', 'lt', 0.8),
    ('volume_price_exhaustion', 'gt', 0.2),
    ('momentum_peak_decay_3d', 'gt', 0),
    ('money_flow_pressure', 'lt', 1.0),
    ('efficiency_ratio', 'lt', 1.5),
    ('pullback_depth_from_10d_high', 'lt', -0.1),
    ('prior_high_proximity_20d', 'lt', 0),
    ('distance_from_52w_high', 'lt', 1.0),
    ('sector_leadership_persistence_v3', 'lt', -0.05),
    ('max_drawdown_20d', 'lt', 0),
]

best = []
for mf in market_factors:
    if mf not in rdf.columns: continue
    mf_vals = rdf[mf].dropna()

    # For each market factor, try "weak" thresholds at different percentiles
    for mf_pct in [0.20, 0.25, 0.30, 0.35, 0.40]:
        mf_th = mf_vals.quantile(mf_pct)

        for sf, sd, st in stock_candidates:
            if sf not in rdf.columns: continue

            # market low (weak) + stock weak
            mask = (rdf[mf] < mf_th)
            if sd == 'lt':
                mask = mask & (rdf[sf] < st)
            else:
                mask = mask & (rdf[sf] > st)

            catch = (mask & (rdf['return_pct'] < -5)).sum()
            fp = (mask & (rdf['return_pct'] > 10)).sum()
            fp_all = (mask & (rdf['return_pct'] > 0)).sum()
            total = mask.sum()
            if catch >= 4 and fp <= 2:
                score = catch - 2 * fp
                desc = f'{mf}<{mf_th:.2f} & {sf[:20]}{"<" if sd == "lt" else ">"}{st}'
                best.append((score, desc, catch, fp, fp_all, total))

    # Also try market high for volatility-type factors
    if 'volatility' in mf or 'shock' in mf:
        for mf_pct in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mf_th = mf_vals.quantile(mf_pct)
            for sf, sd, st in stock_candidates:
                if sf not in rdf.columns: continue
                mask = (rdf[mf] > mf_th)
                if sd == 'lt':
                    mask = mask & (rdf[sf] < st)
                else:
                    mask = mask & (rdf[sf] > st)
                catch = (mask & (rdf['return_pct'] < -5)).sum()
                fp = (mask & (rdf['return_pct'] > 10)).sum()
                fp_all = (mask & (rdf['return_pct'] > 0)).sum()
                total = mask.sum()
                if catch >= 4 and fp <= 2:
                    score = catch - 2 * fp
                    desc = f'{mf}>{mf_th:.2f} & {sf[:20]}{"<" if sd == "lt" else ">"}{st}'
                    best.append((score, desc, catch, fp, fp_all, total))

best.sort(key=lambda x: (x[0], -x[4]), reverse=True)

print(f'\n=== 2. Market+Stock Combination (catch>=4, FP_bigwin<=2) ===')
print(f'Base: {len(big_loss)} big_loss, {len(big_win)} big_win')
print(f'{"Rule":<78} {"Catch":>8} {"FPbw":>5} {"FPall":>5} {"Tot":>5} {"Scr":>4}')
print('-' * 110)
seen = set()
for score, desc, catch, fp, fp_all, total in best[:30]:
    if desc in seen: continue
    seen.add(desc)
    print(f'{desc:<78} {catch:>3}/28   {fp:>2}/42 {fp_all:>3}/59 {total:>4}  {score:>+4}')

# === 3. Best single consolidated rule ===
# Pick the top 1-2 market+stock combos and show detail
print(f'\n=== 3. Top Rules Detail ===')

# Show top 3 rules in detail
top_rules = best[:5]
seen_descs = set()
for score, desc, catch, fp, fp_all, total in top_rules:
    if desc in seen_descs: continue
    seen_descs.add(desc)
    # Reconstruct the mask
    # Parse desc back to conditions
    print(f'\n--- {desc} (catch={catch}/28, fp_bigwin={fp}/42) ---')

# === 4. Compare: V3 + 1 best new rule vs V3 alone ===
# V3 catches 16 trades (10 removed from 162->127=10, but some are not big loss)
# Actually V3 removes 10 trades total, 6 of which are big loss

# Let me find the single best rule that complements V3
# Focus on the 28-6=22 big losses that V3 DOESN'T catch (in 162 but V3 doesn't filter them)
# Actually V3 removes 10 trades, 6 are big loss -> 28-6=22 big losses remain
# But we need to check: in 169, 20 big losses remain

# The 8 big losses V3 catches (162 has 28, 169 has 20 = 8 removed)
v3_removed_big_loss = set()
set162 = set(df162[['stock_code','buy_date']].apply(tuple, axis=1))
set169 = set(df169[['stock_code','buy_date']].apply(tuple, axis=1))
removed = set162 - set169
for stock, buy_date in removed:
    row = df162[(df162['stock_code']==stock) & (df162['buy_date']==buy_date)]
    if len(row) > 0 and row['return_pct'].values[0] < -5:
        v3_removed_big_loss.add((stock, buy_date))
print(f'\nV3 removes {len(v3_removed_big_loss)} big losses from 162')
for s, b in v3_removed_big_loss:
    r = df162[(df162['stock_code']==s) & (df162['buy_date']==b)]['return_pct'].values[0]
    print(f'  {s} buy={b} ret={r:+.2f}%')

# === 5. Now: for the 20 V3-leaked big losses, which single rule catches most? ===
# These are the 20 big losses in 169
bl169 = df169[df169['return_pct'] < -5]
print(f'\nV3-leaked big losses in 169: {len(bl169)}')

# Match factor values for 169 trades
results169 = []
for _, trade in df169.iterrows():
    stock = trade['stock_code']
    buy_dt = pd.Timestamp(trade['buy_date']) if trade['buy_date'] else None
    if buy_dt is None: continue
    stock_data = df[df['stock_code'] == stock].sort_values('date')
    before = stock_data[stock_data['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    factor_row = stock_data[stock_data['date'] == pred_date]
    if len(factor_row) == 0: continue
    r = factor_row.iloc[0]
    rec = {'stock_code': stock, 'buy_date': trade['buy_date'], 'return_pct': trade['return_pct']}
    for nf in test_factors:
        if nf in df.columns: rec[nf] = r.get(nf, np.nan)
    results169.append(rec)

rdf169 = pd.DataFrame(results169)
bl169_rdf = rdf169[rdf169['return_pct'] < -5]
bw169_rdf = rdf169[rdf169['return_pct'] > 10]
allw169 = rdf169[rdf169['return_pct'] > 0]

print(f'169: {len(bl169_rdf)} big_loss, {len(bw169_rdf)} big_win, {len(allw169)} all_win')

# Scan best single rule on 169 leaked big losses
best169 = []
for mf in market_factors:
    if mf not in rdf169.columns: continue
    mf_vals = rdf169[mf].dropna()
    for mf_pct in [0.20, 0.25, 0.30, 0.35, 0.40]:
        mf_th = mf_vals.quantile(mf_pct)
        for sf, sd, st in stock_candidates:
            if sf not in rdf169.columns: continue
            mask = (rdf169[mf] < mf_th)
            if sd == 'lt':
                mask = mask & (rdf169[sf] < st)
            else:
                mask = mask & (rdf169[sf] > st)
            catch = (mask & (rdf169['return_pct'] < -5)).sum()
            fp = (mask & (rdf169['return_pct'] > 10)).sum()
            total = mask.sum()
            if catch >= 3 and fp <= 2:
                score = catch - 2 * fp
                desc = f'{mf}<{mf_th:.2f} & {sf[:20]}{"<" if sd == "lt" else ">"}{st}'
                best169.append((score, desc, catch, fp, total))

    if 'volatility' in mf or 'shock' in mf:
        for mf_pct in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mf_th = mf_vals.quantile(mf_pct)
            for sf, sd, st in stock_candidates:
                if sf not in rdf169.columns: continue
                mask = (rdf169[mf] > mf_th)
                if sd == 'lt':
                    mask = mask & (rdf169[sf] < st)
                else:
                    mask = mask & (rdf169[sf] > st)
                catch = (mask & (rdf169['return_pct'] < -5)).sum()
                fp = (mask & (rdf169['return_pct'] > 10)).sum()
                total = mask.sum()
                if catch >= 3 and fp <= 2:
                    score = catch - 2 * fp
                    desc = f'{mf}>{mf_th:.2f} & {sf[:20]}{"<" if sd == "lt" else ">"}{st}'
                    best169.append((score, desc, catch, fp, total))

best169.sort(key=lambda x: x[0], reverse=True)
print(f'\n=== Best Market+Stock Rule on V3-leaked big losses (169) ===')
print(f'{"Rule":<78} {"Catch":>8} {"FP":>5} {"Tot":>5} {"Scr":>4}')
print('-' * 105)
seen = set()
for score, desc, catch, fp, total in best169[:20]:
    if desc in seen: continue
    seen.add(desc)
    print(f'{desc:<78} {catch:>3}/20   {fp:>2}/42 {total:>4}  {score:>+4}')

# === 6. Show detail of top rule: which trades it catches ===
# Pick the absolute best rule
if best169:
    best_score, best_desc, best_catch, best_fp, best_total = best169[0]
    print(f'\n=== Top Rule Detail: {best_desc} ===')
    # Reconstruct
    # This is messy, let me just show the top 3 rules' caught trades
    for score, desc, catch, fp, total in best169[:3]:
        if desc in seen: continue
        print(f'\n--- {desc} (catch={catch}/20, fp={fp}/42) ---')
        # Parse and re-evaluate
        parts = desc.split(' & ')
        mask = pd.Series(True, index=rdf169.index)
        for part in parts:
            if '<' in part:
                fname, th = part.split('<')
                fname = fname.strip()
                th = float(th)
                # Find the matching column
                col = [c for c in rdf169.columns if c.startswith(fname[:15])]
                if col:
                    mask = mask & (rdf169[col[0]] < th)
            elif '>' in part:
                fname, th = part.split('>')
                fname = fname.strip()
                th = float(th)
                col = [c for c in rdf169.columns if c.startswith(fname[:15])]
                if col:
                    mask = mask & (rdf169[col[0]] > th)

        for _, row in rdf169[mask].iterrows():
            label = 'BIG_LOSS' if row['return_pct'] < -5 else ('BIG_WIN' if row['return_pct'] > 10 else ('win' if row['return_pct'] > 0 else 'loss'))
            print(f'  {row["stock_code"]} ret={row["return_pct"]:+.2f}% [{label}]')
