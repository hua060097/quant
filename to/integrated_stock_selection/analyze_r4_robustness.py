"""分析Rule4 (mkt_liq<0.95 & mfp<1.0) 的鲁棒性"""
import re, pandas as pd, numpy as np

# === 1. 加载影子交易 ===
shadow = pd.read_csv('./results/walkforward_shadow_skipped_opportunities_20260513_230111.csv')
shadow['pred_date'] = pd.to_datetime(shadow['shadow_prediction_date'])

# === 2. 加载因子 ===
df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

# === 3. 为影子交易匹配因子值 ===
need_factors = ['market_liquidity_change', 'money_flow_pressure',
                'post_accumulation_supply_tightness', 'tightness_3d']
shadow_results = []
for _, row in shadow.iterrows():
    stock = row['stock_code']
    pred_dt = row['pred_date']
    mask = (df['stock_code'] == stock) & (df['date'] == pred_dt)
    fp = df[mask]
    if len(fp) == 0:
        # Try 1 day before
        pred_dt2 = pred_dt - pd.Timedelta(days=1)
        mask = (df['stock_code'] == stock) & (df['date'] == pred_dt2)
        fp = df[mask]
    if len(fp) > 0:
        r = fp.iloc[0]
        rec = {
            'stock_code': stock,
            'buy_date': row['buy_datetime'][:10],
            'return_pct': row['actual_return'],
            'exhaustion_blocked': row['exhaustion_blocked'],
            'shadow_status': row['shadow_status'],
        }
        for nf in need_factors:
            rec[nf] = r.get(nf, np.nan)
        shadow_results.append(rec)
    else:
        rec = {
            'stock_code': stock,
            'buy_date': row['buy_datetime'][:10],
            'return_pct': row['actual_return'],
            'exhaustion_blocked': row['exhaustion_blocked'],
            'shadow_status': row['shadow_status'],
        }
        for nf in need_factors:
            rec[nf] = np.nan
        shadow_results.append(rec)

srdf = pd.DataFrame(shadow_results)
print(f'=== 1. 影子满仓机会 (n={len(srdf)}) ===')
print(f'匹配率: {srdf["market_liquidity_change"].notna().sum()}/{len(srdf)}')

# Apply Rule 4
srdf['r4_blocked'] = (srdf['market_liquidity_change'] < 0.95) & (srdf['money_flow_pressure'] < 1.0)

# Show all shadow trades with R4 result
print(f'\n{"Stock":<12} {"Ret":>7} {"ExhV3":>5} {"R4":>4} {"liq":>6} {"mfp":>7} {"Status"}')
print('-'*70)
for _, row in srdf.sort_values('return_pct').iterrows():
    r4 = 'R4' if row['r4_blocked'] else ''
    exh = 'V3' if row['exhaustion_blocked'] else ''
    tag = 'BIG_LOSS' if row['return_pct']<-5 else ('BIG_WIN' if row['return_pct']>10 else ('win' if row['return_pct']>0 else 'loss'))
    liq = f'{row["market_liquidity_change"]:.3f}' if pd.notna(row['market_liquidity_change']) else 'NaN'
    mfp = f'{row["money_flow_pressure"]:.3f}' if pd.notna(row['money_flow_pressure']) else 'NaN'
    print(f'{row["stock_code"]:<12} {row["return_pct"]:>+6.1f}% {exh:>5} {r4:>4} {liq:>6} {mfp:>7} [{tag}]')

# Summary
r4_blocked = srdf[srdf['r4_blocked']]
r4_big_loss = (r4_blocked['return_pct'] < -5).sum()
r4_big_win = (r4_blocked['return_pct'] > 10).sum()
r4_loss = (r4_blocked['return_pct'] < 0).sum()
r4_win = (r4_blocked['return_pct'] > 0).sum()
print(f'\nR4拦截影子交易: {len(r4_blocked)}/{len(srdf)}')
print(f'  其中大亏(<-5%): {r4_big_loss}')
print(f'  其中大赚(>10%): {r4_big_win}')
print(f'  其中亏损(<0%): {r4_loss}')
print(f'  其中盈利(>0%): {r4_win}')

# V3+R4 combined
srdf['v3_or_r4'] = srdf['exhaustion_blocked'] | srdf['r4_blocked']
combined = srdf[srdf['v3_or_r4']]
c_big_loss = (combined['return_pct'] < -5).sum()
c_big_win = (combined['return_pct'] > 10).sum()
print(f'\nV3+R4合计拦截影子交易: {len(combined)}/{len(srdf)}')
print(f'  其中大亏: {c_big_loss}')
print(f'  其中大赚: {c_big_win}')

# === 2. 分位鲁棒性分析 ===
# 用162日志(无风控)的全部交易

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

# Match factors for all 162 trades
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
    rec['market_liquidity_change'] = r.get('market_liquidity_change', np.nan)
    rec['money_flow_pressure'] = r.get('money_flow_pressure', np.nan)
    results.append(rec)

rdf = pd.DataFrame(results)
big_loss = rdf[rdf['return_pct'] < -5]
big_win = rdf[rdf['return_pct'] > 10]

print(f'\n\n=== 2. 分位鲁棒性分析 (162日志, {len(rdf)} trades) ===')
print(f'Big loss: {len(big_loss)}, Big win: {len(big_win)}')

# 2a. market_liquidity_change 分布
liq_vals = rdf['market_liquidity_change'].dropna()
print(f'\n--- market_liquidity_change 分布 ---')
for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    v = liq_vals.quantile(p)
    print(f'  p{int(p*100):>2}: {v:.3f}')
print(f'  当前阈值 0.95 对应分位: {(liq_vals < 0.95).mean()*100:.1f}%')

# 2b. money_flow_pressure 分布
mfp_vals = rdf['money_flow_pressure'].dropna()
print(f'\n--- money_flow_pressure 分布 ---')
for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    v = mfp_vals.quantile(p)
    print(f'  p{int(p*100):>2}: {v:.3f}')
print(f'  当前阈值 1.0 对应分位: {(mfp_vals < 1.0).mean()*100:.1f}%')

# 2c. Grid search: vary both thresholds
print(f'\n--- Grid Search: mkt_liq阈值 x mfp阈值 ---')
print(f'{"liq_th":>8} {"liq_pct":>8} {"mfp_th":>8} {"mfp_pct":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*70)

# Get percentile-based thresholds for both
liq_pcts = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
mfp_pcts = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for liq_pct in liq_pcts:
    liq_th = liq_vals.quantile(liq_pct)
    for mfp_pct in mfp_pcts:
        mfp_th = mfp_vals.quantile(mfp_pct)
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['money_flow_pressure'] < mfp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 3:
            print(f'{liq_th:>8.3f} {liq_pct*100:>7.0f}% {mfp_th:>8.3f} {mfp_pct*100:>7.0f}% {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/59 {total:>4}  {score:>+3}')

# 2d. Focus on the current threshold neighborhood
print(f'\n--- 精细扫描: 当前阈值 0.95 附近 ---')
print(f'{"liq_th":>8} {"mfp_th":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*60)

for liq_th in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]:
    for mfp_th in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 6.0]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['money_flow_pressure'] < mfp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 2:
            print(f'{liq_th:>8.3f} {mfp_th:>8.1f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/59 {total:>4}  {score:>+3}')

# 2e. 在全市场数据上看这两个因子的分位
print(f'\n--- 全市场因子分位 (3.5M rows) ---')
full_liq = df['market_liquidity_change'].dropna()
full_mfp = df['money_flow_pressure'].dropna()
print(f'market_liquidity_change:')
print(f'  0.95 对应分位: {(full_liq < 0.95).mean()*100:.1f}%')
for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    print(f'  p{int(p*100)}={full_liq.quantile(p):.3f}', end='')
print()

print(f'money_flow_pressure:')
print(f'  1.0 对应分位: {(full_mfp < 1.0).mean()*100:.1f}%')
for p in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    print(f'  p{int(p*100)}={full_mfp.quantile(p):.3f}', end='')
print()
