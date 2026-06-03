"""分析R4三个候选方案的鲁棒性对比"""
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

test_factors = ['market_liquidity_change', 'money_flow_pressure', 'close_position_60d']

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
all_win = rdf[rdf['return_pct'] > 0]

# 计算mfp的每日截面分位
print('计算money_flow_pressure每日截面分位...')
mfp_daily = df.groupby('date')['money_flow_pressure'].rank(pct=True)
df_with_idx = df.reset_index()
mfp_cs_rank = df.groupby('date')['money_flow_pressure'].rank(pct=True).values
rdf['mfp_cs_rank'] = np.nan
for i, (_, trade) in enumerate(rdf.iterrows()):
    stock = trade['stock_code']
    buy_dt = pd.Timestamp(trade['buy_date']) if trade['buy_date'] else None
    if buy_dt is None: continue
    stock_data = df[(df['stock_code'] == stock)].sort_values('date')
    before = stock_data[stock_data['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    # Get the cross-sectional rank of mfp on that date
    day_data = df[df['date'] == pred_date]
    day_mfp = day_data[['stock_code', 'money_flow_pressure']].copy()
    day_mfp['mfp_cs_rank'] = day_mfp['money_flow_pressure'].rank(pct=True)
    match = day_mfp[day_mfp['stock_code'] == stock]
    if len(match) > 0:
        rdf.loc[rdf.index[i], 'mfp_cs_rank'] = match['mfp_cs_rank'].values[0]

# 同样计算close_position_60d的每日截面分位
print('计算close_position_60d每日截面分位...')
rdf['cp60_cs_rank'] = np.nan
for i, (_, trade) in enumerate(rdf.iterrows()):
    stock = trade['stock_code']
    buy_dt = pd.Timestamp(trade['buy_date']) if trade['buy_date'] else None
    if buy_dt is None: continue
    stock_data = df[(df['stock_code'] == stock)].sort_values('date')
    before = stock_data[stock_data['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    day_data = df[df['date'] == pred_date]
    day_cp = day_data[['stock_code', 'close_position_60d']].copy()
    day_cp['cp60_cs_rank'] = day_cp['close_position_60d'].rank(pct=True)
    match = day_cp[day_cp['stock_code'] == stock]
    if len(match) > 0:
        rdf.loc[rdf.index[i], 'cp60_cs_rank'] = match['cp60_cs_rank'].values[0]

print(f'Big loss: {len(big_loss)}, Big win: {len(big_win)}, All win: {len(all_win)}')
print(f'mfp_cs_rank valid: {rdf["mfp_cs_rank"].notna().sum()}/{len(rdf)}')
print(f'cp60_cs_rank valid: {rdf["cp60_cs_rank"].notna().sum()}/{len(rdf)}')

# ============================================================
# 方案A: mkt_liq < threshold & close_position_60d < threshold
# ============================================================
print('\n' + '='*100)
print('方案A: market_liquidity_change < X AND close_position_60d < Y')
print('='*100)

liq_vals = rdf['market_liquidity_change'].dropna()
cp_vals = rdf['close_position_60d'].dropna()

print(f'\n{"liq_th":>8} {"liq%":>6} {"cp60_th":>8} {"cp60%":>6} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*70)

for liq_pct in [0.20, 0.25, 0.30, 0.35, 0.40]:
    liq_th = liq_vals.quantile(liq_pct)
    for cp_pct in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        cp_th = cp_vals.quantile(cp_pct)
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['close_position_60d'] < cp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 3:
            print(f'{liq_th:>8.3f} {liq_pct*100:>5.0f}% {cp_th:>8.3f} {cp_pct*100:>5.0f}% {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# 精细扫描
print(f'\n--- 方案A 精细扫描 ---')
print(f'{"liq_th":>8} {"cp60_th":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*60)

for liq_th in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    for cp_th in [0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['close_position_60d'] < cp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 2:
            print(f'{liq_th:>8.3f} {cp_th:>8.2f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# ============================================================
# 方案B: mkt_liq < threshold & mfp_cs_rank < threshold (截面分位)
# ============================================================
print('\n' + '='*100)
print('方案B: market_liquidity_change < X AND mfp_cs_rank < Y (截面分位版)')
print('='*100)

print(f'\n{"liq_th":>8} {"liq%":>6} {"mfp_rank":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*70)

for liq_pct in [0.20, 0.25, 0.30, 0.35, 0.40]:
    liq_th = liq_vals.quantile(liq_pct)
    for mfp_rank_th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['mfp_cs_rank'] < mfp_rank_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 3:
            print(f'{liq_th:>8.3f} {liq_pct*100:>5.0f}% {mfp_rank_th:>8.2f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# 精细扫描
print(f'\n--- 方案B 精细扫描 ---')
print(f'{"liq_th":>8} {"mfp_rank":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*60)

for liq_th in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    for mfp_rank_th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['mfp_cs_rank'] < mfp_rank_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 2:
            print(f'{liq_th:>8.3f} {mfp_rank_th:>8.2f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# ============================================================
# 方案C (原方案): mkt_liq < X AND mfp < Y (绝对值)
# ============================================================
print('\n' + '='*100)
print('方案C (原方案): market_liquidity_change < X AND money_flow_pressure < Y (绝对值)')
print('='*100)

print(f'\n{"liq_th":>8} {"mfp_th":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*60)

for liq_th in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    for mfp_th in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['money_flow_pressure'] < mfp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 2:
            print(f'{liq_th:>8.3f} {mfp_th:>8.1f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# ============================================================
# 方案D: mkt_liq < X AND cp60_cs_rank < Y (截面分位版)
# ============================================================
print('\n' + '='*100)
print('方案D: market_liquidity_change < X AND cp60_cs_rank < Y (截面分位版)')
print('='*100)

print(f'\n{"liq_th":>8} {"cp60_rank":>8} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6} {"Score":>6}')
print('-'*60)

for liq_th in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]:
    for cp_rank_th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['cp60_cs_rank'] < cp_rank_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        score = catch - 2 * fp
        if catch >= 2:
            print(f'{liq_th:>8.3f} {cp_rank_th:>8.2f} {catch:>3}/{len(big_loss):<2} {fp:>2}/{len(big_win):<2} {fp_all:>2}/{len(all_win):<2} {total:>4}  {score:>+3}')

# ============================================================
# 最终对比: 三个方案在零误杀区域的最佳表现
# ============================================================
print('\n' + '='*100)
print('最终对比: 零误杀大赚区域')
print('='*100)

# 找每个方案在 FP=0 时的最大catch
print(f'\n{"方案":<55} {"Catch":>8} {"FPbw":>6} {"FPall":>6} {"Total":>6}')
print('-'*85)

# 方案C: 原方案
for liq_th in [0.94, 0.95, 0.96]:
    for mfp_th in [0.5, 0.8, 1.0]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['money_flow_pressure'] < mfp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        if fp == 0 and catch >= 3:
            print(f'C: liq<{liq_th:.2f} & mfp<{mfp_th:.1f} (绝对值){"":>20} {catch:>3}/28   {fp:>2}/42 {fp_all:>2}/59 {total:>4}')

# 方案A: cp60绝对值
for liq_th in [0.94, 0.95, 0.96]:
    for cp_th in [0.75, 0.80, 0.85]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['close_position_60d'] < cp_th)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        if fp == 0 and catch >= 3:
            print(f'A: liq<{liq_th:.2f} & cp60<{cp_th:.2f} (绝对值){"":>19} {catch:>3}/28   {fp:>2}/42 {fp_all:>2}/59 {total:>4}')

# 方案B: mfp截面分位
for liq_th in [0.94, 0.95, 0.96]:
    for mfp_rank in [0.20, 0.25, 0.30, 0.35]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['mfp_cs_rank'] < mfp_rank)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        if fp == 0 and catch >= 3:
            print(f'B: liq<{liq_th:.2f} & mfp_rank<{mfp_rank:.2f} (截面分位){"":>13} {catch:>3}/28   {fp:>2}/42 {fp_all:>2}/59 {total:>4}')

# 方案D: cp60截面分位
for liq_th in [0.94, 0.95, 0.96]:
    for cp_rank in [0.20, 0.25, 0.30, 0.35]:
        mask = (rdf['market_liquidity_change'] < liq_th) & (rdf['cp60_cs_rank'] < cp_rank)
        catch = (mask & (rdf['return_pct'] < -5)).sum()
        fp = (mask & (rdf['return_pct'] > 10)).sum()
        fp_all = (mask & (rdf['return_pct'] > 0)).sum()
        total = mask.sum()
        if fp == 0 and catch >= 3:
            print(f'D: liq<{liq_th:.2f} & cp60_rank<{cp_rank:.2f} (截面分位){"":>12} {catch:>3}/28   {fp:>2}/42 {fp_all:>2}/59 {total:>4}')
