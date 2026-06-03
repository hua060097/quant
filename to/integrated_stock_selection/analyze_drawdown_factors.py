"""分析最大回撤段的因子异常"""
import re, pandas as pd, numpy as np

def parse_log(fp):
    trades, cb = [], {}
    with open(fp, 'r', encoding='utf-8') as f:
        for l in f:
            m = re.search(r'🔎 (\d+\.\w+) 开盘风控检查: buy_date=(\d{4}-\d{2}-\d{2})', l)
            if m: cb[m.group(1)] = m.group(2)
            m = re.search(r'卖出: (\d+\.\w+),.*?收益: (-?\d+\.?\d*) \((-?\d+\.?\d*)%\).*?原因: (.+)', l)
            if m:
                s, r = m.group(1), float(m.group(3))
                bd = cb.pop(s, None)
                trades.append({'stock_code': s, 'buy_date': bd, 'return_pct': r, 'exit_reason': m.group(4).strip()})
    return pd.DataFrame(trades)

d172 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子172_optuna.log')

# Focus on max drawdown streak: buy dates 2026-03-31, 04-01, 04-02, 04-07, 04-08
# Prediction dates are 1 trading day before each buy date
dd_trades = [
    ('600722.SH', '2026-03-31', -9.87),
    ('603538.SH', '2026-04-01', -2.20),
    ('603115.SH', '2026-04-02', -4.28),
    ('603618.SH', '2026-04-07', -7.63),
    ('300834.SZ', '2026-04-08', -6.11),
]

df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

# Market factors to check + individual stock factors
market_factors = [
    'market_liquidity_change', 'market_trend_strength', 'market_sentiment_ad',
    'market_volatility_context', 'market_advance_decline_ratio',
    'mkt_ma5_followthrough_v3', 'mkt_regime_trend_pressure_v2',
    'mkt_breadth_thrust_v2', 'mkt_emotion_stability_v2',
    'mkt_dispersion_shock_v2', 'mkt_liquidity_impulse_v2',
    'mkt_sector_dispersion',
    'market_60d_new_high_ratio',
]
stock_factors = [
    'post_accumulation_supply_tightness', 'tightness_3d', 'true_turnover_10d',
    'close_position_60d', 'close_position_20d',
    'money_flow_pressure', 'money_flow_pressure_pct',
    'volume_price_exhaustion', 'pullback_depth_from_10d_high',
    'efficiency_ratio', 'max_drawdown_20d', 'max_rally_10d',
    'runup_ma5_exhaustion', 'cpv_factor',
    'sector_momentum_accel_rank', 'sector_leadership_persistence_v3',
    'return_20d_intra_z', 'distance_from_52w_high',
    'stock_volume_ratio_20d', 'turnover_z_20',
]

print('=== 最大回撤段 (2026-03-27 ~ 2026-04-07 prediction dates) ===')
print()

# 1. Market factor values on each prediction date
print('--- 市场环境因子 (T日值) ---')
seen_dates = set()
for stock, buy_date, ret in dd_trades:
    buy_dt = pd.Timestamp(buy_date)
    sd = df[df['stock_code'] == stock].sort_values('date')
    before = sd[sd['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    if pred_date in seen_dates: continue
    seen_dates.add(pred_date)

    # Get any stock's row on that date for market factors (market factors are same for all stocks)
    day_data = df[df['date'] == pred_date]
    r = day_data.iloc[0]
    print(f'\npred_date={pred_date.strftime("%Y-%m-%d")}:')
    for mf in market_factors:
        if mf in r.index:
            val = r.get(mf, np.nan)
            if pd.notna(val):
                print(f'  {mf:<40} = {val:.4f}')

# 2. Individual stock factor values
print()
print('--- 个股因子值 (T日值) ---')
for stock, buy_date, ret in dd_trades:
    buy_dt = pd.Timestamp(buy_date)
    sd = df[df['stock_code'] == stock].sort_values('date')
    before = sd[sd['date'] < buy_dt]
    if len(before) == 0: continue
    pred_date = before['date'].max()
    r = sd[sd['date'] == pred_date].iloc[0]
    print(f'\n{stock} buy={buy_date} pred={pred_date.strftime("%Y-%m-%d")} ret={ret:+.2f}%')
    for sf in stock_factors:
        if sf in r.index:
            val = r.get(sf, np.nan)
            if pd.notna(val):
                print(f'  {sf:<40} = {val:.4f}')

# 3. Compare market factors: drawdown period vs normal periods
print()
print('=== 全时段市场因子对比: 回撤段 vs 正常段 ===')
all_dates = sorted(df['date'].unique())
dd_dates = sorted(seen_dates)
dd_mask = df['date'].isin(dd_dates)
normal_mask = ~dd_mask

dd_strs = [d.strftime('%Y-%m-%d') for d in dd_dates]
print(f'DD dates ({len(dd_dates)}): {dd_strs}')
print()
print(f'{"Factor":<40} {"DD_mean":>10} {"Normal_mean":>10} {"Diff":>10} {"DD_pct":>8}')
print('-'*80)
for mf in market_factors:
    if mf not in df.columns: continue
    dd_vals = df.loc[dd_mask, mf].dropna()
    norm_vals = df.loc[normal_mask, mf].dropna()
    if len(dd_vals) < 2: continue
    dd_m = dd_vals.mean()
    n_m = norm_vals.mean()
    # What percentile is dd_mean in the normal distribution?
    pct = (norm_vals < dd_m).mean() * 100
    print(f'{mf:<40} {dd_m:>10.4f} {n_m:>10.4f} {dd_m-n_m:>+10.4f} {pct:>7.1f}%')

# 4. For each market factor, test: would filtering on DD dates have helped?
print()
print('=== 市场因子阈值过滤测试: 如果拦截DD日期的所有交易 ===')
for mf in market_factors:
    if mf not in df.columns: continue
    dd_vals = df.loc[dd_mask, mf].dropna()
    if len(dd_vals) < 2: continue

    # Try different thresholds
    for pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
        th = dd_vals.quantile(pct)
        # How many DD-day trades would this block?
        blocked = (dd_vals < th).sum()
        # But this is per-stock... let's check at the date level
        date_blocked = set()
        for d in dd_dates:
            day_vals = df.loc[(df['date'] == d) & df[mf].notna(), mf]
            if len(day_vals) == 0: continue
            if (day_vals < th).mean() > 0.5:  # Most stocks below threshold
                date_blocked.add(d)
        print(f'  {mf[:35]} p{pct*100:.0f}(th={th:.3f}): blocks {len(date_blocked)}/{len(dd_dates)} DD dates')
        if len(date_blocked) > 0:
            break
