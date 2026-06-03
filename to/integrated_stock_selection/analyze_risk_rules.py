"""分析新因子+已有因子的组合风控规则"""
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
df169 = parse_log('stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log')

print(f'162(无风控): {len(df162)} trades | 169(V3): {len(df169)} trades')
print(f'162 大亏(<-5%): {len(df162[df162["return_pct"]<-5])}, 大赚(>10%): {len(df162[df162["return_pct"]>10])}')
print(f'169 大亏(<-5%): {len(df169[df169["return_pct"]<-5])}, 大赚(>10%): {len(df169[df169["return_pct"]>10])}')

df = pd.read_parquet('./factor_lab_data/factors.parquet')
df['date'] = pd.to_datetime(df['date'])

test_factors = [
    'post_accumulation_supply_tightness', 'true_turnover_10d', 'tightness_3d',
    'runup_ma5_exhaustion', 'cpv_factor', 'prev_day_blowoff_risk',
    'sector_momentum_accel_rank', 'sector_leadership_persistence_v3', 'return_20d_intra_z',
    'market_trend_strength', 'market_volatility_context',
    'mkt_ma5_followthrough_v3', 'mkt_regime_trend_pressure_v2',
    'mkt_breadth_thrust_v2', 'mkt_emotion_stability_v2',
    'close_position_20d', 'close_position_60d', 'upper_shadow_pressure_5d',
    'high_volume_no_gain_3d', 'pullback_depth_from_10d_high',
    'close_to_high_3d_mean', 'high_pos_upper_shadow', 'high_pos_volume_no_gain',
    'high_pos_reversal_volume', 'shadow_reversal_volume_pressure',
    'momentum_peak_decay_3d', 'runup_low_break',
    'close_below_ma5_after_runup', 'low_break_prev_low_after_runup',
    'ma5_ma10_flattening_after_runup', 'carryover_overheat_risk',
    'distribution_after_strength_risk', 'consecutive_up_days_5d',
    'max_drawdown_20d', 'ulcer_index', 'efficiency_ratio',
    'volume_price_exhaustion', 'money_flow_pressure', 'abnormal_turnover', 'turnover_z_20',
    'stock_volume_ratio_20d', 'high_position_volume_surge',
    'failed_breakout_pressure', 'volatility_structure_alpha',
    'ret_3d_vs_10d_accel', 'prior_high_proximity_20d', 'prior_high_volume_contest_20d',
    'distance_from_52w_high', 'distance_from_52w_low',
    'severe_abnormal_pressure_10d', 'max_rally_10d', 'max_rally_20d',
    'net_d5_amount_to_turnover', 'buy_lg_amount_rate', 'buy_sm_amount_rate',
    'market_advance_decline_ratio', 'market_liquidity_change',
    'mkt_dispersion_shock_v2', 'mkt_liquidity_impulse_v2',
    'regime_aware_reversal_alpha', 'high_volatility_pullback_alpha',
    'volume_confirmed_pullback_alpha', 'volume_structure_alpha',
]

# Match factor values for 162 (no filter baseline)
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
print(f'\nMatched {len(rdf)} trades, big_loss={len(big_loss)}, big_win={len(big_win)}, all_win={len(all_win)}')

# === 1. 逐笔列出大亏和大赚的3新因子+top分离因子 ===
top_factors = ['post_accumulation_supply_tightness', 'tightness_3d',
               'pullback_depth_from_10d_high', 'sector_momentum_accel_rank',
               'close_position_60d', 'money_flow_pressure',
               'momentum_peak_decay_3d', 'volume_price_exhaustion',
               'efficiency_ratio', 'prior_high_proximity_20d']

print('\n=== 28笔大亏交易的因子值 ===')
for _, row in big_loss.sort_values('return_pct').iterrows():
    vals = ' | '.join([f'{nf[:12]}={row.get(nf, np.nan):.2f}' for nf in top_factors if pd.notna(row.get(nf, np.nan))])
    print(f'{row["stock_code"]} ret={row["return_pct"]:+.1f}%: {vals}')

# === 2. 误杀检查: supply<-1.0 的那1笔误杀盈利是什么 ===
print('\n=== supply<-1.0 误杀检查 ===')
supply_low = rdf[rdf['post_accumulation_supply_tightness'] < -1.0]
for _, row in supply_low.iterrows():
    label = '大亏' if row['return_pct'] < -5 else ('大赚' if row['return_pct'] > 10 else ('小赚' if row['return_pct'] > 0 else '小亏'))
    print(f'{row["stock_code"]} ret={row["return_pct"]:+.2f}% [{label}] supply={row["post_accumulation_supply_tightness"]:.3f}')

# === 3. 组合规则扫描 ===
def eval_rule(rdf, conditions):
    mask = pd.Series(True, index=rdf.index)
    for factor, direction, threshold in conditions:
        if factor not in rdf.columns: return pd.Series(False, index=rdf.index)
        if direction == 'lt':
            mask = mask & (rdf[factor] < threshold)
        else:
            mask = mask & (rdf[factor] > threshold)
    return mask

combos = [
    # === 新因子 + 新因子 ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('tightness_3d', 'lt', -1.0)],
    [('post_accumulation_supply_tightness', 'lt', -1.0), ('tightness_3d', 'lt', -0.5)],
    # === 新因子 + position因子 ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('close_position_60d', 'lt', 0.8)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('close_position_20d', 'lt', 0.75)],
    [('tightness_3d', 'lt', -0.5), ('close_position_60d', 'lt', 0.8)],
    # === 新因子 + sector因子 ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('tightness_3d', 'lt', -0.5), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('sector_leadership_persistence_v3', 'lt', -0.05)],
    # === 新因子 + pullback/depth ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('pullback_depth_from_10d_high', 'lt', -0.1)],
    [('tightness_3d', 'lt', -0.5), ('pullback_depth_from_10d_high', 'lt', -0.1)],
    # === 新因子 + money_flow ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('money_flow_pressure', 'lt', 1.0)],
    [('tightness_3d', 'lt', -0.5), ('money_flow_pressure', 'lt', 1.0)],
    # === 新因子 + exhaustion (V3因子) ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('volume_price_exhaustion', 'gt', 0.2)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('momentum_peak_decay_3d', 'gt', 0)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('runup_ma5_exhaustion', 'gt', -1.0)],
    [('tightness_3d', 'lt', -0.5), ('momentum_peak_decay_3d', 'gt', 0)],
    [('tightness_3d', 'lt', -0.5), ('volume_price_exhaustion', 'gt', 0.2)],
    # === 新因子 + efficiency ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('efficiency_ratio', 'lt', 1.5)],
    [('tightness_3d', 'lt', -0.5), ('efficiency_ratio', 'lt', 1.5)],
    # === 新因子 + prior_high ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('prior_high_proximity_20d', 'lt', 0)],
    [('tightness_3d', 'lt', -0.5), ('prior_high_proximity_20d', 'lt', 0)],
    # === position + sector (无新因子) ===
    [('close_position_60d', 'lt', 0.8), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('close_position_60d', 'lt', 0.8), ('money_flow_pressure', 'lt', 1.0)],
    # === momentum_peak + sector ===
    [('momentum_peak_decay_3d', 'gt', 1.0), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('momentum_peak_decay_3d', 'gt', 0.5), ('close_position_60d', 'lt', 0.8)],
    # === V3+新因子 组合 ===
    [('runup_ma5_exhaustion', 'gt', 0.8), ('post_accumulation_supply_tightness', 'lt', 0)],
    [('runup_ma5_exhaustion', 'gt', 0.8), ('tightness_3d', 'lt', -0.5)],
    # === 3因子组合 ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('close_position_60d', 'lt', 0.8), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('money_flow_pressure', 'lt', 1.0), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('tightness_3d', 'lt', -0.5), ('close_position_60d', 'lt', 0.8), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('momentum_peak_decay_3d', 'gt', 0), ('sector_momentum_accel_rank', 'lt', 0.3)],
    [('tightness_3d', 'lt', -0.5), ('momentum_peak_decay_3d', 'gt', 0), ('sector_momentum_accel_rank', 'lt', 0.3)],
    # === supply + high_volume_no_gain ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('high_volume_no_gain_3d', 'gt', 0.5)],
    # === supply + max_drawdown ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('max_drawdown_20d', 'lt', 0)],
    # === supply + runup_low_break ===
    [('post_accumulation_supply_tightness', 'lt', -0.65), ('runup_low_break', 'gt', -0.5)],
]

print('\n=== 组合规则扫描结果 ===')
print(f'基准: {len(big_loss)}大亏, {len(big_win)}大赚, {len(all_win)}全盈利')
print()

results_list = []
for conditions in combos:
    mask = eval_rule(rdf, conditions)
    catch = (mask & (rdf['return_pct'] < -5)).sum()
    fp_big = (mask & (rdf['return_pct'] > 10)).sum()
    fp_all = (mask & (rdf['return_pct'] > 0)).sum()
    total = mask.sum()
    if total == 0: continue
    desc = ' & '.join([f'{f[:15]}{"<" if d=="lt" else ">"}{t}' for f,d,t in conditions])
    catch_pct = catch/len(big_loss)*100
    fp_pct = fp_big/len(big_win)*100
    # Score: catch - 2*fp_bigwin (penalize killing big wins more)
    score = catch - 2*fp_big
    results_list.append((score, desc, catch, len(big_loss), fp_big, len(big_win), fp_all, total))

results_list.sort(key=lambda x: x[0], reverse=True)
print(f'{"Rule":<75} {"Catch":>8} {"FP_big":>8} {"FP_all":>7} {"Total":>6} {"Score":>6}')
print('-'*115)
for score, desc, catch, ctot, fpb, fwt, fpa, total in results_list[:30]:
    print(f'{desc:<75} {catch:>3}/{ctot:<3} {fpb:>3}/{fwt:<3} {fpa:>4} {total:>5} {score:>+6}')

# === 4. V3漏网的大亏，新规则能补多少 ===
# V3 catches: 10 trades removed from 162->169 (127->117)
# Find which 10 trades V3 removed
print('\n=== V3拦截的10笔交易 ===')
set162 = set(df162[['stock_code','buy_date']].apply(tuple, axis=1))
set169 = set(df169[['stock_code','buy_date']].apply(tuple, axis=1))
removed = set162 - set169
for stock, buy_date in removed:
    row = df162[(df162['stock_code']==stock) & (df162['buy_date']==buy_date)]
    if len(row) > 0:
        ret = row['return_pct'].values[0]
        label = '大亏' if ret < -5 else ('大赚' if ret > 10 else ('小赚' if ret > 0 else '小亏'))
        print(f'  {stock} buy={buy_date} ret={ret:+.2f}% [{label}]')

# Big losses in 169 (V3 leaks)
print('\n=== V3漏网的大亏 (169中的大亏) ===')
bl169 = df169[df169['return_pct'] < -5]
for _, row in bl169.sort_values('return_pct').iterrows():
    print(f'  {row["stock_code"]} buy={row["buy_date"]} ret={row["return_pct"]:+.2f}%')
