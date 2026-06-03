"""
分析：因子极端值是否支持收紧开盘跌幅阈值

核心问题：当前Top1的买入规则是 gap_ratio < -0.06 跳过。
在某些因子处于极端值时，是否需要将阈值从-6%收紧（如-4%）？

分析步骤：
1. 构建主数据集（gap_ratio + 因子值 + 收益）
2. 描述性分析（gap分档 × 因子极端 × 盈亏）
3. 因子条件阈值收紧评估
4. 决策表汇总
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
LOG_FILE = BASE_DIR / "stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log"
FACTOR_FILE = BASE_DIR / "factor_lab_data" / "factors.parquet"
SHADOW_FILE = BASE_DIR / "results" / "walkforward_shadow_skipped_opportunities_20260513_230111.csv"

KEY_FACTORS = [
    'sector_momentum_interaction',
    'return_20d_intra_z',
    'price_ma60_ratio',
    'ma60_slope',
    'trend_acceleration',
    'ma_convergence_score'
]

# ============================================================
# Step 1: 从日志提取交易 + gap_ratio
# ============================================================
print("=" * 80)
print("Step 1: 从日志提取交易记录和gap_ratio")
print("=" * 80)

lines = open(LOG_FILE, 'r', encoding='utf-8').readlines()

# 1a: Parse all "跳空计算" lines to build a lookup: (stock_code, date_context) -> gap_ratio
# The log lines look like:
# "📊 688189.SH: 跳空计算 (buy_open=16.91 - prev_close=17.57) / prev_close = -0.0376 (-3.76%)"
gap_records = {}
current_date_context = None

for i, line in enumerate(lines):
    # Track date context from lines like "2025-08-01 ..."
    m_date = re.search(r'^\d{4}-\d{2}-\d{2}', line)
    if m_date:
        current_date_context = m_date.group(0)

    m_gap = re.search(r'📊 (\S+): 跳空计算.*?=\s*(-?[\d.]+)\s*\((-?[\d.]+)%\)', line)
    if m_gap:
        stock = m_gap.group(1)
        gap_ratio = float(m_gap.group(2))
        gap_pct = float(m_gap.group(3))
        gap_records[(stock, current_date_context)] = {
            'gap_ratio': gap_ratio,
            'gap_pct': gap_pct
        }

print(f"解析到 {len(gap_records)} 条跳空计算记录")

# 1b: Extract buys (same pattern as analyze_big_loss_with_factors.py)
buys = []
current_stock = None
current_buy_date = None
current_buy_price = None

for i, line in enumerate(lines):
    m_stock = re.search(r'分层并行仓位生效: (\S+) 使用', line)
    if m_stock:
        current_stock = m_stock.group(1)

    m_tday = re.search(r'并行持仓买入:.*?T日=(\d{4}-\d{2}-\d{2})', line)
    if m_tday:
        current_buy_date = m_tday.group(1)

    m_buyp = re.search(r'实际买入价[：:]?\s*([\d.]+)', line)
    if m_buyp:
        current_buy_price = float(m_buyp.group(1))

    m_confirm = re.search(r'买入后持仓状态', line)
    if m_confirm and current_stock:
        buys.append({
            'stock_code': current_stock,
            'buy_date': current_buy_date,
            'buy_price': current_buy_price,
        })
        current_stock = None
        current_buy_date = None
        current_buy_price = None

print(f"提取到买入记录: {len(buys)} 条")

# 1c: Extract sells
sells = []
for line in lines:
    m = re.search(r'卖出: (\S+), 价格: ([\d.]+), 收益: -?[\d.]+ \((-?[\d.]+)%\), 原因: (.+)', line)
    if m:
        sells.append({
            'stock_code': m.group(1),
            'sell_price': float(m.group(2)),
            'return_pct': float(m.group(3)),
            'exit_reason': m.group(4).strip()
        })
print(f"提取到卖出记录: {len(sells)} 条")

# 1d: Match buys to sells
df_buys = pd.DataFrame(buys)
df_sells = pd.DataFrame(sells)

matched = []
buy_used = [False] * len(df_buys)
for _, sell in df_sells.iterrows():
    for j, buy in df_buys.iterrows():
        if not buy_used[j] and buy['stock_code'] == sell['stock_code']:
            matched.append({
                'stock_code': sell['stock_code'],
                'buy_date': buy['buy_date'],
                'buy_price': buy['buy_price'],
                'sell_price': sell['sell_price'],
                'return_pct': sell['return_pct'],
                'exit_reason': sell['exit_reason'],
            })
            buy_used[j] = True
            break

df_trades = pd.DataFrame(matched)
print(f"成功配对交易: {len(df_trades)} 条")

# 1e: Match gap_ratio to each trade
# For each trade, find the gap_record matching (stock_code, buy_date)
# The buy_date in trades IS the T date (same as date_context in log)

def find_gap_for_trade(row):
    stock = row['stock_code']
    buy_date = str(row['buy_date'])[:10]

    # Try exact match first
    key = (stock, buy_date)
    if key in gap_records:
        return gap_records[key]['gap_ratio']

    # Try matching just by stock_code to the closest date context
    # (fallback: find any gap record for this stock)
    for (s, d), v in gap_records.items():
        if s == stock:
            return v['gap_ratio']
    return None

df_trades['gap_ratio'] = df_trades.apply(find_gap_for_trade, axis=1)
n_matched = df_trades['gap_ratio'].notna().sum()
print(f"gap_ratio匹配成功: {n_matched}/{len(df_trades)}")
print(f"gap_ratio缺失的trade数: {len(df_trades) - n_matched}")

# 1f: Load shadow trades (they have entry_gap_ratio already)
df_shadow = pd.read_csv(SHADOW_FILE)
print(f"\n影子交易: {len(df_shadow)} 条")

# gap_ratio is in 'entry_gap_ratio' column
df_shadow['gap_ratio'] = df_shadow['entry_gap_ratio'] if 'entry_gap_ratio' in df_shadow.columns else None

df_shadow['source'] = 'shadow'
df_trades['source'] = 'real'

# For shadow trades, use buy_price (already in CSV), profit_loss_pct as return_pct
shadow_for_merge = df_shadow[['stock_code', 'shadow_prediction_date', 'buy_price', 'sell_price', 'profit_loss_pct', 'exit_reason', 'gap_ratio']].copy()
shadow_for_merge = shadow_for_merge.rename(
    columns={'shadow_prediction_date': 'buy_date', 'profit_loss_pct': 'return_pct'}
)
shadow_for_merge['source'] = 'shadow'
shadow_for_merge['buy_date'] = shadow_for_merge['buy_date'].astype(str).str[:10]

# Combine
df_all = pd.concat([
    df_trades[['stock_code', 'buy_date', 'buy_price', 'sell_price', 'return_pct', 'exit_reason', 'gap_ratio', 'source']],
    shadow_for_merge[['stock_code', 'buy_date', 'buy_price', 'sell_price', 'return_pct', 'exit_reason', 'gap_ratio', 'source']]
], ignore_index=True)

# Clean up buy_date
df_all['buy_date'] = df_all['buy_date'].apply(lambda x: str(x)[:10])

print(f"\n合并后总交易: {len(df_all)} (real={len(df_trades)}, shadow={len(df_shadow)})")

# ============================================================
# Step 2: 合并因子数据
# ============================================================
print("\n" + "=" * 80)
print("Step 2: 合并因子数据")
print("=" * 80)

print("加载因子文件...")
df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}
print(f"因子文件: {df_factors.shape}, 交易日: {len(all_dates)}, 范围: {all_dates[0]} ~ {all_dates[-1]}")

# For each trade, get factors at buy_date (prediction_date = T-1, but the factor values
# at buy_date T are what the model saw at T-1's close, which should be factors at date=T-1)
# Actually: the factors.parquet has date = the trading day. The model predicts at day T_close
# for T+1 execution. So the factors at prediction_date (T-1) are what matters.
# But for buy-entry gap analysis, we should use factors at buy_date T (the day the stock opens).
# The gap_ratio uses (open_T - close_{T-1}) / close_{T-1}, so the proper context is the
# factors at T-1 (prediction date).

# Get prediction_date (T-1) for each trade
def get_prediction_date(buy_date_str):
    bd = pd.Timestamp(buy_date_str)
    if bd not in date_to_idx:
        return None
    idx = date_to_idx[bd]
    if idx > 0:
        return all_dates[idx - 1]
    return None

df_all['prediction_date'] = df_all['buy_date'].apply(get_prediction_date)
n_pred = df_all['prediction_date'].notna().sum()
print(f"成功计算prediction_date: {n_pred}/{len(df_all)}")

# Merge factors at prediction_date
available_factor_cols = [c for c in KEY_FACTORS if c in df_factors.columns]
missing = [c for c in KEY_FACTORS if c not in df_factors.columns]
if missing:
    print(f"WARNING: 因子文件缺失以下列: {missing}")
    # Check for similar names
    for m in missing:
        similar = [c for c in df_factors.columns if m[:10] in c]
        if similar:
            print(f"  可能的替代列 '{m}': {similar[:5]}")
print(f"可用关键因子: {available_factor_cols}")

# Merge factors
trade_factors_list = []
for _, row in df_all.iterrows():
    stock = row['stock_code']
    pred_date = row['prediction_date']

    if pred_date is None:
        trade_factors_list.append({f: None for f in available_factor_cols})
        continue

    mask = (df_factors['stock_code'] == stock) & (df_factors['date'] == pred_date)
    factor_row = df_factors.loc[mask, available_factor_cols]

    if len(factor_row) > 0:
        vals = factor_row.iloc[0].to_dict()
    else:
        vals = {f: None for f in available_factor_cols}
    trade_factors_list.append(vals)

df_factors_merged = pd.DataFrame(trade_factors_list)
df_all = pd.concat([df_all.reset_index(drop=True), df_factors_merged.reset_index(drop=True)], axis=1)
print(f"因子合并完成: {len(df_all)} 笔交易, {len(available_factor_cols)} 个因子")

# ============================================================
# Step 3: 计算因子截面分位数
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 计算因子截面分位数（在同日全市场中的百分位）")
print("=" * 80)

# For each prediction_date, compute cross-sectional percentiles
pred_dates = df_all['prediction_date'].dropna().unique()
pred_dates_set = set(pred_dates)

df_factors_sub = df_factors[df_factors['date'].isin(pred_dates_set)].copy()

# Precompute percentiles for each (date, factor)
percentile_cache = {}
for dt in pred_dates:
    dt_data = df_factors[df_factors['date'] == dt]
    for f in available_factor_cols:
        vals = dt_data[f].dropna()
        if len(vals) > 0:
            percentile_cache[(dt, f)] = vals

def get_percentile(pred_date, factor_name, value):
    """Return cross-sectional percentile of value on pred_date"""
    if value is None or pd.isna(value):
        return None
    key = (pred_date, factor_name)
    if key not in percentile_cache:
        return None
    try:
        return stats.percentileofscore(percentile_cache[key], value)
    except:
        return None

for f in available_factor_cols:
    df_all[f'{f}_pct'] = df_all.apply(
        lambda row: get_percentile(row['prediction_date'], f, row[f]), axis=1
    )

print(f"分位数计算完成: {len(available_factor_cols)} 个因子")

# ============================================================
# Step 4: 描述性分析 - gap分档 × 因子极端 × 盈亏
# ============================================================
print("\n" + "=" * 80)
print("Step 4: 描述性分析 - gap分布与盈亏关系")
print("=" * 80)

has_gap = df_all['gap_ratio'].notna()
print(f"有gap_ratio的交易: {has_gap.sum()}/{len(df_all)}")

df_with_gap = df_all[has_gap].copy()

# Define gap bands
bins = [-float('inf'), -0.10, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, float('inf')]
labels = ['<-10%', '-10%~-6%', '-6%~-4%', '-4%~-2%', '-2%~0%', '0%~2%', '2%~4%', '4%~6%', '>6%']
df_with_gap['gap_band'] = pd.cut(df_with_gap['gap_ratio'], bins=bins, labels=labels)

# Outcome labels
df_with_gap['is_big_loss'] = df_with_gap['return_pct'] < -5
df_with_gap['is_big_win'] = df_with_gap['return_pct'] > 10
df_with_gap['is_ok'] = (df_with_gap['return_pct'] >= -5) & (df_with_gap['return_pct'] <= 10)

print("\nGap分档 × 盈亏分布:")
gap_summary = df_with_gap.groupby('gap_band').agg(
    trade_count=('return_pct', 'count'),
    mean_return=('return_pct', 'mean'),
    median_return=('return_pct', 'median'),
    big_loss_count=('is_big_loss', 'sum'),
    big_win_count=('is_big_win', 'sum'),
    ok_count=('is_ok', 'sum'),
).reset_index()

pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.float_format', '{:.2f}'.format)
print(gap_summary.to_string())

# Focus on the (-6%, -4%] band
key_band = df_with_gap[df_with_gap['gap_band'] == '-6%~-4%'].copy()
print(f"\n关键区间 (-6%, -4%] 交易: {len(key_band)} 笔")
if len(key_band) > 0:
    print(f"  大亏: {key_band['is_big_loss'].sum()}, 大赚: {key_band['is_big_win'].sum()}")
    print(f"  平均收益: {key_band['return_pct'].mean():.2f}%")
    print("\n每笔明细:")
    detail_cols = ['stock_code', 'buy_date', 'gap_ratio', 'return_pct', 'exit_reason']
    detail_cols += [f for f in available_factor_cols if f in key_band.columns]
    detail_cols += [f'{f}_pct' for f in available_factor_cols if f'{f}_pct' in key_band.columns]
    existing = [c for c in detail_cols if c in key_band.columns]
    print(key_band[existing].to_string())

# Also check the (-10%, -6%] band (currently blocked by gate)
gated_band = df_with_gap[df_with_gap['gap_band'] == '-10%~-6%'].copy()
print(f"\n当前被门控拦截区间 (-10%, -6%]: {len(gated_band)} 笔")
if len(gated_band) > 0:
    print(f"  大亏: {gated_band['is_big_loss'].sum()}, 大赚: {gated_band['is_big_win'].sum()}")
    existing = [c for c in detail_cols if c in gated_band.columns]
    print(gated_band[existing].to_string())

# ============================================================
# Step 5: 因子条件与gap交互分析
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 因子条件 × gap × 盈亏的交叉分析")
print("=" * 80)

# For trades in (-6%, -4%] band: look at factor values for big_loss vs non-big_loss
if len(key_band) >= 3:
    print(f"\n关键区间 (-6%, -4%] 因子差异 (大亏 vs 非大亏):")
    for f in available_factor_cols:
        big_loss_vals = key_band.loc[key_band['is_big_loss'], f].dropna()
        non_bl_vals = key_band.loc[~key_band['is_big_loss'], f].dropna()

        if len(big_loss_vals) > 0 and len(non_bl_vals) > 0:
            bl_mean = big_loss_vals.mean()
            non_mean = non_bl_vals.mean()
            diff = bl_mean - non_mean
            print(f"  {f:40s} | 大亏均值={bl_mean:+.4f} | 非大亏均值={non_mean:+.4f} | 差异={diff:+.4f}")

# ============================================================
# Step 6: 因子条件阈值收紧评估（核心测试）
# ============================================================
print("\n" + "=" * 80)
print("Step 6: 因子条件阈值收紧评估")
print("=" * 80)

# Test: For trades with gap in (-6%, -4%], if factor F is in extreme region,
# should we tighten the threshold from -6% to -4%?

# The trades in (-6%, -4%] are the ones that currently pass but would be blocked
# if we tighten to -4%.
# Among those, we only want to block those where factor F is extreme.

# Also consider trades in (-10%, -6%] that are already blocked. Tightening doesn't affect them.

# What about trades with gap < -6% that are currently blocked but have factor
# suggesting they should be allowed? That's a separate question (relaxing).

# For now: focus on tightening from -6% to -4% conditionally.

# Define extreme: top/bottom N% of cross-sectional distribution
# We'll test various thresholds for each factor

print("\n说明: 测试'gap在(-6%,-4%] 且 因子F极端 → 收紧到-4%跳过'")
print("即：当前通过-6%门控但在[-6%,-4%]区间的交易，如果因子极端则拦截\n")

# We only test on real trades since shadow trades don't have all factor columns
df_real_gap = df_with_gap[df_with_gap['source'] == 'real'].copy()
trades_in_band = df_real_gap[df_real_gap['gap_band'] == '-6%~-4%']
print(f"真实交易中(-6%,-4%]区间: {len(trades_in_band)} 笔")
print(f"  其中大亏: {trades_in_band['is_big_loss'].sum()}, 大赚: {trades_in_band['is_big_win'].sum()}")

# For context: how many trades are in (-6%, -4%] band?
# The unconditional tightening would block ALL of them
print(f"\n无条件收紧到-4%: 拦截{trades_in_band['return_pct'].count()}笔, "
      f"拦截大亏{trades_in_band['is_big_loss'].sum()}笔, "
      f"拦截大赚{trades_in_band['is_big_win'].sum()}笔, "
      f"净效果={trades_in_band['return_pct'].sum():.2f}%")

# Now test factor-conditional rules
decisions = []
# Also test in the (-4%, -2%] band for reference
trades_in_minus4_2 = df_real_gap[df_real_gap['gap_band'] == '-4%~-2%']

for f in available_factor_cols:
    f_pct = f'{f}_pct'

    if f_pct not in df_real_gap.columns:
        continue

    # Test extreme factor conditions (two-sided for each factor)
    # High extreme: factor > Q90, factor > Q80, factor > Q70
    # Low extreme: factor < Q10, factor < Q20, factor < Q30
    # We test these on trades in (-6%, -4%] band

    band_trades = trades_in_band.copy()
    if len(band_trades) < 3:
        continue

    for extreme_label, extreme_mask_fn in [
        ('Pct>90', lambda df: df[f_pct] > 90),
        ('Pct>80', lambda df: df[f_pct] > 80),
        ('Pct>70', lambda df: df[f_pct] > 70),
        ('Pct<30', lambda df: df[f_pct] < 30),
        ('Pct<20', lambda df: df[f_pct] < 20),
        ('Pct<10', lambda df: df[f_pct] < 10),
        ('Pct>85', lambda df: df[f_pct] > 85),
        ('Pct<15', lambda df: df[f_pct] < 15),
    ]:
        try:
            extreme = extreme_mask_fn(band_trades)
            blocked = band_trades[extreme]
        except Exception as e:
            continue

        if len(blocked) < 1:
            continue

        bl_blocked = blocked['is_big_loss'].sum()
        bw_blocked = blocked['is_big_win'].sum()
        ok_blocked = len(blocked) - bl_blocked - bw_blocked
        net_effect = blocked['return_pct'].sum()

        # Selectivity: bl_blocked / total_big_loss_in_band vs bw_blocked / total_big_win_in_band
        total_bl = band_trades['is_big_loss'].sum()
        total_bw = band_trades['is_big_win'].sum()
        bl_rate = bl_blocked / total_bl if total_bl > 0 else 0
        bw_rate = bw_blocked / total_bw if total_bw > 0 else 0

        selectivity = bl_rate / bw_rate if bw_rate > 0 else (999 if bl_rate > 0 else 0)

        decisions.append({
            'factor': f,
            'condition': extreme_label,
            'blocked_n': len(blocked),
            'bl_blocked': int(bl_blocked),
            'bw_blocked': int(bw_blocked),
            'ok_blocked': int(ok_blocked),
            'bl_rate': bl_rate,
            'bw_rate': bw_rate,
            'selectivity': selectivity,
            'net_effect_pct': net_effect,
            'avg_return_blocked': blocked['return_pct'].mean() if len(blocked) > 0 else 0,
            'total_bl_in_band': int(total_bl),
            'total_bw_in_band': int(total_bw),
        })

# Also test on (-6%, -4%] AND (-4%, -2%] combined (wider band for more data)
band_neg6_neg2 = df_real_gap[df_real_gap['gap_band'].isin(['-6%~-4%', '-4%~-2%'])].copy()
band_neg6_neg2['in_tight_band'] = band_neg6_neg2['gap_band'] == '-6%~-4%'
print(f"\n(-6%,-2%]区间总交易: {len(band_neg6_neg2)} 笔")

for f in available_factor_cols:
    f_pct = f'{f}_pct'
    if f_pct not in df_real_gap.columns:
        continue

    band_trades = band_neg6_neg2.copy()

    for extreme_label, extreme_mask_fn in [
        ('Pct>90', lambda df: df[f_pct] > 90),
        ('Pct>80', lambda df: df[f_pct] > 80),
        ('Pct>70', lambda df: df[f_pct] > 70),
        ('Pct<30', lambda df: df[f_pct] < 30),
        ('Pct<20', lambda df: df[f_pct] < 20),
        ('Pct<10', lambda df: df[f_pct] < 10),
    ]:
        try:
            extreme = extreme_mask_fn(band_trades)
            blocked = band_trades[extreme & band_trades['in_tight_band']]
        except Exception as e:
            continue

        if len(blocked) < 1:
            continue

        bl_blocked = blocked['is_big_loss'].sum()
        bw_blocked = blocked['is_big_win'].sum()
        net_effect = blocked['return_pct'].sum()

        total_bl = band_trades[band_trades['in_tight_band']]['is_big_loss'].sum()
        total_bw = band_trades[band_trades['in_tight_band']]['is_big_win'].sum()
        bl_rate = bl_blocked / total_bl if total_bl > 0 else 0
        bw_rate = bw_blocked / total_bw if total_bw > 0 else 0
        selectivity = bl_rate / bw_rate if bw_rate > 0 else (999 if bl_rate > 0 else 0)

        decisions.append({
            'factor': f + ' (band:-6~-2%)',
            'condition': extreme_label,
            'blocked_n': len(blocked),
            'bl_blocked': int(bl_blocked),
            'bw_blocked': int(bw_blocked),
            'ok_blocked': len(blocked) - bl_blocked - bw_blocked,
            'bl_rate': bl_rate,
            'bw_rate': bw_rate,
            'selectivity': selectivity,
            'net_effect_pct': net_effect,
            'avg_return_blocked': blocked['return_pct'].mean() if len(blocked) > 0 else 0,
            'total_bl_in_band': int(total_bl),
            'total_bw_in_band': int(total_bw),
        })

df_decisions = pd.DataFrame(decisions)
print(f"\n共测试 {len(df_decisions)} 个因子条件组合")

# ============================================================
# Step 7: 决策表
# ============================================================
print("\n" + "=" * 80)
print("Step 7: 决策表 - 按selectivity排序")
print("=" * 80)

df_decisions['net_effect_str'] = df_decisions['net_effect_pct'].apply(lambda x: f'{x:+.2f}%')

# Sort by selectivity descending, then by net effect
df_sorted = df_decisions.sort_values(['selectivity', 'net_effect_pct'], ascending=[False, False])

# Only show rules with blocked_n >= 1
df_show = df_sorted[df_sorted['blocked_n'] >= 1].copy()

col_order = ['factor', 'condition', 'blocked_n', 'bl_blocked', 'bw_blocked', 'ok_blocked',
             'total_bl_in_band', 'total_bw_in_band', 'selectivity', 'net_effect_str', 'avg_return_blocked']
existing_cols = [c for c in col_order if c in df_show.columns]
print(df_show[existing_cols].head(30).to_string())

# ============================================================
# Step 8: 最佳规则详细分析
# ============================================================
print("\n" + "=" * 80)
print("Step 8: 最佳规则详细分析")
print("=" * 80)

# Find the best rules: selectivity > 1, net_effect positive, blocked at least 2 trades
best = df_sorted[(df_sorted['selectivity'] > 1) & (df_sorted['net_effect_pct'] > 0) & (df_sorted['blocked_n'] >= 2)]

if len(best) > 0:
    print(f"找到 {len(best)} 个有效规则（选择性>1 且 净效果为正 且 拦截>=2笔）:")
    for idx, row in best.head(10).iterrows():
        print(f"\n  规则: {row['factor']} {row['condition']}")
        print(f"  拦截{row['blocked_n']}笔 | 大亏{row['bl_blocked']}笔 | 大赚{row['bw_blocked']}笔 | "
              f"选择性={row['selectivity']:.1f}x | 净效果={row['net_effect_str']}")
        print(f"  拦截率: 大亏 {row['bl_rate']:.1%} / 大赚 {row['bw_rate']:.1%}")
else:
    print("没有找到满足条件的规则（选择性>1, 净效果为正, 拦截>=2笔）")
    print("这说明因子条件收紧阈值在当前数据中无法有效区分大亏和大赚。")

# ============================================================
# Step 9: 时间稳定性检查 (对候选规则)
# ============================================================
print("\n" + "=" * 80)
print("Step 9: 时间稳定性检查")
print("=" * 80)

# Split the (-6%, -4%] band trades into first half and second half
band_trades_sorted = trades_in_band.sort_values('buy_date').copy()
n = len(band_trades_sorted)
if n >= 6:
    mid = n // 2
    first_half = band_trades_sorted.iloc[:mid]
    second_half = band_trades_sorted.iloc[mid:]
    print(f"前半段: {len(first_half)} 笔 (日期: {first_half['buy_date'].min()} ~ {first_half['buy_date'].max()})")
    print(f"后半段: {len(second_half)} 笔 (日期: {second_half['buy_date'].min()} ~ {second_half['buy_date'].max()})")

    # For each promising factor, check stability
    for f in available_factor_cols:
        f_pct = f'{f}_pct'
        if f_pct not in band_trades_sorted.columns:
            continue

        print(f"\n  因子 {f}:")
        print(f"    前半段: 大亏均值={first_half.loc[first_half['is_big_loss'], f].mean():.4f} vs "
              f"非大亏均值={first_half.loc[~first_half['is_big_loss'], f].mean():.4f}")
        print(f"    后半段: 大亏均值={second_half.loc[second_half['is_big_loss'], f].mean():.4f} vs "
              f"非大亏均值={second_half.loc[~second_half['is_big_loss'], f].mean():.4f}")
elif n < 3:
    print("关键区间(-6%,-4%]交易数不足，无法做时间稳定性检查")

# ============================================================
# Step 10: 总结
# ============================================================
print("\n" + "=" * 80)
print("Step 10: 总结与建议")
print("=" * 80)

print(f"""
关键发现：
1. (6%,-4%]区间共 {len(trades_in_band)} 笔交易，大亏 {trades_in_band['is_big_loss'].sum()} 笔，大赚 {trades_in_band['is_big_win'].sum()} 笔
2. 无条件收紧到-4%会拦截所有 {len(trades_in_band)} 笔，净效果={trades_in_band['return_pct'].sum():.2f}%
""")

if len(best) > 0:
    print("3. 因子条件收紧有 {len(best)} 个候选规则")
    print(f"   最佳规则: {best.iloc[0]['factor']} {best.iloc[0]['condition']}")
else:
    print("3. 没有找到有效的因子条件收紧规则")
    print("   结论：当前的-6%阈值在现有数据中是最优的，因子极端值不支持进一步收紧")

print("\n分析完成!")
