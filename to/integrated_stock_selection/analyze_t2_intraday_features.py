"""
T+2盘中分时特征分析：12笔真恶化 vs 11笔已修复
目标：找出T+2早盘能区分"继续恶化"和"即将修复"的特征
"""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
MINUTE_DIR = BASE_DIR.parent / "local_data" / "minute"
DAILY_DIR = BASE_DIR.parent / "local_data" / "daily"
FACTOR_FILE = BASE_DIR / "factor_lab_data" / "factors.parquet"
ANALYSIS_FILE = BASE_DIR / "results" / "big_loss_analysis_t2_t3.csv"

# ============================================================
# Step 0: 加载分析结果，分为两组
# ============================================================
print("=" * 80)
print("Step 0: 加载分析结果，分为真恶化 vs 已修复")
print("=" * 80)

df_analysis = pd.read_csv(ANALYSIS_FILE)

# 真恶化组：T+3收盘仍亏损超5%
truly_bad = df_analysis[df_analysis['T+3_return_pct'] < -5].copy()
# 已修复组：T+3收盘收复到-3%以上
recovered = df_analysis[(df_analysis['T+3_return_pct'] >= -3) & (df_analysis['T+3_return_pct'].notna())].copy()

print(f"真恶化组: {len(truly_bad)} 笔")
for _, r in truly_bad.iterrows():
    print(f"  {r['stock_code']:10s} | buy={r['buy_date']} | actual={r['actual_return_pct']:+.1f}% | T+3={r['T+3_return_pct']:+.1f}%")

print(f"已修复组: {len(recovered)} 笔")
for _, r in recovered.iterrows():
    print(f"  {r['stock_code']:10s} | buy={r['buy_date']} | actual={r['actual_return_pct']:+.1f}% | T+3={r['T+3_return_pct']:+.1f}%")

# ============================================================
# Step 1: 获取每笔交易的T+1/T+2日期，加载日线和分钟数据
# ============================================================
print("\n" + "=" * 80)
print("Step 1: 确定T+1/T+2日期并加载数据")
print("=" * 80)

# 加载因子文件获取交易日历
df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# 加载日线数据（获取T+1收盘价、T+1低点等）
# 确定需要的月份
all_trade_dates = set()
for _, r in pd.concat([truly_bad, recovered]).iterrows():
    bd = pd.Timestamp(str(r['buy_date'])[:10])
    if bd in date_to_idx:
        idx = date_to_idx[bd]
        for offset in range(5):
            if idx + offset < len(all_dates):
                all_trade_dates.add(all_dates[idx + offset])

needed_months_daily = set()
needed_months_minute = set()
for d in all_trade_dates:
    ts = pd.Timestamp(d)
    ym = f"{ts.year}{ts.month:02d}"
    needed_months_daily.add(ym)
    needed_months_minute.add(ym)

# 加载日线数据
print("加载日线数据...")
daily_frames = []
for month_dir in sorted(DAILY_DIR.iterdir()):
    if month_dir.is_dir() and month_dir.name in needed_months_daily:
        for f in month_dir.glob("*.parquet"):
            try:
                df_d = pd.read_parquet(f, columns=['stock_code', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'amount', 'volume'])
                daily_frames.append(df_d)
            except:
                pass

daily_data = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['lookup'] = daily_data['stock_code'] + '_' + daily_data['date'].dt.strftime('%Y-%m-%d')
daily_dict = daily_data.set_index('lookup').to_dict('index')
print(f"日线数据: {daily_data.shape}")

# ============================================================
# Step 2: 提取T+2盘中分时特征
# ============================================================
print("\n" + "=" * 80)
print("Step 2: 提取T+1/T+2盘中分时特征")
print("=" * 80)

def get_trading_date(buy_date_str, offset):
    bd = pd.Timestamp(buy_date_str[:10])
    if bd not in date_to_idx:
        return None
    idx = date_to_idx[bd]
    target_idx = idx + offset
    if 0 <= target_idx < len(all_dates):
        return all_dates[target_idx]
    return None

def load_minute_data(stock_code, date_str):
    """加载某股票某日的分钟数据"""
    ts = pd.Timestamp(date_str)
    ym = f"{ts.year}{ts.month:02d}"
    f = MINUTE_DIR / ym / "minute_data.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f, columns=['stock_code', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        mask = (df['stock_code'] == stock_code) & (df['datetime'].dt.date == ts.date())
        return df[mask].sort_values('datetime')
    except:
        return None

def extract_t2_features(stock_code, buy_date_str, buy_price):
    """提取T+1和T+2的盘中分时特征"""
    t1_date = get_trading_date(buy_date_str, 1)  # T+1
    t2_date = get_trading_date(buy_date_str, 2)  # T+2

    if t1_date is None or t2_date is None:
        return None

    features = {}

    # --- T+1 日线特征 ---
    t1_key = f"{stock_code}_{pd.Timestamp(t1_date).strftime('%Y-%m-%d')}"
    t1_daily = daily_dict.get(t1_key, {})
    if t1_daily:
        features['T1_close'] = t1_daily.get('close')
        features['T1_low'] = t1_daily.get('low')
        features['T1_high'] = t1_daily.get('high')
        features['T1_open'] = t1_daily.get('open')
        features['T1_return_pct'] = (t1_daily.get('close', 0) / buy_price - 1) * 100 if buy_price else None
        features['T1_low_pct'] = (t1_daily.get('low', 0) / buy_price - 1) * 100 if buy_price else None
        features['T1_range_pct'] = ((t1_daily.get('high', 0) - t1_daily.get('low', 0)) / t1_daily.get('low', 1)) * 100 if t1_daily.get('low') else None
        features['T1_volume'] = t1_daily.get('volume')
        features['T1_amount'] = t1_daily.get('amount')

    # --- T+2 日线特征 ---
    t2_key = f"{stock_code}_{pd.Timestamp(t2_date).strftime('%Y-%m-%d')}"
    t2_daily = daily_dict.get(t2_key, {})
    if t2_daily:
        features['T2_open'] = t2_daily.get('open')
        features['T2_close'] = t2_daily.get('close')
        features['T2_low'] = t2_daily.get('low')
        features['T2_high'] = t2_daily.get('high')
        features['T2_return_pct'] = (t2_daily.get('close', 0) / buy_price - 1) * 100 if buy_price else None
        features['T2_low_pct'] = (t2_daily.get('low', 0) / buy_price - 1) * 100 if buy_price else None
        features['T2_open_gap_pct'] = (t2_daily.get('open', 0) / t1_daily.get('close', 1) - 1) * 100 if t1_daily.get('close') and t2_daily.get('open') else None

    # --- T+2 分钟级别特征 ---
    t2_minute = load_minute_data(stock_code, str(t2_date)[:10])
    if t2_minute is not None and len(t2_minute) > 0:
        # 09:35 第1根5分钟K线
        first_bar = t2_minute.iloc[0]
        features['T2_0935_open'] = first_bar['open']
        features['T2_0935_close'] = first_bar['close']
        features['T2_0935_low'] = first_bar['low']
        features['T2_0935_return_pct'] = (first_bar['close'] / buy_price - 1) * 100 if buy_price else None
        features['T2_0935_vs_T1close_pct'] = (first_bar['open'] / t1_daily.get('close', first_bar['open']) - 1) * 100 if t1_daily.get('close') else None
        features['T2_0935_vs_T1low'] = first_bar['low'] < t1_daily.get('low', float('inf')) if t1_daily.get('low') else None
        features['T2_0935_vs_T1low_pct'] = (first_bar['low'] / t1_daily.get('low', 1) - 1) * 100 if t1_daily.get('low') else None

        # 09:35-10:00 (前6根K线, 30分钟)
        first_30min = t2_minute.head(6)
        if len(first_30min) > 0:
            features['T2_1000_low'] = first_30min['low'].min()
            features['T2_1000_high'] = first_30min['high'].max()
            features['T2_1000_close'] = first_30min.iloc[-1]['close']
            features['T2_1000_return_pct'] = (first_30min.iloc[-1]['close'] / buy_price - 1) * 100 if buy_price else None
            features['T2_1000_vs_T1low_pct'] = (first_30min['low'].min() / t1_daily.get('low', 1) - 1) * 100 if t1_daily.get('low') else None
            features['T2_1000_vs_T1close_pct'] = (first_30min.iloc[-1]['close'] / t1_daily.get('close', 1) - 1) * 100 if t1_daily.get('close') else None

            # 是否收复T+1低点
            features['T2_1000_recovered_T1low'] = first_30min.iloc[-1]['close'] > t1_daily.get('low', float('inf')) if t1_daily.get('low') else None

            # VWAP (前30分钟成交量加权均价)
            total_amount = first_30min['amount'].sum()
            total_volume = first_30min['volume'].sum()
            features['T2_1000_vwap'] = total_amount / total_volume / 100 if total_volume > 0 else None
            features['T2_1000_close_vs_vwap_pct'] = ((first_30min.iloc[-1]['close'] / (features['T2_1000_vwap'] or 1)) - 1) * 100 if features['T2_1000_vwap'] else None

            # 早盘最低点后反弹幅度
            min_low_idx = first_30min['low'].idxmin()
            min_low_bar = first_30min.loc[min_low_idx]
            bars_after_low = first_30min[first_30min['datetime'] > min_low_bar['datetime']]
            if len(bars_after_low) > 0:
                features['T2_1000_bounce_from_low_pct'] = (bars_after_low.iloc[-1]['close'] / min_low_bar['low'] - 1) * 100
            else:
                features['T2_1000_bounce_from_low_pct'] = 0

            # 早盘量价关系：放量下跌 vs 放量上涨
            down_bars = first_30min[first_30min['close'] < first_30min['open']]
            up_bars = first_30min[first_30min['close'] > first_30min['open']]
            down_volume = down_bars['volume'].sum()
            up_volume = up_bars['volume'].sum()
            features['T2_1000_down_up_volume_ratio'] = down_volume / up_volume if up_volume > 0 else (999 if down_volume > 0 else 1)

        # 全天特征
        features['T2_day_low'] = t2_minute['low'].min()
        features['T2_day_high'] = t2_minute['high'].max()
        features['T2_day_close'] = t2_minute.iloc[-1]['close']

        # 09:35是否跌破T+1低点
        features['T2_0935_broke_T1low'] = first_bar['low'] < t1_daily.get('low', float('inf')) if t1_daily.get('low') else None

        # 10:00是否跌破T+1低点
        if len(first_30min) > 0:
            features['T2_1000_broke_T1low'] = first_30min['low'].min() < t1_daily.get('low', float('inf')) if t1_daily.get('low') else None

        # 09:35-10:00是否收复分时均线(简单均线)
        if len(first_30min) >= 3:
            avg_price = first_30min['close'].mean()
            features['T2_1000_above_avg'] = first_30min.iloc[-1]['close'] > avg_price

    return features

# 提取所有交易的特征
all_trades = pd.concat([truly_bad, recovered], ignore_index=True)
all_features = []

for _, trade in all_trades.iterrows():
    stock = trade['stock_code']
    buy_date = str(trade['buy_date'])[:10]
    buy_price = trade['buy_price']

    features = extract_t2_features(stock, buy_date, buy_price)
    if features:
        features['stock_code'] = stock
        features['buy_date'] = buy_date
        features['buy_price'] = buy_price
        features['actual_return_pct'] = trade['actual_return_pct']
        features['T3_return_pct'] = trade['T+3_return_pct']
        features['group'] = 'truly_bad' if trade['T+3_return_pct'] < -5 else 'recovered'
        all_features.append(features)
    else:
        print(f"  WARNING: 无法提取特征: {stock} {buy_date}")

df_features = pd.DataFrame(all_features)
print(f"成功提取特征: {len(df_features)} 笔 (truly_bad={len(df_features[df_features['group']=='truly_bad'])}, recovered={len(df_features[df_features['group']=='recovered'])})")

# ============================================================
# Step 3: 对比两组的T+2盘中特征差异
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 真恶化 vs 已修复 的T+2盘中特征对比")
print("=" * 80)

bad = df_features[df_features['group'] == 'truly_bad']
rec = df_features[df_features['group'] == 'recovered']

# 数值型特征对比
numeric_cols = [c for c in df_features.columns if c not in ('stock_code', 'buy_date', 'group') and df_features[c].dtype in ('float64', 'int64', 'bool')]

comparisons = []
for col in numeric_cols:
    bad_vals = bad[col].dropna()
    rec_vals = rec[col].dropna()

    if len(bad_vals) < 3 or len(rec_vals) < 3:
        continue

    bad_mean = bad_vals.mean()
    rec_mean = rec_vals.mean()
    pooled_std = np.sqrt((bad_vals.std()**2 + rec_vals.std()**2) / 2)

    if pooled_std > 1e-10:
        cohens_d = (bad_mean - rec_mean) / pooled_std
    else:
        cohens_d = 0

    # t-test
    try:
        t_stat, p_value = stats.ttest_ind(bad_vals, rec_vals, equal_var=False)
    except:
        t_stat, p_value = 0, 1

    comparisons.append({
        'feature': col,
        'truly_bad_mean': bad_mean,
        'recovered_mean': rec_mean,
        'diff': bad_mean - rec_mean,
        'cohens_d': cohens_d,
        'abs_cohens_d': abs(cohens_d),
        't_stat': t_stat,
        'p_value': p_value,
        'bad_n': len(bad_vals),
        'rec_n': len(rec_vals),
    })

df_comp = pd.DataFrame(comparisons).sort_values('abs_cohens_d', ascending=False)

print(f"\nTop 25 区分真恶化 vs 已修复的T+2盘中特征 (按Cohen's d排序):")
print(f"{'Feature':40s} | {'Bad Mean':>10s} | {'Rec Mean':>10s} | {'d':>6s} | {'p-val':>8s} | Direction")
print("-" * 100)
for _, row in df_comp.head(25).iterrows():
    direction = "BAD>>" if row['diff'] > 0 else "REC>>"
    sig = "***" if row['p_value'] < 0.05 else "**" if row['p_value'] < 0.1 else ""
    print(f"  {row['feature']:38s} | {row['truly_bad_mean']:10.3f} | {row['recovered_mean']:10.3f} | {row['cohens_d']:+5.2f} | {row['p_value']:8.4f} | {direction} {sig}")

# ============================================================
# Step 4: 逐笔展示关键特征
# ============================================================
print("\n" + "=" * 80)
print("Step 4: 逐笔关键特征详情")
print("=" * 80)

key_features = [
    'T2_0935_vs_T1close_pct', 'T2_0935_vs_T1low_pct', 'T2_0935_broke_T1low',
    'T2_1000_vs_T1close_pct', 'T2_1000_vs_T1low_pct', 'T2_1000_recovered_T1low',
    'T2_1000_bounce_from_low_pct', 'T2_1000_down_up_volume_ratio',
    'T2_open_gap_pct', 'T1_low_pct', 'T1_return_pct',
]

print(f"\n{'Stock':10s} {'Group':10s} | {'T1_ret':>7s} {'T1_low':>7s} | {'T2_gap':>7s} {'935vsT1c':>8s} {'935vsT1l':>8s} {'1000vsT1c':>9s} {'1000recT1l':>10s} {'bounce':>7s} {'vol_ratio':>9s} | T3")
print("-" * 130)
for _, r in df_features.sort_values(['group', 'T3_return_pct']).iterrows():
    t1_ret = f"{r.get('T1_return_pct', np.nan):+.1f}%" if pd.notna(r.get('T1_return_pct')) else "N/A"
    t1_low = f"{r.get('T1_low_pct', np.nan):+.1f}%" if pd.notna(r.get('T1_low_pct')) else "N/A"
    t2_gap = f"{r.get('T2_open_gap_pct', np.nan):+.1f}%" if pd.notna(r.get('T2_open_gap_pct')) else "N/A"
    v1 = f"{r.get('T2_0935_vs_T1close_pct', np.nan):+.1f}%" if pd.notna(r.get('T2_0935_vs_T1close_pct')) else "N/A"
    v2 = f"{r.get('T2_0935_vs_T1low_pct', np.nan):+.1f}%" if pd.notna(r.get('T2_0935_vs_T1low_pct')) else "N/A"
    v3 = f"{r.get('T2_1000_vs_T1close_pct', np.nan):+.1f}%" if pd.notna(r.get('T2_1000_vs_T1close_pct')) else "N/A"
    v4 = f"{'Y' if r.get('T2_1000_recovered_T1low') else 'N'}" if pd.notna(r.get('T2_1000_recovered_T1low')) else "N/A"
    v5 = f"{r.get('T2_1000_bounce_from_low_pct', np.nan):+.1f}%" if pd.notna(r.get('T2_1000_bounce_from_low_pct')) else "N/A"
    v6 = f"{r.get('T2_1000_down_up_volume_ratio', np.nan):.2f}" if pd.notna(r.get('T2_1000_down_up_volume_ratio')) else "N/A"
    t3 = f"{r['T3_return_pct']:+.1f}%"

    print(f"  {r['stock_code']:10s} {r['group']:10s} | {t1_ret:>7s} {t1_low:>7s} | {t2_gap:>7s} {v1:>8s} {v2:>8s} {v3:>9s} {v4:>10s} {v5:>7s} {v6:>9s} | {t3}")

# ============================================================
# Step 5: 总结规则建议
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 风控优化规则建议")
print("=" * 80)

# 分析关键特征的最佳分割点
print("\n--- 关键特征分布 ---")
for feat in ['T2_0935_vs_T1close_pct', 'T2_1000_vs_T1close_pct', 'T2_1000_vs_T1low_pct',
             'T2_1000_bounce_from_low_pct', 'T2_1000_recovered_T1low', 'T2_open_gap_pct']:
    bad_vals = bad[feat].dropna()
    rec_vals = rec[feat].dropna()
    if len(bad_vals) == 0 or len(rec_vals) == 0:
        continue
    print(f"\n  {feat}:")
    print(f"    Truly_bad:  mean={bad_vals.mean():.3f}, median={bad_vals.median():.3f}, range=[{bad_vals.min():.3f}, {bad_vals.max():.3f}]")
    print(f"    Recovered:  mean={rec_vals.mean():.3f}, median={rec_vals.median():.3f}, range=[{rec_vals.min():.3f}, {rec_vals.max():.3f}]")

# ============================================================
# Step 6: 保存
# ============================================================
print("\n" + "=" * 80)
print("Step 6: 保存结果")
print("=" * 80)

output_dir = BASE_DIR / "results"
df_features.to_csv(output_dir / "t2_intraday_features_comparison.csv", index=False, encoding='utf-8-sig')
print("已保存: t2_intraday_features_comparison.csv")

df_comp.to_csv(output_dir / "t2_feature_importance_bad_vs_recovered.csv", index=False, encoding='utf-8-sig')
print("已保存: t2_feature_importance_bad_vs_recovered.csv")

print("\n分析完成!")
