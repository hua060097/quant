"""
分析回测中的大亏交易：
1. 从日志提取所有买入/卖出交易
2. 用T+2/T+3收盘价验证：大亏是风控止损导致的，还是真的亏了
3. 非风控原因的大亏，关联因子文件中的所有因子
"""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
LOG_FILE = BASE_DIR / "stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log"
FACTOR_FILE = BASE_DIR / "factor_lab_data" / "factors.parquet"
SHADOW_FILE = BASE_DIR / "results" / "walkforward_shadow_skipped_opportunities_20260513_230111.csv"

# ============================================================
# Step 1: 从日志提取所有交易（买入+卖出配对）
# ============================================================
print("=" * 80)
print("Step 1: 从日志提取交易记录")
print("=" * 80)

lines = open(LOG_FILE, 'r', encoding='utf-8').readlines()

# 提取买入记录
buys = []
current_stock = None
current_buy_date = None
current_buy_price = None

for i, line in enumerate(lines):
    # 匹配买入股票代码
    m_stock = re.search(r'分层并行仓位生效: (\S+) 使用', line)
    if m_stock:
        current_stock = m_stock.group(1)

    # 匹配 T日
    m_tday = re.search(r'并行持仓买入:.*?T日=(\d{4}-\d{2}-\d{2})', line)
    if m_tday:
        current_buy_date = m_tday.group(1)

    # 匹配买入价格
    m_buyp = re.search(r'实际买入价[：:]?\s*([\d.]+)', line)
    if m_buyp:
        current_buy_price = float(m_buyp.group(1))

    # 匹配买入确认
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

# 提取卖出记录
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

# 配对买入和卖出
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

# 补充影子交易
df_shadow = pd.read_csv(SHADOW_FILE)
print(f"影子跳过交易: {len(df_shadow)} 条")

# ============================================================
# Step 2: 合并所有交易，标记大亏
# ============================================================
print("\n" + "=" * 80)
print("Step 2: 合并交易，标记大亏")
print("=" * 80)

df_trades['source'] = 'real'
df_shadow_renamed = df_shadow.rename(columns={
    'buy_datetime': 'buy_date_raw',
    'profit_loss_pct': 'return_pct',
})[['stock_code', 'buy_date_raw', 'buy_price', 'sell_price', 'return_pct', 'exit_reason', 'shadow_skip_reason']].copy()
df_shadow_renamed['buy_date'] = df_shadow_renamed['buy_date_raw'].str[:10]
df_shadow_renamed['source'] = 'shadow'
df_shadow_renamed = df_shadow_renamed.drop(columns=['buy_date_raw'], errors='ignore')

all_trades = pd.concat([
    df_trades[['stock_code', 'buy_date', 'buy_price', 'sell_price', 'return_pct', 'exit_reason', 'source']],
    df_shadow_renamed[['stock_code', 'buy_date', 'buy_price', 'sell_price', 'return_pct', 'exit_reason', 'source']],
], ignore_index=True)

print(f"总交易数: {len(all_trades)} (real={len(df_trades)}, shadow={len(df_shadow)})")

big_loss = all_trades[all_trades['return_pct'] < -5].copy()
print(f"大亏交易数 (return < -5%): {len(big_loss)}")

def classify_exit(reason):
    if pd.isna(reason):
        return 'unknown'
    risk_keywords = ['止损', '失败止损', '走坏止损', '累计收益低于', '跌破', '连续阴线']
    for kw in risk_keywords:
        if kw in reason:
            return 'risk_control'
    if '到期' in reason or '未转正' in reason:
        return 'expiry'
    if '利润保护' in reason:
        return 'profit_protection'
    if '正常' in reason:
        return 'normal'
    return 'other'

big_loss['exit_type'] = big_loss['exit_reason'].apply(classify_exit)
print("大亏交易按退出类型分布:")
print(big_loss['exit_type'].value_counts().to_string())

# ============================================================
# Step 3: 加载因子文件获取交易日历 + 加载日线收盘价
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 加载数据")
print("=" * 80)

print("加载因子文件...")
df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}
print(f"因子文件: {df_factors.shape}, 交易日: {len(all_dates)}, 范围: {all_dates[0]} ~ {all_dates[-1]}")

# 收集需要加载的月份分区
needed_months = set()
for _, trade in big_loss.iterrows():
    bd = pd.Timestamp(str(trade['buy_date'])[:10])
    if bd in date_to_idx:
        idx = date_to_idx[bd]
        for offset in range(5):  # T ~ T+4
            if idx + offset < len(all_dates):
                ts = pd.Timestamp(all_dates[idx + offset])
                needed_months.add(f"{ts.year}{ts.month:02d}")

print(f"需要加载 {len(needed_months)} 个月份分区的日线数据")

# 加载日线数据
daily_dir = BASE_DIR.parent / "local_data" / "daily"
if not daily_dir.exists():
    daily_dir = BASE_DIR / "local_data" / "daily"

daily_frames = []
if daily_dir.exists():
    for month_dir in sorted(daily_dir.iterdir()):
        if month_dir.is_dir() and month_dir.name in needed_months:
            for f in month_dir.glob("*.parquet"):
                try:
                    df_d = pd.read_parquet(f, columns=['stock_code', 'date', 'close'])
                    daily_frames.append(df_d)
                except Exception as e:
                    print(f"  加载失败 {f}: {e}")

if daily_frames:
    daily_data = pd.concat(daily_frames, ignore_index=True)
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    # 构建快速查找索引
    daily_data['lookup_key'] = daily_data['stock_code'] + '_' + daily_data['date'].dt.strftime('%Y-%m-%d')
    daily_lookup = daily_data.set_index('lookup_key')['close'].to_dict()
    print(f"日线数据: {daily_data.shape}, 日期范围: {daily_data['date'].min()} ~ {daily_data['date'].max()}")
else:
    daily_lookup = {}
    print("WARNING: 未加载到日线收盘价数据！")

# ============================================================
# Step 4: 计算T+2/T+3真实收益
# ============================================================
print("\n" + "=" * 80)
print("Step 4: 计算T+2/T+3真实收盘价收益")
print("=" * 80)

results = []
for _, trade in big_loss.iterrows():
    buy_date_str = str(trade['buy_date'])[:10]
    buy_price = trade['buy_price']
    stock = trade['stock_code']

    bd_ts = pd.Timestamp(buy_date_str)
    if bd_ts not in date_to_idx:
        # 尝试从交易日历找到最近日期
        print(f"  WARNING: {buy_date_str} 不在交易日历中, stock={stock}")
        results.append({
            'stock_code': stock, 'buy_date': buy_date_str, 'buy_price': buy_price,
            'actual_return_pct': trade['return_pct'], 'exit_type': trade['exit_type'],
            'exit_reason': trade['exit_reason'], 'source': trade['source'],
            'T+2_date': None, 'T+2_close': None, 'T+2_return_pct': None,
            'T+3_date': None, 'T+3_close': None, 'T+3_return_pct': None,
        })
        continue

    idx = date_to_idx[bd_ts]

    # T+1, T+2, T+3
    t1_date = all_dates[idx + 1] if idx + 1 < len(all_dates) else None
    t2_date = all_dates[idx + 2] if idx + 2 < len(all_dates) else None
    t3_date = all_dates[idx + 3] if idx + 3 < len(all_dates) else None

    def get_close(stock_code, date_ts):
        if date_ts is None:
            return None
        key = f"{stock_code}_{pd.Timestamp(date_ts).strftime('%Y-%m-%d')}"
        return daily_lookup.get(key)

    t2_close = get_close(stock, t2_date)
    t3_close = get_close(stock, t3_date)
    t1_close = get_close(stock, t1_date)

    t2_return = (t2_close / buy_price - 1) * 100 if t2_close and buy_price else None
    t3_return = (t3_close / buy_price - 1) * 100 if t3_close and buy_price else None
    t1_return = (t1_close / buy_price - 1) * 100 if t1_close and buy_price else None

    results.append({
        'stock_code': stock, 'buy_date': buy_date_str, 'buy_price': buy_price,
        'actual_return_pct': trade['return_pct'], 'exit_type': trade['exit_type'],
        'exit_reason': trade['exit_reason'], 'source': trade['source'],
        'T+1_date': t1_date, 'T+1_close': t1_close, 'T+1_return_pct': t1_return,
        'T+2_date': t2_date, 'T+2_close': t2_close, 'T+2_return_pct': t2_return,
        'T+3_date': t3_date, 'T+3_close': t3_close, 'T+3_return_pct': t3_return,
    })

df_analysis = pd.DataFrame(results)

pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 80)

print("\n大亏交易详情（含T+2/T+3真实收益）:")
display_cols = ['stock_code', 'buy_date', 'buy_price', 'actual_return_pct', 'exit_type',
                'T+1_return_pct', 'T+2_return_pct', 'T+3_return_pct', 'exit_reason']
print(df_analysis[display_cols].to_string())

# ============================================================
# Step 5: 分类分析
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 分类分析 - 风控止损 vs 真实亏损")
print("=" * 80)

# T+3仍亏损超过3%的交易（真实亏损，风控也没救回来）
real_loss_t3 = df_analysis[
    (df_analysis['T+3_return_pct'].notna()) &
    (df_analysis['T+3_return_pct'] < -3)
].copy()

# T+3收复到-3%以上（风控过早止损）
early_stop_t3 = df_analysis[
    (df_analysis['T+3_return_pct'].notna()) &
    (df_analysis['T+3_return_pct'] >= -3)
].copy()

# T+3仍大亏（>5%）
big_loss_t3 = df_analysis[
    (df_analysis['T+3_return_pct'].notna()) &
    (df_analysis['T+3_return_pct'] < -5)
].copy()

# 非风控的真大亏（到期/未转正退出 + T+3仍大亏）
non_risk_big_loss = df_analysis[
    (df_analysis['T+3_return_pct'].notna()) &
    (df_analysis['T+3_return_pct'] < -5) &
    (~df_analysis['exit_type'].isin(['risk_control']))
].copy()

print(f"T+3仍亏损超3%: {len(real_loss_t3)} 笔")
print(f"T+3收复到-3%以上（过早止损）: {len(early_stop_t3)} 笔")
print(f"T+3仍亏损超5%（真大亏）: {len(big_loss_t3)} 笔")
print(f"非风控原因+T+3仍大亏（漏网之鱼）: {len(non_risk_big_loss)} 笔")

if len(non_risk_big_loss) > 0:
    print("\n非风控真大亏交易:")
    for _, row in non_risk_big_loss.iterrows():
        print(f"  {row['stock_code']} | 买入={row['buy_date']} | 实际={row['actual_return_pct']:.1f}% | "
              f"T+2={row['T+2_return_pct']:.1f}% | T+3={row['T+3_return_pct']:.1f}% | "
              f"原因={row['exit_reason']}")

# ============================================================
# Step 6: 因子关联分析（所有T+3仍大亏 + 非风控真大亏）
# ============================================================
print("\n" + "=" * 80)
print("Step 6: 因子关联分析")
print("=" * 80)

# 分析目标：优先非风控真大亏，其次所有T+3仍大亏，再其次所有大亏
if len(non_risk_big_loss) > 0:
    targets = non_risk_big_loss
    target_label = "非风控真大亏"
elif len(big_loss_t3) > 0:
    targets = big_loss_t3
    target_label = "T+3仍大亏"
else:
    targets = df_analysis
    target_label = "所有大亏"

print(f"分析目标: {len(targets)} 笔 ({target_label})")

# 提取因子值
factor_cols = [c for c in df_factors.columns if c not in ('date', 'stock_code')]
print(f"因子列数: {len(factor_cols)}")

target_dates = set()
for _, row in targets.iterrows():
    bd = pd.Timestamp(str(row['buy_date'])[:10])
    target_dates.add(bd)

df_factors_sub = df_factors[df_factors['date'].isin(target_dates)].copy()
print(f"因子数据（目标日期）: {df_factors_sub.shape}")

trade_factors = []
for _, row in targets.iterrows():
    stock = row['stock_code']
    buy_date = pd.Timestamp(str(row['buy_date'])[:10])

    mask = (df_factors_sub['stock_code'] == stock) & (df_factors_sub['date'] == buy_date)
    factor_row = df_factors_sub.loc[mask, factor_cols]

    if len(factor_row) > 0:
        factor_vals = factor_row.iloc[0].to_dict()
        factor_vals['stock_code'] = stock
        factor_vals['buy_date'] = str(row['buy_date'])[:10]
        factor_vals['actual_return_pct'] = row['actual_return_pct']
        factor_vals['T+3_return_pct'] = row.get('T+3_return_pct', None)
        factor_vals['exit_type'] = row['exit_type']
        trade_factors.append(factor_vals)
    else:
        print(f"  WARNING: 未找到因子数据: {stock} {buy_date}")

df_trade_factors = pd.DataFrame(trade_factors)
print(f"成功匹配因子: {len(df_trade_factors)} 笔")

if len(df_trade_factors) == 0:
    print("没有匹配到因子数据，退出分析。")
    exit(0)

# 因子差异分析
df_all_same_dates = df_factors[df_factors['date'].isin(target_dates)].copy()

factor_importance = []
for col in factor_cols:
    big_loss_vals = df_trade_factors[col].dropna()
    all_vals = df_all_same_dates[col].dropna()

    if len(big_loss_vals) == 0 or len(all_vals) == 0:
        continue

    big_loss_mean = big_loss_vals.mean()
    all_mean = all_vals.mean()
    all_std = all_vals.std()

    if all_std > 1e-10:
        cohens_d = (big_loss_mean - all_mean) / all_std
    else:
        cohens_d = 0

    try:
        percentiles = [stats.percentileofscore(all_vals, v) for v in big_loss_vals if not np.isnan(v)]
        avg_percentile = np.mean(percentiles) if percentiles else 50
    except:
        avg_percentile = 50

    factor_importance.append({
        'factor': col,
        'big_loss_mean': big_loss_mean,
        'all_mean': all_mean,
        'all_std': all_std,
        'cohens_d': cohens_d,
        'abs_cohens_d': abs(cohens_d),
        'avg_percentile': avg_percentile,
        'deviation_from_50': abs(avg_percentile - 50),
    })

df_importance = pd.DataFrame(factor_importance)
df_importance = df_importance.sort_values('abs_cohens_d', ascending=False)

print(f"\nTop 30 与大亏最相关的因子（按Cohen's d排序）:")
top30 = df_importance.head(30)
for _, row in top30.iterrows():
    direction = "+" if row['cohens_d'] > 0 else "-"
    print(f"  {row['factor']:45s} | d={row['cohens_d']:+.3f} | "
          f"big_loss={row['big_loss_mean']:.4f} vs all={row['all_mean']:.4f} | "
          f"pct={row['avg_percentile']:.1f}% {direction}")

print(f"\nTop 30 与大亏最相关的因子（按百分位偏离排序）:")
df_importance_pct = df_importance.sort_values('deviation_from_50', ascending=False)
top30_pct = df_importance_pct.head(30)
for _, row in top30_pct.iterrows():
    direction = "+" if row['avg_percentile'] > 50 else "-"
    print(f"  {row['factor']:45s} | pct={row['avg_percentile']:.1f}% {direction} | "
          f"d={row['cohens_d']:+.3f} | "
          f"big_loss={row['big_loss_mean']:.4f} vs all={row['all_mean']:.4f}")

# ============================================================
# Step 7: 保存结果
# ============================================================
print("\n" + "=" * 80)
print("Step 7: 保存分析结果")
print("=" * 80)

output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

df_analysis.to_csv(output_dir / "big_loss_analysis_t2_t3.csv", index=False, encoding='utf-8-sig')
print("已保存: big_loss_analysis_t2_t3.csv")

df_importance.to_csv(output_dir / "big_loss_factor_importance.csv", index=False, encoding='utf-8-sig')
print("已保存: big_loss_factor_importance.csv")

df_trade_factors.to_csv(output_dir / "big_loss_trade_factors.csv", index=False, encoding='utf-8-sig')
print("已保存: big_loss_trade_factors.csv")

print("\n分析完成!")
