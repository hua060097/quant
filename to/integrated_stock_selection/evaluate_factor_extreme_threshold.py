"""
用极端分位数作为买入过滤规则的回测评估：
- 用Top因子的大亏极端值作为阈值
- 检查如果应用这些阈值，会拦截多少大亏，同时误杀多少大涨
- 在因子文件的全量数据上评估拦截率和误杀率
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection")
LOG = BASE / "stock_selection_test_mode_walkforward_分类模型_原因子169_optuna.log"
FACTOR_FILE = BASE / "factor_lab_data" / "factors.parquet"
SHADOW_FILE = BASE / "results" / "walkforward_shadow_skipped_opportunities_20260513_230111.csv"

# ============================================================
# Step 1: 加载数据
# ============================================================
print("加载数据...")

# 提取正式交易
lines = open(LOG, 'r', encoding='utf-8').readlines()
buys = []
cs = cd = cp = None
for line in lines:
    m = re.search(r'分层并行仓位生效: (\S+) 使用', line)
    if m: cs = m.group(1)
    m = re.search(r'并行持仓买入:.*?T日=(\d{4}-\d{2}-\d{2})', line)
    if m: cd = m.group(1)
    m = re.search(r'实际买入价[：:]?\s*([\d.]+)', line)
    if m: cp = float(m.group(1))
    m = re.search(r'买入后持仓状态', line)
    if m and cs:
        buys.append({'stock_code': cs, 'buy_date': cd, 'buy_price': cp})
        cs = cd = cp = None

sells = []
for line in lines:
    m = re.search(r'卖出: (\S+), 价格: ([\d.]+), 收益: -?[\d.]+ \((-?[\d.]+)%\), 原因: (.+)', line)
    if m:
        sells.append({'stock_code': m.group(1), 'sell_price': float(m.group(2)),
                       'return_pct': float(m.group(3)), 'exit_reason': m.group(4).strip()})

used = [False] * len(buys)
trades = []
for sell in sells:
    for j, buy in enumerate(buys):
        if not used[j] and buy['stock_code'] == sell['stock_code']:
            trades.append({**buy, **sell})
            used[j] = True
            break

df_trades = pd.DataFrame(trades)

# 加载影子交易
df_shadow = pd.read_csv(SHADOW_FILE)

# 加载因子文件
df_factors = pd.read_parquet(FACTOR_FILE)
all_dates = sorted(df_factors['date'].unique())
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# 加载日线获取T+2/T+3收盘价
DAILY_DIR = BASE.parent / "local_data" / "daily"
needed_months = set()
for _, t in df_trades.iterrows():
    bd = pd.Timestamp(str(t['buy_date'])[:10])
    if bd in date_to_idx:
        idx = date_to_idx[bd]
        for offset in range(5):
            if idx + offset < len(all_dates):
                ts = pd.Timestamp(all_dates[idx + offset])
                needed_months.add(f"{ts.year}{ts.month:02d}")

daily_frames = []
for month_dir in sorted(DAILY_DIR.iterdir()):
    if month_dir.is_dir() and month_dir.name in needed_months:
        for f in month_dir.glob("*.parquet"):
            try:
                df_d = pd.read_parquet(f, columns=['stock_code', 'date', 'close'])
                daily_frames.append(df_d)
            except:
                pass
daily_data = pd.concat(daily_frames, ignore_index=True)
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data['lookup'] = daily_data['stock_code'] + '_' + daily_data['date'].dt.strftime('%Y-%m-%d')
daily_dict = daily_data.set_index('lookup')['close'].to_dict()

# 给每笔交易加T+3收盘收益
for i, row in df_trades.iterrows():
    bd = pd.Timestamp(str(row['buy_date'])[:10])
    if bd not in date_to_idx: continue
    idx = date_to_idx[bd]
    stock = row['stock_code']
    for offset, label in [(1, 'T1'), (2, 'T2'), (3, 'T3')]:
        if idx + offset < len(all_dates):
            ds = pd.Timestamp(all_dates[idx + offset]).strftime('%Y-%m-%d')
            close = daily_dict.get(f"{stock}_{ds}")
            if close and row['buy_price']:
                df_trades.at[i, f'{label}_return_pct'] = (close / row['buy_price'] - 1) * 100

# 合并因子值：用buy_date-1（买入日前一天收盘的因子，买入时可见）
factor_cols = [c for c in df_factors.columns if c not in ('date', 'stock_code')]

for i, row in df_trades.iterrows():
    bd = pd.Timestamp(str(row['buy_date'])[:10])
    stock = row['stock_code']
    # 买入日的因子（T日开盘时可见的是T-1收盘的因子，但因子文件中日期=交易日的因子
    # 在T日9:35可用，因为因子是基于前日收盘价计算的）
    # 用buy_date对应的因子行
    mask = (df_factors['stock_code'] == stock) & (df_factors['date'] == bd)
    factor_row = df_factors.loc[mask, factor_cols]
    if len(factor_row) > 0:
        for col in factor_cols:
            df_trades.at[i, col] = factor_row.iloc[0][col]

print(f"总交易: {len(df_trades)}, 含因子值: {df_trades[factor_cols[0]].notna().sum()}")
print(f"大亏(<-5%): {(df_trades.return_pct < -5).sum()}")
print(f"大涨(>+10%): {(df_trades.return_pct > 10).sum()}")

# ============================================================
# Step 2: 用大亏交易的极端分位数定义阈值
# ============================================================
print("\n" + "=" * 80)
print("Step 2: 关键因子的极端分位数阈值分析")
print("=" * 80)

# 从之前的分析中，最相关的6个因子
key_factors = [
    'sector_momentum_interaction',
    'return_20d_intra_z',
    'price_ma60_ratio',
    'ma60_slope',
    'trend_acceleration',
    'ma_convergence_score',
]

big_loss = df_trades[df_trades['return_pct'] < -5]
all_trades_with_factors = df_trades[df_trades[factor_cols[0]].notna()]

# 对每个因子，计算大亏交易的百分位，并测试不同阈值
print(f"\n{'Factor':40s} | {'BigLoss_Med%':>11s} | {'BigLoss_P90':>11s} | {'Threshold':>10s} | {'Block_BigLoss':>12s} | {'Block_BigWin':>12s} | {'Block_All':>9s} | {'Block_OK':>8s} | {'Net_Effect':>10s}")
print("-" * 135)

for factor in key_factors:
    big_loss_vals = big_loss[factor].dropna()
    all_vals = all_trades_with_factors[factor].dropna()

    if len(big_loss_vals) == 0:
        continue

    # 大亏交易中的中位数和P90（更极端）
    bl_median_pct = all_vals.quantile(big_loss_vals.median() / all_vals.max()) if all_vals.max() > 0 else 50
    # 直接用大亏交易的百分位
    bl_median = big_loss_vals.median()
    bl_p90 = big_loss_vals.quantile(0.9) if len(big_loss_vals) >= 5 else big_loss_vals.max()

    # trend_acceleration 和 ma_convergence_score 是反向的（大亏值更低）
    if factor in ('trend_acceleration', 'ma_convergence_score'):
        # 低值 = 危险，阈值用P10
        bl_p10 = big_loss_vals.quantile(0.1) if len(big_loss_vals) >= 5 else big_loss_vals.min()

        # 测试多个阈值
        for pct_label, threshold in [
            ("P10", bl_p10),
            ("P25", big_loss_vals.quantile(0.25)),
            ("Median", bl_median),
        ]:
            # 拦截条件: factor <= threshold
            blocked = all_trades_with_factors[all_trades_with_factors[factor] <= threshold]
            not_blocked = all_trades_with_factors[all_trades_with_factors[factor] > threshold]

            blocked_big_loss = blocked[blocked['return_pct'] < -5]
            blocked_big_win = blocked[blocked['return_pct'] > 10]
            blocked_ok = blocked[(blocked['return_pct'] >= -5) & (blocked['return_pct'] <= 10)]

            n_big_loss_total = len(all_trades_with_factors[all_trades_with_factors['return_pct'] < -5])
            n_big_win_total = len(all_trades_with_factors[all_trades_with_factors['return_pct'] > 10])

            # 净效果 = 拦截大亏的收益改善 - 误杀大涨的收益损失
            saved_loss = blocked_big_loss['return_pct'].sum() if len(blocked_big_loss) > 0 else 0  # 负值，拦截后少亏
            lost_profit = blocked_big_win['return_pct'].sum() if len(blocked_big_win) > 0 else 0  # 正值，误杀后少赚
            net = -saved_loss - lost_profit  # 正=改善, 负=恶化

            print(f"  {factor:38s} | {bl_median:11.3f} | {bl_p10:11.3f} | <={threshold:8.3f} | "
                  f"{len(blocked_big_loss):3d}/{n_big_loss_total:3d} | "
                  f"{len(blocked_big_win):3d}/{n_big_win_total:3d} | "
                  f"{len(blocked):3d}/{len(all_trades_with_factors):3d} | "
                  f"{len(blocked_ok):3d} | "
                  f"{'+' if net >= 0 else ''}{net:.1f}%")
    else:
        # 高值 = 危险，阈值用P90
        for pct_label, threshold in [
            ("P90", bl_p90),
            ("P75", big_loss_vals.quantile(0.75)),
            ("Median", bl_median),
        ]:
            # 拦截条件: factor >= threshold
            blocked = all_trades_with_factors[all_trades_with_factors[factor] >= threshold]
            not_blocked = all_trades_with_factors[all_trades_with_factors[factor] < threshold]

            blocked_big_loss = blocked[blocked['return_pct'] < -5]
            blocked_big_win = blocked[blocked['return_pct'] > 10]
            blocked_ok = blocked[(blocked['return_pct'] >= -5) & (blocked['return_pct'] <= 10)]

            n_big_loss_total = len(all_trades_with_factors[all_trades_with_factors['return_pct'] < -5])
            n_big_win_total = len(all_trades_with_factors[all_trades_with_factors['return_pct'] > 10])

            saved_loss = blocked_big_loss['return_pct'].sum() if len(blocked_big_loss) > 0 else 0
            lost_profit = blocked_big_win['return_pct'].sum() if len(blocked_big_win) > 0 else 0
            net = -saved_loss - lost_profit

            print(f"  {factor:38s} | {bl_median:11.3f} | {bl_p90:11.3f} | >={threshold:8.3f} | "
                  f"{len(blocked_big_loss):3d}/{n_big_loss_total:3d} | "
                  f"{len(blocked_big_win):3d}/{n_big_win_total:3d} | "
                  f"{len(blocked):3d}/{len(all_trades_with_factors):3d} | "
                  f"{len(blocked_ok):3d} | "
                  f"{'+' if net >= 0 else ''}{net:.1f}%")

# ============================================================
# Step 3: 组合因子过滤测试
# ============================================================
print("\n" + "=" * 80)
print("Step 3: 组合因子过滤测试")
print("=" * 80)

# 测试ChatGPT提出的组合:
# overheat_exhaustion = return_20d_intra_z >= 0.6 AND sector_momentum_interaction >= 8
#                       AND ma_convergence_score <= 0.18 AND trend_acceleration <= -4

# 也测试更宽松和更严格的版本
combos = [
    ("ChatGPT原始", lambda r: r.get('return_20d_intra_z', 999) >= 0.6 and r.get('sector_momentum_interaction', 0) >= 8 and r.get('ma_convergence_score', 999) <= 0.18 and r.get('trend_acceleration', 0) <= -4),
    ("宽松:ret20>=0.6+smi>=6", lambda r: r.get('return_20d_intra_z', 999) >= 0.6 and r.get('sector_momentum_interaction', 0) >= 6),
    ("宽松:ret20>=0.6+mac<=0.3", lambda r: r.get('return_20d_intra_z', 999) >= 0.6 and r.get('ma_convergence_score', 999) <= 0.3),
    ("宽松:ta<=-2+ret20>=0.5", lambda r: r.get('trend_acceleration', 0) <= -2 and r.get('return_20d_intra_z', 999) >= 0.5),
    ("中档:ret20>=0.7+smi>=7+mac<=0.25", lambda r: r.get('return_20d_intra_z', 999) >= 0.7 and r.get('sector_momentum_interaction', 0) >= 7 and r.get('ma_convergence_score', 999) <= 0.25),
    ("单因子:ret20>=0.9", lambda r: r.get('return_20d_intra_z', 999) >= 0.9),
    ("单因子:smi>=9", lambda r: r.get('sector_momentum_interaction', 0) >= 9),
    ("单因子:mac<=0.1", lambda r: r.get('ma_convergence_score', 999) <= 0.1),
    ("单因子:ta<=-4", lambda r: r.get('trend_acceleration', 0) <= -4),
]

print(f"\n{'Combo':40s} | {'Blocked':>7s} | {'BigLoss':>12s} | {'BigWin':>12s} | {'OK':>5s} | {'SavedLoss':>9s} | {'LostProfit':>10s} | {'Net':>8s} | {'AvgRet_Kept':>11s}")
print("-" * 135)

n_big_loss = len(all_trades_with_factors[all_trades_with_factors['return_pct'] < -5])
n_big_win = len(all_trades_with_factors[all_trades_with_factors['return_pct'] > 10])

for name, condition in combos:
    blocked = all_trades_with_factors[all_trades_with_factors.apply(condition, axis=1)]
    kept = all_trades_with_factors[~all_trades_with_factors.index.isin(blocked.index)]

    bl = blocked[blocked['return_pct'] < -5]
    bw = blocked[blocked['return_pct'] > 10]
    ok = blocked[(blocked['return_pct'] >= -5) & (blocked['return_pct'] <= 10)]

    saved = -bl['return_pct'].sum() if len(bl) > 0 else 0
    lost = bw['return_pct'].sum() if len(bw) > 0 else 0
    net = saved - lost

    avg_ret_kept = kept['return_pct'].mean() if len(kept) > 0 else 0

    print(f"  {name:38s} | {len(blocked):7d} | "
          f"{len(bl):3d}/{n_big_loss:3d} | "
          f"{len(bw):3d}/{n_big_win:3d} | "
          f"{len(ok):5d} | "
          f"{saved:+8.1f}% | {lost:+9.1f}% | "
          f"{'+' if net >= 0 else ''}{net:.1f}% | "
          f"{avg_ret_kept:+.2f}%")

# ============================================================
# Step 4: 被拦截的交易详情
# ============================================================
print("\n" + "=" * 80)
print("Step 4: 被ChatGPT组合拦截的交易详情")
print("=" * 80)

condition_chatgpt = lambda r: r.get('return_20d_intra_z', 999) >= 0.6 and r.get('sector_momentum_interaction', 0) >= 8 and r.get('ma_convergence_score', 999) <= 0.18 and r.get('trend_acceleration', 0) <= -4

blocked_chatgpt = all_trades_with_factors[all_trades_with_factors.apply(condition_chatgpt, axis=1)]

print(f"被拦截: {len(blocked_chatgpt)} 笔")
print(f"\n{'Stock':10s} | {'Return':>7s} | {'ret20z':>7s} | {'SMI':>6s} | {'MAC':>6s} | {'TA':>7s} | Category")
print("-" * 70)
for _, r in blocked_chatgpt.sort_values('return_pct').iterrows():
    cat = "BIG_LOSS" if r['return_pct'] < -5 else "BIG_WIN" if r['return_pct'] > 10 else "OK"
    print(f"  {r['stock_code']:10s} | {r['return_pct']:+7.1f}% | {r.get('return_20d_intra_z', 0):7.2f} | {r.get('sector_momentum_interaction', 0):6.1f} | {r.get('ma_convergence_score', 0):6.3f} | {r.get('trend_acceleration', 0):+7.2f} | {cat}")

# ============================================================
# Step 5: 在全量因子数据上评估拦截率
# ============================================================
print("\n" + "=" * 80)
print("Step 5: 在全量因子数据上评估拦截率")
print("=" * 80)

# 取最后120个交易日的因子数据（回测区间）
last_120_dates = all_dates[-120:]
df_recent = df_factors[df_factors['date'].isin(last_120_dates)].copy()
print(f"最后120个交易日因子数据: {df_recent.shape}")

for name, condition_str, col, op, threshold in [
    ("return_20d_intra_z >= 0.6", "return_20d_intra_z", "return_20d_intra_z", ">=", 0.6),
    ("sector_momentum_interaction >= 8", "sector_momentum_interaction", "sector_momentum_interaction", ">=", 8),
    ("ma_convergence_score <= 0.18", "ma_convergence_score", "ma_convergence_score", "<=", 0.18),
    ("trend_acceleration <= -4", "trend_acceleration", "trend_acceleration", "<=", -4),
    ("ChatGPT组合", None, None, None, None),
]:
    if name == "ChatGPT组合":
        blocked = df_recent[
            (df_recent['return_20d_intra_z'] >= 0.6) &
            (df_recent['sector_momentum_interaction'] >= 8) &
            (df_recent['ma_convergence_score'] <= 0.18) &
            (df_recent['trend_acceleration'] <= -4)
        ]
    elif op == ">=":
        blocked = df_recent[df_recent[col] >= threshold]
    else:
        blocked = df_recent[df_recent[col] <= threshold]

    pct = len(blocked) / len(df_recent) * 100
    # 按日期统计每天拦截比例
    daily_blocked = blocked.groupby('date').size()
    daily_total = df_recent.groupby('date').size()
    daily_pct = (daily_blocked / daily_total * 100).describe()
    print(f"\n  {name}:")
    print(f"    总拦截: {len(blocked)}/{len(df_recent)} ({pct:.2f}%)")
    print(f"    每日拦截比例: mean={daily_pct['mean']:.2f}%, max={daily_pct['max']:.2f}%, min={daily_pct['min']:.2f}%")
    if len(blocked) > 0:
        print(f"    涉及股票数: {blocked['stock_code'].nunique()}")

print("\n分析完成!")
