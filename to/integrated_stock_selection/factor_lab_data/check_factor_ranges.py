"""
查看指定因子在两个时间段内的最大、最小值
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径
factor_file = Path(__file__).parent / "factors_correct.parquet"

print("=" * 80)
print("因子范围统计")
print("=" * 80)

# 读取因子文件
print(f"\n读取文件: {factor_file}")
df = pd.read_parquet(factor_file)

# 重置索引，方便按日期过滤
df = df.reset_index()

# 转换日期列为字符串格式（假设索引名为'date'或第一列是日期）
if 'date' in df.columns:
    df['date_str'] = df['date'].astype(str).str.replace('-', '').str[:8]
else:
    # 假设第一列是日期
    df['date_str'] = df.iloc[:, 0].astype(str).str.replace('-', '').str[:8]

print(f"数据形状: {df.shape}")
print(f"日期范围: {df['date_str'].min()} ~ {df['date_str'].max()}")

# 目标因子
target_factors = [
    'market_vols_adaptive_iv',
    'market_trend_adaptive_beta',
    'sector_excess_intra_rank',
    'coskewness',
    'sector_limit_overflow_alpha',
    'sector_broken_resilience_alpha',
    'limit_sustain_alpha',
    'market_emotion_resonance_alpha',
    'mkt_sector_dispersion', 'sector_mkt_corr_20d', 'intra_sector_breadth', 'stock_sector_pos_align',
    'sector_vols_divergence_alpha',
    'sector_turnover_surge_pv_alpha',
    'sector_dispersion_leader_alpha',
    'sector_trend_gated_momentum_intra_z',
    'sector_rotation_breakout_intra_z',
    'sector_leader_capture_rank',
    'sector_strong_leader_alpha',
    'sector_liquidity_gated_turnover_z',
    'sector_weak_reversal_alpha'
]

# 时间段
periods = [
    ('20230601', '20251226', '20230601~20251226')#,
    # ('20250605', '20260117', '20250605~20260117')
]

# 检查因子是否存在
missing_factors = [f for f in target_factors if f not in df.columns]
if missing_factors:
    print(f"\n警告: 以下因子不存在: {missing_factors}")
    print(f"可用的因子列: {df.columns.tolist()[:20]}...")  # 只显示前20个

# 统计每个时间段
for start_date, end_date, period_name in periods:
    print(f"\n{'=' * 80}")
    print(f"时间段: {period_name}")
    print(f"{'=' * 80}")

    # 筛选时间段
    mask = (df['date_str'] >= start_date) & (df['date_str'] <= end_date)
    period_df = df[mask]

    if len(period_df) == 0:
        print(f"  该时间段无数据")
        continue

    print(f"  数据量: {len(period_df):,} 条")

    # 计算每个因子的统计值
    for factor in target_factors:
        if factor not in df.columns:
            continue

        # 提取因子值（排除NaN）
        factor_data = period_df[['date_str', factor]].dropna(subset=[factor])

        if len(factor_data) == 0:
            print(f"\n  {factor}:")
            print(f"    无有效数据（全为NaN）")
            continue

        min_val = factor_data[factor].min()
        max_val = factor_data[factor].max()
        mean_val = factor_data[factor].mean()
        std_val = factor_data[factor].std()

        # 找到最小值和最大值对应的日期
        min_date_rows = factor_data[factor_data[factor] == min_val]
        max_date_rows = factor_data[factor_data[factor] == max_val]

        # 获取日期（可能有多个相同值，取第一个）
        min_dates = min_date_rows['date_str'].unique().tolist()[:3]  # 最多显示3个
        max_dates = max_date_rows['date_str'].unique().tolist()[:3]

        print(f"\n  {factor}:")
        print(f"    最小值: {min_val:.6f} (日期: {', '.join(min_dates)})")
        print(f"    最大值: {max_val:.6f} (日期: {', '.join(max_dates)})")
        print(f"    均值:   {mean_val:.6f}")
        print(f"    标准差: {std_val:.6f}")
        print(f"    有效数据量: {len(factor_data):,} / {len(period_df):,}")

print(f"\n{'=' * 80}")
print("统计完成")
print("=" * 80)
