#!/usr/bin/env python3
"""简化的行业中性化分析，避免编码问题"""

import pandas as pd
import numpy as np

def simple_sector_analysis():
    """简化的行业中性化分析"""
    print("=== 简化行业中性化分析 ===\n")

    try:
        # 1. 读取数据
        df = pd.read_parquet("C:/Users/86151/.trae-cn/pythons/path/to/local_data/features/daily/20241114.parquet")
        print(f"数据形状: {df.shape}")

        # 2. 加载行业映射
        from main import load_sw_sector_mapping
        sector_mapping = load_sw_sector_mapping()
        print(f"行业映射长度: {len(sector_mapping)}")

        # 3. 添加行业信息
        df['sector'] = df['stock_code'].map(sector_mapping)

        # 4. 创建pct_chg_1d_raw（这是原始数据，实际应该是前一日涨幅）
        df['pct_chg_1d_raw'] = df['pct_chg']

        # 5. 检查001279.SZ
        stock_data = df[df['stock_code'] == '001279.SZ']
        if len(stock_data) > 0:
            stock_info = stock_data.iloc[0]
            print(f"001279.SZ信息:")
            print(f"  原始pct_chg: {stock_info['pct_chg']:.6f}")
            print(f"  pct_chg_1d_raw: {stock_info['pct_chg_1d_raw']:.6f}")
            print(f"  行业: {stock_info['sector']}")

        # 6. 检查行业数据质量
        print(f"\n行业数据统计:")
        sector_stats = df.groupby('sector')['pct_chg_1d_raw'].agg(['count', 'mean', 'std', 'min', 'max'])
        print(sector_stats.head(10))

        # 7. 检查SW1机械设备行业（001279.SZ的行业）
        if 'SW1机械设备' in df['sector'].values:
            machinery_data = df[df['sector'] == 'SW1机械设备']
            print(f"\nSW1机械设备行业分析:")
            print(f"  股票数量: {len(machinery_data)}")
            print(f"  pct_chg_1d_raw统计:")
            print(f"    均值: {machinery_data['pct_chg_1d_raw'].mean():.6f}")
            print(f"    标准差: {machinery_data['pct_chg_1d_raw'].std():.6f}")
            print(f"    范围: [{machinery_data['pct_chg_1d_raw'].min():.6f}, {machinery_data['pct_chg_1d_raw'].max():.6f}]")

            # 检查该行业的极端值
            extreme_mask = abs(machinery_data['pct_chg_1d_raw']) > 9
            extreme_stocks = machinery_data[extreme_mask][['stock_code', 'pct_chg_1d_raw']]
            print(f"  极端值 (|值| > 9):")
            for _, row in extreme_stocks.iterrows():
                print(f"    {row['stock_code']}: {row['pct_chg_1d_raw']:.6f}")

        # 8. 检查所有行业是否有异常的均值
        print(f"\n行业均值检查:")
        sector_means = df.groupby('sector')['pct_chg_1d_raw'].mean()
        extreme_sectors = sector_means[abs(sector_means) > 5]
        print(f"异常行业均值 (|值| > 5):")
        for sector, mean_val in extreme_sectors.items():
            count = sector_stats.loc[sector, 'count']
            print(f"  {sector}: {mean_val:.6f} (股票数: {count})")

        # 9. 手动计算简单的行业中性化（使用行业均值调整）
        print(f"\n简单行业中性化测试:")
        clean_df = df.dropna(subset=['sector', 'pct_chg_1d_raw']).copy()

        # 计算行业均值
        sector_means = clean_df.groupby('sector')['pct_chg_1d_raw'].mean()

        # 计算残差（行业中性化后的值）
        clean_df['sector_mean'] = clean_df['sector'].map(sector_means)
        clean_df['neutralized'] = clean_df['pct_chg_1d_raw'] - clean_df['sector_mean']

        # 检查001279.SZ的结果
        stock_neutralized = clean_df[clean_df['stock_code'] == '001279.SZ']
        if len(stock_neutralized) > 0:
            stock_result = stock_neutralized.iloc[0]
            print(f"001279.SZ行业中性化结果:")
            print(f"  原始值: {stock_result['pct_chg_1d_raw']:.6f}")
            print(f"  行业均值: {stock_result['sector_mean']:.6f}")
            print(f"  中性化后残差: {stock_result['neutralized']:.6f}")

        # 检查所有中性化后的极端值
        print(f"\n中性化后极端值检查:")
        extreme_neutralized = clean_df[abs(clean_df['neutralized']) > 100]
        print(f"极端残差数量 (|值| > 100): {len(extreme_neutralized)}")

        if len(extreme_neutralized) > 0:
            print(f"极端残差详情 (前10个):")
            extreme_sample = extreme_neutralized.nlargest(10, 'neutralized')[['stock_code', 'sector', 'pct_chg_1d_raw', 'sector_mean', 'neutralized']]
            for _, row in extreme_sample.iterrows():
                print(f"  {row['stock_code']} ({row['sector']}): 原始={row['pct_chg_1d_raw']:.6f}, 行业均值={row['sector_mean']:.6f}, 残差={row['neutralized']:.6f}")

        # 10. 检查是否有可能导致1729这种异常值的计算
        print(f"\n异常值计算检查:")
        # 检查是否有行业数据质量问题
        zero_stocks = []
        single_stock_sectors = []

        for sector, group in clean_df.groupby('sector'):
            if len(group) == 1:
                single_stock_sectors.append(sector)
                print(f"单股票行业: {sector}, 股票: {group['stock_code'].iloc[0]}, 涨跌幅: {group['pct_chg_1d_raw'].iloc[0]:.6f}")

        print(f"\n发现 {len(single_stock_sectors)} 个单股票行业")

    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    simple_sector_analysis()