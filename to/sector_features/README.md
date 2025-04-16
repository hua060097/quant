# 板块特征使用说明

## 1. 板块特征概述

板块特征是量化交易系统中重要的分析和决策依据，本模块提供了丰富的板块特征计算和管理功能，包括基础特征（价格、成交量等）和高级特征（涨跌停比例、相对强度等）。

## 2. 特征类型

### 2.1 基础特征

- `open`：板块成分股开盘价的加权平均值
- `high`：板块成分股最高价的加权平均值
- `low`：板块成分股最低价的加权平均值
- `close`：板块成分股收盘价的加权平均值
- `volume`：板块成分股成交量的总和
- `amount`：板块成分股成交额的总和
- `change_percent`：板块当日涨跌幅，基于加权平均收盘价计算

### 2.2 技术指标特征

- `ma5`、`ma10`、`ma20`：板块价格的移动平均线
- `vol_ma5`、`vol_ma10`：板块成交量的移动平均线
- `macd`、`macd_signal`、`macd_hist`：MACD指标
- `rsi6`、`rsi12`：相对强弱指数
- `kdj_k`、`kdj_d`：KDJ随机指标
- `boll_upper`、`boll_middle`、`boll_lower`：布林带指标

### 2.3 市场关联特征（新增）

- `cyb_index_change_pct`：创业板指数当日涨跌幅
- `relative_to_cyb_change_pct`：板块相对于创业板指数的涨跌幅差值，用于衡量板块的相对强弱

### 2.4 涨跌停特征

- `limit_up_ratio`：板块中涨停股票的比例
- `limit_down_ratio`：板块中跌停股票的比例
- `near_limit_up_ratio`：板块中接近涨停的股票比例（涨幅>7%）
- `limit_open_ratio`：板块中一字板开盘的股票比例

### 2.5 排名特征

- `ten_day_rise_count`：板块内进入十日涨幅榜的股票数量
- `ten_day_rise_ratio`：板块内进入十日涨幅榜的股票比例
- `daily_rise_count`、`daily_rise_ratio`：日涨幅榜相关指标
- `daily_fall_count`、`daily_fall_ratio`：日跌幅榜相关指标
- `daily_amount_count`、`daily_amount_ratio`：日成交额榜相关指标

## 3. 相对涨跌幅特征详解

### 3.1 功能说明

相对涨跌幅特征通过比较板块与大盘（创业板指数）的涨跌幅差异，可以有效识别超越大盘表现的强势板块和弱于大盘表现的弱势板块，是构建板块轮动策略的重要指标。

### 3.2 计算方法

```python
# 相对涨跌幅计算
relative_change_pct = sector_change_pct - market_change_pct

# 其中：
# sector_change_pct: 板块当日涨跌幅
# market_change_pct: 创业板指数当日涨跌幅
```

### 3.3 使用场景

1. **板块强弱分析**：持续相对涨幅为正的板块具有相对强势
2. **板块轮动模型输入**：作为板块轮动模型的重要特征
3. **趋势反转识别**：相对强弱转变可能预示板块轮动机会
4. **风险分散**：选择与大盘相关性低的板块构建组合

### 3.4 特征解读

- **正值**：板块表现强于创业板指数，相对强势
  - 例如：板块涨3%，创业板涨1%，相对涨跌幅为+2%
  - 例如：板块跌1%，创业板跌3%，相对涨跌幅为+2%

- **负值**：板块表现弱于创业板指数，相对弱势
  - 例如：板块涨1%，创业板涨3%，相对涨跌幅为-2%
  - 例如：板块跌3%，创业板跌1%，相对涨跌幅为-2%

- **接近0**：板块表现与创业板指数接近，高度相关

## 4. 使用方法

### 4.1 加载板块特征

```python
from to.sector_loader import load_sector_features

# 加载单个板块特征
sector_data = load_sector_features(
    sector_name="TGN人工智能",
    start_date="20250101",
    end_date="20250331"
)

# 查看相对涨跌幅特征
print(sector_data[['date', 'change_percent', 'cyb_index_change_pct', 'relative_to_cyb_change_pct']])
```

### 4.2 筛选相对强势板块

```python
# 筛选最近10个交易日相对强势的板块
strong_sectors = []

for sector in sector_list:
    sector_data = load_sector_features(sector, start_date, end_date)
    # 计算最近10个交易日的平均相对涨跌幅
    avg_relative_change = sector_data.tail(10)['relative_to_cyb_change_pct'].mean()
    if avg_relative_change > 1.0:  # 相对涨跌幅平均超过1%
        strong_sectors.append((sector, avg_relative_change))

# 按相对强度降序排列
strong_sectors.sort(key=lambda x: x[1], reverse=True)
```

### 4.3 板块轮动策略中使用

```python
# 在板块轮动策略中使用相对涨跌幅特征
def select_rotation_sectors(date):
    all_sectors = get_available_sectors()
    sector_scores = []
    
    for sector in all_sectors:
        # 获取最近30天数据
        features = load_sector_features_for_period(sector, date, lookback_days=30)
        if features.empty:
            continue
            
        # 结合相对涨跌幅和其他指标计算得分
        relative_strength = features['relative_to_cyb_change_pct'].mean()
        momentum = features['change_percent'].tail(5).mean()
        volume_ratio = features['volume'].tail(5).mean() / features['vol_ma20'].tail(5).mean()
        
        # 综合得分计算
        total_score = 0.5 * relative_strength + 0.3 * momentum + 0.2 * volume_ratio
        sector_scores.append((sector, total_score))
    
    # 按得分降序排序
    sector_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 返回得分最高的3个板块
    return [s[0] for s in sector_scores[:3]]
```

## 5. 注意事项

1. **数据获取**：相对涨跌幅特征依赖于市场状态数据，确保`MarketStatusProcessor`正常工作
2. **数据缺失**：当无法获取创业板指数数据时，相对涨跌幅会设为0
3. **历史数据**：使用此特征前应确保市场状态数据已完整生成
4. **极端情况**：在极端市场环境下，相对涨跌幅可能会有较大波动，建议结合其他指标使用
5. **长期趋势**：分析中应关注相对涨跌幅的趋势变化，而非单日数值

## 6. 代码示例：相对强度热力图

以下是一个创建板块相对强度热力图的示例代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from to.sector_loader import load_sector_features, get_available_sectors

# 获取所有可用板块
sectors = get_available_sectors()

# 设定日期范围
start_date = "20250301"
end_date = "20250331"

# 创建相对强度数据框
relative_strength_data = []

for sector in sectors[:15]:  # 取前15个板块演示
    features = load_sector_features(sector, start_date, end_date)
    if features.empty:
        continue
        
    # 仅保留需要的列
    df = features[['date', 'relative_to_cyb_change_pct']].copy()
    df['sector'] = sector
    relative_strength_data.append(df)

# 合并所有板块数据
combined_data = pd.concat(relative_strength_data)

# 透视为板块 x 日期的矩阵
pivot_data = combined_data.pivot(index='sector', columns='date', values='relative_to_cyb_change_pct')

# 绘制热力图
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_data, cmap='RdYlGn', center=0, annot=False)
plt.title('板块相对创业板指数的相对强度热力图')
plt.xlabel('日期')
plt.ylabel('板块')
plt.tight_layout()
plt.savefig('sector_relative_strength.png', dpi=300)
plt.show()
```

## 7. 未来增强计划

1. 支持多指数对比：除创业板指数外，增加上证指数、深证成指等对比
2. 动态相关性分析：计算板块与指数的滚动相关系数
3. Beta系数计算：衡量板块对市场变动的敏感度
4. 波动率对比：增加板块与大盘的波动率比较
5. 板块联动性分析：识别板块间的联动关系和传导效应 