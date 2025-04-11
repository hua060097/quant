# 市场状态指标指南

本文档详细介绍了市场状态处理器生成的各项指标的含义和用途，这些指标用于描述市场状态并可以被模型训练使用。

## 1. 核心市场类型指标

### 市场类型编码 (market_type_code)

市场类型编码是一个两位数字，第一位表示大类，第二位表示细分类型：

| 编码 | 名称 | 描述 |
|------|------|------|
| 11 | 强势突破上涨 | 突破重要压力位，量价配合，持续性强 |
| 12 | 加速上涨 | 快速上涨，但未到过热 |
| 13 | 稳健上涨 | 均线多头排列，量价良好 |
| 21 | 盘升上涨 | 缓慢向上，成交量温和 |
| 22 | 回调上涨 | 短期回调后继续上行 |
| 23 | 整固上涨 | 高位整理后继续上行 |
| 31 | 强势盘整 | 高位盘整，支撑强 |
| 32 | 中性盘整 | 箱体震荡，方向不明 |
| 33 | 弱势盘整 | 低位盘整，压力大 |
| 41 | 盘跌下行 | 缓慢向下，暂无明显企稳 |
| 42 | 回抽下跌 | 反弹后继续下行 |
| 43 | 整固下跌 | 低位整理后继续下行 |
| 51 | 加速下跌 | 快速下跌，但未到恐慌 |
| 52 | 恐慌下跌 | 快速大幅下跌，成交量萎缩 |
| 53 | 超跌反弹 | 大幅下跌后的反弹修复 |

### 市场周期编码 (market_cycle_code)

市场周期编码表示市场所处的大周期阶段：

| 编码 | 名称 | 描述 |
|------|------|------|
| 1 | 积累期 | 底部区域，价格低迷，成交量低，但有机构悄悄建仓 |
| 2 | 上升期 | 突破上行，成交量逐渐放大，趋势确立 |
| 3 | 分配期 | 高位区域，成交量显著放大，波动加剧，机构悄悄卖出 |
| 4 | 下降期 | 明确下跌，成交量先放大后萎缩，趋势向下 |

## 2. 市场评分指标

所有评分指标都经过归一化处理，范围为0-5，其中0表示极弱/消极，5表示极强/积极。

### trend_score (趋势得分)

评估市场当前趋势的强度和方向。

**范围**：0-5
- **5分**：极强的上升趋势，多头排列明确，动能强
- **4分**：强劲上升趋势，但有轻微回调迹象
- **3分**：中等上升趋势，或高位盘整
- **2分**：弱上升趋势，或中性盘整
- **1分**：弱下降趋势，或弱势盘整
- **0分**：强烈下降趋势，空头排列明确

计算因素：
- 价格与各周期均线的关系
- 均线系统的排列形态
- MACD柱状图的方向和幅度
- 近期价格走势的方向性和一致性

### position_score (位置得分)

评估当前价格在近期波动区间中的相对位置。

**范围**：0-5
- **5分**：位于上轨突破位置，有新高潜力
- **4分**：接近上轨，但未突破
- **3分**：位于中轨上方，呈现上行趋势
- **2分**：位于中轨附近，方向不明
- **1分**：位于中轨下方，呈现下行趋势
- **0分**：位于下轨或更低，超跌状态

计算因素：
- 布林带中的相对位置
- 相对于最近高点和低点的位置
- 价格是否处于支撑或阻力位附近

### volume_score (量能得分)

评估交易量的强度和特征。

**范围**：0-5
- **5分**：成交量放大并持续，并与价格上涨配合
- **4分**：成交量稳定增长，交投活跃
- **3分**：成交量中等，略高于近期平均
- **2分**：成交量中等，与近期平均相当
- **1分**：成交量萎缩，交投不活跃
- **0分**：成交量极度萎缩，几乎无人交易

计算因素：
- 近期成交量相对于均量的水平
- 量价配合是否合理
- 成交量变化趋势

### momentum_score (动能得分)

评估价格动能和涨跌速度。

**范围**：0-5
- **5分**：强劲上升动能，快速上涨，超买但仍有动力
- **4分**：良好上升动能，上涨速度适中
- **3分**：中等上升动能，或开始减缓
- **2分**：中性动能，横盘整理
- **1分**：下降动能，开始加速下跌
- **0分**：强烈下降动能，快速下跌，超卖状态

计算因素：
- RSI指标值及变化
- MACD值及变化速度
- KDJ指标的数值和方向
- 价格变化的加速度

### volatility_score (波动得分)

评估市场波动性的大小。

**范围**：0-5
- **5分**：极高波动性，剧烈震荡，高风险高回报
- **4分**：高波动性，明显震荡
- **3分**：中等波动性，波动开始增加
- **2分**：低中等波动性，相对稳定
- **1分**：低波动性，波动很小
- **0分**：极低波动性，几乎无波动

计算因素：
- 布林带宽度
- 近期价格波动范围
- 价格变化的标准差
- 高低点之间的距离

### structure_score (结构得分)

评估价格形态的完整性和可靠性。

**范围**：0-5
- **5分**：完美的上涨结构，高点抬高，低点抬高
- **4分**：良好的上涨结构，有小幅回调
- **3分**：中性结构，可能处于形态转变过程中
- **2分**：弱趋势结构，形态不明确
- **1分**：下跌结构开始形成，低点降低
- **0分**：完整的下跌结构，高点降低，低点降低

计算因素：
- 高低点排列模式
- 是否形成特定形态（如头肩顶、双底等）
- 突破确认情况
- 支撑位和阻力位的表现

### cycle_score (周期得分)

评估市场在周期中的位置。

**范围**：0-5
- **5分**：处于上升周期初期，有望继续上涨
- **4分**：处于上升周期中期，上涨动力仍在
- **3分**：处于上升周期后期或顶部区域
- **2分**：开始转入下降周期，但尚未确认
- **1分**：处于下降周期中期
- **0分**：处于下降周期后期或底部区域

计算因素：
- 对大周期转折点的识别
- 周期指标的读数
- 价格波动的周期性特征
- 与历史周期的对比分析

### total_score (总分)

综合以上各项得分的加权平均，反映市场的整体状态。

**范围**：0-5
- **4-5分**：强势市场，适合积极做多
- **3-4分**：偏强市场，可谨慎做多
- **2-3分**：中性市场，观望或轻仓参与
- **1-2分**：偏弱市场，减仓为主
- **0-1分**：弱势市场，适当规避风险

计算方法：
```
total_score = (
    trend_weight * trend_score + 
    position_weight * position_score + 
    volume_weight * volume_score +
    momentum_weight * momentum_score +
    volatility_weight * volatility_score +
    structure_weight * structure_score +
    cycle_weight * cycle_score
) / sum_of_weights
```

其中权重根据市场环境动态调整，一般趋势得分和周期得分的权重较高。

## 3. 技术指标数据

以下原始技术指标数据会被保存在`technical_data`字典中，用于进一步分析和模型训练。所有数值数据均保留两位小数。

### MA系列 (移动平均线)
- `ma5`: 5日移动平均线
- `ma10`: 10日移动平均线
- `ma20`: 20日移动平均线
- `ma60`: 60日移动平均线

### MACD (指数平滑异同平均线)
- `macd`: MACD线，快线(EMA12)减慢线(EMA26)
- `macd_signal`: 信号线，MACD的9日EMA
- `macd_hist`: MACD柱状图，MACD减信号线

### RSI (相对强弱指数)
- `rsi6`: 6日相对强弱指数
- `rsi12`: 12日相对强弱指数

### 布林带指标
- `bb_width`: 布林带宽度，(上轨-下轨)/中轨
- `bb_position`: 价格在布林带中的相对位置，0表示在下轨，1表示在上轨

### KDJ指标
- `kdj_k`: K值，当前收盘价在最近n天的高低点范围内的相对位置
- `kdj_d`: D值，K值的移动平均

### 价格相关指标
- `close`: 当日收盘价
- `prev_close`: 前一日收盘价
- `support`: 支撑位，当前重要的价格支撑水平，价格下跌到此位置可能会获得支撑反弹
- `resistance`: 压力位，当前重要的价格压力水平，价格上涨到此位置可能会遇到阻力回落
- `change_percent`: 涨跌幅，相对于前一交易日的百分比变化，正值表示上涨，负值表示下跌

## 4. 市场数据

以下市场数据会被保存在`market_data`字典中，提供当日基础行情数据。所有数值数据均保留两位小数。

- `open`: 当日开盘价
- `high`: 当日最高价
- `low`: 当日最低价
- `close`: 当日收盘价
- `volume`: 当日成交量
- `daily_change_pct`: 当日涨跌幅百分比

## 5. 市场状态信息

市场状态信息会被保存在`market_status_info`字典中，提供对市场类型和周期的编码与描述。

- `type_code`: 市场类型编码，如11(强势突破上涨)、32(中性盘整)等
- `type_desc`: 市场类型描述文本
- `cycle_code`: 市场周期编码，如1(积累期)、2(上升期)等
- `cycle_desc`: 市场周期描述文本

为了兼容性，以下字段也会单独保存在顶层：
- `market_description`: 与`type_desc`相同，市场类型描述
- `cycle_description`: 与`cycle_desc`相同，市场周期描述

## 6. 数据格式示例

以下是保存的市场状态数据结构示例：

```json
 "20240122": {
        "market_type": 41,
        "market_description": 41,
        "market_cycle": 4,
        "cycle_description": 4,
        "trend_strength": 2.35,
        "signals": {
            "trading_advice": "观望为主，等待明确信号",
            "risk_level": "low",
            "strength": 0
        },
        "individual_status": {
            "399006.SZ": {
                "trend_score": -7.5,
                "position_score": -1.0,
                "volume_score": 1.0,
                "momentum_score": -1.0,
                "volatility_score": 3.0,
                "structure_score": -1.0,
                "cycle_score": -1.0,
                "total_score": -2.35,
                "market_status_info": {
                    "type_code": 41,
                    "type_desc": "盘跌下行",
                    "cycle_code": 4,
                    "cycle_desc": "下降期"
                },
                "market_description": "盘跌下行",
                "cycle_description": "下降期",
                "technical_data": {
                    "close": 1666.88,
                    "prev_close": 1715.44,
                    "ma5": 1713.3,
                    "ma10": 1734.45,
                    "ma20": 1778.89,
                    "ma60": 1881.43,
                    "rsi6": 25.19,
                    "rsi12": 29.8,
                    "macd": -44.68,
                    "macd_signal": -38.77,
                    "macd_hist": -11.82,
                    "kdj_k": 21.37,
                    "kdj_d": 22.4,
                    "bb_width": 0.13,
                    "bb_position": 0.02
                },
                "market_data": {
                    "close": 1666.88,
                    "open": 1710.29,
                    "high": 1714.93,
                    "low": 1657.35,
                    "volume": 117732596.0,
                    "daily_change_pct": -2.83
                },
                "support": 1678.43,
                "resistance": 1666.88,
                "change_percent": -2.83
            }
        }
    }
```

## 7. 模型训练使用指南

在 `train_sector_models.py` 中，以下特征被用于训练预测模型：

1. **市场类型和周期特征**：
   - `market_status_info.type_code`: 市场类型编码(数值)
   - `market_status_info.cycle_code`: 市场周期编码(数值)

2. **市场评分指标**：
   - `trend_score`: 趋势得分
   - `position_score`: 位置得分
   - `volume_score`: 量能得分
   - `momentum_score`: 动能得分
   - `volatility_score`: 波动得分
   - `structure_score`: 结构得分
   - `cycle_score`: 周期得分
   - `total_score`: 总分

3. **技术指标派生特征**：
   - `technical_data.close/technical_data.ma5`: 价格与5日均线的比值
   - `technical_data.close/technical_data.ma20`: 价格与20日均线的比值
   - `technical_data.close/technical_data.ma60`: 价格与60日均线的比值
   - `technical_data.ma5/technical_data.ma20`: 短期均线与中期均线的比值
   - `technical_data.macd_hist`: MACD柱状图数值，反映动量变化
   - `technical_data.rsi6/50-1`: RSI相对中性位置的偏离度
   - `technical_data.bb_position*2-1`: 布林带位置归一化到[-1,1]区间
   - `technical_data.kdj_k-technical_data.kdj_d`: KD值差，反映短期动量变化

4. **价格支撑压力特征**：
   - `support`: 支撑位水平
   - `resistance`: 压力位水平
   - `change_percent`: 涨跌幅百分比
   - `technical_data.close/support-1`: 当前价格与支撑位的距离比率
   - `resistance/technical_data.close-1`: 压力位与当前价格的距离比率

5. **市场数据特征**：
   - `market_data.volume/previous_volume-1`: 成交量变化率
   - `market_data.high-market_data.low)/market_data.close`: 日内波动率
   - `(market_data.close-market_data.open)/market_data.open`: 日内涨跌幅
   - `market_data.daily_change_pct`: 当日涨跌幅百分比

通过组合这些特征的不同时间窗口(短期、中期、长期)值，可以有效地训练出预测市场未来走势的模型。所有数值特征在训练前会进行归一化和标准化处理，以提高模型的稳定性和性能。

## 8. 使用场景和最佳实践

### 回测系统集成

市场状态数据特别适合集成到回测系统中，作为交易决策的重要依据：

```python
# 获取当日市场状态
market_status = processor.get_market_status(date_str)

# 根据市场状态调整策略参数
if market_status['total_score'] > 3.5:
    # 强势市场，更激进的策略参数
    position_size = 1.0
    stop_loss = 0.95
elif market_status['total_score'] > 2.5:
    # 中性市场，中等策略参数
    position_size = 0.7
    stop_loss = 0.97
else:
    # 弱势市场，保守策略参数
    position_size = 0.3
    stop_loss = 0.98
```

### 板块轮动模型增强

通过结合市场状态分析，可以增强板块轮动模型的性能：

```python
# 获取市场状态和板块预测
market_status = processor.get_market_status(date_str)
sector_predictions = model.predict_sectors(features)

# 根据市场状态调整板块选择
if market_status['market_status_info']['type_code'] in [11, 12, 13]:
    # 强势上涨市场，选择成长型板块
    selected_sectors = filter_growth_sectors(sector_predictions)
elif market_status['market_status_info']['type_code'] in [31, 32, 33]:
    # 盘整市场，选择防御型板块
    selected_sectors = filter_defensive_sectors(sector_predictions)
else:
    # 下跌市场，减少选股数量，提高选股标准
    selected_sectors = filter_top_sectors(sector_predictions, threshold=0.8)
```

### 风险管理应用

市场状态数据对风险管理特别有价值，可以用于动态调整风险暴露：

```python
# 获取市场状态
market_status = processor.get_market_status(date_str)

# 根据波动性得分调整仓位
volatility = market_status['volatility_score']
if volatility > 4.0:
    # 高波动环境，减少仓位
    risk_exposure = 0.6
elif volatility > 3.0:
    # 中等波动环境，保持适中仓位
    risk_exposure = 0.8
else:
    # 低波动环境，可以增加仓位
    risk_exposure = 1.0
```

## 9. 常见问题与解决方案

### Q: 市场状态数据如何处理缺失值？
A: 当某些技术指标无法计算时(如数据不足)，相应字段会设置为`null`。在使用时应当检查并提供合理的默认值。

### Q: 如何解决历史数据回溯中的前瞻偏差？
A: 确保在回测中，只使用截至当前日期可获得的市场状态数据，避免使用包含未来信息的特征。

### Q: 不同市场环境下应该关注哪些指标？
A: 
- 上涨市场: `trend_score`, `momentum_score`, `volume_score`
- 盘整市场: `position_score`, `volatility_score`, `support/resistance`
- 下跌市场: `trend_score`, `structure_score`, `volume_score` 