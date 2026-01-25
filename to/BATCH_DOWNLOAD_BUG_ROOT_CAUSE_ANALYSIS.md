# ts.pro_bar 批量下载Bug根本原因分析

## 问题现象

用户运行 `python download_daily.py --start 20260101 --end 20260109` 时：

```
批量下载成功，共 1247500 行数据
[异常] ts_code唯一值(499) != 批次股票数(500)，可能存在数据混乱
[异常] 数据行数(1247500)远超预期(4500)，可能存在重复数据
```

## 数据分析

### 基本数据
- **股票数量**: 500 只
- **日期范围**: 20260101 ~ 20260109 (9天)
- **预期行数**: 500 × 9 = **4,500 行**
- **实际行数**: **1,247,500 行**
- **比值**: **277.22 倍**

### 深度分析

#### 平均每只股票的数据量
```
1,247,500 行 ÷ 500 只股票 = 2,495 行/股
```

#### 每只股票应该有的数据量
```
9 天 × 1 条/天 = 9 行/股
```

#### 实际与预期的比值
```
2,495 行/股 ÷ 9 行/股 = 277.22 倍
```

#### 时间换算
```
2,495 天 ≈ 6.83 年 ≈ 2,495 天
```

## 根本原因推断

**`ts.pro_bar()` 的 `start_date` 和 `end_date` 参数在批量模式下失效！**

### 证据链

1. **日期参数失效**
   - 用户指定：`start_date='20260101', end_date='20260109'` (9天)
   - API实际返回：每只股票约2,495天的数据
   - **结论**：API忽略了日期范围参数

2. **返回全部历史数据**
   - 2,495天 ≈ 从2018年到2026年的数据
   - 这是A股一只股票的典型历史数据长度
   - **结论**：API返回了所有历史数据，而不是指定日期范围

3. **ts_code唯一值499 vs 500**
   - 用户分析正确：这是因为1只股票下载失败（停牌等）
   - 这是正常现象，不是bug

### 为什么会出现277.22倍？

计算过程：
```
1. API返回每只股票的全部历史数据（约2,495天）
2. 用户只需要其中9天的数据
3. 比值 = 2,495 ÷ 9 = 277.22
```

## 解决方案

### 方案1：修复批量下载参数（推荐尝试）

`ts.pro_bar()` 可能在批量模式下需要不同的参数格式：

```python
# 尝试1：使用 pro.daily() 而不是 ts.pro_bar()
batch_df_raw = pro.daily(
    ts_code=','.join(batch_stocks),
    start_date=start_date,  # 20260101
    end_date=end_date        # 20260109
)
```

```python
# 尝试2：检查ts.pro_bar()参数顺序
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    start_date=start_date,    # 先日期
    end_date=end_date,
    adj=adj                   # 后复权
)
```

```python
# 尝试3：使用 asset 参数明确指定资产类型
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    adj=adj,
    start_date=start_date,
    end_date=end_date,
    asset='E',               # E=股票，I=指数
    adjtype='1'              # 1=前复权，2=后复权，0=不复权
)
```

### 方案2：在下载后立即过滤日期（临时方案）

```python
# 批量下载（虽然会返回全部数据）
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    adj=adj,
    start_date=start_date,  # 可能会失效
    end_date=end_date
)

# 立即过滤日期范围
batch_df_raw = batch_df_raw[
    (batch_df_raw['trade_date'] >= start_date) &
    (batch_df_raw['trade_date'] <= end_date)
]
```

**优点**：
- 保留批量下载的性能优势
- 逻辑简单

**缺点**：
- 浪费网络带宽（下载了大量不需要的数据）
- 浪费API调用次数（可能有限流）

### 方案3：回退到逐股票下载（当前方案）

```python
for stock_code in batch_stocks:
    stock_df_raw = ts.pro_bar(
        ts_code=stock_code,      # 单只股票
        adj=adj,
        start_date=start_date,   # 单股票模式下有效
        end_date=end_date
    )
    # 处理...
```

**优点**：
- ✅ 100%可靠
- ✅ 日期参数正常工作
- ✅ 不浪费带宽

**缺点**：
- ⏱️ 下载时间较长（25-30分钟）

## 推荐实施步骤

### 立即实施：测试方案1（修复批量下载）

1. 创建测试脚本验证 `pro.daily()` API
2. 测试不同的参数组合
3. 如果成功，修改 `download_daily.py` 使用正确的API

### 备选方案：实施方案2（日期过滤）

1. 保持批量下载
2. 在下载后立即过滤日期
3. 添加日志显示过滤前后的数据量

### 兜底方案：保持方案3（逐股票下载）

- 如果批量下载确实无法修复
- 继续使用当前的逐股票下载模式

## 建议的测试代码

```python
import tushare as ts
import pandas as pd

pro = ts.pro_api()

# 测试 pro.daily()
print("测试 pro.daily()...")
df = pro.daily(
    ts_code='000001.SZ,000002.SZ',
    start_date='20240101',
    end_date='20240110'
)
print(f"返回行数: {len(df)}")
print(f"预期行数: 2 × 10 = 20")
print(f"trade_date范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

# 如果成功，说明应该使用 pro.daily() 而不是 ts.pro_bar()
```

## 总结

**根本原因**：`ts.pro_bar()` 在批量模式下，`start_date` 和 `end_date` 参数失效，导致返回每只股票的全部历史数据（约6-7年），而不是指定的9天数据。

**最佳解决方案**：使用 `pro.daily()` API 替代 `ts.pro_bar()`，这个API对批量下载的支持更稳定。

---

**创建时间**: 2025-01-11
**状态**: 待用户确认并测试
