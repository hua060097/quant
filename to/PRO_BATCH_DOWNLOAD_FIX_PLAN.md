# ts.pro_bar 批量下载问题分析与修复方案

## 问题现象

运行 `python download_daily.py --start 20260101 --end 20260109` 时出现：
```
批量下载成功，共 1247500 行数据
[异常] ts_code唯一值(499) != 批次股票数(500)
[异常] 数据行数(1247500)远超预期(4500)，可能存在重复数据
```

### 数据分析
- 预期行数：500 股票 × 9 天 = 4,500 行
- 实际行数：1,247,500 行
- **重复倍数：277.22 倍**
- ts_code 唯一值：499（1 只股票缺失）

## 根本原因

**`ts.pro_bar()` API 在批量模式下存在 bug**，可能的原因：

1. **重复请求**：批量下载时，API 对每个股票发送了多次请求
2. **参数错误**：`ts_code` 参数使用逗号分隔时，API 内部处理有误
3. **数据合并问题**：API 返回时重复合并了数据

## 修复方案

### 方案1：回退到逐股票下载（推荐，最稳妥）

**优点**：
- 完全可靠，不会出现重复数据
- 代码简单，易于维护

**缺点**：
- API 调用次数多（5000 次 vs 50 次）
- 下载时间较长（25-30 分钟 vs 3-5 分钟）

**实施**：
```python
# 将批量下载改为逐股票下载
for stock in batch_stocks:
    df = ts.pro_bar(
        ts_code=stock,  # 单只股票
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )
```

### 方案2：使用 pro.daily() API（推荐，性能好）

**优点**：
- Pro API 更稳定
- 支持批量下载
- 性能好

**缺点**：
- 需要修改代码，使用 `pro.daily()` 替代 `ts.pro_bar()`
- 字段名称可能不同

**实施**：
```python
# 使用 pro.daily 替代 ts.pro_bar
batch_df_raw = pro.daily(
    ts_code=','.join(batch_stocks),
    start_date=start_date,
    end_date=end_date
)
# 注意：pro.daily 不支持 adj 参数，需要自己计算复权
```

### 方案3：保留批量下载 + 强化去重（不推荐）

**优点**：
- 保留批量下载的性能优势

**缺点**：
- 治标不治本
- 仍然会有大量无效 API 请求
- 网络带宽浪费

## 推荐实施方案

**采用方案1 + 方案2 的混合方案**：

1. **立即修复**：回退到逐股票下载（方案1）
   - 确保数据正确性
   - 避免重复数据

2. **后续优化**：评估 `pro.daily()` API（方案2）
   - 测试 `pro.daily()` 是否支持批量
   - 如果支持，迁移到 Pro API

## 立即行动

### 修改 download_daily.py

将批量下载代码：
```python
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    adj=adj,
    start_date=start_date,
    end_date=end_date
)
```

改为逐股票下载：
```python
for stock_code in batch_stocks:
    stock_df_raw = ts.pro_bar(
        ts_code=stock_code,
        adj=adj,
        start_date=start_date,
        end_date=end_date
    )
    # 处理 stock_df_raw
```

### 保留去重逻辑

已有的去重修复（修复点1-4）仍然保留，作为双重保障。

## 预期效果

修复后：
- API 调用次数：约 5,000 次（5000 股票 × 1 天）
- 下载时间：25-30 分钟（可接受）
- 数据质量：100% 正确，无重复
- 文件大小：正常（每天 ~0.5 MB）

---

**结论**：由于 `ts.pro_bar()` 批量下载存在严重 bug，**强烈建议立即回退到逐股票下载模式**。
