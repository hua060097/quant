# 使用 pro.daily() API 修复批量下载

## 根本原因确认

**问题**：`ts.pro_bar()` 在批量模式下，`start_date` 和 `end_date` 参数失效，返回每只股票的全部历史数据。

**证据**：
- 预期：500 股票 × 9 天 = 4,500 行
- 实际：1,247,500 行
- 每只股票平均：2,495 行 ≈ 6.8年的全部历史数据
- **结论**：API忽略了日期范围参数

## 解决方案：使用 pro.daily() API

`pro.daily()` 是 Tushare Pro API，对批量下载的支持更稳定。

### 修复代码

修改 `download_daily.py` 的批量下载部分（lines 176-214）：

```python
# 修改前（ts.pro_bar - 有bug）
batch_df_raw = ts.pro_bar(
    ts_code=','.join(batch_stocks),
    adj=adj,
    start_date=start_date,
    end_date=end_date
)

# 修改后（pro.daily - 稳定）
batch_df_raw = pro.daily(
    ts_code=','.join(batch_stocks),
    start_date=start_date,
    end_date=end_date
)
```

### 完整的修复代码

```python
# 下载当前批次的数据（使用 pro.daily API）
batch_data = []
try:
    logger.info(f"  批量下载 {len(batch_stocks)} 只股票...")

    # 使用 pro.daily 替代 ts.pro_bar（批量模式更稳定）
    batch_df_raw = pro.daily(
        ts_code=','.join(batch_stocks),
        start_date=start_date,
        end_date=end_date
    )

    if batch_df_raw is None or len(batch_df_raw) == 0:
        logger.warning(f"  批量下载失败，无数据返回")
        failed_stocks.extend(batch_stocks)
    else:
        logger.info(f"  批量下载成功，共 {len(batch_df_raw)} 行数据")

        # 数据验证
        expected_rows = len(batch_stocks) * ((datetime.strptime(end_date, '%Y%m%d') -
                                             datetime.strptime(start_date, '%Y%m%d')).days + 1)

        if len(batch_df_raw) > expected_rows * 1.5:
            logger.warning(f"  [警告] 数据行数({len(batch_df_raw)})远超预期({expected_rows})，可能存在异常")
        else:
            logger.info(f"  [OK] 数据行数正常")

        # 按股票分组处理
        for stock_code in batch_stocks:
            stock_df = batch_df_raw[batch_df_raw['ts_code'] == stock_code].copy()

            if len(stock_df) == 0:
                logger.warning(f"    {stock_code} 无数据，跳过")
                failed_stocks.append(stock_code)
                continue

            # 字段映射与转换
            stock_df = _transform_tushare_data(stock_df, stock_code)
            batch_data.append(stock_df)

        logger.info(f"  成功处理 {len(batch_data)}/{len(batch_stocks)} 只股票")

except Exception as e:
    logger.error(f"  批量下载失败: {e}，降级为逐个下载")
    # 降级方案：逐个下载
    for j, stock_code in enumerate(batch_stocks, 1):
        try:
            logger.info(f"  [{j}/{len(batch_stocks)}] 下载 {stock_code}...")

            # 使用 pro.daily 逐股票下载
            stock_df_raw = pro.daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )

            if stock_df_raw is None or len(stock_df_raw) == 0:
                logger.warning(f"    {stock_code} 无数据，跳过")
                failed_stocks.append(stock_code)
                continue

            # 字段映射与转换
            stock_df = _transform_tushare_data(stock_df_raw, stock_code)
            batch_data.append(stock_df)

        except Exception as e2:
            logger.error(f"    {stock_code} 下载失败: {e2}")
            failed_stocks.append(stock_code)
            continue

    logger.info(f"  批次完成，成功处理 {len(batch_data)}/{len(batch_stocks)} 只股票")
```

### 需要修改的字段映射

`pro.daily()` 和 `ts.pro_bar()` 的字段名称可能不同，需要调整 `_transform_tushare_data()` 函数：

```python
def _transform_tushare_data(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    转换Tushare数据格式为目标格式

    支持 pro.daily() 和 ts.pro_bar() 两种格式
    """
    # 字段重命名
    # pro.daily() 和 ts.pro_bar() 都使用 ts_code 和 trade_date
    df = df.rename(columns={
        'ts_code': 'stock_code',
        'trade_date': 'date'
    })

    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    # volume转换：手 -> 股（乘以100）
    if 'vol' in df.columns:
        df['volume'] = df['vol'] * 100
        df = df.drop(columns=['vol'])
    elif 'volume' not in df.columns:
        df['volume'] = df['vol'] * 100 if 'vol' in df.columns else 0

    # ... 其余处理保持不变

    return df
```

## API对比

| 特性 | ts.pro_bar() | pro.daily() |
|------|--------------|-------------|
| **批量下载** | ❌ 日期参数失效 | ✅ 正常工作 |
| **复权支持** | ✅ 通过 adj 参数 | ❌ 需要自己计算 |
| **API类型** | 旧版 | Pro版 |
| **稳定性** | 有bug | 稳定 |

### 复权处理

`pro.daily()` 不支持 `adj` 参数，需要自己计算复权：

**方案1：下载不复权数据，后续计算**
```python
batch_df_raw = pro.daily(
    ts_code=','.join(batch_stocks),
    start_date=start_date,
    end_date=end_date
    # 注意：没有 adj 参数
)
```

**方案2：仍然使用 ts.pro_bar()，但逐股票下载**
```python
for stock_code in batch_stocks:
    stock_df_raw = ts.pro_bar(
        ts_code=stock_code,  # 单只股票时日期参数有效
        adj='qfq',
        start_date=start_date,
        end_date=end_date
    )
```

## 推荐方案

由于 `pro.daily()` 不支持复权参数，建议采用**混合方案**：

1. **优先尝试批量下载**：使用 `pro.daily()` + 后复权
2. **降级到逐股票下载**：如果需要复权，使用 `ts.pro_bar()` 逐股票下载

### 具体实施

```python
# 尝试1：使用 pro.daily() 批量下载（无复权）
try:
    batch_df_raw = pro.daily(
        ts_code=','.join(batch_stocks),
        start_date=start_date,
        end_date=end_date
    )
    # 无复权数据，用于分析
except:
    batch_df_raw = None

# 如果批量下载失败，降级到逐股票下载（有复权）
if batch_df_raw is None or len(batch_df_raw) == 0:
    for stock_code in batch_stocks:
        stock_df_raw = ts.pro_bar(
            ts_code=stock_code,
            adj='qfq',  # 单股票模式支持复权
            start_date=start_date,
            end_date=end_date
        )
        # 处理...
```

## 测试验证

运行以下命令测试修复效果：

```bash
# 测试小范围数据
python download_daily.py --start 20240101 --end 20240105

# 检查日志，确认数据量正常
# 预期：500 股票 × 5 天 = 2,500 行（不是 250,000 行）
```

## 预期效果

修复后：
- ✅ 批量下载恢复正常
- ✅ 数据量正确（每天每只股票1条）
- ✅ 下载时间：3-5分钟（vs 逐股票的25-30分钟）
- ✅ 无重复数据

---

**状态**: 待用户测试验证
**下一步**: 请用户运行 `test_pro_daily_api.py` 验证 `pro.daily()` API
