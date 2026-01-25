# 从Tushare下载日线数据 - 使用说明

## 功能概述

从Tushare Pro API下载股票日线数据，并保存为与`advanced_quant_data_fixed.py`中`download_daily_data`相同格式的Parquet文件。

## 主要特性

### 1. 内存优化
- **分批处理**：每批处理100只股票，可配置
- **增量保存**：每累积50000行数据后保存一次，释放内存
- **避免OOM**：5000只股票×3年约275万行数据，分批处理避免内存溢出

### 2. 断点续传
- **进度保存**：下载进度自动保存到`download_progress_tushare.pkl`
- **自动恢复**：程序中断后重新运行，自动从上次中断处继续
- **增量更新**：检查已存在的parquet文件，只下载缺失的数据

### 3. 数据格式兼容
- **字段映射**：自动转换Tushare字段为目标格式
- **市值计算**：自动加载股本数据，计算总市值和流通市值
- **换手率计算**：根据成交量和流通股本计算换手率
- **按月存储**：与原有格式一致，按YYYYMM分区存储

## 数据字段对照

| Tushare字段 | 目标字段 | 转换说明 |
|------------|---------|---------|
| ts_code | stock_code | 直接映射 |
| trade_date | date | 转为datetime格式 |
| open | open | 直接映射 |
| high | high | 直接映射 |
| low | low | 直接映射 |
| close | close | 直接映射 |
| pre_close | preClose | 直接映射 |
| vol | volume | 乘以100（手→股） |
| amount | amount | 直接映射（元） |
| pct_chg | pct_chg | 直接映射（%） |
| - | total_mv | 计算得出：close × total_capital |
| - | circ_mv | 计算得出：close × circulating_capital |
| - | turnover_rate | 计算得出：(volume / circulating_capital) × 100 |

## 使用方法

### 基本使用

```python
python download_daily.py
```

### 自定义参数

修改`download_daily.py`的main函数中的参数：

```python
# 配置下载参数
START_DATE = '20230101'  # 开始日期
END_DATE = '20231231'    # 结束日期
ADJ = 'qfq'             # 复权类型：'qfq'-前复权, 'hfq'-后复权, 'none'-不复权

success = download_daily_from_tushare(
    stock_list=stock_list,
    start_date=START_DATE,
    end_date=END_DATE,
    adj=ADJ,
    batch_size=100,          # 每批处理多少只股票
    rows_per_save=50000,     # 累积多少行后保存一次
    resume=True               # 是否启用断点续传
)
```

## 输出文件结构

```
to/local_data/
├── daily/
│   ├── 202301/
│   │   └── daily_data.parquet
│   ├── 202302/
│   │   └── daily_data.parquet
│   └── ...
├── download_progress_tushare.pkl  # 下载进度文件
└── stock_capital_data.parquet     # 股本数据（用于计算市值）
```

## 进度监控

### 日志输出

程序会实时输出进度信息：

```
[批次 1/52] 下载股票 1-100
  [1/100] 下载 600000.SH...
  [2/100] 下载 600004.SH...
  ...
累积数据达到 52340 行，开始保存...
市值和换手率计算完成
  分区 202301: 新增 12345 行
  分区 202302: 新增 12456 行
进度: 100/5000 (2.0%)
```

### 进度文件

`download_progress_tushare.pkl`包含：
- `completed_stocks`: 各分区已完成的股票代码集合 `{partition_key: set of stock_codes}`
- `failed_stocks`: 失败的股票列表
- `start_time`: 开始时间
- `end_time`: 结束时间

**批次去重机制**：
- 使用 `(stock_code, date_str)` 组合进行精确去重
- 支持分批下载，每批数据会自动合并到已有分区
- 第1批、第2批...第N批数据会逐步累积到同一个parquet文件

## 断点续传

### 自动续传

程序中断后，直接重新运行即可：

```bash
python download_daily.py
```

程序会：
1. 加载进度文件
2. 跳过已完成的股票
3. 从上次中断处继续下载
4. 检查已存在的数据，只添加缺失部分

### 禁用断点续传

如果需要重新下载所有数据，禁用断点续传：

```python
success = download_daily_from_tushare(
    stock_list=stock_list,
    start_date=START_DATE,
    end_date=END_DATE,
    resume=False  # 禁用断点续传
)
```

## 性能优化建议

### 1. 调整批处理大小

```python
batch_size=200  # 增加到200只股票/批，减少API调用次数
```

### 2. 调整保存频率

```python
rows_per_save=100000  # 增加到10万行保存一次，减少I/O操作
```

### 3. 并行下载（高级）

需要修改代码支持多线程下载，但要注意API限流。

## 常见问题

### Q1: 下载失败如何重试？

A: 程序会自动记录失败的股票到`failed_stocks`，可以单独重试：

```python
# 重新下载失败的股票
failed_stocks = progress['failed_stocks']
if failed_stocks:
    download_daily_from_tushare(
        stock_list=failed_stocks,
        start_date=START_DATE,
        end_date=END_DATE,
        resume=False
    )
```

### Q2: 如何验证数据完整性？

A: 检查下载的数据：

```python
import pandas as pd
from pathlib import Path

# 检查某个分区的数据
df = pd.read_parquet('to/local_data/daily/202301/daily_data.parquet')
print(f"股票数量: {df['stock_code'].nunique()}")
print(f"数据行数: {len(df)}")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
print(f"字段: {list(df.columns)}")
```

### Q3: 内存不足怎么办？

A: 调整参数：

```python
batch_size=50         # 减少批处理大小
rows_per_save=20000   # 减少保存前的累积行数
```

### Q4: API限流怎么办？

A: 增加请求间隔，修改代码中的sleep时间：

```python
# 在download_daily_from_tushare函数中
time.sleep(0.2)  # 从0.1秒增加到0.2秒
```

## 技术细节

### 数据流程

```
1. 获取股票列表
   └─ 从sw_sector.json获取
   └─ 过滤北交所(.BJ)
   └─ 过滤ST股票

2. 分批下载
   └─ 每批100只股票
   └─ 调用ts.pro_bar()
   └─ 字段映射与转换
   └─ 累积到内存

3. 增量保存
   └─ 累积到50000行
   └─ 计算市值和换手率
   └─ 按月分区存储
   └─ 使用 (stock_code, date_str) 组合去重
   └─ 释放内存

4. 断点续传
   └─ 保存进度到pkl文件
   └─ 下次启动时恢复
```

### 与download_daily_data的对比

| 特性 | download_daily_data | download_daily_from_tushare |
|-----|---------------------|----------------------------|
| 数据源 | 迅投xtdata | Tushare Pro |
| API限流 | 无明显限制 | 200次/分钟（需控制频率） |
| 复权处理 | 需要自己下载 | adj参数控制 |
| pct_chg | 自己计算 | API直接返回 |
| 断点续传 | 不支持 | 支持 |
| 内存优化 | 一次性加载 | 分批处理 |
| 适用场景 | 大批量快速下载 | 稳定可靠下载 |

## 日志文件

下载日志保存在：`to/log/download_tushare_daily.log`

查看日志：
```bash
tail -f to/log/download_tushare_daily.log
```

## 注意事项

1. **Tushare API限制**：
   - 普通用户：200次/分钟
   - 需要控制请求频率，已设置0.1秒间隔

2. **磁盘空间**：
   - 5000只股票×3年约275万行
   - 预计需要约5-10GB磁盘空间

3. **网络稳定性**：
   - 下载过程中断网：重新运行即可恢复
   - 部分股票失败：检查failed_stocks列表

4. **数据质量**：
   - 下载完成后建议抽查数据
   - 检查字段完整性
   - 验证市值和换手率计算

## 更新日志

- **v1.1** (2025-01-04): 修复批次去重BUG
  - 修复了分批下载时，第2批及后续批次数据被错误跳过的严重BUG
  - 改用 `(stock_code, date_str)` 组合进行精确去重
  - 进度跟踪从 `completed_dates` 改为 `completed_stocks`
  - 确保每批数据都能正确合并到parquet文件

- **v1.0** (2025-01-04): 初始版本
  - 支持从Tushare下载日线数据
  - 支持分批处理和断点续传
  - 支持内存优化
  - 支持按月分区存储
  - 支持市值和换手率计算
