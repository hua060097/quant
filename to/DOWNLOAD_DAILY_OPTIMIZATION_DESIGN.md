# download_daily.py 优化方案设计

## 📋 文档信息
- **创建日期**: 2025-01-11
- **目标**: 优化 download_daily.py 的下载速度
- **原则**: 只设计，不修改原文件

---

## 🔍 问题回顾

### 当前性能瓶颈
1. **逐股票串行下载**（最严重）
   - 5000只股票 × 0.3秒/次 = 25分钟
   - 占总耗时 80-90%

2. **重复读取parquet文件**
   - 每批次都读取全部历史数据
   - 影响较小（5-10%）

3. **低效去重和merge**
   - 全量数据操作
   - 影响较小（5%）

---

## 🧪 测试方案

### 测试脚本：test_tushare_batch_download.py

已创建测试脚本，验证 `ts.pro_bar` 是否支持批量下载：

```bash
# 运行测试
python test_tushare_batch_download.py
```

**测试内容**：
1. ✅ 单个股票下载（基准）
2. ✅ 多个股票用逗号分隔（`"600000.SH,000001.SZ"`）
3. ✅ 多个股票用列表（`["600000.SH", "000001.SZ"]`）
4. ✅ 性能对比（逐个 vs 批量）
5. ✅ 大批量测试（10只股票）

**预期结果**：
- **如果支持批量**：优先使用批量下载（方案1）
- **如果不支持**：使用并发下载（方案2-5）

---

## 💡 优化方案设计

### 方案1：批量API下载（如果支持）⭐⭐⭐⭐⭐

#### 适用条件
```python
# 测试脚本返回
test2['success'] == True and test2['stocks'] > 1
```

#### 核心思路
**从逐股票下载改为批量下载**

```python
# 当前代码（download_daily.py 第138-166行）
for stock in batch_stocks:
    df = ts.pro_bar(ts_code=stock, ...)  # 5000次API调用

# 优化后
df = ts.pro_bar(ts_code=','.join(batch_stocks), ...)  # 50次API调用
```

#### 实现细节

**步骤1：确定最佳批次大小**
```python
def find_optimal_batch_size():
    """测试不同批次大小的性能，找到最优值"""

    test_sizes = [10, 50, 100, 200, 500]
    results = {}

    for size in test_sizes:
        stocks = stock_list[:size]
        start = time.time()

        df = ts.pro_bar(
            ts_code=','.join(stocks),
            start_date='20250101',
            end_date='20250110'
        )

        elapsed = time.time() - start
        results[size] = {
            'time': elapsed,
            'per_stock': elapsed / size
        }

    # 选择每只股票平均耗时最少的批次大小
    best_size = min(results.items(), key=lambda x: x[1]['per_stock'])
    return best_size[0]
```

**建议值**：
- 保守：50只/批（确保稳定性）
- 激进：100-200只/批（如果API稳定）
- 避免超过500只/批（可能触发限流）

**步骤2：修改下载循环**
```python
def download_daily_optimized_batch(stock_list, start_date, end_date, batch_size=100):
    """使用批量API下载"""

    accumulated_data = []
    total_rows = 0

    for i in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"[批次 {batch_num}] 下载 {len(batch_stocks)} 只股票...")

        try:
            # 一次性下载整批
            df = ts.pro_bar(
                ts_code=','.join(batch_stocks),  # ← 批量下载
                adj='qfq',
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                # 按股票分组处理
                for stock_code in batch_stocks:
                    stock_df = df[df['ts_code'] == stock_code].copy()

                    if len(stock_df) > 0:
                        # 转换格式
                        stock_df = _transform_tushare_data(stock_df, stock_code)
                        accumulated_data.append(stock_df)
                        total_rows += len(stock_df)
                    else:
                        logger.warning(f"  {stock_code} 无数据")

                logger.info(f"  批次完成，获取 {len(df)} 行数据")

                # 检查是否需要保存
                if total_rows >= rows_per_save or (i + batch_size) >= len(stock_list):
                    all_data_df = pd.concat(accumulated_data, ignore_index=True)
                    all_data_df = _calculate_derived_fields(all_data_df, capital_df)
                    _save_by_partition(all_data_df, progress['completed_stocks'])

                    # 清理内存
                    del all_data_df
                    accumulated_data = []
                    total_rows = 0

        except Exception as e:
            logger.error(f"批次下载失败: {e}")
            # 降级到逐股票下载
            for stock in batch_stocks:
                try:
                    df = ts.pro_bar(ts_code=stock, ...)
                    # ... 单独处理
                except:
                    pass
```

**步骤3：错误处理和降级**
```python
def download_with_fallback(stock_list, start_date, end_date):
    """批量下载失败时自动降级到逐股票下载"""

    # 尝试批量下载
    try:
        df = ts.pro_bar(ts_code=','.join(stock_list), ...)
        return df, 'batch'
    except Exception as e:
        logger.warning(f"批量下载失败: {e}，降级到逐股票下载")

        # 降级到逐股票下载
        results = []
        for stock in stock_list:
            try:
                df = ts.pro_bar(ts_code=stock, ...)
                results.append(df)
            except:
                pass

        if results:
            return pd.concat(results, ignore_index=True), 'individual'
        else:
            return None, 'failed'
```

#### 预期效果

**性能提升**：
- API调用次数：5000次 → 50次（100倍减少）
- 预计耗时：25分钟 → 3-5分钟（5-8倍提升）
- 网络开销大幅降低

**优势**：
- ✅ 代码改动小（只需修改下载部分）
- ✅ 不增加复杂度
- ✅ 更少API调用 = 更少限流风险
- ✅ 代码更简洁

**劣势**：
- ⚠️ 依赖Tushare API支持批量下载
- ⚠️ 如果API不支持，此方案无效

---

### 方案2：多线程并发下载（通用方案）⭐⭐⭐⭐

#### 适用条件
```python
# 测试脚本返回
test2['success'] == False or test2['stocks'] <= 1
```

#### 核心思路
**使用ThreadPoolExecutor并发下载多只股票**

```python
# 当前代码（串行）
for stock in batch_stocks:
    df = ts.pro_bar(ts_code=stock, ...)  # 串行

# 优化后（并发）
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_stock, stock): stock for stock in batch_stocks}
    results = [f.result() for f in as_completed(futures)]
```

#### 实现细节

**步骤1：定义下载函数**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def download_single_stock(stock_code: str, start_date: str, end_date: str, adj: str = 'qfq'):
    """
    下载单只股票数据

    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        adj: 复权类型

    Returns:
        tuple: (stock_code, DataFrame or None)
    """
    try:
        df = ts.pro_bar(
            ts_code=stock_code,
            adj=adj,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) > 0:
            # 转换格式
            df = _transform_tushare_data(df, stock_code)
            return (stock_code, df)
        else:
            return (stock_code, None)

    except Exception as e:
        logger.error(f"{stock_code} 下载失败: {e}")
        return (stock_code, None)


def download_batch_concurrent(stock_list: List[str], start_date: str, end_date: str,
                              adj: str = 'qfq', max_workers: int = 10) -> pd.DataFrame:
    """
    并发下载一批股票数据

    Args:
        stock_list: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        adj: 复权类型
        max_workers: 最大并发数

    Returns:
        DataFrame: 合并后的数据
    """
    results = []
    failed_stocks = []

    logger.info(f"并发下载 {len(stock_list)} 只股票（并发数={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(download_single_stock, stock, start_date, end_date, adj): stock
            for stock in stock_list
        }

        # 收集结果（按完成顺序）
        completed = 0
        for future in as_completed(future_to_stock):
            stock_code = future_to_stock[future]
            completed += 1

            try:
                stock_code, df = future.result()

                if df is not None:
                    results.append(df)
                    logger.info(f"  [{completed}/{len(stock_list)}] {stock_code} ✅ ({len(df)}行)")
                else:
                    failed_stocks.append(stock_code)
                    logger.warning(f"  [{completed}/{len(stock_list)}] {stock_code} ⚠️ 无数据")

            except Exception as e:
                failed_stocks.append(stock_code)
                logger.error(f"  [{completed}/{len(stock_list)}] {stock_code} ❌ {e}")

    # 合并结果
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        logger.info(f"批次完成: {len(results)}/{len(stock_list)} 只股票成功，{len(failed_stocks)} 只失败")
        return combined_df, failed_stocks
    else:
        return None, failed_stocks
```

**步骤2：集成到主流程**
```python
def download_daily_optimized_concurrent(stock_list, start_date, end_date,
                                       batch_size=500, max_workers=10):
    """使用多线程并发下载"""

    accumulated_data = []
    total_rows = 0

    for i in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"\n[批次 {batch_num}] 并发下载 {len(batch_stocks)} 只股票...")

        # 并发下载当前批次
        batch_df, failed_stocks = download_batch_concurrent(
            stock_list=batch_stocks,
            start_date=start_date,
            end_date=end_date,
            adj='qfq',
            max_workers=max_workers
        )

        if batch_df is not None and len(batch_df) > 0:
            accumulated_data.append(batch_df)
            total_rows += len(batch_df)
            logger.info(f"  批次完成，获取 {len(batch_df)} 行数据")

            # 检查是否需要保存
            if total_rows >= rows_per_save or (i + batch_size) >= len(stock_list):
                all_data_df = pd.concat(accumulated_data, ignore_index=True)
                all_data_df = _calculate_derived_fields(all_data_df, capital_df)
                _save_by_partition(all_data_df, progress['completed_stocks'])

                # 清理内存
                del all_data_df
                accumulated_data = []
                total_rows = 0

        # 记录失败股票
        if failed_stocks:
            progress['failed_stocks'].extend(failed_stocks)

        # 保存进度
        _save_progress(progress, progress_file)
```

**步骤3：线程安全考虑**
```python
# 创建线程锁（如果需要共享资源）
download_lock = threading.Lock()

def download_single_stock_safe(stock_code, start_date, end_date, adj):
    """线程安全的下载函数"""
    try:
        # ... 下载逻辑 ...

        # 如果需要写入共享资源，加锁
        with download_lock:
            # 共享资源操作
            pass

        return (stock_code, df)
    except Exception as e:
        return (stock_code, None)
```

#### 确定最佳并发数

**测试不同并发数的性能**：
```python
def find_optimal_workers():
    """测试不同并发数的性能"""

    test_workers = [1, 5, 10, 20, 50]
    test_stocks = stock_list[:100]  # 测试100只股票

    results = {}

    for workers in test_workers:
        start = time.time()

        df, failed = download_batch_concurrent(
            stock_list=test_stocks,
            start_date='20250101',
            end_date='20250110',
            max_workers=workers
        )

        elapsed = time.time() - start
        success_count = len(test_stocks) - len(failed)

        results[workers] = {
            'time': elapsed,
            'success': success_count,
            'per_stock': elapsed / success_count if success_count > 0 else float('inf')
        }

        logger.info(f"并发数={workers}: {elapsed:.2f}秒, 成功={success_count}/{len(test_stocks)}")

        # 避免限流，等待几秒
        time.sleep(5)

    # 选择最优并发数
    best_workers = min(results.items(), key=lambda x: x[1]['per_stock'])
    logger.info(f"最优并发数: {best_workers[0]}")
    return best_workers[0]
```

**建议值**：
- 保守：5-10线程（稳定，不易限流）
- 平衡：10-20线程（性能与稳定性平衡）
- 激进：20-50线程（高性能，但可能触发限流）

#### 预期效果

**性能提升**：
- 并发数10：25分钟 → 3-5分钟（5-8倍提升）
- 并发数20：25分钟 → 2-3分钟（8-12倍提升）

**优势**：
- ✅ 通用方案，不依赖API支持
- ✅ 灵活调整并发数
- ✅ 代码改动可控

**劣势**：
- ⚠️ 增加代码复杂度
- ⚠️ 需要处理线程安全问题
- ⚠️ 可能触发API限流

---

### 方案3：多进程并发下载（高性能）⭐⭐⭐

#### 核心思路
**使用ProcessPoolExecutor绕过GIL限制**

```python
from concurrent.futures import ProcessPoolExecutor

def download_batch_multiprocess(stock_list, start_date, end_date, max_workers=4):
    """使用多进程并发下载"""

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_stock, stock, start_date, end_date): stock
            for stock in stock_list
        }

        results = [f.result() for f in as_completed(futures)]

    return pd.concat([r[1] for r in results if r[1] is not None])
```

#### 适用场景
- CPU密集型任务（如数据处理）
- 需要绕过Python GIL限制
- 机器有多核CPU

#### 建议
- 对于纯IO任务（网络下载），多线程更合适（方案2）
- 进程开销大，不适合短期任务
- **不推荐用于此场景**

---

### 方案4：异步IO下载（高级）⭐⭐⭐

#### 核心思路
**使用asyncio和aiohttp异步下载**

```python
import asyncio
import aiohttp

async def download_stock_async(session, stock_code, start_date, end_date):
    """异步下载单只股票"""
    url = f"http://api.tushare.pro/?ts_code={stock_code}&start_date={start_date}&end_date={end_date}"

    async with session.get(url) as response:
        data = await response.json()
        return data

async def download_batch_async(stock_list, start_date, end_date):
    """异步批量下载"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_stock_async(session, stock, start_date, end_date)
            for stock in stock_list
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results
```

#### 优势
- ✅ 最高的并发性能（可支持数千并发）
- ✅ 资源占用少
- ✅ 适合高IO场景

#### 劣势
- ❌ Tushare官方SDK不支持异步
- ❌ 需要手动处理API调用
- ❌ 代码复杂度高
- ❌ 不推荐用于此场景

---

### 方案5：混合优化方案（最优）⭐⭐⭐⭐⭐

#### 核心思路
**批量API + 多线程 + 优化保存逻辑**

```python
def download_daily_hybrid_optimization(stock_list, start_date, end_date):
    """混合优化方案"""

    # 1. 如果API支持批量，使用批量下载
    if check_batch_support():
        batch_size = 100  # 每批100只
    else:
        batch_size = 1  # 逐个下载

    # 2. 使用多线程并发下载多批
    max_workers = 10
    accumulated_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将股票列表分成多个批次
        batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]

        # 并发下载多个批次
        futures = {
            executor.submit(download_batch, batch, start_date, end_date): batch
            for batch in batches
        }

        for future in as_completed(futures):
            batch_df = future.result()
            if batch_df is not None:
                accumulated_data.append(batch_df)

                # 定期保存（避免内存占用过大）
                if sum(len(df) for df in accumulated_data) >= 10000:
                    # 合并并保存
                    all_data = pd.concat(accumulated_data, ignore_index=True)
                    _save_by_partition_optimized(all_data)
                    accumulated_data = []

    # 3. 保存剩余数据
    if accumulated_data:
        all_data = pd.concat(accumulated_data, ignore_index=True)
        _save_by_partition_optimized(all_data)
```

#### 优化保存逻辑

**问题**：每次保存都读取完整parquet文件

**解决方案**：维护内存缓存
```python
class PartitionCache:
    """分区数据缓存"""

    def __init__(self):
        self.cache = {}  # {partition_key: DataFrame}

    def append(self, partition_key, df):
        """追加数据到缓存"""
        if partition_key not in self.cache:
            self.cache[partition_key] = []

        self.cache[partition_key].append(df)

    def flush(self):
        """将所有缓存数据写入磁盘（只写一次）"""
        for partition_key, dfs in self.cache.items():
            if dfs:
                # 合并所有该分区的新数据
                new_df = pd.concat(dfs, ignore_index=True)

                # 读取已有数据（只读一次）
                partition_file = Path(f'local_data/daily/{partition_key}/daily_data.parquet')

                if partition_file.exists():
                    existing_df = pd.read_parquet(partition_file)

                    # 去重（只对新数据去重）
                    new_keys = set(zip(new_df['stock_code'], new_df['date_str']))
                    existing_keys = set(zip(existing_df['stock_code'], existing_df['date_str']))

                    only_new_keys = new_keys - existing_keys

                    if only_new_keys:
                        only_new_df = new_df[
                            new_df.set_index(['stock_code', 'date_str']).index.isin(only_new_keys)
                        ]

                        combined_df = pd.concat([existing_df, only_new_df], ignore_index=True)
                        combined_df = combined_df.sort_values(['stock_code', 'date'])
                        combined_df.to_parquet(partition_file, index=False)
                else:
                    # 首次创建
                    new_df = new_df.sort_values(['stock_code', 'date'])
                    partition_file.parent.mkdir(parents=True, exist_ok=True)
                    new_df.to_parquet(partition_file, index=False)

        # 清空缓存
        self.cache = {}
```

#### 预期效果

**综合提升**：
- 批量API：减少100倍API调用
- 多线程：提升10倍并发
- 优化保存：减少50%磁盘IO

**总提升**：25分钟 → 1-2分钟（15-25倍）

---

## 📊 方案对比

| 方案 | 复杂度 | 性能提升 | 稳定性 | 推荐度 |
|------|-------|---------|--------|--------|
| **方案1：批量API** | ⭐ 低 | 5-8倍 | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ |
| **方案2：多线程** | ⭐⭐ 中 | 5-10倍 | ⭐⭐⭐⭐ 中高 | ⭐⭐⭐⭐ |
| **方案3：多进程** | ⭐⭐⭐ 高 | 3-5倍 | ⭐⭐⭐ 中 | ⭐⭐ |
| **方案4：异步IO** | ⭐⭐⭐⭐ 很高 | 10-20倍 | ⭐⭐ 低 | ⭐⭐ |
| **方案5：混合** | ⭐⭐⭐⭐ 高 | 15-25倍 | ⭐⭐⭐ 中高 | ⭐⭐⭐⭐⭐ |

---

## 🎯 推荐实施路线

### 阶段1：验证（必须先做）
```bash
# 运行测试脚本
python test_tushare_batch_download.py
```

**根据测试结果选择方案**：
- ✅ 支持批量 → **方案1**（批量API）
- ❌ 不支持 → **方案2**（多线程）

### 阶段2：基础优化
- 实施选定的基础方案（方案1或方案2）
- 预期提升：5-10倍
- 预期耗时：25分钟 → 3-5分钟

### 阶段3：深度优化（可选）
- 优化保存逻辑（避免重复读取）
- 调整批次参数
- 预期额外提升：2-3倍
- 预期耗时：3-5分钟 → 1-2分钟

### 阶段4：高级优化（可选）
- 混合方案（方案5）
- 预期额外提升：1.5-2倍
- 预期耗时：1-2分钟 → <1分钟

---

## 📝 注意事项

### 1. API限流
- Tushare有每分钟500-1000次的限制
- 使用多线程时注意控制并发数
- 建议添加限流保护：
  ```python
  import time
  from threading import Semaphore

  rate_limiter = Semaphore(10)  # 最多10个并发请求

  def download_with_rate_limit(stock):
      with rate_limiter:
          time.sleep(0.1)  # 每次请求间隔0.1秒
          return ts.pro_bar(ts_code=stock, ...)
  ```

### 2. 内存占用
- 大批次下载会占用大量内存
- 建议：
  - 每批不超过500只股票
  - 每10000行数据保存一次
  - 及时释放内存（`del df`）

### 3. 错误处理
- 网络不稳定时需要重试机制
- API限流时需要等待
- 建议添加：
  ```python
  from tenacity import retry, stop_after_attempt, wait_fixed

  @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
  def download_single_stock_retry(stock):
      return ts.pro_bar(ts_code=stock, ...)
  ```

### 4. 日志记录
- 并发下载时日志会混乱
- 建议每批次记录一次汇总日志
- 详细日志写入文件

---

## 🔧 快速实施指南

### 如果批量API支持（最简单）

**只需修改3处**：

1. 修改下载循环（第138-166行）
2. 调整批次大小（第493行）
3. 测试和调优

**代码量**：约20行
**工作量**：1-2小时
**效果**：5-8倍提升

### 如果批量API不支持（中等复杂）

**需要修改**：

1. 新增 `download_single_stock()` 函数
2. 新增 `download_batch_concurrent()` 函数
3. 修改主下载循环
4. 测试并发数
5. 添加线程安全保护

**代码量**：约100行
**工作量**：3-5小时
**效果**：5-10倍提升

---

## 📚 参考资料

1. **Tushare官方文档**
   - https://tushare.pro/document/2?date=20200101

2. **Python并发编程**
   - https://docs.python.org/3/library/concurrent.futures.html

3. **Pandas性能优化**
   - https://pandas.pydata.org/docs/user_guide/enhancingperf.html

---

**文档创建日期**: 2025-01-11
**最后更新**: 2025-01-11
**状态**: ✅ 设计完成，等待测试结果
