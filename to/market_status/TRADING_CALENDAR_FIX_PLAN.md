# 交易日历回溯修复方案

**问题分析时间**: 2026-01-30
**文件**: `to/market_status/market_status_processor.py`
**目标**: 使用交易日历准确回溯180个**交易日**，而非180个**自然日**

---

## 📋 问题分析

### 当前代码问题

**位置**: `market_status_processor.py` 第291-300行

```python
# 计算历史数据开始日期 (需要至少120天 + 一些缓冲)
required_history_days = 180
try:
    request_start_date = self._str_to_date(start_date_str)
    history_start_date = request_start_date - timedelta(days=required_history_days)  # ❌ 问题：自然日
    history_start_date_str = self._date_to_str(history_start_date)
except (ValueError, TypeError):
     self.logger.warning(f"无法解析请求的开始日期 {start_date_str}，将使用默认历史回溯期")
     history_start_date_str = (datetime.now() - timedelta(days=required_history_days*2)).strftime('%Y%m%d')
```

### 问题根源

**错误1**: 使用 `timedelta(days=180)` 回溯自然日
- 自然日 ≠ 交易日
- 180个自然日 ≈ 130个交易日（考虑周末和节假日）
- 导致历史数据不足

**错误2**: `history_start_date` 可能不是交易日
- 如果 `history_start_date` 是周末/节假日
- xtdata 可能返回空数据或填充数据
- 导致 `start_date_str` 的 `prev_close` 缺失

**错误3**: 导致第一个交易日涨跌幅为0
- `start_date_str` 在数据中是第一条记录
- `prev_close = close_series.iloc[i-1]` 其中 `i=0`
- 由于没有前一交易日数据，`daily_change_pct = 0`

### 具体影响

**场景示例**:
```
请求: start_date_str = 20221205
当前逻辑:
  1. request_start_date = 2022-12-05
  2. history_start_date = 2022-12-05 - 180天 = 2022-06-09
  3. history_start_date_str = 20220609
  4. xtdata.get_market_data(start_time="20220609", end_time="20221205")

问题:
  - 2022-06-09 可能不是交易日
  - 实际第一个交易日可能是 2022-06-10 或更晚
  - 导致 2022-06-10 的 prev_close 缺失（因为 2022-06-09 无数据）
  - 涨跌幅计算为 0
```

---

## 🎯 修复方案

### 方案概述

使用 `trading_calendar.parquet` 准确回溯180个**交易日**：

```
start_date_str (20221205)
    ↓
在交易日历中找到 20221205 的索引
    ↓
回溯 180 个交易日
    ↓
获取第 (index - 180) 个交易日的日期字符串
    ↓
使用该日期作为 history_start_date_str
```

### 交易日历文件信息

**路径**: `to/local_data/calendar/trading_calendar.parquet`

**结构**:
```
   date_str       date market
0  20220705 2022-07-05     SH
1  20220706 2022-07-06     SH
2  20220707 2022-07-07     SH
...
```

**字段**:
- `date_str`: YYYYMMDD 格式的日期字符串
- `date`: datetime 类型的日期
- `market`: 市场标识（SH/SZ）

**范围**: 2022-07-05 ~ 2026-01-26（866个交易日）

### 实现步骤

#### 步骤1: 添加辅助方法加载交易日历

**位置**: 在 `MarketStatusProcessor` 类中添加新方法

```python
def _load_trading_calendar(self):
    """
    加载交易日历

    返回:
        pd.DataFrame: 交易日历，包含 date_str, date, market 列
        None: 加载失败
    """
    try:
        import pandas as pd
        from pathlib import Path

        # 构建交易日历文件路径
        # 当前文件: to/market_status/market_status_processor.py
        # 交易日历: to/local_data/calendar/trading_calendar.parquet
        current_dir = Path(__file__).resolve().parent  # to/market_status/
        project_root = current_dir.parent.parent  # path/
        calendar_file = project_root / "to" / "local_data" / "calendar" / "trading_calendar.parquet"

        self.logger.debug(f"加载交易日历: {calendar_file}")

        if not calendar_file.exists():
            self.logger.error(f"交易日历文件不存在: {calendar_file}")
            return None

        # 读取交易日历
        trading_calendar = pd.read_parquet(calendar_file)

        # 验证必要列
        if 'date_str' not in trading_calendar.columns:
            self.logger.error("交易日历缺少 date_str 列")
            return None

        self.logger.info(f"成功加载交易日历: {len(trading_calendar)} 个交易日")
        self.logger.debug(f"交易日历范围: {trading_calendar['date_str'].min()} ~ {trading_calendar['date_str'].max()}")

        return trading_calendar

    except Exception as e:
        self.logger.error(f"加载交易日历失败: {str(e)}")
        return None
```

#### 步骤2: 添加辅助方法回溯交易日

```python
def _get_history_start_date_by_trading_days(self, start_date_str, trading_days_back=180):
    """
    根据交易日历回溯指定数量的交易日

    参数:
        start_date_str: str, 开始日期字符串 (YYYYMMDD)
        trading_days_back: int, 回溯的交易日数量，默认180

    返回:
        str: 回溯后的日期字符串 (YYYYMMDD)
        None: 失败或交易日历不足
    """
    try:
        # 加载交易日历
        trading_calendar = self._load_trading_calendar()
        if trading_calendar is None:
            self.logger.warning("无法加载交易日历，使用自然日回溯（备用方案）")
            return None

        # 找到 start_date_str 在交易日历中的位置
        matching_rows = trading_calendar[trading_calendar['date_str'] == start_date_str]

        if matching_rows.empty:
            self.logger.warning(f"开始日期 {start_date_str} 不在交易日历中")
            # 尝试找到最接近的交易日
            # 将 start_date_str 转换为 datetime
            from datetime import datetime
            start_date_dt = datetime.strptime(start_date_str, '%Y%m%d')

            # 找到最接近的交易日（之后或等于）
            future_dates = trading_calendar[trading_calendar['date'] >= start_date_dt]
            if not future_dates.empty:
                closest_date_str = future_dates.iloc[0]['date_str']
                self.logger.info(f"开始日期 {start_date_str} 不是交易日，使用最接近的交易日 {closest_date_str}")
                start_date_str = closest_date_str
                matching_rows = trading_calendar[trading_calendar['date_str'] == start_date_str]
            else:
                self.logger.error(f"无法找到接近 {start_date_str} 的交易日")
                return None

        # 获取索引
        start_index = matching_rows.index[0]

        # 回溯交易日
        history_index = start_index - trading_days_back

        # 检查边界
        if history_index < 0:
            self.logger.warning(
                f"交易日历不足 {trading_days_back} 个交易日 "
                f"(可用: {start_index + 1} 个)，将使用最早交易日"
            )
            history_index = 0

        # 获取历史开始日期
        history_start_date_str = trading_calendar.iloc[history_index]['date_str']

        self.logger.info(
            f"交易日历回溯: {start_date_str} (索引 {start_index}) "
            f"-> {history_start_date_str} (索引 {history_index}), "
            f"回溯 {start_index - history_index} 个交易日"
        )

        return history_start_date_str

    except Exception as e:
        self.logger.error(f"根据交易日历回溯日期失败: {str(e)}")
        return None
```

#### 步骤3: 修改 `_get_market_data` 方法

**修改位置**: 第291-300行

**原代码**:
```python
# 计算历史数据开始日期 (需要至少120天 + 一些缓冲)
required_history_days = 180
try:
    request_start_date = self._str_to_date(start_date_str)
    history_start_date = request_start_date - timedelta(days=required_history_days)
    history_start_date_str = self._date_to_str(history_start_date)
except (ValueError, TypeError):
     self.logger.warning(f"无法解析请求的开始日期 {start_date_str}，将使用默认历史回溯期")
     history_start_date_str = (datetime.now() - timedelta(days=required_history_days*2)).strftime('%Y%m%d')
```

**修改后**:
```python
# 计算历史数据开始日期 (需要至少120天 + 一些缓冲)
required_history_days = 180
try:
    # ✅ 修复：使用交易日历回溯，而非自然日
    history_start_date_str = self._get_history_start_date_by_trading_days(
        start_date_str,
        trading_days_back=required_history_days
    )

    # 如果交易日历方法失败，回退到自然日回溯（备用方案）
    if history_start_date_str is None:
        self.logger.warning("交易日历回溯失败，使用自然日回溯（备用方案）")
        request_start_date = self._str_to_date(start_date_str)
        history_start_date = request_start_date - timedelta(days=required_history_days)
        history_start_date_str = self._date_to_str(history_start_date)

except (ValueError, TypeError):
     self.logger.warning(f"无法解析请求的开始日期 {start_date_str}，将使用默认历史回溯期")
     # 备用方案：使用自然日回溯
     history_start_date_str = (datetime.now() - timedelta(days=required_history_days*2)).strftime('%Y%m%d')
```

---

## 🏗️ 代码层级问题分析

### 项目结构

```
path/
  to/
    local_data/
      calendar/
        trading_calendar.parquet  ← 交易日历文件
    market_status/
      market_status_processor.py  ← 当前文件
      market_analyzer.py
    integrated_stock_selection/
      main.py
```

### 路径构建策略

**当前文件位置**: `to/market_status/market_status_processor.py`

**目标文件位置**: `to/local_data/calendar/trading_calendar.parquet`

**路径构建方法**:
```python
current_dir = Path(__file__).resolve().parent  # to/market_status/
project_root = current_dir.parent.parent  # path/
calendar_file = project_root / "to" / "local_data" / "calendar" / "trading_calendar.parquet"
```

### 导入问题

**潜在问题**: 导入循环或相对导入失败

**解决方案**:
1. 在方法内部导入 `pandas`（延迟导入）
2. 使用绝对路径构建，不依赖导入
3. 添加异常处理，容错处理

```python
def _load_trading_calendar(self):
    try:
        import pandas as pd  # ← 延迟导入，避免循环依赖
        from pathlib import Path

        # 使用绝对路径构建
        ...
```

---

## ⚠️ 边界条件和异常处理

### 场景1:交易日历不足

**问题**: 如果 `start_date_str` 是 20220705（交易日历的第一天），回溯180个交易日会超出范围

**处理**:
```python
if history_index < 0:
    self.logger.warning(
        f"交易日历不足 {trading_days_back} 个交易日 "
        f"(可用: {start_index + 1} 个)，将使用最早交易日"
    )
    history_index = 0  # 使用最早交易日
```

### 场景2: start_date_str 不在交易日历中

**问题**: `start_date_str` 可能是周末或节假日

**处理**:
```python
if matching_rows.empty:
    # 找到最接近的交易日（之后或等于）
    start_date_dt = datetime.strptime(start_date_str, '%Y%m%d')
    future_dates = trading_calendar[trading_calendar['date'] >= start_date_dt]

    if not future_dates.empty:
        closest_date_str = future_dates.iloc[0]['date_str']
        self.logger.info(f"开始日期 {start_date_str} 不是交易日，使用 {closest_date_str}")
        start_date_str = closest_date_str
```

### 场景3: 交易日历文件不存在

**问题**: `trading_calendar.parquet` 文件缺失

**处理**:
```python
if not calendar_file.exists():
    self.logger.error(f"交易日历文件不存在: {calendar_file}")
    return None  # 返回 None，触发备用方案
```

**备用方案**: 如果交易日历加载失败，自动回退到自然日回溯

```python
history_start_date_str = self._get_history_start_date_by_trading_days(...)

if history_start_date_str is None:
    # 使用自然日回溯作为备用方案
    self.logger.warning("交易日历回溯失败，使用自然日回溯")
    ...
```

### 场景4: Pandas 导入失败

**问题**: pandas 未安装或导入失败

**处理**:
```python
try:
    import pandas as pd
    ...
except ImportError as e:
    self.logger.error(f"无法导入 pandas: {e}")
    return None
```

---

## 📊 预期效果

### 修复前

```
请求: start_date_str = "20221205"
当前逻辑:
  history_start_date = 20221205 - 180自然日 = 20220609

xtdata.get_market_data(start_time="20220609", ...)

问题:
  - 2022-06-09 可能不是交易日
  - 第一个交易日的 prev_close 缺失
  - daily_change_pct = 0
```

### 修复后

```
请求: start_date_str = "20221205"
新逻辑:
  1. 在交易日历中找到 20221205 的索引 = 150
  2. 回溯 180 个交易日: 150 - 180 = -30
  3. 边界检查: -30 < 0，使用索引 0
  4. history_start_date_str = 交易日历[0]['date_str'] = "20220705"

xtdata.get_market_data(start_time="20220705", ...)

结果:
  - history_start_date 保证是交易日
  - start_date 的 prev_close 存在
  - daily_change_pct 正确计算
```

### 性能影响

**额外开销**:
- 读取 parquet 文件: ~10ms（866行数据）
- 查找索引: ~1ms
- **总开销**: ~11ms（可忽略）

**优化建议**（可选）:
- 缓存交易日历到内存（类属性）
- 只在第一次调用时加载，后续复用

```python
def __init__(self, ...):
    ...
    self._trading_calendar_cache = None  # 缓存交易日历

def _load_trading_calendar(self):
    if self._trading_calendar_cache is not None:
        return self._trading_calendar_cache

    trading_calendar = ...  # 加载逻辑
    self._trading_calendar_cache = trading_calendar
    return trading_calendar
```

---

## ✅ 验证清单

### 功能验证

- [ ] 交易日历正确加载
- [ ] `start_date_str` 在交易日历中，正确回溯180个交易日
- [ ] `start_date_str` 不在交易日历中，找到最接近的交易日
- [ ] 交易日历不足180个，使用最早交易日
- [ ] 交易日历文件不存在，回退到自然日回溯
- [ ] 第一个交易日的 `daily_change_pct` 不再为0

### 日志验证

**应该看到的日志**:
```
[INFO] 成功加载交易日历: 866 个交易日
[INFO] 交易日历范围: 20220705 ~ 20260126
[INFO] 交易日历回溯: 20221205 (索引 150) -> 20220615 (索引 -30), 回溯 180 个交易日
[INFO] 交易日历不足 180 个交易日 (可用: 120 个)，将使用最早交易日
```

**不应该出现的日志**:
```
[ERROR] 交易日历文件不存在
[ERROR] 交易日历缺少 date_str 列
```

### 数据验证

```python
# 验证第一个交易日的涨跌幅
for date_str, data in market_status.items():
    daily_change = data['market_data']['daily_change_pct']
    if daily_change == 0.0:
        print(f"❌ 日期 {date_str} 涨跌幅仍为0")
    else:
        print(f"✅ 日期 {date_str} 涨跌幅: {daily_change}%")
```

---

## 🚀 实施步骤

### 步骤1: 添加 `_load_trading_calendar` 方法
- 位置：`MarketStatusProcessor` 类中
- 功能：加载交易日历文件

### 步骤2: 添加 `_get_history_start_date_by_trading_days` 方法
- 位置：`MarketStatusProcessor` 类中
- 功能：根据交易日历回溯

### 步骤3: 修改 `_get_market_data` 方法
- 位置：第291-300行
- 修改：使用新方法计算 `history_start_date_str`
- 保留：自然日回溯作为备用方案

### 步骤4: 测试
- 运行 `market_status_processor.py`
- 检查日志输出
- 验证第一个交易日的涨跌幅

### 步骤5: （可选）性能优化
- 添加交易日历缓存
- 避免重复读取文件

---

## 🔄 回滚方案

如果修改后出现问题：

**快速回滚**:
1. 恢复第291-300行的原始代码
2. 注释掉新方法调用
3. 使用自然日回溯

**验证点**:
- 检查交易日历文件路径是否正确
- 检查交易日历文件是否存在
- 检查 pandas 是否可用

---

**方案完成时间**: 2026-01-30
**预计影响**: 修复第一个交易日涨跌幅为0的问题
**风险等级**: 低（保留自然日回溯作为备用方案）
