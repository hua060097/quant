# 修复相对路径问题 - 使用绝对路径方案

## 修改日期
2025-12-28

## 问题描述

### 原始问题
在`generate_daily_features_simple.py`中调用`self.storage.get_trading_calendar`时，获取日历文件的地址会因为调用脚本所在目录不同而不同：
- **错误路径1**：`to/to/local_data/calendar`（从to/目录调用）
- **错误路径2**：`integrated_stock_selection/to/local_data/calendar`（从子目录调用）
- **正确路径**：`C:\Users\86151\.trae-cn\pythons\path\to\local_data\calendar`

### 根本原因
`advanced_quant_data.py`第58行使用了**相对路径**：
```python
DATA_ROOT = Path("to/local_data")  # ❌ 相对路径，依赖CWD
```

相对路径基于**当前工作目录（CWD）**，而非脚本文件所在目录。从不同目录调用脚本时，CWD不同，导致解析出的路径不同。

---

## 解决方案：方案1 - 基于__file__的绝对路径

### 核心思想
使用Python的`__file__`内置变量获取脚本文件的绝对路径，从而定位项目根目录，构建绝对路径。

### 修改文件
`advanced_quant_data.py`

### 修改内容

#### 1. 数据目录定义（第57-65行）

**修改前：**
```python
# 定义数据目录
DATA_ROOT = Path("to/local_data")
DAILY_DIR = DATA_ROOT / "daily"
MINUTE_DIR = DATA_ROOT / "minute"
CALENDAR_DIR = DATA_ROOT / "calendar"
METADATA_DIR = DATA_ROOT / "metadata"
```

**修改后：**
```python
# 定义数据目录 - 使用绝对路径，避免因调用目录不同导致的路径问题
_SCRIPT_DIR = Path(__file__).resolve().parent  # 获取脚本所在目录 (advanced_quant_data.py所在目录)
PROJECT_ROOT = _SCRIPT_DIR.parent  # 获取项目根目录 (to/目录)

DATA_ROOT = PROJECT_ROOT / "local_data"  # ✅ 绝对路径，不依赖CWD
DAILY_DIR = DATA_ROOT / "daily"
MINUTE_DIR = DATA_ROOT / "minute"
CALENDAR_DIR = DATA_ROOT / "calendar"
METADATA_DIR = DATA_ROOT / "metadata"
```

**说明：**
- `_SCRIPT_DIR`: 获取`advanced_quant_data.py`文件所在目录的绝对路径
  - 例如：`C:\Users\86151\.trae-cn\pythons\path\to\`
- `PROJECT_ROOT`: 获取项目根目录（脚本所在目录的父目录）
  - 例如：`C:\Users\86151\.trae-cn\pythens\path\to\`
- `DATA_ROOT`: 基于PROJECT_ROOT构建数据根目录
  - 最终：`C:\Users\86151\.trae-cn\pythons\path\to\local_data` ✅

#### 2. get_data_path函数优化（第73-79行）

**修改前：**
```python
def get_data_path(relative_path=None):
    """获取数据目录路径"""
    project_root = get_project_root()
    print(f"project_root:{project_root}")
    data_path = project_root / "to/local_data"
    print(f"data_path: {data_path}")
    if relative_path:
        return data_path / relative_path
    return data_path
```

**修改后：**
```python
def get_data_path(relative_path=None):
    """获取数据目录路径"""
    # 使用全局的DATA_ROOT，确保路径一致性
    print(f"DATA_ROOT (绝对路径): {DATA_ROOT.absolute()}")
    if relative_path:
        return DATA_ROOT / relative_path
    return DATA_ROOT
```

**说明：**
- 直接使用全局变量`DATA_ROOT`，确保路径一致性
- 输出绝对路径便于调试
- 移除了冗余的`project_root / "to/local_data"`逻辑

---

## 修改效果

### 修改前
```python
# 从 to/ 目录调用
CWD: C:\Users\86151\.trae-cn\pythons\path\to
DATA_ROOT解析为: C:\Users\86151\.trae-cn\pythons\path\to\to\local_data ❌

# 从 integrated_stock_selection/ 目录调用
CWD: C:\Users\86151\.trae-cn\pythens\path\to\integrated_stock_selection
DATA_ROOT解析为: C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\to\local_data ❌
```

### 修改后
```python
# 无论从哪个目录调用
_SCRIPT_DIR: C:\Users\86151\.trae-cn\pythons\path\to\  (脚本所在目录)
PROJECT_ROOT: C:\Users\86151\.trae-cn\pythons\path\to\  (项目根目录)
DATA_ROOT: C:\Users\86151\.trae-cn\pythons\path\to\local_data ✅

# 路径始终一致，不依赖CWD
CALENDAR_DIR: C:\Users\86151\.trae-cn\pythons\path\to\local_data\calendar
```

---

## 技术细节

### __file__ 变量
- `__file__` 是Python的内置变量，表示当前脚本的路径
- `Path(__file__).resolve()` 将其转换为绝对路径，消除符号链接等
- `.parent` 获取父目录

### 路径解析示例
```python
# 假设脚本位置：
# C:\Users\86151\.trae-cn\pythons\path\to\advanced_quant_data.py

__file__ = "advanced_quant_data.py"  # 或相对路径，取决于调用方式
Path(__file__).resolve()
# → C:\Users\86151\.trae-cn\pythons\path\to\advanced_quant_data.py

_SCRIPT_DIR = Path(__file__).resolve().parent
# → C:\Users\86151\.trae-cn\pythons\path\to\

PROJECT_ROOT = _SCRIPT_DIR.parent
# → C:\Users\86151\.trae-cn\pythons\path\to\

DATA_ROOT = PROJECT_ROOT / "local_data"
# → C:\Users\86151\.trae-cn\pythons\path\to\local_data
```

---

## 影响范围

### 直接影响
1. ✅ `get_trading_calendar()` - 交易日历加载
2. ✅ `get_stock_list()` - 股票列表加载
3. ✅ `download_daily_data()` - 日线数据下载
4. ✅ `download_minute_data()` - 分钟数据下载
5. ✅ 所有使用`DATA_ROOT`, `DAILY_DIR`, `MINUTE_DIR`, `CALENDAR_DIR`, `METADATA_DIR`的函数

### 间接受益
- 所有依赖`advanced_quant_data.py`的模块都将受益
- 包括但不限于：
  - `generate_daily_features_simple.py`
  - `generate_daily_features.py`
  - `train_sector_models.py`
  - `main.py`
  - 所有调用`ParquetDataStorage`的脚本

---

## 验证方法

### 1. 检查路径输出
运行任何调用`get_trading_calendar`的脚本，查看日志：
```python
# 修改前可能看到：
address: to\to\local_data\calendar\trading_calendar.parquet  # ❌

# 修改后应该看到：
address: C:\Users\86151\.trae-cn\pythons\path\to\local_data\calendar\trading_calendar.parquet  # ✅
```

### 2. 从不同目录测试
```bash
# 测试1：从 to/ 目录调用
cd C:\Users\86151\.trae-cn\pythons\path\to
python generate_daily_features_simple.py

# 测试2：从子目录调用
cd C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection
python ../generate_daily_features_simple.py

# 测试3：从任意目录调用
cd C:\Users\86151\
python .trae-cn\pythons\path\to\generate_daily_features_simple.py

# 三次测试都应该使用相同的绝对路径 ✅
```

### 3. 检查数据目录是否存在
```python
import os
from pathlib import Path

# 检查正确的数据目录是否存在
correct_path = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\local_data")
print(f"正确的数据目录存在: {correct_path.exists()}")

# 检查错误目录是否被创建（不应该存在）
wrong_path1 = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\to\local_data")
print(f"错误的目录1存在: {wrong_path1.exists()}")  # 应该是False

wrong_path2 = Path(r"C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection\to\local_data")
print(f"错误的目录2存在: {wrong_path2.exists()}")  # 应该是False
```

---

## 注意事项

### 1. 脚本位置敏感性
⚠️ **如果移动`advanced_quant_data.py`文件**，需要同步修改`_SCRIPT_DIR`的计算逻辑：
- 如果移动到子目录，可能需要`.parent.parent.parent`
- 如果移动到其他位置，需要调整层级关系

### 2. 与get_project_root()函数的关系
代码中已有`get_project_root()`函数（第68-71行），其逻辑与新代码一致：
```python
def get_project_root():
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent
```

未来可以考虑统一使用这个函数，避免重复逻辑：
```python
# 可选的进一步优化
PROJECT_ROOT = get_project_root()
DATA_ROOT = PROJECT_ROOT / "local_data"
```

### 3. 调试信息
修改后的`get_data_path`函数会输出绝对路径：
```python
print(f"DATA_ROOT (绝对路径): {DATA_ROOT.absolute()}")
```
如果需要静默运行，可以移除这行或改用logger.debug()。

---

## 后续优化建议

### 短期（已完成）
- ✅ 修复相对路径问题
- ✅ 统一使用绝对路径
- ✅ 确保所有路径基于脚本位置

### 中期（可选）
1. **统一路径函数**
   - 将所有路径获取逻辑集中到`get_project_root()`和相关函数
   - 避免散落的路径计算代码

2. **添加路径验证**
   ```python
   def validate_data_paths():
       """验证关键路径是否存在"""
       required_paths = [DATA_ROOT, CALENDAR_DIR, DAILY_DIR]
       for path in required_paths:
           if not path.exists():
               logger.warning(f"路径不存在: {path}")
   ```

3. **移除调试打印**
   - 将`get_data_path`中的`print`改为`logger.debug()`

### 长期（可选）
1. **配置文件支持**
   - 参考"方案3"，将路径配置移到配置文件
   - 支持多环境配置

2. **环境变量覆盖**
   - 参考"方案5"，支持环境变量覆盖默认路径
   - 便于Docker部署和多环境管理

---

## 总结

### 修改前
- ❌ 使用相对路径`Path("to/local_data")`
- ❌ 路径依赖当前工作目录（CWD）
- ❌ 从不同目录调用产生不同路径
- ❌ 可能创建多个错误的数据目录

### 修改后
- ✅ 使用绝对路径，基于`__file__`定位
- ✅ 路径独立于当前工作目录
- ✅ 无论从哪调用都使用相同路径
- ✅ 始终使用正确的数据目录

### 核心优势
1. **一致性** - 所有调用使用相同的绝对路径
2. **可靠性** - 不依赖CWD，避免路径混乱
3. **可维护性** - 逻辑清晰，易于理解
4. **兼容性** - 对现有代码影响最小，无需修改调用方

---

## 相关文件
- `advanced_quant_data.py` - 主要修改文件
- `generate_daily_features_simple.py` - 问题发现文件
- `docs/ABSOLUTE_PATH_FIX_SUMMARY.md` - 本文档

## 参考资料
- Python `__file__` 文档：https://docs.python.org/3/reference/import.html#__file__
- `pathlib.Path` 文档：https://docs.python.org/3/library/pathlib.html
