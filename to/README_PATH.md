# 板块轮动模型系统路径使用规范

本文档说明项目中路径使用的规范和最佳实践，以避免创建冗余目录和路径混乱问题。

## 问题背景

项目中存在路径混乱问题，特别是在`to/`目录下的文件中使用以`to/`开头的相对路径，导致创建了`to/to/`这样的冗余目录结构。这会造成:

1. 存储空间浪费
2. 代码逻辑混乱
3. 数据访问不一致
4. 维护困难

## 正确的路径使用规范

### 1. 使用`path_utils.py`中提供的路径助手函数

```python
from to.path_utils import get_project_path, get_data_path, get_model_path, get_log_path, get_results_path

# 获取数据目录下的某个文件路径
data_file_path = get_data_path('features/stock_data.parquet')

# 获取模型目录下的某个模型文件
model_file_path = get_model_path('sector_rotation/model_20240101.pkl')

# 获取日志文件路径
log_file_path = get_log_path('training_202404.log')

# 获取结果目录
results_dir = get_results_path()

# 获取项目根目录下的任意路径
config_path = get_project_path('to/config.py')
```

### 2. 路径字符串标准化

如果你需要处理路径字符串，可以使用`normalize_path`函数:

```python
from to.path_utils import normalize_path

# 标准化路径字符串，避免to/to这样的重复目录
safe_path = normalize_path('to/local_data/features')
```

### 3. 确保目录存在

```python
from to.path_utils import ensure_dir

# 确保目录存在，不存在则创建
output_dir = ensure_dir(get_data_path('new_features'))
```

## 要避免的做法

1. **直接使用硬编码的相对路径**:

```python
# 不要这样做，特别是在to/目录下的文件中
output_dir = "to/local_data/features"  # 可能创建to/to/local_data/features
```

2. **混合使用不同风格的路径**:

```python
# 不要在同一个项目中混合使用不同风格的路径
path1 = "to/local_data/features"
path2 = Path(__file__).parent / "local_data" / "features" 
```

3. **忽略操作系统路径分隔符差异**:

```python
# 不要硬编码路径分隔符
path = "to\\local_data\\features"  # 在Linux上会出问题
```

## 现有路径修复措施

对于已经存在的`to/to/`等冗余目录，建议执行以下步骤:

1. 将`to/to/`目录下的内容迁移到正确的位置
2. 删除冗余的`to/to/`目录
3. 修改代码中的路径引用，使用`path_utils.py`中的工具函数

## 自动化检测与修复

`path_utils.py`模块会自动检测并警告可能导致路径混乱的用法，并在日志中记录相关信息。在一些情况下，它还能自动修复路径问题。

## 参考资料

- [Python项目结构最佳实践](https://docs.python-guide.org/writing/structure/)
- [Python中的路径处理](https://docs.python.org/3/library/pathlib.html) 