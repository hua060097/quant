# 市值数据更新脚本使用说明

## 📋 功能说明

`update_mv_data.py` 脚本用于将原始数据中的市值信息更新到特征文件中：
- 从 `to/local_data/daily/` 读取 `total_mv`, `circ_mv`, `turnover_rate`
- 更新到 `to/local_data/features/daily/` 对应的特征文件中

## 🚀 使用方法

### 方法1：交互式运行
```bash
python to/update_mv_data.py
```

然后选择：
1. **试运行** (检查但不保存)
2. **正式运行** (实际更新文件)  
3. **指定日期范围运行**

### 方法2：编程调用
```python
from update_mv_data import update_mv_data

# 试运行模式
update_mv_data(dry_run=True)

# 正式更新所有文件
update_mv_data(dry_run=False)

# 更新指定日期范围
update_mv_data(
    start_date='20250723',
    end_date='20250725', 
    dry_run=False
)
```

## 📊 更新效果

### 更新前：
```
total_mv: 全部为 0
circ_mv: 全部为 0  
turnover_rate: 全部为 0
```

### 更新后：
```
total_mv: 5,212个正值 (总市值，单位：元)
circ_mv: 5,212个正值 (流通市值，单位：元)
turnover_rate: 5,205个正值 (换手率，单位：%)
```

## ⚠️ 注意事项

1. **数据类型兼容性**：
   - 脚本会自动处理数据类型转换
   - 可能出现 FutureWarning，但不影响功能

2. **备份建议**：
   - 建议先在少量文件上测试
   - 重要数据建议先备份

3. **性能考虑**：
   - 全量更新620个文件约需10-15分钟
   - 建议分批处理或在空闲时间运行

4. **数据依赖**：
   - 需要原始数据文件存在且包含有效的市值数据
   - 如果原始数据中市值为空，对应记录不会被更新

## 📈 验证更新效果

更新完成后，可以运行以下代码验证：

```python
import pandas as pd
from pathlib import Path

# 检查最新的特征文件
feature_files = list(Path("to/local_data/features/daily").glob("*.parquet"))
latest_file = sorted(feature_files)[-1]

df = pd.read_parquet(latest_file)

# 检查市值数据
print(f"total_mv 正值数量: {(df['total_mv'] > 0).sum()}")
print(f"circ_mv 正值数量: {(df['circ_mv'] > 0).sum()}")  
print(f"turnover_rate 正值数量: {(df['turnover_rate'] > 0).sum()}")

# 显示样本数据
sample = df[['stock_code', 'total_mv', 'circ_mv', 'turnover_rate']].head(3)
print(sample)
```

## 🎯 成功标准

更新成功的标志：
- ✅ 处理文件数 > 0
- ✅ 更新记录数 > 0  
- ✅ 错误数 = 0
- ✅ 特征文件中市值字段不再全为0
- ✅ 数值范围合理（市值>0，换手率0-100%）

## 🔧 故障排除

### 问题1：找不到原始数据文件
**解决**：确保 `to/local_data/daily/YYYYMM/daily_data.parquet` 文件存在

### 问题2：数据类型警告
**解决**：这是正常现象，不影响功能，脚本会自动处理类型转换

### 问题3：更新记录数为0
**解决**：检查原始数据中是否包含有效的市值数据

### 问题4：内存不足
**解决**：分批处理，使用日期范围参数限制处理文件数量

## 📞 技术支持

如遇问题，请检查：
1. 文件路径是否正确
2. 原始数据是否完整
3. 磁盘空间是否充足
4. Python环境是否正常
