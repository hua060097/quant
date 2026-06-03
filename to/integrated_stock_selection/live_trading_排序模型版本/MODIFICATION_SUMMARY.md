# live_trading select模式修改说明

## 修改概述

将 `live_trading` 的 `select` 模式改为与 `main.py` 中 `pred_stocks` 完全一致的逻辑。

**主要变更**:
1. ✅ 使用 **LightGBM单模型** 替代ensemble模型
2. ✅ 数据加载 **复用main.py的_load_factor_data_for_prediction函数**
3. ✅ 过滤逻辑 **与main.py pred_stocks完全一致**
4. ✅ 选股结果 **保持保存到 `live_trading/stock_selections.json`**

---

## 详细修改内容

### 1. 修改文件清单

| 文件 | 修改内容 | 变更类型 |
|------|----------|----------|
| `live_trading/model_loader.py` | 添加 `load_lightgbm_model()` 方法 | 新增 |
| `live_trading/main.py` | 修改 `load_factor_data()` 函数，复用main.py逻辑 | 重写 |
| `live_trading/main.py` | 修改 `run_select_mode()` 函数，使用LightGBM单模型 | 修改 |
| `live_trading/stock_selector.py` | 修改 `_apply_filters()` 过滤逻辑 | 重写 |
| `live_trading/stock_selector.py` | 修改 `_predict_stocks()` 预测逻辑 | 重写 |

---

### 2. 修改详情

#### 2.1 `live_trading/model_loader.py`

**新增方法**:
```python
def load_lightgbm_model(self, model_path: Optional[Path] = None) -> bool:
    """
    加载单个LightGBM模型（与main.py pred_stocks一致）

    Args:
        model_path: 模型文件路径，如果为None则自动查找最新的LightGBM模型

    Returns:
        bool: 是否加载成功
    """
```

**功能**:
- 自动查找最新的 `lightgbm_model_*.pkl` 文件
- 加载模型、scaler、label_transformer、feature_names等
- 与main.py的模型加载逻辑完全一致

---

#### 2.2 `live_trading/main.py`

**修改前**:
```python
def load_factor_data():
    # 直接从 factors.parquet 加载
    factor_file = PathConfig.PROJECT_ROOT / "factor_system" / "factor_lab" / "store" / "factors.parquet"
    data = pd.read_parquet(factor_file)
    return data
```

**修改后**:
```python
def load_factor_data(date: str):
    """
    加载因子数据（复用main.py pred_stocks的逻辑）

    Args:
        date: 日期字符串 (YYYYMMDD)

    Returns:
        pd.DataFrame: 包含因子和基础行情的数据
    """
    # 动态导入main.py的_load_factor_data_for_prediction函数
    from main import _load_factor_data_for_prediction

    # 使用main.py的函数加载数据
    factor_data = _load_factor_data_for_prediction(date)

    return factor_data
```

**关键变化**:
1. ✅ 添加 `date` 参数
2. ✅ 导入并使用 `main._load_factor_data_for_prediction()`
3. ✅ 同时加载因子数据和基础行情数据（is_limit_up, is_limit_down, amount, list_date等）
4. ✅ 正确处理相对路径问题

---

**修改前**:
```python
def run_select_mode(args):
    # 加载ensemble模型
    model_loader = ModelLoader()
    if not model_loader.load_latest_ensemble_models():
        logger.error("模型加载失败")
        return False

    # 加载因子数据
    factor_data = load_factor_data()  # 无参数
```

**修改后**:
```python
def run_select_mode(args):
    # 加载LightGBM单模型
    model_loader = ModelLoader()
    if not model_loader.load_lightgbm_model():  # 使用新方法
        logger.error("模型加载失败")
        return False

    # 加载因子数据（传入date参数）
    factor_data = load_factor_data(args.date)  # 传入date

    # 显示结果时添加SW行业信息
    from main import load_sw_sector_mapping
    stock_to_industry = load_sw_sector_mapping()
```

---

#### 2.3 `live_trading/stock_selector.py`

**修改前** (过滤逻辑):
```python
def _apply_filters(self, data: pd.DataFrame, date: str) -> pd.DataFrame:
    # 1. 北交所过滤
    # 2. 新股过滤 (60天)
    # 3. ST股过滤
    # 4. 涨停股过滤 (pct_chg < 9.9%)
```

**修改后** (与main.py一致):
```python
def _apply_filters(self, data: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    应用过滤条件（与main.py pred_stocks完全一致）

    过滤规则:
    1. ST股票
    2. T日涨停股票 (is_limit_up == 1)
    3. T日跌停股票 (is_limit_down == 1)
    4. 新股 (上市不足30天)
    5. 流动性不足 (amount < 1000万)
    """
    # 1. ST股票过滤
    # 2. 涨停股过滤 (使用is_limit_up字段)
    # 3. 跌停股过滤 (使用is_limit_down字段) ✨新增
    # 4. 新股过滤 (阈值改为30天)
    # 5. 流动性过滤 (amount < 1000万) ✨新增
```

**关键变化**:
1. ✅ **移除** 北交所过滤
2. ✅ **改用** `is_limit_up` 字段（而不是 pct_chg）
3. ✅ **新增** 跌停过滤 (`is_limit_down`)
4. ✅ 新股阈值从60天改为 **30天**
5. ✅ **新增** 流动性过滤 (`amount < 1000万`)

---

**修改前** (预测逻辑):
```python
def _predict_stocks(self, data: pd.DataFrame) -> Optional[np.ndarray]:
    # 使用ensemble模型（多个XGBoost排序模型）
    # 基于RankIC的加权预测
    # 置信度过滤
    # 风险调整
```

**修改后** (LightGBM单模型):
```python
def _predict_stocks(self, data: pd.DataFrame) -> Optional[np.ndarray]:
    """
    模型预测（使用LightGBM单模型，与main.py pred_stocks一致）

    Returns:
        np.ndarray: 预测分数（ATR单位）
    """
    # 1. 加载LightGBM单模型
    model = self.model_loader.models[0]

    # 2. 特征对齐
    X = self._align_features(data, feature_names)

    # 3. 标准化
    X_scaled = scaler.transform(X)

    # 4. LightGBM预测
    y_pred_raw = model.predict(X_scaled, num_iteration=model.best_iteration)

    # 5. 逆变换
    if label_transformer is not None:
        y_pred = label_transformer.inverse_transform(y_pred_raw.reshape(-1, 1)).flatten()

    return y_pred
```

**关键变化**:
1. ✅ 使用 **LightGBM单模型**（不再是ensemble）
2. ✅ 移除RankIC加权逻辑
3. ✅ 移除置信度过滤
4. ✅ 移除风险调整
5. ✅ 直接返回预测值

---

## 路径处理

### 目录结构
```
integrated_stock_selection/
├── main.py                          # pred_stocks逻辑（不修改）
├── local_data/
│   └── features/daily/              # 基础行情数据
├── factor_system/
│   └── factor_lab/store/            # 因子数据
│       └── factors.parquet
└── live_trading/
    ├── main.py                      # 实盘主程序（已修改）
    ├── model_loader.py              # 模型加载器（已修改）
    └── stock_selector.py            # 选股器（已修改）
```

### 相对路径处理

1. **live_trading/main.py**:
   ```python
   PathConfig.PROJECT_ROOT = Path(__file__).parent.parent
   # 指向: integrated_stock_selection/
   ```

2. **数据加载流程**:
   ```
   live_trading/main.py::load_factor_data(date)
     └─> 导入 main._load_factor_data_for_prediction(date)
           └─> main._load_factor_cache_data(...)
                 ├─> FactorDatasetStore(FACTOR_LAB_STORE_PATH)
                 │     └─> PROJECT_ROOT / "factor_system/factor_lab/store"
                 └─> IntegratedDataLoader().load_stock_features(...)
                       └─> PROJECT_ROOT / "local_data/features/daily"
   ```

3. **关键点**:
   - 所有路径都是相对于 `PROJECT_ROOT` 计算
   - `PROJECT_ROOT` 在两边都是 `integrated_stock_selection/`
   - 通过动态导入 `from main import ...` 实现代码复用
   - 路径完全一致，不会有问题

---

## 与main.py pred_stocks的差异对比

| 维度 | main.py pred_stocks | live_trading select（修改后） |
|------|---------------------|------------------------------|
| **模型类型** | LightGBM单模型 | ✅ LightGBM单模型 |
| **数据加载** | `_load_factor_data_for_prediction()` | ✅ 复用main.py函数 |
| **ST过滤** | ✓ | ✅ ✓ |
| **涨停过滤** | `is_limit_up == 1` | ✅ `is_limit_up == 1` |
| **跌停过滤** | ✓ | ✅ ✓ |
| **新股过滤** | 30天 | ✅ 30天 |
| **流动性过滤** | amount < 1000万 | ✅ amount < 1000万 |
| **北交所过滤** | ✗ | ✅ ✗ (已移除) |
| **预测逻辑** | 单模型预测 | ✅ 单模型预测 |
| **结果保存** | 打印到控制台 | ✅ JSON文件 |

**结论**: 修改后的live_trading select模式与main.py pred_stocks的**核心逻辑完全一致**，唯一的区别是结果保存方式不同。

---

## 使用方法

### 运行选股

```bash
# 在integrated_stock_selection目录下运行
cd C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection

# 运行选股模式
python live_trading\main.py --mode select --date 20250108 --num_stocks 3
```

### 参数说明

- `--mode select`: 选股模式
- `--date YYYYMMDD`: 选股日期（T日）
- `--num_stocks N`: 选股数量（1-5，默认3）

### 输出结果

1. **控制台输出**:
   - 模型加载信息
   - 数据加载信息
   - 过滤统计信息
   - 选股结果（包含SW行业）

2. **JSON文件**: `live_trading/stock_selections.json`
   ```json
   {
     "20250108": {
       "stocks": [
         {
           "code": "600000.SH",
           "score": 2.345,
           "rank": 1
         }
       ],
       "num_stocks": 3,
       "timestamp": "2025-01-08 20:00:00"
     }
   }
   ```

---

## 测试验证

使用提供的测试脚本验证修改:

```bash
python live_trading\test_select_mode.py
```

测试内容:
1. ✅ LightGBM模型加载
2. ✅ 数据加载（复用main.py逻辑）
3. ✅ 过滤逻辑（与main.py一致）
4. ✅ 预测逻辑（LightGBM单模型）
5. ✅ 结果文件保存

---

## 注意事项

1. **不要修改main.py**: 所有修改都在live_trading目录内
2. **路径问题**: 确保在 `integrated_stock_selection/` 目录下运行
3. **模型文件**: 确保 `results/lightgbm_model_*.pkl` 存在
4. **数据文件**: 确保因子数据和基础行情数据都已生成
5. **测试日期**: 使用实际存在的交易日进行测试

---

## 完成状态

- ✅ model_loader.py: 添加LightGBM单模型加载
- ✅ main.py: 修改load_factor_data，复用main.py逻辑
- ✅ main.py: 修改run_select_mode，使用LightGBM单模型
- ✅ stock_selector.py: 修改过滤逻辑，与main.py一致
- ✅ stock_selector.py: 修改预测逻辑，使用LightGBM单模型
- ✅ 测试脚本: test_select_mode.py
- ✅ 文档: MODIFICATION_SUMMARY.md

**修改完成！** 🎉
