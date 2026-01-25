# SW编码标准化修改方案

## 概述
将现有代码中的动态SW编码生成逻辑替换为预先生成的标准化编码，确保模型训练、回测、预测使用一致的编码。

## 文件结构
```
integrated_stock_selection/
├── generate_sw_encoding.py           # 编码生成器
├── sw_encoding_integration.py        # 编码集成接口
├── config/
│   └── sw_industry_encoding.pkl      # 标准化编码文件
├── main.py                           # 需要修改的主文件
└── SW_ENCODING_MODIFICATION_PLAN.md   # 本修改方案
```

## 修改步骤

### 步骤1: 生成标准化编码文件
```bash
# 首先生成标准化编码
cd C:\Users\86151\.trae-cn\pythons\path\to\integrated_stock_selection
python generate_sw_encoding.py --output-dir ./config
```

### 步骤2: 修改main.py中的关键函数

#### 2.1 修改导入部分
在main.py顶部添加：
```python
# 导入标准化SW编码接口
try:
    from sw_encoding_integration import (
        load_standardized_sw_encoding,
        get_sw_mapping_for_model,
        get_sw_industry_code,
        validate_encoding_compatibility,
        load_sw_sector_mapping as load_standardized_sw_sector_mapping
    )
    USE_STANDARDIZED_ENCODING = True
    logger.info("[SW编码] 使用标准化编码系统")
except ImportError:
    USE_STANDARDIZED_ENCODING = False
    logger.warning("[SW编码] 标准化编码系统不可用，使用原始方法")
```

#### 2.2 修改load_sw_sector_mapping函数
```python
def load_sw_sector_mapping() -> Optional[Dict[str, str]]:
    """
    加载SW申万行业数据（优先使用标准化编码）

    数据格式: {"920351.BJ": "SW1交通运输", ...}
    返回: 股票代码到SW行业名称的映射字典，或None（加载失败时）
    """
    # 优先使用标准化编码
    if USE_STANDARDIZED_ENCODING:
        try:
            encoding_data = load_standardized_sw_encoding()
            if encoding_data:
                return encoding_data['mappings']['stock_to_industry']
            else:
                logger.warning("[SW编码] 标准化编码加载失败，回退到原始方法")
        except Exception as e:
            logger.error(f"[SW编码] 标准化编码加载异常: {e}")
            logger.warning("[SW编码] 回退到原始方法")

    # 原始加载方法（保持向后兼容）
    return _load_original_sw_sector_mapping()

def _load_original_sw_sector_mapping() -> Optional[Dict[str, str]]:
    """原始的SW加载方法，作为回退方案"""
    # 这里是现有的load_sw_sector_mapping实现
    # ... 保持原有代码不变
    pass
```

#### 2.3 修改_train_lightgbm_model函数中的SW编码部分
```python
def _train_lightgbm_model(raw_data, labels, feature_columns, label_stats=None,
                         preprocessed_data=None, optimize_thresholds=False,
                         sw_sector_mapping=None):
    """训练LightGBM模型"""
    # ... 其他代码保持不变

    # SW行业编码处理（修改部分）
    if USE_STANDARDIZED_ENCODING and sw_sector_mapping is not None:
        logger.info(f"[SW行业编码] 使用标准化编码系统...")

        # 获取标准化的模型映射
        standardized_mapping = get_sw_mapping_for_model()
        if standardized_mapping:
            logger.info(f"[SW行业编码] 成功加载标准化映射")

            # 替换原有的动态生成逻辑
            train_stock_codes = preprocessed_data.get('train_stock_codes')
            val_stock_codes = preprocessed_data.get('val_stock_codes')

            if train_stock_codes is not None and val_stock_codes is not None:
                # 直接使用预生成的编码
                train_industry_codes = []
                val_industry_codes = []

                for code in train_stock_codes:
                    industry_code, _ = get_sw_industry_code(code)
                    train_industry_codes.append(industry_code)

                for code in val_stock_codes:
                    industry_code, _ = get_sw_industry_code(code)
                    val_industry_codes.append(industry_code)

                # 添加行业特征到数据
                preprocessed_data['train_features']['industry_sw_code'] = train_industry_codes
                preprocessed_data['val_features']['industry_sw_code'] = val_industry_codes

                # 保存标准化编码到模型结果中
                lgb_results['industry_mapping'] = standardized_mapping

                logger.info(f"[SW行业编码] 标准化编码应用完成")
                logger.info(f"  训练集行业分布: {pd.Series(train_industry_codes).value_counts().to_dict()}")
        else:
            logger.warning(f"[SW行业编码] 标准化映射获取失败，使用原始方法")
            # 继续使用原始的动态生成逻辑
    else:
        # 使用原始的动态生成逻辑（现有代码）
        # ... 保持原有代码不变
```

#### 2.4 修改_prepare_prediction_data函数
```python
def _prepare_prediction_data(factor_data: pd.DataFrame, feature_columns: list,
                            sw_sector_mapping: dict, logger, industry_mapping: dict = None):
    """准备预测数据，包括SW行业适配"""

    # 使用标准化编码的简化逻辑
    if USE_STANDARDIZED_ENCODING and sw_sector_mapping:
        logger.info(f"[SW行业编码] 使用标准化编码进行预测数据准备...")

        # 加载标准化编码
        standardized_mapping = get_sw_mapping_for_model()

        # 验证兼容性
        if industry_mapping and not validate_encoding_compatibility(industry_mapping):
            logger.warning(f"[SW行业编码] 模型编码与当前编码不兼容")
            # 可以选择继续或中止

        # 直接使用预先生成的编码，无需重新计算
        industry_feature_name = standardized_mapping.get('industry_feature_name', 'industry_sw_code')
        industry_name_to_code = standardized_mapping.get('industry_name_to_code', {})

        # 为预测数据添加行业编码
        if 'stock_code' in factor_data.columns:
            stock_codes = factor_data['stock_code'].tolist()
            industry_codes = []

            for stock_code in stock_codes:
                industry_code, _ = get_sw_industry_code(stock_code)
                industry_codes.append(industry_code)

            factor_data[industry_feature_name] = industry_codes
            logger.info(f"[SW行业编码] 成功为 {len(factor_data)} 只股票添加行业编码")

        # 更新特征列表（如果需要）
        if industry_feature_name not in feature_columns:
            feature_columns = feature_columns + [industry_feature_name]
            logger.info(f"[SW行业编码] 添加行业特征: {industry_feature_name}")

        return factor_data, factor_data[feature_columns].values

    else:
        # 使用原始逻辑（向后兼容）
        # ... 保持原有代码不变
```

#### 2.5 修改run_test_model_walkforward_mode函数
```python
def run_test_model_walkforward_mode(args):
    """运行Walk-Forward Analysis模式"""

    # ... 其他代码保持不变

    # [SW行业编码] 加载SW申万行业数据（修改部分）
    logger.info("\n[SW行业编码] 加载SW申万行业数据...")

    if USE_STANDARDIZED_ENCODING:
        sw_sector_mapping = load_standardized_sw_encoding()
        if sw_sector_mapping:
            logger.info(f"[OK] 标准化SW行业数据加载成功")
            encoding_info = sw_sector_mapping.get('metadata', {})
            logger.info(f"  版本: {encoding_info.get('version', 'unknown')}")
            logger.info(f"  股票数: {encoding_info.get('total_stocks', 0):,}")
            logger.info(f"  行业数: {encoding_info.get('total_industries', 0)}")
        else:
            logger.error("[FAIL] 标准化SW行业数据加载失败")
            sw_sector_mapping = None
    else:
        # 使用原始加载方法
        sw_sector_mapping = None
        try:
            sw_sector_mapping = load_sw_sector_mapping()
            if sw_sector_mapping and len(sw_sector_mapping) > 0:
                logger.info(f"[OK] SW行业数据加载成功: {len(sw_sector_mapping)}只股票")
            else:
                logger.warning("[WARNING] SW行业数据为空或加载失败")
                sw_sector_mapping = None
        except Exception as e:
            logger.warning(f"[WARNING] SW行业数据加载失败: {e}")
            sw_sector_mapping = None

    # ... 继续其他处理逻辑
```

#### 2.6 修改run_pred_stocks_mode函数
```python
def run_pred_stocks_mode(args):
    """股票预测模式"""

    # ... 其他代码保持不变

    # 3. 处理SW行业映射数据（简化版）
    logger.info("步骤3: 处理SW行业映射数据")

    if USE_STANDARDIZED_ENCODING:
        # 使用标准化编码，无需区分训练时和预测时
        encoding_data = load_standardized_sw_encoding()
        if encoding_data:
            logger.info("[OK] 使用标准化SW行业编码")

            # 直接使用标准化的映射
            industry_mapping = get_sw_mapping_for_model()
            sw_sector_mapping = encoding_data['mappings']['stock_to_industry']

            logger.info(f"  编码版本: {industry_mapping.get('encoding_version', 'unknown')}")
            logger.info(f"  行业数量: {industry_mapping.get('total_industries', 0)}")
        else:
            logger.error("[FAIL] 标准化SW编码加载失败")
            sw_sector_mapping = None
            industry_mapping = None
    else:
        # 使用原始逻辑（向后兼容）
        # ... 保持原有代码不变
```

### 步骤3: 更新函数调用

#### 3.1 在所有使用sw_sector_mapping的地方添加编码可用性检查
```python
# 例如在数据过滤之前
if sw_sector_mapping:
    # 原有逻辑保持不变
    valid_stocks_with_sw = set(sw_sector_mapping.keys())
    # ... 其他逻辑
else:
    logger.warning("[SW编码] SW行业映射不可用，跳过相关过滤")
```

### 步骤4: 测试和验证

#### 4.1 编码一致性测试
```python
def test_sw_encoding_consistency():
    """测试SW编码一致性"""
    # 生成测试数据
    test_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

    # 加载标准化编码
    encoding_data = load_standardized_sw_encoding()

    # 测试多次加载的一致性
    for i in range(3):
        for stock in test_stocks:
            code1, industry1 = get_sw_industry_code(stock, encoding_data)
            code2, industry2 = get_sw_industry_code(stock)  # 使用缓存

            assert code1 == code2, f"编码不一致: {stock}"
            assert industry1 == industry2, f"行业不一致: {stock}"

    print("✅ SW编码一致性测试通过")
```

## 修改影响评估

### 需要修改的函数数量
1. **load_sw_sector_mapping** - 核心加载函数
2. **_train_lightgbm_model** - 模型训练时的SW编码处理
3. **_prepare_prediction_data** - 预测时的SW编码处理
4. **run_test_model_walkforward_mode** - 训练模式的SW编码加载
5. **run_pred_stocks_mode** - 预测模式的SW编码处理

### 代码行数估算
- 总修改量：约 **200-300行**
- 新增代码：约 **150行**（主要是集成逻辑）
- 修改现有代码：约 **100-150行**

### 风险评估
1. **低风险**：向后兼容，保留原始方法作为回退
2. **可控风险**：可以通过开关控制是否使用标准化编码
3. **测试友好**：可以渐进式部署和验证

## 优势

### 1. 彻底解决编码不一致问题
- 所有模块使用相同的预生成编码
- 消除了动态生成的不确定性

### 2. 性能提升
- 避免每次重新计算编码
- 减少内存占用和计算时间

### 3. 可维护性增强
- 集中管理编码逻辑
- 便于版本控制和问题追踪

### 4. 向后兼容
- 保留原始方法作为回退
- 渐进式部署，降低风险

## 部署建议

### 阶段1：准备阶段
1. 运行编码生成器生成标准化文件
2. 在测试环境验证编码正确性

### 阶段2：试点部署
1. 先在`run_pred_stocks_mode`中部署
2. 验证预测结果的一致性

### 阶段3：全面部署
1. 在所有相关函数中部署
2. 全面测试训练、回测、预测的一致性

### 阶段4：监控和优化
1. 监控系统运行稳定性
2. 根据反馈优化编码生成逻辑

## 总结

这个修改方案通过预生成标准化SW编码文件，彻底解决了现有的编码不一致问题，同时保持了向后兼容性和渐进式部署的可能性。方案风险可控，修改量适中，是解决SW编码一致性问题的最佳方案。