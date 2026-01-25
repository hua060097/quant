# main.py 修改补丁

## 1. 在文件顶部添加导入（在现有imports之后）

```python
# 导入标准化SW编码接口
try:
    from sw_encoding_loader import (
        load_standardized_sw_encoding,
        get_sw_mapping_for_model,
        get_sw_industry_code,
        load_sw_sector_mapping as load_standardized_sw_sector_mapping
    )
    USE_STANDARDIZED_ENCODING = True
    logger.info("[SW编码] 使用标准化编码系统")
except ImportError as e:
    USE_STANDARDIZED_ENCODING = False
    logger.warning(f"[SW编码] 标准化编码系统不可用: {e}")
    logger.warning("[SW编码] 将使用原始动态生成方法")
```

## 2. 修改load_sw_sector_mapping函数（约6800行）

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
    # 将现有的实现内容移动到这里，保持原有代码不变
    try:
        # 现有的原始加载逻辑
        # ... 保留所有原始代码
        import os
        import json
        from pathlib import Path

        # 尝试从多个可能的位置加载
        possible_paths = [
            "./local_data/sw_sector_mapping.json",
            "./data/sw_sector_mapping.json",
            "./config/sw_sector_mapping.json"
        ]

        for path in possible_paths:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"[SW编码] 从 {path} 加载SW行业映射: {len(data)}只股票")
                    return data

        logger.warning(f"[SW编码] 未找到SW行业映射文件")
        return {}

    except Exception as e:
        logger.error(f"[SW编码] 加载SW行业映射失败: {e}")
        return {}
```

## 3. 修改_train_lightgbm_model函数中的SW编码部分（约2890行）

```python
def _train_lightgbm_model(raw_data, labels, feature_columns, label_stats=None, preprocessed_data=None, optimize_thresholds=False, sw_sector_mapping=None):
    """训练LightGBM模型(使用默认参数)"""

    # ... 其他代码保持不变

    # 如果提供了SW行业映射，添加行业分类特征
    if sw_sector_mapping is not None and len(sw_sector_mapping) > 0:
        logger.info(f"[SW行业编码] 开始添加SW行业分类特征...")

        # 使用标准化编码系统
        if USE_STANDARDIZED_ENCODING:
            logger.info(f"[SW行业编码] 使用标准化编码系统...")

            # 获取标准化的模型映射
            standardized_mapping = get_sw_mapping_for_model()
            if standardized_mapping:
                logger.info(f"[SW行业编码] 成功加载标准化映射")

                # 从preprocessed_data中提取训练集和验证集股票代码
                train_stock_codes = preprocessed_data.get('train_stock_codes')
                val_stock_codes = preprocessed_data.get('val_stock_codes')

                if train_stock_codes is not None and val_stock_codes is not None:
                    logger.info(f"[SW行业编码] 处理训练集({len(train_stock_codes)})和验证集({len(val_stock_codes)})")

                    # 直接使用预生成的编码
                    train_industry_codes = []
                    val_industry_codes = []

                    for code in train_stock_codes:
                        industry_code, industry_name = get_sw_industry_code(code)
                        train_industry_codes.append(industry_code)

                    for code in val_stock_codes:
                        industry_code, industry_name = get_sw_industry_code(code)
                        val_industry_codes.append(industry_code)

                    # 添加行业特征到数据
                    industry_feature_name = standardized_mapping.get('industry_feature_name', 'industry_sw_code')
                    preprocessed_data['train_features'][industry_feature_name] = train_industry_codes
                    preprocessed_data['val_features'][industry_feature_name] = val_industry_codes

                    # 更新特征列
                    if industry_feature_name not in feature_columns:
                        feature_columns = feature_columns + [industry_feature_name]

                    # 保存标准化编码到模型结果中
                    lgb_results['industry_mapping'] = standardized_mapping

                    logger.info(f"[SW行业编码] 标准化编码应用完成")
                    logger.info(f"  行业特征名: {industry_feature_name}")
                    logger.info(f"  训练集行业分布: {pd.Series(train_industry_codes).value_counts().to_dict()}")
                else:
                    logger.warning(f"[SW行业编码] 训练集或验证集股票代码为空")
            else:
                logger.warning(f"[SW行业编码] 标准化映射获取失败，使用原始方法")
                # 继续使用原始的动态生成逻辑
        else:
            # 使用原始的动态生成逻辑（现有代码，保持不变）
            # ... 保留所有原始代码
```

## 4. 修改_prepare_prediction_data函数（约13992行）

```python
def _prepare_prediction_data(factor_data: pd.DataFrame, feature_columns: list,
                            sw_sector_mapping: dict, logger, industry_mapping: dict = None):
    """准备预测数据，包括SW行业适配"""

    # 使用标准化编码的简化逻辑
    if USE_STANDARDIZED_ENCODING and sw_sector_mapping:
        logger.info(f"[SW行业编码] 使用标准化编码进行预测数据准备...")

        # 加载标准化编码
        standardized_mapping = get_sw_mapping_for_model()

        if standardized_mapping:
            logger.info(f"[SW行业编码] 成功加载标准化映射")

            # 直接使用预先生成的编码，无需重新计算
            industry_feature_name = standardized_mapping.get('industry_feature_name', 'industry_sw_code')
            industry_name_to_code = standardized_mapping.get('industry_name_to_code', {})

            # 为预测数据添加行业编码
            if 'stock_code' in factor_data.columns:
                stock_codes = factor_data['stock_code'].tolist()
                industry_codes = []

                for stock_code in stock_codes:
                    industry_code, industry_name = get_sw_industry_code(stock_code)
                    industry_codes.append(industry_code)

                factor_data[industry_feature_name] = industry_codes
                logger.info(f"[SW行业编码] 成功为 {len(factor_data)} 只股票添加行业编码")

            # 更新特征列表（如果需要）
            if industry_feature_name not in feature_columns:
                feature_columns = feature_columns + [industry_feature_name]
                logger.info(f"[SW行业编码] 添加行业特征: {industry_feature_name}")

            return factor_data, factor_data[feature_columns].values
        else:
            logger.warning(f"[SW行业编码] 标准化映射获取失败")
            # 继续使用原始逻辑
    else:
        # 使用原始逻辑（向后兼容）
        # ... 保留所有原始代码
```

## 5. 修改run_test_model_walkforward_mode函数（约13698行）

```python
def run_test_model_walkforward_mode(args):
    """运行Walk-Forward Analysis + 交叉验证模式"""

    # ... 其他代码保持不变

    # [SW行业编码] 加载SW申万行业数据用于LightGBM分类特征
    logger.info("\n[SW行业编码] 加载SW申万行业数据...")

    if USE_STANDARDIZED_ENCODING:
        sw_sector_mapping = None
        try:
            encoding_data = load_standardized_sw_encoding()
            if encoding_data and len(encoding_data) > 0:
                sw_sector_mapping = encoding_data['mappings']['stock_to_industry']
                logger.info(f"[OK] 标准化SW行业数据加载成功: {len(sw_sector_mapping)}只股票")

                metadata = encoding_data.get('metadata', {})
                logger.info(f"  版本: {metadata.get('version', 'unknown')}")
                logger.info(f"  生成时间: {metadata.get('generated_at', 'unknown')}")
                logger.info(f"  行业数: {metadata.get('total_industries', 0)}")
            else:
                logger.warning("[WARNING] 标准化SW行业数据为空或加载失败")
                sw_sector_mapping = None
        except Exception as e:
            logger.warning(f"[WARNING] 标准化SW行业数据加载失败: {e}")
            sw_sector_mapping = None
    else:
        # 使用原始加载方法（现有代码，保持不变）
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

## 6. 修改run_pred_stocks_mode函数（约13867行）

```python
def run_pred_stocks_mode(args):
    """股票预测模式：根据指定日期预测top5股票"""

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

            metadata = encoding_data.get('metadata', {})
            logger.info(f"  编码版本: {metadata.get('version', 'unknown')}")
            logger.info(f"  行业数量: {metadata.get('total_industries', 0)}")
        else:
            logger.error("[FAIL] 标准化SW编码加载失败")
            sw_sector_mapping = None
            industry_mapping = None
    else:
        # 使用原始逻辑（向后兼容）
        # ... 保留所有原始代码
```

## 实施步骤

1. **备份main.py**: 复制一份原始的main.py作为备份
2. **应用修改**: 按照上述补丁修改main.py
3. **测试验证**: 运行测试确保修改正常工作
4. **逐步部署**: 可以先在预测模式中测试，然后全面部署

## 修改优势

1. **向后兼容**: 保留原始方法作为回退方案
2. **渐进部署**: 可以通过开关控制是否使用标准化编码
3. **风险可控**: 如果标准化编码出现问题，可以立即回退
4. **性能提升**: 避免重复计算编码，提升加载速度

## 验证方法

修改完成后，可以通过以下方式验证：

```python
# 测试编码加载
from sw_encoding_loader import load_standardized_sw_encoding, get_encoding_info
encoding_data = load_standardized_sw_encoding()
print(f"编码信息: {get_encoding_info()}")

# 测试主函数
from main import load_sw_sector_mapping
sw_mapping = load_sw_sector_mapping()
print(f"加载的股票数: {len(sw_mapping) if sw_mapping else 0}")
```