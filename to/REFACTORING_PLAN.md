# 🔧 基于现有特征架构的重构方案

## 📋 现有特征架构分析

### **已有的特征加载体系**
1. **板块特征**：`_batch_load_features` → 调用 `_process_features_with_layers`
2. **个股特征**：`load_stock_features` → 调用 `_enhance_stock_features`
3. **市场特征**：`_load_all_market_status` → 在 `_process_features_with_layers` 中使用

### **_process_features_with_layers 的市场状态参数**
```python
market_status = {
    'market_type': '中性盘整',           # 市场类型
    'market_cycle': '蓄势',             # 市场周期
    'trend_strength': 0.0,              # 趋势强度
    'signals': {
        'trading_advice': '观望为主',    # 交易建议
        'risk_level': 'low',            # 风险等级
        'strength': 0                   # 信号强度
    },
    'individual_status': {              # 个股状态
        '399006.SZ': {...}             # 创业板指数等
    }
}
```

### **现有特征生成流程**
1. **板块特征生成**：`generate_sector_features.py` → 保存为 `{sector}_features.parquet`
2. **个股特征生成**：`generate_daily_features.py` → 保存为日期文件
3. **市场状态生成**：`market_status_processor.py` → 保存为 `market_status.json`

## 🎯 重构目标

### **核心问题**
1. ❌ **relations_df未被充分利用**：在训练和预测中都没有使用股票-板块关联关系
2. ❌ **训练预测逻辑不一致**：训练时分析变化点，预测时未充分利用关联信息
3. ❌ **特征工程重复**：我添加的新模块与现有特征加载体系重复

### **重构原则**
1. ✅ **严格使用已有特征**：基于 `_batch_load_features`、`load_stock_features`、`_load_all_market_status`
2. ✅ **删除重复代码**：移除我添加的重复特征工程模块
3. ✅ **充分利用relations_df**：在现有架构基础上增强关联特征
4. ✅ **保持架构简洁**：不破坏现有的特征加载逻辑

## 🛠️ 具体重构方案

### **方案1：重构_analyze_stock_change_point_patterns**

#### **当前问题**
```python
def _analyze_stock_change_point_patterns(self, stock_features_df, sector_features_df, relations_df):
    # relations_df参数未被使用
    # 没有利用现有的特征加载体系
```

#### **重构后**
```python
def _analyze_stock_change_point_patterns(self, stock_features_df, sector_features_df, relations_df):
    """
    分析股票变化点模式 - 基于现有特征架构重构
    
    充分利用：
    1. relations_df构建股票-板块关联特征
    2. 现有的_batch_load_features加载板块特征
    3. 现有的load_stock_features加载个股特征
    4. 现有的_load_all_market_status加载市场特征
    """
    
    # 1. 使用relations_df构建股票-板块映射
    stock_sector_mapping = self._build_stock_sector_mapping(relations_df)
    
    # 2. 为每只股票增强关联特征
    enhanced_stock_features = self._enhance_stock_features_with_relations(
        stock_features_df, sector_features_df, stock_sector_mapping
    )
    
    # 3. 使用现有的变化点检测逻辑
    change_points = self._identify_change_points(enhanced_stock_features)
    
    # 4. 提取变化点特征（包含关联特征）
    change_point_features = self._extract_change_point_features(
        change_points, enhanced_stock_features, sector_features_df
    )
    
    return {
        'patterns': change_point_features,
        'stock_sector_mapping': stock_sector_mapping,
        'enhanced_features': enhanced_stock_features
    }
```

### **方案2：重构预测方法**

#### **当前问题**
```python
def _predict_with_similarity_matching(self, predict_features, relations_df, feature_patterns):
    # relations_df参数未被使用
    # 没有构建完整的关联特征
```

#### **重构后**
```python
def _predict_with_similarity_matching(self, predict_features, relations_df, feature_patterns):
    """
    相似度匹配预测 - 基于现有特征架构重构
    """
    
    # 1. 使用relations_df获取股票关联的板块
    stock_sector_mapping = feature_patterns.get('stock_sector_mapping', {})
    
    # 2. 为预测股票加载关联板块特征
    enhanced_predict_features = self._enhance_predict_features_with_existing_loaders(
        predict_features, relations_df, stock_sector_mapping
    )
    
    # 3. 使用增强后的特征进行相似度匹配
    return self._perform_similarity_matching(enhanced_predict_features, feature_patterns)

def _enhance_predict_features_with_existing_loaders(self, predict_features, relations_df, stock_sector_mapping):
    """使用现有的特征加载器增强预测特征"""
    
    enhanced_features = predict_features.copy()
    
    # 获取预测日期范围
    dates = predict_features['date_str'].unique()
    start_date = min(dates)
    end_date = max(dates)
    
    # 获取涉及的板块
    involved_sectors = set()
    for _, row in predict_features.iterrows():
        stock_code = row['stock_code']
        if stock_code in stock_sector_mapping:
            involved_sectors.update(stock_sector_mapping[stock_code])
    
    # 使用现有的_batch_load_features加载板块特征
    if involved_sectors:
        sector_features = self._batch_load_features(
            list(involved_sectors), start_date, end_date
        )
        
        # 将板块特征映射到个股
        enhanced_features = self._map_sector_features_to_stocks(
            enhanced_features, sector_features, relations_df
        )
    
    # 使用现有的_load_all_market_status加载市场特征
    market_status_data = self._load_all_market_status()
    if market_status_data:
        enhanced_features = self._add_market_features_to_stocks(
            enhanced_features, market_status_data
        )
    
    return enhanced_features
```

### **方案3：重构模型数据准备**

#### **重构_prepare_test_data_for_model_with_features**
```python
def _prepare_test_data_for_model_with_features(self, predict_features, relations_df, expected_features):
    """
    为模型预测准备测试数据 - 基于现有特征架构重构
    """
    
    # 1. 使用现有特征加载器增强特征
    enhanced_features = self._enhance_predict_features_with_existing_loaders(
        predict_features, relations_df, {}
    )
    
    # 2. 特征对齐和标准化
    aligned_features = self._align_features_with_expected(enhanced_features, expected_features)
    
    return aligned_features, self._extract_stock_info(predict_features)
```

### **方案4：新增关联特征处理方法**

#### **_build_stock_sector_mapping**
```python
def _build_stock_sector_mapping(self, relations_df):
    """构建股票-板块映射关系"""
    mapping = {}
    for _, row in relations_df.iterrows():
        stock_code = row['stock_code']
        sector_code = row['sector_code']
        if stock_code not in mapping:
            mapping[stock_code] = []
        mapping[stock_code].append(sector_code)
    return mapping
```

#### **_map_sector_features_to_stocks**
```python
def _map_sector_features_to_stocks(self, stock_features, sector_features, relations_df):
    """将板块特征映射到个股"""
    
    enhanced_features = stock_features.copy()
    
    # 构建映射关系
    stock_sector_mapping = self._build_stock_sector_mapping(relations_df)
    
    # 为每只股票添加关联板块特征
    for _, stock_row in stock_features.iterrows():
        stock_code = stock_row['stock_code']
        date_str = stock_row['date_str']
        
        if stock_code in stock_sector_mapping:
            for sector_code in stock_sector_mapping[stock_code]:
                # 查找对应的板块特征
                sector_row = sector_features[
                    (sector_features['sector_code'] == sector_code) &
                    (sector_features['date_str'] == date_str)
                ]
                
                if not sector_row.empty:
                    # 添加板块特征到股票特征中
                    for col in ['change_percent', 'volume', 'amount']:
                        if col in sector_row.columns:
                            enhanced_features.loc[
                                enhanced_features['stock_code'] == stock_code,
                                f'sector_{sector_code}_{col}'
                            ] = sector_row[col].iloc[0]
    
    return enhanced_features
```

#### **_add_market_features_to_stocks**
```python
def _add_market_features_to_stocks(self, stock_features, market_status_data):
    """将市场特征添加到个股特征中"""
    
    enhanced_features = stock_features.copy()
    
    for _, stock_row in stock_features.iterrows():
        date_str = stock_row['date_str']
        
        if date_str in market_status_data:
            market_data = market_status_data[date_str]
            
            # 添加市场特征
            enhanced_features.loc[
                enhanced_features['date_str'] == date_str,
                'market_trend_strength'
            ] = market_data.get('trend_strength', 0)
            
            enhanced_features.loc[
                enhanced_features['date_str'] == date_str,
                'market_risk_level'
            ] = self._encode_risk_level(market_data.get('signals', {}).get('risk_level', 'low'))
    
    return enhanced_features
```

## 🗑️ 需要删除的重复代码

### **删除的文件**
1. `advanced_feature_engineering.py` - 与现有特征工程重复
2. `relation_feature_processor.py` - 功能整合到主文件中
3. `hierarchical_data_structure.py` - 现有架构已足够

### **删除的方法**
1. `_generate_market_data_from_stocks` - 使用现有的`_load_all_market_status`
2. `_enhance_predict_features_with_relations` - 重构为使用现有加载器
3. 所有重复的特征标准化逻辑

## 📊 重构后的架构

### **特征加载流程**
```
训练阶段：
stock_features (load_stock_features) + 
sector_features (_batch_load_features) + 
market_features (_load_all_market_status) + 
relations_df (新增关联逻辑)
↓
enhanced_stock_features (包含关联特征)
↓
change_point_analysis

预测阶段：
predict_features + 
relations_df + 
现有特征加载器
↓
enhanced_predict_features (包含关联特征)
↓
similarity_matching / model_prediction
```

### **关键改进**
1. ✅ **relations_df充分利用**：在训练和预测中都构建关联特征
2. ✅ **架构统一**：训练和预测使用相同的特征增强逻辑
3. ✅ **代码简洁**：删除重复模块，基于现有架构扩展
4. ✅ **性能优化**：复用现有的缓存和优化机制

## 🎯 实施步骤

### **第1步：删除重复代码**
- 删除 `advanced_feature_engineering.py`
- 删除 `relation_feature_processor.py`  
- 删除 `hierarchical_data_structure.py`

### **第2步：重构核心方法**
- 重构 `_analyze_stock_change_point_patterns`
- 重构 `_predict_with_similarity_matching`
- 重构 `_prepare_test_data_for_model_with_features`

### **第3步：新增关联特征方法**
- 添加 `_build_stock_sector_mapping`
- 添加 `_map_sector_features_to_stocks`
- 添加 `_add_market_features_to_stocks`

### **第4步：测试验证**
- 验证relations_df被充分利用
- 验证训练预测逻辑一致性
- 验证特征数量和质量

这个重构方案严格基于现有架构，删除了重复代码，确保了逻辑的通畅和简洁。
