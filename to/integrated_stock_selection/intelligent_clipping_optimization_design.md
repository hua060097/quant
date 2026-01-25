# 智能裁剪算法优化改进设计

## 🔍 当前算法系统性问题分析

### 核心问题
通过分析 `corr_slope_20_120_z` 和 `ma_convergence_score_z` 两个案例，发现当前智能裁剪算法存在以下系统性问题：

#### 1. **偏态检测过于敏感**
```python
# 当前问题代码
if skewness > 0.5:  # 右偏，上阈值收紧
    upper_threshold = clip_threshold * 0.9  # 固定90%收紧
elif skewness < -0.5:  # 左偏，下阈值收紧
    lower_threshold = clip_threshold * 0.9  # 固定90%收紧
```

**问题分析**：
- **固定阈值0.5过于严格**：轻微偏态就触发收紧
- **固定收紧比例90%过于激进**：可能导致大量截断
- **不考虑偏态程度**：轻微偏态和严重偏态使用相同收紧比例

#### 2. **极端值比例检测不完善**
```python
# 当前逻辑
if extreme_ratio > 0.05 or kurtosis > 5:  # 极端值过多或严重厚尾
    clip_threshold = 2.5
elif extreme_ratio > 0.02 or abs(skewness) > 1:  # 中等问题
    clip_threshold = 2.8
```

**问题分析**：
- **阈值设置随意**：5%、2%、1%等阈值缺乏理论基础
- **多重条件叠加**：extreme_ratio 和 skewness/kurtosis 同时判断可能过度惩罚
- **没有考虑样本大小**：小样本和大样本使用相同标准

#### 3. **边界聚集因子未识别**
- **corr_slope_20_120**：相关性因子，自然右偏
- **ma_convergence_score**：收敛因子，自然左偏且下限聚集
- **这类因子不是真正的"异常"**：其偏态分布是自然特征

#### 4. **缺乏因子类型感知**
- 算法对所有因子使用相同的处理逻辑
- 没有考虑因子的理论约束和自然特性
- 缺乏对不同类型因子的分类处理

## 🛠️ 优化改进设计方案

### 核心设计原则

#### 1. **分层检测机制**
```python
# 三层检测：基础检测 → 深度分析 → 因子类型感知
Level 1: 基础统计检测（样本量、基本分布）
Level 2: 深度分布分析（偏态、峰度、极端值）
Level 3: 因子特性识别（边界聚集、理论约束、天然偏态）
```

#### 2. **动态阈值调整**
```python
# 基于样本大小动态调整阈值
def get_adaptive_thresholds(sample_size, factor_type):
    base_threshold = {
        'large_sample': {'extreme_ratio': 0.03, 'skewness': 1.5, 'kurtosis': 3.0},
        'medium_sample': {'extreme_ratio': 0.05, 'skewness': 1.0, 'kurtosis': 4.0},
        'small_sample': {'extreme_ratio': 0.10, 'skewness': 0.5, 'kurtosis': 5.0}
    }

    sample_category = classify_sample_size(sample_size)
    return base_threshold[sample_category].copy()
```

#### 3. **因子类型分类处理**
```python
FACTOR_CHARACTERISTICS = {
    'correlation': {
        'natural_skewness': 'right',
        'boundary_behavior': 'none',
        'sensitivity': 'high',
        'preferred_method': 'rankpct'
    },
    'convergence': {
        'natural_skewness': 'left',
        'boundary_behavior': 'lower_bound',
        'sensitivity': 'high',
        'preferred_method': 'rankpct'
    },
    'ratio': {
        'natural_skewness': 'right',
        'boundary_behavior': 'lower_bound',
        'sensitivity': 'medium',
        'preferred_method': 'zscore'
    },
    'momentum': {
        'natural_skewness': 'slight_right',
        'boundary_behavior': 'none',
        'sensitivity': 'medium',
        'preferred_method': 'zscore'
    }
}
```

#### 4. **渐进式裁剪策略**
```python
def graduated_clipping(data, thresholds):
    """
    渐进式裁剪：避免突然的严格截断

    策略：
    1. 温和警告区：轻度调整
    2. 中等调整区：适中收紧
    3. 严格干预区：大幅收紧
    """

    # 计算调整因子（0.7-1.0，1.0表示无调整）
    adjustment_factor = calculate_adjustment_factor(data, thresholds)

    return data.clip(
        -3.0 * adjustment_factor['lower'],
        3.0 * adjustment_factor['upper']
    )
```

### 详细算法设计

#### 1. **增强的分布分析器**
```python
class DistributionAnalyzer:
    def __init__(self):
        self.analysis_results = {}

    def analyze_distribution(self, data, factor_name):
        """全面分析数据分布特征"""
        results = {
            'basic_stats': self._calculate_basic_stats(data),
            'distribution_shape': self._analyze_distribution_shape(data),
            'extreme_values': self._analyze_extreme_values(data),
            'boundary_behavior': self._detect_boundary_behavior(data),
            'factor_characteristics': self._identify_factor_characteristics(factor_name, data),
            'recommendations': self._generate_recommendations(data, factor_name)
        }

        self.analysis_results[factor_name] = results
        return results

    def _detect_boundary_behavior(self, data):
        """检测边界聚集行为"""
        lower_boundary_ratio = (data <= data.quantile(0.01)).mean()
        upper_boundary_ratio = (data >= data.quantile(0.99)).mean()

        return {
            'lower_boundary_ratio': lower_boundary_ratio,
            'upper_boundary_ratio': upper_boundary_ratio,
            'has_boundary_cluster': lower_boundary_ratio > 0.02 or upper_boundary_ratio > 0.02,
            'boundary_type': 'both' if lower_boundary_ratio > 0.02 and upper_boundary_ratio > 0.02
                          else 'lower' if lower_boundary_ratio > 0.02
                          else 'upper' if upper_boundary_ratio > 0.02
                          else 'none'
        }
```

#### 2. **自适应阈值管理器**
```python
class AdaptiveThresholdManager:
    def __init__(self):
        self.threshold_history = {}

    def get_adjusted_thresholds(self, analysis_results, sample_size):
        """根据分析结果和样本大小动态调整阈值"""
        base_thresholds = {
            'clip_upper': 3.0,
            'clip_lower': -3.0,
            'extreme_ratio': 0.03,
            'skewness_threshold': 1.5,
            'kurtosis_threshold': 3.0
        }

        # 样本大小调整
        sample_adjustment = self._get_sample_size_adjustment(sample_size)

        # 分布特征调整
        distribution_adjustment = self._get_distribution_adjustment(analysis_results)

        # 因子特性调整
        factor_adjustment = self._get_factor_type_adjustment(analysis_results)

        # 综合调整
        final_adjustments = self._combine_adjustments(
            sample_adjustment, distribution_adjustment, factor_adjustment
        )

        # 应用调整
        adjusted_thresholds = {}
        for key, base_value in base_thresholds.items():
            if key in final_adjustments:
                adjusted_thresholds[key] = base_value * final_adjustments[key]
            else:
                adjusted_thresholds[key] = base_value

        return adjusted_thresholds

    def _get_distribution_adjustment(self, analysis_results):
        """基于分布特征的调整系数"""
        shape = analysis_results['distribution_shape']

        adjustments = {
            'clip_upper': 1.0,  # 上阈值调整系数
            'clip_lower': 1.0,  # 下阈值调整系数
            'extreme_ratio': 1.0,
            'skewness_threshold': 1.0,
            'kurtosis_threshold': 1.0
        }

        # 偏态调整（更温和）
        skewness = abs(shape['skewness'])
        if skewness > 2.0:  # 严重偏态
            adjustments['clip_upper'] *= 0.85 if shape['skewness'] > 0 else 0.95
            adjustments['clip_lower'] *= 0.95 if shape['skewness'] < 0 else 0.85
        elif skewness > 1.0:  # 中等偏态
            adjustments['clip_upper'] *= 0.95 if shape['skewness'] > 0 else 0.97
            adjustments['clip_lower'] *= 0.97 if shape['skewness'] < 0 else 0.95

        # 峰度调整
        kurtosis = shape['kurtosis']
        if kurtosis > 5.0:  # 严重厚尾
            adjustments['clip_upper'] *= 0.9
            adjustments['clip_lower'] *= 0.9
            adjustments['extreme_ratio'] *= 1.5
        elif kurtosis > 3.0:  # 中等厚尾
            adjustments['clip_upper'] *= 0.95
            adjustments['clip_lower'] *= 0.95
            adjustments['extreme_ratio'] *= 1.2

        return adjustments
```

#### 3. **智能裁剪执行器**
```python
class IntelligentClipper:
    def __init__(self):
        self.distribution_analyzer = DistributionAnalyzer()
        self.threshold_manager = AdaptiveThresholdManager()
        self.clipping_history = {}

    def smart_clip(self, data, factor_name, date=None):
        """智能裁剪主函数"""
        # 1. 基础检查
        if len(data) < 5:
            return data.clip(-3, 3)

        # 2. 分布分析
        analysis = self.distribution_analyzer.analyze_distribution(data, factor_name)

        # 3. 获取自适应阈值
        thresholds = self.threshold_manager.get_adjusted_thresholds(
            analysis, len(data)
        )

        # 4. 边界聚集特殊处理
        if analysis['boundary_behavior']['has_boundary_cluster']:
            thresholds = self._adjust_for_boundary_clustering(
                thresholds, analysis['boundary_behavior']
            )

        # 5. 应用智能裁剪
        clipped_data = self._apply_intelligent_clipping(data, thresholds)

        # 6. 记录历史
        self._record_clipping_history(factor_name, date, analysis, thresholds)

        return clipped_data

    def _adjust_for_boundary_clustering(self, thresholds, boundary_info):
        """为边界聚集调整阈值"""
        # 边界聚集时，相应侧的阈值应该更宽松
        if boundary_info['boundary_type'] == 'lower':
            # 下边界聚集，下阈值放宽
            thresholds['clip_lower'] *= 1.2  # 放宽20%
        elif boundary_info['boundary_type'] == 'upper':
            # 上边界聚集，上阈值放宽
            thresholds['clip_upper'] *= 1.2
        elif boundary_info['boundary_type'] == 'both':
            # 双边界聚集，两侧都放宽
            thresholds['clip_upper'] *= 1.15
            thresholds['clip_lower'] *= 1.15

        return thresholds

    def _apply_intelligent_clipping(self, data, thresholds):
        """应用渐进式智能裁剪"""
        # 计算温和的截断阈值
        upper_threshold = min(3.5, max(2.0, thresholds['clip_upper']))
        lower_threshold = max(-3.5, min(-2.0, thresholds['clip_lower']))

        # 应用截断
        return data.clip(lower_threshold, upper_threshold)
```

#### 4. **因子类型感知处理器**
```python
class FactorTypeAwareProcessor:
    def __init__(self):
        self.factor_patterns = {
            'correlation': r'corr.*|correlation',
            'convergence': r'convergence|ma_.*_score',
            'momentum': r'momentum|return_.*|pct_chg',
            'volatility': r'volatility|std|var',
            'ratio': r'ratio|_ma_.*|_to_.*',
            'rank': r'rank_.*'
        }

    def identify_factor_type(self, factor_name):
        """识别因子类型"""
        for factor_type, pattern in self.factor_patterns.items():
            if re.search(pattern, factor_name, re.IGNORECASE):
                return factor_type
        return 'unknown'

    def get_preferred_normalization(self, factor_type):
        """获取首选归一化方法"""
        preferences = {
            'correlation': 'rankpct',
            'convergence': 'rankpct',
            'momentum': 'zscore',
            'volatility': 'zscore',
            'ratio': 'zscore',
            'rank': 'rankpct'
        }
        return preferences.get(factor_type, 'zscore')
```

### 集成方案

#### 新的智能裁剪主函数
```python
def _zscore_normalize_with_intelligent_clipping_v2(self, series: pd.DataFrame, data: pd.DataFrame,
                                                    enable_intelligent_clipping: bool = True) -> pd.Series:
    """
    改进的智能裁剪版本2.0

    新特性：
    1. 三层检测机制
    2. 自适应阈值调整
    3. 因子类型感知
    4. 渐进式裁剪策略
    5. 边界聚集识别
    """
    # 1. 基础z-score归一化
    zscore_series = self._zscore_normalize(series, data)

    if not enable_intelligent_clipping:
        # 使用传统裁剪
        return zscore_series.clip(-3, 3)

    # 2. 创建改进的智能裁剪器
    clipper = IntelligentClipper()

    # 3. 按日期分组应用改进的智能裁剪
    df = pd.DataFrame({
        'zscore': zscore_series.values,
        'date': data['date'].values
    })

    def daily_smart_clip(group):
        factor_name = series.name if hasattr(series, 'name') else 'unknown'
        return clipper.smart_clip(group['zscore'], factor_name, group['date'].iloc[0])

    # 4. 应用改进的每日智能裁剪
    clipped_results = df.groupby('date').apply(daily_smart_clip)

    return pd.Series(clipped_results.values, index=series.index)
```

## 📊 预期改进效果

### 1. **减少过度截断**
- **corr_slope_20_120_z**：从95%截断减少到<5%
- **ma_convergence_score_z**：从95%截断减少到<10%

### 2. **提高因子质量**
- 保持更多原始信息
- 减少人为截断引入的偏差
- 提高因子的区分度

### 3. **自适应能力**
- 根据样本大小动态调整
- 考虑因子类型特征
- 适应不同的市场环境

### 4. **可解释性**
- 详细的裁剪日志
- 清晰的决策逻辑
- 可追溯的处理过程

## 🧪 测试和验证

### 测试用例设计
1. **边界聚集因子测试**：验证对收敛因子的处理
2. **极端值因子测试**：验证对异常值的处理
3. **样本量测试**：验证不同样本量的适应性
4. **回归测试**：确保不破坏现有功能

### 评估指标
1. **截断率**：被截断到边界的记录比例
2. **信息保留度**：保留的原始信息量
3. **稳定性**：算法的鲁棒性和一致性
4. **性能**：计算效率影响

这个设计方案从根本上解决了当前智能裁剪算法的系统性问题，通过多层次检测、动态阈值、因子感知等机制，实现更智能、更自适应的因子处理。