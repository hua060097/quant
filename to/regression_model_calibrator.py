"""
回归模型校准工具

此模块提供了对回归模型预测结果进行校准的功能，能够提高收益率预测的准确性。
主要支持以下校准方法：
1. 等回归校准 - 使用非参数化的单调映射校准预测结果
2. 线性缩放校准 - 简单的线性变换，调整均值和标准差
3. 分位数匹配校准 - 通过分位数映射调整整个分布
4. 集成校准 - 组合多种校准方式获得更稳健的结果

使用方法:
```python
from to.regression_model_calibrator import ReturnCalibrator

# 创建校准器
calibrator = ReturnCalibrator(method='isotonic')

# 校准预测
calibrated_pred = calibrator.fit_transform(y_pred, y_true)

# 保存校准器
calibrator.save('path/to/calibrator')

# 加载校准器
calibrator.load('path/to/calibrator')
```
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import json
import traceback
import logging

# 配置日志
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ReturnCalibrator(BaseEstimator, TransformerMixin):
    """用于校准收益率预测的类，可以使用多种校准方式
    
    支持的校准方法:
    - 'isotonic': 等回归校准，非参数化的单调映射
    - 'scaling': 线性缩放方法，调整预测值以匹配历史收益率分布
    - 'quantile': 分位数匹配方法，通过分位数映射调整预测值分布
    """
    
    def __init__(self, method='isotonic', n_bins=20, clip_quantile=0.01):
        """初始化校准器
        
        Args:
            method: 校准方法，可选 'isotonic', 'scaling', 'quantile'
            n_bins: 用于分位数校准的区间数量
            clip_quantile: 用于裁剪极端值的分位数阈值
        """
        self.method = method
        self.n_bins = n_bins
        self.clip_quantile = clip_quantile
        self.calibrator = None
        self.scale_factor = None
        self.shift_factor = None
        self.quantile_map = None
        self.target_stats = None
        self.is_fitted = False
        
    def fit(self, y_pred, y_true):
        """拟合校准器
        
        Args:
            y_pred: 模型预测值
            y_true: 真实收益率值
            
        Returns:
            self: 校准器对象
        """
        if len(y_pred) == 0 or len(y_true) == 0:
            raise ValueError("输入数组为空")
            
        # 移除NaN值
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred = np.array(y_pred)[mask]
        y_true = np.array(y_true)[mask]
        
        if len(y_pred) == 0 or len(y_true) == 0:
            raise ValueError("去除NaN后输入数组为空")
            
        # 储存目标分布的统计信息
        self.target_stats = {
            'mean': float(np.mean(y_true)),
            'std': float(np.std(y_true)),
            'min': float(np.min(y_true)),
            'max': float(np.max(y_true)),
            'median': float(np.median(y_true)),
            'q1': float(np.percentile(y_true, 25)),
            'q3': float(np.percentile(y_true, 75)),
        }
        
        # 根据选择的方法拟合校准器
        if self.method == 'isotonic':
            # 等回归校准，单调但不限制为线性关系
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true)
            
        elif self.method == 'scaling':
            # 线性缩放方法
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)
            true_mean = np.mean(y_true)
            true_std = np.std(y_true)
            
            # 防止除零
            if pred_std == 0:
                self.scale_factor = 1.0
            else:
                self.scale_factor = true_std / pred_std
                
            self.shift_factor = true_mean - pred_mean * self.scale_factor
            
        elif self.method == 'quantile':
            # 分位数映射校准
            # 将预测值和真实值映射到相同分位数分布
            self.quantile_map = {}
            
            pred_percentiles = np.linspace(0, 100, self.n_bins + 1)
            pred_quantiles = np.percentile(y_pred, pred_percentiles)
            true_quantiles = np.percentile(y_true, pred_percentiles)
            
            # 储存分位数映射
            for i in range(len(pred_quantiles)):
                self.quantile_map[float(pred_quantiles[i])] = float(true_quantiles[i])
                
        self.is_fitted = True
        return self
        
    def transform(self, y_pred):
        """应用校准转换到新的预测值
        
        Args:
            y_pred: 需要校准的预测值数组
            
        Returns:
            calibrated_pred: 校准后的预测值
        """
        if not self.is_fitted:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
            
        # 转为numpy数组
        y_pred = np.array(y_pred)
        
        # 处理NaN值
        has_nan = np.any(np.isnan(y_pred))
        if has_nan:
            nan_mask = np.isnan(y_pred)
            y_pred_valid = y_pred[~nan_mask]
        else:
            y_pred_valid = y_pred
            
        # 应用对应的校准方法
        if self.method == 'isotonic':
            if len(y_pred_valid) > 0:
                calibrated_valid = self.calibrator.transform(y_pred_valid)
            else:
                calibrated_valid = np.array([])
                
        elif self.method == 'scaling':
            calibrated_valid = y_pred_valid * self.scale_factor + self.shift_factor
            
        elif self.method == 'quantile':
            # 分位数映射
            calibrated_valid = np.zeros_like(y_pred_valid)
            pred_keys = sorted(list(self.quantile_map.keys()))
            
            for i, pred in enumerate(y_pred_valid):
                # 找到预测值所处的分位数区间
                if pred <= pred_keys[0]:
                    calibrated_valid[i] = self.quantile_map[pred_keys[0]]
                elif pred >= pred_keys[-1]:
                    calibrated_valid[i] = self.quantile_map[pred_keys[-1]]
                else:
                    # 线性插值
                    for j in range(len(pred_keys) - 1):
                        if pred_keys[j] <= pred < pred_keys[j + 1]:
                            left_key = pred_keys[j]
                            right_key = pred_keys[j + 1]
                            left_val = self.quantile_map[left_key]
                            right_val = self.quantile_map[right_key]
                            
                            # 线性插值公式
                            ratio = (pred - left_key) / (right_key - left_key) if right_key != left_key else 0
                            calibrated_valid[i] = left_val + ratio * (right_val - left_val)
                            break
                            
        # 重建包含NaN的完整数组
        if has_nan:
            calibrated_pred = np.empty_like(y_pred)
            calibrated_pred[~nan_mask] = calibrated_valid
            calibrated_pred[nan_mask] = np.nan
        else:
            calibrated_pred = calibrated_valid
            
        # 裁剪极端值
        if self.clip_quantile > 0 and self.clip_quantile < 0.5:
            valid_mask = ~np.isnan(calibrated_pred)
            if np.any(valid_mask):
                valid_values = calibrated_pred[valid_mask]
                low, high = np.percentile(valid_values, [self.clip_quantile*100, (1-self.clip_quantile)*100])
                calibrated_pred[valid_mask] = np.clip(valid_values, low, high)
                
        return calibrated_pred
        
    def fit_transform(self, y_pred, y_true):
        """同时执行拟合和转换
        
        Args:
            y_pred: 模型预测值
            y_true: 真实收益率值
            
        Returns:
            calibrated_pred: 校准后的预测值
        """
        return self.fit(y_pred, y_true).transform(y_pred)
        
    def save(self, filepath):
        """保存校准器到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'method': self.method,
                'n_bins': self.n_bins,
                'clip_quantile': self.clip_quantile,
                'is_fitted': self.is_fitted,
                'scale_factor': self.scale_factor,
                'shift_factor': self.shift_factor,
                'target_stats': self.target_stats
            }
            
            # 分别保存不同类型的数据
            if self.quantile_map is not None:
                model_data['quantile_map'] = {str(k): float(v) for k, v in self.quantile_map.items()}
                
            # 保存为JSON格式
            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=4)
                
            # 如果有IsotonicRegression对象，则单独保存
            if self.calibrator is not None:
                joblib.dump(self.calibrator, f"{filepath}.joblib")
                
            logger.info(f"校准器已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存校准器失败: {e}")
            logger.error(traceback.format_exc())
            
    def load(self, filepath):
        """从文件加载校准器
        
        Args:
            filepath: 文件路径
            
        Returns:
            self: 校准器对象
        """
        # 加载JSON格式的数据
        try:
            with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                
            self.method = model_data.get('method', 'isotonic')
            self.n_bins = model_data.get('n_bins', 20)
            self.clip_quantile = model_data.get('clip_quantile', 0.01)
            self.is_fitted = model_data.get('is_fitted', False)
            self.scale_factor = model_data.get('scale_factor')
            self.shift_factor = model_data.get('shift_factor')
            self.target_stats = model_data.get('target_stats')
            
            # 加载分位数映射
            if 'quantile_map' in model_data:
                self.quantile_map = {float(k): v for k, v in model_data['quantile_map'].items()}
                
            # 加载IsotonicRegression对象
            if self.method == 'isotonic' and os.path.exists(f"{filepath}.joblib"):
                self.calibrator = joblib.load(f"{filepath}.joblib")
                
            logger.info(f"校准器已从 {filepath} 加载")
            return self
            
        except Exception as e:
            logger.error(f"加载校准器失败: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"加载校准器失败: {e}")
            
class EnsembleCalibrator:
    """集成多个校准器的校准器
    
    这个类可以组合多种校准方法的结果，获得更稳健的校准效果
    """
    
    def __init__(self, methods=None, weights=None):
        """初始化集成校准器
        
        Args:
            methods: 校准方法列表，默认使用['isotonic', 'scaling', 'quantile']
            weights: 各方法的权重，默认平均权重
        """
        self.methods = methods or ['isotonic', 'scaling', 'quantile']
        self.weights = weights or [1.0/len(self.methods)] * len(self.methods)
        self.calibrators = []
        self.is_fitted = False
        
    def fit(self, y_pred, y_true):
        """拟合所有校准器
        
        Args:
            y_pred: 模型预测值
            y_true: 真实收益率值
            
        Returns:
            self: 校准器对象
        """
        self.calibrators = []
        
        for method in self.methods:
            calibrator = ReturnCalibrator(method=method)
            calibrator.fit(y_pred, y_true)
            self.calibrators.append(calibrator)
            
        self.is_fitted = True
        return self
        
    def transform(self, y_pred):
        """应用集成校准转换到新的预测值
        
        Args:
            y_pred: 需要校准的预测值数组
            
        Returns:
            calibrated_pred: 校准后的预测值
        """
        if not self.is_fitted or not self.calibrators:
            raise ValueError("校准器尚未拟合，请先调用fit方法")
            
        # 获取各校准器的结果
        calibrated_preds = []
        for calibrator in self.calibrators:
            calibrated_preds.append(calibrator.transform(y_pred))
            
        # 加权平均集成
        ensemble_pred = np.zeros_like(y_pred, dtype=float)
        for i, pred in enumerate(calibrated_preds):
            ensemble_pred += self.weights[i] * pred
            
        return ensemble_pred
        
    def fit_transform(self, y_pred, y_true):
        """同时执行拟合和转换
        
        Args:
            y_pred: 模型预测值
            y_true: 真实收益率值
            
        Returns:
            calibrated_pred: 校准后的预测值
        """
        return self.fit(y_pred, y_true).transform(y_pred)
        
    def save(self, filepath):
        """保存集成校准器到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存集成器配置
            model_data = {
                'methods': self.methods,
                'weights': self.weights,
                'is_fitted': self.is_fitted
            }
            
            with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=4)
                
            # 保存各个校准器
            for i, calibrator in enumerate(self.calibrators):
                calibrator.save(f"{filepath}_calibrator_{i}")
                
            logger.info(f"集成校准器已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存集成校准器失败: {e}")
            logger.error(traceback.format_exc())
            
    def load(self, filepath):
        """从文件加载集成校准器
        
        Args:
            filepath: 文件路径
            
        Returns:
            self: 校准器对象
        """
        try:
            with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                
            self.methods = model_data.get('methods', ['isotonic', 'scaling', 'quantile'])
            self.weights = model_data.get('weights', [1.0/len(self.methods)] * len(self.methods))
            self.is_fitted = model_data.get('is_fitted', False)
            
            # 加载各个校准器
            self.calibrators = []
            for i, method in enumerate(self.methods):
                calibrator = ReturnCalibrator(method=method)
                calibrator.load(f"{filepath}_calibrator_{i}")
                self.calibrators.append(calibrator)
                
            logger.info(f"集成校准器已从 {filepath} 加载")
            return self
            
        except Exception as e:
            logger.error(f"加载集成校准器失败: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"加载集成校准器失败: {e}")
            
def create_default_calibrator():
    """创建默认的集成校准器
    
    Returns:
        calibrator: 默认配置的校准器
    """
    return EnsembleCalibrator(
        methods=['isotonic', 'scaling', 'quantile'],
        weights=[0.5, 0.3, 0.2]  # 权重更偏向等回归校准
    )
    
def calibrate_predictions(y_pred, y_true=None, calibrator=None, method='ensemble'):
    """便捷函数，校准预测结果
    
    Args:
        y_pred: 需要校准的预测值
        y_true: 真实值，如果提供则重新拟合校准器
        calibrator: 预先训练好的校准器，如果为None且y_true提供，则创建新校准器
        method: 如果需要创建新校准器，使用的方法
        
    Returns:
        calibrated_pred: 校准后的预测值
        calibrator: 使用的校准器
    """
    if calibrator is None and y_true is not None:
        # 需要创建并拟合新校准器
        if method == 'ensemble':
            calibrator = create_default_calibrator()
        else:
            calibrator = ReturnCalibrator(method=method)
            
        calibrated_pred = calibrator.fit_transform(y_pred, y_true)
        return calibrated_pred, calibrator
        
    elif calibrator is not None:
        # 使用已有校准器
        calibrated_pred = calibrator.transform(y_pred)
        return calibrated_pred, calibrator
        
    else:
        # 无法校准，返回原始预测
        logger.warning("无法校准预测值：没有提供校准器或真实值")
        return y_pred, None 