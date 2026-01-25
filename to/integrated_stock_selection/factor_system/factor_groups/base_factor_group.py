#!/usr/bin/env python3
"""
因子组基类

定义因子组的标准接口和通用功能
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import hashlib

class BaseFactorGroup(ABC):
    """因子组基类"""
    
    def __init__(self, group_name: str = None):
        self.group_name = group_name or self.__class__.__name__
        self.name = self.group_name  # 添加name属性以保持兼容性
        self.logger = logging.getLogger(f"{__name__}.{self.group_name}")
        self.version = self._get_version()
        self.dependencies = self._get_dependencies()
        self.feature_names = self._get_feature_names()
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子组的所有特征
        
        Args:
            data: 输入数据，必须包含stock_code和date列
            
        Returns:
            包含新特征的数据
        """
        pass
    
    @abstractmethod
    def _get_feature_names(self) -> List[str]:
        """获取因子组包含的特征名称列表"""
        pass
    
    def _get_version(self) -> str:
        """获取因子组版本号"""
        # 子类可以重写此方法提供自定义版本
        return "v1.0"
    
    def _get_dependencies(self) -> List[str]:
        """获取依赖的其他因子组"""
        # 子类可以重写此方法提供依赖关系
        return []
    
    def get_version_hash(self) -> str:
        """获取版本哈希值"""
        features = sorted(self.feature_names)
        hash_string = f"{self.group_name}|{self.version}|{'|'.join(features)}"
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据格式"""
        try:
            if data.empty:
                self.logger.warning("输入数据为空")
                return False
            
            required_columns = ['stock_code', 'date']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"输入数据缺少必需列: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证输入数据失败: {e}")
            return False
    
    def validate_dependencies(self, data: pd.DataFrame) -> bool:
        """验证依赖特征是否存在"""
        try:
            if not self.dependencies:
                return True
            
            # 这里简化处理，实际应该检查依赖的特征是否存在
            # 在实际实现中，可以通过因子管理器来检查依赖
            return True
            
        except Exception as e:
            self.logger.error(f"验证依赖失败: {e}")
            return False
    
    def validate_output_data(self, data: pd.DataFrame) -> bool:
        """验证输出数据格式"""
        try:
            if data.empty:
                self.logger.warning("输出数据为空")
                return False
            
            # 检查是否包含预期的特征
            missing_features = [feat for feat in self.feature_names if feat not in data.columns]
            if missing_features:
                self.logger.warning(f"输出数据缺少预期特征: {missing_features}")
            
            # 检查是否有NaN值过多的特征
            for feature in self.feature_names:
                if feature in data.columns:
                    nan_ratio = data[feature].isna().sum() / len(data)
                    if nan_ratio > 0.5:  # 超过50%的NaN值
                        self.logger.warning(f"特征 {feature} 包含过多NaN值: {nan_ratio:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证输出数据失败: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        try:
            # 基础预处理
            processed_data = data.copy()
            
            # 确保日期格式正确
            if 'date' in processed_data.columns:
                processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
            
            # 按股票代码和日期排序
            if 'stock_code' in processed_data.columns and 'date' in processed_data.columns:
                processed_data = processed_data.sort_values(['stock_code', 'date'])
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"预处理数据失败: {e}")
            return data
    
    def postprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """后处理数据"""
        try:
            # 基础后处理
            processed_data = data.copy()
            
            # 处理无穷大值
            for feature in self.feature_names:
                if feature in processed_data.columns:
                    # 替换无穷大值为NaN
                    processed_data[feature] = processed_data[feature].replace([np.inf, -np.inf], np.nan)
                    
                    # 可选：填充NaN值
                    # processed_data[feature] = processed_data[feature].fillna(0)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"后处理数据失败: {e}")
            return data
    
    def calculate_with_validation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """带验证的计算方法"""
        try:
            # 验证输入
            if not self.validate_input_data(data):
                self.logger.error("输入数据验证失败")
                return data
            
            if not self.validate_dependencies(data):
                self.logger.error("依赖验证失败")
                return data
            
            # 预处理
            processed_data = self.preprocess_data(data)
            
            # 计算特征
            self.logger.debug(f"开始计算因子组 {self.group_name} 的特征")
            result_data = self.calculate(processed_data, **kwargs)
            
            # 后处理
            result_data = self.postprocess_data(result_data)
            
            # 验证输出
            if not self.validate_output_data(result_data):
                self.logger.warning("输出数据验证失败，但继续返回结果")
            
            self.logger.debug(f"因子组 {self.group_name} 计算完成")
            return result_data
            
        except Exception as e:
            self.logger.error(f"计算因子组 {self.group_name} 失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return data
    
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """获取特征元数据"""
        metadata = {}
        
        for feature in self.feature_names:
            metadata[feature] = {
                'group': self.group_name,
                'version': self.version,
                'dependencies': self.dependencies,
                'description': f"特征 {feature} 来自因子组 {self.group_name}"
            }
        
        return metadata
    
    def get_summary(self) -> Dict[str, Any]:
        """获取因子组摘要信息"""
        return {
            'group_name': self.group_name,
            'version': self.version,
            'version_hash': self.get_version_hash(),
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'dependencies': self.dependencies,
            'class_name': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        return f"{self.group_name}(v{self.version}, {len(self.feature_names)} features)"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(group_name='{self.group_name}', version='{self.version}')"
