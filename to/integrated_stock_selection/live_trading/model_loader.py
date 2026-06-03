"""
模型加载模块

负责加载训练好的集成模型、scaler、label_transformer等
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .config import PathConfig, StrategyConfig


class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.scalers = []
        self.label_transformers = []
        self.model_types = []  # 新增：模型类型列表(xgboost/lightgbm)
        self.model_performances = []  # 🚀 新增：模型RankIC列表(排序模型专用)
        self.feature_names = None
        self.label_stats = None
        self.winsorization_thresholds = None
    
    def load_latest_ensemble_models(self) -> bool:
        """
        加载最新的集成模型
        
        Returns:
            bool: 是否加载成功
        """
        try:
            model_dir = PathConfig.MODEL_DIR
            if not model_dir.exists():
                self.logger.error(f"模型目录不存在: {model_dir}")
                return False
            
            # 查找集成模型文件（支持XGBoost和LightGBM）
            xgb_files = list(model_dir.glob("xgboost_model_*.pkl"))
            lgb_files = list(model_dir.glob("lightgbm_model_*.pkl"))
            model_files = xgb_files + lgb_files

            if not model_files:
                self.logger.error("未找到模型文件")
                return False
            
            # 按时间戳分组
            model_groups = defaultdict(list)
            for f in model_files:
                # 提取时间戳
                # 格式: xgboost_model_YYYYMMDD_HHMMSS.pkl 或 lightgbm_model_YYYYMMDD_HHMMSS.pkl
                parts = f.stem.split('_')
                if len(parts) >= 4:
                    timestamp = f"{parts[2]}_{parts[3]}"
                    model_groups[timestamp].append(f)
            
            # 获取最新时间戳的所有模型
            latest_timestamp = sorted(model_groups.keys())[-1]
            latest_models = sorted(model_groups[latest_timestamp])

            self.logger.info(f"找到最新模型: {len(latest_models)}个")
            self.logger.info(f"时间戳: {latest_timestamp}")
            
            # 加载所有模型
            for model_file in latest_models:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                self.models.append(model_data['model'])
                self.scalers.append(model_data['scaler'])
                self.label_transformers.append(model_data.get('label_transformer', None))

                # 🚀 新增：加载模型类型和性能(使用RankIC)
                model_type = model_data.get('model_type', 'xgboost')  # 默认xgboost
                self.model_types.append(model_type)

                # 🚀 排序模型使用RankIC作为性能指标
                performance = model_data.get('performance', {})
                rank_ic = performance.get('rank_ic', None)
                self.model_performances.append(rank_ic)

                if self.feature_names is None:
                    self.feature_names = model_data['feature_names']
                    self.label_stats = model_data.get('label_stats', None)
                    self.winsorization_thresholds = model_data.get('winsorization_thresholds', {})

                self.logger.info(f"  ✅ 加载: {model_file.name} (类型:{model_type.upper()}, RankIC:{rank_ic:.4f if rank_ic else 'N/A'})")
            
            self.logger.info(f"✅ 模型加载成功: {len(self.models)}个")
            self.logger.info(f"  XGBoost模型: {self.model_types.count('xgboost')}个")
            self.logger.info(f"  LightGBM模型: {self.model_types.count('lightgbm')}个")
            self.logger.info(f"✅ 特征数量: {len(self.feature_names)}")

            if self.label_stats:
                self.logger.info(f"✅ 标签统计量: μ={self.label_stats['mean']:.6f}, σ={self.label_stats['std']:.6f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def load_lightgbm_model(self, model_path: Optional[Path] = None) -> bool:
        """
        加载单个LightGBM模型（与main.py pred_stocks一致）

        Args:
            model_path: 模型文件路径，如果为None则自动查找最新的LightGBM模型

        Returns:
            bool: 是否加载成功
        """
        try:
            model_dir = PathConfig.MODEL_DIR
            if not model_dir.exists():
                self.logger.error(f"模型目录不存在: {model_dir}")
                return False

            # 如果没有指定模型路径，查找最新的LightGBM模型
            if model_path is None:
                preferred_model = model_dir / StrategyConfig.DEFAULT_LIGHTGBM_MODEL_FILE
                if preferred_model.exists():
                    model_path = preferred_model
                    self.logger.info(f"使用默认LightGBM模型: {model_path.name}")
                else:
                    lgb_files = list(model_dir.glob("lightgbm_model_window3_*.pkl"))
                    if not lgb_files:
                        self.logger.error("未找到LightGBM模型文件")
                        return False
                    # 按修改时间排序，取最新的
                    model_path = max(lgb_files, key=lambda f: f.stat().st_mtime)
                    self.logger.warning(
                        f"默认模型 {StrategyConfig.DEFAULT_LIGHTGBM_MODEL_FILE} 不存在，回退到最新窗口3模型: {model_path.name}"
                    )

            if not model_path.exists():
                self.logger.error(f"模型文件不存在: {model_path}")
                return False

            # 加载模型
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.models = [model_data['model']]
            self.scalers = [model_data['scaler']]
            self.label_transformers = [model_data.get('label_transformer', None)]
            self.model_types = ['lightgbm']
            self.model_performances = [None]  # 单模型不需要RankIC
            self.feature_names = model_data.get('feature_names', [])
            self.label_stats = model_data.get('label_stats', None)
            self.winsorization_thresholds = model_data.get('winsorization_thresholds', {})

            self.logger.info(f"✅ 成功加载LightGBM模型: {model_path.name}")
            self.logger.info(f"  特征数量: {len(self.feature_names)}")

            if self.label_stats:
                self.logger.info(f"  标签统计量: μ={self.label_stats['mean']:.6f}, σ={self.label_stats['std']:.6f}")

            return True

        except Exception as e:
            self.logger.error(f"加载LightGBM模型失败: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict: 模型信息字典
        """
        return {
            'num_models': len(self.models),
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'has_label_transformer': any(lt is not None for lt in self.label_transformers),
            'has_winsorization': bool(self.winsorization_thresholds)
        }
