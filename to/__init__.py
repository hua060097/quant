"""
To项目核心模块包

此包包含了时间序列数据处理和特征工程的核心组件。
"""

# 版本信息
__version__ = '0.1.0'

# 导出常用的路径工具函数，方便其他模块使用
try:
    from .path_utils import (
        get_project_path,
        get_data_path,
        get_model_path,
        get_log_path,
        get_results_path,
        ensure_dir,
        normalize_path
    )
except ImportError:
    # 如果path_utils.py不存在，则提供兼容性处理
    import os
    from pathlib import Path
    
    def get_project_path(relative_path=None):
        project_root = Path(__file__).resolve().parent.parent
        if relative_path:
            return project_root / relative_path
        return project_root
