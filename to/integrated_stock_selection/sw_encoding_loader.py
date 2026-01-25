#!/usr/bin/env python3
"""
SW行业编码标准化加载模块

提供统一的SW编码加载接口，替换现有的动态生成逻辑
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# 全局变量缓存已加载的编码
_SW_ENCODING_CACHE: Optional[Dict] = None

# 使用基于 __file__ 的绝对路径（避免工作目录依赖问题）
_CURRENT_DIR = Path(__file__).parent.resolve()
_ENCODING_FILE_PATH = _CURRENT_DIR / "config" / "sw2_industry_encoding.pkl"


def load_standardized_sw_encoding(
    encoding_file: str = _ENCODING_FILE_PATH,
    force_reload: bool = False
) -> Optional[Dict]:
    """
    加载标准化的SW行业编码

    Args:
        encoding_file: 编码文件路径
        force_reload: 强制重新加载

    Returns:
        Dict: 编码数据，加载失败返回None
    """
    global _SW_ENCODING_CACHE

    # 使用缓存避免重复加载
    if _SW_ENCODING_CACHE is not None and not force_reload:
        return _SW_ENCODING_CACHE

    try:
        encoding_path = Path(encoding_file)
        if not encoding_path.exists():
            logger.error(f"[SW编码] 编码文件不存在: {encoding_file}")
            logger.error(f"[SW编码] 请先运行: python generate_sw_encoding_simple.py")
            return None

        # 优先使用pickle格式（更快）
        if encoding_path.suffix == '.pkl':
            with open(encoding_path, 'rb') as f:
                encoding_data = pickle.load(f)
        else:
            # 支持JSON格式
            with open(encoding_path, 'r', encoding='utf-8') as f:
                encoding_data = json.load(f)

        # 验证数据完整性
        if not _validate_encoding_data(encoding_data):
            logger.error(f"[SW编码] 编码数据验证失败: {encoding_file}")
            return None

        # 缓存数据
        _SW_ENCODING_CACHE = encoding_data

        # 记录日志
        metadata = encoding_data.get('metadata', {})
        logger.info(f"[SW编码] 成功加载标准化编码")
        logger.info(f"  版本: {metadata.get('version', 'unknown')}")
        logger.info(f"  生成时间: {metadata.get('generated_at', 'unknown')}")
        logger.info(f"  股票数: {metadata.get('total_stocks', 0):,}")
        logger.info(f"  行业数: {metadata.get('total_industries', 0)}")

        return encoding_data

    except Exception as e:
        logger.error(f"[SW编码] 加载编码文件失败: {e}")
        return None


def _validate_encoding_data(encoding_data: Dict) -> bool:
    """验证编码数据的完整性"""
    try:
        # 检查必要字段
        required_fields = ['metadata', 'mappings']
        for field in required_fields:
            if field not in encoding_data:
                logger.error(f"[SW编码] 缺少必要字段: {field}")
                return False

        mappings = encoding_data['mappings']
        required_mappings = ['stock_to_industry', 'industry_to_code', 'code_to_industry']
        for mapping in required_mappings:
            if mapping not in mappings:
                logger.error(f"[SW编码] 缺少必要映射: {mapping}")
                return False

        # 验证双向映射一致性
        industry_to_code = mappings['industry_to_code']
        code_to_industry = mappings['code_to_industry']

        for industry, code in industry_to_code.items():
            if code_to_industry.get(code) != industry:
                logger.error(f"[SW编码] 双向映射不一致: {industry} <-> {code}")
                return False

        return True

    except Exception as e:
        logger.error(f"[SW编码] 数据验证异常: {e}")
        return False


def get_sw_industry_code(stock_code: str, encoding_data: Optional[Dict] = None) -> Tuple[int, str]:
    """
    获取股票对应的SW行业编码

    Args:
        stock_code: 股票代码
        encoding_data: 编码数据（可选，未提供时会自动加载）

    Returns:
        Tuple[int, str]: (行业编码, 行业名称)
    """
    if encoding_data is None:
        encoding_data = load_standardized_sw_encoding()

    if encoding_data is None:
        logger.warning(f"[SW编码] 无法加载编码数据，返回默认值")
        return -1, '未知'

    try:
        mappings = encoding_data['mappings']

        # 1. 股票代码 -> 行业名称
        industry_name = mappings['stock_to_industry'].get(stock_code, '未知')

        # 2. 行业名称 -> 编码
        industry_code = mappings['industry_to_code'].get(industry_name, -1)

        return industry_code, industry_name

    except Exception as e:
        logger.error(f"[SW编码] 获取行业编码失败: {e}")
        return -1, '未知'


def get_sw_mapping_for_model(
    encoding_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    获取适用于模型的SW映射数据

    Args:
        encoding_data: 编码数据（可选，未提供时会自动加载）

    Returns:
        Dict: 模型需要的映射数据格式
    """
    if encoding_data is None:
        encoding_data = load_standardized_sw_encoding()

    if encoding_data is None:
        logger.error(f"[SW编码] 无法加载编码数据")
        return {}

    try:
        mappings = encoding_data['mappings']
        metadata = encoding_data['metadata']

        # 构建模型需要的格式
        model_mapping = {
            'industry_feature_name': 'industry_sw_code',
            'stock_to_industry_name': mappings['stock_to_industry'],
            'industry_name_to_code': mappings['industry_to_code'],
            'code_to_industry_name': mappings['code_to_industry'],
            'encoding_version': metadata.get('version', 'unknown'),
            'generated_at': metadata.get('generated_at', 'unknown'),
            'total_industries': metadata.get('total_industries', 0)
        }

        return model_mapping

    except Exception as e:
        logger.error(f"[SW编码] 构建模型映射失败: {e}")
        return {}


# 向后兼容的接口
def load_sw_sector_mapping() -> Optional[Dict[str, str]]:
    """
    向后兼容接口，加载股票代码到行业名称的映射

    Returns:
        Dict[str, str]: 股票代码 -> 行业名称映射
    """
    encoding_data = load_standardized_sw_encoding()
    if encoding_data is None:
        # 回退到原始方法
        logger.warning(f"[SW编码] 标准化编码加载失败，回退到原始方法")
        return _load_original_sw_mapping()

    return encoding_data['mappings']['stock_to_industry']


def _load_original_sw_mapping() -> Optional[Dict[str, str]]:
    """
    原始的SW映射加载方法（作为回退方案）

    注意：这个函数需要从main.py中复制原始的load_sw_sector_mapping逻辑
    """
    try:
        # 这里需要复制原始的load_sw_sector_mapping函数逻辑
        # 为了保持向后兼容，我们暂时返回空字典
        logger.warning(f"[SW编码] 原始加载方法尚未实现")
        return {}
    except Exception as e:
        logger.error(f"[SW编码] 原始加载方法失败: {e}")
        return None


def is_encoding_available() -> bool:
    """检查标准化编码是否可用"""
    return Path(_ENCODING_FILE_PATH).exists()


def get_encoding_info() -> Dict[str, Any]:
    """获取当前编码的详细信息"""
    encoding_data = load_standardized_sw_encoding()
    if encoding_data is None:
        return {}

    metadata = encoding_data.get('metadata', {})
    mappings = encoding_data.get('mappings', {})

    return {
        'version': metadata.get('version', 'unknown'),
        'generated_at': metadata.get('generated_at', 'unknown'),
        'total_stocks': metadata.get('total_stocks', 0),
        'total_industries': metadata.get('total_industries', 0),
        'encoding_file': _ENCODING_FILE_PATH,
        'industry_list': list(mappings.get('industry_to_code', {}).keys())
    }