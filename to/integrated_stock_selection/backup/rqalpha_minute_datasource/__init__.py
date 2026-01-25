"""
RQAlpha 5分钟数据源包

提供5分钟K线数据的RQAlpha数据源支持。

使用方法:
    from rqalpha_minute_datasource import load_mod
    mod = load_mod()

作者: Claude Code Assistant
日期: 2025-11-22
"""

from .mod import MinuteDataMod
from .minute_data_source import MinuteKDataSource

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"


def load_mod():
    """
    加载5分钟数据源Mod的入口函数

    RQAlpha会调用此函数来加载Mod

    Returns:
        MinuteDataMod: Mod实例
    """
    return MinuteDataMod()


__all__ = [
    'load_mod',
    'MinuteDataMod',
    'MinuteKDataSource'
]