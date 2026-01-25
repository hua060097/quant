"""
RQAlpha 5分钟数据源Mod模块

用于将MinuteKDataSource集成到RQAlpha系统中。

作者: Claude Code Assistant
日期: 2025-11-22
"""

import logging
from rqalpha.interface import AbstractMod
from rqalpha.const import DEFAULT_ACCOUNT_TYPE
from .minute_data_source import MinuteKDataSource

logger = logging.getLogger(__name__)


class MinuteDataMod(AbstractMod):
    """
    5分钟数据源Mod

    配置和使用5分钟K线数据源
    """

    def __init__(self):
        super(MinuteDataMod, self).__init__()
        self.data_source = None
        self.env = None

    def start_up(self, env, mod_config):
        """
        Mod启动函数

        Args:
            env: RQAlpha环境对象
            mod_config: Mod配置参数
        """
        self.env = env

        # 获取数据路径配置
        data_path = mod_config.get('data_path', '../local_data/minute/')
        cache_size = mod_config.get('cache_size', 100)

        logger.info(f"启动5分钟数据源Mod，数据路径: {data_path}, 缓存大小: {cache_size}")

        try:
            # 创建5分钟数据源
            self.data_source = MinuteKDataSource(data_path, cache_size)

            # 设置数据源到环境中
            env.set_data_source(self.data_source)

            logger.info("5分钟数据源Mod启动成功")

        except Exception as e:
            logger.error(f"5分钟数据源Mod启动失败: {e}")
            raise

    def tear_down(self, code, exception=None):
        """
        Mod清理函数

        Args:
            code: 退出代码
            exception: 异常信息（如果有）
        """
        logger.info("5分钟数据源Mod正在关闭...")

        if self.data_source:
            # 清理数据源缓存
            if hasattr(self.data_source, 'data_cache'):
                self.data_source.data_cache.clear()
            logger.info("数据源缓存已清理")

        if exception:
            logger.error(f"Mod退出时发生异常: {exception}")

        logger.info("5分钟数据源Mod已关闭")

    def get_mod_config_schema(self):
        """
        返回Mod配置模式

        Returns:
            dict: 配置参数定义
        """
        return {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "5分钟数据存储路径",
                    "default": "../local_data/minute/"
                },
                "cache_size": {
                    "type": "integer",
                    "description": "数据缓存股票数量",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 1000
                }
            }
        }

    def get_default_mod_config(self):
        """
        返回Mod默认配置

        Returns:
            dict: 默认配置参数
        """
        return {
            "data_path": "../local_data/minute/",
            "cache_size": 100
        }