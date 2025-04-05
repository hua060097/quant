#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ray配置测试脚本

用于测试Ray配置是否正常工作，检查系统资源，并提供诊断信息。
"""

import os
import sys
import time
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ray_test")

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    logger.info(f"添加当前目录到路径: {current_dir}")

def test_ray_imports():
    """测试Ray模块导入"""
    logger.info("===== 测试Ray模块导入 =====")
    
    try:
        logger.info("尝试导入ray模块...")
        import ray
        logger.info(f"成功导入ray模块，版本: {ray.__version__}")
        return True
    except ImportError as e:
        logger.error(f"导入ray模块失败: {e}")
        logger.info("请使用以下命令安装Ray: pip install ray==2.0.0 (或更高版本)")
        return False

def test_system_resources():
    """测试系统资源"""
    logger.info("===== 测试系统资源 =====")
    
    try:
        import psutil
        
        # CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        logger.info(f"CPU核心数: {cpu_physical} 物理核心，{cpu_count} 逻辑核心")
        
        # 内存信息
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_available_gb = mem.available / (1024**3)
        logger.info(f"内存: 总计 {mem_total_gb:.2f} GB，可用 {mem_available_gb:.2f} GB")
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        logger.info(f"磁盘: 总计 {disk_total_gb:.2f} GB，可用 {disk_free_gb:.2f} GB")
        
        # 系统负载
        try:
            load = psutil.getloadavg()
            logger.info(f"系统负载: {load}")
        except:
            logger.warning("无法获取系统负载")
        
        return {
            "cpu_count": cpu_count,
            "cpu_physical": cpu_physical,
            "mem_total_gb": mem_total_gb,
            "mem_available_gb": mem_available_gb,
            "disk_total_gb": disk_total_gb,
            "disk_free_gb": disk_free_gb
        }
    except Exception as e:
        logger.error(f"获取系统资源信息失败: {e}")
        return None

def test_ray_config():
    """测试Ray配置模块"""
    logger.info("===== 测试Ray配置模块 =====")
    
    try:
        logger.info("尝试导入ray_config模块...")
        from ray_config import (
            get_optimized_ray_config, 
            init_ray_with_fallback, 
            shutdown_ray,
            get_ray_status
        )
        logger.info("成功导入ray_config模块")
        
        # 获取配置
        logger.info("获取优化的Ray配置...")
        config = get_optimized_ray_config()
        logger.info(f"优化配置: {config}")
        
        return True
    except ImportError as e:
        logger.error(f"导入ray_config模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"测试ray_config模块失败: {e}")
        logger.error(traceback.format_exc())
        return False

def test_ray_initialization():
    """测试Ray初始化"""
    logger.info("===== 测试Ray初始化 =====")
    
    try:
        from ray_config import init_ray_with_fallback, shutdown_ray, get_ray_status
        
        # 尝试初始化Ray
        logger.info("尝试初始化Ray...")
        start_time = time.time()
        success, config = init_ray_with_fallback(verbose=True)
        elapsed = time.time() - start_time
        
        if success:
            logger.info(f"Ray初始化成功，耗时: {elapsed:.2f}秒")
            
            # 获取Ray状态
            status = get_ray_status()
            logger.info(f"Ray状态: {status}")
            
            # 尝试简单计算
            logger.info("尝试进行简单计算...")
            import ray
            
            @ray.remote
            def add(a, b):
                import socket
                return {
                    "result": a + b,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid()
                }
            
            result = ray.get(add.remote(1, 2))
            logger.info(f"计算结果: {result}")
            
            # 关闭Ray
            logger.info("关闭Ray...")
            shutdown_ray()
            logger.info("Ray已关闭")
            
            return True
        else:
            logger.error(f"Ray初始化失败，错误: {config.get('error', '未知错误')}")
            return False
    except Exception as e:
        logger.error(f"测试Ray初始化失败: {e}")
        logger.error(traceback.format_exc())
        return False

def test_parallel_compute():
    """测试并行计算"""
    logger.info("===== 测试并行计算 =====")
    
    try:
        from ray_config import init_ray_with_fallback, shutdown_ray
        
        # 初始化Ray
        success, _ = init_ray_with_fallback()
        if not success:
            logger.error("Ray初始化失败，无法进行并行计算测试")
            return False
        
        import ray
        import numpy as np
        
        # 创建计算密集型任务
        @ray.remote
        def compute_intensive(size):
            # 创建一个大矩阵并进行计算
            matrix = np.random.rand(size, size)
            result = np.linalg.svd(matrix)
            return {
                "size": size,
                "trace": np.trace(matrix),
                "hostname": ray.util.get_node_ip_address()
            }
        
        # 运行多个并行任务
        logger.info("提交并行计算任务...")
        start_time = time.time()
        
        futures = [compute_intensive.remote(500) for _ in range(4)]
        results = ray.get(futures)
        
        elapsed = time.time() - start_time
        logger.info(f"并行计算完成，耗时: {elapsed:.2f}秒")
        logger.info(f"计算结果: {results}")
        
        # 关闭Ray
        shutdown_ray()
        
        return True
    except Exception as e:
        logger.error(f"测试并行计算失败: {e}")
        logger.error(traceback.format_exc())
        
        # 确保Ray被关闭
        try:
            from ray_config import shutdown_ray
            shutdown_ray()
        except:
            pass
        
        return False

def diagnose_ray_issues():
    """诊断Ray相关问题"""
    logger.info("===== Ray问题诊断 =====")
    
    # 检查临时目录问题
    try:
        import tempfile
        temp_dir = tempfile.gettempdir()
        logger.info(f"临时目录: {temp_dir}")
        
        # 检查临时目录权限
        if os.access(temp_dir, os.W_OK):
            logger.info("临时目录可写")
        else:
            logger.warning("临时目录不可写，这可能导致Ray启动问题")
        
        # 检查Ray临时目录
        ray_temp = os.path.join(temp_dir, "ray")
        if os.path.exists(ray_temp):
            logger.info(f"Ray临时目录已存在: {ray_temp}")
            
            # 检查是否存在过期的Ray会话
            import glob
            sessions = glob.glob(os.path.join(ray_temp, "session_*"))
            if sessions:
                logger.info(f"发现 {len(sessions)} 个Ray会话目录")
                for session in sessions:
                    try:
                        # 尝试清理过期会话
                        import shutil
                        shutil.rmtree(session)
                        logger.info(f"已清理过期会话: {session}")
                    except Exception as e:
                        logger.warning(f"无法清理会话 {session}: {e}")
    except Exception as e:
        logger.error(f"诊断临时目录时出错: {e}")
    
    # 检查网络相关问题
    try:
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        logger.info(f"主机名: {hostname}, IP: {ip}")
        
        # 检查本地端口是否可用
        def check_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                    return True
                except:
                    return False
        
        # 检查Ray常用端口
        common_ports = [6379, 8265, 10001, 8076]
        for port in common_ports:
            available = check_port(port)
            logger.info(f"端口 {port} 可用: {available}")
    except Exception as e:
        logger.error(f"诊断网络问题时出错: {e}")
    
    # 检查环境变量
    try:
        ray_env_vars = [var for var in os.environ if var.startswith("RAY_")]
        if ray_env_vars:
            logger.info("发现Ray相关环境变量:")
            for var in ray_env_vars:
                logger.info(f"  {var} = {os.environ[var]}")
        else:
            logger.info("未发现Ray相关环境变量")
    except Exception as e:
        logger.error(f"诊断环境变量时出错: {e}")

def main():
    """主函数"""
    logger.info("开始Ray配置测试")
    
    # 测试Ray模块导入
    if not test_ray_imports():
        logger.error("Ray模块导入失败，测试终止")
        return
    
    # 测试系统资源
    resources = test_system_resources()
    
    # 基于系统资源提供建议
    if resources:
        if resources["cpu_count"] < 4:
            logger.warning("CPU核心数较少，Ray可能无法高效运行，建议使用串行或进程池模式")
        
        if resources["mem_available_gb"] < 2:
            logger.warning("可用内存不足，可能导致Ray内存不足错误，建议关闭其他应用或增加内存")
    
    # 测试Ray配置模块
    test_ray_config()
    
    # 测试Ray初始化
    test_ray_initialization()
    
    # 测试并行计算
    test_parallel_compute()
    
    # 诊断可能的问题
    diagnose_ray_issues()
    
    logger.info("Ray配置测试完成")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error(traceback.format_exc()) 