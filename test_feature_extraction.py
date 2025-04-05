import logging
import sys
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置日志
os.makedirs('to/log', exist_ok=True)  # 确保日志目录存在
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('to/log/feature_extraction_test.log')
    ]
)

logger = logging.getLogger('test_features')

def test_feature_extraction():
    """测试特征提取功能"""
    try:
        logger.info("=" * 60)
        logger.info("开始特征提取测试")
        logger.info("=" * 60)
        
        # 导入模块
        logger.info("添加'to'目录到模块搜索路径...")
        to_dir = os.path.join(os.getcwd(), 'to')
        sys.path.insert(0, to_dir)
        logger.info(f"添加模块路径: {to_dir}")
        
        logger.info("导入AdvancedQuantModel...")
        from advanced_quant_model import AdvancedQuantModel
        
        # 创建模型实例
        logger.info("创建模型实例...")
        model = AdvancedQuantModel()
        
        # 定义测试参数
        test_cases = [
            {
                "name": "正常A股测试",
                "stocks": ['000001.SZ', '600000.SH', '600036.SH', '601318.SH', '600519.SH', 
                           '000651.SZ', '000333.SZ', '000858.SZ', '002230.SZ', '300059.SZ', 
                           '600276.SH', '002415.SZ'],
                "date": "20240320",
                "window": 30,
                "threshold": 5
            },
            {
                "name": "少量股票测试",
                "stocks": ['000001.SZ', '600000.SH'],
                "date": "20240320",
                "window": 30,
                "threshold": 1  # 降低阈值使其能够处理少量股票
            },
            {
                "name": "包含新上市股票测试",
                # 添加一些可能是新上市的股票
                "stocks": ['000001.SZ', '600000.SH', '688981.SH', '301506.SZ', '301503.SZ'],
                "date": "20240320",
                "window": 30,
                "threshold": 1
            },
            {
                "name": "长窗口测试",
                "stocks": ['000001.SZ', '600000.SH', '600036.SH', '601318.SH'],
                "date": "20240320",
                "window": 60,
                "threshold": 1
            },
            {
                "name": "历史日期测试",
                "stocks": ['000001.SZ', '600000.SH', '600036.SH', '601318.SH'],
                "date": "20230101",  # 历史日期
                "window": 30,
                "threshold": 1
            }
        ]
        
        # 运行测试
        results = []
        for idx, test_case in enumerate(test_cases):
            logger.info(f"运行测试 {idx+1}/{len(test_cases)}: {test_case['name']}")
            
            start_time = time.time()
            try:
                features, labels = model._extract_features(
                    test_case['stocks'],
                    test_case['date'],
                    test_case['window'],
                    test_case['threshold']
                )
                
                # 记录结果
                elapsed = time.time() - start_time
                feature_shape = str(features.shape) if features is not None else "None"
                labels_shape = str(labels.shape) if labels is not None else "None"
                
                result = {
                    "测试名称": test_case['name'],
                    "状态": "成功" if features is not None else "失败",
                    "特征形状": feature_shape,
                    "标签形状": labels_shape,
                    "耗时(秒)": round(elapsed, 2)
                }
                results.append(result)
                
                logger.info(f"测试结果: {result}")
                
                # 如果特征不为空，分析特征数据
                if features is not None and len(features) > 0:
                    logger.info(f"特征统计信息:")
                    # 计算每个特征的基本统计信息
                    features_df = pd.DataFrame(features)
                    stats = features_df.describe().T
                    logger.info(f"特征数量: {features_df.shape[1]}")
                    logger.info(f"非空值比例: {features_df.count().mean() / len(features_df):.2%}")
                    logger.info(f"均值范围: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
                    logger.info(f"标准差范围: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
                    
                    # 检查是否有无穷值或NaN
                    has_inf = np.isinf(features).any()
                    has_nan = np.isnan(features).any()
                    logger.info(f"包含无穷值: {has_inf}")
                    logger.info(f"包含NaN值: {has_nan}")
                    
                    if has_nan:
                        nan_count = np.isnan(features).sum()
                        logger.warning(f"NaN值数量: {nan_count} ({nan_count/(features.size)*100:.2f}%)")
            
            except Exception as e:
                logger.error(f"测试出错: {str(e)}", exc_info=True)
                results.append({
                    "测试名称": test_case['name'],
                    "状态": "错误",
                    "特征形状": "N/A",
                    "标签形状": "N/A",
                    "耗时(秒)": round(time.time() - start_time, 2),
                    "错误": str(e)
                })
        
        # 输出汇总结果
        logger.info("\n" + "=" * 60)
        logger.info("特征提取测试汇总")
        logger.info("=" * 60)
        
        # 创建结果DataFrame并显示
        results_df = pd.DataFrame(results)
        logger.info("\n" + results_df.to_string())
        
        # 计算成功率
        success_rate = sum(1 for r in results if r['状态'] == '成功') / len(results)
        logger.info(f"\n测试成功率: {success_rate:.2%}")
        
        logger.info("=" * 60)
        logger.info("特征提取测试完成")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"测试主程序发生错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_feature_extraction() 