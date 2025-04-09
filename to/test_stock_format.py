import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_stock_format")

# 指定路径
minute_dir = Path("to/local_data/minute")

def check_all_partitions():
    """检查所有分区中的000001股票"""
    partitions = [d for d in minute_dir.iterdir() if d.is_dir()]
    logger.info(f"找到 {len(partitions)} 个分区目录")
    
    # 要搜索的目标股票
    target_stocks = ["000001.SZ", "sz.000001", "000001", "000001.XSHE"]
    
    # 检查每个分区
    for partition_dir in partitions:
        partition_file = partition_dir / "minute_data.parquet"
        
        if not partition_file.exists():
            logger.warning(f"分区 {partition_dir.name} 不存在数据文件")
            continue
        
        logger.info(f"检查分区: {partition_dir.name}")
        try:
            # 读取全部股票代码列获取唯一值
            table = pq.read_table(partition_file, columns=["stock_code"])
            df = table.to_pandas()
            
            unique_stocks = df["stock_code"].unique()
            # 记录前几个作为样本
            logger.info(f"  股票代码样例: {unique_stocks[:5]}")
            
            # 检查是否包含目标股票
            for stock in target_stocks:
                count = (df["stock_code"] == stock).sum()
                if count > 0:
                    logger.info(f"  找到股票 '{stock}'，出现 {count} 次")
            
            # 检查是否有以000001开头的股票代码
            pattern_stocks = [s for s in unique_stocks if "000001" in s]
            if pattern_stocks:
                logger.info(f"  包含000001的股票代码: {pattern_stocks}")
        
        except Exception as e:
            logger.error(f"检查分区 {partition_dir.name} 时出错: {e}")
            continue

def check_stock_format():
    """检查分区文件中的股票代码格式"""
    partition_path = minute_dir / "202301" / "minute_data.parquet"
    logger.info(f"检查文件: {partition_path}")
    
    if not partition_path.exists():
        logger.error(f"文件不存在: {partition_path}")
        return
    
    try:
        # 只读取必要的列
        table = pq.read_table(partition_path, columns=["code", "stock_code"])
        df = table.to_pandas()
        
        # 输出基本信息
        logger.info(f"读取成功，数据包含 {len(df)} 行")
        logger.info(f"列名: {df.columns.tolist()}")
        
        # 检查股票代码列
        if "stock_code" in df.columns:
            unique_stocks = df["stock_code"].unique()
            logger.info(f"股票代码样例 (前20个): {unique_stocks[:20]}")
            logger.info(f"股票代码总数: {len(unique_stocks)}")
            
            # 寻找特定股票
            target_stocks = ["000001.SZ", "sz.000001", "000001"]
            for stock in target_stocks:
                count = (df["stock_code"] == stock).sum()
                logger.info(f"股票 '{stock}' 出现次数: {count}")
        
        # 检查code列
        if "code" in df.columns:
            unique_codes = df["code"].unique()
            logger.info(f"code列样例 (前20个): {unique_codes[:20]}")
            logger.info(f"code列总数: {len(unique_codes)}")
            
            # 寻找特定股票
            target_stocks = ["000001.SZ", "sz.000001", "000001"]
            for stock in target_stocks:
                count = (df["code"] == stock).sum()
                logger.info(f"code '{stock}' 出现次数: {count}")
        
        # 检查两列的关系
        if "stock_code" in df.columns and "code" in df.columns:
            sample = df.head(10)
            logger.info("stock_code与code列对应关系:")
            for _, row in sample.iterrows():
                logger.info(f"  stock_code: '{row['stock_code']}', code: '{row['code']}'")
    
    except Exception as e:
        logger.error(f"检查过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # check_stock_format()
    check_all_partitions() 