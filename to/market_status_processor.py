def get_market_status_data(self, start_date, end_date):
    """获取指定日期范围的市场状态数据
    
    Args:
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        
    Returns:
        市场状态数据字典 {日期: 状态数据}，如果没有找到数据则返回空字典
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import json
    
    logger.info(f"获取市场状态数据: {start_date} - {end_date}")
    
    # 标准化日期格式
    if isinstance(start_date, str):
        start_date = start_date.replace('-', '')
    
    if isinstance(end_date, str):
        end_date = end_date.replace('-', '')
    
    try:
        # 尝试从文件加载数据
        metadata_dir = self.metadata_dir
        
        # 检查元数据目录是否存在
        if not os.path.exists(metadata_dir):
            logger.warning(f"元数据目录不存在: {metadata_dir}")
            return {}
        
        # 查找可用的元数据文件
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        
        if not metadata_files:
            logger.warning(f"没有找到市场状态元数据文件")
            return {}
        
        # 加载并合并数据
        market_status_data = {}
        
        for file_name in metadata_files:
            file_path = os.path.join(metadata_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                if isinstance(file_data, dict):
                    market_status_data.update(file_data)
            except Exception as e:
                logger.error(f"加载元数据文件 {file_path} 时出错: {str(e)}")
        
        # 过滤日期范围
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        filtered_data = {}
        for date_str, status in market_status_data.items():
            try:
                date_dt = datetime.strptime(date_str, '%Y%m%d')
                if start_dt <= date_dt <= end_dt:
                    filtered_data[date_str] = status
            except:
                pass
        
        if not filtered_data:
            logger.warning(f"过滤后没有找到指定日期范围内的市场状态数据")
            return {}
        
        logger.info(f"成功加载 {len(filtered_data)} 个日期的市场状态数据")
        return filtered_data
        
    except Exception as e:
        logger.error(f"获取市场状态数据时出错: {str(e)}")
        return {} 