import concurrent.futures
import os
import multiprocessing as mp
from threading import Lock
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial
import time
import mmap
import pickle
import tempfile
import json
import psutil
import math
import queue  # 确保正确导入queue模块

# Set the multiprocessing start method to 'spawn' to avoid deadlock issues
# This is especially important on Windows
mp.set_start_method('spawn', force=True)

# Increase recursion limit for pickle to handle large objects
import sys
sys.setrecursionlimit(10000)

# 在模块顶部定义全局变量
_g_feature_weights = None
_g_positive_patterns = None
_g_negative_patterns = None
_g_clusters = None
_g_cluster_analysis = None

# 添加NumPy类型的JSON序列化支持
class NumpyEncoder(json.JSONEncoder):
    """用于JSON序列化含NumPy数据类型的数据"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if obj is np.nan:  # 处理NaN值
            return None
        if pd.isna(obj):  # 处理pandas的NA值
            return None
        return super(NumpyEncoder, self).default(obj)

# 确保所有返回结果都不含NumPy类型
def safe_convert_numpy_types(obj):
    """递归转换所有NumPy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: safe_convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [safe_convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

def init_worker(feature_weights, positive_patterns, negative_patterns, clusters, cluster_analysis):
    """工作进程初始化函数，将共享数据加载为全局变量
    
    Args:
        feature_weights: 特征权重字典
        positive_patterns: 正向模式列表
        negative_patterns: 负向模式列表
        clusters: 聚类信息
        cluster_analysis: 聚类分析结果
    """
    global _g_feature_weights, _g_positive_patterns, _g_negative_patterns, _g_clusters, _g_cluster_analysis
    
    # 将传入的数据赋值给全局变量
    _g_feature_weights = feature_weights
    _g_positive_patterns = positive_patterns
    _g_negative_patterns = negative_patterns
    _g_clusters = clusters
    _g_cluster_analysis = cluster_analysis
    
    # 记录内存使用情况
    worker_memory = get_memory_usage()
    print(f"工作进程初始化完成，当前内存使用: {worker_memory:.2f} MB")

# Process stocks in batches for better parallelism
def process_stock_batch(stock_batch, date_str, feature_weights=None, positive_patterns=None, negative_patterns=None, clusters=None, cluster_analysis=None, 
                       feature_weights_path=None, positive_patterns_path=None, negative_patterns_path=None, 
                       clusters_path=None, cluster_analysis_path=None):
    """Process a batch of stocks for a single date
    
    Args:
        stock_batch: DataFrame containing a batch of stocks for the date
        date_str: Date string
        feature_weights: Feature weight dictionary (fallback)
        positive_patterns: List of positive patterns (fallback)
        negative_patterns: List of negative patterns (fallback)
        clusters: Cluster information (fallback)
        cluster_analysis: Cluster analysis results (fallback)
        feature_weights_path: Path to memory mapped feature weights
        positive_patterns_path: Path to memory mapped positive patterns
        negative_patterns_path: Path to memory mapped negative patterns
        clusters_path: Path to memory mapped clusters
        cluster_analysis_path: Path to memory mapped cluster analysis
    
    Returns:
        dict: Prediction results for the batch
    """
    global _g_feature_weights, _g_positive_patterns, _g_negative_patterns, _g_clusters, _g_cluster_analysis
    
    # 1. 首先尝试使用全局变量
    if feature_weights is None and _g_feature_weights is not None:
        feature_weights = _g_feature_weights
    if positive_patterns is None and _g_positive_patterns is not None:
        positive_patterns = _g_positive_patterns
    if negative_patterns is None and _g_negative_patterns is not None:
        negative_patterns = _g_negative_patterns
    if clusters is None and _g_clusters is not None:
        clusters = _g_clusters
    if cluster_analysis is None and _g_cluster_analysis is not None:
        cluster_analysis = _g_cluster_analysis
    
    # 2. 如果全局变量不可用，尝试从内存映射加载数据
    if feature_weights is None and feature_weights_path and os.path.exists(feature_weights_path):
        feature_weights = load_from_mmap(feature_weights_path)
        
    if positive_patterns is None and positive_patterns_path and os.path.exists(positive_patterns_path):
        positive_patterns = load_from_mmap(positive_patterns_path)
        
    if negative_patterns is None and negative_patterns_path and os.path.exists(negative_patterns_path):
        negative_patterns = load_from_mmap(negative_patterns_path)
        
    if clusters is None and clusters_path and os.path.exists(clusters_path):
        clusters = load_from_mmap(clusters_path)
        
    if cluster_analysis is None and cluster_analysis_path and os.path.exists(cluster_analysis_path):
        cluster_analysis = load_from_mmap(cluster_analysis_path)
        
    batch_predictions = {}
    
    # 验证参数是否正常加载
    if not feature_weights:
        print(f"警告: feature_weights为空或未正确加载")
    if not positive_patterns:
        print(f"警告: positive_patterns为空或未正确加载")
    if not negative_patterns:
        print(f"警告: negative_patterns为空或未正确加载")
    
    for _, stock_row in stock_batch.iterrows():
        
        stock_code = stock_row['stock_code']
        stock_name = stock_row['stock_name'] if 'stock_name' in stock_row else ''
        print(f"date: {date_str}, stock_code: {stock_code}")
        # print(f"feature_weights: {feature_weights}, positive_patterns: {positive_patterns}, negative_patterns: {negative_patterns}, clusters: {clusters}, cluster_analysis: {cluster_analysis}")
        
        # Extract stock features - use vectorized operations
        stock_features = {}
        for col in stock_row.index:
            if col not in ['date_str', 'stock_code', 'stock_name']:
                try:
                    # 确保值是数值类型
                    value = stock_row[col]
                    if isinstance(value, (int, float, np.number)) and not pd.isna(value):
                        stock_features[col] = float(value)
                except Exception:
                    continue
        
        # 确保有足够的特征
        if len(stock_features) < 10:
            continue
            
        # 1. Calculate similarities with positive patterns
        positive_similarities = []
        positive_returns = []
        positive_pattern_details = []
        
        for pattern in positive_patterns:
            if not isinstance(pattern, dict):
                continue
                
            # 确保pattern包含必需的字段
            if 'date_str' not in pattern or pattern['date_str'] is None:
                continue
                
            # Calculate weighted similarity
            similarity, matched_features, feature_contribs = calculate_weighted_similarity(
                stock_features, pattern, feature_weights)
            
            if similarity > 0.5:  # Only consider patterns with similarity > 0.5
                positive_similarities.append(similarity)
                future_return = pattern.get('future_return_2d', 0)
                if not isinstance(future_return, (int, float)) or pd.isna(future_return):
                    future_return = 0
                positive_returns.append(future_return)
                
                # 确保模式细节的值是安全的JSON类型
                pattern_detail = {
                    'similarity': float(similarity),
                    'return': float(future_return),
                    'matched_features': matched_features,
                    'feature_contributions': feature_contribs,
                    'date': pattern.get('date_str', ''),
                    'stock_code': pattern.get('stock_code', ''),
                    'tech_combo_state': float(pattern.get('tech_combo_state', 0)) if pattern.get('tech_combo_state') is not None else 0,
                    'market_group': int(pattern.get('market_group', 0)) if pattern.get('market_group') is not None else 0
                }
                positive_pattern_details.append(pattern_detail)
                
                # 减少内存使用: 只保留前10个匹配项以节省内存
                if len(positive_pattern_details) >= 10:
                    break
        
        # 2. Calculate similarities with negative patterns
        negative_similarities = []
        negative_returns = []
        negative_pattern_details = []
        
        for pattern in negative_patterns:
            if not isinstance(pattern, dict):
                continue
                
            # 确保pattern包含必需的字段
            if 'date_str' not in pattern or pattern['date_str'] is None:
                continue
            
            similarity, matched_features, feature_contribs = calculate_weighted_similarity(
                stock_features, pattern, feature_weights)
            
            if similarity > 0.5:
                negative_similarities.append(similarity)
                future_return = pattern.get('future_return_2d', 0)
                if not isinstance(future_return, (int, float)) or pd.isna(future_return):
                    future_return = 0
                negative_returns.append(future_return)
                
                # 确保模式细节的值是安全的JSON类型
                pattern_detail = {
                    'similarity': float(similarity),
                    'return': float(future_return),
                    'matched_features': matched_features,
                    'feature_contributions': feature_contribs,
                    'date': pattern.get('date_str', ''),
                    'stock_code': pattern.get('stock_code', ''),
                    'tech_combo_state': float(pattern.get('tech_combo_state', 0)) if pattern.get('tech_combo_state') is not None else 0,
                    'market_group': int(pattern.get('market_group', 0)) if pattern.get('market_group') is not None else 0
                }
                negative_pattern_details.append(pattern_detail)
                
                # 减少内存使用: 只保留前10个匹配项以节省内存
                # if len(negative_pattern_details) >= 10:
                #     break
        
        # 3. Cluster-based prediction - vectorized calculation
        cluster_prediction = 0
        cluster_confidence = 0
        
        if clusters and 'feature_names' in cluster_analysis:
            cluster_features = cluster_analysis['feature_names']
            stock_cluster_vector = np.array([
                float(stock_row[feature]) if feature in stock_row else 0
                for feature in cluster_features if feature != 'future_return_2d'
            ])
            
            if len(stock_cluster_vector) > 0:
                best_cluster = None
                best_similarity = -1
                
                for cluster_id, cluster_info in clusters.items():
                    cluster_center = np.array(cluster_info.get('center', []))
                    # Exclude future_return_2d feature
                    center_vector = np.array([
                        x for i, x in enumerate(cluster_center) 
                        if i < len(cluster_features) and cluster_features[i] != 'future_return_2d'
                    ])
                    
                    # Ensure vectors have consistent length
                    min_len = min(len(stock_cluster_vector), len(center_vector))
                    if min_len > 0:
                        # Calculate cosine similarity - vectorized
                        dot_product = np.dot(stock_cluster_vector[:min_len], center_vector[:min_len])
                        norm_product = np.linalg.norm(stock_cluster_vector[:min_len]) * np.linalg.norm(center_vector[:min_len])
                        similarity = dot_product / norm_product if norm_product > 0 else 0
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster = cluster_id
                
                # If best cluster found, use its average return as prediction
                if best_cluster is not None and best_similarity > 0.6:
                    cluster_prediction = clusters[best_cluster].get('avg_return', 0)
                    cluster_confidence = best_similarity
        
        # 4. Calculate weighted expected return - vectorized calculation
        expected_return = 0
        probability = 0.5  # Default probability
        confidence = 0
        
        # Based on similar patterns - use numpy vectorized operations
        pos_expected, pos_confidence = 0, 0
        if positive_similarities:
            pos_similarities_array = np.array(positive_similarities)
            pos_returns_array = np.array(positive_returns)
            pos_weights = pos_similarities_array / np.sum(pos_similarities_array) if np.sum(pos_similarities_array) > 0 else np.zeros_like(pos_similarities_array)
            pos_expected = np.sum(pos_returns_array * pos_weights) if np.sum(pos_weights) > 0 else 0
            pos_confidence = np.mean(pos_similarities_array) if positive_similarities else 0
            
        neg_expected, neg_confidence = 0, 0
        if negative_similarities:
            neg_similarities_array = np.array(negative_similarities)
            neg_returns_array = np.array(negative_returns)
            neg_weights = neg_similarities_array / np.sum(neg_similarities_array) if np.sum(neg_similarities_array) > 0 else np.zeros_like(neg_similarities_array)
            neg_expected = np.sum(neg_returns_array * neg_weights) if np.sum(neg_weights) > 0 else 0
            neg_confidence = np.mean(neg_similarities_array) if negative_similarities else 0
        
        # Combine similarity-based scores
        similarity_based_return = 0
        similarity_confidence = 0
        
        if pos_confidence + neg_confidence > 0:
            similarity_based_return = (pos_expected * pos_confidence + neg_expected * neg_confidence) / (pos_confidence + neg_confidence)
            probability = pos_confidence / (pos_confidence + neg_confidence) if pos_confidence > 0 else 0
            similarity_confidence = max(pos_confidence, neg_confidence)
        
        # Combine both prediction methods
        if similarity_confidence > 0 or cluster_confidence > 0:
            total_confidence = similarity_confidence + cluster_confidence
            if total_confidence > 0:
                expected_return = (similarity_based_return * similarity_confidence + 
                                cluster_prediction * cluster_confidence) / total_confidence
                confidence = (similarity_confidence + cluster_confidence) / 2
            else:
                expected_return = similarity_based_return
                confidence = similarity_confidence
        
        # 5-7. 调整因素部分保持不变，但添加类型转换
        # 5. Adjust based on current market environment
        if 'market_group' in stock_row:
            try:
                market_group = int(stock_row['market_group'])
                # Adjust expected return based on market conditions
                market_adjustment = {
                    1: 1.2,   # Bull market: amplify positive prediction
                    2: 0.8,   # Bear market: reduce positive prediction
                    3: 1.0,   # Oscillating market: no adjustment
                    4: 0.7    # Extreme market: reduce prediction certainty
                }.get(market_group, 1.0)
                
                # Only adjust positive expectations
                if expected_return > 0:
                    expected_return *= market_adjustment
                    
                # Reduce confidence in extreme market conditions
                if market_group == 4:
                    confidence *= 0.9
            except Exception as e:
                # 忽略市场环境调整的错误
                pass
        
        # 6. Adjust based on sector structure
        if 'board_heat_index' in stock_row and expected_return > 0:
            try:
                board_heat = float(stock_row['board_heat_index'])
                if board_heat > 60:  # High heat sectors
                    expected_return *= 1.1
                    confidence *= 1.05
            except Exception:
                pass
        
        if 'limit_up_ratio' in stock_row and expected_return > 0:
            try:
                limit_ratio = float(stock_row['limit_up_ratio'])
                if limit_ratio > 0.15:  # >15% limit up ratio
                    expected_return *= 1.15
                    confidence *= 1.05
            except Exception:
                pass
        
        if 'broken_ratio' in stock_row and expected_return > 0:
            try:
                broken_ratio = float(stock_row['broken_ratio'])
                if broken_ratio > 0.3:  # >30% broken ratio
                    expected_return *= 0.9
                    confidence *= 0.95
            except Exception:
                pass
        
        # 7. Adjust based on stock technical indicators
        if 'tech_combo_state' in stock_row:
            try:
                tech_state = float(stock_row['tech_combo_state'])
                if tech_state > 2 and expected_return > 0:
                    expected_return *= 1.1
                    confidence *= 1.05
                elif tech_state < -2 and expected_return < 0:
                    expected_return *= 1.1
                    confidence *= 1.05
            except Exception:
                pass
        
        # Combine other key technical indicators
        for composite_feature in ['rsi_macd_bull_signal', 'oversold_macd_bullish', 'strong_bull_consensus']:
            if composite_feature in stock_row and stock_row[composite_feature] == 1 and expected_return > 0:
                expected_return *= 1.05
        
        for composite_feature in ['rsi_macd_bear_signal', 'overbought_macd_bearish', 'strong_bear_consensus']:
            if composite_feature in stock_row and stock_row[composite_feature] == 1 and expected_return < 0:
                expected_return *= 1.05
        
        # 8. 图表模式调整部分保持不变，但添加安全的类型转换
        
        # Analyze feature contributions
        important_features = {}
        if positive_pattern_details:
            important_pos_features = analyze_feature_contributions(positive_pattern_details)
            for feature, importance in important_pos_features.items():
                important_features[f"pos_{feature}"] = float(importance)
                
        if negative_pattern_details:
            important_neg_features = analyze_feature_contributions(negative_pattern_details)
            for feature, importance in important_neg_features.items():
                important_features[f"neg_{feature}"] = float(importance)
        
        # 确保数值都是Python原生类型，避免JSON序列化问题
        prediction_key = f"{stock_code}_{date_str}"
        prediction_result = {
            'stock_code': str(stock_code),
            'stock_name': str(stock_name),
            'date': str(date_str),
            'expected_return': float(expected_return),
            'probability': float(probability),
            'confidence': float(confidence),
            'score': float(expected_return * probability * confidence),  # Combined score
            'positive_matches': int(len(positive_similarities)),
            'negative_matches': int(len(negative_similarities)),
            'cluster_prediction': float(cluster_prediction),
            'important_features': important_features,
            'positive_pattern_details': positive_pattern_details[:3] if positive_pattern_details else [],
            'negative_pattern_details': negative_pattern_details[:3] if negative_pattern_details else []
        }
        
        # 确保结果不含NumPy类型
        prediction_result = safe_convert_numpy_types(prediction_result)
        batch_predictions[prediction_key] = prediction_result
    
    return batch_predictions


# Process a single date's stock data
def process_single_date(date_str, daily_features=None, feature_weights_path=None, positive_patterns_path=None, 
                       negative_patterns_path=None, clusters_path=None, cluster_analysis_path=None,
                       feature_weights=None, positive_patterns=None, negative_patterns=None, 
                       clusters=None, cluster_analysis=None):
    """Process stock features for a single date
    
    Args:
        date_str: Date string to process
        daily_features: DataFrame with stock features for the date
        feature_weights_path: Path to memory mapped feature weights (可能是元组或字符串)
        positive_patterns_path: Path to memory mapped positive patterns (可能是元组或字符串)
        negative_patterns_path: Path to memory mapped negative patterns (可能是元组或字符串)
        clusters_path: Path to memory mapped clusters (可能是元组或字符串)
        cluster_analysis_path: Path to memory mapped cluster analysis (可能是元组或字符串)
        feature_weights: Feature weight dictionary
        positive_patterns: List of positive patterns
        negative_patterns: List of negative patterns
        clusters: Cluster information
        cluster_analysis: Cluster analysis results
        
    Returns:
        dict: Dictionary of prediction results
    """
    global _g_feature_weights, _g_positive_patterns, _g_negative_patterns, _g_clusters, _g_cluster_analysis
    
    print(f"开始处理日期 {date_str}")
    start_time = time.time()
    
    # 1. Use global variables if available
    if feature_weights is None and _g_feature_weights is not None:
        feature_weights = _g_feature_weights
    if positive_patterns is None and _g_positive_patterns is not None:
        positive_patterns = _g_positive_patterns
    if negative_patterns is None and _g_negative_patterns is not None:
        negative_patterns = _g_negative_patterns
    if clusters is None and _g_clusters is not None:
        clusters = _g_clusters
    if cluster_analysis is None and _g_cluster_analysis is not None:
        cluster_analysis = _g_cluster_analysis
    
    # print(f"feature_weights: {feature_weights}")
    print(f"feature_weights_path: {feature_weights_path}")
    
    # 处理路径参数可能是元组的情况
    if isinstance(feature_weights_path, tuple):
        feature_weights_path = feature_weights_path[0]
    if isinstance(positive_patterns_path, tuple):
        positive_patterns_path = positive_patterns_path[0]
    if isinstance(negative_patterns_path, tuple):
        negative_patterns_path = negative_patterns_path[0]
    if isinstance(clusters_path, tuple):
        clusters_path = clusters_path[0]
    if isinstance(cluster_analysis_path, tuple):
        cluster_analysis_path = cluster_analysis_path[0]
    
    # 2. Load shared objects from memory map if needed
    if feature_weights is None and feature_weights_path and os.path.exists(feature_weights_path):
        print(f"加载特征权重文件: {feature_weights_path}")
        try:
            feature_weights = load_from_mmap(feature_weights_path)
            print(f"特征权重加载成功，包含 {len(feature_weights) if feature_weights else 0} 个特征")
        except Exception as e:
            print(f"加载特征权重失败: {e}")
    
    if positive_patterns is None and positive_patterns_path and os.path.exists(positive_patterns_path):
        try:
            positive_patterns = load_from_mmap(positive_patterns_path)
            print(f"正向模式加载成功，包含 {len(positive_patterns) if positive_patterns else 0} 个模式")
        except Exception as e:
            print(f"加载正向模式失败: {e}")
    
    if negative_patterns is None and negative_patterns_path and os.path.exists(negative_patterns_path):
        try:
            negative_patterns = load_from_mmap(negative_patterns_path)
            print(f"负向模式加载成功，包含 {len(negative_patterns) if negative_patterns else 0} 个模式")
        except Exception as e:
            print(f"加载负向模式失败: {e}")
        
    if clusters is None and clusters_path and os.path.exists(clusters_path):
        try:
            clusters = load_from_mmap(clusters_path)
        except Exception as e:
            print(f"加载聚类数据失败: {e}")
    
    if cluster_analysis is None and cluster_analysis_path and os.path.exists(cluster_analysis_path):
        try:
            cluster_analysis = load_from_mmap(cluster_analysis_path)
        except Exception as e:
            print(f"加载聚类分析失败: {e}")
    
    # 验证数据加载
    if not isinstance(feature_weights, dict) or not feature_weights:
        logging.warning(f"日期 {date_str} 的特征权重为空或格式错误")
        return {}
        
    if not isinstance(positive_patterns, list) or not positive_patterns:
        logging.warning(f"日期 {date_str} 的正向模式为空或格式错误")
        return {}
        
    if not isinstance(negative_patterns, list) or not negative_patterns:
        logging.warning(f"日期 {date_str} 的负向模式为空或格式错误")
        return {}
    
    if daily_features is None or daily_features.empty:
        logging.warning(f"日期 {date_str} 没有有效特征数据")
        return {}
    
    # 3. Process stocks for this date - 改用线程池而不是进程池，避免daemon进程问题
    results = {}
    try:
        # 添加日期列
        if 'date_str' not in daily_features.columns:
            daily_features['date_str'] = date_str
            
        # 确保stock_code和stock_name列存在
        if 'stock_name' not in daily_features.columns:
            daily_features['stock_name'] = ""

        # 设置日期参数，确保每个处理函数都能获取日期
        if not daily_features.empty:
            # 确定可用的CPU核心数，但线程池使用更多线程是安全的
            num_threads = min(os.cpu_count() * 2, 10)  # 线程数量可以比CPU核心多
            num_stocks = len(daily_features)
            print(f"日期 {date_str} 有 {num_stocks} 只股票需要处理，将使用线程池并行处理")
            
            # 为每个处理批次准备参数
            batch_size = max(20, num_stocks // (num_threads * 2))  # 批次大小
            
            # 使用分组创建处理批次
            batches = []
            for i in range(0, num_stocks, batch_size):
                end_idx = min(i + batch_size, num_stocks)
                batches.append(daily_features.iloc[i:end_idx])
            
            print(f"将 {num_stocks} 只股票分成 {len(batches)} 个批次处理")
            
            # 准备批处理参数
            process_batch_func = partial(
                process_stock_batch,
                date_str=date_str,
                feature_weights=feature_weights,
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                clusters=clusters,
                cluster_analysis=cluster_analysis
            )
            
            # 使用线程池而不是进程池，避免多进程嵌套问题
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # 提交所有批次任务
                    future_to_batch = {}
                    for i, batch in enumerate(batches):
                        future = executor.submit(process_batch_func, batch)
                        future_to_batch[future] = i
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_result = future.result()
                            if batch_result:
                                results.update(batch_result)
                                print(f"日期 {date_str} 的批次 {batch_idx+1}/{len(batches)} 处理完成，新增 {len(batch_result)} 条预测")
                        except Exception as e:
                            print(f"错误: 日期 {date_str} 的批次 {batch_idx+1} 处理出错: {str(e)}")
                            import traceback
                            print(traceback.format_exc())
            except Exception as e:
                print(f"批处理过程中发生严重错误: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                # 回退到串行处理
                print("回退到串行处理模式")
                results = {}
                for i, batch in enumerate(batches):
                    try:
                        batch_result = process_batch_func(batch)
                        if batch_result:
                            results.update(batch_result)
                            print(f"串行模式 - 日期 {date_str} 的批次 {i+1}/{len(batches)} 处理完成")
                    except Exception as e:
                        print(f"串行处理批次 {i+1} 时出错: {e}")
            
            # 检查结果的有效性
            if not results:
                logging.warning(f"日期 {date_str} 的所有股票预测结果为空")
            
            processing_time = time.time() - start_time
            print(f"日期 {date_str} 处理完成，耗时 {processing_time:.2f} 秒，处理了 {len(results)} 只股票")
    except Exception as e:
        logging.error(f"处理日期 {date_str} 时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    # 4. 确保结果不包含NumPy类型
    try:
        results = safe_convert_numpy_types(results)
    except Exception as e:
        logging.error(f"转换预测结果类型时出错: {str(e)}")
    
    # 5. 记录内存使用情况
    log_memory_usage(logging, f"日期 {date_str} 处理完成后")
    
    # 6. 强制清理内存
    release_memory()
    
    return results


# Helper function to calculate weighted similarity
def calculate_weighted_similarity(vec1_dict, vec2_dict, feature_weights=None):
    """Calculate weighted similarity between two feature dictionaries
    
    Args:
        vec1_dict: First feature dictionary
        vec2_dict: Second feature dictionary
        feature_weights: Dictionary of feature weights
        
    Returns:
        tuple: (similarity, matched_features, feature_contributions)
    """
    if not feature_weights:
        feature_weights = {}
    
    # 安全检查 - 确保输入是字典
    if not isinstance(vec1_dict, dict) or not isinstance(vec2_dict, dict):
        return 0, [], {}
        
    # 找出两个特征字典中都有的特征
    common_features = set(vec1_dict.keys()) & set(vec2_dict.keys())
    if not common_features:
        return 0, [], {}
        
    # 计算加权点积和向量模长
    dot_product = 0
    vec1_squared_sum = 0
    vec2_squared_sum = 0
    matched_features = []
    feature_contributions = {}
    
    try:
        for feature in common_features:
            # 获取特征权重
            weight = float(feature_weights.get(feature, 1.0))
            
            # 确保值是浮点数
            try:
                v1 = float(vec1_dict[feature]) if vec1_dict[feature] is not None else 0
                v2 = float(vec2_dict[feature]) if vec2_dict[feature] is not None else 0
            except (ValueError, TypeError):
                continue
            
            # 跳过NaN值
            if pd.isna(v1) or pd.isna(v2) or not np.isfinite(v1) or not np.isfinite(v2):
                continue
                
            # 应用权重
            weighted_v1 = v1 * weight
            weighted_v2 = v2 * weight
            
            # 计算点积和特征贡献
            feature_contrib = weighted_v1 * weighted_v2
            dot_product += feature_contrib
            vec1_squared_sum += weighted_v1 * weighted_v1
            vec2_squared_sum += weighted_v2 * weighted_v2
            
            # 如果特征值相似，添加到匹配特征列表
            if abs(v1 - v2) < 0.5 * (abs(v1) + abs(v2) + 0.0001):  # Consider features similar if values are within 50%
                matched_features.append(feature)
                feature_contributions[feature] = float(feature_contrib)
        
        # 计算余弦相似度
        magnitude = np.sqrt(vec1_squared_sum * vec2_squared_sum)
        similarity = dot_product / magnitude if magnitude > 0 else 0
        
        # 确保相似度在0-1范围内
        similarity = max(0, min(1, similarity))
        
        # 确保返回值是基本类型
        return float(similarity), matched_features, {k: float(v) for k, v in feature_contributions.items()}
    
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return 0, [], {}


# Helper function to analyze feature contributions
def analyze_feature_contributions(pattern_details, top_n=20):
    """Analyze feature contributions from pattern details
    
    Args:
        pattern_details: List of pattern detail dictionaries
        top_n: Number of top features to return
        
    Returns:
        dict: Dictionary of feature importance values
    """
    if not pattern_details:
        return {}
    
    try:
        # 聚合所有模式中的特征贡献
        feature_contributions = {}
        pattern_weights = {}
        
        for pattern in pattern_details:
            if not isinstance(pattern, dict):
                continue
                
            # 获取该模式的权重（基于相似度）
            similarity = pattern.get('similarity', 0)
            if not similarity:
                continue
                
            # 遍历该模式的特征贡献
            for feature, contribution in pattern.get('feature_contributions', {}).items():
                if not feature or pd.isna(contribution):
                    continue
                    
                # 累加加权特征贡献
                if feature not in feature_contributions:
                    feature_contributions[feature] = 0
                    pattern_weights[feature] = 0
                
                try:
                    feature_contributions[feature] += float(similarity * contribution)
                    pattern_weights[feature] += float(similarity)
                except (ValueError, TypeError):
                    continue
        
        # 计算每个特征的平均贡献
        avg_contributions = {}
        for feature, total_contribution in feature_contributions.items():
            weight_sum = pattern_weights.get(feature, 0)
            if weight_sum > 0:
                avg_contributions[feature] = total_contribution / weight_sum
        
        # 获取top_n个特征
        top_features = dict(sorted(avg_contributions.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)[:top_n])
        
        return {k: float(v) for k, v in top_features.items()}
    
    except Exception as e:
        print(f"分析特征贡献时出错: {e}")
        return {}


def predict_stocks_with_change_points_mt(self, predict_features, feature_patterns, relations_df=None):
    """使用变化点的特征模式预测股票表现 - 多进程版本
    Args:
        predict_features: 预测期特征数据
        feature_patterns: 历史特征模式数据
        relations_df: 股票板块关联结果DataFrame，可选参数
        
    Returns:
        dict: 预测结果，股票代码为键，预测信息为值
    """
    print(f"feature_patterns原始内容: {feature_patterns.keys()}")
    print(f"正向模式数量: {len(feature_patterns.get('positive_patterns', []))}")
    print(f"负向模式数量: {len(feature_patterns.get('negative_patterns', []))}")
    print(f"特征重要性数量: {len(feature_patterns.get('feature_importance', {}))}")
    start_time = time.time()
    self.logger.info(f"开始预测 {len(predict_features)} 只股票的表现...")
    
    # 记录初始内存使用情况
    log_memory_usage(self.logger, "预测开始时内存使用")
    
    # 检查是否提供了股票板块关联数据
    if relations_df is not None and not relations_df.empty:
        self.logger.info(f"使用单独的股票板块关联数据(包含{len(relations_df)}条记录)进行预测")
    
    if predict_features.empty or not feature_patterns:
        self.logger.warning("预测数据或特征模式为空，无法进行预测")
        return {}
        
    # 确保交易日历加载
    if self.trading_days is None:
        self._load_trading_calendar()
        
    # 对所有日期的数据进行预测，而不仅仅是最新日期
    unique_dates = sorted(predict_features['date_str'].unique())
    
    # 获取当前预测日期范围内的实际交易日
    trading_days = [day for day in self.trading_days if day in unique_dates]
    
    # === 临时测试修改：只取前5个交易日 ===
    # 要恢复完整预测逻辑，只需注释掉下面这一行
    trading_days = trading_days[:5]  # 临时限制为5个日期进行测试
    # === 临时测试修改结束 ===
    
    if not trading_days:
        self.logger.warning("预测日期范围内没有有效交易日")
        return {}
        
    self.logger.info(f"将对 {len(trading_days)} 个交易日的数据进行预测: {', '.join(trading_days)}")
    
    # 提取训练数据中的聚类信息
    clusters = feature_patterns.get('clusters', {})
    cluster_analysis = feature_patterns.get('cluster_analysis', {})
    
    # 设置特征权重 - 可以根据经验或特征重要性分析设置
    # 初始化基础权重
    feature_weights = {
        # 技术指标权重
        'rsi14': 1.5,
        'macd_line': 1.2,
        'momentum_10d': 1.8,
        'volatility_20d': 0.8,
        'volume_ratio_5': 1.3,
        
        # 趋势指标权重
        'trend_direction_score': 2.0,
        'ma_bull_trend': 1.5,
        'ma_bear_trend': 1.5,
        
        # 动量指标权重
        'momentum_5d': 1.7,
        'momentum_20d': 1.5,
        
        # 价格指标权重
        'price_ma5_ratio': 1.3,
        'price_ma20_ratio': 1.2,
        
        # 板块相关指标权重
        'sector_strength': 1.8,
        'board_heat_index': 1.7,
        'limit_up_ratio': 1.6,
        'broken_ratio': 1.4,
        
        # 市场指标权重
        'market_group': 1.5,
        'market_type': 1.2,
        
        # 复合信号权重
        'rsi_macd_bull_signal': 2.0,
        'rsi_macd_bear_signal': 2.0,
        'strong_bull_consensus': 2.2,
        'strong_bear_consensus': 2.2,
        
        # 形态指标权重
        'pattern_hammer': 1.2,
        'pattern_morning_star': 1.5,
        'pattern_evening_star': 1.5
    }
    
    # 如果有历史数据中的特征重要性信息，优先使用
    if 'feature_importance' in feature_patterns:
        for feature, importance in feature_patterns['feature_importance'].items():
            # 将重要性转换为权重 (重要性乘以基础权重或使用默认值)
            feature_weights[feature] = importance * 5  # 调整系数

    # 提前将模式数据转换为共享对象，减少进程间通信开销
    positive_patterns = feature_patterns.get('positive_patterns', [])
    negative_patterns = feature_patterns.get('negative_patterns', [])
    
    # 记录数据加载后内存使用情况
    log_memory_usage(self.logger, "数据准备完成后内存使用")
    
    # 提前检查所有需要的数据是否有效
    if not isinstance(feature_weights, dict) or len(feature_weights) == 0:
        self.logger.error("特征权重为空或格式错误")
        return {}
        
    if not isinstance(positive_patterns, list) or len(positive_patterns) == 0:
        self.logger.error("正向模式数据为空或格式错误")
        return {}
        
    if not isinstance(negative_patterns, list) or len(negative_patterns) == 0:
        self.logger.error("负向模式数据为空或格式错误")
        return {}
        
    # 根据日期预先组织数据，减少进程间的数据传输
    daily_features_by_date = {}
    for date in trading_days:
        daily_features_by_date[date] = predict_features[predict_features['date_str'] == date].copy()
            
    # 按日期处理数据
    all_predictions = {}
    
    # 初始化全局变量 for global variable initializer process
    global _g_feature_weights, _g_positive_patterns, _g_negative_patterns, _g_clusters, _g_cluster_analysis
    _g_feature_weights = feature_weights
    _g_positive_patterns = positive_patterns
    _g_negative_patterns = negative_patterns
    _g_clusters = clusters
    _g_cluster_analysis = cluster_analysis
    
    # 简化多进程模型，直接使用单一进程来处理日期，避免嵌套进程
    try:
        # 按顺序处理每个日期，简化模型
        self.logger.info(f"开始按顺序处理 {len(trading_days)} 个交易日")
        
        for idx, date in enumerate(trading_days):
            self.logger.info(f"准备处理第 {idx+1}/{len(trading_days)} 个日期: {date}")
            
            # 获取当前日期的特征数据
            daily_features = daily_features_by_date.get(date)
            if daily_features is None or len(daily_features) == 0:
                self.logger.warning(f"日期 {date} 没有有效的特征数据，跳过")
                continue
                
            # 直接处理，不使用进程池和executor，避免嵌套进程问题
            try:
                self.logger.info(f"开始处理日期 {date}，包含 {len(daily_features)} 只股票")
                
                # 直接调用处理函数
                result = process_date_with_timeout(
                    date,
                    daily_features,
                    feature_weights=feature_weights,
                    positive_patterns=positive_patterns,
                    negative_patterns=negative_patterns,
                    clusters=clusters,
                    cluster_analysis=cluster_analysis
                )
                
                if result:  # 只在有结果时更新
                    all_predictions.update(result)
                    self.logger.info(f"完成日期 {date} 的预测，处理了 {len(result)} 只股票")
                else:
                    self.logger.warning(f"日期 {date} 处理后没有返回结果")
                
            except Exception as e:
                self.logger.error(f"处理日期 {date} 时发生错误: {str(e)}")
                import traceback
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                # 继续处理下一个日期，不中断整体流程
            
            # 每个日期处理完后强制释放内存
            self.logger.info(f"日期 {date} 处理完成，执行内存释放")
            release_memory()
            log_memory_usage(self.logger, f"日期 {date} 完成后内存使用")
            
            # 添加短暂休眠，让系统有时间清理资源
            if idx < len(trading_days) - 1:  # 如果不是最后一个日期，添加休眠
                self.logger.info(f"日期 {date} 处理完成后暂停3秒，准备处理下一个日期")
                time.sleep(3)
                
    except Exception as e:
        self.logger.error(f"预测过程中发生错误: {str(e)}")
        import traceback
        self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            
    # 记录内存使用情况
    log_memory_usage(self.logger, "预测完成后内存使用")
    
    # 计算总耗时
    total_time = time.time() - start_time
    stocks_per_sec = len(predict_features) / total_time if total_time > 0 else 0
    
    self.logger.info(f"预测完成，共处理了 {len(all_predictions)} 条预测结果，平均每秒处理 {stocks_per_sec:.2f} 只股票")
    self.logger.info(f"总耗时: {total_time:.2f} 秒")
    
    return all_predictions

# 使用内存映射来存储和读取大型数据对象
def save_to_mmap(obj, prefix="temp_data"):
    """将Python对象保存到内存映射文件中
    
    Args:
        obj: 要保存的Python对象
        prefix: 临时文件名前缀
    
    Returns:
        str: 临时文件路径
    """
    # 使用临时文件创建内存映射
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".mmap")
    os.close(fd)  # 关闭文件描述符，我们将使用mmap直接操作
    
    # 序列化数据
    serialized_data = pickle.dumps(obj)
    data_size = len(serialized_data)
    
    # 写入数据到内存映射文件
    with open(temp_path, "wb") as f:
        f.write(serialized_data)
    
    return temp_path, data_size

def load_from_mmap(file_path, data_size=None):
    """从内存映射文件加载Python对象
    
    Args:
        file_path: 内存映射文件路径
        data_size: 数据大小（可选）
    
    Returns:
        object: 加载的Python对象
    """
    if not os.path.exists(file_path):
        return None
    
    # 打开文件并创建内存映射
    with open(file_path, "rb") as f:
        # 如果未指定数据大小，则读取整个文件
        if data_size is None:
            data_size = os.path.getsize(file_path)
        
        # 创建内存映射
        with mmap.mmap(f.fileno(), data_size, access=mmap.ACCESS_READ) as mm:
            # 从内存映射中加载对象
            return pickle.loads(mm[:])

def cleanup_mmap_file(file_path):
    """清理内存映射临时文件
    
    Args:
        file_path: 临时文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"清理内存映射文件时出错: {str(e)}")

def get_memory_usage():
    """获取当前进程的内存使用情况
    
    Returns:
        float: 当前进程内存使用（MB）
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # 转换为MB
    memory_mb = memory_info.rss / (1024 * 1024)
    return memory_mb

def log_memory_usage(logger, message="当前内存使用"):
    """记录当前内存使用情况
    
    Args:
        logger: 日志记录器
        message: 日志消息前缀
    """
    memory_mb = get_memory_usage()
    logger.info(f"{message}: {memory_mb:.2f} MB")

def release_memory():
    """强制触发垃圾回收以释放内存"""
    import gc
    
    # 主动删除一些大型数据类型的全局变量
    global _g_feature_weights, _g_positive_patterns, _g_negative_patterns, _g_clusters, _g_cluster_analysis
    
    # 第一次垃圾回收
    collected1 = gc.collect(0)  # 收集第0代（最年轻的对象）
    
    # 清理全局缓存变量
    if '_g_feature_weights' in globals() and _g_feature_weights is not None:
        _g_feature_weights = None
    if '_g_positive_patterns' in globals() and _g_positive_patterns is not None:
        _g_positive_patterns = None  
    if '_g_negative_patterns' in globals() and _g_negative_patterns is not None:
        _g_negative_patterns = None
    if '_g_clusters' in globals() and _g_clusters is not None:
        _g_clusters = None
    if '_g_cluster_analysis' in globals() and _g_cluster_analysis is not None:
        _g_cluster_analysis = None
        
    # 第二次垃圾回收，所有代
    collected2 = gc.collect()
    
    # 尝试直接调用Python解释器的内存回收
    import sys
    if hasattr(sys, 'intern'):
        sys.intern('')
        
    # 尝试释放未使用内存给操作系统
    if os.name == 'posix':  # Linux, Mac OS X
        try:
            # 释放未使用内存给操作系统
            os.system('sync')
        except:
            pass
    elif os.name == 'nt':  # Windows
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except:
            pass
            
    print(f"垃圾回收释放了 {collected1 + collected2} 个对象")

def _process_and_return_worker(queue, date_str, df, 
                        fw_path, pp_path, np_path, c_path, ca_path):
    """处理单个日期的工作函数，用于多进程处理
    
    Args:
        queue: 结果队列（已不再使用，但保留参数保持兼容性）
        date_str: 日期字符串
        df: 股票特征DataFrame
        fw_path: 特征权重文件路径(元组或字符串)
        pp_path: 正向模式文件路径(元组或字符串)
        np_path: 负向模式文件路径(元组或字符串)
        c_path: 聚类文件路径(元组或字符串)
        ca_path: 聚类分析文件路径(元组或字符串)
    """
    print(f"工作进程开始处理日期 {date_str}")
    
    # 创建结果文件路径
    import tempfile
    import os
    import pickle
    import time
    
    # 创建临时结果文件
    result_file = os.path.join(tempfile.gettempdir(), f"prediction_result_{date_str}_{int(time.time())}.pkl")
    status_file = os.path.join(tempfile.gettempdir(), f"process_status_{date_str}_{int(time.time())}.txt")
    
    # 写入初始状态
    with open(status_file, 'w') as f:
        f.write(f"RUNNING:{os.getpid()}")
    
    print(f"工作进程 {date_str} 创建状态文件: {status_file}")
    print(f"工作进程 {date_str} 创建结果文件: {result_file}")
    
    try:
        # 如果路径是元组，提取第一个元素作为路径
        if isinstance(fw_path, tuple):
            fw_path = fw_path[0]
        if isinstance(pp_path, tuple):
            pp_path = pp_path[0]
        if isinstance(np_path, tuple):
            np_path = np_path[0]
        if isinstance(c_path, tuple):
            c_path = c_path[0]
        if isinstance(ca_path, tuple):
            ca_path = ca_path[0]
            
        print(f"工作进程开始调用process_single_date处理日期 {date_str}")
        results = process_single_date(
            date_str, df, 
            feature_weights_path=fw_path,
            positive_patterns_path=pp_path,
            negative_patterns_path=np_path,
            clusters_path=c_path,
            cluster_analysis_path=ca_path
        )
        print(f"工作进程完成日期 {date_str} 处理，准备保存结果，结果大小: {len(results) if isinstance(results, dict) else '未知'}")
        
        # 通过文件保存结果，避免队列问题
        try:
            # 尝试将结果转为JSON再转回字典，确保序列化安全
            try:
                import json
                json_str = json.dumps(results, cls=NumpyEncoder)
                results = json.loads(json_str)
                print(f"工作进程将结果转换为JSON并重新加载，确保序列化安全")
            except Exception as json_err:
                print(f"工作进程JSON转换失败: {str(json_err)}，将直接使用原始结果")
            
            # 检查文件夹是否存在
            dir_path = os.path.dirname(result_file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"工作进程创建目录: {dir_path}")
            
            # 检查文件是否可写
            try:
                with open(result_file, 'wb') as test_f:
                    test_f.write(b"TEST")
                print(f"工作进程确认结果文件可写: {result_file}")
            except Exception as test_err:
                print(f"工作进程测试文件写入失败: {str(test_err)}")
                # 尝试使用不同的目录
                result_file = os.path.join(os.getcwd(), f"prediction_result_{date_str}_{int(time.time())}.pkl")
                print(f"工作进程改用当前工作目录创建结果文件: {result_file}")
            
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"工作进程已将日期 {date_str} 的结果保存到文件: {result_file}")
            
            # 验证结果文件是否成功创建
            if os.path.exists(result_file):
                file_size = os.path.getsize(result_file)
                print(f"工作进程确认结果文件已创建，大小: {file_size} 字节")
                
                # 测试文件是否可读
                try:
                    with open(result_file, 'rb') as check_f:
                        test_content = check_f.read(10)
                        print(f"工作进程确认结果文件可读")
                except Exception as check_err:
                    print(f"工作进程检查文件可读性失败: {str(check_err)}")
            else:
                print(f"警告：工作进程写入文件后，结果文件不存在: {result_file}")
            
            # 更新状态为完成
            with open(status_file, 'w') as f:
                f.write(f"COMPLETED:{result_file}")
            print(f"工作进程更新状态文件为完成状态: {status_file}")
            
        except Exception as save_err:
            print(f"工作进程保存结果时出错: {str(save_err)}")
            import traceback
            print(f"工作进程保存结果错误详情: {traceback.format_exc()}")
            with open(status_file, 'w') as f:
                f.write(f"ERROR:{str(save_err)}")
        
    except Exception as e:
        print(f"工作进程处理日期 {date_str} 时出错: {str(e)}")
        import traceback
        print(f"工作进程处理错误详情: {traceback.format_exc()}")
        try:
            # 记录错误信息到状态文件
            with open(status_file, 'w') as f:
                f.write(f"ERROR:{str(e)}")
        except:
            print(f"工作进程尝试写入错误信息时失败")
    finally:
        # 确保在工作进程退出前释放内存
        print(f"工作进程 {date_str} 退出前执行内存释放")
        try:
            import gc
            gc.collect()
        except:
            pass
        
        # 确保状态文件存在，即使之前的写入失败
        try:
            if not os.path.exists(status_file):
                with open(status_file, 'w') as f:
                    f.write("UNKNOWN:状态未知")
        except:
            pass
            
        # 队列已不再使用，但为保持兼容性，放入一个简单消息
        try:
            queue.put({"status": "completed", "result_file": result_file, "status_file": status_file})
        except:
            pass
            
        print(f"工作进程 {date_str} 即将退出")

def process_date_with_timeout(date, daily_features, feature_weights=None, positive_patterns=None, 
                             negative_patterns=None, clusters=None, cluster_analysis=None):
    """Process a single date with timeout to prevent hanging
    
    Args:
        date: Date string to process
        daily_features: DataFrame with stock features for the date
        feature_weights: Feature weight dictionary
        positive_patterns: List of positive patterns
        negative_patterns: List of negative patterns
        clusters: Cluster information
        cluster_analysis: Cluster analysis results
        
    Returns:
        dict: Dictionary of prediction results
    """
    start_time = time.time()
    print(f"===开始处理日期 {date}，包含 {len(daily_features) if daily_features is not None else 0} 只股票===")
    
    print(f"特征权重内容: {list(feature_weights.keys())[:10] if isinstance(feature_weights, dict) else 'None'}")
    print(f"特征权重示例: {list(feature_weights.items())[:3] if isinstance(feature_weights, dict) else 'None'}")
    
    # 存储临时文件路径列表，确保最后能清理所有文件
    temp_file_paths = []
    status_file = None
    result_file = None
    
    try:
        # 创建临时内存映射文件以共享大型数据
        # 使用临时文件路径确保它们会在系统重启后自动清理
        if feature_weights and isinstance(feature_weights, dict) and len(feature_weights) > 0:
            try:
                feature_weights_tuple = save_to_mmap(feature_weights, prefix=f"feature_weights_{date}")
                feature_weights_path = feature_weights_tuple
                temp_file_paths.append(feature_weights_path[0] if isinstance(feature_weights_path, tuple) else feature_weights_path)
                print(f"特征权重保存到临时文件: {feature_weights_path}")
                feature_weights = None  # Release memory
            except Exception as e:
                print(f"保存特征权重时出错: {e}")
                feature_weights_path = None
        else:
            feature_weights_path = None
            
        if positive_patterns and isinstance(positive_patterns, list) and len(positive_patterns) > 0:
            try:
                positive_patterns_tuple = save_to_mmap(positive_patterns, prefix=f"positive_patterns_{date}")
                positive_patterns_path = positive_patterns_tuple
                temp_file_paths.append(positive_patterns_path[0] if isinstance(positive_patterns_path, tuple) else positive_patterns_path)
                positive_patterns = None  # Release memory
            except Exception as e:
                print(f"保存正向模式时出错: {e}")
                positive_patterns_path = None
        else:
            positive_patterns_path = None
            
        if negative_patterns and isinstance(negative_patterns, list) and len(negative_patterns) > 0:
            try:
                negative_patterns_tuple = save_to_mmap(negative_patterns, prefix=f"negative_patterns_{date}")
                negative_patterns_path = negative_patterns_tuple
                temp_file_paths.append(negative_patterns_path[0] if isinstance(negative_patterns_path, tuple) else negative_patterns_path)
                negative_patterns = None  # Release memory
            except Exception as e:
                print(f"保存负向模式时出错: {e}")
                negative_patterns_path = None
        else:
            negative_patterns_path = None
            
        if clusters and isinstance(clusters, dict) and len(clusters) > 0:
            try:
                clusters_tuple = save_to_mmap(clusters, prefix=f"clusters_{date}")
                clusters_path = clusters_tuple
                temp_file_paths.append(clusters_path[0] if isinstance(clusters_path, tuple) else clusters_path)
                clusters = None  # Release memory
            except Exception as e:
                print(f"保存聚类数据时出错: {e}")
                clusters_path = None
        else:
            clusters_path = None
            
        if cluster_analysis and isinstance(cluster_analysis, dict) and len(cluster_analysis) > 0:
            try:
                cluster_analysis_tuple = save_to_mmap(cluster_analysis, prefix=f"cluster_analysis_{date}")
                cluster_analysis_path = cluster_analysis_tuple
                temp_file_paths.append(cluster_analysis_path[0] if isinstance(cluster_analysis_path, tuple) else cluster_analysis_path)
                cluster_analysis = None  # Release memory
            except Exception as e:
                print(f"保存聚类分析时出错: {e}")
                cluster_analysis_path = None
        else:
            cluster_analysis_path = None
            
        # 强制垃圾回收
        release_memory()
        print(f"日期 {date} 的临时数据文件创建完成，准备启动处理进程")
        
        # 创建单独的进程来处理此日期 - 如果进程挂起，我们可以终止它
        try:
            # 队列不再用于结果传递，但为保持兼容性保留
            result_queue = mp.Queue()
            
            # 启动处理进程
            date_process = mp.Process(
                target=_process_and_return_worker,
                args=(result_queue, date, daily_features,
                     feature_weights_path, positive_patterns_path,
                     negative_patterns_path, clusters_path,
                     cluster_analysis_path)
            )
            date_process.start()
            print(f"日期 {date} 的处理进程已启动 (PID: {date_process.pid})")
            
            # 设置超时时间，根据股票数量调整
            if daily_features is not None and not daily_features.empty:
                # 减少超时时间，避免进程挂起太久
                timeout = max(180, min(len(daily_features) * 0.3, 600))  # 3-10分钟，根据股票数量调整
                print(f"日期 {date} 设置超时时间为 {timeout:.1f} 秒")
            else:
                timeout = 180  # 默认3分钟
                
            # 检测进程状态文件路径
            import glob
            import os
            
            # 等待进程完成或超时
            start_wait = time.time()
            status_found = False
            
            while time.time() - start_wait < timeout:
                # 检查进程是否仍在运行
                if not date_process.is_alive():
                    print(f"日期 {date} 处理进程已结束，进程状态: {'活跃' if date_process.is_alive() else '已结束'}")
                    break
                
                # 尝试查找状态文件
                if not status_found:
                    status_files = glob.glob(os.path.join(tempfile.gettempdir(), f"process_status_{date}_*.txt"))
                    if status_files:
                        status_file = status_files[0]
                        status_found = True
                        print(f"找到日期 {date} 的状态文件: {status_file}")
                    
                # 如果找到状态文件，检查进程状态
                if status_found:
                    try:
                        with open(status_file, 'r') as f:
                            status = f.read().strip()
                            print(f"日期 {date} 当前状态: {status}")
                            
                            if status.startswith('COMPLETED:'):
                                # 进程已完成，获取结果文件路径
                                result_file = status.split(':', 1)[1]
                                print(f"日期 {date} 处理已完成，结果文件: {result_file}")
                                break
                            elif status.startswith('ERROR:'):
                                # 进程出错
                                error_msg = status.split(':', 1)[1]
                                print(f"日期 {date} 处理出错: {error_msg}")
                                break
                    except Exception as e:
                        print(f"读取状态文件出错: {e}")
                
                # 短暂休眠，避免频繁检查
                time.sleep(2)
            
            # 检查是否超时
            if time.time() - start_wait >= timeout:
                print(f"警告: 处理日期 {date} 超时 ({timeout:.1f}秒)，正在终止进程...")
                date_process.terminate()
                time.sleep(1)
                
                # 如果进程仍然存活，强制终止
                if date_process.is_alive():
                    print(f"错误: 进程无法正常终止，强制杀死进程")
                    date_process.kill()
                    time.sleep(1)
                
                # 最终检查
                print(f"日期 {date} 进程终止后状态: {'仍然活跃' if date_process.is_alive() else '已结束'}")
                
                results = {}
            else:
                # 进程已结束，尝试获取结果
                print(f"日期 {date} 进程已结束，准备获取结果")
                results = {}
                
                # 从文件获取结果
                if result_file and os.path.exists(result_file):
                    try:
                        print(f"正在从文件加载日期 {date} 的结果: {result_file}")
                        # 检查文件大小
                        file_size = os.path.getsize(result_file)
                        print(f"结果文件大小: {file_size} 字节")
                        
                        # 检查文件前10个字节，确认是否是pickle格式
                        with open(result_file, 'rb') as f_check:
                            header = f_check.read(10)
                            print(f"文件头部字节: {' '.join(f'{b:02x}' for b in header)}")
                        
                        # 尝试加载结果
                        with open(result_file, 'rb') as f:
                            try:
                                results = pickle.load(f)
                                print(f"日期 {date} 处理成功，从文件加载了 {len(results) if isinstance(results, dict) else 'unknown'} 条预测结果")
                                print(f"结果类型: {type(results)}")
                                
                                # 如果不是字典，尝试转换
                                if not isinstance(results, dict):
                                    print(f"警告: 结果不是字典类型，尝试转换")
                                    try:
                                        import json
                                        # 尝试先转JSON再转回
                                        if hasattr(results, '__dict__'):
                                            results = results.__dict__
                                            print(f"将对象转换为字典")
                                        else:
                                            print(f"结果无法直接转换为字典")
                                    except Exception as convert_err:
                                        print(f"转换结果类型失败: {convert_err}")
                                else:
                                    print(f"结果包含 {len(results)} 个键: {list(results.keys())[:10]}")
                            except Exception as pickle_err:
                                print(f"pickle加载失败，错误: {pickle_err}")
                                # 尝试使用不同的加载方式
                                f.seek(0)
                                raw_data = f.read()
                                print(f"读取了 {len(raw_data)} 字节的原始数据")
                                try:
                                    # 尝试使用marshal加载
                                    import marshal
                                    results = marshal.loads(raw_data)
                                    print(f"使用marshal成功加载了结果")
                                except:
                                    print(f"marshal加载也失败")
                        
                        # 处理完成后删除结果文件
                        try:
                            os.unlink(result_file)
                            print(f"已删除结果文件: {result_file}")
                        except Exception as e:
                            print(f"删除结果文件失败: {e}")
                    except Exception as e:
                        print(f"从文件加载结果时出错: {e}")
                        import traceback
                        print(f"加载错误详情: {traceback.format_exc()}")
                else:
                    if result_file:
                        print(f"结果文件不存在: {result_file}")
                        # 检查临时目录下的所有prediction_result文件
                        try:
                            import glob
                            prediction_files = glob.glob(os.path.join(tempfile.gettempdir(), f"prediction_result_{date}_*.pkl"))
                            if prediction_files:
                                print(f"找到可能的结果文件: {prediction_files}")
                                # 尝试从第一个文件加载
                                alternate_file = prediction_files[0]
                                print(f"尝试加载替代文件: {alternate_file}")
                                try:
                                    with open(alternate_file, 'rb') as alt_f:
                                        alt_results = pickle.load(alt_f)
                                    print(f"从替代文件成功加载了 {len(alt_results)} 条结果")
                                    results = alt_results
                                    result_file = alternate_file
                                except Exception as alt_err:
                                    print(f"加载替代文件失败: {alt_err}")
                            else:
                                print(f"未找到任何备选结果文件")
                        except Exception as glob_err:
                            print(f"查找备选文件时出错: {glob_err}")
                    else:
                        print(f"没有提供结果文件路径")
                        
                    # 如果没有找到结果文件，尝试从队列获取(兼容性保留)
                    if not results:
                        try:
                            if not result_queue.empty():
                                queue_result = result_queue.get(timeout=1)
                                print(f"从队列获取到信息: {queue_result}")
                                # 从队列通知中获取结果文件路径
                                if isinstance(queue_result, dict) and 'result_file' in queue_result:
                                    queue_result_file = queue_result.get('result_file')
                                    if queue_result_file and os.path.exists(queue_result_file):
                                        print(f"从队列获取到结果文件路径: {queue_result_file}")
                                        try:
                                            with open(queue_result_file, 'rb') as f:
                                                results = pickle.load(f)
                                            print(f"从队列提供的文件中加载了 {len(results) if isinstance(results, dict) else 'unknown'} 条预测结果")
                                            # 更新结果文件路径，以便后续删除
                                            result_file = queue_result_file
                                        except Exception as e:
                                            print(f"从队列提供的结果文件加载失败: {e}")
                                    else:
                                        print(f"队列提供的结果文件不存在: {queue_result_file}")
                                        
                                    # 获取状态文件路径
                                    if 'status_file' in queue_result:
                                        status_file = queue_result.get('status_file')
                                        if not status_found and status_file and os.path.exists(status_file):
                                            status_found = True
                                # 注意：队列中不再包含实际结果，只是通知
                        except Exception as e:
                            print(f"从队列获取结果时出错: {e}")
                            import traceback
                            print(f"队列处理错误详情: {traceback.format_exc()}")
                
                # 如果结果为空，尝试再次查找状态文件
                if not results and not status_found:
                    status_files = glob.glob(os.path.join(tempfile.gettempdir(), f"process_status_{date}_*.txt"))
                    if status_files:
                        status_file = status_files[0]
                        print(f"找到日期 {date} 的状态文件: {status_file}")
                        
                        try:
                            with open(status_file, 'r') as f:
                                status = f.read().strip()
                                print(f"日期 {date} 最终状态: {status}")
                                
                                if status.startswith('COMPLETED:'):
                                    result_file = status.split(':', 1)[1]
                                    if os.path.exists(result_file):
                                        with open(result_file, 'rb') as f:
                                            results = pickle.load(f)
                                        print(f"最终从文件加载了 {len(results)} 条预测结果")
                        except Exception as e:
                            print(f"读取最终状态文件出错: {e}")
                
                if not results:
                    print(f"警告: 日期 {date} 没有返回有效结果")
            
            # 清理状态和结果文件
            try:
                if status_file and os.path.exists(status_file):
                    os.unlink(status_file)
                    print(f"已删除状态文件: {status_file}")
                if result_file and os.path.exists(result_file):
                    os.unlink(result_file)
                    print(f"已删除结果文件: {result_file}")
            except Exception as e:
                print(f"清理状态或结果文件失败: {e}")
                    
        except Exception as proc_err:
            print(f"创建或管理日期 {date} 的子进程时出错: {str(proc_err)}")
            results = {}
        finally:
            # 确保不留下任何子进程
            try:
                if 'date_process' in locals() and date_process.is_alive():
                    date_process.terminate()
                    date_process.join(1)
                    print(f"日期 {date} 的处理进程被显式终止")
            except:
                pass
            
            # 安全关闭队列
            try:
                if 'result_queue' in locals():
                    if hasattr(result_queue, 'close'):
                        result_queue.close()
                    print(f"日期 {date} 的结果队列已关闭")
            except:
                pass
                
        # 清理临时内存映射文件
        print(f"准备清理日期 {date} 的 {len(temp_file_paths)} 个临时文件")
        for path in temp_file_paths:
            if path:
                try:
                    cleanup_mmap_file(path)
                    print(f"清理临时文件: {path}")
                except Exception as e:
                    print(f"清理临时文件 {path} 失败: {e}")
        
        processing_time = time.time() - start_time
        print(f"===日期 {date} 总处理时间: {processing_time:.2f} 秒，处理完成===")
        return results
    except Exception as e:
        logging.error(f"处理日期 {date} 的外部进程失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 确保清理所有临时文件
        for path in temp_file_paths:
            if path:
                try:
                    cleanup_mmap_file(path)
                except Exception:
                    pass
                    
        # 清理状态和结果文件
        try:
            if status_file and os.path.exists(status_file):
                os.unlink(status_file)
            if result_file and os.path.exists(result_file):
                os.unlink(result_file)
        except:
            pass
                    
        print(f"===日期 {date} 处理异常退出===")
        return {} 